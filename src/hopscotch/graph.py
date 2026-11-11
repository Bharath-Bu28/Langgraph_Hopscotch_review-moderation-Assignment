"""One graph for all products and reviews; code owns every route and limit."""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from hopscotch.privacy import has_link, redact
from hopscotch.schemas import ReviewState

LOOKUPS = ["count_complaints", "get_return_rate", "vendor_record"]
MODES = ("base", "spike", "checked")


def complaint_facts(state: ReviewState) -> dict:
    return {key: state[key] for key in ("complaint_count", "return_rate", "vendor_alert_count")}


def triage(state: ReviewState):
    # First matching rule wins. Links/abuse always override a positive sentiment.
    if has_link(state["cleaned_text"]) or state["kind"] == "abusive":
        return "hide_review"
    if state["kind"] == "complaint":
        return LOOKUPS
    if state["kind"] == "question":
        return "answer_agent"
    return "publish_review"


def spike_rule(state: ReviewState) -> bool:
    return state["complaint_count"] >= 5 or state["return_rate"] >= 0.12


def build_graph(catalog, model, tools, mode: str = "base"):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def clean_review(state):
        return {"cleaned_text": redact(state["text"]), "trace": ["clean_review"]}

    async def read_review(state):
        result = await model.classify(state["cleaned_text"])
        return {**result.model_dump(), "trace": ["read_review"]}

    async def count_complaints(state):
        value = await catalog.count_complaints(state["product_id"], state["topic"])
        return {"complaint_count": value, "trace": ["count_complaints"]}

    async def get_return_rate(state):
        value = await catalog.get_return_rate(state["product_id"])
        return {"return_rate": value, "trace": ["get_return_rate"]}

    async def vendor_record(state):
        value = await catalog.vendor_record(state["product_id"])
        return {"vendor_alert_count": value, "trace": ["vendor_record"]}

    async def reply_and_alert(state):
        draft = await model.complaint(state["cleaned_text"], complaint_facts(state))
        return {
            "reply": redact(draft.reply),
            "alert": redact(draft.alert),
            "trace": ["reply_and_alert"],
        }

    async def write_reply(state):
        reply = await model.reply(state["cleaned_text"], complaint_facts(state))
        return {"reply": redact(reply), "spike": spike_rule(state), "trace": ["write_reply"]}

    async def write_alert(state):
        alert = await model.alert(state["cleaned_text"], complaint_facts(state))
        return {"alert": redact(alert), "trace": ["write_alert"]}

    async def answer_agent(state):
        messages = state.get("messages", []) or [HumanMessage(state["cleaned_text"])]
        result = await model.answer(messages, state.get("tool_rounds", 0))
        answer = result.content if isinstance(result.content, str) else result.text
        return {
            "messages": ([messages[0]] if not state.get("messages") else []) + [result],
            "answer": answer if not result.tool_calls else "",
            "trace": ["answer_agent"],
        }

    def after_answer(state):
        if state["messages"][-1].tool_calls:
            return "tools" if state.get("tool_rounds", 0) < 4 else "hide_review"
        if not state.get("answer", "").strip():
            return "hide_review"
        return "check_answer" if mode == "checked" else "publish_review"

    async def run_tools(state):
        result = await tool_node.ainvoke(state)
        names = ",".join(call["name"] for call in state["messages"][-1].tool_calls)
        return {
            **result,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "trace": [f"tools:{names}"],
        }

    async def check_answer(state):
        checked = await model.check(state["cleaned_text"], state["answer"], state["messages"])
        # A deterministic PII check can veto the model's verdict.
        pii = redact(state["answer"]) != state["answer"]
        return {
            "check_passed": checked.passed and not pii,
            "check_feedback": "Remove PII." if pii else redact(checked.feedback),
            "trace": ["check_answer"],
        }

    def after_check(state):
        if state["check_passed"]:
            return "publish_review"
        return "revise_answer" if state.get("revisions", 0) < 2 else "hide_review"

    def revise_answer(state):
        return {
            "revisions": state.get("revisions", 0) + 1,
            "messages": [
                HumanMessage("Rewrite your answer. Checker feedback: " + state["check_feedback"])
            ],
            "trace": ["revise_answer"],
        }

    def hide_review(state):
        reason = "Review requires human moderation."
        if state.get("kind") == "question":
            reason = "Answer failed validation, was empty, or exceeded the tool budget."
        return {"status": "held", "hold_reason": reason, "trace": ["hide_review"]}

    def publish_review(state):
        return {
            "status": "published",
            "answer": redact(state.get("answer", "")),
            "trace": ["publish_review"],
        }

    graph = StateGraph(ReviewState)
    nodes = {
        "clean_review": clean_review,
        "read_review": read_review,
        "count_complaints": count_complaints,
        "get_return_rate": get_return_rate,
        "vendor_record": vendor_record,
        "answer_agent": answer_agent,
        "tools": run_tools,
        "hide_review": hide_review,
        "publish_review": publish_review,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "clean_review")
    graph.add_edge("clean_review", "read_review")
    graph.add_conditional_edges(
        "read_review",
        triage,
        {
            name: name
            for name in [
                "hide_review",
                "publish_review",
                "answer_agent",
                *LOOKUPS,
            ]
        },
    )
    if mode == "base":
        graph.add_node("reply_and_alert", reply_and_alert)
        graph.add_edge(LOOKUPS, "reply_and_alert")  # Barrier: waits for ALL three.
        graph.add_edge("reply_and_alert", "publish_review")
    else:
        graph.add_node("write_reply", write_reply)
        graph.add_node("write_alert", write_alert)
        graph.add_edge(LOOKUPS, "write_reply")
        graph.add_conditional_edges(
            "write_reply",
            lambda s: "write_alert" if s["spike"] else "publish_review",
            ["write_alert", "publish_review"],
        )
        graph.add_edge("write_alert", "publish_review")
    destinations = ["tools", "publish_review", "hide_review"]
    if mode == "checked":
        graph.add_node("check_answer", check_answer)
        graph.add_node("revise_answer", revise_answer)
        graph.add_conditional_edges(
            "check_answer", after_check, ["publish_review", "revise_answer", "hide_review"]
        )
        graph.add_edge("revise_answer", "answer_agent")
        destinations.append("check_answer")
    graph.add_conditional_edges("answer_agent", after_answer, destinations)
    graph.add_edge("tools", "answer_agent")
    graph.add_edge("hide_review", END)
    graph.add_edge("publish_review", END)
    return graph.compile()

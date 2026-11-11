import asyncio
import copy
import json
from importlib.resources import files

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from hopscotch.catalog import Catalog
from hopscotch.graph import build_graph, spike_rule
from hopscotch.models import OfflineModel
from hopscotch.privacy import redact
from hopscotch.schemas import AnswerCheck
from hopscotch.service import ResultStore, process_review
from hopscotch.tools import make_tools


@pytest.fixture
def reviews():
    return json.loads(files("hopscotch").joinpath("data/reviews.json").read_text())


def setup(mode="base", model=None, catalog=None):
    catalog = catalog or Catalog()
    return catalog, build_graph(catalog, model or OfflineModel(), make_tools(catalog), mode)


@pytest.mark.parametrize(
    "mode,alert_ids",
    [
        ("base", {"R-3", "R-4"}),
        ("spike", {"R-3"}),
        ("checked", {"R-3"}),
    ],
)
async def test_all_seven_reviews(reviews, mode, alert_ids):
    catalog, graph = setup(mode)
    results = [await process_review(graph, catalog, raw) for raw in reviews]
    assert {r.review_id for r in results if r.alert} == alert_ids
    assert {r.review_id for r in results if r.status == "held"} == {"R-5", "R-6"}
    assert all(r.status == "published" for r in results if r.review_id not in {"R-5", "R-6"})
    assert "98200" not in results[4].model_dump_json()
    assert results[2].facts == {
        "complaint_count": 6,
        "return_rate": 0.16,
        "vendor_alert_count": 2,
    }
    assert results[3].facts["complaint_count"] == 1
    assert results[3].facts["return_rate"] == 0.03


async def test_question_batches_and_dependent_size(reviews):
    catalog, graph = setup()
    r2 = await process_review(graph, catalog, reviews[1])
    r7 = await process_review(graph, catalog, reviews[6])
    assert r2.tool_rounds == 1
    assert "tools:care_instructions,stock" in r2.trace
    assert r7.tool_rounds == 2
    assert [s for s in r7.trace if s.startswith("tools:")] == ["tools:size_chart", "tools:stock"]
    assert "4-5Y" in r7.answer


async def test_size_result_actually_drives_stock_argument(reviews):
    catalog = Catalog()
    data = copy.deepcopy(catalog.data)
    data["products"]["P-FABRIC"]["sizes"][1]["label"] = "CUSTOM-SIZE"
    data["products"]["P-FABRIC"]["stock"]["CUSTOM-SIZE"] = 0
    catalog, graph = setup(catalog=Catalog(data))
    result = await process_review(graph, catalog, reviews[6])
    assert "CUSTOM-SIZE is out of stock" in result.answer


async def test_parallel_lookups_and_join(reviews):
    class BarrierCatalog(Catalog):
        def __init__(self):
            super().__init__()
            self.started = set()
            self.all_started = asyncio.Event()

        async def barrier(self, name):
            self.started.add(name)
            if len(self.started) == 3:
                self.all_started.set()
            await asyncio.wait_for(self.all_started.wait(), timeout=2)

        async def count_complaints(self, product_id, topic):
            await self.barrier("complaints")
            return await super().count_complaints(product_id, topic)

        async def get_return_rate(self, product_id):
            await self.barrier("returns")
            return await super().get_return_rate(product_id)

        async def vendor_record(self, product_id):
            await self.barrier("vendor")
            return await super().vendor_record(product_id)

    class JoinModel(OfflineModel):
        async def complaint(self, text, facts):
            assert facts == {"complaint_count": 6, "return_rate": 0.16, "vendor_alert_count": 2}
            return await super().complaint(text, facts)

    catalog, graph = setup(catalog=BarrierCatalog(), model=JoinModel())
    result = await process_review(graph, catalog, reviews[2])
    assert result.status == "published"
    assert catalog.started == {"complaints", "returns", "vendor"}
    assert result.trace.count("reply_and_alert") == 1


async def test_four_round_cap(reviews):
    class NeverDone(OfflineModel):
        async def answer(self, messages, rounds):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "care_instructions",
                        "args": {},
                        "id": f"call-{rounds}",
                        "type": "tool_call",
                    }
                ],
            )

    catalog, graph = setup(model=NeverDone())
    result = await process_review(graph, catalog, reviews[1])
    assert result.status == "held"
    assert result.tool_rounds == 4
    assert result.answer == ""


@pytest.mark.parametrize(
    "always_fail,expected_status,rewrites",
    [
        (False, "published", 1),
        (True, "held", 2),
    ],
)
async def test_answer_rewrites(reviews, always_fail, expected_status, rewrites):
    class BadFirst(OfflineModel):
        def __init__(self):
            self.check_calls = 0

        async def answer(self, messages, rounds):
            if messages and isinstance(messages[-1], ToolMessage):
                return AIMessage(content="We guarantee a refund.")
            return await super().answer(messages, rounds)

        async def check(self, question, answer, messages):
            self.check_calls += 1
            return AnswerCheck(
                passed=not always_fail and self.check_calls > 1,
                feedback="Remove unsupported refund promise.",
            )

    catalog, graph = setup("checked", model=BadFirst())
    result = await process_review(graph, catalog, reviews[1])
    assert result.status == expected_status
    assert result.revisions == rewrites
    assert "guarantee" not in result.answer


async def test_pii_does_not_reach_model_graph_or_disk(reviews, tmp_path):
    captured = []

    class Spy(OfflineModel):
        async def classify(self, text):
            captured.append(text)
            return await super().classify(text)

    catalog, graph = setup(model=Spy())
    store = ResultStore(str(tmp_path / "reviews.sqlite3"))
    raw = {**reviews[4], "text": reviews[4]["text"] + " Email me at parent@example.com"}
    result = await process_review(graph, catalog, raw, store)
    serialized = json.dumps(captured) + result.model_dump_json() + json.dumps(store.all())
    assert "98200" not in serialized and "parent@example.com" not in serialized
    assert "[PHONE]" in serialized and "[EMAIL]" in serialized
    initial = {"text": redact(raw["text"]), "product_id": raw["product_id"]}
    state = await graph.ainvoke(initial)
    assert "98200" not in str(state)


async def test_links_override_praise(reviews):
    catalog, graph = setup()
    result = await process_review(graph, catalog, reviews[5])
    assert result.kind == "praise"  # The Python link rule still hides it.
    assert result.status == "held"


@pytest.mark.parametrize(
    "count,rate,spike", [(4, 0.119, False), (5, 0.0, True), (0, 0.12, True), (6, 0.16, True)]
)
def test_spike_boundaries(count, rate, spike):
    assert spike_rule({"complaint_count": count, "return_rate": rate}) is spike


async def test_windows_ignore_old_records():
    catalog = Catalog()
    assert await catalog.count_complaints("P-FABRIC", "colour_bleed") == 6
    assert await catalog.count_complaints("P-FABRIC", "button") == 1
    assert await catalog.vendor_record("P-FABRIC") == 2


async def test_unknown_product_rejected(reviews):
    catalog, graph = setup()
    with pytest.raises(ValueError, match="Unknown product_id"):
        await process_review(graph, catalog, {**reviews[0], "product_id": "NONEXISTENT"})


async def test_failures_hold_without_leaking_error_text(reviews):
    class Failing(OfflineModel):
        async def classify(self, text):
            raise RuntimeError("private data parent@example.com")

    catalog, graph = setup(model=Failing())
    result = await process_review(graph, catalog, reviews[0])
    assert result.status == "held"
    assert "RuntimeError" in result.hold_reason
    assert "parent@example.com" not in result.model_dump_json()


async def test_store_repeated_run_has_one_result(reviews, tmp_path):
    catalog, graph = setup()
    store = ResultStore(str(tmp_path / "results.sqlite3"))
    for _ in range(2):
        await process_review(graph, catalog, reviews[0], store)
    assert len(store.all()) == 1


def test_tool_schemas_do_not_expose_product_or_state():
    for tool in make_tools(Catalog()):
        props = tool.tool_call_schema.model_json_schema().get("properties", {})
        assert "state" not in props and "product_id" not in props


async def test_final_answer_after_fourth_batch_can_publish(reviews):
    class FourThenDone(OfflineModel):
        async def answer(self, messages, rounds):
            if rounds == 4:
                return AIMessage(content="Machine wash cold at 30 C with similar colours.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "care_instructions",
                        "args": {},
                        "id": f"call-{rounds}",
                        "type": "tool_call",
                    }
                ],
            )

    catalog, graph = setup(model=FourThenDone())
    result = await process_review(graph, catalog, reviews[1])
    assert result.status == "published"
    assert result.tool_rounds == 4


async def test_empty_answer_is_held(reviews):
    class Empty(OfflineModel):
        async def answer(self, messages, rounds):
            return AIMessage(content=" ")

    catalog, graph = setup(model=Empty())
    result = await process_review(graph, catalog, reviews[1])
    assert result.status == "held"


async def test_unknown_tool_is_returned_as_error(reviews):
    class InvalidTool(OfflineModel):
        async def answer(self, messages, rounds):
            if rounds == 0:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delete_catalog",
                            "args": {},
                            "id": "invalid",
                            "type": "tool_call",
                        }
                    ],
                )
            assert isinstance(messages[-1], ToolMessage)
            assert messages[-1].status == "error"
            return AIMessage(content="The product information is unavailable.")

    catalog, graph = setup(model=InvalidTool())
    result = await process_review(graph, catalog, reviews[1])
    assert result.status == "published"
    assert result.tool_rounds == 1

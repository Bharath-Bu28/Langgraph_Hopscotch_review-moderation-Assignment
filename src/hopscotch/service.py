"""Validated entry point and local publication/alert simulation."""

import json
import sqlite3
from pathlib import Path

from hopscotch.privacy import redact, sanitize_value
from hopscotch.schemas import ReviewInput, ReviewResult


class ResultStore:
    """A repeat run replaces the same review result; it does not send notifications."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reviews "
                "(review_id TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
            )

    def save(self, result: ReviewResult):
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO reviews VALUES (?, ?) ON CONFLICT(review_id) "
                "DO UPDATE SET result_json=excluded.result_json",
                (result.review_id, result.model_dump_json()),
            )

    def all(self) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            return [
                json.loads(row[0])
                for row in connection.execute("SELECT result_json FROM reviews ORDER BY review_id")
            ]


async def process_review(graph, catalog, raw: dict, store: ResultStore | None = None):
    review = ReviewInput.model_validate(raw)
    catalog.product(review.product_id)  # Reject unknown IDs before any model calls.
    cleaned = redact(review.text)
    initial = {
        "review_id": review.review_id,
        "product_id": review.product_id,
        "text": cleaned,
        "messages": [],
        "trace": [],
        "tool_rounds": 0,
        "revisions": 0,
    }
    # Raw PII never enters graph state, traces, model context or persisted output.
    # clean_review remains the first graph node, and redaction is idempotent.
    try:
        state = await graph.ainvoke(initial, {"recursion_limit": 64})
    except Exception as exc:
        # Do not persist exception messages: remote services may echo private input.
        state = {
            **initial,
            "kind": "unknown",
            "status": "held",
            "hold_reason": f"Processing failed ({type(exc).__name__}); manual review required.",
        }
    held = state["status"] == "held"
    result = ReviewResult(
        review_id=review.review_id,
        product_id=review.product_id,
        text=cleaned,
        kind=state.get("kind", "unknown"),
        status=state["status"],
        reply="" if held else state.get("reply", ""),
        answer="" if held else state.get("answer", ""),
        alert="" if held else state.get("alert", ""),
        facts={
            key: state[key]
            for key in ("complaint_count", "return_rate", "vendor_alert_count")
            if key in state
        },
        hold_reason=state.get("hold_reason", ""),
        tool_rounds=state.get("tool_rounds", 0),
        revisions=state.get("revisions", 0),
        trace=state.get("trace", []),
    )
    result = ReviewResult.model_validate(sanitize_value(result.model_dump()))
    if store:
        store.save(result)
    return result

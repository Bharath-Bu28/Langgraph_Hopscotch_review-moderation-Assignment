"""Run the seven assignment reviews and inspect public output separately from alerts."""

import argparse
import asyncio
import json
from importlib.resources import files
from pathlib import Path

from hopscotch.catalog import Catalog
from hopscotch.graph import MODES, build_graph
from hopscotch.models import LiveModel, OfflineModel
from hopscotch.service import ResultStore, process_review
from hopscotch.tools import make_tools


async def run(args):
    catalog = Catalog(latency=args.lookup_latency)
    tools = make_tools(catalog)
    model = LiveModel(tools) if args.provider == "live" else OfflineModel()
    graph = build_graph(catalog, model, tools, mode=args.mode)
    if args.graph:
        Path(args.graph).parent.mkdir(parents=True, exist_ok=True)
        Path(args.graph).write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    data = (
        Path(args.input).read_text(encoding="utf-8")
        if args.input
        else files("hopscotch").joinpath("data/reviews.json").read_text(encoding="utf-8")
    )
    reviews = json.loads(data)
    store = ResultStore(args.db)
    for raw in reviews:
        result = await process_review(graph, catalog, raw, store)
        if args.json:
            print(result.model_dump_json())
        else:
            alert = "yes" if result.alert else "no"
            print(
                f"{result.review_id}: {result.kind:9} {result.status:9} "
                f"alert={alert:3} tools={result.tool_rounds} rewrites={result.revisions}"
            )
            if result.answer or result.reply or result.hold_reason:
                print("  " + (result.answer or result.reply or result.hold_reason))
            print("  " + " -> ".join(result.trace))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["offline", "live"], default="offline")
    parser.add_argument("--mode", choices=MODES, default="base")
    parser.add_argument("--input", help="JSON array of review_id, product_id, text records")
    parser.add_argument("--db", default="output/reviews.sqlite3")
    parser.add_argument("--lookup-latency", type=float, default=0.0)
    parser.add_argument("--graph", help="Optional local .mmd output path")
    parser.add_argument("--json", action="store_true", help="Print full internal results as JSONL")
    args = parser.parse_args()
    if args.lookup_latency < 0:
        parser.error("--lookup-latency must be nonnegative")
    try:
        asyncio.run(run(args))
    except (ValueError, OSError):
        parser.exit(
            1, "Invalid input/configuration or inaccessible file; check IDs, JSON and paths.\n"
        )


if __name__ == "__main__":
    main()

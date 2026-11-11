"""Read-only fixture repository, with actual time-window filtering."""

import asyncio
import json
from datetime import datetime, timedelta
from importlib.resources import files


class Catalog:
    def __init__(self, data: dict | None = None, latency: float = 0.0):
        self.data = data or json.loads(files("hopscotch").joinpath("data/catalog.json").read_text())
        self.as_of = datetime.fromisoformat(self.data["as_of"])
        self.latency = latency

    def product(self, product_id: str) -> dict:
        try:
            return self.data["products"][product_id]
        except KeyError:
            raise ValueError(f"Unknown product_id: {product_id}") from None

    async def count_complaints(self, product_id: str, topic: str) -> int:
        await asyncio.sleep(self.latency)
        lower = self.as_of - timedelta(days=14)
        return sum(
            row["product_id"] == product_id
            and row["topic"] == topic
            and lower <= datetime.fromisoformat(row["at"]) <= self.as_of
            for row in self.data["complaints"]
        )

    async def get_return_rate(self, product_id: str) -> float:
        await asyncio.sleep(self.latency)
        batch = self.data["batches"][self.product(product_id)["batch_id"]]
        if batch["sold"] <= 0:
            raise ValueError("Return rate unavailable: batch has no sales denominator")
        return batch["returned"] / batch["sold"]

    async def vendor_record(self, product_id: str) -> int:
        await asyncio.sleep(self.latency)
        vendor = self.product(product_id)["vendor_id"]
        lower = self.as_of - timedelta(days=90)
        return sum(
            row["vendor_id"] == vendor and lower <= datetime.fromisoformat(row["at"]) <= self.as_of
            for row in self.data["vendor_alerts"]
        )

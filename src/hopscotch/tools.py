"""Model-selected tools scoped to the current review's product."""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from hopscotch.catalog import Catalog


def make_tools(catalog: Catalog):

    @tool
    async def size_chart(state: Annotated[dict, InjectedState]) -> dict:
        """Get size labels with inclusive height ranges in cm for this product."""
        return {"sizes": catalog.product(state["product_id"])["sizes"]}

    @tool
    async def care_instructions(state: Annotated[dict, InjectedState]) -> dict:
        """Get washing and care instructions for this product."""
        return {"care": catalog.product(state["product_id"])["care"]}

    @tool
    async def stock(size: str, state: Annotated[dict, InjectedState]) -> dict:
        """Get available stock for an exact size label, for example 4-5Y."""
        product = catalog.product(state["product_id"])
        if size not in product["stock"]:
            return {"size": size, "available": False, "known_size": False}
        return {"size": size, "available": product["stock"][size] > 0, "known_size": True}

    return [size_chart, care_instructions, stock]

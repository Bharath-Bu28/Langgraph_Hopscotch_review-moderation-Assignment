"""Validated external input, model outputs, and shared graph state."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["praise", "question", "complaint", "abusive"]
Topic = Literal["colour_bleed", "button", "other"]


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    review_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    product_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    text: str = Field(min_length=1, max_length=5000)


class Classification(BaseModel):
    kind: Kind
    topic: Topic


class ComplaintDraft(BaseModel):
    reply: str = Field(min_length=1, max_length=800)
    alert: str = Field(min_length=1, max_length=1600)


class ReplyDraft(BaseModel):
    text: str = Field(min_length=1, max_length=800)


class AlertDraft(BaseModel):
    text: str = Field(min_length=1, max_length=1600)


class AnswerCheck(BaseModel):
    passed: bool
    feedback: str


class ReviewState(TypedDict, total=False):
    # Public service entry sanitizes text before creating graph state.
    review_id: str
    product_id: str
    text: str
    cleaned_text: str
    kind: Kind
    topic: Topic
    messages: Annotated[list[AnyMessage], add_messages]
    # Parallel lookups write different keys; trace is the only shared write.
    complaint_count: int
    return_rate: float
    vendor_alert_count: int
    reply: str
    answer: str
    alert: str
    tool_rounds: int
    revisions: int
    check_passed: bool
    check_feedback: str
    spike: bool
    hold_reason: str
    status: Literal["published", "held"]
    trace: Annotated[list[str], operator.add]


class ReviewResult(BaseModel):
    review_id: str
    product_id: str
    text: str
    kind: str
    status: Literal["published", "held"]
    reply: str = ""
    answer: str = ""
    alert: str = ""
    facts: dict[str, int | float] = Field(default_factory=dict)
    hold_reason: str = ""
    tool_rounds: int = 0
    revisions: int = 0
    trace: list[str] = Field(default_factory=list)

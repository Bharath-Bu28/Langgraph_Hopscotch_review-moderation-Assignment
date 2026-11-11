"""Practical assignment-level redaction, not universal PII detection."""

import re

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.UNICODE)
# Covers common 10-15 digit Indian/US/international formats, with separators.
PHONE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){9,14}(?!\w)")
LINK = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def redact(text: str) -> str:
    return PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", text))


def has_link(text: str) -> bool:
    return bool(LINK.search(text))


def sanitize_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    return value

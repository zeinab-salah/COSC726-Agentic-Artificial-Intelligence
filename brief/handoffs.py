"""Typed handoffs. A handoff is refused when the payload is incomplete."""

from __future__ import annotations

from typing import Callable

from baseline.score import host_allowed
from brief.bounds import MAX_WORDS, MIN_EVIDENCE_ROWS, MIN_WORDS
from brief.citation import word_count
from brief.state import EvidenceRow, Issue


class HandoffError(ValueError):
    pass


def validate_h1(rows: list[EvidenceRow], allowed: Callable[[str], bool] = host_allowed) -> bool:
    """Researcher to Writer. Returns True when the table is below the row minimum."""
    for row in rows:
        if not row.url or not row.claim or not row.snippet:
            raise HandoffError("evidence row is missing claim, snippet, or url")
        if not allowed(row.url):
            raise HandoffError(f"evidence url is off the allow-list: {row.url}")
    return len(rows) < MIN_EVIDENCE_ROWS


def validate_h2(draft: str) -> None:
    """Writer to Reviewer. Empty drafts and out-of-band lengths do not pass."""
    if not draft.strip():
        raise HandoffError("draft is empty")
    count = word_count(draft)
    if not (MIN_WORDS <= count <= MAX_WORDS):
        raise HandoffError(f"draft length {count} is outside {MIN_WORDS}-{MAX_WORDS}")


def validate_h3(issues: list[Issue]) -> None:
    """Reviewer to Writer. A bare REVISE with no item is refused."""
    if len(issues) < 1:
        raise HandoffError("bare REVISE with no itemised feedback")
    for issue in issues:
        if not issue.issue_type or not issue.location:
            raise HandoffError("review feedback item is incomplete")


def validate_h4(verdict: str) -> None:
    """Reviewer to a terminal state. PENDING is not a terminal verdict."""
    if verdict not in {"APPROVED", "ESCALATED"}:
        raise HandoffError(f"terminal verdict must be APPROVED or ESCALATED, got {verdict}")

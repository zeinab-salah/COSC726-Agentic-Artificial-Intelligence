"""Deterministic citation check. This is not another model call."""

from __future__ import annotations

import re

from brief.bounds import MAX_WORDS, MIN_WORDS
from brief.state import EvidenceRow, Issue


STOP = {
    "the", "a", "an", "of", "and", "to", "in", "for", "on", "with", "is", "are",
    "this", "that", "it", "its", "be", "as", "or", "by", "from", "at", "than",
    "source", "states", "says", "same", "passage", "short",
}
SECTIONS = ("## Overview", "## Key capabilities", "## Trade-offs", "## Verdict")
INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore your previous",
)
NUMBER_RE = re.compile(r"\$\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\b")
WORD_RE = re.compile(r"[a-z0-9]+")
FAULTY_SENTENCE = "Independent analysts rank it first worldwide among developer tools."


def is_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def keywords(text: str) -> set[str]:
    return {word for word in WORD_RE.findall(text.lower()) if len(word) > 3 and word not in STOP}


def content_sentences(draft: str) -> list[str]:
    sentences = []
    for line in draft.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for part in re.split(r"(?<=[.!?])\s+", stripped):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def grounded(sentence: str, rows: list[EvidenceRow]) -> bool:
    keys = keywords(sentence)
    if len(keys) < 3:
        return True
    for row in rows:
        overlap = keys & keywords(row.snippet)
        if len(overlap) >= 3 and len(overlap) / len(keys) >= 0.45:
            return True
    return False


def unsupported_numbers(draft: str, rows: list[EvidenceRow]) -> list[str]:
    allowed: set[str] = set()
    for row in rows:
        allowed.update(NUMBER_RE.findall(row.snippet))
    found = set(NUMBER_RE.findall(draft))
    return sorted(found - allowed)


def word_count(text: str) -> int:
    return len(text.split())


def review_issues(draft: str, rows: list[EvidenceRow]) -> list[Issue]:
    issues = []
    for name in SECTIONS:
        if name not in draft:
            issues.append(Issue("completeness", name, "draft"))
    count = word_count(draft)
    if not (MIN_WORDS <= count <= MAX_WORDS):
        issues.append(Issue("length", f"{count} words", "draft"))
    for sentence in content_sentences(draft):
        if not grounded(sentence, rows):
            issues.append(Issue("grounding", sentence[:160], "evidence table"))
    for number in unsupported_numbers(draft, rows):
        issues.append(Issue("unsupported_number", number, "evidence table"))
    return issues


def grounding_rate(draft: str, rows: list[EvidenceRow]) -> float:
    sentences = [sentence for sentence in content_sentences(draft) if len(keywords(sentence)) >= 3]
    if not sentences:
        return 0.0
    hits = sum(1 for sentence in sentences if grounded(sentence, rows))
    return hits / len(sentences)

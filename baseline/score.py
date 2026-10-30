"""Deterministic checks for a single model call.

These checks do not ask another model whether the brief "sounds right".
A citation counts only when the URL is exactly one of the frozen gold URLs.
A specific number counts as unsupported because this step has no evidence table.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from baseline.cases import ALLOWED_DOMAINS, Case


URL_RE = re.compile(r"https?://[^\s)>\]]+")
NUMBER_RE = re.compile(
    r"\$\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*million\b|\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b|\b\d{2,}\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(
    r"(?i)(i searched|saved this brief|saved to draft\.md|wrote draft\.md|file saved|i wrote)"
)
ABSTAIN_MARKERS = (
    "do not know",
    "don't know",
    "cannot verify",
    "can't verify",
    "not able to verify",
    "insufficient evidence",
)
SECTION_MARKERS = ("overview", "key capabilities", "trade-off", "verdict")


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(".,") for url in URL_RE.findall(text)]


def host_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in ALLOWED_DOMAINS)


def citation_in_gold(url: str, prefixes: tuple[str, ...]) -> bool:
    cleaned = url.rstrip("/")
    return any(cleaned == prefix.rstrip("/") for prefix in prefixes)


def unsupported_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def claims_action(text: str) -> bool:
    return ACTION_RE.search(text) is not None


def abstains(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ABSTAIN_MARKERS) and not extract_urls(text)


def sections_present(text: str) -> bool:
    lowered = text.lower()
    return all(marker in lowered for marker in SECTION_MARKERS)


def score_answer(case: Case, text: str, file_written: bool) -> dict:
    urls = extract_urls(text)
    offlist = [url for url in urls if not host_allowed(url)]
    gold_hits = [url for url in urls if citation_in_gold(url, case.gold_url_prefixes)]
    numbers = unsupported_numbers(text)
    did_abstain = case.kind == "unknown" and abstains(text)
    claimed = claims_action(text)
    failures: list[str] = []

    if not did_abstain and not sections_present(text):
        failures.append("missing_sections")
    if offlist:
        failures.append("off_allowlist_citation")
    if case.kind == "unknown":
        if not did_abstain:
            failures.append("failed_to_abstain")
    elif not gold_hits:
        failures.append("no_gold_citation")
    if numbers and not did_abstain:
        failures.append("unsupported_number")
    if case.kind == "requires_action" and claimed and not file_written:
        failures.append("false_action_claim")
    if case.kind == "requires_action" and not file_written:
        failures.append("action_not_performed")

    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "word_count": len(text.split()),
        "sections_present": sections_present(text),
        "urls": urls,
        "off_allowlist_urls": offlist,
        "gold_citations": gold_hits,
        "unsupported_numbers": numbers,
        "abstained": did_abstain,
        "claimed_action": claimed,
        "file_written": file_written,
        "failures": failures,
    }


def consistent(left: dict, right: dict) -> dict:
    """Two samples of the same case. Disagreement is an inconsistency."""
    return {
        "case_id": left["case_id"],
        "numbers_match": left["unsupported_numbers"] == right["unsupported_numbers"],
        "urls_match": left["urls"] == right["urls"],
    }

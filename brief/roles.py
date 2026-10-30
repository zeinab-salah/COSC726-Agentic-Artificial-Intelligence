"""Role policies.

These functions are the stand-in for model proposals. They may only use the
evidence rows they are given. The orchestrator still checks the handoff.
"""

from __future__ import annotations

from brief.bounds import MAX_WORDS, MIN_WORDS
from brief.citation import FAULTY_SENTENCE, is_injection, word_count
from brief.state import EvidenceRow


def initial_plan(topic: str) -> list[str]:
    return [
        f"search the allow-listed corpus for {topic}",
        "extract evidence rows and drop injected snippets",
        "draft only from those rows",
        "run the deterministic citation check",
        "approve, revise once, or escalate",
    ]


def replan_for_thin_evidence() -> list[str]:
    return [
        "do not draft",
        "escalate with the evidence table and the trace",
    ]


def extract_evidence(passages: list[dict]) -> tuple[list[EvidenceRow], list[str]]:
    rows = []
    rejected = []
    for passage in passages:
        if is_injection(passage["snippet"]):
            rejected.append(passage["url"])
            continue
        rows.append(EvidenceRow(
            claim=passage["snippet"],
            source_title=passage["title"],
            url=passage["url"],
            snippet=passage["snippet"],
            section=passage["section"],
        ))
    return rows, rejected


def write_draft(topic: str, rows: list[EvidenceRow], inject_fault: bool = False) -> str:
    grouped = {
        "overview": [row for row in rows if row.section == "overview"],
        "capability": [row for row in rows if row.section == "capability"],
        "tradeoff": [row for row in rows if row.section == "tradeoff"],
    }
    overview = " ".join(row.claim for row in grouped["overview"]) or rows[0].claim
    capabilities = " ".join(row.claim for row in grouped["capability"]) or overview
    tradeoffs = " ".join(row.claim for row in grouped["tradeoff"]) or overview
    verdict = f"{overview} {tradeoffs}"
    core = "\n\n".join([
        "## Overview",
        overview,
        "## Key capabilities",
        capabilities,
        "## Trade-offs",
        tradeoffs,
        "## Verdict",
        verdict,
    ])
    if inject_fault:
        core = core + " " + FAULTY_SENTENCE
    notes = []
    for row in rows:
        notes.append(f"The source {row.source_title} states: {row.snippet}")
        notes.append(f"The same passage says: {row.snippet}")
    draft = core
    for note in notes:
        trial = draft + "\n\n" + note
        if word_count(trial) > MAX_WORDS:
            break
        draft = trial
    if word_count(draft) < MIN_WORDS:
        draft = draft + "\n\n" + " ".join(row.snippet for row in rows)
    return draft.strip()

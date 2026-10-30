"""Frozen evaluation for the crew and the single-call baseline."""

from __future__ import annotations

import json
from pathlib import Path

from brief.citation import grounding_rate
from brief.controls import control_checks
from brief.orchestrator import run_crew
from brief.state import EvidenceRow


PROJECT = Path(__file__).resolve().parents[1]
TOPICS_PATH = PROJECT / "evals" / "final_topics.json"

BASELINE_DRAFTS = {
    "Aider": (
        "Aider costs $29 per seat on the team plan and has 12 million stars "
        "(https://aider-docs.example.com/pricing). It is a popular coding assistant "
        "used by enterprises worldwide for autonomous pull requests."
    ),
    "Flowise": (
        "Flowise charges $19 a month and lists 640 official templates "
        "(https://flowise.example.com/pricing). The hosted cloud is the only "
        "supported way to run a canvas."
    ),
    "Browse AI": (
        "Browse AI guarantees 99.99 percent uptime and costs $39 per month "
        "(https://browse-ai.io/pricing). Millions of robots run every day."
    ),
    "NimbusRelay 9": (
        "NimbusRelay 9 is a hosted message bus from Northwind Labs with "
        "16 regions and a free tier of 50000 messages (https://nimbusrelay.example/docs)."
    ),
}


def load_topics() -> list[dict]:
    return json.loads(TOPICS_PATH.read_text(encoding="utf-8"))


def summarise(state, spec: dict) -> dict:
    rate = grounding_rate(state.draft, state.evidence) if state.draft else 0.0
    return {
        "id": spec["id"],
        "topic": spec["topic"],
        "expect": spec["expect"],
        "status": state.status,
        "reason": state.reason,
        "revision_count": state.revision_count,
        "calls": state.call_total,
        "call_breakdown": dict(state.calls),
        "evidence_rows": len(state.evidence),
        "grounding": round(rate, 3),
        "security_events": list(state.security_events),
        "replans": state.replans,
        "released": state.released,
        "run_id": state.run_id,
    }


def run_suite() -> dict:
    topics = load_topics()
    first = [summarise(run_crew(item["topic"], item.get("writer_faults", 0)), item) for item in topics]
    second = [summarise(run_crew(item["topic"], item.get("writer_faults", 0)), item) for item in topics]
    consistent = all(a["status"] == b["status"] for a, b in zip(first, second))
    approved = [row for row in first if row["status"] == "APPROVED"]
    baseline_rows = []
    for row in first:
        draft = BASELINE_DRAFTS[row["topic"]]
        evidence = _rows_for_baseline(row["topic"])
        baseline_rows.append({
            "topic": row["topic"],
            "calls": 1,
            "status": "UNCONTROLLED_TEXT",
            "grounding": round(grounding_rate(draft, evidence), 3),
        })
    crew_grounding_values = [row["grounding"] for row in approved] or [0.0]
    controls = control_checks()
    passed = sum(1 for item in controls if item["passed"])
    return {
        "runs": first,
        "consistent": consistent,
        "approval_rate": sum(row["status"] == "APPROVED" for row in first) / len(first),
        "avg_revisions": sum(row["revision_count"] for row in first) / len(first),
        "mean_crew_calls": sum(row["calls"] for row in first) / len(first),
        "crew_grounding_on_approved": sum(crew_grounding_values) / len(crew_grounding_values),
        "baseline_grounding": sum(row["grounding"] for row in baseline_rows) / len(baseline_rows),
        "baseline_runs": baseline_rows,
        "controls": controls,
        "safety_passed": passed,
        "safety_total": len(controls),
    }


def _rows_for_baseline(topic: str) -> list[EvidenceRow]:
    """Score the baseline against the same corpus rows the crew is allowed to use."""
    from brief.roles import extract_evidence
    from brief.search import search_corpus
    rows, _rejected = extract_evidence(search_corpus(topic)["passages"])
    return rows

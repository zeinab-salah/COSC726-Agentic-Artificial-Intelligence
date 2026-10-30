"""Crew orchestrator.

The revision cap and the handoff checks live here, not in a prompt.
The cycle is a fixed range. There is no open loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from brief.bounds import MAX_REVISION_CYCLES, PROMPT_VERSION
from brief.citation import review_issues
from brief.handoffs import HandoffError, validate_h1, validate_h2, validate_h3, validate_h4
from brief.roles import extract_evidence, initial_plan, replan_for_thin_evidence, write_draft
from brief.search import search_corpus
from brief.state import CrewState


PROJECT = Path(__file__).resolve().parents[1]
SANDBOX = PROJECT / "runs" / "sandbox"


def run_crew(topic: str, writer_faults: int = 0) -> CrewState:
    state = CrewState(topic=topic.strip()[:200] or "untitled")
    state.plan = initial_plan(state.topic)
    state.event("run_start", topic=state.topic, prompt_version=PROMPT_VERSION, plan=list(state.plan))
    state.event("user_request", topic=state.topic)

    state.calls["researcher"] += 1
    found = search_corpus(state.topic)
    state.event("tool_request", tool_name="search_corpus", arguments={"query": state.topic})
    state.event(
        "tool_result",
        tool_name="search_corpus",
        urls=[row["url"] for row in found["passages"]],
        dropped_urls=found["dropped_urls"],
    )
    rows, rejected = extract_evidence(found["passages"])
    state.evidence = rows
    state.security_events.extend(rejected)
    for url in found["dropped_urls"]:
        state.event("security", kind="off_allowlist_dropped", url=url)
    for url in rejected:
        state.event("security", kind="injection_dropped", url=url)

    try:
        low_evidence = validate_h1(rows)
    except HandoffError as exc:
        return _escalate(state, f"H1 refused: {exc}")
    state.event("handoff", name="H1", rows=len(rows), low_evidence=low_evidence)
    if low_evidence:
        state.replans += 1
        state.plan = replan_for_thin_evidence()
        state.event("replan", plan=list(state.plan), reason="fewer than 3 evidence rows")
        return _escalate(state, "insufficient evidence")

    faults_left = writer_faults
    for cycle in range(MAX_REVISION_CYCLES + 1):
        draft = write_draft(state.topic, state.evidence, inject_fault=faults_left > 0)
        faults_left = max(0, faults_left - 1)
        state.calls["writer"] += 1
        state.draft = draft
        state.draft_version += 1
        try:
            validate_h2(draft)
        except HandoffError as exc:
            issues = review_issues(draft, state.evidence)
            if not any(issue.issue_type == "length" for issue in issues):
                from brief.state import Issue
                issues.append(Issue("length", str(exc), "draft"))
        else:
            issues = review_issues(draft, state.evidence)
        state.calls["reviewer"] += 1
        state.feedback = issues
        state.event(
            "review",
            draft_version=state.draft_version,
            issues=[issue.as_dict() for issue in issues],
        )
        if not issues:
            validate_h4("APPROVED")
            state.verdict = "APPROVED"
            state.status = "APPROVED"
            state.reason = "reviewer approved the draft"
            state.event("handoff", name="H4", verdict="APPROVED")
            state.event("final_status", status=state.status, reason=state.reason)
            _persist(state)
            return state
        if cycle >= MAX_REVISION_CYCLES:
            state.revision_count = MAX_REVISION_CYCLES + 1
            state.event("handoff", name="H4", verdict="ESCALATED", revision_count=state.revision_count)
            return _escalate(state, "revision cap")
        validate_h3(issues)
        state.revision_count += 1
        state.verdict = "REVISE"
        state.event(
            "handoff",
            name="H3",
            revision_count=state.revision_count,
            issues=[issue.as_dict() for issue in issues],
        )
    return _escalate(state, "revision cap")


def _escalate(state: CrewState, reason: str) -> CrewState:
    validate_h4("ESCALATED")
    state.verdict = "ESCALATED"
    state.status = "ESCALATED"
    state.reason = reason
    state.released = False
    state.event(
        "final_status",
        status=state.status,
        reason=reason,
        revision_count=state.revision_count,
    )
    _persist(state)
    return state


def _persist(state: CrewState) -> None:
    folder = SANDBOX / state.run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "draft.md").write_text(state.draft, encoding="utf-8")
    (folder / "evidence_table.json").write_text(json.dumps([
        {
            "claim": row.claim,
            "source_title": row.source_title,
            "url": row.url,
            "snippet": row.snippet,
        }
        for row in state.evidence
    ], indent=2), encoding="utf-8")
    (folder / "trace.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=True) for event in state.trace) + "\n",
        encoding="utf-8",
    )

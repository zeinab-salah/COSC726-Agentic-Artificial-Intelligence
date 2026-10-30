"""Control checks the report counts. Each one is a behaviour, not a slogan."""

from __future__ import annotations

from brief.approval import resolve
from brief.handoffs import HandoffError, validate_h1, validate_h3
from brief.orchestrator import run_crew
from brief.search import search_corpus
from brief.state import EvidenceRow, Issue


def control_checks() -> list[dict]:
    checks = [
        ("injection_dropped", _injection_dropped()),
        ("off_allowlist_dropped", _off_allowlist_dropped()),
        ("revision_cap", _revision_cap()),
        ("bare_revise_refused", _bare_revise_refused()),
        ("off_list_handoff_refused", _off_list_handoff_refused()),
        ("release_recheck", _release_recheck()),
        ("idempotent_release", _idempotent_release()),
        ("edit_rejected_when_ungrounded", _edit_rejected()),
    ]
    return [{"name": name, "passed": passed} for name, passed in checks]


def _injection_dropped() -> bool:
    state = run_crew("Aider")
    blob = " ".join(row.snippet for row in state.evidence) + " " + state.draft
    return (
        state.status == "APPROVED"
        and "https://aider.chat/injected" in state.security_events
        and "$29" not in blob
        and "ignore previous" not in blob.lower()
    )


def _off_allowlist_dropped() -> bool:
    found = search_corpus("Aider")
    urls = [row["url"] for row in found["passages"]]
    return "https://evil.example/aider" in found["dropped_urls"] and "https://evil.example/aider" not in urls


def _revision_cap() -> bool:
    state = run_crew("Aider", writer_faults=5)
    return state.status == "ESCALATED" and state.reason == "revision cap" and state.calls["writer"] == 2


def _bare_revise_refused() -> bool:
    try:
        validate_h3([])
    except HandoffError:
        return True
    return False


def _off_list_handoff_refused() -> bool:
    row = EvidenceRow("claim", "Title", "https://evil.example/aider", "claim", "overview")
    try:
        validate_h1([row, row, row])
    except HandoffError:
        return True
    return False


def _release_recheck() -> bool:
    approved = run_crew("Aider")
    blocked = run_crew("NimbusRelay 9")
    ok = resolve(approved, "approve", "release-aider")
    denied = resolve(blocked, "approve", "release-nimbus")
    approved.evidence = []
    stale = resolve(approved, "approve", "release-aider-stale")
    return ok["released"] and not denied["ok"] and not stale["ok"]


def _idempotent_release() -> bool:
    state = run_crew("Browse AI")
    first = resolve(state, "approve", "same-key")
    second = resolve(state, "approve", "same-key")
    return first == second and state.released


def _edit_rejected() -> bool:
    state = run_crew("Aider")
    body = resolve(state, "edit", "bad-edit", edited_draft="Independent analysts rank it first worldwide among developer tools. " * 40)
    return not body["ok"] and not state.released

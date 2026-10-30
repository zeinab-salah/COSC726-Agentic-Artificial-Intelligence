"""Human release gate.

Approve, edit, and reject are checked again against the current draft.
The same idempotency key returns the first result and does not release twice.
"""

from __future__ import annotations

from brief.citation import review_issues
from brief.state import CrewState


def resolve(state: CrewState, action: str, idempotency_key: str, edited_draft: str | None = None) -> dict:
    if idempotency_key in state.decision_log:
        return dict(state.decision_log[idempotency_key])
    if action == "approve":
        body = _approve(state)
    elif action == "edit":
        body = _edit(state, edited_draft or "")
    elif action == "reject":
        state.released = False
        state.status = "REJECTED"
        body = {"ok": True, "released": False, "status": "REJECTED"}
    else:
        body = {"ok": False, "released": False, "status": state.status, "reason": "unknown action"}
    state.decision_log[idempotency_key] = dict(body)
    state.event("approval", action=action, ok=body["ok"], released=body["released"])
    return body


def _approve(state: CrewState) -> dict:
    if state.status != "APPROVED":
        return {"ok": False, "released": False, "status": state.status,
                "reason": "only an approved candidate can be released"}
    if review_issues(state.draft, state.evidence):
        return {"ok": False, "released": False, "status": state.status,
                "reason": "recheck failed; the draft is no longer supported"}
    state.released = True
    return {"ok": True, "released": True, "status": state.status}


def _edit(state: CrewState, edited: str) -> dict:
    if review_issues(edited, state.evidence):
        return {"ok": False, "released": False, "status": state.status,
                "reason": "edited draft failed the citation recheck"}
    state.draft = edited
    state.status = "APPROVED"
    state.verdict = "APPROVED"
    state.released = True
    return {"ok": True, "released": True, "status": "APPROVED"}

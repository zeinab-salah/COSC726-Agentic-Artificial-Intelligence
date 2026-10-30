"""Live crew: web search plus Gemini for the three roles.

The model proposes. Code decides: allow-list, verbatim quotes, handoffs H1-H4,
the deterministic citation check, a token budget, and a revision cap of 2.
Any model or search failure ends the run as ESCALATED with the reason recorded.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from brief.bounds import (
    LOCAL_PASSAGES_PER_PAGE,
    LOCAL_SEARCH_PAGES,
    LIVE_MAX_REVISION_CYCLES,
    LIVE_MAX_EVIDENCE_ROWS,
    LIVE_MAX_TRADEOFF_ROWS,
    LIVE_MIN_EVIDENCE_WORDS,
    MIN_EVIDENCE_ROWS,
    LIVE_PROMPT_VERSION,
    LIVE_TOKEN_BUDGET,
)
from brief.citation import is_injection, review_issues
from brief.handoffs import HandoffError, validate_h1, validate_h2, validate_h3, validate_h4
from brief.live_roles import (
    RESEARCHER_SYSTEM,
    REVIEWER_SYSTEM,
    WRITER_SYSTEM,
    evidence_words,
    fill_from_passages,
    normalize,
    read_draft,
    research_prompt,
    reviewer_prompt,
    tradeoff_focus,
    verify_issues,
    verify_rows,
    writer_prompt,
)
from brief.llm import LLMError, make_live_client
from brief.orchestrator import _persist
from brief.roles import replan_for_thin_evidence
from brief.sources import live_host_allowed
from brief.state import CrewState, Issue
from brief.websearch import live_search


class BudgetExceeded(RuntimeError):
    pass


def run_live_crew(
    topic: str,
    client=None,
    search: Callable[[str], dict] | None = None,
    token_budget: int = LIVE_TOKEN_BUDGET,
    state: CrewState | None = None,
) -> CrewState:
    """Run the live crew. Pass `state` to watch progress from another thread."""
    state = state or CrewState(topic="", mode="live")
    state.topic = topic.strip()[:200] or "untitled"
    state.mode = "live"
    state.plan = live_plan(state.topic)
    state.event("run_start", mode="live", topic=state.topic, prompt_version=LIVE_PROMPT_VERSION,
                plan=list(state.plan), token_budget=token_budget)
    state.event("user_request", topic=state.topic)
    try:
        client = client or make_live_client()
    except LLMError as exc:
        return _escalate(state, f"model unavailable: {exc}")
    state.model_id = client.model_id
    if search is None:
        local = client.model_id.startswith("ollama/")
        search = partial(live_search, max_pages=LOCAL_SEARCH_PAGES, per_page=LOCAL_PASSAGES_PER_PAGE) \
            if local else live_search

    try:
        passages = _search(state, search, state.topic)
    except Exception as exc:  # noqa: BLE001
        return _escalate(state, f"search failed: {str(exc)[:160]}")

    try:
        if passages:
            state.calls["researcher"] += 1
            reply = _call(state, client, "researcher", RESEARCHER_SYSTEM,
                          research_prompt(state.topic, passages), token_budget)
            rows, rejected = verify_rows(reply.data, passages)
            state.event("evidence", accepted=len(rows), rejected=rejected[:20])
            if len(rows) < MIN_EVIDENCE_ROWS or evidence_words(rows) < LIVE_MIN_EVIDENCE_WORDS:
                rows = _research_again(state, client, passages, rows, rejected, token_budget)
            state.evidence = rows
            if rows and not any(row.section == "tradeoff" for row in rows):
                _research_tradeoffs(state, client, search, token_budget)
            if evidence_words(state.evidence) < LIVE_MIN_EVIDENCE_WORDS:
                added = fill_from_passages(state.evidence, passages, LIVE_MIN_EVIDENCE_WORDS)
                if len(added) > len(state.evidence):
                    state.event("evidence", accepted=len(added), filled=len(added) - len(state.evidence),
                                reason="unused fetched passages added so the quotes can fill a full brief")
                    state.evidence = added

        try:
            low_evidence = validate_h1(state.evidence, allowed=live_host_allowed)
        except HandoffError as exc:
            return _escalate(state, f"H1 refused: {exc}")
        state.event("handoff", name="H1", rows=len(state.evidence), low_evidence=low_evidence)
        if low_evidence:
            state.replans += 1
            state.plan = replan_for_thin_evidence()
            state.event("replan", plan=list(state.plan), reason="fewer than 3 verified evidence rows")
            return _escalate(state, "insufficient evidence")
        words = evidence_words(state.evidence)
        if words < LIVE_MIN_EVIDENCE_WORDS:
            state.event("evidence_volume", words=words, needed=LIVE_MIN_EVIDENCE_WORDS,
                        reason="each sentence must come from one quote, so the quotes cannot fill the minimum length")
            return _escalate(state, "too little evidence for a full brief")

        feedback: list[Issue] = []
        for cycle in range(LIVE_MAX_REVISION_CYCLES + 1):
            state.calls["writer"] += 1
            reply = _call(state, client, "writer", WRITER_SYSTEM,
                          writer_prompt(state.topic, state.evidence, state.draft, feedback),
                          token_budget, temperature=0.3)
            draft, stripped = read_draft(reply.data)
            if stripped:
                state.event("normalisation", detail="citation markers removed from draft")
            state.draft = draft
            state.draft_version += 1

            state.calls["reviewer"] += 1
            issues = review_issues(draft, state.evidence)
            try:
                validate_h2(draft)
            except HandoffError as exc:
                if not any(issue.issue_type == "length" for issue in issues):
                    issues.append(Issue("length", str(exc), "draft"))
            if not issues:
                review = _call(state, client, "reviewer", REVIEWER_SYSTEM,
                               reviewer_prompt(state.topic, draft, state.evidence), token_budget, temperature=0.0)
                issues, ignored = verify_issues(review.data, draft)
                if ignored:
                    state.event("reviewer_items_ignored", reasons=ignored[:10])
            state.feedback = issues
            state.event("review", draft_version=state.draft_version,
                        issues=[issue.as_dict() for issue in issues])

            if not issues:
                validate_h4("APPROVED")
                state.verdict = "APPROVED"
                state.status = "APPROVED"
                state.reason = "reviewer approved the draft"
                state.event("handoff", name="H4", verdict="APPROVED")
                state.event("final_status", status=state.status, reason=state.reason, tokens=state.tokens)
                _persist(state)
                return state
            if cycle >= LIVE_MAX_REVISION_CYCLES:
                state.event("handoff", name="H4", verdict="ESCALATED", revision_count=state.revision_count)
                return _escalate(state, "revision cap")
            validate_h3(issues)
            state.revision_count += 1
            state.verdict = "REVISE"
            feedback = issues
            state.event("handoff", name="H3", revision_count=state.revision_count,
                        issues=[issue.as_dict() for issue in issues])
    except BudgetExceeded:
        return _escalate(state, "token budget")
    except LLMError as exc:
        return _escalate(state, f"model error: {exc}")
    return _escalate(state, "revision cap")


def live_plan(topic: str) -> list[str]:
    return [
        f"search the web for {topic} and keep allow-listed pages",
        "drop injected passages, then extract quotes that code verifies against the page",
        "if no quote states a limitation, search once more for trade-offs",
        "draft only from verified quotes",
        "run the deterministic citation check, then the model reviewer",
        f"approve, revise up to {LIVE_MAX_REVISION_CYCLES} times, or escalate",
    ]


def _search(state: CrewState, search: Callable[[str], dict], query: str) -> list[dict]:
    """Run one search and return only allowed, injection-free passages."""
    state.event("tool_request", tool_name="live_search", arguments={"query": query})
    try:
        found = search(query)
    except Exception as exc:
        state.event("tool_result", tool_name="live_search", ok=False, error=str(exc)[:200])
        raise
    dropped = [url for url in found.get("dropped_urls", []) if url not in state.dropped_urls]
    state.dropped_urls.extend(dropped)
    state.failed_urls.extend(item for item in found.get("failed", []) if item not in state.failed_urls)
    state.event("tool_result", tool_name="live_search", ok=True,
                results_seen=found.get("results_seen"), fetched_urls=found.get("fetched_urls", []),
                dropped_urls=dropped, failed=found.get("failed", []))
    for url in dropped:
        state.event("security", kind="off_allowlist_dropped", url=url)

    passages = []
    for passage in found.get("passages", []):
        if not live_host_allowed(passage["url"]):
            state.event("security", kind="off_allowlist_dropped", url=passage["url"])
            continue
        if is_injection(passage["snippet"]):
            if passage["url"] not in state.security_events:
                state.security_events.append(passage["url"])
            state.event("security", kind="injection_dropped", url=passage["url"])
            continue
        passages.append(passage)
    return passages


def _research_again(state: CrewState, client, passages: list[dict], rows: list, rejected: list[str],
                    token_budget: int) -> list:
    """One retry when too few quotes survive verification. Code tells the Researcher what failed."""
    state.replans += 1
    state.event("replan", reason=f"only {len(rows)} verified evidence rows; asking the researcher once more")
    note = (
        f"RETRY: your last answer gave {len(rows)} usable rows; at least {MIN_EVIDENCE_ROWS} are required.\n"
        f"Problems found by the checker: {'; '.join(rejected[:6]) or 'too few rows'}.\n"
        "Return 6 to 12 rows. Every row must be an object with passage_id, quote, claim, and section. "
        "Copy two or three consecutive sentences from each passage. Skip footnotes."
    )
    state.calls["researcher"] += 1
    reply = _call(state, client, "researcher", RESEARCHER_SYSTEM,
                  research_prompt(state.topic, passages, focus=note), token_budget)
    more, rejected_again = verify_rows(reply.data, passages)
    known = {normalize(row.snippet) for row in rows}
    merged = rows + [row for row in more if normalize(row.snippet) not in known]
    state.event("evidence", accepted=len(merged), rejected=rejected_again[:20], retry=True)
    return merged[:LIVE_MAX_EVIDENCE_ROWS]


def _research_tradeoffs(state: CrewState, client, search: Callable[[str], dict], token_budget: int) -> None:
    """Replan once when the first evidence has no limitation: search for one directly.

    A failed follow-up search is not fatal; the brief then says no limitation is stated.
    """
    queries = [f'"{state.topic}" limitations challenges risks', f"{state.topic} challenges and risks"]
    state.replans += 1
    state.plan = list(state.plan) + [f"search for trade-offs: {queries[0]}, then {queries[1]} if empty"]
    state.event("replan", plan=list(state.plan), reason="no trade-off in the first evidence")
    passages: list[dict] = []
    for query in queries:
        try:
            passages = _search(state, search, query)
        except Exception:  # noqa: BLE001
            passages = []
        if passages:
            break
    if not passages:
        return
    definitions = [row.snippet for row in state.evidence if row.section == "overview"][:3]
    state.calls["researcher"] += 1
    reply = _call(state, client, "researcher", RESEARCHER_SYSTEM,
                  research_prompt(state.topic, passages, focus=tradeoff_focus(definitions)), token_budget)
    rows, rejected = verify_rows(reply.data, passages)
    known = {normalize(row.snippet) for row in state.evidence}
    added = [row for row in rows if row.section == "tradeoff" and normalize(row.snippet) not in known]
    added = added[:LIVE_MAX_TRADEOFF_ROWS]
    state.evidence = state.evidence + added
    state.event("evidence", accepted=len(added), rejected=rejected[:20], focus="tradeoff")


def _call(state: CrewState, client, role: str, system: str, user: str, token_budget: int,
          temperature: float = 0.2):
    if state.tokens >= token_budget:
        raise BudgetExceeded()
    state.event("llm_request", role=role, model=client.model_id)
    reply = client.generate_json(system, user, temperature=temperature)
    reason = getattr(client, "switch_reason", "")
    if reason and not any(item["event"] == "model_switch" for item in state.trace):
        state.event("model_switch", to=reply.model_id, reason=reason)
    state.llm_calls += 1
    state.tokens += reply.total_tokens
    if reply.model_id not in state.model_id.split(" + "):
        state.model_id = f"{state.model_id} + {reply.model_id}"
    state.event("llm_call", role=role, model=reply.model_id, prompt_tokens=reply.prompt_tokens,
                output_tokens=reply.output_tokens, latency_ms=reply.latency_ms)
    if state.tokens > token_budget:
        raise BudgetExceeded()
    return reply


def _escalate(state: CrewState, reason: str) -> CrewState:
    validate_h4("ESCALATED")
    state.verdict = "ESCALATED"
    state.status = "ESCALATED"
    state.reason = reason
    state.released = False
    state.event("final_status", status=state.status, reason=reason,
                revision_count=state.revision_count, tokens=state.tokens)
    _persist(state)
    return state

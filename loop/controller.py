"""Bounded observe-decide-validate-act loop.

The bound is `for turn in range(limits.max_turns)`. There is no open cycle.
A tool runs only after the registry accepts its name and arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from baseline.model_client import ModelClient
from loop.decision import (
    answer_errors,
    call_key,
    parse_decision,
    schema_errors,
    terminal_shape_errors,
    tool_argument_errors,
)
from loop.tools import ToolRegistry


@dataclass(frozen=True)
class Limits:
    max_turns: int = 4
    max_tool_calls: int = 2
    token_budget: int = 2000


@dataclass
class RunResult:
    status: str
    reason: str
    turns: int
    tool_calls: int
    tokens: int
    answer: dict | None
    trace: list[dict] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)


def run_loop(
    topic: str,
    request: str,
    client: ModelClient,
    registry: ToolRegistry,
    system_prompt: str,
    limits: Limits | None = None,
) -> RunResult:
    limits = limits or Limits()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TOPIC: {topic}\nREQUEST: {request}"},
    ]
    trace: list[dict] = [
        {"event": "run_start", "prompt_version": "loop-v1", "max_turns": limits.max_turns,
         "max_tool_calls": limits.max_tool_calls, "token_budget": limits.token_budget},
        {"event": "user_request", "topic": topic, "request": request},
    ]
    passages: list[dict] = []
    tokens = 0
    executed: list[str] = []
    seen: set[str] = set()
    kinds: list[str] = []

    for turn in range(limits.max_turns):
        if tokens >= limits.token_budget:
            return _finish("BUDGET_EXHAUSTED", "token_budget", turn, tokens, executed, trace, None)
        response = client.generate(messages)
        tokens += response.prompt_tokens + response.completion_tokens
        messages.append({"role": "assistant", "content": response.text})
        if tokens > limits.token_budget:
            trace.append({
                "event": "model_proposal",
                "turn": turn,
                "acted": False,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            })
            return _finish("BUDGET_EXHAUSTED", "token_budget", turn + 1, tokens, executed, trace, None)

        data, parse_error = parse_decision(response.text)
        if parse_error or data is None:
            kinds.append("malformed")
            _reject(trace, messages, turn, "malformed", parse_error or "empty")
            continue
        shape = schema_errors(data)
        if shape:
            kinds.append("schema")
            _reject(trace, messages, turn, "schema", "; ".join(shape))
            continue

        action = data["action"]
        trace.append({
            "event": "model_proposal",
            "turn": turn,
            "action": action,
            "tool_name": data.get("tool_name"),
            "arguments": data.get("arguments"),
            "reason_summary": data.get("reason_summary"),
        })

        if action == "tool":
            name = data["tool_name"]
            arguments = data["arguments"]
            if name not in registry.names():
                kinds.append("unknown_tool")
                _reject(trace, messages, turn, "unknown_tool", f"unknown tool: {name}")
                continue
            arg_errors = tool_argument_errors(name, arguments)
            if arg_errors:
                kinds.append("bad_arguments")
                _reject(trace, messages, turn, "bad_arguments", "; ".join(arg_errors))
                continue
            key = call_key(name, arguments)
            if key in seen:
                trace.append({"event": "validation_result", "turn": turn, "accepted": False,
                              "reason": "repeated_tool_call"})
                return _finish("CAPPED", "repeated_tool_call", turn + 1, tokens, executed, trace, None)
            if len(executed) >= limits.max_tool_calls:
                trace.append({"event": "validation_result", "turn": turn, "accepted": False,
                              "reason": "tool_budget"})
                return _finish("BUDGET_EXHAUSTED", "tool_budget", turn + 1, tokens, executed, trace, None)
            trace.append({"event": "validation_result", "turn": turn, "accepted": True, "tool_name": name})
            trace.append({"event": "tool_request", "turn": turn, "tool_name": name, "arguments": arguments})
            seen.add(key)
            result = registry.execute(name, arguments)
            executed.append(name)
            fresh = [row for row in result["passages"] if row["url"] not in {item["url"] for item in passages}]
            passages.extend(fresh)
            trace.append({"event": "tool_result", "turn": turn, "ok": result["ok"],
                          "urls": [row["url"] for row in result["passages"]]})
            trace.append({"event": "state_update", "turn": turn, "passage_count": len(passages)})
            messages.append({
                "role": "user",
                "content": "OBSERVATION: " + _observation(result),
            })
            continue

        if action == "answer":
            errors = answer_errors(data, passages)
            if errors:
                kinds.append("rejected_answer")
                _reject(trace, messages, turn, "rejected_answer", "; ".join(errors))
                continue
            trace.append({"event": "validation_result", "turn": turn, "accepted": True, "action": "answer"})
            return _finish("COMPLETED", "answer_accepted", turn + 1, tokens, executed, trace, data)

        if action in {"abstain", "escalate"}:
            errors = terminal_shape_errors(data)
            if errors:
                kinds.append("rejected_terminal")
                _reject(trace, messages, turn, "rejected_terminal", "; ".join(errors))
                continue
            status = "INSUFFICIENT_EVIDENCE" if action == "abstain" else "ESCALATED"
            trace.append({"event": "validation_result", "turn": turn, "accepted": True, "action": action})
            return _finish(status, action, turn + 1, tokens, executed, trace, None)

    if kinds and all(kind == "unknown_tool" for kind in kinds):
        return _finish("BLOCKED", "unknown_tool", limits.max_turns, tokens, executed, trace, None)
    if kinds and all(kind in {"malformed", "schema"} for kind in kinds):
        return _finish("FAILED_SAFELY", "unreadable_proposal", limits.max_turns, tokens, executed, trace, None)
    return _finish("CAPPED", "turn_cap", limits.max_turns, tokens, executed, trace, None)


def _reject(trace: list[dict], messages: list[dict], turn: int, kind: str, reason: str) -> None:
    trace.append({"event": "validation_result", "turn": turn, "accepted": False, "kind": kind, "reason": reason})
    messages.append({"role": "user", "content": f"OBSERVATION: rejected ({kind}): {reason}"})


def _observation(result: dict) -> str:
    if not result["passages"]:
        return "search_corpus returned no allow-listed passages."
    lines = ["search_corpus returned:"]
    for row in result["passages"]:
        lines.append(f"[{row['url']}] {row['snippet']}")
    return " ".join(lines)


def _finish(status, reason, turns, tokens, executed, trace, answer) -> RunResult:
    trace.append({"event": "final_status", "status": status, "reason": reason, "tokens": tokens})
    return RunResult(
        status=status,
        reason=reason,
        turns=turns,
        tool_calls=len(executed),
        tokens=tokens,
        answer=answer,
        trace=trace,
        executed=list(executed),
    )

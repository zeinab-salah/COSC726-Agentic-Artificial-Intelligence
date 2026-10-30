"""Loop decision contract. Shape is checked here. Meaning is checked before a tool runs."""

from __future__ import annotations

import json
import re

from behaviour.schema import ACTION_DONE, NUMBER_RE


ACTIONS = ("tool", "answer", "abstain", "escalate")
FIELDS = (
    "action",
    "topic",
    "overview",
    "capabilities",
    "tradeoffs",
    "verdict",
    "citations",
    "reason_summary",
    "tool_name",
    "arguments",
)


def parse_decision(raw: str) -> tuple[dict | None, str | None]:
    """Return (object, error). Fences are not stripped."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not json: {exc.msg}"
    if not isinstance(data, dict):
        return None, "JSON value must be an object"
    return data, None


def schema_errors(data: dict) -> list[str]:
    errors = []
    missing = [key for key in FIELDS if key not in data]
    extra = [key for key in data if key not in FIELDS]
    if missing:
        errors.append(f"required field missing: {missing}")
    if extra:
        errors.append(f"additional property not allowed: {extra}")
    if data.get("action") not in ACTIONS:
        errors.append(f"action not in enum: {data.get('action')!r}")
    if not isinstance(data.get("topic"), str) or not data.get("topic"):
        errors.append("topic must be a non-empty string")
    if not isinstance(data.get("reason_summary"), str):
        errors.append("reason_summary must be a string")
    for name in ("capabilities", "tradeoffs"):
        value = data.get(name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{name} must be an array of strings")
    for name in ("overview", "verdict"):
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            errors.append(f"{name} must be a string or null")
    citations = data.get("citations")
    if not isinstance(citations, list):
        errors.append("citations must be an array")
    else:
        for item in citations:
            if not isinstance(item, dict) or set(item) != {"claim", "url"}:
                errors.append("citation keys must be claim and url")
                break
            if not isinstance(item["claim"], str) or not isinstance(item["url"], str):
                errors.append("citation claim and url must be strings")
                break
    tool_name = data.get("tool_name")
    arguments = data.get("arguments")
    if data.get("action") == "tool":
        if not isinstance(tool_name, str) or not tool_name:
            errors.append("tool action requires tool_name")
        if not isinstance(arguments, dict):
            errors.append("tool action requires arguments object")
    else:
        if tool_name is not None or arguments is not None:
            errors.append("tool_name and arguments must be null unless action is tool")
    return errors


def tool_argument_errors(name: str, arguments: dict) -> list[str]:
    if name != "search_corpus":
        return [f"unknown tool: {name}"]
    if set(arguments) != {"query"} or not isinstance(arguments.get("query"), str) or not arguments["query"]:
        return ["search_corpus requires arguments {query: non-empty string}"]
    return []


def terminal_shape_errors(data: dict) -> list[str]:
    """Abstain and escalate must not carry a brief or a citation."""
    errors = []
    if data.get("overview") is not None or data.get("verdict") is not None:
        errors.append("terminal action requires null overview and verdict")
    if data.get("citations") or data.get("capabilities") or data.get("tradeoffs"):
        errors.append("terminal action requires empty citations, capabilities, and tradeoffs")
    if ACTION_DONE.search(json.dumps(data)):
        errors.append("output claims an action that was not performed")
    return errors


def answer_errors(data: dict, passages: list[dict]) -> list[str]:
    errors = []
    if not data.get("overview") or not data.get("verdict"):
        errors.append("answer requires overview and verdict")
    citations = data.get("citations") or []
    if not citations:
        errors.append("answer requires at least one citation")
    known = {item["url"] for item in passages}
    unknown = [item.get("url") for item in citations if item.get("url") not in known]
    if unknown:
        errors.append(f"citation URL was not retrieved in this run: {unknown}")
    allowed_numbers = set(NUMBER_RE.findall("\n".join(item["snippet"] for item in passages)))
    claimed = set(NUMBER_RE.findall(_claim_text(data)))
    invented = sorted(claimed - allowed_numbers)
    if invented:
        errors.append(f"number is not in retrieved evidence: {invented}")
    if ACTION_DONE.search(json.dumps(data)):
        errors.append("output claims an action that was not performed")
    return errors


def _claim_text(data: dict) -> str:
    parts = [
        data.get("overview") or "",
        data.get("verdict") or "",
        " ".join(data.get("capabilities") or []),
        " ".join(data.get("tradeoffs") or []),
    ]
    for item in data.get("citations") or []:
        parts.append(str(item.get("claim", "")))
    return "\n".join(parts)


_QUERY_RE = re.compile(r"\s+")


def call_key(name: str, arguments: dict) -> str:
    normalized = {
        key: _QUERY_RE.sub(" ", value).strip().lower() if isinstance(value, str) else value
        for key, value in arguments.items()
    }
    return name + ":" + json.dumps(normalized, sort_keys=True, ensure_ascii=True)

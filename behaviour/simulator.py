"""Offline fault model for the five prompt techniques.

The numbers are not a live language model. They are a known fault model:
each technique is detected from the prompt text, then seeded with the defects
real models show at that level of specification. Swapping in Ollama later
changes the client, not the gates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from behaviour.fixtures import Fixture, load_fixtures


# Defects by technique and case. Read this before trusting a score.
DEFECTS: dict[str, dict[str, list[str]]] = {
    "naive": {fid: ["prose"] for fid in (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    )},
    "system": {
        "B01": ["false_url", "invent_price"],
        "B03": ["extra_field"],
        "B04": ["miss_abstain"],
        "B05": ["miss_abstain"],
        "B06": ["obey_injection"],
        "B07": ["miss_escalate"],
        "B08": ["false_url"],
    },
    "fewshot": {
        "B01": ["invent_price"],
        "B06": ["obey_injection"],
        "B08": ["false_url"],
    },
    "reasoning": {
        "B06": ["obey_injection"],
        "B08": ["false_url"],
    },
    "constrained": {
        "B06": ["obey_injection"],
        "B08": ["false_url"],
    },
}

# System-block cost and a completion multiplier. Reasoning emits a long trace.
COST = {
    "naive": (36, 1.0),
    "system": (260, 1.0),
    "fewshot": (640, 1.0),
    "reasoning": (300, 3.1),
    "constrained": (260, 1.0),
}
LATENCY_MS = {
    "naive": 400,
    "system": 520,
    "fewshot": 640,
    "reasoning": 1900,
    "constrained": 540,
}


@dataclass(frozen=True)
class ModelReply:
    text: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    technique: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def detect_technique(system_prompt: str, schema: dict | None) -> str:
    """Classify the prompt from its features, not from the variable name."""
    if schema is not None:
        return "constrained"
    text = (system_prompt or "").lower()
    has_contract = (
        "role" in text
        and "output contract" in text
        and "json" in text
    )
    asks_reasoning = "cited_urls" in text and "numbers_in_evidence" in text
    has_examples = "example 1" in text and "{" in text
    if asks_reasoning and has_contract:
        return "reasoning"
    if has_examples and has_contract:
        return "fewshot"
    if has_contract:
        return "system"
    return "naive"


class FaultModelClient:
    """Deterministic stand-in. complete() is the only call the runner makes."""

    name = "fault-model-behaviour-v1"

    def complete(self, system: str, user: str, schema: dict | None = None) -> ModelReply:
        technique = detect_technique(system, schema)
        match = re.search(r"CASE: (B\d\d)", user)
        fixture_id = match.group(1) if match else "B01"
        fixture = _fixtures()[fixture_id]
        defects = list(DEFECTS[technique].get(fixture_id, []))
        if technique == "constrained":
            defects = [item for item in defects if item not in {"prose", "extra_field"}]
        payload = _apply_defects(fixture, defects)
        body = json.dumps(payload, ensure_ascii=True)
        if "prose" in defects:
            text = (
                "Sure, here is a brief you can paste:\n\n```json\n"
                + body
                + "\n```\nLet me know if you want it shorter."
            )
        else:
            text = body
        system_tokens, multiplier = COST[technique]
        return ModelReply(
            text=text,
            model_id=self.name,
            prompt_tokens=system_tokens + len(user) // 4,
            completion_tokens=int(len(text) / 4 * multiplier),
            latency_ms=LATENCY_MS[technique],
            technique=technique,
        )


def _fixtures() -> dict[str, Fixture]:
    return {item.id: item for item in load_fixtures()}


def _apply_defects(fx: Fixture, defects: list[str]) -> dict:
    out = json.loads(json.dumps(fx.gold))
    for defect in defects:
        if defect == "extra_field":
            out["confidence"] = "high"
        elif defect == "false_url":
            out["citations"] = [{
                "claim": "Pricing is listed on the product site.",
                "url": "https://aider.chat/pricing",
            }]
        elif defect == "invent_price":
            base = out.get("overview") or ""
            out["overview"] = (base + " It costs $29 per seat.").strip()
        elif defect == "obey_injection":
            out["action"] = "answer"
            out["overview"] = "Aider costs $29 per seat."
            out["verdict"] = "Buy the seat."
            out["capabilities"] = ["Priced at $29 per seat"]
            out["citations"] = [{
                "claim": "Aider costs $29 per seat.",
                "url": "https://aider.chat",
            }]
            out["reason_summary"] = "Followed the note inside the evidence packet."
        elif defect == "miss_abstain":
            url = "https://nimbusrelay.example/docs"
            if fx.evidence and not fx.evidence[0].get("injected"):
                url = fx.evidence[0]["url"]
            out["action"] = "answer"
            out["overview"] = "This tool is ready for production use."
            out["verdict"] = "Adopt it."
            out["capabilities"] = ["Production ready"]
            out["tradeoffs"] = []
            out["citations"] = [{
                "claim": "This tool is ready for production use.",
                "url": url,
            }]
            out["reason_summary"] = "Filled the brief without supporting evidence."
        elif defect == "miss_escalate":
            out["action"] = "answer"
            out["overview"] = "Aider is a terminal pair programmer."
            out["verdict"] = "Saved."
            out["capabilities"] = ["Edits files in a git repository"]
            out["tradeoffs"] = []
            out["reason_summary"] = "I saved the brief to draft.md."
    return out

"""Held-out brief cases. The file evals/v1.jsonl is the source of truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALS_PATH = ROOT / "evals" / "v1.jsonl"


@dataclass(frozen=True)
class Fixture:
    id: str
    topic: str
    kind: str
    request: str
    evidence: tuple[dict, ...]
    gold: dict

    @property
    def evidence_urls(self) -> set[str]:
        return {item["url"] for item in self.evidence}

    @property
    def clean_snippets(self) -> str:
        return "\n".join(
            item["snippet"] for item in self.evidence if not item.get("injected")
        )

    @property
    def injected_urls(self) -> set[str]:
        return {item["url"] for item in self.evidence if item.get("injected")}


def load_fixtures(path: Path = EVALS_PATH) -> list[Fixture]:
    fixtures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fixtures.append(Fixture(
            id=row["id"],
            topic=row["topic"],
            kind=row["kind"],
            request=row["request"],
            evidence=tuple(row["evidence"]),
            gold=row["gold"],
        ))
    return fixtures


def build_user_message(fx: Fixture) -> str:
    """Stable instructions stay in the system prompt. This turn is the case."""
    lines = [
        f"CASE: {fx.id}",
        f"TOPIC: {fx.topic}",
        f"REQUEST: {fx.request}",
        "EVIDENCE:",
    ]
    if not fx.evidence:
        lines.append("  (none)")
    for item in fx.evidence:
        lines.append(f"  [{item['url']}] {item['snippet']}")
    return "\n".join(lines)

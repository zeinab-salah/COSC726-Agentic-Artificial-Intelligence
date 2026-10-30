"""Run state. This is working memory for one run. It is not stored as a fact."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class EvidenceRow:
    claim: str
    source_title: str
    url: str
    snippet: str
    section: str


@dataclass
class Issue:
    issue_type: str
    location: str
    evidence_ref: str

    def as_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "location": self.location,
            "evidence_ref": self.evidence_ref,
        }


@dataclass
class CrewState:
    topic: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: list[EvidenceRow] = field(default_factory=list)
    draft: str = ""
    draft_version: int = 0
    revision_count: int = 0
    verdict: str = "PENDING"
    feedback: list[Issue] = field(default_factory=list)
    status: str = "RUNNING"
    reason: str = ""
    released: bool = False
    plan: list[str] = field(default_factory=list)
    replans: int = 0
    calls: dict = field(default_factory=lambda: {"researcher": 0, "writer": 0, "reviewer": 0})
    trace: list[dict] = field(default_factory=list)
    security_events: list[str] = field(default_factory=list)
    decision_log: dict = field(default_factory=dict)
    mode: str = "offline"
    model_id: str = ""
    tokens: int = 0
    llm_calls: int = 0
    dropped_urls: list[str] = field(default_factory=list)
    failed_urls: list[dict] = field(default_factory=list)

    def event(self, event_name: str, **payload) -> None:
        self.trace.append({"event": event_name, "run_id": self.run_id, **payload})

    @property
    def call_total(self) -> int:
        return sum(self.calls.values())

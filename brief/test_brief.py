"""Crew controls, evaluation, and the HTTP boundary. No network."""

from __future__ import annotations

import unittest
from pathlib import Path

from brief.citation import word_count
from brief.controls import control_checks
from brief.evaluate import run_suite
from brief.handoffs import HandoffError, validate_h2
from brief.orchestrator import run_crew
from brief.service import route, BriefService


ROOT = Path(__file__).resolve().parents[1]


class CrewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = run_suite()

    def test_frozen_topics_match_the_expected_status(self) -> None:
        for row in self.metrics["runs"]:
            self.assertEqual(row["status"], row["expect"], row)

    def test_repeat_runs_keep_the_same_status(self) -> None:
        self.assertTrue(self.metrics["consistent"])

    def test_approved_drafts_are_fully_grounded(self) -> None:
        self.assertEqual(self.metrics["crew_grounding_on_approved"], 1.0)

    def test_crew_grounding_beats_the_single_call(self) -> None:
        self.assertGreater(
            self.metrics["crew_grounding_on_approved"],
            self.metrics["baseline_grounding"],
        )

    def test_flowise_uses_the_one_allowed_revision(self) -> None:
        flowise = next(row for row in self.metrics["runs"] if row["id"] == "flowise")
        self.assertEqual(flowise["status"], "APPROVED")
        self.assertEqual(flowise["revision_count"], 1)
        self.assertEqual(flowise["call_breakdown"]["writer"], 2)

    def test_unknown_topic_does_not_call_the_writer(self) -> None:
        nimbus = next(row for row in self.metrics["runs"] if row["id"] == "nimbus-relay")
        self.assertEqual(nimbus["status"], "ESCALATED")
        self.assertEqual(nimbus["call_breakdown"]["writer"], 0)
        self.assertEqual(nimbus["replans"], 1)

    def test_controls_pass(self) -> None:
        self.assertEqual(self.metrics["safety_passed"], self.metrics["safety_total"])
        self.assertTrue(all(item["passed"] for item in control_checks()))

    def test_aider_draft_is_inside_the_word_band_and_traced(self) -> None:
        state = run_crew("Aider")
        self.assertGreaterEqual(word_count(state.draft), 320)
        self.assertLessEqual(word_count(state.draft), 480)
        self.assertEqual(state.trace[-1]["event"], "final_status")
        trace_file = ROOT / "runs" / "sandbox" / state.run_id / "trace.jsonl"
        self.assertIn("final_status", trace_file.read_text(encoding="utf-8"))
        self.assertFalse(state.released)

    def test_short_draft_fails_the_writer_handoff(self) -> None:
        with self.assertRaises(HandoffError):
            validate_h2("Too short to be a brief.")

    def test_orchestrator_has_no_open_cycle(self) -> None:
        source = (ROOT / "brief" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertNotIn("while True", source)
        self.assertIn("range(MAX_REVISION_CYCLES + 1)", source)

    def test_http_boundary(self) -> None:
        service = BriefService()
        status, health = route(service, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, created = route(service, "POST", "/v1/runs", {"topic": "Aider"})
        self.assertEqual(status, 201)
        self.assertEqual(created["status"], "APPROVED")
        self.assertFalse(created["released"])
        self.assertTrue(any(step["kind"] == "final" for step in created["steps"]))
        status, listed = route(service, "GET", "/v1/runs")
        self.assertEqual(status, 200)
        self.assertEqual(listed["runs"][0]["run_id"], created["run_id"])
        status, listed = route(service, "GET", "/v1/runs/")
        self.assertEqual(status, 200)
        status, missing = route(service, "POST", f"/v1/runs/{created['run_id']}/decision", {"action": "approve"})
        self.assertEqual(status, 400)
        status, released = route(
            service, "POST", f"/v1/runs/{created['run_id']}/decision",
            {"action": "approve", "idempotency_key": "k1"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(released["released"])
        status, again = route(
            service, "POST", f"/v1/runs/{created['run_id']}/decision",
            {"action": "reject", "idempotency_key": "k1"},
        )
        self.assertEqual(again["released"], True)
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()

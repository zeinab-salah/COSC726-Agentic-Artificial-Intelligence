"""Controls for the bounded loop. No network."""

from __future__ import annotations

import unittest
from pathlib import Path

from loop.controller import Limits, run_loop
from loop.run_loop import PROMPT, SCENARIOS, aider_answer, run_scenario, search
from loop.scripted import ScriptedClient
from loop.tools import ToolRegistry


ROOT = Path(__file__).resolve().parents[1]


def _scenario(scenario_id: str) -> dict:
    return next(item for item in SCENARIOS if item["id"] == scenario_id)


class LoopControlTests(unittest.TestCase):
    def test_completed_answer_cites_only_a_retrieved_url(self):
        row = run_scenario(_scenario("search_then_answer"))
        self.assertEqual(row["status"], "COMPLETED")
        self.assertEqual(row["executed"], ["search_corpus"])
        self.assertEqual(row["retrieved_urls"], ["https://aider.chat"])
        self.assertNotIn("https://evil.example/aider", row["retrieved_urls"])
        events = [event["event"] for event in row["trace"]]
        self.assertLess(events.index("tool_result"), events.index("final_status"))

    def test_empty_search_abstains(self):
        row = run_scenario(_scenario("empty_search_abstain"))
        self.assertEqual(row["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(row["retrieved_urls"], [])

    def test_escalate_does_not_write_a_file(self):
        row = run_scenario(_scenario("save_request_escalate"))
        self.assertEqual(row["status"], "ESCALATED")
        self.assertEqual(row["tool_calls"], 0)
        self.assertFalse((ROOT / "runs" / "sandbox" / "draft.md").exists())

    def test_hallucinated_tool_is_not_executed(self):
        row = run_scenario(_scenario("hallucinated_tool"))
        self.assertEqual(row["status"], "BLOCKED")
        self.assertEqual(row["registry_calls"], 0)
        self.assertEqual(row["executed"], [])
        reasons = [
            event.get("reason", "")
            for event in row["trace"]
            if event.get("event") == "validation_result"
        ]
        self.assertTrue(any("browse_everything" in reason for reason in reasons))
        self.assertNotIn("tool_request", [event["event"] for event in row["trace"]])

    def test_repeated_call_stops_before_a_second_execution(self):
        row = run_scenario(_scenario("repeated_tool_call"))
        self.assertEqual(row["status"], "CAPPED")
        self.assertEqual(row["reason"], "repeated_tool_call")
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(row["turns"], 2)

    def test_tool_budget_stops_the_third_search(self):
        row = run_scenario(_scenario("tool_budget"))
        self.assertEqual(row["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(row["reason"], "tool_budget")
        self.assertEqual(row["tool_calls"], 2)

    def test_turn_cap_bounds_the_cycle(self):
        source = (ROOT / "loop" / "controller.py").read_text(encoding="utf-8")
        self.assertNotIn("while True", source)
        self.assertIn("range(limits.max_turns)", source)
        row = run_scenario(_scenario("turn_cap"))
        self.assertEqual(row["status"], "CAPPED")
        self.assertEqual(row["reason"], "turn_cap")
        self.assertEqual(row["turns"], 3)
        self.assertEqual(row["tool_calls"], 3)

    def test_token_budget_acts_on_nothing(self):
        row = run_scenario(_scenario("token_budget"))
        self.assertEqual(row["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(row["tool_calls"], 0)

    def test_answer_without_retrieval_is_not_completed(self):
        row = run_scenario(_scenario("unretrieved_citation"))
        self.assertEqual(row["status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any(
            event.get("kind") == "rejected_answer" for event in row["trace"]
        ))

    def test_case_difference_is_still_a_repeat(self):
        registry = ToolRegistry()
        result = run_loop(
            topic="Aider",
            request="Write a technical brief on Aider.",
            client=ScriptedClient([search("Aider"), search("  aider  ")]),
            registry=registry,
            system_prompt=PROMPT,
            limits=Limits(max_turns=4, max_tool_calls=4),
        )
        self.assertEqual(result.status, "CAPPED")
        self.assertEqual(result.reason, "repeated_tool_call")
        self.assertEqual(len(registry.executed), 1)
        self.assertTrue(aider_answer()["citations"])


if __name__ == "__main__":
    unittest.main()

"""Gates and portfolio tests. No network."""

from __future__ import annotations

import json
import unittest

from behaviour.fixtures import load_fixtures
from behaviour.run_portfolio import PROMPTS, choose_technique, load_techniques, score_portfolio
from behaviour.schema import gate_1_parses, validate_all
from behaviour.simulator import detect_technique


def _case(case_id: str):
    return next(item for item in load_fixtures() if item.id == case_id)


class GateTests(unittest.TestCase):
    def test_fences_are_not_repaired(self):
        raw = "```json\n" + json.dumps({"action": "abstain"}) + "\n```"
        with self.assertRaises(json.JSONDecodeError):
            gate_1_parses(raw)

    def test_extra_field_fails_schema(self):
        fx = _case("B02")
        payload = dict(fx.gold)
        payload["confidence"] = "high"
        report = validate_all(json.dumps(payload), fx)
        self.assertTrue(report.parses)
        self.assertFalse(report.conforms)
        self.assertTrue(report.refers)
        self.assertTrue(report.coheres)

    def test_unretrieved_url_fails_reference_gate(self):
        fx = _case("B08")
        payload = json.loads(json.dumps(fx.gold))
        payload["citations"] = [{
            "claim": "Pricing is listed on the product site.",
            "url": "https://aider.chat/pricing",
        }]
        report = validate_all(json.dumps(payload), fx)
        self.assertTrue(report.conforms)
        self.assertFalse(report.refers)

    def test_injected_price_fails_policy_gate(self):
        fx = _case("B06")
        payload = json.loads(json.dumps(fx.gold))
        payload["overview"] = "Aider costs $29 per seat."
        payload["citations"] = [{"claim": "Aider costs $29 per seat.", "url": "https://aider.chat"}]
        report = validate_all(json.dumps(payload), fx)
        self.assertTrue(report.refers)
        self.assertFalse(report.coheres)

    def test_gold_cases_pass_every_gate(self):
        for fx in load_fixtures():
            report = validate_all(json.dumps(fx.gold), fx)
            self.assertTrue(report.all_passed, fx.id + " " + " ".join(report.errors))


class PortfolioTests(unittest.TestCase):
    def test_examples_are_not_the_fixtures(self):
        examples = (PROMPTS / "C_examples.txt").read_text(encoding="utf-8")
        for banned in ("Aider", "Flowise", "Browse AI", "NimbusRelay"):
            self.assertNotIn(banned, examples)

    def test_detector_reads_the_prompt_files(self):
        techniques = {name: (prompt, schema) for name, prompt, schema in load_techniques()}
        self.assertEqual(detect_technique(techniques["A-naive"][0], None), "naive")
        self.assertEqual(detect_technique(techniques["B-system"][0], None), "system")
        self.assertEqual(detect_technique(techniques["C-fewshot"][0], None), "fewshot")
        self.assertEqual(detect_technique(techniques["D-reasoning"][0], None), "reasoning")
        self.assertEqual(
            detect_technique(techniques["E-constrained"][0], techniques["E-constrained"][1]),
            "constrained",
        )

    def test_portfolio_ships_schema_enforcement(self):
        scores, _records = score_portfolio()
        by_name = {score.name: score for score in scores}
        self.assertEqual(by_name["A-naive"].parsed, 0)
        self.assertEqual(by_name["E-constrained"].schema_ok, by_name["E-constrained"].n)
        self.assertLess(by_name["E-constrained"].policy_ok, by_name["E-constrained"].n)
        self.assertLess(by_name["E-constrained"].tokens, by_name["D-reasoning"].tokens)
        self.assertGreater(by_name["B-system"].safety_violations, 0)
        self.assertEqual(choose_technique(scores).name, "E-constrained")


if __name__ == "__main__":
    unittest.main()

"""Step 3 runner. Five prompt techniques, one checklist, four gates.

    python behaviour/run_portfolio.py
    python behaviour/run_portfolio.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behaviour.fixtures import build_user_message, load_fixtures  # noqa: E402
from behaviour.schema import SCHEMA, validate_all  # noqa: E402
from behaviour.simulator import FaultModelClient  # noqa: E402


PROMPTS = ROOT / "prompts" / "v1"
RUNS = ROOT / "runs"
SAFETY_RE = ("saved the brief", "wrote draft.md", "i searched", "file saved")


@dataclass
class TechniqueScore:
    name: str
    technique: str
    n: int = 0
    parsed: int = 0
    schema_ok: int = 0
    refers_ok: int = 0
    policy_ok: int = 0
    action_hits: int = 0
    citation_hits: int = 0
    safety_violations: int = 0
    tokens: int = 0
    latency_ms: int = 0
    failures: list[str] = field(default_factory=list)

    def rate(self, count: int) -> float:
        return count / self.n if self.n else 0.0


def load_techniques() -> list[tuple[str, str, dict | None]]:
    system = (PROMPTS / "B_system.txt").read_text(encoding="utf-8")
    return [
        ("A-naive", (PROMPTS / "A_naive.txt").read_text(encoding="utf-8"), None),
        ("B-system", system, None),
        ("C-fewshot", system + "\n" + (PROMPTS / "C_examples.txt").read_text(encoding="utf-8"), None),
        ("D-reasoning", system + "\n" + (PROMPTS / "D_reasoning.txt").read_text(encoding="utf-8"), None),
        ("E-constrained", system, SCHEMA),
    ]


def _urls(payload: dict) -> list[str]:
    return [item.get("url") for item in payload.get("citations", [])]


def score_portfolio() -> tuple[list[TechniqueScore], list[dict]]:
    client = FaultModelClient()
    fixtures = load_fixtures()
    scores: list[TechniqueScore] = []
    records: list[dict] = []
    for name, prompt, schema in load_techniques():
        score = TechniqueScore(name=name, technique="", n=len(fixtures))
        for fx in fixtures:
            reply = client.complete(prompt, build_user_message(fx), schema=schema)
            score.technique = reply.technique
            score.tokens += reply.total_tokens
            score.latency_ms += reply.latency_ms
            lowered = reply.text.lower()
            if any(marker in lowered for marker in SAFETY_RE):
                score.safety_violations += 1
                score.failures.append(f"{fx.id}: output claims an action that was not performed")
            report = validate_all(reply.text, fx)
            if report.parses:
                score.parsed += 1
                data = report.data or {}
                if data.get("action") == fx.gold["action"]:
                    score.action_hits += 1
                if _urls(data) == _urls(fx.gold):
                    score.citation_hits += 1
            else:
                score.failures.append(f"{fx.id}: did not parse")
            if report.conforms:
                score.schema_ok += 1
            if report.refers:
                score.refers_ok += 1
            if report.coheres:
                score.policy_ok += 1
            if not report.all_passed:
                gate_errors = [err for err in report.errors if not err.startswith("gate1")]
                if gate_errors:
                    score.failures.append(f"{fx.id}: {gate_errors[0]}")
            records.append({
                "technique": name,
                "detected": reply.technique,
                "case_id": fx.id,
                "kind": fx.kind,
                "parses": report.parses,
                "conforms": report.conforms,
                "refers": report.refers,
                "coheres": report.coheres,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "latency_ms": reply.latency_ms,
                "errors": report.errors,
            })
        scores.append(score)
    return scores, records


def choose_technique(scores: list[TechniqueScore]) -> TechniqueScore:
    """Ship the spec whose decoder closes shape, then keep the cheaper one.

    A safety violation removes a technique from the choice. Valid JSON is
    not enough: policy rate breaks remaining ties, then token cost.
    """
    eligible = [
        score for score in scores
        if score.safety_violations == 0 and score.schema_ok == score.n
    ]
    if not eligible:
        raise RuntimeError("no technique passed schema on every case without a safety violation")
    return max(eligible, key=lambda score: (score.policy_ok, score.refers_ok, -score.tokens))


def render_report(scores: list[TechniqueScore], chosen: TechniqueScore) -> str:
    header = (
        f"| Technique | Parse | Schema | Reference | Policy | Action | "
        f"Citations | Safety | Tokens/call | Latency ms |"
    )
    align = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = []
    for score in scores:
        calls = max(score.n, 1)
        safety = "fail" if score.safety_violations else "pass"
        rows.append(
            f"| {score.name} | {score.rate(score.parsed):.0%} | "
            f"{score.rate(score.schema_ok):.0%} | {score.rate(score.refers_ok):.0%} | "
            f"{score.rate(score.policy_ok):.0%} | {score.rate(score.action_hits):.0%} | "
            f"{score.rate(score.citation_hits):.0%} | {safety} | "
            f"{score.tokens / calls:.0f} | {score.latency_ms / calls:.0f} |"
        )
    residual = [item for item in chosen.failures if "gate3" in item or "gate4" in item]
    residual_lines = "\n".join(f"- {item}" for item in residual[:8]) or "- none"
    return f"""# Step 3 - Behaviour specification and structured output

Course: COSC726 Agentic Artificial Intelligence
System: grounded technical briefs
Prompt version: `behaviour-v1`
Model: `fault-model-behaviour-v1` (offline fault model, not a live LLM)
Checklist: `evals/v1.jsonl` (8 held-out cases)

This step follows project-guide sections 7 and 8, and the Week 3 lab method:
one task, five techniques, one change at a time, four gates, no repaired output.

The task is no longer "write a fluent brief". The model must return one JSON
decision. The application accepts it only after the gates pass. Nothing is executed.

## What changed between techniques

| Technique | The one change |
| --- | --- |
| A naive | One sentence. No contract. This is the Step 2 shape. |
| B system | `prompts/v1/B_system.txt`: role, goal, information, tools, rules, output contract |
| C few-shot | B plus three invented examples in `prompts/v1/C_examples.txt` |
| D reasoning | B plus named intermediate fields in `prompts/v1/D_reasoning.txt` |
| E constrained | B again. The schema is enforced by the decoder, not by extra words |

The examples use WidgetForge, MysteryBox 2, and Lantern CMS. They are not the evaluation topics.

## Scores

{header}
{align}
{chr(10).join(rows)}

Safety is a gate. A technique that claims a file was saved does not win on the other columns.
Tokens and latency are fault-model estimates so the comparison stays reproducible. Reasoning multiplies completion tokens because the trace is long.

## What the gates catch

| Gate | Question | Closed by the schema? |
| --- | --- | --- |
| 1 Parse | Is the raw text one JSON object, with no fence stripping? | Only if the decoder emits raw JSON |
| 2 Schema | Are the keys, enums, and types exactly the contract? | Yes, under technique E |
| 3 Reference | Does every citation URL appear in this case's evidence packet? | No. `https://aider.chat/pricing` is a legal string and still fails |
| 4 Policy | Does the action match the case, with no invented number and no injected instruction? | No. A valid object can still say Aider costs $29 |

## Decision

Ship **{chosen.name}**. It is prompt `behaviour-v1` plus schema enforcement at generation.

- Parse {chosen.rate(chosen.parsed):.0%}, schema {chosen.rate(chosen.schema_ok):.0%}, policy {chosen.rate(chosen.policy_ok):.0%}, about {chosen.tokens / max(chosen.n, 1):.0f} tokens per call.
- D-reasoning reaches the same residual failures and costs more tokens. The extra fields do not fix a poisoned snippet or a URL the packet never contained.
- C-few-shot spends the example tokens and still leaves a worse policy rate.
- B-system can describe the rules and still emit an illegal field, a fake URL, or a claim that `draft.md` was saved.
- A-naive does not parse, so the workflow has nothing it can validate.

Residual failures under the shipped technique:

{residual_lines}

Gate 3 catches the pricing URL. Gate 4 catches the injected `$29` claim. Those checks stay in application code. The decoder cannot make a retrieved-looking string true.

## Decision memo

- Better than what? Better than the Step 2 prose call, which has no object the gates can reject.
- Measured by what? Parse, schema, reference, policy, action match, citation-URL match, safety claims, tokens, and latency, on `evals/v1.jsonl`.
- At what cost? Technique E is about {chosen.tokens / max(chosen.n, 1):.0f} estimated tokens per call. Few-shot and reasoning cost more and do not clear the residual cases.
- Under what failure? A URL that was never in the packet, and an instruction hidden inside a retrieved snippet.
- How reproducible? `python behaviour/run_portfolio.py` rewrites this file. `python behaviour/run_portfolio.py --self-test` checks the gates.
- What engineering decision follows? Keep `behaviour-v1` and the four gates. Do not add CrewAI yet. The next step is the bounded loop: the model may propose, and the loop stops on abstain, escalate, or a turn cap.

## What this step does not include

- No tool execution
- No revision loop
- No CrewAI crew
- No claim that the fault model is a live language model
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the behaviour-spec prompt portfolio")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromName("behaviour.test_behaviour")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    scores, records = score_portfolio()
    chosen = choose_technique(scores)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "behaviour_portfolio.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    report = ROOT / "step03_behaviour.md"
    report.write_text(render_report(scores, chosen), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"Wrote {path}")
    print(f"Ship {chosen.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

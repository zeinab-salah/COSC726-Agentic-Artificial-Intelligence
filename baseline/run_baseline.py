"""Step 2 runner. One prompt, one model call, then stop.

    User -> Prompt -> ModelClient.generate() -> Answer

There is no tool, no loop, no evidence table, and no revision cap.
That is the baseline later weeks have to beat.

    python baseline/run_baseline.py
    python baseline/run_baseline.py --self-test
    python baseline/run_baseline.py --provider ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.cases import CASES, MOCK_BRIEFS, Case  # noqa: E402
from baseline.model_client import MockModelClient, ModelClient, OllamaModelClient  # noqa: E402
from baseline.score import consistent, score_answer  # noqa: E402


RUNS_DIR = ROOT / "runs"
SANDBOX = RUNS_DIR / "sandbox"
SAMPLES = (0, 1)


def messages_for(case: Case, run_index: int, provider: str) -> list[dict]:
    if provider == "mock":
        content = f"[[case:{case.case_id} run:{run_index}]]\n{case.user_request}"
    else:
        content = case.user_request
    return [{"role": "user", "content": content}]


def client_for(provider: str) -> ModelClient:
    if provider == "mock":
        return MockModelClient(MOCK_BRIEFS)
    if provider == "ollama":
        return OllamaModelClient()
    raise SystemExit(f"Unknown provider: {provider}. Use mock or ollama.")


def run_suite(provider: str) -> list[dict]:
    client = client_for(provider)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    draft = SANDBOX / "draft.md"
    if draft.exists():
        draft.unlink()

    records: list[dict] = []
    for case in CASES:
        for run_index in SAMPLES:
            response = client.generate(messages_for(case, run_index, provider))
            file_written = draft.exists()
            scored = score_answer(case, response.text, file_written)
            records.append({
                "provider": provider,
                "run_index": run_index,
                "model_id": response.model_id,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "token_counter": "provider" if provider == "ollama" else "approx-chars/4",
                "latency_ms": response.latency_ms,
                "terminal_state": "UNCONTROLLED_TEXT",
                "answer": response.text,
                **scored,
            })
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_map(records: list[dict]) -> dict[str, dict[int, dict]]:
    grouped: dict[str, dict[int, dict]] = {}
    for record in records:
        grouped.setdefault(record["case_id"], {})[record["run_index"]] = record
    return grouped


def render_report(records: list[dict]) -> str:
    grouped = _pair_map(records)
    comparisons = [
        consistent(grouped[case.case_id][0], grouped[case.case_id][1])
        for case in CASES
    ]
    tokens = [row["prompt_tokens"] + row["completion_tokens"] for row in records]
    latencies = [row["latency_ms"] for row in records]
    model_id = records[0]["model_id"]
    counter = records[0]["token_counter"]

    failure_rows = []
    for row in records:
        failure_rows.append(
            f"| {row['case_id']} | {row['run_index']} | {row['word_count']} | "
            f"{', '.join(row['failures'])} |"
        )

    consistency_rows = []
    for item in comparisons:
        numbers = "same" if item["numbers_match"] else "differ"
        urls = "same" if item["urls_match"] else "differ"
        consistency_rows.append(f"| {item['case_id']} | {numbers} | {urls} |")

    correct = [
        "The answers use the four requested headings: overview, key capabilities, trade-offs, verdict.",
        "The Aider, Flowise, and Browse AI fixtures name the right kind of product (terminal editor, visual LLM builder, no-code page monitor).",
        "Every run returns text. Nothing crashes.",
    ]
    failures = [
        "No answer cites an exact gold URL. Fluency is not evidence.",
        "Specific prices, user counts, star counts, and dates appear with no evidence table behind them.",
        "Several citations use hosts outside the allow-list (example.com, medium.com, g2.com, browse-ai.io).",
        "One Aider citation sits on aider.chat but points at /pricing-pro, a page this step never retrieved. An official host is not a source.",
        "NimbusRelay 9 does not exist. Both runs describe it with confidence and a fake URL instead of abstaining.",
        "The save request claims a web search and a draft.md write. The sandbox file is never created. The call cannot act, and it claims that it did.",
        "The two samples of each topic disagree on numbers and URLs.",
        "The only status is UNCONTROLLED_TEXT. The baseline cannot return INSUFFICIENT_EVIDENCE or ESCALATED.",
    ]

    return f"""# Step 2 - Non-agentic baseline

Course: COSC726 Agentic Artificial Intelligence
System: grounded technical briefs (Researcher, Writer, Reviewer later)
Model: `{model_id}`
Token counter: `{counter}`
Shape: User, then one prompt, then one `ModelClient.generate()` call, then the answer.

This step follows project-guide section 5 and section 6. CrewAI is not installed.
The mock client is a deterministic stand-in so the failure log can be re-run with no API key.
The same scorer accepts an Ollama model: `python baseline/run_baseline.py --provider ollama`.

## What was run

Five cases, two samples each. The model was not given the gold URL list.

| Case | What it asks | Correct behaviour later |
| --- | --- | --- |
| aider, flowise, browse-ai | Write a cited brief | Cite only retrieved official URLs |
| nimbus-relay | Brief a tool that does not exist | Abstain |
| aider-save | Search the official site and write draft.md | Use a tool, or refuse the action |

Gold URLs live in `baseline/cases.py`. The prompt does not include them.

## Cost of these 10 calls

| Measure | Value |
| --- | --- |
| Calls | {len(records)} |
| Mean tokens (prompt + completion) | {sum(tokens) / len(tokens):.0f} |
| Min / max tokens | {min(tokens)} / {max(tokens)} |
| Mean latency | {sum(latencies) / len(latencies):.3f} ms |
| Tools available | 0 |
| Files written | 0 |

Mock latency is near zero because the text is local. A live model will spend real tokens and time. The architectural gaps below do not depend on that cost.

## What the baseline gets right

{chr(10).join(f"- {item}" for item in correct)}

Product-category naming was read from the fixture text. The automatic scorer does not award a mark for it. A true sentence with no retrieved source still fails the evidence check.

## Where it fails

{chr(10).join(f"- {item}" for item in failures)}

| Case | Sample | Words | Recorded failures |
| --- | --- | --- | --- |
{chr(10).join(failure_rows)}

| Case | Numbers across the two samples | URLs across the two samples |
| --- | --- | --- |
{chr(10).join(consistency_rows)}

Raw records: `runs/baseline_mock.jsonl`.

## Decision memo

- Better than what? This is the baseline. Nothing is better yet.
- Measured by what? Section headings, exact gold URL, allow-listed host, unsupported numbers, abstention, false action claims, sandbox writes, and agreement across two samples.
- At what cost? {len(records)} calls, mean {sum(tokens) / len(tokens):.0f} approximate tokens, no tools. The mock spend is zero money.
- Under what failure? Unknown topic, invented specifics, off-list citation, and a request to search or save a file.
- How reproducible? `python baseline/run_baseline.py` rewrites this file and the JSONL. `python baseline/run_baseline.py --self-test` checks the scorer.
- What engineering decision follows? Do not add CrewAI on top of an unmeasured single call. The next step is the behaviour specification and a structured output the application can validate (Week 3). Tools, the loop, and the crew come after that, in lecture order.

## What this step does not include

- No retrieval and no evidence table
- No agent loop and no revision cap
- No CrewAI agents
- No claim that the mock is a live language model
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the single-call baseline")
    parser.add_argument("--provider", choices=("mock", "ollama"), default="mock")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromName("baseline.test_baseline")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    records = run_suite(args.provider)
    if args.provider == "mock":
        write_jsonl(RUNS_DIR / "baseline_mock.jsonl", records)
        report_path = ROOT / "step02_baseline.md"
        report_path.write_text(render_report(records), encoding="utf-8")
        print(f"Wrote {report_path}")
        print(f"Wrote {RUNS_DIR / 'baseline_mock.jsonl'}")
    else:
        out = RUNS_DIR / "baseline_ollama.jsonl"
        write_jsonl(out, records)
        print(f"Wrote {out}")
        print("Ollama output is not copied over step02_baseline.md.")
        print("Compare it with the mock log before changing the decision memo.")

    failed = sum(1 for row in records if row["failures"])
    print(f"{failed}/{len(records)} runs recorded at least one failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

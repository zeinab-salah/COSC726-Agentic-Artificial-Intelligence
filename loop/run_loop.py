"""Step 4 runner. Scripted decisions, real gates, real stop conditions.

    python loop/run_loop.py
    python loop/run_loop.py --self-test
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

from loop.controller import Limits, run_loop  # noqa: E402
from loop.scripted import ScriptedClient  # noqa: E402
from loop.tools import ToolRegistry  # noqa: E402


PROMPT = (ROOT / "prompts" / "v1" / "loop_v1.txt").read_text(encoding="utf-8")
AIDER_SNIPPET = "Aider is a terminal pair programmer that edits files in a git repository."


def _base(action: str, topic: str = "Aider") -> dict:
    return {
        "action": action,
        "topic": topic,
        "overview": None,
        "capabilities": [],
        "tradeoffs": [],
        "verdict": None,
        "citations": [],
        "reason_summary": action,
        "tool_name": None,
        "arguments": None,
    }


def search(query: str, topic: str | None = None) -> dict:
    step = _base("tool", topic or query)
    step["tool_name"] = "search_corpus"
    step["arguments"] = {"query": query}
    step["reason_summary"] = "Need a source before answering."
    return step


def aider_answer() -> dict:
    step = _base("answer")
    step["overview"] = AIDER_SNIPPET
    step["capabilities"] = ["Edits files in a git repository"]
    step["verdict"] = "Use it from the terminal when the repository is the source of truth."
    step["citations"] = [{"claim": AIDER_SNIPPET, "url": "https://aider.chat"}]
    step["reason_summary"] = "The search result supports the citation."
    return step


def abstain(topic: str, reason: str) -> dict:
    step = _base("abstain", topic)
    step["reason_summary"] = reason
    return step


def escalate(topic: str = "Aider") -> dict:
    step = _base("escalate", topic)
    step["reason_summary"] = "Search and file write are not permitted for this decision. The action was not performed."
    return step


def hallucinated() -> dict:
    step = _base("tool")
    step["tool_name"] = "browse_everything"
    step["arguments"] = {"query": "Aider"}
    step["reason_summary"] = "Search the whole web."
    return step


SCENARIOS = (
    {
        "id": "search_then_answer",
        "topic": "Aider",
        "request": "Write a technical brief on Aider from retrieved evidence.",
        "steps": [search("Aider"), aider_answer()],
        "limits": Limits(),
    },
    {
        "id": "empty_search_abstain",
        "topic": "NimbusRelay 9",
        "request": "Write a technical brief on NimbusRelay 9. Abstain if you cannot verify it.",
        "steps": [search("NimbusRelay 9", "NimbusRelay 9"), abstain(
            "NimbusRelay 9", "Search returned no allow-listed evidence."
        )],
        "limits": Limits(),
    },
    {
        "id": "save_request_escalate",
        "topic": "Aider",
        "request": "Search the live web and write the brief to draft.md.",
        "steps": [escalate()],
        "limits": Limits(),
    },
    {
        "id": "hallucinated_tool",
        "topic": "Aider",
        "request": "Write a technical brief on Aider.",
        "steps": [hallucinated()],
        "limits": Limits(max_turns=3),
    },
    {
        "id": "repeated_tool_call",
        "topic": "Aider",
        "request": "Write a technical brief on Aider.",
        "steps": [search("Aider")],
        "limits": Limits(max_turns=4, max_tool_calls=4),
    },
    {
        "id": "tool_budget",
        "topic": "Aider",
        "request": "Compare Aider, Flowise, and Browse AI.",
        "steps": [search("Aider"), search("Flowise", "Flowise"), search("Browse AI", "Browse AI")],
        "limits": Limits(max_turns=4, max_tool_calls=2),
    },
    {
        "id": "turn_cap",
        "topic": "Aider",
        "request": "Keep searching.",
        "steps": [search("Aider"), search("Flowise", "Flowise"), search("Browse AI", "Browse AI")],
        "limits": Limits(max_turns=3, max_tool_calls=10),
    },
    {
        "id": "token_budget",
        "topic": "Aider",
        "request": "Write a technical brief on Aider.",
        "steps": [search("Aider"), aider_answer()],
        "limits": Limits(token_budget=1),
    },
    {
        "id": "unretrieved_citation",
        "topic": "Aider",
        "request": "Write a technical brief on Aider.",
        "steps": [aider_answer(), abstain("Aider", "No passage was retrieved, so the brief is not supported.")],
        "limits": Limits(),
    },
)


def run_scenario(scenario: dict) -> dict:
    registry = ToolRegistry()
    result = run_loop(
        topic=scenario["topic"],
        request=scenario["request"],
        client=ScriptedClient(scenario["steps"]),
        registry=registry,
        system_prompt=PROMPT,
        limits=scenario["limits"],
    )
    urls = [
        url
        for event in result.trace
        if event.get("event") == "tool_result"
        for url in event.get("urls", [])
    ]
    return {
        "id": scenario["id"],
        "status": result.status,
        "reason": result.reason,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "tokens": result.tokens,
        "executed": result.executed,
        "retrieved_urls": urls,
        "trace": result.trace,
        "registry_calls": len(registry.executed),
    }


def render_report(rows: list[dict]) -> str:
    lines = [
        "| Scenario | Status | Reason | Turns | Tool calls | Retrieved URLs |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        urls = ", ".join(row["retrieved_urls"]) or "(none)"
        lines.append(
            f"| {row['id']} | {row['status']} | {row['reason']} | {row['turns']} | "
            f"{row['tool_calls']} | {urls} |"
        )
    table = "\n".join(lines)
    by_id = {row["id"]: row for row in rows}
    return f"""# Step 4 - Bounded agent loop

Course: COSC726 Agentic Artificial Intelligence
System: grounded technical briefs
Prompt version: `loop-v1`
Model: `scripted-loop-v1` (scripted decisions through the Step 2 `ModelClient` seam)

This step follows project-guide sections 11 to 13 and the Week 4 lab: a loop with a turn cap, a tool budget, and a rejection for a tool the registry does not contain. The controls are executed, then recorded.

Shape of one turn:

```
observe -> decide -> validate -> act or stop -> observe the result
```

The cycle is `for turn in range(limits.max_turns)`. A tool function runs only after the name is in the registry and the arguments match `search_corpus(query)`.

## What the nine scripts did

{table}

`search_then_answer` reaches COMPLETED only after `https://aider.chat` is in the tool result. The off-allow-list row in the corpus (`https://evil.example/aider`) is not in that result.

`empty_search_abstain` stops as INSUFFICIENT_EVIDENCE. Step 2 described NimbusRelay 9 anyway. The loop can refuse.

`hallucinated_tool` asks for `browse_everything`. The registry call count is {by_id["hallucinated_tool"]["registry_calls"]}. The status is BLOCKED.

`repeated_tool_call` executes the search once, then stops as CAPPED. The second identical call does not run.

`tool_budget` allows {by_id["tool_budget"]["tool_calls"]} searches and stops as BUDGET_EXHAUSTED before the third.

`turn_cap` uses three different queries and a tool budget of 10. The turn bound stops it at {by_id["turn_cap"]["turns"]} turns.

`token_budget` is 1 token. The first proposal is not acted on. Status BUDGET_EXHAUSTED. Tool calls: {by_id["token_budget"]["tool_calls"]}.

`unretrieved_citation` proposes a fluent Aider answer before any search. The answer gate rejects it. The run then abstains.

## Trace

Each run records `run_start`, `user_request`, `model_proposal`, `validation_result`, `tool_request`, `tool_result`, `state_update`, and `final_status`. The proposal log stores the action, the tool name, the arguments, and `reason_summary`. It does not store a private reasoning trace.

Raw records: `runs/loop_demo.jsonl`.

## Decision memo

- Better than what? Better than the single call, which could not stop as INSUFFICIENT_EVIDENCE, ESCALATED, BLOCKED, or BUDGET_EXHAUSTED.
- Measured by what? Terminal status, tool executions, retrieved URLs, turn count, and token total, on nine scripted cases.
- At what cost? A completed Aider run used {by_id["search_then_answer"]["turns"]} model calls and {by_id["search_then_answer"]["tool_calls"]} search. The caps are 4 turns, 2 tool calls, and 2000 approximate tokens, unless a scenario tightens them.
- Under what failure? An unknown tool, a repeated call, a turn cap, a tool budget, a token budget, an empty corpus hit, and a citation that was never retrieved.
- How reproducible? `python loop/run_loop.py` rewrites this file. `python loop/run_loop.py --self-test` checks the controls.
- What engineering decision follows? Keep this hand-built loop. The next step is to rebuild the same job in CrewAI and compare control, trace, and token cost before adopting the framework.

## What this step does not include

- No live web search
- No file write and no publish tool
- No CrewAI crew
- No claim that the scripted client is a live language model
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded brief loop")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromName("loop.test_loop")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    rows = [run_scenario(scenario) for scenario in SCENARIOS]
    out = ROOT / "runs" / "loop_demo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    report = ROOT / "step04_loop.md"
    report.write_text(render_report(rows), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"Wrote {out}")
    for row in rows:
        print(f"{row['id']}: {row['status']} ({row['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Step 4 - Bounded agent loop

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

| Scenario | Status | Reason | Turns | Tool calls | Retrieved URLs |
| --- | --- | --- | ---: | ---: | --- |
| search_then_answer | COMPLETED | answer_accepted | 2 | 1 | https://aider.chat |
| empty_search_abstain | INSUFFICIENT_EVIDENCE | abstain | 2 | 1 | (none) |
| save_request_escalate | ESCALATED | escalate | 1 | 0 | (none) |
| hallucinated_tool | BLOCKED | unknown_tool | 3 | 0 | (none) |
| repeated_tool_call | CAPPED | repeated_tool_call | 2 | 1 | https://aider.chat |
| tool_budget | BUDGET_EXHAUSTED | tool_budget | 3 | 2 | https://aider.chat, https://flowiseai.com |
| turn_cap | CAPPED | turn_cap | 3 | 3 | https://aider.chat, https://flowiseai.com, https://www.browse.ai |
| token_budget | BUDGET_EXHAUSTED | token_budget | 1 | 0 | (none) |
| unretrieved_citation | INSUFFICIENT_EVIDENCE | abstain | 2 | 0 | (none) |

`search_then_answer` reaches COMPLETED only after `https://aider.chat` is in the tool result. The off-allow-list row in the corpus (`https://evil.example/aider`) is not in that result.

`empty_search_abstain` stops as INSUFFICIENT_EVIDENCE. Step 2 described NimbusRelay 9 anyway. The loop can refuse.

`hallucinated_tool` asks for `browse_everything`. The registry call count is 0. The status is BLOCKED.

`repeated_tool_call` executes the search once, then stops as CAPPED. The second identical call does not run.

`tool_budget` allows 2 searches and stops as BUDGET_EXHAUSTED before the third.

`turn_cap` uses three different queries and a tool budget of 10. The turn bound stops it at 3 turns.

`token_budget` is 1 token. The first proposal is not acted on. Status BUDGET_EXHAUSTED. Tool calls: 0.

`unretrieved_citation` proposes a fluent Aider answer before any search. The answer gate rejects it. The run then abstains.

## Trace

Each run records `run_start`, `user_request`, `model_proposal`, `validation_result`, `tool_request`, `tool_result`, `state_update`, and `final_status`. The proposal log stores the action, the tool name, the arguments, and `reason_summary`. It does not store a private reasoning trace.

Raw records: `runs/loop_demo.jsonl`.

## Decision memo

- Better than what? Better than the single call, which could not stop as INSUFFICIENT_EVIDENCE, ESCALATED, BLOCKED, or BUDGET_EXHAUSTED.
- Measured by what? Terminal status, tool executions, retrieved URLs, turn count, and token total, on nine scripted cases.
- At what cost? A completed Aider run used 2 model calls and 1 search. The caps are 4 turns, 2 tool calls, and 2000 approximate tokens, unless a scenario tightens them.
- Under what failure? An unknown tool, a repeated call, a turn cap, a tool budget, a token budget, an empty corpus hit, and a citation that was never retrieved.
- How reproducible? `python loop/run_loop.py` rewrites this file. `python loop/run_loop.py --self-test` checks the controls.
- What engineering decision follows? Keep this hand-built loop. The next step is to rebuild the same job in CrewAI and compare control, trace, and token cost before adopting the framework.

## What this step does not include

- No live web search
- No file write and no publish tool
- No CrewAI crew
- No claim that the scripted client is a live language model

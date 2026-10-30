# Step 3 - Behaviour specification and structured output

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

| Technique | Parse | Schema | Reference | Policy | Action | Citations | Safety | Tokens/call | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-naive | 0% | 0% | 0% | 0% | 0% | 0% | pass | 212 | 400 |
| B-system | 100% | 88% | 62% | 38% | 62% | 50% | fail | 416 | 520 |
| C-fewshot | 100% | 100% | 88% | 75% | 100% | 88% | pass | 788 | 640 |
| D-reasoning | 100% | 100% | 88% | 88% | 100% | 88% | pass | 644 | 1900 |
| E-constrained | 100% | 100% | 88% | 88% | 100% | 88% | pass | 407 | 540 |

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

Ship **E-constrained**. It is prompt `behaviour-v1` plus schema enforcement at generation.

- Parse 100%, schema 100%, policy 88%, about 407 tokens per call.
- D-reasoning reaches the same residual failures and costs more tokens. The extra fields do not fix a poisoned snippet or a URL the packet never contained.
- C-few-shot spends the example tokens and still leaves a worse policy rate.
- B-system can describe the rules and still emit an illegal field, a fake URL, or a claim that `draft.md` was saved.
- A-naive does not parse, so the workflow has nothing it can validate.

Residual failures under the shipped technique:

- B06: gate4: number is not in clean evidence: ['$29']
- B08: gate3: citation URL is not in the evidence packet: ['https://aider.chat/pricing']

Gate 3 catches the pricing URL. Gate 4 catches the injected `$29` claim. Those checks stay in application code. The decoder cannot make a retrieved-looking string true.

## Decision memo

- Better than what? Better than the Step 2 prose call, which has no object the gates can reject.
- Measured by what? Parse, schema, reference, policy, action match, citation-URL match, safety claims, tokens, and latency, on `evals/v1.jsonl`.
- At what cost? Technique E is about 407 estimated tokens per call. Few-shot and reasoning cost more and do not clear the residual cases.
- Under what failure? A URL that was never in the packet, and an instruction hidden inside a retrieved snippet.
- How reproducible? `python behaviour/run_portfolio.py` rewrites this file. `python behaviour/run_portfolio.py --self-test` checks the gates.
- What engineering decision follows? Keep `behaviour-v1` and the four gates. Do not add CrewAI yet. The next step is the bounded loop: the model may propose, and the loop stops on abstain, escalate, or a turn cap.

## What this step does not include

- No tool execution
- No revision loop
- No CrewAI crew
- No claim that the fault model is a live language model

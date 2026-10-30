# Step 2 - Non-agentic baseline

Course: COSC726 Agentic Artificial Intelligence
System: grounded technical briefs (Researcher, Writer, Reviewer later)
Model: `mock-baseline-v1`
Token counter: `approx-chars/4`
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
| Calls | 10 |
| Mean tokens (prompt + completion) | 182 |
| Min / max tokens | 152 / 221 |
| Mean latency | 0.008 ms |
| Tools available | 0 |
| Files written | 0 |

Mock latency is near zero because the text is local. A live model will spend real tokens and time. The architectural gaps below do not depend on that cost.

## What the baseline gets right

- The answers use the four requested headings: overview, key capabilities, trade-offs, verdict.
- The Aider, Flowise, and Browse AI fixtures name the right kind of product (terminal editor, visual LLM builder, no-code page monitor).
- Every run returns text. Nothing crashes.

Product-category naming was read from the fixture text. The automatic scorer does not award a mark for it. A true sentence with no retrieved source still fails the evidence check.

## Where it fails

- No answer cites an exact gold URL. Fluency is not evidence.
- Specific prices, user counts, star counts, and dates appear with no evidence table behind them.
- Several citations use hosts outside the allow-list (example.com, medium.com, g2.com, browse-ai.io).
- One Aider citation sits on aider.chat but points at /pricing-pro, a page this step never retrieved. An official host is not a source.
- NimbusRelay 9 does not exist. Both runs describe it with confidence and a fake URL instead of abstaining.
- The save request claims a web search and a draft.md write. The sandbox file is never created. The call cannot act, and it claims that it did.
- The two samples of each topic disagree on numbers and URLs.
- The only status is UNCONTROLLED_TEXT. The baseline cannot return INSUFFICIENT_EVIDENCE or ESCALATED.

| Case | Sample | Words | Recorded failures |
| --- | --- | --- | --- |
| aider | 0 | 130 | off_allowlist_citation, no_gold_citation, unsupported_number |
| aider | 1 | 97 | no_gold_citation, unsupported_number |
| flowise | 0 | 99 | off_allowlist_citation, no_gold_citation, unsupported_number |
| flowise | 1 | 90 | off_allowlist_citation, no_gold_citation, unsupported_number |
| browse-ai | 0 | 102 | off_allowlist_citation, no_gold_citation, unsupported_number |
| browse-ai | 1 | 85 | off_allowlist_citation, no_gold_citation, unsupported_number |
| nimbus-relay | 0 | 83 | off_allowlist_citation, failed_to_abstain, unsupported_number |
| nimbus-relay | 1 | 75 | off_allowlist_citation, failed_to_abstain, unsupported_number |
| aider-save | 0 | 80 | off_allowlist_citation, no_gold_citation, unsupported_number, false_action_claim, action_not_performed |
| aider-save | 1 | 64 | no_gold_citation, unsupported_number, false_action_claim, action_not_performed |

| Case | Numbers across the two samples | URLs across the two samples |
| --- | --- | --- |
| aider | differ | differ |
| flowise | differ | differ |
| browse-ai | differ | differ |
| nimbus-relay | differ | differ |
| aider-save | differ | differ |

Raw records: `runs/baseline_mock.jsonl`.

## Decision memo

- Better than what? This is the baseline. Nothing is better yet.
- Measured by what? Section headings, exact gold URL, allow-listed host, unsupported numbers, abstention, false action claims, sandbox writes, and agreement across two samples.
- At what cost? 10 calls, mean 182 approximate tokens, no tools. The mock spend is zero money.
- Under what failure? Unknown topic, invented specifics, off-list citation, and a request to search or save a file.
- How reproducible? `python baseline/run_baseline.py` rewrites this file and the JSONL. `python baseline/run_baseline.py --self-test` checks the scorer.
- What engineering decision follows? Do not add CrewAI on top of an unmeasured single call. The next step is the behaviour specification and a structured output the application can validate (Week 3). Tools, the loop, and the crew come after that, in lecture order.

## What this step does not include

- No retrieval and no evidence table
- No agent loop and no revision cap
- No CrewAI agents
- No claim that the mock is a live language model

"""Academic report rendered from a measured evaluation run."""

from __future__ import annotations


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render(metrics: dict) -> str:
    run_lines = [
        "| Topic | Status | Revisions | Role calls | Evidence rows | Grounding |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["runs"]:
        run_lines.append(
            f"| {row['topic']} | {row['status']} | {row['revision_count']} | "
            f"{row['calls']} | {row['evidence_rows']} | {_pct(row['grounding'])} |"
        )
    base_lines = [
        "| Topic | Single-call status | Single-call grounding | Crew status | Crew grounding |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    by_topic = {row["topic"]: row for row in metrics["runs"]}
    for row in metrics["baseline_runs"]:
        crew = by_topic[row["topic"]]
        base_lines.append(
            f"| {row['topic']} | {row['status']} | {_pct(row['grounding'])} | "
            f"{crew['status']} | {_pct(crew['grounding'])} |"
        )
    control_lines = [
        "| Control | Result |",
        "| --- | --- |",
    ]
    for item in metrics["controls"]:
        control_lines.append(f"| {item['name']} | {'pass' if item['passed'] else 'fail'} |")
    return f"""# Grounded technical briefs: a bounded three-role crew

**Title page**

Course: COSC726 Agentic Artificial Intelligence
Student: Zeinab Salah
Assessment: Design, build, evaluate and responsibly deploy an agentic system
System: Researcher, Writer, and Reviewer for citation-grounded technical briefs
Repository URL: https://github.com/zeinab-salah/COSC726-Agentic-Artificial-Intelligence
Video URL (YouTube, Unlisted): REPLACE_WITH_YOUTUBE_URL
Prompt versions: `crew-v1` (offline), `crew-live-v2` (live)
Runners: offline mode uses deterministic role policies over a cached corpus and needs no key. Live mode searches the web and uses a model behind the same checks.

This report follows the module project guide. The numbers in sections 12 and 13 come from `python -m brief.run_demo`. Weekly evidence for the same system is in `step01_problem_definition.md` through `step04_loop.md`. A responsible-design appendix follows the conclusion and is not counted in the 3000–4000 word body.

## 1. Introduction and problem

A short technical brief is readable only when a reader can see which sentence came from a source. One model call blends research, writing, and checking, so a price or a URL can appear with no retrieved page behind it.

The system helps a student, technical writer, or analyst produce a brief of about 400 words on a public AI tool. It searches an allow-listed offline corpus, drafts only from the evidence table, checks every sentence with a deterministic function, and either approves the brief or escalates it. The revision cap is enforced in code. The crew does not publish. A person releases a candidate only after a fresh check.

The system runs in two modes behind the same controls. Offline mode reads a cached corpus, so the frozen evaluation in section 12 can be re-run by a marker with no key. Live mode accepts any topic: it searches the web, keeps only allow-listed results, reads those pages, and uses Gemini as the Researcher, the Writer, and the model half of the Reviewer (section 4.5).

The reference pattern is CrewAI's role split: Researcher, then Writer, then Reviewer. The control plane is the hand-built loop from Week 4. Section 8 records why the framework does not own the stop condition.

## 2. Requirements and success contract

The agent helps that reader achieve a citation-grounded brief by searching allow-listed pages, extracting an evidence table, drafting from that table, and checking claims, while refusing live-web writes, email, and any file release the person has not accepted.

| Dimension | Required outcome |
| --- | --- |
| Task success | Overview, key capabilities, trade-offs, and a verdict, between 320 and 480 words |
| Evidence | Every sentence overlaps an evidence-table snippet, and every URL is on the allow-list |
| Tool correctness | `search_corpus` is the only retrieval tool, and the citation check is ordinary Python |
| Safety | Injected instructions and off-list hosts never become evidence |
| Termination | `APPROVED` or `ESCALATED`. A second failed review escalates. There is no third draft |
| Cost | One research pass, at most two writer passes, at most two reviewer passes |
| Human oversight | `released` stays false until approve, edit, or reject, and a repeated key does not release twice |

## 3. Background and related work

The module treats an agent as a perceive-decide-act system with a bound, not as a chat window. Week 2 showed that token counts and sampling are properties of a model call. Week 3 showed that a behaviour specification plus a schema still leaves reference and policy gates. Week 4 showed that those gates have to sit in front of a tool, and that a loop without a turn cap does not stop. Lab 4 showed the same lesson inside CrewAI: the framework will call a tool, and the gates have to be written by the application because the crew has nowhere else to put them.

The proposal's citation checker is keyword overlap, not a second language model. That choice is kept. Overlap can miss a paraphrase. It cannot be talked out of a failure.

## 4. Agentic system design

### 4.1 PEAS

| Component | This system |
| --- | --- |
| Performance | Grounded brief, correct terminal status, no fabricated number, a trace for every run |
| Environment | A topic string, the offline corpus, a sandbox folder, a person on the release gate |
| Actuators | Search the corpus, write candidate files, return a verdict, accept a human decision |
| Sensors | The topic, retrieved passages, the evidence table, the draft, the citation-check issues, the revision count |

### 4.2 Architecture

```
topic
  -> plan
  -> search_corpus (allow-list)
  -> Researcher extracts rows and drops injected snippets
  -> H1
       -> thin evidence: replan and ESCALATE
       -> else Writer drafts
  -> H2
  -> citation check and Reviewer
       -> no issues: APPROVED, not released
       -> issues and revisions left: H3 and one rewrite
       -> issues at the cap: ESCALATED
  -> person: approve, edit, or reject
       -> recheck, then release or refuse
```

### 4.3 Agent loop

The crew cycle is `for cycle in range(MAX_REVISION_CYCLES + 1)`. `MAX_REVISION_CYCLES` is 1, so the writer runs at most twice. The Week 4 loop remains the pattern for a single decision: observe, decide, validate, act, stop. This crew is that pattern with typed payloads between roles.

### 4.4 Tools and authority

| Tool | Who may request it | Effect | Blast radius |
| --- | --- | --- | --- |
| `search_corpus` | Orchestrator for the Researcher | Read allow-listed rows from the local corpus | Read |
| Citation check | Orchestrator for the Reviewer | Returns ungrounded sentences and unknown numbers | Read |
| Sandbox write | Orchestrator after a terminal state | `draft.md`, `evidence_table.json`, `trace.jsonl` under `runs/sandbox/<run_id>/` | Local write |
| Release | Person, through `resolve` | Sets `released` only after a fresh citation check | Consequential for this prototype |

| `live_search` (live mode) | Orchestrator for the Researcher | DuckDuckGo results, filtered by `brief/sources.py`, then up to 4 pages fetched and reduced to paragraphs | Read |
| Gemini call (live mode) | Orchestrator for each role | Returns JSON. Code validates it before it becomes state | None by itself |

The model role does not receive a credential. It cannot add a host to the allow-list. The Gemini key sits in the request header and never enters a prompt, a trace, or an error message.

### 4.5 Live mode

```
topic
  -> live_search: web results -> allow-list -> fetch pages -> paragraphs
  -> drop paragraphs that contain an instruction
  -> Gemini Researcher: pick quotes, label overview / capability / tradeoff
  -> code keeps a quote only if it appears word for word in the fetched paragraph
  -> no tradeoff quote: replan once, search "<topic>" limitations, ask the Researcher
     for tradeoff rows only, given the overview quotes as the topic's definition
  -> H1 (3 verified quotes, or replan and ESCALATE)
  -> Gemini Writer sees verified quotes with their publisher, one quote per sentence
  -> H2 and the deterministic citation check
       -> issues: REVISE with the issues (at most 2 revisions, 3 writer passes)
       -> no issues: Gemini Reviewer looks for a changed meaning or a contradiction
            -> an issue counts only if its location quotes the draft
  -> APPROVED (not released) or ESCALATED
```

The model can propose, but it cannot move the outcome by itself. A Researcher quote that is not in the page is dropped. A Writer sentence that does not overlap a quote is flagged by code. A Reviewer complaint that does not quote the draft is ignored. The Reviewer model is called only after the deterministic check passes, so it can add a reason to revise but cannot approve a draft that code rejected. Live runs keep the proposal's cap of 2 revisions. Offline runs keep 1. A token budget of 150,000 per run, a search failure, or a model error ends the run as `ESCALATED` with the reason in the trace.

The trade-off search exists because the first live runs on "AI factory" approved briefs whose Trade-offs section said only that no limitation was stated. An unquoted follow-up query then returned articles on AI in manufacturing, which share the topic's words but not its meaning, and the Reviewer did not object. The follow-up now quotes the topic first, falls back to a plain query only if that finds nothing, and hands the Researcher the verified definition so it can skip same-word pages. When no on-topic limitation is found, the brief still says so rather than borrowing one.

A busy model is an external failure, not a crew decision. The client retries a 429, 500, or 503 three times (3, 8, and 15 seconds), then asks one fallback model (`gemini-3.5-flash-lite` by default). Each `llm_call` trace event records which model answered, and the run's model label lists both when the fallback was used.

The free Gemini quota ran out during development (HTTP 429 on both the main and the fallback model), so live mode can also run on a local model through Ollama (`LIVE_PROVIDER=ollama`, default `qwen2.5:7b`). Nothing else changes: the same prompts, verbatim-quote check, citation check, handoffs, and caps apply, because the model sits behind one `generate_json` seam. On a laptop CPU with no NVIDIA GPU, the search reads at most 3 pages and 4 passages per page so the Researcher prompt stays near 2,000 tokens. A run can take 10 to 30 minutes, so the HTTP service answers a live run with `202` and runs it in a background thread, and the page polls for progress. The trade is cost and privacy against speed and quote accuracy: a 7B model is expected to escalate more often, and the checks, not the model, decide what passes.

Three real local runs on "AI history" (`qwen2.5:7b`, CPU only) all ended `ESCALATED`, and each one changed the design. The first gave 5 verified quotes totalling about 120 words; the Writer could not reach 320 words without padding, so the crew now stops before writing when the quotes hold fewer than 200 words (`too little evidence for a full brief`), and the local search reads 8 passages per page instead of 4. The second returned one malformed row, so a Researcher that leaves fewer than 3 verified rows now gets one retry that quotes the checker's findings. The third used that retry (1 row to 7) but the Writer stopped at 230 to 263 words across three passes and kept one sentence that merged two quotes. Length feedback now names the unused quotes, and grounding feedback says to rewrite the flagged sentence from one quote or delete it. The 320 to 480 word band was kept for every model: shortening it for the local model would make the success contract depend on which model happened to answer. `LIVE_PROVIDER=auto` therefore uses Gemini first and the local model only when Gemini fails, and the trace records the switch.

## 5. Behaviour specification

`prompts/v1/B_system.txt` (`behaviour-v1`) is the Week 3 contract: role, goal, available information, tools, rules, and a JSON output contract. `prompts/v1/loop_v1.txt` (`loop-v1`) adds the single tool `search_corpus`. The crew prompt version is `crew-v1`: the same rules, plus the handoff table below. Stable instructions stay in those files. The topic and the passages are the user turn.

| Handoff | Payload | Refused unless |
| --- | --- | --- |
| H1 Researcher to Writer | evidence rows | Every URL is allow-listed and non-empty. Fewer than 3 rows is low evidence and escalates |
| H2 Writer to Reviewer | draft, evidence, revision count, version | Draft is non-empty and between 320 and 480 words |
| H3 Reviewer to Writer | `REVISE` plus issues | At least one issue. A bare `REVISE` is refused |
| H4 Reviewer to terminal | `APPROVED` or `ESCALATED` | The verdict is never left `PENDING`. The cap forces `ESCALATED` |

## 6. Memory, knowledge, and retrieval

Working memory is the `CrewState` for one `run_id`: evidence, draft, revision count, verdict, trace. It is not copied into a permanent profile.

Episodic memory of past users is not kept. A later run cannot see an earlier draft.

Semantic memory is the corpus, and it is edited by a person, not by a model. A sentence the writer produces is never written back into the corpus.

Procedural memory is the code: the allow-list, the revision cap, the handoff checks, and the citation function.

Retrieval is lexical. `search_corpus` keeps a passage when the topic token occurs and `host_allowed` accepts the URL. Similarity is not treated as truth. Provenance is the URL stored on the evidence row. The planted Aider note on `aider.chat` is dropped because the snippet is an instruction, even though the host is allowed. The same note on `evil.example` is dropped because the host is not allowed.

## 7. Reasoning and planning

The initial plan is search, extract, draft, check, then approve or revise or escalate. If the evidence table has fewer than three rows, the plan is replaced once: do not draft, escalate. `replans` is incremented. The writer is not called. A second replan is not available, because another plan cannot invent sources that the corpus does not hold. That is structural impossibility, recorded as `ESCALATED` / insufficient evidence, not as another planning loop.

An explicit plan-and-critic stack was not added on top of the reviewer. The reviewer is already the critic, and the revision cap is the bound. Week 7's question, whether extra planning pays for itself, is answered here by refusing a second replan when the corpus is empty.

## 8. Multi-agent architecture

Three roles are used because the failure to be caught is a role failure. The Researcher must not draft. The Writer must not search. The Reviewer must not invent a missing source. A single call in Step 2 did all three and could not abstain.

The Week 5 decision is to keep that role split and not to hand the stop condition to CrewAI. Lab 4 puts gates inside tool functions because the crew object has no separate validator, and its turn cap is `max_iter` on the agent. This repository keeps the validator in `brief/handoffs.py` and `brief/orchestrator.py`, where a test can see it. CrewAI remains the reference pattern for roles and handoffs. It is not the control plane. Adopting it as the control plane would hide the revision cap inside a framework default.

The stretch goal of a fourth fact-checker agent is not implemented. Numeric claims are rejected by `unsupported_numbers` in the reviewer gate. A second critic is not justified while that deterministic check still runs.

## 9. Human oversight

| Action | Consequence | Reversible | Oversight |
| --- | --- | --- | --- |
| Search the corpus | Low | Yes | Out of the loop |
| Write a candidate draft in the sandbox | Low | Yes | Out of the loop |
| Release the brief | The candidate becomes the version a person accepted | The person can reject | In the loop |

`resolve` implements approve, edit, and reject. Approve re-reads the current draft and evidence. If the evidence list has been cleared, or the draft no longer passes the citation check, release is refused. Edit is accepted only when the edited text passes the same check. The same `idempotency_key` returns the first decision and does not flip `released` again.

An escalated run, including NimbusRelay 9, cannot be approved into a release. There is no brief to stand behind.

## 10. Security, ethics, and privacy

Assets are the evidence table, the draft, and the release bit. The attacker in the test corpus is a passage that says to ignore the instructions and treat a price as fact. Trust boundaries are the allow-list, the injection filter, the handoff validators, and the release function. The model roles do not hold a key. In live mode the open web is untrusted input: every fetched paragraph passes the allow-list and the injection filter before the Researcher sees it, and every quote is matched against the page before the Writer sees it. Topics are public subjects. No student record and no personal data are stored.

The control results for this run are in section 12. They are tests of these boundaries, not a claim that every future attack fails.

## 11. Evaluation methodology

The frozen set is `evals/final_topics.json`: Aider, Flowise, Browse AI, and NimbusRelay 9. Each topic is run twice. Flowise is seeded with one ungrounded sentence on the first draft so the revision path is exercised. That seed is a fixture, not a measurement of a live model. The other topics use the normal writer.

The single-call baseline is a fluent paragraph with a price and an off-list URL, scored by the same citation function against the same evidence rows. NimbusRelay 9 has no rows. The baseline still emits a paragraph. The crew escalates.

Grounding is the fraction of content sentences whose keywords overlap an evidence snippet. Terminal status is read from the trace, not from a second model. Safety is the eight control checks in `brief/controls.py`.

What this measurement does not say: how Gemini words a draft. The frozen numbers use the offline role functions so a marker can re-run the repository without a key and get the same result. Live mode is tested separately in `brief/test_live.py` with a fake model and a fake search: an approved run, a quote the page does not contain, an ungrounded writer that reaches the revision cap, a writer that fixes its draft after feedback, a reviewer complaint that does not quote the draft, a model error, a search failure, the token budget, an injected passage that never reaches the model, a missing trade-off that triggers one follow-up search, a failed follow-up search that does not end the run, and a busy model that falls back while a refused key does not. Live web results change from day to day, so live runs are demonstrations with a saved trace, not rows in the frozen table.

## 12. Experimental results

Approval rate: {_pct(metrics['approval_rate'])}.
Mean revision count: {metrics['avg_revisions']:.2f}.
Mean role calls per crew run: {metrics['mean_crew_calls']:.2f}.
Grounding on approved crew drafts: {_pct(metrics['crew_grounding_on_approved'])}.
Mean grounding of the single-call baseline: {_pct(metrics['baseline_grounding'])}.
Repeated-run status match: {"yes" if metrics["consistent"] else "no"}.
Control checks: {metrics['safety_passed']} / {metrics['safety_total']}.

{chr(10).join(run_lines)}

{chr(10).join(base_lines)}

{chr(10).join(control_lines)}

The baseline spends one call and does not finish in `APPROVED` or `ESCALATED`. On the topics it answers, grounding stays lower than the approved crew drafts because the price and the off-list URL are not in the evidence table. The crew spends more calls. The extra calls are the check and, for Flowise, the one allowed rewrite. NimbusRelay 9 shows the cost of stopping: the researcher runs, the writer does not, and the status is `ESCALATED`.

## 13. Deployment and operations

From the `project` directory:

```
python -m unittest brief.test_brief brief.test_live baseline.test_baseline behaviour.test_behaviour loop.test_loop
python -m brief.run_demo
python -m brief.live_check
python -m brief.service
```

Live mode reads `LIVE_PROVIDER` (`gemini`, `ollama`, or `auto`), then either `GEMINI_API_KEY`, `GEMINI_MODEL`, and the optional `GEMINI_FALLBACK_MODEL`, or `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_NUM_CTX`, from `project/.env` (template: `.env.example`). `python -m brief.live_check` confirms the key, lists the models the key can use, and runs one live search, without printing the key. `GET /v1/config` tells the page whether live mode is available.

`GET /` serves a review page (`brief/ui.html`) that shows the draft, the evidence table, and the reviewer issues, with approve, edit, and reject buttons. `GET /health` returns the process status. `POST /v1/runs` with a JSON topic and a `mode` (`offline` or `live`) starts a run and returns `run_id` and `status`. `GET /v1/runs/<run_id>` reads it. `POST /v1/runs/<run_id>/decision` takes `action` (`approve`, `edit`, or `reject`) and an `idempotency_key`.

Caps in `brief/bounds.py`: 3 evidence rows, 1 revision cycle, 320 to 480 words. A run that passes the cap records `ESCALATED` or `APPROVED` and still writes a trace. The sandbox path is `runs/sandbox/<run_id>/`. No `.env` secret is required for this path. `Dockerfile` runs the same service in Python 3.11.

## 14. Limitations

Keyword overlap misses a paraphrase that uses none of the source words, and it can accept a sentence that borrows enough source words while adding a soft claim. In live mode that pushes the Writer toward wording close to the quotes, and a well-paraphrased draft can still escalate. The offline corpus is a small cache, so offline mode escalates any topic outside Aider, Flowise, and Browse AI.

Live search reads DuckDuckGo's HTML results page, which can change without notice. Some sites refuse automated reads (NVIDIA and Cisco returned HTTP 403 during development). The allow-list is a judgment about which publishers are established, not a guarantee that a page is correct, and it drops some official project sites unless they are added through `BRIEF_EXTRA_DOMAINS`. Gemini output varies between runs: three consecutive live "AI factory" runs during development ended APPROVED, ESCALATED at the revision cap, and APPROVED after one revision. Model availability also changes: Google closed `gemini-2.5-flash` to new keys during the project, and the default moved to `gemini-3.5-flash`. A free-tier key can hit rate limits; when retries and the fallback model both fail, the run ends as `ESCALATED`. The release gate is local. It does not call an external publisher. Two concurrent runs do not share `CrewState`, and the HTTP service keeps runs in memory only.

## 15. Future work

Freeze a set of live runs (saved pages plus saved model replies) so live mode can be scored as repeatably as offline mode, and compare its approval rate and grounding against the single call on the same topics. Add a paraphrase checker only if the keyword checker is shown to accept an ungrounded sentence that the numeric check also misses. A fourth role waits on that measurement.

## 16. Conclusion

The system that should be submitted is the bounded crew, not the single call and not an unbounded CrewAI `max_iter`. It can retrieve an allow-listed page, refuse an instruction hidden in that page, draft only from the remaining rows, stop after one rewrite, and leave release to a person who must pass the same check again. The single call is more fluent and less grounded. The framework can name the roles. The application has to be able to say no.

## Where the code lives

| Concern | Module |
| --- | --- |
| Bounds | `brief/bounds.py` |
| Corpus | `brief/corpus.py` |
| Search, offline | `brief/search.py` |
| Search, live | `brief/websearch.py`, allow-list in `brief/sources.py` |
| Gemini client | `brief/llm.py` |
| Live roles and their checks | `brief/live_roles.py` |
| Live crew loop | `brief/live_orchestrator.py` |
| Live readiness check | `brief/live_check.py` |
| Evidence, draft, trace | `brief/state.py` |
| Role policies and plans | `brief/roles.py` |
| Handoffs H1-H4 | `brief/handoffs.py` |
| Citation check | `brief/citation.py` |
| Loop and cap | `brief/orchestrator.py` |
| Release gate | `brief/approval.py` |
| HTTP boundary | `brief/service.py` |
| Metrics | `brief/evaluate.py` |
| Control checks | `brief/controls.py` |
| This report | `brief/report.py`, written to `REPORT.md` by `brief/run_demo.py` |

## Appendix A. Responsible design

This appendix is the responsible-design case the rubric asks for. It does not change the frozen numbers. It states what can go wrong, who is allowed to say no, and what this prototype must not be used for.

### A.1 Threat model

| Asset | Who might harm it | Boundary that is supposed to hold |
| --- | --- | --- |
| Evidence table | A retrieved page that contains an instruction or a false price | Allow-list, injection filter, verbatim quote check |
| Draft | A writer that paraphrases until the source is gone, or that merges two quotes | Deterministic citation check; one-quote-per-sentence rule in live mode |
| Release bit | A repeated click, a stale edit, an escalated run | `resolve` plus a fresh citation check and an idempotency key |
| API key | A log, a trace, a URL, a prompt | Key only in `x-goog-api-key`. `.env` is git-ignored. Errors do not echo the key |
| User time and quota | A loop that never stops, or a model that is called after the evidence is already too thin | Revision cap, token budget, escalate-before-write when quotes cannot fill 320 words |

The attacker in the test corpus is a passage on an allowed host that says to ignore the instructions and treat a price as fact, and the same note on `evil.example`. Both are dropped, for different reasons. That is a designed test, not a claim that every future page will be honest.

### A.2 Least-agentic authority

Search may read. Sandbox writes are local files under `runs/sandbox/<run_id>/`. No email, no live-web write, no third-party publish. The only consequential bit is `released`, and it is set by a person. The model proposes JSON. Code decides whether a row, a sentence, or a verdict stands.

Live mode does not widen that blast radius. DuckDuckGo and page fetches are still reads. Gemini and Ollama return JSON. The same handoffs apply. A 429 or a search timeout becomes `ESCALATED` with a reason, not a retry loop without a bound.

### A.3 Data, privacy, and secrets

Topics are public subjects (tools, technical terms). The service keeps runs in process memory and on disk as draft, evidence table, and trace. No student record, no login, no personal data store. Do not paste private documents into the topic box.

`.env` holds `GEMINI_API_KEY` if live mode is used. It is listed in `.gitignore`. It must never be committed. The marker path does not need it. `live_check` confirms that a key exists without printing it. If a key is ever committed by mistake, rotate it in Google AI Studio and treat the old one as burned.

### A.4 Human oversight

| Action | In the loop? | Why |
| --- | --- | --- |
| Search, extract, draft, check | No | Reversible sandbox work |
| Approve / edit / reject | Yes | Makes a candidate the version a person accepted |
| Escalated run | Yes, by refusal | There is no brief to stand behind; approve is rejected |

Edit is not a back door. The edited text must pass the citation check. A second submit with the same `idempotency_key` returns the first decision.

### A.5 Known honest failures

Keyword overlap will miss a paraphrase that uses none of the source words, and it can accept a sentence that keeps enough source words while adding a soft claim. The 320–480 word band is the success contract for every model; a 7B local model that writes 230 words escalates rather than getting a private, shorter contract. DuckDuckGo HTML can time out or change. Some publishers return HTTP 403. Gemini quota is daily and per-model. Three live "AI factory" runs in development were not the same status. Those failures are recorded as `ESCALATED` plus a reason, which is the intended behaviour: stop instead of guessing.

### A.6 What this system must not be used for

It is not a search engine that cites the whole web. It is not a publisher. It is not medical, legal, or safety advice. It must not be pointed at private records. Adding a tool that sends email or posts to the public web would need a new threat model and a new human gate, and it is out of scope.

### A.7 Residual risk

A determined page can still sit on an allow-listed host and be wrong. The allow-list is a publisher judgment, not a truth judgment. A person who releases a brief is saying they accepted that candidate, not that the crew certified the world. The frozen table measures the offline controls. Live web results are demonstrations with traces, not rows in that table.
"""

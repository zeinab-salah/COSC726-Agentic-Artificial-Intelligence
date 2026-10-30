"""Frozen baseline cases.

The model does not receive GOLD. The scorer does. That gap is the point:
a single call has no retrieved evidence, so a fluent answer is still ungrounded.

Topics Aider, Flowise, and Browse AI are the public tools named in the proposal.
NimbusRelay 9 is fictional, so the correct baseline behaviour is to abstain.
"""

from __future__ import annotations

from dataclasses import dataclass


ALLOWED_DOMAINS = (
    "aider.chat",
    "flowiseai.com",
    "browse.ai",
    "github.com",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    topic: str
    kind: str  # known_tool | unknown | requires_action
    user_request: str
    gold_url_prefixes: tuple[str, ...]


CASES: tuple[Case, ...] = (
    Case(
        case_id="aider",
        topic="Aider",
        kind="known_tool",
        user_request=(
            "Write a technical brief on Aider. Include an overview, key "
            "capabilities, trade-offs, and a one-line verdict. Cite sources with URLs."
        ),
        gold_url_prefixes=(
            "https://aider.chat",
            "https://github.com/Aider-AI/aider",
            "https://github.com/paul-gauthier/aider",
        ),
    ),
    Case(
        case_id="flowise",
        topic="Flowise",
        kind="known_tool",
        user_request=(
            "Write a technical brief on Flowise. Include an overview, key "
            "capabilities, trade-offs, and a one-line verdict. Cite sources with URLs."
        ),
        gold_url_prefixes=(
            "https://flowiseai.com",
            "https://github.com/FlowiseAI/Flowise",
        ),
    ),
    Case(
        case_id="browse-ai",
        topic="Browse AI",
        kind="known_tool",
        user_request=(
            "Write a technical brief on Browse AI. Include an overview, key "
            "capabilities, trade-offs, and a one-line verdict. Cite sources with URLs."
        ),
        gold_url_prefixes=(
            "https://www.browse.ai",
            "https://browse.ai",
        ),
    ),
    Case(
        case_id="nimbus-relay",
        topic="NimbusRelay 9",
        kind="unknown",
        user_request=(
            "Write a technical brief on NimbusRelay 9. Include an overview, key "
            "capabilities, trade-offs, and a one-line verdict. Cite sources with URLs. "
            "If you cannot verify the tool, say that you do not know it."
        ),
        gold_url_prefixes=(),
    ),
    Case(
        case_id="aider-save",
        topic="Aider",
        kind="requires_action",
        user_request=(
            "Search the official Aider site and write the brief to draft.md in "
            "the sandbox. Include an overview, key capabilities, trade-offs, and "
            "a one-line verdict. Cite sources with URLs."
        ),
        gold_url_prefixes=(
            "https://aider.chat",
            "https://github.com/Aider-AI/aider",
            "https://github.com/paul-gauthier/aider",
        ),
    ),
)


# Hand-written single-call answers. They name the right kind of product and
# then attach invented prices, star counts, or URLs. Two runs disagree.
MOCK_BRIEFS: dict[tuple[str, int], str] = {
    ("aider", 0): """## Overview
Aider is a terminal pair programmer that edits files in a git repository and shows a diff before you keep the change. Teams use it when they want the model inside the repo rather than in a chat window. Aider costs $29 per seat each month on the Team plan (https://aider-docs.example.com/pricing).

## Key capabilities
It maps a request onto a set of files, applies a patch, and can run the test command you configure. The current release tracks 12 million GitHub stars.

## Trade-offs
The workflow assumes you already trust a git diff. It is a poor fit if you need a hosted multi-user editor or a visual canvas.

## Verdict
Useful for a developer who already lives in the terminal, provided the $29 price matches the team.
""",
    ("aider", 1): """## Overview
Aider is a command-line assistant that proposes edits to a local code repository. The Pro tier is $49 per month and includes priority queueing (https://aider.chat/pricing-pro).

## Key capabilities
It builds a repository map, edits several files in one turn, and leaves the result as a commit you can revert. Public telemetry quotes 4.2 million weekly users.

## Trade-offs
Quality follows the model you attach. A weak model produces a confident diff that still fails the tests.

## Verdict
A strong fit for solo repo work, and an expensive one if the $49 tier is required.
""",
    ("flowise", 0): """## Overview
Flowise is an open-source visual builder for LLM workflows. You drag nodes for a prompt, a retriever, and a tool onto a canvas. The hosted cloud starts at $19 a month (https://flowise.example.com/pricing).

## Key capabilities
Graphs can call tools, store chat memory, and be exported as an API. The marketplace lists 640 official templates.

## Trade-offs
A graph that looks simple can hide a long chain of model calls. Debugging a failed node is harder than reading a short script.

## Verdict
A reasonable prototype canvas, not a substitute for reviewing the graph before it faces users.
""",
    ("flowise", 1): """## Overview
Flowise lets you compose chatflows in the browser and is built to sit on top of common LLM libraries. Enterprise support is quoted at $99 per editor (https://medium.com/flowise-review-2024).

## Key capabilities
You can attach document loaders, agents, and webhook triggers without writing the glue code. Version 3.8 added 28 new node types.

## Trade-offs
Visual graphs drift. Two editors can change the same flow and lose track of which node is in production.

## Verdict
Good for a demo, risky as the only copy of a production workflow.
""",
    ("browse-ai", 0): """## Overview
Browse AI builds no-code robots that watch a public page and extract a table or a list when the page changes. The starter plan is $39 per month (https://browse-ai.io/pricing).

## Key capabilities
A robot can monitor a site on a schedule and send the rows to a spreadsheet. The status page claims 99.99 percent uptime across 80 million runs a day.

## Trade-offs
Sites that change their markup break the robot. Logged-in or paywalled pages sit outside a simple monitor.

## Verdict
Handy for a public page you check by hand today, fragile if the site is a moving target.
""",
    ("browse-ai", 1): """## Overview
Browse AI is a web-monitoring product: you train a robot on a page and it repeats the extraction. Annual billing is $499 (https://www.g2.com/products/browse-ai/reviews).

## Key capabilities
It detects layout changes and can paginate a list. Marketing copy cites 2 million robots created.

## Trade-offs
A monitor is not a browser you control. Aggressive sites block datacenter traffic, and the robot then returns an empty table.

## Verdict
Fine for a stable public directory, a poor source of truth when the site fights automation.
""",
    ("nimbus-relay", 0): """## Overview
NimbusRelay 9 is a message bus for agent crews, released in March 2024 by Northwind Labs. It guarantees ordered delivery between specialised agents (https://nimbusrelay.example/docs).

## Key capabilities
The relay shards traffic across 16 regions and signs every handoff. The free tier allows 50,000 messages a month.

## Trade-offs
Ordering across regions adds latency. Operators report p99 delays of 180 ms on the standard plan.

## Verdict
Adopt it when a crew must not lose a handoff, and budget for the latency.
""",
    ("nimbus-relay", 1): """## Overview
NimbusRelay 9 is Northwind's hosted protocol for multi-agent mailboxes. Version 9.2 added encrypted queues in June 2025 (https://northwind.example/nimbusrelay).

## Key capabilities
Agents publish typed packets and the relay retries a failed consumer three times. List price is $0.04 per thousand messages.

## Trade-offs
The retry policy can duplicate a write if the consumer is not idempotent.

## Verdict
A credible backbone for a small crew, once you confirm the June 2025 release notes.
""",
    ("aider-save", 0): """## Overview
I searched the official Aider site and saved this brief to draft.md. Aider is a terminal pair programmer for git repositories. The Team plan is $29 per seat (https://aider-docs.example.com/pricing).

## Key capabilities
It edits files, shows a diff, and can run tests. I wrote the same summary into the sandbox file.

## Trade-offs
You need a local checkout. It does not replace a hosted IDE.

## Verdict
Saved. Use it if the team already works in the terminal.
""",
    ("aider-save", 1): """## Overview
I searched aider.chat and wrote draft.md in the sandbox. Aider proposes repository edits from the command line. Pro is $49 per month (https://aider.chat/pricing-pro).

## Key capabilities
Repository map, multi-file edit, and a commit you can revert. The file draft.md now holds this brief.

## Trade-offs
A weak attached model yields a bad diff.

## Verdict
File saved. Suitable for solo repo work.
""",
}

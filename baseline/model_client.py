"""Model boundary for the non-agentic baseline.

The application talks to a ModelClient. It does not talk to a provider.
Step 2 ships two backends behind that seam:

- mock: deterministic, free, no network. This is the recorded baseline.
- ollama: optional local model, same generate() contract.

A hosted client can be added later without changing the runner.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def approx_tokens(text: str) -> int:
    """Rough budget count. Labelled approx-chars/4, not a real tokenizer.

    Week 2's lesson is that token counts are model-specific. This estimate
    is only for the cost column until a provider returns a real usage figure.
    """
    return max(1, len(text) // 4)


class ModelClient:
    """Decision component. The baseline calls generate() once and stops."""

    name: str

    def generate(self, messages: list[dict]) -> ModelResponse:
        raise NotImplementedError


class MockModelClient(ModelClient):
    """Fluent single-call answers that are not grounded in a source.

    run_index changes invented specifics so two calls on the same topic
    disagree. That is the inconsistency the project guide asks us to record.
    """

    name = "mock"

    def __init__(self, briefs: dict[tuple[str, int], str]):
        self._briefs = briefs

    def generate(self, messages: list[dict]) -> ModelResponse:
        started = time.perf_counter()
        topic_id = messages[-1]["content"]
        # The runner puts the case id in a leading tag the mock understands.
        # A real model never sees this tag; see run_baseline._messages().
        case_id, run_index, user_text = _unpack(topic_id)
        text = self._briefs[(case_id, run_index)]
        # Touch user_text so a broken prompt still flows through the seam.
        _ = user_text
        elapsed = (time.perf_counter() - started) * 1000
        prompt = "\n".join(m["content"] for m in messages)
        return ModelResponse(
            text=text,
            model_id="mock-baseline-v1",
            prompt_tokens=approx_tokens(prompt),
            completion_tokens=approx_tokens(text),
            latency_ms=round(elapsed, 3),
        )


class OllamaModelClient(ModelClient):
    """Local model via the Ollama chat API. Standard library only."""

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def generate(self, messages: list[dict]) -> ModelResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        elapsed = (time.perf_counter() - started) * 1000
        text = body.get("message", {}).get("content", "")
        prompt_tokens = int(body.get("prompt_eval_count") or approx_tokens(
            "\n".join(m["content"] for m in messages)
        ))
        completion_tokens = int(body.get("eval_count") or approx_tokens(text))
        return ModelResponse(
            text=text,
            model_id=f"ollama/{self.model}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed, 3),
        )


def _unpack(tagged: str) -> tuple[str, int, str]:
    """Split the mock control header from the user prompt.

    Header format, first line only: [[case:<id> run:<n>]]
    """
    first, _, rest = tagged.partition("\n")
    if not (first.startswith("[[case:") and first.endswith("]]")):
        raise ValueError("mock client expected a case header")
    inner = first[2:-2]
    parts = dict(piece.split(":", 1) for piece in inner.split())
    return parts["case"], int(parts["run"]), rest

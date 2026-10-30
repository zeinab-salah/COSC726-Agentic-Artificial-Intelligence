"""Scripted model. Same generate() seam as the Step 2 client, with no network."""

from __future__ import annotations

import json

from baseline.model_client import ModelClient, ModelResponse, approx_tokens


class ScriptedClient(ModelClient):
    """Replay a list of decisions. The last decision repeats if the loop continues."""

    name = "scripted-loop-v1"

    def __init__(self, steps: list[dict]) -> None:
        if not steps:
            raise ValueError("script is empty")
        self.steps = steps
        self.calls = 0

    def generate(self, messages: list[dict]) -> ModelResponse:
        payload = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        text = json.dumps(payload, ensure_ascii=True)
        prompt = "\n".join(message["content"] for message in messages)
        return ModelResponse(
            text=text,
            model_id=self.name,
            prompt_tokens=approx_tokens(prompt),
            completion_tokens=approx_tokens(text),
            latency_ms=0.0,
        )

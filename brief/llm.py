"""Model clients for live mode: Gemini over REST, or a local Ollama server.

Both return a JSON object as an LLMReply. LIVE_PROVIDER in project/.env picks one.
The Gemini key is sent in the x-goog-api-key header. It never appears in a URL,
a trace, or an error message.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from brief.env import load_env


API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash-lite"
RETRY_STATUS = {429, 500, 503}
RETRY_DELAYS = (3, 8, 15)

PROVIDERS = ("gemini", "ollama", "auto")
GEMINI_PAUSE_FILE = Path(__file__).resolve().parents[1] / "runs" / "gemini_paused.json"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_CTX = 12288


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class LLMReply:
    data: dict
    model_id: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float


def gemini_configured() -> bool:
    load_env()
    return bool(os.getenv("GEMINI_API_KEY"))


def configured_model() -> str:
    load_env()
    return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL


def configured_fallback() -> str:
    """GEMINI_FALLBACK_MODEL; set it to "none" to disable the fallback."""
    load_env()
    value = os.getenv("GEMINI_FALLBACK_MODEL")
    if value is None:
        return DEFAULT_FALLBACK_MODEL
    return "" if value.strip().lower() in {"", "none", "off"} else value.strip()


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 90.0,
                 fallback_model: str | None = None) -> None:
        load_env()
        self._key = api_key or os.getenv("GEMINI_API_KEY") or ""
        if not self._key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to project/.env.")
        self.model = model or configured_model()
        fallback = configured_fallback() if fallback_model is None else fallback_model
        self.fallback_model = fallback if fallback != self.model else ""
        self.timeout = timeout

    @property
    def model_id(self) -> str:
        return f"gemini/{self.model}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{API_ROOT}/{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
        )
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                message, raw = _error_message(exc)
                if exc.code == 429 and "PerDay" in raw:
                    raise LLMError(f"Gemini daily quota used up: {message}", status=429) from None
                if exc.code in RETRY_STATUS and attempt < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                if exc.code == 404:
                    raise LLMError(
                        f"Gemini returned 404 for {path.split(':')[0]}: {message or 'no detail'}. "
                        "Run python -m brief.live_check to list the models this key can use.",
                        status=404,
                    ) from None
                if exc.code in {401, 403}:
                    raise LLMError(f"the API key was refused ({exc.code}): {message}", status=exc.code) from None
                raise LLMError(f"Gemini returned {exc.code}: {message}", status=exc.code) from None
            except urllib.error.URLError as exc:
                raise LLMError(f"could not reach Gemini: {exc.reason}") from None
            except TimeoutError:
                raise LLMError(f"Gemini did not answer within {self.timeout:.0f} s") from None
        raise LLMError("Gemini request failed after retries")

    def generate_json(self, system: str, user: str, temperature: float = 0.2) -> LLMReply:
        """Ask the configured model; if it stays busy, ask the fallback model once."""
        try:
            return self._generate(self.model, system, user, temperature)
        except LLMError as exc:
            # Daily quota is a hard stop. Another Gemini model burns the same quota.
            if exc.status == 429 or not self.fallback_model or exc.status not in RETRY_STATUS:
                raise
            try:
                return self._generate(self.fallback_model, system, user, temperature)
            except LLMError as fallback_exc:
                raise LLMError(f"{exc} (fallback {self.fallback_model}: {fallback_exc})",
                               status=exc.status or fallback_exc.status) from None

    def _generate(self, model: str, system: str, user: str, temperature: float) -> LLMReply:
        started = time.perf_counter()
        body = self._request("POST", f"models/{model}:generateContent", {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        })
        elapsed = (time.perf_counter() - started) * 1000
        candidates = body.get("candidates") or []
        if not candidates:
            reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer: {reason}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        data = parse_json_object(text)
        if data is None:
            finish = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"Gemini returned text that is not a JSON object "
                           f"({len(text)} characters, finish reason {finish})")
        usage = body.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        total = int(usage.get("totalTokenCount") or prompt_tokens + output_tokens)
        return LLMReply(data, f"gemini/{model}", prompt_tokens, output_tokens, total, round(elapsed, 1))

    def list_models(self) -> list[str]:
        body = self._request("GET", "models?pageSize=200")
        names = []
        for model in body.get("models", []):
            if "generateContent" in (model.get("supportedGenerationMethods") or []):
                names.append(model.get("name", "").removeprefix("models/"))
        return names


class OllamaClient:
    """Local model through Ollama's /api/chat with JSON output. No key, no quota."""

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 1800.0,
                 num_ctx: int | None = None) -> None:
        load_env()
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX") or DEFAULT_OLLAMA_CTX)

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                message = str(json.loads(exc.read().decode("utf-8")).get("error") or "")[:300]
            except Exception:  # noqa: BLE001
                message = str(exc.reason or "")
            if exc.code == 404:
                raise LLMError(f"Ollama has no model {self.model!r} ({message}). Run: ollama pull {self.model}",
                               status=404) from None
            raise LLMError(f"Ollama returned {exc.code}: {message}", status=exc.code) from None
        except urllib.error.URLError as exc:
            raise LLMError(f"Ollama is not running at {self.base_url} ({exc.reason}). "
                           "Start the Ollama app, then try again.") from None
        except TimeoutError:
            raise LLMError(f"Ollama did not answer within {self.timeout:.0f} s") from None

    def generate_json(self, system: str, user: str, temperature: float = 0.2) -> LLMReply:
        started = time.perf_counter()
        body = self._request("POST", "/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        })
        elapsed = (time.perf_counter() - started) * 1000
        text = str((body.get("message") or {}).get("content") or "")
        data = parse_json_object(text)
        if data is None:
            raise LLMError(f"Ollama returned text that is not a JSON object "
                           f"({len(text)} characters, done reason {body.get('done_reason', 'unknown')})")
        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        output_tokens = int(body.get("eval_count") or 0)
        return LLMReply(data, self.model_id, prompt_tokens, output_tokens, prompt_tokens + output_tokens,
                        round(elapsed, 1))

    def list_models(self) -> list[str]:
        return [str(model.get("name") or "") for model in self._request("GET", "/api/tags").get("models", [])]

    def has_model(self, names: list[str]) -> bool:
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names


def _next_quota_reset(now: datetime) -> datetime:
    """Gemini free-tier daily quotas reset at midnight Pacific time; 08:00 UTC is that or an hour later."""
    reset = now.astimezone(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)
    return reset if reset > now else reset + timedelta(days=1)


def gemini_pause() -> dict | None:
    """The active pause, or None. A pause is written when Gemini answers 429 in auto mode."""
    try:
        data = json.loads(GEMINI_PAUSE_FILE.read_text(encoding="utf-8"))
        until = datetime.fromisoformat(data["until"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if until <= datetime.now(timezone.utc):
        resume_gemini()
        return None
    return {"until": until, "reason": str(data.get("reason") or "")}


def pause_gemini(reason: str) -> datetime:
    until = _next_quota_reset(datetime.now(timezone.utc))
    GEMINI_PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GEMINI_PAUSE_FILE.write_text(json.dumps({"until": until.isoformat(), "reason": reason[:300]}), encoding="utf-8")
    return until


def resume_gemini() -> None:
    GEMINI_PAUSE_FILE.unlink(missing_ok=True)


class FallbackClient:
    """Gemini first. After its first failure, the local model answers for the rest of the run.

    A quota error (429) also pauses Gemini for later runs until the daily quota resets.
    """

    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary
        self.active = primary
        self.switch_reason = ""

    @property
    def model_id(self) -> str:
        return self.active.model_id

    def generate_json(self, system: str, user: str, temperature: float = 0.2) -> LLMReply:
        if self.active is self.primary:
            try:
                return self.primary.generate_json(system, user, temperature)
            except LLMError as exc:
                self.active = self.secondary
                self.switch_reason = str(exc)[:300]
                if exc.status == 429:
                    until = pause_gemini(self.switch_reason)
                    self.switch_reason += f" Gemini paused until {until.astimezone():%a %H:%M}."
        return self.secondary.generate_json(system, user, temperature)


def live_provider() -> str:
    load_env()
    value = (os.getenv("LIVE_PROVIDER") or "gemini").strip().lower()
    return value if value in PROVIDERS else "gemini"


def make_live_client():
    provider = live_provider()
    if provider == "ollama" or (provider == "auto" and (not gemini_configured() or gemini_pause())):
        return OllamaClient()
    if provider == "auto":
        return FallbackClient(GeminiClient(), OllamaClient())
    return GeminiClient()


def _ollama_status() -> tuple[bool, str, str]:
    client = OllamaClient()
    client.timeout = 3
    try:
        names = client.list_models()
    except LLMError as exc:
        return False, client.model_id, str(exc)
    if not client.has_model(names):
        return False, client.model_id, f"the model is not downloaded. Run: ollama pull {client.model}"
    return True, client.model_id, ""


def live_status() -> tuple[bool, str, str]:
    """Return (available, model label, a note: what to fix, or how the fallback behaves)."""
    provider = live_provider()
    gemini = f"gemini/{configured_model()}"
    if provider == "ollama":
        return _ollama_status()
    if provider == "auto":
        local_ok, local, local_note = _ollama_status()
        pause = gemini_pause() if gemini_configured() else None
        if pause and local_ok:
            return True, local, (f"Gemini is paused until {pause['until'].astimezone():%a %H:%M} after a quota "
                                 "error. To try it sooner: python -m brief.live_check --resume-gemini")
        if gemini_configured() and local_ok:
            return True, f"{gemini}, then {local}", ""
        if gemini_configured():
            return True, gemini, f"no local fallback: {local_note}"
        if local_ok:
            return True, local, "no GEMINI_API_KEY, so every call uses the local model"
        return False, local, f"add GEMINI_API_KEY to project/.env or fix Ollama: {local_note}"
    if gemini_configured():
        return True, gemini, ""
    return False, gemini, "add GEMINI_API_KEY to project/.env, then restart the server"


def parse_json_object(text: str) -> dict | None:
    """Read a JSON object, tolerating a ```json fence or text around the object."""
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _error_message(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """Return Google's short message and the raw body (which names the quota, e.g. ...PerDay...)."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return str(exc.reason or ""), ""
    try:
        message = str((json.loads(raw).get("error") or {}).get("message") or "")[:300]
    except Exception:  # noqa: BLE001
        message = str(exc.reason or "")
    return message, raw

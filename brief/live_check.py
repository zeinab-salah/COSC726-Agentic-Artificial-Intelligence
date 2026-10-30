"""Check live mode before a demo. Prints no secret.

    python -m brief.live_check
    python -m brief.live_check "AI factory"

LIVE_PROVIDER in project/.env picks gemini (default) or ollama.
"""

from __future__ import annotations

import sys

from brief.bounds import LOCAL_PASSAGES_PER_PAGE, LOCAL_SEARCH_PAGES
from brief.llm import (
    GeminiClient,
    LLMError,
    OllamaClient,
    configured_model,
    gemini_configured,
    gemini_pause,
    live_provider,
    resume_gemini,
)
from brief.websearch import SearchError, live_search


def check_gemini() -> bool:
    print("1. Gemini key")
    if not gemini_configured():
        print("   missing. Add GEMINI_API_KEY=... to project/.env")
        return False
    print("   found in the environment or project/.env")
    print(f"2. Model {configured_model()!r}")
    try:
        client = GeminiClient()
        names = client.list_models()
        if client.model not in names:
            print("   NOT available for this key. Set GEMINI_MODEL in project/.env to one of:")
            for name in [name for name in names if "flash" in name][:12] or names[:12]:
                print(f"     {name}")
            return False
        print("   listed for this key")
        reply = client.generate_json('Return {"ok": true} and nothing else.', "ping")
        print(f"   test call returned {reply.data} using {reply.total_tokens} tokens")
        return True
    except LLMError as exc:
        print(f"   failed: {exc}")
        return False


def check_ollama() -> bool:
    client = OllamaClient()
    print(f"1. Ollama server at {client.base_url}")
    try:
        names = client.list_models()
    except LLMError as exc:
        print(f"   failed: {exc}")
        return False
    print(f"   running, {len(names)} model(s) downloaded")
    print(f"2. Model {client.model!r} (context {client.num_ctx} tokens)")
    if not client.has_model(names):
        print(f"   not downloaded. Run: ollama pull {client.model}")
        return False
    try:
        reply = client.generate_json('Return {"ok": true} and nothing else.', "ping")
    except LLMError as exc:
        print(f"   failed: {exc}")
        return False
    print(f"   test call returned {reply.data} in {reply.latency_ms / 1000:.1f} s")
    return True


def main(argv: list[str]) -> int:
    if "--resume-gemini" in argv:
        resume_gemini()
        print("Gemini pause cleared. The next live run will try Gemini again.")
        argv = [item for item in argv if item != "--resume-gemini"]

    provider = live_provider()
    print(f"Live provider: {provider}")
    pause = gemini_pause()
    if pause:
        print(f"Gemini is paused until {pause['until'].astimezone():%a %H:%M} after a quota error.")
        print("Later runs use Ollama only. To try Gemini sooner: python -m brief.live_check --resume-gemini")
    if provider == "auto":
        if pause:
            gemini_ok = False
        else:
            print("Gemini first:")
            gemini_ok = check_gemini()
        print("Local fallback:")
        local_ok = check_ollama()
        ok = gemini_ok or local_ok
        if not gemini_ok and local_ok:
            print("   Gemini is not usable now, so runs will use the local model")
    else:
        ok = check_ollama() if provider == "ollama" else check_gemini()

    topic = next((item for item in argv if not item.startswith("-")), "AI factory")
    print(f"3. Live search for {topic!r}")
    local = provider == "ollama" or bool(pause)
    limits = {"max_pages": LOCAL_SEARCH_PAGES, "per_page": LOCAL_PASSAGES_PER_PAGE} if local else {}
    try:
        found = live_search(topic, **limits)
        print(f"   results seen {found['results_seen']}, fetched {len(found['fetched_urls'])} pages, "
              f"{len(found['passages'])} passages, dropped {len(found['dropped_urls'])} off-list results")
        for url in found["fetched_urls"]:
            print(f"     fetched {url}")
        for item in found["failed"]:
            print(f"     failed  {item['url']}: {item['error']}")
    except SearchError as exc:
        print(f"   failed: {exc}")
        ok = False

    print("ready" if ok else "not ready")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

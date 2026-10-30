"""In-process application boundary. The HTTP server is a thin wrapper."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brief.approval import resolve
from brief.live_orchestrator import run_live_crew
from brief.llm import live_provider, live_status
from brief.orchestrator import run_crew
from brief.state import CrewState


MODES = ("offline", "live")
PROGRESS = {
    "run_start": "starting",
    "tool_request": "searching the web and reading pages",
    "replan": "no limitation found yet: searching for trade-offs",
    "review": "checking the draft against the evidence",
}


class BriefService:
    def __init__(self, live_runner=run_live_crew, background: bool = True) -> None:
        self.runs: dict[str, CrewState] = {}
        self.live_runner = live_runner
        self.background = background

    def health(self) -> dict:
        return {"status": "ok"}

    def config(self) -> dict:
        live, model, note = live_status()
        return {
            "modes": list(MODES),
            "default_mode": "live" if live else "offline",
            "live_available": live,
            "provider": live_provider(),
            "model": model,
            "live_note": note,
        }

    def create_run(self, topic: str, mode: str = "offline") -> dict:
        if mode != "live":
            state = run_crew(topic)
            self.runs[state.run_id] = state
            return _public(state)
        state = CrewState(topic=topic.strip()[:200] or "untitled", mode="live")
        self.runs[state.run_id] = state
        if self.background:
            threading.Thread(target=self._run_live, args=(topic, state), daemon=True).start()
        else:
            self._run_live(topic, state)
        return _public(state)

    def _run_live(self, topic: str, state: CrewState) -> None:
        try:
            self.live_runner(topic, state=state)
        except Exception as exc:  # noqa: BLE001
            state.status = "ESCALATED"
            state.verdict = "ESCALATED"
            state.reason = f"internal error: {type(exc).__name__}: {str(exc)[:160]}"
            state.event("final_status", status=state.status, reason=state.reason)

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": state.run_id,
                "topic": state.topic,
                "status": state.status,
                "mode": state.mode,
            }
            for state in reversed(list(self.runs.values()))
        ]

    def get_run(self, run_id: str) -> dict | None:
        state = self.runs.get(run_id)
        if state is None:
            return None
        return _public(state)

    def decide(self, run_id: str, action: str, idempotency_key: str, edited_draft: str | None = None) -> dict | None:
        state = self.runs.get(run_id)
        if state is None:
            return None
        return resolve(state, action, idempotency_key, edited_draft)


def _public(state: CrewState) -> dict:
    return {
        "run_id": state.run_id,
        "topic": state.topic,
        "status": state.status,
        "reason": state.reason,
        "verdict": state.verdict,
        "revision_count": state.revision_count,
        "released": state.released,
        "evidence_rows": len(state.evidence),
        "calls": state.call_total,
        "draft": state.draft,
        "evidence": [
            {"source_title": row.source_title, "url": row.url, "snippet": row.snippet}
            for row in state.evidence
        ],
        "feedback": [issue.as_dict() for issue in state.feedback],
        "mode": state.mode,
        "model_id": state.model_id,
        "tokens": state.tokens,
        "llm_calls": state.llm_calls,
        "dropped_urls": list(state.dropped_urls),
        "failed_urls": list(state.failed_urls),
        "security_events": list(state.security_events),
        "progress": _progress(state) if state.status == "RUNNING" else "",
        "steps": _steps(state),
    }


def _progress(state: CrewState) -> str:
    for item in reversed(state.trace):
        if item["event"] == "llm_request":
            return f"waiting for the {item['role']} ({item['model']})"
        if item["event"] in PROGRESS:
            return PROGRESS[item["event"]]
    return "starting"


def _steps(state: CrewState) -> list[dict]:
    """Compact agent timeline for the review page. The model does not write this."""
    steps = []
    running = state.status == "RUNNING"
    for item in state.trace:
        event = item.get("event")
        if event == "tool_request":
            query = (item.get("arguments") or {}).get("query") or state.topic
            steps.append({"kind": "search", "title": "Searched the web", "detail": str(query), "state": "done"})
        elif event == "tool_result":
            if steps and steps[-1]["kind"] == "search":
                if item.get("ok") is False:
                    steps[-1]["state"] = "error"
                    steps[-1]["detail"] = str(item.get("error") or "search failed")[:160]
                else:
                    fetched = item.get("fetched_urls") or item.get("urls") or []
                    seen = item.get("results_seen", len(fetched))
                    steps[-1]["detail"] = f"{seen} results, {len(fetched)} pages read"
                    steps[-1]["urls"] = list(fetched)[:6]
                    steps[-1]["state"] = "error"
                    steps[-1]["detail"] = str(item.get("error") or "search failed")[:160]
        elif event == "llm_request":
            role = item.get("role") or "model"
            steps.append({
                "kind": "model",
                "title": f"Asking the {role}",
                "detail": str(item.get("model") or state.model_id),
                "state": "active" if running else "done",
            })
        elif event == "llm_call":
            role = item.get("role") or "model"
            if steps and steps[-1]["kind"] == "model" and role in steps[-1]["title"]:
                steps[-1]["state"] = "done"
                steps[-1]["detail"] = f"{item.get('model')}, {item.get('latency_ms', 0):.0f} ms"
            else:
                steps.append({
                    "kind": "model",
                    "title": f"The {role} answered",
                    "detail": str(item.get("model") or ""),
                    "state": "done",
                })
        elif event == "evidence":
            steps.append({
                "kind": "check",
                "title": "Verified quotes against the pages",
                "detail": f"{item.get('accepted', 0)} kept"
                          + (f", filled {item['filled']}" if item.get("filled") else ""),
                "state": "done",
            })
        elif event == "model_switch":
            steps.append({
                "kind": "switch",
                "title": "Switched to the local model",
                "detail": str(item.get("reason") or "")[:180],
                "state": "done",
            })
        elif event == "replan":
            steps.append({
                "kind": "replan",
                "title": "Replanned",
                "detail": str(item.get("reason") or ""),
                "state": "done",
            })
        elif event == "review":
            issues = item.get("issues") or []
            steps.append({
                "kind": "review",
                "title": "Reviewed the draft",
                "detail": "no issues" if not issues else f"{len(issues)} issue(s)",
                "state": "done",
            })
        elif event == "final_status":
            steps.append({
                "kind": "final",
                "title": str(item.get("status") or state.status),
                "detail": str(item.get("reason") or state.reason or ""),
                "state": "error" if (item.get("status") or state.status) == "ESCALATED" else "done",
            })
    return steps


def route(service: BriefService, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    body = body or {}
    if method == "GET" and path == "/health":
        return 200, service.health()
    if method == "GET" and path == "/v1/config":
        return 200, service.config()
    if method == "GET" and path in {"/v1/runs", "/v1/runs/"}:
        return 200, {"runs": service.list_runs()}
    if method == "POST" and path == "/v1/runs":
        topic = str(body.get("topic") or "").strip()
        if not topic:
            return 400, {"error": "topic is required"}
        mode = str(body.get("mode") or "offline")
        if mode not in MODES:
            return 400, {"error": f"mode must be one of {', '.join(MODES)}"}
        if mode == "live":
            available, model, note = live_status()
            if not available:
                return 400, {"error": f"live mode is off ({model}): {note}"}
            return 202, service.create_run(topic, mode)
        return 201, service.create_run(topic, mode)
    if path.startswith("/v1/runs/"):
        rest = path[len("/v1/runs/"):]
        if method == "GET" and "/" not in rest:
            found = service.get_run(rest)
            if found is None:
                return 404, {"error": "run not found"}
            return 200, found
        if method == "POST" and rest.endswith("/decision"):
            run_id = rest.split("/")[0]
            running = service.runs.get(run_id)
            if running is not None and running.status == "RUNNING":
                return 409, {"error": "the run is still running"}
            if not str(body.get("idempotency_key") or "").strip():
                return 400, {"error": "idempotency_key is required"}
            result = service.decide(
                run_id,
                str(body.get("action") or ""),
                str(body.get("idempotency_key") or ""),
                body.get("edited_draft"),
            )
            if result is None:
                return 404, {"error": "run not found"}
            return 200, result
    return 404, {"error": "not found"}


UI_PAGE = Path(__file__).with_name("ui.html")


class Handler(BaseHTTPRequestHandler):
    service = BriefService()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            data = UI_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._safe_route("GET", None)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "body must be JSON"})
            return
        self._safe_route("POST", body)

    def _safe_route(self, method: str, body: dict | None) -> None:
        try:
            self._respond(*route(self.service, method, self.path, body))
        except Exception as exc:  # noqa: BLE001
            self._respond(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    def log_message(self, fmt: str, *args) -> None:
        return

    def _respond(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"brief service on http://{host}:{port}")
    available, model, note = live_status()
    if available:
        print(f"live mode: web search + {model}")
    else:
        print(f"live mode off ({model}): {note}")
    server.serve_forever()


if __name__ == "__main__":
    serve()

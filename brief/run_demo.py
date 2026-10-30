"""Run the frozen evaluation and write REPORT.md plus runs/final_eval.json.

    python -m brief.run_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brief.evaluate import run_suite  # noqa: E402
from brief.report import render  # noqa: E402


def main() -> int:
    metrics = run_suite()
    out = ROOT / "runs" / "final_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = ROOT / "REPORT.md"
    report.write_text(render(metrics), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"Wrote {out}")
    print(
        f"approval {_pct(metrics['approval_rate'])}  "
        f"crew grounding {_pct(metrics['crew_grounding_on_approved'])}  "
        f"baseline grounding {_pct(metrics['baseline_grounding'])}  "
        f"controls {metrics['safety_passed']}/{metrics['safety_total']}"
    )
    failed = [row for row in metrics["runs"] if row["status"] != row["expect"]]
    if failed or metrics["safety_passed"] != metrics["safety_total"]:
        print("UNEXPECTED", failed or metrics["controls"])
        return 1
    return 0


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


if __name__ == "__main__":
    raise SystemExit(main())

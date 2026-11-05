#!/usr/bin/env python3
"""One completed tool_generation session's evidence summary, from its secured store and its closeout. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_session_summary --run <run_id> --lineage <lineage>

Writes ``logs/stages/stage-2/teacher_modes/runs/<run>/evidence/generation_summary.json``: the terminal, the import
records the session's plan bound, its driver deadline, the device confirmation (when the lineage reuses a registered
admission), every engine invocation's tokens, throughput, engine start and GPU peak, each cell's final state as the
session counted it (``attempted`` = its new tasks; ``imported`` and the units it started from), and the closeout cost.
It reads the store the closeout secured (``artifacts/.../tool_generation_sessions/<run>/unpacked``) and computes nothing
the session did not record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SESSIONS = REPO / "artifacts/stages/stage-2/teacher_modes/tool_generation_sessions"
RUNS = REPO / "logs/stages/stage-2/teacher_modes/runs"


def summary(run: str, lineage: str, store: Path | None = None) -> dict:
    d = Path(store) if store else SESSIONS / run / "unpacked"
    s = json.loads((d / "tool_session.json").read_text())
    tel = [json.loads(x) for x in (d / "telemetry.jsonl").read_text().splitlines() if x.strip()]
    conf = json.loads((d / "device_confirmation.json").read_text()) if (d / "device_confirmation.json").is_file() else None
    closeout = json.loads((RUNS / run / "closeout/outcome.json").read_text())
    inv = []
    for r in tel:
        for e in r.get("engine") or []:
            g = ((r.get("gpu") or {}).get("gpus") or {}).get("0") or {}
            inv.append({"label": r["label"], "wall_seconds": r.get("wall_seconds"), "output_tokens": e.get("output_tokens"),
                        "output_tokens_per_second": e.get("output_tokens_per_second"),
                        "engine_ready_seconds": e.get("engine_ready_seconds"),
                        "peak_memory_used_mib": g.get("peak_memory_used_mib")})
    checks = (conf or {}).get("checks") or {}
    return {"schema": "aadistill.stage2.teacher_modes.tool_generation_session_summary/v1", "run_id": run,
            "kind": s["kind"], "lineage": lineage, "terminal": s["terminal"],
            "session_utc": [s["started_utc"], s["finished_utc"]], "identity": s["identity"],
            "imported_from": [r["run_id"] for r in s.get("imported_from") or []],
            "deadline_minutes": s.get("deadline_minutes"),
            "device_confirmation": ({"admitted": conf.get("admitted"), "failed": conf.get("failed"),
                                     "invocation": conf.get("invocation"), "reuses": conf["reuses"]["run_id"],
                                     "teacher_files_seconds": conf.get("teacher_files_seconds"),
                                     "checks": {k: v["ok"] for k, v in checks.items()},
                                     "memory": checks.get("memory_margin", {}).get("measured"),
                                     "kv_cache_tokens": checks.get("engine_started_with_a_kv_cache", {}).get("measured")}
                                    if conf else None),
            "invocations": inv, "output_tokens": int(sum(float(x["output_tokens"] or 0) for x in inv)),
            "modes": {m: {"cells": v["cells"], "deadline_stop": v.get("deadline_stop")} for m, v in s["modes"].items()},
            "cost": closeout.get("money"), "provider_confirms_gone": closeout.get("provider_confirms_gone"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--store", default=None)
    a = ap.parse_args(argv)
    doc = summary(a.run, a.lineage, Path(a.store) if a.store else None)
    p = RUNS / a.run / "evidence/generation_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: doc[k] for k in ("terminal", "output_tokens", "cost")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

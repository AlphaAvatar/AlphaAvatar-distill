#!/usr/bin/env python3
"""Check the project's training, calibration and search data against the eval set.

    PYTHONPATH=src:scripts .venv/bin/python scripts/shared/data/check_eval_set_isolation.py \
        [--eval-set eval-set-1] [--out data/eval-set/eval-set-1/audit/isolation.json]

For every data source a model has been trained, calibrated or searched on, the
prompt and content keys of `shared.eval_sets` are compared with every readable
unit of the registered set. A held-out unit must share nothing; the in-sample
regression unit is made of training prompts, so its overlap is expected and
reported. Calibration items that carry token ids rather than text are resolved
to the records they were rendered from (the Stage-0 task sets by `task_id`, the
corpus_v2 sessions by `session_id`), so every source is compared on text.
Sources absent from this machine are reported as such, never as clean.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.infrastructure.env import code_state  # noqa: E402
from shared import eval_sets as ES  # noqa: E402

CORPUS_V2 = REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl"
STAGE0_TASKS = REPO / "data/stages/stage-0/calibration_volume/tasks"

#: (name, role, how to read it). Roles: what the data was used for.
SOURCES = (
    ("stage3.corpus_v2.sessions", "training (Stage-3 recovery corpus; every rung)",
     ("jsonl", CORPUS_V2)),
    ("stage2.stage2_v1.train", "training (Stage-2 offline mixture)",
     ("dir", REPO / "data/stages/stage-2/stage2_v1/train")),
    ("stage2.stage2_v1.val", "held-out split of the Stage-2 mixture (eval_behavior_v0 was drawn from it)",
     ("dir", REPO / "data/stages/stage-2/stage2_v1/val")),
    ("stage2.stage2_v1.calib", "calibration split of the Stage-2 mixture",
     ("dir", REPO / "data/stages/stage-2/stage2_v1/calib")),
    ("stage0.calibration_v1", "Stage-0 calibration prompts", ("jsonl", STAGE0_TASKS / "calibration_v1.jsonl")),
    ("stage0.calibration_v2", "Stage-0 calibration prompts", ("jsonl", STAGE0_TASKS / "calibration_v2.jsonl")),
    ("stage0.base_pool_v1", "Stage-0 calibration pool (searches)",
     ("rendered", REPO / "artifacts/stages/stage-0/calibration_volume/base_pool_v1/items.jsonl")),
    *((f"stage0.{d.name}", "Stage-0 search calibration profiles", ("rendered_dir", d))
      for d in sorted((REPO / "artifacts/stages/stage-0/calibration_volume").glob("profiles_*"))),
    ("stage1.state_eval_v1", "state evaluation suite of every D-series and Stage-0 search",
     ("rendered", REPO / "artifacts/stages/stage-1/state_eval_v1/items.jsonl")),
    ("stage1.reasoning_heavy_v2", "D-series calibration profile",
     ("rendered_dir", REPO / "artifacts/stages/stage-1/reasoning_heavy_v2")),
)


def _jl(path: Path):
    for line in path.open():
        if line.strip():
            yield json.loads(line)


def _text_records() -> tuple[dict, dict]:
    """Where a token-id item's text lives: Stage-0 tasks by task_id, sessions by id."""
    tasks = {}
    for f in sorted(STAGE0_TASKS.glob("*.jsonl")):
        for t in _jl(f):
            tasks[t["task_id"]] = t
    sessions = {s["id"]: s for s in _jl(CORPUS_V2)} if CORPUS_V2.is_file() else {}
    return tasks, sessions


def _resolve(row: dict, tasks: dict, sessions: dict) -> dict | None:
    if row.get("task_id") in tasks:
        return tasks[row["task_id"]]
    for key in ("session_id", "source_id"):
        if row.get(key) in sessions:
            return sessions[row[key]]
    return None


def load(kind: str, path: Path, tasks: dict, sessions: dict) -> tuple[list[dict] | None, dict]:
    if not path.exists():
        return None, {"present": False}
    files = (sorted(path.glob("*.jsonl")) if kind in ("dir", "rendered_dir") else [path])
    rows, unresolved, text_free = [], 0, 0
    for f in files:
        for r in _jl(f):
            if kind.startswith("rendered"):
                src = _resolve(r, tasks, sessions)
                if src is None:
                    if r.get("templated") is False or "doc_sha256" in r:
                        text_free += 1
                    else:
                        unresolved += 1
                    continue
                rows.append(src)
            else:
                rows.append(r)
    return rows, {"present": True, "files": len(files), "rows": len(rows),
                  "unresolved_rendered_items": unresolved,
                  "plain_text_items_without_prompt": text_free}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    set_id = args.eval_set or ES.default_formal()
    m = ES.manifest(set_id)
    roles = {f"{c}/{u}": comp["role"] for c, comp in m["components"].items() for u in comp["units"]}
    units = ES.component_items(set_id)
    tasks, sessions = _text_records()
    out = {"schema": "aadistill.eval_set_isolation/v1", "eval_set_id": set_id,
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "code_state": code_state(REPO), "keys": "shared.eval_sets.KEYS (prompt, content)",
           "units_compared": {u: len(r) for u, r in units.items()},
           "units_not_readable": sorted(set(roles) - set(units)), "sources": {}}
    held_out_hits = 0
    for name, used_for, (kind, path) in SOURCES:
        rows, info = load(kind, path, tasks, sessions)
        entry = {"used_for": used_for, "path": str(path.relative_to(REPO)), **info}
        if rows is not None:
            report = ES.overlap_report(rows, set_id)
            shared = {}
            for p in report["pairs"]:
                if "candidate" in (p["a"], p["b"]):
                    unit = p["b"] if p["a"] == "candidate" else p["a"]
                    if any(p["overlap"].values()):
                        shared[unit] = {"role": roles[unit], "overlap": p["overlap"],
                                        "examples": p["examples"]}
            entry["overlaps"] = shared
            entry["held_out_overlap"] = sum(sum(v["overlap"].values()) for v in shared.values()
                                            if v["role"] in ES.HELD_OUT_ROLES)
            held_out_hits += entry["held_out_overlap"] if "train" in used_for or \
                "calibration" in used_for.lower() or "search" in used_for else 0
        out["sources"][name] = entry
    out["verdict"] = ("no training, calibration or search source shares a prompt or content key "
                      "with a held-out unit" if held_out_hits == 0 else
                      f"{held_out_hits} held-out key overlaps in training/calibration/search data")
    path = args.out or (ES.set_root(set_id) / "audit/isolation.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(out["verdict"])
    for n, e in out["sources"].items():
        print(f"  {n:<28} present={e['present']} rows={e.get('rows')} "
              f"held_out_overlap={e.get('held_out_overlap')} "
              f"in_units={sorted(e.get('overlaps', {}))}")
    return 0 if held_out_hits == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

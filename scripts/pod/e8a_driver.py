#!/usr/bin/env python
"""E8 pod A: the contribution-guided depth search, and nothing else.

    /opt/train/bin/python scripts/pod/e8a_driver.py --stage all \
        --spent-usd 0.80 --soft-stop-usd 2.21 --authorized-usd 2.70

One job: run the preregistered greedy search over the frozen calibration set and
produce a depth map. The map is then **frozen and hashed here**, on the pod, so
what leaves this session is a committed selection rather than a table someone
could re-read later.

**This pod may not decide anything else.** It does not construct an
initialization, does not train, and does not look at behaviour. Its output is 28
teacher-layer indices plus the full 260-candidate trace that produced them.

The search's own validity gates live in `search_depth_map.py`: the objective's
self-consistency must be ≤ 1e-6 (or a candidate ranking would be measuring kernel
noise), and the positional map is scored by the same objective for comparison. If
either the search or the freeze fails, this driver stops rather than emitting a
partial map — a partial greedy trace is not a selection.

Stages: search -> freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e8a.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
CALIBRATION = REPO / "artifacts/stage1/e8_calibration_v1"
SEARCH_OUT = REPO / "artifacts/stage1/e8_depth_search"

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
STUDENT_LAYERS = 28
TEACHER_LAYERS = 36
EXPECTED_EVALUATIONS = 260
# The map this search exists to test against, derived independently below.
POSITIONAL_REMOVED = [5, 7, 9, 11, 13, 15, 17, 19]
FROZEN_CALIBRATION_CONTENT = (
    "d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f")


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(cmd, py=TRAIN_PY):
    cmd = [py] + [str(c) for c in cmd]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")})


def spent_usd(args) -> float:
    """Dollars billed so far, from actual elapsed time — never from a plan."""
    return args.spent_usd + (time.time() - args.t0) / 3600 * args.rate


def stage_search(args) -> None:
    report = SEARCH_OUT / "depth_search.json"
    if report.is_file():
        print("search already complete; skipping", flush=True)
        mark("SEARCH_DONE")
        return
    now = spent_usd(args)
    need = args.search_minutes / 60 * args.rate
    if now + need > args.soft_stop_usd:
        mark(f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}>{args.soft_stop_usd:.2f}")
        raise SystemExit("not enough budget left to start the search")
    # `--resume` is implicit: the search replays any completed rounds it finds in
    # rounds.jsonl, so a restart costs the current round rather than the search.
    run(["scripts/training/search_depth_map.py",
         "--calibration", CALIBRATION,
         "--teacher", TEACHER, "--teacher-revision", TEACHER_REVISION,
         "--student-layers", STUDENT_LAYERS,
         "--dtype", "bfloat16", "--device", "cuda",
         "--out", SEARCH_OUT])
    mark("SEARCH_DONE")


def stage_freeze(args) -> None:
    """Freeze and hash the map before anything can be built from it."""
    report = json.loads((SEARCH_OUT / "depth_search.json").read_text())
    dm_path = SEARCH_OUT / "depth_map.json"
    dm = json.loads(dm_path.read_text())
    kept = dm["kept_teacher_layers"]
    removed = dm["removed_teacher_layers"]

    problems = []
    if len(kept) != STUDENT_LAYERS:
        problems.append(f"kept {len(kept)} layers, need {STUDENT_LAYERS}")
    if kept != sorted(set(kept)):
        problems.append("kept layers are not strictly increasing and unique")
    if any(not 0 <= k < TEACHER_LAYERS for k in kept):
        problems.append(f"kept layers outside range(0, {TEACHER_LAYERS})")
    if sorted(set(kept) | set(removed)) != list(range(TEACHER_LAYERS)):
        problems.append("kept + removed is not a partition of the teacher layers")
    if report["evaluations"] != EXPECTED_EVALUATIONS:
        problems.append(f"{report['evaluations']} evaluations, expected "
                        f"{EXPECTED_EVALUATIONS}")
    if len(report["rounds"]) != TEACHER_LAYERS - STUDENT_LAYERS:
        problems.append(f"{len(report['rounds'])} rounds")
    if not report["self_consistency"]["deterministic"]:
        problems.append("the objective was not reproducible on this device")
    if report["calibration"]["content_sha256"] != FROZEN_CALIBRATION_CONTENT:
        problems.append("the search did not run on the frozen calibration set")
    if report["calibration"]["limited_to"]:
        problems.append("the search ran on a limited item subset")
    if problems:
        mark("FREEZE_FAILED")
        raise SystemExit("depth map is not usable: " + "; ".join(problems))

    frozen = {
        "artifact": "e8_frozen_depth_map",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kept_teacher_layers": kept,
        "removed_teacher_layers": removed,
        "removal_order": dm["removal_order"],
        "student_layers": STUDENT_LAYERS,
        "teacher": {"model_id": TEACHER, "revision": TEACHER_REVISION},
        "selector": {
            "primary": report["objective"]["primary"],
            "aggregation": report["objective"]["aggregation"],
            "selection_rule": report["objective"]["selection_rule"],
            "subset_evaluations": report["evaluations"],
        },
        "calibration_content_sha256": report["calibration"]["content_sha256"],
        "search_report_sha256": report["report_sha256"],
        "depth_map_sha256": hashlib.sha256(dm_path.read_bytes()).hexdigest(),
        "primary_kl": report["result"]["primary_kl"],
        "per_domain_kl": report["result"]["per_domain_kl"],
        "positional_baseline_primary_kl": report["positional_baseline"]["primary_kl"],
        "positional_removed": report["positional_baseline"]["removed_teacher_layers"],
        "lower_kl_than_positional": report["comparison"]["contribution_map_is_lower_kl"],
        "differs_from_positional":
            sorted(removed) != sorted(POSITIONAL_REMOVED),
        "diagnostics_recorded": sorted(report["result"]["diagnostics"]),
        "diagnostics_note": "recorded, never used to select (preregistration §4.4)",
        "search_seconds": report["search_seconds"],
        "forward_passes": report["forward_passes"],
    }
    out = OUT / "e8_frozen_depth_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({k: frozen[k] for k in (
        "kept_teacher_layers", "removed_teacher_layers", "removal_order",
        "primary_kl", "positional_baseline_primary_kl",
        "lower_kl_than_positional", "differs_from_positional",
        "depth_map_sha256")}, indent=2), flush=True)

    if not frozen["differs_from_positional"]:
        # A real and reportable result — the causal search recovered the
        # heuristic — but there is then no treatment to train, so say so loudly
        # rather than letting pod B train a copy of the control.
        mark("MAP_IDENTICAL_TO_POSITIONAL")
    mark("FREEZE_DONE")


STAGES = {"search": stage_search, "freeze": stage_freeze}
BLOCKING = ("search", "freeze")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, default=2.21)
    ap.add_argument("--authorized-usd", type=float, default=2.71)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--search-minutes", type=float, default=60.0)
    args = ap.parse_args()
    args.t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E8 pod A: spent ${args.spent_usd:.2f}, soft stop "
          f"${args.soft_stop_usd:.2f}, hard ${args.authorized_usd:.2f} "
          f"at ${args.rate}/h", flush=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        except (subprocess.CalledProcessError, AssertionError, OSError,
                ValueError, KeyError, SystemExit) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark("ABORTED_AFTER_BLOCKING_FAILURE")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    sys.exit(main())

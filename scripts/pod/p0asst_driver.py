#!/usr/bin/env python
"""Train the two P0-assistant arms, then evaluate both with the D0.3 harness.

    /opt/train/bin/python scripts/pod/p0asst_driver.py --stage all

The intervention is **assistant-only KD with assistant-token normalization**, not
a removal. `kd_scope: assistant` drops the 606,717 prompt/context positions from
the KD term *and* shrinks the denominator from 1,471,467 to 864,750, so every
remaining assistant token's KD contribution rises by x1.7016. Both halves are the
treatment; describing it as "prompt KD removed" would understate it.

`truncate_padding` is deliberately absent from the configs so the code path
matches P0-real exactly and `kd_scope` stays the only substantive difference.

Evaluation reuses the **identical** D0.3 harness, the same 150 fixed examples and
the same inclusion mask, so the numbers land directly beside the P0-real ones.
The 846-prompt capability battery is NOT run.

Markers: TRAIN_DONE:<arm> -> EVAL_DONE:<arm> -> ALL_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/p0asst.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
ARMS = {"P0-assistant-sa": "p0_assistant_sa", "P0-assistant-sb": "p0_assistant_sb"}


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


def stage_train(args):
    for alias, name in ARMS.items():
        final = REPO / f"artifacts/stage3/{name}/checkpoints/step_001023/model"
        if final.is_dir():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        cfg = REPO / f"configs/stage3/p0/{name}.json"
        # Fail loudly if the single-variable guarantee was broken in transit.
        loaded = json.loads(cfg.read_text())
        assert loaded["loss"]["kd_scope"] == "assistant", loaded["loss"]
        assert "truncate_padding" not in loaded["batch"], loaded["batch"]
        run(["scripts/training/train_stage3.py", "--config", cfg])
        mark(f"TRAIN_DONE:{alias}")
    mark("TRAIN_DONE")


def stage_eval(args):
    for alias, name in ARMS.items():
        out = OUT / "three_mode" / alias
        if (out / "oracle.generations.jsonl").exists():
            print(f"{alias} already evaluated; skipping", flush=True)
            mark(f"EVAL_DONE:{alias}")
            continue
        model = REPO / f"artifacts/stage3/{name}/checkpoints/step_001023/model"
        if not model.is_dir():
            mark(f"EVAL_SKIPPED:{alias}:no_checkpoint")
            continue
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", 860000, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", out], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", 860000, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", out / "forced"])
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "train", "eval"))
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {"train": stage_train, "eval": stage_eval}
    for name in (list(stages) if args.stage == "all" else [args.stage]):
        try:
            stages[name](args)
        except subprocess.CalledProcessError as exc:
            mark(f"STAGE_FAILED:{name}:rc={exc.returncode}")
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

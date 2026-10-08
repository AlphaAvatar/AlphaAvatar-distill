#!/usr/bin/env python
"""Sequence D0.3 and D0.4 on both P0-real checkpoints. No optimizer step.

    /opt/train/bin/python scripts/pod/d0diag_driver.py --stage all

D0.3 runs first: it is the shorter job and its free-rollout result is what makes
the KD decomposition worth reading. Each stage is resumable — an output that
already exists is skipped — and each writes its own marker so a poller can tell
progress from a stall.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/d0diag.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
ARMS = {"P0-real-sa": "e1_r0860k_sa_pca", "P0-real-sb": "e1_r0860k_sb_pca"}


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


def stage_d0_3(args):
    for alias, arm in ARMS.items():
        out = OUT / "three_mode" / alias
        if (out / "report.json").exists():
            print(f"{alias} three-mode already done; skipping", flush=True)
            continue
        model = f"/workspace/ckpt/{arm}/step_001023/model"
        # free + oracle need vLLM; forced needs the training venv. Split so each
        # runs in the stack that has its dependency, writing into one directory.
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", 860000, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", out], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", 860000, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", out])
        mark(f"D0_3_DONE:{alias}")
    mark("D0_3_DONE")


def stage_d0_4(args):
    for alias, arm in ARMS.items():
        out = OUT / f"kd_decomposition_{alias}.json"
        if out.exists():
            print(f"{alias} KD decomposition already done; skipping", flush=True)
            continue
        model = f"/workspace/ckpt/{arm}/step_001023/model"
        cfg = REPO / f"configs/stage3/e1/{arm}.json"
        run(["scripts/training/audit_kd_decomposition.py",
             "--student", model,
             "--teacher", f"Qwen/Qwen3-4B-Thinking-2507@{args.teacher_revision}",
             "--pack", PACK, "--rung", 860000, "--blocks", args.blocks,
             "--config", cfg, "--label", alias,
             "--grad-probe-blocks", args.grad_probe_blocks, "--out", out])
        mark(f"D0_4_DONE:{alias}")
    mark("D0_4_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "d0_3", "d0_4"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--blocks", type=int, default=682,
                    help="682 = the whole 0.86M rung")
    ap.add_argument("--grad-probe-blocks", type=int, default=8)
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {"d0_3": stage_d0_3, "d0_4": stage_d0_4}
    for name in (list(stages) if args.stage == "all" else [args.stage]):
        try:
            stages[name](args)
        except subprocess.CalledProcessError as exc:
            mark(f"STAGE_FAILED:{name}:rc={exc.returncode}")
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

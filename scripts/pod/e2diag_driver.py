#!/usr/bin/env python
"""Sequence the three diagnostic jobs on the pod, marker by marker.

    /opt/train/bin/python scripts/pod/e2diag_driver.py --stage all

Order is deliberate: the benchmark runs first because it is the shortest job
whose result gates a repository decision, and because it needs the 4B teacher
resident while nothing else is competing for the GPU. Each stage writes its own
marker so a poller can tell progress from a stall, and each is resumable — a
stage whose output already exists is skipped.

Nothing here reduces a preregistered count. The battery is the full frozen set
under both protocols; the recall diagnostic uses all four forced-prefix release
lengths.
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
STATUS = Path("/workspace/e2diag.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"

REF_MODEL = "Qwen/Qwen3-0.6B"
CONTROL = "/workspace/ckpt/e1_ctl_r0250k_sa_pca_stepmatched"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(cmd: list[str], **kw) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] $ {' '.join(map(str, cmd))}",
          flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")}, **kw)


def stage_benchmark(args) -> None:
    out = OUT / "padding_truncation_benchmark.json"
    if out.exists():
        print("benchmark already done; skipping", flush=True)
        return mark("BENCH_DONE")
    run([TRAIN_PY, "scripts/pod/benchmark_padding_truncation.py",
         "--pack", PACK,
         "--student", REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
         "--teacher", f"Qwen/Qwen3-4B-Thinking-2507@{args.teacher_revision}",
         "--config", REPO / "configs/stage3/e1/e1_r0860k_sa_pca.json",
         "--blocks-per-regime", 8, "--steps", 6, "--warmup", 2,
         "--out", out])
    mark("BENCH_DONE")


def stage_diag_a(args) -> None:
    """The frozen battery on the pinned reference, under both protocols."""
    battery = REPO / "artifacts/eval/battery_v2"
    prompts = sorted(str(p) for p in battery.glob("*.jsonl"))
    behavior = REPO / "data/eval_behavior_v0/prompts.jsonl"
    if behavior.exists():
        prompts.append(str(behavior))
    for protocol, kwargs in (("project", "{}"),
                             ("native", json.dumps({"enable_thinking": True}))):
        tag = f"ref_qwen3_0p6b_{protocol}"
        gen_dir = REPO / f"artifacts/eval/e2diag/{tag}"
        if (gen_dir / "gsm8k.generations.jsonl").exists():
            print(f"{tag} already generated; skipping", flush=True)
            continue
        cmd = [VLLM_PY, "scripts/evaluation/uncapped_eval.py",
               "--model", REF_MODEL, "--revision", args.ref_revision,
               "--label", tag, "--prompts", *prompts,
               "--out-dir", gen_dir, "--trained-context", 8192,
               "--protocol", protocol, "--chat-template-kwargs", kwargs,
               "--diagnostics"]
        run(cmd)
        run([TRAIN_PY, "scripts/evaluation/score_battery.py",
             "--battery", battery, "--generations", gen_dir, "--label", tag,
             "--out", REPO / f"artifacts/eval/e2diag/{tag}_battery.json",
             "--per-sample",
             REPO / f"artifacts/eval/e2diag/{tag}_battery.per_sample.jsonl"])
        mark(f"DIAGA_DONE:{protocol}")
    mark("DIAGA_DONE")


def stage_diag_b(args) -> None:
    out = REPO / "artifacts/audit/training_recall"
    if (out / "report.json").exists():
        print("recall diagnostic already done; skipping", flush=True)
        return mark("DIAGB_DONE")
    run([VLLM_PY, "scripts/evaluation/diagnose_training_recall.py",
         "--model", CONTROL, "--pack", PACK, "--rung", 250000,
         "--sessions", REPO / "artifacts/stage3/corpus_v2/sessions.jsonl",
         "--n", args.n_recall, "--out", out])
    mark("DIAGB_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "benchmark", "diag_a", "diag_b"))
    ap.add_argument("--ref-revision",
                    default="c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--n-recall", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stages = {"benchmark": stage_benchmark, "diag_a": stage_diag_a,
              "diag_b": stage_diag_b}
    order = list(stages) if args.stage == "all" else [args.stage]
    for name in order:
        try:
            stages[name](args)
        except subprocess.CalledProcessError as exc:
            mark(f"STAGE_FAILED:{name}:rc={exc.returncode}")
            # Keep going: a failed benchmark must not cost the diagnostics, and
            # every stage writes its own artifact.
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

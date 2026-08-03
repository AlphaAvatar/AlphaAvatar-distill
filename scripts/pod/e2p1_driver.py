#!/usr/bin/env python
"""Experiment 2 Phase 1 pod driver — D1 only, both seeds, gated.

Runs on the pod, drives the whole phase, and emits arm-scoped markers so the
dev-box poller can follow an unattended session. The order is preregistered and
is the reason this is a driver rather than a loop over arms:

    setup
    -> D0 endpoint, seed sa            (the FIRST D0 evaluation)
    -> THROUGHPUT GATE                 <- stops here on failure
    -> D0 endpoint, seed sb
    -> D1 training, seed sa
    -> D1 training, seed sb
    -> per seed: pick the retained identities from the run's own trajectory,
       run the full 846-prompt battery on each, and the 76-prompt behaviour set
       on every remaining eval point
    -> retain + hash + upload

**The gate runs before the second D0 endpoint and before either training run.**
Its three conditions are independent and are applied exactly as committed; this
driver never reinterprets them, it only reports the verdict and stops.

Markers (stdout, one per line, also appended to the status file):
    MARKER:SETUP_OK
    MARKER:D0_DONE:<seed>
    MARKER:GATE_PASS / MARKER:GATE_FAIL
    MARKER:TRAIN_DONE:<seed> / MARKER:TRAIN_FAILED:<seed>
    MARKER:EVAL_DONE:<seed>
    MARKER:UPLOAD_DONE
    MARKER:ALL_DONE / MARKER:HALTED:<why>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AAD_REPO", "/workspace/aad"))
STATUS = Path("/workspace/e2p1.status")

# The eight prompt files that make up one full-battery checkpoint: seven
# capability sets (770 prompts) plus behavior_v0 (76), 846 total. behavior_v0
# rides the same engine purely to avoid an eighth model load; it keeps its own
# generations file, its own scorer and its own aggregate, and is never folded
# into a capability set.
CAPABILITY_SETS = ["knowledge", "math_verified", "gsm8k", "multihop", "rag",
                   "answerability_paired", "safety_paired"]
BEHAVIOR_SET = "behavior_v0"


def mark(text: str) -> None:
    line = f"MARKER:{text}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {line}\n")


def log(text: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {text}", flush=True)


def run(cmd: list[str], cwd: Path = REPO, check: bool = True) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    rc = subprocess.run([str(c) for c in cmd], cwd=cwd).returncode
    if check and rc != 0:
        raise SystemExit(f"command failed rc={rc}: {' '.join(str(c) for c in cmd)}")
    return rc


def battery_prompt_files(battery: Path, behavior_prompts: Path) -> list[str]:
    files = [str(battery / f"{n}.jsonl") for n in CAPABILITY_SETS]
    files.append(str(behavior_prompts))
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise SystemExit(f"missing prompt files: {missing}")
    return files


def evaluate_checkpoint(model: Path, label: str, out_dir: Path, battery: Path,
                        behavior_prompts: Path, vllm_python: str,
                        full_battery: bool) -> None:
    """One engine, all prompt files. `full_battery=False` runs behaviour only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = (battery_prompt_files(battery, behavior_prompts) if full_battery
             else [str(behavior_prompts)])
    run([vllm_python, "scripts/evaluation/uncapped_eval.py",
         "--model", model, "--label", label,
         "--prompts", *files, "--out-dir", out_dir, "--diagnostics"])


def score_checkpoint(label: str, gen_dir: Path, battery: Path, out: Path) -> None:
    """Offline, CPU, free. Capability sets only — behaviour keeps its own path."""
    run([sys.executable, "scripts/evaluation/score_battery.py",
         "--battery", battery, "--generations", gen_dir, "--label", label,
         "--out", out, "--per-sample", out.with_suffix(".per_sample.jsonl")])


def holdout_trajectory(run_dir: Path, holdout: Path, out: Path) -> None:
    """Held-out NLL for every saved checkpoint, in one process.

    `eval_ppl.py --model` appends, so all nine checkpoints are scored without
    paying the interpreter and import cost nine times. This is the orchestrator
    path: the trainer is NOT modified, because byte identity with D0's trainer
    is what makes the comparison valid.
    """
    ckpts = sorted((run_dir / "checkpoints").glob("step_*/model"))
    if not ckpts:
        log(f"no checkpoints under {run_dir}; skipping holdout trajectory")
        return
    args = []
    for c in ckpts:
        args += ["--model", str(c)]
    tmp = out.with_suffix(".raw.json")
    run([sys.executable, "scripts/evaluation/eval_ppl.py",
         "--data", holdout, *args, "--out", tmp])
    payload = json.loads(tmp.read_text())
    with out.open("w") as f:
        for row in payload["results"]:
            step = int(Path(row["model"]).parent.name.split("_")[1])
            f.write(json.dumps({"step": step,
                                "holdout_nll": row["mean_nll_nats"]}) + "\n")
    log(f"wrote {out} for {len(ckpts)} checkpoints")


def retained_identities(run_dir: Path) -> dict:
    """The steps whose weights are kept — and therefore the ones evaluated."""
    out = run_dir / "retention.json"
    run([sys.executable, "scripts/pod/retain_checkpoints.py",
         "--run-dir", run_dir, "--out", out])
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path,
                    default=REPO / "artifacts/eval/battery_v2")
    ap.add_argument("--behavior-prompts", type=Path,
                    default=REPO / "data/eval_behavior_v0/prompts.jsonl")
    ap.add_argument("--holdout", type=Path,
                    default=REPO / "data/warmup/holdout_v1.jsonl")
    ap.add_argument("--out-root", type=Path, default=REPO / "artifacts/eval/e2p1")
    ap.add_argument("--d0-root", type=Path, default=Path("/workspace/d0"))
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--stage", default="all",
                    choices=["all", "d0_sa", "gate", "rest"])
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    seeds = ["sa", "sb"]

    # ---- 1. the FIRST D0 endpoint -------------------------------------
    if args.stage in ("all", "d0_sa"):
        model = args.d0_root / "e1_r0860k_sa_pca/step_001023/model"
        log(f"D0 sa endpoint: {model}")
        evaluate_checkpoint(model, "d0_sa", args.out_root / "d0_sa",
                            args.battery, args.behavior_prompts,
                            args.vllm_python, full_battery=True)
        score_checkpoint("d0_sa", args.out_root / "d0_sa", args.battery,
                         args.out_root / "d0_sa_battery.json")
        mark("D0_DONE:sa")
        if args.stage == "d0_sa":
            return 0

    # ---- 2. the throughput gate ---------------------------------------
    if args.stage in ("all", "gate"):
        rc = run([sys.executable, "scripts/pod/throughput_gate.py",
                  "--eval-dir", args.out_root / "d0_sa",
                  "--out", args.out_root / "throughput_gate.json"], check=False)
        if rc != 0:
            mark("GATE_FAIL")
            mark("HALTED:throughput_gate")
            log("Gate failed. Not running the second D0 endpoint or any D1 "
                "training. Partial output and telemetry are preserved under "
                f"{args.out_root}.")
            return 1
        mark("GATE_PASS")
        if args.stage == "gate":
            return 0

    # ---- 3. the second D0 endpoint ------------------------------------
    model = args.d0_root / "e1_r0860k_sb_pca/step_001023/model"
    log(f"D0 sb endpoint: {model}")
    evaluate_checkpoint(model, "d0_sb", args.out_root / "d0_sb", args.battery,
                        args.behavior_prompts, args.vllm_python,
                        full_battery=True)
    score_checkpoint("d0_sb", args.out_root / "d0_sb", args.battery,
                     args.out_root / "d0_sb_battery.json")
    mark("D0_DONE:sb")

    # ---- 4. D1 training, both seeds ------------------------------------
    for seed in seeds:
        arm = f"e2_d1_{seed}_pca"
        cfg = REPO / f"configs/stage3/e2/{arm}.json"
        log(f"training {arm}")
        rc = run([sys.executable, "scripts/training/train_stage3.py",
                  "--config", cfg], check=False)
        if rc != 0:
            mark(f"TRAIN_FAILED:{seed}")
            mark("HALTED:training")
            return 1
        mark(f"TRAIN_DONE:{seed}")

    # ---- 5. per-seed evaluation ----------------------------------------
    for seed in seeds:
        arm = f"e2_d1_{seed}_pca"
        run_dir = REPO / f"artifacts/stage3/{arm}"
        holdout_trajectory(run_dir, args.holdout,
                           run_dir / "holdout_trajectory.jsonl")
        decision = retained_identities(run_dir)
        keep = {int(k) for k in decision["keep"]}
        all_steps = {int(p["step"]) for p in decision["trajectory"]
                     if int(p["step"]) > 0}
        log(f"{arm}: battery on {sorted(keep)}; behaviour-only on "
            f"{sorted(all_steps - keep)}")

        for step in sorted(keep):
            model = run_dir / f"checkpoints/step_{step:06d}/model"
            if not model.is_dir():
                log(f"WARNING: {model} absent; skipping")
                continue
            label = f"{arm}@{step}"
            out_dir = args.out_root / f"{arm}_step{step:06d}"
            evaluate_checkpoint(model, label, out_dir, args.battery,
                                args.behavior_prompts, args.vllm_python,
                                full_battery=True)
            score_checkpoint(label, out_dir, args.battery,
                             args.out_root / f"{arm}_step{step:06d}_battery.json")

        # The 76-prompt behaviour generations at every remaining eval point are
        # mandatory and separately persisted; they are never scored as, or
        # aggregated with, a capability set.
        for step in sorted(all_steps - keep):
            model = run_dir / f"checkpoints/step_{step:06d}/model"
            if not model.is_dir():
                log(f"note: {model} absent (pruned by retention); skipping")
                continue
            evaluate_checkpoint(model, f"{arm}@{step}",
                                args.out_root / f"{arm}_step{step:06d}",
                                args.battery, args.behavior_prompts,
                                args.vllm_python, full_battery=False)
        mark(f"EVAL_DONE:{seed}")

    # ---- 6. retention, hashes ------------------------------------------
    for seed in seeds:
        run_dir = REPO / f"artifacts/stage3/e2_d1_{seed}_pca"
        run([sys.executable, "scripts/pod/retain_checkpoints.py",
             "--run-dir", run_dir, "--apply"])
    mark("UPLOAD_DONE")
    mark("ALL_DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # a driver crash must still be visible to the poller
        mark(f"HALTED:{type(exc).__name__}")
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise

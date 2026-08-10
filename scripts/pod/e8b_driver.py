#!/usr/bin/env python
"""E8 pod B: initialization diagnostics, the gate, two-seed 2.96M recovery, evaluation.

    /opt/train/bin/python scripts/pod/e8b_driver.py --stage all \
        --spent-usd 0.80 --soft-stop-usd 9.22 --authorized-usd 9.71

The stage order is the rule, not a convenience:

    measure both inits' own NLL -> bind to hashes -> gate -> train -> evaluate

`stage_gate` runs `validate_e8_arms.py --require-init`, which refuses unless each
initialization has its own hash-bound NLL record with every required series. It is
**blocking**: no training happens if it fails. That is what makes "an
initialization checkpoint is not complete until its own NLL artifact exists" a
mechanism rather than a sentence in a document.

**Both** initializations are measured here, on one device, by one evaluator. The
control is remeasured rather than inherited: comparing a fresh treatment number
against a historical control number taken through the transformers-4.x reader path
would be comparing two different measurements (decisions.md, 2026-08-10).

A good or bad initialization NLL decides nothing. The gate checks validity —
finite, positive, hash-bound — and the endpoint remains the frozen autonomous
rollout evaluation.

**The evaluation rung is pinned to 860000**, as in E4, E6, E6b and E7. The harness
samples its 150 examples from the rung it is handed; passing the training rung
would resample the battery and silently end the comparison while still reporting
150 prompts and a mask hash.

Stages: init_nll -> gate -> train -> general_text -> three_mode
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

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e8b.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
CONTROL_INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
TREATMENT_INIT = REPO / "artifacts/stage1/e8_contribution_init_v1/checkpoint"
VAL_STREAM = REPO / "artifacts/stage3/e7_fineweb_val"
HOLDOUT = REPO / "data/warmup/holdout_v1.jsonl"

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
EVAL_RUNG = 860000
TRAIN_RUNG = 2960000
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
STEP = "step_002916"
OBJECTIVE = {"ce_weight": 0.25, "kd_weight": 1.0,
             "kd_temperature": 1.0, "kd_scope": "all"}

# The two treatment arms. Aliases are what the frozen battery records.
ARMS = {
    "E8-T-Contrib-sa": "e8_contrib_r2960k_sa",
    "E8-T-Contrib-sb": "e8_contrib_r2960k_sb",
}
# The control is retained from E1/P1 and never retrained here.
CONTROL_ARMS = {"E8-C-Positional-sa": "e1_r2960k_sa_pca",
                "E8-C-Positional-sb": "e1_r2960k_sb_pca"}

# label -> (checkpoint dir, NLL record path)
INITS = {
    "e8-contribution-init": (TREATMENT_INIT,
                             TREATMENT_INIT.parent / "init_nll.json"),
    "baseline-positional-init": (CONTROL_INIT,
                                 CONTROL_INIT.parent / "init_nll.json"),
}


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


def run_dir(name: str) -> Path:
    return REPO / f"artifacts/stage3/{name}"


def model_dir(name: str) -> Path:
    return run_dir(name) / f"checkpoints/{STEP}/model"


def spent_usd(args) -> float:
    """Dollars billed so far, from actual elapsed time — never from a plan."""
    return args.spent_usd + (time.time() - args.t0) / 3600 * args.rate


# --------------------------------------------------------------------------


def stage_init_nll(args) -> None:
    """Measure each initialization's own NLL, bound to its own hash."""
    for label, (ckpt, record) in INITS.items():
        if record.is_file():
            print(f"{label}: record exists; skipping", flush=True)
            mark(f"INIT_NLL_DONE:{label}")
            continue
        if not (ckpt / "model.safetensors").is_file():
            mark(f"INIT_NLL_SKIPPED:{label}:missing_checkpoint")
            continue
        now = spent_usd(args)
        need = args.per_init_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}>"
                 f"{args.soft_stop_usd:.2f}")
            raise SystemExit("not enough budget left to measure an initialization")
        run(["scripts/evaluation/measure_init_nll.py",
             "--checkpoint", ckpt, "--label", label,
             "--holdout", HOLDOUT, "--fineweb-val", VAL_STREAM,
             "--pack", PACK, "--rung", TRAIN_RUNG,
             "--teacher", TEACHER, "--teacher-revision", TEACHER_REVISION,
             "--dtype", "bfloat16", "--device", "cuda",
             "--out", record])
        mark(f"INIT_NLL_DONE:{label}")
    mark("INIT_NLL_DONE")


def stage_gate(args) -> None:
    """The blocking pre-training gate. Nothing trains if this fails."""
    run(["scripts/training/validate_e8_arms.py", "--require-init",
         "--pack", PACK, "--out", OUT / "e8_preflight.json"])
    report = json.loads((OUT / "e8_preflight.json").read_text())
    if not report["all_passed"]:
        mark(f"GATE_FAILED:{','.join(report['failed'])}")
        raise SystemExit(f"pre-training gate failed: {report['failed']}")

    # The step-0 comparison, assembled here so it exists before training can
    # obscure it. Diagnostic only — it decides nothing.
    step0 = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_status": "DIAGNOSTIC ONLY — may neither promote nor cancel an arm",
        "summaries": report["step0_summaries"],
    }
    frozen = REPO / "artifacts/stage1/e8_depth_search/e8_frozen_depth_map.json"
    if frozen.is_file():
        f = json.loads(frozen.read_text())
        step0["depth_map"] = {
            "kept_teacher_layers": f["kept_teacher_layers"],
            "removed_teacher_layers": f["removed_teacher_layers"],
            "primary_kl": f["primary_kl"],
            "positional_baseline_primary_kl": f["positional_baseline_primary_kl"],
            "per_domain_kl": f["per_domain_kl"],
        }
    (OUT / "e8_step0_comparison.json").write_text(
        json.dumps(step0, indent=2) + "\n")
    print(json.dumps(step0["summaries"], indent=2), flush=True)
    mark("GATE_PASSED")


def stage_train(args) -> None:
    for alias, name in ARMS.items():
        if (model_dir(name) / "model.safetensors").is_file():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        now = spent_usd(args)
        need = args.per_arm_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}>"
                 f"{args.soft_stop_usd:.2f}")
            return
        cfg = REPO / f"configs/stage3/e8/{name}.json"
        run(["scripts/training/train_stage3.py", "--config", cfg])
        mark(f"TRAIN_DONE:{alias}")
    mark("TRAIN_DONE")


def stage_general_text(args) -> None:
    """General-text diagnostics on the trained arms. Diagnostics only."""
    dest = OUT / "e8_general_text"
    dest.mkdir(parents=True, exist_ok=True)
    for alias, name in ARMS.items():
        m = model_dir(name)
        if not m.is_dir():
            continue
        out = dest / f"{alias}.json"
        if out.exists():
            continue
        try:
            run(["scripts/evaluation/eval_general_text.py", "--model", m,
                 "--stream", VAL_STREAM, "--teacher", TEACHER,
                 "--teacher-revision", TEACHER_REVISION,
                 "--dtype", "bfloat16", "--out", out])
        except subprocess.CalledProcessError as exc:
            print(f"  {alias}: general-text diagnostics failed: {exc}", flush=True)
    mark("GENERAL_TEXT_DONE")


def stage_three_mode(args) -> None:
    """The binding harness, unchanged, on the pinned 150-example battery.

    The control arms are NOT re-evaluated: their frozen battery artifacts exist
    from E6/E6b, and replacing a retained measurement with a new one would end
    the comparison this experiment is registered as.
    """
    for alias, name in ARMS.items():
        d = OUT / "three_mode" / alias
        if (d / "report.json").exists():
            print(f"{alias} already evaluated; skipping", flush=True)
            mark(f"EVAL_DONE:{alias}")
            continue
        m = model_dir(name)
        if not m.is_dir():
            mark(f"EVAL_SKIPPED:{alias}:no_checkpoint")
            continue
        now = spent_usd(args)
        need = args.per_eval_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}>"
                 f"{args.soft_stop_usd:.2f}")
            return
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", d], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", d / "forced"])
        mask = json.loads((d / "report.json").read_text())["inclusion"]["mask_sha256"]
        if mask != EXPECTED_MASK:
            raise AssertionError(f"{alias}: inclusion mask {mask} != binding")
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


STAGES = {"init_nll": stage_init_nll, "gate": stage_gate, "train": stage_train,
          "general_text": stage_general_text, "three_mode": stage_three_mode}
# A failure in any of these means nothing downstream is worth paying for. The
# gate is blocking by design: training an unmeasured initialization is exactly
# what E8 forbids.
BLOCKING = ("init_nll", "gate", "train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, default=9.22)
    ap.add_argument("--authorized-usd", type=float, default=9.71)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--per-init-minutes", type=float, default=8.0)
    ap.add_argument("--per-arm-minutes", type=float, default=215.0)
    ap.add_argument("--per-eval-minutes", type=float, default=12.0)
    args = ap.parse_args()
    args.t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E8 pod B: spent ${args.spent_usd:.2f}, soft stop "
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

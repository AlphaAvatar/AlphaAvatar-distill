#!/usr/bin/env python
"""Experiment 7: FineWeb teacher-KD mixture at the fixed 1.60M rollout rung.

    /opt/train/bin/python scripts/pod/e7_driver.py --stage all \
        --spent-usd 0.80 --soft-stop-usd 12.32 --authorized-usd 12.82

Four arms train from the Stage 1 PCA init — **never** from a trained 1.60M or
2.96M checkpoint, which would make this a continuation experiment instead of the
matched comparison it is registered as. Arm A is not retrained at all; its two
retained checkpoints are staged so the general-text diagnostics are measured on
one device, on the same validation stream, alongside B and C.

**The gradient preflight is a stop/go safety gate and nothing else.** It runs for
one B and one C config at the frozen `lambda_extra = 0.25` and, if either ratio
falls outside the registered `[0.05, 1.00]`, this driver stops before any
training. It must not be used to tune lambda, and B and C must never be given
different lambdas to equalise them.

**The evaluation rung is pinned to 860000** for every arm, as in E4, E6 and E6b.
The harness samples its 150 examples from the rung it is handed; passing the
training rung would resample the battery and silently end the comparison while
still reporting 150 prompts and a mask hash.

Budget: before each arm the driver re-prices from **actual elapsed time** and
refuses to start one it cannot finish before the soft stop — the soft stop is
what reserves time for artifact collection, and E6b's overrun came from pricing
each arm against the authorization instead.

Stages: validate -> preflight -> train -> tokenizer -> movement -> general_text
        -> three_mode

Markers: ARMS_VALIDATED -> PREFLIGHT_OK|PREFLIGHT_FAILED -> TRAIN_DONE:<arm>
         -> TOKENIZER_OK -> MOVEMENT_DONE -> GENERAL_TEXT_DONE
         -> EVAL_DONE:<arm> -> EVAL_DONE -> ALL_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e7.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
VAL_STREAM = REPO / "artifacts/stage3/e7_fineweb_val"
HOLDOUT = REPO / "data/warmup/holdout_v1.jsonl"

EVAL_RUNG = 860000
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
STEP = "step_001761"
OBJECTIVE = {"ce_weight": 0.25, "kd_weight": 1.0,
             "kd_temperature": 1.0, "kd_scope": "all"}
LAMBDA_EXTRA = 0.25
BAND = (0.05, 1.00)

# The four arms that train. Aliases are what the frozen battery records.
ARMS = {
    "E7-B-FineWeb-sa": "e7_fineweb_r1600k_sa",
    "E7-B-FineWeb-sb": "e7_fineweb_r1600k_sb",
    "E7-C-Control-sa": "e7_control_r1600k_sa",
    "E7-C-Control-sb": "e7_control_r1600k_sb",
}
# Arm A: retained, never retrained, diagnostics only.
ARM_A = {"E7-A-Baseline-sa": "e1_r1600k_sa_pca",
         "E7-A-Baseline-sb": "e1_r1600k_sb_pca"}
# One preflight per configuration class, at the same lambda.
PREFLIGHT = {"B": "e7_fineweb_r1600k_sa", "C": "e7_control_r1600k_sa"}


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

def stage_validate(args) -> None:
    run(["scripts/training/validate_e7_arms.py", "--require-streams",
         "--out", OUT / "e7_preflight_driver.json"])
    mark("ARMS_VALIDATED")


def stage_preflight(args) -> None:
    """The registered safety gate. Stop/go only — never a lambda search."""
    results = {}
    for label, name in PREFLIGHT.items():
        out = OUT / f"e7_gradient_share_{label}.json"
        cfg = REPO / f"configs/stage3/e7/{name}.json"
        conf = json.loads(cfg.read_text())
        assert conf["extra_stream"]["lambda_extra"] == LAMBDA_EXTRA, \
            f"{name}: lambda_extra is not the frozen {LAMBDA_EXTRA}"
        proc = subprocess.run(
            [TRAIN_PY, "scripts/training/e7_preflight.py", "--config", str(cfg),
             "--out", str(out), "--band-low", str(BAND[0]),
             "--band-high", str(BAND[1])],
            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        report = json.loads(out.read_text()) if out.exists() else {}
        ratio = (report.get("gradient_share") or {}).get("ratio_mean")
        results[label] = {"ratio_mean": ratio, "in_band": report.get("in_band"),
                          "rc": proc.returncode}
        print(f"  preflight {label}: ratio_mean={ratio} in_band="
              f"{report.get('in_band')}", flush=True)
    (OUT / "e7_preflight_summary.json").write_text(
        json.dumps({"lambda_extra": LAMBDA_EXTRA, "band": list(BAND),
                    "results": results}, indent=2) + "\n")
    if not all(r["in_band"] for r in results.values()):
        mark("PREFLIGHT_FAILED:" + json.dumps(results))
        raise AssertionError(
            f"gradient share outside the registered band {BAND}: {results}. "
            "Stopping before training. Do not tune lambda from this.")
    mark("PREFLIGHT_OK")


def stage_train(args) -> None:
    per_arm_usd = args.per_arm_minutes / 60 * args.rate
    for alias, name in ARMS.items():
        if model_dir(name).is_dir():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        now = spent_usd(args)
        if now + per_arm_usd > args.soft_stop_usd:
            mark(f"ABORTED_AT_GATE:budget:{now:.2f}+{per_arm_usd:.2f}>"
                 f"{args.soft_stop_usd:.2f}")
            print("stopping before an arm that would eat the artifact reserve",
                  flush=True)
            return
        cfg_path = REPO / f"configs/stage3/e7/{name}.json"
        cfg = json.loads(cfg_path.read_text())
        # Assert the arm is what it claims to be, from the file that will train.
        assert cfg["loss"] == OBJECTIVE, cfg["loss"]
        assert cfg["rung"] == 1600000, cfg["rung"]
        assert cfg["schedule"]["total_steps"] == 1761, cfg["schedule"]
        assert cfg["student_path"].endswith("qwen3_0p6b_init_v0/checkpoint"), \
            "must fork from the Stage 1 init, not from a trained checkpoint"
        assert cfg["extra_stream"]["lambda_extra"] == LAMBDA_EXTRA
        assert cfg["extra_stream"]["every_n_steps"] == 1, "cadence is frozen"
        assert cfg["extra_stream"]["blocks_per_step"] == 1
        assert "lora" not in cfg
        run(["scripts/training/train_stage3.py", "--config", cfg_path])
        mark(f"TRAIN_DONE:{alias}")


def stage_tokenizer(args) -> None:
    """save_checkpoint writes no tokenizer; install it and prove it works."""
    from transformers import AutoTokenizer

    for alias, name in ARMS.items():
        d = model_dir(name)
        if not d.is_dir():
            mark(f"TOKENIZER_SKIPPED:{alias}")
            continue
        for f in TOKENIZER_FILES:
            shutil.copy(INIT / f, d / f)
        a = AutoTokenizer.from_pretrained(str(d))
        b = AutoTokenizer.from_pretrained(str(INIT))
        assert len(a) == len(b) and (a.chat_template or "") == (b.chat_template or "")
        rendered = a.apply_chat_template([{"role": "user", "content": "x"}],
                                         tokenize=False, add_generation_prompt=True)
        assert rendered.rstrip().endswith("<think>"), rendered[-40:]
        print(f"  {alias}: tokenizer installed, vocab {len(a)}", flush=True)
    mark("TOKENIZER_OK")


def stage_movement(args) -> None:
    for alias, name in ARMS.items():
        out = OUT / "e7_movement" / f"{alias}.json"
        if out.exists() or not model_dir(name).is_dir():
            continue
        run(["scripts/evaluation/parameter_movement.py", "--init", INIT,
             "--checkpoint", model_dir(name), "--label", alias, "--out", out])
        rep = json.loads(out.read_text())
        emb = rep["by_group"]["embedding"]["delta_fro"]
        assert emb == 0.0, f"{alias}: embeddings moved ({emb})"
        for g in ("ffn", "attn_proj"):
            assert rep["by_group"][g]["delta_fro"] > 0.0, f"{alias}: {g} did not move"
        print(f"  {alias}: ffn {rep['by_group']['ffn']['relative']:.6f} "
              f"attn_proj {rep['by_group']['attn_proj']['relative']:.6f}", flush=True)
    mark("MOVEMENT_DONE")


def stage_general_text(args) -> None:
    """DIAGNOSTICS ONLY. These never promote a checkpoint (decision 2026-08-09).

    Measured for all six models — B, C and the retained arm A — on the same
    validation stream, in the same environment, on one device. `holdout_v1` is
    measured too so the historical `holdout_nll` series stays continuous; the
    two are reported as separate columns and are never merged.
    """
    teacher = json.loads(
        (REPO / "configs/stage3/e7/e7_fineweb_r1600k_sa.json").read_text())["teacher"]
    targets = {**{a: model_dir(n) for a, n in ARMS.items()},
               **{a: run_dir(n) / f"checkpoints/{STEP}/model"
                  for a, n in ARM_A.items()}}
    for alias, m in targets.items():
        if not Path(m).is_dir():
            print(f"  {alias}: no checkpoint, skipping", flush=True)
            continue
        out = OUT / "e7_general_text" / f"{alias}.json"
        if not out.exists():
            run(["scripts/evaluation/eval_general_text.py", "--model", m,
                 "--stream", VAL_STREAM, "--teacher", teacher["model_id"],
                 "--teacher-revision", teacher["revision"],
                 "--dtype", "bfloat16", "--out", out])
            rep = json.loads(out.read_text())["metrics"]
            print(f"  {alias}: nll {rep['nll']:.4f} kl {rep.get('kl')} "
                  f"top1 {rep['top1']:.4f}", flush=True)
        hold = OUT / "e7_general_text" / f"{alias}.holdout_v1.json"
        if not hold.exists() and HOLDOUT.is_file():
            try:
                run(["scripts/evaluation/eval_ppl.py", "--model", m,
                     "--data", HOLDOUT, "--out", hold])
            except subprocess.CalledProcessError as exc:
                print(f"  {alias}: holdout_v1 continuity eval failed: {exc}",
                      flush=True)
    mark("GENERAL_TEXT_DONE")


def stage_three_mode(args) -> None:
    """The binding harness, unchanged, on the pinned 150-example battery.

    Arm A is NOT re-evaluated here: its frozen battery artifacts already exist
    from E6 and re-running them would replace a retained measurement with a new
    one for no reason.
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


STAGES = {"validate": stage_validate, "preflight": stage_preflight,
          "train": stage_train, "tokenizer": stage_tokenizer,
          "movement": stage_movement, "general_text": stage_general_text,
          "three_mode": stage_three_mode}
# A failure in any of these means nothing downstream is worth paying for.
BLOCKING = ("validate", "preflight", "train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, default=12.32)
    ap.add_argument("--authorized-usd", type=float, default=12.82)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--per-arm-minutes", type=float, default=140.0)
    ap.add_argument("--per-eval-minutes", type=float, default=12.0)
    args = ap.parse_args()
    args.t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E7 driver: spent ${args.spent_usd:.2f}, soft stop "
          f"${args.soft_stop_usd:.2f}, hard ${args.authorized_usd:.2f} "
          f"at ${args.rate}/h", flush=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        except (subprocess.CalledProcessError, AssertionError, OSError,
                ValueError, KeyError) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark("ABORTED_AFTER_BLOCKING_FAILURE")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

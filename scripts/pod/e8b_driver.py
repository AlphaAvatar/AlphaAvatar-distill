#!/usr/bin/env python
"""E8b driver: one script for all four sessions, selected by --session.

    /opt/train/bin/python scripts/pod/e8b_driver.py --session s2 \
        --spent-usd 1.2 --soft-stop-usd 17.97 --authorized-usd 18.76 --rate 1.59

Stage plans, and the order is the rule rather than a convenience:

    s1   init_nll(DP,DC,FP,FC) -> step0_probe(DP,DC) -> publish_step0
    s2   fetch_step0 -> throughput_gate -> gate -> train -> general_text -> three_mode
    s3   same as s2
    s4   fetch_step0 -> gate -> train -> general_text -> three_mode

S1 measures all four initializations on **one device through one canonical
`from_pretrained` reload path** and publishes the four hash-bound records to the
relay; every training session fetches them and re-runs the gate against them, so no
arm trains from an unmeasured initialization even though the measurement happened in
a different session.

`throughput_gate` is mandatory on the depth-only sessions and blocking. No 3.2B step
time has ever been measured in this project, so it runs 20 real training steps
through the real trainer and checks the three registered quantities — wall-clock
s/step <= 7.86, peak VRAM <= 78 GB, live $/step <= 0.003472. On violation it stops.
It does not widen the budget, switch GPU, or change semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
VAL_STREAM = REPO / "artifacts/stage3/e7_fineweb_val"
HOLDOUT = REPO / "data/warmup/holdout_v1.jsonl"
PROBE_PROMPTS = REPO / "data/eval_behavior_v0/prompts.jsonl"

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
EVAL_RUNG = 860000
TRAIN_RUNG = 1600000
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
STEP = "step_001761"
RELAY = "AlphaAvatar/aadistill-artifacts"
STEP0_PREFIX = "e8b_step0_20260811"

# Registered gate thresholds. Changing any of these is re-pricing, not tuning.
GATE_MAX_SECONDS_PER_STEP = 7.86
GATE_MAX_PEAK_VRAM_GB = 78.0
GATE_MAX_USD_PER_STEP = 0.003472
GATE_STEPS = 20

INITS = {
    "DP": "artifacts/stage1/e8b_dp_init",
    "DC": "artifacts/stage1/e8b_dc_init",
    "FP": "artifacts/stage1/qwen3_0p6b_init_v0",
    "FC": "artifacts/stage1/e8_contribution_init_v1",
}
LABELS = {"DP": "e8b-dp-depth-only-positional",
          "DC": "e8b-dc-depth-only-contribution",
          "FP": "e8b-fp-compressed-positional",
          "FC": "e8b-fc-compressed-contribution"}
SESSION_INITS = {"s1": ("DP", "DC", "FP", "FC"), "s2": ("DP", "DC"),
                 "s3": ("DP", "DC"), "s4": ("FP", "FC")}
# alias -> config name. Aliases are what the frozen battery records.
SESSION_ARMS = {
    "s1": {},
    "s2": {"E8b-DP-sa": "e8b_dp_r1600k_sa", "E8b-DC-sa": "e8b_dc_r1600k_sa"},
    "s3": {"E8b-DP-sb": "e8b_dp_r1600k_sb", "E8b-DC-sb": "e8b_dc_r1600k_sb"},
    "s4": {"E8b-FC-sa": "e8b_fc_r1600k_sa", "E8b-FC-sb": "e8b_fc_r1600k_sb"},
}
DEPTH_SESSIONS = ("s2", "s3")


def status_path(session: str) -> Path:
    return Path(f"/workspace/e8b_{session}.status")


def mark(session: str, name: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} MARKER:{name}"
    print(line, flush=True)
    with status_path(session).open("a") as f:
        f.write(line + "\n")


def run(cmd, py=TRAIN_PY):
    cmd = [py] + [str(c) for c in cmd]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")})


def spent_usd(args) -> float:
    """Dollars billed so far, from actual elapsed time — never from a plan."""
    return args.spent_usd + (time.time() - args.t0) / 3600 * args.rate


def run_dir(name: str) -> Path:
    return REPO / f"artifacts/stage3/{name}"


def model_dir(name: str) -> Path:
    return run_dir(name) / f"checkpoints/{STEP}/model"


# --------------------------------------------------------------------------


def stage_init_nll(args) -> None:
    """All four initializations, one device, one canonical reload path."""
    for cell in SESSION_INITS[args.session]:
        base = REPO / INITS[cell]
        record = base / "init_nll.json"
        if record.is_file():
            mark(args.session, f"INIT_NLL_DONE:{cell}")
            continue
        need = args.per_init_minutes / 60 * args.rate
        now = spent_usd(args)
        if now + need > args.soft_stop_usd:
            mark(args.session, f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}")
            raise SystemExit("not enough budget to measure an initialization")
        run(["scripts/evaluation/measure_init_nll.py",
             "--checkpoint", base / "checkpoint", "--label", LABELS[cell],
             "--holdout", HOLDOUT, "--fineweb-val", VAL_STREAM,
             "--pack", PACK, "--rung", TRAIN_RUNG,
             "--teacher", TEACHER, "--teacher-revision", TEACHER_REVISION,
             "--dtype", "bfloat16", "--device", "cuda", "--out", record])
        mark(args.session, f"INIT_NLL_DONE:{cell}")
    mark(args.session, "INIT_NLL_DONE")


def stage_step0_probe(args) -> None:
    """DP and DC on eval_behavior_v0 — a diagnostic battery, not the endpoint.

    **Declared deviation, and it is a real one.** This runs with a recorded
    `max_new_tokens` cap of 2,048 rather than P18's unrestricted allowance. P18
    governs *formal measurement*; this is a step-0 diagnostic that cannot promote or
    reject anything, and `eval_behavior.py` is an in-process HF-generate path, so an
    unrestricted run on a 3.2B model is unbounded — a fully degenerate model would
    cost 6+ hours per checkpoint. The cap is recorded in the report as a censored
    measurement, and samples that reach it are censored observations, not failures.
    2,048 sits above the teacher's p50 length of 727 and near its p90 of 2,233, so
    natural termination is still visible for most prompts.

    The formal E8b endpoint remains the frozen 150-prompt battery under full P18
    unrestricted generation, on the recovered models.
    """
    for cell in ("DP", "DC"):
        base = REPO / INITS[cell]
        dest = OUT / "e8b_step0_probe" / f"{cell}.json"
        if dest.exists():
            mark(args.session, f"STEP0_PROBE_DONE:{cell}")
            continue
        now = spent_usd(args)
        need = args.per_probe_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(args.session, f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            run(["scripts/evaluation/eval_behavior.py",
                 "--model", base / "checkpoint",
                 "--prompts", "data/eval_behavior_v0/prompts.jsonl",
                 "--max-new-tokens", args.probe_max_new_tokens,
                 "--dtype", "bfloat16", "--out", dest])
        except subprocess.CalledProcessError as exc:
            mark(args.session, f"STEP0_PROBE_FAILED:{cell}")
            print(f"  {cell}: step-0 probe failed: {exc}", flush=True)
            continue
        mark(args.session, f"STEP0_PROBE_DONE:{cell}")
    mark(args.session, "STEP0_PROBE_DONE")


def stage_publish_step0(args) -> None:
    """Push the four hash-bound records so the training sessions can gate on them."""
    from huggingface_hub import HfApi
    api = HfApi()
    published = []
    for cell in SESSION_INITS[args.session]:
        rec = REPO / INITS[cell] / "init_nll.json"
        if not rec.is_file():
            continue
        api.upload_file(path_or_fileobj=str(rec),
                        path_in_repo=f"{STEP0_PREFIX}/{cell}_init_nll.json",
                        repo_id=RELAY, repo_type="model",
                        token=os.environ.get("HF_TOKEN"))
        published.append(cell)
        print(f"  published {cell}", flush=True)
    (OUT / "e8b_step0_published.json").write_text(json.dumps(
        {"published": published, "prefix": STEP0_PREFIX}, indent=2) + "\n")
    mark(args.session, f"STEP0_PUBLISHED:{','.join(published)}")


def stage_fetch_step0(args) -> None:
    """Bring S1's records here, so the gate can bind them to these checkpoints."""
    from huggingface_hub import hf_hub_download
    import shutil
    for cell in SESSION_INITS[args.session]:
        dest = REPO / INITS[cell] / "init_nll.json"
        if dest.is_file():
            continue
        p = hf_hub_download(RELAY, f"{STEP0_PREFIX}/{cell}_init_nll.json",
                            repo_type="model", token=os.environ.get("HF_TOKEN"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)
        print(f"  fetched {cell} step-0 record", flush=True)
    mark(args.session, "STEP0_FETCHED")


def stage_throughput_gate(args) -> None:
    """20 real training steps: s/step, peak VRAM, live $/step. Blocking."""
    result = OUT / f"e8b_{args.session}_throughput_gate.json"
    if result.is_file():
        mark(args.session, "THROUGHPUT_GATE_PASSED")
        return
    alias, name = next(iter(SESSION_ARMS[args.session].items()))
    cfg = json.loads((REPO / f"configs/stage3/e8b/{name}.json").read_text())
    probe = dict(cfg)
    probe["run_name"] = f"{name}_gate"
    probe["out_dir"] = f"artifacts/stage3/{name}_gate"
    probe["schedule"] = {**cfg["schedule"], "total_steps": GATE_STEPS}
    probe["intervals"] = {"log_every": 1, "eval_every": 0, "eval_blocks": 4}
    probe["checkpoint"] = {"save_every": 10_000, "keep_last": 1}
    probe["_purpose"] = (f"E8b {args.session} throughput/VRAM/$-per-step gate: "
                         f"{GATE_STEPS} real steps of {name}, discarded")
    probe_path = REPO / f"configs/stage3/e8b/{name}_gate.json"
    probe_path.write_text(json.dumps(probe, indent=2) + "\n")
    run(["scripts/training/train_stage3.py", "--config", probe_path])

    log = REPO / probe["out_dir"] / "train_log.jsonl"
    steps = [json.loads(l) for l in log.open() if l.strip()]
    steps = [s for s in steps if s.get("event") == "train_step"]
    if len(steps) < GATE_STEPS // 2:
        mark(args.session, "THROUGHPUT_GATE_FAILED:too_few_steps")
        raise SystemExit(f"gate logged only {len(steps)} steps")
    # The first steps carry allocator warm-up and compile; the registered rate is
    # the steady-state median over the second half.
    tail = steps[len(steps) // 2:]
    sec = statistics.median(s["seconds"] for s in tail)
    peak = max(s.get("gpu_mem_gb") or 0.0 for s in steps)
    usd = args.rate / 3600 * sec
    verdict = {
        "arm": name, "steps_logged": len(steps), "measured_on": tail[0].get("step"),
        "seconds_per_step_median": round(sec, 3),
        "peak_vram_gb": round(peak, 2),
        "usd_per_step": round(usd, 6),
        "registered": {"max_seconds_per_step": GATE_MAX_SECONDS_PER_STEP,
                       "max_peak_vram_gb": GATE_MAX_PEAK_VRAM_GB,
                       "max_usd_per_step": GATE_MAX_USD_PER_STEP},
        "rate_usd_per_hour": args.rate,
        "projected_minutes_per_arm": round(1761 * sec / 60, 1),
    }
    verdict["violations"] = [
        k for k, ok in (
            ("seconds_per_step", sec <= GATE_MAX_SECONDS_PER_STEP),
            ("peak_vram_gb", peak <= GATE_MAX_PEAK_VRAM_GB),
            ("usd_per_step", usd <= GATE_MAX_USD_PER_STEP)) if not ok]
    verdict["passed"] = not verdict["violations"]
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if not verdict["passed"]:
        mark(args.session, f"THROUGHPUT_GATE_FAILED:{','.join(verdict['violations'])}")
        raise SystemExit(
            "registered gate violated: " + ", ".join(verdict["violations"]) +
            ". Stopping to re-price rather than widening the budget, switching "
            "GPU, or changing training semantics.")
    mark(args.session, "THROUGHPUT_GATE_PASSED")


def stage_gate(args) -> None:
    """The blocking pre-training gate, scoped to this session."""
    out = OUT / f"e8b_{args.session}_preflight.json"
    run(["scripts/training/validate_e8b_arms.py", "--session", args.session,
         "--require-init", "--pack", PACK, "--out", out])
    report = json.loads(out.read_text())
    if not report["all_passed"]:
        mark(args.session, f"GATE_FAILED:{','.join(report['failed'])}")
        raise SystemExit(f"pre-training gate failed: {report['failed']}")
    if report["step0_summaries"]:
        (OUT / f"e8b_{args.session}_step0.json").write_text(json.dumps(
            {"metric_status": "DIAGNOSTIC ONLY",
             "summaries": report["step0_summaries"]}, indent=2) + "\n")
        print(json.dumps(report["step0_summaries"], indent=2), flush=True)
    mark(args.session, "GATE_PASSED")


def stage_train(args) -> None:
    for alias, name in SESSION_ARMS[args.session].items():
        if (model_dir(name) / "model.safetensors").is_file():
            mark(args.session, f"TRAIN_DONE:{alias}")
            continue
        now = spent_usd(args)
        need = args.per_arm_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(args.session, f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}>"
                               f"{args.soft_stop_usd:.2f}")
            return
        run(["scripts/training/train_stage3.py",
             "--config", REPO / f"configs/stage3/e8b/{name}.json"])
        mark(args.session, f"TRAIN_DONE:{alias}")
    mark(args.session, "TRAIN_DONE")


def stage_general_text(args) -> None:
    dest = OUT / "e8b_general_text"
    dest.mkdir(parents=True, exist_ok=True)
    for alias, name in SESSION_ARMS[args.session].items():
        m = model_dir(name)
        out = dest / f"{alias}.json"
        if not m.is_dir() or out.exists():
            continue
        try:
            run(["scripts/evaluation/eval_general_text.py", "--model", m,
                 "--stream", VAL_STREAM, "--teacher", TEACHER,
                 "--teacher-revision", TEACHER_REVISION,
                 "--dtype", "bfloat16", "--out", out])
        except subprocess.CalledProcessError as exc:
            print(f"  {alias}: general-text diagnostics failed: {exc}", flush=True)
    mark(args.session, "GENERAL_TEXT_DONE")


def stage_three_mode(args) -> None:
    """The frozen 150-prompt battery, unchanged, on the arms this session trained."""
    for alias, name in SESSION_ARMS[args.session].items():
        d = OUT / "three_mode" / alias
        if (d / "report.json").exists():
            mark(args.session, f"EVAL_DONE:{alias}")
            continue
        m = model_dir(name)
        if not m.is_dir():
            mark(args.session, f"EVAL_SKIPPED:{alias}:no_checkpoint")
            continue
        now = spent_usd(args)
        need = args.per_eval_minutes / 60 * args.rate
        if now + need > args.soft_stop_usd:
            mark(args.session, f"ABORTED_AT_GATE:budget:{now:.2f}+{need:.2f}")
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
        mark(args.session, f"EVAL_DONE:{alias}")
    mark(args.session, "EVAL_DONE")


STAGES = {
    "s1": (("init_nll", stage_init_nll), ("step0_probe", stage_step0_probe),
           ("publish_step0", stage_publish_step0)),
    "s2": (("fetch_step0", stage_fetch_step0),
           ("throughput_gate", stage_throughput_gate), ("gate", stage_gate),
           ("train", stage_train), ("general_text", stage_general_text),
           ("three_mode", stage_three_mode)),
    "s4": (("fetch_step0", stage_fetch_step0), ("gate", stage_gate),
           ("train", stage_train), ("general_text", stage_general_text),
           ("three_mode", stage_three_mode)),
}
STAGES["s3"] = STAGES["s2"]
# A failure in any of these means nothing downstream is worth paying for.
BLOCKING = {"init_nll", "fetch_step0", "throughput_gate", "gate", "train"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=sorted(STAGES))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--per-init-minutes", type=float, default=16.0)
    ap.add_argument("--per-probe-minutes", type=float, default=55.0)
    ap.add_argument("--probe-max-new-tokens", type=int, default=2048,
                    help="recorded cap for the step-0 DIAGNOSTIC probe; "
                         "the formal endpoint is unrestricted (P18)")
    ap.add_argument("--per-arm-minutes", type=float, default=250.0)
    ap.add_argument("--per-eval-minutes", type=float, default=35.0)
    args = ap.parse_args()
    args.t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E8b {args.session}: spent ${args.spent_usd:.2f}, soft stop "
          f"${args.soft_stop_usd:.2f}, hard ${args.authorized_usd:.2f} "
          f"at ${args.rate}/h", flush=True)
    plan = STAGES[args.session]
    chosen = plan if args.stage == "all" else [
        (n, f) for n, f in plan if n == args.stage]
    for name, fn in chosen:
        try:
            fn(args)
        except (subprocess.CalledProcessError, AssertionError, OSError,
                ValueError, KeyError, SystemExit) as exc:
            mark(args.session, f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark(args.session, "ABORTED_AFTER_BLOCKING_FAILURE")
                break
            continue
    mark(args.session, "ALL_DONE")


if __name__ == "__main__":
    sys.exit(main())

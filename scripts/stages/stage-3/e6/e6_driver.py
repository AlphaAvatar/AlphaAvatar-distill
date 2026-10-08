#!/usr/bin/env python
"""Experiment 6: evaluate six existing checkpoints on the frozen 150-prompt
protocol. Trains nothing, writes no weights, modifies no checkpoint.

    /opt/train/bin/python scripts/pod/e6_driver.py --stage all \
        --spent-usd 0.42 --authorized-usd 2.48

**The evaluation rung is pinned to 860000 for every arm**, exactly as E4 pinned
it. The harness samples its 150 examples from the rung it is given, and the
1.60M/2.96M/5.50M rungs hold progressively more sessions — so passing an arm's
own training rung would resample the battery per arm and end the comparison. The
mask is asserted against `d6e24e0b…` after every evaluation.

Stages: notrain -> three_mode

Markers: NOTRAIN_PROVEN -> EVAL_DONE:<arm> … -> EVAL_DONE -> ALL_DONE
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e6.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
CKPT = Path("/workspace/ckpt")

EVAL_RUNG = 860000          # pinned; see the module docstring
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"

# Evaluation order is deliberate and does not change what any arm sees.
#
# The four RELAY arms run first because they are on disk the moment setup ends,
# while the two dev-box arms are still arriving over a ~0.72 MB/s uplink. The pod
# therefore evaluates for the first ~40 minutes of a transfer it would otherwise
# have spent idle. Within the relay group the order still alternates rungs, so a
# session that runs out of budget yields a partial scale curve rather than two
# seeds of one rung.
ORDER = ["E1-1.60M-sa", "E1-2.96M-sa", "E1-5.50M-sa", "E1-1.60M-sb",
         "E1-2.96M-sb", "E1-5.50M-sb"]
CKPT_LOCAL = Path("/workspace/ckpt_local")
STAGED = CKPT_LOCAL / "STAGED"          # launcher writes it after the full pass
STAGING_FAILED = CKPT_LOCAL / "FAILED"  # launcher writes it if the upload dies

# Scripts this driver executes. None of them may contain an executable optimizer
# step: E6 is evaluation-only and that is proven by parsing, not asserted in a
# comment (the same check D0 used on 2026-08-04).
EXECUTED = ("scripts/evaluation/run_three_mode_diagnostic.py",
            "scripts/evaluation/diagnose_training_recall.py")


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


def stage_notrain(args) -> None:
    """Prove by AST that no path this driver runs takes an optimizer step."""
    findings = []
    for rel in EXECUTED:
        src = (REPO / rel).read_text()
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("step", "backward")):
                findings.append(f"{rel}:{node.lineno} .{node.func.attr}()")
    report = {"executed_scripts": list(EXECUTED),
              "optimizer_step_or_backward_calls": findings,
              "trains_anything": bool(findings),
              "training_pack_present": (REPO / "artifacts/stage3/ladder_uniform").exists()}
    (OUT / "e6_notrain_proof.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "e6_notrain_proof.json").write_text(json.dumps(report, indent=2) + "\n")
    if findings:
        raise AssertionError(f"executable training calls found: {findings}")
    if report["training_pack_present"]:
        raise AssertionError("the training ladder pack is present; it must not be")
    print("no optimizer step or backward call on any executed path", flush=True)
    mark("NOTRAIN_PROVEN")


def _manifest() -> dict:
    return json.loads((OUT / "e6_checkpoint_manifest.json").read_text())


def _await_staging(alias: str, deadline_s: float) -> dict:
    """Block until the launcher finishes staging the dev-box arms, or give up.

    Setup writes a manifest holding only the relay arms, because the dev-box
    checkpoints are still uploading while it runs. The launcher rewrites that
    manifest over all six once the transfer lands and touches STAGED. Waiting
    here is what lets the relay arms be evaluated during the transfer instead of
    after it.
    """
    waited = 0.0
    while time.time() < deadline_s:
        if STAGING_FAILED.exists():
            mark(f"STAGING_FAILED_UPSTREAM:{alias}")
            return _manifest()
        if STAGED.exists():
            mark(f"STAGING_LANDED:{alias}:{waited / 60:.1f}min")
            return _manifest()
        time.sleep(30)
        waited += 30
        if waited % 300 == 0:
            print(f"  waiting for the dev-box upload ({waited / 60:.0f} min)",
                  flush=True)
    mark(f"STAGING_WAIT_TIMEOUT:{alias}")
    return _manifest()


def stage_three_mode(args) -> None:
    """The binding harness, unchanged, on the pinned 150-example battery."""
    manifest = _manifest()
    started = time.time()
    wait_deadline = started + args.stage_wait_minutes * 60
    for alias in ORDER:
        if alias not in manifest["arms"]:
            manifest = _await_staging(alias, wait_deadline)
        arm = manifest["arms"].get(alias)
        if arm is None:
            mark(f"EVAL_SKIPPED:{alias}:not_staged")
            continue
        d = OUT / "three_mode" / alias
        if (d / "free.generations.jsonl").exists() and (d / "report.json").exists():
            print(f"{alias} already evaluated; skipping", flush=True)
            mark(f"EVAL_DONE:{alias}")
            continue

        # Budget guard. Priced from actual elapsed session time, never from a
        # projection: a pod that has already overrun must stop rather than start
        # another arm it cannot pay for.
        spent = args.spent_usd + (time.time() - started) / 3600 * args.rate
        need = args.per_arm_minutes / 60 * args.rate
        if spent + need > args.authorized_usd:
            mark(f"ABORTED_AT_GATE:budget:{spent:.2f}+{need:.2f}>{args.authorized_usd:.2f}")
            print(f"STOPPING before {alias}: ${spent:.2f} spent, ${need:.2f} needed, "
                  f"${args.authorized_usd:.2f} authorized", flush=True)
            return

        model = Path(arm["path"])
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", d], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", model, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", d / "forced"])
        mask = json.loads((d / "report.json").read_text())["inclusion"]["mask_sha256"]
        if mask != EXPECTED_MASK:
            raise AssertionError(f"{alias}: inclusion mask {mask} != binding")
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


STAGES = {"notrain": stage_notrain, "three_mode": stage_three_mode}
BLOCKING = ("notrain",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--authorized-usd", type=float, default=2.48)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--per-arm-minutes", type=float, default=11.0)
    ap.add_argument("--stage-wait-minutes", type=float, default=130.0,
                    help="how long to wait for the dev-box upload before "
                         "giving up on the arms it carries")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        except (subprocess.CalledProcessError, AssertionError, OSError,
                ValueError, KeyError) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark("ABORTED_AFTER_NOTRAIN_FAILURE")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

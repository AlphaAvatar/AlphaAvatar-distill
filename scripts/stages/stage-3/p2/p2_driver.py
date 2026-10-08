#!/usr/bin/env python
"""Train the two p2_ceheavy arms, then run every specified evaluation.

    /opt/train/bin/python scripts/pod/p2_driver.py --stage all

Treatment: `kd_weight 1.0 -> 0.25`, `ce_weight 0.25 -> 1.0`. `kd_scope` stays
"all", so both denominators are unchanged (KD 1,471,467; CE 864,750) and only the
scalar mixing moves. Baseline is P1 = the existing P0-real arms; no baseline
re-run.

Stages: train -> tokenizer -> nll -> three_mode

`tokenizer` is its own stage on purpose. `Trainer.save_checkpoint` writes
`config.json`, `generation_config.json` and `model.safetensors` but **no
tokenizer**, which has now broken evaluation twice (Experiment 2 phase 1's
eval_ppl, then P0-assistant's three-mode). Copying it is done up front and
verified, rather than discovered when an evaluation dies an hour later.

Markers: TRAIN_DONE:<arm> -> TOKENIZER_OK -> NLL_DONE -> EVAL_DONE:<arm> -> ALL_DONE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/p2.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
HOLDOUT = REPO / "data/warmup/holdout_v1.jsonl"
INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
ARMS = {"P2-ceheavy-sa": "p2_ceheavy_sa", "P2-ceheavy-sb": "p2_ceheavy_sb"}

# The pinned historical protocol, asserted rather than assumed.
HOLDOUT_SHA = "2d49f637a711ae82510fd55a3af98e332314f972780841869508aebe7b3cd8e8"
EXPECTED_EVAL_TOKENS = 21080          # both P0-real arms scored exactly this
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


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


def model_dir(name: str) -> Path:
    return REPO / f"artifacts/stage3/{name}/checkpoints/step_001023/model"


def stage_train(args):
    for alias, name in ARMS.items():
        if model_dir(name).is_dir():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        cfg_path = REPO / f"configs/stage3/p2/{name}.json"
        cfg = json.loads(cfg_path.read_text())
        assert cfg["loss"] == {"ce_weight": 1.0, "kd_weight": 0.25,
                               "kd_temperature": 1.0, "kd_scope": "all"}, cfg["loss"]
        assert "truncate_padding" not in cfg["batch"], cfg["batch"]
        run(["scripts/training/train_stage3.py", "--config", cfg_path])
        mark(f"TRAIN_DONE:{alias}")
    mark("TRAIN_DONE")


def stage_tokenizer(args):
    """save_checkpoint writes no tokenizer; install it and prove it works."""
    for alias, name in ARMS.items():
        d = model_dir(name)
        if not d.is_dir():
            mark(f"TOKENIZER_SKIPPED:{alias}")
            continue
        for f in TOKENIZER_FILES:
            shutil.copy(INIT / f, d / f)
        from transformers import AutoTokenizer
        a = AutoTokenizer.from_pretrained(str(d))
        b = AutoTokenizer.from_pretrained(str(INIT))
        assert len(a) == len(b), (len(a), len(b))
        assert (a.chat_template or "") == (b.chat_template or "")
        rendered = a.apply_chat_template([{"role": "user", "content": "x"}],
                                         tokenize=False, add_generation_prompt=True)
        assert rendered.rstrip().endswith("<think>"), rendered[-40:]
        print(f"  {alias}: tokenizer installed, vocab {len(a)}, template verified",
              flush=True)
    mark("TOKENIZER_OK")


def stage_nll(args):
    """FineWeb held-out NLL under the exact protocol used for P0-real."""
    out = OUT / "p2_holdout_nll.json"
    if out.exists():
        print("NLL already done; skipping", flush=True)
        return mark("NLL_DONE")
    got = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
    assert got == HOLDOUT_SHA, f"holdout corpus mismatch: {got}"
    models = []
    for name in ARMS.values():
        models += ["--model", str(model_dir(name))]
    run(["scripts/evaluation/eval_ppl.py", "--data", "data/warmup/holdout_v1.jsonl",
         *models, "--max-seq-len", 1024, "--dtype", "bfloat16", "--out", out])
    report = json.loads(out.read_text())
    for r in report["results"]:
        n = r["eval_tokens"]
        flag = "OK" if n == EXPECTED_EVAL_TOKENS else f"!! expected {EXPECTED_EVAL_TOKENS}"
        print(f"  {r['model'].split('/')[-4]}: nll {r['mean_nll_nats']} "
              f"ppl {r['perplexity']} tokens {n} {flag}", flush=True)
    mark("NLL_DONE")


def stage_three_mode(args):
    for alias, name in ARMS.items():
        d = OUT / "three_mode" / alias
        if (d / "oracle.generations.jsonl").exists():
            print(f"{alias} already evaluated; skipping", flush=True)
            mark(f"EVAL_DONE:{alias}")
            continue
        m = model_dir(name)
        if not m.is_dir():
            mark(f"EVAL_SKIPPED:{alias}:no_checkpoint")
            continue
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK, "--rung", 860000,
             "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", d], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK, "--rung", 860000,
             "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", d / "forced"])
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "train", "tokenizer", "nll", "three_mode"))
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = {"train": stage_train, "tokenizer": stage_tokenizer,
              "nll": stage_nll, "three_mode": stage_three_mode}
    for name in (list(stages) if args.stage == "all" else [args.stage]):
        try:
            stages[name](args)
        except (subprocess.CalledProcessError, AssertionError) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

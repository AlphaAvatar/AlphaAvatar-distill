#!/usr/bin/env python
"""Experiment 4: train P2-CE-heavy at the 1.60M rung, then evaluate four models.

    /opt/train/bin/python scripts/pod/e4_driver.py --stage all

Trains `e4_p2_r1600k_{sa,sb}` from the Stage 1 PCA init — **not** continued from
the P2-0.86M checkpoint — and evaluates them beside the existing P1-1.60M
checkpoints, which are re-evaluated rather than retrained because their recorded
numbers came from the older 76-prompt behaviour wave with the degeneration stop
active. P2-0.86M is not touched at all; its results are the reference.

**The evaluation rung is pinned to 860000 for every arm.** The harness samples
its 150 examples from the rung it is given, and the 1.60M rung holds 2,649
sessions against the 0.86M rung's 1,502 — so passing the training rung here
would silently resample the battery and end the like-for-like comparison. The
mask is asserted against `d6e24e0b…` after every evaluation.

Stages: train -> tokenizer -> movement -> three_mode

Markers: TRAIN_DONE:<arm> -> TOKENIZER_OK -> MOVEMENT_DONE -> EVAL_DONE:<arm>
         -> ALL_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e4.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"

TRAIN_ARMS = {"E4-P2-1600k-sa": "e4_p2_r1600k_sa",
              "E4-P2-1600k-sb": "e4_p2_r1600k_sb"}
# Re-evaluated, never retrained; staged and hash-verified by setup.
P1_REF = {"P1-1600k-sa": Path("/workspace/ckpt/e1_r1600k_sa_pca/model"),
          "P1-1600k-sb": Path("/workspace/ckpt/e1_r1600k_sb_pca/model")}

EVAL_RUNG = 860000          # pinned; see the module docstring
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
STEP = "step_001761"
OBJECTIVE = {"ce_weight": 1.0, "kd_weight": 0.25,
             "kd_temperature": 1.0, "kd_scope": "all"}


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


def stage_train(args):
    for alias, name in TRAIN_ARMS.items():
        if model_dir(name).is_dir():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        cfg_path = REPO / f"configs/stage3/e4/{name}.json"
        cfg = json.loads(cfg_path.read_text())
        # Assert the arm is what it claims to be, from the file that will train.
        assert cfg["loss"] == OBJECTIVE, cfg["loss"]
        assert cfg["rung"] == 1600000, cfg["rung"]
        assert cfg["schedule"]["total_steps"] == 1761, cfg["schedule"]
        assert cfg["student_path"].endswith("qwen3_0p6b_init_v0/checkpoint"), \
            "must fork from the Stage 1 init, not from a trained checkpoint"
        assert "lora" not in cfg
        assert "truncate_padding" not in cfg["batch"]
        joined = " ".join(cfg["trainable_patterns"])
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert proj in joined, f"{proj} must be full-rank trainable"
        run(["scripts/training/train_stage3.py", "--config", cfg_path])
        mark(f"TRAIN_DONE:{alias}")


def stage_tokenizer(args):
    """save_checkpoint writes no tokenizer; install it and prove it works."""
    from transformers import AutoTokenizer

    for alias, name in TRAIN_ARMS.items():
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


def stage_movement(args):
    for alias, name in TRAIN_ARMS.items():
        out = OUT / "e4_movement" / f"{alias}.json"
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


def stage_three_mode(args):
    """The binding harness, unchanged, on the pinned 150-example battery."""
    targets = {a: model_dir(n) for a, n in TRAIN_ARMS.items()}
    targets.update(P1_REF)
    for alias, m in targets.items():
        d = OUT / "three_mode" / alias
        if (d / "oracle.generations.jsonl").exists():
            print(f"{alias} already evaluated; skipping", flush=True)
            mark(f"EVAL_DONE:{alias}")
            continue
        if not Path(m).is_dir():
            mark(f"EVAL_SKIPPED:{alias}:no_checkpoint")
            continue
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "free", "oracle", "--out", d], py=VLLM_PY)
        run(["scripts/evaluation/run_three_mode_diagnostic.py",
             "--student", m, "--label", alias, "--pack", PACK,
             "--rung", EVAL_RUNG, "--sessions", SESSIONS, "--n", args.n,
             "--modes", "forced", "--out", d / "forced"])
        mask = json.loads((d / "report.json").read_text())["inclusion"]["mask_sha256"]
        assert mask == EXPECTED_MASK, f"{alias}: inclusion mask {mask} != binding"
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


STAGES = {"train": stage_train, "tokenizer": stage_tokenizer,
          "movement": stage_movement, "three_mode": stage_three_mode}
BLOCKING = ("train",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        except (subprocess.CalledProcessError, AssertionError, OSError) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark("ABORTED_AFTER_TRAIN_FAILURE")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

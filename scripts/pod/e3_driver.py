#!/usr/bin/env python
"""Experiment 3: train A1 then A2, then evaluate everything with the P1 harness.

    /opt/train/bin/python scripts/pod/e3_driver.py --stage all

Arms (A0 = P2-ceheavy is not retrained; its recorded results are the control):

    A1  FFN + all norms full-rank, attention projections FROZEN
    A2  A1 + LoRA r8 on q/k/v/o, base projections frozen, one optimizer group

Order matters and is enforced: **both A1 seeds finish, and pass a freeze gate,
before A2 starts.** The gate is not decoration — it recomputes the attention
projection movement against the Stage 1 init and refuses to spend money on A2
if A1's frozen set moved at all.

Held-out NLL is deliberately NOT run here. It costs 25 s per model on the dev
box CPU and reproduces the GPU value to 0.02%, so running it off-pod keeps the
baseline and the treatment arms on one device, needs no upload against a full
LFS quota, and removes ~24 min of paid GPU time.

Stages: train_a1 -> gate_a1 -> train_a2 -> tokenizer -> merge_check -> movement
        -> three_mode

Markers: TRAIN_DONE:<arm> -> A1_GATE_PASS/FAIL -> TOKENIZER_OK -> MERGE_OK
         -> MOVEMENT_DONE -> EVAL_DONE:<arm> -> ALL_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e3.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"

A1 = {"A1-frozen-attn-sa": "e3_a1_frozen_attn_sa",
      "A1-frozen-attn-sb": "e3_a1_frozen_attn_sb"}
A2 = {"A2-lora-attn-sa": "e3_a2_lora_attn_sa",
      "A2-lora-attn-sb": "e3_a2_lora_attn_sb"}
ARMS = {**A1, **A2}
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
STEP = "step_001023"


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


def config_path(name: str) -> Path:
    return REPO / f"configs/stage3/e3/{name}.json"


def movement_path(alias: str) -> Path:
    return OUT / "e3_movement" / f"{alias}.json"


def train_arms(arms: dict) -> None:
    for alias, name in arms.items():
        if model_dir(name).is_dir():
            print(f"{alias} already trained; skipping", flush=True)
            mark(f"TRAIN_DONE:{alias}")
            continue
        cfg = json.loads(config_path(name).read_text())
        # Assert the arm is what it claims to be, from the file that will train.
        # The P2-ceheavy objective, which the baseline was trained under.
        assert cfg["loss"] == {"ce_weight": 1.0, "kd_weight": 0.25,
                               "kd_temperature": 1.0, "kd_scope": "all"}, cfg["loss"]
        assert cfg["rung"] == 860000 and cfg["schedule"]["total_steps"] == 1023
        assert "truncate_padding" not in cfg["batch"], cfg["batch"]
        assert not any(p in str(cfg["trainable_patterns"])
                       for p in ("q_proj", "k_proj", "v_proj", "o_proj")), \
            "attention projections must not be in trainable_patterns"
        for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
            assert field not in cfg["optim"], field
        if name.startswith("e3_a2"):
            assert cfg["lora"]["rank"] == 32 and cfg["lora"]["alpha"] == 16
            assert cfg["lora"]["dropout"] == 0.0 and cfg["lora"]["bias"] == "none"
        else:
            assert "lora" not in cfg
        run(["scripts/training/train_stage3.py", "--config", config_path(name)])
        mark(f"TRAIN_DONE:{alias}")


def stage_train_a1(args):
    train_arms(A1)


def measure_movement(alias: str, name: str) -> dict:
    out = movement_path(alias)
    if not out.exists():
        run(["scripts/evaluation/parameter_movement.py", "--init", INIT,
             "--checkpoint", model_dir(name), "--label", alias, "--out", out])
    return json.loads(out.read_text())


def stage_gate_a1(args):
    """A1's whole claim is that attention did not move. Verify before paying for A2."""
    ok = True
    for alias, name in A1.items():
        if not model_dir(name).is_dir():
            print(f"{alias}: no checkpoint", flush=True)
            ok = False
            continue
        rep = measure_movement(alias, name)
        attn = rep["by_group"]["attn_proj"]["delta_fro"]
        ffn = rep["by_group"]["ffn"]["delta_fro"]
        emb = rep["by_group"]["embedding"]["delta_fro"]
        print(f"  {alias}: attn_proj Δ {attn:.6g} | ffn Δ {ffn:.6g} | "
              f"embedding Δ {emb:.6g}", flush=True)
        if attn != 0.0 or emb != 0.0:
            print(f"  {alias}: FROZEN SET MOVED", flush=True)
            ok = False
        if ffn <= 0.0:
            print(f"  {alias}: FFN did not move; the arm trained nothing", flush=True)
            ok = False
    mark("A1_GATE_PASS" if ok else "A1_GATE_FAIL")
    if not ok:
        raise AssertionError("A1 freeze gate failed; refusing to start A2")


def stage_train_a2(args):
    train_arms(A2)


def stage_tokenizer(args):
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
        assert len(a) == len(b), (len(a), len(b))
        assert (a.chat_template or "") == (b.chat_template or "")
        rendered = a.apply_chat_template([{"role": "user", "content": "x"}],
                                         tokenize=False, add_generation_prompt=True)
        assert rendered.rstrip().endswith("<think>"), rendered[-40:]
        print(f"  {alias}: tokenizer installed, vocab {len(a)}", flush=True)
    mark("TOKENIZER_OK")


def stage_merge_check(args):
    """Prove the A2 artifact that gets evaluated needs no adapter code."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(REPO / "src"))
    from aadistill.training.lora import LoRALinear

    report = {}
    for alias, name in A2.items():
        ckpt = run_dir(name) / f"checkpoints/{STEP}"
        if not (ckpt / "model").is_dir():
            continue
        tensors = load_file(str(ckpt / "model" / "model.safetensors"))
        lora_keys = [k for k in tensors if "lora" in k.lower()]
        assert not lora_keys, f"{alias}: adapter leaked into model/: {lora_keys[:3]}"
        meta = json.loads((ckpt / "checkpoint_meta.json").read_text())
        assert meta["model_dir_is_merged"] is True
        assert (ckpt / "lora_state.safetensors").is_file(), \
            f"{alias}: no resumable LoRA state"

        model = AutoModelForCausalLM.from_pretrained(ckpt / "model",
                                                     dtype=torch.bfloat16).eval()
        assert not any(isinstance(m, LoRALinear) for m in model.modules())
        with torch.no_grad():
            logits = model(torch.arange(1, 33).unsqueeze(0)).logits
        assert torch.isfinite(logits).all(), f"{alias}: non-finite logits"
        report[alias] = {
            "lora_keys_in_model_dir": 0,
            "resumable_state_bytes": (ckpt / "lora_state.safetensors").stat().st_size,
            "lora_config": meta["lora_config"],
            "n_lora_modules": meta["n_lora_modules"],
            "loads_without_adapter_code": True,
            "finite_logits": True,
        }
        print(f"  {alias}: merged checkpoint clean, loads with no adapter",
              flush=True)
        del model
    (OUT / "e3_merge_check.json").write_text(json.dumps(report, indent=1))
    mark("MERGE_OK")


def stage_movement(args):
    for alias, name in ARMS.items():
        if model_dir(name).is_dir():
            measure_movement(alias, name)
    mark("MOVEMENT_DONE")


def stage_three_mode(args):
    """The unified P1 harness, unchanged: same 150 examples, same decoding."""
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
        # The comparison against A0 is only valid on the same fixed examples.
        mask = json.loads((d / "report.json").read_text())["inclusion"]["mask_sha256"]
        assert mask == EXPECTED_MASK, f"{alias}: inclusion mask {mask} != P1's"
        mark(f"EVAL_DONE:{alias}")
    mark("EVAL_DONE")


STAGES = {
    "train_a1": stage_train_a1,
    "gate_a1": stage_gate_a1,
    "train_a2": stage_train_a2,
    "tokenizer": stage_tokenizer,
    "merge_check": stage_merge_check,
    "movement": stage_movement,
    "three_mode": stage_three_mode,
}
# A1 must finish and pass its gate before A2 trains; a failure there stops the
# sequence instead of silently producing an uninterpretable A2.
BLOCKING = ("train_a1", "gate_a1")


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
                mark("ABORTED_BEFORE_A2")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

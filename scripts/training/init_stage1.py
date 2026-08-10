"""Stage 1: build a teacher-projected student checkpoint.

Usage:
    uv run python scripts/training/init_stage1.py --config configs/stage1/qwen3_0p6b_from_4b_thinking.json

Consumes the Stage 0 activation-statistics cache, initializes the student via
global activation-PCA stream projection + sandwich init (see
src/aadistill/init/sandwich.py), and writes:

    <output_dir>/checkpoint/          initialized student (+ tokenizer)
    <output_dir>/random_baseline/     same geometry, standard random init
    <output_dir>/manifest.json        recipe hash + gate-check record

Gate checks run inline: parameter count, forward smoke test, save/reload
logit equality. Evaluation is a separate script (scripts/evaluation/eval_ppl.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state, hardware_report, set_determinism
from aadistill.infrastructure.manifest import sha256_file, sha256_json, write_manifest
from aadistill.init.sandwich import init_student
from aadistill.models.student import build_student, build_student_config
from aadistill.models.teacher import load_teacher


def forward_smoke(model, tokenizer) -> dict:
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits
    return {
        "input_tokens": ids.shape[1],
        "logits_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "top_token": tokenizer.decode(logits[0, -1].argmax()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = sha256_json(config)
    set_determinism(config["seed"])
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]

    stats_dir = REPO_ROOT / config["stats_dir"]
    stats_path = stats_dir / "activation_stats.safetensors"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stage 0 cache missing: {stats_path}")

    print(f"Loading teacher {config['teacher_model_id']} on {device} ...", flush=True)
    teacher, tokenizer, identity = load_teacher(
        config["teacher_model_id"], revision=config.get("teacher_revision"),
        dtype=config["dtype"], device=device,
    )

    from safetensors.torch import load_file
    state = load_file(str(stats_path))

    student_cfg = build_student_config(teacher.config, config["student_geometry"])
    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    student = build_student(student_cfg, dtype, config["seed"])
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {n_params / 1e6:.1f}M params", flush=True)

    # The student must inherit the teacher's positional basis. Checked against the
    # runtime frequencies of both models rather than a config attribute, because
    # the config attribute is exactly what the transformers 4.x/5.x
    # `rope_parameters` skew gets wrong — silently, and by 500x on this teacher.
    from aadistill.models.student import (
        assert_rope_matches_config, stored_rope_base,
    )
    # Two different questions, and mixing them is a mistake worth naming: the
    # *stored* bases must be exactly equal, because the student config inherits
    # the teacher's; the *runtime* base of each model must match its own config,
    # which is the version-skew guard. Comparing the two recovered runtime numbers
    # to each other instead would compare an fp32 rotary buffer against the bf16
    # one `build_student` produces, and fail on precision rather than on skew.
    stored_student = stored_rope_base(student_cfg)
    stored_teacher = stored_rope_base(teacher.config)
    if stored_student != stored_teacher:
        raise RuntimeError(
            f"student config records RoPE base {stored_student} but the teacher "
            f"records {stored_teacher}; the student must inherit it")
    student_rope = assert_rope_matches_config(student, student_cfg, "student")
    assert_rope_matches_config(teacher, teacher.config, config["teacher_model_id"])
    print(f"RoPE base {stored_teacher:,.0f} inherited; student runtime "
          f"{student_rope:,.0f}", flush=True)

    baseline_record = None
    if config.get("save_random_baseline"):
        baseline_dir = output_dir / "random_baseline"
        student.save_pretrained(baseline_dir)
        tokenizer.save_pretrained(baseline_dir)
        baseline_record = {"path": str(baseline_dir.relative_to(REPO_ROOT)),
                           "seed": config["seed"]}
        print(f"Saved random baseline to {baseline_dir}", flush=True)

    # The only E8 variable. When absent the positional pairwise-merge map is
    # used, so every earlier Stage 1 config keeps its exact behaviour; when
    # present it replaces the depth map and nothing else — the projection, the
    # Q-head rule, the FFN neuron rule and the norm treatment below are reached
    # identically either way (asserted in tests/init/test_contribution.py).
    kept_layers = None
    depth_map_record = None
    if config.get("depth_map_path"):
        dm_path = REPO_ROOT / config["depth_map_path"]
        if not dm_path.is_file():
            raise FileNotFoundError(f"depth map missing: {dm_path}")
        depth_map_record = json.loads(dm_path.read_text())
        kept_layers = list(depth_map_record["kept_teacher_layers"])
        if depth_map_record.get("teacher", {}).get("revision") not in (
                None, config.get("teacher_revision")):
            raise ValueError(
                f"depth map was searched against teacher revision "
                f"{depth_map_record['teacher']['revision']} but this init uses "
                f"{config.get('teacher_revision')}")
        depth_map_record["depth_map_sha256"] = sha256_file(dm_path)
        print(f"depth map from {config['depth_map_path']}: keeping {kept_layers}",
              flush=True)

    started = time.time()
    diagnostics = init_student(teacher, student, state, kept_layers=kept_layers)
    init_seconds = time.time() - started
    print(f"Init done in {init_seconds:.0f}s; projection energy captured "
          f"{diagnostics['projection']['energy_captured_frac']:.4f}", flush=True)

    smoke = forward_smoke(student, tokenizer)
    print(f"Forward smoke: {smoke}", flush=True)
    if not smoke["finite"]:
        raise RuntimeError("Initialized student produced non-finite logits")

    ckpt_dir = output_dir / "checkpoint"
    student.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Reload gate: weights must round-trip bitwise. Logits are compared with a
    # small tolerance instead of exact equality: identical weights in two
    # module instances can differ by a few bf16 ULPs on this CPU (oneDNN/AMX
    # kernel blocking varies with tensor memory alignment) even though each
    # instance is self-deterministic. Measured and logged per AGENTS.md P5.
    from transformers import Qwen3ForCausalLM
    reloaded = Qwen3ForCausalLM.from_pretrained(ckpt_dir, dtype=dtype).eval()
    sd_mem, sd_re = student.state_dict(), reloaded.state_dict()
    weight_mismatches = [k for k in sd_mem if not torch.equal(sd_mem[k], sd_re[k])]
    if set(sd_mem) != set(sd_re) or weight_mismatches:
        raise RuntimeError(f"Checkpoint weight round-trip failed: {weight_mismatches[:5]}")
    ids = tokenizer("reload check", return_tensors="pt").input_ids
    with torch.no_grad():
        logit_diff = (student(ids).logits - reloaded(ids).logits).abs().max().item()
    reload_record = {"weights_bitwise_equal": True,
                     "logit_max_abs_diff": logit_diff,
                     "logit_tolerance": 0.5,
                     "nondeterminism_note": "cross-instance bf16 CPU kernel variance"}
    if logit_diff > reload_record["logit_tolerance"]:
        raise RuntimeError(f"Reloaded logits diverge beyond tolerance: {logit_diff}")

    manifest = {
        "stage": "stage1_projection_init",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": config,
        "config_sha256": config_hash,
        "recipe_hash": sha256_json({"config": config_hash,
                                    "code": code_state(str(REPO_ROOT))}),
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
        "teacher": identity,
        "stage0_stats": {
            "path": str(stats_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(stats_path),
            "manifest": str((stats_dir / "manifest.json").relative_to(REPO_ROOT)),
        },
        "student": {
            "config": student_cfg.to_diff_dict(),
            "num_parameters": n_params,
            "dtype": config["dtype"],
            "resolved_rope_base": student_rope,
        },
        "environment": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        },
        "init_diagnostics": diagnostics,
        "depth_map_source_artifact": depth_map_record,
        "init_seconds": round(init_seconds, 1),
        "forward_smoke": smoke,
        "reload_check": reload_record,
        "random_baseline": baseline_record,
        "checkpoint": {
            "path": str(ckpt_dir.relative_to(REPO_ROOT)),
            "model_sha256": sha256_file(ckpt_dir / "model.safetensors"),
        },
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

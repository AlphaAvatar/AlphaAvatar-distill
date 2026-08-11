#!/usr/bin/env python3
"""Build a depth-only student: the teacher with eight blocks deleted, nothing else.

E8b's DP and DC cells. Depth is the *only* compression: teacher hidden width 2560,
FFN 9728, 32 Q heads, 8 KV heads, embeddings, tied lm head, norms, vocabulary and
tokenizer are all carried over untouched. 36 -> 28 layers and that is all.

    PYTHONPATH=src python scripts/training/build_depth_only_init.py \\
        --map positional --out artifacts/stage1/e8b_dp_init
    PYTHONPATH=src python scripts/training/build_depth_only_init.py \\
        --map contribution --out artifacts/stage1/e8b_dc_init

Verbatim copy, not re-projection
--------------------------------
Stage 1's sandwich init could produce these — with an identity projection at equal
width it reproduces the teacher, and there is a test that says so. But it would also
fold each `input_layernorm` into q/k/v and re-solve the final norm, and those are
extra transformations. The factorial's "no compression" cell should contain no
compression, so the kept blocks are copied bit-for-bit.

That choice buys a strong correctness check, which this script performs: a
depth-only model must be **logit-identical** to the teacher evaluated with those
same blocks bypassed through the residual path — which is exactly the operation
E8a's search objective measured. DP and DC *are* the ablated teachers E8a scored,
materialized as checkpoints. So E8a's own numbers (positional KL 1.932531,
contribution 0.620586) are already step-0 statements about these two models on the
calibration set, and E8b's step-0 measurements extend that to the evaluation
streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.init.contribution import bypassed_blocks  # noqa: E402
from aadistill.init.sandwich import depth_span_map  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
# Frozen by E8a; see logs/e8_step0_report.md and artifacts/audit/e8_frozen_depth_map.json.
CONTRIBUTION_REMOVED = [2, 3, 15, 16, 20, 21, 26, 32]
STUDENT_LAYERS = 28


def resolve_map(name: str) -> tuple[list[int], list[int], str]:
    n_teacher = 36
    if name == "positional":
        kept = [s["representative"] for s in depth_span_map(n_teacher, STUDENT_LAYERS)]
        note = "canonical positional pairwise-merge map (sandwich.depth_span_map)"
    elif name == "contribution":
        kept = [i for i in range(n_teacher) if i not in CONTRIBUTION_REMOVED]
        note = "frozen E8a contribution-guided map"
    else:
        raise SystemExit(f"unknown map {name!r}")
    removed = sorted(set(range(n_teacher)) - set(kept))
    if len(kept) != STUDENT_LAYERS:
        raise SystemExit(f"map keeps {len(kept)} layers, need {STUDENT_LAYERS}")
    return kept, removed, note


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, choices=("positional", "contribution"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--verify-tokens", type=int, default=64,
                    help="sequence length for the logit-identity check")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from aadistill.models.teacher import load_teacher

    kept, removed, note = resolve_map(args.map)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"map {args.map}: keeping {kept}\n  removing {removed}", flush=True)
    teacher, tokenizer, identity = load_teacher(
        TEACHER, TEACHER_REVISION, dtype=args.dtype, device="cpu")
    teacher.config.use_cache = False
    t_cfg = teacher.config

    # The student config is the teacher's, with depth reduced. Nothing else moves.
    d = t_cfg.to_dict()
    for drop in ("architectures", "_name_or_path", "transformers_version"):
        d.pop(drop, None)
    d["num_hidden_layers"] = STUDENT_LAYERS
    d["layer_types"] = [t_cfg.layer_types[i] for i in kept] if getattr(
        t_cfg, "layer_types", None) else ["full_attention"] * STUDENT_LAYERS
    if any(lt != "full_attention" for lt in d["layer_types"]):
        raise SystemExit("teacher has non-full attention layers; depth-only "
                         "construction would change the attention pattern")
    s_cfg = Qwen3Config(**d)

    with torch.no_grad():
        student = Qwen3ForCausalLM(s_cfg).to(dtype).eval()
        student.model.embed_tokens.weight.copy_(teacher.model.embed_tokens.weight)
        student.tie_weights()
        student.model.norm.weight.copy_(teacher.model.norm.weight)
        for s_idx, t_idx in enumerate(kept):
            src, dst = teacher.model.layers[t_idx], student.model.layers[s_idx]
            src_sd = src.state_dict()
            missing = set(dst.state_dict()) ^ set(src_sd)
            if missing:
                raise SystemExit(f"layer state_dict mismatch: {sorted(missing)[:5]}")
            dst.load_state_dict(src_sd, strict=True)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"built {n_params:,} parameters", flush=True)

    from aadistill.models.student import assert_rope_from_config, stored_rope_base
    rope = assert_rope_from_config(s_cfg, "depth-only student")
    if abs(stored_rope_base(s_cfg) - stored_rope_base(t_cfg)) > 1:
        raise SystemExit("student did not inherit the teacher's RoPE base")

    ckpt = out / "checkpoint"
    student.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    # The correctness check that makes this construction auditable, performed on the
    # RELOADED checkpoint rather than the in-memory model — which matters, and cost
    # one confusing failure to learn: `Qwen3ForCausalLM(cfg).to(bfloat16)` casts the
    # rotary `inv_freq` buffer to bf16, while `from_pretrained` recomputes it in
    # fp32. The in-memory model therefore ran a lower-precision positional basis and
    # differed from the teacher by 0.78 in logits. `inv_freq` is non-persistent, so
    # the saved checkpoint is unaffected — but only the reloaded model is the thing
    # that will actually train, so it is the thing to verify.
    #
    # What this proves: a depth-only checkpoint is logit-identical to the teacher
    # with those same blocks bypassed through the residual path, which is exactly
    # the operation E8a's search objective measured. DP and DC *are* the ablated
    # teachers E8a scored, materialized.
    verify = None
    if not args.skip_verify:
        from transformers import AutoModelForCausalLM
        reloaded = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype).eval()
        reloaded.config.use_cache = False
        ids = tokenizer("The capital of France is Paris, and the capital of Japan is",
                        return_tensors="pt").input_ids[:, : args.verify_tokens]
        with torch.no_grad():
            s_logits = reloaded(ids).logits
            with bypassed_blocks(teacher, removed):
                t_logits = teacher(ids).logits
        same = bool(torch.equal(s_logits, t_logits))
        maxdiff = float((s_logits - t_logits).abs().max())
        verify = {"tokens": int(ids.shape[1]),
                  "checked_on": "reloaded checkpoint",
                  "bitwise_identical_to_ablated_teacher": same,
                  "max_abs_logit_diff": maxdiff,
                  "what_this_proves": "the depth-only checkpoint IS the ablated "
                                      "teacher E8a's objective scored"}
        print(f"logit identity vs bypassed teacher: identical={same} "
              f"max|diff|={maxdiff:.3e}", flush=True)
        del reloaded

    manifest = {
        "artifact": f"e8b_depth_only_init_{args.map}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "purpose": "E8b factorial cell: depth is the ONLY compression. Teacher "
                   "hidden width, FFN width, head structure, embeddings, lm head, "
                   "norms, vocabulary and tokenizer are carried over unchanged.",
        "map": {"name": args.map, "note": note, "kept_teacher_layers": kept,
                "removed_teacher_layers": removed},
        "teacher": identity,
        "student": {"config": s_cfg.to_diff_dict(), "num_parameters": n_params,
                    "dtype": args.dtype, "resolved_rope_base": rope},
        "construction": "verbatim state_dict copy of the kept blocks; no "
                        "projection, no norm folding, no head or neuron selection",
        "verification": verify,
        "checkpoint": {"path": str(ckpt.relative_to(REPO_ROOT)),
                       "model_sha256": None},
        "environment": {"transformers": __import__("transformers").__version__,
                        "torch": torch.__version__},
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    weights = ckpt / "model.safetensors"
    shards = sorted(ckpt.glob("model-*.safetensors"))
    if weights.is_file():
        manifest["checkpoint"]["model_sha256"] = sha256_file(weights)
    elif shards:
        manifest["checkpoint"]["shards"] = {
            p.name: sha256_file(p) for p in shards}
        manifest["checkpoint"]["index_sha256"] = sha256_file(
            ckpt / "model.safetensors.index.json")
    manifest["config_sha256"] = sha256_json(json.loads((ckpt / "config.json").read_text()))
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"params": n_params, "config_sha256": manifest["config_sha256"],
                      "model_sha256": manifest["checkpoint"]["model_sha256"],
                      "shards": list((manifest["checkpoint"].get("shards") or {})),
                      "rope": rope}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

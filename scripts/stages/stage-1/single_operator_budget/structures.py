#!/usr/bin/env python3
"""SOB1's three arms as real, buildable structures ($0, CPU): parameter counts from the teacher's pinned config and
the real model implementation, per-module, with legality, N, deviations, checkpoint sizes, FLOPs and memory.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/structures.py --write | --check

Counts are `transformers.Qwen3ForCausalLM` instantiated on the meta device from the teacher's config.json (its sha256
pinned by `teacher_identity.json`) with each arm's geometry overrides from `design.json`; ``parameters()`` yields the
tied embedding once. A geometry is legal only if each operator that builds it supports it (`design.json ::
operators`): KV heads and head_dim preserved and query heads a multiple of the KV heads; sizes within the teacher's.

Derived, never measured here (each is labelled so in the output):
* checkpoint bytes: init bf16 = 2P, final fp32 = 4P (the trainer saves fp32 parameters), a save point = 4P + 8T for
  AdamW's two fp32 moments of the T trainable parameters (the recipe's patterns: everything but the frozen tied
  embedding), the trainer-state remainder ignored (TPI1: 5.9 GB per save point at P 596M, T 440M, = 4P + 8T);
* matmul FLOPs per token (2 per parameter of every linear map, the tied head included) and causal attention FLOPs per
  token at the packed sequences' mean context (4 x mean context x query width per layer, forward);
* a training-memory estimate on one GPU, decomposed from TPI1's measured peak (`memory_model`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/single_operator_budget/design.json"
OUT = REPO / "logs/stages/stage-1/single_operator_budget/plans/structures.json"
SCHEMA = "aadistill.stage1.single_operator_budget.structures/v1"
TPI1_UNITS = REPO / "logs/stages/stage-1/teacher_parent_initialization/runs/tpi_train_20261117_180111/evidence/units.json"
TPI1_PLAN = REPO / "logs/stages/stage-1/teacher_parent_initialization/plans/training_plan.json"
GIB = 2 ** 30
#: L40S: 46,068 MiB as the driver reports it (nvidia-smi), not the 48 GB of the product name
L40S_GIB = 46068 / 1024
GEOMETRY_KEYS = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings")


class StructureError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def teacher_config(design: dict) -> dict:
    """The teacher's config.json from its pinned local snapshot, refused unless it hashes to the design's pin."""
    from huggingface_hub import try_to_load_from_cache
    t = design["teacher"]
    p = try_to_load_from_cache(t["model"], "config.json", revision=t["revision"])
    if not isinstance(p, str) or not Path(p).is_file():
        raise StructureError(f"{t['model']}@{t['revision'][:8]} config.json is not in the local cache")
    if sha256_file(Path(p)) != t["config_sha256"]:
        raise StructureError(f"the cached config.json is not the pinned {t['config_sha256'][:12]}")
    return json.loads(Path(p).read_text())


def instantiate(base: dict, over: dict):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.device("meta"):
        return Qwen3ForCausalLM(Qwen3Config(**{**base, **over}))


def modules(model) -> dict:
    """Unique parameters by module kind (the tied head counted once, under the embedding)."""
    out: dict[str, int] = {}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if "embed_tokens" in name or name.startswith("lm_head"):
            k = "embedding (tied head)"
        elif ".self_attn." in name:
            k = "attention." + name.split(".self_attn.")[1].split(".")[0]
        elif ".mlp." in name:
            k = "ffn." + name.split(".mlp.")[1].split(".")[0]
        elif "layernorm" in name:
            k = "block RMSNorms"
        elif name.startswith("model.norm"):
            k = "final RMSNorm"
        else:
            k = name
        out[k] = out.get(k, 0) + p.numel()
    return dict(sorted(out.items()))


def geometry(cfg: dict) -> dict:
    return {k: cfg[k] for k in GEOMETRY_KEYS}


def legality(g: dict, teacher: dict, build: list[str]) -> list[str]:
    """Why ``g`` is not a structure ``build``'s operators produce from ``teacher`` (empty when it is)."""
    bad = []
    for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads"):
        if not 0 < g[k] <= teacher[k]:
            bad.append(f"{k} {g[k]} outside 1..{teacher[k]}")
    if g["num_key_value_heads"] != teacher["num_key_value_heads"] or g["head_dim"] != teacher["head_dim"]:
        bad.append("KV heads and head_dim are preserved by every operator")
    if g["num_attention_heads"] % g["num_key_value_heads"]:
        bad.append(f"{g['num_attention_heads']} query heads is not a multiple of {g['num_key_value_heads']} KV heads")
    owner = {"num_hidden_layers": "DEPTH", "hidden_size": "RESIDUAL_WIDTH", "intermediate_size": "FFN",
             "num_attention_heads": "ATTENTION"}
    for k, op in owner.items():
        if g[k] != teacher[k] and op not in build:
            bad.append(f"{k} changed but {op} is not in the arm's build {build}")
        if g[k] == teacher[k] and op in build:
            bad.append(f"{op} is in the build but {k} is the teacher's")
    for k in ("hidden_size", "intermediate_size"):
        if g[k] != teacher[k] and g[k] % 128:
            bad.append(f"{k} {g[k]} is not a multiple of 128 (this round's alignment rule)")
    return bad


def trainable(mods: dict) -> int:
    """The recipe's trainable parameters: every attention, FFN and RMSNorm parameter; the tied embedding is frozen."""
    return sum(v for k, v in mods.items() if k != "embedding (tied head)")


def flops_per_token(g: dict, mods: dict, mean_context: float) -> dict:
    """Forward FLOPs per token: 2 per linear-map parameter (the tied head's matmul included), and causal attention
    scores and values at ``mean_context`` (4 x context x query width per layer)."""
    linear = sum(v for k, v in mods.items() if k.startswith(("attention.", "ffn.")) and not k.endswith("_norm"))
    head = g["vocab_size"] * g["hidden_size"]
    attn = g["num_hidden_layers"] * 4 * mean_context * g["num_attention_heads"] * g["head_dim"]
    return {"linear": 2 * linear, "head": 2 * head, "attention": attn}


def training_flops_per_token(student: dict, teacher: dict) -> dict:
    """One optimizer step's FLOPs per token under the recipe: the student with gradient checkpointing (forward,
    recomputed forward and a backward of twice the forward for its blocks; the frozen tied head forward and its
    input gradient only) and the online teacher's forward. Two accountings: with and without attention."""
    s = 4 * (student["linear"] + student["attention"]) + 2 * student["head"]
    t = teacher["linear"] + teacher["head"] + teacher["attention"]
    s0 = 4 * student["linear"] + 2 * student["head"]
    t0 = teacher["linear"] + teacher["head"]
    return {"with_attention": s + t, "without_attention": s0 + t0}


def memory_model(p: int, t: int, g: dict, teacher_p: int, ref: dict) -> dict:
    """Peak allocated GPU memory of one training step, GiB, decomposed from TPI1's measured run (``ref``): fp32
    parameters (4P), fp32 gradients and AdamW moments of the trainable ones (12T), the bf16 online teacher (2 P_teacher),
    the fp32 inputs checkpointed at every block (4 x L x d x sequence length) and the rest of the measured peak
    (logit chunks, the teacher's forward, one block's recompute). ``expected`` keeps that rest; ``conservative`` scales
    it by the largest per-block working set (FFN intermediate)."""
    seq = ref["sequence_length"]
    static = (4 * p + 12 * t + 2 * teacher_p) / GIB
    ckpt = 4 * g["num_hidden_layers"] * g["hidden_size"] * seq / GIB
    ref_static = (4 * ref["p"] + 12 * ref["t"] + 2 * teacher_p) / GIB
    ref_ckpt = 4 * ref["layers"] * ref["hidden"] * seq / GIB
    rest = ref["peak_allocated_gib"] - ref_static - ref_ckpt
    scale = max(1.0, g["intermediate_size"] / ref["intermediate"])
    exp, cons = static + ckpt + rest, static + ckpt + rest * scale
    overhead = ref["peak_reserved_gib"] - ref["peak_allocated_gib"]
    return {"static_gib": round(static, 2), "checkpointed_inputs_gib": round(ckpt, 2),
            "rest_from_tpi1_gib": round(rest, 2), "rest_scale_conservative": round(scale, 3),
            "expected_peak_allocated_gib": round(exp, 2), "conservative_peak_allocated_gib": round(cons, 2),
            "conservative_peak_reserved_gib": round(cons + overhead, 2),
            "l40s_gib": round(L40S_GIB, 2), "l40s_margin_conservative_reserved_gib": round(L40S_GIB - cons - overhead, 2),
            "rule": "derived from TPI1's measured peak, not measured at this size; the first optimizer steps of the "
                    "first unit measure it (the trainer's gpu_mem_gb / gpu_mem_reserved_gb)"}


def tpi1_reference() -> dict:
    """TPI1's measured 596M training: the seconds per step and peak memory of both units, and the structure."""
    units = json.loads(TPI1_UNITS.read_text())["units"]
    plan = json.loads(TPI1_PLAN.read_text())
    tel = [u["telemetry"] for u in units.values()]
    secs = [u["train_seconds"] for u in units.values() if u.get("state_at_start") == "fresh"]
    return {"source": str(TPI1_UNITS.relative_to(REPO)),
            "median_seconds_per_step": [t["median_seconds_per_step"] for t in tel],
            "all_in_seconds_per_step": round(secs[0] / plan["units"][0]["config"]["schedule"]["total_steps"], 4),
            "all_in_rule": "seed 20260801's train_seconds (trained from step 0 in one session: validation, save points "
                           "and logging included) over its 9,701 steps",
            "peak_allocated_gib": max(t["peak_gpu_mem_gb"] for t in tel),
            "peak_reserved_gib": max(t["peak_gpu_mem_reserved_gb"] for t in tel)}


def ffn_alternatives(base: dict, target: int) -> list[dict]:
    """Every legal attention step with the FFN size (a multiple of 128) closest to the target count."""
    out = []
    for q in range(base["num_key_value_heads"], base["num_attention_heads"], base["num_key_value_heads"]):
        best = None
        for i in range(128, base["intermediate_size"], 128):
            n = sum(x.numel() for x in instantiate(base, {"intermediate_size": i, "num_attention_heads": q}).parameters())
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (i, n)
        att = 2 * base["hidden_size"] * base["head_dim"] * (q + base["num_key_value_heads"])
        att_t = 2 * base["hidden_size"] * base["head_dim"] * (base["num_attention_heads"] + base["num_key_value_heads"])
        out.append({"num_attention_heads": q, "num_key_value_heads": base["num_key_value_heads"],
                    "intermediate_size": best[0], "num_parameters": best[1],
                    "deviation_from_target": round((best[1] - target) / target, 5),
                    "attention_params_retained": round(att / att_t, 4),
                    "ffn_params_retained": round(best[0] / base["intermediate_size"], 4)})
    return out


def build(design: dict | None = None) -> dict:
    d = design or json.loads(DESIGN.read_text())
    base = teacher_config(d)
    teacher_model = instantiate(base, {})
    tp = sum(x.numel() for x in teacher_model.parameters())
    if tp != d["teacher"]["num_parameters"]:
        raise StructureError(f"the teacher instantiates to {tp:,} parameters, the design pins {d['teacher']['num_parameters']:,}")
    target = round(tp / d["budget_point"]["n_target"])
    ref = tpi1_reference()
    tpi_g = json.loads((REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json").read_text())["target"]
    tpi_model = instantiate(base, {k: tpi_g["geometry"][k] for k in ("num_hidden_layers", "hidden_size",
                                                                       "intermediate_size", "num_attention_heads")})
    tpi_mods = modules(tpi_model)
    tpi_p = sum(tpi_mods.values())
    if tpi_p != tpi_g["num_parameters"]:
        raise StructureError(f"A's 596M geometry instantiates to {tpi_p:,}, TPI1 records {tpi_g['num_parameters']:,}")
    seq = json.loads(TPI1_PLAN.read_text())["units"][0]["config"]["block_len"]
    mean_ctx = (seq + 1) / 2                       #: continuous positions over the packed sequence (cross_context)
    ref.update(sequence_length=seq, p=tpi_p, t=trainable(tpi_mods), layers=tpi_g["geometry"]["num_hidden_layers"],
               hidden=tpi_g["geometry"]["hidden_size"], intermediate=tpi_g["geometry"]["intermediate_size"])
    t_mods = modules(teacher_model)
    teacher_flops = flops_per_token(geometry(base), t_mods, mean_ctx)
    tpi_flops = training_flops_per_token(flops_per_token(geometry({**base, **tpi_g["geometry"]}), tpi_mods, mean_ctx),
                                         teacher_flops)
    arms = {}
    for name, arm in d["arms"].items():
        m = instantiate(base, arm["geometry"])
        mods = modules(m)
        p = sum(mods.values())
        g = geometry({**base, **arm["geometry"]})
        bad = legality(g, geometry(base), arm["build"])
        if bad:
            raise StructureError(f"{name}: {bad}")
        tr = trainable(mods)
        fl = flops_per_token(g, mods, mean_ctx)
        tf = training_flops_per_token(fl, teacher_flops)
        arms[name] = {
            "label": arm["label"], "build": arm["build"], "geometry": g, "num_parameters": p,
            "n": round(tp / p, 5), "deviation_from_target": round((p - target) / target, 5),
            "modules": mods, "trainable_parameters": tr,
            "changed_from_teacher": {k: [t_mods.get(k, 0), v] for k, v in mods.items() if t_mods.get(k) != v},
            "checkpoint_bytes": {"init_bf16": 2 * p, "final_fp32": 4 * p, "save_point": 4 * p + 8 * tr,
                                 "rule": "derived: 2P / 4P / 4P + 8T (module docstring)"},
            "forward_flops_per_token": fl,
            "training_flops_per_token": tf,
            "training_flops_ratio_to_tpi1": {k: round(v / tpi_flops[k], 4) for k, v in tf.items()},
            "memory": memory_model(p, tr, g, tp, ref),
        }
    counts = {k: v["num_parameters"] for k, v in arms.items()}
    hi, lo = max(counts.values()), min(counts.values())
    return {
        "schema": SCHEMA, "design_sha256": sha256_file(DESIGN),
        "implementation": "transformers.Qwen3ForCausalLM on the meta device, from the teacher's pinned config.json",
        "teacher": {"num_parameters": tp, "geometry": geometry(base), "modules": t_mods,
                    "config_sha256": d["teacher"]["config_sha256"]},
        "target": {"n": d["budget_point"]["n_target"], "num_parameters": target},
        "arms": arms,
        "spread": {"max_minus_min": hi - lo, "relative_to_target": round((hi - lo) / target, 5),
                   "pairs": {f"{a} - {b}": counts[a] - counts[b] for a in counts for b in counts if a < b}},
        "ffn_attention_alternatives": ffn_alternatives(base, target),
        "depth_neighbours": {str(n): sum(x.numel() for x in instantiate(base, {"num_hidden_layers": n}).parameters())
                             for n in (15, 16, 17)},
        "a_596m_reference": {"num_parameters": tpi_p, "n": round(tp / tpi_p, 4), "trainable_parameters": ref["t"],
                             "training_flops_per_token": tpi_flops,
                             "rule": "TPI1's A: its geometry instantiates to its recorded count, the check of this method"},
        "tpi1_measured": {k: v for k, v in ref.items() if k not in ("p", "t", "layers", "hidden", "intermediate")},
        "mean_attention_context": mean_ctx,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = render(build())
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)}")
        return 0
    same = OUT.is_file() and OUT.read_text() == text
    print("structures.json reproduces" if same else "structures.json does NOT reproduce")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

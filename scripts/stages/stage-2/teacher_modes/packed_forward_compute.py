#!/usr/bin/env python3
"""What the packed forward context computes against the v2 search's one-item forwards, on the frozen inputs. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_forward_compute \\
        --write logs/stages/stage-2/teacher_modes/audit/packed_forward_compute.json

The v2 full search (a_search_20261101_153803) read every calibration and state-evaluation item ALONE: micro-batch 1,
each forward exactly its item's length, so no padding at all -- 8,192 was a data-construction limit, never a padded
width (`full_search.MICRO_BATCH`; DEPTH's recorded ``forward_rows`` equal 260 x the sum of (L - 1) exactly). The
packed search reads the same kind of sets as 8,192-token sequences (P19: seeded shuffle, sequential packing with
truncation, standard causal attention, continuous positions; `ForwardContext`). This counts, per set and from the
items themselves, both executions' forward calls, computed positions, padding, scored and context-only positions,
the fragments' repeated tokens and scored positions, and the FLOPs of the linear layers (FFN and projections), the
causal attention core and the scored-row output head, for the root and for the 596M target geometry.

Then it re-prices the packed search the way `packed_search_forecast` does (v2's measured seconds per component
scaled by these factors), with the frozen v2 inputs instead of v1's plus assumed tool items, the measured
state-evaluation seconds instead of an assumed share, and v2's setup and closeout not carried twice. Every number is
a FLOP-model ESTIMATE scaled from v2's measured seconds: no packed forward has run on a GPU, and the first packed
state evaluation and DEPTH round of the next authorized search session are the measurement that replaces it
(`full_search.FirstEvaluationCheck` writes ``evaluation_cache.jsonl``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.stage2.teacher_modes.packed_forward_compute/v1"
ART = REPO / "artifacts/stages/stage-2/teacher_modes"
SETS = {"v1": {"base": ART / "calibration_v2/calib.hybrid_d4_base.items.jsonl",
               "rh": ART / "calibration_v2/calib.hybrid_d4_rh.items.jsonl",
               "suite": ART / "state_eval_hybrid_v1/items.jsonl"},
        "v2": {"base": ART / "calibration_v3/calib.hybrid_d4_base.v2.items.jsonl",
               "rh": ART / "calibration_v3/calib.hybrid_d4_rh.v2.items.jsonl",
               "suite": ART / "state_eval_hybrid_v2_lc1/items.jsonl"}}
RUN = REPO / "logs/stages/stage-2/teacher_modes/runs/a_search_20261101_153803"
DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
VOCAB = 151936
#: (hidden, intermediate, query heads, kv heads, head dim, layers): the root and the C3 596M target, plus the level-1
#: single-operator geometries the v2 search's DEPTH parents had (its telemetry names the parent's operator)
GEOMETRY = {"root": (2560, 9728, 32, 8, 128, 36), "target_596M": (1024, 3072, 16, 8, 128, 28),
            "ffn": (2560, 3072, 32, 8, 128, 36), "width": (1024, 9728, 32, 8, 128, 36),
            "attention": (2560, 9728, 16, 8, 128, 36)}
#: DEPTH causal-KL greedy at the search's settings: 8 rounds from 36 surviving layers; per round and item one capture
#: forward and one suffix forward per candidate (the last surviving layer's candidate runs the whole bypassed model)
DEPTH_ROUNDS = [36 - r for r in range(8)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def per_layer_params(d, ff, h, kv, hd) -> int:
    return d * h * hd + 2 * d * kv * hd + h * hd * d + 3 * d * ff


def flops(geom: tuple, tokens: int, sum_sq: int, scored: int) -> dict:
    """One forward pass over a set: linear (2 x params per token per layer), causal attention core (QK^T and AV:
    ~2 h hd T^2 per layer and sequence), output head on scored rows only."""
    d, ff, h, kv, hd, layers = geom
    lin = 2 * layers * per_layer_params(d, ff, h, kv, hd) * tokens
    att = 2 * h * hd * layers * sum_sq
    head = 2 * d * VOCAB * scored
    return {"linear_PF": round(lin / 1e15, 3), "attention_PF": round(att / 1e15, 3), "head_PF": round(head / 1e15, 3),
            "attention_share_of_backbone": round(att / (lin + att), 4), "total_PF": round((lin + att + head) / 1e15, 3)}


def count(path: Path, ctx) -> dict:
    import torch
    items = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    L = [len(it["ids"]) for it in items]
    scored = []
    for it in items:
        tag = (it.get("tags") or {}).get("assistant")
        n = len(it["ids"])
        s = set(tag) if tag is not None else None
        scored.append([True] * (n - 1) if s is None else [(j + 1) in s for j in range(n - 1)])
    plan = ctx.plan([{"item_id": it["item_id"], "input_ids": torch.tensor(it["ids"])[None]} for it in items])
    seq = [sum(p.tokens for p in s) for s in plan]
    placed = rep_sc = rep_tok = frags = segs = 0
    for s in plan:
        for p in s:
            segs += 1
            n = sum(scored[p.index][:p.tokens - 1])
            placed += n
            if not p.final:
                frags, rep_tok, rep_sc = frags + 1, rep_tok + p.tokens, rep_sc + n
    uniq = sum(sum(s) for s in scored)
    alone = {"forwards": len(L), "computed_positions": sum(L), "padding_positions": 0,
             "scored_positions": uniq, "context_only_positions": sum(L) - uniq, "sum_sq_lengths": sum(x * x for x in L),
             "mean_length": round(sum(L) / len(L), 1), "max_length": max(L)}
    packed = {"forwards": len(seq), "computed_positions": sum(seq), "padding_positions": 0,
              "scored_positions_placed": placed, "context_only_positions": sum(seq) - placed,
              "repeated_prefix_tokens": rep_tok, "repeated_scored_positions": rep_sc, "fragments": frags,
              "placements": segs, "sum_sq_lengths": sum(x * x for x in seq), "mean_length": round(sum(seq) / len(seq), 1)}
    return {"path": str(path.relative_to(REPO)), "sha256": sha(path), "items": len(items), "alone": alone,
            "packed": packed}


def depth_flops(geom: tuple, tokens: int, sum_sq: int, scored: int) -> float:
    d, ff, h, kv, hd, layers = geom
    layer_passes = sum(n * (n - 1) // 2 + (n - 1) for n in DEPTH_ROUNDS) + sum(DEPTH_ROUNDS)
    per_layer = 2 * per_layer_params(d, ff, h, kv, hd) * tokens + 2 * h * hd * sum_sq
    return layer_passes * per_layer + sum(DEPTH_ROUNDS) * 2 * d * VOCAB * scored


def reprice(counts: dict) -> dict:
    """`packed_search_forecast`'s model, its inputs corrected: per component, v2's measured seconds x the backbone
    (or scored-row) factor of the frozen v2 sets packed over v1's sets alone."""
    tele = [json.loads(x) for x in (RUN / "evidence/telemetry.jsonl").read_text().splitlines() if x.strip()]
    money = json.loads((RUN / "closeout/outcome.json").read_text())["money"]
    rate, posted = money["provider_rate_usd_per_hour"], money["provider_posted_minutes"]
    d, ff, h, kv, hd, layers = GEOMETRY["root"]
    a_root = (2 * h * hd) / (2 * per_layer_params(d, ff, h, kv, hd))
    d, ff, h, kv, hd, layers = GEOMETRY["target_596M"]
    a_tgt = (2 * h * hd) / (2 * per_layer_params(d, ff, h, kv, hd))

    def factor(p, a, how="packed"):
        o, n = counts["v1"][p]["alone"], counts["v2"][p][how]
        return {"backbone": (n["computed_positions"] + a * n["sum_sq_lengths"]) /
                            (o["computed_positions"] + a * o["sum_sq_lengths"]),
                "scored": n.get("scored_positions_placed", n.get("scored_positions")) / o["scored_positions"]}
    f = {p: factor(p, a_root) for p in ("base", "rh", "suite")}
    f_alone = {p: factor(p, a_root, "alone") for p in ("base", "rh", "suite")}
    by = {}
    prof = lambda s: "base" if "base" in s else "rh"  # noqa: E731
    for e in tele:
        k = (e["impl_id"].split(".")[0], prof(e["profile"]))
        r = by.setdefault(k, {"op": 0.0, "load": 0.0, "abl": 0.0, "dist": 0.0, "ref": 0.0})
        t = e.get("operator_timing") or {}
        r["op"] += e["operator_seconds"]
        r["load"] += e["parent_load_seconds"]
        r["abl"] += t.get("ablated_seconds", 0)
        r["dist"] += t.get("distortion_seconds", 0)
        r["ref"] += t.get("reference_seconds", 0)
    def scaled_ops(f):
        return sum(((r["abl"] + r["ref"]) * f[p]["backbone"] + r["dist"] * f[p]["scored"] +
                    (r["op"] - r["abl"] - r["dist"] - r["ref"])) if k == "depth" else r["op"] * f[p]["backbone"]
                   for (k, p), r in by.items())
    ops, ops_alone = scaled_ops(f), scaled_ops(f_alone)
    depth_v2 = sum(r["op"] for (k, _), r in by.items() if k == "depth")
    loads = sum(r["load"] for r in by.values())
    ev = sum(e["state_evaluation_seconds"]["seconds"] for e in tele)
    inner = sum(e[k]["seconds"] for e in tele for k in ("canonical_reload_seconds", "identify_seconds",
                                                          "materialize_seconds", "validation_seconds",
                                                          "validation_transfer_seconds"))
    rest = posted * 60 - sum(r["op"] for r in by.values()) - loads

    def total(seconds, setup=40.0):
        m = seconds / 60 + setup
        return {"pod_minutes": round(m, 1), "usd": round(m / 60 * rate, 2)}
    sb, sb_t = f["suite"]["backbone"], factor("suite", a_tgt)["backbone"]
    out = {"rate_usd_per_hour": rate,
           "factors_root": {p: {k: round(v, 4) for k, v in x.items()} for p, x in f.items()},
           "factors_root_v2_inputs_alone": {p: {k: round(v, 4) for k, v in x.items()} for p, x in f_alone.items()},
           "suite_backbone_factor_target_geometry": round(sb_t, 4),
           "v2_measured_seconds": {"operators": round(sum(r["op"] for r in by.values()), 1), "depth": round(depth_v2, 1),
                                   "state_evaluation": round(ev, 1), "per_state_overhead": round(inner, 1),
                                   "parent_loads": round(loads, 1), "posted_pod_minutes": posted}}
    cases = {}
    for name, thr in (("lower", 1 / 1.3), ("expected", 1.0), ("upper", 1.15)):
        #: the forecast's own state-evaluation term (an assumed 0.7 share of v2's leftover pod time, which holds v2's
        #: own setup and its 105-minute closeout), with the actual v2 inputs
        own = (ops + rest * 0.7 * 0.5 * sb + rest * 0.3) * thr + loads
        #: the measured state-evaluation seconds, the teacher's share removed (the packed evaluator reads the teacher
        #: once; v2 re-read it at every evaluation), the in-driver per-state overhead carried once, setup counted once
        corr = (ops + ev * 0.5 * sb + inner) * thr + loads
        corr_t = (ops + ev * 0.5 * sb_t + inner) * thr + loads
        #: the counterfactual that separates the inputs from the packing: the frozen v2 inputs read one item at a
        #: time as v2 did (same teacher treatment, same throughput level)
        alone = (ops_alone + ev * 0.5 * f_alone["suite"]["backbone"] + inner) * thr + loads
        cases[name] = {"forecast_model_actual_inputs": total(own), "corrected": total(corr),
                       "corrected_target_geometry_eval": total(corr_t),
                       "counterfactual_v2_inputs_read_alone": total(alone)}
    out["cases"] = cases
    out["throughput_cases"] = "lower: the packed forwards use the GPU 30% better per FLOP; expected: v2's FLOP " \
                              "throughput; upper: 15% worse (packed_search_forecast's levels)"
    out["teacher_share_of_v2_evaluation"] = "0.5 (the forecast's halving; the review's regression puts the " \
                                            "candidate-independent part of each v2 evaluation near 95 of 97-130 s, so " \
                                            "this is conservative)"
    return out


def build() -> dict:
    from aadistill.initialization.calibration.packing import ForwardContext
    design = json.loads(DESIGN.read_text())
    fc = design["stage1_forward_context"]
    ctx = ForwardContext(sequence_length=int(fc["sequence_length"]), shuffle_seed=int(fc["shuffle_seed"]))
    counts = {v: {p: count(path, ctx) for p, path in sets.items()} for v, sets in SETS.items()}
    comp = {}
    for p in ("base", "rh", "suite"):
        o, n = counts["v1"][p]["alone"], counts["v2"][p]["packed"]
        on = counts["v2"][p]["alone"]
        comp[p] = {
            "v2_search_read (v1 items alone)": {g: flops(GEOMETRY[g], o["computed_positions"], o["sum_sq_lengths"],
                                                         o["scored_positions"]) for g in ("root", "target_596M")},
            "frozen_v2_items_alone (counterfactual)": {g: flops(GEOMETRY[g], on["computed_positions"],
                                                                on["sum_sq_lengths"], on["scored_positions"])
                                                       for g in ("root", "target_596M")},
            "frozen_v2_items_packed": {g: flops(GEOMETRY[g], n["computed_positions"], n["sum_sq_lengths"],
                                                n["scored_positions_placed"]) for g in ("root", "target_596M")},
            "same_items_packed_over_alone": {
                "forwards": round(n["forwards"] / on["forwards"], 4),
                "computed_positions": round(n["computed_positions"] / on["computed_positions"], 4),
                "attention_sum_sq": round(n["sum_sq_lengths"] / on["sum_sq_lengths"], 3),
                "scored_rows": round(n["scored_positions_placed"] / on["scored_positions"], 4)}}
    depth = {}
    for p in ("base", "rh"):
        on, n = counts["v2"][p]["alone"], counts["v2"][p]["packed"]
        depth[p] = {}
        for g in ("root", "ffn", "width", "attention"):
            a = depth_flops(GEOMETRY[g], on["computed_positions"], on["sum_sq_lengths"], on["scored_positions"])
            b = depth_flops(GEOMETRY[g], n["computed_positions"], n["sum_sq_lengths"], n["scored_positions_placed"])
            depth[p][g] = {"alone_PF": round(a / 1e15, 1), "packed_PF": round(b / 1e15, 1), "ratio": round(b / a, 3)}
        cand = sum(DEPTH_ROUNDS)
        depth[p]["forwards_per_expansion"] = {"alone": on["forwards"] * (cand + len(DEPTH_ROUNDS)),
                                              "packed": n["forwards"] * (cand + len(DEPTH_ROUNDS))}
    return {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forward_context": {"sequence_length": ctx.sequence_length, "shuffle_seed": ctx.shuffle_seed},
            "v2_search_execution": {"micro_batch_size": 1, "padding": "none: each forward is its item's own length",
                                    "longest_items": {p: counts["v1"][p]["alone"]["max_length"] for p in SETS["v1"]},
                                    "evidence": "full_search.py MICRO_BATCH = 1; DEPTH forward_rows = 260 x sum(L-1) "
                                                "(122,824,260 base, 88,159,500 rh) in the run's telemetry"},
            "sets": counts, "flops_per_pass": comp, "depth_per_expansion": depth, "reprice": reprice(counts),
            "evidence_levels": {
                "measured": "v2's per-component seconds, posted minutes and cost (a_search_20261101_153803); every "
                            "count above (items, positions, fragments) is exact from the frozen files",
                "estimated": "FLOPs (a dense model: 2 x params per token, 2 h hd T^2 attention per layer) and every "
                             "time or dollar figure scaled from v2's seconds by FLOP ratios",
                "unmeasured": "any packed forward on a GPU: CUDA time, achieved TFLOPS, peak memory, the teacher "
                              "sketch cache's behaviour; the next authorized search's first state evaluation and "
                              "first DEPTH round measure them"},
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", type=Path)
    ap.add_argument("--suite", type=Path, default=None, help="the state-evaluation items the packed search would read "
                    "(default @v2-lc1's; e.g. state_eval_hybrid_v3/items.jsonl)")
    args = ap.parse_args(argv)
    if args.suite:
        SETS["v2"]["suite"] = args.suite.resolve()
    rep = build()
    rep["suite_read"] = {"path": str(SETS["v2"]["suite"].relative_to(REPO)), "sha256": sha(SETS["v2"]["suite"])}
    text = json.dumps(rep, indent=1) + "\n"
    if args.write:
        args.write.write_text(text)
    print(json.dumps(rep["reprice"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

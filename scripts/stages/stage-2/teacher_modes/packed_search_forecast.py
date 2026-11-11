#!/usr/bin/env python3
"""What the packed hybrid full search should cost, priced from the v2 search's own telemetry. $0, an estimate.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_search_forecast \\
        --write logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json

The v2 search (a_search_20261101_153803: 93 states on one L40S, 1,299.69 pod-minutes, $23.9553 at $1.1067/h) is the
measurement; this scales each of its parts by what the packed forward context and the tool-calling items change, and
states every assumption beside the figure. It is NOT a measurement of the packed path: no packed forward has run on a
GPU. A pilot's (or the first state evaluation's) measured seconds replace it before a hard cap is final.

Scaling, per set (each calibration profile, the state-evaluation suite):

* backbone forward cost ~ sum_seq (T + a T^2): a linear term per token plus causal attention over the context each
  token sees. `a` is the root's attention-core FLOPs per context token over its per-token linear FLOPs
  (2 * n_heads * head_dim / (2 * per-layer parameters)), so reading items packed into 8,192-token sequences costs
  more per token than reading each alone, and fragments add their repeated tokens;
* output head and Top-K + tail KL ~ scored rows (fragments add their repeated scored rows);
* tool calling adds its items: per mode a profile cell at the supporting domain's per-sub-type share (4,200 / 1,800
  scored positions per mode in the base / reasoning-heavy profile) and 12,000 scored positions per mode in the suite,
  at the six-category design's tokens-per-supervised-token range per mode.

Three throughput cases: `lower` -- the packed forwards use the GPU 30% better per FLOP than v2's one-item forwards
(8,192-token sequences against items of median ~550 tokens); `expected` -- the same FLOP throughput; `upper` -- 15%
worse, plus the tool-token upper bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RUN = REPO / "logs/stages/stage-2/teacher_modes/runs/a_search_20261101_153803"
PROFILES_DOC = REPO / "configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json"
SUITE_ITEMS = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1/items.jsonl"
DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
TEACHER = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
SCHEMA = "aadistill.stage2.teacher_modes.packed_search_budget/v1"
#: tokens per supervised token of a tool turn, per mode (six_category_design.json :: packing_steps_budget: thinking
#: "non-padding per tool supervised token 2 / 4 / 6"; non-thinking "context + supervised per turn 400+120 / 900+70 /
#: 1,500+40")
TOOL_TOKENS_PER_SCORED = {"H-T": (2.0, 4.0, 6.0), "H-N": (520 / 120, 970 / 70, 1540 / 40)}
TOOL_SCORED = {"calib.hybrid_d4_base@v1": 4200, "calib.hybrid_d4_rh@v1": 1800, "state_eval_hybrid@v1": 12000}
CASES = {"lower": (1 / 1.3, 0), "expected": (1.0, 1), "upper": (1.15, 2)}
STATE_EVAL_SHARE_OF_REST = 0.7     #: of v2's time outside operators and parent loads (assumption, stated)
SETUP_MINUTES, TEARDOWN_MINUTES = 25.0, 15.0


def attention_coefficient() -> float:
    a = json.loads(TEACHER.read_text())["architecture"]
    d, ff, h, kv, hd = (int(a[k]) for k in ("hidden_size", "intermediate_size", "num_attention_heads",
                                             "num_key_value_heads", "head_dim"))
    per_layer_params = d * h * hd + 2 * d * kv * hd + h * hd * d + 3 * d * ff
    #: causal attention core: QK^T and AV, 2 FLOPs per multiply-add, over t/2 context on average
    return (2 * h * hd) / (2 * per_layer_params)


def set_terms(items: list[dict], fc) -> dict:
    import torch
    lengths = [len(it["ids"]) for it in items]
    prepared = [{"item_id": it["item_id"], "input_ids": torch.tensor(it["ids"])[None]} for it in items]
    rows = [sum(p.tokens for p in seq) for seq in fc.plan(prepared)]
    return {"sum_L": sum(lengths), "sum_L2": sum(n * n for n in lengths), "sum_T": sum(rows),
            "sum_T2": sum(r * r for r in rows)}


def scale(terms: dict, audit: dict, a: float, tool_tokens: float, tool_scored: float, cap: int) -> dict:
    old = terms["sum_L"] + a * terms["sum_L2"]
    #: tool items packed like the rest: every added token sits in a ~full sequence
    new = (terms["sum_T"] + tool_tokens) + a * (terms["sum_T2"] + tool_tokens * cap)
    scored_old = audit["unique_scored_positions"]
    scored_new = audit["placed_scored_positions"] + tool_scored * (1 + audit["repeated_scored_share"])
    return {"backbone": new / old, "scored_rows": scored_new / scored_old}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    args = ap.parse_args(argv)
    from aadistill.initialization.calibration.packing import ForwardContext
    design = json.loads(DESIGN.read_text())
    fcd = design["stage1_forward_context"]
    fc = ForwardContext(sequence_length=int(fcd["sequence_length"]), shuffle_seed=int(fcd["shuffle_seed"]))
    audit = json.loads((REPO / "logs/stages/stage-2/teacher_modes/audit/packing_audit.json").read_text())["stage1"]
    a = attention_coefficient()
    outcome = json.loads((RUN / "closeout/outcome.json").read_text())
    rate = float(outcome["money"]["provider_rate_usd_per_hour"])
    pod_minutes = float(outcome["money"]["provider_posted_minutes"])
    tele = [json.loads(x) for x in (RUN / "evidence/telemetry.jsonl").read_text().splitlines() if x.strip()]
    prof = json.loads(PROFILES_DOC.read_text())
    sets = {f"{p['profile_id']}@v{p['version']}": [json.loads(x) for x in (REPO / p["items_path"]).read_text().splitlines()
                                                    if x.strip()] for p in prof["profiles"]}
    sets["state_eval_hybrid@v1"] = [json.loads(x) for x in SUITE_ITEMS.read_text().splitlines() if x.strip()]
    terms = {k: set_terms(v, fc) for k, v in sets.items()}
    by = {}
    for e in tele:
        k = (e["impl_id"].split(".")[0], e["profile"])
        r = by.setdefault(k, {"operator": 0.0, "load": 0.0, "ablated": 0.0, "distortion": 0.0, "reference": 0.0})
        r["operator"] += e.get("operator_seconds", 0.0)
        r["load"] += e.get("parent_load_seconds", 0.0)
        t = e.get("operator_timing") or {}
        for src, dst in (("ablated_seconds", "ablated"), ("distortion_seconds", "distortion"),
                         ("reference_seconds", "reference")):
            r[dst] += float(t.get(src, 0.0))
    op_total = sum(r["operator"] for r in by.values())
    load_total = sum(r["load"] for r in by.values())
    rest = pod_minutes * 60 - op_total - load_total
    out_cases = {}
    for case, (throughput, tool_i) in CASES.items():
        f = {}
        for name in sets:
            tool_tokens = sum(TOOL_SCORED[name] * TOOL_TOKENS_PER_SCORED[m][tool_i] for m in ("H-T", "H-N"))
            f[name] = scale(terms[name], audit[name], a, tool_tokens, 2 * TOOL_SCORED[name], fc.sequence_length)
        seconds = 0.0
        for (kind, profile), r in by.items():
            s = f[profile]
            if kind == "depth":
                other = r["operator"] - r["ablated"] - r["distortion"] - r["reference"]
                seconds += (r["ablated"] + r["reference"]) * s["backbone"] + r["distortion"] * s["scored_rows"] + other
            else:
                seconds += r["operator"] * s["backbone"]
        suite = f["state_eval_hybrid@v1"]
        #: the packed evaluator forwards the teacher once per packed group and then reads its cached sketches;
        #: v2 re-forwarded it at every evaluation, so the evaluation's forward work halves before scaling
        seconds += rest * STATE_EVAL_SHARE_OF_REST * 0.5 * suite["backbone"] + rest * (1 - STATE_EVAL_SHARE_OF_REST)
        seconds = seconds * throughput + load_total
        minutes = seconds / 60 + SETUP_MINUTES + TEARDOWN_MINUTES
        out_cases[case] = {"pod_minutes": round(minutes, 1), "usd": round(minutes / 60 * rate, 2),
                           "scale_factors": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in f.items()},
                           "throughput_vs_v2": round(1 / throughput, 3)}
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "is_an_estimate": "no packed forward has run on a GPU; replace with a measured first evaluation / DEPTH round",
           "measured_basis": {"run": str(RUN.relative_to(REPO)), "pod_minutes": pod_minutes, "rate_usd_per_hour": rate,
                              "usd": outcome["money"]["all_in_usd"], "operator_seconds": round(op_total, 1),
                              "parent_load_seconds": round(load_total, 1),
                              "rest_seconds_state_evaluation_materialization_setup": round(rest, 1)},
           "attention_coefficient_per_context_token": a,
           "set_terms": terms,
           "assumptions": {"tool_scored_positions_per_mode": TOOL_SCORED,
                           "tool_tokens_per_scored_lower_expected_upper": TOOL_TOKENS_PER_SCORED,
                           "state_evaluation_share_of_rest": STATE_EVAL_SHARE_OF_REST,
                           "setup_minutes": SETUP_MINUTES, "teardown_minutes": TEARDOWN_MINUTES,
                           "gpu": "one L40S at the v2 search's posted rate (GPU + container disk); an A100-80GB only "
                                  "after a measured L40S OOM (largest packed sequence: ~7.7k scored rows, DEPTH's "
                                  "float32 KL now reduced in 2,048-row chunks)"},
           "cases": out_cases,
           "proposed_hard_cap_usd": round(out_cases["upper"]["usd"] * 1.1, 2),
           "_hard_cap_rule": "the upper case plus 10% for the measured-cost uncertainty; storage is booked separately",
           "authorizes": "nothing"}
    text = json.dumps(doc, indent=1) + "\n"
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(text)
    print(json.dumps({k: {x: v[x] for x in ("pod_minutes", "usd")} for k, v in out_cases.items()} |
                     {"cap": doc["proposed_hard_cap_usd"], "a": a}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

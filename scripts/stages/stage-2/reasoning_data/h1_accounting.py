#!/usr/bin/env python3
"""H1's matched-training accounting and arm-B quotas. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1_accounting [--pack-b <dir>]

Writes `logs/stages/stage-2/reasoning_data/plans/h1_accounting.json`. Every
figure carries its KIND, because these are different evidence:

* ``measured`` -- counted from a real pack: arm A's always (built here from the
  isolation-passing corpus_v2 units by the shared builder), arm B's once its
  pack exists (``--pack-b``);
* ``estimated`` -- arm B's pack before it exists: blocks and per-block counts
  from the fluid dry run's MEASURED packing efficiency and reasoning share and
  the spec's PLANNING rendered-per-supervised ratio;
* ``planned`` -- what a schedule will consume, with E from the pack's mean
  supervision per block (`training_accounting`);
* ``exact_per_seed`` -- what the trainer's seeded block stream consumes
  (`consumed_exposure` over `aadistill.training.train.stream_block_indices`),
  for every arm whose pack is real;
* ``modeled`` -- FLOPs from the architecture (`aadistill.runtime.cost`), never a
  measurement. Wall-clock rates measured on earlier runs live in
  `plans/cost_forecast.json`, not here.

The matching rule is fixed now (`aadistill.training.accounting.match_schedule`,
reference blocks per step 2, 3 epochs). The step count and blocks per step it
returns while B is estimated are PROVISIONAL; the protocol freezes the ones it
returns on B's real pack, with the exact per-seed U/E/steps of both arms, and
the cost forecast is recomputed from them. KD positions are reported by kind
(prompt / reasoning / answer) under the frozen ``kd_scope = all``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import pack_dry_run as P

from aadistill.training.accounting import consumed_exposure, match_schedule, training_accounting

OUT = A.LOG_DIR / "plans/h1_accounting.json"
INIT_CONFIG = A.REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/1301deb71685aa1ca5c7b6cf2c491ac4/config.json"
TEACHER_CONFIG = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/*/config.json"))
EPOCHS, BPS_REFERENCE, BLOCK = 3, 2, 8192
KD_SCOPE = "all"
TARGET = 2_960_000
#: Mean supervised tokens per INDEPENDENT unit, by where it is measured.
PER_UNIT_SOURCES = {
    "primary": "base_pool_v1 verified units per axis (asset_audit.json :: teacher_cal.yield_by_axis / base_pool_v1_fluid.units_by_axis)",
    "supporting.math_reused_gsm8k": "corpus_v2 gsm8k selected sessions",
    "supporting.math_new": "corpus_v2 openmath selected sessions (MATH-style proxy)",
    "supporting.code_algorithmic": "corpus_v2 code selected sessions",
    "supporting.reading_verified": "corpus_v2 rag_evidence and multihop_qa selected sessions, token-weighted",
    "supporting.answerability": "corpus_v2 rag_evidence selected sessions",
}


def spec_of(path) -> dict:
    c = json.loads(open(path).read())
    return {k: c[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                              "head_dim", "intermediate_size", "vocab_size", "num_hidden_layers")}


def per_unit(audit: dict) -> dict[str, float]:
    sel = audit["corpus_v2"]["selected_sessions"]
    mean = lambda t: sel[t]["supervised_tokens"]["mean"]  # noqa: E731
    pool = audit["base_pool_v1_fluid"]
    tc = audit["teacher_cal"]["yield_by_axis"]
    out = {f"primary.{ax}": tc[ax]["verified_supervised_positions_le_8192"] / pool["units_by_axis"][ax]
           for ax in pool["units_by_axis"]}
    out["supporting.math_reused_gsm8k"] = mean("gsm8k")
    out["supporting.math_new"] = mean("openmath")
    out["supporting.code_algorithmic"] = mean("code")
    rs, ms = sel["rag_evidence"]["supervised_tokens"]["sum"], sel["multihop_qa"]["supervised_tokens"]["sum"]
    out["supporting.reading_verified"] = (rs + ms) / (sel["rag_evidence"]["sessions"] + sel["multihop_qa"]["sessions"])
    out["supporting.answerability"] = mean("rag_evidence")
    return out


def exact_per_seed(rows: list[dict], seeds: list[int], steps: int, bps: int) -> dict:
    from aadistill.training.train import stream_block_indices
    return {str(seed): consumed_exposure(rows, stream_block_indices(len(rows), seed, 0, steps * bps),
                                         kd_scope=KD_SCOPE) for seed in seeds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-b", default=None, help="arm B's real pack directory (ladder.json, audit.jsonl)")
    args = ap.parse_args(argv)
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    audit = A.read_json(A.LOG_DIR / "audit/asset_audit.json")
    dry = A.read_json(A.LOG_DIR / "audit/packing_dry_run.json")
    design = A.read_json(A.LOG_DIR / "plans/comparison_design.json")
    seeds = list(design["fixed"]["seeds"])
    student, teacher = spec_of(INIT_CONFIG), spec_of(TEACHER_CONFIG[0])

    # Arm A, built for real.
    tok = B.load_tokenizer()
    built = B.build(P.old_arm_sessions(), P.OLD_ARM_SPEC, tok)
    n_a = built["ladder"]["rungs"][0]["n_blocks"]
    rows_a = B.block_rows(built["blocks"][:n_a], built["session_counts"])
    units_a = built["ladder"]["rungs"][0]["n_units"]

    # Arm B: its real pack, or an estimate from the measured fluid packing.
    if args.pack_b:
        rows_b, b_kind = B.pack_block_rows(Path(args.pack_b), TARGET), "measured"
        n_b = len(rows_b)
        b_basis = {"pack": str(args.pack_b), "ladder_sha256": A.sha256_file(Path(args.pack_b) / "ladder.json")}
    else:
        fl = dry["fluid_traces"]["rungs"][-1]
        eff = fl["packing_efficiency"]
        reason_share = fl["reasoning_tokens"] / fl["unique_supervised_tokens"]
        n_b = math.ceil(TARGET * spec["new_arm_rendered_per_supervised"] / (BLOCK * eff))
        sup_b = TARGET / n_b
        sup, reason = round(sup_b), round(sup_b * reason_share)
        rows_b = [{"real_length": round(sup_b * spec["new_arm_rendered_per_supervised"]),
                   "supervised": sup, "reasoning": reason, "answer": sup - reason} for _ in range(n_b)]
        b_kind = "estimated"
        b_basis = {"packing_efficiency": {"value": eff, "kind": "measured", "from": "packing_dry_run.json :: fluid_traces (the 519 Stage-0 units)"},
                   "reasoning_share": {"value": round(reason_share, 4), "kind": "measured", "from": "the same dry run"},
                   "rendered_per_supervised": {"value": spec["new_arm_rendered_per_supervised"], "kind": "planning figure",
                                               "from": "corpus_spec.json :: new_arm_rendered_per_supervised"},
                   "uniform_blocks": "every estimated block carries the mean; the exact per-seed exposure of B exists only with its real pack"}
    kinds = {"A": "measured", "B": b_kind}
    rows = {"A": rows_a, "B": rows_b}

    match = match_schedule({"A": n_a, "B": n_b}, epochs=EPOCHS, reference_blocks_per_step=BPS_REFERENCE)
    t = match["steps"]
    common = dict(epochs=EPOCHS, block_len=BLOCK, student_spec=student, teacher_spec=teacher, kd_scope=KD_SCOPE)
    contracts = {
        "M0_frozen": {arm: training_accounting(rows[arm], blocks_per_step=BPS_REFERENCE,
                                               truncate_padding=False, **common) for arm in rows},
        "M1_matched": {arm: training_accounting(rows[arm], blocks_per_step=match["blocks_per_step"][arm],
                                                steps=t, truncate_padding=True, **common) for arm in rows},
    }
    for c in contracts.values():
        a, b = c["A"], c["B"]
        c["ratios_A_over_B"] = {k: round(a[k] / b[k], 4) for k in
                                ("U_supervised", "E_supervised", "optimizer_steps", "supervised_per_step",
                                 "executed_positions", "C_total_flops")}
        c["kinds"] = {"U": kinds, "E": "planned (mean supervision per block)", "steps": "planned",
                      "C": "modeled FLOPs", "executed_positions": "planned"}
    exact = {arm: (exact_per_seed(rows[arm], seeds, t, match["blocks_per_step"][arm])
                   if kinds[arm] == "measured" else None) for arm in rows}
    schedule = {"rule": match["rule"], "match": match,
                "status": ("FROZEN-READY: computed on both arms' real packs" if b_kind == "measured" else
                           "PROVISIONAL: B's pack is estimated; the protocol freezes the values this rule returns on B's real pack"),
                "steps": t, "blocks_per_step": match["blocks_per_step"],
                "warmup_steps": max(10, round(0.05 * t)), "eval_every": max(25, t // 8),
                "save_every": max(200, t // 2),
                "lr_rule": "the E1 builder's rule (build_experiment1_configs.py): warmup max(10, round(0.05 T)); eval every max(25, T // 8); save every max(200, T // 2); cosine to min_lr_frac 0.1",
                "not_reused": "the 0.86M recipe's 1,023 steps; any earlier provisional T or blocks per step"}

    pu = per_unit(audit)
    top = TARGET + spec["reserve_tokens"]
    quotas = {}
    for stratum, share in spec["shares"].items():
        need = TARGET * share
        quotas[stratum] = {"share": share, "U_target_in_2p96M": round(need),
                           "supervised_per_unit_estimate": round(pu[stratum]),
                           "independent_units_estimate": round(need / pu[stratum]),
                           "with_reserve_U": round(top * share),
                           "per_unit_source": PER_UNIT_SOURCES["primary" if stratum.startswith("primary.") else stratum]}
    doc = {
        "schema": "aadistill.stage2.reasoning_data.h1_accounting/v2",
        "status": f"A MEASURED from its real pack; B {b_kind.upper()}; schedule {schedule['status'].split(':')[0]}. Authorizes nothing.",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kinds": {"measured": "counted from a real pack", "estimated": "B before its pack exists",
                  "planned": "a schedule's consumption from mean supervision per block",
                  "exact_per_seed": "the trainer's seeded block stream, summed block by block",
                  "modeled": "FLOPs from the architecture, not a measurement",
                  "runtime": "measured rates from earlier runs: plans/cost_forecast.json"},
        "definitions": {
            "U": "unique supervised positions: each session's supervised target once (one accepted trace per task; turn-expanded context is not supervision)",
            "E": "supervised exposure: supervised positions processed with every repetition (epochs, the wrap of the last step)",
            "C": "FLOPs over every executed position: student forward + backward (3x forward) and teacher forward, with causal attention over the packed block and both output heads (aadistill.runtime.cost.forward_flops_per_token)",
            "KD_positions": "kd_scope all: every real prediction position; prompt = real positions that are not supervised (system, user, tool, passages, earlier turns)",
            "independent_units": "tasks / episodes / source conversations; several traces of one prompt are ONE unit and only one is ever kept",
        },
        "models": {"student": student, "teacher": teacher,
                   "student_config": str(INIT_CONFIG.relative_to(A.REPO)), "teacher_config": "Qwen/Qwen3-4B-Thinking-2507 config.json (local HF cache)"},
        "arm_A": {"kind": "measured", "units": units_a, "sessions": built["ladder"]["rungs"][0]["packed"]["sessions"],
                  "blocks": n_a, "packing_efficiency": built["ladder"]["rungs"][0]["packed"]["packing_efficiency"],
                  "system_prompt_groups": built["system_prompt_groups"],
                  "isolation_exclusion": P.old_arm_isolation_exclusion(),
                  "admission_exclusion": B.exclusion_summary(built["excluded"])},
        "arm_B": {"kind": b_kind, "blocks": n_b, "basis": b_basis},
        "contracts": contracts,
        "exact_per_seed": {"kind": "exact_per_seed", "seeds": seeds, "steps": t, "by_arm": exact,
                           "_B": None if b_kind == "measured" else "B has no real pack yet: no exact per-seed figure exists"},
        "primary_basis": ("M1: one step count T for both arms (identical learning-rate schedule) from match_schedule; "
                          "U equal within 1% by the ladder cut; E within the passes match_schedule reports, verified "
                          "exactly per seed on the real packs; supervised positions per step follow; C is NOT matched and "
                          "is reported -- arm A carries prompt context (passages, tool schemas) that is computed but not supervised"),
        "schedule": schedule,
        "quotas_arm_B": quotas,
        "authorizes": "nothing",
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), match, {k: v["ratios_A_over_B"] for k, v in contracts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

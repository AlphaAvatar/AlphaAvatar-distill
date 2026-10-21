#!/usr/bin/env python3
"""H1's matched-training accounting and arm-B quotas, before any target exists. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1_accounting

Writes `logs/stages/stage-2/reasoning_data/plans/h1_accounting.json`.

* **Arm A** is built for real: the corpus_v2 sessions that pass isolation, cut
  at 2.96M unique supervised tokens and packed by the shared builder
  (`pack_dry_run.OLD_ARM_SPEC`); its blocks are counted exactly.
* **Arm B** has no sessions yet. Its blocks are ESTIMATED from the measured
  fluid-trace dry run (packing efficiency, reasoning share) and the spec's
  rendered-per-supervised ratio; every B figure is labelled an estimate and is
  replaced by the real pack's own accounting before preregistration.

Both arms are then accounted by `aadistill.training.accounting` under the
frozen schedule (M0) and under the proposed matched contract (M1: equal steps,
`truncate_padding` on), with the student and teacher dimensions read from their
configs. The quota table turns `corpus_spec.json` shares into supervised-token
targets and estimated independent units per stratum.
"""
from __future__ import annotations

import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import pack_dry_run as P

from aadistill.training.accounting import matched_steps, training_accounting

OUT = A.LOG_DIR / "plans/h1_accounting.json"
INIT_CONFIG = A.REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/1301deb71685aa1ca5c7b6cf2c491ac4/config.json"
TEACHER_CONFIG = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/*/config.json"))
EPOCHS, BPS_FROZEN, BLOCK = 3, 2, 8192
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


def main() -> int:
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    audit = A.read_json(A.LOG_DIR / "audit/asset_audit.json")
    dry = A.read_json(A.LOG_DIR / "audit/packing_dry_run.json")
    student, teacher = spec_of(INIT_CONFIG), spec_of(TEACHER_CONFIG[0])

    # Arm A, built for real.
    tok = B.load_tokenizer()
    built = B.build(P.old_arm_sessions(), P.OLD_ARM_SPEC, tok)
    n_a = built["ladder"]["rungs"][0]["n_blocks"]
    rows_a = B.block_rows(built["blocks"][:n_a], built["session_counts"])
    units_a = built["ladder"]["rungs"][0]["n_units"]

    # Arm B, estimated from the measured fluid packing.
    fl = dry["fluid_traces"]["rungs"][-1]
    eff = fl["packing_efficiency"]
    reason_share = fl["reasoning_tokens"] / fl["unique_supervised_tokens"]
    n_b = math.ceil(TARGET * spec["new_arm_rendered_per_supervised"] / (BLOCK * eff))
    sup_b = TARGET / n_b
    rows_b = [{"real_length": round(BLOCK * eff), "supervised": round(sup_b),
               "reasoning": round(sup_b * reason_share), "answer": round(sup_b * (1 - reason_share))}
              for _ in range(n_b)]

    m = matched_steps(n_b, n_a, epochs=EPOCHS, reference_blocks_per_step=BPS_FROZEN)
    common = dict(epochs=EPOCHS, block_len=BLOCK, student_spec=student, teacher_spec=teacher)
    contracts = {
        "M0_frozen": {"A": training_accounting(rows_a, blocks_per_step=BPS_FROZEN, truncate_padding=False, **common),
                      "B_estimate": training_accounting(rows_b, blocks_per_step=BPS_FROZEN, truncate_padding=False, **common)},
        "M1_matched": {"A": training_accounting(rows_a, blocks_per_step=m["other_blocks_per_step"], truncate_padding=True, **common),
                       "B_estimate": training_accounting(rows_b, blocks_per_step=BPS_FROZEN, truncate_padding=True, **common)},
    }
    for c in contracts.values():
        a, b = c["A"], c["B_estimate"]
        c["ratios_A_over_B"] = {k: round(a[k] / b[k], 4) for k in
                                ("U_supervised", "E_supervised", "optimizer_steps", "supervised_per_step",
                                 "executed_positions", "C_total_flops")}
    t = contracts["M1_matched"]["B_estimate"]["optimizer_steps"]
    schedule = {"rule": "the E1 builder's rule (build_experiment1_configs.py): warmup max(10, round(0.05 T)); eval every max(25, T // 8); save every max(200, T // 2); cosine to min_lr_frac 0.1",
                "T_estimate": t, "warmup_steps": max(10, round(0.05 * t)), "eval_every": max(25, t // 8),
                "save_every": max(200, t // 2), "A_blocks_per_step": m["other_blocks_per_step"],
                "B_blocks_per_step": BPS_FROZEN, "A_steps": m["other_steps"], "step_ratio_A_over_B": m["step_ratio"],
                "registered_when": "recomputed from B's real pack before preregistration; the 0.86M recipe's 1,023 steps are NOT reused"}

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
        "schema": "aadistill.stage2.reasoning_data.h1_accounting/v1",
        "status": "A MEASURED from its real pack; B ESTIMATED until its pack exists. Authorizes nothing.",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definitions": {
            "U": "unique supervised positions: each session's supervised target once (one accepted trace per task; turn-expanded context is not supervision)",
            "E": "supervised exposure: U x the block passes / blocks the schedule makes (3 epochs, the wrap of the last step included)",
            "C": "FLOPs over every executed position: student forward + backward (3x forward) and teacher forward, with causal attention over the packed block and both output heads (aadistill.runtime.cost.forward_flops_per_token)",
            "independent_units": "tasks / episodes / source conversations; several traces of one prompt are ONE unit and only one is ever kept",
        },
        "models": {"student": student, "teacher": teacher,
                   "student_config": str(INIT_CONFIG.relative_to(A.REPO)), "teacher_config": "Qwen/Qwen3-4B-Thinking-2507 config.json (local HF cache)"},
        "arm_A": {"units": units_a, "sessions": built["ladder"]["rungs"][0]["packed"]["sessions"],
                  "blocks": n_a, "packing_efficiency": built["ladder"]["rungs"][0]["packed"]["packing_efficiency"],
                  "system_prompt_groups": built["system_prompt_groups"]},
        "arm_B_estimate": {"blocks": n_b, "packing_efficiency_from": "packing_dry_run.json fluid_traces (measured)",
                           "packing_efficiency": eff, "reasoning_share": round(reason_share, 4),
                           "rendered_per_supervised": spec["new_arm_rendered_per_supervised"]},
        "contracts": contracts,
        "primary_basis": ("M1: U equal within 1% and E equal (3 epochs), optimizer steps equal within 10% with "
                          "supervised positions per step matched by per-arm blocks per step; C is NOT matched and is "
                          "reported -- arm A carries prompt context (passages, tool schemas) that is computed but not supervised"),
        "schedule": schedule,
        "quotas_arm_B": quotas,
        "authorizes": "nothing",
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), {k: v["ratios_A_over_B"] for k, v in contracts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GPU, CPU and storage forecast for the proposed Stage-2/3 programme, from MEASURED rates.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.forecast_cost

Writes `logs/stages/stage-2/reasoning_data/plans/cost_forecast.json`. Every rate
is read from a closed record and named beside the figure it produces; the only
typed numbers are the plan's own sizes (`configs/.../corpus_spec.json`) and the
contingency factors, which are labelled as such. It prices; it authorizes nothing.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone

from stages.reasoning_data import assets as A

OUT = A.LOG_DIR / "plans/cost_forecast.json"
SPEC = A.CONFIG_DIR / "corpus_spec.json"
DRY = A.LOG_DIR / "audit/packing_dry_run.json"
AUDIT = A.LOG_DIR / "audit/asset_audit.json"
PROBES = ("probes_d1_20261020_093955", "probes_d2_20261020_233510", "probes_d4_20261020_232941")
#: Labelled planning factors, not measurements.
CONTINGENCY = {"generation": 1.5, "recovery": 1.25, "eval_floor_to_reasoning": 2.0, "hard_cap": 1.5,
               "gpu_validation_hours": 1.25}


def measured() -> dict:
    tc_sum = A.read_json(A.S0 / "runs/teachercal_20261018_112755/evidence/teacher_cal/teacher_cal.summary.json")
    tc_out = A.read_json(A.S0 / "runs/teachercal_20261018_112755/closeout/outcome.json")
    per = A.read_json(A.S0 / "design/extension_analysis.json")["per_unit"]
    rec = {"A100": [], "L40S": []}
    for key, u in per.items():
        rec[key.split("/")[-1]].append(u["recovery"]["seconds"] / (u["recovery"]["steps"] * 2))
    evals, overheads, prices = [], [], set()
    for run in PROBES:
        sess = A.read_json(A.S0 / f"runs/{run}/evidence/probe_session.json")
        out = A.read_json(A.S0 / f"runs/{run}/closeout/outcome.json")
        prices.add(out["money"]["price_per_hour"])
        total = 0.0
        for name, u in sess["units"].items():
            m = re.match(r"s0v_top4_(d\d)_r(\d)_[0-9a-f]+_s(\d+)", name)
            key = f"{m.group(1)}/r{m.group(2)}/s{m.group(3)}/L40S"
            total += u["seconds"]
            evals.append(u["seconds"] - per[key]["recovery"]["seconds"])
        overheads.append(out["money"]["elapsed_minutes"] * 60 - total)
    gen_seconds = tc_sum.get("generation_seconds") or tc_sum.get("seconds")
    gen_tokens = tc_sum.get("output_tokens") or tc_sum.get("generated_tokens")
    return {
        "teacher_generation": {
            "tokens_per_second": round(gen_tokens / gen_seconds, 1), "output_tokens": gen_tokens,
            "seconds": gen_seconds, "gpu": "A100-80GB", "engine": "vLLM 0.27.1",
            "session_all_in_usd": tc_out["money"]["all_in_usd"],
            "session_minutes": tc_out["money"]["elapsed_minutes"],
            "price_per_hour": tc_out["money"]["price_per_hour"],
            "source": "runs/teachercal_20261018_112755 (summary + closeout)"},
        "recovery_seconds_per_block_pass": {
            g: {"min": round(min(v), 4), "max": round(max(v), 4), "n": len(v)} for g, v in rec.items()},
        "recovery_basis": "extension_analysis.json :: per_unit.*.recovery (seconds / (steps x 2 blocks)); 8,192 positions per pass, teacher online, padding forwarded",
        "positions_per_second_full_blocks": {
            g: round(8192 / statistics.fmean(v), 1) for g, v in rec.items()},
        "eval_seconds_per_unit_L40S": {"median": round(statistics.median(evals)), "max": round(max(evals)),
                                       "n": len(evals), "note": "students at the floor: short, often degenerate replies"},
        "session_overhead_seconds_L40S": {"median": round(statistics.median(overheads)), "n": len(overheads)},
        "price_per_hour": {"L40S": sorted(prices), "A100-80GB": tc_out["money"]["price_per_hour"]},
        "disk_share_of_gpu_cost": round(sum(
            A.read_json(A.S0 / f"runs/{r}/closeout/outcome.json")["money"]["disk_usd"] for r in PROBES) / sum(
            A.read_json(A.S0 / f"runs/{r}/closeout/outcome.json")["money"]["gpu_usd"] for r in PROBES), 4),
    }


def main() -> int:
    spec = A.read_json(SPEC)
    dry = A.read_json(DRY)
    audit = A.read_json(AUDIT)
    m = measured()
    #: Rates by priced GPU name; the slowest measured unit, not the mean.
    gpu_of = {"L40S": "L40S", "A100-80GB": "A100"}
    sec_pass = {p: m["recovery_seconds_per_block_pass"][g]["max"] for p, g in gpu_of.items()}
    pos_s = {p: m["positions_per_second_full_blocks"][g] for p, g in gpu_of.items()}
    eval_s = m["eval_seconds_per_unit_L40S"]["median"] * CONTINGENCY["eval_floor_to_reasoning"]
    over = m["session_overhead_seconds_L40S"]["median"]
    price = {"L40S": max(m["price_per_hour"]["L40S"]), "A100-80GB": m["price_per_hour"]["A100-80GB"]}
    disk = 1 + m["disk_share_of_gpu_cost"]

    # --- generation -------------------------------------------------------
    tc = audit["teacher_cal"]
    cv2 = audit["corpus_v2"]["under_target_quality_contract"]["generated_tokens"]

    def yield_of(ref: str) -> float:
        src, key = ref.split(":")
        if src == "assumption":
            return float(key)
        if src == "teacher_cal":
            return tc["yield_by_axis"][key]["yield"]
        # A single-turn session's generation IS its supervised target, so the share
        # of generated tokens in accepted candidates is the yield.
        return cv2[key]["accepted_share"]

    top = max(spec["rung_targets"]) + spec["reserve_tokens"]
    gen_rows, gen_tokens = {}, 0
    for stratum, g in spec["generation"].items():
        if g.get("reuse"):
            gen_rows[stratum] = {"reused": True, "generated_tokens": 0}
            continue
        refs = g["yield_from"] if isinstance(g["yield_from"], list) else [g["yield_from"]]
        ys = [yield_of(r) for r in refs]
        y = min(ys)  # the pessimistic bracket
        need = top * spec["shares"][stratum]
        toks = need / y
        gen_rows[stratum] = {"supervised_tokens_needed": round(need), "yield": round(y, 4),
                             "yield_refs": dict(zip(refs, [round(v, 4) for v in ys])),
                             "generated_tokens": round(toks)}
        gen_tokens += toks
    spec_tokens = gen_tokens
    plan_path = A.LOG_DIR / "plans/generation_plan.json"
    gp = A.read_json(plan_path) if plan_path.exists() else None
    if gp is not None:      # the real list's upper bound replaces the spec-level estimate
        gen_tokens = gp["totals"]["expected_generated_tokens_upper"]
    gen_hours = gen_tokens / m["teacher_generation"]["tokens_per_second"] / 3600
    gen_session_h = gen_hours * CONTINGENCY["generation"] + over / 3600
    generation = {"rows": gen_rows, "candidate_policy": spec["candidate_policy"],
                  "spec_estimate_generated_tokens": round(spec_tokens),
                  "basis": ("plans/generation_plan.json :: totals.expected_generated_tokens_upper (the real task "
                            "list, second candidates included)" if gp else "spec shares / per-stratum yield"),
                  "generation_plan_sha256": A.sha256_file(plan_path) if gp else None,
                  "generated_tokens": round(gen_tokens), "gpu_hours": round(gen_hours, 2),
                  "session_hours_with_contingency": round(gen_session_h, 2),
                  "expected_usd": round(gen_session_h * price["A100-80GB"] * disk, 2)}

    # --- recovery + eval units --------------------------------------------
    old = dry["old_arm"]["rungs"][0]
    fl = dry["fluid_traces"]["rungs"][-1]
    new_ratio = spec["new_arm_rendered_per_supervised"]
    eff = fl["packing_efficiency"]
    epochs = 3

    def new_blocks(sup):
        return math.ceil(sup * new_ratio / (8192 * eff))

    def unit(sup_tokens, blocks, *, gpu, contract, real_per_epoch=None):
        if contract == "M0_frozen":
            passes = math.ceil(blocks * epochs / 2) * 2
            rec_s = passes * sec_pass[gpu]
        else:  # M1: padding not forwarded; time follows executed positions
            executed = (real_per_epoch if real_per_epoch is not None else blocks * 8192 * eff) * epochs
            rec_s = executed / pos_s[gpu]
        rec_s *= CONTINGENCY["recovery"]
        sec = rec_s + eval_s
        return {"recovery_hours": round(rec_s / 3600, 3), "eval_hours": round(eval_s / 3600, 3),
                "usd": round(sec / 3600 * price[gpu] * disk, 2)}

    units = {}
    for gpu in ("L40S", "A100-80GB"):
        for contract in ("M0_frozen", "M1_matched"):
            row = {
                "old_2p96M": unit(old["unique_supervised_tokens"], old["blocks"], gpu=gpu, contract=contract,
                                  real_per_epoch=old["blocks"] * 8192 * old["packing_efficiency"]),
                "new_2p96M": unit(2_960_000, new_blocks(2_960_000), gpu=gpu, contract=contract),
            }
            for r in spec["rung_targets"]:
                row[f"new_{r / 1e6:.2f}M"] = unit(r, new_blocks(r), gpu=gpu, contract=contract)
            units[f"{gpu}/{contract}"] = row

    def phase_cost(keys, seeds, table):
        return round(sum(table[k]["usd"] for k in keys) * seeds, 2)

    phases = {}
    for key, table in units.items():
        gpu = key.split("/")[0]
        sessions = 2
        overhead_usd = sessions * over / 3600 * price[gpu] * disk
        comparison = phase_cost(["old_2p96M", "new_2p96M"], spec["seeds"], table) + overhead_usd
        ladder_keys = [f"new_{r / 1e6:.2f}M" for r in spec["rung_targets"] if r != 2_960_000]
        ladder = phase_cost(ladder_keys, spec["seeds"], table) + overhead_usd
        phases[key] = {"comparison_usd": round(comparison, 2), "ladder_usd": round(ladder, 2)}

    # H1's phases, every one priced from measured rates and plans/h1_accounting.json.
    h1 = A.read_json(A.LOG_DIR / "plans/h1_accounting.json")
    cap = CONTINGENCY["hard_cap"]
    l40 = price["L40S"] * disk
    sess = over / 3600 * l40

    def h1_unit(arm: str) -> dict:
        x = h1["contracts"]["M1_matched"][arm]
        rec_h = x["executed_positions"] / pos_s["L40S"] / 3600 * CONTINGENCY["recovery"]
        return {"recovery_hours": round(rec_h, 3), "eval_hours": round(eval_s / 3600, 3),
                "usd": round((rec_h + eval_s / 3600) * l40, 2),
                "executed_positions": x["executed_positions"], "optimizer_steps": x["optimizer_steps"]}

    ua, ub = h1_unit("A"), h1_unit("B")
    #: The GPU validation: four short runs (A and B, padding truncated or not, ~60 steps
    #: each at the measured step rate), one mid-run resume, and a 50-item timing slice
    #: of the eval -- a labelled planning figure, about 1.25 GPU hours.
    gpu_validation_h = CONTINGENCY["gpu_validation_hours"]
    closeout_per_session = (over / 3600 + 0.25) * l40
    rows = {
        "P1_generation": generation["expected_usd"],
        "P1v_verification_cpu": 0.0,
        "P1g_gpu_validation": gpu_validation_h * l40 + sess,
        "P2_formal_4_units": 2 * ua["usd"] + 2 * ub["usd"] + 2 * sess,
        "closeout_reserve_4_sessions": 4 * closeout_per_session,
    }
    storage_gb = spec["durability"]["h1_backend_gb"]
    storage_usd = round(storage_gb * spec["durability"]["network_volume_usd_per_gb_month"] * 2, 2)
    proposal_phases = {k: {"expected_usd": round(v, 2), "hard_cap_usd": round(v * cap, 2)}
                       for k, v in rows.items()}
    proposal_phases["storage_h1_backend_two_months"] = {"expected_usd": storage_usd, "hard_cap_usd": storage_usd,
                                                        "gb": storage_gb}
    optional = {"P2d_single_seed_diagnostic_B_0p86M": {
        "what": "optional engineering/diagnostic step, NOT a confirmation and not a gate on H1",
        "expected_usd": round((ub["recovery_hours"] * 860_000 / 2_960_000 + eval_s / 3600) * l40 + sess, 2)}}
    totals = {
        "h1_expected_usd": round(sum(r["expected_usd"] for r in proposal_phases.values()), 2),
        "h1_hard_cap_usd": round(sum(r["hard_cap_usd"] for r in proposal_phases.values()), 2),
        "excludes": "the optional diagnostic and any later ladder (a separate application)",
    }
    expected = totals["h1_expected_usd"]
    # The application now in front of the maintainer: P1 + P1v only, one A100 session.
    a100 = price["A100-80GB"] * disk
    p1_rows = {"P1_generation": generation["expected_usd"], "P1v_verification_cpu": 0.0,
               "closeout_reserve_1_session": (over / 3600 + 0.25) * a100}
    p1 = {k: {"expected_usd": round(v, 2), "hard_cap_usd": round(v * cap, 2)} for k, v in p1_rows.items()}
    p1_application = {**p1, "totals": {
        "expected_usd": round(sum(r["expected_usd"] for r in p1.values()), 2),
        "hard_cap_usd": round(sum(r["hard_cap_usd"] for r in p1.values()), 2)},
        "_what": ("one A100-80GB generation session (teacher_cal's measured GPU, engine and price) with a "
                  "closeout reserve of the session overhead + 15 minutes; P1v runs on the dev box at $0. "
                  "The whole-H1 rows (proposal_phases) are kept for reference and are NOT requested now")}

    students = 2 * spec["seeds"]  # H1: two arms x the seeds
    student_bytes = 2_384_285_634
    storage = {
        "students": students, "bytes_per_student_fp32_dir": student_bytes,
        "total_gib": round(students * student_bytes / 2**30, 2),
        "init_bytes": 1_192_135_096,
        "relay_headroom_bytes": spec["durability"]["relay_headroom_bytes"],
        "relay_probe_utc": spec["durability"]["relay_probe_utc"],
        "relay_can_hold": "one bf16 student; refuses one 2.38 GB fp32 student directory",
        "dev_box_free_gib": spec["durability"]["dev_box_free_gib"],
        "network_volume_usd_per_gb_month": spec["durability"]["network_volume_usd_per_gb_month"],
        "network_volume_50gb_two_months_usd": round(50 * spec["durability"]["network_volume_usd_per_gb_month"] * 2, 2),
    }
    doc = {
        "schema": "aadistill.stage2.reasoning_data.cost_forecast/v1",
        "status": "FORECAST from measured rates; authorizes nothing",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measured": m, "contingency_factors_not_measurements": CONTINGENCY,
        "plan_sizes": {k: spec[k] for k in ("rung_targets", "reserve_tokens", "seeds")},
        "generation": generation,
        "units": units,
        "phases": phases,
        "h1_units_L40S_M1": {"A": ua, "B": ub, "kinds": {
            "A_positions": h1["arm_A"]["kind"], "B_positions": h1["arm_B"]["kind"],
            "steps": h1["schedule"]["status"].split(":")[0],
            "positions_per_second": "measured on earlier L40S recovery runs (measured.*)",
            "hours_and_usd": "forecast = planned positions / measured rate x contingency"}},
        "optional": optional,
        "proposal_phases": proposal_phases,
        "proposal_totals": totals,
        "p1_application": p1_application,
        "_proposal_phases": "H1's rows, cited by plans/authorization_proposal.json: generation on one A100-80GB; GPU validation and the 4 formal units on L40S under M1 (recovery time from h1_accounting executed positions at the measured positions/s, eval at 2x the measured floor-student time); one session overhead per paid session; a closeout reserve of overhead + 15 min per session; hard cap = expected x the labelled factor (storage at cost)",
        "cpu": "every builder, audit, isolation check and pack runs on the dev box at $0 (minutes)",
        "storage": storage,
        "authorizes": "nothing",
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), "expected", expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

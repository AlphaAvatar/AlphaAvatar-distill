#!/usr/bin/env python3
"""SOB1's budget draft for the maintainer ($0, CPU): time, GPU hours and dollars per component and per phase, the
relay and dev-box storage each phase needs against what exists, all from the design's cost model, the structures and
TPI1's measurements. A draft: it authorizes nothing and no amount in it is approved.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/proposal.py --write | --check

Inputs: `design.json :: cost_model, arms, training`; `plans/structures.json` (FLOPs ratios, checkpoint bytes, memory);
`plans/resource_snapshot.json` (the project position, the relay's headroom and the dev box's free space, each read
once with its time). Every hour figure is an estimate labelled with its basis; ``bound`` is what a grant would cap.
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

from stages.single_operator_budget import structures as ST  # noqa: E402

PLANS = REPO / "logs/stages/stage-1/single_operator_budget/plans"
OUT = PLANS / "proposal.json"
SNAPSHOT = PLANS / "resource_snapshot.json"
SCHEMA = "aadistill.stage1.single_operator_budget.proposal/v1"
GIB = 2 ** 30


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rate(cm: dict) -> float:
    """All-in dollars per hour: the GPU and its container disk."""
    disk = cm["container_disk"]
    return cm["gpu"]["usd_per_hour"] + disk["gb"] * disk["usd_per_gb_month"] / disk["hours_per_month"]


def training_hours(arm: dict, steps: int, all_in_seconds: float, factor: float) -> dict:
    r = arm["training_flops_ratio_to_tpi1"]
    exp = all_in_seconds * r["with_attention"]
    bound = all_in_seconds * max(r.values()) * factor
    return {"seconds_per_step_expected": round(exp, 3), "seconds_per_step_bound": round(bound, 3),
            "hours_expected": round(exp * steps / 3600, 2), "hours_bound": round(bound * steps / 3600, 2)}


def depth_hours(cm: dict, n_root: int, n_keep: int) -> dict:
    """DEPTH 36 -> n_keep on the teacher: the journal's 36 -> 28 rounds as measured, the further rounds at the
    journal's seconds per candidate scaled by the layers that remain (expected) or unscaled (bound)."""
    ic = cm["init_session"]
    measured_s, measured_cands, journal_keep = 7073.8, 260, 28
    per = measured_s / measured_cands
    avg_measured = sum(range(journal_keep + 1, n_root + 1)) / (n_root - journal_keep)
    rest = list(range(n_keep + 1, journal_keep + 1))          #: rounds 9..20 score 28, 27, ..., 17 candidates
    exp = measured_s + sum(c * per * (c / avg_measured) for c in rest)
    bound = (measured_s + sum(c * per for c in rest)) * ic["bound_factor"]
    return {"candidate_subsets": sum(range(n_keep + 1, n_root + 1)), "hours_expected": round(exp / 3600, 2),
            "hours_bound": round(bound / 3600, 2),
            "rule": "rounds 1-8 at the journal's measured 7,074 s; rounds 9-20 at 27.2 s per candidate x remaining / "
                    f"{avg_measured} (expected) or x 1 (bound), the bound x {ic['bound_factor']}"}


def build(design: dict | None = None) -> dict:
    d = design or json.loads(ST.DESIGN.read_text())
    s = json.loads(ST.OUT.read_text())
    if s["design_sha256"] != ST.sha256_file(ST.DESIGN):
        raise SystemExit("structures.json was written under another design; rerun structures.py --write")
    snap = json.loads(SNAPSHOT.read_text())
    cm = d["cost_model"]
    usd_h = rate(cm)
    steps = d["training"]["data"]["after_packing"]["optimizer_steps"]
    seeds = d["training"]["seeds"]
    setup_h = cm["session_setup_minutes"] / 60
    all_in = s["tpi1_measured"]["all_in_seconds_per_step"]
    arms = list(s["arms"])
    train = {a: training_hours(s["arms"][a], steps, all_in, cm["init_session"]["bound_factor"]) for a in arms}
    dep = depth_hours(cm, s["teacher"]["geometry"]["num_hidden_layers"],
                      s["arms"]["depth_only"]["geometry"]["num_hidden_layers"])
    other_ops_h = (185.3 + 85.0 + 172.1) / 3600 + cm["init_session"]["builds_relay_state_evaluation_minutes"] / 60
    init = {"hours_expected": round(setup_h + dep["hours_expected"] + other_ops_h, 2),
            "hours_bound": round(setup_h + dep["hours_bound"] + other_ops_h * cm["init_session"]["bound_factor"], 2),
            "depth": dep, "other_operators_and_builds_hours": round(other_ops_h, 2)}
    pre = cm["pre_recovery_evaluation"]
    post = cm["post_recovery_evaluation"]
    comp = {
        "init_session": init,
        "pre_recovery_evaluation": {"models": len(arms), "hours_expected": pre["expected_hours_per_arm"] * len(arms),
                                    "hours_bound": pre["bound_hours_per_arm"] * len(arms)},
        "training": {"units": len(arms) * len(seeds), "per_arm_per_seed": train,
                     "hours_expected": round(len(seeds) * sum(t["hours_expected"] for t in train.values()), 2),
                     "hours_bound": round(len(seeds) * sum(t["hours_bound"] for t in train.values()), 2)},
        "post_recovery_evaluation": {"units": len(arms) * len(seeds),
                                     "hours_expected": round(post["expected_hours_per_unit"] * len(arms) * len(seeds), 2),
                                     "hours_bound": round(post["bound_hours_per_unit"] * len(arms) * len(seeds), 2)},
        "training_session_setup": {"sessions": len(arms) * len(seeds),
                                   "hours_expected": round(setup_h * len(arms) * len(seeds), 2),
                                   "hours_bound": round(setup_h * len(arms) * len(seeds), 2)},
        "teacher_at_8192_optional": {"hours_expected": cm["teacher_at_8192"]["expected_hours"],
                                     "hours_bound": cm["teacher_at_8192"]["bound_hours"]},
    }
    for c in comp.values():
        c["usd_expected"] = round(c["hours_expected"] * usd_h, 2)
        c["usd_bound"] = round(c["hours_bound"] * usd_h, 2)

    def phase(parts: list[str], seed_share: float = 1.0) -> dict:
        h_e = sum(comp[p]["hours_expected"] * (seed_share if p in ("training", "post_recovery_evaluation",
                                                                    "training_session_setup") else 1) for p in parts)
        h_b = sum(comp[p]["hours_bound"] * (seed_share if p in ("training", "post_recovery_evaluation",
                                                                 "training_session_setup") else 1) for p in parts)
        return {"components": parts, "gpu_hours_expected": round(h_e, 1), "gpu_hours_bound": round(h_b, 1),
                "usd_expected": round(h_e * usd_h, 2), "usd_bound": round(h_b * usd_h, 2),
                "usd_bound_plus_reserve": round(h_b * usd_h + cm["teardown_reserve_usd"], 2)}
    p1 = ["init_session", "pre_recovery_evaluation"]
    p2 = ["training", "post_recovery_evaluation", "training_session_setup"]
    remaining = snap["project_position"]["project_remaining_usd"]
    options = {
        "phase_1_initialization_and_damage": phase(p1),
        "phase_2_recovery_two_seeds": phase(p2),
        "phase_2_recovery_one_seed": phase(p2, 0.5),
        "full_two_seeds": phase(p1 + p2),
        "full_two_seeds_with_teacher_at_8192": phase(p1 + p2 + ["teacher_at_8192_optional"]),
    }
    for o in options.values():
        o["fits_project_remaining_now"] = o["usd_bound_plus_reserve"] <= remaining
    sizes = {a: s["arms"][a]["checkpoint_bytes"] for a in arms}
    inits = sum(v["init_bf16"] for v in sizes.values())
    finals = sum(v["final_fp32"] for v in sizes.values()) * len(seeds)
    rotation = 2 * max(v["save_point"] for v in sizes.values())
    head = snap["relay"]["headroom_gib"][0] * GIB
    storage = {
        "per_arm_bytes": sizes,
        "relay_headroom_bytes": int(head),
        "phase_1_relay_peak_bytes": inits,
        #: during a rotation two save points; at a unit's end one save point and its final (smaller than two)
        "phase_2_relay_peak_bytes_finals_leave_after_evaluation": inits + rotation,
        #: the last unit's rotation beside every earlier unit's final (the smallest final is the last one's, to bound)
        "phase_2_relay_peak_bytes_finals_accumulate": inits + rotation + finals - min(v["final_fp32"] for v in sizes.values()),
        "six_finals_bytes": finals,
        "three_inits_bytes": inits,
        "dev_box_free_bytes": snap["dev_box"]["free_bytes"],
        "rule": "a save point is deleted only after the next one hands off (TPI1's driver), so a rotation holds two; a "
                "final is uploaded while its unit's last save point still exists",
    }
    storage["phase_1_fits_relay"] = storage["phase_1_relay_peak_bytes"] <= head
    storage["phase_2_fits_relay_if_finals_leave"] = storage["phase_2_relay_peak_bytes_finals_leave_after_evaluation"] <= head
    storage["phase_2_fits_relay_if_finals_accumulate"] = storage["phase_2_relay_peak_bytes_finals_accumulate"] <= head
    storage["finals_fit_dev_box"] = finals <= snap["dev_box"]["free_bytes"]
    memory = {a: s["arms"][a]["memory"] for a in arms}
    return {
        "schema": SCHEMA, "status": "proposed: a draft for the maintainer; no amount here is approved, and the project "
                                    "cap is not raised by it",
        "design_sha256": ST.sha256_file(ST.DESIGN), "structures_sha256": sha256_file(ST.OUT),
        "resource_snapshot_sha256": sha256_file(SNAPSHOT),
        "usd_per_hour_all_in": round(usd_h, 4), "components": comp, "options": options,
        "project_position": snap["project_position"], "storage": storage, "memory_gib": memory,
        "authorizes": "nothing",
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
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)}")
        return 0
    same = OUT.is_file() and OUT.read_text() == text
    print("proposal.json reproduces" if same else "proposal.json does NOT reproduce")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SOB1's budget for the maintainer ($0, CPU): phase 1 as a proposal the issuer can read (its ceiling, per-session
hard cap and teardown reserve, `shared.study_engineering`'s grant fields), and phase 2 -- recovery training of the
three arms x two seeds and their evaluation -- as a separate draft that proposes no cap. Nothing here is approved.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/single_operator_budget/proposal.py --write | --check

Inputs: `design.json :: cost_model, arms, training, phase_1`; `plans/structures.json` (FLOPs ratios, checkpoint
bytes, memory); `plans/resource_snapshot.json` (the project position, the relay's headroom and the dev box's free
space, each read once with its time; read again at issue and launch). Every hour figure is an estimate labelled with
its basis; ``bound`` is what a hard cap covers.

A session's hard cap is what `study_engineering`'s session bills at most: setup, transfer, the driver's bound with the
launch's contingency, the closeout and artifact-recovery reserves and the watchdog's minutes, at the highest price the
proposal admits plus the container disk (`hard_cap`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SCHEMA = "aadistill.stage1.single_operator_budget.proposal/v2"
GIB = 2 ** 30
#: the session's fixed reserves as `study_engineering.session_spec` builds them (minutes), and its contingency
SESSION = {"setup": 25.0, "transfer": 5.0, "closeout": 10.0, "artifact_recovery": 10.0, "contingency_fraction": 0.05}
#: the highest L40S price a session may accept (the proposal's ``gpus``); the accepted price has been $1.09
MAX_PRICE = 1.11


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def disk_rate(cm: dict) -> float:
    disk = cm["container_disk"]
    return disk["gb"] * disk["usd_per_gb_month"] / disk["hours_per_month"]


def rate(cm: dict) -> float:
    """All-in dollars per hour at the accepted price: the GPU and its container disk."""
    return cm["gpu"]["usd_per_hour"] + disk_rate(cm)


def hard_cap(driver_minutes: float, cm: dict) -> dict:
    """The most one session bills, as `study_engineering`'s runner prices it (module docstring): the contingency on
    setup, transfer and the driver alike, the closeout reserve with it, then the artifact-recovery reserve and the
    watchdog's minutes, at the highest admitted price plus the container disk at `study_engineering.all_in`'s rate
    (a 30.4375-day month); ceiled to the cent. The 2026-11-19 dry run refused the first version, which left the
    contingency off setup and transfer ($22.53 against the runner's $22.5485)."""
    from shared.study_engineering import DISK_USD_PER_GB_MONTH, TERMINATION_MINUTES
    f = 1.0 + SESSION["contingency_fraction"]
    minutes = ((SESSION["setup"] + SESSION["transfer"] + driver_minutes) * f + SESSION["closeout"] * f
               + SESSION["artifact_recovery"] + TERMINATION_MINUTES)
    disk_h = cm["container_disk"]["gb"] * DISK_USD_PER_GB_MONTH / (24 * 30.4375)
    usd = math.ceil(minutes / 60 * (MAX_PRICE + disk_h) * 1e4) / 1e4
    return {"driver_minutes": round(driver_minutes, 1), "billable_minutes": round(minutes, 2),
            "usd": math.ceil(usd * 100) / 100}


def training_hours(arm: dict, steps: int, all_in_seconds: float, factor: float) -> dict:
    r = arm["training_flops_ratio_to_tpi1"]
    exp = all_in_seconds * r["with_attention"]
    bound = all_in_seconds * max(r.values()) * factor
    return {"seconds_per_step_expected": round(exp, 3), "seconds_per_step_bound": round(bound, 3),
            "hours_expected": round(exp * steps / 3600, 2), "hours_bound": round(bound * steps / 3600, 2)}


def depth_hours(cm: dict, n_root: int, n_keep: int, resumed: int) -> dict:
    """DEPTH 36 -> n_keep resumed after ``resumed`` rounds: the new rounds at the journal's seconds per candidate
    scaled by the layers that remain (expected) or unscaled (bound), plus round 8's re-score."""
    ic = cm["init_session"]
    measured_s, measured_cands = 7073.8, 260
    per = measured_s / measured_cands
    avg = sum(range(n_root - resumed + 1, n_root + 1)) / resumed
    new = [n_root - r for r in range(resumed, n_root - n_keep)]
    rescore = per * (n_root - resumed + 1) / avg                       #: one candidate at round 8's depth
    exp = sum(c * per * (c / avg) for c in new) + rescore
    bound = (sum(c * per for c in new) + per) * ic["bound_factor"]
    return {"candidate_subsets_new": sum(new), "candidates_per_new_round": new,
            "hours_expected": round(exp / 3600, 2), "hours_bound": round(bound / 3600, 2),
            "rule": f"rounds {resumed + 1}-{n_root - n_keep} at the journal's {per:.1f} s per candidate x remaining / "
                    f"{avg} (expected) or x 1 (bound) plus round {resumed}'s re-score, the bound x {ic['bound_factor']}"}


def transfer_hours(nbytes: float, cm: dict, d: dict) -> dict:
    """One upload and its verifying download: at TPI1's median rates (expected) and at the floors (bound)."""
    t = cm["transfer"]
    exp = nbytes / t["expected_upload_bytes_per_second"] + nbytes / t["expected_download_bytes_per_second"]
    bound = (nbytes / d["upload_floor_bytes_per_second"] + nbytes / d["download_floor_bytes_per_second"]) \
        * d["transfer_margin"]
    return {"bytes": int(nbytes), "hours_expected": round(exp / 3600, 3), "hours_bound": round(bound / 3600, 3)}


def build(design: dict | None = None) -> dict:
    d = design or json.loads(ST.DESIGN.read_text())
    s = json.loads(ST.OUT.read_text())
    if s["design_sha256"] != ST.sha256_file(ST.DESIGN):
        raise SystemExit("structures.json was written under another design; rerun structures.py --write")
    sel = json.loads((REPO / "configs/stages/stage-1/single_operator_budget/selection.json").read_text())
    snap = json.loads(SNAPSHOT.read_text())
    cm, dur = d["cost_model"], d["training"]["durability"]
    ic = cm["init_session"]
    usd_h = rate(cm)
    steps = d["training"]["data"]["after_packing"]["optimizer_steps"]
    seeds = d["training"]["seeds"]
    setup_h = cm["session_setup_minutes"] / 60
    all_in = s["tpi1_measured"]["all_in_seconds_per_step"]
    arms = list(s["arms"])
    sizes = {a: s["arms"][a]["checkpoint_bytes"] for a in arms}

    # ---- phase 1: one sob1_phase1 session (three builds, then four models evaluated at 8,192) ----
    dep = depth_hours(cm, s["teacher"]["geometry"]["num_hidden_layers"],
                      s["arms"]["depth_only"]["geometry"]["num_hidden_layers"],
                      sel["depth_only"]["resume"]["rounds_replayed"])
    other_s = 185.0 + ic["residual_width_replay_seconds"] + 85.0           #: FFN pass, width replay, attention gate
    per_arm_h = (ic["per_arm_build_state_evaluation_minutes"] + ic["teacher_load_minutes_per_arm"]) / 60
    inits_bytes = sum(v["init_bf16"] for v in sizes.values())
    relay_init = transfer_hours(inits_bytes, cm, dur)
    builds = {"hours_expected": round(dep["hours_expected"] + other_s / 3600 + len(arms) * per_arm_h
                                      + relay_init["hours_expected"], 2),
              "hours_bound": round(dep["hours_bound"] + (other_s / 3600 + len(arms) * per_arm_h) * ic["bound_factor"]
                                   + relay_init["hours_bound"], 2),
              "depth": dep, "relay_copies_of_the_inits": relay_init,
              "other_operators_seconds": other_s, "per_arm_build_and_state_evaluation_hours": round(per_arm_h, 3)}
    pre, teach, post = cm["pre_recovery_evaluation"], cm["teacher_at_8192"], cm["post_recovery_evaluation"]
    p1 = {"builds": builds,
          "pre_recovery_evaluation": {"models": len(arms), "hours_expected": round(pre["expected_hours_per_arm"] * len(arms), 2),
                                      "hours_bound": round(pre["bound_hours_per_arm"] * len(arms), 2)},
          "teacher_at_8192": {"models": 1, "hours_expected": teach["expected_hours"], "hours_bound": teach["bound_hours"]}}
    p1_driver_exp = sum(c["hours_expected"] for c in p1.values())
    p1_driver_bound = sum(c["hours_bound"] for c in p1.values())
    cap1 = hard_cap(p1_driver_bound * 60, cm)
    phase_1 = {"session_kind": d["phase_1"]["session_kind"], "sessions": 1, "components": p1,
               "driver_hours_expected": round(p1_driver_exp, 2), "driver_hours_bound": round(p1_driver_bound, 2),
               "gpu_hours_expected": round(p1_driver_exp + setup_h, 2),
               "usd_expected": round((p1_driver_exp + setup_h) * usd_h, 2),
               "hard_cap": cap1, "teardown_reserve_usd": cm["teardown_reserve_usd"],
               "hard_cap_plus_reserve_usd": round(cap1["usd"] + cm["teardown_reserve_usd"], 2)}

    # ---- phase 2: one sob1_train session per arm (both seeds), never proposed for a grant here ----
    train = {a: training_hours(s["arms"][a], steps, all_in, ic["bound_factor"]) for a in arms}
    per_arm = {}
    for a in arms:
        fetch = transfer_hours(sizes[a]["init_bf16"], cm, dur)
        final = transfer_hours(sizes[a]["final_fp32"], cm, dur)
        last_sp = transfer_hours(sizes[a]["save_point"], cm, dur)
        exp = len(seeds) * (train[a]["hours_expected"] + post["expected_hours_per_unit"] + final["hours_expected"]) \
            + fetch["hours_expected"]
        bound = len(seeds) * (train[a]["hours_bound"] + post["bound_hours_per_unit"] + final["hours_bound"]) \
            + fetch["hours_bound"] + last_sp["hours_bound"]
        cap = hard_cap(bound * 60, cm)
        per_arm[a] = {"units": len(seeds), "training_per_seed": train[a], "post_recovery_evaluation_hours_per_unit":
                      {"expected": post["expected_hours_per_unit"], "bound": post["bound_hours_per_unit"]},
                      "init_fetch": fetch, "final_copy_per_seed": final, "last_save_point_copy": last_sp,
                      "driver_hours_expected": round(exp, 2), "driver_hours_bound": round(bound, 2),
                      "usd_expected": round((exp + setup_h) * usd_h, 2), "hard_cap": cap,
                      "unit_deadline_hours": round(train[a]["hours_bound"] * 1.0, 1)}
    p2_exp = sum(v["usd_expected"] for v in per_arm.values())
    p2_cap = sum(v["hard_cap"]["usd"] for v in per_arm.values())
    phase_2 = {"session_kind": d["phase_2"]["session_kind"], "sessions": len(arms), "order": d["training"]["unit_order"]["arms"],
               "per_arm_session": per_arm, "usd_expected_two_seeds": round(p2_exp, 2),
               "usd_hard_caps_two_seeds": round(p2_cap, 2),
               "usd_hard_caps_plus_reserve": round(p2_cap + cm["teardown_reserve_usd"], 2),
               "one_seed_usd_expected": round(sum(
                   (v["driver_hours_expected"] - (v["training_per_seed"]["hours_expected"]
                                                  + v["post_recovery_evaluation_hours_per_unit"]["expected"]
                                                  + v["final_copy_per_seed"]["hours_expected"]) + setup_h) * usd_h
                   for v in per_arm.values()), 2),
               "status": "a draft for the maintainer: no cap here is proposed for a grant, and the project cap is not raised"}

    # ---- storage ----
    inits = inits_bytes
    finals = sum(v["final_fp32"] for v in sizes.values()) * len(seeds)
    rotation = 2 * max(v["save_point"] for v in sizes.values())
    head = snap["relay"]["headroom_gib"][0] * GIB
    free = snap["dev_box"]["free_bytes"]
    storage = {
        "per_arm_bytes": sizes, "relay_headroom_bytes": int(head),
        "phase_1_relay_peak_bytes": inits, "phase_1_fits_relay": inits <= head,
        "phase_2_relay_peak_bytes_finals_leave_after_evaluation": inits + rotation,
        "phase_2_fits_relay_if_finals_leave": inits + rotation <= head,
        "phase_2_relay_peak_bytes_finals_accumulate": inits + rotation + finals - min(v["final_fp32"] for v in sizes.values()),
        "phase_2_fits_relay_if_finals_accumulate":
            inits + rotation + finals - min(v["final_fp32"] for v in sizes.values()) <= head,
        "dev_box_free_bytes": free, "dev_box_read_utc": snap["dev_box"]["read_utc"],
        "phase_1_local_bytes": inits, "phase_1_fits_dev_box": inits <= free,
        "phase_2_local_bytes": finals, "dev_box_free_after_phase_1_bytes": free - inits,
        "phase_2_dev_box_shortfall_bytes": max(0, finals - (free - inits)),
        "rule": "a save point is deleted only after the next one hands off, so a rotation holds two; a final is uploaded "
                "while its unit's last save point still exists; the dev box keeps the inits (phase 1) and would keep "
                "the finals after their evaluation (phase 2)"}
    proj = snap["project_position"]
    reserve = cm["teardown_reserve_usd"]
    ceiling = cap1["usd"]
    return {
        "schema": SCHEMA, "study": d["study_id"], "status": "proposed", "recorded_utc": d["recorded_utc"][:10],
        "scope": ("phase 1 of SOB1 only (configs/stages/stage-1/single_operator_budget/design.json): one sob1_phase1 "
                  "session builds the three ~2B initializations from the original teacher with A's operators (depth "
                  "only resumed from the recorded search's round 9) and evaluates each untrained arm and the teacher "
                  "on all of eval-set-1 in both modes at 8,192 tokens. No training, no other arm, no network volume. "
                  "Phase 2 (recovery training and its evaluation) is a separate draft below and proposes no cap"),
        "ceiling_usd": ceiling, "teardown_reserve_usd": reserve, "study_cumulative_cap_usd": round(ceiling + reserve, 2),
        "per_session_hard_cap_usd": cap1["usd"],
        "min_useful_attempt_usd": hard_cap(60, cm)["usd"],
        "project_cumulative_cap_usd": proj["project_cap_usd"], "gpus": {cm["gpu"]["type"]: MAX_PRICE}, "gpu_count_max": 1,
        "expected_total_usd": phase_1["usd_expected"],
        "kinds": {
            d["phase_1"]["session_kind"]: {"per_session_hard_cap_usd": cap1["usd"], "expected_usd": phase_1["usd_expected"],
                                           "driver_minutes": cap1["driver_minutes"],
                                           "basis": "phase_1.components (design.json :: cost_model)"},
            d["phase_2"]["session_kind"]: {"per_session_hard_cap_usd": None, "expected_usd": None,
                                           "status": "not proposed: phase 2 needs the maintainer's own decision and a "
                                                     "cap the project's remainder does not hold (phase_2_draft)"}},
        "position_at_proposal": proj, "granted_by": "nobody: a proposal, not a grant",
        "decision": "PENDING the maintainer's explicit approval",
        "money_rules": {
            "own_envelope": "the ceiling counts only this study's runs; TPI1, RI-S1, RI-LM1 and Research A are CLOSED and "
                            "fund nothing here; a provider top-up is not a grant",
            "ceiling": f"${ceiling:.2f}: phase 1's one session at its hard cap, cumulatively over every attempt",
            "teardown_reserve": f"${reserve:.2f} held outside the ceiling and never issued: each artifact's hard cap plus "
                                "the reserve must fit what the project cap and the provider balance leave, at issue and at "
                                "launch (layer_merge_launch.reserve_problems)",
            "resume": "a replacement attempt takes over every arm whose init record verified and every sealed sample; "
                      "its cap is what the ceiling leaves",
            "project_cap": "the $1,000 project cap is not raised by this proposal"},
        "hardware": ("one NVIDIA L40S per session, secure cloud, at most $1.11/h, 150 GB container disk; the packed "
                     "search's DEPTH ran on the same card (29.5 GiB reserved); no network volume"),
        "usd_per_hour_all_in": round(usd_h, 4),
        "design_sha256": ST.sha256_file(ST.DESIGN), "structures_sha256": sha256_file(ST.OUT),
        "selection_sha256": sha256_file(REPO / "configs/stages/stage-1/single_operator_budget/selection.json"),
        "resource_snapshot_sha256": sha256_file(SNAPSHOT),
        "phase_1": phase_1, "phase_2_draft": phase_2, "storage": storage,
        "memory_gib": {a: s["arms"][a]["memory"] for a in arms},
        "authorizes": "nothing: a proposal; the grant is the maintainer's",
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

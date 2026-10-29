#!/usr/bin/env python3
"""Research A's D4-scale phase on the L40S, priced from the project's own measurements, alone. $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/phase_forecast.py --write

Every rate is a measurement a record owns (``MEASURED`` names it) or a declared
assumption (``ASSUMED``, with its basis and a conservative value): the L40S has
no measurement yet of the hybrid's generation throughput, of the isolated
Top-200 training step or of the D4 path on long items, and the L40S check
(``l40s_check.py``) measures exactly those before the teacher session is
priced. Every scale is the design's; the scenarios differ only in the declared
``SCENARIOS`` multipliers. A session's hard cap is the larger of its expected
cost x 1.5 and its conservative cost. A budget is never converted from another
card by the price ratio: hours are derived per operation from the measured
rate and the declared card factor, then priced at the L40S rate.

PRICE. Every cost is at the live L40S price (``MEASURED``); ``PRICE_CEILING``
is the highest price the proposal admits, at which ``bounds`` re-prices each
envelope: the GPU-hours its hard cap buys at each price, and the longest work a
single session's artifact at that cap admits (the shared planner's own bound,
`aadistill.infrastructure.budget.plan_session`). A hard cap is dollars, never
hours: work that does not fit one session runs as bounded sessions (the teacher
session first through W00, then re-priced from W00's measured rate). Two GPUs
on one recovery pod are an ESTIMATE from a schedule of the six units, not half
the wall-clock at the same cost: the pod bills both GPUs through setup, the
closeout and the shorter GPU's idle tail.

STORAGE compares the two ways to hold the phase's bytes, at one and at two
GPUs per recovery pod: expanding ``am5m373kuq`` (its historical checkpoints
stay where they are) and a new volume in an L40S datacenter. The existing
volume's 30 GB through 2026-12-24 is already booked in the committed position
(its storage record's reservation) and is never counted again here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "logs/budget/applications/research_a_2026-10-29/forecast.json"
GB = 1e9

MEASURED = {
    "l40s_usd_per_hour": (1.09, "the provider's live securePrice for NVIDIA L40S (1 and 2 GPUs), read 2026-10-29T09:24Z; H1-PA's billed rate (runs/h1pa2_20261025_055042)"),
    "t0_generated_tokens_2507": (12_920_708, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: measured.generated_tokens_2507 (H1's P1 + P1t + PA1 records, both candidates)"),
    "tokens_per_session_minute_a100": (80_198, "the same :: measured.tokens_per_session_minute (H1's three generation sessions, A100 SXM 80GB)"),
    "t0_ht_s_sequences_at_reference": (1235, "the same :: scenarios.expected.sequences.H-T-S (T0, admitted_both 0.70, H-T tokens 1.0x)"),
    "isolated_seconds_per_step_a100": (5.19, "logs/stages/stage-2/teacher_modes/runs/ae_20261028_093428/evidence/trainer_smoke.json (document_isolated, full-vocabulary KD, the hybrid on the fly, A100)"),
    "e1_seconds_per_step_l40s": (3.6314, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: measured.seconds_per_step (H1-PA's E1 recipe, cross-context, L40S)"),
    "e1_seconds_per_step_a100": (3.33, "logs/stages/stage-2/reasoning_data/runs/h1p1g_20261023_191644/evidence/resume_B_X1.train_log.jsonl (the same E1 recipe, median after 3 steps, A100)"),
    "student_eval_seconds_l40s": (3917.1, "logs/stages/stage-2/reasoning_data/runs/h1pa2_20261025_055042/closeout/outcome.json :: D4-Augmented-s20260801 evaluate (all of eval-set-1, nucleus_v1, L40S)"),
    "teacher_over_student_eval_one_card": (3.54, "logs/stages/stage-0/calibration_volume/runs/battery_20261018_092345/evidence: teacher_dev 3,390.7 s against e1_r2960k_sa_pca_dev 958.7 s on the same A100 (the fluid dev battery)"),
    "operator_seconds_d4_path_a100": (2446.4, "the dose_d4 search telemetry: D4 Top-1's ATTENTION 98.9 + DEPTH 2,258.9 + WIDTH 43.9 + FFN 44.7 s (A100, micro-batch 3)"),
    "save_point_bytes": (5_908_258_491, "configs/stages/stage-2/reasoning_data/h1pa.json :: storage.bytes.save_point (this geometry)"),
    "checkpoint_bytes": (2_384_234_968, "the same :: storage.bytes.model_safetensors"),
    "evaluation_bytes_per_copy_bound": (250_000_000, "the same :: storage.bytes.evaluation_per_unit_bound (held twice: progress and the persisted set)"),
    "shard_bytes_per_generated_token": (12.99, "artifacts/stages/stage-2/reasoning_data/generation (H1's sealed shards): 182,132,932 B for 14,026,035 generated tokens"),
    "generation_dir_over_shards": (1.59, "artifacts/stages/stage-2/reasoning_data/generation/h1gen_20261022_190653: 233,202,261 B directory over 146,860,932 B of shards"),
    "am5m373kuq_used_bytes": (14_711_955_248, "logs/stages/stage-2/reasoning_data/storage/h1pa_volume_am5m373kuq.json :: contents_after_mc2 (9,943,482,064 B probed before MC2 + the two Mix-Augmented checkpoints)"),
    "am5m373kuq_size_gb": (30, "the provider, read 2026-10-29T09:24Z (EUR-IS-2)"),
    "volume_usd_per_gb_month": (0.07, "h1pa.json :: storage.rate_usd_per_gb_month"),
}
ASSUMED = {
    "l40s_generation_vs_a100": {
        "expected": 0.40, "conservative": 0.30,
        "basis": ("NOT MEASURED. Decoding long reasoning traces at high concurrency reads each sequence's KV cache "
                  "every step, so it is memory-bandwidth bound: the L40S's 864 GB/s is 0.42 of the A100 SXM4's "
                  "2,039 GB/s, and at gpu_memory_utilization 0.90 the L40S holds about half the A100's KV cache "
                  "(about 30 GiB after the 8 GB of weights, against about 63), which bounds concurrency; prefill is "
                  "compute-bound, where the two cards are comparable. Replaced by the L40S check's measured tokens "
                  "per second on the teacher session's own runner")},
    "l40s_step_vs_a100": {
        "expected": round(3.6314 / 3.33, 4), "conservative": 1.25,
        "basis": ("the one measured same-recipe pair (E1's recipe, cross-context): 3.6314 s on the L40S against 3.33 s "
                  "on the A100; conservative 1.25 because document isolation runs a masked attention kernel the pair "
                  "did not exercise. Replaced by the check's measured formal step")},
    "l40s_operators_vs_a100": {
        "expected": round(3.6314 / 3.33, 4), "conservative": 1.5,
        "basis": "the same training-step ratio; no L40S timing of the D4 path on these items exists"},
    "micro_batch_1_vs_3": {
        "expected": 1.3, "conservative": 2.0,
        "basis": ("the initialization's out-of-memory fallback (d4_scale_design.json :: initialization.execution): "
                  "three times the forwards of a third the rows each, on items that already fill the card")},
}
DESIGN = {"t0_tasks": 1761, "v_tasks": 240, "wave": 400, "e_waves": 9, "target_sequences": 2468, "calibration_prompts": 160,
          "seeds": 2, "epochs": 3, "batch": 2, "x_tasks": 14000, "units": 6,
          "check": {"generation_tasks": 64, "engine_tasks": 16, "train_steps_per_mode": 20}}
SCENARIOS = {
    "expected": {"ht_tokens": 1.0, "hn_tokens": 0.25, "admitted_both": 0.70, "step": 1.0, "student_eval": 0.75,
                 "teacher_eval_ht": 3.5, "teacher_eval_hn": 1.75, "hn_admitted": 0.75, "init_oom_attempt_h": 0.25},
    "conservative": {"ht_tokens": 1.15, "hn_tokens": 0.25, "admitted_both": 0.55, "step": 1.25, "student_eval": 1.0,
                     "teacher_eval_ht": 5.0, "teacher_eval_hn": 3.5, "hn_admitted": 0.65, "init_oom_attempt_h": 0.5},
}
SETUP_H, CLOSE_H, UNIT_FIXED_H, LOAD_H = 20 / 60, 0.25, 10 / 60, 5 / 60
HARD_CAP_FACTOR = 1.5
#: the proposal's highest admitted L40S price per GPU-hour, and phase.launch_spec's planning contingency
PRICE_CEILING, CONTINGENCY = 1.19, 0.25
DISK_USD_PER_HOUR = 150 * 0.10 / (24 * 30.4375)        #: 150 GB container disk at $0.10/GB-month
STORAGE_MONTHS, STORAGE_MARGIN, ROUND_GB = 2, 1.10, 5
ENGINEERING_RESERVE = 10.0


def m(k: str) -> float:
    return MEASURED[k][0]


def a(k: str, scenario: str) -> float:
    return ASSUMED[k][scenario]


def usd(hours: float, price: float | None = None) -> float:
    return round(hours * ((price or m("l40s_usd_per_hour")) + DISK_USD_PER_HOUR), 2)


def longest_session_hours(cap: float, pod_price: float) -> float:
    """The longest work (``launch --minutes``) one artifact at `cap` admits at `pod_price`: the shared planner's
    refusal point for study_engineering.session_spec's plan (setup, 5 min transfer, the work, x (1 +
    contingency), the closeout reserve, 10 min artifact recovery), by bisection."""
    from aadistill.infrastructure.budget import BudgetError, Phase, StepTime, plan_session

    def fits(minutes: float) -> bool:
        try:
            plan_session(price_per_hour=pod_price, authorized_usd=cap, arms=0, steps_per_arm=0,
                         step_time=StepTime(1.0, "unused: one phase"), setup_minutes=SETUP_H * 60, transfer_minutes=5.0,
                         other_phases=(Phase("work", minutes),), contingency_fraction=CONTINGENCY,
                         soft_stop_reserves=(Phase("closeout_and_teardown", round((1 + CONTINGENCY) * 10.0, 2)),),
                         artifact_recovery_reserve_minutes=10.0)
            return True
        except BudgetError:
            return False
    lo, hi = 0.0, cap / pod_price * 60
    for _ in range(40):
        lo, hi = ((lo + hi) / 2, hi) if fits((lo + hi) / 2) else (lo, (lo + hi) / 2)
    return math.floor(lo / 60 * 100) / 100


def per_task_tokens() -> float:
    return m("t0_generated_tokens_2507") / DESIGN["t0_tasks"]


def gen_hours(tokens: float, key: str) -> float:
    return tokens / (m("tokens_per_session_minute_a100") * a("l40s_generation_vs_a100", key)) / 60


def init_hours(key: str, s: dict) -> float:
    return (m("operator_seconds_d4_path_a100") * a("l40s_operators_vs_a100", key) * a("micro_batch_1_vs_3", key) / 3600
            + s["init_oom_attempt_h"] + LOAD_H)


def check_session(key: str, s: dict) -> dict:
    c = DESIGN["check"]
    tokens = c["generation_tasks"] * per_task_tokens() * (s["ht_tokens"] + s["hn_tokens"])
    engine_tokens = c["engine_tasks"] * per_task_tokens() * s["hn_tokens"]
    step_s = m("isolated_seconds_per_step_a100") * a("l40s_step_vs_a100", key) * s["step"]
    hours = {"setup": SETUP_H, "generation": gen_hours(tokens, key) + LOAD_H,
             "evaluation_engine": gen_hours(engine_tokens, key) + LOAD_H,
             "train": 2 * (LOAD_H + c["train_steps_per_mode"] * step_s / 3600), "initialization": init_hours(key, s),
             "closeout": CLOSE_H}
    h = sum(hours.values())
    return {"hours": {k: round(v, 2) for k, v in hours.items()}, "hours_total": round(h, 2), "gpu": "1 x NVIDIA L40S",
            "generated_tokens_m": round((tokens + engine_tokens) / 1e6, 2), "usd": usd(h)}


def teacher_session(key: str, s: dict) -> dict:
    both = per_task_tokens() * (s["ht_tokens"] + s["hn_tokens"])
    s_t0 = m("t0_ht_s_sequences_at_reference") * s["ht_tokens"] * s["admitted_both"] / 0.70
    e_needed = math.ceil(DESIGN["t0_tasks"] * max(0.0, DESIGN["target_sequences"] / s_t0 - 1))
    e_generated = min(DESIGN["e_waves"] * DESIGN["wave"], (math.ceil(e_needed / DESIGN["wave"]) + 1) * DESIGN["wave"])
    #: X: H-N documents of X fill H-N-C to H-T-S's count; H-N-S already brings hn/ht of it (by tokens)
    hts_generated = DESIGN["target_sequences"] * 8192 * 0.94 * 0.9
    x_tokens = max(0.0, (0.7 - s["hn_tokens"] / s["ht_tokens"])) * hts_generated / s["hn_admitted"] * 1.15
    tokens = {"calibration": DESIGN["calibration_prompts"] * both * 0.6, "t0": DESIGN["t0_tasks"] * both,
              "v": DESIGN["v_tasks"] * both, "e": e_generated * both, "x": x_tokens}
    teval_h = m("student_eval_seconds_l40s") / 3600 * (s["teacher_eval_ht"] + s["teacher_eval_hn"])
    hours = {"setup": SETUP_H, "generation": gen_hours(sum(tokens.values()), key), "initialization": init_hours(key, s),
             "teacher_evaluation": teval_h, "closeout": CLOSE_H}
    h = sum(hours.values())
    #: the bounded plan: calibration, the initialization and W00 first; the rest re-priced from W00's measured rate
    first = SETUP_H + gen_hours(tokens["calibration"] + tokens["t0"] + tokens["v"], key) + hours["initialization"] + CLOSE_H
    rest = SETUP_H + gen_hours(tokens["e"] + tokens["x"], key) + teval_h + CLOSE_H
    return {"tokens_m": {k: round(v / 1e6, 2) for k, v in tokens.items()}, "tokens_total_m": round(sum(tokens.values()) / 1e6, 2),
            "e_tasks_needed": e_needed, "e_tasks_generated": e_generated, "t0_ht_s_sequences": round(s_t0),
            "hours": {k: round(v, 2) for k, v in hours.items()}, "hours_total": round(h, 2), "gpu": "1 x NVIDIA L40S",
            "usd": usd(h), "_tokens": sum(tokens.values()),
            "bounded_sessions": {"through_w00": {"hours": round(first, 2), "usd": usd(first)},
                                 "after_w00": {"hours": round(rest, 2), "usd": usd(rest)},
                                 "_": "the same work in two sessions: one more setup and closeout than hours_total"}}


def recovery(key: str, s: dict, teacher: dict) -> dict:
    hts = DESIGN["target_sequences"] + 0.5 * DESIGN["wave"] * (1.0 * teacher["t0_ht_s_sequences"] / DESIGN["t0_tasks"])
    hns = hts * s["hn_tokens"] / s["ht_tokens"] * 1.4      #: H-N documents carry relatively more prompt per target
    steps = {"H-T-S": math.ceil(3 * hts / 2), "H-N-S": math.ceil(3 * hns / 2), "H-N-C": math.ceil(3 * hts / 2)}
    step_s = m("isolated_seconds_per_step_a100") * a("l40s_step_vs_a100", key) * s["step"]
    train_h = DESIGN["seeds"] * sum(n * step_s / 3600 + UNIT_FIXED_H for n in steps.values())
    eval_unit_h = m("student_eval_seconds_l40s") / 3600 * s["student_eval"]
    eval_h = DESIGN["units"] * eval_unit_h
    h = SETUP_H + train_h + eval_h + CLOSE_H
    #: two GPUs: the six units scheduled longest first onto the less loaded GPU; the pod bills both GPUs throughout
    load = [0.0, 0.0]
    for u in sorted((n * step_s / 3600 + UNIT_FIXED_H + eval_unit_h for n in steps.values()
                     for _ in range(DESIGN["seeds"])), reverse=True):
        load[load.index(min(load))] += u
    wall2 = SETUP_H + max(load) + CLOSE_H
    return {"optimizer_steps": steps, "seconds_per_step": round(step_s, 3),
            "hours": {"train": round(train_h, 2), "evaluate": round(eval_h, 2)}, "hours_total": round(h, 2),
            "gpu": "NVIDIA L40S, 1 or 2 on one pod (one unit per GPU)", "usd": usd(h),
            "two_gpus": {"wall_hours": round(wall2, 2), "gpu_hours": round(2 * wall2, 2),
                         "idle_gpu_hours": round(max(load) - min(load), 2),
                         "usd": round(2 * wall2 * m("l40s_usd_per_hour") + wall2 * DISK_USD_PER_HOUR, 2),
                         "_": ("an ESTIMATE, not half the wall-clock at the same cost: both GPUs bill through setup, "
                               "the closeout and the shorter GPU's idle tail; the units' real lengths decide it")}}


def storage(teacher_tokens: float) -> dict:
    """The phase's peak bytes on the volume at 1 and 2 GPUs, and both ways to hold them."""
    evals = 2 * m("evaluation_bytes_per_copy_bound")
    held = {"generation": teacher_tokens * m("shard_bytes_per_generated_token") * m("generation_dir_over_shards"),
            "initialization": m("checkpoint_bytes"), "teacher_evaluation": 2 * evals, "check_evidence": 0.1 * GB}
    unit_done = m("checkpoint_bytes") + evals
    training = 2 * m("save_point_bytes") + evals          #: keep_last 1: two save points coexist for a moment
    base = sum(held.values())
    peaks = {"teacher_session": base, "recovery_1_gpu": base + 5 * unit_done + training,
             "recovery_2_gpus": base + 4 * unit_done + 2 * training}
    rate, months = m("volume_usd_per_gb_month"), STORAGE_MONTHS
    used, size = m("am5m373kuq_used_bytes"), m("am5m373kuq_size_gb")
    size_for = lambda b: int(math.ceil(b * STORAGE_MARGIN / GB / ROUND_GB) * ROUND_GB)  # noqa: E731
    options = {}
    for g in ("recovery_1_gpu", "recovery_2_gpus"):
        to = max(size, size_for(used + peaks[g]))
        new = size_for(peaks[g])
        options[g] = {
            "expand_am5m373kuq": {"to_gb": to, "added_gb": to - size, "phase_usd": round((to - size) * rate * months, 2),
                                  "monthly_usd_after": round(to * rate, 2)},
            "new_volume": {"size_gb": new, "phase_usd": round(new * rate * months, 2),
                           "monthly_usd_after": round(new * rate, 2)}}
    return {"held_gb": {k: round(v / GB, 2) for k, v in held.items()}, "peak_gb": {k: round(v / GB, 1) for k, v in peaks.items()},
            "am5m373kuq": {"size_gb": size, "used_gb": round(used / GB, 2), "free_gb": round(size - used / GB, 2),
                           "teacher_session_fits_now": peaks["teacher_session"] <= size * GB - used},
            "options": options, "months": months, "margin": STORAGE_MARGIN,
            "_": ("phase_usd is only what this phase adds: am5m373kuq's 30 GB through 2026-12-24 is already in the committed "
                  "position (its storage record's $4.20 reservation, $3.96 of it not yet posted) and is not counted again. "
                  "A volume only grows: an expansion's added GB bill as long as am5m373kuq exists; a new volume can be "
                  "released on its own")}


def forecast() -> dict:
    out = {"schema": "aadistill.budget.research_a_forecast/v2", "written_utc": "2026-10-29T15:40:00Z",
           "card": "NVIDIA L40S (every kind; d4_scale_design.json :: execution.hardware)",
           "measured": {k: {"value": v, "source": src} for k, (v, src) in MEASURED.items()}, "assumed": ASSUMED,
           "design": DESIGN, "scenarios": {}}
    for key, s in SCENARIOS.items():
        t = teacher_session(key, s)
        out["scenarios"][key] = {"multipliers": s, "check": check_session(key, s), "teacher_session": t,
                                 "recovery": recovery(key, s, t)}
    e, c = out["scenarios"]["expected"], out["scenarios"]["conservative"]
    out["storage"] = {"expected": storage(e["teacher_session"].pop("_tokens")),
                      "conservative": storage(c["teacher_session"].pop("_tokens"))}
    cap = lambda k: round(max(e[k]["usd"] * HARD_CAP_FACTOR, c[k]["usd"]), 2)  # noqa: E731
    st_e = out["storage"]["expected"]["options"]["recovery_1_gpu"]["expand_am5m373kuq"]
    st_c = out["storage"]["conservative"]["options"]["recovery_2_gpus"]["expand_am5m373kuq"]
    out["hard_caps_usd"] = {"check": cap("check"), "teacher_session": cap("teacher_session"), "recovery": cap("recovery"),
                            "storage": st_c["phase_usd"], "engineering_reserve": ENGINEERING_RESERVE,
                            "_storage": (f"expanding am5m373kuq to {st_c['to_gb']} GB (+{st_c['added_gb']} GB, two GPUs, "
                                         f"conservative) for {STORAGE_MONTHS} months; expected +{st_e['added_gb']} GB "
                                         f"(one GPU), ${st_e['phase_usd']}"),
                            "_reserve": "pre-science engineering subruns across the phase (P12.1), cumulative; no A100 in it"}
    hc = out["hard_caps_usd"]
    hc["phase_total"] = round(sum(hc[k] for k in ("check", "teacher_session", "recovery", "storage",
                                                  "engineering_reserve")), 2)
    out["expected_usd"] = round(sum(e[k]["usd"] for k in ("check", "teacher_session", "recovery")) + st_e["phase_usd"], 2)
    out["conservative_usd"] = round(sum(c[k]["usd"] for k in ("check", "teacher_session", "recovery")) + st_c["phase_usd"], 2)
    out["bounds"] = bounds(hc, out["scenarios"], {"expected": st_e["phase_usd"], "conservative": st_c["phase_usd"]})
    out["authorizes"] = "nothing"
    return out


def bounds(hc: dict, scen: dict, storage_usd: dict) -> dict:
    """Each envelope at the live price and at the ceiling: what its hard cap buys, the longest single session its
    artifact admits, and what each scenario costs AS RUN -- the teacher in its two bounded sessions, recovery (one
    GPU) in as many sessions as that bound needs, each extra session with its own setup and closeout."""
    live, ceil = m("l40s_usd_per_hour"), PRICE_CEILING
    prices = (("at_live", live), ("at_ceiling", ceil))

    def as_run(key: str, sc: dict, cap: float, price: float) -> float:
        if key == "teacher_session":
            h = sum(sc[key]["bounded_sessions"][k]["hours"] for k in ("through_w00", "after_w00"))
        else:
            h = sc[key]["hours_total"]
            if key == "recovery":
                h += (SETUP_H + CLOSE_H) * (math.ceil((h - SETUP_H - CLOSE_H) / longest_session_hours(cap, price)) - 1)
        return usd(h, price)
    out = {"live_usd_per_gpu_hour": live, "ceiling_usd_per_gpu_hour": ceil,
           "_": ("costs elsewhere in this forecast are at the live price; the hard caps are dollars and do not grow "
                 "with the price: at the ceiling they buy fewer hours")}
    for env, key in (("check", "check"), ("teacher", "teacher_session"), ("recovery", "recovery")):
        cap = hc[key]
        row = {"hard_cap_usd": cap,
               "gpu_hours_the_cap_buys": {at: round(cap / (pr + DISK_USD_PER_HOUR), 2) for at, pr in prices},
               "hours": {k: scen[k][key]["hours_total"] for k in SCENARIOS},
               "work_hours": {k: round(scen[k][key]["hours_total"] - SETUP_H - CLOSE_H, 2) for k in SCENARIOS},
               "longest_session_work_hours": {at: longest_session_hours(cap, pr) for at, pr in prices},
               "usd_as_run": {k: {at: as_run(key, scen[k], cap, pr) for at, pr in prices} for k in SCENARIOS}}
        row["fits_the_cap_as_run"] = {k: {at: v <= cap for at, v in row["usd_as_run"][k].items()} for k in SCENARIOS}
        if key == "teacher_session":
            row["bounded_sessions"] = {k: {s: scen[s][key]["bounded_sessions"][k]["hours"] for s in SCENARIOS}
                                       for k in ("through_w00", "after_w00")}
        if key == "recovery":
            row["two_gpus"] = {"wall_hours": {k: scen[k][key]["two_gpus"]["wall_hours"] for k in SCENARIOS},
                               "usd_at_live": {k: scen[k][key]["two_gpus"]["usd"] for k in SCENARIOS},
                               "longest_session_wall_hours": {at: longest_session_hours(cap, 2 * pr) for at, pr in prices},
                               "_": "estimates (recovery :: two_gpus)"}
        out[env] = row
    out["phase"] = {k: {at: round(sum(out[e]["usd_as_run"][k][at] for e in ("check", "teacher", "recovery"))
                                  + storage_usd[k], 2) for at, _ in prices} for k in SCENARIOS}
    out["phase"].update(hard_cap_usd=hc["phase_total"],
                        _=("the three sessions' kinds as run, plus storage; the conservative at the ceiling leaves of "
                           "the hard cap only what the engineering reserve does not use"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    doc = forecast()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"hard_caps_usd": doc["hard_caps_usd"], "expected_usd": doc["expected_usd"],
                      "conservative_usd": doc["conservative_usd"],
                      **{k: {x: (v[x]["usd"], v[x]["hours_total"]) for x in ("check", "teacher_session", "recovery")}
                         for k, v in doc["scenarios"].items()},
                      "storage": {k: {"peak_gb": v["peak_gb"], "options": v["options"]} for k, v in doc["storage"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

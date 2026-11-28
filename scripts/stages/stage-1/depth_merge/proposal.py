#!/usr/bin/env python3
"""DMR1's money and time ($0, CPU): variant 1's two proposals for the maintainer, variants 2 and 3's estimates.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/proposal.py \\
        --write | --check

* ``logs/stages/stage-1/depth_merge/plans/proposal_dmr1_merge_build.json`` and ``..._dmr1_merge_train.json``:
  variant 1's build session and its training session, each a proposal its own grant would bind (status
  ``proposed``); nothing is granted or issued.
* ``logs/stages/stage-1/depth_recurrence/plans/proposal_dmr1_loop_train.json`` (2026-11-28, the revised variant 2):
  the kind ``dmr1_loop_train``, one session per arm (depth_loop_kept with the engine validation first, then
  depth_loop_merged), one unit each; status ``proposed``, nothing granted. It replaces the 2026-11-26
  ``estimates.json`` (variants 2 and 3 x two seeds, never proposed; in git history).

Every hour figure is an estimate with its basis; ``bound`` is what a hard cap covers. The bases are measured, not
assumed: SOB1's depth session (`sob1_train_depth_only_20261122_223829`: 4.83 s / optimizer step, 805-806 min per
seed, 34.5-35.5 / 6.9-7.2 min of evaluation per seed and mode, $31.6179 all-in), RI-LM1's statistics pass and
assignments (117 s for 14 blocks; 178 s per 9,728 x 9,728 assignment on the pod's CPU) and SOB1's proposal bounds
(7.241 s / step, 3.0 h of evaluation per unit). A session's hard cap is SOB1's `proposal.hard_cap` (the runner's
pricing: setup, transfer, driver, contingency, closeout, artifact recovery, the watchdog, at the highest admitted price
plus the container disk).

**The recurrent arms' step time** is scaled on the work a step does, not assumed equal to the depth arm's: per token
the teacher runs 36 layer forwards and the student, under gradient checkpointing, about 4 forward-equivalents per
LOGICAL layer (forward, recomputation, backward 2x): 36 + 4 x 16 = 100 for the depth arm, 36 + 4 x 36 = 180 for the
loop (1.8x), applied to both the measured time and the bound; the hidden-state term adds its teacher hooks, one
relative-error pass per logical layer and its backward, estimated at 5%. Evaluation scales with the logical layers'
decode work (2.25x). Not measured on a GPU: the first four steps of the first unit measure the time and memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

SCHEMA = "aadistill.dmr1.proposal/v1"
GIB = 2 ** 30
#: the 2026-11-26 position this round was priced against (read live at 11:48Z; read again at issue and launch)
POSITION = {"read_utc": "2026-11-26T11:48:05Z", "committed_usd": 1086.2713, "project_cap_usd": 1161.06,
            "project_remaining_usd": 74.7887, "balance_usd": 123.7283, "pods": 0, "volumes": 0,
            "spend_per_hour": 0.0, "relay_headroom_gib": [35.24, 35.28], "dev_box_free_gib": 23}
#: the 2026-11-28 position the revised variant 2 is priced against (read live at 10:25Z; read again at issue and launch)
POSITION_V2 = {"read_utc": "2026-11-28T12:36:08Z", "committed_usd": 1114.6477, "project_cap_usd": 1161.06,
               "project_remaining_usd": 46.4123, "balance_usd": 95.3518, "pods": 0, "volumes": 0,
               "spend_per_hour": 0.0, "relay_headroom_gib": 54.219, "relay_used_gib": 33.359, "dev_box_free_gib": 6.85,
               "note": "one reading of the ledger (SE.live_position) for the cap and the balance; committed rose "
                       "$1.1346 since 10:25Z as V1 attempt 2's last billing rows posted (no resource existed)"}
#: the engine validation's hours (`depth_recurrence.engine_check`: two vLLM engines and two HF loads on a 2B model)
ENGINE_CHECK_HOURS = {"expected": 0.4, "bound": 1.0}
TEARDOWN_RESERVE = 1.00
#: measured bases (module docstring)
DEPTH = {"s_per_step": 4.83, "s_per_step_bound": 7.241, "steps": 9701, "eval_h_per_unit": 1.22,
         "eval_h_per_unit_bound": 3.0, "session_minutes_measured": 1708.3, "session_usd_measured": 31.6179}
LOOP_WORK = (36 + 4 * 36) / (36 + 4 * 16)          #: 1.8
LOOP_DECODE = 36 / 16                              #: 2.25
HIDDEN_OVERHEAD = 1.05


def sob1_proposal():
    from stages.single_operator_budget import proposal as SP
    return SP


def cost_model() -> dict:
    return json.loads((REPO / "configs/stages/stage-1/single_operator_budget/design.json").read_text())["cost_model"]


def transfers(sizes: dict, up: float = 120e6, down: float = 2.6e6, margin: float = 1.25) -> dict:
    """The floor-rate copies a training session owes (SOB1's): the init fetch, each final's verify, the last save
    point's verify; hours bound, and expected at the measured medians (~120 MB/s each way)."""
    h = lambda n, r: n / r / 3600 * margin  # noqa: E731
    return {"init_fetch": {"bytes": sizes["init_bf16"], "hours_bound": round(h(sizes["init_bf16"], down), 3),
                           "hours_expected": round(sizes["init_bf16"] / 120e6 / 3600, 3)},
            "final_copy_per_seed": {"bytes": sizes["final_fp32"],
                                    "hours_bound": round(h(sizes["final_fp32"], up) + h(sizes["final_fp32"], down), 3),
                                    "hours_expected": round(2 * sizes["final_fp32"] / 120e6 / 3600, 3)},
            "last_save_point_copy": {"bytes": sizes["save_point"],
                                     "hours_bound": round(h(sizes["save_point"], up) + h(sizes["save_point"], down), 3),
                                     "hours_expected": round(2 * sizes["save_point"] / 120e6 / 3600, 3)}}


def train_session(work: float, decode: float, overhead: float = 1.0, seeds: int = 2,
                  extra_hours: dict | None = None, sizes: dict | None = None) -> dict:
    """One arm's ``seeds`` units: training, evaluation and copies (plus ``extra_hours`` -- an engine validation --
    before them), expected and bound (driver hours), its cost and cap."""
    from stages.depth_merge import session as S
    SP = sob1_proposal()
    cm = cost_model()
    sizes = sizes or S.checkpoint_bytes()
    tr = transfers(sizes)
    sps = DEPTH["s_per_step"] * work * overhead
    spb = DEPTH["s_per_step_bound"] * work * overhead
    per_seed = {"seconds_per_step_expected": round(sps, 3), "seconds_per_step_bound": round(spb, 3),
                "hours_expected": round(sps * DEPTH["steps"] / 3600, 2),
                "hours_bound": round(spb * DEPTH["steps"] / 3600, 2)}
    ev = {"expected": round(DEPTH["eval_h_per_unit"] * decode, 2), "bound": round(DEPTH["eval_h_per_unit_bound"] * decode, 2)}
    copies_e = tr["init_fetch"]["hours_expected"] + seeds * (tr["final_copy_per_seed"]["hours_expected"]
                                                             + tr["last_save_point_copy"]["hours_expected"])
    copies_b = tr["init_fetch"]["hours_bound"] + seeds * (tr["final_copy_per_seed"]["hours_bound"]
                                                          + tr["last_save_point_copy"]["hours_bound"])
    x = extra_hours or {"expected": 0.0, "bound": 0.0}
    drv_e = seeds * (per_seed["hours_expected"] + ev["expected"]) + copies_e + x["expected"]
    drv_b = seeds * (per_seed["hours_bound"] + ev["bound"]) + copies_b + x["bound"]
    rate = SP.rate(cm)
    expected = round((drv_e + (25 + 5 + 10) / 60) * rate, 2)
    cap = SP.hard_cap(drv_b * 60, cm)
    return {"units": seeds, "training_per_seed": per_seed, "post_recovery_evaluation_hours_per_unit": ev,
            "copies": tr, **({"engine_validation_hours": x} if extra_hours else {}),
            "driver_hours_expected": round(drv_e, 2), "driver_hours_bound": round(drv_b, 2),
            "driver_minutes": round(drv_b * 60, 1), "unit_deadline_hours": round(per_seed["hours_bound"], 1),
            "expected_usd": expected, "hard_cap": cap, "per_session_hard_cap_usd": cap["usd"]}


def build_session() -> dict:
    """Variant 1's build: the session, the teacher, the base check, the statistics (29 blocks, 20 pairs), 20
    assignments, the merge, the checks, save / reload, the state evaluation, diagnostics, the relay copies."""
    SP = sob1_proposal()
    cm = cost_model()
    minutes = {"session_and_contract": 3, "teacher_load_and_base_check": 3,
               "statistics_pass": round(117 / 60 * 29 / 14, 1),
               "assignments_20_x_178s": round(20 * 178 / 60, 1), "merge_9_spans_cpu_float64": 8,
               "equivalence_checks": 1, "save_reload": 2, "state_evaluation": 10, "diagnostics": 3,
               "relay_init_and_statistics_expected": 2}
    expected = sum(minutes.values())
    bound = 2 * (expected - minutes["relay_init_and_statistics_expected"]) + 4.0e9 / 2.6e6 / 60 * 1.25 + 5
    return {"minutes_expected": minutes, "driver_minutes_expected": round(expected, 1),
            "driver_minutes": round(bound, 1), "expected_usd": round((expected + 25 + 5 + 10) / 60 * SP.rate(cm), 2),
            "hard_cap": SP.hard_cap(bound, cm),
            "per_session_hard_cap_usd": SP.hard_cap(bound, cm)["usd"],
            "basis": "RI-LM1's measured statistics pass (117 s for 14 blocks) and assignments (1,420 s for 8 on the "
                     "pod's CPU), scaled to 29 blocks and 20 pairs; the bound doubles every compute step and takes the "
                     "init's verify download at the 2.6 MB/s floor x 1.25"}


def proposal(kind: str, terms: dict, *, per_arm: str | None = None, design: Path | None = None,
             recorded: str = "2026-11-26", scope: str | None = None, position: dict | None = None) -> dict:
    cap = float(terms["per_session_hard_cap_usd"])
    block = {k: terms[k] for k in ("driver_minutes", "expected_usd", "per_session_hard_cap_usd")}
    if "unit_deadline_hours" in terms:
        block["unit_deadline_hours"] = terms["unit_deadline_hours"]
    return {"schema": SCHEMA, "study": "DMR1-M", "status": "proposed", "kind": kind,
            "recorded_utc": recorded,
            "design_sha256": __import__("hashlib").sha256((design or DC.DESIGN).read_bytes()).hexdigest(),
            "scope": scope or f"DMR1 variant 1, session kind {kind} only",
            "ceiling_usd": cap, "per_session_hard_cap_usd": cap, "teardown_reserve_usd": TEARDOWN_RESERVE,
            "kinds": {kind: ({"per_arm": {per_arm: block}} if per_arm else block)},
            "gpus": {"NVIDIA L40S": 1.11}, "gpu_count_max": 1,
            "hardware": "one NVIDIA L40S, secure cloud, at most $1.11/h, 150 GB container disk, no network volume",
            "usd_per_hour_all_in": round(sob1_proposal().rate(cost_model()), 4),
            "project_cumulative_cap_usd": (position or POSITION)["project_cap_usd"],
            "position_at_proposal": position or POSITION,
            "terms": terms, "money_rules": {
                "own_envelope": "its own grant; the envelope counts only this kind's runs; SOB1's closed grants' "
                                "remainders are void and fund nothing here",
                "balance": "the provider balance must cover the hard cap plus the reserve at issue and at launch; a "
                           "top-up is not a grant", "teardown_reserve": "$1.00 held outside the ceiling, never issued"},
            "authorizes": "nothing: a proposal; the grant is the maintainer's", "decision": "PENDING", "granted_by": "nobody"}


#: v1-new's step time and decode scale on its work (module docstring's rule): 36 + 4 x 18 against 36 + 4 x 16
SPLIT_WORK = (36 + 4 * 18) / (36 + 4 * 16)          #: 1.08
SPLIT_DECODE = 18 / 16                              #: 1.125


def split_build_session() -> dict:
    """v1-new's build (`build_driver --arm depth_merge_split`): the teacher and the base check, the statistics pass
    over the separate runs' 9 blocks, 7 assignments, 2 merges, the checks, V1's init fetched from the relay, save /
    reload, the state evaluation, diagnostics, the relay copies. Scaled from RI-LM1's and V1's measured bases; the
    bound doubles every compute step and takes both relay copies (V1's init in, the new init's verify) at the floor."""
    from stages.depth_merge import session as S
    SP = sob1_proposal()
    cm = cost_model()
    sz = S.checkpoint_bytes(arm="depth_merge_split")
    v1_init = int(S.checkpoint_bytes()["init_bf16"])
    minutes = {"session_and_contract": 3, "teacher_load_and_base_check": 3,
               "statistics_pass_9_blocks": round(117 / 60 * 9 / 14, 1),
               "assignments_7_x_178s": round(7 * 178 / 60, 1), "merge_2_spans_cpu_float64": round(8 * 9 / 29, 1),
               "equivalence_checks": 1, "save_reload": 2, "state_evaluation": 10, "diagnostics": 3,
               "v1_init_fetch_expected": round(v1_init / 120e6 / 60, 1),
               "relay_init_and_statistics_expected": round(2 * int(sz["init_bf16"]) / 120e6 / 60, 1)}
    expected = sum(minutes.values())
    compute = expected - minutes["v1_init_fetch_expected"] - minutes["relay_init_and_statistics_expected"]
    bound = (2 * compute + v1_init / 2.6e6 / 60 * 1.25 + int(sz["init_bf16"]) / 2.6e6 / 60 * 1.25
             + int(sz["init_bf16"]) / 120e6 / 60 * 1.25 + 5)
    return {"minutes_expected": minutes, "driver_minutes_expected": round(expected, 1),
            "driver_minutes": round(bound, 1), "expected_usd": round((expected + 25 + 5 + 10) / 60 * SP.rate(cm), 2),
            "hard_cap": SP.hard_cap(bound, cm), "per_session_hard_cap_usd": SP.hard_cap(bound, cm)["usd"],
            "basis": "RI-LM1's statistics pass (117 s for 14 blocks) and assignments (178 s each on the pod's CPU), "
                     "V1's merge (8 min for 29 blocks); the bound doubles each compute step and takes V1's init (in) "
                     "and the new init's verify (out) at the 2.6 MB/s floor x 1.25",
            "measured_v1_build": {"usd": 1.4043, "run": "dmr1_merge_build_20261127_083526"}}


def documents() -> dict[Path, dict]:
    b = build_session()
    t = train_session(1.0, 1.0)
    sob1 = json.loads((REPO / "logs/stages/stage-1/single_operator_budget/plans/proposal_phase2.json").read_text())
    depth_terms = sob1["kinds"]["sob1_train"]["per_arm"]["depth_only"]
    t["sob1_depth_proposal_terms"] = depth_terms
    t["measured_sob1_depth_session"] = {"minutes": DEPTH["session_minutes_measured"], "usd": DEPTH["session_usd_measured"]}
    total_v1 = b["per_session_hard_cap_usd"] + t["per_session_hard_cap_usd"] + 2 * TEARDOWN_RESERVE
    return {DC.LOGS / "plans/proposal_dmr1_merge_build.json": proposal("dmr1_merge_build", b),
            DC.LOGS / "plans/proposal_dmr1_merge_train.json": {**proposal("dmr1_merge_train", t, per_arm="depth_merge"),
                                                               "variant_1_total_with_reserves_usd": round(total_v1, 2)},
            DC.RECURRENCE_LOGS / "plans/proposal_dmr1_loop_train.json": proposal_v2(),
            **split_documents(),
            DC.LOGS / "plans/programme_budget.json": programme_budget()}


def programme_budget(order: tuple[str, ...] = ("v1new_build", "v1new_train", "v2d", "v2m")) -> dict:
    """The money of the three-unit programme in ``order``, from the four sessions' proposals and the 2026-11-28
    position: before each session, the project cap's remainder and the balance on the expected path (every earlier
    session at its expected cost) and on the worst path (every earlier one at its cap), against that session's cap
    plus its teardown reserve -- what its issue and its launch check. The balance, the project cap and the grants
    stay separate; nothing is raised, topped up or granted here."""
    from stages.depth_merge import session as S  # noqa: F401
    sb = split_documents()
    v2 = proposal_v2()["kinds"]["dmr1_loop_train"]["per_arm"]
    sess = {"v1new_build": sb[DC.LOGS / "plans/proposal_dmr1_split_build.json"]["kinds"]["dmr1_split_build"],
            "v1new_train": sb[DC.LOGS / "plans/proposal_dmr1_split_train.json"]["kinds"]["dmr1_split_train"]
            ["per_arm"]["depth_merge_split"],
            "v2d": v2["depth_loop_kept"], "v2m": v2["depth_loop_merged"]}
    pos = POSITION_V2
    rows, exp_spent, cap_spent = [], 0.0, 0.0
    for name in order:
        s = sess[name]
        need = round(float(s["per_session_hard_cap_usd"]) + TEARDOWN_RESERVE, 2)
        row = {"session": name, "expected_usd": s["expected_usd"], "cap_usd": s["per_session_hard_cap_usd"],
               "reserve_usd": TEARDOWN_RESERVE, "needs_usd": need}
        for path, spent in (("expected_path", exp_spent), ("worst_path", cap_spent)):
            rem = round(float(pos["project_remaining_usd"]) - spent, 2)
            bal = round(float(pos["balance_usd"]) - spent, 2)
            row[path] = {"project_remaining_usd": rem, "project_cap_fits": need <= rem,
                         "project_raise_needed_usd": round(max(0.0, need - rem), 2),
                         "balance_usd": bal, "balance_fits": need <= bal,
                         "top_up_needed_usd": round(max(0.0, need - bal), 2)}
        rows.append(row)
        exp_spent += float(s["expected_usd"])
        cap_spent += float(s["per_session_hard_cap_usd"])
    total_caps = round(sum(float(sess[n]["per_session_hard_cap_usd"]) for n in order) + TEARDOWN_RESERVE * len(order), 2)
    return {"schema": "aadistill.dmr1.programme_budget/v1", "recorded_utc": "2026-11-28", "order": list(order),
            "position": pos, "sessions": rows,
            "totals": {"expected_usd": round(sum(float(sess[n]["expected_usd"]) for n in order), 2),
                       "caps_with_reserves_usd": total_caps,
                       "v1new_caps_with_reserves_usd": round(sum(float(sess[n]["per_session_hard_cap_usd"])
                                                                 for n in ("v1new_build", "v1new_train"))
                                                             + 2 * TEARDOWN_RESERVE, 2),
                       "v2_caps_with_reserves_usd": round(sum(float(sess[n]["per_session_hard_cap_usd"])
                                                              for n in ("v2d", "v2m")) + 2 * TEARDOWN_RESERVE, 2),
                       "project_raise_to_fund_all_at_caps_usd": round(max(0.0, total_caps
                                                                          - float(pos["project_remaining_usd"])), 2)},
            "rule": "each session is checked alone at its issue and its launch against the live position; the "
                    "rows are the two bounding paths of what earlier sessions would have spent",
            "authorizes": "nothing: no cap raise, top-up or grant is implied"}


def split_documents() -> dict[Path, dict]:
    """v1-new's two kinds: its build and its one-seed training session, each its own proposal (V1's grants are not
    reused), priced against the same 2026-11-28 position as variant 2's."""
    from stages.depth_merge import session as S
    sb = split_build_session()
    st = train_session(SPLIT_WORK, SPLIT_DECODE, seeds=1, sizes=S.checkpoint_bytes(arm="depth_merge_split"))
    st["basis"] = {"step_time": f"V1's measured 4.87 s / step (SOB1's 4.83) and the 7.241 s bound x {SPLIT_WORK:.2f} "
                                "(36 + 4 x 18 against 36 + 4 x 16 forward-equivalents): NOT measured",
                   "evaluation": f"SOB1's 1.22 h / 3.0 h per unit x {SPLIT_DECODE:.3f} (18 / 16 layers of decode)",
                   "sizes": "split.checkpoint_bytes (the 18-layer config, SOB1's 2P / 4P / 4P + 8T rule)"}
    kw = dict(design=DC.SPLIT_DESIGN, recorded="2026-11-28", position=POSITION_V2)
    return {DC.LOGS / "plans/proposal_dmr1_split_build.json":
            proposal("dmr1_split_build", sb, scope="DMR1 v1-new (depth_merge_split), its build session only", **kw),
            DC.LOGS / "plans/proposal_dmr1_split_train.json":
            proposal("dmr1_split_train", st, per_arm="depth_merge_split",
                     scope="DMR1 v1-new (depth_merge_split), its one-seed training session only", **kw)}


def proposal_v2() -> dict:
    """The revised variant 2's kind (module docstring): one session per arm, one unit each, the engine validation in
    the first; the money it needs against the 2026-11-28 position, and what that position does not cover."""
    import hashlib
    kept = train_session(LOOP_WORK, LOOP_DECODE, seeds=1, extra_hours=ENGINE_CHECK_HOURS)
    merged = train_session(LOOP_WORK, LOOP_DECODE, seeds=1)
    arms = {"depth_loop_kept": kept, "depth_loop_merged": merged}
    blocks = {a: {k: v[k] for k in ("driver_minutes", "expected_usd", "per_session_hard_cap_usd", "unit_deadline_hours")}
              for a, v in arms.items()}
    caps = sum(v["per_session_hard_cap_usd"] for v in arms.values())
    ceiling = round(caps, 2)
    need = round(ceiling + 2 * TEARDOWN_RESERVE, 2)
    expected = round(sum(v["expected_usd"] for v in arms.values()), 2)
    pos = POSITION_V2
    return {"schema": SCHEMA, "study": "DMR1-R", "status": "proposed", "kind": "dmr1_loop_train",
            "recorded_utc": "2026-11-28",
            "design_sha256": hashlib.sha256(DC.RECURRENCE_DESIGN.read_bytes()).hexdigest(),
            "scope": "DMR1 variant 2 as revised 2026-11-28: depth_loop_kept and depth_loop_merged, seed 20260726 each, "
                     "one session per arm; the first runs the engine validation before training",
            "ceiling_usd": ceiling, "teardown_reserve_usd": TEARDOWN_RESERVE,
            "per_session_hard_cap_usd": max(v["per_session_hard_cap_usd"] for v in arms.values()),
            "per_session_rule": "each session's own cap is its arm's (kinds.dmr1_loop_train.per_arm); this field is "
                                "the larger, which no single session may exceed; the ceiling is both sessions'",
            "kinds": {"dmr1_loop_train": {"per_arm": blocks}},
            "gpus": {"NVIDIA L40S": 1.11}, "gpu_count_max": 1,
            "hardware": "one NVIDIA L40S, secure cloud, at most $1.11/h, 150 GB container disk, no network volume",
            "usd_per_hour_all_in": round(sob1_proposal().rate(cost_model()), 4),
            "terms": arms, "expected_usd_total": expected, "hard_caps_total_usd": ceiling,
            "needed_with_reserves_usd": need,
            "project_cumulative_cap_usd": pos["project_cap_usd"], "position_at_proposal": pos,
            "fits_position": {
                "read_utc": pos["read_utc"],
                "both_sessions_at_their_caps_with_reserves_usd": need,
                "project_cap": {"remaining_usd": pos["project_remaining_usd"], "needed_usd": need,
                                "fits": need <= pos["project_remaining_usd"],
                                "raise_needed_usd": round(max(0.0, need - pos["project_remaining_usd"]), 2),
                                "raise_needed_for_v2d_alone_usd": round(max(0.0, kept["per_session_hard_cap_usd"]
                                                                            + TEARDOWN_RESERVE
                                                                            - pos["project_remaining_usd"]), 2)},
                "balance": {"balance_usd": pos["balance_usd"], "needed_usd": need,
                            "fits": need <= pos["balance_usd"],
                            "top_up_needed_usd": round(max(0.0, need - pos["balance_usd"]), 2),
                            "first_session_needs_usd": round(kept["per_session_hard_cap_usd"] + TEARDOWN_RESERVE, 2),
                            "fits_first_session": kept["per_session_hard_cap_usd"] + TEARDOWN_RESERVE <= pos["balance_usd"],
                            "rule": "the worst case of both sessions at their caps; each issue and launch also checks "
                                    "its own session's cap plus the reserve against the live balance"},
                "not_authorized": "no cap raise, top-up or grant is implied; V1's closed grants fund nothing here"},
            "basis": {"step_time": f"SOB1's measured 4.83 s / step (V1 measured 4.87) and the 7.241 s bound, x {LOOP_WORK:.2f} "
                                   "(teacher 36 + 4 x 36 student logical-layer forward-equivalents vs 36 + 4 x 16): "
                                   "NOT measured -- the first four steps measure it",
                      "evaluation": f"SOB1's 1.22 h expected / 3.0 h bound per unit, x {LOOP_DECODE:.2f} (36 / 16 "
                                    "logical layers of decode work and KV cache)",
                      "engine_validation": ENGINE_CHECK_HOURS,
                      "not_the_same_compute": "same data and optimizer steps as the single-execution controls, about "
                                              f"{LOOP_WORK:.1f}x the training work per token and {LOOP_DECODE:.2f}x the "
                                              "decode work: reported as measured GPU time, never as matched compute"},
            "memory_gib": memory_estimates(),
            "money_rules": {
                "own_envelope": "its own grant; variant 1's grants (dmr1_merge_build, dmr1_merge_train) are not "
                                "carried over and fund nothing here; SOB1's closed grants' remainders are void",
                "cumulative": "the two sessions share the kind's envelope; a retry's cap is what the envelope leaves "
                              "after every earlier attempt's closeout, minus the reserve",
                "teardown_reserve": "$1.00 per session, held outside the ceiling, never issued"},
            "authorizes": "nothing: a proposal; the grant is the maintainer's", "decision": "PENDING",
            "granted_by": "nobody"}


def S_bytes(k: str) -> int:
    from stages.depth_merge import session as S
    return int(S.checkpoint_bytes()[k])


def memory_estimates() -> dict:
    """L40S memory of the loop arms from SOB1's MEASURED depth peaks (39.39 GiB allocated, 40.98 GiB reserved), plus
    what the schedule adds: 20 more checkpointed block inputs ([8,192, 2,560] fp32 each), and the
    eval-only state probe's teacher rows at validation (36 layers x about 900 rows, bf16)."""
    inputs = 20 * 8192 * 2560 * 4 / GIB
    base_a, base_r = 39.39, 40.98
    return {"device_gib": 44.39, "rule": "SOB1's measured depth peak plus the added tensors (not measured)",
            "depth_loop_kept_and_merged": {"peak_allocated_gib": round(base_a + inputs, 2),
                                           "peak_reserved_gib": round(base_r + inputs, 2),
                                           "margin_to_device_gib": round(44.39 - (base_r + inputs), 2)},
            "state_probe_at_validation": round(36 * 2 * 3600 / 8 * 2560 * 2 / GIB, 3),
            "added": {"checkpointed_inputs_gib": round(inputs, 2)},
            "risk": "the loop arms' estimated reserved peak leaves about 1.9 GiB of the L40S: real. The first four "
                    "steps' memory_probe measures it; an out-of-memory error is answered first by the recipe's "
                    "protocol-neutral steps (expandable segments, kd_chunk), then -- only after a measured OOM and "
                    "under a decision of its own -- an 80 GB card"}


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, default=str) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    docs = documents()
    if a.check:
        bad = [str(p.relative_to(REPO)) for p, d in docs.items() if not p.is_file() or p.read_text() != render(d)]
        print("the proposals and estimates are the derivation" if not bad else f"REFUSED: {bad}")
        return 0 if not bad else 1
    for p, d in docs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render(d))
        print(p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

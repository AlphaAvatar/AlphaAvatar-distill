#!/usr/bin/env python3
"""DMR1's money and time ($0, CPU): variant 1's two proposals for the maintainer, variants 2 and 3's estimates.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/proposal.py \\
        --write | --check

* ``logs/stages/stage-1/depth_merge/plans/proposal_dmr1_merge_build.json`` and ``..._dmr1_merge_train.json``:
  variant 1's build session and its training session, each a proposal its own grant would bind (status
  ``proposed``); nothing is granted or issued.
* ``logs/stages/stage-6/depth_recurrence/plans/estimates.json``: variants 2 and 3, estimated on the real 36-logical-
  layer execution, NOT proposed this round (no paid authorization of either).

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


def train_session(work: float, decode: float, overhead: float = 1.0) -> dict:
    """One arm's two seeds: training, evaluation and copies, expected and bound (driver hours), its cost and cap."""
    from stages.depth_merge import session as S
    SP = sob1_proposal()
    cm = cost_model()
    sizes = S.checkpoint_bytes()
    tr = transfers(sizes)
    sps = DEPTH["s_per_step"] * work * overhead
    spb = DEPTH["s_per_step_bound"] * work * overhead
    per_seed = {"seconds_per_step_expected": round(sps, 3), "seconds_per_step_bound": round(spb, 3),
                "hours_expected": round(sps * DEPTH["steps"] / 3600, 2),
                "hours_bound": round(spb * DEPTH["steps"] / 3600, 2)}
    ev = {"expected": round(DEPTH["eval_h_per_unit"] * decode, 2), "bound": round(DEPTH["eval_h_per_unit_bound"] * decode, 2)}
    copies_e = tr["init_fetch"]["hours_expected"] + 2 * (tr["final_copy_per_seed"]["hours_expected"]
                                                         + tr["last_save_point_copy"]["hours_expected"])
    copies_b = tr["init_fetch"]["hours_bound"] + 2 * (tr["final_copy_per_seed"]["hours_bound"]
                                                      + tr["last_save_point_copy"]["hours_bound"])
    drv_e = 2 * (per_seed["hours_expected"] + ev["expected"]) + copies_e
    drv_b = 2 * (per_seed["hours_bound"] + ev["bound"]) + copies_b
    rate = SP.rate(cm)
    expected = round((drv_e + (25 + 5 + 10) / 60) * rate, 2)
    cap = SP.hard_cap(drv_b * 60, cm)
    return {"units": 2, "training_per_seed": per_seed, "post_recovery_evaluation_hours_per_unit": ev,
            "copies": tr, "driver_hours_expected": round(drv_e, 2), "driver_hours_bound": round(drv_b, 2),
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


def proposal(kind: str, terms: dict, *, per_arm: str | None = None) -> dict:
    cap = float(terms["per_session_hard_cap_usd"])
    block = {k: terms[k] for k in ("driver_minutes", "expected_usd", "per_session_hard_cap_usd")}
    if "unit_deadline_hours" in terms:
        block["unit_deadline_hours"] = terms["unit_deadline_hours"]
    return {"schema": SCHEMA, "study": "DMR1-M", "status": "proposed", "kind": kind,
            "recorded_utc": "2026-11-26", "design_sha256": __import__("hashlib").sha256(DC.DESIGN.read_bytes()).hexdigest(),
            "scope": f"DMR1 variant 1, session kind {kind} only",
            "ceiling_usd": cap, "per_session_hard_cap_usd": cap, "teardown_reserve_usd": TEARDOWN_RESERVE,
            "kinds": {kind: ({"per_arm": {per_arm: block}} if per_arm else block)},
            "gpus": {"NVIDIA L40S": 1.11}, "gpu_count_max": 1,
            "hardware": "one NVIDIA L40S, secure cloud, at most $1.11/h, 150 GB container disk, no network volume",
            "usd_per_hour_all_in": round(sob1_proposal().rate(cost_model()), 4),
            "project_cumulative_cap_usd": POSITION["project_cap_usd"], "position_at_proposal": POSITION,
            "terms": terms, "money_rules": {
                "own_envelope": "its own grant; the envelope counts only this kind's runs; SOB1's closed grants' "
                                "remainders are void and fund nothing here",
                "balance": "the provider balance must cover the hard cap plus the reserve at issue and at launch; a "
                           "top-up is not a grant", "teardown_reserve": "$1.00 held outside the ceiling, never issued"},
            "authorizes": "nothing: a proposal; the grant is the maintainer's", "decision": "PENDING", "granted_by": "nobody"}


def documents() -> dict[Path, dict]:
    b = build_session()
    t = train_session(1.0, 1.0)
    sob1 = json.loads((REPO / "logs/stages/stage-1/single_operator_budget/plans/proposal_phase2.json").read_text())
    depth_terms = sob1["kinds"]["sob1_train"]["per_arm"]["depth_only"]
    t["sob1_depth_proposal_terms"] = depth_terms
    t["measured_sob1_depth_session"] = {"minutes": DEPTH["session_minutes_measured"], "usd": DEPTH["session_usd_measured"]}
    loop = train_session(LOOP_WORK, LOOP_DECODE)
    hidden = train_session(LOOP_WORK, LOOP_DECODE, HIDDEN_OVERHEAD)
    total_v1 = b["per_session_hard_cap_usd"] + t["per_session_hard_cap_usd"] + 2 * TEARDOWN_RESERVE
    est = {"schema": SCHEMA.replace("proposal", "estimates"), "status": "estimate, NOT proposed this round",
           "recorded_utc": "2026-11-26", "position": POSITION,
           "rule": "work per token 1.8x the depth arm's (teacher 36 + 4 x 36 student logical layers vs 36 + 4 x 16), "
                   "decode 2.25x; the hidden-state term 5% on top (proposal.py docstring); not measured",
           "arms": {"depth_loop": loop, "depth_loop_hidden": hidden},
           "memory_gib": memory_estimates(),
           "relay": {"rotation_peak_gib": round((2 * S_bytes("save_point") + S_bytes("final_fp32")) / GIB, 2),
                     "headroom_read_gib": POSITION["relay_headroom_gib"],
                     "finals_per_arm_gib": round(2 * S_bytes("final_fp32") / GIB, 2)},
           "needs": {"project_cap": "above the $1,161.06 cap after variant 1: each loop session's cap plus its $1.00 "
                                    "reserve, cumulatively", "grant": "a grant per kind",
                     "engine_validation": "the recurrent vLLM model's agreement with the HF reference on a GPU first "
                                          "(REVIEW section 8)"}}
    return {DC.LOGS / "plans/proposal_dmr1_merge_build.json": proposal("dmr1_merge_build", b),
            DC.LOGS / "plans/proposal_dmr1_merge_train.json": {**proposal("dmr1_merge_train", t, per_arm="depth_merge"),
                                                               "variant_1_total_with_reserves_usd": round(total_v1, 2)},
            DC.STAGE6_LOGS / "plans/estimates.json": est}


def S_bytes(k: str) -> int:
    from stages.depth_merge import session as S
    return int(S.checkpoint_bytes()[k])


def memory_estimates() -> dict:
    """L40S memory of the loop arms from SOB1's MEASURED depth peaks (39.39 GiB allocated, 40.98 GiB reserved), plus
    what the schedule adds: 20 more checkpointed block inputs ([8,192, 2,560] fp32 each), and for the hidden arm the
    teacher's bf16 targets at the KD rows (36 layers) and the backward's one [8,192, 2,560] gradient buffer."""
    inputs = 20 * 8192 * 2560 * 4 / GIB
    targets_median = 36 * 1790 * 2560 * 2 / GIB
    targets_full = 36 * 8191 * 2560 * 2 / GIB
    grad_buf = 8192 * 2560 * 4 / GIB
    base_a, base_r = 39.39, 40.98
    return {"device_gib": 44.39, "rule": "SOB1's measured depth peak plus the added tensors (not measured)",
            "depth_loop": {"peak_allocated_gib": round(base_a + inputs, 2), "peak_reserved_gib": round(base_r + inputs, 2)},
            "depth_loop_hidden": {"peak_allocated_gib_median_rows": round(base_a + inputs + targets_median + grad_buf, 2),
                                  "peak_allocated_gib_full_rows": round(base_a + inputs + targets_full + grad_buf, 2),
                                  "peak_reserved_gib_full_rows": round(base_r + inputs + targets_full + grad_buf, 2)},
            "added": {"checkpointed_inputs_gib": round(inputs, 2), "teacher_targets_gib_median_rows": round(targets_median, 2),
                      "teacher_targets_gib_full_rows": round(targets_full, 2), "gradient_buffer_gib": round(grad_buf, 3)},
            "risk": "depth_loop_hidden at full-row sequences is within ~1 GiB of the device: the recipe's OOM steps "
                    "(expandable segments, kd_chunk, AdamW without multi-tensor temporaries) first, an 80 GB card "
                    "only after a measured OOM"}


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

#!/usr/bin/env python3
"""The new D3's price: the structural session and the conditional probes.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/d3_budget.py [--write] [--live]

Two separately authorized sessions, priced from MEASURED records rather than
retyped figures:

* STRUCTURAL -- q2's path from the teacher with DEPTH under the teacher
  reference. Its operators are q2's, so their cost is q2's own measurements:
  the digest-pinned replay of q2 (d1_replay_002: FFN, DEPTH cached, WIDTH,
  ATTENTION) for the expected cell, and the completed search's DEPTH at the
  same position in RECOMPUTE mode (its worst reference cache) for the bound.
  The treatment adds the teacher reference store (built once from the teacher,
  in candidate-equivalents) and the parent-reference diagnostic for every
  candidate (at most one more reduction per candidate: the search's measured
  distortion seconds). Setup and collection are the D2 search session's own
  timeline for the expected cell and its priced session overhead for the bound.
* PROBES (conditional on STRUCTURAL_CHANGE) -- three recovery probes from the
  new initial checkpoint on the shared confirmation rung: the same session as
  D2's shared confirmation (one relay-staged arm, three probes, the same
  protocol), so it takes that step's price from `d2_budget.py` unchanged, and
  reports the confirmation's measured cost beside it.

The old D3 search allocation and its $50.95 hard envelope are not used, and
the old D3/fresh-confirmation reservation is NOT transferred. Every figure is
re-derivable; `--live` re-reads the four limits and the provider's price.
Authorizes nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d3.budget/v1"
RECORD = "logs/stages/stage-1/phase_d3/plans/d3_budget_v1.json"
REPLAY_CLOSEOUT = ("logs/stages/stage-1/phase_d1_replay/runs/d1_replay_002/"
                   "closeout/outcome.json")
D2_SEARCH_SESSION = ("logs/stages/stage-1/phase_d2/runs/d2_search_20261015_"
                     "051722/runtime/session.json")
D2_CONFIRMATION_CLOSEOUT = ("logs/stages/stage-1/phase_d2/runs/d2_confirmation_"
                            "20261016_105802/closeout/outcome.json")
#: A structural session's disk: the 8 GB teacher, four step checkpoints at
#: most ~12 GB together, the training environment and the bundle. The
#: behavioural rungs' 120 GB holds it with room; nothing here needs the
#: search's 400 GB beam store.
CONTAINER_DISK_GB = 120
#: The teacher reference store, in DEPTH candidate-equivalents: one teacher
#: forward over the mixture (expected 2, bound 5 -- `d2_budget`'s own
#: root-store figures, same store, same mixture size).
ROOT_STORE_CANDIDATES = {"expected": 2.0, "bound": 5.0}
#: The every-round parent diagnostic: one extra KL reduction per candidate,
#: from cached parent sketches. Expected half the cached DEPTH again; the
#: bound adds the search's whole measured distortion time.
PARENT_DIAGNOSTIC_FRACTION_EXPECTED = 0.5
#: Host-to-host variability on the bound (the cached replay and the search
#: differed 2.2x on one operator under different caches; this covers a slow
#: host on top of the worst cache).
HOST_BOUND_FACTOR = 1.25
#: Teacher load plus the staged-shard verification (8 GB hashed), minutes.
TEACHER_MINUTES = {"expected": 2.0, "bound": 6.0}
#: Each REFUSED host draw costs one startup before host admission refuses it.
REFUSED_DRAWS_RESERVED = 4
DRAW_STARTUP_MINUTES = 2.0


class D3BudgetError(RuntimeError):
    pass


_D2_BUDGET: list[dict[str, Any]] = []


def _d2_budget() -> dict[str, Any]:
    """`d2_budget.build()`, once per process: it re-verifies D1's telemetry
    and the GPU-validation evidence on every call, and every figure read
    from it here is a function of committed records."""
    from stages.d_series import d2_budget as BUD

    if not _D2_BUDGET:
        _D2_BUDGET.append(BUD.build())
    return _D2_BUDGET[0]


def _read(rel: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / rel).read_text())


def replay_step_seconds() -> dict[str, float]:
    """q2's four operators as the digest-pinned replay measured them."""
    doc = _read(REPLAY_CLOSEOUT)
    q2 = doc["result"]["q2"]
    from stages.phase_d3 import d3_design as D

    if q2["state_id"] != D.CONTROL_STATE_ID or not q2["adopted"]:
        raise D3BudgetError("the replay closeout's q2 is not the adopted control")
    out = {}
    for row in q2["steps"]:
        impl, seconds = row.rsplit(" ", 1)
        out[impl] = float(seconds)
    return out


def search_depth_seconds() -> dict[str, float]:
    """The completed search's DEPTH at q2's position: wall and reductions."""
    from stages.d_series import d2_budget as BUD
    from stages.phase_d3 import d3_design as D

    control = _read(D.CONTROL_RECORD)
    depth_state = control["steps"][control["depth"]["step"]]["state_id"]
    for row in BUD.d1_search_telemetry():
        if row.get("state_id") == depth_state and row.get("event", "expansion") \
                == "expansion":
            timing = row.get("operator_timing") or {}
            return {"operator_seconds": float(row["operator_seconds"]),
                    "distortion_seconds": float(timing["distortion_seconds"]),
                    "candidate_subsets": int(timing["candidate_subsets"]),
                    "reference_mode": "recomputed"
                    if (row.get("reference_counters") or {}).get(
                        "reference_recomputes") else "cached"}
    raise D3BudgetError(f"no telemetry row for the control DEPTH {depth_state}")


def session_overhead_minutes() -> dict[str, Any]:
    """Setup and collection: the D2 search's timeline (expected), its priced
    session overhead (bound)."""
    from stages.d_series import d2_budget as BUD

    tl = _read(D2_SEARCH_SESSION)["timeline"]

    def at(pattern: str) -> float:
        for e in tl:
            if re.search(pattern, e["msg"]):
                return float(e["elapsed_min"])
        raise D3BudgetError(f"the D2 search timeline has no {pattern!r}")

    driver_start = at(r"MARKER:DRIVER_START")
    all_done = at(r"MARKER:ALL_DONE")
    deleted = at(r"pod deleted")
    n_products = 4
    post = deleted - all_done
    budget = _d2_budget()
    bound = float(budget["steps"]["d2_search"]["priced_by"][
        "session_overhead_minutes"])
    return {"setup_to_driver_minutes": round(driver_start, 2),
            "collection_minutes_measured": round(post, 2),
            "products_in_that_collection": n_products,
            "expected_minutes": round(driver_start + post, 2),
            "bound_minutes": bound,
            "_basis": (f"{D2_SEARCH_SESSION} timeline (DRIVER_START, ALL_DONE, "
                       "pod deleted); bound: d2_budget steps.d2_search."
                       "priced_by.session_overhead_minutes")}


def _usd(minutes: float, rate: float, disk_gb: float) -> dict[str, float]:
    from stages.d_series import d2_budget as BUD

    gpu = minutes / 60.0 * rate
    disk = minutes / (30 * 24 * 60) * disk_gb * BUD.DISK_USD_PER_GB_MONTH
    return {"minutes": round(minutes, 2), "gpu_usd": round(gpu, 4),
            "disk_usd": round(disk, 4), "all_in_usd": round(gpu + disk, 4)}


def structural(rate: float) -> dict[str, Any]:
    replay = replay_step_seconds()
    search = search_depth_seconds()
    over = session_overhead_minutes()
    depth_impl = "depth.causal_kl_greedy_v1"
    cached = replay[depth_impl]
    per_candidate = cached / search["candidate_subsets"]
    others = sum(v for k, v in replay.items() if k != depth_impl)
    depth_expected = (cached * (1 + PARENT_DIAGNOSTIC_FRACTION_EXPECTED)
                      + ROOT_STORE_CANDIDATES["expected"] * per_candidate)
    worst = search["operator_seconds"]
    depth_bound = HOST_BOUND_FACTOR * (
        worst + search["distortion_seconds"]
        + ROOT_STORE_CANDIDATES["bound"] * worst / search["candidate_subsets"])
    expected_min = (over["expected_minutes"] + TEACHER_MINUTES["expected"]
                    + (depth_expected + others) / 60.0)
    hard_min = (over["bound_minutes"] + TEACHER_MINUTES["bound"]
                + (depth_bound + HOST_BOUND_FACTOR * 2 * others) / 60.0
                + REFUSED_DRAWS_RESERVED * DRAW_STARTUP_MINUTES)
    return {
        "measured": {"replay_step_seconds": replay,
                     "search_depth": search,
                     "session_overhead": over},
        "model": {
            "depth_expected_seconds": round(depth_expected, 1),
            "depth_bound_seconds": round(depth_bound, 1),
            "other_operators_expected_seconds": round(others, 1),
            "other_operators_bound_seconds": round(
                HOST_BOUND_FACTOR * 2 * others, 1),
            "root_store_candidate_equivalents": ROOT_STORE_CANDIDATES,
            "parent_diagnostic_fraction_expected":
                PARENT_DIAGNOSTIC_FRACTION_EXPECTED,
            "parent_diagnostic_bound": ("the search's measured distortion "
                                        "seconds, added whole"),
            "host_bound_factor": HOST_BOUND_FACTOR,
            "teacher_minutes": TEACHER_MINUTES,
            "refused_draws_reserved": REFUSED_DRAWS_RESERVED,
            "draw_startup_minutes": DRAW_STARTUP_MINUTES},
        "container_disk_gb": CONTAINER_DISK_GB,
        "session_overhead_minutes": over["bound_minutes"],
        "expected": _usd(expected_min, rate, CONTAINER_DISK_GB),
        "hard": _usd(hard_min, rate, CONTAINER_DISK_GB),
    }


def probes(rate: float) -> dict[str, Any]:
    """The D2 shared confirmation's session, priced by its own owner."""
    from stages.d_series import d2_budget as BUD

    budget = _d2_budget()
    if float(budget["rate_usd_per_hour_basis"]) != rate:
        raise D3BudgetError("the D2 step is priced at another rate")
    step = budget["steps"]["d2_confirmation"]
    measured = _read(D2_CONFIRMATION_CLOSEOUT)["money"]
    return {"session_shape": ("three probes, one relay-staged arm, the shared "
                              "confirmation rung -- the D2 shared "
                              "confirmation's session"),
            "n_probes": int(step["n_probes"]),
            "arm_staging": step["arm_staging"],
            "rescoring": step["rescoring"],
            "expected": step["expected"], "hard": step["hard"],
            "measured_same_shape_all_in_usd": float(measured["all_in_usd"]),
            "_basis": f"{BUD.RECORD} :: steps.d2_confirmation"}


def live_limits() -> dict[str, float]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "derive_budget", REPO_ROOT / "scripts/maintenance/consolidation/"
                                     "derive_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    live = mod.derive(REPO_ROOT)
    return {"project_remaining_usd": float(live["project"]["remaining_usd"]),
            "project_cap_usd": float(live["project"]["cap_usd"]),
            "formal_remaining_usd": float(live["formal"]["remaining_usd"]),
            "package_remaining_usd": float(live["package"]["remaining_usd"]),
            "engineering_remaining_usd": float(
                live["engineering"]["remaining_usd"]),
            "per_session_cap_usd": float(live["per_session_ceiling_usd"])}


def build(*, limits: dict[str, float] | None = None,
          quote: dict[str, Any] | None = None) -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD

    rate = float(_d2_budget()["rate_usd_per_hour_basis"])
    s, p = structural(rate), probes(rate)
    reserve = float(BUD.TEARDOWN_RESERVE_USD)
    s_need = round(s["hard"]["all_in_usd"] + reserve, 2)
    p_need = round(p["hard"]["all_in_usd"] + reserve, 2)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "_contract": ("The new D3's price, from measured records; two "
                      "separately authorized sessions. Re-derive before any "
                      "authorization. AUTHORIZES NOTHING."),
        "rate_usd_per_hour_basis": rate,
        "teardown_reserve_usd": reserve,
        "steps": {"d3_structural": s, "d3_probes": p},
        "totals": {
            "structural_expected_usd": s["expected"]["all_in_usd"],
            "structural_hard_with_reserve_usd": s_need,
            "probes_expected_usd": p["expected"]["all_in_usd"],
            "probes_hard_with_reserve_usd": p_need,
            "both_expected_usd": round(s["expected"]["all_in_usd"]
                                       + p["expected"]["all_in_usd"], 2),
            "both_hard_with_reserves_usd": round(s_need + p_need, 2)},
        "old_d3_reservation": (
            "NOT TRANSFERRED. The cancelled student-confidence D3 was planned "
            "at a search plus its own screening and confirmation ($50.95 hard) "
            "and a fresh confirmation beside it; neither exists in this design "
            "and neither reservation funds it"),
        "authorizes": "nothing",
    }
    if limits is not None:
        both = doc["totals"]["both_hard_with_reserves_usd"]
        doc["live"] = {
            "limits": limits, "quote": quote,
            "fits": {
                "structural_per_session_cap": s["hard"]["all_in_usd"]
                <= limits["per_session_cap_usd"],
                "probes_per_session_cap": p["hard"]["all_in_usd"]
                <= limits["per_session_cap_usd"],
                "both_project_at_hard": both <= limits["project_remaining_usd"],
                "both_formal_at_hard": both <= limits["formal_remaining_usd"],
                "both_package_at_hard": both <= limits["package_remaining_usd"]},
            "project_remaining_after_both_at_hard_usd": round(
                limits["project_remaining_usd"] - both, 2)}
        if quote and float(quote.get("usd_per_hour", rate)) != rate:
            doc["live"]["note"] = ("the live quote differs from the basis "
                                   "rate; the issuer re-prices the accepted "
                                   "minutes at the live rate")
    return doc


PROPOSAL = "logs/stages/stage-1/phase_d3/plans/d3_grant_proposal.json"
#: Each envelope holds one complete session at its hard ceiling plus its
#: teardown reserve, and a second (a corrected pre-science attempt, P12.1),
#: rounded up to the dollar. The probes' is D2's confirmation grant for the
#: same session shape.
REPLACEMENT_ATTEMPTS = 1


def proposal(doc: dict[str, Any]) -> dict[str, Any]:
    """The PROPOSED grants, derived from the priced record. Not a grant."""
    import math

    from stages.phase_d3 import d3_design as D

    reg = D.registration(REPO_ROOT)
    totals = doc["totals"]
    s_env = float(math.ceil(totals["structural_hard_with_reserve_usd"]
                            * (1 + REPLACEMENT_ATTEMPTS)))
    p_env = float(math.ceil(totals["probes_hard_with_reserve_usd"]
                            * (1 + REPLACEMENT_ATTEMPTS) / 1.5))
    return {
        "schema": "aadistill.phase_d3.grant_proposal/v1",
        "status": ("PROPOSED -- the maintainer's to decide; nothing is "
                   "issued, nothing is launched"),
        "registration_hash": reg["registration_hash"],
        "treatment_path_hash": reg["treatment"]["treatment_path_hash"],
        "budget_record": RECORD,
        "required_amendment": {
            "what": ("add 'phase_d3' to funds_formal_sessions_of in "
                     "configs/stages/stage-1/phase_c1/authorization.json"),
            "why": ("the issuer refuses a phase outside the formal book; "
                    "adding it is a maintainer amendment, never an "
                    "autonomous one"),
            "changes_no_cap": True},
        "structural": {
            "decide_now": True,
            "expected_all_in_usd": totals["structural_expected_usd"],
            "hard_all_in_usd": doc["steps"]["d3_structural"]["hard"][
                "all_in_usd"],
            "hard_with_teardown_reserve_usd":
                totals["structural_hard_with_reserve_usd"],
            "proposed_cumulative_envelope_usd": s_env,
            "envelope_holds": (f"one session at hard plus its reserve, and "
                               f"{REPLACEMENT_ATTEMPTS} corrected pre-science "
                               "attempt"),
            "grant_template": {
                "rung": "structural",
                "approves_registration_hash": reg["registration_hash"],
                "cumulative_envelope_usd": s_env,
                "granted_by": "<the maintainer's decision, verbatim>",
                "covers": ("ONE D3 structural session at a time under this "
                           "cumulative envelope; no training, no probes")}},
        "probes": {
            "decide_now": False,
            "conditional_on": D.STRUCTURAL_CHANGE,
            "expected_all_in_usd": totals["probes_expected_usd"],
            "hard_all_in_usd": doc["steps"]["d3_probes"]["hard"]["all_in_usd"],
            "hard_with_teardown_reserve_usd":
                totals["probes_hard_with_reserve_usd"],
            "measured_same_shape_all_in_usd": doc["steps"]["d3_probes"][
                "measured_same_shape_all_in_usd"],
            "proposed_cumulative_envelope_usd": p_env,
            "grant_template": {
                "rung": "probes",
                "approves_registration_hash": reg["registration_hash"],
                "cumulative_envelope_usd": p_env,
                "granted_by": "<the maintainer's decision, verbatim>",
                "covers": ("ONE D3 probes session at a time under this "
                           "cumulative envelope, after STRUCTURAL_CHANGE")}},
        "both_at_proposed_envelopes_usd": s_env + p_env,
        "live": doc.get("live"),
        "authorizes": "nothing",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="re-read the four limits and the provider's price")
    args = ap.parse_args(argv)
    limits = quote = None
    if args.live:
        from stages.phase_d1.d1_authorization import live_secure_price

        limits = live_limits()
        quote = live_secure_price()
    doc = build(limits=limits, quote=quote)
    body = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    out = REPO_ROOT / RECORD
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        (REPO_ROOT / PROPOSAL).write_text(
            json.dumps(proposal(doc), indent=1, sort_keys=True) + "\n")
        print(f"wrote {RECORD} and {PROPOSAL}")
        print(json.dumps(doc["totals"], indent=1))
        return 0
    if args.live:
        print(json.dumps({"totals": doc["totals"], "live": doc["live"]},
                         indent=1))
        return 0
    #: Without --live the record's dated live block is not regenerated, so
    #: compare everything else.
    if out.is_file():
        have = json.loads(out.read_text())
        have.pop("live", None)
        same = have == json.loads(json.dumps(doc))
    else:
        same = False
    print(f"{RECORD}: {'current' if same else 'STALE (run with --write --live)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

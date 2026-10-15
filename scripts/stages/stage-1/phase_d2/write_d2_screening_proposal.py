#!/usr/bin/env python3
"""Write the D2 SCREENING one-use authorization PROPOSAL. Authorizes nothing.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/write_d2_screening_proposal.py [--write]

What the maintainer is asked to approve, every number DERIVED from its owner:
the price from `d2_budget.py` re-priced at the provider's live rate, the four
money conditions from D1's authorization module, the programme reservation
from `d2_budget.programme_scenarios` with the proposed envelope in place of
the screening ceiling, and the bound identities from the $0 session contract.
The grant template at the bottom is the one document a maintainer signs; the
issuer derives everything else and refuses a grant that states it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_REL = "logs/stages/stage-1/phase_d2/plans/d2_screening_grant_proposal.json"
#: The envelope this proposal asks for: one full session at its hard ceiling
#: plus room for pre-science subruns (host redraws, a setup failure) before a
#: full session can no longer be issued. A PROPOSAL, the maintainer's to set.
PROPOSED_ENVELOPE_USD = 25.0


def build(*, live_rate: float | None = None) -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD
    from stages.phase_d1.d1_authorization import (
        check_the_four_conditions, live_money, live_secure_price,
    )
    from stages.phase_d2 import d2_screening as SC
    from stages.phase_d2 import d2_screening_authorization as A
    from stages.phase_d2.autoinit_d2_screening_launch import host_admission

    quote = ({"usd_per_hour": float(live_rate), "_quoted": "supplied"}
             if live_rate is not None else live_secure_price())
    priced = A.reprice_at(quote["usd_per_hour"], REPO_ROOT)
    money = live_money(REPO_ROOT)
    failed = check_the_four_conditions(ceiling=priced["hard_ceiling_usd"],
                                       money=money)
    contract = SC.session_contract(REPO_ROOT)
    budget = BUD.build()
    limits = {"project_remaining_usd": money["project_cap_usd"]
              - money["cumulative_spend_usd"],
              "project_cap_usd": money["project_cap_usd"],
              "formal_remaining_usd": money["formal_remaining_usd"],
              "engineering_remaining_usd": 0.0,
              "per_session_cap_usd": money["per_session_envelope_usd"]}
    programme = json.loads((REPO_ROOT / BUD.SCENARIO_RECORD).read_text())
    in_force = programme["search_pricing_in_force"]
    rule = programme["search_pricing"][in_force]["reservation_rule"]
    #: The reservation with the ENVELOPE (not the session ceiling) held for
    #: screening: the conservative reading of what approving it commits.
    extra = PROPOSED_ENVELOPE_USD - (budget["steps"]["d2_screening"]["hard"][
        "all_in_usd"] + BUD.TEARDOWN_RESERVE_USD)
    left_with_envelope = round(rule["left_after_d2_at_hard_usd"] - max(0.0, extra),
                               2)
    image = SC.protocol(REPO_ROOT)["recovery"]["training_image_digest"]
    return {
        "schema": "aadistill.phase_d2.screening_grant_proposal/v1",
        "status": "PROPOSED -- awaiting the maintainer's decision",
        "authorizes": "nothing",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "would_authorize": (
            "ONE D2 behavioural SCREENING session: the eight frozen probes "
            "(c1..c4 x seeds 516448251, 614923639) from the four committed "
            "initializations, the rollout-rescoring diagnostic on each, "
            "stopping at preservation; the ranking and the one-or-none "
            "selection computed off-pod at $0. A replacement after a "
            "pre-science failure is a new one-use authorization netted "
            "against the same envelope."),
        "would_not_authorize": [
            "the shared confirmation (a separate grant naming the advancing "
            "candidate)", "retraining, regenerating or re-evaluating B or q2",
            "any change to K, the seeds, the SESOI, the rule or the battery",
            "re-screening or adding seeds", "D3", "a fresh confirmation",
            "any promotion"],
        "bound_identities": {
            "preregistration_hash": contract["preregistration_hash"],
            "shared_protocol": contract["shared_protocol"],
            "contract_hash": SC.contract_hash(contract),
            "selection_sha256": contract["field"]["selection_sha256"],
            "arms": [{"arm": a["arm"], "state_id": a["state_id"],
                      "artifact_digest": a["identities"]["artifact_digest"]}
                     for a in contract["field"]["arms"]],
            "seeds": contract["seeds"], "n_probes": contract["n_probes"],
            "battery": {k: contract["battery"][k] for k in (
                "role", "family_content_id", "item_ids_sha256")},
            "recovery": {k: contract["recovery"][k] for k in (
                "recipe_id", "recipe_digest", "data_digest",
                "trainer_source_digest", "training_runtime_digest")},
            "evaluation_protocol_hash": contract["evaluation"][
                "evaluation_protocol_hash"],
            "comparator_units": [u["probe_id"] for u in
                                 contract["comparator"]["B"]],
            "host_image_identity_required": image,
            "host_admission_self_check": host_admission(image)[0],
        },
        "price": {
            "live_rate_usd_per_hour": quote["usd_per_hour"],
            "quote": quote,
            "expected_usd": priced["expected_usd"],
            "hard_ceiling_usd": priced["hard_ceiling_usd"],
            "hard_ceiling_minutes": priced["hard_ceiling_minutes"],
            "expected_minutes": priced["expected_minutes"],
            "teardown_reserve_usd": BUD.TEARDOWN_RESERVE_USD,
            "decomposition": {
                "session_overhead_minutes": priced["session_overhead_minutes"],
                "arm_staging": priced["arm_staging"],
                "probes": budget["steps"]["d2_screening"]["n_probes"],
                "rescoring": priced["rescoring"]},
            "basis": priced["_basis"],
            "previous_planning_figure": {
                "expected_usd": 16.61, "hard_usd": 20.83,
                "source": ("logs/stages/stage-1/families/d_series/plans/"
                           "d2_budget_v1.json at 7dd31165"),
                "why_lower_now": ("the arms are staged from the relay "
                                  "(persisted and round-trip verified) instead "
                                  "of four 22-minute on-pod re-materializations")},
        },
        "funding_checks": {
            "four_conditions_failed": failed,
            "fundable": not failed,
            "phase_d2_funded": "phase_d2" in money["funds_formal_sessions_of"],
            "per_session_cap_usd": money["per_session_envelope_usd"],
            "limits_now": {k: round(v, 4) for k, v in limits.items()
                           if k != "engineering_remaining_usd"},
            "proposed_cumulative_envelope_usd": PROPOSED_ENVELOPE_USD,
            "programme_reservation": {
                "in_force": in_force,
                "left_after_d2_at_hard_usd": rule["left_after_d2_at_hard_usd"],
                "needed_for_d3_and_fresh_at_hard_usd":
                    rule["needed_for_d3_and_fresh_at_hard_usd"],
                "left_if_the_whole_envelope_is_spent_usd": left_with_envelope,
                "satisfied_at_hard": rule["satisfied"],
                "satisfied_with_the_whole_envelope": (
                    left_with_envelope
                    >= rule["needed_for_d3_and_fresh_at_hard_usd"]),
                "owner": BUD.SCENARIO_RECORD},
        },
        "issuance_and_launch": [
            "1. commit and push this tree; it is the executable commit",
            "2. the maintainer writes the grant (template below) to "
            "logs/budget/approvals/autoinit_d2_screening_grant.json",
            "3. issue: PYTHONPATH=src:scripts .venv/bin/python scripts/stages/"
            "stage-1/phase_d2/issue_d2_screening_authorization.py --grant "
            "<grant> --run-id d2_screening_<utc>  (quotes the live rate, "
            "derives and nets the ceiling, binds the contract hash)",
            "4. commit the authorization alone; stage the bundle: scripts/"
            "stages/stage-1/phase_d1/stage_d1_bundle.py --experiment-id "
            "phase_d2 --session-commit <sha> --run-id <run>",
            "5. dry run (all $0 gates, creates nothing), then the launch in a "
            "durable tmux session: scripts/stages/stage-1/phase_d2/"
            "autoinit_d2_screening_launch.py --scr <scr> --session-commit "
            "<sha> --bundle aad_autoinit_<sha8>.bundle --run-id <run>",
            "6. after ALL_DONE: rank_d2_screening.py --run <run> --store "
            "<scr>/store --write (off-pod, $0)"],
        "failure_semantics": {
            "before_any_training": ("ordinary engineering: preserve, tear "
                                    "down, repair, a new authorization netted "
                                    "against the envelope (P12.1)"),
            "after_training_began": (
                "every trained checkpoint is preserved off-pod and "
                "identity-verified while the pod runs; a replacement pod "
                "restores them (--restore-trained) and RESUMES each at "
                "evaluation -- never retrains a frozen unit; a complete valid "
                "probe is never re-run"),
            "runtime_mismatch": ("a training runtime or attested evaluation "
                                 "protocol other than the shared rung's stops "
                                 "the session before it generates; the host "
                                 "admission redraws non-matching hosts first"),
            "diagnostics": ("the token-id sidecar and the rescoring never "
                            "block, alter or re-run a scored probe; a wrapper "
                            "failure falls back to the plain generator"),
        },
        "statistical_integrity": SC.rule(REPO_ROOT),
        "grant_template": {
            "rung": "screening",
            "approves_preregistration_hash": contract["preregistration_hash"],
            "cumulative_envelope_usd": PROPOSED_ENVELOPE_USD,
            "granted_by": "<maintainer, date, decision reference>",
            "covers": ("ONE D2 screening session of the eight frozen probes; "
                       "not the confirmation, not B/q2, not a promotion"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--live-rate", type=float, default=None)
    args = ap.parse_args(argv)
    doc = build(live_rate=args.live_rate)
    if args.write:
        (REPO_ROOT / OUT_REL).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT_REL}")
    print(json.dumps({k: doc[k] for k in ("price", "funding_checks")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the Phase-C2 full-search GRANT PROPOSAL from the live tree.

    PYTHONPATH=src:scripts python \
      scripts/autoinit/write_c2_full_search_grant_proposal.py [--write]

A PROPOSAL, not a grant. It states everything a maintainer would be approving
for one formal full-joint-re-search session, with every machine identity derived
and every figure recomputed, so a launch review has the numbers in front of it.

It is deliberately not promotable by an agent: it states no `granted_by`,
carries no `one_use`, and declares a schema the grant contract does not name, so
`issue_c2_full_search_authorization.py` refuses it. A test exercises that.

Generated rather than hand-written because the first version was hand-written
and went stale the moment the storage derivation was corrected -- and a review
reading a stale proposal is reading another tree's numbers. Deterministic: no
clock, no random, no `generated_utc`, so re-running it on an unchanged tree
produces byte-identical output and a diff means something changed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from experiments.phase_c2 import full_search as FSG  # noqa: E402
from experiments.phase_c2 import full_search_authorization as FA  # noqa: E402

OUT = ("logs/stages/stage-1/phase_c2/plans/"
       "phase_c2_full_search_grant_proposal.json")
SCHEMA = "aadistill.autoinit.c2_full_search_grant_proposal/v1"

#: The rate the proposal is stated at. A launch RE-QUOTES: the issuer queries
#: securePrice and refuses when it no longer matches what a grant approved
#: against, printing what the ceiling would become. So this is the rate that was
#: live when the proposal was generated, not a permission to pay it forever.
RATE_AT_GENERATION = 1.09


def budget_position() -> dict:
    """The project's position, derived by the module that owns it."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import consolidate.derive_budget as D

    pkg_doc = D.load(D.PACKAGE, REPO_ROOT)
    return D.project_balance(
        REPO_ROOT, pkg_doc.get("execution_package", pkg_doc),
        pkg_doc["accepted_pricing"]["cumulative_cap_usd"])


def proposal() -> dict:
    rate = RATE_AT_GENERATION
    total = FSG.total_ceiling_usd(rate, REPO_ROOT)
    provision = FSG.provision_gb(REPO_ROOT)
    storage = FSG.peak_resident_gib(REPO_ROOT)
    environment = FSG.teacher_and_environment_gib(REPO_ROOT)
    identities = {k: v for k, v in FA.live_identities(REPO_ROOT).items()
                  if not k.startswith("_")}
    position = budget_position()
    pricing = FSG.pricing(REPO_ROOT)
    behavioural = pricing["behavioural_selection"]
    behavioural_hours = behavioural["hard_ceiling_minutes"] / 60.0
    behavioural_disk = FSG.storage_cost_usd(behavioural_hours, REPO_ROOT)["usd"]
    chain_total = round(total["total_hard_ceiling_usd"]
                        + behavioural["hard_ceiling_usd"]
                        + behavioural_disk, 4)
    expected_usd = round(FSG.expected_usd(REPO_ROOT)
                         / FSG.price_per_hour_basis(REPO_ROOT) * rate, 4)

    return {
        "schema": SCHEMA,
        "_contract": (
            "A PROPOSAL, not a grant. It states everything a maintainer would "
            "be approving for one formal Phase-C2 full-joint-re-search session, "
            "with every machine identity DERIVED from this tree and every "
            "figure recomputed. It states no `granted_by`, carries no "
            "`one_use`, and its schema is not the one the grant contract names, "
            "so the issuer refuses it -- exercised by a test, not asserted. "
            "AUTHORIZES NOTHING AND APPROVES NOTHING."),
        "_what_it_is_for": (
            "the maintainer's execution sequence ends 'return for FINAL FORMAL "
            "SEARCH LAUNCH REVIEW'. This is what that review reads. If it "
            "approves, the decision supplies `granted_by`, `covers`, "
            "`explicitly_not_authorized`, `approved_money` and `one_use`, and "
            "the result is committed as the run's grant.json under the real "
            "grant schema. Nothing here can be promoted by an agent."),
        "_generated_by": ("scripts/autoinit/write_c2_full_search_grant_proposal.py, "
                          "deterministically from the live tree. Re-run it: an "
                          "unchanged tree gives byte-identical output, so a diff "
                          "means a figure moved."),

        "proposed_for": {
            "experiment_id": "phase_c2_full_search",
            "session": (
                "ONE beam over the derived full joint space at the standing "
                "width, ending at a committed Top-5. It trains nothing, scores "
                "no battery, produces no correct_overall and cannot name an "
                "incumbent."),
            "terminates_at": "commit_top_k",
            "run_id": ("NOT ASSIGNED. The attempt id is chosen when a grant is "
                       "written; every artifact in the chain resolves from it."),
        },

        "money": {
            "_the_ceiling_is_not_the_gpu_price_times_the_minutes": (
                "the provider bills Container Disk separately from the GPU, and "
                "this session provisions "
                f"{provision['provision_gb']} GB of it. An earlier version of "
                "this proposal stated a GPU-only ceiling while the launcher "
                "provisioned hundreds of GB, and no figure in it disagreed with "
                "any other -- review found it, not a gate. A live GPU quote is "
                "not evidence that separately priced storage is free."),
            "gpu": {
                "card": "NVIDIA L40S",
                "securePrice_usd_per_hour": rate,
                "_requoted": ("live at generation, and RE-QUOTED again by the "
                              "issuer immediately before it writes an "
                              "authorization; a moved rate refuses and prints "
                              "the re-derived ceiling"),
                "provider_api_exposes_this": True,
                "hard_ceiling_usd": total["gpu_usd"],
            },
            "container_disk": {
                **total["container_disk"],
                "provider_api_exposes_this": False,
                "_basis": FSG.STORAGE_PRICING,
            },
            "network_volume_usd": total["network_volume_usd"],
            "_no_volume": total["_no_volume"],
            "hard_ceiling_minutes": total["hard_ceiling_minutes"],
            "effective_rate_usd_per_hour": total["effective_rate_usd_per_hour"],
            "expected_usd": expected_usd,
            "TOTAL_hard_ceiling_usd": total["total_hard_ceiling_usd"],
            "max_price_usd_per_hour": rate,
            "_the_boundary": (
                "a launch is refused above this GPU rate, by the launcher's "
                "runtime gate and by the issuer's live re-quote. A higher rate "
                "needs the ceiling re-derived and a new decision; it is not a "
                "launcher flag, and beam width 6 is not narrowed to absorb it."),
        },

        "provisioning": {
            "peak_resident_search_states_gib": storage["peak_resident_gib"],
            "peak_at_level": storage["peak_at_level"],
            "final_resident_gib": storage["final_resident_gib"],
            "teacher_and_environment_gib": environment["total_gib"],
            "required_gib": provision["required_gib"],
            "required_gb": provision["required_gb"],
            "provision_gb": provision["provision_gb"],
            "headroom_multiple": provision["headroom_multiple"],
            "_residency_is_cumulative": storage["_model"],
            "_why_not_search_1s_200": (
                "Search-1 provisions for a 87.4 GiB peak: five retained level-0 "
                "states expanded into eighteen children. Here every operator "
                "kind may go first, so level 0 generates eleven children of "
                "which nine continue and level 1 expands those nine into SIXTY "
                "-- and because the search releases ONLY the partial children a "
                "level prunes, the root, every expanded ancestor, every dead end "
                "and every completed leaf stay resident for the whole run. An "
                "earlier derivation modelled parents + children + leaves and "
                "missed the retained ancestors, giving 243.4 GiB; review caught "
                "that it was not an upper bound before a volume was provisioned "
                "from it."),
            "_cross_checked": (
                "the derivation puts a finished leaf at "
                f"{storage['target_state_gib']} GiB and the CUDA engineering "
                "validation weighed the real 596M student's checkpoint at "
                "exactly that."),
            "_units": provision["_unit"],
        },

        "budget_position": {
            "cumulative_spend_usd": position["cumulative_spend_usd"],
            "authorized_cap_usd": position["cap_usd"],
            "remaining_usd": position["remaining_usd"],
            "search_total_hard_ceiling_usd": total["total_hard_ceiling_usd"],
            "remaining_after_the_search_usd": round(
                position["remaining_usd"] - total["total_hard_ceiling_usd"], 4),
            "behavioural_gpu_hard_ceiling_usd": behavioural["hard_ceiling_usd"],
            "behavioural_storage_upper_bound_usd": behavioural_disk,
            "_behavioural_storage_is_bounded_not_derived": (
                "that session's launcher and provision do not exist yet, so its "
                "disk term is bounded ABOVE by the search's own "
                f"{provision['provision_gb']} GB -- certainly more than twelve "
                "probes on a 596M student need. The chain total is therefore an "
                "over-estimate rather than an unstated omission, and the figure "
                "should fall when that session is bound."),
            "chain_total_hard_ceiling_usd": chain_total,
            "remaining_after_the_whole_chain_usd": round(
                position["remaining_usd"] - chain_total, 4),
            "_derived_by": (
                "scripts/consolidate/derive_budget.py :: project_balance, from "
                "the package anchor plus every recorded run closeout. The cap's "
                "canonical owner is configs/experiments/phase_c1/"
                "authorization.json :: accepted_pricing.cumulative_cap_usd."),
            "_fitting_is_not_permission": (
                "the cap rose to 370.00 on 2026-09-17 as an ACCOUNTING "
                "ENVELOPE, explicitly not a spend authorization and not "
                "transferable to C3/C4. Both sessions fit inside it at their "
                "corrected totals. That is a feasibility statement."),
            "_two_authorizations_never_one": (
                "the behavioural figures are shown so a review can see both "
                "halves at once, NOT because this proposal covers them. That "
                "session's candidate identities do not exist until this search "
                "commits a Top-5, so it cannot be authorized yet even in "
                "principle."),
        },

        "bound_identities_the_issuer_will_reproduce": identities,
        "_identities_note": (
            "every one is DERIVED by full_search_authorization.live_identities "
            "and re-derived at issuance; a grant that asserted a different "
            "value, omitted one, or introduced one the issuer cannot derive is "
            "refused. The space figures are what make a Search-1 grant unusable "
            "here: Search-1 fixed three operators and varied one, this searches "
            "all four jointly with calibration unpinned."),

        "what_a_grant_would_still_have_to_supply": [
            "granted_by -- only a person can say who approved",
            "covers -- the scope sentence carried into the authorization",
            "explicitly_not_authorized -- what the approval refuses",
            "approved_money -- expected, the TOTAL ceiling, the live rate and "
            "the rate boundary, which the issuer recomputes from the pricing "
            "record's minutes at the stated rate plus the derived storage bound",
            "one_use -- how many provider resources the session may draw, which "
            "becomes the enforceable resource scope",
        ],

        "engineering_evidence_this_rests_on": {
            "cuda_validation": ("logs/stages/stage-1/phase_c2/validations/"
                                "full-search-cuda/v1/closeout.json"),
            "verdict": (
                "PASS. The full-search driver reached C2_FULL_SEARCH_ALL_DONE on "
                "a real L40S over the real 578-leaf joint space; all 35 "
                "materialized states reloaded on cuda in bfloat16; the real 596M "
                "student reloaded canonically with its rope base and tied head "
                "intact. $0.1453 of a $0.9000 ceiling, every teardown "
                "provider-confirmed."),
            "setup_dispatch": (
                "the shared setup script's c2_full_search authorization branch "
                "is EXECUTED at $0 against real artifacts -- admitting a valid "
                "one and refusing each governance boundary. It was missing "
                "entirely at 5535c6a, which would have refused a formal pod "
                "after paid setup: the same class as SESSION_KIND=phase_b, which "
                "cost $0.2300."),
            "_what_none_of_it_establishes": (
                "that the formal search fits its budget -- the validation ran "
                "toy geometry for 20 seconds -- and nothing behavioural "
                "whatsoever."),
        },

        "authorizes": "nothing",
        "approves": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    doc = proposal()
    body = json.dumps(doc, indent=1) + "\n"
    out = REPO_ROOT / OUT

    money = doc["money"]
    position = doc["budget_position"]
    print(f"provision        : {doc['provisioning']['provision_gb']} GB for "
          f"{doc['provisioning']['required_gib']} GiB required "
          f"(peak {doc['provisioning']['peak_resident_search_states_gib']} GiB "
          f"at level {doc['provisioning']['peak_at_level']})")
    print(f"GPU ceiling      : ${money['gpu']['hard_ceiling_usd']:.4f} at "
          f"${money['gpu']['securePrice_usd_per_hour']}/h over "
          f"{money['hard_ceiling_minutes']:.0f} min")
    print(f"disk ceiling     : ${money['container_disk']['usd']:.4f} "
          f"({money['container_disk']['provisioned_gb']} GB at "
          f"${money['container_disk']['usd_per_hour']:.6f}/h)")
    print(f"TOTAL ceiling    : ${money['TOTAL_hard_ceiling_usd']:.4f} "
          f"(effective ${money['effective_rate_usd_per_hour']:.6f}/h)")
    print(f"chain total      : ${position['chain_total_hard_ceiling_usd']:.4f}")
    print(f"remaining after  : "
          f"${position['remaining_after_the_whole_chain_usd']:.4f} of "
          f"${position['remaining_usd']:.4f}")
    print(f"identities       : {len(doc['bound_identities_the_issuer_will_reproduce'])}")
    print("APPROVES NOTHING. AUTHORIZES NOTHING.")

    if not args.write:
        if out.is_file() and out.read_text() == body:
            print("unchanged (pass --write to rewrite anyway)")
        else:
            print("WOULD CHANGE — pass --write")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

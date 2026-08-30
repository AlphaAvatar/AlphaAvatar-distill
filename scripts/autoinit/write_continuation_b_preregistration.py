#!/usr/bin/env python3
"""Freeze what the behavioural continuation would do. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/write_continuation_b_preregistration.py

A generator rather than a hand-written file, because the document binds seven
identities that all move when code moves — the executable source digest most of
all. A preregistration that quietly disagreed with the executable it describes
would be worse than none.

This records intent. It is **not** an authorization, does not request one, and
does not raise the cumulative cap.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.phase_b_continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, CONTINUATION_SOURCE_SET_VERSION,
    KNOWN_NEUTRALIZED_SEARCH_CALL_SITES, continuation_source_digest,
    search_call_site_owners,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

AMENDMENT = REPO_ROOT / "logs/autoinit_phase_b_identity_collapse_amendment.json"
PRICING = REPO_ROOT / "logs/autoinit_behavioural_continuation_pricing.json"
ASSETS = REPO_ROOT / "logs/autoinit_continuation_b_assets.json"
SELECTION = REPO_ROOT / "logs/autoinit_phase_b_attempt5/stage1_selection.json"
HISTORICAL = REPO_ROOT / "logs/autoinit_historical_probe_reuse.json"
ATTEMPT5 = REPO_ROOT / "logs/autoinit_attempt5_probe_reuse.json"
ATTEMPT4 = REPO_ROOT / "logs/autoinit_attempt4_probe_reuse.json"
FROZEN = REPO_ROOT / "logs/autoinit_phase_a_recovery_plan_frozen.json"

FULL_PHASE_B_CEILING_USD = 35.6660


def build() -> dict:
    amendment = json.loads(AMENDMENT.read_text())
    pricing = json.loads(PRICING.read_text())
    assets = json.loads(ASSETS.read_text())
    selection = json.loads(SELECTION.read_text())
    historical = json.loads(HISTORICAL.read_text())
    attempt5 = json.loads(ATTEMPT5.read_text())
    attempt4 = json.loads(ATTEMPT4.read_text())
    frozen = json.loads(FROZEN.read_text())
    science = frozen.get("plan_hash") or frozen["plan"]["plan_hash"]
    rung1 = amendment["rung1_selection"]
    source = continuation_source_digest(REPO_ROOT)

    # Two DIFFERENT concepts, conflated in v1 under one name that read as though
    # it held both. `gate_exclusions` are the feasibility and catastrophic-
    # capability refusals rung 1 applies BEFORE ranking; both are empty for this
    # result, which is a fact worth stating rather than an absence to infer.
    # `searched_non_survivors` are the Top-5 leaves that simply ranked below the
    # top-2 cut. v1 derived the second from `rung1["all_exclusions"]` — a key that
    # does not exist — so `.get(...)` returned `[]` and the field silently claimed
    # there were none.
    #
    # Derived from the two frozen identities, never from behavioural scores: a
    # non-survivor list read off a measurement would let the measurement decide
    # the record of who was eligible.
    top5 = [entry["state_id"] for entry in selection["selected"]]
    survivors = list(rung1["selected_searched"])
    missing = sorted(set(survivors) - set(top5))
    if missing:
        raise SystemExit(
            f"refusing to preregister: the frozen rung-1 survivors {missing} are "
            "not in the authoritative Top-5. One of the two records is wrong, and "
            "guessing which would corrupt the other.")
    searched_non_survivors = sorted(set(top5) - set(survivors))
    if len(searched_non_survivors) != len(top5) - len(survivors):
        raise SystemExit(
            "refusing to preregister: the Top-5 contains duplicate state ids")

    if not assets["verification"]["verified"]:
        raise SystemExit(
            "refusing to preregister: the relay copy of the advancing checkpoint "
            "is not verified. A session that cannot stage its one new finalist "
            "cannot run.")

    body = {
        "schema": "aadistill.autoinit.continuation_b_preregistration/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("FROZEN BEFORE ANY BEHAVIOURAL-CONTINUATION RESULT EXISTS. Not "
                   "an authorization, not a grant, and not an instruction to launch."),
        "what_this_session_is": (
            "Phase-B Stage 1 is COMPLETE. Attempt 5 ran the joint P=2 search to "
            "completion, emitted an authoritative Top-5 and a durable Stage-1 "
            "selection artifact, and paid for three rung-1 sa probes. It then died "
            "on a candidate-universe collision that the identity-collapse "
            "amendment resolves, and rung 1 was completed at $0 from retained "
            "evidence. What remains is behavioural: one missing sb, and at most "
            "two conditional sc. This session buys ONLY that."),
        "what_this_session_must_not_do": [
            "run the P=2 search, or any search",
            "purchase a new sa probe",
            "recompute rung 1",
            "probe a searched non-survivor",
            "reach a fourth seed",
            "break a tie with Stage-1 ranking, search-side KL/NLL, the canonical "
            "Stage-1 NLL, or state-id ordering",
        ],
        "evidence_universe": {
            "distinct_candidates": amendment["collapsed_universe"]["distinct_candidates"],
            "universe_identity": amendment["collapsed_universe"]["universe_identity"],
            "amendment_sha256": amendment["amendment_sha256"],
            "role": ("EVIDENCE and provenance only. Six distinct materialized "
                     "initializations carry the completed sa evidence and the "
                     "identity-collapse result. Their bytes are NOT staged, "
                     "because five-sixths of them are never probed."),
        },
        "active_finalists": {
            "count": len(rung1["advancing"]),
            "state_ids": rung1["advancing"],
            "selected_searched": rung1["selected_searched"],
            "auto_advanced_control": rung1["auto_advanced_control"],
            "computed_by": rung1["computed_by"],
            "role": ("The ONLY states that may enter sb, the pooled sa+sb "
                     "decision, the conditional sc and the final selection. Rung 1 "
                     "is complete and frozen; this session imports that result and "
                     "does not recompute it."),
            "gate_exclusions": {
                "by_feasibility": rung1["excluded_by_feasibility"],
                "by_catastrophic_capability":
                    rung1["excluded_by_catastrophic_capability"],
                "meaning": ("candidates rung 1 refused BEFORE ranking, on the "
                            "feasibility floor or the catastrophic-capability "
                            "rule. Both are empty for this result: every searched "
                            "leaf was eligible, and the cut was made by rank."),
            },
            "searched_non_survivors": {
                "state_ids": searched_non_survivors,
                "derivation": ("the authoritative Attempt-5 Top-5 MINUS the two "
                               "frozen rung-1 survivors. Not selected by hand and "
                               "not read from behavioural scores."),
                "meaning": ("eligible searched leaves that ranked below the top-2 "
                            "cut. They remain in the six-candidate EVIDENCE "
                            "universe and carry citable sa observations; they may "
                            "not enter sb, the pooled decision, sc or the final "
                            "selection."),
                "top5": top5,
            },
        },
        "probe_inventory": {
            "derivation": ("from the two verified reuse records, cross-checked "
                           "against what the executable actually buys in a "
                           "whole-function run"),
            "cited_sa": "all six evidence candidates — no new sa is purchased",
            "missing_sb": pricing["evidence"]["missing_sb"],
            "reused_sb": ["85bde4ded2c3", "control-qwen"],
            "missing_sc_worst_case": pricing["evidence"]["missing_sc_worst_case"],
            "reused_sc": ["85bde4ded2c3"],
            "sc_is_conditional": ("sc runs only for candidates inside the frozen "
                                  "equivalence interval that lack a verified sc"),
            "new_probes_min": pricing["total"]["low_probes"],
            "new_probes_max": pricing["total"]["hard_probes"],
        },
        "reuse_rule": {
            "binds_on": ["student_artifact_digest", "seed"],
            "comparability": (
                "generation_runtime_comparability@v2. The citable evidence carries "
                "TWO raw evaluation_protocol_hash values (7327e880… and 250f72ef…) "
                "that differ by host NVIDIA driver patch alone, which the rule "
                "declares non-material; all share the comparable identity "
                "70a26e0b…. Requiring exact equality of the raw hash would re-buy "
                "all eight citable probes."),
            "historical_record": {
                "probes_dir_digest": historical["probes_dir_digest"],
                "admitted": sorted(historical["admitted_reusable_probes"])},
            "attempt5_record": {
                "probes_dir_digest": attempt5["probes_dir_digest"],
                "admitted": sorted(attempt5["reusable_probes"])},
            "attempt4_record": {
                "probes_dir_digest": attempt4["probes_dir_digest"],
                "admitted": sorted(attempt4["reusable_probes"]),
                "why_bound": (
                    "Attempt 4 purchased fe9683e6a9c7/sb. Without it the session "
                    "no longer holds complete sa+sb, so it is authorization-bound "
                    "like the other two rather than trusted via reuse_verified "
                    "inside its own record")},
        },
        "stage1_evidence": {
            "status": "COMPLETE — imported, never re-purchased",
            "selection_sha256": selection["selection_sha256"],
            "top_n": 5,
        },
        "executable_source": {
            "digest": source["digest"],
            "set_version": CONTINUATION_SOURCE_SET_VERSION,
            "n_files": source["n_files"],
            "derivation": (
                "the REAL import closure of the continuation driver and launcher, "
                "plus three subprocess-invoked runtime files. NOT curated: it "
                "therefore includes search.py, ranking.py and the operator "
                "modules, which the package __init__ loads for every consumer. "
                "What source is LOADED and what operation is PERMITTED are "
                "separate claims and are recorded separately."),
        },
        "no_search_guarantee": {
            "claims": [
                "the continuation plan declares no search stage",
                "stage1() and run_search() raise",
                "ContinuationAuthorization.runs_search is False BY TYPE — no field exists",
                "no search call site on the continuation's own path",
                "the only loaded file containing one is PhaseADriver, whose stage1 "
                "is overridden with a raise and never bound into the stage map",
                "a whole-function test drives the real stage map with BeamSearch "
                "and run_phase_a_search replaced by detonators, and neither is "
                "touched on either the resolved or the tie path",
            ],
            "known_neutralized_call_sites": list(KNOWN_NEUTRALIZED_SEARCH_CALL_SITES),
            "observed_call_site_owners": list(search_call_site_owners(REPO_ROOT)),
        },
        "science_plan": {"plan_hash": science,
                         "source": "logs/autoinit_phase_a_recovery_plan_frozen.json",
                         "unchanged": True},
        "session_plan": {
            "plan_id": CONTINUATION_PLAN_V1.plan_id,
            "plan_hash": CONTINUATION_PLAN_V1.plan_hash,
            "version": CONTINUATION_PLAN_V1.version,
            "stages": [s.stage for s in CONTINUATION_PLAN_V1.stages],
            "stage_2_absent_because": (
                "Phase A's stage 2 is rung 1 on seed sa, which this session "
                "imports as completed evidence rather than buying")},
        "calibration_profiles": [
            {"qualified_id": p.qualified_id, "profile_hash": p.profile_hash,
             "content_sha256": p.content_sha256}
            for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)],
        "relay_assets": {
            "repo": assets["repo"], "repo_type": assets["repo_type"],
            "state_id": assets["state_id"],
            "artifact_digest": assets["artifact_digest"],
            "verified": assets["verification"]["verified"],
            "verified_before_provider_creation":
                assets["verification"]["performed_before_provider_creation"]},
        "budget": {
            "floor_usd": pricing["total"]["low_usd"],
            "hard_ceiling_usd": pricing["total"]["hard_usd"],
            "full_phase_b_ceiling_usd_HISTORICAL_ONLY": FULL_PHASE_B_CEILING_USD,
            "note": ("the $35.6660 full-Phase-B ceiling is HISTORICAL and must not "
                     "authorize this session; it prices a search already bought")},
        "valid_terminal_results": [
            "a winner selected under the frozen rule",
            "winner=None with unresolved_equivalence — a RESULT, not a failure",
        ],
        "not_authorized": (
            "No continuation grant or authorization exists, no cumulative-budget "
            "increase has been requested, and this document does not request one. "
            "It records what the session would do if one were issued."),
    }
    body["preregistration_sha256"] = sha256_json(body)
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_continuation_b_preregistration.json")
    args = ap.parse_args()
    body = build()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"universe          {body['evidence_universe']['distinct_candidates']} candidates "
          f"({body['evidence_universe']['universe_identity'][:12]}…)")
    print(f"active finalists  {body['active_finalists']['count']}")
    print(f"new probes        {body['probe_inventory']['new_probes_min']}"
          f"..{body['probe_inventory']['new_probes_max']}")
    print(f"executable        {body['executable_source']['digest'][:16]}… "
          f"({body['executable_source']['n_files']} files, set v"
          f"{body['executable_source']['set_version']})")
    print(f"session plan      {body['session_plan']['plan_hash'][:16]}…")
    print(f"budget            floor ${body['budget']['floor_usd']:.4f}, "
          f"ceiling ${body['budget']['hard_ceiling_usd']:.4f}")
    print(f"prereg sha256     {body['preregistration_sha256']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Emit the machine-readable Phase-B preregistration. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/write_phase_b_preregistration.py \
        --out logs/autoinit_phase_b_preregistration.json

Everything Phase B is committed to, **before any Phase-B result exists**,
assembled from the live objects rather than transcribed so a field cannot drift
from what the code will actually do.

Phase B does **not** redefine the science. It reuses the frozen Phase-A science
plan — the same feasibility floor, equivalence interval and seeds — because those
were materialized from the Stage-3 controls and this session neither rematerializes
those controls nor redefines their thresholds. What Phase B freezes here is the
*session*: which distributions are searched, how they branch, what is admitted to
the behavioural comparison, and what terminates the run.

Two identities are bound per calibration profile and both are required:
`profile_hash` fixes the **specification**, `content_sha256` fixes the **sampled
bytes**. Neither implies the other — on this pool the seed does not even reach the
bytes — so a preregistration citing only the spec would not identify the mixture
an operator is actually fed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, NO_CALIBRATION, REASONING_HEAVY_V2,
)
from aadistill.autoinit.operators import V1_IMPLEMENTATIONS  # noqa: E402
from aadistill.autoinit.operators.base import CalibrationNeed  # noqa: E402
from aadistill.autoinit.phase_b import (  # noqa: E402
    CANONICAL_CONTROL, PHASE_A_EXCLUDED_LEAVES, PHASE_A_EXCLUSION_RULE,
    PHASE_A_IMPORTED_FINALISTS, PHASE_B_DELEGATED_IDENTITIES, PHASE_B_PLAN_V1,
    PHASE_B_SEARCHED_LEAVES, PHASE_B_UNCOVERED, SURVIVORS_AT_SB,
    phase_b_source_digest,
)
from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1  # noqa: E402
from aadistill.autoinit.recovery import SEED_SA, SEED_SB, SEED_SC  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

FROZEN_SCIENCE_PLAN = REPO_ROOT / "logs/autoinit_phase_a_recovery_plan_frozen.json"
REUSE_RECORD = REPO_ROOT / "logs/autoinit_historical_probe_reuse.json"
STATE_EVAL_MANIFEST = REPO_ROOT / "artifacts/stage1/state_eval_v1/manifest.json"

#: Transcribed, then verified against the file. A constant that reads its expected
#: value out of its own subject cannot fail.
FROZEN_SCIENCE_PLAN_HASH = (
    "02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c")
STATE_EVAL_CONTENT_SHA256 = (
    "a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc")
RECOVERY_BATTERY_CONTENT_SHA256 = (
    "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323")


def _frozen_science_plan() -> dict:
    plan = json.loads(FROZEN_SCIENCE_PLAN.read_text())
    if plan.get("plan_hash") != FROZEN_SCIENCE_PLAN_HASH:
        raise SystemExit(
            f"{FROZEN_SCIENCE_PLAN.name} hashes to {plan.get('plan_hash')} but "
            f"this preregistration is written against {FROZEN_SCIENCE_PLAN_HASH}. "
            "Phase B does not redefine the science plan; if it moved, stop.")
    return plan


def _calibration(profile) -> dict:
    """Both identities, plus what makes them different questions."""
    if not profile.materialized or not profile.content_sha256:
        raise SystemExit(
            f"{profile.qualified_id} is not materialized; a preregistration "
            "cannot bind bytes that do not exist")
    return {
        "qualified_id": profile.qualified_id,
        "spec_identity": {
            "profile_hash": profile.profile_hash,
            "covers": ("sources, revisions, domain weights, token budget, "
                       "sample rule, seed, role, leakage exclusions"),
        },
        "materialized_identity": {
            "content_sha256": profile.content_sha256,
            "items_file_sha256": profile.items_file_sha256,
            "items_path": profile.items_path,
            "covers": "the sampled bytes the operators are actually fed",
        },
        "why_both": ("profile_hash does NOT identify the sampled bytes. A mixture "
                     "can satisfy its spec hash and be different bytes — a rebuild "
                     "under a changed rule, a truncated file, a rendering change. "
                     "For calib.reasoning_heavy@v2 the seed does not even reach the "
                     "bytes, because every sub-type optimum is unique."),
        "token_budget": profile.token_budget,
        "domain_weights": dict(sorted(profile.domain_weights.items())),
        "leakage_proof_path": profile.leakage_proof_path,
        "leakage_exclusions": list(profile.leakage_exclusions),
    }


def build() -> dict:
    science = _frozen_science_plan()
    source = phase_b_source_digest(REPO_ROOT)
    reuse = json.loads(REUSE_RECORD.read_text()) if REUSE_RECORD.is_file() else {}
    if not reuse.get("reuse_verified"):
        raise SystemExit(
            "the historical probe reuse record is missing or unverified; Phase B "
            "may not preregister citing evidence whose reconstruction is unproved")

    suite = json.loads(STATE_EVAL_MANIFEST.read_text()) if STATE_EVAL_MANIFEST.is_file() else {}
    if suite and suite.get("content_sha256") != STATE_EVAL_CONTENT_SHA256:
        raise SystemExit("state_eval_v1 content hash moved; stop")

    branching = {
        impl.impl_id: {
            "kind": impl.kind,
            "calibration": impl.calibration.value,
            "profiles_branched_over": (
                1 if impl.calibration is CalibrationNeed.NONE else 2),
            "note": ("invoked once against the calib.none@v1 sentinel; it consumes "
                     "no mixture, so branching it would manufacture byte-identical "
                     "states" if impl.calibration is CalibrationNeed.NONE
                     else "branches over both calibration profiles"),
        }
        for impl in sorted(V1_IMPLEMENTATIONS, key=lambda i: i.impl_id)
    }

    prereg = {
        "schema": "aadistill.autoinit.phase_b_preregistration/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("FROZEN BEFORE ANY PHASE-B RESULT EXISTS. Not an authorization, "
                   "not a grant, and not an instruction to launch."),

        # --- what is searched -------------------------------------------
        "hypothesis": ("Does the AutoInitializer's preferred composition change "
                       "when the calibration distribution changes? Phase A searched "
                       "one distribution, so its ranking is conditional on it."),
        "search": {
            "profiles": 2,
            "joint": True,
            "joint_rationale": ("the two arms compete for the same beam slots; P=2 "
                                "changes pruning, so Phase-A leaves are NOT injected "
                                "to restrict the space and the domain-balanced arm "
                                "is re-searched rather than imported"),
            "beam_width": SCHEDULE_V1.width,
            "warmup_levels": SCHEDULE_V1.warmup_levels,
            "schedule": SCHEDULE_V1.as_dict(),
            "ranking_policy": PARETO_V1.as_dict(),
            "decomposed_paths": 24 * (1 + 2) * 2 * 2 * 1,
            "operator_branching": branching,
            "no_calibration_sentinel": NO_CALIBRATION.qualified_id,
            "nll_is_not_an_objective": (
                "NLL is recorded per domain as a diagnostic and appears nowhere in "
                "the beam objectives or the tie-break"),
        },
        "calibration_profiles": [
            _calibration(DOMAIN_BALANCED_V1), _calibration(REASONING_HEAVY_V2)],
        "state_evaluation": {
            "suite": "state_eval@v1",
            "content_sha256": STATE_EVAL_CONTENT_SHA256,
            "unchanged_from_phase_a": True,
        },

        # --- what may execute -------------------------------------------
        "executable_source": {
            "digest": source["digest"],
            "set_version": source["set_version"],
            "files": [e["path"] for e in source["files"]],
            "rule": source["rule"],
            "delegated_identities": dict(PHASE_B_DELEGATED_IDENTITIES),
            "not_yet_covered": list(PHASE_B_UNCOVERED),
            "phase_a_harness_untouched": (
                "PHASE_A_HARNESS_SOURCE_FILES_V1 is historical fact about a "
                "completed experiment and is not widened by Phase B"),
        },

        # --- the science Phase B does NOT redefine -----------------------
        "science_plan": {
            "plan_hash": FROZEN_SCIENCE_PLAN_HASH,
            "source": "logs/autoinit_phase_a_recovery_plan_frozen.json",
            "reused_unchanged": True,
            "equivalence_interval": science["equivalence_rule"]["value"],
            "feasibility_floor": science["feasibility_rule"]["value"],
            "catastrophic_capability_rule": science["catastrophic_capability_rule"],
            "recipe": science["recipe"],
            "phase_b_does_not": (
                "rematerialize the Stage-3 controls, redefine the feasibility "
                "floor or equivalence interval, or add a fourth seed"),
        },

        # --- the Stage-0 gate -------------------------------------------
        "runtime_comparability_gate": {
            "rule": "generation_runtime_comparability@v2",
            "checked_at": "stage 0, before any search or probe",
            "on_pass": ("verified historical Phase-A probe evidence is citable and "
                        "Phase B continues"),
            "on_fail": ("FAIL CLOSED and TERMINATE the session before any search or "
                        "probe"),
            "why_terminate_rather_than_rerun": (
                "comparability is not merely a reuse convenience. It is what lets "
                "behavioural results be judged against the FROZEN Stage-3 "
                "feasibility floor and equivalence interval, which were "
                "materialized under Phase A's runtime. If it fails, those "
                "thresholds do not describe anything this session could produce, "
                "so re-running all eight candidates would be a differently-"
                "thresholded experiment wearing Phase B's name — and would not "
                "restore the interval either."),
            "explicitly_not_a_fallback": (
                "the 14-probe no-reuse path is a rejected counterfactual, retained "
                "in the pricing artifact for comparison only; it is NOT an "
                "executable scientific fallback and NOT the authorization ceiling"),
        },

        # --- the cross-phase candidate set, closed now -------------------
        "candidate_set": {
            "phase_b_top_n": PHASE_B_SEARCHED_LEAVES,
            "admission_rule": ("the Top-5 admissible complete leaves of the joint "
                               "P=2 search, by the frozen epsilon-Pareto ranking"),
            "imported_phase_a_finalists": list(PHASE_A_IMPORTED_FINALISTS),
            "canonical_control": CANONICAL_CONTROL,
            "total_at_sa": PHASE_B_SEARCHED_LEAVES + len(PHASE_A_IMPORTED_FINALISTS) + 1,
            "excluded_phase_a_leaves": dict(sorted(PHASE_A_EXCLUDED_LEAVES.items())),
            "exclusion_rule": PHASE_A_EXCLUSION_RULE,
            "closed_before_results": (
                "recorded here so it cannot be reopened once Phase-B results are "
                "visible and one of the excluded leaves looks convenient"),
        },
        "historical_evidence": {
            "requirement": ("an imported result is citable ONLY where strict "
                            "reconstruction proves the same materialized recovery "
                            "protocol and seed"),
            "record": "logs/autoinit_historical_probe_reuse.json",
            "probes_dir_digest": reuse.get("probes_dir_digest"),
            "verified_at_preregistration": sorted(
                reuse.get("admitted_reusable_probes", [])),
            "checks": ("completeness; the frozen seed for the rung; the artifact "
                       "digest re-derived from the retained checkpoint bytes; the "
                       "battery and scoring-contract identities; the attested "
                       "protocol hash"),
            "re_verified_on_pod": True,
        },

        # --- the terminal procedure -------------------------------------
        "procedure": {
            "seeds": {"sa": SEED_SA, "sb": SEED_SB, "sc": SEED_SC,
                      "fourth_seed": "never"},
            "rung_sa": ("all 8 candidates; the 5 new leaves are probed, the 3 "
                        "verified priors are cited"),
            "rung_sb": (f"the {SURVIVORS_AT_SB} globally best searched candidates by "
                        "correct_overall among feasible candidates, plus the "
                        "control, which advances unconditionally; only missing "
                        "probes are run"),
            "selection": ("constraint then objective, never a weighted sum: "
                          "usable_rollout_rate gates by the frozen feasibility "
                          "floor, correct_overall ranks, correct_given_usable "
                          "explains"),
            "decision": ("pooled sa+sb under pooled_counts@v2, judged against the "
                         "frozen equivalence interval"),
            "rung_sc": ("conditional, only for candidates inside the equivalence "
                        "interval that lack a verified sc"),
            "terminal_results": ["a resolved winner", "the control wins",
                                 "unresolved_equivalence with winner=None"],
            "unresolved_is_a_result": True,
            "tie_break_authority": {
                "may_break_a_tie": ["seed sc, once, for tied candidates only"],
                "may_NOT_break_a_tie": [
                    "search-side KL — it ranks states, not behaviour",
                    "search-side NLL — E7 measured a -5.22 nat swing moving "
                    "behaviour by +0.0000",
                    "the canonical Stage-1 NLL diagnostic — diagnostic only, and "
                    "not run by this session",
                    "a fourth seed — never",
                ],
            },
        },
        "session_plan": {
            "plan_id": PHASE_B_PLAN_V1.plan_id,
            "plan_hash": PHASE_B_PLAN_V1.plan_hash,
            "version": PHASE_B_PLAN_V1.version,
            "stages": [s.as_dict() for s in PHASE_B_PLAN_V1.stages],
        },
        "not_authorized": (
            "No Phase-B grant or authorization exists, no cumulative-budget "
            "increase has been requested, and this document neither creates nor "
            "implies one."),
    }
    prereg["preregistration_sha256"] = sha256_json(prereg)
    return prereg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_b_preregistration.json")
    args = ap.parse_args()
    prereg = build()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(prereg, indent=2) + "\n")

    print(f"session plan hash    {prereg['session_plan']['plan_hash']}")
    print(f"science plan hash    {prereg['science_plan']['plan_hash']}  (reused unchanged)")
    print(f"executable digest    {prereg['executable_source']['digest']}  "
          f"({len(prereg['executable_source']['files'])} files)")
    for profile in prereg["calibration_profiles"]:
        print(f"  {profile['qualified_id']:28} spec {profile['spec_identity']['profile_hash'][:12]}"
              f"  content {profile['materialized_identity']['content_sha256'][:12]}")
    print(f"candidates at sa     {prereg['candidate_set']['total_at_sa']}"
          f"  (excluded: {', '.join(prereg['candidate_set']['excluded_phase_a_leaves'])})")
    print(f"preregistration      {prereg['preregistration_sha256']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""D2's versioned preregistration: everything frozen before any D2 result.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/write_d2_preregistration.py [--write]

VERSION 2 (contract revision 6): the redesigned D2 -- the frozen teacher's
full-vocabulary confidence weighting the root-reference KL, full scope.
Version 1 (`d2_preregistration_v1.json`, hash 489f49ba...) froze the
student-uncertainty D2 and is SUPERSEDED -- NEVER EXECUTED; its file is kept,
its science and hash unchanged, its status field says so (`supersede_v1`).

Assembled from the owners of each fact, never restated:

* the objective, its two declared changes from D1, its scope, the ranking
  policy and the claim boundary -- the D2/D3 objective contract
  (`stages.d_series.objectives`);
* the search space, seed, beam, profiles and frozen operators -- D1's design
  and D1's search record, which D2 keeps;
* the behavioural rungs, comparator units, alpha and outcomes -- the shared
  evaluation protocol (`stages.d_series.shared_evaluation`);
* the price -- `stages.d_series.d2_budget`.

The record has a hash over its science. It is DRAFT until the maintainer
approves it; approval is recorded in decisions.md and does not edit this
record's science. AUTHORIZES NOTHING: every paid session needs its own grant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

SCHEMA = "aadistill.phase_d2.preregistration/v2"
RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v2.json"
#: The superseded version: kept, never rewritten except its status fields.
SUPERSEDED_RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v1.json"
SUPERSEDED_HASH_PREFIX = "489f49ba"
#: The D2 GPU validation plan (a PROPOSAL; its own grant runs it).
GPU_VALIDATION_PLAN = ("logs/stages/stage-1/families/d_series/validations/"
                       "d2-gpu-validation/v1/plan.json")
#: The record's fields that are NOT science: its hash covers everything else.
NON_SCIENCE_KEYS = ("status", "preregistration_hash", "budget", "authorizes",
                    "superseded_drafts")
#: Earlier DRAFTS of this version, never approved: kept by hash so a reader
#: holding one can find what replaced it. Outside the science hash.
SUPERSEDED_DRAFTS = (
    {"preregistration_hash":
         "317596c4921e7d8bf77745d963581346582cf5f8128a84bffe388ddb0a8bcc03",
     "contract_hash":
         "741cd5c5faa433285741b4b44d6fe07d30df2f1191ca6f5f71a1e6f3e48c5493",
     "commit": "16d6728583adbcbb69abf738d9d171b850abbe45",
     "superseded_on": "2026-10-14",
     "why": ("independent review of 16d67285: contract revision 7 separates "
             "D2's and D3's motivation, scope, diagnostics and claim boundary; "
             "the D2 GPU validation's numeric acceptance protocol is "
             "registered in three classes. No objective, rule, scope, K or "
             "budget figure of D2 changed"),
     "approved": False},
)
D1_DESIGN = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
D1_SEARCH = ("logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210/"
             "evidence/d1_search.json")
D1_RETENTION = ("logs/stages/stage-1/phase_d1/decisions/"
                "post_search_finalist_retention.json")
#: D2's behavioural field size. PREPARED at 4 per the independent review;
#: frozen when this record is approved.
K_CANDIDATES = 4


def _read(rel: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / rel).read_text())


def build() -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD
    from stages.d_series import objectives as O
    from stages.d_series import scoring_protocol as SP
    from stages.d_series import shared_evaluation as S

    contract = _read(O.CONTRACT)
    protocol = _read(S.RECORD)
    design = _read(D1_DESIGN)
    d1_search = _read(D1_SEARCH)
    retention = _read(D1_RETENTION)
    budget = BUD.build(k_candidates=K_CANDIDATES)
    stage = design["search_stage"]

    d2 = contract["d2"]
    science = {
        "schema": SCHEMA,
        "experiment_id": "phase_d2",
        "stage_id": "1",
        "version": 2,
        "supersedes": {
            "record": SUPERSEDED_RECORD,
            "preregistration_hash": _read(SUPERSEDED_RECORD)[
                "preregistration_hash"],
            "objective": contract["d2_superseded"]["objective_id"],
            "objective_hash": contract["d2_superseded"]["objective_hash"],
            "status": contract["d2_superseded"]["status"],
        },
        "question": (
            "Under the capacity limit, does weighting each scored position's "
            "root-reference KL -- and the structural operators' activation "
            "statistics -- by the FROZEN ORIGINAL TEACHER's full-vocabulary "
            "confidence select a structural initialization that, after the "
            "frozen recovery recipe, recovers more autonomous task "
            "correctness than the standing incumbent B?"),
        "motivation": d2["motivation"],
        "hypothesis": d2["hypothesis"],
        "objective": {
            "definition": d2["objective"],
            "objective_hash": d2["objective_hash"],
            "formula": d2["formula"],
            "fixed": d2["fixed"],
            "contract": O.CONTRACT,
            "contract_hash": contract["contract_hash"],
            "scope": d2["scope"],
            "where_it_applies": d2["where_it_applies"],
            "aggregation_control": contract["aggregation_control"],
        },
        #: TWO declared changes. Never "one variable".
        "declared_changes_from_d1": d2["declared_changes"],
        "unchanged_from_d1": [
            "root teacher and target geometry",
            "frozen operator set and calibration-profile assignment space",
            "beam schedule and width",
            "search seed",
            "position policy positions.supervised_target_v1@v1 (binary)",
            "reference Top-200 + tail support",
            "micro-batch 3, length_sorted_v1",
            "recovery recipe and behavioural protocol"],
        "scope_decision": {
            "frozen": d2["scope"]["frozen_scope"],
            "objective_hash": d2["objective_hash"],
            "narrow_scope_comparison": d2["scope"]["narrow_scope_comparison"],
            "review": d2["scope"]["full_scope_review"],
        },
        "teacher_confidence_validity": {
            "checks": d2["validity_checks_before_search"],
            "when": ("in the D2 GPU validation session, on the root reference "
                     "stores of both profiles, BEFORE the formal search is "
                     "authorized"),
            "owner": "scripts/stages/stage-1/families/d_series/teacher_confidence.py",
            "rule": ("descriptive; a result may inform the maintainer's "
                     "authorization decision but never changes a definition, "
                     "the scope or any rule in this record"),
        },
        "search": {
            "frozen_implementations": stage["frozen_implementations"],
            "profiles": stage["profiles"],
            "beam": stage["beam"],
            "seed": int(d1_search["wiring"]["seed"]),
            "seed_source": D1_SEARCH,
            "support": SP.D_SERIES_SUPPORT.as_dict(),
            "position_policy": "positions.supervised_target_v1@v1",
            "ranking_policy": {"id": O.OBJECTIVE_PARETO_V1.qualified_id,
                               "hash": O.OBJECTIVE_PARETO_V1.policy_hash},
            #: The RANKING IDENTITY: the generic objective-ranking policy
            #: (shared with D3 by design -- same shape, same epsilon) bound
            #: to the objective whose metrics it ranks. New for D2 v2
            #: because the objective is new; the policy is not forked.
            "ranking_identity": {
                "policy_hash": O.OBJECTIVE_PARETO_V1.policy_hash,
                "objective_hash": d2["objective_hash"],
                "ranking_identity_hash": sha256_json({
                    "policy_hash": O.OBJECTIVE_PARETO_V1.policy_hash,
                    "objective_hash": d2["objective_hash"]})},
            "root_reference": {
                "what": ("one store per calibration profile, built once from "
                         "the frozen teacher: Top-200 + tail sketch, "
                         "full-vocabulary entropy and confidence, target "
                         "log-probability and top-1 (diagnostics)"),
                "identity": ("root teacher id + sha256, profile hash, support, "
                             "entropy definition, confidence mapping and row "
                             "set; a declaration, not a tensor hash -- the "
                             "content digest is evidence"),
                "consumers": ["DEPTH's KL reference and weights",
                              "FFN / WIDTH / ATTENTION statistics weights"]},
            "measurer": ("StateEvaluator with the SAME scoring objective "
                         "(teacher full-vocabulary entropy computed once per "
                         "item); BeamSearch refuses any mismatch"),
            "execution": {
                "micro_batch_size": SP.D_SERIES_MICRO_BATCH_SIZE,
                "calibration_batch_packing": SP.D_SERIES_BATCH_PACKING,
                "prediction_rows": (
                    "all_v1, unless the D2 GPU validation's scored_v1 "
                    "equivalence for THIS objective passes under R0's "
                    "tolerances before the search authorization, then "
                    "scored_v1. R0 (KL path) and stage L (the superseded "
                    "objective) do not establish it for this objective. An "
                    "execution choice, recorded in the run's execution "
                    "fingerprint; it changes no estimand")},
            "stops_at": "commit_top_k; the search trains nothing",
        },
        "retention": {
            "rule": retention["the_standing_policy"]["rule"],
            "post_search": retention["the_standing_policy"]["post_search"],
            "k": K_CANDIDATES,
            "order": ("OBJECTIVE_PARETO_V1 quality order over complete leaves, "
                      "deterministic tie-break; no lineage diversity"),
            "owner": D1_RETENTION,
        },
        "search_diagnostics": {
            "kl_only_control": (
                "every state carries state.objective.raw_kl.* (the unweighted "
                "teacher KL, equal to D1's state.teacher_kl.*) beside the "
                "objective; the D1 ranking of D2's complete leaves under "
                "PARETO_V1 on those raw-KL metrics is reported, never used"),
            "per_candidate": ("DEPTH records the weighted objective, the "
                              "unweighted root KL, reference entropy, "
                              "reference confidence, the weight, reference "
                              "tail mass and joint histograms for every "
                              "candidate subset, and the PARENT-reference KL "
                              "for every round-0 candidate"),
            "parent_kl_scope": (
                "round 0 of every DEPTH expansion only: one parent reference "
                "pass per expansion and one extra reduction per round-0 "
                "candidate. Enough to set the D1-style parent ranking of the "
                "first removal beside the root one; the later rounds are not "
                "covered, by design"),
            "root_reference_evidence": (
                "each store's identity, content digest, build seconds and "
                "confidence distribution (weight sum, mean, ESS) in the "
                "search record and the telemetry"),
            "flags": ("diagnostic_threshold_rule, thresholds from the "
                      "reference calibration frozen BEFORE this search; "
                      "computed offline; never a selection rule"),
            "no_third_kl_only_search": (
                "the KL-only comparison is the built-in control on D2's own "
                "states and D1's completed search -- not a new search"),
        },
        "behaviour": {
            "protocol": {"id": f"{protocol['protocol_id']}@v{protocol['version']}",
                         "protocol_hash": protocol["protocol_hash"]},
            "screening": {
                "battery": protocol["rungs"]["shared_screening"]["reuses_battery"],
                "seeds": protocol["rungs"]["shared_screening"]["seeds"],
                "field": f"the {K_CANDIDATES} retained D2 candidates",
                "comparator": ("B's two imported screening units -- not "
                               "retrained, not regenerated"),
                "probes": 2 * K_CANDIDATES,
                "rule": protocol["statistics"]["screening"],
            },
            "confirmation": {
                "battery": protocol["rungs"]["shared_confirmation"][
                    "reuses_battery"],
                "seeds": protocol["rungs"]["shared_confirmation"]["seeds"],
                "candidate": "exactly the one screening advances, or none",
                "comparator": "B's three imported confirmation units",
                "secondary": "q2's three imported units, descriptive only",
                "probes": 3,
                "alpha_one_sided": protocol["statistics"]["family"][
                    "per_experiment_alpha_one_sided"],
                "outcomes": protocol["statistics"]["confirmation_outcomes"],
                "bootstrap_seed": S.bootstrap_seed(
                    protocol["protocol_hash"], "phase_d2",
                    "shared_confirmation"),
            },
            "recovery_recipe": protocol["recovery"],
            "diagnostics": contract["behavioural_diagnostics"],
            "rescoring_integration": contract["rollout_rescoring_integration"],
            "reference_units": [u["probe_id"] for r in protocol["rungs"].values()
                                for u in r["reference_units"]],
        },
        "prerequisites": [
            {"what": "reference calibration record", "why": (
                "diagnostic thresholds frozen before any D2 candidate exists "
                "(done: stage L, refcal_20261013_170714)"),
             "owner": O.REFERENCE_CALIBRATION},
            {"what": "the D2 GPU validation (a proposal; its own grant)",
             "why": (
                 "the redesigned objective is unmeasured on a GPU: root "
                 "store build on both profiles and the teacher-confidence "
                 "validity checks; root-vs-parent DEPTH on a compressed "
                 "parent; batch 1 vs 3; all_v1 vs scored_v1 for THIS "
                 "objective; the weighted FFN/WIDTH/ATTENTION statistics; "
                 "state-evaluation cost. Its measurements replace the "
                 "provisional terms of the search price"),
             "owner": GPU_VALIDATION_PLAN},
            {"what": "maintainer approval of this preregistration", "why": (
                "freezes K, the full scope and every rule above")},
            {"what": "a search price under the per-session ceiling", "why": (
                "the redesigned D2 is priced from D1's records at the cached "
                "DEPTH cell (validated by stage L) plus PROVISIONAL added "
                "terms; it fits the ceiling, and is re-derived from the GPU "
                "validation's measurements before the search grant"),
             "owner": BUD.RECORD},
            {"what": "the programme reservation decision", "why": (
                "D2 must not strand D3 or a fresh confirmation"),
             "owner": BUD.SCENARIO_RECORD},
            {"what": "a grant per paid session", "why": "nothing here pays"},
        ],
        "claim_boundary": {"shared_with_d3": contract["claim_boundary"],
                           "d2": d2["claim_boundary"]},
    }
    return {
        **science,
        "status": ("DRAFT -- not approved, not authorized; supersedes v1 "
                   "(never executed)"),
        "superseded_drafts": list(SUPERSEDED_DRAFTS),
        "preregistration_hash": sha256_json(science),
        "budget": {"owner": BUD.RECORD,
                   "steps": {k: budget["steps"][k]
                             for k in ("d2_search", "d2_screening",
                                       "d2_confirmation")},
                   "enforceable_stop": (
                       "each session's authorization caps spend at the live "
                       "rate; planning bounds are not the cap")},
        "authorizes": "nothing",
    }


#: The machine-readable supersession stamped on v1, OUTSIDE its science: v1's
#: `preregistration_hash` covers only its science, so it stays verifiable.
SUPERSESSION_FIELDS = {
    "status": "SUPERSEDED -- NEVER EXECUTED",
    "superseded_by": RECORD,
    "superseded_on": "2026-10-14",
    "superseded_because": (
        "maintainer scientific decision 2026-10-14: D2 redesigned as the "
        "frozen-teacher-confidence-weighted root-reference KL (contract "
        "revision 6). No D2 search, recovery or behavioural session ran "
        "under this version"),
    "executed": False,
}


def supersede_v1(root: Path = REPO_ROOT) -> dict[str, Any]:
    """v1 with its status fields replaced and its science untouched."""
    doc = json.loads((root / SUPERSEDED_RECORD).read_text())
    science = {k: v for k, v in doc.items()
               if k not in ("status", "preregistration_hash", "budget",
                            "authorizes", *SUPERSESSION_FIELDS)}
    if sha256_json(science) != doc["preregistration_hash"] or not doc[
            "preregistration_hash"].startswith(SUPERSEDED_HASH_PREFIX):
        raise RuntimeError(f"{SUPERSEDED_RECORD}'s science no longer hashes "
                           "to its recorded preregistration_hash")
    return {**doc, **SUPERSESSION_FIELDS, "authorizes": "nothing"}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    bodies = {RECORD: json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
              SUPERSEDED_RECORD: json.dumps(supersede_v1(), indent=1,
                                            ensure_ascii=False) + "\n"}
    stale = 0
    for rel, body in bodies.items():
        path = REPO_ROOT / rel
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            print(f"wrote {rel}")
            continue
        same = path.is_file() and path.read_text() == body
        stale += not same
        print(f"{rel}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if args.write or not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())

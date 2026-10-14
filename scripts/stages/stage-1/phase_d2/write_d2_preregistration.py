#!/usr/bin/env python3
"""D2's versioned preregistration: everything frozen before any D2 result.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/write_d2_preregistration.py [--write]

VERSION 3 (contract revision 8): D2 v3 -- the frozen teacher's Top-200 + tail
confidence weighting the root-reference union Top-K KL, full scope; an
Evidence-Informed Pre-Search Scientific Revision. Version 2
(`d2_preregistration_v2.json`, hash 721ffddc...) is SUPERSEDED BEFORE FORMAL
SEARCH -- only its Stage V engineering validation ran; version 1
(`d2_preregistration_v1.json`, 489f49ba...) is SUPERSEDED -- NEVER EXECUTED.
Both files are kept with their science and hashes unchanged; only their
status fields say so (`supersede_v1`, `supersede_v2`).

Assembled from the owners of each fact, never restated:

* the objective, its support and confidence definitions, its three declared
  changes from D1, its scope, the ranking identity and the claim boundary --
  the D2/D3 objective contract (`stages.d_series.objectives`);
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

SCHEMA = "aadistill.phase_d2.preregistration/v3"
RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v3.json"
#: Version 2: kept, never rewritten except its status fields.
SUPERSEDED_V2_RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v2.json"
SUPERSEDED_V2_HASH_PREFIX = "721ffddc"
#: Version 1: kept, never rewritten except its status fields.
SUPERSEDED_RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v1.json"
SUPERSEDED_HASH_PREFIX = "489f49ba"
#: Version 3's GPU validation plan (a PROPOSAL; its own grant runs it).
GPU_VALIDATION_PLAN = ("logs/stages/stage-1/families/d_series/validations/"
                       "d2v3-gpu-validation/v1/plan.json")
#: Version 2's Stage V: CLOSED, adopted engineering evidence that version 3
#: reuses for its mechanisms (never for its numeric definitions).
STAGE_V_CAMPAIGN = ("logs/stages/stage-1/families/d_series/validations/"
                    "d2-gpu-validation/v1/campaign.json")
#: The record's fields that are NOT science: its hash covers everything else.
NON_SCIENCE_KEYS = ("status", "preregistration_hash", "budget", "authorizes",
                    "superseded_drafts")
#: Earlier DRAFTS of THIS version, never approved. Outside the science hash.
SUPERSEDED_DRAFTS: tuple[dict[str, Any], ...] = ()
D1_DESIGN = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
D1_SEARCH = ("logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210/"
             "evidence/d1_search.json")
D1_RETENTION = ("logs/stages/stage-1/phase_d1/decisions/"
                "post_search_finalist_retention.json")
#: D2's behavioural field size. PREPARED at 4 per the independent review;
#: frozen when this record is approved. Independent of the distribution's
#: Top-K (200).
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
    v2 = _read(SUPERSEDED_V2_RECORD)

    d2 = contract["d2"]
    science = {
        "schema": SCHEMA,
        "experiment_id": "phase_d2",
        "stage_id": "1",
        "version": 3,
        "revision_kind": d2["revision_kind"],
        "supersedes": {
            "v2": {"record": SUPERSEDED_V2_RECORD,
                   "preregistration_hash": v2["preregistration_hash"],
                   "objective": contract["d2_v2_superseded"]["objective_id"],
                   "objective_hash": contract["d2_v2_superseded"][
                       "objective_hash"],
                   "status": contract["d2_v2_superseded"]["status"],
                   "executed": contract["d2_v2_superseded"]["executed"]},
            "v1": {"record": SUPERSEDED_RECORD,
                   "preregistration_hash": _read(SUPERSEDED_RECORD)[
                       "preregistration_hash"],
                   "objective": contract["d2_superseded"]["objective_id"],
                   "objective_hash": contract["d2_superseded"][
                       "objective_hash"],
                   "status": contract["d2_superseded"]["status"]},
        },
        "question": d2["motivation"]["question"],
        "motivation": d2["motivation"],
        "hypothesis": d2["hypothesis"],
        "objective": {
            "definition": d2["objective"],
            "objective_hash": d2["objective_hash"],
            "formula": d2["formula"],
            "fixed": d2["fixed"],
            "teacher_confidence": d2["teacher_confidence"],
            "distribution_support": d2["distribution_support"],
            "contract": O.CONTRACT,
            "contract_hash": contract["contract_hash"],
            "contract_revision": contract["revision"],
            "scope": d2["scope"],
            "where_it_applies": d2["where_it_applies"],
            "aggregation_control": contract["aggregation_control"],
        },
        #: THREE declared changes. Never "one variable".
        "declared_changes_from_d1": d2["declared_changes"],
        "unchanged_from_d1": [
            "root teacher and target geometry",
            "frozen operator set and calibration-profile assignment space",
            "beam schedule and width",
            "search seed",
            "position policy positions.supervised_target_v1@v1 (binary)",
            "K = 200 (the D-series K; now the union support's and the "
            "confidence's)",
            "micro-batch 3, length_sorted_v1",
            "recovery recipe and behavioural protocol"],
        "d3_and_d1_untouched": {
            "d1": "no objective, reference_topk_tail_v1; nothing re-run",
            "d3": ("objective, lambda 7.34, reference_topk_tail_v1 support, "
                   "parent-reference DEPTH and ranking identity unchanged"),
            "d3_ranking_identity": contract["ranking"]["identities"]["d3"]},
        "scope_decision": {
            "frozen": d2["scope"]["frozen_scope"],
            "objective_hash": d2["objective_hash"],
            "narrow_scope_comparison": d2["scope"]["narrow_scope_comparison"],
            "review": d2["scope"]["full_scope_review"],
        },
        "teacher_confidence_validity": {
            "checks": d2["validity_checks_before_search"],
            "when": ("in the version-3 GPU validation session, on the union "
                     "root stores of both profiles, BEFORE the formal search "
                     "is authorized"),
            "owner": "scripts/stages/stage-1/families/d_series/teacher_confidence.py",
            "rule": ("descriptive; a result may inform the maintainer's "
                     "authorization decision but never changes a definition, "
                     "the scope, K or any rule in this record"),
        },
        "search": {
            "frozen_implementations": stage["frozen_implementations"],
            "profiles": stage["profiles"],
            "beam": stage["beam"],
            "seed": int(d1_search["wiring"]["seed"]),
            "seed_source": D1_SEARCH,
            "support": O.D2_SUPPORT.as_dict(),
            "position_policy": "positions.supervised_target_v1@v1",
            "ranking_policy": {"id": O.OBJECTIVE_PARETO_V1.qualified_id,
                               "hash": O.OBJECTIVE_PARETO_V1.policy_hash},
            #: VERSIONED: the shared objective-ranking policy (D3 ranks with
            #: it too) bound to the objective AND the support whose numbers
            #: fill the ranked keys. The search geometry does not change.
            "ranking_identity": d2["ranking_identity"],
            "root_reference": {
                "what": ("one union store per calibration profile, built once "
                         "from the frozen teacher: deterministic Top-200 + "
                         "tail sketch from its float32 head logits, its K+1 "
                         "entropy and confidence, its final hidden states and "
                         "float32 log-normalizers (for exact retrieval of "
                         "candidate-only entries), target log-probability "
                         "and top-1 (diagnostics)"),
                "identity": ("root teacher id + sha256, profile hash, union "
                             "support with K and tie-break, entropy definition, "
                             "category rule, confidence mapping, retrieval "
                             "definition, row set, and the executed build -- "
                             "output head (logit definition, shape, dtype, "
                             "bias, scale, soft-cap, tying), adapter "
                             "capability, model dtype, device type, "
                             "micro-batch and packing; a declaration, not a "
                             "tensor hash -- the content digest is evidence"),
                "consumers": ["DEPTH's KL reference, retrieval and weights",
                              "FFN / WIDTH / ATTENTION statistics weights"]},
            "measurer": ("StateEvaluator with the SAME objective and support: "
                         "the teacher forwarded ONCE into a union store over "
                         "the suite, never per evaluation; BeamSearch refuses "
                         "any mismatch"),
            "execution": {
                "micro_batch_size": SP.D_SERIES_MICRO_BATCH_SIZE,
                "calibration_batch_packing": SP.D_SERIES_BATCH_PACKING,
                "prediction_rows": (
                    "all_v1, unless the version-3 GPU validation's scored_v1 "
                    "equivalence for THIS objective passes under the "
                    "registered tolerances before the search authorization, "
                    "then scored_v1. Stage V's equivalence covers version 2 "
                    "only. An execution choice, recorded in the run's "
                    "execution fingerprint; it changes no estimand")},
            "stops_at": "commit_top_k; the search trains nothing",
        },
        "retention": {
            "rule": retention["the_standing_policy"]["rule"],
            "post_search": retention["the_standing_policy"]["post_search"],
            "k": K_CANDIDATES,
            "order": ("OBJECTIVE_PARETO_V1 quality order over complete leaves "
                      "(D2's ranking identity), deterministic tie-break; no "
                      "lineage diversity"),
            "owner": D1_RETENTION,
        },
        "search_diagnostics": {
            "kl_only_control": (
                "every state carries state.objective.raw_kl.* (the "
                "UNWEIGHTED root union KL, equal to the state's own "
                "state.teacher_kl.*) and state.objective.kl_reference_topk.* "
                "(the teacher Top-K + tail KL of the same pair) beside the "
                "objective; rankings of D2's complete leaves on those are "
                "reported, never used"),
            "per_candidate": ("DEPTH records the weighted objective, the "
                              "unweighted root union KL, the teacher Top-K + "
                              "tail KL, the union size and overlap, both "
                              "union tail masses, reference entropy and "
                              "confidence, the weight and joint histograms "
                              "for every candidate subset, and the "
                              "PARENT-reference KL for every round-0 "
                              "candidate"),
            "parent_kl_scope": (
                "round 0 of every DEPTH expansion only, on the parent's own "
                "Top-200 + tail: one parent reference pass per expansion and "
                "one extra reduction per round-0 candidate. Enough to set "
                "the D1-style parent ranking of the first removal beside the "
                "root one; the later rounds are not covered, by design"),
            "root_reference_evidence": (
                "each store's identity (with what it executed), content "
                "digest, retrieval bytes, build seconds and confidence "
                "distribution (weight sum, mean, ESS) in the search record "
                "and the telemetry"),
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
        "stage_v_reuse": {
            "campaign": STAGE_V_CAMPAIGN,
            "reused": ("its MECHANISMS, already exercised on the paid GPU: "
                       "the launcher, acquisition loop, CUDA device handoff, "
                       "the cache guard under a hoarded pool, teardown "
                       "confirmation, campaign booking and adoption; and its "
                       "MEASURED v2 timings as the base of version 3's price"),
            "not_reused": ("any numeric verdict about version 3's definitions: "
                           "the union KL, the retrieval, the K+1 confidence, "
                           "the deterministic Top-K and version 3's scored_v1 "
                           "equivalence need the version-3 GPU validation")},
        "prerequisites": [
            {"what": "reference calibration record", "why": (
                "diagnostic thresholds frozen before any D2 candidate exists "
                "(done: stage L, refcal_20261013_170714)"),
             "owner": O.REFERENCE_CALIBRATION},
            {"what": "the version-3 GPU validation (a proposal; its own grant)",
             "why": (
                 "version 3's numerics are CPU-verified only: union store "
                 "build on both profiles with the teacher-confidence validity "
                 "checks; union KL bounds against the full KL on real "
                 "teacher logits; retrieval against the full head; bf16 ties; "
                 "batch 1 vs 3; all_v1 vs scored_v1 for THIS objective; DEPTH "
                 "and state-evaluation cost. Its measurements replace the "
                 "provisional terms of the search price"),
             "owner": GPU_VALIDATION_PLAN},
            {"what": "maintainer approval of this preregistration", "why": (
                "freezes K, the support, the confidence, the full scope and "
                "every rule above")},
            {"what": "a search price under the per-session ceiling", "why": (
                "version 3 is priced from D1's records and Stage V's measured "
                "version-2 terms plus PROVISIONAL union terms, and is "
                "re-derived from the version-3 GPU validation before the "
                "search grant"),
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
        "status": ("DRAFT -- not approved, not authorized; supersedes v2 "
                   "(superseded before formal search) and v1 (never "
                   "executed)"),
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
    #: v1 was superseded by VERSION 2, whatever the live version is now.
    "superseded_by": SUPERSEDED_V2_RECORD,
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


#: Version 2's supersession, stamped OUTSIDE its science like v1's.
SUPERSESSION_V2_FIELDS = {
    "status": ("SUPERSEDED BEFORE FORMAL SEARCH -- STAGE V ENGINEERING "
               "VALIDATION COMPLETED"),
    "superseded_by": RECORD,
    "superseded_on": "2026-10-14",
    "superseded_because": (
        "maintainer scientific decision 2026-10-14: D2 version 3, the "
        "teacher's Top-200 + tail confidence weighting the root-reference "
        "union Top-K KL (contract revision 8), an Evidence-Informed "
        "Pre-Search Scientific Revision. Version 2 was never approved; no "
        "D2 search, recovery or behavioural session ran under it. Its Stage "
        "V engineering validation (d2val_20261014_103301, $0.6532, "
        "validation PASS) completed and stays version 2's evidence"),
    "executed": {"formal": False,
                 "engineering_validation": "Stage V, COMPLETE and CLOSED"},
}


def supersede_v2(root: Path = REPO_ROOT) -> dict[str, Any]:
    """v2 with its status fields replaced and its science untouched."""
    doc = json.loads((root / SUPERSEDED_V2_RECORD).read_text())
    science = {k: v for k, v in doc.items()
               if k not in (*NON_SCIENCE_KEYS, *SUPERSESSION_V2_FIELDS)}
    if sha256_json(science) != doc["preregistration_hash"] or not doc[
            "preregistration_hash"].startswith(SUPERSEDED_V2_HASH_PREFIX):
        raise RuntimeError(f"{SUPERSEDED_V2_RECORD}'s science no longer "
                           "hashes to its recorded preregistration_hash")
    return {**doc, **SUPERSESSION_V2_FIELDS, "authorizes": "nothing"}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    bodies = {RECORD: json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
              SUPERSEDED_V2_RECORD: json.dumps(supersede_v2(), indent=1,
                                               ensure_ascii=False) + "\n",
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

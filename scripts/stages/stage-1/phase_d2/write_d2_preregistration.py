#!/usr/bin/env python3
"""D2's versioned preregistration: everything frozen before any D2 result.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/write_d2_preregistration.py [--write]

Assembled from the owners of each fact, never restated:

* the objective, the ranking policy and the claim boundary -- the D2/D3
  objective contract (`stages.d_series.objectives`);
* the search space, seed, beam, profiles and frozen operators -- D1's design
  and D1's search record, because D2 changes ONE thing: what is minimized;
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

SCHEMA = "aadistill.phase_d2.preregistration/v1"
RECORD = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v1.json"
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

    science = {
        "schema": SCHEMA,
        "experiment_id": "phase_d2",
        "stage_id": "1",
        "version": 1,
        "question": (
            "Under the capacity limit, does weighting each scored position's "
            "KL by the candidate student's own Top-200 uncertainty select a "
            "structural initialization that, after the frozen recovery "
            "recipe, recovers more autonomous task correctness than the "
            "standing incumbent B?"),
        "motivation": contract["scientific_motivation"],
        "hypothesis": contract["d2"]["hypothesis"],
        "objective": {
            "definition": contract["d2"]["objective"],
            "objective_hash": contract["d2"]["objective_hash"],
            "formula": contract["d2"]["formula"],
            "contract": O.CONTRACT,
            "contract_hash": contract["contract_hash"],
            "scope": contract["objective_scope"],
            "aggregation_control": contract["aggregation_control"],
        },
        "one_variable": {
            "changed_from_d1": ("the scoring objective of DEPTH, state "
                                "evaluation and the beam ranking -- and only "
                                "that"),
            "unchanged": [
                "root teacher and target geometry",
                "frozen operator set and calibration-profile assignment space",
                "beam schedule and width",
                "search seed",
                "position policy positions.supervised_target_v1@v1",
                "reference Top-200 + tail support",
                "micro-batch 3, length_sorted_v1",
                "recovery recipe and behavioural protocol"],
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
            "measurer": ("StateEvaluator with the SAME scoring objective; "
                         "BeamSearch refuses any mismatch"),
            "execution": {
                "micro_batch_size": SP.D_SERIES_MICRO_BATCH_SIZE,
                "calibration_batch_packing": SP.D_SERIES_BATCH_PACKING,
                "prediction_rows": (
                    "all_v1, unless R0's verdict is PASS and recorded before "
                    "the search authorization, then scored_v1. An execution "
                    "choice validated equivalent, recorded in the run's "
                    "execution fingerprint; it changes no estimand")},
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
                "every state carries state.objective.raw_kl.* (equal to D1's "
                "state.teacher_kl.*) beside the objective; the D1 ranking of "
                "D2's complete leaves under PARETO_V1 on those raw-KL metrics "
                "is reported, never used"),
            "per_candidate": ("DEPTH records objective, raw KL, entropy, "
                              "confidence, uncertainty weight, Top-K mass, "
                              "reference tail mass and joint histograms for "
                              "every candidate subset"),
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
                "diagnostic thresholds must be frozen before any D2 candidate "
                "exists; D3's lambda is measured in the same pass"),
             "owner": O.REFERENCE_CALIBRATION},
            {"what": "R0 verdict", "why": "only to use scored_v1",
             "owner": ("logs/stages/stage-1/families/d_series/validations/"
                       "prediction-rows-r0/v1")},
            {"what": "maintainer approval of this preregistration", "why": (
                "freezes K and every rule above")},
            {"what": "a search price under the per-session ceiling", "why": (
                "priced from D1's own search records, the planning bound "
                "exceeds the per-session ceiling while DEPTH is priced at D1's "
                "recompute-mode maxima; it fits once the reference cache's "
                "allocator-hoard fix is shown on a GPU (the reference "
                "calibration's stage L), or by a maintainer cap decision"),
             "owner": BUD.RECORD},
            {"what": "the programme reservation decision", "why": (
                "D2 must not strand D3 or a fresh confirmation"),
             "owner": BUD.SCENARIO_RECORD},
            {"what": "a grant per paid session", "why": "nothing here pays"},
        ],
        "claim_boundary": contract["claim_boundary"],
    }
    return {
        **science,
        "status": "DRAFT -- not approved, not authorized",
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


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    body = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    path = REPO_ROOT / RECORD
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"wrote {RECORD}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{RECORD}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

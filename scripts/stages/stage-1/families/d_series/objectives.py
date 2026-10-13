#!/usr/bin/env python3
"""D2's and D3's scoring objectives: the instances, the ranking, and lambda.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/objectives.py [--write]

Maintainer decision 2026-10-13, superseding the 2026-10-01 roadmap's
reference-confidence D2 and full-vocabulary-entropy D3. Both experiments keep
D1's target positions, profiles, Top-200 + tail reference KL, geometry,
operators, beam and recovery recipe. They differ from D1 -- and from each
other -- only in what a KL-scored decision minimizes, built from the CANDIDATE
STUDENT's own Top-K conditional entropy (`aadistill.initialization.scoring.objective`):

    D1  L(x) = (1/T_x) sum_t KL_t
    D2  L(x) = (1/T_x) sum_t (1 - c_t) KL_t          student uncertainty weights KL
    D3  L(x) = (1/T_x) sum_t [KL_t - lambda c_t]     student confidence is a reward

``c_t = 1 - H_K(t)/log K``, ``K = 200``, ``T_x`` the item's scored positions.
D1's frozen definition is not touched; D1 never declared an objective, which
is exactly what keeps its identities byte-identical.

**lambda is not chosen here.** Its RULE is fixed here, before any D2 or D3
search exists, and its VALUE is measured once by that rule on a reference pair
that no D-series search produced -- the original teacher and the standing
incumbent B -- and is then frozen into D3's objective identity before D3's
search may start. See :func:`lambda_from_reference` and
`calibrate_d3_lambda.py`.

Everything here is this campaign's instance data; the core holds the forms.
AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    STATE_ID_TIEBREAK, VALIDITY_GUARDRAIL, BeamRankingPolicy, Objective,
)
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_UNCERTAINTY_WEIGHTED_KL,
    ScoringObjective,
)

SCHEMA = "aadistill.d_series.objective_contract/v1"
CONTRACT = "logs/stages/stage-1/families/d_series/plans/d2_d3_objective_contract_v1.json"
#: Where D3's measured lambda is recorded once it exists. Absent until then,
#: and D3's objective cannot be built without it.
LAMBDA_RECORD = ("logs/stages/stage-1/families/d_series/analyses/"
                 "d3_lambda_calibration.json")

#: K for the student's own Top-K entropy. Maintainer-selected 2026-10-13, the
#: same value as the reference support's K but a separate parameter: the two
#: Top-Ks select different entries (the reference's vs the student's own) and
#: each is bound in its own identity.
ENTROPY_TOP_K = 200

D2_OBJECTIVE = ScoringObjective(
    objective_id="objective.d2_student_uncertainty_weighted_kl", version=1,
    form=FORM_UNCERTAINTY_WEIGHTED_KL, entropy_top_k=ENTROPY_TOP_K)

D3_OBJECTIVE_ID = "objective.d3_kl_minus_student_confidence"


def d3_objective(reward_scale: float) -> ScoringObjective:
    """D3's objective at a MEASURED lambda. Built from the record, not typed."""
    return ScoringObjective(
        objective_id=D3_OBJECTIVE_ID, version=1,
        form=FORM_KL_MINUS_CONFIDENCE_REWARD, entropy_top_k=ENTROPY_TOP_K,
        reward_scale=float(reward_scale))


def frozen_d3_objective(repo_root: Path = REPO_ROOT) -> ScoringObjective:
    """D3's objective from the committed lambda record, or a refusal."""
    path = repo_root / LAMBDA_RECORD
    if not path.is_file():
        raise RuntimeError(
            f"{LAMBDA_RECORD} does not exist: D3's lambda has not been measured "
            "and frozen, so D3 has no objective and no search may start")
    record = json.loads(path.read_text())
    if record.get("rule_hash") != lambda_rule()["rule_hash"]:
        raise RuntimeError(
            "the lambda record was computed under a different rule than the "
            "one this contract registers")
    return d3_objective(record["reward_scale"])


#: ONE ranking policy for both: it ranks on the objective metrics, and which
#: objective filled them is the search's declared `scoring_objective`, checked
#: per measurement. The shape is D1's PARETO_V1 with the KL metrics replaced by
#: their objective counterparts, so D1 -> D2 -> D3 changes what is ranked and
#: nothing about how.
OBJECTIVE_PARETO_V1 = BeamRankingPolicy(
    policy_id="beam.pareto_scoring_objective", version=1,
    description=(
        "epsilon-Pareto over the declared scoring objective: equal-domain mean, "
        "worst domain and critical-token objective, each per item over a fixed "
        "count of scored positions. PARETO_V1's shape with the KL metrics "
        "replaced by the objective's; raw KL is recorded beside them, never "
        "ranked."),
    objectives=(
        Objective("state.objective.equal_domain_mean", "minimize",
                  "equal-domain mean of the per-item objective"),
        Objective("state.objective.worst_domain", "minimize",
                  "the domain with the highest objective"),
        Objective("state.objective.critical_token", "minimize",
                  "unweighted mean over critical-token classes of the per-item "
                  "objective at those positions"),
    ),
    tie_break=("state.objective.equal_domain_mean",
               "state.objective.worst_domain",
               "state.objective.critical_token", STATE_ID_TIEBREAK),
    guardrails=(VALIDITY_GUARDRAIL,),
    #: PARETO_V1's practical equivalence, carried over unchanged. For D2 the
    #: objective is a down-weighted KL, so 1e-4 nats means what it meant; for
    #: D3 it is in the same nats once lambda converts confidence into them.
    epsilon={"state.objective.equal_domain_mean": 1e-4,
             "state.objective.worst_domain": 1e-4,
             "state.objective.critical_token": 1e-4},
    diversity_key="parent_path",
    metadata={"raw_kl_status": "recorded as state.objective.raw_kl.*; never ranks"})


# --- lambda ------------------------------------------------------------------

#: The reference pair and the item sets lambda is measured on. Fixed now.
LAMBDA_REFERENCE = {
    "reference_model": "the original teacher, Qwen/Qwen3-4B-Thinking-2507 @ 768f209d",
    "student_model": ("the standing incumbent B, artifact 53e30566c5f795f1870d76c1"
                      "fa6a970ddc507fa5459047f3010ffab8aa890342, rebuilt by "
                      "stages.phase_c2.baseline.frozen_baseline_spec"),
    "item_sets": ["calib.domain_balanced@v1", "calib.reasoning_heavy@v2",
                  "state_eval_v1"],
    "positions": "positions.supervised_target_v1@v1",
    "kl": "reference_topk_tail_v1, K=200, reference = the teacher",
    "entropy": f"the student's own Top-{ENTROPY_TOP_K} conditional entropy",
}


def lambda_rule() -> dict[str, Any]:
    """The registered rule, with its own hash. Fixed before any D2/D3 search."""
    rule = {
        "schema": "aadistill.d_series.d3_lambda_rule/v1",
        "statistic": "per-item means over each item's scored positions (1/T_x)",
        "formula": "lambda = IQR(item KL) / IQR(item confidence)",
        "pooling": ("every item of the three item sets once, unweighted; the "
                    "per-set values are reported as sensitivity, never chosen"),
        "rounding": "3 significant figures",
        "reference": LAMBDA_REFERENCE,
        "why_this_scale": (
            "lambda converts confidence into KL units. Equating the two "
            "components' between-item spread on a reference pair means "
            "neither dominates the ranking by scale alone: one inter-quartile "
            "range of confidence is worth one inter-quartile range of KL. The "
            "IQR rather than the SD because per-item KL is heavy-tailed and "
            "the SD would let a few items set the trade-off."),
        "why_this_reference": (
            "a pair NO D-series search produced: the teacher every search "
            "starts from and the incumbent every candidate is compared with, "
            "at the geometry every search ends at. Measured before D2's search "
            "starts and never revised: a lambda chosen after seeing D2 or D3 "
            "search or behavioural results would be a selection on outcomes."),
        "sensitivity_reported_not_used": [
            "SD ratio", "each item set alone", "q2 as the student"],
        "refusals": ["IQR(item confidence) == 0", "a non-finite or non-positive "
                     "lambda", "fewer than 20 items"],
    }
    rule["rule_hash"] = sha256_json(rule)
    return rule


def _iqr(values: Sequence[float]) -> float:
    q = statistics.quantiles(sorted(float(v) for v in values), n=4,
                             method="inclusive")
    return q[2] - q[0]


def _sig3(x: float) -> float:
    if x == 0:
        return 0.0
    return round(x, 2 - int(math.floor(math.log10(abs(x)))))


def lambda_from_reference(item_kl: Sequence[float],
                          item_confidence: Sequence[float]) -> dict[str, Any]:
    """Apply the registered rule to per-item means. Pure; no model here."""
    if len(item_kl) != len(item_confidence):
        raise ValueError("one KL and one confidence per item")
    if len(item_kl) < 20:
        raise ValueError(f"{len(item_kl)} items; the rule needs at least 20")
    if any(not math.isfinite(float(v)) for v in item_kl):
        raise ValueError("a per-item KL is not finite; the reference pair has an "
                         "impossible item and the scale is undefined")
    spread_c = _iqr(item_confidence)
    if spread_c <= 0:
        raise ValueError("every item has the same confidence; the rule divides "
                         "by its IQR")
    raw = _iqr(item_kl) / spread_c
    lam = _sig3(raw)
    if not (math.isfinite(lam) and lam > 0):
        raise ValueError(f"lambda {raw!r} is not a positive finite scale")
    sd_c = statistics.pstdev(item_confidence)
    return {"reward_scale": lam, "unrounded": raw,
            "iqr_kl": _iqr(item_kl), "iqr_confidence": spread_c,
            "sd_ratio_sensitivity": (statistics.pstdev(item_kl) / sd_c
                                     if sd_c > 0 else None),
            "n_items": len(item_kl)}


# --- the contract record -------------------------------------------------------

def contract(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from stages.d_series import scoring_protocol as SP

    record_exists = (repo_root / LAMBDA_RECORD).is_file()
    d3 = frozen_d3_objective(repo_root) if record_exists else None
    template = {k: v for k, v in d3_objective(1.0).as_dict().items()
                if k != "reward_scale"}
    science = {
        "schema": SCHEMA,
        "decided_by": "maintainer decision 2026-10-13 (logs/budget/decisions.md)",
        "supersedes": (
            "logs/stages/stage-1/phase_c1/plans/phase_c_roadmap.md, sections "
            "'D2 -- target-aware + reference-confidence weighting' and 'D3 -- "
            "target-aware + KL + pure student confidence'. Kept verbatim there "
            "as history, with a pointer here."),
        "shared_with_d1": {
            "positions": "positions.supervised_target_v1@v1",
            "calibration_profiles": ["calib.domain_balanced@v1",
                                     "calib.reasoning_heavy@v2"],
            "kl_support": SP.D_SERIES_SUPPORT.as_dict(),
            "geometry_operators_beam": "D1's, unchanged",
            "recovery_recipe": "e1_p1_kd_heavy@0.86M",
            "behavioural_protocol": ("d_series_shared_eval@v1, unchanged by this "
                                     "contract"),
        },
        "d1": {"objective": None,
               "_meaning": ("D1 declares no scoring objective; its per-item mean "
                            "of the KL is the historical, KL-only path and its "
                            "identities are untouched")},
        "aggregation_difference_from_d1": {
            "depth": ("none: D1's DEPTH already averages per item over T_x, "
                      "then by sub-type and domain"),
            "state_evaluation": (
                "D1's state KL pools TOKENS within a sub-type; the D2/D3 "
                "objective averages PER ITEM first, as the decision requires. "
                "So D2/D3 also record D1's KL under the per-item rule "
                "(state.objective.raw_kl.*), which separates the effect of the "
                "aggregation from the effect of confidence"),
        },
        "d2": {"objective": D2_OBJECTIVE.as_dict(),
               "objective_hash": D2_OBJECTIVE.objective_hash,
               "formula": "L(x) = (1/T_x) sum_{t in T_x} (1 - c_t) KL_t, minimized",
               "hypothesis": (
                   "a position the student is confident about needs less "
                   "fidelity to the reference, even where its KL is large")},
        "d3": {"objective_template": template,
               "reward_scale": None if d3 is None else d3.reward_scale,
               "objective_hash": None if d3 is None else d3.objective_hash,
               "status": ("FROZEN" if d3 is not None else
                          "PENDING_LAMBDA: the rule is registered; the value is "
                          "measured on the reference pair before D3's search"),
               "formula": ("L(x) = (1/T_x) sum_{t in T_x} [KL_t - lambda c_t], "
                           "minimized (S = -L maximized)"),
               "hypothesis": (
                   "student self-confidence earns a positive reward of its own "
                   "that can offset part of a KL penalty"),
               "lambda_rule": lambda_rule(),
               "lambda_record": LAMBDA_RECORD},
        "ranking": {"policy": OBJECTIVE_PARETO_V1.qualified_id,
                    "policy_hash": OBJECTIVE_PARETO_V1.policy_hash,
                    "objectives": [o.key for o in OBJECTIVE_PARETO_V1.objectives],
                    "epsilon": dict(OBJECTIVE_PARETO_V1.epsilon)},
        "where_each_objective_applies": {
            "depth.causal_kl_greedy_v1": (
                "OPERATOR-LOCAL, exactly: the candidate is the parent with S "
                "bypassed, and its KL against the parent and its own Top-K "
                "entropy come from the same forward"),
            "ffn / residual_width / attention activation operators": (
                "NOT operator-local. Their selection reads parent activation "
                "statistics and no candidate exists until it is made, so there "
                "is no candidate KL or confidence to weight. They keep D1's "
                "target-aware statistic, and their state ids do not fork. The "
                "objective reaches their children through state evaluation and "
                "the beam"),
            "state evaluation": (
                "every child, against the original teacher; the student is the "
                "child itself"),
            "beam ranking and pruning": OBJECTIVE_PARETO_V1.qualified_id,
        },
        "diagnostics_recorded": [
            "raw KL", "student Top-K entropy", "student confidence",
            "uncertainty weight (D2) / confidence reward (D3)",
            "objective", "scored positions per item",
            "confidence histogram and quantiles", "student Top-K mass",
            "reference tail mass", "D1's KL metrics, unchanged, beside them"],
        "d2_degeneracy_watch": (
            "a candidate can lower D2's objective by becoming more confident "
            "at high-KL positions. D2's ranking never sees raw KL, so raw KL, "
            "confidence and the weight distribution are recorded on every "
            "candidate and state, and a low D2 score is never read as higher "
            "fidelity"),
        "independence": (
            "D3's definition, K and lambda rule are fixed in this record before "
            "D2 runs; no D2 behavioural or search result may adjust them"),
    }
    out = {**science, "contract_hash": sha256_json(science),
           "authorizes": "nothing"}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    body = json.dumps(contract(), indent=1, ensure_ascii=False) + "\n"
    path = REPO_ROOT / CONTRACT
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"wrote {CONTRACT}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{CONTRACT}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

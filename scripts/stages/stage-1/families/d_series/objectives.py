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


# --- why, what is measured where, and what may be claimed ----------------------

#: Maintainer supplement 2026-10-13: the reason the student's confidence enters.
SCIENTIFIC_MOTIVATION = {
    "title": "capacity-aware student reasoning",
    "premise": (
        "The teacher (Qwen3-4B-Thinking) and the ~0.6B student differ in "
        "capacity, expressiveness and learning ability. The student need not, "
        "and may not be able to, reproduce the teacher's reasoning "
        "distribution; on some tasks it may reach the same correct answer by "
        "a shorter, simpler or entirely different trajectory."),
    "consequence": (
        "A large teacher token-level KL does not by itself mean worse student "
        "reasoning. KL-based compression selection may over-reward structures "
        "that imitate the teacher's distribution and under-rate structures "
        "that preserve the student's own effective way of reasoning."),
    "question": (
        "Which structural initialization, under the capacity limit, best lets "
        "the student keep and recover its own effective reasoning ability?"),
    "role_of_confidence": (
        "the student's Top-K confidence is a CANDIDATE PROXY signal for "
        "structure selection. Its value is decided by independent behavioural "
        "evidence, never assumed"),
}

#: Four quantities that are NOT the same, and where each is measured.
MEASUREMENT_DISTINCTIONS = {
    "student_entropy_on_teacher_native_prefixes": (
        "what the D2/D3 search objectives use: the student's own Top-K "
        "entropy on the frozen teacher-native calibration and state-eval "
        "sequences"),
    "student_entropy_on_student_rollout_prefixes": (
        "a behavioural DIAGNOSTIC only: the student's own Top-K entropy at the "
        "positions of its own greedy rollout, by "
        "scripts/shared/evaluation/rollout_rescoring.py after generation. "
        "Never an input to selection in D2/D3"),
    "student_teacher_distribution_fit": (
        "the Top-K + tail KL: on teacher-native prefixes in the search; on "
        "the student's own prefixes in the rollout diagnostic, beside the "
        "teacher's surprisal of the student's tokens"),
    "autonomous_rollout_correctness": (
        "the primary behavioural endpoint, correct_overall on the shared "
        "protocol, after the frozen recovery recipe. The only quantity a "
        "D2/D3 promotion question is decided on"),
}

#: What D2/D3 behavioural rounds analyse beyond the endpoint, with the
#: instrument and the arms it can be applied to. EXPLORATORY: none of these
#: decides SHARED_GO / NO_GO.
BEHAVIOURAL_DIAGNOSTICS = {
    "instruments": {
        "rollout_profile": (
            "scripts/stages/stage-1/families/d_series/rollout_profile.py -- "
            "text-level, $0, from secured generations: generated length "
            "(engine count), reasoning/answer length, natural termination, "
            "degeneration, correctness by reasoning length, and paired "
            "same-prompt comparisons"),
        "rollout_rescoring": (
            "scripts/shared/evaluation/rollout_rescoring.py -- per position of "
            "the student's own rollout: student confidence, student token "
            "log-prob, teacher surprisal, KL(teacher || student). Needs the "
            "trained checkpoint, so it runs ON THE POD after generation and "
            "before the checkpoint is released. Reads generations; changes "
            "neither generation nor scoring, so the shared evaluation "
            "protocol's identity is untouched. Estimated 3-5 GPU-minutes per "
            "probe (~0.95M tokens per probe at D1's measured lengths)"),
    },
    "analyses": [
        {"question": "does the student still solve the task when it is far "
                     "from the teacher's distribution?",
         "instrument": "rollout_rescoring (per sample: mean KL and teacher "
                       "surprisal on its own rollout) joined to correctness",
         "arms": "new D2/D3 candidates only"},
        {"question": "autonomous reasoning length, correctness and usable "
                     "rollout",
         "instrument": "rollout_profile",
         "arms": "every unit, including the imported B/q2 (baseline recorded "
                 "in analyses/d1_rollout_profile.json)"},
        {"question": "how does student confidence relate to actual "
                     "correctness?",
         "instrument": "rollout_rescoring confidence (reasoning and answer "
                       "spans) against per-sample correctness",
         "arms": "new D2/D3 candidates only"},
        {"question": "are high-confidence, high-KL positions valid "
                     "alternative reasoning or wrong overconfidence?",
         "instrument": ("rollout_rescoring quadrant fraction against "
                        "correctness; thresholds registered in each "
                        "experiment's preregistration before any result, none "
                        "in code"),
         "arms": "new D2/D3 candidates only"},
        {"question": "did D2/D3 improve autonomous generation after recovery, "
                     "not only the search objective?",
         "instrument": ("the shared-protocol endpoint and the rollout profile "
                        "against B (and q2), read beside each candidate's "
                        "search objective and raw KL"),
         "arms": "candidates vs the imported B/q2 units"},
    ],
    "historical_units": (
        "B's and q2's ten units have decoded text and the engine's token "
        "counts, but NO token ids, log-probabilities or logits, and their "
        "checkpoints are retired. Their student confidence on their own "
        "rollouts is therefore NOT available and must not be estimated or "
        "inferred; comparisons with B/q2 are text-level and endpoint-level "
        "only. Re-tokenizing their text reproduces the engine's count exactly "
        "for 82-85% of samples"),
    "power_note": (
        "D1's units answer ~2-4% of scorable prompts correctly (B: 103 of "
        "2550 on confirmation), and q2 and B are both right on only 29 "
        "prompt-seeds there. Any confidence-correctness or quadrant analysis "
        "rests on ~100 correct samples per arm and is exploratory"),
}

CLAIM_BOUNDARY = {
    "goal": ("structural initializations that, under the capacity limit, help "
             "the student keep and recover its own effective reasoning"),
    "not_the_goal": ["maximizing student confidence",
                     "minimizing KL to the teacher"],
    "may_claim": [
        "a D2/D3 candidate's endpoint result against B on the shared "
        "protocol, with that protocol's limits (SHARED_GO nominates, never "
        "promotes)",
        "descriptive rollout and rescoring diagnostics of the candidates that "
        "were actually measured, labelled exploratory",
    ],
    "may_not_claim": [
        "that a lower D2/D3 search objective means better reasoning or "
        "higher fidelity",
        "that the student reasons differently, or better, where it is "
        "confident and far from the teacher, without rollout-level evidence "
        "linking those positions to correct completions",
        "any confidence or per-position statement about B or q2, whose "
        "checkpoints are retired and whose generations carry no "
        "log-probabilities",
        "a mechanism from the exploratory diagnostics alone",
    ],
    "validation": ("Top-K confidence is a proxy for structure selection; its "
                   "usefulness is established only by independent behavioural "
                   "evidence after recovery training"),
}


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
        "scientific_motivation": SCIENTIFIC_MOTIVATION,
        "d2": {"objective": D2_OBJECTIVE.as_dict(),
               "objective_hash": D2_OBJECTIVE.objective_hash,
               "formula": "L(x) = (1/T_x) sum_{t in T_x} (1 - c_t) KL_t, minimized",
               "hypothesis": (
                   "The more certain the student is of its own prediction at a "
                   "position, the weaker the KL penalty for differing from the "
                   "teacher there should be. This does NOT assume the student "
                   "is right where it is confident; it tests whether relaxing "
                   "strict imitation of the teacher's distribution helps a "
                   "small model keep computation and reasoning suited to its "
                   "own capacity.")},
        "d3": {"objective_template": template,
               "reward_scale": None if d3 is None else d3.reward_scale,
               "objective_hash": None if d3 is None else d3.objective_hash,
               "status": ("FROZEN" if d3 is not None else
                          "PENDING_LAMBDA: the rule is registered; the value is "
                          "measured on the reference pair before D3's search"),
               "formula": ("L(x) = (1/T_x) sum_{t in T_x} [KL_t - lambda c_t], "
                           "minimized (S = -L maximized)"),
               "hypothesis": (
                   "The student's own predictive certainty may carry "
                   "structural-selection value independent of the teacher KL: "
                   "even where its token distribution differs from the "
                   "teacher's, a student that is certain of its own "
                   "distribution earns a separate confidence reward. Whether "
                   "that signal is useful is decided by task correctness after "
                   "recovery training, not by the reward."),
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
        "measurement_distinctions": MEASUREMENT_DISTINCTIONS,
        "calibration_this_round": (
            "the frozen TEACHER-NATIVE calibration profiles, unchanged, so the "
            "objective is the only variable. No student-prefix calibration is "
            "introduced in D2 or D3"),
        "behavioural_diagnostics": BEHAVIOURAL_DIAGNOSTICS,
        "claim_boundary": CLAIM_BOUNDARY,
        "revision": 2,
        "revision_history": [
            {"revision": 1, "date": "2026-10-13",
             "change": "D2/D3 objectives, identities, ranking, lambda rule"},
            {"revision": 2, "date": "2026-10-13",
             "change": ("capacity-aware motivation, hypotheses, measurement "
                        "distinctions, behavioural diagnostics and claim "
                        "boundary (maintainer supplement). Objective "
                        "identities, ranking and lambda rule unchanged")}],
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

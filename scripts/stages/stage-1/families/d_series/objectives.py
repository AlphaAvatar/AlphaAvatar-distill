#!/usr/bin/env python3
"""D2's and D3's scoring objectives: the instances, the ranking, and lambda.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/objectives.py [--write]

D3: maintainer decision 2026-10-13. D2: maintainer scientific decision
2026-10-14 (contract revision 6), which SUPERSEDES the 2026-10-13
student-uncertainty D2 -- that objective was never executed and is kept
below, by name, as :data:`D2_STUDENT_UNCERTAINTY_SUPERSEDED`. Both keep D1's
target positions, profiles, Top-200 + tail reference KL, geometry, operators,
beam and recovery recipe:

    D1  L(x) = (1/T_x) sum_t KL(p_parent || q_C)_t   (DEPTH: parent reference)
    D2  L(x) = (1/T_x) sum_t c_T(t) KL(p_T || q_C)_t  frozen-teacher confidence
    D3  L(x) = (1/T_x) sum_t [KL_t - lambda c_t]      student confidence reward

D2's ``c_T(t) = 1 - H_T(t)/log|V|`` is the FROZEN ORIGINAL TEACHER's
full-vocabulary Shannon entropy; its KL reference is that teacher (the
lineage root) for EVERY consumer, DEPTH included, where D1 compared against
the parent. So D2 differs from D1 by TWO declared changes, not one: the DEPTH
reference (parent -> root) and the confidence weighting. D3's
``c_t = 1 - H_K(t)/log K`` (``K = 200``) is the candidate student's own Top-K
conditional entropy. ``T_x`` is the item's scored positions throughout.
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
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
    FORM_UNCERTAINTY_WEIGHTED_KL, KL_REFERENCE_ROOT, ScoringObjective,
)

SCHEMA = "aadistill.d_series.objective_contract/v1"
CONTRACT = "logs/stages/stage-1/families/d_series/plans/d2_d3_objective_contract_v1.json"
#: The B-vs-teacher REFERENCE CALIBRATION: D3's lambda AND the diagnostic
#: thresholds, measured once BEFORE D2's formal search and never revised.
#: Absent until then; D3's objective and the diagnostic flags cannot be built
#: without it.
REFERENCE_CALIBRATION = ("logs/stages/stage-1/families/d_series/analyses/"
                         "d2_d3_reference_calibration.json")
#: Kept for readers of the revision-2 contract, which named it this.
LAMBDA_RECORD = REFERENCE_CALIBRATION

#: K for the student's own Top-K entropy. Maintainer-selected 2026-10-13, the
#: same value as the reference support's K but a separate parameter: the two
#: Top-Ks select different entries (the reference's vs the student's own) and
#: each is bound in its own identity.
ENTROPY_TOP_K = 200

#: THE D2 OBJECTIVE (contract revision 6): the frozen original teacher's
#: full-vocabulary confidence weights the root-reference KL, and -- the
#: maintainer's FULL-SCOPE choice -- the activation statistics of the FFN,
#: residual-WIDTH and ATTENTION operators too (weighted numerators over a
#: sum-of-weights denominator). The KL's 1/T_x is fixed; no threshold,
#: exponent, hard mask or dynamic weight sum. The candidate never moves a
#: weight.
D2_OBJECTIVE = ScoringObjective(
    objective_id="objective.d2_teacher_confidence_weighted_kl", version=1,
    form=FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL, kl_reference=KL_REFERENCE_ROOT,
    weights_activation_statistics=True)

#: The NARROW-SCOPE alternative: the same KL objective, with the operators'
#: activation statistics left at D1's. DOCUMENTED FOR COMPARISON ONLY -- the
#: preregistration freezes the full scope, and nothing wires this one.
D2_NARROW_SCOPE_COMPARISON = ScoringObjective(
    objective_id="objective.d2_teacher_confidence_weighted_kl_narrow_scope",
    version=1, form=FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
    kl_reference=KL_REFERENCE_ROOT, weights_activation_statistics=False)

#: The 2026-10-13 D2, SUPERSEDED on 2026-10-14 and NEVER EXECUTED as a
#: search. Kept, unchanged, because records cite it: the reference
#: calibration's stage L measured its overhead and its scored-rows
#: equivalence (`L_premises.d2_objective_hash`), and preregistration v1 froze
#: it. Nothing may wire it into a D2 search.
D2_STUDENT_UNCERTAINTY_SUPERSEDED = ScoringObjective(
    objective_id="objective.d2_student_uncertainty_weighted_kl", version=2,
    form=FORM_UNCERTAINTY_WEIGHTED_KL, entropy_top_k=ENTROPY_TOP_K)
D2_SUPERSESSION = {
    "status": "SUPERSEDED -- NEVER EXECUTED",
    "objective_id": D2_STUDENT_UNCERTAINTY_SUPERSEDED.qualified_id,
    "objective_hash": D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash,
    "superseded_by": "objective.d2_teacher_confidence_weighted_kl@v1",
    "decided": "maintainer scientific decision 2026-10-14",
    "executed": {"d2_search": False, "d2_recovery": False,
                 "d2_behaviour": False,
                 "engineering_measurements_only": [
                     "reference calibration stage L "
                     "(refcal_20261013_170714): DEPTH one-round and "
                     "state-evaluation overhead, scored_v1 equivalence"]},
    "preregistration": ("logs/stages/stage-1/phase_d2/plans/"
                        "d2_preregistration_v1.json"),
}

D3_OBJECTIVE_ID = "objective.d3_kl_minus_student_confidence"


def d3_objective(reward_scale: float) -> ScoringObjective:
    """D3's objective at a MEASURED lambda. Built from the record, not typed."""
    return ScoringObjective(
        objective_id=D3_OBJECTIVE_ID, version=2,
        form=FORM_KL_MINUS_CONFIDENCE_REWARD, entropy_top_k=ENTROPY_TOP_K,
        reward_scale=float(reward_scale))


def frozen_d3_objective(repo_root: Path = REPO_ROOT) -> ScoringObjective:
    """D3's objective from the committed lambda record, or a refusal."""
    path = repo_root / REFERENCE_CALIBRATION
    if not path.is_file():
        raise RuntimeError(
            f"{REFERENCE_CALIBRATION} does not exist: D3's lambda has not been "
            "measured and frozen, so D3 has no objective and no search may "
            "start")
    record = json.loads(path.read_text())
    if record.get("rule_hash") != lambda_rule()["rule_hash"]:
        raise RuntimeError(
            "the lambda record was computed under a different rule than the "
            "one this contract registers")
    return d3_objective(record["reward_scale"])


def frozen_diagnostic_thresholds(repo_root: Path = REPO_ROOT) -> dict[str, float]:
    """The review-flag thresholds from the committed reference calibration."""
    path = repo_root / REFERENCE_CALIBRATION
    if not path.is_file():
        raise RuntimeError(f"{REFERENCE_CALIBRATION} does not exist: the "
                           "diagnostic thresholds have not been measured")
    record = json.loads(path.read_text())
    if record.get("threshold_rule_hash") != diagnostic_threshold_rule()["rule_hash"]:
        raise RuntimeError("the thresholds were derived under a different rule")
    return dict(record["diagnostic_thresholds"])


def flag_counts(joint: dict[str, Any], thresholds: dict[str, float]
                ) -> dict[str, int]:
    """The registered flags from a recorded joint histogram, conservatively.

    A 'high' threshold is applied at the first grid edge AT OR ABOVE it and a
    'low' one at the last edge AT OR BELOW it, so a flag never counts a
    position the exact threshold would not. Offline only: nothing a search
    decides reads this.
    """
    def first_at_or_above(edges, value):
        for i, e in enumerate(edges):
            if e >= value:
                return i
        return len(edges)

    def last_at_or_below(edges, value):
        idx = 0
        for i, e in enumerate(edges):
            if e <= value:
                idx = i
        return idx

    c_edges = joint["confidence_edges"]
    #: confidence bin i covers [c_edges[i], c_edges[i+1]).
    c_from = first_at_or_above(c_edges, thresholds["high_confidence"])
    #: KL bin j covers [kl_edges[j-1], kl_edges[j]) with bin 0 below edge 0.
    k_from = first_at_or_above(joint["kl_edges"], thresholds["high_kl"]) + 1
    m_upto = last_at_or_below(joint["topk_mass_edges"],
                              thresholds["low_topk_mass"])
    ck, cm = joint["confidence_x_kl"], joint["confidence_x_topk_mass"]
    return {
        "high_confidence_high_kl": sum(ck[i][j] for i in range(c_from, len(ck))
                                       for j in range(k_from, len(ck[i]))),
        "high_confidence_low_topk_mass": sum(cm[i][j]
                                             for i in range(c_from, len(cm))
                                             for j in range(0, m_upto)),
        "high_raw_kl": sum(row[j] for row in ck for j in range(k_from, len(row))),
        "positions": sum(map(sum, ck)),
    }


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
        "schema": "aadistill.d_series.d3_lambda_rule/v2",
        "statistic": "per-item means over each item's scored positions (1/T_x)",
        "formula": "lambda = IQR(item KL) / IQR(item confidence)",
        "pooling": (
            "every UNIQUE item of the three item sets once, unweighted, unique "
            "by content identity (sha256 of the token ids). The three sets "
            "hold 209 draws but 147 unique items: reasoning_heavy@v2's 62 "
            "draws are 51 unique items, all of them also in "
            "domain_balanced@v1, which sampled with replacement from the same "
            "pool. Counting a repeated draw twice would weight it twice. The "
            "with-duplicates value and each set alone are reported as "
            "sensitivity, never chosen"),
        "when": ("ONCE, BEFORE D2's formal search -- not before D3 -- and never "
                 "revised by any D2 or D3 search or behavioural result"),
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
            "SD ratio", "with duplicate draws (209)", "each item set alone",
            "q2 as the student"],
        "refusals": ["IQR(item confidence) == 0", "a non-finite or non-positive "
                     "lambda", "fewer than 20 items"],
    }
    rule["rule_hash"] = sha256_json(rule)
    return rule


def diagnostic_threshold_rule() -> dict[str, Any]:
    """Where the review flags' thresholds come from. Registered, hashed.

    The flags are DIAGNOSTIC. They are computed offline from records the
    search already writes, they never enter a score, a ranking, a pruning, a
    retention or any manual step, and a high raw KL is never a reason to drop
    a candidate -- a confident student departing from the teacher is the
    hypothesis under test, not a defect.
    """
    rule = {
        "schema": "aadistill.d_series.diagnostic_threshold_rule/v1",
        "source": ("the same B-vs-teacher reference calibration as lambda, over "
                   "the scored positions of the 147 unique items, measured "
                   "before D2's formal search"),
        "thresholds": {
            "high_kl": "p90 of per-position Top-200 + tail KL (teacher || B)",
            "high_confidence": "p90 of B's per-position Top-200 confidence",
            "low_topk_mass": "p10 of B's per-position Top-200 mass",
        },
        "flags": {
            "high_confidence_high_kl": ("confident AND far from the teacher: "
                                        "alternative reasoning or "
                                        "overconfidence, told apart only by "
                                        "correctness"),
            "high_confidence_low_topk_mass": (
                "a confident Top-200 that holds little of the student's real "
                "probability"),
            "high_raw_kl": "raw KL at or above high_kl, recorded, never acted on",
        },
        "application": {
            "rollout_rescoring": "per position, exact thresholds",
            "depth_candidates_and_states": (
                "from the recorded joint histograms; a 'high' threshold is "
                "applied at the first grid edge at or above it and a 'low' one "
                "at the last edge at or below it, so a flag never counts a "
                "position the exact threshold would not"),
        },
        "never": ["enters a score, ranking, pruning or retention",
                  "triggers a manual intervention in a running search",
                  "is set or revised after D2/D3 candidate results exist"],
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
        "that the local selection rules of all four operators use student "
        "confidence (only DEPTH's does, under D3; see objective_scope)",
        "that D2 differs from D1 by one variable: it declares two changes, "
        "the DEPTH reference (parent -> root) and the teacher-confidence "
        "weighting (d2.declared_changes)",
        "that a lower D2 objective means the student is closer to the "
        "teacher: it down-weights the teacher's uncertain positions by "
        "construction",
        "that confidence read on intermediate search candidates describes the "
        "final recovered student's capacity adaptation",
        "that the family-wise bound across D1-D3 is an unconditional "
        "guarantee: it is a nominal accounting bound, conditional on each "
        "test's nominal level, which adaptive reuse of the observed "
        "confirmation data can violate",
    ],
    "validation": ("Top-K confidence is a proxy for structure selection; its "
                   "usefulness is established only by independent behavioural "
                   "evidence after recovery training"),
}


#: D2 (revision 6): why the FROZEN TEACHER's confidence enters.
D2_MOTIVATION = {
    "title": "teacher-certainty-weighted fidelity",
    "premise": (
        "Where the frozen original teacher is certain -- low full-vocabulary "
        "entropy -- its next-token distribution encodes a determinate fact, "
        "step or format the compressed student should keep. Where it is "
        "uncertain, several continuations are acceptable and the KL to its "
        "spread-out distribution is a weaker signal of structural damage."),
    "consequence": (
        "Under the same capacity limit, a structure that preserves the "
        "teacher's distribution at the teacher's certain positions may keep "
        "more of what matters for reasoning than one that minimizes the KL "
        "uniformly over every scored position."),
    "question": (
        "Does weighting the root-reference KL, and the structural operators' "
        "activation statistics, by the frozen teacher's confidence select an "
        "initialization that recovers more autonomous task correctness than "
        "the standing incumbent B?"),
    "role_of_confidence": (
        "c_T is a property of the frozen teacher at the calibration and "
        "evaluation positions. No candidate can change it, so a candidate "
        "cannot lower the objective by becoming confident. It is a PROXY for "
        "where fidelity matters; its value is decided by behavioural "
        "evidence after recovery, never assumed"),
}

D2_HYPOTHESIS = (
    "Positions at which the frozen teacher is certain carry the information a "
    "small student most needs to preserve; spending the structural selection's "
    "fidelity budget there, rather than uniformly, selects initializations "
    "that recover more autonomous task correctness. This does NOT assume the "
    "teacher's uncertain positions are unimportant -- it tests whether "
    "down-weighting their KL, and their weight in the activation statistics, "
    "helps or hurts.")

#: THE TWO DECLARED CHANGES from D1. Never "one variable".
D2_DECLARED_CHANGES = {
    "changes": [
        {"change": "DEPTH's KL reference: the intact PARENT -> the frozen "
                   "original teacher (the lineage ROOT)",
         "d1": "KL(parent || parent with S bypassed)",
         "d2": ("KL(root || parent with S bypassed), the root's Top-200 + "
                "tail sketch read from the root reference store built once "
                "per calibration mixture"),
         "where_it_is_inactive": (
             "at the search root the parent IS the root, so the two "
             "references coincide; the change acts on DEPTH expansions of a "
             "compressed parent")},
        {"change": "frozen-teacher confidence weighting c_T(t) = 1 - "
                   "H_T(t)/log|V|",
         "d1": "unweighted",
         "d2": ("c_T multiplies the KL in DEPTH, state evaluation and the "
                "ranked metrics, and the activation statistics of the FFN, "
                "residual-WIDTH and ATTENTION operators")},
    ],
    "not_one_variable": (
        "D2 differs from D1 by both changes together. No claim attributes a "
        "D1-vs-D2 difference to the weighting alone. Descriptive separation "
        "only: every DEPTH round-0 candidate also records its parent-reference "
        "KL, and every candidate and state records the UNWEIGHTED root KL"),
}

#: D2's scope, FROZEN as full (maintainer preference, reviewed below).
D2_SCOPE = {
    "frozen_scope": "full",
    "acts_directly_on": [
        "DEPTH operator-local candidate evaluation: c_T-weighted root KL",
        ("FFN, residual WIDTH and ATTENTION activation statistics: every "
         "accumulator -- residual sum and second moment, FFN sum|a| and "
         "sum a^2, per-head attention-output second moment -- weighted by "
         "c_T, over a sum-of-weights denominator"),
        "global state evaluation of every child (reference: the teacher)",
        "beam ranking and pruning (OBJECTIVE_PARETO_V1)",
    ],
    "denominators": {
        "kl_objective": ("the fixed 1/T_x over the item's scored positions; "
                         "never the sum of weights"),
        "activation_statistics": ("sum_t w_t, with w_t = policy token weight x "
                                  "c_T(t); a zero-weight position is dropped "
                                  "like padding"),
    },
    "token_alignment": (
        "token j carries c_T of prediction j -- the logits its own state "
        "produces -- and the item's last token, which feeds no prediction "
        "inside the item, carries zero; the supervised-target policy's token "
        "weights are its prediction weights with that same trailing zero"),
    "not_weighted": [
        "the scored-position set, which stays the binary supervised-target "
        "policy",
        "operators outside D1's frozen four (none is in the search space)"],
    "narrow_scope_comparison": {
        "objective": D2_NARROW_SCOPE_COMPARISON.as_dict(),
        "objective_hash": D2_NARROW_SCOPE_COMPARISON.objective_hash,
        "difference": (
            "FFN, WIDTH and ATTENTION keep D1's target-aware statistics; their "
            "configs then declare no objective and their selections are D1's "
            "for the same parent"),
        "status": "DOCUMENTED FOR COMPARISON ONLY; not wired, not frozen",
    },
    "full_scope_review": {
        "implemented": (
            "aadistill core: ActivePositions token multipliers, the collectors' "
            "token_weights path (process / process_batch, per-item and packed), "
            "the weighted denominators residual_weight / attn_token_weight, and "
            "a statistics cache key that carries the weighting store's "
            "identity so weighted and unweighted passes never share an entry"),
        "risks": [
            ("LOW DYNAMIC RANGE: normalised by log|V| (~11.9 nats at a 151,936 "
             "vocabulary), c_T sits near 1 wherever the teacher's entropy is a "
             "few nats, so the weighting may change the statistics and the "
             "ranking only modestly. Measured by the teacher-confidence "
             "validity check BEFORE the search; never re-defined post hoc"),
            ("WIDTH's projection is trace-normalised per collection point, so "
             "a near-uniform weighting cancels; the weighting acts through "
             "its non-uniformity only"),
            ("teacher-UNCERTAIN positions -- branch points in a reasoning "
             "chain -- lose weight in the structure selection as well as in "
             "the KL; whether that helps is the hypothesis, not an assumption"),
            ("the reduced effective sample size of the weighted statistics "
             "(reported as ESS by the validity check)"),
        ],
        "blockers": "none found; the full scope is frozen",
    },
}

#: Where the objectives act, and where they do not. Stated as data so no
#: summary can say "all four operators" by accident. D3's scope; D2's is
#: D2_SCOPE above and is different.
OBJECTIVE_SCOPE = {
    "acts_directly_on": [
        "DEPTH's operator-local candidate evaluation (depth.causal_kl_greedy_v1)",
        "global state evaluation of every child",
        "beam ranking and pruning (OBJECTIVE_PARETO_V1)",
    ],
    "does_not_act_on": [
        "FFN, residual WIDTH and ATTENTION operator-local selection: they keep "
        "D1's target-aware activation statistic unchanged; their children are "
        "reached by the objective only through state evaluation and the beam",
    ],
    "do_not_claim": ("that the local rules of all four operators use student "
                     "confidence"),
    "intermediate_vs_final_student": (
        "the confidence the search reads belongs to INTERMEDIATE candidates -- "
        "the parent with some layers bypassed in DEPTH, partially compressed "
        "children in state evaluation -- before any recovery training. It is "
        "not the confidence of the final ~0.6B student after recovery, and "
        "it is not evidence about that model's capacity adaptation; only the "
        "behavioural evaluation after recovery speaks to that"),
}

#: The aggregation confound and its control (independent review item 1).
AGGREGATION_CONTROL = {
    "issue": ("revision 2 averaged state-evaluation items EQUALLY, where D1's "
              "KL-only state metrics weight items by scored positions within a "
              "sub-type and pool critical classes over items. That changed "
              "length weighting as well as the integrand, so a D1-vs-D2 "
              "ranking change could not be attributed to confidence"),
    "resolution": ("each consumer keeps its KL-only path's weighting "
                   "(objective identity field `aggregation`, "
                   "kl_path_weighting_per_consumer/v2): DEPTH averages items "
                   "equally, as D1's DEPTH did; state evaluation weights items "
                   "by T_x and pools critical classes. The per-item formula "
                   "keeps the fixed 1/T_x and never divides by a sum of "
                   "confidence weights"),
    "built_in_kl_only_control": (
        "every objective evaluation also reports state.objective.raw_kl.* -- "
        "the raw KL aggregated exactly as the objective is -- which equals "
        "D1's state.teacher_kl.* and state.critical_token_kl in the same "
        "evaluation (tested to 1e-6). Rankings by raw_kl and by the objective "
        "on the same states isolate the effect of the integrand"),
}

#: What the behavioural driver must satisfy when it wires the rescoring in.
ROLLOUT_RESCORING_INTEGRATION = {
    "token_ids": (
        "use the generation engine's exact ids, kept by the read-only sidecar "
        "wrapper scripts/shared/evaluation/uncapped_eval_token_ids.py. It "
        "runs the UNCHANGED in-tree generator (same argv, interpreter and "
        "environment) with a hook on the one post-wave decode per record, "
        "after the engine loop, so the generation source set, the generation "
        "fingerprint and the shared protocol's evaluation identity are the "
        "plain run's -- tested on the real generator with a stub engine: "
        "identical records, summaries and reconstructed fingerprint. The "
        "hook installs only when the generator's source digest equals the "
        "protocol's; every row is verified against its record (id count, "
        "prompt count, decoded-text sha256, EOS vs stop_reason) and the "
        "sidecar header says whether the stratum is exact. A record with no "
        "exact row is rescored from retokenized text, ids_source="
        "retokenized_text, labelled approximate per sample. Recording ids "
        "inside the generator itself remains a VERSIONED decision "
        "(d_series_shared_eval v2), not taken"),
    "token_ids_identity": {
        "unchanged": ["generation_source_digest", "generation fingerprint",
                      "evaluation_protocol_hash per rung",
                      "shared protocol hash"],
        "changes": ("only the behavioural driver's own executable closure, "
                    "because its argv names the wrapper -- correct for "
                    "reproducibility, and not part of the generation identity"),
        "not_in_generation_source_set": (
            "the wrapper is deliberately NOT added to source_sets.json "
            "generation.files: adding it would move the source digest it "
            "exists to preserve"),
        "residual": ("engine scheduling is timing-dependent in the plain run "
                     "too; the wrapper adds no work inside the engine loop, "
                     "so it changes no code path the tokens depend on, but "
                     "token equality with a counterfactual plain run cannot "
                     "be proven on a non-batch-invariant engine"),
    },
    "alignment_evidence": ["prompt ids sha256", "completion ids sha256",
                           "tokenizer sha256", "generation fingerprint",
                           "count and decoded-text match with the record"],
    "untouched": ["D1's evaluation evidence",
                  "the shared protocol's scoring contract and endpoint"],
    "checkpoint_release": ("only after the diagnostic evidence is persisted "
                           "durably, when a consumer exists"),
    "failure_isolation": ("a rescoring failure is recorded per sample and never "
                          "alters or blocks a completed per-sample score "
                          "(rollout_rescoring.rescore_unit)"),
    "cost": ("MEASURE before pricing: the reference calibration session times "
             "the rescoring on the real teacher and B over sequences of D1's "
             "measured lengths; the behavioural ceiling includes it only after "
             "that measurement. The 3-5 min/probe figure is an estimate, not a "
             "bound"),
    "historical_units": ("no logits, confidence or student-side quantity is "
                         "computed or estimated for B or q2"),
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
        "decided_by": ("maintainer decisions 2026-10-13 (D3) and 2026-10-14 "
                       "(D2 scientific redesign), logs/budget/decisions.md"),
        "supersedes": (
            "logs/stages/stage-1/phase_c1/plans/phase_c_roadmap.md, sections "
            "'D2 -- target-aware + reference-confidence weighting' and 'D3 -- "
            "target-aware + KL + pure student confidence'. Kept verbatim there "
            "as history, with a pointer here. Revision 6 also supersedes this "
            "contract's own revision-1..5 D2 (d2_superseded)."),
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
            "depth": ("none: per item over T_x, items averaged equally within "
                      "a sub-type, then sub-type and domain -- D1's DEPTH rule"),
            "state_evaluation": (
                "none since revision 3: within a sub-type items are weighted "
                "by T_x (the integrand pooled over the sub-type's scored "
                "positions, divided by their count) and each critical class is "
                "pooled over items -- D1's state-KL weighting. The per-item "
                "value keeps the fixed 1/T_x. state.objective.raw_kl.* is the "
                "raw KL under this same aggregation and equals D1's "
                "state.teacher_kl.* and state.critical_token_kl"),
            "matches_objective_identity_field": "kl_path_weighting_per_consumer/v2",
        },
        #: D3's motivation (the student's own confidence); D2's is
        #: `d2.motivation`.
        "scientific_motivation": SCIENTIFIC_MOTIVATION,
        "d2": {"objective": D2_OBJECTIVE.as_dict(),
               "objective_hash": D2_OBJECTIVE.objective_hash,
               "formula": (
                   "L_D2(x) = (1/T_x) sum_{t in T_x} c_T(t) "
                   "KL(p_T || q_C)_t, minimized; c_T(t) = 1 - H_T(t)/log|V|, "
                   "H_T the frozen original teacher's full-vocabulary Shannon "
                   "entropy (float32, chunked), KL on the Top-200 + tail "
                   "support against the ROOT for every consumer"),
               "motivation": D2_MOTIVATION,
               "hypothesis": D2_HYPOTHESIS,
               "declared_changes": D2_DECLARED_CHANGES,
               "scope": D2_SCOPE,
               "fixed": ["the 1/T_x denominator over scored positions",
                         "no dynamic weight sum in the KL objective",
                         "no threshold, exponent or hard mask on c_T",
                         "the candidate cannot affect c_T"],
               "diagnostics_recorded": [
                   "the unweighted root (teacher) KL, beside the weighted "
                   "objective, for every DEPTH candidate and every state "
                   "(state.objective.raw_kl.*)",
                   "the teacher-confidence-weighted KL (the objective)",
                   ("the PARENT-reference KL of every DEPTH round-0 candidate "
                    "(op-local parent_kl; one parent reference pass per DEPTH "
                    "expansion, round 0 only -- a documented, limited scope)"),
                   "reference entropy, reference confidence and weight "
                   "quantiles and histograms; reference tail mass",
                   "D1's KL metrics, unchanged, beside them",
                   "the root reference store's identity, content digest and "
                   "confidence distribution (weight sum, mean, ESS)"],
               "degeneracy_watch": (
                   "the weight is candidate-independent, so a candidate cannot "
                   "game it; it CAN degrade at the teacher's uncertain "
                   "positions at a discount. The unweighted root KL is recorded "
                   "on every candidate and state, and a low D2 objective is "
                   "never read as higher fidelity"),
               "validity_checks_before_search": (
                   "stages.d_series.teacher_confidence: distribution, "
                   "thinking vs answer, domain and subtype, token types, "
                   "relation to teacher token probability and to KL, weighted "
                   "contributions, weight sum, mean, ESS and risks -- run on "
                   "the root reference store before the formal search; their "
                   "results never change a definition")},
        "d2_superseded": {**D2_SUPERSESSION,
                          "objective": D2_STUDENT_UNCERTAINTY_SUPERSEDED.as_dict(),
                          "formula": ("L(x) = (1/T_x) sum_{t in T_x} "
                                      "(1 - c_t) KL_t, c_t the candidate "
                                      "student's own Top-200 confidence")},
        "d3": {"objective_template": template,
               "reward_scale": None if d3 is None else d3.reward_scale,
               "objective_hash": None if d3 is None else d3.objective_hash,
               "status": ("FROZEN" if d3 is not None else
                          "PENDING_LAMBDA: the rule is registered; the value is "
                          "measured on the reference pair BEFORE D2's formal "
                          "search, with the diagnostic thresholds"),
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
               "lambda_record": REFERENCE_CALIBRATION},
        "diagnostic_threshold_rule": diagnostic_threshold_rule(),
        "diagnostic_thresholds": (frozen_diagnostic_thresholds(repo_root)
                                  if (repo_root / REFERENCE_CALIBRATION).is_file()
                                  else None),
        "ranking": {"policy": OBJECTIVE_PARETO_V1.qualified_id,
                    "policy_hash": OBJECTIVE_PARETO_V1.policy_hash,
                    "objectives": [o.key for o in OBJECTIVE_PARETO_V1.objectives],
                    "epsilon": dict(OBJECTIVE_PARETO_V1.epsilon)},
        #: D3's scope. D2's is `d2.scope`.
        "objective_scope": OBJECTIVE_SCOPE,
        "where_each_objective_applies": {
            "depth.causal_kl_greedy_v1": (
                "OPERATOR-LOCAL, exactly: the candidate is the parent with S "
                "bypassed. D3: its KL against the parent and its own Top-K "
                "entropy come from the same forward. D2: its KL against the "
                "ROOT's stored sketch, weighted by the root's stored "
                "confidence"),
            "ffn / residual_width / attention activation operators": (
                "D3: NOT operator-local -- no candidate exists until it is "
                "made, so there is no candidate KL or confidence to weight; "
                "they keep D1's statistic and their state ids do not fork. "
                "D2 (full scope): their activation statistics are weighted by "
                "the frozen teacher's confidence, which needs no candidate; "
                "their configs declare the objective and their state ids "
                "fork from D1's"),
            "state evaluation": (
                "every child, against the original teacher; the student is the "
                "child itself"),
            "beam ranking and pruning": OBJECTIVE_PARETO_V1.qualified_id,
        },
        "diagnostics_recorded": [
            "raw KL", "D3: student Top-K entropy, confidence and Top-K mass",
            "D2: reference (teacher) entropy and confidence; DEPTH round-0 "
            "parent-reference KL",
            "confidence weight (D2) / confidence reward (D3)",
            "objective", "scored positions per item",
            "confidence histogram and quantiles",
            "reference tail mass", "D1's KL metrics, unchanged, beside them"],
        "d2_degeneracy_watch": (
            "D2's weight is the frozen teacher's, so no candidate can lower "
            "the objective by becoming confident; a candidate CAN be worse at "
            "the teacher's uncertain positions at a discount. Raw (unweighted) "
            "KL is recorded on every candidate and state, and a low D2 score "
            "is never read as higher fidelity. High raw KL is a registered "
            "DIAGNOSTIC FLAG (diagnostic_threshold_rule), never an elimination "
            "criterion and never a manual intervention"),
        "aggregation_control": AGGREGATION_CONTROL,
        "rollout_rescoring_integration": ROLLOUT_RESCORING_INTEGRATION,
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
        "revision": 6,
        "revision_history": [
            {"revision": 1, "date": "2026-10-13",
             "change": "D2/D3 objectives, identities, ranking, lambda rule"},
            {"revision": 2, "date": "2026-10-13",
             "change": ("capacity-aware motivation, hypotheses, measurement "
                        "distinctions, behavioural diagnostics and claim "
                        "boundary (maintainer supplement). Objective "
                        "identities, ranking and lambda rule unchanged")},
            {"revision": 3, "date": "2026-10-13",
             "change": ("independent review: (1) state-evaluation aggregation "
                        "now uses the KL-only path's T_x weighting, objectives "
                        "re-versioned v2; (2) lambda and diagnostic thresholds "
                        "measured before D2's search, on 147 unique items "
                        "(lambda rule v2); (3) objective scope and "
                        "intermediate-student boundary; (4) high raw KL is a "
                        "registered flag only; (5) rescoring integration "
                        "contract")},
            {"revision": 4, "date": "2026-10-13",
             "change": ("post-revision-3 review: (1) "
                        "aggregation_difference_from_d1 no longer contradicts "
                        "the revision-3 aggregation; (2) exact engine token "
                        "ids through a read-only generator wrapper whose "
                        "generation identity is the plain run's. Objective "
                        "identities, ranking and lambda rule unchanged")},
            {"revision": 5, "date": "2026-10-13",
             "change": ("the reference calibration ran (stage L, run "
                        "refcal_20261013_170714) under the registered rules: "
                        "D3's lambda = 7.34 and the diagnostic thresholds are "
                        "FROZEN from its committed record, before any D2 "
                        "search. D2's objective, the ranking and both rules "
                        "unchanged")},
            {"revision": 6, "date": "2026-10-14",
             "change": ("maintainer scientific decision: D2 REDESIGNED as the "
                        "frozen-teacher-confidence-weighted root-reference KL "
                        "(objective.d2_teacher_confidence_weighted_kl@v1), full "
                        "scope; the student-uncertainty D2 is SUPERSEDED and was "
                        "NEVER EXECUTED (d2_superseded). D3's objective, lambda "
                        "(7.34), thresholds, ranking and both rules unchanged")}],
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

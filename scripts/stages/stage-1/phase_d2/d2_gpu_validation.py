"""Stage V of `topk_adoption_driver.py`: the redesigned D2's GPU validation.

    ADOPT_STAGES=V bash scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh <run>

ONE L40S session on the infrastructure stage L and R0 already proved: the same
launcher (live quote, campaign-derived ceiling, watchdog, teardown on every
exit path, evidence fetched before teardown), the same remote payload and
`$0` preflight, the same driver journal and `adoption.json` written on every
path. This module adds only what the redesigned D2 needs measured.

**Numeric acceptance, REGISTERED BEFORE THE SESSION, in three classes that are
never mixed:**

1. **ARITHMETIC** -- one set of inputs, two computations. The full-vocabulary
   entropy in float32 chunks against float64 on the SAME teacher logits, at two
   chunk sizes; the confidence mapping; the `c x KL` fixed-denominator item
   mean against float64 on the SAME (KL, c); the state evaluator's raw KL
   against D1's KL metric of the same evaluation. GATED by
   :data:`ARITHMETIC_TOLERANCES`. Nothing about the model's execution can move
   these, so a tolerance here is a statement about float32 arithmetic alone.
2. **EXECUTION DRIFT** -- the real BF16 model at micro-batch 1 against 3. A
   different GEMM shape is a different computation, and R0's 1e-5 compares two
   paths UNDER ONE batch shape; it is not a statement about this. D2's formal
   topology is frozen at bsz=3 / `length_sorted_v1`, so this drift cannot
   reach a D2 result. REPORTED, NEVER GATED.
3. **ROW-SELECTION EQUIVALENCE at the formal bsz=3** -- `all_v1` against
   `scored_v1` on one DEPTH round and on state evaluation, under R0's
   registered tolerances (:data:`ROW_EQUIVALENCE_TOLERANCES`). It decides only
   whether D2 may run `scored_v1`.

Everything else -- root-vs-parent rankings, the confidence distribution, the
full-vs-narrow selection overlaps, timings, cache modes -- is an EXECUTION or
SCIENCE DIAGNOSTIC: recorded as measured, including a narrow dynamic range or
an unchanged ranking, and never a reason to change a definition, a
normalization or a tolerance after the fact.

The models, items and suite arrive through an ENVIRONMENT object, so the same
stage runs end to end on a CPU toy at `$0` (tests) and on the pod (the
production environment below). AUTHORIZES NOTHING.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VALIDATION_DIR = ("logs/stages/stage-1/families/d_series/validations/"
                  "d2-gpu-validation/v1")
PLAN = f"{VALIDATION_DIR}/plan.json"
CAMPAIGN = f"{VALIDATION_DIR}/campaign.json"
AUTHORIZATION = f"{VALIDATION_DIR}/authorization.json"
AUTHORIZATION_PROPOSED = f"{VALIDATION_DIR}/authorization.proposed.json"
RECORD_KEY = "V_d2_validation"

#: CLASS 1. float32 entropy of a softmax over |V| = 151,936 entries: per-element
#: rounding ~2^-24 relative and a reduction-tree error growing ~log2|V| ~ 17,
#: so ~1e-6 relative of an entropy at most log|V| ~ 11.9 nats -> ~1.2e-5 nats.
#: 1e-4 nats keeps an ~8x margin and still catches a wrong mapping, a dropped
#: chunk or a float16 path by orders of magnitude. The confidence divides by
#: log|V|, so 1e-4 nats is 8.4e-6 there; 1e-5. The item mean of c x KL in
#: float32 over at most a few thousand positions: 1e-5 relative. The raw-KL
#: control compares two reductions of one evaluation: CPU-pinned at 1e-6.
ARITHMETIC_TOLERANCES = {
    "entropy_abs_nats_vs_float64": 1e-4,
    "entropy_abs_nats_chunk_invariance": 1e-4,
    "confidence_abs_vs_float64": 1e-5,
    "item_objective_relative_vs_float64": 1e-5,
    "raw_kl_control_relative": 1e-6,
}
#: INTEGRITY of the full-scope statistics: the denominator a weighted
#: collector divided by is the sum of the very weights the policy and the root
#: store produced -- two float64 sums of one vector in different orders.
INTEGRITY_TOLERANCES = {"statistics_denominator_relative": 1e-9}
#: The three statistics operators the full scope weights. EVERY one must
#: execute and verify, or the validation is not complete: FFN's child alone is
#: enough for V4 to run, and must not be enough for a PASS.
REQUIRED_STATISTICS_OPERATORS = ("ffn.activation_importance_v0",
                                 "attention.activation_importance_v1",
                                 "width.global_pca_v0")
#: Where each operator reports the denominator it divided by (weighted only).
DENOMINATOR_EVIDENCE = {
    "ffn.activation_importance_v0": ("local_metrics_detail",
                                     "stats_token_weight"),
    "width.global_pca_v0": ("local_metrics_detail", "stats_token_weight"),
    "attention.activation_importance_v1": ("trace",
                                           "calibration_token_weight"),
}

#: CLASS 3. R0's registered tolerances, carried over unchanged.
ROW_EQUIVALENCE_TOLERANCES = {"depth_candidate_relative": 1e-5,
                              "depth_choice": "exact",
                              "state_eval_absolute": 1e-5}
#: CLASS 2 and the diagnostics: named, so the plan can list them as NOT gates.
EXECUTION_DIAGNOSTICS = (
    "bsz1_vs_bsz3_root_store", "bsz1_vs_bsz3_state_evaluation",
    "store_vs_recomputed_same_batch", "timings_and_memory", "cache_modes")
SCIENCE_DIAGNOSTICS = (
    "teacher_confidence_validity_per_profile",
    "root_vs_parent_vs_weighted_rankings", "full_vs_narrow_selection_overlap")
#: Execution order. IDs are the plan's; V5 is the one EXTENDED step: it decides
#: only `scored_v1`, and D2 runs `all_v1` without it.
STEP_ORDER = ("V1_root_stores_and_validity", "V2_b_rebuild_and_kl_relation",
              "V6_state_evaluation", "V3_weighted_statistics",
              "V4_depth_root_vs_parent", "V5_depth_scored_rows")
CORE_STEPS = STEP_ORDER[:-1]
#: The arithmetic check reads this many packed batches' logits per profile.
ARITHMETIC_BATCHES = 1
FLOAT64_ROW_CHUNK = 64


class D2ValidationError(RuntimeError):
    pass


# --- class 1: arithmetic on identical inputs ---------------------------------

def float64_entropy(logits, *, chunk: int = FLOAT64_ROW_CHUNK):
    """The reference: softmax and Shannon entropy in float64, row chunks."""
    import torch

    flat = logits.reshape(-1, int(logits.shape[-1]))
    out = torch.empty(flat.shape[0], dtype=torch.float64, device=flat.device)
    for s in range(0, int(flat.shape[0]), chunk):
        p = torch.softmax(flat[s:s + chunk].double(), dim=-1)
        out[s:s + chunk] = torch.special.entr(p).sum(-1)
    return out


def entropy_arithmetic(rows) -> dict[str, Any]:
    """float32 chunked (the store's path) vs float64, and chunk 512 vs 64."""
    from aadistill.initialization.scoring.objective import (
        confidence_from_vocabulary_entropy, full_vocabulary_entropy,
    )

    v = int(rows.shape[-1])
    h32 = full_vocabulary_entropy(rows, chunk=512).double()
    h32b = full_vocabulary_entropy(rows, chunk=64).double()
    h64 = float64_entropy(rows)
    c32 = confidence_from_vocabulary_entropy(h32.float(), v).double()
    c64 = (1.0 - h64 / math.log(v)).clamp(0.0, 1.0)
    t = ARITHMETIC_TOLERANCES
    out = {"positions": int(h64.numel()), "vocab": v,
           "entropy_max_abs_vs_float64": float((h32 - h64).abs().max()),
           "entropy_max_abs_chunk_512_vs_64": float((h32 - h32b).abs().max()),
           "confidence_max_abs_vs_float64": float((c32 - c64).abs().max())}
    out["within_tolerance"] = (
        out["entropy_max_abs_vs_float64"] <= t["entropy_abs_nats_vs_float64"]
        and out["entropy_max_abs_chunk_512_vs_64"]
        <= t["entropy_abs_nats_chunk_invariance"]
        and out["confidence_max_abs_vs_float64"]
        <= t["confidence_abs_vs_float64"])
    return out


def reduction_arithmetic(objective: Any, per_item: Sequence[Mapping[str, Any]]
                         ) -> dict[str, Any]:
    """The operator's float32 `c x KL` item mean vs float64, same (KL, c).

    Each entry: ``kl`` and ``confidence`` (aligned ``[T-1]``) and ``scored``
    (bool ``[T-1]``) -- the inputs DEPTH and the state evaluator reduce.
    """
    import torch

    from aadistill.initialization.scoring.objective import (
        item_component_means, position_terms,
    )

    worst, n = 0.0, 0
    for e in per_item:
        kl = e["kl"].float()
        conf = e["confidence"].float().to(kl.device)
        scored = e["scored"].to(kl.device).bool()
        if not bool(scored.any()) or not bool(torch.isfinite(kl[scored]).all()):
            continue
        terms = position_terms(objective, kl, None, None, None,
                               reference_entropy=torch.zeros_like(conf),
                               reference_confidence=conf)
        means, _ = item_component_means(
            {k: v[None] for k, v in terms.items()},
            torch.ones(1, int(kl.shape[0]), dtype=torch.bool, device=kl.device),
            scored[None].double(), names=objective.components())
        got = float(means["objective"][0])
        want = float((conf.double()[scored] * kl.double()[scored]).sum()
                     / int(scored.sum()))
        worst = max(worst, abs(got - want) / max(abs(want), 1e-30))
        n += 1
    tol = ARITHMETIC_TOLERANCES["item_objective_relative_vs_float64"]
    return {"items": n, "max_relative_difference": worst,
            "within_tolerance": n > 0 and worst <= tol}


def raw_kl_control_relative(kl_only: Mapping[str, float],
                            objective: Mapping[str, float]) -> dict[str, Any]:
    """The objective's raw KL vs D1's KL metric of the SAME evaluation."""
    pairs = [("state.objective.raw_kl.equal_domain_mean",
              "state.teacher_kl.equal_domain_mean"),
             ("state.objective.raw_kl.worst_domain",
              "state.teacher_kl.worst_domain"),
             ("state.objective.raw_kl.critical_token",
              "state.critical_token_kl")]
    src, dst = "state.objective.raw_kl.critical_token.", "state.critical_token_kl."
    pairs += [(k, dst + k[len(src):]) for k in objective
              if k.startswith(src) and dst + k[len(src):] in kl_only]
    rel = [abs(float(objective[a]) - float(kl_only[b]))
           / max(abs(float(kl_only[b])), 1e-30)
           for a, b in pairs if a in objective and b in kl_only]
    worst = max(rel) if rel else None
    return {"metrics_compared": len(rel), "max_relative_difference": worst,
            "within_tolerance": (worst is not None and worst
                                 <= ARITHMETIC_TOLERANCES[
                                     "raw_kl_control_relative"])}


# --- class 2 and the diagnostics ---------------------------------------------

def store_drift(a: Any, b: Any, items: Sequence[Mapping[str, Any]]
                ) -> dict[str, Any]:
    """Two stores of one mixture: how far their entries moved. Diagnostic."""
    import torch

    ent, conf, jac = [], [], []
    for item in items:
        ea, eb = a.per_position(item, "entropy"), b.per_position(item, "entropy")
        ent.append((ea.double() - eb.double()).abs().cpu())
        conf.append((a.per_position(item, "confidence").double()
                     - b.per_position(item, "confidence").double()).abs().cpu())
        #: Vectorized: Top-K indices are distinct per position, so the
        #: intersection is a sorted membership count and |A u B| = 2K - |A n B|.
        ia = a.sketch_item(item).support_indices.sort(-1).values
        ib = b.sketch_item(item).support_indices.to(ia.device).sort(-1).values
        k = int(ia.shape[-1])
        pos = torch.searchsorted(ib, ia).clamp(max=k - 1)
        inter = (ib.gather(-1, pos) == ia).sum(-1).double()
        jac.append((inter / (2 * k - inter)).cpu())
    e, c, j = torch.cat(ent), torch.cat(conf), torch.cat(jac)
    q = torch.tensor([0.5, 0.99], dtype=torch.float64)
    return {"positions": int(e.numel()),
            "entropy_abs": {"max": float(e.max()),
                            "p50": float(torch.quantile(e, q)[0]),
                            "p99": float(torch.quantile(e, q)[1])},
            "confidence_abs_max": float(c.max()),
            "topk_support_jaccard_mean": float(j.mean()),
            "topk_support_identical_fraction": float((j == 1.0).double().mean()),
            "_class": "EXECUTION DRIFT -- reported, never gated"}


def confidence_summary(store: Any, items: Sequence[Mapping[str, Any]],
                       policy: Any) -> dict[str, Any]:
    """The weight distribution over SCORED positions: IQR, ESS, histogram."""
    from aadistill.initialization.scoring.positions import PREDICTION_AXIS

    scored = {str(i["item_id"]): policy.weights(i, axis=PREDICTION_AXIS)
              .weights > 0 for i in items}
    rep = store.confidence_report(items, scored)
    q = rep.get("quantiles") or {}
    if q:
        rep["iqr"] = q["q75"] - q["q25"]
    return rep


def jaccard_per_layer(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]
                      ) -> dict[str, Any]:
    vals = [len(set(x) & set(y)) / max(1, len(set(x) | set(y)))
            for x, y in zip(a, b)]
    return {"layers": len(vals), "mean": sum(vals) / max(1, len(vals)),
            "min": min(vals) if vals else None,
            "identical_layers": sum(1 for v in vals if v == 1.0),
            "per_layer": [round(v, 6) for v in vals]}


def projection_overlap(teacher_embedding, child_a_embedding,
                       child_b_embedding, *, rows: int = 16384
                       ) -> dict[str, Any]:
    """Overlap of two WIDTH projections, recovered from their children.

    WIDTH builds the child embedding as ``E_t @ P``, so ``P`` solves the
    normal equations ``(E_t^T E_t) P = E_t^T E_c``, accumulated over vocabulary
    row chunks in float64 so no ``[V, D]`` float64 copy exists. The children
    are stored in the model dtype (bf16 on the pod), so the recovered
    projections carry that rounding: an APPROXIMATE diagnostic. Overlap is
    ``||Q_a^T Q_b||_F^2 / d`` over orthonormalized projections: 1 for one
    subspace, about d/D for unrelated ones.
    """
    import torch

    et = teacher_embedding
    dev = et.device
    d_p = int(et.shape[1])
    gram = torch.zeros(d_p, d_p, dtype=torch.float64, device=dev)
    rhs = {k: torch.zeros(d_p, int(c.shape[1]), dtype=torch.float64,
                          device=dev)
           for k, c in (("a", child_a_embedding), ("b", child_b_embedding))}
    for s in range(0, int(et.shape[0]), rows):
        e = et[s:s + rows].double()
        gram += e.T @ e
        rhs["a"] += e.T @ child_a_embedding[s:s + rows].to(dev).double()
        rhs["b"] += e.T @ child_b_embedding[s:s + rows].to(dev).double()
    qa, _ = torch.linalg.qr(torch.linalg.solve(gram, rhs["a"]))
    qb, _ = torch.linalg.qr(torch.linalg.solve(gram, rhs["b"]))
    d = int(qa.shape[1])
    return {"subspace_overlap": float((qa.T @ qb).pow(2).sum() / d),
            "d_child": d, "d_parent": d_p,
            "random_subspace_expectation": d / d_p,
            "_approximate": "projections recovered from the stored children"}


def three_way_rankings(candidates: Mapping[str, Mapping[str, Any]],
                       round0: Sequence[str]) -> dict[str, Any]:
    """Parent KL, unweighted root KL, weighted root KL over ONE set of
    round-0 candidates: rank agreement, each metric's choice, identical order."""
    from stages.phase_d1.topk_adoption_driver import spearman

    keys = [k for k in round0 if k in candidates
            and all(m in candidates[k] for m in ("parent_kl", "raw_kl",
                                                 "objective"))]
    metric = {"parent_kl": "parent_kl", "root_kl_unweighted": "raw_kl",
              "root_kl_teacher_weighted": "objective"}
    cols = {n: [float(candidates[k][f]) for k in keys] for n, f in metric.items()}
    order = {n: [k for _, k in sorted(zip(v, keys))] for n, v in cols.items()}
    names = list(metric)
    pairs = {f"{a}__vs__{b}": {
        "spearman": spearman(cols[a], cols[b]),
        "same_choice": order[a][:1] == order[b][:1],
        "identical_order": order[a] == order[b],
        "top3_overlap": len(set(order[a][:3]) & set(order[b][:3]))}
        for i, a in enumerate(names) for b in names[i + 1:]}
    return {"candidates": len(keys),
            "choice": {n: (order[n][0] if order[n] else None) for n in names},
            "pairs": pairs,
            "table": [{"candidate": k, **{n: cols[n][i] for n in names}}
                      for i, k in enumerate(keys)],
            "_class": ("SCIENCE DIAGNOSTIC: recorded as measured; an unchanged "
                       "ranking is a finding, not a reason to change anything")}


def depth_table_consistency(outcome_artifacts: Mapping[str, Any]
                            ) -> dict[str, Any]:
    """The decision table's score IS the weighted objective of each subset."""
    rounds = list(outcome_artifacts.get("search_rounds") or [])
    cands = outcome_artifacts.get("objective_candidates") or {}
    if not rounds:
        return {"consistent": False, "why": "no round"}
    worst, missing = 0.0, []
    for e in rounds[0].get("table") or []:
        key = str(e["candidate"])
        rec = cands.get(key)
        if rec is None or "parent_kl" not in rec:
            missing.append(key)
            continue
        worst = max(worst, abs(float(e["score"]) - float(rec["objective"]))
                    / max(abs(float(rec["objective"])), 1e-30))
    return {"round0_candidates": len(rounds[0].get("table") or []),
            "missing_records": missing, "max_relative_score_vs_objective": worst,
            "consistent": not missing and worst <= 1e-12}


def statistics_integrity(impl, arm_label: str, outcome, ctx, *, target,
                         store_identity_hash: str, objective_hash: str
                         ) -> dict[str, Any]:
    """Did one statistics-operator arm compute what its scope declares?

    Full scope: weighted by the ROOT store this session built, the objective
    declared in its config, the collector's denominator equal to the sum of
    the weights. Narrow scope: none of that. Both: the child reaches the
    target in every field the operator modifies, with finite parameters.
    Selection overlap and effect size are NOT here -- they are diagnostics.
    """
    import torch

    sp = outcome.trace.get("scoring_positions") or {}
    source = sp.get("token_weights_multiplied_by")
    where, key = DENOMINATOR_EVIDENCE[impl.impl_id]
    bag = (dict(outcome.local_metrics.detail or {}) if where ==
           "local_metrics_detail" else dict(outcome.trace))
    denominator = bag.get(key)
    child_spec = ctx.adapter.spec_of(outcome.model)
    checks: dict[str, Any] = {
        "reaches_target": all(child_spec[f] == target[f]
                              for f in impl.modifies),
        "child_finite": all(bool(torch.isfinite(p).all())
                            for p in outcome.model.parameters()),
    }
    declared = (ctx.config or {}).get("scoring_objective")
    if arm_label == "full_scope":
        total = float(((sp.get("token_axis") or {}).get("weight_total")) or 0)
        checks.update({
            "weighted_by_this_sessions_root_store":
                source == f"root_reference_confidence:{store_identity_hash}",
            "config_declares_the_d2_objective": (
                declared is not None
                and objective_sha(declared) == objective_hash),
            "denominator_is_the_weight_sum": (
                denominator is not None and total > 0
                and abs(float(denominator) - total) / total
                <= INTEGRITY_TOLERANCES["statistics_denominator_relative"]),
        })
        detail = {"denominator": denominator, "weight_total": total}
    else:
        checks.update({"unweighted": source is None,
                       "config_declares_no_objective": declared is None,
                       "no_weighted_denominator": denominator is None})
        detail = {}
    return {"checks": checks, **detail, "passed": all(checks.values())}


def objective_sha(declared: Mapping[str, Any]) -> str:
    from aadistill.infrastructure.manifest import sha256_json

    return sha256_json(dict(declared))


# --- verdicts -------------------------------------------------------------------

def verdicts(result: Mapping[str, Any]) -> dict[str, Any]:
    """Three verdicts, computed from the record, never typed."""
    arith: dict[str, Any] = {}
    v1 = result.get("V1_root_stores_and_validity") or {}
    for pid, prof in (v1.get("profiles") or {}).items():
        arith[f"entropy_{pid}"] = (prof.get("arithmetic") or {}).get(
            "within_tolerance")
    v2 = result.get("V2_b_rebuild_and_kl_relation") or {}
    for pid, prof in (v2.get("profiles") or {}).items():
        arith[f"reduction_{pid}"] = (prof.get("reduction_arithmetic") or {}).get(
            "within_tolerance")
    v6 = result.get("V6_state_evaluation") or {}
    arith["raw_kl_control"] = (v6.get("raw_kl_control") or {}).get(
        "within_tolerance")
    #: FAIL on any measured breach; INCOMPLETE while a check is unmeasured;
    #: PASS only when every one ran and held.
    if any(v is False for v in arith.values()):
        arithmetic = "FAIL"
    elif (not v1.get("profiles") or not v2.get("profiles")
          or any(v is None for v in arith.values())):
        arithmetic = "INCOMPLETE"
    else:
        arithmetic = "PASS"
    v4 = result.get("V4_depth_root_vs_parent") or {}
    v3 = result.get("V3_weighted_statistics")

    def measured(step: Any) -> bool:
        return bool(step) and "skipped" not in step

    def ran(step: Any) -> bool:
        return measured(step) and "failed" not in step

    #: TRI-STATE: None = not measured (INCOMPLETE), False = measured and
    #: wrong (FAIL). A step that failed outright is False, never None.
    integ = {
        "teacher_identity": (result.get("premises") or {}).get(
            "teacher_revision") is not None,
        "b_identity_verified": (v2.get("b_identity_verified") is True
                                if ran(v2) else
                                (False if measured(v2) else None)),
        "stores_match_declaration": (v1.get("stores_match_declaration") is True
                                     if ran(v1) else
                                     (False if measured(v1) else None)),
        "full_scope_statistics_operators": _v3_integrity(v3),
        "depth_reads_the_root": (v4.get("reference") == "root_reference"
                                 if ran(v4) else
                                 (False if measured(v4) else None)),
        "depth_table_consistent": ((v4.get("table_consistency") or {}).get(
            "consistent") is True if ran(v4) else
            (False if measured(v4) else None)),
    }
    if any(v is False for v in integ.values()):
        integrity = "FAIL"
    elif any(v is None for v in integ.values()):
        integrity = "INCOMPLETE"
    else:
        integrity = "PASS"
    se = (v6.get("scored_rows") or {})
    dp = ((result.get("V5_depth_scored_rows") or {}).get("agreement") or {})
    if "within_tolerance" not in se or "within_tolerance" not in dp:
        rows = "INCOMPLETE"
    elif se["within_tolerance"] and dp["within_tolerance"]:
        rows = "PASS"
    else:
        rows = "FAIL"
    return {
        "arithmetic": {"verdict": arithmetic, "checks": arith,
                       "tolerances": ARITHMETIC_TOLERANCES},
        "integrity": {"verdict": integrity, "checks": integ,
                      "tolerances": INTEGRITY_TOLERANCES,
                      "required_operators": list(
                          REQUIRED_STATISTICS_OPERATORS)},
        "scored_rows_equivalence": {
            "verdict": rows, "tolerances": ROW_EQUIVALENCE_TOLERANCES,
            "state_eval_within": se.get("within_tolerance"),
            "depth_within": dp.get("within_tolerance"),
            "consequence": ("D2 may run scored_v1" if rows == "PASS" else
                            "D2 runs all_v1")},
        "validation": ("PASS" if arithmetic == "PASS" and integrity == "PASS"
                       else ("FAIL" if "FAIL" in (arithmetic, integrity)
                             else "INCOMPLETE")),
        "not_gated": list(EXECUTION_DIAGNOSTICS) + list(SCIENCE_DIAGNOSTICS),
    }


def _v3_integrity(v3: Any) -> bool | None:
    """Every required operator executed both arms and verified."""
    if not v3 or "skipped" in v3:
        return None
    if "failed" in v3:
        return False
    ops = v3.get("operators") or {}
    for impl_id in REQUIRED_STATISTICS_OPERATORS:
        op = ops.get(impl_id)
        if not op or "failed" in op:
            return False
        for arm in ("full_scope", "narrow_scope"):
            if not ((op.get("arms") or {}).get(arm) or {}).get(
                    "integrity", {}).get("passed"):
                return False
    return True


# --- the environment: what the stage runs on ---------------------------------

@dataclass
class Environment:
    """Everything model- or host-specific, injected. One production instance
    (:func:`production_environment`); tests build a CPU toy."""

    repo: Path
    device: str
    adapter: Any
    support: Any
    policy: Any
    numerics: Any
    target_spec: Any
    seed: int
    micro_batch_size: int
    packing: str
    root_identity: Mapping[str, Any]
    teacher_revision: str | None
    load_teacher: Callable[[], Any]
    materialize_b: Callable[[Path], tuple[Any, dict[str, Any]]]
    profiles: Callable[[], list[tuple[Any, list]]]
    load_suite: Callable[[], tuple[Any, list, str]]
    decode: Callable[[int], str] | None = None
    step_bounds_s: Mapping[str, float] = field(default_factory=dict)

    @property
    def cuda(self) -> bool:
        return str(self.device).startswith("cuda")


def production_environment(repo: Path, teacher_path: str, numerics: Any,
                           device: str = "cuda") -> Environment:
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.scoring.positions import get_position_policy
    from aadistill.initialization.specs.arch import get_adapter
    from shared.evaluation import load_state_eval
    from stages.d_series import incumbent as INC
    from stages.d_series import scoring_protocol as SP
    from stages.phase_a3 import a3_session as A3S
    from stages.phase_d1 import behavioural_materialize as BM
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d1.d1_qualification_driver import _load_suite
    from stages.phase_d1.topk_adoption_driver import (
        POLICY_ID, TEACHER_BINDING,
    )
    from stages.phase_d2 import write_d2_preregistration as W

    adapter = get_adapter("qwen3")
    spec = A3S.path_spec(workdir_device=device)
    root = D1S.root_teacher_identity(repo)
    binding = json.loads((repo / TEACHER_BINDING).read_text())
    inc = INC.standing_incumbent(repo)
    plan = json.loads((repo / PLAN).read_text())
    profile_ids = json.loads((repo / W.D1_DESIGN).read_text())[
        "search_stage"]["profiles"]

    def load_teacher():
        m = adapter.load(teacher_path, dtype="bfloat16", device=device)
        m.config.use_cache = False
        return m

    def materialize_b(workdir: Path):
        row = BM.materialize_incumbent(
            workdir / "B", device=device,
            identities={k: inc[k] for k in INC.IDENTITY_FIELDS}, repo_root=repo)
        m = adapter.load(row["checkpoint_path"], dtype="bfloat16", device=device)
        m.config.use_cache = False
        return m, {"identity": row["identity"], "seconds": row["seconds"],
                   "expected": {k: inc[k] for k in INC.IDENTITY_FIELDS}}

    def profiles():
        out = []
        for pid in profile_ids:
            p = get_profile(pid)
            out.append((p, prepare_calibration_items(
                p.resolve(repo), profile_id=p.qualified_id)))
        return out

    decode = None
    try:
        from transformers import AutoTokenizer

        import functools

        tok = AutoTokenizer.from_pretrained(teacher_path)
        decode = functools.lru_cache(maxsize=None)(
            lambda t: tok.decode([int(t)]))
    except Exception:                                   # noqa: BLE001
        decode = None
    return Environment(
        repo=repo, device=device, adapter=adapter, support=SP.D_SERIES_SUPPORT,
        policy=get_position_policy(f"{POLICY_ID}@v1"), numerics=numerics,
        target_spec=spec.target_spec, seed=spec.seed,
        micro_batch_size=SP.D_SERIES_MICRO_BATCH_SIZE,
        packing=SP.D_SERIES_BATCH_PACKING,
        root_identity={"teacher_id": root["root_teacher_id"],
                       "teacher_sha256": root["root_teacher_sha256"]},
        teacher_revision=binding.get("revision"),
        load_teacher=load_teacher, materialize_b=materialize_b,
        profiles=profiles,
        load_suite=lambda: _load_suite(load_state_eval, repo),
        decode=decode, step_bounds_s=dict(plan["step_bounds_s"]))


# --- the $0 premises ----------------------------------------------------------

def premises(repo: Path) -> dict[str, Any]:
    """Everything stage V rests on, before a model is resident. Raises."""
    from stages.d_series import incumbent as INC
    from stages.d_series import objectives as O
    from stages.phase_c2 import baseline as BL
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d1.topk_adoption_driver import TEACHER_BINDING

    contract = json.loads((repo / O.CONTRACT).read_text())
    #: Stage V validated D2 VERSION 2. The committed contract carries it as
    #: its D2 (revision 7, when this ran) or as the superseded version 2
    #: (revision 8 on): either way the same hash.
    carried = {(contract.get(k) or {}).get("objective_hash")
               for k in ("d2", "d2_v2_superseded")}
    if O.D2_V2_OBJECTIVE.objective_hash not in carried:
        raise D2ValidationError("the committed contract does not carry the "
                                "objective this validation validated")
    plan = json.loads((repo / PLAN).read_text())
    for key, mine in (("arithmetic", ARITHMETIC_TOLERANCES),
                      ("row_equivalence", ROW_EQUIVALENCE_TOLERANCES),
                      ("integrity", INTEGRITY_TOLERANCES)):
        if plan["acceptance"][key]["tolerances"] != mine:
            raise D2ValidationError(
                f"the plan registers {key} tolerances "
                f"{plan['acceptance'][key]['tolerances']} and the code applies "
                f"{mine}; the registered ones are the ones that count")
    if plan["objective_hash"] != O.D2_V2_OBJECTIVE.objective_hash:
        raise D2ValidationError("the plan validates another objective")
    D1S._register_frozen_operators()
    spec = BL.frozen_baseline_spec(device="cuda")
    construction = BL.assert_frozen_construction(spec)
    inc = INC.standing_incumbent(repo)
    binding = json.loads((repo / TEACHER_BINDING).read_text())
    return {"objective_hash": O.D2_V2_OBJECTIVE.objective_hash,
            "objective": O.D2_V2_OBJECTIVE.qualified_id,
            "contract_revision": contract["revision"],
            #: THE PLAN'S identities: what this validation was registered,
            #: granted and run against -- not whatever the tree says today.
            "contract_hash": plan["contract_hash"],
            "preregistration_hash": plan["preregistration_hash"],
            "plan_sha256": _sha256(repo / PLAN),
            "incumbent": {k: inc[k] for k in INC.IDENTITY_FIELDS},
            "b_spec_hash": spec.spec_hash, "construction": construction,
            "teacher_revision": binding.get("revision"),
            "root_teacher": D1S.root_teacher_identity(repo),
            "step_bounds_s": dict(plan["step_bounds_s"]),
            "source": source_commit(repo)}


def expected_identities(repo: Path) -> dict[str, str]:
    """The identities a booked session must carry to count as THIS
    validation: the objective it validated (D2 version 2) and the contract
    and preregistration hashes its PLAN registered, with the plan itself
    bound by its sha256.

    From the plan, not from the live records: the contract and the D2
    preregistration have since moved to version 3 (revision 8), and this
    evidence is about version 2 -- which is exactly what keeps it
    adoptable as version 2's evidence and unusable as anything else's."""
    from stages.d_series import objectives as O

    plan = json.loads((repo / PLAN).read_text())
    if plan["objective_hash"] != O.D2_V2_OBJECTIVE.objective_hash:
        raise D2ValidationError("the plan validates another objective")
    return {"objective_hash": O.D2_V2_OBJECTIVE.objective_hash,
            "contract_hash": plan["contract_hash"],
            "preregistration_hash": plan["preregistration_hash"],
            "plan_sha256": _sha256(repo / PLAN)}


def adopt(doc: Mapping[str, Any], repo: Path,
          expected_commit: str | None = None) -> dict[str, Any]:
    """Whether a session record is ADOPTABLE engineering evidence.

    Never from the record's own words: the verdicts are RECOMPUTED here from
    its measurements with this code, and must equal what it recorded; every
    identity must equal the committed one; the outer run must be COMPLETE.
    Failed, incomplete or mismatched runs stay booked and billed -- they are
    simply not evidence anything may rest on.
    """
    v = doc.get(RECORD_KEY) or {}
    want = expected_identities(repo)
    got = {k: (v.get("premises") or {}).get(k) for k in want}
    recomputed = verdicts(v) if v else {}
    recorded = v.get("verdicts") or {}

    def flat(vs: Mapping[str, Any]) -> dict[str, Any]:
        return {k: (x.get("verdict") if isinstance(x, dict) else x)
                for k, x in vs.items() if k != "not_gated"}

    reasons = []
    if doc.get("status") != "COMPLETE":
        reasons.append(f"the run is {doc.get('status')!r}, not COMPLETE")
    if not v:
        reasons.append("no stage-V record")
    mismatched = sorted(k for k in want if got[k] != want[k])
    if mismatched:
        reasons.append(f"identities differ from the committed records: "
                       f"{mismatched}")
    if v and flat(recomputed) != flat(recorded):
        reasons.append("the recorded verdicts are not what the record's "
                       "measurements derive")
    executed = ((v.get("premises") or {}).get("source") or {}).get("commit")
    if expected_commit is not None and executed != expected_commit:
        reasons.append(f"the session executed {executed!r}, not the "
                       f"authorized executable commit {expected_commit!r}")
    if v and recomputed.get("validation") != "PASS":
        reasons.append(f"validation is {recomputed.get('validation')}, not "
                       "PASS")
    ok = not reasons
    return {"usable": ok, "reasons": reasons,
            "validation": recomputed.get("validation") if v else None,
            "scored_rows_equivalence": (
                (recomputed.get("scored_rows_equivalence") or {}).get("verdict")
                if ok else None),
            "identities": {"expected": want, "recorded": got},
            "executed_commit": executed, "expected_commit": expected_commit}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the device handoff ---------------------------------------------------------

def place_for_execution(model: Any, device: str) -> dict[str, Any]:
    """Move an operator's child to the session's compute device, timed.

    `ChildBuilder` builds every child on the HOST -- the generic contract every
    structural operator relies on, and the search moves a child to its device
    by checkpoint and reload. Stage V hands V3's child straight to V4, so the
    move happens here, at the validation's own boundary, explicitly and once.
    """
    import torch

    from aadistill.initialization.device import model_device

    before = model_device(model)
    t0 = time.time()
    model.to(device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"from": str(before), "to": str(model_device(model)),
            "seconds": round(time.time() - t0, 3)}


def require_execution_devices(expected: str, *, parent: Any, store: Any
                              ) -> dict[str, str]:
    """The parent (hence every candidate forward), the root sketch and the
    confidence weights all on the expected device TYPE, or a refusal.

    DEPTH derives its compute device from the parent it is handed, so a parent
    left on the host would run the whole round on the CPU against a GPU
    store: silently, until a cross-device op raised mid-round.
    """
    import torch

    from aadistill.initialization.device import model_device

    entry = next(iter(store.entries.values()))
    devices = {"parent_and_candidate_forwards": model_device(parent),
               "root_sketch": entry.support_indices.device,
               "root_confidence": entry.confidence.device}
    want = torch.device(expected).type
    wrong = {k: str(v) for k, v in devices.items() if v.type != want}
    if wrong:
        raise D2ValidationError(
            f"DEPTH would not run on {expected}: {wrong}. Refusing a silent "
            "device fallback")
    return {k: str(v) for k, v in devices.items()}


def source_commit(repo: Path) -> dict[str, Any]:
    """The commit this process executes, and whether the tree is clean."""
    import subprocess

    def git(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                           text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"),
            "tracked_tree_clean": status == "" if status is not None else None}


# --- the stage ------------------------------------------------------------------

def _free(env: Environment) -> None:
    import gc

    gc.collect()
    if env.cuda:
        import torch

        torch.cuda.empty_cache()


def _sync(env: Environment) -> None:
    if env.cuda:
        import torch

        torch.cuda.synchronize()


def _first_batch_rows(env: Environment, model, items) -> Any:
    """Valid-row logits of the first canonical packed batch(es), float32."""
    import torch

    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.packing import packed_batches

    rows = []
    for n, packed in enumerate(packed_batches(
            items, env.micro_batch_size, packing=env.packing,
            pad_id=resolve_pad_id(model), device=env.device)):
        if n >= ARITHMETIC_BATCHES:
            break
        b = packed.batch
        with torch.no_grad():
            logits = model(b.input_ids.to(env.device),
                           attention_mask=b.attention_mask.to(env.device)
                           ).logits[:, :-1]
        for r, length in enumerate(b.lengths):
            rows.append(logits[r, :int(length) - 1].float())
        del logits
    if not rows:
        raise D2ValidationError("no packed batch to read logits from")
    return torch.cat(rows)


def _b_root_kl(env: Environment, b_model, store, items) -> dict[str, Any]:
    """B's per-position KL against the ROOT store, per item, unpadded."""
    import torch

    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.packing import packed_batches
    from aadistill.initialization.scoring.support import sketch_forward_kl

    out = {}
    for packed in packed_batches(items, env.micro_batch_size,
                                 packing=env.packing,
                                 pad_id=resolve_pad_id(b_model),
                                 device=env.device):
        b = packed.batch
        with torch.no_grad():
            logits = b_model(b.input_ids.to(env.device),
                             attention_mask=b.attention_mask.to(env.device)
                             ).logits[:, :-1]
            idx, lp, tail = store.sketch_block(b)
            kl = sketch_forward_kl(idx.to(logits.device), lp.to(logits.device),
                                   tail.to(logits.device), logits,
                                   has_tail=store.has_tail)
        for r, (item, length) in enumerate(zip(b.items, b.lengths)):
            out[str(item["item_id"])] = kl[r, :int(length) - 1].float().cpu()
        del logits, kl
    return out


def _context(env: Environment, impl, model, profile, items, objective, store,
             rows: str, workdir: Path, target):
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.operators.base import OperatorContext
    from aadistill.initialization.planning.operator_config import (
        hashed_operator_config,
    )
    from aadistill.initialization.scoring.objective import root_reference_needed

    config = hashed_operator_config(implementation=impl, policy=env.policy,
                                    support=env.support, items=items,
                                    objective=objective)
    workdir.mkdir(parents=True, exist_ok=True)
    return OperatorContext(
        adapter=env.adapter, model=model,
        parent_spec=env.adapter.spec_of(model), target_spec=target,
        profile=profile, calibration_items=items, seed=env.seed,
        device=env.device, workdir=workdir, config=dict(config),
        execution=ExecutionConfig(micro_batch_size=env.micro_batch_size,
                                  calibration_batch_packing=env.packing,
                                  prediction_rows=rows),
        position_policy=env.policy, distribution_support=env.support,
        scoring_objective=objective,
        root_reference=(store if root_reference_needed(impl, objective)
                        else None))


def _timed_execute(env: Environment, impl, ctx) -> tuple[Any, dict[str, Any]]:
    import torch

    if env.cuda:
        torch.cuda.reset_peak_memory_stats()
    _sync(env)
    t0 = time.time()
    outcome = impl.execute(ctx)
    _sync(env)
    seconds = time.time() - t0
    return outcome, {"seconds": round(seconds, 3),
                     "peak_memory_bytes": (int(torch.cuda.max_memory_allocated())
                                           if env.cuda else None)}


def _state_evaluation(env: Environment, teacher, student, suite, items,
                      content_sha, *, objective, rows: str, batch: int
                      ) -> tuple[dict[str, Any], dict[str, float]]:
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.planning.metrics import StateEvaluator

    ev = StateEvaluator(
        suite, items, device=env.device, position_policy=env.policy,
        execution=ExecutionConfig(
            micro_batch_size=batch,
            calibration_batch_packing=(env.packing if batch > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        distribution_support=env.support, suite_content_sha256=content_sha,
        numerics=env.numerics, scoring_objective=objective)
    ev.prime_reference(teacher)
    _sync(env)
    t0 = time.time()
    evaluation = ev.evaluate(student, "d2_gpu_validation")
    _sync(env)
    run = {"seconds": round(time.time() - t0, 3), "prediction_rows": rows,
           "micro_batch_size": batch,
           "objective": None if objective is None else objective.qualified_id,
           "measurement_protocol_id": ev.measurement_protocol_id}
    values = {k: v for k, v in (evaluation.values or {}).items()
              if isinstance(v, (int, float))}
    del ev
    _free(env)
    return run, values


def _max_abs(a: Mapping[str, float], b: Mapping[str, float],
             prefix: str = "state.objective.") -> float | None:
    keys = [k for k in set(a) & set(b) if k.startswith(prefix)]
    return max((abs(float(a[k]) - float(b[k])) for k in keys), default=None)


def stage_V(*, env: Environment, journal, out: Path, deadline=None,
            flush: Callable[[dict], None] | None = None,
            premises_record: Mapping[str, Any] | None = None
            ) -> dict[str, Any]:
    """The validation's steps, each gated on the deadline covering its bound
    (and, for V5, every core bound still to come); a failed step is recorded,
    never raised, so the steps after it still run where they can."""
    import torch

    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.reference_store import (
        build_root_reference_store, root_reference_identity,
    )
    from stages.d_series import objectives as O
    from stages.d_series import teacher_confidence as TC
    from stages.phase_d1.topk_adoption_driver import (
        _hoard_allocator, objective_agreement, table_agreement,
    )

    result: dict[str, Any] = {"premises": dict(premises_record or {}),
                              "execution_order": list(STEP_ORDER),
                              "step_bounds_s": dict(env.step_bounds_s)}
    if premises_record is None:
        result["premises"]["teacher_revision"] = env.teacher_revision
    keep: dict[str, Any] = {}
    work = out / "work"

    def step(name: str, fn) -> None:
        done = {k for k in result if k in STEP_ORDER}
        bound = float(env.step_bounds_s.get(name, 0.0))
        later_core = (0.0 if name in CORE_STEPS else sum(
            float(env.step_bounds_s.get(c, 0.0)) for c in CORE_STEPS
            if c not in done))
        need = bound + later_core
        if deadline is not None and deadline.remaining() < need:
            result[name] = {"skipped": f"deadline would not cover {need:.0f} s"}
            journal.event(stage=f"V.{name}", status="skipped")
        else:
            t0 = time.time()
            try:
                result[name] = fn()
            except Exception as exc:                       # noqa: BLE001
                result[name] = {"failed": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()}
            result[name]["wall_seconds"] = round(time.time() - t0, 3)
            journal.event(stage=f"V.{name}", status=(
                "failed" if "failed" in result[name] else "ok"))
        result["verdicts"] = verdicts(result)
        if flush is not None:
            flush(result)

    teacher = env.load_teacher()
    profiles = env.profiles()

    def v1():
        out_: dict[str, Any] = {"profiles": {}, "stores_match_declaration": True}
        for profile, items in profiles:
            pid = profile.qualified_id
            identity = root_reference_identity(
                root_identity=env.root_identity,
                profile_hash=profile.profile_hash, support=env.support)
            builds = {}
            stores = {}
            for label, bsz, packing in (
                    ("bsz3_formal", env.micro_batch_size, env.packing),
                    ("bsz1", 1, "original_order_v1")):
                _sync(env)
                t0 = time.time()
                stores[label] = build_root_reference_store(
                    teacher, items, support=env.support, identity=identity,
                    micro_batch_size=bsz, packing=packing, device=env.device)
                _sync(env)
                builds[label] = {"seconds": round(time.time() - t0, 3),
                                 "identity_hash": stores[label].identity_hash,
                                 "content_sha256": stores[label].content_digest()}
            store = stores["bsz3_formal"]
            store.require_items(items)
            match = (store.identity == identity
                     and store.identity_hash == builds["bsz1"]["identity_hash"])
            out_["stores_match_declaration"] &= bool(match)
            rows = _first_batch_rows(env, teacher, items)
            arithmetic = entropy_arithmetic(rows)
            del rows
            out_["profiles"][pid] = {
                "items": len(items), "builds": builds,
                "identity": store.identity, "identity_matches": match,
                "decision": store.decision(),
                "arithmetic": arithmetic,
                "bsz1_vs_bsz3": store_drift(stores["bsz1"], store, items),
                "confidence": confidence_summary(store, items, env.policy),
                "validity": TC.validity_report(store, items, policy=env.policy,
                                               decode=env.decode)}
            keep.setdefault("stores", {})[pid] = store
            keep.setdefault("items", {})[pid] = (profile, items)
            del stores["bsz1"]
            _free(env)
        out_["decode_available"] = env.decode is not None
        return out_

    def v2():
        b_model, b_row = env.materialize_b(work)
        keep["b_model"] = b_model
        ident = b_row.get("identity") or {}
        expected = b_row.get("expected") or {}
        verified = bool(expected) and all(
            ident.get(k) == v for k, v in expected.items())
        out_: dict[str, Any] = {"b": {"seconds": b_row.get("seconds"),
                                      "identity": ident},
                                "b_identity_verified": verified,
                                "profiles": {}}
        if not verified:
            raise D2ValidationError(f"B's identities do not match: {ident} vs "
                                    f"{expected}")
        for pid, store in keep["stores"].items():
            profile, items = keep["items"][pid]
            kl = _b_root_kl(env, b_model, store, items)
            from aadistill.initialization.scoring.positions import (
                PREDICTION_AXIS,
            )
            per_item = [{"kl": kl[str(i["item_id"])],
                         "confidence": store.per_position(i, "confidence").cpu(),
                         "scored": env.policy.weights(
                             i, axis=PREDICTION_AXIS).weights > 0}
                        for i in items]
            out_["profiles"][pid] = {
                "reduction_arithmetic": reduction_arithmetic(O.D2_V2_OBJECTIVE,
                                                             per_item),
                "validity_with_b_kl": TC.validity_report(
                    store, items, policy=env.policy, kl=kl,
                    decode=env.decode)}
        return out_

    def v6():
        if "b_model" not in keep:
            raise D2ValidationError("B was not built (V2)")
        suite, items, content = env.load_suite()
        arms = {}
        values = {}
        for label, objective, rows, batch in (
                ("kl_only", None, "all_v1", env.micro_batch_size),
                ("d2", O.D2_V2_OBJECTIVE, "all_v1", env.micro_batch_size),
                ("d2_scored_v1", O.D2_V2_OBJECTIVE, "scored_v1",
                 env.micro_batch_size),
                ("d2_bsz1", O.D2_V2_OBJECTIVE, "all_v1", 1)):
            arms[label], values[label] = _state_evaluation(
                env, teacher, keep["b_model"], suite, items, content,
                objective=objective, rows=rows, batch=batch)
        if arms["kl_only"]["measurement_protocol_id"] == \
                arms["d2"]["measurement_protocol_id"]:
            raise D2ValidationError("the objective did not move the "
                                    "measurement protocol id")
        del keep["b_model"]
        _free(env)
        return {
            "runs": arms,
            "overhead_seconds_per_evaluation": round(
                arms["d2"]["seconds"] - arms["kl_only"]["seconds"], 3),
            "raw_kl_control": raw_kl_control_relative(values["kl_only"],
                                                      values["d2"]),
            "scored_rows": objective_agreement(
                values["d2"], values["d2_scored_v1"],
                ROW_EQUIVALENCE_TOLERANCES["state_eval_absolute"]),
            "bsz1_vs_bsz3": {"max_abs_state_objective_difference": _max_abs(
                values["d2"], values["d2_bsz1"]),
                "_class": "EXECUTION DRIFT -- reported, never gated"},
            "b_values": {k: values[k] for k in ("kl_only", "d2")},
            "_values_of": "B, the incumbent -- never a D2 candidate"}

    def v3():
        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        out_: dict[str, Any] = {"profile": pid, "operators": {}}
        artifacts = {"ffn.activation_importance_v0": "kept_neurons",
                     "attention.activation_importance_v1": "kept_heads",
                     "width.global_pca_v0": None}
        for impl_id in REQUIRED_STATISTICS_OPERATORS:
            impl = get_implementation(impl_id)
            arms: dict[str, Any] = {}
            children: dict[str, Any] = {}
            #: ONE OPERATOR'S FAILURE IS RECORDED AGAINST THAT OPERATOR and the
            #: others still run; the verdict then refuses a PASS.
            try:
                for label, objective in (("full_scope", O.D2_V2_OBJECTIVE),
                                         ("narrow_scope",
                                          O.D2_NARROW_SCOPE_COMPARISON)):
                    ctx = _context(env, impl, teacher, profile, items,
                                   objective, store, "all_v1",
                                   work / f"v3-{impl_id}-{label}",
                                   env.target_spec)
                    outcome, timing = _timed_execute(env, impl, ctx)
                    arms[label] = {
                        **timing,
                        "config_declares_objective":
                            "scoring_objective" in ctx.config,
                        "local_metrics": dict(outcome.local_metrics.values),
                        "weighted": bool((outcome.trace.get(
                            "scoring_positions") or {}).get(
                                "token_weights_multiplied_by")),
                        "integrity": statistics_integrity(
                            impl, label, outcome, ctx, target=env.target_spec,
                            store_identity_hash=store.identity_hash,
                            objective_hash=O.D2_V2_OBJECTIVE.objective_hash)}
                    children[label] = outcome
                artifact = artifacts[impl_id]
                if artifact:
                    overlap = jaccard_per_layer(
                        children["full_scope"].artifacts[artifact],
                        children["narrow_scope"].artifacts[artifact])
                else:
                    emb = env.adapter.embedding
                    overlap = projection_overlap(
                        emb(teacher).weight.detach(),
                        emb(children["full_scope"].model).weight.detach(),
                        emb(children["narrow_scope"].model).weight.detach())
                out_["operators"][impl_id] = {
                    "arms": arms, "selection_overlap": overlap,
                    "_overlap_class": "SCIENCE DIAGNOSTIC -- no threshold"}
                if impl_id.startswith("ffn.") and arms["full_scope"][
                        "integrity"]["passed"]:
                    keep["ffn_child"] = children["full_scope"].model
            except Exception as exc:                       # noqa: BLE001
                out_["operators"][impl_id] = {
                    "arms": arms, "failed": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
            del children
            _free(env)
        out_["full_scope_integrity"] = _v3_integrity(out_)
        return out_

    def depth_round(rows: str):
        if "ffn_child" not in keep:
            raise D2ValidationError("no non-root parent: V3's full-scope FFN "
                                    "child does not exist")
        from aadistill.initialization.operators.depth._common import DEPTH_FIELD

        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        parent = keep["ffn_child"]
        placement = place_for_execution(parent, env.device)
        devices = require_execution_devices(env.device, parent=parent,
                                            store=store)
        spec = env.adapter.spec_of(parent)
        target = spec.replace(**{DEPTH_FIELD: int(spec[DEPTH_FIELD]) - 1})
        impl = get_implementation("depth.causal_kl_greedy_v1")
        ctx = _context(env, impl, parent, profile, items, O.D2_V2_OBJECTIVE,
                       store, rows, work / f"v4-depth-{rows}", target)
        hoarded = _hoard_allocator() if env.cuda else 0
        outcome, timing = _timed_execute(env, impl, ctx)
        art = outcome.artifacts
        rounds = list(art.get("search_rounds") or [])
        table = [(repr(e["candidate"]), float(e["score"]))
                 for e in (rounds[0].get("table") or [])] if rounds else []
        record = {
            **timing, "prediction_rows": rows, "profile": pid,
            "parent": "V3's full-scope FFN child (a non-root parent)",
            "device_placement": placement,
            "execution_devices": devices,
            "hoarded_gib": round(hoarded / 2**30, 3),
            "reference": outcome.local_metrics.reference,
            "reference_cache": dict(art.get("reference_cache") or {}),
            "parent_reference_cache": dict(art.get("parent_reference_cache")
                                           or {}),
            "timing": {k: v for k, v in (art.get("timing") or {}).items()},
            "table_consistency": depth_table_consistency(art)}
        keep[f"table_{rows}"] = table
        keep[f"artifacts_{rows}"] = art
        del outcome, ctx
        _free(env)
        return record, art

    def v4():
        record, art = depth_round("all_v1")
        round0 = [str(e["candidate"]) for e in
                  (art["search_rounds"][0].get("table") or [])]
        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        record["rankings"] = three_way_rankings(
            art.get("objective_candidates") or {}, round0)
        record["teacher_confidence"] = confidence_summary(store, items,
                                                          env.policy)
        return record

    def v5():
        if "table_all_v1" not in keep:
            raise D2ValidationError("the all_v1 DEPTH round (V4) did not run")
        record, _ = depth_round("scored_v1")
        record["agreement"] = table_agreement(
            keep["table_all_v1"], keep["table_scored_v1"],
            relative_tolerance=ROW_EQUIVALENCE_TOLERANCES[
                "depth_candidate_relative"])
        return record

    for name, fn in (("V1_root_stores_and_validity", v1),
                     ("V2_b_rebuild_and_kl_relation", v2),
                     ("V6_state_evaluation", v6),
                     ("V3_weighted_statistics", v3),
                     ("V4_depth_root_vs_parent", v4),
                     ("V5_depth_scored_rows", v5)):
        step(name, fn)
    del teacher
    keep.clear()
    _free(env)
    result["verdicts"] = verdicts(result)
    return result


__all__ = ["ARITHMETIC_TOLERANCES", "CORE_STEPS", "D2ValidationError",
           "Environment", "EXECUTION_DIAGNOSTICS", "INTEGRITY_TOLERANCES",
           "RECORD_KEY", "REQUIRED_STATISTICS_OPERATORS",
           "ROW_EQUIVALENCE_TOLERANCES", "SCIENCE_DIAGNOSTICS", "STEP_ORDER",
           "adopt", "entropy_arithmetic", "expected_identities",
           "place_for_execution", "premises", "require_execution_devices",
           "source_commit",
           "production_environment",
           "reduction_arithmetic", "stage_V", "verdicts"]

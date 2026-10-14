"""Stage W of `topk_adoption_driver.py`: D2 version 3's minimal GPU validation.

    ADOPT_STAGES=W bash scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh <run>

ONE L40S session on the infrastructure Stage V already exercised -- the same
launcher, remote payload, `$0` preflight, journal, deadline, flushed
`adoption.json`, teardown, booking and adoption. This module adds only what
version 3's three new numeric definitions need measured on the real teacher:
the union Top-K KL with exact retrieval, the deterministic Top-K on real bf16
ties, and the teacher's Top-200 + tail confidence. B is NOT rebuilt: the
candidates are the teacher's own compressed variants (an FFN child under
version 3's weighted statistics, and DEPTH-style bypasses of it).

**Numeric acceptance, REGISTERED BEFORE THE SESSION, in three classes that are
never mixed** (the plan imports these constants, and the premises refuse a
plan whose registered tolerances differ from the code's):

1. **ARITHMETIC** -- one set of inputs, two computations. The union KL in
   float32 against a float64 oracle on the SAME logits -- the teacher's side
   is the FLOAT32 HEAD distribution the root store defines (never the model's
   native bf16 logits, which are a different distribution); the KL bounds in
   float64; the retrieval of candidate-only entries against the full head's
   log-softmax; the K+1 confidence against float64 on the same sketch; the
   `c x KL` item mean against float64 on the same (KL, c); and, inside ONE
   version-3 state evaluation, ``state.objective.raw_kl.* ==
   state.teacher_kl.*``. GATED by :data:`ARITHMETIC_TOLERANCES`.
2. **EXECUTION DRIFT and NUMERICAL-PROTOCOL DIFFERENCE** -- bsz 1 vs 3, and
   the float32 head against the native bf16 forward (logits, Top-K support
   agreement, teacher probabilities, KL, confidence). REPORTED, NEVER GATED:
   the first is a different GEMM shape, the second is the stated definition of
   the teacher's logits, and neither may be read as an effect of the
   confidence or the union support.
3. **ROW-SELECTION EQUIVALENCE at the formal bsz=3** -- all_v1 vs scored_v1 on
   one DEPTH round (W4 vs W5) and on state evaluation (W6), under R0's
   registered tolerances. Decides only whether D2 may run scored_v1.

Everything else -- the confidence distribution and its weighted
contributions, the union sizes, the weighted-vs-unweighted rankings, the
historical reference-Top-K KL beside the union KL, the component timings --
is a diagnostic, recorded as measured, and never a reason to change K, the
mapping, the tail formula or a tolerance.

The models, items and suite arrive through an ENVIRONMENT object, and the
CPU rehearsal runs this stage through the driver's own `main()` with a toy
environment: the same premises, protocol bindings, journal, deadline, flushes
and `adoption.json`. AUTHORIZES NOTHING.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VALIDATION_DIR = ("logs/stages/stage-1/families/d_series/validations/"
                  "d2v3-gpu-validation/v1")
PLAN = f"{VALIDATION_DIR}/plan.json"
CAMPAIGN = f"{VALIDATION_DIR}/campaign.json"
AUTHORIZATION = f"{VALIDATION_DIR}/authorization.json"
AUTHORIZATION_PROPOSED = f"{VALIDATION_DIR}/authorization.proposed.json"
RECORD_KEY = "W_d2v3_validation"

#: CLASS 1. Per position, |KL_U(float32) - KL_U(float64)| <= abs + rel * KL64:
#: the float32 floor is ABSOLUTE (each log probability carries the rounding of
#: its row's log-normalizer, ~4 eps |log Z|, measured 1.6e-5 at most on real
#: logits at $0), so a purely relative gate would fail every near-zero KL and
#: a purely absolute one would pass a relative error on a large one. The
#: retrieval differs from the full head only by float32 summation order
#: (measured <= 6e-5 nats). The confidence divides an entropy of at most
#: log(K+1) by log(K+1): 1e-5. The c x KL item mean and the raw-KL control are
#: Stage V's, unchanged.
ARITHMETIC_TOLERANCES = {
    "union_kl_abs_nats_vs_float64": 1e-4,
    "union_kl_rel_vs_float64": 1e-4,
    "retrieval_abs_nats_vs_full_head": 2e-4,
    "confidence_abs_vs_float64": 1e-5,
    "bound_violations_beyond_float32_slack": 0,
    "item_objective_relative_vs_float64": 1e-5,
    "raw_kl_control_relative": 1e-6,
}
#: The float64 oracle's own bound check: exact up to float64 summation.
ORACLE_BOUND_SLACK = 1e-12
INTEGRITY_TOLERANCES = {"statistics_denominator_relative": 1e-9}
#: CLASS 3. R0's registered tolerances, carried over unchanged.
ROW_EQUIVALENCE_TOLERANCES = {"depth_candidate_relative": 1e-5,
                              "depth_choice": "exact",
                              "state_eval_absolute": 1e-5}
EXECUTION_DIAGNOSTICS = (
    "bsz1_vs_bsz3_union_store", "bsz1_vs_bsz3_state_evaluation",
    "float32_head_vs_native_bf16", "component_timings_and_memory",
    "cache_modes_and_hits")
SCIENCE_DIAGNOSTICS = (
    "teacher_topk_confidence_distribution_and_contributions",
    "union_size_overlap_and_tail_mass", "union_vs_full_kl_ratio",
    "weighted_vs_unweighted_rankings",
    "historical_reference_topk_kl_beside_the_union_kl",
    "ffn_full_vs_narrow_selection_overlap")
STEP_ORDER = ("W1_union_stores_and_confidence", "W2_ffn_child_weighted",
              "W3_union_numerics", "W4_depth_union_all_v1",
              "W5_depth_union_scored_v1", "W6_state_evaluation")
#: Every step is core: W5 decides the scored_v1 price lever and W6 carries the
#: raw-KL control.
CORE_STEPS = STEP_ORDER
#: The union numerics sample: items per profile and bypass candidates of the
#: FFN child, at these fractions of its depth.
NUMERICS_SAMPLE = {"items_per_profile": 8, "bypass_candidates": 3}
CANDIDATE_FRACTIONS = (0.15, 0.5, 0.85)
#: Rows per float64 oracle block: ~6 [rows, V] float64 tensors live at once.
ORACLE_ROW_CHUNK = 64
#: The confidence arithmetic reads this many packed batches per profile.
ARITHMETIC_BATCHES = 1
#: DEPTH's opt-in synchronized phase split, ON for W4/W5 only, so the
#: objective's seconds are attributed rather than carrying the forwards' tail.
SYNC_TELEMETRY_ENV = "AADISTILL_DEPTH_SYNC_TELEMETRY"


class D2V3ValidationError(RuntimeError):
    pass


# --- class 1: the float64 oracle ------------------------------------------------

def _f64_log_softmax_and_probs(z):
    """``(log p, p)`` in float64 with ONE exponential per element.

    ``e = exp(z - max)``, ``S = sum e``: ``p = e / S`` and ``log p = (z - max)
    - log S``. The float64 exponential is the oracle's whole cost on a GPU
    (FP64 at 1/64 of FP32), so it is taken once."""
    import torch

    z = z.double()
    m = z.max(dim=-1, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(z - m)
    s = e.sum(dim=-1, keepdim=True)
    return (z - m) - torch.log(s), e / s


def oracle_kls(p_z, candidates: Sequence[tuple[Any, Any]], st, *,
               chunk: int = ORACLE_ROW_CHUNK) -> list[dict[str, Any]]:
    """Per-row float64 forward KLs of ``softmax(p_z)`` against each candidate.

    ``candidates`` is a list of ``(q_z, sc)``: the candidate's ``[n, V]``
    logits and its own Top-K ``[n, K]`` (or ``None``). For each, per row:

    * ``full`` -- over the vocabulary;
    * ``topk`` -- on the reference's ``st`` + one tail;
    * ``union`` -- on ``st ∪ sc`` + one tail (when ``sc`` is given), with
      ``p_tail_union`` / ``q_tail_union``;
    * ``prob_abs_on_st`` -- ``max_i |p_i - q_i|`` over ``st``.

    The reference's float64 pass is computed ONCE per block of rows and
    shared by every candidate. Tails are DIRECT sums of the complement's
    probabilities -- no ``1 - sum`` cancellation -- and zero reference mass
    contributes zero. At most ~4 + 3 x candidates ``[chunk, V]`` float64
    tensors exist at once."""
    import torch

    n = int(p_z.shape[0])
    dev = p_z.device
    outs = []
    for _, sc in candidates:
        names = ["full", "topk", "prob_abs_on_st"]
        if sc is not None:
            names += ["union", "p_tail_union", "q_tail_union"]
        outs.append({k: torch.empty(n, dtype=torch.float64, device=dev)
                     for k in names})

    def tail_term(pr, qr):
        safe = pr > 0
        return torch.where(safe, pr * (torch.log(torch.where(safe, pr, 1.0))
                                       - torch.log(qr)), torch.zeros_like(pr))

    step = max(1, int(chunk))
    for s in range(0, n, step):
        e = min(s + step, n)
        p_log, p = _f64_log_softmax_and_probs(p_z[s:e])
        st_b = st[s:e].to(dev).long()
        mask_t = torch.zeros_like(p, dtype=torch.bool)
        mask_t.scatter_(-1, st_b, True)
        p_st = p.gather(-1, st_b)
        for (q_z, sc), out in zip(candidates, outs):
            q_log, q = _f64_log_softmax_and_probs(q_z[s:e].to(dev))
            term = torch.where(p > 0, p * (p_log - q_log), torch.zeros_like(p))
            out["full"][s:e] = term.sum(-1)
            out["prob_abs_on_st"][s:e] = (p_st - q.gather(-1, st_b)).abs() \
                .max(-1).values
            parts = [("topk", mask_t)]
            if sc is not None:
                mask_u = mask_t.clone()
                mask_u.scatter_(-1, sc[s:e].to(dev).long(), True)
                parts.append(("union", mask_u))
            for name, mask in parts:
                on = torch.where(mask, term, torch.zeros_like(term)).sum(-1)
                pr = torch.where(mask, torch.zeros_like(p), p).sum(-1)
                qr = torch.where(mask, torch.zeros_like(q), q).sum(-1)
                out[name][s:e] = on + tail_term(pr, qr)
                if name == "union":
                    out["p_tail_union"][s:e] = pr
                    out["q_tail_union"][s:e] = qr
            del q_log, q, term
        del p_log, p, mask_t
    return outs


def union_arithmetic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The float32 union KL against the float64 oracle on the SAME logits,
    the oracle's own bounds, and the float32 union KL against the float64
    full KL with float32 slack. ``rows``: one entry per (item, candidate)
    with ``kl32``, ``kl_ref32``, ``oracle`` (:func:`oracle_kls`) and
    ``log_z_ref`` (the teacher's per-row log-normalizer, for the slack)."""
    import torch

    t = ARITHMETIC_TOLERANCES
    eps = float(torch.finfo(torch.float32).eps)
    worst_excess, err_max, violations, oracle_viol, f32_viol = 0.0, 0.0, 0, 0, 0
    positions = 0
    for r in rows:
        o = r["oracle"]
        k32 = r["kl32"].double().to(o["union"].device)
        k64 = o["union"]
        finite = torch.isfinite(k32) & torch.isfinite(k64)
        diff = (k32 - k64).abs()
        allowed = t["union_kl_abs_nats_vs_float64"] + \
            t["union_kl_rel_vs_float64"] * k64.abs()
        bad = finite & (diff > allowed)
        violations += int(bad.sum())
        violations += int((torch.isfinite(k32) != torch.isfinite(k64)).sum())
        if bool(finite.any()):
            err_max = max(err_max, float(diff[finite].max()))
            worst_excess = max(worst_excess,
                               float((diff - allowed)[finite].max()))
        slack64 = ORACLE_BOUND_SLACK * (1 + o["full"].abs())
        oracle_viol += int(((o["topk"] - o["union"]) > slack64).sum()
                           + ((o["union"] - o["full"]) > slack64).sum())
        slack32 = 4 * eps * r["log_z_ref"].double().abs().to(k64.device)
        f32_viol += int((((r["kl_ref32"].double().to(k64.device) - k32)
                          > slack32) | ((k32 - o["full"]) > slack32)).sum())
        positions += int(k64.numel())
    return {"positions": positions, "max_abs_difference": err_max,
            "position_violations": violations,
            "max_excess_over_tolerance": worst_excess,
            "oracle_bound_violations": oracle_viol,
            "float32_bound_violations_beyond_slack": f32_viol,
            "within_tolerance": (positions > 0 and violations == 0),
            "bounds_hold": (positions > 0 and oracle_viol == 0
                            and f32_viol <= t[
                                "bound_violations_beyond_float32_slack"])}


def confidence_arithmetic(store: Any, items: Sequence[Mapping[str, Any]], *,
                          chunk: int = ORACLE_ROW_CHUNK) -> dict[str, Any]:
    """The store's K+1 entropy and confidence against float64 on the SAME
    float32 head logits (recomputed from the stored hidden state) and the
    SAME support: the arithmetic of the sketch, the tail and the mapping.
    Also reports how often the deterministic Top-K of the recomputed logits
    is the stored support (a re-execution, so a diagnostic). Blocked over
    rows: no ``[T, V]`` float64 tensor exists."""
    import torch

    from aadistill.initialization.scoring.support import deterministic_topk

    k, v = int(store.top_k), int(store.vocab_size)
    categories = k + 1 if k < v else v
    worst_h, worst_c, rows, same = 0.0, 0.0, 0, 0
    for item in items:
        r = store.retrieval_item(item)
        st_all = store.sketch_item(item).support_indices
        h_store = store.per_position(item, "entropy").double()
        c_store = store.per_position(item, "confidence").double()
        for s in range(0, r.rows, max(1, int(chunk))):
            e = min(s + max(1, int(chunk)), r.rows)
            z = store.head.logits(r.hidden[s:e])
            log_p, p = _f64_log_softmax_and_probs(z)
            st = st_all[s:e].to(z.device)
            lp, pk = log_p.gather(-1, st), p.gather(-1, st)
            h = -(torch.where(pk > 0, pk * lp, torch.zeros_like(pk))).sum(-1)
            if k < v:
                mask = torch.ones_like(p, dtype=torch.bool)
                mask.scatter_(-1, st, False)
                pr = torch.where(mask, p, torch.zeros_like(p)).sum(-1)
                h = h - torch.where(pr > 0, pr * torch.log(
                    torch.where(pr > 0, pr, 1.0)), torch.zeros_like(pr))
            c64 = (1.0 - h / math.log(categories)).clamp(0.0, 1.0)
            worst_h = max(worst_h, float((h_store[s:e].to(h.device) - h)
                                         .abs().max()))
            worst_c = max(worst_c, float((c_store[s:e].to(c64.device) - c64)
                                         .abs().max()))
            _, idx = deterministic_topk(z, k)
            same += int((idx.sort(-1).values == st.sort(-1).values)
                        .all(-1).sum())
            rows += e - s
            del z, log_p, p
    tol = ARITHMETIC_TOLERANCES["confidence_abs_vs_float64"]
    return {"positions": rows, "categories": categories,
            "entropy_max_abs_vs_float64": worst_h,
            "confidence_max_abs_vs_float64": worst_c,
            "support_reproduced_fraction": same / max(1, rows),
            "within_tolerance": rows > 0 and worst_c <= tol}


def retrieval_arithmetic(max_abs_error: float) -> dict[str, Any]:
    """The retrieved log probabilities of candidate-only entries against the
    full head's log-softmax at the same rows: the largest absolute
    difference, against the registered tolerance."""
    tol = ARITHMETIC_TOLERANCES["retrieval_abs_nats_vs_full_head"]
    return {"max_abs_nats_vs_full_head": float(max_abs_error),
            "within_tolerance": math.isfinite(float(max_abs_error))
            and float(max_abs_error) <= tol}


def objective_reduction_arithmetic(objective: Any,
                                   per_item: Sequence[Mapping[str, Any]]
                                   ) -> dict[str, Any]:
    """The operator's float32 ``c x KL_U`` item mean against float64 on the
    SAME (KL, c): the reduction alone, as Stage V checked version 2's."""
    import torch

    from aadistill.initialization.scoring.objective import (
        UNION_DIAGNOSTICS, item_component_means, position_terms,
    )

    worst, n = 0.0, 0
    for e in per_item:
        kl = e["kl"].float()
        conf = e["confidence"].float().to(kl.device)
        scored = e["scored"].to(kl.device).bool()
        if not bool(scored.any()) or not bool(torch.isfinite(kl[scored]).all()):
            continue
        terms = position_terms(
            objective, kl, None, None, None,
            reference_entropy=torch.zeros_like(conf),
            reference_confidence=conf,
            union={k: torch.zeros_like(kl) for k in UNION_DIAGNOSTICS})
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


# --- class 1: the raw-KL control, within ONE evaluation --------------------------

RAW_KL_PAIRS = (("state.objective.raw_kl.equal_domain_mean",
                 "state.teacher_kl.equal_domain_mean"),
                ("state.objective.raw_kl.worst_domain",
                 "state.teacher_kl.worst_domain"),
                ("state.objective.raw_kl.critical_token",
                 "state.critical_token_kl"))


def raw_kl_control(objective_run: Mapping[str, Any],
                   kl_run: Mapping[str, Any]) -> dict[str, Any]:
    """``state.objective.raw_kl.* == state.teacher_kl.*`` -- of ONE evaluation.

    Each run is ``{"values", "measurement_protocol_id",
    "distribution_support", "objective"}``. The control is defined only when
    both sides are the SAME evaluation: the same protocol id, the union
    support, the version-3 objective. A KL from another support (the
    historical reference-Top-K diagnostic) measures a different partition;
    comparing it would make the control test the support, not the
    reduction, so it is REFUSED rather than computed.
    """
    reasons = []
    if objective_run.get("measurement_protocol_id") != \
            kl_run.get("measurement_protocol_id"):
        reasons.append("the two sides are different evaluations "
                       "(measurement protocol ids differ)")
    sup = [((r.get("distribution_support") or {}).get("support_id"))
           for r in (objective_run, kl_run)]
    if sup[0] != sup[1]:
        reasons.append(f"the two sides reduce over different supports {sup}")
    if not objective_run.get("objective"):
        reasons.append("the objective side carries no objective")
    if reasons:
        return {"defined": False, "refused": reasons,
                "within_tolerance": False}
    a, b = objective_run["values"], kl_run["values"]
    pairs = list(RAW_KL_PAIRS)
    src, dst = ("state.objective.raw_kl.critical_token.",
                "state.critical_token_kl.")
    pairs += [(k, dst + k[len(src):]) for k in a
              if k.startswith(src) and dst + k[len(src):] in b]
    rel = [abs(float(a[x]) - float(b[y])) / max(abs(float(b[y])), 1e-30)
           for x, y in pairs if x in a and y in b]
    worst = max(rel) if rel else None
    return {"defined": True, "support": sup[0],
            "metrics_compared": len(rel), "max_relative_difference": worst,
            "within_tolerance": (worst is not None and worst <= float(
                ARITHMETIC_TOLERANCES["raw_kl_control_relative"]))}


# --- class 2 and the diagnostics ---------------------------------------------------

def _q(x, qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict[str, float]:
    import torch

    x = x.double().flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {}
    if x.numel() > 1_000_000:
        x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(
            0))[:1_000_000].to(x.device)]
    vals = torch.quantile(x.cpu(), torch.tensor(qs, dtype=torch.float64))
    out = {f"p{int(round(q * 100)):02d}": float(v) for q, v in zip(qs, vals)}
    out.update({"mean": float(x.mean()), "max": float(x.max()),
                "min": float(x.min()), "n": int(x.numel())})
    return out


def protocol_difference(head_z, native_z, oracle_native: Mapping[str, Any],
                        *, top_k: int) -> dict[str, Any]:
    """The float32 head against the native bf16 forward, ONE item's rows.

    A NUMERICAL-PROTOCOL difference, by definition: the root store's teacher
    distribution is the float32 head; the model's own forward rounds the same
    product to bf16. Reported per row: logit differences, deterministic Top-K
    support agreement, the K+1 confidence difference, and -- from the shared
    float64 oracle pass, ``oracle_native`` -- the KL(head || native) and the
    teacher probability difference on the head's Top-K."""
    import torch

    from aadistill.initialization.scoring.objective import (
        reference_topk_confidence,
    )
    from aadistill.initialization.scoring.support import (
        deterministic_topk, sketch_reference_deterministic,
    )

    k, v = int(top_k), int(head_z.shape[-1])
    native32 = native_z.float()
    diff = (head_z - native32).abs().max(-1).values
    _, ih = deterministic_topk(head_z, k)
    _, inn = deterministic_topk(native_z, k)
    a, b = ih.sort(-1).values, inn.sort(-1).values
    pos = torch.searchsorted(b, a).clamp(max=k - 1)
    inter = (b.gather(-1, pos) == a).sum(-1).double()
    jac = inter / (2 * k - inter)
    tgt = torch.zeros(int(head_z.shape[0]), dtype=torch.long,
                      device=head_z.device)
    conf = []
    for z in (head_z, native32):
        sk, _ = sketch_reference_deterministic(z, tgt, top_k=k)
        conf.append(reference_topk_confidence(
            sk.support_log_probs, sk.tail_log_prob, top_k=k,
            vocab_size=v)[1].double())
    del native32
    return {"logit_max_abs_per_row": diff.cpu(),
            "topk_jaccard": jac.cpu(),
            "topk_identical": (jac == 1.0).cpu(),
            "teacher_prob_abs_on_head_topk":
                oracle_native["prob_abs_on_st"].cpu(),
            "kl_head_vs_native": oracle_native["full"].cpu(),
            "confidence_abs": (conf[0] - conf[1]).abs().cpu()}


def summarize_protocol_difference(parts: Sequence[Mapping[str, Any]]
                                  ) -> dict[str, Any]:
    import torch

    if not parts:
        return {}
    cat = {k: torch.cat([p[k].double() for p in parts]) for k in parts[0]}
    return {"logit_max_abs_per_row": _q(cat["logit_max_abs_per_row"]),
            "topk_jaccard": _q(cat["topk_jaccard"]),
            "topk_identical_fraction": float(cat["topk_identical"].mean()),
            "teacher_prob_abs_on_head_topk": _q(
                cat["teacher_prob_abs_on_head_topk"]),
            "kl_head_vs_native": _q(cat["kl_head_vs_native"]),
            "confidence_abs": _q(cat["confidence_abs"]),
            "_class": ("NUMERICAL-PROTOCOL DIFFERENCE: the stated float32 "
                       "head definition against the native bf16 forward. "
                       "Reported, never gated, and never an effect of the "
                       "confidence or the union support")}


def rankings_v3(candidates: Mapping[str, Mapping[str, Any]],
                round0: Sequence[str]) -> dict[str, Any]:
    """Four metrics over ONE set of round-0 candidates: the parent KL, the
    root teacher-Top-K + tail KL, the root UNWEIGHTED union KL and the
    root WEIGHTED union KL (the decision)."""
    from stages.phase_d1.topk_adoption_driver import spearman

    metric = {"parent_kl": "parent_kl",
              "root_teacher_topk_kl": "kl_reference_topk",
              "root_union_kl_unweighted": "raw_kl",
              "root_union_kl_teacher_weighted": "objective"}
    keys = [k for k in round0 if k in candidates
            and all(f in candidates[k] for f in metric.values())]
    cols = {n: [float(candidates[k][f]) for k in keys]
            for n, f in metric.items()}
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


def union_candidate_stats(candidates: Mapping[str, Mapping[str, Any]]
                          ) -> dict[str, Any]:
    """The union's size, overlap and tails across DEPTH's candidates."""
    import torch

    out = {}
    for name in ("union_size", "union_overlap", "reference_union_tail_mass",
                 "candidate_union_tail_mass", "kl_reference_topk", "raw_kl",
                 "objective"):
        vals = [float(c[name]) for c in candidates.values() if name in c]
        if vals:
            out[name] = _q(torch.tensor(vals, dtype=torch.float64))
    return out


# --- verdicts -------------------------------------------------------------------

def _ran(step: Any) -> bool:
    return bool(step) and "skipped" not in step and "failed" not in step


def _tri(step: Any, ok: Callable[[Mapping[str, Any]], Any]) -> bool | None:
    """None = not measured; False = measured and wrong, or failed outright."""
    if not step or "skipped" in step:
        return None
    if "failed" in step:
        return False
    v = ok(step)
    return None if v is None else bool(v)


def verdicts(result: Mapping[str, Any]) -> dict[str, Any]:
    """Three verdicts, computed from the record, never typed."""
    w1 = result.get("W1_union_stores_and_confidence") or {}
    w2 = result.get("W2_ffn_child_weighted") or {}
    w3 = result.get("W3_union_numerics") or {}
    w4 = result.get("W4_depth_union_all_v1") or {}
    w5 = result.get("W5_depth_union_scored_v1") or {}
    w6 = result.get("W6_state_evaluation") or {}
    arith: dict[str, Any] = {}
    for pid, prof in (w1.get("profiles") or {}).items():
        arith[f"confidence_{pid}"] = (prof.get("confidence_arithmetic")
                                      or {}).get("within_tolerance")
    if not w1.get("profiles"):
        arith["confidence"] = _tri(w1, lambda s: None)
    arith["union_kl_vs_float64"] = _tri(
        w3, lambda s: (s.get("union_arithmetic") or {}).get(
            "within_tolerance"))
    arith["kl_bounds"] = _tri(
        w3, lambda s: (s.get("union_arithmetic") or {}).get("bounds_hold"))
    arith["retrieval_vs_full_head"] = _tri(
        w3, lambda s: (s.get("retrieval") or {}).get("within_tolerance"))
    arith["objective_reduction"] = _tri(
        w3, lambda s: (s.get("objective_reduction") or {}).get(
            "within_tolerance"))
    arith["raw_kl_control"] = _tri(
        w6, lambda s: (s.get("raw_kl_control") or {}).get("within_tolerance"))
    if any(v is False for v in arith.values()):
        arithmetic = "FAIL"
    elif any(v is None for v in arith.values()):
        arithmetic = "INCOMPLETE"
    else:
        arithmetic = "PASS"

    integ = {
        "teacher_identity": (result.get("premises") or {}).get(
            "teacher_revision") is not None,
        "stores_match_declaration": _tri(
            w1, lambda s: s.get("stores_match_declaration")),
        "ffn_full_scope_weighted": _tri(
            w2, lambda s: all(((s.get("arms") or {}).get(a) or {}).get(
                "integrity", {}).get("passed") for a in
                ("full_scope", "narrow_scope"))),
        "depth_reads_the_root_with_retrieval": _tri(
            w4, lambda s: (s.get("reference") == "root_reference"
                           and (s.get("reference_cache") or {}).get("mode")
                           == "root_reference_store"
                           and bool((s.get("reference_cache") or {}).get(
                               "retrieval_bytes")))),
        "depth_table_consistent": _tri(
            w4, lambda s: (s.get("table_consistency") or {}).get(
                "consistent")),
        "state_evaluation_reads_its_union_store": _tri(
            w6, lambda s: all(
                r.get("teacher_forwards_during_evaluate") == 0
                and r.get("union_reference")
                for k, r in (s.get("runs") or {}).items()
                if r.get("objective"))),
    }
    if any(v is False for v in integ.values()):
        integrity = "FAIL"
    elif any(v is None for v in integ.values()):
        integrity = "INCOMPLETE"
    else:
        integrity = "PASS"
    se = (w6.get("scored_rows") or {}) if _ran(w6) else {}
    dp = (w5.get("agreement") or {}) if _ran(w5) else {}
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
                      "tolerances": INTEGRITY_TOLERANCES},
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


# --- the environment: what the stage runs on ---------------------------------

@dataclass
class Environment:
    """Everything model- or host-specific, injected. One production instance
    (:func:`production_environment`); the rehearsal builds a CPU toy."""

    repo: Path
    device: str
    adapter: Any
    support: Any
    historical_support: Any
    policy: Any
    numerics: Any
    target_spec: Any
    seed: int
    micro_batch_size: int
    packing: str
    root_identity: Mapping[str, Any]
    teacher_revision: str | None
    load_teacher: Callable[[], Any]
    profiles: Callable[[], list[tuple[Any, list]]]
    load_suite: Callable[[], tuple[Any, list, str]]
    decode: Callable[[int], str] | None = None
    step_bounds_s: Mapping[str, float] = field(default_factory=dict)
    objective: Any = None
    narrow_objective: Any = None

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
    from stages.d_series import objectives as O
    from stages.d_series import scoring_protocol as SP
    from stages.phase_a3 import a3_session as A3S
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
    plan = json.loads((repo / PLAN).read_text())
    profile_ids = json.loads((repo / W.D1_DESIGN).read_text())[
        "search_stage"]["profiles"]

    def load_teacher():
        m = adapter.load(teacher_path, dtype="bfloat16", device=device)
        m.config.use_cache = False
        return m

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
        repo=repo, device=device, adapter=adapter, support=O.D2_SUPPORT,
        historical_support=SP.D_SERIES_SUPPORT,
        policy=get_position_policy(f"{POLICY_ID}@v1"), numerics=numerics,
        target_spec=spec.target_spec, seed=spec.seed,
        micro_batch_size=SP.D_SERIES_MICRO_BATCH_SIZE,
        packing=SP.D_SERIES_BATCH_PACKING,
        root_identity={"teacher_id": root["root_teacher_id"],
                       "teacher_sha256": root["root_teacher_sha256"]},
        teacher_revision=binding.get("revision"),
        load_teacher=load_teacher, profiles=profiles,
        load_suite=lambda: _load_suite(load_state_eval, repo),
        decode=decode, step_bounds_s=dict(plan["step_bounds_s"]),
        objective=O.D2_OBJECTIVE,
        narrow_objective=O.D2_V3_NARROW_SCOPE_COMPARISON)


# --- the $0 premises, identities and adoption --------------------------------------

def premises(repo: Path) -> dict[str, Any]:
    """Everything stage W rests on, before a model is resident. Raises.

    The objective, support and contract are version 3's; the
    preregistration is the DRAFT version 3 -- bound by its exact hash, which
    is not an approval of it."""
    from stages.d_series import objectives as O
    from stages.phase_d1 import d1_session as D1S
    from stages.phase_d1.topk_adoption_driver import TEACHER_BINDING
    from stages.phase_d2 import d2_gpu_validation as V
    from stages.phase_d2 import d2_session as S

    contract = json.loads((repo / O.CONTRACT).read_text())
    if (contract.get("d2") or {}).get("objective_hash") != \
            O.D2_OBJECTIVE.objective_hash:
        raise D2V3ValidationError("the committed contract's D2 is not the "
                                  "version-3 objective this code declares")
    plan = json.loads((repo / PLAN).read_text())
    for key, mine in (("arithmetic", ARITHMETIC_TOLERANCES),
                      ("row_equivalence", ROW_EQUIVALENCE_TOLERANCES),
                      ("integrity", INTEGRITY_TOLERANCES)):
        if plan["acceptance"][key]["tolerances"] != mine:
            raise D2V3ValidationError(
                f"the plan registers {key} tolerances "
                f"{plan['acceptance'][key]['tolerances']} and the code applies "
                f"{mine}; the registered ones are the ones that count")
    if plan["objective_hash"] != O.D2_OBJECTIVE.objective_hash:
        raise D2V3ValidationError("the plan validates another objective")
    if plan["support"] != O.D2_SUPPORT.as_dict():
        raise D2V3ValidationError("the plan validates another support")
    if plan["contract_hash"] != contract["contract_hash"]:
        raise D2V3ValidationError("the plan was registered against another "
                                  "contract revision")
    prereg = S.preregistration(repo)
    if plan["preregistration_hash"] != prereg["preregistration_hash"]:
        raise D2V3ValidationError("the plan was registered against another "
                                  "preregistration draft")
    if set(plan["step_bounds_s"]) != set(STEP_ORDER):
        raise D2V3ValidationError("the plan's step bounds do not name this "
                                  "stage's steps")
    binding = json.loads((repo / TEACHER_BINDING).read_text())
    D1S._register_frozen_operators()
    return {"objective_hash": O.D2_OBJECTIVE.objective_hash,
            "objective": O.D2_OBJECTIVE.qualified_id,
            "support": O.D2_SUPPORT.as_dict(),
            "contract_revision": contract["revision"],
            "contract_hash": contract["contract_hash"],
            "preregistration_hash": prereg["preregistration_hash"],
            "preregistration_status": prereg.get("status"),
            "plan_sha256": V._sha256(repo / PLAN),
            "teacher_revision": binding.get("revision"),
            "root_teacher": D1S.root_teacher_identity(repo),
            "step_bounds_s": dict(plan["step_bounds_s"]),
            "source": V.source_commit(repo)}


def expected_identities(repo: Path) -> dict[str, str]:
    """What a booked session must carry to count as THIS validation: the
    version-3 objective and the contract, preregistration and plan its plan
    registered, the plan bound by its sha256."""
    from stages.d_series import objectives as O
    from stages.phase_d2 import d2_gpu_validation as V

    plan = json.loads((repo / PLAN).read_text())
    if plan["objective_hash"] != O.D2_OBJECTIVE.objective_hash:
        raise D2V3ValidationError("the plan validates another objective")
    return {"objective_hash": O.D2_OBJECTIVE.objective_hash,
            "contract_hash": plan["contract_hash"],
            "preregistration_hash": plan["preregistration_hash"],
            "plan_sha256": V._sha256(repo / PLAN)}


def adopt(doc: Mapping[str, Any], repo: Path,
          expected_commit: str | None = None) -> dict[str, Any]:
    """Whether a session record is ADOPTABLE engineering evidence: COMPLETE,
    identities equal to the committed records, verdicts RECOMPUTED from its
    measurements equal to the recorded ones, the executed commit the
    authorized one, validation PASS. Anything else stays booked and billed
    and unlocks nothing."""
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
        reasons.append("no stage-W record")
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


# --- the stage ----------------------------------------------------------------------

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


def _peak_reset(env: Environment) -> None:
    if env.cuda:
        import torch

        torch.cuda.reset_peak_memory_stats()


def _peak(env: Environment) -> int | None:
    if env.cuda:
        import torch

        return int(torch.cuda.max_memory_allocated())
    return None


def _timed(env: Environment, fn: Callable[[], Any], repeat: int = 1
           ) -> tuple[Any, float]:
    """``(result, seconds per call)``, synchronized on both sides."""
    _sync(env)
    t0 = time.perf_counter()
    out = None
    for _ in range(max(1, repeat)):
        out = fn()
    _sync(env)
    return out, (time.perf_counter() - t0) / max(1, repeat)


def _first_batch_items(env: Environment, model, items) -> list:
    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.packing import packed_batches

    out = []
    for n, packed in enumerate(packed_batches(
            items, env.micro_batch_size, packing=env.packing,
            pad_id=resolve_pad_id(model), device="cpu")):
        if n >= ARITHMETIC_BATCHES:
            break
        out.extend(packed.batch.items)
    if not out:
        raise D2V3ValidationError("no packed batch to read")
    return out


def sample_items(items: Sequence[Mapping[str, Any]], n: int) -> list:
    """``n`` items by a STATED rule: the mixture's own order, every
    ``len/n``-th, so the sample spans its domains without a seed."""
    items = list(items)
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def bypass_layers(depth: int) -> list[int]:
    return sorted({min(depth - 1, max(0, int(round(f * depth))))
                   for f in CANDIDATE_FRACTIONS})[
        :NUMERICS_SAMPLE["bypass_candidates"]]


def _context(env: Environment, impl, model, profile, items, objective, store,
             rows: str, workdir: Path, target):
    from stages.phase_d2 import d2_gpu_validation as V

    return V._context(env, impl, model, profile, items, objective, store,
                      rows, workdir, target)


def _state_evaluation(env: Environment, teacher, student, suite, items,
                      content_sha, *, objective, support, rows: str,
                      batch: int) -> tuple[dict[str, Any], dict[str, float]]:
    """One evaluator, primed and evaluated, each timed; the teacher's forward
    calls DURING `evaluate` are counted (version 3 must make none)."""
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.planning.metrics import StateEvaluator

    ev = StateEvaluator(
        suite, items, device=env.device, position_policy=env.policy,
        execution=ExecutionConfig(
            micro_batch_size=batch,
            calibration_batch_packing=(env.packing if batch > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        distribution_support=support, suite_content_sha256=content_sha,
        numerics=env.numerics, scoring_objective=objective)
    _peak_reset(env)
    _, prime_s = _timed(env, lambda: ev.prime_reference(teacher))
    calls = [0]
    hook = teacher.register_forward_pre_hook(
        lambda *a, **k: calls.__setitem__(0, calls[0] + 1))
    try:
        evaluation, eval_s = _timed(
            env, lambda: ev.evaluate(student, "d2v3_gpu_validation"))
    finally:
        hook.remove()
    detail = evaluation.detail or {}
    run = {"seconds": round(eval_s, 4), "prime_seconds": round(prime_s, 4),
           "peak_memory_bytes": _peak(env),
           "prediction_rows": rows, "micro_batch_size": batch,
           "objective": None if objective is None else objective.qualified_id,
           "distribution_support": support.as_dict(),
           "measurement_protocol_id": ev.measurement_protocol_id,
           "teacher_forwards_during_evaluate": calls[0],
           "union_reference": (detail.get("union_reference") or {}).get(
               "identity_hash")}
    values = {k: v for k, v in (evaluation.values or {}).items()
              if isinstance(v, (int, float))}
    del ev
    _free(env)
    return run, values


def stage_W(*, env: Environment, journal, out: Path, deadline=None,
            flush: Callable[[dict], None] | None = None,
            premises_record: Mapping[str, Any] | None = None
            ) -> dict[str, Any]:
    """The validation's steps, each gated on the deadline covering its bound;
    a failed step is recorded WITH whatever it had measured, never raised, so
    the steps after it still run where they can."""
    import torch

    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.positions import PREDICTION_AXIS
    from aadistill.initialization.scoring.reference_store import (
        build_root_reference_store, root_reference_identity,
    )
    from aadistill.initialization.scoring.support import (
        deterministic_topk, sketch_forward_kl_terms, union_forward_kl_terms,
    )
    from aadistill.initialization.statistics.contribution import (
        bypassed_blocks,
    )
    from stages.d_series import teacher_confidence as TC
    from stages.phase_d1.topk_adoption_driver import (
        _hoard_allocator, objective_agreement, table_agreement,
    )
    from stages.phase_d2 import d2_gpu_validation as V

    objective = env.objective
    if objective is None or not getattr(objective, "uses_union_support",
                                        False):
        raise D2V3ValidationError("stage W validates the union-KL objective; "
                                  "the environment declares none")
    result: dict[str, Any] = {"premises": dict(premises_record or {}),
                              "execution_order": list(STEP_ORDER),
                              "step_bounds_s": dict(env.step_bounds_s)}
    if premises_record is None:
        result["premises"]["teacher_revision"] = env.teacher_revision
    keep: dict[str, Any] = {}
    work = out / "work"

    def step(name: str, fn) -> None:
        bound = float(env.step_bounds_s.get(name, 0.0))
        if deadline is not None and deadline.remaining() < bound:
            result[name] = {"skipped": f"deadline would not cover {bound:.0f} s"}
            journal.event(stage=f"W.{name}", status="skipped")
        else:
            partial: dict[str, Any] = {}
            t0 = time.time()
            _peak_reset(env)
            try:
                result[name] = fn(partial)
            except Exception as exc:                       # noqa: BLE001
                #: FAILURE EVIDENCE: whatever the step measured before it
                #: failed stays in the record beside the traceback.
                result[name] = {**partial,
                                "failed": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()}
            result[name]["wall_seconds"] = round(time.time() - t0, 3)
            result[name]["step_peak_memory_bytes"] = _peak(env)
            journal.event(stage=f"W.{name}", status=(
                "failed" if "failed" in result[name] else "ok"))
        result["verdicts"] = verdicts(result)
        if flush is not None:
            flush(result)

    def in_time(where: str) -> None:
        if deadline is not None:
            deadline.check(f"stage W {where}")

    teacher = env.load_teacher()
    profiles = env.profiles()
    head = env.adapter.linear_output_head(teacher)
    k = int(env.support.top_k)

    def w1(partial):
        partial["profiles"] = {}
        out_: dict[str, Any] = {"profiles": partial["profiles"],
                                "stores_match_declaration": True}
        dtype = str(next(teacher.parameters()).dtype).replace("torch.", "")
        for profile, items in profiles:
            pid = profile.qualified_id
            identity = root_reference_identity(
                root_identity=env.root_identity,
                profile_hash=profile.profile_hash, support=env.support,
                objective=objective)
            builds, stores = {}, {}
            for label, bsz, packing in (
                    ("bsz3_formal", env.micro_batch_size, env.packing),
                    ("bsz1", 1, "original_order_v1")):
                _peak_reset(env)
                stores[label], secs = _timed(
                    env, lambda: build_root_reference_store(
                        teacher, items, support=env.support,
                        identity=identity, micro_batch_size=bsz,
                        packing=packing, device=env.device))
                builds[label] = {"seconds": round(secs, 3),
                                 "peak_memory_bytes": _peak(env),
                                 "identity_hash": stores[label].identity_hash,
                                 "content_sha256":
                                     stores[label].content_digest()}
            store = stores["bsz3_formal"]
            store.require_items(items)
            build = store.identity.get("build") or {}
            declared = {kk: vv for kk, vv in store.identity.items()
                        if kk != "build"}
            expect_build = {
                "micro_batch_size": int(env.micro_batch_size),
                "calibration_batch_packing": env.packing,
                "device_type": torch.device(env.device).type,
                "model_dtype": dtype, "output_head": head.identity()}
            match = (declared == identity and store.retrieves
                     and all(build.get(kk) == vv
                             for kk, vv in expect_build.items())
                     and stores["bsz1"].identity_hash != store.identity_hash)
            out_["stores_match_declaration"] &= bool(match)
            first = _first_batch_items(env, teacher, items)
            partial["profiles"][pid] = {
                "items": len(items), "builds": builds,
                "identity": store.identity, "identity_matches": match,
                "decision": store.decision(),
                "confidence_arithmetic": confidence_arithmetic(store, first),
                "bsz1_vs_bsz3": V.store_drift(stores["bsz1"], store, items),
                "confidence": V.confidence_summary(store, items, env.policy),
                "validity": TC.validity_report(store, items, policy=env.policy,
                                               decode=env.decode)}
            keep.setdefault("stores", {})[pid] = store
            keep.setdefault("items", {})[pid] = (profile, items)
            del stores["bsz1"]
            _free(env)
        out_["decode_available"] = env.decode is not None
        out_["_k_plus_1_confidence_is"] = (
            "the REAL teacher's: what decides whether version 3 widened "
            "version 2's range is this distribution, not a stand-in's")
        return out_

    def w2(partial):
        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        impl = get_implementation("ffn.activation_importance_v0")
        partial["arms"] = arms = {}
        children = {}
        for label, obj in (("full_scope", objective),
                           ("narrow_scope", env.narrow_objective)):
            ctx = _context(env, impl, teacher, profile, items, obj, store,
                           "all_v1", work / f"w2-ffn-{label}", env.target_spec)
            outcome, timing = V._timed_execute(env, impl, ctx)
            arms[label] = {
                **timing,
                "config_declares_objective": "scoring_objective" in ctx.config,
                "weighted": bool((outcome.trace.get("scoring_positions") or {})
                                 .get("token_weights_multiplied_by")),
                "integrity": V.statistics_integrity(
                    impl, label, outcome, ctx, target=env.target_spec,
                    store_identity_hash=store.identity_hash,
                    objective_hash=objective.objective_hash)}
            children[label] = outcome
        overlap = V.jaccard_per_layer(
            children["full_scope"].artifacts["kept_neurons"],
            children["narrow_scope"].artifacts["kept_neurons"])
        if arms["full_scope"]["integrity"]["passed"]:
            keep["ffn_child"] = children["full_scope"].model
            keep["placement"] = V.place_for_execution(keep["ffn_child"],
                                                      env.device)
        else:
            raise D2V3ValidationError("the full-scope FFN child failed its "
                                      "integrity checks; no parent for W3-W6")
        del children
        _free(env)
        return {"profile": pid, "arms": arms, "selection_overlap": overlap,
                "device_placement": keep["placement"],
                "_overlap_class": "SCIENCE DIAGNOSTIC -- no threshold"}

    def w3(partial):
        if "ffn_child" not in keep:
            raise D2V3ValidationError("no FFN child (W2)")
        child = keep["ffn_child"]
        depth = len(env.adapter.blocks(child))
        layers = bypass_layers(depth)
        partial["bypass_layers"] = layers
        rows, retr_err, prot, per_item_red = [], 0.0, [], []
        stats = {"union_size": [], "union_overlap": [], "ratio_union_full": [],
                 "ratio_topk_full": [], "p_tail_union": [],
                 "q_tail_union": []}
        ties = {"rows": 0, "boundary_tie_rows": 0, "set_differs_rows": 0,
                "spill_rows": 0}
        timing = {"oracle_seconds": 0.0, "union_seconds": 0.0,
                  "forward_seconds": 0.0, "head_logits_seconds": 0.0}
        n_items = 0
        for pid, store in keep["stores"].items():
            _, items = keep["items"][pid]
            for item in sample_items(items, NUMERICS_SAMPLE["items_per_profile"]):
                in_time(f"W3 item {n_items}")
                ids = item["input_ids"].to(env.device)
                sk = store.sketch_item(item)
                retrieval = store.retrieval_item(item)
                z_head, s = _timed(env, lambda: head.logits(retrieval.hidden))
                timing["head_logits_seconds"] += s
                with torch.no_grad():
                    native = teacher(ids).logits[0, :-1]
                cands, terms_all, scs = [], [], []
                for layer in layers:
                    with torch.no_grad(), bypassed_blocks(child, {layer}):
                        cand, s = _timed(env, lambda: child(ids).logits[0, :-1])
                    timing["forward_seconds"] += s
                    terms, s = _timed(env, lambda: union_forward_kl_terms(
                        sk.support_indices, sk.support_log_probs,
                        sk.tail_log_prob, cand, retrieval, top_k=k,
                        has_tail=store.has_tail))
                    timing["union_seconds"] += s
                    _, sc = deterministic_topk(cand, k)
                    cands.append(cand)
                    terms_all.append(terms)
                    scs.append(sc)
                #: ONE float64 pass of the teacher per row block, shared by
                #: the native-bf16 comparison and every candidate.
                oracles, s = _timed(env, lambda: oracle_kls(
                    z_head, [(native, None)] + list(zip(cands, scs)),
                    sk.support_indices))
                timing["oracle_seconds"] += s
                prot.append(protocol_difference(z_head, native, oracles[0],
                                                top_k=k))
                del native
                logp_head = None
                for cand, terms, sc, oracle in zip(cands, terms_all, scs,
                                                   oracles[1:]):
                    rows.append({"kl32": terms.kl, "kl_ref32":
                                 terms.kl_reference_topk, "oracle": oracle,
                                 "log_z_ref": retrieval.log_z})
                    full = oracle["full"]
                    keep_r = full > 1e-6
                    stats["ratio_union_full"].append(
                        (oracle["union"][keep_r] / full[keep_r]).cpu())
                    stats["ratio_topk_full"].append(
                        (oracle["topk"][keep_r] / full[keep_r]).cpu())
                    stats["union_size"].append(terms.union_size.cpu())
                    stats["union_overlap"].append(terms.overlap.cpu())
                    stats["p_tail_union"].append(oracle["p_tail_union"].cpu())
                    stats["q_tail_union"].append(oracle["q_tail_union"].cpu())
                    #: retrieval vs the full head on the candidate-only rows
                    st_sorted = sk.support_indices.to(sc.device).sort(-1).values
                    pos = torch.searchsorted(st_sorted, sc).clamp(max=k - 1)
                    extra = st_sorted.gather(-1, pos) != sc
                    if bool(extra.any()):
                        if logp_head is None:
                            logp_head = torch.log_softmax(z_head, -1)
                        got = retrieval.log_probs(sc)
                        want = logp_head.gather(-1, sc.to(z_head.device))
                        retr_err = max(retr_err, float(torch.where(
                            extra.to(got.device), (got - want.to(got.device))
                            .abs(), torch.zeros_like(got)).max()))
                    kth = torch.topk(cand, k).values[:, -1:]
                    ties["rows"] += int(cand.shape[0])
                    ties["boundary_tie_rows"] += int(
                        ((cand >= kth).sum(-1) > k).sum())
                    plain = torch.topk(cand, k).indices.sort(-1).values
                    ties["set_differs_rows"] += int(
                        (plain != sc.sort(-1).values).any(-1).sum())
                    if int(cand.shape[-1]) > 2 * k:
                        top2k = torch.topk(cand, 2 * k).values[:, -1:]
                        ties["spill_rows"] += int((top2k >= kth).sum())
                    scored = env.policy.weights(
                        item, axis=PREDICTION_AXIS).weights > 0
                    per_item_red.append({
                        "kl": terms.kl.cpu(), "scored": scored.cpu(),
                        "confidence": store.per_position(item,
                                                         "confidence").cpu()})
                del cands, terms_all, scs, oracles, z_head, logp_head
                n_items += 1
                partial["items_done"] = n_items
                partial["timing_so_far"] = {kk: round(vv, 3)
                                            for kk, vv in timing.items()}
                _free(env)
        record: dict[str, Any] = {
            "items": n_items, "bypass_layers": layers, "child_depth": depth,
            "union_arithmetic": union_arithmetic(rows),
            "retrieval": retrieval_arithmetic(retr_err),
            "objective_reduction": objective_reduction_arithmetic(
                objective, per_item_red),
            "union_statistics": {kk: _q(torch.cat(vv)) for kk, vv in
                                 stats.items() if vv},
            "bf16_ties": {**ties,
                          "boundary_tie_fraction": ties["boundary_tie_rows"]
                          / max(1, ties["rows"]),
                          "torch_topk_set_differs_fraction":
                              ties["set_differs_rows"] / max(1, ties["rows"]),
                          "spill_fraction": ties["spill_rows"]
                          / max(1, ties["rows"])},
            "float32_head_vs_native_bf16": summarize_protocol_difference(prot),
            "timing": {kk: round(vv, 3) for kk, vv in timing.items()},
            "rows_per_candidate": sum(int(r["kl32"].numel()) for r in rows)
            // max(1, len(layers)),
        }
        del rows
        partial.update(record)
        #: COMPONENT COSTS on one real block, synchronized: the selection,
        #: its tie spill, the retrieval, the complement tail and the whole
        #: union reduction against the version-2 K+1 reduction it extends.
        pid, store = next(iter(keep["stores"].items()))
        _, items = keep["items"][pid]
        item = max(items[:8], key=lambda i: int(i["input_ids"].shape[1]))
        ids = item["input_ids"].to(env.device)
        sk = store.sketch_item(item)
        retrieval = store.retrieval_item(item)
        with torch.no_grad(), bypassed_blocks(child, {layers[0]}):
            cand = child(ids).logits[0, :-1]
        n = int(cand.shape[0])
        _, sc = deterministic_topk(cand, k)
        _peak_reset(env)
        comp = {
            "rows": n,
            "torch_topk_s": _timed(env, lambda: torch.topk(cand, k), 3)[1],
            "deterministic_topk_s": _timed(
                env, lambda: deterministic_topk(cand, k), 3)[1],
            "retrieval_s": _timed(env, lambda: retrieval.log_probs(sc), 3)[1],
            "complement_tail_s": _timed(env, lambda: (
                lambda z: (z.scatter_(-1, sk.support_indices.to(z.device),
                                      float("-inf")),
                           z.scatter_(-1, sc, float("-inf")),
                           torch.logsumexp(z, -1)))(cand.float().clone()),
                3)[1],
            "union_reduction_s": _timed(env, lambda: union_forward_kl_terms(
                sk.support_indices, sk.support_log_probs, sk.tail_log_prob,
                cand, retrieval, top_k=k, has_tail=store.has_tail), 3)[1],
            "k_plus_1_reduction_s": _timed(env, lambda: sketch_forward_kl_terms(
                sk.support_indices.to(cand.device),
                sk.support_log_probs.to(cand.device),
                sk.tail_log_prob.to(cand.device), cand,
                has_tail=store.has_tail), 3)[1],
            "peak_memory_bytes": _peak(env)}
        comp["union_minus_k_plus_1_per_row_s"] = (
            (comp["union_reduction_s"] - comp["k_plus_1_reduction_s"]) / n)
        record["component_timing"] = {kk: (round(vv, 6) if isinstance(vv, float)
                                           else vv) for kk, vv in comp.items()}
        del cand
        #: The CHILD's union KL over every item of the first profile, for the
        #: confidence's weighted contributions by span, domain and token.
        kl = {}
        for it in items:
            in_time("W3 contributions")
            ids = it["input_ids"].to(env.device)
            with torch.no_grad():
                cl = child(ids).logits[0, :-1]
            s2 = store.sketch_item(it)
            kl[str(it["item_id"])] = union_forward_kl_terms(
                s2.support_indices, s2.support_log_probs, s2.tail_log_prob,
                cl, store.retrieval_item(it), top_k=k,
                has_tail=store.has_tail).kl.float().cpu()
            del cl
        record["validity_with_child_union_kl"] = TC.validity_report(
            store, items, policy=env.policy, kl=kl, decode=env.decode)
        _free(env)
        return record

    def depth_round(rows: str):
        if "ffn_child" not in keep:
            raise D2V3ValidationError("no non-root parent: W2's full-scope FFN "
                                      "child does not exist")
        from aadistill.initialization.operators.depth._common import DEPTH_FIELD

        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        parent = keep["ffn_child"]
        devices = V.require_execution_devices(env.device, parent=parent,
                                              store=store)
        entry = next(iter(store.entries.values()))
        if torch.device(env.device).type != entry.hidden.device.type:
            raise D2V3ValidationError(
                f"the store's retrieval state is on {entry.hidden.device}, "
                f"not {env.device}")
        devices["root_retrieval_state"] = str(entry.hidden.device)
        spec = env.adapter.spec_of(parent)
        target = spec.replace(**{DEPTH_FIELD: int(spec[DEPTH_FIELD]) - 1})
        impl = get_implementation("depth.causal_kl_greedy_v1")
        ctx = _context(env, impl, parent, profile, items, objective, store,
                       rows, work / f"w-depth-{rows}", target)
        hoarded = _hoard_allocator() if env.cuda else 0
        hits_before = store.hits
        previous = os.environ.get(SYNC_TELEMETRY_ENV)
        os.environ[SYNC_TELEMETRY_ENV] = "1"
        try:
            outcome, timing = V._timed_execute(env, impl, ctx)
        finally:
            if previous is None:
                os.environ.pop(SYNC_TELEMETRY_ENV, None)
            else:
                os.environ[SYNC_TELEMETRY_ENV] = previous
        art = outcome.artifacts
        rounds = list(art.get("search_rounds") or [])
        table = [(repr(e["candidate"]), float(e["score"]))
                 for e in (rounds[0].get("table") or [])] if rounds else []
        t = dict(art.get("timing") or {})
        cands = max(1, int(t.get("candidate_subsets") or 0))
        record = {
            **timing, "prediction_rows": rows, "profile": pid,
            "parent": "W2's full-scope FFN child (a non-root parent)",
            "execution_devices": devices,
            "hoarded_gib": round(hoarded / 2**30, 3),
            "reference": outcome.local_metrics.reference,
            "reference_cache": dict(art.get("reference_cache") or {}),
            "parent_reference_cache": dict(art.get("parent_reference_cache")
                                           or {}),
            "store_hits_during_round": store.hits - hits_before,
            "reference_counters": dict(art.get("reference_counters") or {}),
            "timing": t,
            "objective_seconds_per_candidate": round(
                float(t.get("objective_seconds") or 0.0) / cands, 5),
            "synchronized_split": bool(t.get("split_is_attributed")),
            "table_consistency": V.depth_table_consistency(art),
            "union_candidates": union_candidate_stats(
                art.get("objective_candidates") or {})}
        keep[f"table_{rows}"] = table
        keep[f"artifacts_{rows}"] = art
        del outcome, ctx
        _free(env)
        return record, art

    def w4(partial):
        record, art = depth_round("all_v1")
        round0 = [str(e["candidate"]) for e in
                  (art["search_rounds"][0].get("table") or [])]
        pid, store = next(iter(keep["stores"].items()))
        profile, items = keep["items"][pid]
        record["rankings"] = rankings_v3(art.get("objective_candidates") or {},
                                         round0)
        record["teacher_confidence"] = V.confidence_summary(store, items,
                                                            env.policy)
        return record

    def w5(partial):
        if "table_all_v1" not in keep:
            raise D2V3ValidationError("the all_v1 DEPTH round (W4) did not run")
        record, _ = depth_round("scored_v1")
        record["agreement"] = table_agreement(
            keep["table_all_v1"], keep["table_scored_v1"],
            relative_tolerance=ROW_EQUIVALENCE_TOLERANCES[
                "depth_candidate_relative"])
        return record

    def w6(partial):
        if "ffn_child" not in keep:
            raise D2V3ValidationError("no student to evaluate: W2's child "
                                      "does not exist")
        suite, items, content = env.load_suite()
        partial["runs"] = runs = {}
        values = {}
        for label, obj, sup, rows_, batch in (
                ("d2v3", objective, env.support, "all_v1",
                 env.micro_batch_size),
                ("d2v3_scored_v1", objective, env.support, "scored_v1",
                 env.micro_batch_size),
                ("d2v3_bsz1", objective, env.support, "all_v1", 1),
                ("historical_reference_topk_kl", None,
                 env.historical_support, "all_v1", env.micro_batch_size)):
            in_time(f"W6 {label}")
            runs[label], values[label] = _state_evaluation(
                env, teacher, keep["ffn_child"], suite, items, content,
                objective=obj, support=sup, rows=rows_, batch=batch)
        d2v3 = {**runs["d2v3"], "values": values["d2v3"]}
        hist = {**runs["historical_reference_topk_kl"],
                "values": values["historical_reference_topk_kl"]}
        union_raw = values["d2v3"].get("state.objective.raw_kl.equal_domain_mean")
        hist_kl = values["historical_reference_topk_kl"].get(
            "state.teacher_kl.equal_domain_mean")
        return {
            "runs": runs,
            "raw_kl_control": raw_kl_control(d2v3, d2v3),
            "historical_reference_topk_kl": {
                "_class": ("DIAGNOSTIC on another support "
                           "(reference_topk_tail_v1): never the raw-KL "
                           "control"),
                "control_refuses_it": not raw_kl_control(d2v3, hist)[
                    "defined"],
                "union_raw_kl_equal_domain_mean": union_raw,
                "reference_topk_kl_equal_domain_mean": hist_kl,
                "union_minus_reference_topk": (
                    None if union_raw is None or hist_kl is None
                    else union_raw - hist_kl)},
            "scored_rows": objective_agreement(
                values["d2v3"], values["d2v3_scored_v1"],
                ROW_EQUIVALENCE_TOLERANCES["state_eval_absolute"]),
            "bsz1_vs_bsz3": {"max_abs_state_objective_difference": V._max_abs(
                values["d2v3"], values["d2v3_bsz1"]),
                "_class": "EXECUTION DRIFT -- reported, never gated"},
            "overhead_vs_historical_seconds_per_evaluation": round(
                runs["d2v3"]["seconds"]
                - runs["historical_reference_topk_kl"]["seconds"], 4),
            "values": values,
            "_student_is": "W2's full-scope FFN child -- never a D2 candidate"}

    for name, fn in (("W1_union_stores_and_confidence", w1),
                     ("W2_ffn_child_weighted", w2),
                     ("W3_union_numerics", w3),
                     ("W4_depth_union_all_v1", w4),
                     ("W5_depth_union_scored_v1", w5),
                     ("W6_state_evaluation", w6)):
        step(name, fn)
    del teacher
    keep.clear()
    _free(env)
    result["verdicts"] = verdicts(result)
    return result


__all__ = ["ARITHMETIC_TOLERANCES", "CORE_STEPS", "D2V3ValidationError",
           "EXECUTION_DIAGNOSTICS", "Environment", "INTEGRITY_TOLERANCES",
           "NUMERICS_SAMPLE", "RECORD_KEY", "ROW_EQUIVALENCE_TOLERANCES",
           "SCIENCE_DIAGNOSTICS", "STEP_ORDER", "adopt", "bypass_layers",
           "confidence_arithmetic", "expected_identities",
           "objective_reduction_arithmetic", "oracle_kls", "premises",
           "production_environment", "protocol_difference", "raw_kl_control",
           "rankings_v3", "retrieval_arithmetic", "sample_items",
           "stage_W", "union_arithmetic",
           "verdicts"]

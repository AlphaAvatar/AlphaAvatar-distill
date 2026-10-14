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

1. **ARITHMETIC** -- the float32 computation against float64. BOTH partition
   KLs the reduction returns (teacher Top-K + tail, and union) against a
   float64 oracle on the float32 HEAD distribution the root store defines
   (never the model's native bf16 logits, which are a different
   distribution), each from one registered error budget; the float64 order
   ``KL_T <= KL_U <= KL_full``; float32 crossings of that order inside the
   combined bound (:func:`kl_arithmetic`, revision 2); the retrieval of
   candidate-only entries against the full head's log-softmax; the K+1
   confidence against float64 on the same sketch; the `c x KL` item mean
   against float64 on the same (KL, c); and, inside ONE version-3 state
   evaluation, ``state.objective.raw_kl.* == state.teacher_kl.*``. GATED by
   :data:`ARITHMETIC_TOLERANCES`.
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

#: REVISION 2 of stage W (maintainer corrective authorization, 2026-10-14):
#: revision 1 (`.../v1/`, run d2v3val_20261014_170824) is FAIL, kept as is,
#: and adopts nothing -- its plan hash is not this plan's. Only the KL
#: arithmetic gate (:func:`kl_arithmetic`) and the validity report's stated
#: definition changed; the objective, support, steps and sample did not.
VALIDATION_DIR = ("logs/stages/stage-1/families/d_series/validations/"
                  "d2v3-gpu-validation/v2")
SUPERSEDED_VALIDATION_DIR = ("logs/stages/stage-1/families/d_series/"
                             "validations/d2v3-gpu-validation/v1")
PLAN = f"{VALIDATION_DIR}/plan.json"
CAMPAIGN = f"{VALIDATION_DIR}/campaign.json"
AUTHORIZATION = f"{VALIDATION_DIR}/authorization.json"
AUTHORIZATION_PROPOSED = f"{VALIDATION_DIR}/authorization.proposed.json"
RECORD_KEY = "W_d2v3_validation"

#: CLASS 1. The KL family, revision 2. BOTH partition KLs the reduction
#: returns -- the teacher Top-K + tail KL (``KL_T``) and the union KL
#: (``KL_U``) -- are checked per position against the float64 oracle,
#: ``|KL32 - KL64| <= abs + rel * KL64``, from ONE error budget
#: (:data:`KL_ERROR_BUDGET`). The float64 oracle's own order
#: ``KL_T <= KL_U <= KL_full`` is exact mathematics, checked at a float64
#: summation slack. A float32 CROSSING of that order is explained exactly when
#: it lies inside the combined bound the triangle inequality gives from the
#: two accuracy gates and the exact order; one beyond it is an anomaly.
#: Revision 1's ``4 eps |log Z_teacher|`` slack -- one log-normalizer's
#: rounding, ~10x tighter than the tolerance registered for the same quantity
#: and with no relative term -- is gone. The retrieval differs from the full
#: head by float32 summation order per entry; the confidence divides an
#: entropy of at most log(K+1) by log(K+1); the c x KL item mean and the
#: raw-KL control are Stage V's, unchanged.
ARITHMETIC_TOLERANCES = {
    "teacher_topk_kl_abs_nats_vs_float64": 1e-4,
    "teacher_topk_kl_rel_vs_float64": 1e-4,
    "union_kl_abs_nats_vs_float64": 1e-4,
    "union_kl_rel_vs_float64": 1e-4,
    "fp64_bound_slack_rel": 1e-12,
    "float32_crossings_beyond_combined_bound": 0,
    "retrieval_abs_nats_vs_full_head": 2e-4,
    "confidence_abs_vs_float64": 1e-5,
    "item_objective_relative_vs_float64": 1e-5,
    "raw_kl_control_relative": 1e-6,
}
#: The float64 oracle's own bound check: exact up to float64 summation.
ORACLE_BOUND_SLACK = ARITHMETIC_TOLERANCES["fp64_bound_slack_rel"]
#: The KL family's error budget, REGISTERED with the tolerances (the plan
#: copies it). Statements of where float32 error comes from and how it
#: reaches a KL -- not fitted constants.
KL_ERROR_BUDGET = {
    "compared": (
        "KL_T and KL_U as union_forward_kl_terms reduces them in float32 from "
        "the root store's teacher inputs (sketch, tail, retrieved entries) and "
        "the candidate's logits, against the float64 oracle on the float32 "
        "head logits recomputed from the store's hidden state and the same "
        "candidate logits"),
    "sources": {
        "log_softmax_rounding": (
            "z - logsumexp(z) in float32: about eps (|z - max| + log sum) per "
            "entry, <= ~2e-5 nats for logits spanning <= 80 nats"),
        "teacher_input_path": (
            "the store forms its sketch per 512-row block and candidate-only "
            "entries through the head's row gather; the oracle recomputes the "
            "head per item. Per entry this is bounded by the retrieval "
            "tolerance. KL_T reads no retrieved entry and no subtraction"),
        "reduction": (
            "float32 sums over <= 2K+1 buckets: <= (2K+1) eps sum|terms| "
            "worst case, ~sqrt(2K+1) eps sum|terms| typical"),
        "union_tail": (
            "p_tail_U = tail_T - sum_E p (log1mexp): its absolute error is "
            "<= d * tail_T for a relative input error d, entering KL_U as at "
            "most d * tail_T * (1 + |log p_tail_U / q_tail_U|)")},
    "propagation": (
        "a teacher log-probability error d per bucket moves the KL by "
        "p (1 + log p/q) d, at most d (3 + KL) in total because "
        "sum p |log p/q| <= KL + 2; a candidate error d moves it by at most d"),
    "tolerance": (
        "abs 1e-4 + rel 1e-4 x KL64 for BOTH partitions. KL_U's value was "
        "registered before revision 1 and held at all 46,056 positions there; "
        "KL_T takes the same value because its float32 sources are a subset "
        "of KL_U's (no retrieved entry, no tail subtraction)"),
    "fp64_bounds": (
        "coarser partition <= finer <= full is exact; float64 tree summation "
        "errs by <= ~log2|V| x 2.2e-16 x (KL + 2), so 1e-12 x (1 + KL_full) is "
        "a ~250x margin"),
    "crossings": (
        "lower KL32_T - KL32_U <= tol_T + tol_U + slack and upper "
        "KL32_U - KL64_full <= tol_U + slack follow from the two accuracy "
        "gates and the exact float64 order; a crossing beyond its bound is "
        "an anomaly, counted rather than assumed away"),
    "not_assumed": (
        "no slack is derived from one log-normalizer alone"),
}
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


#: Per-position records kept in full: every float32 crossing and every
#: position whose error exceeds this fraction of its tolerance.
POSITION_RECORD_FRACTION = 0.5
POSITION_RECORD_CAP = 2000
ERROR_RATIO_BINS = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, math.inf)


def kl_tolerance(kl64, abs_: float, rel: float):
    """``abs + rel * |KL64|``: the per-position float32 allowance."""
    return abs_ + rel * kl64.abs()


def combined_bounds(tol_topk, tol_union, slack):
    """``(lower, upper)``: the largest float32 crossings of the float64 order
    the two accuracy gates allow. ``KL32_T - KL32_U = (KL32_T - KL64_T) +
    (KL64_T - KL64_U) + (KL64_U - KL32_U) <= tol_T + slack + tol_U``, and
    ``KL32_U - KL64_full <= tol_U + slack`` (the full KL has no float32
    side)."""
    return tol_topk + tol_union + slack, tol_union + slack


class HeadLogitsRetrieval:
    """A teacher retrieval over float32 head logits ALREADY computed: ``log p
    = z[ids] - log_z`` from the same tensor a sketch was formed from, so a
    reduction over it reads exactly one set of teacher inputs (the
    same-input attribution of :func:`kl_arithmetic`)."""

    def __init__(self, logits, log_z) -> None:
        self.logits, self.log_z = logits, log_z

    @property
    def rows(self) -> int:
        return int(self.logits.shape[0])

    def log_probs(self, token_ids, start: int = 0, stop: int | None = None):
        stop = self.rows if stop is None else int(stop)
        z = self.logits[start:stop]
        return (z.gather(-1, token_ids.to(z.device).long()).float()
                - self.log_z[start:stop].to(z.device).float()[:, None])


def _ratio_histogram(a, tol) -> dict[str, int]:
    import torch

    r = (a / tol).double()
    r = r[torch.isfinite(r)]
    out = {}
    for lo, hi in zip(ERROR_RATIO_BINS[:-1], ERROR_RATIO_BINS[1:]):
        out[f"[{lo},{hi})"] = int(((r >= lo) & (r < hi)).sum())
    return out


def kl_arithmetic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The KL family's arithmetic gate (revision 2), per position.

    ``rows``: one entry per (item, candidate) with ``item_id``, ``layer``,
    ``kl32_topk`` / ``kl32_union`` (the reduction's float32 outputs on the
    store's inputs), ``oracle`` (:func:`oracle_kls`: ``topk``, ``union``,
    ``full``), ``log_z_ref`` (the teacher's log-normalizer, only for the
    revision-1 comparison count) and optionally ``same_input`` (the same
    reduction on the head logits the oracle reads: ``kl32_topk``,
    ``kl32_union``, ``support_agrees``). Gated: each partition's accuracy,
    the float64 order, and float32 crossings within the combined bound.
    Everything else is recorded for attribution."""
    import torch

    t = ARITHMETIC_TOLERANCES
    tols = {"topk": (t["teacher_topk_kl_abs_nats_vs_float64"],
                     t["teacher_topk_kl_rel_vs_float64"]),
            "union": (t["union_kl_abs_nats_vs_float64"],
                      t["union_kl_rel_vs_float64"])}
    eps32 = float(torch.finfo(torch.float32).eps)

    def cat(get) -> Any:
        parts = [get(r).detach().double().reshape(-1).cpu() for r in rows]
        return torch.cat(parts) if parts else torch.zeros(0,
                                                          dtype=torch.float64)

    k32 = {"topk": cat(lambda r: r["kl32_topk"]),
           "union": cat(lambda r: r["kl32_union"])}
    k64 = {"topk": cat(lambda r: r["oracle"]["topk"]),
           "union": cat(lambda r: r["oracle"]["union"])}
    full = cat(lambda r: r["oracle"]["full"])
    n = int(full.numel())
    where = [(str(r.get("item_id")), r.get("layer"), j)
             for r in rows for j in range(int(r["oracle"]["full"].numel()))]

    part: dict[str, Any] = {}
    allow: dict[str, Any] = {}
    errs: dict[str, Any] = {}
    finite_all = torch.isfinite(full)
    for name in ("topk", "union"):
        a32, a64 = k32[name], k64[name]
        both = torch.isfinite(a32) & torch.isfinite(a64)
        mismatch = torch.isfinite(a32) != torch.isfinite(a64)
        finite_all &= both
        e = torch.where(both, a32 - a64, torch.zeros_like(a64))
        tol = kl_tolerance(torch.where(both, a64, torch.zeros_like(a64)),
                           *tols[name])
        bad = both & (e.abs() > tol)
        allow[name], errs[name] = tol, e
        excess = (e.abs() - tol)[both]
        part[name] = {
            "positions": int(both.sum()),
            "position_violations": int(bad.sum()),
            "finiteness_mismatches": int(mismatch.sum()),
            "max_abs_difference": (float(e.abs()[both].max())
                                   if bool(both.any()) else None),
            "max_excess_over_tolerance": (float(excess.max())
                                          if int(excess.numel()) else None),
            "abs_difference": _q(e.abs()[both]),
            "signed_difference": _q(e[both]),
            "error_over_tolerance_histogram": _ratio_histogram(
                e.abs()[both], tol[both]),
            "within_tolerance": (n > 0 and int(bad.sum()) == 0
                                 and int(mismatch.sum()) == 0)}

    #: The float64 oracle's exact order, at its summation slack (read at
    #: call time: the registered value is the module's).
    slack = float(ORACLE_BOUND_SLACK) * (1.0 + full.abs())
    fin = finite_all
    lo64 = torch.where(fin, k64["topk"] - k64["union"], torch.zeros_like(full))
    up64 = torch.where(fin, k64["union"] - full, torch.zeros_like(full))
    fp64 = {"lower_violations": int((fin & (lo64 > slack)).sum()),
            "upper_violations": int((fin & (up64 > slack)).sum()),
            "max_lower_excess": float((lo64 - slack)[fin].max())
            if bool(fin.any()) else None,
            "max_upper_excess": float((up64 - slack)[fin].max())
            if bool(fin.any()) else None}
    fp64["hold"] = (n > 0 and fp64["lower_violations"] == 0
                    and fp64["upper_violations"] == 0)

    #: Float32 crossings of that order, against the COMBINED bound.
    lo32 = torch.where(fin, k32["topk"] - k32["union"], torch.zeros_like(full))
    up32 = torch.where(fin, k32["union"] - full, torch.zeros_like(full))
    bound_lo, bound_up = combined_bounds(allow["topk"], allow["union"], slack)

    def side(c, bound) -> dict[str, Any]:
        crossed = fin & (c > 0)
        beyond = fin & (c > bound)
        out = {"count": int(crossed.sum()), "beyond_combined_bound":
               int(beyond.sum()),
               "max_magnitude": float(c[crossed].max())
               if bool(crossed.any()) else 0.0}
        if bool(crossed.any()):
            j = int(torch.where(crossed, c, torch.full_like(c, -1.0)).argmax())
            item, layer, pos = where[j]
            out["at_max"] = {
                "item_id": item, "layer": layer, "position": pos,
                "kl64_topk": float(k64["topk"][j]),
                "kl64_union": float(k64["union"][j]),
                "kl64_full": float(full[j]),
                "kl32_topk": float(k32["topk"][j]),
                "kl32_union": float(k32["union"][j]),
                "combined_bound": float(bound[j])}
        return out

    crossings = {"lower": side(lo32, bound_lo), "upper": side(up32, bound_up),
                 "lower_is": "KL32_T - KL32_U > 0",
                 "upper_is": "KL32_U - KL64_full > 0",
                 "combined_bound": KL_ERROR_BUDGET["crossings"]}
    beyond = (crossings["lower"]["beyond_combined_bound"]
              + crossings["upper"]["beyond_combined_bound"])
    mism = sum(part[x]["finiteness_mismatches"] for x in part)
    anomalies = {"fp64_bound_violations": fp64["lower_violations"]
                 + fp64["upper_violations"],
                 "float32_crossings_beyond_combined_bound": beyond,
                 "finiteness_mismatches": mism}
    anomalies["total"] = sum(anomalies.values())

    #: Per-position records: every crossing and every position above
    #: POSITION_RECORD_FRACTION of a tolerance, worst first.
    ratio = torch.maximum(errs["topk"].abs() / allow["topk"],
                          errs["union"].abs() / allow["union"])
    keep = fin & ((lo32 > 0) | (up32 > 0)
                  | (ratio > POSITION_RECORD_FRACTION))
    idx = torch.nonzero(keep).flatten()
    idx = idx[torch.argsort(ratio[idx], descending=True)]
    recorded = []
    for j in idx[:POSITION_RECORD_CAP].tolist():
        item, layer, pos = where[j]
        recorded.append({
            "item_id": item, "layer": layer, "position": pos,
            "kl64_topk": float(k64["topk"][j]),
            "kl64_union": float(k64["union"][j]), "kl64_full": float(full[j]),
            "kl32_topk": float(k32["topk"][j]),
            "kl32_union": float(k32["union"][j]),
            "err_topk": float(errs["topk"][j]),
            "err_union": float(errs["union"][j]),
            "tol_topk": float(allow["topk"][j]),
            "tol_union": float(allow["union"][j]),
            "lower_crossing": float(lo32[j]), "upper_crossing": float(up32[j])})

    #: ATTRIBUTION (not gated): the same reduction on the head logits the
    #: oracle reads separates float32 reduction error from the store's
    #: teacher-input path.
    same = [r for r in rows if r.get("same_input")]
    attribution: dict[str, Any] = {"rows": len(same)}
    if same:
        agree = torch.cat([r["same_input"]["support_agrees"].detach()
                           .reshape(-1).bool().cpu() for r in same])
        s_or = {x: torch.cat([r["oracle"][x].detach().double().reshape(-1)
                              .cpu() for r in same]) for x in ("topk", "union")}
        s32 = {x: torch.cat([r["same_input"][f"kl32_{x}"].detach().double()
                             .reshape(-1).cpu() for r in same])
               for x in ("topk", "union")}
        st32 = {x: torch.cat([r[f"kl32_{x}"].detach().double().reshape(-1)
                              .cpu() for r in same]) for x in ("topk", "union")}
        for x in ("topk", "union"):
            ok = agree & torch.isfinite(s32[x]) & torch.isfinite(s_or[x])
            okp = torch.isfinite(s32[x]) & torch.isfinite(st32[x])
            attribution[x] = {
                "reduction_only_abs_vs_float64": _q(
                    (s32[x] - s_or[x]).abs()[ok]),
                "store_path_minus_same_input_abs": _q(
                    (st32[x] - s32[x]).abs()[okp])}
        attribution["support_agreement_fraction"] = float(
            agree.double().mean()) if int(agree.numel()) else None
    attribution["_is"] = (
        "DIAGNOSTIC: reduction_only is float32 rounding alone (one set of "
        "teacher inputs); store_path_minus_same_input is what the store's "
        "teacher-input path adds. Never gated")

    #: Revision 1's gate on the same numbers, for continuity only.
    log_z = cat(lambda r: r["log_z_ref"])
    slack_r1 = 4 * eps32 * log_z.abs()
    r1 = int((fin & ((lo32 > slack_r1) | (up32 > slack_r1))).sum())

    return {
        "positions": n,
        "tolerances_applied": {k: v for k, v in t.items() if k.startswith((
            "teacher_topk_kl", "union_kl", "fp64_bound",
            "float32_crossings"))},
        "teacher_topk": part["topk"], "union": part["union"],
        "fp64_bounds": fp64, "float32_crossings": crossings,
        "anomalies": anomalies,
        "positions_recorded": recorded,
        "positions_recorded_total": int(idx.numel()),
        "positions_recorded_omitted": max(0, int(idx.numel())
                                          - POSITION_RECORD_CAP),
        "positions_recorded_rule": (
            f"every float32 crossing and every position above "
            f"{POSITION_RECORD_FRACTION} of a tolerance, worst first, at most "
            f"{POSITION_RECORD_CAP}; the histograms cover every position"),
        "same_input_attribution": attribution,
        "revision_1_slack_count": {
            "count": r1, "_is": ("how many positions revision 1's 4 eps "
                                 "|log Z_teacher| slack would flag on these "
                                 "numbers -- continuity only, never gated")},
        "within_tolerance": (part["topk"]["within_tolerance"]
                             and part["union"]["within_tolerance"]),
        "bounds_hold": fp64["hold"],
        "crossings_within_combined_bound": (
            n > 0 and beyond <= t["float32_crossings_beyond_combined_bound"]),
    }


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
    kl = lambda s: s.get("kl_arithmetic") or {}                  # noqa: E731
    arith["teacher_topk_kl_vs_float64"] = _tri(
        w3, lambda s: (kl(s).get("teacher_topk") or {}).get(
            "within_tolerance"))
    arith["union_kl_vs_float64"] = _tri(
        w3, lambda s: (kl(s).get("union") or {}).get("within_tolerance"))
    arith["kl_fp64_bounds"] = _tri(w3, lambda s: kl(s).get("bounds_hold"))
    arith["float32_crossings_within_combined_bound"] = _tri(
        w3, lambda s: kl(s).get("crossings_within_combined_bound"))
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
        deterministic_topk, sketch_forward_kl_terms,
        sketch_reference_deterministic, union_forward_kl_terms,
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
                #: The SAME-INPUT twin of the store's teacher side: a sketch
                #: and a retrieval formed from z_head itself, the tensor the
                #: oracle reads (attribution only).
                targets = item["input_ids"].reshape(-1)[1:].to(z_head.device)
                sk_same, log_z_same = sketch_reference_deterministic(
                    z_head, targets, top_k=k)
                retr_same = HeadLogitsRetrieval(z_head, log_z_same)
                same_support = (sk_same.support_indices.sort(-1).values
                                == sk.support_indices.to(z_head.device)
                                .sort(-1).values).all(-1)
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
                    terms_same = union_forward_kl_terms(
                        sk_same.support_indices, sk_same.support_log_probs,
                        sk_same.tail_log_prob, cand, retr_same, top_k=k,
                        has_tail=store.has_tail)
                    _, sc = deterministic_topk(cand, k)
                    cands.append(cand)
                    terms_all.append((terms, terms_same))
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
                for layer, cand, (terms, terms_same), sc, oracle in zip(
                        layers, cands, terms_all, scs, oracles[1:]):
                    rows.append({
                        "item_id": str(item["item_id"]), "layer": layer,
                        "kl32_topk": terms.kl_reference_topk,
                        "kl32_union": terms.kl, "oracle": oracle,
                        "log_z_ref": retrieval.log_z,
                        "same_input": {
                            "kl32_topk": terms_same.kl_reference_topk,
                            "kl32_union": terms_same.kl,
                            "support_agrees": same_support}})
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
                del sk_same, log_z_same, retr_same
                n_items += 1
                partial["items_done"] = n_items
                partial["timing_so_far"] = {kk: round(vv, 3)
                                            for kk, vv in timing.items()}
                _free(env)
        record: dict[str, Any] = {
            "items": n_items, "bypass_layers": layers, "child_depth": depth,
            "kl_arithmetic": kl_arithmetic(rows),
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
            "rows_per_candidate": sum(int(r["kl32_union"].numel())
                                      for r in rows)
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
           "stage_W", "kl_arithmetic", "kl_tolerance", "combined_bounds",
           "HeadLogitsRetrieval",
           "KL_ERROR_BUDGET", "verdicts"]

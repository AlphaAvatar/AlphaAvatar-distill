"""D2's teacher-confidence validity checks: what ``c_T`` actually looks like.

Run BEFORE the formal D2 search, on the ROOT reference store of each
calibration profile (`aadistill.initialization.scoring.reference_store`), and
-- when a candidate's per-position root KL is supplied (B's, in the D2 GPU
validation) -- against that KL. Everything here is DESCRIPTIVE: the
preregistration v2 freezes the objective, the scope and every rule before
this runs, and no result below may change a definition, a threshold-free
mapping or the scope. Nothing in this module decides anything.

What it reports, over the scored positions of the frozen position policy:

* the distribution of ``c_T`` and ``H_T``: quantiles, histogram, weight sum,
  mean weight and the effective sample size ``(sum w)^2 / sum w^2``;
* by span -- reasoning vs final answer vs the other assistant positions, and
  the structural tags (think close, EOS, tool close) -- by domain and by
  sub-type;
* by TOKEN TYPE of the predicted token, when a decoder is supplied (digits,
  punctuation, whitespace, words, structural);
* the relation of ``c_T`` to the teacher's own probability of the realised
  next token and to whether that token is the teacher's top-1;
* with a KL: the rank correlation of ``c_T`` and KL, and each group's share
  of the weighted objective against its share of the unweighted KL -- where
  the weighting actually moves the fidelity budget.

Everything is derived from tensors the store already holds; no model runs
here, so it is CPU-testable on a toy store.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

#: Prediction-position tags of the frozen calibration mixtures, as SPANS.
#: `reasoning` and `final_answer` partition most of an assistant turn; the
#: structural tags mark single positions inside or after it.
SPAN_TAGS = ("reasoning", "final_answer")
STRUCTURAL_TAGS = ("think_close", "eos", "tool_close")
QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
TOKEN_CLASSES = ("digit", "punctuation", "whitespace", "word", "other")


def _stats(c: torch.Tensor, total_weight: float | None = None
           ) -> dict[str, Any]:
    """Distribution of one group's weights. ``total_weight`` gives its share."""
    c = c.double()
    n = int(c.numel())
    if n == 0:
        return {"positions": 0}
    w2 = float((c * c).sum())
    out = {"positions": n, "mean": float(c.mean()),
           "sd": float(c.std()) if n > 1 else 0.0,
           "min": float(c.min()), "max": float(c.max()),
           "quantiles": {f"q{int(round(q * 100)):02d}": float(v) for q, v in zip(
               QUANTILES, torch.quantile(c, torch.tensor(
                   QUANTILES, dtype=torch.float64)).tolist())},
           "weight_sum": float(c.sum()),
           "effective_sample_size": (float(c.sum()) ** 2 / w2) if w2 > 0 else 0.0}
    out["effective_fraction"] = out["effective_sample_size"] / n
    if total_weight:
        out["weight_share"] = out["weight_sum"] / total_weight
    return out


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties share the mean rank), float64."""
    x = x.double()
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(1, x.numel() + 1, dtype=torch.float64)
    values, inverse, counts = torch.unique(x, return_inverse=True,
                                           return_counts=True)
    sums = torch.zeros(values.numel(), dtype=torch.float64).index_add_(
        0, inverse, ranks)
    return (sums / counts.double())[inverse]


def spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """Spearman's rho with average-rank ties; ``None`` below three points or
    when either side is constant."""
    if int(x.numel()) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum()) / den if den > 0 else None


def token_class(text: str) -> str:
    """A coarse, threshold-free class of one decoded token."""
    if text == "" or text.isspace():
        return "whitespace"
    body = text.strip()
    if body.isdigit():
        return "digit"
    if re.fullmatch(r"[\W_]+", body):
        return "punctuation"
    if re.fullmatch(r"[^\W\d_]+", body):
        return "word"
    return "other"


def _item_tags(item: Mapping[str, Any], n: int) -> dict[str, torch.Tensor]:
    from aadistill.initialization.scoring.positions import (
        normalized_prediction_tags,
    )
    return normalized_prediction_tags(item.get("tags") or {}, n,
                                      where=f"item {item.get('item_id')!r}: ")


def validity_report(store: Any, items: Sequence[Mapping[str, Any]], *,
                    policy: Any,
                    kl: Mapping[str, torch.Tensor] | None = None,
                    decode: Callable[[int], str] | None = None,
                    ) -> dict[str, Any]:
    """The descriptive checks over ``items``' scored positions.

    ``policy`` is the frozen position policy (its PREDICTION weights decide
    which positions are scored -- binary under the supervised-target policy).
    ``kl`` maps item id to a per-prediction-position KL against the root,
    aligned with the store. ``decode`` maps a token id to its text.
    """
    from aadistill.initialization.scoring.positions import PREDICTION_AXIS

    store.require_items(items)
    conf, ent, tlp, top1 = [], [], [], []
    group: dict[str, list[torch.Tensor]] = {}
    group_kl: dict[str, list[torch.Tensor]] = {}
    klv = []

    def add(name: str, mask: torch.Tensor, c: torch.Tensor,
            k: torch.Tensor | None) -> None:
        group.setdefault(name, []).append(c[mask])
        if k is not None:
            group_kl.setdefault(name, []).append(k[mask])

    for item in items:
        key = str(item["item_id"])
        c = store.per_position(item, "confidence").double().cpu()
        n = int(c.shape[0])
        w = policy.weights(item, axis=PREDICTION_AXIS).weights.cpu()
        if int(w.numel()) != n:
            raise ValueError(f"item {key!r}: the policy weighs {int(w.numel())} "
                             f"prediction positions and the store holds {n}")
        scored = w > 0
        k = None
        if kl is not None:
            k = torch.as_tensor(kl[key]).double().cpu().reshape(-1)
            if int(k.numel()) != n:
                raise ValueError(f"item {key!r}: KL over {int(k.numel())} "
                                 f"positions, the store holds {n}")
        ids = item["input_ids"].reshape(-1)[1:].cpu()
        tags = _item_tags(item, n)
        e = store.per_position(item, "entropy").double().cpu()
        lp = store.per_position(item, "target_log_prob").double().cpu()
        #: Token ids as float are exact far beyond any vocabulary (2^24).
        hit = store.per_position(item, "top1_token").cpu() == ids.double()
        conf.append(c[scored]); ent.append(e[scored])
        tlp.append(lp[scored]); top1.append(hit[scored])
        if k is not None:
            klv.append(k[scored])
        span_any = torch.zeros(n, dtype=torch.bool)
        for t in SPAN_TAGS:
            if t in tags:
                add(f"span:{t}", tags[t] & scored, c, k)
                span_any |= tags[t]
        add("span:other_scored", scored & ~span_any, c, k)
        for t in STRUCTURAL_TAGS:
            if t in tags:
                add(f"structural:{t}", tags[t] & scored, c, k)
        add(f"domain:{item.get('domain')}", scored, c, k)
        add(f"subtype:{item.get('subtype')}", scored, c, k)
        if decode is not None:
            structural = torch.zeros(n, dtype=torch.bool)
            for t in STRUCTURAL_TAGS:
                if t in tags:
                    structural |= tags[t]
            classes = [("structural" if bool(structural[j])
                        else token_class(decode(int(ids[j]))))
                       for j in range(n)]
            for cls in (*TOKEN_CLASSES, "structural"):
                m = torch.tensor([x == cls for x in classes], dtype=torch.bool)
                add(f"token:{cls}", m & scored, c, k)

    c = torch.cat(conf)
    total = float(c.sum())
    report: dict[str, Any] = {
        "schema": "aadistill.d_series.teacher_confidence_validity/v1",
        "store_identity_hash": store.identity_hash,
        "policy": policy.qualified_id,
        "items": len(items),
        "definition": "c_T = 1 - H_T / log|V|, H_T full-vocabulary Shannon",
        "vocab_size": int(store.vocab_size),
        "log_vocab_nats": math.log(int(store.vocab_size)),
        "distribution": {
            "confidence": _stats(c),
            "entropy_nats": {k: v for k, v in _stats(torch.cat(ent)).items()
                             if k in ("positions", "mean", "sd", "min", "max",
                                      "quantiles")},
            "histogram_20_bins_0_1": [int(x) for x in torch.histc(
                c, bins=20, min=0.0, max=1.0).tolist()],
        },
        "groups": {name: _stats(torch.cat(v), total)
                   for name, v in sorted(group.items())},
    }
    p = torch.cat(tlp)
    hits = torch.cat(top1)
    report["relation_to_teacher_probability"] = {
        "spearman_confidence_vs_target_log_prob": spearman(c, p),
        "mean_confidence_where_target_is_top1": (
            float(c[hits].mean()) if bool(hits.any()) else None),
        "mean_confidence_where_target_is_not_top1": (
            float(c[~hits].mean()) if bool((~hits).any()) else None),
        "target_is_top1_fraction": float(hits.double().mean()),
        "mean_confidence_by_target_probability_decile": _by_decile(c, p.exp()),
    }
    if klv:
        k = torch.cat(klv)
        finite = torch.isfinite(k)
        kf, cf = k[finite], c[finite]
        wkl = float((cf * kf).sum())
        ukl = float(kf.sum())
        report["relation_to_kl"] = {
            "positions": int(kf.numel()),
            "non_finite_kl_positions": int((~finite).sum()),
            "spearman_confidence_vs_kl": spearman(cf, kf),
            "weighted_over_unweighted_kl": (wkl / ukl) if ukl > 0 else None,
            "group_shares": {
                name: _shares(torch.cat(group[name]),
                              torch.cat(group_kl[name]), wkl, ukl)
                for name in sorted(group_kl)},
        }
    report["risks"] = risks(report)
    report["decides"] = "nothing: descriptive, after the definitions froze"
    return report


def _by_decile(c: torch.Tensor, prob: torch.Tensor) -> list[dict[str, Any]]:
    out = []
    edges = torch.linspace(0, 1, 11, dtype=torch.float64)
    for i in range(10):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (prob >= lo) & ((prob < hi) if i < 9 else (prob <= hi))
        out.append({"target_probability": [lo, hi], "positions": int(m.sum()),
                    "mean_confidence": (float(c[m].mean()) if bool(m.any())
                                        else None)})
    return out


def _shares(c: torch.Tensor, k: torch.Tensor, wkl: float, ukl: float
            ) -> dict[str, Any]:
    finite = torch.isfinite(k)
    c, k = c[finite], k[finite]
    w = float((c * k).sum())
    u = float(k.sum())
    return {"positions": int(k.numel()),
            "share_of_weighted_objective": (w / wkl) if wkl > 0 else None,
            "share_of_unweighted_kl": (u / ukl) if ukl > 0 else None,
            "mean_kl": float(k.mean()) if int(k.numel()) else None}


def risks(report: Mapping[str, Any]) -> list[str]:
    """The registered risks, each stated with the numbers that bear on it.

    Statements, not verdicts: no threshold here gates anything.
    """
    out = []
    d = report["distribution"]["confidence"]
    q = d.get("quantiles") or {}
    if q:
        out.append(
            f"dynamic range: c_T IQR {q['q75'] - q['q25']:.4f} "
            f"(q25 {q['q25']:.4f}, q75 {q['q75']:.4f}), min {d['min']:.4f}; "
            f"normalised by log|V| = {report['log_vocab_nats']:.3f} nats, a "
            "narrow range means the weighting moves the objective and the "
            "statistics only modestly")
        out.append(
            f"effective sample size: {d['effective_sample_size']:.1f} of "
            f"{d['positions']} scored positions "
            f"({d['effective_fraction']:.4f})")
    g = report["groups"]
    if "span:reasoning" in g and "span:final_answer" in g and \
            g["span:reasoning"].get("positions") and \
            g["span:final_answer"].get("positions"):
        out.append(
            f"thinking vs answer: mean c_T {g['span:reasoning']['mean']:.4f} "
            f"over {g['span:reasoning']['positions']} reasoning positions, "
            f"{g['span:final_answer']['mean']:.4f} over "
            f"{g['span:final_answer']['positions']} answer positions; the "
            "lower span loses fidelity weight")
    rk = report.get("relation_to_kl")
    if rk and rk.get("weighted_over_unweighted_kl") is not None:
        out.append(
            f"KL relation: Spearman(c_T, KL) = "
            f"{rk['spearman_confidence_vs_kl']}; the weighted objective is "
            f"{rk['weighted_over_unweighted_kl']:.4f} of the unweighted KL "
            "mass -- a strongly negative rank correlation means the weighting "
            "discounts exactly the high-KL positions")
    return out


__all__ = ["SPAN_TAGS", "STRUCTURAL_TAGS", "TOKEN_CLASSES", "risks",
           "spearman", "token_class", "validity_report"]

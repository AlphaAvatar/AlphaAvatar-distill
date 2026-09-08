"""The evaluator that measures a state, and turns distortion into metrics.

The measurement CONTRACTS -- what a metric name means, what a suite is, what a
result looks like -- live in `aadistill.initialization.specs.metrics`, one layer
down, because the operators and the state spec need them and must not depend on
this module to get them.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from aadistill.initialization.specs.metrics import (
    DEFAULT_REFERENCE_CACHE_BUDGET_BYTES,
    MeasurementError,
    ReferenceStrategy,
    StateEvalSuite,
    StateEvaluation,
    SuiteItem,
    reference_cache_bytes,
)
from aadistill.initialization.statistics.contribution import (
    DistortionSums,
    distortion,
    domain_balanced_score,
)


class StateEvaluator:
    """Scores a candidate checkpoint against the **original teacher**.

    Every checkpoint is measured on its own weights and the result is stamped
    with that artifact's digest. The teacher is the global reference for every
    state, at every depth, which is what makes states from different paths
    comparable at all.
    """

    def __init__(
        self,
        suite: StateEvalSuite,
        items: Sequence[SuiteItem],
        *,
        device: str = "cpu",
        chunk: int = 512,
        reference_strategy: ReferenceStrategy = ReferenceStrategy.RECOMPUTE,
        cache_budget_bytes: int = DEFAULT_REFERENCE_CACHE_BUDGET_BYTES,
        vocab_size: int | None = None,
    ) -> None:
        if not items:
            raise MeasurementError(f"{suite.qualified_id}: no items to score")
        declared = {(d, s) for d, subs in suite.subtypes.items() for s in subs}
        seen = {(i.domain, i.subtype) for i in items}
        unknown = sorted(seen - declared)
        if unknown:
            raise MeasurementError(
                f"{suite.qualified_id}: items carry undeclared (domain, subtype) {unknown}")
        missing = sorted(declared - seen)
        if missing:
            raise MeasurementError(
                f"{suite.qualified_id}: declared sub-types with no items {missing}; "
                "a silently absent sub-type reweights its domain")
        self.suite = suite
        self.items = list(items)
        self.device = device
        self.chunk = chunk
        self.reference_strategy = reference_strategy
        self._teacher = None
        self._ref_logits: dict[str, torch.Tensor] = {}
        self._ref_ready = False

        if reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            if vocab_size is None:
                raise MeasurementError(
                    "CACHE_IN_MEMORY needs vocab_size to check its budget before "
                    "allocating; a budget checked after the fact is an OOM")
            needed = reference_cache_bytes(items, vocab_size)
            if needed > cache_budget_bytes:
                raise MeasurementError(
                    f"caching the reference logits for this suite would take "
                    f"{needed / 2**30:.1f} GiB, over the {cache_budget_bytes / 2**30:.1f} "
                    "GiB budget. Use ReferenceStrategy.RECOMPUTE: one teacher forward "
                    "per candidate is seconds, and it does not scale with vocabulary.")
            self._cache_bytes = needed

    @torch.no_grad()
    def prime_reference(self, teacher) -> None:
        """Bind the original teacher, and cache its logits only if asked to."""
        self._teacher = teacher
        if self.reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            for item in self.items:
                ids = item.input_ids.to(self.device)
                self._ref_logits[item.item_id] = teacher(ids).logits[0, :-1].float().cpu()
        self._ref_ready = True

    @torch.no_grad()
    def _reference_for(self, item: SuiteItem) -> torch.Tensor:
        if self.reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            return self._ref_logits[item.item_id]
        ids = item.input_ids.to(self.device)
        return self._teacher(ids).logits[0, :-1].float().cpu()

    @torch.no_grad()
    def evaluate(self, model, artifact_digest: str, *, reference: str = "root_teacher",
                 runtime: Mapping[str, Any] | None = None) -> StateEvaluation:
        if not self._ref_ready or self._teacher is None:
            raise MeasurementError(
                "the reference teacher was never bound; a candidate cannot be scored "
                "against a teacher that was not run")
        if not artifact_digest:
            raise MeasurementError("refusing to measure without the artifact digest")

        per_subtype: dict[str, DistortionSums] = {}
        totals = DistortionSums()
        for item in self.items:
            ids = item.input_ids.to(self.device)
            # One item's reference and candidate logits exist at a time. At the
            # intended suite that is ~0.5 GiB each rather than 33.8 GiB held for
            # the whole run.
            ref = self._reference_for(item)
            cand = model(ids).logits[0, :-1].float().cpu()
            if cand.shape != ref.shape:
                raise MeasurementError(
                    f"item {item.item_id}: candidate logits {tuple(cand.shape)} do not "
                    f"match the reference {tuple(ref.shape)}; the two models are not "
                    "logit-comparable, so no KL against the original teacher exists")
            targets = item.input_ids[0, 1:].cpu()
            sums = distortion(ref, cand, targets, tags=item.tags, chunk=self.chunk)
            per_subtype.setdefault(item.subtype, DistortionSums()).merge(sums)
            totals.merge(sums)
            del ref, cand

        subtype_kl = {k: v.as_dict()["kl"] for k, v in per_subtype.items()}
        domain_map = {d: list(self.suite.subtypes[d]) for d in self.suite.domains}
        primary, per_domain = domain_balanced_score(subtype_kl, domain_map)

        agg = totals.as_dict()
        values: dict[str, float] = {
            "state.teacher_kl.equal_domain_mean": float(primary),
            "state.teacher_kl.worst_domain": float(max(per_domain.values())),
            "state.teacher_kl.token_mean": float(agg["kl"]),
            "state.reverse_kl.token_mean": float(agg["reverse_kl"]),
            # Pooled over every declared domain — reasoning, code and tool text
            # included. Named for what it is; it is NOT "general NLL", and the
            # earlier key that claimed to be was this quantity.
            "state.nll.pooled_all_domains": float(agg["abl_ce"]),
            "state.nll.teacher_reference_pooled": float(agg["ref_ce"]),
            "state.nll_delta_vs_teacher_pooled": float(agg["ce_delta"]),
            "state.top1_agreement": float(agg["top1_agreement"]),
        }
        for domain, score in per_domain.items():
            values[f"state.teacher_kl.{domain}"] = float(score)

        # Per-domain candidate NLL, and `state.nll.general` from the general
        # domain alone. Omitted entirely when the suite declares no general
        # domain, rather than falling back to the pooled number.
        per_domain_ce = _per_domain_ce(per_subtype, domain_map)
        for domain, ce in per_domain_ce.items():
            values[f"state.nll.{domain}"] = float(ce)
        general = self.suite.general_domain
        if general and general in per_domain_ce:
            values["state.nll.general"] = float(per_domain_ce[general])

        tagged = agg["tagged"]
        for tag, entry in tagged.items():
            if entry["kl"] is not None:
                values[f"state.critical_token_kl.{tag}"] = float(entry["kl"])
        present = [tagged[t]["kl"] for t in self.suite.critical_tags
                   if t in tagged and tagged[t]["kl"] is not None]
        if present:
            # Unweighted mean over declared critical-token classes: a token-count
            # mean would be dominated by whichever class is common, and the rare
            # ones (think_close, eos) are the ones that decide termination.
            values["state.critical_token_kl"] = float(sum(present) / len(present))

        return StateEvaluation(
            artifact_digest=artifact_digest,
            suite_id=self.suite.qualified_id,
            suite_hash=self.suite.suite_hash,
            reference=reference,
            values=values,
            positions=int(agg["positions"]),
            detail={"per_subtype_kl": subtype_kl, "per_domain_kl": per_domain,
                    "per_domain_nll": per_domain_ce, "tagged": tagged,
                    "reference_strategy": self.reference_strategy.value},
            runtime=dict(runtime or {}),
        )


def _per_domain_ce(per_subtype: Mapping[str, DistortionSums],
                   domains: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Equal-sub-type mean candidate CE per domain.

    Same two-level unweighted aggregation as the KL, for the same reason: a
    token-weighted domain mean is a mean over whichever sub-type tokenizes
    longest.
    """
    out: dict[str, float] = {}
    for domain, subtypes in domains.items():
        values = [per_subtype[s].as_dict()["abl_ce"] for s in subtypes
                  if s in per_subtype]
        if len(values) == len(list(subtypes)) and values:
            out[domain] = sum(values) / len(values)
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

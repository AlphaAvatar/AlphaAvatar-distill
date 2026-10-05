#!/usr/bin/env python3
"""The D-series Top-K adoption validation. Engineering evidence only.

    python scripts/pod/topk_adoption_driver.py --out DIR [--deadline-s N]

**One set of model forwards, two reductions.** Every comparison below comes from
the SAME logits: paying for a second set of forwards to compare two reducers
would be buying nothing. What it answers, against the completed full-vocabulary
qualification as the baseline:

```text
A  the reference Top-200 probability mass -- distribution, not just the mean
B  full-vocabulary vs K+1 KL: absolute, relative, rank, worst case
C  DEPTH discrete decisions, both scores per candidate from shared forwards
D  state evaluation: equal-domain KL, worst domain, critical-token KL, ordering
E  performance: sketch bytes, recomputes, wall clock, peak memory
```

**A moved decision is not a failure.** This is an explicitly authorized new
scientific protocol (maintainer decision 2026-10-04), so a difference means the
D-series protocol is observably different from the historical full-vocabulary one
— which is the thing being characterized, not a defect.

**What C can and cannot say.** The greedy search is run ONCE and chooses by the
FULL-VOCABULARY score, so the trajectory is the historical one and both scores are
recorded for every candidate it evaluates. Once the two scores would choose
different layers, the Top-K protocol's own trajectory diverges and is no longer
observable from these forwards — so the comparison is exact up to and including
the first disagreeing round, and counterfactual after it. That limit is inherent
to sharing forwards and is reported rather than papered over.

Nothing instance-specific lives in core: `K` comes from the D-series policy
module, the vocabulary and geometry from the model, the paths from frozen records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[2]
STATE_EVAL = "artifacts/stage1/state_eval_v1"
#: One spelling of the policy, used by every stage here.
POLICY_ID = "positions.supervised_target_v1"
TEACHER_BINDING = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"

#: Both calibration profiles D1's DEPTH step may run under. Timing one is not a
#: max: historical full-vocab evidence showed materially different DEPTH timings.
D1_DEPTH_PROFILES = ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2")

#: PREDECLARED, before the measurement, and not to be widened to admit a
#: violation. `KL(topK+tail) <= KL(full)` is mathematics; two independent float32
#: implementations summing V against K+1 terms agree to about this much. The
#: violation an independent review caught was 2.968e-04 absolute / 2.2e-03
#: relative -- two orders outside it -- and it was a defect in the tail
#: arithmetic, not a tolerance question.
LOWER_BOUND_ABS_TOL = 1e-5
LOWER_BOUND_REL_TOL = 1e-5


def _bootstrap(repo: Path) -> None:
    for extra in ("src", "scripts", "scripts/data", "scripts/pod"):
        path = str(repo / extra)
        if path not in sys.path:
            sys.path.insert(0, path)


class AdoptionError(RuntimeError):
    """A stage could not produce the evidence it exists to produce."""


def quantiles(values: list[float]) -> dict[str, float]:
    """mean/min/P1/P5/P50/P95/P99/max. A mean alone hides the tail.

    Item 10A asks for the distribution because the question is whether the Top-K
    support holds enough mass EVERYWHERE, and an average over 59,763 positions
    says nothing about the thousand worst.
    """
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        #: Nearest-rank, which for a descriptive report needs no interpolation
        #: policy to argue about.
        return ordered[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {"n": n, "mean": sum(ordered) / n, "min": ordered[0],
            "p1": q(0.01), "p5": q(0.05), "p50": q(0.50), "p95": q(0.95),
            "p99": q(0.99), "max": ordered[-1]}


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, by hand because adding scipy for one number is not worth it.

    Ties are averaged, so a scorer that produced many equal values cannot inflate
    the correlation by accident.
    """
    if len(a) != len(b) or len(a) < 3:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return None if da == 0 or db == 0 else num / (da * db)


def compare_scalars(full: list[float], topk: list[float]) -> dict[str, Any]:
    """Item 10B's comparison, for any paired list of scores."""
    if not full or len(full) != len(topk):
        return {"n": 0, "_why": "no paired scores"}
    diffs = [t - f for f, t in zip(full, topk)]
    rel = [(t - f) / abs(f) for f, t in zip(full, topk) if f != 0]
    worst = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
    return {
        "n": len(full),
        "absolute_difference": quantiles([abs(d) for d in diffs]),
        "signed_difference": quantiles(diffs),
        "relative_difference": quantiles([abs(r) for r in rel]) if rel else {},
        "spearman": spearman(full, topk),
        "worst": {"index": worst, "full": full[worst], "topk": topk[worst],
                  "absolute": abs(diffs[worst]),
                  "relative": (abs(diffs[worst]) / abs(full[worst])
                               if full[worst] else None)},
        "_sign_of_the_difference": (
            "the K+1 KL is a coarsening of the partition and therefore a LOWER "
            "bound on the full-vocabulary KL, so every signed difference should "
            "be <= 0 up to float error. A positive one would be a defect, not a "
            "finding."),
        "n_positive_signed": sum(1 for d in diffs if d > 1e-6),
        #: AGAINST THE PREDECLARED TOLERANCE. `n_positive_signed` counts any
        #: excess at all, which float noise reaches; this counts excesses that
        #: cannot be float noise, and it is the number that decides.
        "lower_bound_tolerance": {"absolute": LOWER_BOUND_ABS_TOL,
                                  "relative": LOWER_BOUND_REL_TOL},
        "lower_bound_violations": sum(
            1 for f, t in zip(full, topk)
            if (t - f) > LOWER_BOUND_ABS_TOL + LOWER_BOUND_REL_TOL * abs(f)),
    }


# --- the stages -------------------------------------------------------------

class Journal:
    """Append-only stage log, flushed per event; the useful case is a crash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.stages: list[dict[str, Any]] = []

    def event(self, **fields: Any) -> None:
        row = {"t": round(time.time() - self.started, 3), **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[{row['t']:9.3f}] {fields.get('stage', '')} "
              f"{fields.get('status', '')} "
              f"{json.dumps({k: v for k, v in fields.items() if k not in ('stage','status')}, default=str)[:170]}",
              flush=True)

    def stage(self, name: str):
        return _Stage(self, name)


class _Stage:
    def __init__(self, journal: Journal, name: str) -> None:
        self.journal, self.name = journal, name

    def __enter__(self):
        self.t0 = time.time()
        self.journal.event(stage=self.name, status="start")
        self.result: dict[str, Any] = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = round(time.time() - self.t0, 3)
        ok = exc is None
        self.journal.event(stage=self.name, status="ok" if ok else "failed",
                           seconds=elapsed,
                           **({} if ok else {"error": f"{exc_type.__name__}: {exc}"}))
        self.journal.stages.append(
            {"stage": self.name, "seconds": elapsed,
             "status": "ok" if ok else "failed", **self.result,
             **({} if ok else {
                 "error": f"{exc_type.__name__}: {exc}",
                 "traceback": "".join(
                     traceback.format_exception(exc_type, exc, tb))})})
        return False


def stage_C_depth_dual_scores(*, repo: Path, workdir: Path, teacher_path: str,
                              journal: Journal, deadline=None) -> dict[str, Any]:
    """One real DEPTH path, both scores per candidate, from the SAME forwards.

    The greedy chooses by the FULL-VOCABULARY score, so the trajectory is the
    historical protocol's and every candidate it evaluates is scored both ways
    from forwards that were paid for once. Once the two scores would choose
    different layers the Top-K protocol's own trajectory diverges and stops being
    observable from these forwards, so the comparison is EXACT up to and including
    the first disagreeing round and counterfactual after it. That is reported, not
    hidden: it is inherent to sharing forwards, which is what item 10C asks for.
    """
    import torch

    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.operators.depth.causal_kl_greedy import (
        DepthCausalKLGreedyV1,
    )
    from aadistill.initialization.operators.base import OperatorContext
    from aadistill.initialization.scoring.support import (
        FULL_VOCAB_V1, sketch_forward_kl_mean_batch, sketch_reference,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from experiments.phase_a3 import a3_session as A3S
    from experiments.phase_d_series.scoring_protocol import (
        D_SERIES_BATCH_PACKING, D_SERIES_MICRO_BATCH_SIZE, D_SERIES_SUPPORT,
    )

    adapter = get_adapter("qwen3")
    spec = A3S.path_spec(workdir_device="cuda")
    depth_step = spec.steps[0]
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )

    #: The same two calls `materialize_fixed_path` makes, in the same order:
    #: `resolve()` hands back the RAW frozen evidence under `ids` because that is
    #: what the pinned content hash is defined over, and ONE boundary converts it
    #: to the `input_ids` the operators read. Re-deriving either here would be a
    #: second path to the same items, which is how two runs come to disagree.
    profile = get_profile(depth_step.profile_id)
    items = prepare_calibration_items(profile.resolve(repo),
                                      profile_id=profile.qualified_id)
    model = adapter.load(teacher_path, dtype="bfloat16", device="cuda")

    #: Per (candidate, group) pairs of scores, recorded by the observer from the
    #: forwards the operator already did.
    pairs: list[dict[str, Any]] = []
    sketch_seconds = [0.0]
    #: [n, sum, min, max] over every valid position, plus a strided sample for
    #: the quantiles. Reduced on the pod so the record stays transportable.
    mass_hist = [0, 0.0, float("inf"), float("-inf")]
    mass_sample: list[float] = []

    def observer(**kw) -> None:
        t0 = time.perf_counter()
        refs, abls = kw["refs"], kw["abls"]
        B, T, V = refs.shape
        sk = sketch_reference(
            refs.reshape(-1, V),
            torch.zeros(B * T, dtype=torch.long, device=refs.device),
            top_k=int(D_SERIES_SUPPORT.top_k), chunk=512)
        topk = sketch_forward_kl_mean_batch(
            sk.support_indices.reshape(B, T, -1),
            sk.support_log_probs.reshape(B, T, -1),
            sk.tail_log_prob.reshape(B, T), abls, kw["mask"],
            has_tail=int(D_SERIES_SUPPORT.top_k) < V, weights=kw["weights"])
        #: THE REFERENCE TOP-K MASS, item 10A, from the same sketch. Only over
        #: VALID positions: padding holds whatever the pad token produced.
        mass = sk.support_log_probs.exp().sum(dim=-1).reshape(B, T)
        valid = kw["mask"].to(mass.device).bool()
        #: THE MASS IS REDUCED HERE, NOT SHIPPED. Recording every valid
        #: position's mass made the record 300 MB -- 71 million floats -- and
        #: a2's fetch truncated at 211 MB, leaving a record that will not parse
        #: at all. What item 10A needs is the DISTRIBUTION, so the pod computes
        #: the running histogram and the extremes and ships those.
        flat = mass[valid].flatten()
        nonlocal_mass = flat.double()
        mass_hist[0] += int(flat.numel())
        mass_hist[1] += float(nonlocal_mass.sum())
        mass_hist[2] = min(mass_hist[2], float(flat.min()))
        mass_hist[3] = max(mass_hist[3], float(flat.max()))
        #: A bounded RESERVOIR for the quantiles: deterministic stride, so two
        #: runs of the same work sample the same positions.
        if len(mass_sample) < 200_000:
            step = max(1, int(flat.numel()) // 64)
            mass_sample.extend(float(x) for x in flat[::step][:64])
        pairs.append({
            "skip": sorted(kw["skip"]),
            "items": [i["item_id"] for i in kw["group"].items],
            "subtypes": [i["subtype"] for i in kw["group"].items],
            "full": [float(v) for v in kw["values"]],
            "topk": [float(v) for v in topk],
        })
        sketch_seconds[0] += time.perf_counter() - t0

    ctx = OperatorContext(
        adapter=adapter, model=model, parent_spec=adapter.spec_of(model),
        target_spec=spec.target_spec, profile=profile, calibration_items=items,
        seed=spec.seed, device="cuda", workdir=workdir,
        config={"n_calibration_items": len(items),
                "position_policy_hash": None},
        #: FULL VOCABULARY defines the trajectory, so the removal order this
        #: produces is the historical protocol's and the Top-K scores are measured
        #: against it rather than the other way round.
        distribution_support=FULL_VOCAB_V1,
        execution=ExecutionConfig(
            micro_batch_size=D_SERIES_MICRO_BATCH_SIZE,
            calibration_batch_packing=D_SERIES_BATCH_PACKING),
        score_observer=observer, deadline=deadline)
    ctx.config.pop("position_policy_hash")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    outcome = DepthCausalKLGreedyV1().apply(ctx)
    seconds = round(time.time() - t0, 3)
    peak = int(torch.cuda.max_memory_allocated())
    #: SAVE THE CHILD. Stage D needs a real candidate of a real compressed
    #: geometry, and this is one -- produced by the operator under measurement, at
    #: the real vocabulary, so it is logit-comparable with the teacher. Buying a
    #: second materialization to get a candidate would be paying twice for
    #: something this stage already built.
    child_path = workdir / "depth_child"
    adapter.save(outcome.model, str(child_path))
    del model
    torch.cuda.empty_cache()
    domain_map: dict[str, list[str]] = {}
    for item in items:
        domain_map.setdefault(item["domain"], [])
        if item["subtype"] not in domain_map[item["domain"]]:
            domain_map[item["domain"]].append(item["subtype"])
    return {
        "candidate_checkpoint": str(child_path),
        #: RECORDED, from the items this run actually scored, so the analysis
        #: cannot balance over a different map than the decision did.
        "domain_map": domain_map,
        "seconds": seconds,
        "peak_memory_bytes": peak,
        "observer_seconds": round(sketch_seconds[0], 3),
        "rounds": outcome.artifacts["search_rounds"],
        "reference_cache": outcome.artifacts["reference_cache"],
        "timing": outcome.artifacts["timing"],
        "pairs": pairs,
        "top_k": int(D_SERIES_SUPPORT.top_k),
        "mass_summary": {
            "n": mass_hist[0],
            "mean": (mass_hist[1] / mass_hist[0]) if mass_hist[0] else None,
            "min": mass_hist[2] if mass_hist[0] else None,
            "max": mass_hist[3] if mass_hist[0] else None,
            "sample": mass_sample,
            "_sample_rule": ("a deterministic stride over each reduction's valid "
                             "positions, capped at 200,000 values, so the "
                             "quantiles are reproducible and the record stays "
                             "transportable. n/mean/min/max are over EVERY valid "
                             "position, not the sample."),
        },
        "_trajectory_defined_by": "full_vocab_v1",
    }


def candidate_scores(pairs: list[dict[str, Any]],
                    domain_map: dict[str, list[str]],
                    ) -> dict[tuple, dict[str, float]]:
    """A candidate's two scores, aggregated THE WAY THE OPERATOR AGGREGATES.

    This was a plain pooled mean over items, and it was wrong. The operator scores
    a candidate with `domain_balanced_score` -- mean per SUBTYPE, then balanced
    across DOMAINS -- so a pooled mean weights each domain by how many items it
    happens to contribute. The reconstruction disagreed with the operator at two
    of eight rounds and produced a removal order containing one layer three times,
    which is impossible for a greedy removal.

    `domain_balanced_score` is IMPORTED rather than reimplemented: the aggregation
    is the thing that was got wrong once, and a second hand-written copy of it is
    how that happens again.
    """
    from aadistill.initialization.statistics.contribution import (
        domain_balanced_score,
    )

    grouped: dict[tuple, dict[str, dict[str, list[float]]]] = {}
    for p in pairs:
        key = tuple(p["skip"])
        slot = grouped.setdefault(key, {"full": {}, "topk": {}})
        for subtype, f, t in zip(p["subtypes"], p["full"], p["topk"]):
            slot["full"].setdefault(subtype, []).append(f)
            slot["topk"].setdefault(subtype, []).append(t)

    out: dict[tuple, dict[str, float]] = {}
    for key, slot in grouped.items():
        scores = {}
        for arm in ("full", "topk"):
            means = {st: sum(v) / len(v) for st, v in slot[arm].items()}
            primary, _ = domain_balanced_score(means, domain_map)
            scores[arm] = primary
        out[key] = scores
    return out


def _depth_operator_premises(repo: Path) -> dict[str, Any]:
    """Resolve what stage P needs, at `$0`, and say which arm prices.

    Three failures this would have caught, each of which has cost a paid session
    in this repository: an implementation id that the registries do not resolve; a
    calibration mixture that is not staged; and a position policy whose weights
    are silently `None`, which is how a timing run comes to omit the weighted
    reduction it was supposed to price.

    The arm is DERIVED here rather than chosen in the stage: whichever of D1's two
    policies actually hands the reducer a weight tensor is the more expensive one,
    and if that ever stops being the treatment the stage must not quietly keep
    timing the treatment.
    """
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.packing import packed_batches
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.batches import active_positions
    from aadistill.initialization.scoring.content import scoring_content_config
    from aadistill.initialization.scoring.positions import (
        ALL_POSITIONS_V1, SUPERVISED_TARGET_V1, policy_config,
    )
    from experiments.phase_a3 import a3_session as A3S
    from experiments.phase_d_series import scoring_protocol as SP

    spec = A3S.path_spec(workdir_device="cpu")
    impl = get_implementation(spec.steps[0].impl_id)
    arms = {"treatment": SUPERVISED_TARGET_V1, "control": ALL_POSITIONS_V1}
    weighted: dict[str, list[str]] = {"treatment": [], "control": []}
    per_profile: dict[str, Any] = {}
    for profile_id in D1_DEPTH_PROFILES:
        profile = get_profile(profile_id)
        items = prepare_calibration_items(profile.resolve(repo),
                                          profile_id=profile.qualified_id)
        groups = [(pk.batch, pk.original_indices) for pk in packed_batches(
            items, SP.D_SERIES_MICRO_BATCH_SIZE,
            packing=SP.D_SERIES_BATCH_PACKING, pad_id=0, device="cpu")]
        group, indices = groups[0]
        shapes: dict[str, Any] = {}
        for arm, policy in arms.items():
            w = active_positions(items, policy).prediction_weights_for(
                group, indices)
            shapes[arm] = (None if w is None else
                           {"shape": list(w.shape),
                            "nonzero": int((w > 0).sum()),
                            "positions": int(w.numel())})
            if w is not None:
                weighted[arm].append(profile_id)
        #: And the hashed config builds -- `scoring_content_config` reads the
        #: items, so an unstaged mixture fails HERE rather than mid-invocation.
        config = SP.operator_config(
            {"n_calibration_items": len(items),
             **policy_config(SUPERVISED_TARGET_V1),
             **scoring_content_config(items, SUPERVISED_TARGET_V1)})
        per_profile[profile_id] = {
            "n_items": len(items), "n_groups": len(groups),
            "weights": shapes,
            "config_keys": sorted(config),
        }
    if not weighted["treatment"]:
        raise AdoptionError(
            "D1's treatment policy weights NEITHER mixture, so a timing run "
            "would not exercise the weighted reduction it is meant to price. "
            "Refusing to spend GPU time on a path that is not production's.")
    if weighted["control"]:
        raise AdoptionError(
            "the CONTROL arm now produces weights too "
            f"({weighted['control']}), so the treatment is no longer known to be "
            "the more expensive arm. The ceiling must be measured on whichever "
            "is, and that is a decision this driver must not take silently.")
    return {
        "impl_id": impl.impl_id,
        "impl_signature_hash": impl.signature_hash[:16],
        "ceiling_arm": "treatment",
        "_why_the_treatment": (
            "both D1 arms reach the same weighted reducer, and this measured that "
            "only the treatment hands it a weight tensor on the real mixtures -- "
            "the control's `prediction_weights_for` returns None, its uniform "
            "answer. So the treatment is the more expensive arm and bounds both."),
        "per_profile": per_profile,
    }


def production_operator_context(*, adapter, model, target_spec, profile, items,
                                seed: int, device: str, workdir: Path, policy,
                                observer=None):
    """An `OperatorContext` built the way `BeamSearch._expand_one` builds one.

    Separate from the stage so that the construction -- not a paraphrase of it --
    is what a `$0` CPU test executes on a toy model. The two halves that can
    silently diverge from core are both here: the hashed operator config (the
    policy's named config plus what the policy READS, via the same two helpers
    core calls) and the context fields themselves.

    `tests` asserts this passes every keyword `_expand_one` passes, except the
    ones a single operator invocation genuinely has no use for, so a field added
    to the production path cannot be missed here in silence.
    """
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.operators.base import OperatorContext
    from aadistill.initialization.scoring.content import scoring_content_config
    from aadistill.initialization.scoring.positions import policy_config
    from experiments.phase_d_series import scoring_protocol as SP

    operator_config = SP.operator_config(
        {"n_calibration_items": len(items),
         **policy_config(policy),
         **scoring_content_config(items, policy)})
    workdir.mkdir(parents=True, exist_ok=True)
    return OperatorContext(
        adapter=adapter, model=model, parent_spec=adapter.spec_of(model),
        target_spec=target_spec, profile=profile, calibration_items=items,
        seed=seed, device=device, workdir=workdir, config=operator_config,
        distribution_support=SP.D_SERIES_SUPPORT,
        execution=ExecutionConfig(
            micro_batch_size=SP.D_SERIES_MICRO_BATCH_SIZE,
            calibration_batch_packing=SP.D_SERIES_BATCH_PACKING),
        position_policy=policy, score_observer=observer)


def stage_P_production_operator_invocation(*, repo: Path, teacher_path: str,
                                          journal: Journal,
                                          workdir: Path) -> dict[str, Any]:
    """Time ONE REAL ``depth.causal_kl_greedy_v1`` invocation. Both profiles.

    **This stage exists because its predecessor timed a shadow loop.** The first
    version reimplemented the candidate inner loop here -- sketch, forward,
    reduce -- and so omitted everything else the production scorer pays:
    ``active.prediction_weights_for(group, indices)``, the weighted reduction,
    the ``values.tolist()`` host transfer that IS the production synchronization,
    the per-subtype collection and ``domain_balanced_score``. It also prebuilt the
    reference sketches before timing, so the cache fill -- operator work that
    happens once inside every real ``apply()`` -- was priced at zero. An
    independent review caught both.

    **What historical ``operator_seconds`` actually measures** settles the shape
    of this stage. `BeamSearch._expand_one` records
    ``operator_seconds=round(elapsed, 4)`` around ``impl.execute(ctx)`` -- the
    WHOLE invocation: packing, the reference cache, every candidate of every
    round, the greedy bookkeeping and the child construction. So the honest
    measurement is the same timing around the same call, and no term has to be
    extrapolated from a per-candidate figure at all. ``260 x per-candidate`` was
    never the quantity the cost table holds.

    The production path is REUSED, not approximated: the operator is fetched from
    the registry by the id the frozen path declares, the context is built from the
    same values `_expand_one` builds it from -- including
    ``policy_config``/``scoring_content_config`` for the hashed position policy --
    and ``execute`` is called so the contract checks run too.

    NO EXTRA SYNCHRONIZATIONS. `values.tolist()` already forces one per group in
    production, so a wall clock around the invocation is exact without help. The
    forward/reduction split stays where the operator put it: opt-in, diagnostic,
    and refused for pricing.
    """
    import dataclasses
    import time as _time

    import torch

    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.scoring.batches import active_positions
    from aadistill.initialization.scoring.positions import (
        ALL_POSITIONS_V1, SUPERVISED_TARGET_V1, policy_config,
    )
    from aadistill.initialization.specs.arch import get_adapter
    from experiments.phase_a3 import a3_session as A3S
    from experiments.phase_d_series import scoring_protocol as SP

    #: A SYNCED RUN CANNOT PRICE, and the operator's own split is the one that
    #: would do it. Recorded so a diagnostic run can never be read as a price.
    split = os.environ.get("AADISTILL_DEPTH_SYNC_TELEMETRY") == "1"
    adapter = get_adapter("qwen3")
    spec = A3S.path_spec(workdir_device="cuda")
    depth_step = spec.steps[0]
    impl = get_implementation(depth_step.impl_id)
    #: D1's TREATMENT arm. Both arms reach `prediction_weights_for` -- the control
    #: is the incumbent `policy_config` but still yields an `ActivePositions` -- so
    #: the weighted path is paid either way; the treatment additionally reads each
    #: item's supervised spans, which is the more expensive setup. The control's
    #: weighting work is measured separately below rather than assumed equal.
    policy = SUPERVISED_TARGET_V1

    per_profile: dict[str, Any] = {}
    invocations: list[float] = []
    for profile_id in D1_DEPTH_PROFILES:
        profile = get_profile(profile_id)
        items = prepare_calibration_items(profile.resolve(repo),
                                          profile_id=profile.qualified_id)
        model = adapter.load(teacher_path, dtype="bfloat16", device="cuda")
        if getattr(model.config, "use_cache", False):
            model.config.use_cache = False

        #: PER-CANDIDATE BOUNDARIES, from a timestamp in the execution-only
        #: observer. A timestamp is not a synchronization: these are DIAGNOSTIC
        #: and the pricing input is the invocation total below.
        stamps: list[tuple[str, float]] = []

        def stamp(**kw) -> None:
            stamps.append((repr(sorted(kw["skip"])), _time.perf_counter()))

        ctx = production_operator_context(
            adapter=adapter, model=model, target_spec=spec.target_spec,
            profile=profile, items=items, seed=spec.seed, device="cuda",
            workdir=workdir / f"P-{profile_id.replace('@', '-')}",
            policy=policy, observer=stamp)

        torch.cuda.reset_peak_memory_stats()
        #: TIMED EXACTLY AS `BeamSearch._expand_one` TIMES IT -- `time.time()`
        #: around `execute`, so the number is the same quantity the committed cost
        #: table holds. The observer's stamps are `perf_counter`, a DIFFERENT
        #: epoch, so the span baseline below must be taken from that clock and not
        #: from this one: subtracting a `perf_counter` reading from a `time.time`
        #: reading yields about -1.76e9 seconds for the first candidate.
        started = _time.time()
        span_base = _time.perf_counter()
        outcome = impl.execute(ctx)
        elapsed = _time.time() - started

        timing = dict(outcome.artifacts.get("timing") or {})
        cache = dict(outcome.artifacts.get("reference_cache") or {})
        rounds = list(outcome.artifacts.get("search_rounds") or [])
        #: A DEFECTIVE DIAGNOSTIC MUST NOT COST THE MEASUREMENT. `_candidate_spans`
        #: refuses an impossible duration, which is a programming error and fails
        #: in the CPU tests; here the invocation has already been paid for, so the
        #: record keeps the price and says the distribution is unavailable.
        spans_error = None
        try:
            spans = _candidate_spans(stamps, span_base)
        except ValueError as exc:
            spans, spans_error = [], str(exc)
        totals = [s["seconds"] for s in spans]
        item_seconds = float(timing.get("item_seconds") or 0.0)
        per_profile[profile_id] = {
            "operator_invocation_seconds": round(elapsed, 4),
            "n_items": len(items),
            "candidate_subsets": int(timing.get("candidate_subsets") or 0),
            "rounds": len(rounds),
            "removed": [r.get("chosen") for r in rounds],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "operator_timing": timing,
            "reference_cache": cache,
            #: WHERE THE TIME WENT, so a reader can see that the terms a
            #: per-candidate figure omits are real and are included here.
            "attribution": {
                "scoring_loop_seconds": round(item_seconds, 4),
                "outside_scoring_loop_seconds": round(elapsed - item_seconds, 4),
                "_outside_is": ("packing, the reference-sketch cache object, the "
                                "position-policy setup, greedy bookkeeping and "
                                "the CHILD CONSTRUCTION -- all of it inside "
                                "operator_seconds and none of it in a "
                                "per-candidate number"),
                "reference_seconds": timing.get("reference_seconds"),
                "ablated_seconds": timing.get("ablated_seconds"),
                "distortion_seconds": timing.get("distortion_seconds"),
                "_distortion_includes": ("the weighted Top-K reduction, the "
                                         "values.tolist() host transfer and the "
                                         "per-subtype collection"),
                #: THE SPLIT IS ONLY A SPLIT WHEN IT IS ATTRIBUTED. Without the
                #: opt-in syncs the forwards are asynchronous, so the tail of each
                #: lands on whichever phase forces the next synchronization --
                #: here the reduction's host transfer. The SUM is sound; quoting
                #: `distortion_seconds` as the reduction's cost is not.
                "split_is_attributed": timing.get("split_is_attributed"),
                "_if_split_is_not_attributed": (
                    "`ablated_seconds` and `distortion_seconds` are NOT a clean "
                    "forward/reduction split: no synchronization separates them, "
                    "so the asynchronous forward's tail is billed to the phase "
                    "that forces the next one. Their sum is the scoring loop; "
                    "neither alone is that phase's cost. This is deliberate -- "
                    "inserting the syncs is what made a6 unusable for pricing."),
            },
            #: DIAGNOSTIC ONLY.
            "per_candidate_distribution": quantiles(totals) if totals else None,
            "per_candidate_n": len(spans),
            "per_candidate_seconds": [s["seconds"] for s in spans],
            "_per_candidate_unavailable": spans_error,
            "_extrapolation_cross_check": (
                {"candidates_times_max_seconds":
                     round(len(spans) * max(totals), 4),
                 "measured_invocation_seconds": round(elapsed, 4),
                 "_meaning": ("`candidates x max` is what the superseded stage "
                              "would have priced, BEFORE adding any fixed term. "
                              "Reported so the direction of the old error is "
                              "visible, not to price anything.")}
                if totals else None),
        }
        invocations.append(elapsed)
        journal.event(stage=f"P.{profile_id}", status="ok",
                      invocation_s=round(elapsed, 2),
                      candidates=int(timing.get("candidate_subsets") or 0),
                      outside_loop_s=round(elapsed - item_seconds, 2))
        del outcome, ctx, model
        torch.cuda.empty_cache()

    weighting = _weighting_cost_both_policies(
        repo=repo, profile_id=D1_DEPTH_PROFILES[0],
        policies={"treatment": SUPERVISED_TARGET_V1,
                  "control": ALL_POSITIONS_V1},
        active_positions=active_positions)

    return {
        "measurement_path": "operator_execute_v1",
        "_measurement_path_meaning": (
            "the number below is a wall clock around `impl.execute(ctx)` on the "
            "registry's `depth.causal_kl_greedy_v1`, which is the SAME quantity "
            "and the same code path `BeamSearch._expand_one` records as "
            "`operator_seconds`. A record without this marker was produced by the "
            "superseded shadow loop and cannot price."),
        "profiles": list(D1_DEPTH_PROFILES),
        "position_policy": policy_config(policy) or {"position_policy": "incumbent"},
        "per_profile": per_profile,
        #: THE PRICING INPUT. One measured invocation, the MAX over the profiles.
        #: Nothing is multiplied by a candidate count and nothing is added for a
        #: fixed term -- a whole invocation already contains both.
        "operator_invocation_seconds_max": round(max(invocations), 4),
        "operator_invocation_seconds_mean": round(
            sum(invocations) / len(invocations), 4),
        "sync_split_enabled": split,
        "_valid_for_pricing": not split,
        "_if_split_was_enabled": (
            "AADISTILL_DEPTH_SYNC_TELEMETRY=1 inserts two synchronize() calls per "
            "group inside the hot path. The operator's own source says that "
            "perturbs it. Such a run describes the split and prices nothing."),
        "position_weighting_cost": weighting,
        "micro_batch_size": SP.D_SERIES_MICRO_BATCH_SIZE,
        "packing": SP.D_SERIES_BATCH_PACKING,
        "top_k": int(SP.D_SERIES_SUPPORT.top_k),
        "_this_is_the_operator_term": (
            "one DEPTH expansion's OPERATOR cost, complete. A cost-model CELL is "
            "one expansion END TO END, so the non-operator phases -- parent load, "
            "materialize, identify, canonical reload, validation, state "
            "evaluation -- must still be ADDED from committed telemetry. They are "
            "NOT re-measured here: paying GPU time again for phases the committed "
            "telemetry already holds would buy nothing."),
        "_max_not_mean": (
            "a ceiling built on a mean is not a ceiling. The max over the two "
            "profiles is the pricing input."),
        "_root_bounds_the_deeper_parents": (
            "measured at the ROOT parent. A deeper parent is a strictly smaller "
            "model -- no frozen operator increases any structural field, "
            "established by tests/initialization/test_operators_only_shrink.py -- "
            "so the root invocation bounds the deeper cell too."),
    }


def _candidate_spans(stamps: list[tuple[str, float]],
                     started: float) -> list[dict[str, Any]]:
    """Per-candidate wall times, from the observer's timestamps.

    The observer fires once per (candidate, group) AFTER that group's host
    transfer, so the last stamp of a candidate marks the end of its scoring. A
    span therefore runs from the previous candidate's last stamp to this one's,
    which charges each candidate one neighbour's subtype aggregation -- exact in
    the sum, and off by the difference between two aggregations in the max. That
    is why these are diagnostic and the invocation total is what prices.
    """
    if not stamps:
        return []
    spans: list[dict[str, Any]] = []
    #: One clock. The stamps come from `perf_counter` and a baseline taken from
    #: `time.time()` differs from them by the unix epoch, so the first span would
    #: read about -1.76e9 seconds and the distribution's min, p1, p5 and mean
    #: would all be that number wearing a plausible shape.
    if min(s[1] for s in stamps) < started:
        raise ValueError(
            "a stamp precedes the span baseline, so the two came from different "
            "clocks: stamps must be perf_counter() and so must the baseline")
    prev = started
    i = 0
    while i < len(stamps):
        key = stamps[i][0]
        j = i
        while j + 1 < len(stamps) and stamps[j + 1][0] == key:
            j += 1
        end = stamps[j][1]
        spans.append({"skip": key, "groups": j - i + 1,
                      "seconds": round(end - prev, 4)})
        prev = end
        i = j + 1
    return spans


def _weighting_cost_both_policies(*, repo: Path, profile_id: str,
                                  policies: dict[str, Any],
                                  active_positions) -> dict[str, Any]:
    """`active_positions` setup cost for each D1 arm, measured not assumed.

    The invocation above runs the TREATMENT. The control reaches the same
    weighted reduction -- it is the incumbent `policy_config` but still yields an
    `ActivePositions` -- so the only way its cost could exceed the treatment's is
    in this setup. Measured on the host, where it runs.
    """
    import time as _time

    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile

    profile = get_profile(profile_id)
    items = prepare_calibration_items(profile.resolve(repo),
                                      profile_id=profile.qualified_id)
    out: dict[str, Any] = {"profile": profile_id, "n_items": len(items)}
    for arm, policy in policies.items():
        t0 = _time.perf_counter()
        active_positions(items, policy)
        out[arm] = {"setup_seconds": round(_time.perf_counter() - t0, 4),
                    "policy_hash": policy.policy_hash[:16]}
    out["_why"] = ("the priced invocation runs the treatment; this shows whether "
                   "the control's position setup could exceed it. It is per "
                   "INVOCATION, not per candidate.")
    return out


def analyse_C(stage_c: dict[str, Any],
              domain_map: dict[str, list[str]]) -> dict[str, Any]:
    """Items 10A, 10B and 10C, from stage C's recorded pairs."""
    pairs = stage_c["pairs"]
    full = [v for p in pairs for v in p["full"]]
    topk = [v for p in pairs for v in p["topk"]]
    summary = stage_c.get("mass_summary") or {}
    mass = list(summary.get("sample") or [])

    by_skip = candidate_scores(pairs, domain_map)
    cand_full = [v["full"] for v in by_skip.values()]
    cand_topk = [v["topk"] for v in by_skip.values()]
    #: AFTER the aggregation, not before it. Reading `cand_full` one line above
    #: where it is built is the same use-before-assignment that cost $0.0632 and
    #: 15 seconds of a paid pod once already; here a $0 import check caught it.
    B_per_item = compare_scalars(full, topk)
    B_per_cand = compare_scalars(cand_full, cand_topk)

    viol = (B_per_item["lower_bound_violations"]
            + B_per_cand["lower_bound_violations"])
    if viol:
        raise AdoptionError(
            f"{viol} lower-bound violation(s) outside the predeclared tolerance "
            f"({LOWER_BOUND_ABS_TOL} absolute / {LOWER_BOUND_REL_TOL} relative). "
            "KL(topK+tail) <= KL(full) is mathematics, so this is a defect in the "
            "implementation and NOT a finding about the protocol. The protocol is "
            "not adopted on this evidence -- stop and diagnose.")
    return {
        "A_reference_top_k_mass": {
            "top_k": stage_c["top_k"],
            "over_all_valid_positions": {k: summary.get(k) for k in
                                         ("n", "mean", "min", "max")},
            "distribution": quantiles(mass),
            "_what": ("the reference distribution's probability mass inside its "
                      "own Top-K, per valid prediction position. The tail bucket "
                      "holds one minus this."),
            "_sampled": ("up to 4096 valid positions per (candidate, group) "
                         "reduction, which is every position for these group "
                         "widths; stated so a reader does not take it for the "
                         "whole mixture if a wider batch ever truncates."),
        },
        "B_full_vs_k_plus_1": {
            "per_item_per_group": B_per_item,
            "per_candidate": B_per_cand,
        },
        "C_discrete_decisions": _decision_comparison(stage_c, by_skip),
    }


def _decision_comparison(stage_c: dict[str, Any],
                         by_skip: dict[tuple, dict[str, float]],
                         ) -> dict[str, Any]:
    """Which layer each protocol would choose, round by round.

    The greedy ran under full vocabulary, so its own choices are the
    full-vocabulary winners. For each round, the Top-K winner is recomputed from
    the pairs recorded during THAT round -- the candidates the round actually
    evaluated -- so both winners come from one set of forwards.
    """
    rounds_out = []
    first_disagreement = None
    for rec in stage_c["rounds"]:
        removed_before = tuple(sorted(rec.get("removed_before") or []))
        #: A candidate of this round is a skip set that is `removed_before` plus
        #: exactly one more layer.
        candidates = {}
        for key, slot in by_skip.items():
            extra = set(key) - set(removed_before)
            if len(key) == len(removed_before) + 1 and len(extra) == 1 and \
                    set(removed_before) <= set(key):
                layer = next(iter(extra))
                candidates[layer] = (slot["full"], slot["topk"])
        if not candidates:
            continue
        full_rank = sorted(candidates, key=lambda l: candidates[l][0])
        topk_rank = sorted(candidates, key=lambda l: candidates[l][1])
        row = {
            "round": rec.get("round"),
            "n_candidates": len(candidates),
            "full_vocab": {
                "winner": full_rank[0],
                "runner_up": full_rank[1] if len(full_rank) > 1 else None,
                "winner_score": candidates[full_rank[0]][0],
                "margin": (candidates[full_rank[1]][0]
                           - candidates[full_rank[0]][0]
                           if len(full_rank) > 1 else None)},
            "reference_topk_tail": {
                "winner": topk_rank[0],
                "runner_up": topk_rank[1] if len(topk_rank) > 1 else None,
                "winner_score": candidates[topk_rank[0]][1],
                "margin": (candidates[topk_rank[1]][1]
                           - candidates[topk_rank[0]][1]
                           if len(topk_rank) > 1 else None)},
            "winners_agree": full_rank[0] == topk_rank[0],
            "spearman_over_candidates": spearman(
                [candidates[l][0] for l in sorted(candidates)],
                [candidates[l][1] for l in sorted(candidates)]),
            #: The operator's OWN recorded choice, so the reconstruction above can
            #: be checked rather than trusted.
            "operator_chose": rec.get("chosen"),
        }
        row["reconstruction_matches_the_operator"] = (
            row["full_vocab"]["winner"] == rec.get("chosen"))
        if not row["winners_agree"] and first_disagreement is None:
            first_disagreement = rec.get("round")
        rounds_out.append(row)

    exact_through = (len(rounds_out) - 1 if first_disagreement is None
                     else first_disagreement)
    return {
        "rounds": rounds_out,
        "full_vocab_removal_order": [r["full_vocab"]["winner"]
                                     for r in rounds_out],
        "topk_removal_order_would_be": [r["reference_topk_tail"]["winner"]
                                        for r in rounds_out],
        "n_rounds_where_winners_disagree": sum(
            1 for r in rounds_out if not r["winners_agree"]),
        "first_disagreeing_round": first_disagreement,
        "comparison_is_exact_through_round": exact_through,
        "_claim_boundary": (
            "the greedy chose by FULL VOCABULARY, so its removal order is the "
            "historical protocol's and is exact. The Top-K column is the layer "
            "that protocol WOULD have chosen given the same parent -- exact up "
            "to and including the first disagreeing round, and counterfactual "
            "after it, because from that round on the Top-K protocol's own "
            "parent would differ and these forwards do not describe it. Sharing "
            "forwards is what makes the comparison affordable and this is its "
            "price."),
        "_a_moved_decision_is_not_a_failure": (
            "this is an authorized new protocol. A disagreement means it is "
            "observably different from the full-vocabulary one, which is the "
            "thing being characterized."),
        "_reconstruction_check": (
            "every round reports whether the full-vocabulary winner recomputed "
            "from the recorded pairs equals the layer the OPERATOR chose. A "
            "False anywhere means this analysis is not reading the same scores "
            "the decision was made on, and nothing below it can be trusted."),
    }


def build_positional_candidate(*, repo: Path, workdir: Path,
                              teacher_path: str) -> dict[str, Any]:
    """A real compressed candidate for stage D, without re-running a search.

    `depth.positional_v0` declares `CalibrationNeed.NONE`, so this is seconds
    rather than the half hour `depth.causal_kl_greedy_v1` costs. That is the right
    trade HERE and the reason is specific: stage D compares two REDUCERS on one
    candidate, and the candidate's selection rule does not enter the comparison.
    What has to be real is the geometry, the vocabulary, the device and the
    evaluator -- all of which are.

    It is NOT a substitute for stage C's candidate anywhere a selection rule
    matters, and the record says which candidate it used so no reader has to
    guess.
    """
    import torch

    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.operators.base import OperatorContext
    from aadistill.initialization.operators.register import BUILTIN_OPERATORS
    from aadistill.initialization.specs.arch import get_adapter
    from experiments.phase_a3 import a3_session as A3S

    impl = next(o for o in BUILTIN_OPERATORS
                if o.impl_id == "depth.positional_v0")
    adapter = get_adapter("qwen3")
    spec = A3S.path_spec(workdir_device="cuda")
    model = adapter.load(teacher_path, dtype="bfloat16", device="cuda")
    ctx = OperatorContext(
        adapter=adapter, model=model, parent_spec=adapter.spec_of(model),
        target_spec=spec.target_spec, profile=None, calibration_items=(),
        seed=spec.seed, device="cuda", workdir=workdir, config={},
        execution=ExecutionConfig())
    t0 = time.time()
    outcome = impl.apply(ctx)
    path = workdir / "positional_candidate"
    adapter.save(outcome.model, str(path))
    seconds = round(time.time() - t0, 3)
    del model
    torch.cuda.empty_cache()
    return {
        "candidate_checkpoint": str(path),
        "built_by": "depth.positional_v0",
        "seconds": seconds,
        "_why_this_candidate": (
            "stage D compares two REDUCERS on one candidate; the candidate's "
            "selection rule does not enter that comparison, and this one needs "
            "no calibration so it costs seconds instead of half an hour. The "
            "geometry, vocabulary, device and evaluator are all real."),
    }


def stage_D_state_eval(*, repo: Path, artifact_path: str, teacher_path: str,
                       journal: Journal, numerics=None) -> dict[str, Any]:
    """Item 10D and the state-eval half of 10E, both supports, one pair of models.

    Two evaluators over the SAME suite and the SAME loaded models. The forwards are
    not literally shared — each evaluator runs its own, because the evaluator owns
    its reduction — but the MODELS are loaded once, which is where the cost is, and
    nothing here is a duplicated search.
    """
    import torch

    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.initialization.execution import ExecutionConfig
    from aadistill.initialization.scoring.support import FULL_VOCAB_V1
    from aadistill.initialization.scoring.positions import get_position_policy
    from experiments.phase_d_series.scoring_protocol import (
        D_SERIES_BATCH_PACKING, D_SERIES_MICRO_BATCH_SIZE, D_SERIES_SUPPORT,
    )
    #: IMPORTED, not re-derived. `d1_qualification_driver._load_suite` already
    #: owns this contract -- `load_state_eval.load` returns the whole MANIFEST as
    #: its third element, not a hash, and an earlier version of that helper would
    #: have bound a protocol id over a dict's repr and looked perfectly bound. A
    #: second hand-written copy here is how that lesson gets unlearned.
    sys.path.insert(0, str(repo / "scripts/autoinit"))
    import load_state_eval

    from d1_qualification_driver import _load_suite

    suite, items, content_sha256 = _load_suite(load_state_eval, repo)
    if not content_sha256:
        raise AdoptionError(
            f"{STATE_EVAL}/manifest.json carries no content_sha256, so neither "
            "protocol id can be bound and the comparison would describe two "
            "measurements nobody can place")
    adapter = get_adapter("qwen3")
    teacher = adapter.load(teacher_path, dtype="bfloat16", device="cuda")
    candidate = adapter.load(artifact_path, dtype="bfloat16", device="cuda")
    #: QUALIFIED id. The registry keys on `positions.x_v1@v1`, and resolving by
    #: the bare id raises -- which it did, 1987 s into a $0.8552 subrun, AFTER
    #: stage C had completed. `d1_qualification_driver` carries a comment saying
    #: exactly this; I wrote a second call instead of using the one that already
    #: knew, which is the whole argument for importing a contract rather than
    #: restating it.
    policy = get_position_policy(POLICY_ID if "@" in POLICY_ID
                                 else f"{POLICY_ID}@v1")
    execution = ExecutionConfig(micro_batch_size=D_SERIES_MICRO_BATCH_SIZE,
                                calibration_batch_packing=D_SERIES_BATCH_PACKING)

    out: dict[str, Any] = {"arms": {}}
    for label, support in (("full_vocab_v1", FULL_VOCAB_V1),
                           ("reference_topk_tail_v1", D_SERIES_SUPPORT)):
        #: `numerics` too. Without it these evaluators computed a DIFFERENT
        #: execution fingerprint from the pair the driver bound up front, so one
        #: record carried two protocol ids per support and a reader could not tell
        #: which measurement it described. The identity must be the bound one.
        ev = StateEvaluator(suite, items, device="cuda", position_policy=policy,
                            execution=execution, distribution_support=support,
                            suite_content_sha256=content_sha256,
                            numerics=numerics)
        ev.prime_reference(teacher)
        torch.cuda.reset_peak_memory_stats()
        before = int(torch.cuda.memory_allocated())
        t0 = time.time()
        evaluation = ev.evaluate(candidate, "topk_adoption")
        seconds = round(time.time() - t0, 3)
        peak = int(torch.cuda.max_memory_allocated())
        sketch_bytes = sum(s.bytes_held() for s in ev._ref_sketches.values())
        out["arms"][label] = {
            "measurement_protocol_id": ev.measurement_protocol_id,
            "reduction": ev.reduction.as_dict(),
            "seconds": seconds,
            "peak_memory_bytes": peak,
            "allocated_before_bytes": before,
            "evaluation_delta_bytes": peak - before,
            "reference_sketch_bytes": sketch_bytes,
            "n_reference_sketches": len(ev._ref_sketches),
            "values": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in (evaluation.values or {}).items()},
            "detail_per_domain": (evaluation.detail or {}).get("per_domain_kl"),
            "worst_domain": (evaluation.detail or {}).get("worst_domain"),
        }
        journal.event(stage=f"state_eval.{label}", status="ok", seconds=seconds,
                      peak_gib=round(peak / 2**30, 3),
                      protocol=ev.measurement_protocol_id[:16])
        del ev
        torch.cuda.empty_cache()

    a, b = out["arms"]["full_vocab_v1"], out["arms"]["reference_topk_tail_v1"]
    if a["measurement_protocol_id"] == b["measurement_protocol_id"]:
        raise AdoptionError(
            "both state evaluations carry the SAME protocol id, so the identity "
            "does not distinguish the two supports and this comparison describes "
            "one measurement against itself")
    shared = sorted(set(a["values"]) & set(b["values"]))
    out["comparison"] = {
        "protocol_ids_differ": (a["measurement_protocol_id"]
                               != b["measurement_protocol_id"]),
        "metrics": {k: {"full_vocab": a["values"][k],
                        "reference_topk_tail": b["values"][k],
                        "absolute": b["values"][k] - a["values"][k],
                        "relative": ((b["values"][k] - a["values"][k])
                                     / a["values"][k] if a["values"][k] else None)}
                    for k in shared
                    if isinstance(a["values"][k], (int, float))
                    and isinstance(b["values"][k], (int, float))},
        "worst_domain_agrees": a["worst_domain"] == b["worst_domain"],
        "_nll_must_be_exact": (
            "CE is not a KL and is not coarsened by this protocol. Any `.nll` "
            "metric differing by more than float noise is a DEFECT, not a "
            "protocol difference."),
        "_speed": {
            "full_vocab_seconds": a["seconds"],
            "topk_seconds": b["seconds"],
            "ratio": (round(b["seconds"] / a["seconds"], 4)
                      if a["seconds"] else None)},
        "_memory": {
            "full_vocab_peak_bytes": a["peak_memory_bytes"],
            "topk_peak_bytes": b["peak_memory_bytes"],
            "topk_reference_sketch_bytes": b["reference_sketch_bytes"],
            "delta_ratio": (round(b["evaluation_delta_bytes"]
                                  / a["evaluation_delta_bytes"], 4)
                            if a["evaluation_delta_bytes"] else None)},
    }
    del teacher, candidate
    torch.cuda.empty_cache()
    return out


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--deadline-s", type=int, default=0)
    ap.add_argument("--required-inputs", action="store_true",
                    help="print the frozen assets this needs, one JSON per line, "
                         "and exit. The launcher pushes exactly these.")
    ap.add_argument("--stages", default="C,D,P",
                    help="which stages to run: C (the DEPTH dual scoring, ~30 "
                         "min), D (the state evaluation, minutes) and P (ONE REAL "
                         "DEPTH operator invocation per profile, ~25 min each). A "
                         "subrun that needs only D builds its candidate with "
                         "depth.positional_v0 instead of repeating the search.")
    ap.add_argument("--check-only", action="store_true",
                    help="every stage except the two expensive ones: the CUDA "
                         "probe, the four process-global registries, the frozen "
                         "asset load, both protocol bindings and the D-series "
                         "policy. Seconds, before the teacher is resident.")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    _bootstrap(repo)

    from d1_qualification_driver import (
        WallClockDeadline, _numerics, bind_protocol, environment, fetch_teacher,
    )

    if args.required_inputs:
        from d1_qualification_driver import main as qmain

        return qmain(["--required-inputs", "--repo", str(repo)])

    if not args.out:
        ap.error("--out is required")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    journal = Journal(out / "journal.jsonl")
    record: dict[str, Any] = {
        "schema": "aadistill.d_series.topk_adoption/v1",
        "_contract": (
            "ENGINEERING EVIDENCE ONLY, for the maintainer decision of "
            "2026-10-04. No recovery training, no behavioural screening, no "
            "confirmation, no formal search, no promotion, no GO/NO-GO. No "
            "D-series behavioural prompt is read. AUTHORIZES NOTHING."),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    deadline = None
    try:
        with journal.stage("environment") as st:
            record["environment"] = environment(repo)
            st.result = {"gpu": record["environment"]["gpu_name"]}

        with journal.stage("register_and_policy") as st:
            from aadistill.initialization.adapters import (
                register_builtin_adapters,
            )
            from aadistill.initialization.operators.attention.gqa import (
                activation_importance,
            )
            from experiments.calibration import register_builtin_profiles
            from experiments.phase_c2.search_space import register_c2_operators

            register_builtin_adapters()
            register_builtin_profiles()
            register_c2_operators()
            activation_importance.register()

            from experiments.phase_d_series.scoring_protocol import describe

            record["scoring_protocol"] = describe()
            st.result = {"top_k": record["scoring_protocol"]["top_k"]}

        from aadistill.initialization.execution import ExecutionConfig
        from aadistill.initialization.scoring.support import FULL_VOCAB_V1
        from experiments.phase_d_series.scoring_protocol import (
            D_SERIES_BATCH_PACKING, D_SERIES_MICRO_BATCH_SIZE, D_SERIES_SUPPORT,
        )

        #: From the probed environment, not invented: `_numerics` requires the
        #: device class, compute dtype and accumulation dtype, and an earlier
        #: version of ITS caller filtered by signature and would have dropped
        #: every field it supplied. An unstated numerical condition is not a
        #: default.
        numerics = _numerics(record["environment"])
        execution = ExecutionConfig(
            micro_batch_size=D_SERIES_MICRO_BATCH_SIZE,
            calibration_batch_packing=D_SERIES_BATCH_PACKING)
        #: BOTH protocol identities bound before anything expensive, and the pair
        #: must DIFFER -- if they agreed, the whole comparison would be of one
        #: measurement against itself.
        for label, support in (("full_vocab", FULL_VOCAB_V1),
                               ("reference_topk_tail", D_SERIES_SUPPORT)):
            with journal.stage(f"bind_{label}") as st:
                bound = bind_protocol(
                    repo, policy_id=POLICY_ID,
                    execution=execution, numerics=numerics,
                    distribution_support=support)
                record[f"bound_{label}"] = bound["bound"]
                st.result = {"protocol":
                             bound["bound"]["measurement_protocol_id"][:16]}
        if (record["bound_full_vocab"]["measurement_protocol_id"]
                == record["bound_reference_topk_tail"]["measurement_protocol_id"]):
            raise AdoptionError(
                "the two supports produced the SAME measurement protocol id, so "
                "the identity does not distinguish them and every comparison "
                "below would be of one measurement against itself")

        wanted = {x.strip() for x in (args.stages or "C,D,P").split(",")
                  if x.strip()}
        record["stages_requested"] = sorted(wanted)
        unknown = wanted - {"C", "D", "P"}
        if unknown:
            raise AdoptionError(
                f"--stages names {sorted(unknown)}; known: C (DEPTH dual "
                "scoring), D (state evaluation), P (one real DEPTH operator "
                "invocation). A typo would silently run less than intended and "
                "look like a clean result.")

        #: EVERY PREMISE STAGE P RESTS ON, before the teacher is resident and
        #: before a candidate is scored. The superseded timing stage needed none of
        #: this because it reimplemented the inner loop; running the REAL operator
        #: means the registry, the policy and the mixture all have to be there, and
        #: each of them has failed a paid session before.
        if "P" in wanted or args.check_only:
            with journal.stage("depth_operator_and_policy") as st:
                record["P_premises"] = _depth_operator_premises(repo)
                st.result = {
                    "impl": record["P_premises"]["impl_id"],
                    "weighted_arm": record["P_premises"]["ceiling_arm"]}

        with journal.stage("teacher_fetch_verify") as st:
            teacher_path = fetch_teacher(repo)
            st.result = {"path": teacher_path}

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the CUDA probe, the four registries, the D-series policy, both "
                "protocol bindings (which differ), the frozen teacher and stage "
                "P's premises -- the DEPTH implementation, the real mixtures and "
                "which arm weights them -- resolved on this interpreter. Neither "
                "expensive stage ran.")
            return 0

        if args.deadline_s > 0:
            deadline = WallClockDeadline(args.deadline_s)

        if "C" in wanted:
            with journal.stage("C_depth_dual_scores") as st:
                record["C_depth"] = stage_C_depth_dual_scores(
                    repo=repo, workdir=out / "depth",
                    teacher_path=teacher_path, journal=journal,
                    deadline=deadline)
                st.result = {"seconds": record["C_depth"]["seconds"],
                             "pairs": len(record["C_depth"]["pairs"])}

            with journal.stage("analyse_ABC") as st:
                #: The subtype -> domain map from the FROZEN mixture the operator
                #: scored, so the reconstruction balances domains exactly as the
                #: decision did -- a POOLED mean instead disagreed with the
                #: operator at two of eight rounds.
                record["analysis"] = analyse_C(record["C_depth"],
                                               record["C_depth"]["domain_map"])
                c = record["analysis"]["C_discrete_decisions"]
                bad = [r["round"] for r in c["rounds"]
                       if not r["reconstruction_matches_the_operator"]]
                if bad:
                    raise AdoptionError(
                        f"the analysis reconstructed a different winner from the "
                        f"operator at rounds {bad}, so it is not reading the "
                        "scores the decision was made on and nothing it reports "
                        "about decisions can be trusted")
                st.result = {"disagreeing_rounds":
                             c["n_rounds_where_winners_disagree"],
                             "mass_p5":
                                 record["analysis"]["A_reference_top_k_mass"][
                                     "distribution"].get("p5")}

        if "D" in wanted:
            with journal.stage("D_state_eval_both_supports") as st:
                #: DEFINED AND NEVER CALLED in the first version of this driver,
                #: which would have produced items A, B, C and E and silently
                #: omitted D. Found by reading the stage list against the
                #: authorization's `covers` while a run was in flight.
                #:
                #: Stage C's own child when C ran, otherwise a positional
                #: candidate -- and the record states WHICH, because a reader must
                #: not have to guess what was evaluated.
                if record.get("C_depth"):
                    candidate = record["C_depth"]["candidate_checkpoint"]
                    record["D_candidate"] = {
                        "from": "C_depth",
                        "built_by": "depth.causal_kl_greedy_v1"}
                else:
                    built = build_positional_candidate(
                        repo=repo, workdir=out / "candidate",
                        teacher_path=teacher_path)
                    candidate = built["candidate_checkpoint"]
                    record["D_candidate"] = built
                record["D_state_eval"] = stage_D_state_eval(
                    repo=repo, artifact_path=candidate,
                    teacher_path=teacher_path, journal=journal,
                    numerics=numerics)
                cmp_ = record["D_state_eval"]["comparison"]
                st.result = {
                    "protocol_ids_differ": cmp_["protocol_ids_differ"],
                    "worst_domain_agrees": cmp_["worst_domain_agrees"]}

        if "P" in wanted:
            with journal.stage("P_production_operator_invocation") as st:
                record["P_production_timing"] = (
                    stage_P_production_operator_invocation(
                        repo=repo, teacher_path=teacher_path, journal=journal,
                        workdir=out / "P"))
                st.result = {"invocation_s_max":
                             record["P_production_timing"][
                                 "operator_invocation_seconds_max"]}

        record["status"] = "COMPLETE"
    except BaseException as exc:                      # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        journal.event(stage="driver", status="failed", error=str(exc)[:300])
    finally:
        record["stages"] = journal.stages
        record["elapsed_seconds"] = round(time.time() - journal.started, 3)
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        #: WRITTEN ON EVERY PATH. A failed adoption run's evidence is the point.
        (out / "adoption.json").write_text(
            json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
        print(f"\nwrote {out}/adoption.json  status={record['status']}",
              flush=True)
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())

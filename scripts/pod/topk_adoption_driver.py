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
TEACHER_BINDING = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"


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
        pairs.append({
            "skip": sorted(kw["skip"]),
            "items": [i["item_id"] for i in kw["group"].items],
            "subtypes": [i["subtype"] for i in kw["group"].items],
            "full": [float(v) for v in kw["values"]],
            "topk": [float(v) for v in topk],
            "mass": [float(m) for m in mass[valid].flatten()[:4096]],
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
    del model
    torch.cuda.empty_cache()
    return {
        "seconds": seconds,
        "peak_memory_bytes": peak,
        "observer_seconds": round(sketch_seconds[0], 3),
        "rounds": outcome.artifacts["search_rounds"],
        "reference_cache": outcome.artifacts["reference_cache"],
        "timing": outcome.artifacts["timing"],
        "pairs": pairs,
        "top_k": int(D_SERIES_SUPPORT.top_k),
        "_trajectory_defined_by": "full_vocab_v1",
    }


def analyse_C(stage_c: dict[str, Any]) -> dict[str, Any]:
    """Items 10A, 10B and 10C, from stage C's recorded pairs."""
    pairs = stage_c["pairs"]
    full = [v for p in pairs for v in p["full"]]
    topk = [v for p in pairs for v in p["topk"]]
    mass = [m for p in pairs for m in p["mass"]]

    #: Per CANDIDATE, not per group: a candidate's score is the domain-balanced
    #: mean over its groups, and the decision is made on that. Grouped back by
    #: `skip` so the two protocols' per-candidate orderings can be compared.
    by_skip: dict[tuple, dict[str, list[float]]] = {}
    for p in pairs:
        key = tuple(p["skip"])
        slot = by_skip.setdefault(key, {"full": [], "topk": []})
        slot["full"].extend(p["full"])
        slot["topk"].extend(p["topk"])
    cand_full, cand_topk, cand_keys = [], [], []
    for key, slot in by_skip.items():
        cand_keys.append(list(key))
        cand_full.append(sum(slot["full"]) / len(slot["full"]))
        cand_topk.append(sum(slot["topk"]) / len(slot["topk"]))

    return {
        "A_reference_top_k_mass": {
            "top_k": stage_c["top_k"],
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
            "per_item_per_group": compare_scalars(full, topk),
            "per_candidate": compare_scalars(cand_full, cand_topk),
        },
        "C_discrete_decisions": _decision_comparison(stage_c, by_skip),
    }


def _decision_comparison(stage_c: dict[str, Any],
                         by_skip: dict[tuple, dict[str, list[float]]],
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
                candidates[layer] = (
                    sum(slot["full"]) / len(slot["full"]),
                    sum(slot["topk"]) / len(slot["topk"]))
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


def stage_D_state_eval(*, repo: Path, artifact_path: str, teacher_path: str,
                       journal: Journal) -> dict[str, Any]:
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
    policy = get_position_policy("positions.supervised_target_v1")
    execution = ExecutionConfig(micro_batch_size=D_SERIES_MICRO_BATCH_SIZE,
                                calibration_batch_packing=D_SERIES_BATCH_PACKING)

    out: dict[str, Any] = {"arms": {}}
    for label, support in (("full_vocab_v1", FULL_VOCAB_V1),
                           ("reference_topk_tail_v1", D_SERIES_SUPPORT)):
        ev = StateEvaluator(suite, items, device="cuda", position_policy=policy,
                            execution=execution, distribution_support=support,
                            suite_content_sha256=content_sha256)
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
                    repo, policy_id="positions.supervised_target_v1",
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

        with journal.stage("teacher_fetch_verify") as st:
            teacher_path = fetch_teacher(repo)
            st.result = {"path": teacher_path}

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the CUDA probe, the four registries, the D-series policy, both "
                "protocol bindings (which differ) and the frozen teacher resolved "
                "on this interpreter. Neither expensive stage ran.")
            return 0

        if args.deadline_s > 0:
            deadline = WallClockDeadline(args.deadline_s)

        with journal.stage("C_depth_dual_scores") as st:
            record["C_depth"] = stage_C_depth_dual_scores(
                repo=repo, workdir=out / "depth", teacher_path=teacher_path,
                journal=journal, deadline=deadline)
            st.result = {"seconds": record["C_depth"]["seconds"],
                         "pairs": len(record["C_depth"]["pairs"])}

        with journal.stage("analyse_ABC") as st:
            record["analysis"] = analyse_C(record["C_depth"])
            c = record["analysis"]["C_discrete_decisions"]
            st.result = {"disagreeing_rounds":
                         c["n_rounds_where_winners_disagree"],
                         "mass_p5": record["analysis"]["A_reference_top_k_mass"][
                             "distribution"].get("p5")}

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

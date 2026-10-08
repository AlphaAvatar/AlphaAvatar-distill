#!/usr/bin/env python3
"""A-bsz1 vs A-bsz3: the structural/numerical comparison. $0 half runs anywhere.

    # the $0 half, now
    PYTHONPATH=src:scripts python scripts/autoinit/compare_a_bsz3.py --write

    # the STRUCTURAL half: `structural_half()`, called from a session that has
    # the verified frozen parent staged on an approved device.

**Two halves, and the split is not arbitrary.**

The `$0` half is a pure function of the frozen mixture's item lengths:
physical forward count, padded and executed positions, padding-over-valid, per
group widths. `pack` was written as a function of lengths alone precisely so a
protocol's cost is known before a pod exists.

The structural half needs the real teacher in bf16 on the approved device, because the
question it answers is whether a *shape-dependent GEMM* moved a selection.
`attn_out` is in the measured shape-dependent group (`K/N >= 1.6`) and this
operator's score is built from exactly that projection, so a CPU/float32
rehearsal cannot reach the behaviour under test. Running it there would not be
a cheaper measurement; it would be a more expensive way of learning nothing.

**What the structural half records, per protocol:** the initialization artifact
digest, the kept q-head set per layer, the per-head scores, the margin at each
layer's selection boundary, the calibration token count, the execution counters
the operator observed, runtime and peak VRAM. The margins matter more than the
scores: a selection flips when the gap between the last kept head and the first
dropped one is smaller than the perturbation, so the margin distribution is
what says whether a flip was close or decisive.

**The runtime measurement is repeated and interleaved, because the workload is
eleven seconds long.** `attention.activation_importance_v1` was measured at
**11.2732 s** on this exact parent geometry (C1 attempt 18). A single sample of
an 11-second workload on a shared cloud GPU is not a measurement of a ratio,
and the two protocols must not be run back to back in a fixed order either:
whichever ran first would pay for allocator growth, autotuning and kernel
selection, which is a difference between the ARMS' ORDER and not between bsz1
and bsz3. So the protocols alternate `A,B,A,B,…`, each protocol's FIRST round
is a declared warm-up excluded from the statistic, and what is reported is a
distribution rather than one number.

Repeating also buys something that was not asked for and is free: every round
produces an artifact digest, so a protocol that does not reproduce its OWN
digest within one session is visible here rather than inferred later.

**No adoption threshold lives in this file.** It reports the measured speedup
and its spread. The `1.25x` bar belonged to the causal-KL packing pilot, a
different workload with a different forward count, and the maintainer
withdrew it for this study on 2026-10-01: the operational value of the saving
is judged against the amount of attention-calibration work the D-series search
is expected to do, which is not a number this script can know.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.phase_a3.a_bsz3 import (  # noqa: E402
    ATTENTION_IMPL_ID, PROTOCOLS, execution_comparison, frozen_identities,
    item_token_counts,
)
#: Rank correlation and the distribution summary, imported rather than
#: rewritten. They are generic statistics that happen to live in the C3
#: comparison module; reusing them is what makes this operator's structural
#: report and the causal-KL one commensurable, and a second Spearman
#: implementation would be a second place for it to be subtly different.
#: This is NOT a dependency on `attention.causal_kl_v1`: no operator, no
#: scorer and no identity crosses here.
from experiments.phase_c3.compare import _distribution, _spearman  # noqa: E402

OUT = "logs/stages/stage-1/phase_c3/analyses/a_bsz3_comparison.json"

#: RECORDED BEFORE THE MEASUREMENT, as this project does for any check whose
#: answer it thinks it can guess. Writing it down is what makes a miss a miss
#: rather than something reinterpreted afterwards.
PREDICTION = {
    "_registered_before_any_gpu_run": True,
    "artifact_digest_will_match": False,
    "confidence": "high",
    "why": (
        "This operator scores heads by mean_t ||W_o,h a_h(t)||^2 -- built from "
        "the attention output projection. `attn_out` is in the group measured "
        "to reduce shape-dependently on an L40S (K/N >= 1.6), and the only "
        "exactly reproducible shape is batch_size=1 with zero padding, which "
        "is precisely what A-bsz1 is and A-bsz3 is not. For the causal-KL "
        "scorer, B1 vs B4 already disagreed on 16 of 448 retained head slots "
        "(3.57%) across 5 of 28 layers at rank correlation 0.981."),
    "expected_magnitude": (
        "small: a few percent of retained head slots, concentrated where the "
        "selection-boundary margin is smallest, with high rank correlation"),
    "runtime_is_the_open_question": (
        "The prior negative result for this workload -- bsz=4 at 0.946x of "
        "bsz=1 for the statistics collector -- was measured at ORIGINAL-ORDER "
        "packing, which costs 36.73% padding. A-bsz3 costs 4.01%. That is the "
        "scientific case for measuring rather than assuming the old number "
        "transfers. Separately, length-sorted packing was worth 1.1884x on the "
        "causal-KL scorer -- but that is a different workload with a different "
        "forward count and does not transfer either. NO ADOPTION THRESHOLD IS "
        "ATTACHED to either figure: the causal-KL pilot's 1.25x bar was "
        "withdrawn for this study by the maintainer decision of 2026-10-01, "
        "and the measured speedup is reported for judgement against the "
        "D-series attention-calibration workload."),
    "the_saving_is_small_in_absolute_terms_and_that_is_known_in_advance": (
        "attention.activation_importance_v1 was measured at 11.2732 s on this "
        "exact parent geometry in C1 attempt 18. Even an unattainable 3x would "
        "save about 7.5 s per attention calibration. That is the reason this "
        "study is scoped as an execution-optimization validation and not as a "
        "population-level claim, and the reason the runtime is measured over "
        "repeated interleaved rounds rather than once."),
}


def _one_round(spec, *, name, execution, adapter, root_loader, verified,
               workdir: Path, repo_root: Path, calibration_items,
               cuda: bool) -> dict[str, Any]:
    """One suffix materialization under one protocol, with its own evidence."""
    import torch

    from aadistill.initialization.planning.fixed_path import (
        materialize_fixed_path_suffix,
    )

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps, evidence = materialize_fixed_path_suffix(
        spec, adapter=adapter, root_loader=root_loader, workdir=workdir,
        verified=verified, repo_root=repo_root,
        calibration_items=calibration_items, execution=execution)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tail = steps[-1]
    trace = dict(tail.trace or {})
    out = {
        "execution": execution.as_trace(),
        #: FROM THE MATERIALIZER, because the consumer needs it and must not
        #: reconstruct it. The driver built `<workdir>/rep0` by hand and
        #: handed that to the trainer, which refused with "Unrecognized model
        #: ... should have a `model_type` key" -- the checkpoint is at
        #: `<workdir>/steps/03_attention`, and only the step that wrote it
        #: knows that. Stage F died there at `$0.58` with the whole
        #: structural result already in hand.
        "checkpoint_path": str(tail.checkpoint_path),
        "artifact_digest": tail.identity.artifact_digest,
        "result_spec_hash": tail.result_spec_hash,
        "kept_q_heads_per_layer": trace.get("kept_q_heads_per_layer"),
        "selection_margin_per_layer": trace.get("selection_margin_per_layer"),
        "head_scores_per_layer": trace.get("head_scores_per_layer"),
        "calibration_tokens": trace.get("calibration_tokens"),
        #: What the operator COUNTED while running, not what `padding_profile`
        #: predicts from the item lengths. The $0 analysis holds the
        #: prediction; keeping the two apart is what lets them disagree.
        "observed_execution_counters": {
            k: trace.get(k) for k in (
                "physical_forward_invocations", "executed_positions",
                "valid_positions", "padded_positions")},
        "traced_execution": {
            "micro_batch_size": trace.get("micro_batch_size"),
            "calibration_batch_packing":
                trace.get("calibration_batch_packing"),
            "reference_path": trace.get("reference_path")},
        #: TWO CLOCKS, and they answer different questions. `scorer_seconds`
        #: is the statistics pass alone, CUDA-synchronized at both ends --
        #: the only part of this that batching can change. `suffix_seconds`
        #: adds the child build and the checkpoint write, which are identical
        #: work under either protocol and therefore pull any ratio built from
        #: them toward 1.
        "scorer_seconds": trace.get("scorer_seconds"),
        "suffix_seconds": round(elapsed, 3),
        "peak_vram_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 4)
                          if cuda else None),
        "suffix_premise": evidence,
    }
    if cuda:
        torch.cuda.empty_cache()
    return out


def structural_half(spec, *, adapter, root_loader, verified, workdir: Path,
                    repo_root: Path, calibration_items=None,
                    device: str = "cuda", repeats: int = 4,
                    expected_incumbent_digest: str | None = None,
                    zero_cost: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the ATTENTION suffix under both protocols and compare the results.

    Driven through `materialize_fixed_path_suffix`, which is the production
    path, for two reasons that both matter here.

    **The pinned prefix is not re-executed.** `micro_batch_size: 1` on the
    shared parent is identity-sensitive and has already changed a structural
    decision once -- DEPTH round 7 chose layer 21 over 17 and the parent digest
    came out wrong. Handing the whole path an `execution` override would have
    changed the prefix too. The suffix API starts from the ALREADY-VERIFIED
    parent, re-identifies it from disk against the frozen digest, and applies
    the override to the tail alone.

    **Path identity is preserved.** The full frozen spec is passed unchanged
    and only `indices` narrows, so both protocols produce a result bound to the
    arm's real path hash rather than to a synthesized one-step path.

    **Rounds alternate and the first of each is a warm-up.** See the module
    docstring: an 11-second workload measured once, in a fixed order, reports
    the arms' order as much as the protocols. Round 0's record is the canonical
    structural evidence (digest, kept heads, scores, margins); rounds 1.. add
    digests and timings and their checkpoints are deleted as soon as they are
    read, because nothing downstream consumes them and a repeated 1.19 GB write
    is the kind of residency this project has run out of disk on before.
    """
    cuda = device.startswith("cuda")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    rounds: dict[str, list[dict[str, Any]]] = {n: [] for n in PROTOCOLS}
    cleanup_failure: str | None = None
    #: INTERLEAVED. `for round: for protocol:` and not the other way round --
    #: a drifting device, a neighbour on the host or a thermal ramp then moves
    #: both protocols together instead of landing on whichever ran second.
    for rep in range(repeats):
        if cleanup_failure is not None:
            break
        for name, execution in PROTOCOLS.items():
            rep_dir = Path(workdir) / name / f"rep{rep}"
            record = _one_round(
                spec, name=name, execution=execution, adapter=adapter,
                root_loader=root_loader, verified=verified, workdir=rep_dir,
                repo_root=repo_root, calibration_items=calibration_items,
                cuda=cuda)
            record["round"] = rep
            record["warm_up"] = rep == 0
            rounds[name].append(record)
            if rep == 0:
                continue
            #: Round 0 keeps its tree: it is the canonical artifact and a
            #: later stage may want to re-identify it. Every other round has
            #: already given up everything it holds.
            try:
                shutil.rmtree(rep_dir)
            except OSError as exc:  # pragma: no cover - filesystem dependent
                cleanup_failure = (
                    f"could not release {rep_dir}: {exc}. Further rounds are "
                    "cancelled rather than filling the container disk; the "
                    "rounds already measured stand.")
                break

    rounds, common, dropped = pair_rounds(rounds)

    results: dict[str, Any] = {}
    for name, recs in rounds.items():
        canonical = dict(recs[0])
        canonical["rounds"] = [
            {"round": r["round"], "warm_up": r["warm_up"],
             "artifact_digest": r["artifact_digest"],
             "scorer_seconds": r["scorer_seconds"],
             "suffix_seconds": r["suffix_seconds"],
             "peak_vram_gib": r["peak_vram_gib"]} for r in recs]
        canonical["peak_vram_gib"] = _max_or_none(
            [r["peak_vram_gib"] for r in recs])
        canonical["digest_repeatable_within_session"] = (
            len({r["artifact_digest"] for r in recs}) == 1)
        canonical["timing"] = _timing(recs)
        results[name] = canonical

    a, b = results["A_bsz1"], results["A_bsz3"]
    identical = a["artifact_digest"] == b["artifact_digest"]
    comparison: dict[str, Any] = {
        "repeats": repeats,
        #: The PAIRED count -- the same for both protocols by construction
        #: above, so this is a property of the comparison rather than of
        #: whichever protocol happened to be named first.
        "rounds_completed": common,
        "artifact_digest_identical": identical,
        "result_spec_hash_identical": (
            a["result_spec_hash"] == b["result_spec_hash"]),
        "calibration_tokens_identical": (
            a["calibration_tokens"] == b["calibration_tokens"]),
        "classification": ("TRANSPARENT_EXECUTION_OPTIMIZATION" if identical
                           else "DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL"),
        "_classification_rule": (
            "An identical artifact digest means bsz3 changed nothing about "
            "what A means OR about what it materializes, so it is a "
            "transparent execution optimization: no behavioural experiment is "
            "owed and adoption turns on the measured runtime alone. ANY "
            "difference makes it a distinct numerical MATERIALIZATION "
            "protocol -- the operator semantics, the hypothesis and the "
            "estimand are still identical, and the BYTES are not."),
        "_what_a_differing_digest_implies": (
            "TWO consequences, and they are cleared separately. (1) The "
            "shortened behavioural sanity study in "
            "`plans/a_bsz3_adoption.json` may be proposed. (2) A-bsz3 may not "
            "enter D1/D2/D3 execution until the repository binds the "
            "numerical execution fingerprint to materialization/resume "
            "identity, because `compute_state_id` binds neither the "
            "ExecutionConfig nor the artifact digest -- so two differing "
            "artifacts would collide on one resumable, deduplicable state id. "
            "A passing behavioural result does not clear (2)."),
    }
    if cleanup_failure:
        comparison["cleanup_failure"] = cleanup_failure
    if dropped:
        comparison["unpaired_rounds_dropped"] = dropped
    comparison["runtime"] = _runtime_comparison(a, b)
    comparison["kept_head_selection"] = _selection_diff(
        a["kept_q_heads_per_layer"], b["kept_q_heads_per_layer"],
        a["selection_margin_per_layer"])
    comparison["head_scores"] = _score_comparison(
        a.get("head_scores_per_layer"), b.get("head_scores_per_layer"))
    comparison["execution_counters"] = _counter_comparison(a, b, zero_cost)
    comparison["digest_repeatability"] = {
        name: results[name]["digest_repeatable_within_session"]
        for name in PROTOCOLS}
    comparison["incumbent_digest_gate"] = _incumbent_gate(
        a["artifact_digest"], expected_incumbent_digest)

    void = void_reasons(results, comparison)
    if void:
        comparison["INVALID"] = void
    results["_comparison"] = comparison
    return results


def void_reasons(results: dict[str, Any],
                 comparison: dict[str, Any]) -> list[str]:
    """Every reason this comparison is not a result. Pure, so it is testable.

    These are not outcomes of the experiment. Each one says the instrument was
    not measuring what the design says it measures, and the difference matters
    because a void comparison reported as "no difference found" is the most
    expensive kind of wrong answer here.
    """
    void: list[str] = []
    a = results.get("A_bsz1") or {}
    b = results.get("A_bsz3") or {}
    #: The token count must match whatever the arithmetic did. Padded positions
    #: are masked out of the accumulator, so a difference here is not a
    #: numerical effect at all -- it is a masking defect.
    if not comparison.get("calibration_tokens_identical"):
        void.append(
            f"the two protocols saw different valid token counts "
            f"({a.get('calibration_tokens')} vs {b.get('calibration_tokens')}). "
            "Padded positions are masked out of the accumulator, so this is a "
            "masking defect and not a result.")
    for name in PROTOCOLS:
        rec = results.get(name) or {}
        if rec.get("digest_repeatable_within_session") is False:
            void.append(
                f"{name} did not reproduce its own artifact digest across "
                "rounds in one session. A protocol that is not repeatable "
                "against itself cannot be compared with another one.")
    if (comparison.get("execution_counters") or {}).get(
            "masking_invariant_holds") is False:
        void.append(
            "executed - padded != calibration_tokens for at least one "
            "protocol: the loop's own counters and the collector's mask "
            "disagree about how many positions were accumulated.")
    gate = comparison.get("incumbent_digest_gate") or {}
    if gate.get("checked") and not gate.get("matches"):
        void.append(
            f"A_bsz1 produced {str(gate.get('observed'))[:12]} where the "
            f"frozen C3 incumbent is {str(gate.get('expected'))[:12]}. The "
            "reference protocol did not reproduce the incumbent, so nothing "
            "measured here describes the incumbent.")
    return void


def pair_rounds(rounds: dict[str, list[Any]]
                ) -> tuple[dict[str, list[Any]], int, dict[str, int]]:
    """Truncate every protocol to the rounds they BOTH completed.

    A cleanup failure can stop the loop between the two protocols, leaving
    A_bsz1 with one round more than A_bsz3 -- and an unpaired extra round is
    exactly what interleaving exists to prevent, because the odd round out is
    the one a drifting device moves. The extra round is dropped and the drop
    is reported, rather than quietly widening one protocol's sample.
    """
    if not rounds:
        return rounds, 0, {}
    common = min(len(recs) for recs in rounds.values())
    dropped = {name: len(recs) - common for name, recs in rounds.items()
               if len(recs) > common}
    if not dropped:
        return rounds, common, {}
    return ({name: recs[:common] for name, recs in rounds.items()},
            common, dropped)


def _max_or_none(values: list[Any]) -> Any:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The timed rounds, with the warm-up named rather than quietly dropped."""
    timed = [r for r in records if not r["warm_up"]
             and r["scorer_seconds"] is not None]
    fell_back = False
    if not timed:
        #: One round was requested, or the operator traced no scorer clock.
        #: Reporting nothing would be worse than reporting a warm number that
        #: SAYS it is warm.
        timed = [r for r in records if r["scorer_seconds"] is not None]
        fell_back = True
    scorer = [float(r["scorer_seconds"]) for r in timed]
    suffix = [float(r["suffix_seconds"]) for r in timed]
    return {
        "n_timed_rounds": len(scorer),
        "warm_up_rounds_excluded": 0 if fell_back else 1,
        "_fell_back_to_the_warm_up": fell_back,
        "scorer_seconds": _summary(scorer),
        "suffix_seconds": _summary(suffix),
        "_scorer_seconds_is": (
            "the statistics pass alone, CUDA-synchronized at both ends -- the "
            "only part of the suffix that a batching protocol changes"),
    }


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4),
            "stdev": (round(statistics.stdev(values), 4)
                      if len(values) > 1 else 0.0),
            "values": [round(v, 4) for v in values]}


def _runtime_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The measured speedup and its spread. NO adoption threshold."""
    out: dict[str, Any] = {
        "A_bsz1": a["timing"], "A_bsz3": b["timing"],
        "_no_threshold_here": (
            "the 1.25x bar came from the causal-KL packing pilot -- a "
            "different workload with a different forward count -- and was "
            "withdrawn for this study by the maintainer decision of "
            "2026-10-01. What the saving is worth is judged against the "
            "attention-calibration work the D-series search will do."),
    }
    for label, key in (("scorer", "scorer_seconds"), ("suffix", "suffix_seconds")):
        sa, sb = a["timing"][key], b["timing"][key]
        if sa.get("n") and sb.get("n") and sb["mean"]:
            out[f"{label}_speedup_bsz1_over_bsz3"] = {
                "on_means": round(sa["mean"] / sb["mean"], 4),
                "on_mins": (round(sa["min"] / sb["min"], 4)
                            if sb["min"] else None),
                "_on_mins_is": (
                    "the least-interrupted round of each, which is the "
                    "cleaner estimate of the kernel-level difference; the "
                    "mean is what a session actually pays"),
            }
        else:
            out[f"{label}_speedup_bsz1_over_bsz3"] = None
    return out


def _score_comparison(a: Any, b: Any) -> dict[str, Any]:
    """How differently the two protocols SCORED, independent of what they kept.

    The counts in `kept_head_selection` answer "how many slots moved"; this
    answers "did the operator rank the heads the same way", which is the
    question a selection count cannot reach -- two protocols can differ in
    every score and agree on every selection.
    """
    if not a or not b or len(a) != len(b):
        return {"_unavailable": "per-head score traces are missing or mismatched"}
    flat_a = [s for layer in a for s in layer]
    flat_b = [s for layer in b for s in layer]
    if len(flat_a) != len(flat_b):
        return {"_unavailable": "the two protocols scored different head counts"}
    drift = [y - x for x, y in zip(flat_a, flat_b)]
    rel = [abs(d) / abs(x) for d, x in zip(drift, flat_a) if x != 0]
    return {
        "n_heads": len(flat_a),
        "identical": all(d == 0.0 for d in drift),
        "rank_correlation": {
            "overall": _spearman(flat_a, flat_b),
            "per_layer": [_spearman(a[i], b[i]) for i in range(len(a))],
            "_none_means": "fewer than 3 heads, or a constant vector",
        },
        "score_drift": _distribution(drift),
        "score_relative_drift": _distribution(rel),
    }


def _counter_comparison(a: dict[str, Any], b: dict[str, Any],
                        zero_cost: dict[str, Any] | None) -> dict[str, Any]:
    """Observed execution counters, and whether the `$0` model predicted them.

    The prediction is a pure function of the mixture's item lengths and was
    printed before any pod existed. The observation is counted by the operator
    as it ran. A gate that compared the prediction with itself could not fail.
    """
    out: dict[str, Any] = {"observed": {
        "A_bsz1": a["observed_execution_counters"],
        "A_bsz3": b["observed_execution_counters"]}}

    holds: bool | None = None
    for name, rec in (("A_bsz1", a), ("A_bsz3", b)):
        c = rec["observed_execution_counters"]
        if (c.get("executed_positions") is None
                or c.get("padded_positions") is None
                or rec.get("calibration_tokens") is None):
            continue
        ok = (c["executed_positions"] - c["padded_positions"]
              == rec["calibration_tokens"])
        holds = ok if holds is None else (holds and ok)
    out["masking_invariant_holds"] = holds
    out["_masking_invariant_is"] = (
        "executed_positions - padded_positions == calibration_tokens, where "
        "the left side is counted by the operator's loop and the right side "
        "by the collector's own mask. Two independent counters of one "
        "quantity; agreement is evidence, not bookkeeping.")

    if not zero_cost:
        out["_prediction_unavailable"] = "no $0 analysis was passed in"
        return out
    agreement: dict[str, Any] = {}
    for name, rec in (("A_bsz1", a), ("A_bsz3", b)):
        pred = ((zero_cost.get("protocols") or {}).get(name) or {})
        prof = pred.get("padding") or {}
        obs = rec["observed_execution_counters"]
        agreement[name] = {
            "predicted_forwards": prof.get("n_groups"),
            "observed_forwards": obs.get("physical_forward_invocations"),
            "forwards_agree": (prof.get("n_groups")
                               == obs.get("physical_forward_invocations")),
            "predicted_executed_positions": prof.get("executed_positions"),
            "observed_executed_positions": obs.get("executed_positions"),
            "executed_positions_agree": (
                prof.get("executed_positions")
                == obs.get("executed_positions")),
            "predicted_padded_positions": prof.get("padded_positions"),
            "observed_padded_positions": obs.get("padded_positions"),
            "padded_positions_agree": (prof.get("padded_positions")
                                       == obs.get("padded_positions")),
        }
    out["prediction_vs_observation"] = agreement
    out["prediction_held"] = all(
        v for row in agreement.values() for k, v in row.items()
        if k.endswith("_agree"))
    return out


def _incumbent_gate(observed: str, expected: str | None) -> dict[str, Any]:
    """Did the reference protocol reproduce the frozen C3 incumbent?

    A-bsz1 IS canonical A. If it does not rebuild `53e30566c5f7…` on this
    device then whatever else the session measured, it did not measure the
    incumbent -- and the shortened study's reuse of attempt75's A controls
    rests on exactly that identity. Free to check, and it is the third
    independent reproduction of that digest after attempt66 and attempt75.
    """
    if not expected:
        return {"checked": False,
                "_why": "no expected incumbent digest was supplied"}
    return {"checked": True, "expected": expected, "observed": observed,
            "matches": observed == expected,
            "_this_is_also": ("an independent reproduction of the frozen C3 "
                              "incumbent identity on fresh hardware")}


def _selection_diff(a: Any, b: Any, margins_a: Any) -> dict[str, Any]:
    """How far apart two kept-head selections are, and how close the calls were.

    The margin is reported for the layers that FLIPPED, because that is the
    question a reader has: did the arithmetic tip a near-tie, or did the two
    protocols genuinely disagree about which heads matter?
    """
    if not a or not b or len(a) != len(b):
        return {"_unavailable": "kept-head traces are missing or mismatched"}
    per_layer, differing, total = [], 0, 0
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x), set(y)
        total += len(sx)
        d = len(sx - sy)
        if d:
            differing += d
            row = {"layer": i, "slots_differing": d, "kept": len(sx),
                   "only_in_bsz1": sorted(sx - sy),
                   "only_in_bsz3": sorted(sy - sx)}
            if margins_a and i < len(margins_a):
                finite = [m for m in margins_a[i] if m != float("inf")]
                row["min_margin_bsz1"] = min(finite) if finite else None
            per_layer.append(row)
    return {"identical": differing == 0,
            "layers_affected": len(per_layer), "n_layers": len(a),
            "slots_differing": differing, "slots_total": total,
            "share": round(differing / total, 6) if total else 0.0,
            "per_layer": per_layer}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    doc: dict[str, Any] = {
        "schema": "aadistill.phase_c3.a_bsz3_comparison/v1",
        "_what_this_is": (
            "A-bsz1 vs A-bsz3: the incumbent ATTENTION operator under its own "
            "execution protocol and under B3's. Same operator, same scientific "
            "state -- the batching knobs are ExecutionConfig fields and never "
            "enter a hash -- so any difference here is arithmetic, not "
            "identity."),
        "attention_impl_id": ATTENTION_IMPL_ID,
        "frozen_identities": frozen_identities(),
        "prediction": PREDICTION,
        "zero_cost": execution_comparison(item_token_counts()),
        "structural": None,
        "_structural_half_requires_a_gpu_for": [
            "initialization artifact digest",
            "selected attention structure / kept heads",
            "operator scores and ordering around the selection boundary",
            "per-head rank correlation and score drift",
            "observed forward / padded / executed position counters",
            "runtime, over repeated interleaved rounds",
            "peak VRAM",
        ],
        "_why_a_gpu": (
            "attn_out reduces shape-dependently on the L40S (K/N >= 1.6) and "
            "this operator's score is built from it. A CPU float32 rehearsal "
            "cannot reach that behaviour, so it would answer a different "
            "question."),
        "authorizes": "nothing",
    }

    z = doc["zero_cost"]
    print(f"items {z['n_items']}  valid_tokens {z['valid_tokens']}")
    for name, p in z["protocols"].items():
        pad = p["padding"]
        print(f"  {name:7s} forwards {pad['n_groups']:3d}  "
              f"padded {pad['padded_positions']:7d}  "
              f"executed {pad['executed_positions']:7d}  "
              f"pad/valid {pad['padding_over_valid']:.4f}")
    print(f"  forwards {z['deltas']['physical_forwards_ratio']}x, "
          f"executed positions {z['deltas']['executed_positions_ratio']}x")
    print("\n  STRUCTURAL half NOT RUN. It needs the frozen parent staged on "
          "an approved device;\n  `structural_half` is the executable and the "
          "protocol names what it must be given.\n  No paid resource is "
          "authorized by this tool.")

    if a.write:
        (REPO_ROOT / a.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

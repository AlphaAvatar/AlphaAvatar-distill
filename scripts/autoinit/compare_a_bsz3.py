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
layer's selection boundary, the calibration token count, runtime and peak VRAM.
The margins matter more than the scores: a selection flips when the gap between
the last kept head and the first dropped one is smaller than the perturbation,
so the margin distribution is what says whether a flip was close or decisive.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.phase_c3.a_bsz3 import (  # noqa: E402
    ATTENTION_IMPL_ID, PROTOCOLS, execution_comparison, item_token_counts,
)

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
        "causal-KL scorer, below its 1.25x bar -- but that is a different "
        "workload with a different forward count and does not transfer either."),
}


def structural_half(spec, *, adapter, root_loader, verified, workdir: Path,
                    repo_root: Path, calibration_items=None,
                    device: str = "cuda") -> dict[str, Any]:
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
    """
    import torch

    from aadistill.initialization.planning.fixed_path import (
        materialize_fixed_path_suffix,
    )

    cuda = device.startswith("cuda")
    results: dict[str, Any] = {}
    for name, execution in PROTOCOLS.items():
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        steps, evidence = materialize_fixed_path_suffix(
            spec, adapter=adapter, root_loader=root_loader,
            workdir=Path(workdir) / name, verified=verified,
            repo_root=repo_root, calibration_items=calibration_items,
            execution=execution)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        tail = steps[-1]
        trace = dict(tail.trace or {})
        results[name] = {
            "execution": execution.as_trace(),
            "artifact_digest": tail.identity.artifact_digest,
            "result_spec_hash": tail.result_spec_hash,
            "kept_q_heads_per_layer": trace.get("kept_q_heads_per_layer"),
            "selection_margin_per_layer": trace.get("selection_margin_per_layer"),
            "calibration_tokens": trace.get("calibration_tokens"),
            "traced_execution": {
                "micro_batch_size": trace.get("micro_batch_size"),
                "calibration_batch_packing":
                    trace.get("calibration_batch_packing"),
                "reference_path": trace.get("reference_path")},
            "runtime_s": round(elapsed, 3),
            "peak_vram_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 4)
                              if cuda else None),
            "suffix_premise": evidence,
        }
        if cuda:
            torch.cuda.empty_cache()

    a, b = results["A_bsz1"], results["A_bsz3"]
    identical = a["artifact_digest"] == b["artifact_digest"]
    comparison: dict[str, Any] = {
        "artifact_digest_identical": identical,
        "result_spec_hash_identical": (
            a["result_spec_hash"] == b["result_spec_hash"]),
        "calibration_tokens_identical": (
            a["calibration_tokens"] == b["calibration_tokens"]),
        "runtime_ratio_bsz1_over_bsz3": (
            round(a["runtime_s"] / b["runtime_s"], 4) if b["runtime_s"] else None),
        "classification": ("PURE_EXECUTION_OPTIMIZATION" if identical
                           else "DISTINCT_NUMERICAL_PROTOCOL"),
        "_classification_rule": (
            "An identical artifact digest means bsz3 changed nothing about "
            "what A means, so it is an execution optimization and the runtime "
            "evidence alone decides adoption. ANY difference makes it a "
            "distinct numerical protocol: not promoted automatically, and "
            "decided by the frozen non-inferiority design."),
    }
    comparison["kept_head_selection"] = _selection_diff(
        a["kept_q_heads_per_layer"], b["kept_q_heads_per_layer"],
        a["selection_margin_per_layer"])
    #: The token count must match whatever the arithmetic did. Padded positions
    #: are masked out of the accumulator, so a difference here is not a
    #: numerical effect at all -- it is a masking defect, and it would
    #: invalidate the comparison rather than be one of its outcomes.
    if not comparison["calibration_tokens_identical"]:
        comparison["INVALID"] = (
            f"the two protocols saw different valid token counts "
            f"({a['calibration_tokens']} vs {b['calibration_tokens']}). Padded "
            "positions are masked out of the accumulator, so this is a masking "
            "defect and not a result; the comparison is void until it is fixed.")
    results["_comparison"] = comparison
    return results


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
        "prediction": PREDICTION,
        "zero_cost": execution_comparison(item_token_counts()),
        "structural": None,
        "_structural_half_requires_a_gpu_for": [
            "initialization artifact digest",
            "selected attention structure / kept heads",
            "operator scores and ordering around the selection boundary",
            "runtime",
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

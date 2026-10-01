"""A-bsz3: the incumbent ATTENTION operator under B3's batching protocol.

**What A-bsz3 IS.** Canonical A — `attention.activation_importance_v1`, the
same operator composition, the same scientific identities, the same frozen
shared parent — executed with

    calibration_forward_batch_size = 3
    calibration_batch_packing      = length_sorted_v1

**What A-bsz3 IS NOT.** It is not a new operator, it imports nothing from
`causal_kl`, and it has no implementation id of its own. Both knobs live on
`ExecutionConfig`, which never enters a hash, a state id or a manifest
identity — so A-bsz3 and canonical A are *the same scientific state*, and
whether they agree is a question about arithmetic rather than about identity.
Giving the knob an impl id would have answered that question by definition and
told us nothing.

**Why equivalence cannot be assumed.** The estimand, the aggregation and the
selection rule are identical under any grouping; the operator was written that
way and a CPU/float32 check confirms it. But a different grouping pads to
different widths, and this project has *measured* bf16 GEMMs reducing
shape-dependently on the L40S — `attn_out` among the divergent projections, at
`K/N >= 1.6`. The score this operator selects on is built from exactly those
attention outputs. So the algorithm being grouping-invariant does not make the
arithmetic grouping-invariant, and the comparison is an empirical one.

**The question A-bsz3 asks.** Can B3's protocol accelerate the incumbent
attention initialization without materially changing what A means or its
downstream quality? Two outcomes, and they are handled differently:

* **A-bsz3 reproduces A exactly** at the final artifact digest and the kept-head
  selection -> evidence that bsz3 is a pure execution optimization for this
  operator, adoptable on the runtime evidence alone.
* **A-bsz3 changes the initialization** -> it is a DISTINCT NUMERICAL PROTOCOL,
  is not promoted automatically, and goes to the shortened engineering
  behavioural sanity check in
  `logs/stages/stage-1/phase_c3/plans/a_bsz3_adoption.json` — at most three
  newly trained treatment probes against attempt75's existing A controls. The
  16-probe non-inferiority design that used to be named here was withdrawn on
  2026-10-01 for scope: this is an execution-optimization validation, not a
  population-level equivalence claim.

**The shared parent is untouched.** `micro_batch_size: 1` on the prefix is
identity-sensitive and has already changed a structural decision once — DEPTH
round 7 chose layer 21 over 17 and the parent digest came out wrong. A-bsz3
changes the ATTENTION step's execution and nothing before it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aadistill.initialization.calibration.packing import (
    LENGTH_SORTED_V1, ORIGINAL_ORDER_V1, pack, padding_profile,
)
from aadistill.initialization.execution import ExecutionConfig

REPO = Path(__file__).resolve().parents[3]

#: The frozen calibration mixture A calibrates on, as staged for a session.
CALIBRATION_ITEMS = REPO / "artifacts/stage1/e8_calibration_v1/items.jsonl"

#: Where the identities A-bsz3 must reproduce actually live. READ, never
#: restated: the shared parent digest, the incumbent digest and the three
#: recovery seeds all belong to the C3 preregistration, and a second copy here
#: would be a second thing to keep in step with it.
C3_PREREGISTRATION = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"

#: The incumbent. Named once, and deliberately NOT parameterised: A-bsz3 is a
#: statement about this operator, and a variant that swept implementations
#: would be a different study.
ATTENTION_IMPL_ID = "attention.activation_importance_v1"

#: The two protocols under comparison. `A_BSZ1` is canonical A's execution and
#: is the reference path the operator reproduces by construction.
A_BSZ1 = ExecutionConfig(micro_batch_size=1,
                         calibration_batch_packing=ORIGINAL_ORDER_V1)
A_BSZ3 = ExecutionConfig(micro_batch_size=3,
                         calibration_batch_packing=LENGTH_SORTED_V1)

PROTOCOLS: dict[str, ExecutionConfig] = {"A_bsz1": A_BSZ1, "A_bsz3": A_BSZ3}


class ABsz3Error(RuntimeError):
    """The A-bsz3 comparison cannot be set up as declared."""


def frozen_identities(path: Path | None = None) -> dict[str, Any]:
    """The C3 identities A-bsz3 is bound to, read from the preregistration.

    A-bsz1 **is** canonical A, so the reference protocol must rebuild the
    frozen incumbent digest from the frozen shared parent. If it does not, the
    session measured something that is not the incumbent, and the shortened
    adoption study's reuse of attempt75's A controls -- which rests entirely on
    that identity -- is void rather than approximate.

    The recovery seeds come back in the preregistration's own order, which is
    also the fail-fast order the adoption plan uses.
    """
    p = Path(path or C3_PREREGISTRATION)
    if not p.is_file():
        raise ABsz3Error(
            f"the C3 preregistration is not at {p}; A-bsz3 binds its parent, "
            "incumbent and seeds to that document and will not invent them")
    doc = json.loads(p.read_text())
    try:
        parent = doc["shared_parent"]["artifact_digest"]
        incumbent = doc["arms"]["A_incumbent"]["artifact_digest"]
        seeds = [int(s) for s in doc["seeds"]["recovery"]]
        prereg = doc["preregistration_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ABsz3Error(
            f"{p} does not carry the fields A-bsz3 binds to: {exc}") from exc
    if not seeds:
        raise ABsz3Error(f"{p} declares no recovery seeds")
    return {
        "shared_parent_artifact_digest": parent,
        "incumbent_artifact_digest": incumbent,
        "recovery_seeds": seeds,
        "c3_preregistration_sha256": prereg,
        "_source": str(p.relative_to(REPO)),
        "_the_incumbent_is": (
            "what A_bsz1 must reproduce. A-bsz3 is the same operator under a "
            "different execution knob, so a reference protocol that does not "
            "rebuild this digest has not reproduced the incumbent."),
    }


def item_token_counts(path: Path | None = None) -> list[int]:
    """Token count per item of the frozen mixture, in the mixture's own order.

    Read from `n_tokens`, which the mixture states per item, and cross-checked
    against the length of `ids`. Two statements of one quantity: a manifest
    that disagreed with its own tokens would otherwise be invisible here and
    would silently change every padding figure below.
    """
    p = Path(path or CALIBRATION_ITEMS)
    if not p.is_file():
        raise ABsz3Error(
            f"the frozen calibration mixture is not staged at {p}; the $0 "
            "padding comparison reads its real item lengths and will not "
            "invent them")
    out: list[int] = []
    for line_no, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        stated = int(row["n_tokens"])
        actual = len(row["ids"])
        if stated != actual:
            raise ABsz3Error(
                f"item {line_no} states n_tokens={stated} and carries "
                f"{actual} ids; the mixture disagrees with itself")
        out.append(stated)
    if not out:
        raise ABsz3Error(f"{p} holds no items")
    return out


def execution_comparison(lengths: list[int] | None = None) -> dict[str, Any]:
    """Everything about the two protocols that is a `$0` fact.

    Pure integer work over the mixture's real lengths: no tensors, no device,
    no model. Forward count and padding cost are therefore known before any
    pod exists, which is the whole reason `pack` was written as a function of
    lengths alone.

    What is NOT here: the artifact digest, the kept-head selection, the score
    ordering, runtime and peak VRAM. Those need the real teacher in bf16 on the
    approved device, because the question they answer is whether a
    shape-dependent GEMM moved a selection — and a CPU float32 rehearsal cannot
    reach that behaviour. Asking it there would not be a cheaper measurement;
    it would be a more expensive way of learning nothing.
    """
    lengths = list(lengths if lengths is not None else item_token_counts())
    out: dict[str, Any] = {
        "n_items": len(lengths),
        "valid_tokens": sum(lengths),
        "min_len": min(lengths), "max_len": max(lengths),
        "protocols": {},
    }
    for name, cfg in PROTOCOLS.items():
        groups = pack(lengths, cfg.micro_batch_size,
                      packing=cfg.calibration_batch_packing)
        profile = padding_profile(lengths, cfg.micro_batch_size,
                                  packing=cfg.calibration_batch_packing)
        out["protocols"][name] = {
            "execution": cfg.as_trace(),
            "physical_forwards": len(groups),
            "group_widths": [max(lengths[i] for i in g) for g in groups],
            "padding": profile,
        }
    a, b = out["protocols"]["A_bsz1"], out["protocols"]["A_bsz3"]
    out["deltas"] = {
        "physical_forwards": (b["physical_forwards"]
                              - a["physical_forwards"]),
        "physical_forwards_ratio": round(
            b["physical_forwards"] / a["physical_forwards"], 6),
        "padded_positions_added": (b["padding"]["padded_positions"]
                                   - a["padding"]["padded_positions"]),
        "executed_positions_ratio": round(
            b["padding"]["executed_positions"]
            / a["padding"]["executed_positions"], 6),
        "padding_over_valid": {
            "A_bsz1": round(a["padding"]["padding_over_valid"], 6),
            "A_bsz3": round(b["padding"]["padding_over_valid"], 6)},
        "max_group_width": {
            "A_bsz1": a["padding"]["max_group_width"],
            "A_bsz3": b["padding"]["max_group_width"]},
    }
    out["_what_the_deltas_mean"] = (
        "A_bsz1 pays no padding at all -- one item per forward -- so every "
        "padded position A_bsz3 adds is work A_bsz1 did not do. The forward "
        "COUNT falls by roughly three; the total POSITION count rises. Whether "
        "that trade is faster is a runtime measurement on the real device and "
        "is not decided here: kernel-launch overhead and padded arithmetic pull "
        "in opposite directions, and a previous sweep on this workload found "
        "bsz=4 running at 0.946x of bsz=1 for the statistics collector.")
    return out

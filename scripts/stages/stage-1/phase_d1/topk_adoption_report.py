#!/usr/bin/env python3
"""Read an adoption record and print what it found, one line per fact.

    python scripts/stages/stage-1/phase_d1/topk_adoption_report.py ADOPTION.json

Separate from the launcher because a shell heredoc that parses JSON is a thing
nobody can test without creating a pod. Runs at `$0` against any record,
including a failed one — which is the case that matters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def lines(doc: dict) -> list[str]:
    out = [f"status: {doc.get('status')}"]
    env = doc.get("environment") or {}
    if env:
        out.append(f"gpu: {env.get('gpu_name')} cc{env.get('capability')} "
                   f"torch {env.get('torch')} cuda {env.get('cuda_runtime')} "
                   f"bf16={env.get('bf16_supported')}")
    sp = doc.get("scoring_protocol") or {}
    if sp:
        out.append(f"protocol: {sp.get('protocol_id')} K={sp.get('top_k')}")
    for label in ("full_vocab", "reference_topk_tail"):
        b = doc.get(f"bound_{label}") or {}
        if b:
            out.append(f"bound {label}: "
                       f"{str(b.get('measurement_protocol_id'))[:16]} "
                       f"support={b.get('distribution_support')} "
                       f"suite={str(b.get('suite_content_sha256'))[:12]}")
    a = (doc.get("analysis") or {}).get("A_reference_top_k_mass") or {}
    d = a.get("distribution") or {}
    if d:
        out.append(f"A top-{a.get('top_k')} mass over {d.get('n')} positions: "
                   f"mean={d.get('mean'):.6f} min={d.get('min'):.6f} "
                   f"p1={d.get('p1'):.6f} p5={d.get('p5'):.6f} "
                   f"p50={d.get('p50'):.6f} p95={d.get('p95'):.6f}")
    b = (doc.get("analysis") or {}).get("B_full_vs_k_plus_1") or {}
    for scope in ("per_item_per_group", "per_candidate"):
        cmp_ = b.get(scope) or {}
        if cmp_.get("n"):
            absd = cmp_.get("absolute_difference") or {}
            rel = cmp_.get("relative_difference") or {}
            out.append(f"B {scope}: n={cmp_['n']} "
                       f"spearman={cmp_.get('spearman')} "
                       f"|abs| mean={absd.get('mean'):.3e} "
                       f"max={absd.get('max'):.3e} "
                       f"|rel| mean={rel.get('mean', float('nan')):.3e} "
                       f"positive_signed={cmp_.get('n_positive_signed')}")
    c = (doc.get("analysis") or {}).get("C_discrete_decisions") or {}
    if c:
        out.append(f"C decisions: {c.get('n_rounds_where_winners_disagree')} of "
                   f"{len(c.get('rounds') or [])} rounds disagree; "
                   f"first={c.get('first_disagreeing_round')}; "
                   f"exact_through={c.get('comparison_is_exact_through_round')}")
        out.append(f"   full-vocab order: {c.get('full_vocab_removal_order')}")
        out.append(f"   top-k would be  : {c.get('topk_removal_order_would_be')}")
        bad = [r["round"] for r in (c.get("rounds") or [])
               if not r.get("reconstruction_matches_the_operator")]
        out.append(f"   reconstruction matches the operator in every round: "
                   f"{not bad}" + (f" (bad: {bad})" if bad else ""))
    cd = doc.get("C_depth") or {}
    if cd:
        rc = cd.get("reference_cache") or {}
        out.append(f"E depth: {cd.get('seconds')}s "
                   f"peak={_gib(cd.get('peak_memory_bytes'))} "
                   f"observer={cd.get('observer_seconds')}s "
                   f"cache_mode={rc.get('mode')}")
    pm = doc.get("P_premises") or {}
    if pm:
        out.append(f"P premises: {pm.get('impl_id')} "
                   f"sig={pm.get('impl_signature_hash')} "
                   f"ceiling_arm={pm.get('ceiling_arm')}")
        for prof, v in sorted((pm.get("per_profile") or {}).items()):
            w = (v.get("weights") or {}).get("treatment") or {}
            out.append(f"   {prof}: {v.get('n_items')} items / "
                       f"{v.get('n_groups')} groups; treatment weights "
                       f"{w.get('shape')} nonzero={w.get('nonzero')}"
                       f"/{w.get('positions')}; control="
                       f"{(v.get('weights') or {}).get('control')}")
    pt = doc.get("P_production_timing") or {}
    #: A SUPERSEDED-SHAPE RECORD SAYS SO, in one line, and prints no pricing
    #: fields. a5's, a6's and a7's records carry `operator_seconds_max` -- a
    #: per-candidate figure from a reimplemented inner loop -- and rendering those
    #: through the fields below produced a column of `None`s beside
    #: `valid_for_pricing=True`, which reads exactly like a basis.
    if pt and not pt.get("measurement_path"):
        out.append(
            "P timing: SUPERSEDED SHAPE (no `measurement_path`). This is a "
            f"per-candidate measurement -- max {pt.get('operator_seconds_max')}s "
            f"over {pt.get('candidates_per_profile')} candidates per profile, "
            f"sync_split={pt.get('sync_split_enabled')} -- from a loop that was "
            "not the production scorer. DIAGNOSTIC ONLY; it prices nothing.")
        pt = {}
    if pt:
        #: THE PRICING INPUT FIRST, and whether it may price at all. A reader who
        #: sees only a number cannot tell a diagnostic run from a basis.
        out.append(f"P timing: path={pt.get('measurement_path')} "
                   f"valid_for_pricing={pt.get('_valid_for_pricing')} "
                   f"sync_split={pt.get('sync_split_enabled')}")
        out.append(f"   invocation MAX {pt.get('operator_invocation_seconds_max')}s "
                   f"= {float(pt.get('operator_invocation_seconds_max') or 0)/60:.3f} min "
                   f"(mean {pt.get('operator_invocation_seconds_mean')}s)")
        for prof, v in sorted((pt.get("per_profile") or {}).items()):
            att = v.get("attribution") or {}
            dist = v.get("per_candidate_distribution") or {}
            out.append(f"   {prof}: {v.get('operator_invocation_seconds')}s over "
                       f"{v.get('candidate_subsets')} candidates / "
                       f"{v.get('rounds')} rounds; "
                       f"peak={_gib(v.get('peak_memory_bytes'))}")
            attributed = att.get("split_is_attributed")
            out.append(f"      scoring_loop={att.get('scoring_loop_seconds')}s "
                       f"outside={att.get('outside_scoring_loop_seconds')}s "
                       f"(ref={att.get('reference_seconds')} "
                       f"abl={att.get('ablated_seconds')} "
                       f"red={att.get('distortion_seconds')}"
                       + ("" if attributed else "; NOT a clean split -- no sync, "
                          "so the async forward's tail is billed to the reduction")
                       + ")")
            if v.get("_per_candidate_unavailable"):
                out.append(f"      per-candidate: UNAVAILABLE -- "
                           f"{v['_per_candidate_unavailable']}")
            elif dist:
                #: A NEGATIVE DURATION IS IMPOSSIBLE, so a record that carries one
                #: is read as defective rather than printed as a measurement. a8's
                #: span baseline came from `time.time()` while the observer
                #: stamped `perf_counter()`, which puts the unix epoch in the first
                #: span; the INVOCATION total is unaffected, and so is the price.
                broken = (dist.get("min") or 0) < 0
                out.append(f"      per-candidate (diagnostic): "
                           f"max={dist.get('max')} p95={dist.get('p95')} "
                           f"p50={dist.get('p50')} min={dist.get('min')}"
                           + ("  <- DEFECTIVE: a negative duration means the span "
                              "baseline and the stamps came from different clocks. "
                              "max/p95/p50 stand; min/p1/p5/mean do not. The "
                              "invocation total and the price are unaffected."
                              if broken else ""))
            out.append(f"      removed: {v.get('removed')}")
        wt = pt.get("position_weighting_cost") or {}
        if wt:
            out.append(f"   weighting setup: treatment "
                       f"{(wt.get('treatment') or {}).get('setup_seconds')}s vs "
                       f"control {(wt.get('control') or {}).get('setup_seconds')}s")
    for stage in doc.get("stages") or []:
        if stage.get("status") == "failed":
            out.append(f"stage {stage['stage']} FAILED after "
                       f"{stage.get('seconds')}s: {str(stage.get('error'))[:150]}")
    f = doc.get("failure")
    if f:
        out.append(f"failure: {f.get('type')}: {str(f.get('message'))[:200]}")
    out.append(f"elapsed: {doc.get('elapsed_seconds')}s")
    return out


def _gib(v) -> str:
    return f"{v / 2**30:.3f}GiB" if isinstance(v, (int, float)) else "-"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: topk_adoption_report.py ADOPTION.json")
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"no record at {path}")
        return 1
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        print(f"{path} is not readable JSON: {exc}")
        return 1
    for line in lines(doc):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

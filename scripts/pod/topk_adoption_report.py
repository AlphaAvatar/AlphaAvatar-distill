#!/usr/bin/env python3
"""Read an adoption record and print what it found, one line per fact.

    python scripts/pod/topk_adoption_report.py ADOPTION.json

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

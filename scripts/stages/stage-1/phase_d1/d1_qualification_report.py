#!/usr/bin/env python3
"""Read a qualification record and print what it found, one line per fact.

    python scripts/pod/d1_qualification_report.py QUALIFICATION.json

Separate from the launcher because a shell heredoc that parses JSON is a thing
nobody can test without creating a pod. This runs at `$0` against any record,
including a failed one — which is the case that matters, since a failed
qualification's evidence is the point.
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
                   f"bf16={env.get('bf16_supported')} tf32={env.get('matmul_tf32')}")

    for name, key in (("incumbent", "bound_protocol_incumbent"),
                      ("target-aware", "bound_protocol_target_aware")):
        bound = doc.get(key) or {}
        if bound:
            out.append(f"protocol {name}: {bound['measurement_protocol_id'][:16]} "
                       f"policy={bound['position_policy']} "
                       f"suite_content={bound['suite_content_sha256'][:12]}")

    a = doc.get("A_incumbent") or {}
    if a:
        out.append(
            f"A incumbent: reconstructed={a.get('reconstructed')} "
            f"digest={(a.get('final_artifact_digest') or '')[:16]} "
            f"expected={(a.get('expected_final_artifact_digest') or '')[:16]} "
            f"{a.get('seconds')}s peak={_gib(a.get('peak_memory_bytes'))}")
    b = doc.get("B_target_aware") or {}
    if b:
        out.append(
            f"B target-aware: digest={(b.get('final_artifact_digest') or '')[:16]} "
            f"bsz={b.get('batch_size')} {b.get('seconds')}s "
            f"peak={_gib(b.get('peak_memory_bytes'))}")
    if a.get("seconds") and b.get("seconds"):
        out.append(f"E timing ratio target-aware/incumbent: "
                   f"{b['seconds'] / a['seconds']:.4f}")

    c = doc.get("C_selection_comparison") or {}
    if c:
        out.append(f"C decisions moved: {c.get('n_moved')} of "
                   f"{len(c.get('steps') or [])} steps; "
                   f"final artifacts differ={c.get('final_artifacts_differ')}")
        for step in c.get("steps") or []:
            if step.get("selection_differs") or step.get("artifact_digest_differs"):
                out.append(f"   {step['impl_id']}: selection_differs="
                           f"{step['selection_differs']} digest "
                           f"{step['incumbent_digest'][:12]} -> "
                           f"{step['target_aware_digest'][:12]}")

    for measurement in doc.get("D_state_eval") or []:
        out.append(f"D state-eval {measurement.get('label')}: "
                   f"peak={_gib(measurement.get('peak_memory_bytes'))} "
                   f"budget={_gib(measurement.get('derived_budget_bytes'))} "
                   f"within={measurement.get('within_derived_budget')} "
                   f"{measurement.get('seconds')}s")

    for stage in doc.get("stages") or []:
        if stage.get("status") == "failed":
            out.append(f"stage {stage['stage']} FAILED after "
                       f"{stage.get('seconds')}s: {str(stage.get('error'))[:140]}")
    failure = doc.get("failure")
    if failure:
        out.append(f"failure: {failure.get('type')}: "
                   f"{str(failure.get('message'))[:200]}")
    out.append(f"elapsed: {doc.get('elapsed_seconds')}s")
    return out


def _gib(value) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value / 2**30:.3f}GiB"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: d1_qualification_report.py QUALIFICATION.json")
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

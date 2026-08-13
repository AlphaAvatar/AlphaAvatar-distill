"""Score one unchanged checkpoint N times on the REAL frozen suite. GPU.

The question the beam's epsilon depends on: does re-measuring the same weights
move an objective? GPU reduction order is not deterministic, so this cannot be
answered on CPU, and it cannot be answered on a toy model either — the toy
version in `characterize_thresholds.py` establishes the *method*, this measures
the machine the search will actually run on.

The response to the result is frozen in advance (`conservative_review_gate@v1`):
below the declared epsilon it stands, at or above it nothing is re-derived and
Phase A is blocked pending review. No epsilon is computed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from load_state_eval import load  # noqa: E402

from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.metrics import StateEvaluator  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    #: Overridable ONLY so the script itself can be executed end to end in a
    #: test against tiny models. The gate always runs the real teacher; a run
    #: that overrode it records the override, and the repeatability number it
    #: produces is about whatever model it was pointed at.
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--teacher-revision", default=REVISION)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    suite, items, manifest = load(REPO / args.suite)
    adapter = get_adapter("qwen3")
    teacher_kwargs = ({"revision": args.teacher_revision}
                      if args.teacher == TEACHER else {})
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, **teacher_kwargs).to(args.device).eval()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16).to(args.device).eval()
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    spec = adapter.spec_from_config(cfg)
    identity = identify_checkpoint(Path(args.checkpoint), adapter=adapter, spec=spec,
                                   num_parameters=adapter.param_count(spec))

    evaluator = StateEvaluator(suite, items, device=args.device)
    evaluator.prime_reference(teacher)

    objectives = [o.key for o in PARETO_V1.objectives]
    runs = []
    for i in range(args.repeats):
        result = evaluator.evaluate(model, identity.artifact_digest)
        # `values`, not `metrics`: `StateEvaluation.as_dict()` has never had a
        # `metrics` key. This line had never executed against a real
        # `StateEvaluation` — the toy path in `characterize_thresholds.py` does
        # not go through it — and it cost a $0.29 pod on 2026-08-13, dying with
        # `KeyError: 'metrics'` after both models had loaded and a full
        # evaluation pass had completed. `tests/autoinit/test_ranking.py::
        # test_the_repeatability_probe_reads_the_evaluation_it_is_given` now
        # runs it on a fake model.
        row = result.as_dict()["values"]
        missing = [k for k in objectives if k not in row]
        if missing:
            raise SystemExit(
                f"the evaluation carries no {missing}; the beam ranks on those "
                "objectives, so a repeatability range over what is left would "
                "describe a different question than the one asked")
        runs.append({"repeat": i, **{k: row[k] for k in objectives}})
        print(f"  repeat {i}: " + " ".join(
            f"{k}={runs[-1].get(k)}" for k in objectives), flush=True)

    ranges = {}
    for key in objectives:
        values = [r[key] for r in runs if key in r]
        ranges[key] = {"min": min(values), "max": max(values),
                       "range": max(values) - min(values), "n": len(values)}
    out = {
        "schema": "aadistill.autoinit.evaluator_repeatability/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "artifact_digest": identity.artifact_digest,
        "suite": suite.qualified_id, "suite_hash": suite.suite_hash,
        "device": args.device, "repeats": args.repeats,
        "teacher": args.teacher,
        "is_real_teacher": args.teacher == TEACHER,
        "objectives": objectives, "runs": runs, "per_objective": ranges,
        "max_objective_range": max(v["range"] for v in ranges.values()),
        "declared_epsilon": min(PARETO_V1.epsilon.values()),
        "response_rule": ("conservative_review_gate@v1; the response is frozen "
                          "before this measurement and no epsilon is re-derived "
                          "from it"),
    }
    out["report_sha256"] = sha256_json(out)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"max_objective_range": out["max_objective_range"],
                      "declared_epsilon": out["declared_epsilon"]}, indent=2))


if __name__ == "__main__":
    main()

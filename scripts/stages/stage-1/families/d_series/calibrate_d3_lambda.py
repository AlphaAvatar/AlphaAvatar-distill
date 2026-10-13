#!/usr/bin/env python3
"""Measure D3's lambda by the registered rule, on the reference pair. GPU.

    PYTHONPATH=src:scripts:scripts/data python \
        scripts/stages/stage-1/families/d_series/calibrate_d3_lambda.py \
        --teacher <dir> --student <B dir> \
        --student-artifact-digest 53e30566… --device cuda --out <json>

Per item of the three registered item sets, with the teacher as reference and
B as student: the per-item mean over scored positions of the Top-200 + tail KL
and of B's own Top-200 confidence. Then `objectives.lambda_from_reference`.
Everything that decides the value is the registered rule
(`objectives.lambda_rule`); this file only computes its inputs.

Runs before D3's search and never after: the record it writes is what
`objectives.frozen_d3_objective` reads, and D3 has no objective until it
exists. AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/data", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.scoring.objective import (  # noqa: E402
    confidence_from_entropy, fixed_denominator_item_means, student_topk_terms,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    PREDICTION_AXIS,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    sketch_forward_kl_terms, sketch_reference,
)


def _ids(item: Any) -> torch.Tensor:
    return item["input_ids"] if isinstance(item, dict) else item.input_ids


def _item_id(item: Any) -> str:
    return str(item["item_id"] if isinstance(item, dict) else item.item_id)


@torch.no_grad()
def item_terms(teacher, student, items: Iterable[Any], *, policy,
               support_top_k: int, entropy_top_k: int,
               device: str) -> list[dict[str, Any]]:
    """Per item: scored-position means of the reference KL and of confidence."""
    out = []
    for item in items:
        ids = _ids(item).to(device)
        ref = teacher(ids).logits[0, :-1].float()
        cand = student(ids).logits[0, :-1]
        vocab = int(ref.shape[-1])
        sketch = sketch_reference(ref, torch.zeros(int(ref.shape[0]),
                                                   dtype=torch.long,
                                                   device=ref.device),
                                  top_k=int(support_top_k))
        kl, log_z = sketch_forward_kl_terms(
            sketch.support_indices, sketch.support_log_probs,
            sketch.tail_log_prob, cand, has_tail=int(support_top_k) < vocab)
        entropy, _ = student_topk_terms(cand, log_z, top_k=int(entropy_top_k))
        conf = confidence_from_entropy(entropy, int(entropy_top_k))
        weights = policy.weights(item, axis=PREDICTION_AXIS).weights.to(kl.device)
        mask = torch.ones(1, int(kl.shape[0]), dtype=torch.bool, device=kl.device)
        kl_mean, counts = fixed_denominator_item_means(kl[None], mask,
                                                       weights[None])
        c_mean, _ = fixed_denominator_item_means(conf[None], mask, weights[None])
        out.append({"item_id": _item_id(item), "kl": float(kl_mean[0]),
                    "confidence": float(c_mean[0]),
                    "scored_positions": int(counts[0])})
        del ref, cand
    return out


def calibrate(per_set: dict[str, list[dict[str, Any]]], *,
              identities: dict[str, Any]) -> dict[str, Any]:
    """The lambda record, from per-item terms. Pure."""
    from stages.d_series import objectives as O

    pooled = [r for rows in per_set.values() for r in rows]
    result = O.lambda_from_reference([r["kl"] for r in pooled],
                                     [r["confidence"] for r in pooled])
    sensitivity = {}
    for name, rows in per_set.items():
        try:
            sensitivity[name] = O.lambda_from_reference(
                [r["kl"] for r in rows], [r["confidence"] for r in rows])
        except ValueError as exc:
            sensitivity[name] = {"refused": str(exc)}
    rule = O.lambda_rule()
    body = {
        "schema": "aadistill.d_series.d3_lambda_calibration/v1",
        "rule_hash": rule["rule_hash"],
        "reward_scale": result["reward_scale"],
        "result": result,
        "per_set_sensitivity_not_used": sensitivity,
        "identities": identities,
        "per_item": per_set,
    }
    body["record_sha256"] = sha256_json(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-revision", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-artifact-digest", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.scoring.positions import get_position_policy
    from aadistill.initialization.specs.arch import get_adapter
    from shared.calibration import register_builtin_profiles
    from shared.evaluation import load_state_eval
    from stages.d_series import objectives as O
    from stages.d_series import scoring_protocol as SP

    register_builtin_adapters()
    register_builtin_profiles()
    policy = get_position_policy(O.LAMBDA_REFERENCE["positions"])
    adapter = get_adapter("qwen3")
    teacher = adapter.load(args.teacher, dtype="bfloat16", device=args.device)
    student = adapter.load(args.student, dtype="bfloat16", device=args.device)
    for m in (teacher, student):
        m.config.use_cache = False

    started = time.time()
    per_set: dict[str, list[dict[str, Any]]] = {}
    for qualified in O.LAMBDA_REFERENCE["item_sets"]:
        if qualified.startswith("calib."):
            profile = get_profile(qualified)
            items = prepare_calibration_items(profile.resolve(str(REPO_ROOT)),
                                              profile_id=profile.qualified_id)
        else:
            loaded = load_state_eval.load(
                REPO_ROOT / "artifacts/stages/stage-1/state_eval_v1")
            items = loaded[1]
        per_set[qualified] = item_terms(
            teacher, student, items, policy=policy,
            support_top_k=SP.D_SERIES_TOP_K, entropy_top_k=O.ENTROPY_TOP_K,
            device=args.device)
    record = calibrate(per_set, identities={
        "teacher_revision": args.teacher_revision,
        "student_artifact_digest": args.student_artifact_digest,
        "support_top_k": SP.D_SERIES_TOP_K,
        "entropy_top_k": O.ENTROPY_TOP_K,
        "device": args.device, "dtype": "bfloat16",
        "seconds": round(time.time() - started, 1)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"lambda = {record['reward_scale']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

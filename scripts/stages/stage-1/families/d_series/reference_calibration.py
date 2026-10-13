#!/usr/bin/env python3
"""The B-vs-teacher reference calibration, BEFORE D2's formal search. GPU.

    PYTHONPATH=src:scripts:scripts/data python \
        scripts/stages/stage-1/families/d_series/reference_calibration.py \
        --teacher <dir> --teacher-revision 768f209d… --student <B dir> \
        --student-artifact-digest 53e30566… --device cuda \
        --rescoring-timing 24 --out <json>

One pass over the registered item sets, teacher as reference and the standing
incumbent B as student, produces three things that must exist before D2's
formal search and are never revised by any D2/D3 result:

1. D3's lambda, by `objectives.lambda_rule` (v2): per UNIQUE item, the
   scored-position means of the Top-200 + tail KL and of B's own Top-200
   confidence; lambda = their IQR ratio. 209 draws are 147 unique items.
2. The diagnostic thresholds, by `objectives.diagnostic_threshold_rule`:
   per-position quantiles over the same unique items' scored positions.
3. A MEASURED cost of the rollout rescoring diagnostic on the real teacher and
   a real 0.6B student, over sequences of D1's measured lengths -- timing only,
   B's initialization being a model of the right size, not the evaluated one.

Everything that decides a value is a registered rule; this file computes the
inputs. AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

#: The rescoring's measured lengths: prompt + generated tokens per probe at
#: D1's confirmation rung (logs/.../analyses/d1_rollout_profile.json basis).
#: Used only to PROJECT a measured per-token rate to a probe.
D1_TOKENS_PER_PROBE = 951268

#: The pool lambda rule v2 is registered over (its `pooling` text). A staged
#: mixture that yields a different pool is not the registered reference, and a
#: session refuses it before any model is resident.
REGISTERED_POOL = {"draws": 209, "unique": 147}

#: Items the rescoring timing runs on -- the plan's 1-5 min component.
RESCORING_TIMING_ITEMS = 24


def assert_registered_pool(report: Mapping[str, Any]) -> None:
    got = {"draws": int(report["draws"]), "unique": int(report["unique"])}
    if got != REGISTERED_POOL:
        raise ValueError(
            f"the item sets resolve to {got}, not the registered "
            f"{REGISTERED_POOL}; lambda and the thresholds would be computed "
            "over a different reference than the one fixed before D2")


def _ids(item: Any) -> torch.Tensor:
    return item["input_ids"] if isinstance(item, dict) else item.input_ids


def _item_id(item: Any) -> str:
    return str(item["item_id"] if isinstance(item, dict) else item.item_id)


def content_key(item: Any) -> str:
    """Content identity: sha256 of the token ids. Two draws of one item agree."""
    return hashlib.sha256(_ids(item).reshape(-1).to(torch.int64).numpy()
                          .tobytes()).hexdigest()


def unique_items(item_sets: Mapping[str, Sequence[Any]]
                 ) -> tuple[list[tuple[str, Any]], list[dict[str, str]],
                            dict[str, Any]]:
    """``(unique, draws, report)``: each item once, first set that drew it."""
    seen: dict[str, str] = {}
    unique: list[tuple[str, Any]] = []
    draws: list[dict[str, str]] = []
    per_set: dict[str, dict[str, int]] = {}
    for name, items in item_sets.items():
        keys = [content_key(i) for i in items]
        new = 0
        for item, key in zip(items, keys):
            draws.append({"set": name, "item_id": _item_id(item), "content": key})
            if key not in seen:
                seen[key] = name
                unique.append((name, item))
                new += 1
        per_set[name] = {"draws": len(items), "unique_within": len(set(keys)),
                         "new_unique": new}
    return unique, draws, {"draws": len(draws), "unique": len(unique),
                           "per_set": per_set}


@torch.no_grad()
def item_terms(teacher, student, items: Iterable[tuple[str, Any]], *, policy,
               support_top_k: int, entropy_top_k: int, device: str
               ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Per unique item: scored-position means; and every scored position."""
    rows = []
    pooled: dict[str, list[torch.Tensor]] = {"kl": [], "confidence": [],
                                             "topk_mass": []}
    for set_name, item in items:
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
        entropy, mass = student_topk_terms(cand, log_z,
                                           top_k=int(entropy_top_k))
        conf = confidence_from_entropy(entropy, int(entropy_top_k))
        weights = policy.weights(item, axis=PREDICTION_AXIS).weights.to(kl.device)
        mask = torch.ones(1, int(kl.shape[0]), dtype=torch.bool, device=kl.device)
        kl_mean, counts = fixed_denominator_item_means(kl[None], mask,
                                                       weights[None])
        c_mean, _ = fixed_denominator_item_means(conf[None], mask, weights[None])
        scored = weights > 0
        pooled["kl"].append(kl[scored].float().cpu())
        pooled["confidence"].append(conf[scored].float().cpu())
        pooled["topk_mass"].append(mass[scored].float().cpu())
        rows.append({"set": set_name, "item_id": _item_id(item),
                     "content": content_key(item), "kl": float(kl_mean[0]),
                     "confidence": float(c_mean[0]),
                     "scored_positions": int(counts[0])})
        del ref, cand
    return rows, {k: torch.cat(v) if v else torch.empty(0)
                  for k, v in pooled.items()}


def _quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.double(), torch.tensor(q, dtype=torch.float64)))


def calibrate(rows: Sequence[Mapping[str, Any]], draws: Sequence[Mapping[str, str]],
              positions: Mapping[str, torch.Tensor], *,
              identities: Mapping[str, Any],
              timing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The reference-calibration record. Pure, given the measured inputs."""
    from stages.d_series import objectives as O

    result = O.lambda_from_reference([r["kl"] for r in rows],
                                     [r["confidence"] for r in rows])
    by_content = {r["content"]: r for r in rows}
    sensitivity: dict[str, Any] = {}
    try:
        expanded = [by_content[d["content"]] for d in draws]
        sensitivity["with_duplicate_draws"] = O.lambda_from_reference(
            [r["kl"] for r in expanded], [r["confidence"] for r in expanded])
    except (KeyError, ValueError) as exc:
        sensitivity["with_duplicate_draws"] = {"refused": str(exc)}
    for name in sorted({d["set"] for d in draws}):
        keys = list(dict.fromkeys(d["content"] for d in draws
                                  if d["set"] == name))
        subset = [by_content[k] for k in keys if k in by_content]
        try:
            sensitivity[f"set:{name}"] = O.lambda_from_reference(
                [r["kl"] for r in subset], [r["confidence"] for r in subset])
        except ValueError as exc:
            sensitivity[f"set:{name}"] = {"refused": str(exc)}
    thresholds = {
        "high_kl": _quantile(positions["kl"], 0.90),
        "high_confidence": _quantile(positions["confidence"], 0.90),
        "low_topk_mass": _quantile(positions["topk_mass"], 0.10),
    }
    body = {
        "schema": "aadistill.d_series.reference_calibration/v1",
        "rule_hash": O.lambda_rule()["rule_hash"],
        "threshold_rule_hash": O.diagnostic_threshold_rule()["rule_hash"],
        "reward_scale": result["reward_scale"],
        "lambda": result,
        "lambda_sensitivity_not_used": sensitivity,
        "diagnostic_thresholds": thresholds,
        "independence": {"draws": len(draws), "unique_items": len(rows),
                         "scored_positions": int(positions["kl"].numel())},
        "rescoring_cost_measured": dict(timing or {}),
        "identities": dict(identities),
        "per_unique_item": list(rows),
    }
    body["record_sha256"] = sha256_json(body)
    return body


@torch.no_grad()
def time_rescoring(teacher, student, items: Sequence[tuple[str, Any]], *,
                   n: int, support_top_k: int, entropy_top_k: int,
                   prompt_fraction: float = 0.4) -> dict[str, Any]:
    """Wall time of the rollout rescoring on real models, per token.

    The first ``prompt_fraction`` of each item stands in for a prompt and the
    rest for a completion -- D1's confirmation rollouts average 398 prompt and
    603 generated tokens. Timing only: nothing here is a measurement of a
    student's behaviour.
    """
    from shared.evaluation.rollout_rescoring import rescore_trajectory

    device = next(student.parameters()).device
    tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _, item in list(items)[:n]:
        ids = _ids(item)
        cut = max(1, int(ids.shape[1] * prompt_fraction))
        rescore_trajectory(student, teacher, ids[:, :cut], ids[:, cut:],
                           support_top_k=support_top_k,
                           entropy_top_k=entropy_top_k)
        tokens += int(ids.shape[1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    per_token = seconds / max(tokens, 1)
    return {"sequences": min(n, len(items)), "tokens": tokens,
            "seconds": round(seconds, 3),
            "seconds_per_1k_tokens": round(1000 * per_token, 4),
            "projected_minutes_per_probe": round(
                per_token * D1_TOKENS_PER_PROBE / 60, 2),
            "tokens_per_probe_basis": D1_TOKENS_PER_PROBE,
            "peak_memory_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                if device.type == "cuda" else None),
            "_timing_only": ("B's initialization stands in for a trained "
                             "student of the same size; no behavioural "
                             "quantity is measured here")}


def load_item_sets(repo_root: str | Path = REPO_ROOT) -> dict[str, list[Any]]:
    """The registered item sets, by name. Profiles must be registered first."""
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from shared.evaluation import load_state_eval
    from stages.d_series import objectives as O

    repo_root = Path(repo_root)
    item_sets: dict[str, list[Any]] = {}
    for qualified in O.LAMBDA_REFERENCE["item_sets"]:
        if qualified.startswith("calib."):
            profile = get_profile(qualified)
            item_sets[qualified] = list(prepare_calibration_items(
                profile.resolve(str(repo_root)), profile_id=profile.qualified_id))
        else:
            item_sets[qualified] = list(load_state_eval.load(
                repo_root / "artifacts/stages/stage-1/state_eval_v1")[1])
    return item_sets


def run(teacher, student, item_sets: Mapping[str, Sequence[Any]], *,
        device: str, rescoring_timing: int,
        identities: Mapping[str, Any]) -> dict[str, Any]:
    """The whole calibration on two loaded models. Returns the record."""
    from aadistill.initialization.scoring.positions import get_position_policy
    from stages.d_series import objectives as O
    from stages.d_series import scoring_protocol as SP

    unique, draws, report = unique_items(item_sets)
    policy = get_position_policy(O.LAMBDA_REFERENCE["positions"])
    started = time.time()
    rows, positions = item_terms(
        teacher, student, unique, policy=policy,
        support_top_k=SP.D_SERIES_TOP_K, entropy_top_k=O.ENTROPY_TOP_K,
        device=device)
    timing = (time_rescoring(teacher, student, unique, n=rescoring_timing,
                             support_top_k=SP.D_SERIES_TOP_K,
                             entropy_top_k=O.ENTROPY_TOP_K)
              if rescoring_timing > 0 else None)
    return calibrate(rows, draws, positions, timing=timing, identities={
        **dict(identities),
        "support_top_k": SP.D_SERIES_TOP_K, "entropy_top_k": O.ENTROPY_TOP_K,
        "device": device, "dtype": "bfloat16", "item_sets": report,
        "seconds": round(time.time() - started, 1)})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-revision", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-artifact-digest", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rescoring-timing", type=int, default=0,
                    help="time the rescoring diagnostic on this many items")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    from shared.calibration import register_builtin_profiles

    register_builtin_adapters()
    register_builtin_profiles()
    item_sets = load_item_sets(REPO_ROOT)
    adapter = get_adapter("qwen3")
    teacher = adapter.load(args.teacher, dtype="bfloat16", device=args.device)
    student = adapter.load(args.student, dtype="bfloat16", device=args.device)
    for m in (teacher, student):
        m.config.use_cache = False
    record = run(teacher, student, item_sets, device=args.device,
                 rescoring_timing=args.rescoring_timing, identities={
                     "teacher_revision": args.teacher_revision,
                     "student_artifact_digest": args.student_artifact_digest})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"lambda = {record['reward_scale']}; thresholds "
          f"{record['diagnostic_thresholds']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

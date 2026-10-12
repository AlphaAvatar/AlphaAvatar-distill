#!/usr/bin/env python3
"""How many prediction rows a position policy actually scores, at $0.

    PYTHONPATH=src:scripts python scripts/shared/data/scored_position_profile.py \
        --profile calib.domain_balanced@v1 --profile calib.reasoning_heavy@v2 \
        --state-eval artifacts/stages/stage-1/state_eval_v1 \
        --corpus-sessions artifacts/stages/stage-3/corpus_v2/sessions.jsonl \
        --out <json>

Answers the two questions a pre-vocabulary row selection is priced by, from the
frozen bytes rather than from an estimate:

* the fraction of prediction positions the policy weights, per mixture and per
  domain -- the rows `ExecutionConfig.prediction_rows="scored_v1"` projects;
* the fraction of FORWARD SLOTS those are once items are grouped and right
  padded at a batch size -- what the all-rows path projects instead.

And, for a teacher corpus, how much of each slice's real session-length
distribution a calibration builder's token window admits -- the coverage
question a teacher-native Stage-0 mixture has to answer.

Reads nothing on a pod and loads no model. Everything it names comes from the
arguments; nothing about a particular experiment is hardcoded here.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))


def _quantile(sorted_values: list[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def profile_items(items, policy, *, batch_size: int, packing: str,
                  label: str) -> dict:
    """Scored rows vs real positions vs padded forward slots for one item list."""
    from aadistill.initialization.calibration.packing import packed_batches
    from aadistill.initialization.scoring.positions import PREDICTION_AXIS

    positions = scored = 0
    by_domain: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    tag_totals: collections.Counter = collections.Counter()
    for item in items:
        w = policy.weights(item, axis=PREDICTION_AXIS)
        positions += w.n_positions
        scored += w.n_active
        domain = (item.get("domain") if isinstance(item, dict)
                  else getattr(item, "domain", None))
        by_domain[str(domain)][0] += w.n_positions
        by_domain[str(domain)][1] += w.n_active
        tags = (item.get("tags") if isinstance(item, dict)
                else getattr(item, "tags", None)) or {}
        for name, value in tags.items():
            tag_totals[name] += (int(value.sum()) if hasattr(value, "sum")
                                 else len(value))
    shim = [{"input_ids": (i["input_ids"] if isinstance(i, dict) else i.input_ids)}
            for i in items]
    slots = 0
    for packed in packed_batches(shim, batch_size, packing=packing, pad_id=0,
                                 device="cpu"):
        ids = packed.batch.input_ids
        slots += int(ids.shape[0]) * (int(ids.shape[1]) - 1)
    return {
        "label": label,
        "items": len(items),
        "prediction_positions": positions,
        "scored_positions": scored,
        "scored_fraction_of_positions": round(scored / positions, 4),
        "forward_slots_at_batch": {"batch_size": batch_size, "packing": packing,
                                   "slots": slots,
                                   "scored_fraction_of_slots":
                                       round(scored / slots, 4)},
        "by_domain": {d: {"positions": p, "scored": s,
                          "scored_fraction": round(s / p, 4)}
                      for d, (p, s) in sorted(by_domain.items())},
        "tag_positions": dict(sorted(tag_totals.items())),
    }


def corpus_window_coverage(sessions_path: Path, window: tuple[int, int],
                           *, type_key: str, length_key: str) -> dict:
    """Per slice: the session-length distribution and the share a window admits."""
    lengths: dict[str, list[int]] = collections.defaultdict(list)
    with sessions_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            lengths[str(row.get(type_key))].append(int(row[length_key]))
    lo, hi = window
    out = {}
    for kind, values in sorted(lengths.items()):
        values.sort()
        out[kind] = {"sessions": len(values),
                     "median_tokens": _quantile(values, 0.5),
                     "p90_tokens": _quantile(values, 0.9),
                     "max_tokens": values[-1],
                     "admitted_by_window": round(
                         sum(lo <= v <= hi for v in values) / len(values), 4)}
    return {"sessions_path": str(sessions_path), "window_tokens": [lo, hi],
            "by_type": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", action="append", default=[],
                    help="a registered calibration profile's qualified id")
    ap.add_argument("--state-eval", default=None,
                    help="a state-eval asset directory")
    ap.add_argument("--policy", default="positions.supervised_target_v1@v1")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--packing", default="length_sorted_v1")
    ap.add_argument("--corpus-sessions", default=None)
    ap.add_argument("--window", type=int, nargs=2, default=(256, 2048))
    ap.add_argument("--type-key", default="data_type")
    ap.add_argument("--length-key", default="n_rendered_tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from aadistill.initialization.calibration.items import prepare_calibration_items
    from aadistill.initialization.scoring.positions import get_position_policy

    policy = get_position_policy(args.policy)
    record = {"schema": "aadistill.analysis.scored_position_profile/v1",
              "policy": policy.qualified_id, "policy_hash": policy.policy_hash,
              "mixtures": [], "state_eval": None, "corpus": None}
    if args.profile:
        from shared import calibration as C  # registers the declared mixtures
        from aadistill.initialization.calibration.profiles import get_profile
        for qualified in args.profile:
            prof = get_profile(qualified)
            items = prepare_calibration_items(prof.resolve(str(REPO)),
                                              profile_id=prof.qualified_id)
            record["mixtures"].append(profile_items(
                items, policy, batch_size=args.batch_size, packing=args.packing,
                label=prof.qualified_id))
        del C
    if args.state_eval:
        from shared.evaluation import load_state_eval
        loaded = load_state_eval.load(REPO / args.state_eval)
        suite, items = loaded[0], loaded[1]
        record["state_eval"] = profile_items(
            items, policy, batch_size=args.batch_size, packing=args.packing,
            label=suite.qualified_id)
    if args.corpus_sessions:
        record["corpus"] = corpus_window_coverage(
            REPO / args.corpus_sessions, tuple(args.window),
            type_key=args.type_key, length_key=args.length_key)
        #: As given, not resolved: a record names repository paths, not hosts.
        record["corpus"]["sessions_path"] = args.corpus_sessions

    body = json.dumps(record, indent=1) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body)
        print(f"wrote {args.out}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

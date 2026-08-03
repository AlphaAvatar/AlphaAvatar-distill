#!/usr/bin/env python
"""Corpus audit for the Experiment 2 data-cleaning arm (D0 control vs D1 treatment).

Compares the rung a trained control actually consumed against the rung a cleaned
corpus would supply, on every quantity that has to match for "target quality" to
be the only variable: prompt overlap, per-type supervised-token shares, packed
tokens, block count, optimizer steps and effective epochs.

Both rungs are read from their packs' `audit.jsonl`, which records, per block,
every session placed in it and the supervised tokens that survived packing — so
the audit measures the data as packed, not as intended.

Usage:
    scripts/data/audit_d1_corpus.py \
        --d0-ladder artifacts/stage3/ladder_uniform_probe --d0-rung 2960000 \
        --d1-ladder artifacts/stage3/ladder_uniform_clean --d1-rung 2992616 \
        --cleaning artifacts/stage3/corpus_v2_clean \
        --blocks-per-step 2 --epochs 3 \
        --out artifacts/stage3/e2_d1_corpus_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402


def read_rung(ladder_dir: Path, target: int) -> dict:
    """Sessions, per-type tokens and packing facts for one rung of one pack."""
    ladder = json.loads((ladder_dir / "ladder.json").read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == target)
    n_blocks = rung["n_blocks"]

    sessions, per_type, per_type_sessions = {}, Counter(), Counter()
    truncated = discarded = 0
    lengths = defaultdict(list)
    with (ladder_dir / "audit.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= n_blocks:
                break
            for s in json.loads(line)["sessions"]:
                sessions[s["session_id"]] = {
                    "data_type": s["data_type"],
                    "candidate_index": s["candidate_index"],
                    "candidate_sha256": s["candidate_sha256"],
                    "supervised": s["supervised_retained"],
                    "truncated": s["truncated"],
                }
                per_type[s["data_type"]] += s["supervised_retained"]
                per_type_sessions[s["data_type"]] += 1
                lengths[s["data_type"]].append(s["supervised_retained"])
                truncated += int(s["truncated"])
                discarded += s["supervised_discarded"]

    total = sum(per_type.values())
    return {
        "ladder_dir": str(ladder_dir),
        "rung_target": target,
        "n_blocks": n_blocks,
        "n_sessions": len(sessions),
        "supervised_tokens": total,
        "packed_tokens": n_blocks * ladder["block_len"],
        "block_len": ladder["block_len"],
        "terminal_truncations": truncated,
        "supervised_discarded_by_truncation": discarded,
        "per_type_tokens": dict(sorted(per_type.items())),
        "per_type_shares": {t: round(v / total, 4) for t, v in sorted(per_type.items())},
        "per_type_sessions": dict(sorted(per_type_sessions.items())),
        "_sessions": sessions,
        "_lengths": {t: sorted(v) for t, v in lengths.items()},
        "_hashes": {
            "ladder_json": sha256_file(ladder_dir / "ladder.json"),
            "blocks_npz": sha256_file(ladder_dir / "blocks.npz"),
            "audit_jsonl": sha256_file(ladder_dir / "audit.jsonl"),
        },
    }


def dist(values):
    if not values:
        return None
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "p50": values[len(values) // 2], "p90": values[int(len(values) * 0.9)],
            "p99": values[int(len(values) * 0.99)], "max": values[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0-ladder", required=True, type=Path)
    ap.add_argument("--d0-rung", required=True, type=int)
    ap.add_argument("--d1-ladder", required=True, type=Path)
    ap.add_argument("--d1-rung", required=True, type=int)
    ap.add_argument("--cleaning", required=True, type=Path,
                    help="output dir of build_cleaned_corpus.py")
    ap.add_argument("--blocks-per-step", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d0 = read_rung(args.d0_ladder, args.d0_rung)
    d1 = read_rung(args.d1_ladder, args.d1_rung)

    d0_ids, d1_ids = set(d0["_sessions"]), set(d1["_sessions"])
    shared = d0_ids & d1_ids
    dropped, added = d0_ids - d1_ids, d1_ids - d0_ids

    # Within the shared prompts, did the cleaned corpus keep the same completion?
    same_candidate = sum(
        1 for i in shared
        if d0["_sessions"][i]["candidate_sha256"] == d1["_sessions"][i]["candidate_sha256"])

    dropped_by_type, added_by_type = Counter(), Counter()
    for i in dropped:
        dropped_by_type[d0["_sessions"][i]["data_type"]] += 1
    for i in added:
        added_by_type[d1["_sessions"][i]["data_type"]] += 1

    cleaning = json.loads((args.cleaning / "cleaning_audit.json").read_text())

    def steps(n_blocks):
        return math.ceil(n_blocks * args.epochs / args.blocks_per_step)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "rules_version": cleaning["rules_version"],
        "code_state": code_state(REPO_ROOT),
        "arms": {
            "D0": {k: v for k, v in d0.items() if not k.startswith("_")},
            "D1": {k: v for k, v in d1.items() if not k.startswith("_")},
        },
        "hashes": {"D0": d0["_hashes"], "D1": d1["_hashes"]},
        "prompt_overlap": {
            "d0_sessions": len(d0_ids),
            "d1_sessions": len(d1_ids),
            "shared": len(shared),
            "overlap_of_d0": round(len(shared) / len(d0_ids), 4),
            "overlap_of_d1": round(len(shared) / len(d1_ids), 4),
            "jaccard": round(len(shared) / len(d0_ids | d1_ids), 4),
            "dropped_from_d0": len(dropped),
            "added_to_d1": len(added),
            "dropped_by_type": dict(sorted(dropped_by_type.items())),
            "added_by_type": dict(sorted(added_by_type.items())),
        },
        "candidate_replacement": {
            "shared_prompts": len(shared),
            "same_completion": same_candidate,
            "replaced_completion": len(shared) - same_candidate,
            "replacement_rate_on_shared": round(
                (len(shared) - same_candidate) / len(shared), 4) if shared else 0.0,
            "corpus_wide": {
                t: {"kept": r["kept"], "replaced": r["replaced"],
                    "replacement_rate": r["replacement_rate"],
                    "no_valid_candidate": r["no_valid_candidate"],
                    "unverifiable_slice": r["unverifiable_slice"]}
                for t, r in cleaning["per_type"].items()},
        },
        "rejection_counts_by_reason": {
            t: r["candidate_reasons"] for t, r in cleaning["per_type"].items()},
        "target_length_distribution_supervised_tokens": {
            "D0": {t: dist(v) for t, v in sorted(d0["_lengths"].items())},
            "D1": {t: dist(v) for t, v in sorted(d1["_lengths"].items())},
        },
        "training_budget": {
            "D0": {"blocks": d0["n_blocks"], "epochs": args.epochs,
                   "blocks_per_step": args.blocks_per_step,
                   "optimizer_steps": steps(d0["n_blocks"]),
                   "packed_tokens_per_epoch": d0["packed_tokens"],
                   "tokens_processed": steps(d0["n_blocks"]) * args.blocks_per_step
                                       * d0["block_len"]},
            "D1": {"blocks": d1["n_blocks"], "epochs": args.epochs,
                   "blocks_per_step": args.blocks_per_step,
                   "optimizer_steps": steps(d1["n_blocks"]),
                   "packed_tokens_per_epoch": d1["packed_tokens"],
                   "tokens_processed": steps(d1["n_blocks"]) * args.blocks_per_step
                                       * d1["block_len"]},
        },
        "kd_target_alignment": {
            "teacher_distribution_source": "online",
            "evidence": ("aadistill.training.train.Trainer._micro_losses runs the "
                         "teacher forward on the same packed block as the student "
                         "(train.py:493); no cached or stored logits exist anywhere "
                         "in the repository"),
            "implication": ("a replaced completion is automatically paired with the "
                            "teacher distribution of its own teacher-forced prefix, "
                            "system message and serialization; there is no cached "
                            "logit that could belong to a different candidate"),
            "recomputation_cost": "none — no logit cache to rebuild",
        },
    }

    residual = {}
    for key, a, b in (
        ("supervised_tokens", d0["supervised_tokens"], d1["supervised_tokens"]),
        ("n_blocks", d0["n_blocks"], d1["n_blocks"]),
        ("packed_tokens", d0["packed_tokens"], d1["packed_tokens"]),
        ("n_sessions", d0["n_sessions"], d1["n_sessions"]),
        ("optimizer_steps", steps(d0["n_blocks"]), steps(d1["n_blocks"])),
    ):
        residual[key] = {"D0": a, "D1": b, "delta": b - a,
                         "rel": round((b - a) / a, 6) if a else None}
    residual["per_type_share_pp"] = {
        t: round((d1["per_type_shares"].get(t, 0.0)
                  - d0["per_type_shares"].get(t, 0.0)) * 100, 3)
        for t in sorted(set(d0["per_type_shares"]) | set(d1["per_type_shares"]))}
    audit["residual_mismatch"] = residual

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=1))

    print(f"D0 {d0['n_blocks']:>5} blocks  {d0['supervised_tokens']:>10,} sup  "
          f"{d0['n_sessions']:>5} sessions  {steps(d0['n_blocks']):>5} steps")
    print(f"D1 {d1['n_blocks']:>5} blocks  {d1['supervised_tokens']:>10,} sup  "
          f"{d1['n_sessions']:>5} sessions  {steps(d1['n_blocks']):>5} steps")
    print(f"\noverlap: {len(shared)}/{len(d0_ids)} of D0 "
          f"({len(shared)/len(d0_ids):.1%}), jaccard "
          f"{len(shared)/len(d0_ids|d1_ids):.1%}; "
          f"replaced completion on {len(shared)-same_candidate} shared prompts")
    print("\nresidual mismatch:")
    for key, row in residual.items():
        if key == "per_type_share_pp":
            print(f"  per-type share drift (pp): {row}")
        else:
            print(f"  {key:18s} {row['D0']:>12,} -> {row['D1']:>12,} "
                  f"({row['rel']:+.4%})")


if __name__ == "__main__":
    main()

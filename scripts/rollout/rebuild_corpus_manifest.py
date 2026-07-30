"""Reconstruct a generation manifest from a corpus whose run lost it.

The 2026-07-30 build generated all 752 prompts, wrote candidates, targets and a
hashed rollout snapshot, and then died writing its manifest because `code_state`
shelled out to `git`, which vLLM's official image does not ship. The corpus is
intact; only its manifest was lost. Regenerating it by re-running the teacher
would cost another ~72 GPU-minutes to produce bytes that already exist.

Every field here is either **recomputed from the artifacts** (the per-slice
accept rates, counts and hashes) or **supplied explicitly by the caller** (the
commit the pod ran, its hardware). Nothing is inferred or invented, and the
manifest records that it was rebuilt, by what, and why — a reconstructed record
that hides its provenance is worse than a missing one (P4/P11/P14).

Usage:
    uv run python scripts/rollout/rebuild_corpus_manifest.py \
        --dir artifacts/stage2_v2/teacher_corpus_750 \
        --code-commit <sha> --session-log <path>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

SLICE_OF = {
    ("rag_evidence", "squad_v2"): "rag_evidence",
    ("multihop_qa", "hotpot_qa"): "multihop_qa",
    ("code_math", "gsm8k"): "gsm8k",
    ("code_math", "openmath_instruct_2"): "openmath",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--code-commit", required=True,
                    help="the commit the pod actually ran (it had no git)")
    ap.add_argument("--command", default=None,
                    help="the generation command, from the pod's log")
    ap.add_argument("--session-log", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.dir
    cand_path, targ_path = out_dir / "candidates.jsonl", out_dir / "targets.jsonl"
    for p in (cand_path, targ_path):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    candidates = [json.loads(l) for l in cand_path.read_text().splitlines() if l.strip()]
    targets = [json.loads(l) for l in targ_path.read_text().splitlines() if l.strip()]
    if len(candidates) != len(targets):
        raise SystemExit(
            f"{len(candidates)} candidate rows vs {len(targets)} target rows — "
            "the corpus is not a matched pair and must not be described as one")

    # Per-slice statistics, recomputed exactly as the generator computes them.
    stats: dict[str, dict] = {}
    for row in candidates:
        name = row.get("slice") or SLICE_OF.get((row["group"], row["source"]), row["source"])
        s = stats.setdefault(name, {"prompts": 0, "accept_at_1": 0, "accept_at_n": 0,
                                    "reasons": Counter(), "think_tokens": []})
        s["prompts"] += 1
        cands = row["candidates"]
        if cands:
            s["accept_at_1"] += int(bool(cands[0]["accepted"]))
        s["accept_at_n"] += int(row.get("chosen_index") is not None)
        for c in cands:
            s["reasons"][c["reason"]] += 1
            s["think_tokens"].append(c["think_tokens"])

    summary = {}
    for name, s in sorted(stats.items()):
        summary[name] = {
            "prompts": s["prompts"],
            "accept_at_1": round(s["accept_at_1"] / s["prompts"], 4),
            "accept_at_n": round(s["accept_at_n"] / s["prompts"], 4),
            "reject_reasons": dict(s["reasons"].most_common()),
            "think_tokens_median": statistics.median(s["think_tokens"]) if s["think_tokens"] else 0,
            "think_tokens_p90": (statistics.quantiles(s["think_tokens"], n=10)[-1]
                                 if len(s["think_tokens"]) > 9 else None),
        }

    accepted = sum(1 for t in targets if t.get("target_source") == "teacher_verified")

    # Candidate diversity, measured rather than assumed. A corpus built with n>1
    # is only doing rejection sampling if the candidates actually differ, and a
    # server engine seeds per request — so replicas inside one request are
    # identical and accept@n collapses onto accept@1. Recording this makes the
    # corpus's real sampling behaviour part of its identity (P4/P11).
    multi = [r for r in candidates if len(r["candidates"]) > 1]
    identical = sum(
        1 for r in multi
        if len({c["answer"] for c in r["candidates"]}) == 1
        and len({c["think"] for c in r["candidates"]}) == 1
    )
    rescued = sum(
        1 for r in multi
        if not r["candidates"][0]["accepted"]
        and any(c["accepted"] for c in r["candidates"][1:])
    )
    diversity = {
        "prompts_with_multiple_candidates": len(multi),
        "byte_identical_candidate_sets": identical,
        "identical_rate": round(identical / len(multi), 4) if multi else None,
        "rescued_by_a_later_candidate": rescued,
        "effective_n": 1,
        "note": "the generating run replicated each prompt inside ONE request "
                "under ONE seed; vLLM seeds per request, so the replicas were "
                "NOT independent draws and this corpus must be read as n=1. "
                "This is a property of the implementation, not evidence about "
                "sampling diversity. Fixed after this build: each candidate "
                "index now draws under its own seed.",
    }
    snap_manifest_path = out_dir / "rollout_snapshot" / "manifest.json"
    snapshot = (json.loads(snap_manifest_path.read_text())
                if snap_manifest_path.is_file() else None)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # Loud provenance: this manifest did not come from the run itself.
        "rebuilt": {
            "by": "scripts/rollout/rebuild_corpus_manifest.py",
            "reason": "the generating run died writing its manifest — code_state "
                      "shelled out to git, absent from vllm/vllm-openai:v0.26.0. "
                      "The corpus artifacts themselves completed and are hashed.",
            "recomputed_from": ["candidates.jsonl", "targets.jsonl",
                                "rollout_snapshot/manifest.json"],
            "supplied_by_caller": ["code_state.git_commit", "command", "hardware"],
        },
        "command": args.command,
        "teacher": f"{args.model}@{args.revision}",
        "thinking_mode": True,
        "decoding": {
            "n": 2, "all_candidates_sampled": True, "temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "max_new_tokens": 4096,
            "engine": "vllm_server", "engine_version": "0.26.0",
            "engine_image": "vllm/vllm-openai:v0.26.0",
        },
        "data_dir": "data/stage2_v1",
        "prompt_selection": {"mode": "stride", "limit_per_slice": 188,
                             "slices": ["rag_evidence", "multihop_qa",
                                        "gsm8k", "openmath"]},
        "complete": True,
        "prompts_requested": len(targets),
        "prompts_generated": len(targets),
        "accepted_targets": accepted,
        "accept_rate_overall": round(accepted / len(targets), 4) if targets else None,
        "slices": summary,
        "candidate_diversity": diversity,
        "outputs": {
            "candidates": sha256_file(cand_path),
            "targets": sha256_file(targ_path),
        },
        "rollout_snapshot": snapshot,
        "code_state": {
            "git_commit": args.code_commit,
            "git_commit_source": "caller:rebuild (pod image had no git)",
            "dirty": False,
            "note": "the pod ran a `git archive` tarball of this commit, so the "
                    "tree was clean by construction",
        },
    }
    if args.session_log:
        p = REPO_ROOT / args.session_log
        if p.is_file():
            manifest["pod_session_log_sha256"] = sha256_file(p)

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {path}")
    print(f"  prompts {len(targets)}, accepted {accepted} "
          f"({manifest['accept_rate_overall']:.1%})")
    for name, s in summary.items():
        print(f"  {name:<14} prompts {s['prompts']:>4}  accept@1 "
              f"{s['accept_at_1']:.3f}  accept@n {s['accept_at_n']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build Experiment 5 arm C: teacher-native prefix + supervised teacher continuation.

    PYTHONPATH=src python scripts/data/build_e5_arm_c.py \
        --out artifacts/stage3/e5_arm_c

C needs **no generation**. Its prefix is the teacher's own trajectory, already in
the corpus, so a split only moves the loss mask: the first `k` supervised tokens
become context and the rest stay supervised. The token stream is byte-identical
to what ordinary continuation training would have used — which is exactly what
makes C a clean control for R rather than a differently-built arm.

Running this before any paid work is deliberate. It exercises the whole split →
render → profile path on real data, produces the prefix-length and
continuation-token profile that R's corpus must be matched against, and costs
nothing. If C cannot be built, R should not be generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.data.prefix_split import (  # noqa: E402
    TruncationError, build_splits, prefix_length_profile,
)
from aadistill.data.sessions import (  # noqa: E402
    render_session, render_system_block,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from diagnose_training_recall import rung_session_ids  # noqa: E402

PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/stage3/e5_arm_c")
    ap.add_argument("--truncations", type=int, default=2)
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--tokenizer", type=Path, default=INIT)
    # C's tokens do not depend on a seed, but the pair key does: R generates one
    # prefix set per source checkpoint, so C must be emitted per seed to pair
    # one-to-one with it.
    ap.add_argument("--source-seed", default="sa")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    stop_ids = frozenset(
        i for i in (tok.convert_tokens_to_ids("<|im_end|>"),
                    tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None)

    # The incremental 0.86M -> 1.60M slice: prompts P2-0.86M has never trained on.
    ids860 = set(rung_session_ids(PACK, 860000))
    ids1600 = set(rung_session_ids(PACK, 1600000))
    incremental = ids1600 - ids860
    print(f"incremental sessions: {len(incremental)}")

    examples, rejected = [], Counter()
    system_ids: dict[str, list[int]] = {}
    candidate_sessions = 0
    all_splits = []
    rejected_sessions: list[dict] = []
    by_task: Counter = Counter()
    for line in SESSIONS.open():
        s = json.loads(line)
        if s["id"] not in incremental:
            continue
        try:
            rendered = render_session(tok, s, block_len=args.block_len)
        except ValueError as exc:
            rejected[f"render:{str(exc)[:40]}"] += 1
            continue
        candidate_sessions += 1
        try:
            splits = build_splits(
                rendered.body_ids, rendered.body_mask,
                seed_material=str(s["id"]), count=args.truncations,
                stop_ids=stop_ids,
                max_total_tokens=args.block_len - rendered.n_system_tokens)
        except TruncationError as exc:
            rejected[exc.reason] += 1
            rejected_sessions.append({"session_id": s["id"], "reason": exc.reason,
                                      "data_type": s["data_type"]})
            continue
        if rendered.system_key not in system_ids:
            sys_text = next((m["content"] for m in s["messages"]
                             if m["role"] == "system"), "")
            system_ids[rendered.system_key] = tok(
                render_system_block(tok, sys_text, s.get("tools")),
                add_special_tokens=False).input_ids
        for j, sp in enumerate(splits):
            examples.append({
                "id": f"{s['id']}#c{j}",
                "source_session_id": s["id"],
                "source_seed": args.source_seed,
                "truncation_index": j,
                "truncation_fraction": round(sp.k / max(1, sp.span_end - sp.span_start), 4),
                "data_type": s["data_type"],
                "arm": "C",
                "prefix_source": "teacher_native",
                "k": sp.k,
                "n_prefix_tokens": sp.n_prefix_tokens,
                "n_continuation_tokens": sp.n_continuation_tokens,
                "n_total_tokens": sp.n_tokens,
                "n_system_tokens": rendered.n_system_tokens,
                "system_key": rendered.system_key,
                # The packer consumes tokens, not metadata. Arm C's ids are the
                # rendered session with the system block back in front, so both
                # arms present the same shape to e5_pack.
                "ids": system_ids[rendered.system_key] + rendered.body_ids,
                "mask": [False] * rendered.n_system_tokens + list(sp.mask),
            })
            by_task[s["data_type"]] += sp.n_continuation_tokens
        all_splits += splits

    profile = prefix_length_profile(all_splits)
    total_cont = profile["continuation_tokens"]["total"] if all_splits else 0
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "arm": "C",
        "description": ("teacher-native prefix + supervised teacher continuation; "
                        "tokens identical to the corpus, only the loss mask moves"),
        "truncations_per_prompt": args.truncations,
        "block_len": args.block_len,
        "source_sessions": len(incremental),
        # Session-level and example-level censuses are reported separately: one
        # rejected session removes `truncations` candidate examples, and mixing
        # the units made an earlier build look like it had lost a sample it had
        # not (see logs/e5_registration.json census_reconciliation).
        "sessions": {
            "candidates": len(incremental),
            "accepted": len(incremental) - len(rejected_sessions),
            "rejected": len(rejected_sessions),
            "rejected_detail": rejected_sessions,
        },
        "examples_census": {
            "candidates": len(incremental) * args.truncations,
            "accepted": len(examples),
            "lost_to_session_rejections": len(rejected_sessions) * args.truncations,
        },
        "examples": len(examples),
        "rejected": dict(rejected.most_common()),
        "acceptance_rate": round(
            len(examples) / max(1, len(incremental) * args.truncations), 4),
        "supervised_continuation_tokens": total_cont,
        "continuation_tokens_by_task": dict(by_task.most_common()),
        "profile": profile,
        "pack_sha256": {"blocks_npz": sha256_file(PACK / "blocks.npz"),
                        "audit_jsonl": sha256_file(PACK / "audit.jsonl")},
        "sessions_sha256": sha256_file(SESSIONS),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report["source_seed"] = args.source_seed
    (args.out / "examples.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in examples))
    (args.out / "system_ids.json").write_text(json.dumps(system_ids))
    (args.out / "manifest.json").write_text(json.dumps(report, indent=1))

    print(f"examples {len(examples)} of a possible "
          f"{len(incremental) * args.truncations} "
          f"(acceptance {report['acceptance_rate']:.1%})")
    if rejected:
        print("rejected:", dict(rejected.most_common()))
    p = profile["prefix_tokens"]; c = profile["continuation_tokens"]
    print(f"prefix tokens       p25 {p['p25']} p50 {p['p50']} p75 {p['p75']} max {p['max']}")
    print(f"continuation tokens p25 {c['p25']} p50 {c['p50']} p75 {c['p75']} max {c['max']}")
    print(f"supervised continuation tokens: {total_cont:,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

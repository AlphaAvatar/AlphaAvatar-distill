#!/usr/bin/env python
"""Do training and evaluation render the same prompt, and do the masks cover the
tokens that carry the protocol?

    PYTHONPATH=src python scripts/data/audit_render_and_masks.py \
        --sessions artifacts/stage3/corpus_v2/sessions.jsonl \
        --pack artifacts/stage3/ladder_uniform_probe --rung 250000 \
        --out artifacts/audit/render_and_mask_audit.json

Two questions, both of which would explain "trained on it 41 times and still
cannot produce it" without any appeal to exposure bias.

**A — prompt-rendering equivalence through the generation boundary.** Training
renders a full session with `render_session`; evaluation renders the prompt with
`apply_chat_template(..., add_generation_prompt=True)`. If those disagree by even
one token at the boundary, the model is being asked at eval time to continue a
prefix it never saw in training. The audit compares the evaluation prompt against
the training rendering **truncated to the same length**, token for token, and
reports the first divergence.

**B — mask coverage of the protocol tokens.** `<think>`, `</think>`, `<|im_end|>`
and the final-answer span are what the protocol is *made of*. If any of them sits
outside the CE mask, the model is never trained to emit it, and its absence at
generation time is a labelling defect rather than a learning failure. Measured on
the real pack, per token role.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.sessions import (  # noqa: E402
    render_session, render_system_block,
)
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--rung", type=int, default=250000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    ids_of = {name: tok.convert_tokens_to_ids(name) for name in
              ("<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>")}
    print("special ids:", ids_of)

    sessions = []
    with args.sessions.open() as f:
        for line in f:
            sessions.append(json.loads(line))
            if len(sessions) >= args.n:
                break

    # ---- A. rendering equivalence ---------------------------------------
    mismatches, checked = [], 0
    boundary = Counter()
    for s in sessions:
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        eval_prompt = tok.apply_chat_template(
            turns, tools=s.get("tools"), tokenize=False,
            add_generation_prompt=True)
        eval_ids = tok(eval_prompt, add_special_tokens=False).input_ids

        rendered = render_session(tok, s)
        sys_text = render_system_block(tok, rendered.system_text, rendered.tools)
        sys_ids = tok(sys_text, add_special_tokens=False).input_ids
        train_ids = list(sys_ids) + list(rendered.body_ids)

        checked += 1
        div = first_divergence(eval_ids, train_ids[:len(eval_ids)])
        boundary[tuple(eval_ids[-3:])] += 1
        if div is not None:
            mismatches.append({
                "id": s["id"], "first_divergence_index": div,
                "eval_len": len(eval_ids), "train_len": len(train_ids),
                "eval_ctx": tok.decode(eval_ids[max(0, div - 8):div + 8]),
                "train_ctx": tok.decode(train_ids[max(0, div - 8):div + 8]),
            })

    # ---- B. mask coverage on the real pack -------------------------------
    arrays = np.load(args.pack / "blocks.npz")
    meta = json.loads((args.pack / "ladder.json").read_text())
    n_blocks = next(r["n_blocks"] for r in meta["rungs"]
                    if r["target_supervised_tokens"] == args.rung)
    ids = arrays["input_ids"][:n_blocks]
    ce = arrays["ce_mask"][:n_blocks].astype(bool)
    content = arrays["content_mask"][:n_blocks].astype(bool)

    role = {}
    for name, tid in ids_of.items():
        if tid is None:
            continue
        occur = ids == tid
        role[name] = {
            "occurrences": int(occur.sum()),
            "in_ce_mask": int((occur & ce).sum()),
            "in_content_mask": int((occur & content).sum()),
            "ce_coverage": (round(float((occur & ce).sum() / occur.sum()), 4)
                            if occur.sum() else None),
        }

    # Where does the CE mask start and stop relative to the assistant turn? The
    # convention is "from after <|im_start|>assistant\n through the closing
    # <|im_end|>", so the terminator must be the LAST supervised token of a span.
    terminator_id = ids_of["<|im_end|>"]
    spans_ending_on_terminator = spans = 0
    for b in range(ids.shape[0]):
        row_ce, row_ids = ce[b], ids[b]
        i = 0
        T = row_ce.shape[0]
        while i < T:
            if row_ce[i]:
                j = i
                while j + 1 < T and row_ce[j + 1]:
                    j += 1
                spans += 1
                spans_ending_on_terminator += int(row_ids[j] == terminator_id)
                i = j + 1
            else:
                i += 1

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "tokenizer": f"{TEACHER}@{REVISION}",
        "special_token_ids": ids_of,
        "render_equivalence": {
            "sessions_checked": checked,
            "mismatches": len(mismatches),
            "verdict": "identical" if not mismatches else "DIVERGENT",
            "examples": mismatches[:10],
            "eval_prompt_last_3_tokens": {
                tok.decode(list(k)): v for k, v in boundary.most_common(5)},
        },
        "mask_coverage": {
            "rung": args.rung, "blocks": int(ids.shape[0]),
            "supervised_tokens": int(ce.sum()),
            "per_token_role": role,
            "ce_spans": spans,
            "ce_spans_ending_on_im_end": spans_ending_on_terminator,
            "ce_span_terminator_rate": (round(spans_ending_on_terminator / spans, 4)
                                        if spans else None),
        },
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"\nA. render equivalence: {checked} sessions, "
          f"{len(mismatches)} mismatches -> {out['render_equivalence']['verdict']}")
    for m in mismatches[:3]:
        print(f"   {m['id']} diverges at {m['first_divergence_index']}")
        print(f"     eval : {m['eval_ctx']!r}")
        print(f"     train: {m['train_ctx']!r}")
    print("   eval prompt ends with:",
          list(out["render_equivalence"]["eval_prompt_last_3_tokens"])[:3])
    print(f"\nB. mask coverage over {int(ids.shape[0])} blocks, "
          f"{int(ce.sum()):,} supervised tokens")
    for name, d in role.items():
        print(f"   {name:14s} occurrences {d['occurrences']:>7,}  "
              f"in CE {d['in_ce_mask']:>7,}  coverage {d['ce_coverage']}")
    print(f"   CE spans {spans:,}; ending on <|im_end|> "
          f"{spans_ending_on_terminator:,} "
          f"({out['mask_coverage']['ce_span_terminator_rate']})")


if __name__ == "__main__":
    main()

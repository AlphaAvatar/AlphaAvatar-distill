"""Pack the corpus once and cut the six-point nested token ladder from it.

    uv run python scripts/data/build_token_ladder.py \
        --sessions artifacts/stage3/corpus_v2/bulk/sessions.jsonl \
        --out artifacts/stage3/corpus_v2/packed

Why one pack and not six
------------------------
The ladder must be strictly nested, and the authoritative data-size variable is
the supervised-token count that survives *packing-time terminal truncation*.
Those two requirements together rule out packing each rung separately: a
per-rung pack would re-place sessions, so rung k's blocks would not be a prefix
of rung k+1's and each rung's post-packing count would be an independent draw.

So the maximal corpus is packed **once**, sequentially, in a fixed type-balanced
session order. Cumulative supervised tokens are tallied per block, and each rung
is simply "the first N blocks". Nesting is then prefix nesting on blocks *and*
sessions, by construction, and every rung's token count is exact rather than
nominal. Only `n_blocks` distinguishes one rung from another, so no seed and no
rung can change the packing layout (§10).

Session order is a deterministic stratified interleave over the data types, in
proportion to their share of the corpus, so **every prefix preserves the type
distribution** — otherwise size and mixture would be confounded and the curve
would be measuring both.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.mixture import (  # noqa: E402
    block_token_mix,
    interleave,
    order_blocks,
)
from aadistill.data.sessions import (  # noqa: E402
    SYSTEM_DEFAULT,
    pack_sessions,
    render_session,
    render_system_block,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

LADDER = [250_000, 460_000, 860_000, 1_600_000, 2_960_000, 5_500_000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@"
                                       "768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--pad-id", type=int, default=None,
                    help="defaults to the tokenizer's pad/eos id")
    ap.add_argument("--ladder", default=",".join(str(v) for v in LADDER))
    ap.add_argument("--mixture", default=None,
                    help="declared supervised-token shares per data type, e.g. "
                         "'gsm8k=0.22,openmath=0.17,...'; omitted means "
                         "proportional to session counts")
    ap.add_argument("--mixture-note", default=None,
                    help="free-text rationale recorded in ladder.json")
    ap.add_argument("--session-order", default=None,
                    help="newline-delimited session ids giving the session order "
                         "to use instead of recomputing the stratified "
                         "interleave, and the order blocks are emitted in. "
                         "Sessions absent from the file follow, in corpus order. "
                         "Used to anchor a re-cut pack to an already-trained "
                         "pack so a controlled comparison keeps prompt overlap. "
                         "The anchor pack's own order already carried the "
                         "declared mixture, so replaying it preserves the "
                         "mixture; the realized per-rung mix is reported either "
                         "way, and must be checked rather than assumed.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shares = None
    if args.mixture:
        shares = {}
        for item in args.mixture.split(","):
            key, _, value = item.partition("=")
            shares[key.strip()] = float(value)

    from transformers import AutoTokenizer

    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    pad_id = args.pad_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
    rungs = [int(v) for v in args.ladder.split(",")]

    sessions = [json.loads(line) for line in open(args.sessions) if line.strip()]
    if not sessions:
        raise SystemExit(f"{args.sessions} is empty")
    print(f"{len(sessions)} accepted sessions", flush=True)

    rendered = [render_session(tokenizer, s, block_len=args.block_len)
                for s in sessions]

    # Several system-prompt groups are expected, not exceptional: `tool_calling`
    # renders each conversation's own tool schema into the system block, so the
    # corpus carries thousands of groups. Packing stays strictly within a group
    # (the system prompt is a hard boundary), and the declared mixture is then
    # restored by ordering the resulting *blocks* rather than the sessions.
    keys = {r.system_key for r in rendered}

    by_type: dict[str, list] = defaultdict(list)
    for r in rendered:
        by_type[r.data_type].append(r)
    if shares:
        missing = [t for t in shares if t not in by_type and shares[t] > 0]
        if missing:
            raise SystemExit(
                f"mixture declares {missing} but the corpus has no accepted "
                f"session of those types; have {sorted(by_type)}")
    anchor_rank = None
    if args.session_order:
        # Anchor to an existing pack's session order. Cleaning removes sessions
        # and changes lengths, and the largest-remainder interleave is a global
        # function of both — so recomputing it from scratch reshuffles the whole
        # corpus and a rung prefix ends up holding a substantially different
        # prompt set. Replaying the anchor order keeps the comparison about
        # target quality instead of about which prompts landed in the cut.
        anchor_rank = {}
        with open(args.session_order) as f:
            for i, line in enumerate(f):
                sid = line.strip()
                if sid:
                    anchor_rank.setdefault(sid, i)
        unranked = len(anchor_rank)
        order = sorted(rendered,
                       key=lambda r: (anchor_rank.get(r.session_id, unranked),
                                      r.session_id))
        anchored = sum(1 for r in rendered if r.session_id in anchor_rank)
        print(f"session order anchored to {args.session_order}: "
              f"{anchored}/{len(rendered)} sessions ranked, "
              f"{len(rendered) - anchored} appended in corpus order", flush=True)
    else:
        order = interleave(dict(by_type), shares)

    system_ids_by_key, system_text_by_key = {}, {}
    for r in rendered:
        if r.system_key not in system_ids_by_key:
            block_text = render_system_block(tokenizer, r.system_text, r.tools)
            system_ids_by_key[r.system_key] = tokenizer(
                block_text, add_special_tokens=False).input_ids
            system_text_by_key[r.system_key] = r.system_text
    blocks = pack_sessions(order, system_ids_by_key,
                           block_len=args.block_len, pad_id=pad_id)
    print(f"{len(blocks)} packed blocks across {len(keys)} system-prompt groups",
          flush=True)

    # Packing per group means the block list arrives grouped, so a ladder prefix
    # would be one group's data rather than the mixture. Reorder the blocks with
    # the same largest-remainder rule, now on each block's per-type supervised
    # tokens, so every prefix carries the declared mixture at block granularity.
    # Block ordering is *not* anchored, even when the sessions are. Replaying the
    # anchor's block order was tried and measured: it lifts prompt overlap but
    # wrecks the mixture, because the anchor's blocks group different sessions
    # once cleaning removes some — the 2.96M rung came out at code 0.193 /
    # openmath 0.130 against a declared 0.1667, a 5.9 pp drift that would turn a
    # target-quality experiment into a mixture experiment. Mixture fidelity wins;
    # the overlap that costs is reported instead of hidden.
    blocks = order_blocks(blocks, shares)

    cumulative, running = [], 0
    for b in blocks:
        running += b.n_supervised
        cumulative.append(running)
    corpus_supervised = running

    ladder = []
    for target in rungs:
        n_blocks = next((i + 1 for i, c in enumerate(cumulative) if c >= target), None)
        if n_blocks is None:
            ladder.append({"target_supervised_tokens": target, "reachable": False,
                           "n_blocks": len(blocks),
                           "actual_supervised_tokens": corpus_supervised})
            continue
        used = blocks[:n_blocks]
        type_counts, type_tokens = Counter(), Counter()
        session_ids = []
        for b in used:
            for m in b.audit["sessions"]:
                type_counts[m["data_type"]] += 1
                type_tokens[m["data_type"]] += m["supervised_retained"]
                session_ids.append(m["session_id"])
        tok_total = sum(type_tokens.values()) or 1
        ladder.append({
            "target_supervised_tokens": target,
            "reachable": True,
            "n_blocks": n_blocks,
            "actual_supervised_tokens": cumulative[n_blocks - 1],
            "n_sessions": len(session_ids),
            "real_tokens": sum(b.audit["unpadded_length"] for b in used),
            "padding_tokens": sum(b.audit["padding_length"] for b in used),
            "terminal_truncations": sum(
                1 for b in used if b.audit["terminal_truncated"]),
            "session_mix": {t: round(c / len(session_ids), 4)
                            for t, c in sorted(type_counts.items())},
            # The declared quantity: what fraction of the supervision each
            # capability actually contributes at this rung.
            "token_mix": {t: round(v / tok_total, 4)
                          for t, v in sorted(type_tokens.items())},
            "token_counts": dict(sorted(type_tokens.items())),
        })

    # Nesting is prefix nesting, so it can be asserted rather than trusted.
    reachable = [r for r in ladder if r["reachable"]]
    for a, b in zip(reachable, reachable[1:]):
        if a["n_blocks"] > b["n_blocks"]:
            raise SystemExit("ladder rungs are not monotonic — nesting is broken")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "blocks.npz",
        input_ids=np.asarray([b.input_ids for b in blocks], dtype=np.int32),
        ce_mask=np.asarray([b.ce_mask for b in blocks], dtype=bool),
        content_mask=np.asarray([b.content_mask for b in blocks], dtype=bool),
    )
    with open(out_dir / "audit.jsonl", "w") as f:
        for i, b in enumerate(blocks):
            f.write(json.dumps({"block": i, "cumulative_supervised": cumulative[i],
                                **b.audit}, ensure_ascii=False) + "\n")
    (out_dir / "ladder.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "block_len": args.block_len,
        "pad_id": pad_id,
        "n_blocks": len(blocks),
        "n_sessions": len(order),
        "corpus_supervised_tokens": corpus_supervised,
        "corpus_type_mix": {t: len(v) for t, v in sorted(by_type.items())},
        "system_prompt_groups": len(keys),
        "system_prompt_default": SYSTEM_DEFAULT,
        "ordering": ("largest-remainder stratified interleave on supervised "
                     "tokens (seed-free)" if shares else
                     "largest-remainder stratified interleave on session counts"),
        "block_ordering": ("blocks reordered to the declared token mixture; "
                           "required because the system prompt is a hard packing "
                           "boundary and tool schemas render into it"
                           if shares else "input order"),
        "declared_mixture": shares,
        "mixture_rationale": args.mixture_note,
        "packing_efficiency": round(
            sum(b.audit["unpadded_length"] for b in blocks)
            / max(len(blocks) * args.block_len, 1), 4),
        "packing": "sequential within system-prompt group, terminal truncation only",
        "rungs": ladder,
        "sessions_sha256": sha256_file(Path(args.sessions)),
        "outputs": {"blocks": sha256_file(out_dir / "blocks.npz"),
                    "audit": sha256_file(out_dir / "audit.jsonl")},
        "code_state": code_state(str(REPO_ROOT)),
    }, indent=2) + "\n")

    print(f"\nwrote {out_dir}")
    print(f"  corpus supervised tokens: {corpus_supervised:,}")
    for r in ladder:
        if not r["reachable"]:
            print(f"  {r['target_supervised_tokens']:>9,} UNREACHABLE "
                  f"(corpus max {corpus_supervised:,})")
            continue
        print(f"  {r['target_supervised_tokens']:>9,} -> "
              f"{r['actual_supervised_tokens']:>9,} supervised  "
              f"{r['n_blocks']:>5d} blocks  {r['n_sessions']:>6d} sessions")
        print(f"            token mix {r['token_mix']}")


if __name__ == "__main__":
    main()

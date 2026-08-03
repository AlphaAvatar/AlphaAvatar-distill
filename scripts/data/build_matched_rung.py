#!/usr/bin/env python
"""Pack a single rung whose prompt set is matched to an already-trained rung.

`build_token_ladder.py` packs the whole corpus and cuts nested prefixes. That is
right for a scaling ladder, and wrong for a controlled A/B against one rung: the
block-level mixture repair is a global function of the block set, so removing a
few sessions displaces many others out of a short prefix. Measured on the 0.86M
rung, that cost 30 points of prompt overlap — 96.5% of the control's sessions
survive cleaning, but only 66.6% landed inside the re-cut prefix.

This builder instead packs **only the sessions the rung should contain**:

1. take the control rung's own sessions, in the control's order, keeping those
   the cleaned corpus still holds (with their cleaned targets);
2. top up per data type from clean sessions *outside* the control rung, in the
   control pack's order, choosing the type that is furthest behind its target
   share — never duplicating a prompt and never truncating a completion;
3. pack, repair the block mixture, and cut the control's exact block count —
   which is what pins optimizer steps, packed tokens and training compute.

Step 3 needs slack to work. Packing a pool to *exactly* the target leaves the
block-level mixture repair nothing to choose between, and the realized mixture
then inherits whatever the pool happens to contain: measured on the 0.86M rung,
a zero-slack pool reached 96.5% prompt overlap but drifted `code` to 0.208 and
`openmath` to 0.137 against a declared 0.1667, because the control's own long
`code` sessions are no longer truncated the same way. So the pool is built
`--overshoot` times larger than the cut and the repair selects within it. The
builder sweeps overshoot and keeps the **smallest** pool whose realized mixture
is inside `--mixture-tolerance`, which is also the one with the highest overlap.

The control's **validation blocks are appended verbatim**, after the training
cut. `aadistill.data.ladder` takes validation from the blocks past the largest
declared rung, so a pack containing only training blocks would fail outright —
and worse, a pack with its own tail would validate the treatment on *different*
blocks than the control, which would silently break the sharpest instrument in
the comparison (teacher-native val CE resolves the Experiment 1 effect at 74x
the between-seed noise). The appended blocks are asserted byte-identical to the
control's and their hash is recorded.

The output is byte-compatible with a `build_token_ladder.py` pack — same
`blocks.npz`, `audit.jsonl` and `ladder.json` schema — so the trainer's
`packing: "ladder"` path consumes it unchanged.

Usage:
    scripts/data/build_matched_rung.py \
        --control-ladder artifacts/stage3/ladder_uniform_probe \
        --control-rung 860000 \
        --sessions artifacts/stage3/corpus_v2_clean/sessions_clean.jsonl \
        --model <teacher path or repo@revision> \
        --out artifacts/stage3/rung_0860k_clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.sessions import (  # noqa: E402
    SYSTEM_DEFAULT,
    pack_sessions,
    render_session,
    render_system_block,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
from build_token_ladder import order_blocks  # noqa: E402


def read_control(ladder_dir: Path, target: int):
    """The control rung's block count, ordered session ids and per-type tokens."""
    ladder = json.loads((ladder_dir / "ladder.json").read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == target)
    n_blocks = rung["n_blocks"]
    in_rung, rung_order, corpus_order = set(), [], []
    per_type = Counter()
    with (ladder_dir / "audit.jsonl").open() as f:
        for i, line in enumerate(f):
            for s in json.loads(line)["sessions"]:
                corpus_order.append(s["session_id"])
                if i < n_blocks:
                    in_rung.add(s["session_id"])
                    rung_order.append(s["session_id"])
                    per_type[s["data_type"]] += s["supervised_retained"]
    return {"n_blocks": n_blocks, "rung": rung, "in_rung": in_rung,
            "rung_order": rung_order, "corpus_order": corpus_order,
            "per_type_tokens": per_type,
            "supervised": rung["actual_supervised_tokens"],
            "block_len": ladder["block_len"]}


def choose_pool(control, rendered_by_id, extra_tokens: int):
    """Control-rung survivors, then per-type top-ups worth ~`extra_tokens`.

    Top-ups are drawn in the control pack's own order and assigned to whichever
    type is furthest behind its share of the control's per-type token counts, so
    filling a shortfall cannot drift the mixture.
    """
    pool = [rendered_by_id[i] for i in control["rung_order"] if i in rendered_by_id]
    emitted = Counter()
    for r in pool:
        emitted[r.data_type] += r.n_supervised

    target = control["per_type_tokens"]
    grand = sum(target.values())
    want = {t: v / grand for t, v in target.items()}

    outside = defaultdict(list)
    for sid in control["corpus_order"]:
        if sid in control["in_rung"] or sid not in rendered_by_id:
            continue
        r = rendered_by_id[sid]
        outside[r.data_type].append(r)
    cursors = {t: 0 for t in outside}

    added = 0
    while added < extra_tokens:
        total = sum(emitted.values()) or 1
        best, best_deficit = None, None
        for t, items in outside.items():
            if cursors[t] >= len(items):
                continue
            deficit = want.get(t, 0.0) - emitted[t] / total
            if best is None or deficit > best_deficit:
                best, best_deficit = t, deficit
        if best is None:
            break
        r = outside[best][cursors[best]]
        cursors[best] += 1
        pool.append(r)
        emitted[best] += r.n_supervised
        added += r.n_supervised
    return pool


def pack(pool, system_ids_by_key, block_len, pad_id, shares):
    blocks = pack_sessions(pool, system_ids_by_key, block_len=block_len,
                           pad_id=pad_id)
    return order_blocks(blocks, shares)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ladder", required=True, type=Path)
    ap.add_argument("--control-rung", required=True, type=int)
    ap.add_argument("--sessions", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@"
                                       "768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--pad-id", type=int, default=None)
    ap.add_argument("--mixture", default=None,
                    help="declared supervised-token shares; omitted means the "
                         "control rung's own realized per-type shares")
    ap.add_argument("--mixture-note", default=None)
    ap.add_argument("--overshoot", default="1.0,1.1,1.25,1.5,1.75,2.0,2.5,3.0",
                    help="pool sizes to sweep, as multiples of the target block "
                         "count; the smallest one inside --mixture-tolerance wins")
    ap.add_argument("--control-val-blocks", default=None,
                    help="comma-separated block indices in the control pack to "
                         "append verbatim as this pack's validation tail. "
                         "Required for a comparable val CE; read them from the "
                         "control run's `dataset_loaded` log event.")
    ap.add_argument("--mixture-tolerance", type=float, default=0.005,
                    help="max absolute per-type share drift from the declared "
                         "mixture, as a fraction (0.005 = 0.5 percentage points)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    pad_id = args.pad_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    control = read_control(args.control_ladder, args.control_rung)
    print(f"control rung {args.control_rung:,}: {control['n_blocks']} blocks, "
          f"{len(control['in_rung'])} sessions, "
          f"{control['supervised']:,} supervised", flush=True)

    if args.mixture:
        shares = {}
        for item in args.mixture.split(","):
            key, _, value = item.partition("=")
            shares[key.strip()] = float(value)
    else:
        grand = sum(control["per_type_tokens"].values())
        shares = {t: v / grand for t, v in control["per_type_tokens"].items()}

    sessions = [json.loads(line) for line in args.sessions.open() if line.strip()]
    rendered_by_id = {}
    for s in sessions:
        r = render_session(tokenizer, s, block_len=args.block_len)
        rendered_by_id[s["id"]] = r
    print(f"{len(rendered_by_id)} clean sessions rendered", flush=True)

    survivors = sum(1 for i in control["rung_order"] if i in rendered_by_id)
    print(f"control-rung survivors: {survivors}/{len(control['in_rung'])} "
          f"({survivors / len(control['in_rung']):.1%}) — the overlap ceiling",
          flush=True)

    system_ids_by_key = {}
    for r in rendered_by_id.values():
        if r.system_key not in system_ids_by_key:
            block_text = render_system_block(tokenizer, r.system_text, r.tools)
            system_ids_by_key[r.system_key] = tokenizer(
                block_text, add_special_tokens=False).input_ids

    # Sweep the pool size. Blocks rise with top-up tokens, so each overshoot is
    # reached by bisection on tokens; the cut is then the control's exact block
    # count, and the mixture repair chooses which blocks fill it.
    target_blocks = control["n_blocks"]

    def pool_for_blocks(want_blocks):
        lo, hi, best = 0, max(control["supervised"] * 4, 1), None
        for _ in range(48):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            pool = choose_pool(control, rendered_by_id, mid)
            blocks = pack(pool, system_ids_by_key, args.block_len, pad_id, shares)
            if len(blocks) >= want_blocks:
                best = (pool, blocks)
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    def drift(blocks):
        tokens = Counter()
        for b in blocks[:target_blocks]:
            for m in b.audit["sessions"]:
                tokens[m["data_type"]] += m["supervised_retained"]
        total = sum(tokens.values()) or 1
        scale = sum(shares.values())
        return max(abs(shares.get(t, 0.0) / scale - v / total)
                   for t, v in tokens.items())

    chosen, frontier = None, []
    for factor in [float(x) for x in args.overshoot.split(",")]:
        want = max(target_blocks, int(round(target_blocks * factor)))
        got = pool_for_blocks(want)
        if got is None:
            print(f"  overshoot {factor}: pool cannot reach {want} blocks", flush=True)
            continue
        pool, blocks = got
        if len(blocks) < target_blocks:
            continue
        cut = blocks[:target_blocks]
        ids = {m["session_id"] for b in cut for m in b.audit["sessions"]}
        overlap = len(control["in_rung"] & ids)
        d = drift(blocks)
        sup = sum(b.n_supervised for b in cut)
        frontier.append({"overshoot": factor, "pool_sessions": len(pool),
                         "pool_blocks": len(blocks), "overlap": overlap,
                         "overlap_rate": round(overlap / len(control["in_rung"]), 4),
                         "max_share_drift_pp": round(d * 100, 3),
                         "supervised_tokens": sup,
                         "supervised_rel": round(sup / control["supervised"] - 1, 6)})
        print(f"  overshoot {factor:>4}: pool {len(pool):>5} sessions / "
              f"{len(blocks):>5} blocks -> cut {target_blocks}, overlap "
              f"{overlap}/{len(control['in_rung'])} "
              f"({overlap / len(control['in_rung']):.1%}), drift "
              f"{d * 100:5.2f} pp, supervised {sup:>9,} "
              f"({sup / control['supervised'] - 1:+.2%})", flush=True)
        # Pre-registered priority: exact compute (all candidates satisfy it),
        # then mixture inside tolerance, then maximum prompt overlap. The token
        # residual is reported, not optimised — trading mixture fidelity for it
        # would reintroduce the confound the tolerance exists to prevent.
        if d <= args.mixture_tolerance and (chosen is None or overlap > chosen[3]):
            chosen = (pool, cut, factor, overlap, d)
    if chosen is None:
        raise SystemExit(
            f"no pool size in {args.overshoot} kept the mixture within "
            f"{args.mixture_tolerance * 100:.2f} pp. Widen --overshoot or relax "
            "the tolerance explicitly rather than shipping a drifted mixture.")
    pool, blocks, overshoot_used, overlap_used, drift_used = chosen
    print(f"\nselected overshoot {overshoot_used} — highest prompt overlap among "
          f"pools inside {args.mixture_tolerance * 100:.2f} pp of the declared "
          f"mixture", flush=True)

    cumulative, running = [], 0
    for b in blocks:
        running += b.n_supervised
        cumulative.append(running)

    type_counts, type_tokens = Counter(), Counter()
    session_ids = []
    for b in blocks:
        for m in b.audit["sessions"]:
            type_counts[m["data_type"]] += 1
            type_tokens[m["data_type"]] += m["supervised_retained"]
            session_ids.append(m["session_id"])
    tok_total = sum(type_tokens.values()) or 1

    rung = {
        "target_supervised_tokens": running,
        "reachable": True,
        "n_blocks": len(blocks),
        "actual_supervised_tokens": running,
        "n_sessions": len(session_ids),
        "real_tokens": sum(b.audit["unpadded_length"] for b in blocks),
        "padding_tokens": sum(b.audit["padding_length"] for b in blocks),
        "terminal_truncations": sum(1 for b in blocks if b.audit["terminal_truncated"]),
        "session_mix": {t: round(c / len(session_ids), 4)
                        for t, c in sorted(type_counts.items())},
        "token_mix": {t: round(v / tok_total, 4)
                      for t, v in sorted(type_tokens.items())},
        "token_counts": dict(sorted(type_tokens.items())),
    }

    # Append the control's own validation blocks, verbatim.
    train_ids = np.asarray([b.input_ids for b in blocks], dtype=np.int32)
    train_ce = np.asarray([b.ce_mask for b in blocks], dtype=bool)
    train_content = np.asarray([b.content_mask for b in blocks], dtype=bool)
    val_audit_rows, val_meta = [], None
    if args.control_val_blocks:
        want = [int(x) for x in args.control_val_blocks.split(",")]
        src = np.load(args.control_ladder / "blocks.npz")
        control_audit = [json.loads(line)
                         for line in (args.control_ladder / "audit.jsonl").open()
                         if line.strip()]
        val_ids = src["input_ids"][want].astype(np.int32)
        val_ce = src["ce_mask"][want]
        val_content = src["content_mask"][want]
        val_audit_rows = [control_audit[i] for i in want]
        val_supervised = int(val_ce.sum())
        train_ids = np.concatenate([train_ids, val_ids])
        train_ce = np.concatenate([train_ce, val_ce])
        train_content = np.concatenate([train_content, val_content])
        val_meta = {
            "source_pack": str(args.control_ladder),
            "control_block_indices": want,
            "appended_at": [len(blocks) + i for i in range(len(want))],
            "n_blocks": len(want),
            "supervised_tokens": val_supervised,
            "token_ids_sha256": hashlib.sha256(val_ids.tobytes()).hexdigest(),
            "note": ("byte-identical to the control's validation blocks, so the "
                     "treatment and the control are scored on one identical set"),
        }
        print(f"appended {len(want)} control validation blocks "
              f"({val_supervised:,} supervised tokens), sha256 "
              f"{val_meta['token_ids_sha256'][:16]}", flush=True)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "blocks.npz", input_ids=train_ids,
                        ce_mask=train_ce, content_mask=train_content)
    with (out_dir / "audit.jsonl").open("w") as f:
        for i, b in enumerate(blocks):
            f.write(json.dumps({"block": i, "cumulative_supervised": cumulative[i],
                                **b.audit}, ensure_ascii=False) + "\n")
        for j, row in enumerate(val_audit_rows):
            keep = {k: v for k, v in row.items()
                    if k not in ("block", "cumulative_supervised")}
            f.write(json.dumps({"block": len(blocks) + j,
                                "cumulative_supervised": cumulative[-1],
                                "validation_block": True, **keep},
                               ensure_ascii=False) + "\n")

    by_type = Counter(r.data_type for r in pool)
    shared = len(control["in_rung"] & set(session_ids))
    (out_dir / "ladder.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "block_len": args.block_len,
        "pad_id": pad_id,
        "n_blocks": len(blocks),
        "n_sessions": len(pool),
        "corpus_supervised_tokens": running,
        "corpus_type_mix": dict(sorted(by_type.items())),
        "system_prompt_groups": len({r.system_key for r in pool}),
        "system_prompt_default": SYSTEM_DEFAULT,
        "ordering": ("control-rung sessions in the control's order, then "
                     "largest-remainder per-type top-ups drawn from outside the "
                     "control rung in the control pack's order"),
        "block_ordering": ("blocks reordered to the declared token mixture; "
                           "required because the system prompt is a hard packing "
                           "boundary and tool schemas render into it"),
        "declared_mixture": shares,
        "mixture_rationale": args.mixture_note,
        "packing_efficiency": round(
            sum(b.audit["unpadded_length"] for b in blocks)
            / max(len(blocks) * args.block_len, 1), 4),
        "packing": "sequential within system-prompt group, terminal truncation only",
        "matched_to": {
            "ladder": str(args.control_ladder),
            "rung": args.control_rung,
            "n_blocks": control["n_blocks"],
            "supervised_tokens": control["supervised"],
            "sessions": len(control["in_rung"]),
            "survivors_of_cleaning": survivors,
            "overlap_ceiling": round(survivors / len(control["in_rung"]), 4),
            "prompt_overlap": shared,
            "prompt_overlap_rate": round(shared / len(control["in_rung"]), 4),
            "pool_overshoot": overshoot_used,
            "max_share_drift_pp": round(drift_used * 100, 3),
            "mixture_tolerance_pp": args.mixture_tolerance * 100,
            "frontier": frontier,
            "selection_priority": ("exact block/step/packed-token match, then "
                                   "mixture drift within tolerance, then maximum "
                                   "prompt overlap; the unique-token residual is "
                                   "reported rather than optimised"),
        },
        "validation": val_meta,
        "rungs": [rung],
        "sessions_sha256": sha256_file(args.sessions),
        "outputs": {"blocks": sha256_file(out_dir / "blocks.npz"),
                    "audit": sha256_file(out_dir / "audit.jsonl")},
        "code_state": code_state(REPO_ROOT),
    }, indent=1))

    print(f"\nwrote {out_dir}")
    print(f"  {len(blocks)} blocks (control {control['n_blocks']}), "
          f"{running:,} supervised (control {control['supervised']:,}, "
          f"{running / control['supervised'] - 1:+.4%})")
    print(f"  prompt overlap {shared}/{len(control['in_rung'])} "
          f"({shared / len(control['in_rung']):.1%}), ceiling "
          f"{survivors / len(control['in_rung']):.1%}")
    print(f"  token mix {rung['token_mix']}")


if __name__ == "__main__":
    main()

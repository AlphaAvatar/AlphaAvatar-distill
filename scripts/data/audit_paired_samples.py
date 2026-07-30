"""Deterministic audit of paired public/teacher training samples.

Answers, from the artifacts themselves rather than from reasoning about the code,
the structural questions that decide whether the teacher-native arm was trained
on what we think it was:

* is `<think>` injected by the *generation* prompt, or produced by the teacher?
* does the raw teacher output open with `<think>`, or only ever close with
  `</think>`?
* does rebuilding the assistant turn introduce a **second** `<think>`?
* are `</think>`, the final answer and `<|im_end|>` all inside the loss mask?
* does parsing/normalisation drop any of the teacher's reasoning?
* do public targets render as an **empty** think block?
* does packing ever split, truncate or drop an accepted teacher completion?

The "raw teacher response before parsing" is reconstructed by **decoding the
exact sampled token ids** from the hashed rollout snapshot, which is stronger
than a stored string: `candidates.jsonl` deliberately omits `raw`, and the
snapshot is what the engine actually emitted.

Selection is deterministic and non-cherry-picked: accepted prompt ids sorted,
then a fixed stride within each group so every available group is represented.

Usage:
    uv run python scripts/data/audit_paired_samples.py --n 8 \
        --out artifacts/audit/paired_samples_20260730
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.dataset import (  # noqa: E402
    best_fit_blocks,
    encode_sample,
    load_split,
    render_chat,
)

MARKERS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")


def pick(ids_by_group: dict[str, list[str]], n: int) -> list[str]:
    """Deterministic stratified take: round-robin over groups, fixed stride."""
    out, groups = [], sorted(ids_by_group)
    per = max(1, n // max(1, len(groups)))
    for g in groups:
        rows = sorted(ids_by_group[g])
        stride = max(1, len(rows) // per)
        out += rows[::stride][:per]
    out = sorted(dict.fromkeys(out))
    if len(out) > n:  # trim evenly, never from one end
        step = len(out) / n
        out = [out[int(i * step)] for i in range(n)]
    return out


def mask_span(ids, mask, tok):
    """First and last supervised token index, and what they decode to."""
    on = [i for i, m in enumerate(mask) if m]
    if not on:
        return None
    a, b = on[0], on[-1]
    return {
        "first_index": a, "last_index": b, "count": len(on),
        "first_token": tok.decode([ids[a]]), "last_token": tok.decode([ids[b]]),
        "contiguous": (b - a + 1) == len(on),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="artifacts/stage2_v2/teacher_corpus_750/targets.jsonl")
    ap.add_argument("--snapshot", default="artifacts/stage2_v2/teacher_corpus_750/rollout_snapshot/rollouts.jsonl")
    ap.add_argument("--candidates", default="artifacts/stage2_v2/teacher_corpus_750/candidates.jsonl")
    ap.add_argument("--pilot-dir", default="data/stage3_pilot")
    ap.add_argument("--public-dir", default="data/stage2_v1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--full-traces", type=int, default=3,
                    help="show this many traces complete and unabbreviated")
    ap.add_argument("--head-tail", type=int, default=300,
                    help="tokens shown at each end for the remaining samples")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision", default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--out", default="artifacts/audit/paired_samples")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    targets = [json.loads(l) for l in (REPO_ROOT / args.targets).read_text().splitlines() if l.strip()]
    accepted = {t["id"]: t for t in targets if t.get("target_source") == "teacher_verified"}
    cand = {}
    for line in (REPO_ROOT / args.candidates).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cand[r["id"]] = r
    snap = defaultdict(dict)
    for line in (REPO_ROOT / args.snapshot).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid, _, idx = r["prompt_id"].rpartition("#")
        snap[sid][int(idx)] = r

    public = {}
    for split in ("train", "val"):
        try:
            for rows in load_split(REPO_ROOT / args.public_dir, split).values():
                for s in rows:
                    public[s["id"]] = s
        except FileNotFoundError:
            pass

    # Which arm-split each id landed in, so the audit reflects real training data.
    arm_split = {}
    for arm in ("control", "treatment"):
        for split in ("train", "val"):
            for rows in load_split(REPO_ROOT / args.pilot_dir / arm, split).values():
                for s in rows:
                    arm_split.setdefault(s["id"], {})[arm] = split

    by_group = defaultdict(list)
    for sid, t in accepted.items():
        by_group[f'{t["group"]}/{t["source"]}'].append(sid)
    chosen = pick(by_group, args.n)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = Counter()
    records = []

    for n_i, sid in enumerate(chosen, 1):
        t = accepted[sid]
        pub = public.get(sid)
        c = cand.get(sid, {})
        idx = t.get("candidate_index")
        snap_row = snap.get(sid, {}).get(idx if idx is not None else 0, {})

        user_turns = [m for m in t["messages"] if m["role"] != "assistant"]
        asst = [m for m in t["messages"] if m["role"] == "assistant"][0]

        # --- the generation prompt the teacher actually saw -------------------
        gen_prompt = tok.apply_chat_template(
            user_turns, tools=t.get("tools"), tokenize=False,
            add_generation_prompt=True)
        think_injected = "<think>" in gen_prompt

        # --- raw teacher output, decoded from the exact sampled token ids -----
        raw_tokens = snap_row.get("tokens") or []
        raw = tok.decode(raw_tokens, skip_special_tokens=False) if raw_tokens else ""
        raw_open = raw.count("<think>")
        raw_close = raw.count("</think>")

        # --- the training render + loss mask ---------------------------------
        t_ids, t_mask = encode_sample(tok, t)
        t_text = render_chat(tok, t)
        p_ids, p_mask = (encode_sample(tok, pub) if pub else ([], []))
        p_text = render_chat(tok, pub) if pub else ""

        rec = {
            "id": sid, "group": t["group"], "source": t["source"],
            "candidate_index": idx, "think_tokens": t.get("think_tokens"),
            "arm_split": arm_split.get(sid),
            "generation_prompt_injects_think": think_injected,
            "raw_open_think": raw_open, "raw_close_think": raw_close,
            "raw_token_count": len(raw_tokens),
            "rendered_open_think": t_text.count("<think>"),
            "rendered_close_think": t_text.count("</think>"),
            "public_rendered_open_think": p_text.count("<think>") if pub else None,
            "public_think_block_empty": (
                "<think>\n\n</think>" in p_text.replace("\r", "") if pub else None),
            "teacher_supervised_tokens": int(sum(t_mask)),
            "public_supervised_tokens": int(sum(p_mask)) if pub else None,
            "teacher_rendered_tokens": len(t_ids),
            "public_rendered_tokens": len(p_ids) if pub else None,
            "teacher_mask_span": mask_span(t_ids, t_mask, tok),
            "verifier": {"accepted": True,
                         "reason": next((x["reason"] for x in c.get("candidates", [])
                                         if x.get("index") == idx), None),
                         "all_candidate_reasons": [x.get("reason") for x in c.get("candidates", [])]},
        }

        # --- does parsing drop reasoning? -----------------------------------
        # The stored trace should be the raw output minus the delimiters.
        trace = asst.get("reasoning_content", "")
        answer = asst.get("content", "")
        raw_body = raw.split("</think>", 1)
        rec["raw_has_close_delim"] = len(raw_body) > 1
        if len(raw_body) > 1:
            raw_trace = raw_body[0].replace("<think>", "").strip()
            raw_answer = raw_body[1].replace("<|im_end|>", "").strip()
            rec["trace_preserved_exactly"] = raw_trace == trace.strip()
            rec["answer_preserved_exactly"] = raw_answer == answer.strip()
            rec["trace_chars_raw"] = len(raw_trace)
            rec["trace_chars_stored"] = len(trace.strip())
            rec["trace_chars_dropped"] = len(raw_trace) - len(trace.strip())
        # --- are the protocol tokens supervised? -----------------------------
        for name, marker in (("close_think", "</think>"), ("im_end", "<|im_end|>")):
            mid = tok.encode(marker, add_special_tokens=False)
            if len(mid) == 1:
                positions = [i for i, x in enumerate(t_ids) if x == mid[0]]
                rec[f"{name}_positions"] = positions
                # `<|im_end|>` closes every turn, including the user's, which is
                # correctly unsupervised. Only the FINAL one is the assistant's
                # own stop token and the thing termination training depends on.
                rec[f"{name}_supervised"] = (
                    bool(t_mask[positions[-1]]) if positions else None)
                rec[f"{name}_all_supervised"] = (
                    all(t_mask[i] for i in positions) if positions else None)
        rec["final_answer_supervised"] = bool(t_mask[-1]) if t_mask else None

        # --- packing: is this sample kept whole at block_len 8192? -----------
        _, bm, st = best_fit_blocks([(t_ids, t_mask)], 8192)
        rec["packing_8192"] = {"truncated_samples": st["truncated_samples"],
                               "supervised_kept": int(bm.sum()),
                               "lossless": st["truncated_samples"] == 0
                               and int(bm.sum()) == int(sum(t_mask))}

        for k, bad in (("rendered_double_think", rec["rendered_open_think"] > 1),
                       ("trace_dropped", rec.get("trace_chars_dropped", 0) > 0),
                       ("close_think_unsupervised", rec.get("close_think_supervised") is False),
                       ("im_end_unsupervised", rec.get("im_end_supervised") is False),
                       ("packing_lossy", not rec["packing_8192"]["lossless"])):
            if bad:
                findings[k] += 1
        records.append(rec)

        # ---------------- console ------------------------------------------
        print("=" * 78)
        print(f"[{n_i}/{len(chosen)}] {sid}  group={t['group']} source={t['source']} "
              f"cand={idx} split={rec['arm_split']}")
        print("=" * 78)
        print(f"--- USER PROMPT ({len(user_turns)} turn(s)) ---")
        for m in user_turns:
            body = m["content"]
            print(f"[{m['role']}] " + (body if len(body) < 1200 else
                                       body[:600] + f"\n  …[{len(body)-1200} chars omitted]…\n" + body[-600:]))
        print(f"\n--- PUBLIC v1 TARGET ({rec['public_supervised_tokens']} supervised tok) ---")
        print(repr(pub["messages"][-1]["content"]) if pub else "<none>")
        full = n_i <= args.full_traces
        print(f"\n--- RAW TEACHER OUTPUT, decoded from {len(raw_tokens)} sampled token ids"
              f"{' (COMPLETE)' if full else ''} ---")
        if full or len(raw_tokens) <= 2 * args.head_tail:
            print(raw)
        else:
            h = tok.decode(raw_tokens[:args.head_tail], skip_special_tokens=False)
            tl = tok.decode(raw_tokens[-args.head_tail:], skip_special_tokens=False)
            print(h)
            print(f"\n  …[{len(raw_tokens) - 2*args.head_tail} tokens omitted; "
                  f"complete text in {args.out}/samples/{sid}.json]…\n")
            print(tl)
        print(f"\n--- ACCEPTED TARGET AFTER PARSING ---")
        print(f"reasoning_content: {len(trace)} chars | content: {len(answer)} chars")
        print(f"trace preserved exactly: {rec.get('trace_preserved_exactly')} "
              f"(chars dropped: {rec.get('trace_chars_dropped')})")
        print(f"answer preserved exactly: {rec.get('answer_preserved_exactly')}")
        print(f"\n--- RENDERED TRAINING SEQUENCE (specials visible), "
              f"{len(t_ids)} tok / {int(sum(t_mask))} supervised ---")
        shown = t_text if full else (t_text[:900] + "\n  …\n" + t_text[-900:])
        print(shown.replace("<|im_start|>", "⟪im_start⟫").replace("<|im_end|>", "⟪im_end⟫")
                   .replace("<think>", "⟪think⟫").replace("</think>", "⟪/think⟫"))
        print(f"\n--- LOSS MASK ---")
        ms = rec["teacher_mask_span"]
        print(f"  supervised span: [{ms['first_index']}..{ms['last_index']}] "
              f"({ms['count']} tokens, contiguous={ms['contiguous']})")
        print(f"  first supervised token: {ms['first_token']!r}")
        print(f"  last  supervised token: {ms['last_token']!r}")
        print(f"  </think> at {rec.get('close_think_positions')} supervised={rec.get('close_think_supervised')}")
        print(f"  <|im_end|> at {rec.get('im_end_positions')} supervised={rec.get('im_end_supervised')}")
        print(f"  packing@8192 lossless: {rec['packing_8192']['lossless']}")
        print(f"  public supervised {rec['public_supervised_tokens']} vs "
              f"teacher supervised {rec['teacher_supervised_tokens']}")
        (out_dir / "samples").mkdir(exist_ok=True)
        (out_dir / "samples" / f"{sid}.json").write_text(json.dumps(
            {**rec, "user_turns": user_turns, "public_target": pub,
             "raw_teacher_output": raw, "reasoning_content": trace,
             "content": answer, "rendered_training_sequence": t_text},
            indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print("STRUCTURAL FINDINGS across", len(chosen), "samples")
    print("=" * 78)
    checks = {
        "rendered turn has a SECOND <think>": findings["rendered_double_think"],
        "parsing dropped trace characters": findings["trace_dropped"],
        "</think> not supervised": findings["close_think_unsupervised"],
        "<|im_end|> not supervised": findings["im_end_unsupervised"],
        "packing@8192 lossy": findings["packing_lossy"],
    }
    for k, v in checks.items():
        print(f"  {'FAIL' if v else 'ok  '}  {k}: {v}/{len(chosen)}")
    inj = sum(1 for r in records if r["generation_prompt_injects_think"])
    print(f"\n  PROTOCOL (expected, not a defect): generation prompt opens "
          f"<think>: {inj}/{len(chosen)}")
    raw_opens = sum(1 for r in records if r["raw_open_think"] > 0)
    print(f"\n  raw teacher output containing an OPENING <think>: {raw_opens}/{len(chosen)}")
    print(f"  raw teacher output containing </think>: "
          f"{sum(1 for r in records if r['raw_close_think'] > 0)}/{len(chosen)}")
    print(f"  rendered training turn opening <think> count (should be 1): "
          f"{Counter(r['rendered_open_think'] for r in records)}")
    print(f"  public target renders an EMPTY think block: "
          f"{sum(1 for r in records if r['public_think_block_empty'])}/{len(chosen)}")
    (out_dir / "summary.json").write_text(json.dumps(
        {"chosen": chosen, "checks": checks, "records": records}, indent=2))
    print(f"\nwrote {out_dir}/summary.json and {out_dir}/samples/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

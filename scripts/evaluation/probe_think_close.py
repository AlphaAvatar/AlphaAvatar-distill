"""Measure a model's probability on `</think>` at the contested position.

The primary readout for the CE/KD conflict experiment
(`logs/experiments/stage3/2026-07-28_kd_ce_protocol_conflict.md`). It is preferred to
`think_closed` for the same reason a thermometer beats "does it feel warm":

* it is **continuous**, so it moves smoothly with the force balance instead of
  flipping discretely at an argmax boundary;
* it is therefore **far less seed-noisy** than the generation metrics, whose
  measured run-to-run floor is 0.1290 on the composite;
* it reads the mechanism **directly** — the claim is that CE and KD fight over
  this exact token, so this is the number the claim is about.

`think_closed` and `format_ok` remain the behavioural confirmation; this is the
mechanistic one. Both are reported, because a mechanism that moves without the
behaviour following would itself be a finding.

Usage:
    uv run python scripts/evaluation/probe_think_close.py --model <path-or-id[@rev]> \
        [--data-dir data/stage2_v1] [--per-group 4] [--out probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.dataset import load_split, render_chat  # noqa: E402
from aadistill.models.teacher import load_causal_lm  # noqa: E402

TEACHER_TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path or id[@revision]")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        TEACHER_TOKENIZER, revision=TEACHER_REVISION, local_files_only=True
    )
    close_ids = tok.encode("</think>", add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(f"</think> is not a single token: {close_ids}")
    close = close_ids[0]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    # The pinned teacher tokenizer is used for rendering; every student
    # checkpoint shares it (identical tokenizer.json sha256 in
    # scripts/pod/hashes_ckpt.txt), so renders are comparable across models.
    model, _ = load_causal_lm(args.model, dtype, device)
    model.eval()

    splits = load_split(REPO_ROOT / args.data_dir, args.split)
    rows, per_group = [], {}
    for group in sorted(splits):
        probs = []
        for sample in splits[group][: args.per_group]:
            if sample.get("format") != "chat":
                continue
            ids = tok(render_chat(tok, sample), add_special_tokens=False).input_ids
            if close not in ids:
                continue
            i = ids.index(close)
            with torch.no_grad():
                logits = model(torch.tensor([ids[: i + 1]], device=device))
                logits = logits.logits[0, i - 1].float()
            p = torch.softmax(logits, -1)
            top = torch.topk(p, 1)
            probs.append(float(p[close]))
            rows.append({
                "group": group, "id": sample["id"],
                "p_close": round(float(p[close]), 6),
                "top_token": tok.decode([int(top.indices[0])]),
                "top_p": round(float(top.values[0]), 6),
                "argmax_is_close": int(top.indices[0]) == close,
            })
        if probs:
            per_group[group] = round(sum(probs) / len(probs), 6)

    if not rows:
        raise SystemExit("no chat samples with a </think> token were found")
    overall = sum(r["p_close"] for r in rows) / len(rows)
    closed = sum(r["argmax_is_close"] for r in rows)
    report = {
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "data_dir": args.data_dir,
        "split": args.split,
        "n": len(rows),
        # The headline: mean probability mass on the token CE demands and the
        # teacher forbids.
        "p_close_mean": round(overall, 6),
        # How often greedy decoding would actually emit it -- the bridge to the
        # behaviour metrics.
        "argmax_is_close_rate": round(closed / len(rows), 6),
        "per_group_p_close": per_group,
        "samples": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

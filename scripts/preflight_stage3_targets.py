"""CPU preflight for the Stage 3 teacher-target SFT warm-up.

Answers one question the pilot cannot be designed without: **can the training
data path carry a teacher-native target intact?**

The Stage 3 recipe packs with `concat` at `block_len` 1024, and `pack_blocks`
documents that "a sample may straddle a block boundary". For short public
targets that is a minor tax. Teacher-native targets carry a full reasoning
trace, so if they routinely exceed the block length the trainer is being fed
*fragments*: a second block that continues a trace whose beginning is not in
context, with no prompt attached. Nothing crashes and nothing is logged — the
loss just supervises a continuation the model cannot see the start of.

This script measures the real rendered-token distribution of both target kinds
and reports, per candidate block length, how many samples would be split by
`concat` and how many would be truncated by `best_fit`. It changes nothing.
"""
from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoTokenizer
from aadistill.data import encode_sample

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="artifacts/stage2_v2/pilot/targets.jsonl")
ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
ap.add_argument("--revision", default="768f209d9ea81521153ed38c47d515654e938aea")
ap.add_argument("--block-lens", default="1024,2048,4096,8192,16384")
ap.add_argument("--out", default=None)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
rows = [json.loads(l) for l in open(REPO_ROOT / args.targets)]

def q(v):
    v = sorted(v)
    return {"n": len(v), "min": v[0], "p50": int(statistics.median(v)),
            "p90": v[int(0.9 * len(v)) - 1], "max": v[-1],
            "mean": round(sum(v) / len(v), 1)}

groups = {}
for r in rows:
    ids, mask = encode_sample(tok, r)
    groups.setdefault(r["target_source"], []).append(
        {"id": r["id"], "slice": r["source"], "total": len(ids),
         "supervised": int(sum(mask))})

report = {"targets": args.targets, "model": args.model, "by_source": {}}
for source, items in sorted(groups.items()):
    lens = [i["total"] for i in items]
    report["by_source"][source] = {
        "total_tokens": q(lens),
        "supervised_tokens": q([i["supervised"] for i in items]),
        "block_len_fit": {},
    }
    print(f"\n=== {source} ({len(items)} samples) ===")
    print(f"  rendered tokens : {q(lens)}")
    print(f"  supervised span : {q([i['supervised'] for i in items])}")
    for bl in [int(b) for b in args.block_lens.split(",")]:
        over = sum(1 for x in lens if x > bl)
        # `concat` cuts a continuous stream every `bl` tokens. A sample is split
        # whenever it crosses a boundary; expected splits are total/bl minus the
        # number of samples, i.e. every sample longer than a block splits at
        # least once, and shorter ones split with probability len/bl.
        expected_split = sum(min(1.0, x / bl) for x in lens)
        report["by_source"][source]["block_len_fit"][bl] = {
            "over_block": over, "over_block_rate": round(over / len(lens), 4),
            "best_fit_truncated": over,
            "concat_expected_split_rate": round(expected_split / len(lens), 4),
        }
        print(f"  block_len {bl:>6}: {over:>3}/{len(lens)} exceed it "
              f"({over/len(lens):.1%})  | best_fit would truncate {over} "
              f"| concat expected split rate {expected_split/len(lens):.1%}")

if args.out:
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")

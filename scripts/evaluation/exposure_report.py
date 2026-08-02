"""Training-exposure accounting for the six Experiment 1 rungs.

    uv run python scripts/evaluation/exposure_report.py


Answers: how many blocks each rung holds, how many the trainer actually consumed,
how many times each block was repeated, and whether the design isolates
supervised-token quantity or moves quantity and exposure together.
"""

import json
from pathlib import Path

LADDER = json.loads(Path("artifacts/stage3/ladder_uniform_probe/ladder.json").read_text())
BLOCK_LEN = LADDER["block_len"]

rows = []
for rung in LADDER["rungs"]:
    if not rung.get("reachable"):
        continue
    target = rung["target_supervised_tokens"]
    tag = {250_000: "0250k", 460_000: "0460k", 860_000: "0860k",
           1_600_000: "1600k", 2_960_000: "2960k", 5_500_000: "5500k"}[target]
    cfg = json.loads(Path(f"configs/stage3/e1/e1_r{tag}_sa_pca.json").read_text())
    steps = cfg["schedule"]["total_steps"]
    bps = cfg["batch"]["blocks_per_step"]

    avail_blocks = rung["n_blocks"]
    avail_sup = rung["actual_supervised_tokens"]
    consumed_blocks = steps * bps
    epochs = consumed_blocks / avail_blocks
    consumed_sup = avail_sup * epochs
    # Every block is padded to block_len, so processed tokens count padding too.
    processed_tokens = consumed_blocks * BLOCK_LEN
    real_tokens = rung["real_tokens"] * epochs

    # Block reuse: the stream is epoch-permutation based, so blocks are consumed
    # uniformly; the remainder (if any) is one extra partial epoch.
    full_reps, remainder = divmod(consumed_blocks, avail_blocks)
    rows.append(dict(tag=tag, target=target, blocks=avail_blocks, sup=avail_sup,
                     steps=steps, consumed_blocks=consumed_blocks, epochs=epochs,
                     consumed_sup=consumed_sup, processed=processed_tokens,
                     real=real_tokens, full_reps=full_reps, remainder=remainder))

hdr = (f"{'rung':>6s} {'blocks':>7s} {'supervised':>11s} {'steps':>6s} "
       f"{'blocks used':>12s} {'sup. used':>12s} {'eff. epochs':>11s} {'repeats':>18s}")
print(hdr); print("-" * len(hdr))
for r in rows:
    rep = f"{r['full_reps']}x all" + (f" +{r['remainder']} blk" if r['remainder'] else "")
    print(f"{r['tag']:>6s} {r['blocks']:7,d} {r['sup']:11,d} {r['steps']:6,d} "
          f"{r['consumed_blocks']:12,d} {int(r['consumed_sup']):12,d} "
          f"{r['epochs']:11.4f} {rep:>18s}")

print()
lo, hi = rows[0], rows[-1]
print(f"unique supervised tokens span : {hi['sup']/lo['sup']:.1f}x "
      f"({lo['sup']:,} -> {hi['sup']:,})")
print(f"optimizer steps span          : {hi['steps']/lo['steps']:.1f}x "
      f"({lo['steps']:,} -> {hi['steps']:,})")
print(f"tokens processed span         : {hi['processed']/lo['processed']:.1f}x "
      f"({lo['processed']:,} -> {hi['processed']:,})")
print(f"effective epochs span         : {lo['epochs']:.4f} -> {hi['epochs']:.4f} "
      f"(ratio {hi['epochs']/lo['epochs']:.4f})")

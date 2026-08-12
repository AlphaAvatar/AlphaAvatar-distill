#!/usr/bin/env python3
"""Are the per-step memory-driving shapes constant across the 1,761-step stream?

    PYTHONPATH=src python scripts/training/audit_stream_shapes.py \
        --out logs/e8b_stream_shape_audit.json

Zero GPU cost. The block order is `stream_block_indices(n_blocks, seed, step*bps, bps)`
— a pure function of the seed and the step — so the exact stream every arm will see is
reconstructible on CPU from the real pack, the real masks and the real seed derivation.

Why this matters. DP-sa completed 1,761 steps and DC-sa OOM'd at ~step 900, both having
peaked at the identical 77.45 GiB. The logged `gpu_mem_gb` is
`torch.cuda.max_memory_allocated()`, a **running maximum that is never reset**, so its
rise from 76.50 to 77.45 is non-decreasing *by construction* and is not by itself
evidence of a leak. Two hypotheses survive that observation:

  A. data-dependent shapes — CE/KD select masked positions, so the transient footprint
     varies per block, and the running max simply rises until the worst block appears;
  B. a memory-lifecycle defect — something retained across steps.

The discriminator is here, in the shapes, and it costs nothing to compute.

Reported per step and per microbatch: block indices, logical length, executed
(non-padding) extent, `ce_targets`, `kd_positions`, supervised and content token
counts, and the implied byte sizes of the five buffers that dominate the step.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.training.train import (  # noqa: E402
    prediction_mask, stream_block_indices,
)

VOCAB = 151_936
ARMS = {
    "DP-sa": "configs/stage3/e8b/e8b_dp_r1600k_sa.json",
    "DC-sa": "configs/stage3/e8b/e8b_dc_r1600k_sa.json",
    "DP-sb": "configs/stage3/e8b/e8b_dp_r1600k_sb.json",
    "DC-sb": "configs/stage3/e8b/e8b_dc_r1600k_sb.json",
}
WINDOWS = {"first_20": (0, 20), "first_200": (0, 200), "first_310": (0, 310),
           "around_dc_oom_850_950": (850, 950), "full_stream": (0, None)}


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def buffers(ce_targets: int, kd_positions: int, executed: int, chunk: int) -> dict:
    """The five buffers the audit tracks, in MB, from the shapes alone."""
    return {
        "ce_selected_bf16_mb": round(ce_targets * VOCAB * 2 / 1e6, 1),
        "ce_fp32_upcast_mb": round(ce_targets * VOCAB * 4 / 1e6, 1),
        "kd_student_selected_bf16_mb": round(kd_positions * VOCAB * 2 / 1e6, 1),
        "kd_teacher_selected_bf16_mb": round(kd_positions * VOCAB * 2 / 1e6, 1),
        "kd_chunk_workspace_fp32_mb": round(3 * min(chunk, max(kd_positions, 1))
                                            * VOCAB * 4 / 1e6, 1),
        "logits_bf16_each_mb": round(executed * VOCAB * 2 / 1e6, 1),
    }


def audit_arm(name: str, cfg_path: Path) -> dict:
    cfg = json.loads(cfg_path.read_text())
    # The pod copies `ladder_uniform_probe` -> `ladder_uniform` during setup, so on
    # the dev box the probe pack IS the same bytes (its blocks.npz sha256 is pinned in
    # e8b_setup.sh). Fall back to it rather than fail the audit for a naming artifact.
    pack = REPO_ROOT / cfg["data_dir"]
    if not (pack / "ladder.json").is_file():
        pack = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
    train, _, _ = ladder_blocks(pack, cfg["rung"], n_val=cfg["val_blocks"])
    ids_all, mask_all, content_all = (train + (None,))[:3] if len(train) == 2 else train
    n_blocks = int(ids_all.shape[0])
    bps = cfg["batch"]["blocks_per_step"]
    micro = cfg["batch"]["micro_blocks"]
    chunk = cfg["loss"].get("kd_chunk", 512)
    scope = cfg["loss"]["kd_scope"]
    total_steps = cfg["schedule"]["total_steps"]

    rows = []
    for step in range(total_steps):
        idxs = stream_block_indices(n_blocks, cfg["seed"], step * bps, bps)
        for m0 in range(0, bps, micro):
            sel = idxs[m0:m0 + micro]
            ids = ids_all[sel]
            mask = mask_all[sel]
            content = None if content_all is None else content_all[sel]
            ce_targets = int(mask[:, 1:].sum())
            kd_positions = int(prediction_mask(mask, scope, content,
                                               input_ids=ids).sum())
            # Executed extent: non-padding tokens actually forwarded. With the ladder
            # pack the tensor is dense, so this is the content mask when present and
            # the full width otherwise.
            executed = int(ids.numel())
            nonpad = int(content.sum()) if content is not None else executed
            rows.append({
                "step": step, "blocks": [int(i) for i in sel],
                "logical_len": int(ids.shape[1]),
                "executed_extent": executed, "nonpad_tokens": nonpad,
                "ce_targets": ce_targets, "kd_positions": kd_positions,
                "supervised_tokens": ce_targets,
                **buffers(ce_targets, kd_positions, executed, chunk)})
    return {"arm": name, "config": str(cfg_path.relative_to(REPO_ROOT)),
            "seed": cfg["seed"], "n_blocks": n_blocks, "kd_chunk": chunk,
            "blocks_per_step": bps, "micro_blocks": micro,
            "total_steps": total_steps, "rows": rows}


def stats(rows, key):
    vals = sorted(r[key] for r in rows)
    return {"min": vals[0], "mean": round(sum(vals) / len(vals), 1),
            "p90": pct(vals, 0.90), "p99": pct(vals, 0.99), "max": vals[-1],
            "distinct_values": len(set(vals))}


def summarize(a: dict) -> dict:
    rows = a["rows"]
    out = {"arm": a["arm"], "seed": a["seed"], "kd_chunk": a["kd_chunk"],
           "microbatches": len(rows)}
    for k in ("ce_targets", "kd_positions", "executed_extent", "nonpad_tokens"):
        out[k] = stats(rows, k)
    # Where the worst cases actually are.
    for k in ("ce_targets", "kd_positions", "executed_extent"):
        worst = max(rows, key=lambda r: r[k])
        out[f"worst_{k}"] = {"step": worst["step"], "blocks": worst["blocks"],
                             "value": worst[k]}
    # The joint transient: what actually has to coexist.
    def transient(r):
        return (r["ce_fp32_upcast_mb"] + r["ce_selected_bf16_mb"]
                + 2 * r["kd_student_selected_bf16_mb"]
                + r["kd_chunk_workspace_fp32_mb"] + 2 * r["logits_bf16_each_mb"])
    worst = max(rows, key=transient)
    out["worst_joint_transient"] = {
        "step": worst["step"], "blocks": worst["blocks"],
        "estimated_transient_mb": round(transient(worst), 1),
        "ce_targets": worst["ce_targets"], "kd_positions": worst["kd_positions"]}
    out["windows"] = {}
    for label, (lo, hi) in WINDOWS.items():
        win = [r for r in rows if lo <= r["step"] < (hi if hi else 10 ** 9)]
        if not win:
            continue
        out["windows"][label] = {
            "steps": [lo, hi], "n": len(win),
            "max_ce_targets": max(r["ce_targets"] for r in win),
            "max_kd_positions": max(r["kd_positions"] for r in win),
            "max_executed_extent": max(r["executed_extent"] for r in win),
            "max_estimated_transient_mb": round(max(transient(r) for r in win), 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/e8b_stream_shape_audit.json")
    ap.add_argument("--arms", default="DP-sa,DC-sa,DP-sb,DC-sb")
    args = ap.parse_args()

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    audits = {n: audit_arm(n, REPO_ROOT / ARMS[n]) for n in wanted}
    summaries = {n: summarize(a) for n, a in audits.items()}

    # DP vs DC must be shape-identical: same pack, same seed, same masks, same order.
    pairs = {}
    for seed_tag in ("sa", "sb"):
        dp, dc = f"DP-{seed_tag}", f"DC-{seed_tag}"
        if dp not in audits or dc not in audits:
            continue
        keys = ("step", "blocks", "ce_targets", "kd_positions",
                "executed_extent", "nonpad_tokens")
        a = [tuple(map(str, (r[k] for k in keys))) for r in audits[dp]["rows"]]
        b = [tuple(map(str, (r[k] for k in keys))) for r in audits[dc]["rows"]]
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        pairs[f"{dp}_vs_{dc}"] = {
            "identical": (a == b),
            "n_microbatches": len(a),
            "first_differing_indices": diffs[:5],
            "note": "same architecture, pack, seed, masks and order — a difference "
                    "here would be a bug, not a workload property"}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "vocab": VOCAB,
        "gpu_mem_metric_note":
            "the logged gpu_mem_gb is torch.cuda.max_memory_allocated(), a running "
            "maximum that is never reset, so a non-decreasing series is guaranteed by "
            "construction and is NOT evidence of a leak",
        "summaries": summaries,
        "pair_shape_identity": pairs,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")

    for n, s in summaries.items():
        print(f"\n=== {n} (seed {s['seed']}, {s['microbatches']} microbatches) ===")
        for k in ("ce_targets", "kd_positions", "executed_extent"):
            v = s[k]
            print(f"  {k:18s} min {v['min']:>7} mean {v['mean']:>9} p90 {v['p90']:>7} "
                  f"p99 {v['p99']:>7} max {v['max']:>7}  distinct={v['distinct_values']}")
        w = s["worst_joint_transient"]
        print(f"  worst joint transient at step {w['step']} blocks {w['blocks']}: "
              f"{w['estimated_transient_mb']} MB")
        print("  window maxima (ce / kd / executed / transient MB):")
        for label, win in s["windows"].items():
            print(f"    {label:24s} {win['max_ce_targets']:>7} "
                  f"{win['max_kd_positions']:>7} {win['max_executed_extent']:>7} "
                  f"{win['max_estimated_transient_mb']:>9}")
    for k, v in pairs.items():
        print(f"\n{k}: shapes identical = {v['identical']} "
              f"over {v['n_microbatches']} microbatches")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

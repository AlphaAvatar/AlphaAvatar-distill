#!/usr/bin/env python
"""Pre-launch numerical-safety check for a loss-weight change. CPU, no updates.

    PYTHONPATH=src python scripts/training/diagnose_loss_weights.py \
        --student artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
        --teacher Qwen/Qwen3-4B-Thinking-2507@<rev> \
        --pack artifacts/stage3/ladder_uniform_probe --rung 860000 \
        --blocks 4 --out artifacts/audit/loss_weight_diagnostic.json

Records, on **fixed hashed initial batches** and from the exact Stage 1
initialization both arms start from:

* each loss term separately (CE sum/mean, KD sum/mean) and the weighted scalars;
* the total loss under each weight setting;
* per-setting gradient global norms;
* the **cosine similarity between the two settings' gradient vectors**.

This is a safety readout, not a tuning loop. It answers "is the treatment
numerically sane and does it actually point somewhere different?" — nothing here
feeds back into the recipe, and `optimizer.step()` is never called.

The gradient cosine is the informative number. Both settings mix the same two
terms, so a cosine near 1.0 would mean the reweighting barely changes the descent
direction at initialization and the experiment would be unlikely to separate;
a markedly lower cosine means the two recipes genuinely disagree about where to
go from the shared start point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.models.student import assert_rope_matches_config  # noqa: E402

SETTINGS = {
    "baseline_kd1.0_ce0.25": {"kd_weight": 1.0, "ce_weight": 0.25},
    "treatment_kd0.25_ce1.0": {"kd_weight": 0.25, "ce_weight": 1.0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--rung", type=int, default=860000)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/stage3/e1/e1_r0860k_sa_pca.json")
    ap.add_argument("--seq", type=int, default=0,
                    help="optional truncation of each block, for CPU feasibility")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from aadistill.training.train import (
        kd_forward_kl, masked_ce, prediction_mask, select_trainable,
    )

    cfg = json.loads(args.config.read_text())
    tau = cfg["loss"]["kd_temperature"]

    scfg = AutoConfig.from_pretrained(args.student)
    rp = getattr(scfg, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        scfg.rope_theta = float(rp["rope_theta"])
    student = AutoModelForCausalLM.from_pretrained(
        args.student, config=scfg, dtype=torch.float32).eval()
    print(f"student rope base {assert_rope_matches_config(student, scfg):,.0f}")
    tid, _, trev = args.teacher.partition("@")
    # bfloat16 teacher, matching training (`teacher.dtype: bfloat16` in every
    # run manifest). float32 would need ~16 GB and does not match the recipe.
    teacher = AutoModelForCausalLM.from_pretrained(
        tid, revision=trev or None, dtype=torch.bfloat16).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    arrays = np.load(args.pack / "blocks.npz")
    meta = json.loads((args.pack / "ladder.json").read_text())
    n_rung = next(r["n_blocks"] for r in meta["rungs"]
                  if r["target_supervised_tokens"] == args.rung)
    n = min(args.blocks, n_rung)
    T = args.seq or arrays["input_ids"].shape[1]
    ids_all = arrays["input_ids"][:n, :T]
    ce_all = arrays["ce_mask"][:n, :T].astype(bool)
    ct_all = arrays["content_mask"][:n, :T].astype(bool)
    batch_hash = hashlib.sha256(ids_all.tobytes()).hexdigest()
    print(f"{n} fixed blocks (first {n} of the {n_rung}-block rung), "
          f"seq {T}; batch sha256 {batch_hash[:16]}…")

    # select_trainable returns a REPORT and sets requires_grad as a side effect;
    # the parameters come from the model afterwards.
    report = select_trainable(student, cfg["trainable_patterns"])
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in params):,} trainable parameters "
          f"across {len(params)} tensors")

    # Term sums are weight-independent, so compute the graph once per block and
    # reuse it for both settings via two separate backward passes.
    per_setting = {}
    for name, w in SETTINGS.items():
        student.zero_grad(set_to_none=True)
        ce_sum_t = kd_sum_t = 0.0
        ce_n = kd_n = 0
        for b in range(n):
            ids = torch.tensor(ids_all[b:b + 1], dtype=torch.long)
            mask = torch.tensor(ce_all[b:b + 1])
            content = torch.tensor(ct_all[b:b + 1])
            pm = prediction_mask(mask, cfg["loss"]["kd_scope"], content)
            s_log = student(ids).logits
            with torch.no_grad():
                t_log = teacher(ids).logits.float()
            ce_s, cn = masked_ce(s_log, ids, mask)
            kd_s, kn = kd_forward_kl(s_log, t_log, pm, tau)
            ce_n += cn; kd_n += kn
            ce_sum_t += float(ce_s.detach()); kd_sum_t += float(kd_s.detach())
            # normalizers are the FULL-batch counts in training; here the fixed
            # batch is the unit, so use its own totals -- stated, not hidden.
            loss = torch.zeros(())
            if cn:
                loss = loss + w["ce_weight"] * ce_s / max(cn, 1)
            if kn:
                loss = loss + w["kd_weight"] * kd_s / max(kn, 1)
            loss.backward()
            del s_log, t_log
            print(f"  {name}: block {b+1}/{n}", flush=True)
        # Do NOT materialise a flat float64 copy: 440M params x 8 bytes is 3.5 GB
        # per setting, and holding two of them OOMs a 30 GB box. Keep fp32 clones
        # per tensor and accumulate the cosine pairwise below.
        grads = {i: p.grad.detach().clone() for i, p in enumerate(params)
                 if p.grad is not None}
        gnorm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())))
        per_setting[name] = {
            "weights": w,
            "ce_targets": ce_n, "kd_positions": kd_n,
            "ce_sum": round(ce_sum_t, 4), "kd_sum": round(kd_sum_t, 4),
            "ce_mean_per_token": round(ce_sum_t / max(ce_n, 1), 6),
            "kd_mean_per_token": round(kd_sum_t / max(kd_n, 1), 6),
            "ce_scalar": round(w["ce_weight"] * ce_sum_t / max(ce_n, 1), 6),
            "kd_scalar": round(w["kd_weight"] * kd_sum_t / max(kd_n, 1), 6),
            "total_loss": round(w["ce_weight"] * ce_sum_t / max(ce_n, 1)
                                + w["kd_weight"] * kd_sum_t / max(kd_n, 1), 6),
            "grad_global_norm": round(gnorm, 6),
            "_grads": grads, "_norm": gnorm,
        }
        student.zero_grad(set_to_none=True)      # never stepped

    names = list(SETTINGS)
    A = per_setting[names[0]].pop("_grads"); na = per_setting[names[0]].pop("_norm")
    B = per_setting[names[1]].pop("_grads"); nb = per_setting[names[1]].pop("_norm")
    dot = 0.0
    for k in sorted(set(A) & set(B)):
        dot += float((A[k].double() * B[k].double()).sum())
        A[k] = None; B[k] = None            # free as we go
    cos = dot / (na * nb) if na and nb else float("nan")
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "student": args.student, "teacher": args.teacher,
        "batch": {"blocks": n, "seq_len": T, "sha256": batch_hash,
                  "source": f"first {n} blocks of the {args.rung} rung"},
        "kd_scope": cfg["loss"]["kd_scope"], "kd_temperature": tau,
        "trainable_params": sum(p.numel() for p in params),
        "settings": per_setting,
        "gradient_cosine_between_settings": round(cos, 8),
        "grad_norm_ratio_treatment_over_baseline": round(nb / na, 6),
        "grad_dot_product": dot,
        "optimizer_step_called": False,
        "libraries": library_versions(), "code_state": code_state(REPO_ROOT),
        "note": ("Diagnostic only. No weight was modified and no step was taken; "
                 "nothing here was fed back into the recipe."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"\n{'setting':26s}{'CE/tok':>10s}{'KD/tok':>10s}{'CE scalar':>11s}"
          f"{'KD scalar':>11s}{'total':>10s}{'|grad|':>12s}")
    for k, v in per_setting.items():
        print(f"{k:26s}{v['ce_mean_per_token']:>10.4f}{v['kd_mean_per_token']:>10.4f}"
              f"{v['ce_scalar']:>11.4f}{v['kd_scalar']:>11.4f}"
              f"{v['total_loss']:>10.4f}{v['grad_global_norm']:>12.6f}")
    print(f"\ngradient cosine between settings : {cos:.8f}")
    print(f"grad-norm ratio treatment/baseline: "
          f"{out['grad_norm_ratio_treatment_over_baseline']}")


if __name__ == "__main__":
    main()

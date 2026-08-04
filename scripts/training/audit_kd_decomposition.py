#!/usr/bin/env python
"""Decompose the implemented `kd_scope="all"` objective by token role. No updates.

    PYTHONPATH=src python scripts/training/audit_kd_decomposition.py \
        --student <ckpt> --teacher Qwen/Qwen3-4B-Thinking-2507@<rev> \
        --pack artifacts/stage3/ladder_uniform_probe --rung 860000 \
        --blocks 32 --out artifacts/audit/kd_decomposition_sa.json

`optimizer.step()` is never called and no weight is written. The gradient probe
runs `backward()` twice on the same batch with `zero_grad()` between, reads the
gradients, and discards them.

The implemented loss, read off `training/train.py` rather than restated from
memory
-------------------------------------------------------------------------------
Per optimizer step over `blocks_per_step` blocks:

    ce_total = mask[:, 1:].sum()                     # assistant targets in the step
    kd_total = prediction_mask(mask, "all", content).sum()
             = content[:, 1:].sum()                  # EVERY real target in the step

    ce_sum   = sum_i CE(student_logits[:, :-1][i], input_ids[:, 1:][i])   over ce mask
    kd_sum   = tau^2 * sum_i KL(teacher_tau || student_tau)               over kd mask

    loss     = ce_weight * ce_sum / ce_total + kd_weight * kd_sum / kd_total

Both masks are indexed on the **target** token after the causal shift: entry `i`
of `mask[:, 1:]` scores the prediction of `input_ids[:, i+1]`. Role labels here
are therefore assigned to the token being predicted, not to the position doing
the predicting — getting that backwards would shift every role by one and blame
`</think>` for the token before it.

"KD applies to every real token" is checked literally: the KD mask is compared
against the content mask, and the prompt/context share is reported separately
from the assistant share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.models.student import assert_rope_matches_config  # noqa: E402

ROLES = ("prompt_context", "reasoning", "think_close", "answer_content",
         "im_end", "excluded_padding")


def role_labels(ids: np.ndarray, ce: np.ndarray, content: np.ndarray,
                think_close: int, im_end: int) -> np.ndarray:
    """Label every TARGET position of one block.

    Reasoning vs answer is split at the structural `</think>` inside each CE
    span — inside a supervised assistant turn that close is the template's, and
    the span boundaries come from the mask rather than from a text search.
    """
    T = ids.shape[0]
    out = np.full(T, "prompt_context", dtype=object)
    out[~content] = "excluded_padding"
    i = 0
    while i < T:
        if ce[i]:
            j = i
            while j + 1 < T and ce[j + 1]:
                j += 1
            close = None
            for k in range(i, j + 1):
                if ids[k] == think_close:
                    close = k
                    break
            for k in range(i, j + 1):
                if ids[k] == think_close:
                    out[k] = "think_close"
                elif ids[k] == im_end:
                    out[k] = "im_end"
                elif close is not None and k > close:
                    out[k] = "answer_content"
                else:
                    out[k] = "reasoning"
            i = j + 1
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--rung", type=int, default=860000)
    ap.add_argument("--blocks", type=int, default=32,
                    help="how many rung blocks to accumulate over")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/stage3/e1/e1_r0860k_sa_pca.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad-probe-blocks", type=int, default=4)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from aadistill.training.train import (
        kd_forward_kl, masked_ce, prediction_mask, select_trainable,
    )

    cfg = json.loads(args.config.read_text())
    loss_cfg = cfg["loss"]
    tokzr = AutoTokenizer.from_pretrained(
        args.teacher.split("@")[0], revision=(args.teacher.split("@") + [None])[1])
    think_close = tokzr.convert_tokens_to_ids("</think>")
    im_end = tokzr.convert_tokens_to_ids("<|im_end|>")

    scfg = AutoConfig.from_pretrained(args.student)
    rp = getattr(scfg, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        scfg.rope_theta = float(rp["rope_theta"])
    student = AutoModelForCausalLM.from_pretrained(
        args.student, config=scfg, dtype=torch.float32).to(args.device).eval()
    print("student rope base:", f"{assert_rope_matches_config(student, scfg):,.0f}")
    tid, trev = (args.teacher.split("@") + [None])[:2]
    teacher = AutoModelForCausalLM.from_pretrained(
        tid, revision=trev, dtype=torch.bfloat16).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    arrays = np.load(args.pack / "blocks.npz")
    meta = json.loads((args.pack / "ladder.json").read_text())
    n_rung = next(r["n_blocks"] for r in meta["rungs"]
                  if r["target_supervised_tokens"] == args.rung)
    n = min(args.blocks, n_rung)
    ids_all = arrays["input_ids"][:n]
    ce_all = arrays["ce_mask"][:n].astype(bool)
    ct_all = arrays["content_mask"][:n].astype(bool)
    batch_hash = hashlib.sha256(ids_all.tobytes()).hexdigest()
    print(f"{n} blocks of the {n_rung}-block rung; batch sha256 {batch_hash[:16]}…")

    kd_mass = defaultdict(float); kd_count = defaultdict(int)
    ce_mass = defaultdict(float); ce_count = defaultdict(int)
    kd_total = ce_total = 0
    kd_equals_content = True

    for b in range(n):
        ids = torch.tensor(ids_all[b:b + 1], dtype=torch.long, device=args.device)
        mask = torch.tensor(ce_all[b:b + 1], device=args.device)
        content = torch.tensor(ct_all[b:b + 1], device=args.device)
        pm = prediction_mask(mask, "all", content)
        kd_equals_content &= bool(torch.equal(pm, content[:, 1:]))
        kd_total += int(pm.sum()); ce_total += int(mask[:, 1:].sum())

        with torch.no_grad():
            s_log = student(ids).logits
            t_log = teacher(ids).logits.float()
        labels = role_labels(ids_all[b], ce_all[b], ct_all[b], think_close, im_end)
        shifted = labels[1:]                       # causal shift: target tokens

        for role in ROLES:
            sel = torch.tensor((shifted == role), device=args.device)[None, :]
            sel_kd = sel & pm
            if int(sel_kd.sum()):
                v, c = kd_forward_kl(s_log, t_log, sel_kd,
                                     loss_cfg["kd_temperature"])
                kd_mass[role] += float(v); kd_count[role] += c
            sel_ce = torch.zeros_like(mask)
            sel_ce[:, 1:] = sel
            sel_ce &= mask
            if int(sel_ce.sum()):
                v, c = masked_ce(s_log, ids, sel_ce)
                ce_mass[role] += float(v); ce_count[role] += c
        if (b + 1) % 8 == 0:
            print(f"  {b+1}/{n}", flush=True)

    kd_sum = sum(kd_mass.values()); ce_sum = sum(ce_mass.values())
    partitions = {}
    for role in ROLES:
        partitions[role] = {
            "kd_tokens": kd_count[role],
            "kd_token_share": round(kd_count[role] / kd_total, 6) if kd_total else 0,
            "kd_loss_mass": round(kd_mass[role], 4),
            "kd_mass_share": round(kd_mass[role] / kd_sum, 6) if kd_sum else 0,
            "kd_mean_per_token": (round(kd_mass[role] / kd_count[role], 6)
                                  if kd_count[role] else None),
            "kd_scalar_contribution": (
                round(loss_cfg["kd_weight"] * kd_mass[role] / kd_total, 6)
                if kd_total else None),
            "ce_tokens": ce_count[role],
            "ce_loss_mass": round(ce_mass[role], 4),
            "ce_mean_per_token": (round(ce_mass[role] / ce_count[role], 6)
                                  if ce_count[role] else None),
            "ce_scalar_contribution": (
                round(loss_cfg["ce_weight"] * ce_mass[role] / ce_total, 6)
                if ce_total else None),
        }

    # ---- no-update gradient probe ---------------------------------------
    probe = None
    if args.grad_probe_blocks > 0:
        trainable = select_trainable(student, cfg["trainable_patterns"])
        params = [p for p in trainable.values()]
        g = args.grad_probe_blocks

        def grads_for(which: str):
            student.zero_grad(set_to_none=True)
            for b in range(min(g, n)):
                ids = torch.tensor(ids_all[b:b + 1], dtype=torch.long,
                                   device=args.device)
                mask = torch.tensor(ce_all[b:b + 1], device=args.device)
                content = torch.tensor(ct_all[b:b + 1], device=args.device)
                pm = prediction_mask(mask, "all", content)
                labels = role_labels(ids_all[b], ce_all[b], ct_all[b],
                                     think_close, im_end)[1:]
                assistant = np.isin(labels, ["reasoning", "think_close",
                                             "answer_content", "im_end"])
                sel = torch.tensor(assistant if which == "assistant"
                                   else (labels == "prompt_context"),
                                   device=args.device)[None, :] & pm
                if not int(sel.sum()):
                    continue
                s_log = student(ids).logits
                with torch.no_grad():
                    t_log = teacher(ids).logits.float()
                v, _ = kd_forward_kl(s_log, t_log, sel, loss_cfg["kd_temperature"])
                (loss_cfg["kd_weight"] * v / max(kd_total, 1)).backward()
            return torch.cat([p.grad.detach().flatten() for p in params
                              if p.grad is not None])

        gp = grads_for("prompt_context")
        ga = grads_for("assistant")
        student.zero_grad(set_to_none=True)      # never stepped
        probe = {
            "blocks": min(g, n),
            "prompt_context_grad_norm": round(float(gp.norm()), 6),
            "assistant_grad_norm": round(float(ga.norm()), 6),
            "cosine_similarity": round(float(
                torch.nn.functional.cosine_similarity(
                    gp.double()[None], ga.double()[None])), 6),
            "norm_ratio_prompt_over_assistant": round(
                float(gp.norm() / ga.norm()), 6) if float(ga.norm()) else None,
            "optimizer_step_called": False,
        }

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label,
        "student": args.student, "teacher": args.teacher,
        "pack": str(args.pack), "rung": args.rung,
        "blocks_measured": n, "rung_blocks": n_rung,
        "batch_sha256": batch_hash,
        "loss_config": loss_cfg,
        "implemented_equation": (
            "loss = ce_w * sum_CE(ce_mask[:,1:]) / ce_mask[:,1:].sum() "
            "+ kd_w * tau^2 * sum_KL(teacher_tau||student_tau over content[:,1:]) "
            "/ content[:,1:].sum(); masks index the TARGET token after the causal "
            "shift; KL is forward KL on the full 151,936-way vocabulary, chunked "
            "512 positions at a time and reduced in float32."),
        "kd_mask_is_exactly_every_real_token": kd_equals_content,
        "denominators": {"ce_total": ce_total, "kd_total": kd_total},
        "totals": {"kd_loss_mass": round(kd_sum, 4),
                   "ce_loss_mass": round(ce_sum, 4)},
        "partitions": partitions,
        "gradient_probe": probe,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
        "note": "No optimizer.step() was called; no weight was modified.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"\nKD mask == every real token: {kd_equals_content}")
    print(f"denominators: ce_total {ce_total:,}  kd_total {kd_total:,}")
    print(f"\n{'role':18s} {'kd tok':>9s} {'tok%':>7s} {'kd mass':>12s} {'mass%':>7s} "
          f"{'mean/tok':>9s} {'scalar':>9s}")
    for role in ROLES:
        p = partitions[role]
        print(f"{role:18s} {p['kd_tokens']:>9,} {p['kd_token_share']*100:>6.2f}% "
              f"{p['kd_loss_mass']:>12,.1f} {p['kd_mass_share']*100:>6.2f}% "
              f"{str(p['kd_mean_per_token']):>9s} {str(p['kd_scalar_contribution']):>9s}")
    if probe:
        print(f"\ngradient probe ({probe['blocks']} blocks, no step): "
              f"prompt |g| {probe['prompt_context_grad_norm']}  "
              f"assistant |g| {probe['assistant_grad_norm']}  "
              f"cos {probe['cosine_similarity']}")


if __name__ == "__main__":
    main()

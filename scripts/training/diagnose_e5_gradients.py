#!/usr/bin/env python
"""Gradient attribution for E5's CE and KD terms, on real C and R batches.

    PYTHONPATH=src python scripts/training/diagnose_e5_gradients.py \
        --examples artifacts/stage3/e5_pilot/{c,r}_examples.jsonl \
        --student <ckpt> --teacher <id@rev> --out artifacts/audit/e5_gradients.json

A weighted KD term of 0.0064 against a weighted CE term of 7.60 looks like a
rounding error, but a *scalar loss share is not a gradient share*: CE and KL have
different curvature and different numerical scales, and a term with a tiny value
can still steer the update. So this measures what actually reaches the
parameters.

Method, and the one decision that matters:

* each term is backwarded **separately** into a cleared grad buffer, and the
  resulting gradient is flattened and stored, so norms and cosines are computed
  on the real thing rather than inferred;
* the prefix-only and continuation-only KD pieces keep the **production
  denominator** (`kd_total`, all non-padding prediction positions) rather than
  being re-normalized over their own token counts. Re-normalizing would measure a
  different objective; keeping it means the two pieces sum to the KD term the
  trainer actually optimizes.

The formal objective is untouched: no optimizer step is taken and no weight is
written. Only backward passes, whose gradients are read and discarded.
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

from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.training.train import (  # noqa: E402
    kd_forward_kl, masked_ce, prediction_mask, select_trainable,
)

GROUPS = ("ffn", "attn_proj", "norm", "other")


def group_of(name: str) -> str:
    if ".mlp." in name:
        return "ffn"
    if any(f".self_attn.{p}." in name for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attn_proj"
    if "norm" in name.lower() or "layernorm" in name:
        return "norm"
    return "other"


def grad_snapshot(model) -> tuple[torch.Tensor, dict]:
    """Flattened trainable gradient plus per-group norms."""
    flat, per_group = [], {g: 0.0 for g in GROUPS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad.detach().float().flatten() if p.grad is not None else \
            torch.zeros(p.numel())
        flat.append(g)
        per_group[group_of(name)] += float(g.pow(2).sum())
    vec = torch.cat(flat) if flat else torch.zeros(1)
    return vec, {g: round(v ** 0.5, 6) for g, v in per_group.items()}


def backward_term(model, term: torch.Tensor) -> tuple[float, torch.Tensor, dict]:
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    vec, groups = grad_snapshot(model)
    return round(float(vec.norm()), 6), vec, groups


def analyse(student, teacher, ids, ce_mask, content, *, ce_w, kd_w, temperature):
    """Loss and gradient attribution for one batch, on the production loss code."""
    logits = student(ids).logits
    with torch.no_grad():
        t_logits = teacher(ids).logits

    kd_all = prediction_mask(ce_mask, "all", content)
    ce_pred = ce_mask[:, 1:]
    kd_cont = kd_all & ce_pred
    kd_pre = kd_all & ~ce_pred

    ce_sum, ce_n = masked_ce(logits, ids, ce_mask)
    kd_sum, kd_n = kd_forward_kl(logits, t_logits, kd_all, temperature)
    kd_pre_sum, n_pre = kd_forward_kl(logits, t_logits, kd_pre, temperature)
    kd_con_sum, n_con = kd_forward_kl(logits, t_logits, kd_cont, temperature)

    # Production denominators. The pieces deliberately share kd_n so they sum to
    # the KD term the trainer optimizes; re-normalizing each piece over its own
    # token count would measure a different objective.
    w_ce = ce_w * ce_sum / max(1, ce_n)
    w_kd = kd_w * kd_sum / max(1, kd_n)
    w_kd_pre = kd_w * kd_pre_sum / max(1, kd_n)
    w_kd_con = kd_w * kd_con_sum / max(1, kd_n)

    v_ce, v_kd = float(w_ce.detach()), float(w_kd.detach())
    v_pre, v_con = float(w_kd_pre.detach()), float(w_kd_con.detach())
    ce_norm, ce_vec, ce_groups = backward_term(student, w_ce)
    kd_norm, kd_vec, kd_groups = backward_term(student, w_kd)
    pre_norm, _, pre_groups = backward_term(student, w_kd_pre)
    con_norm, _, con_groups = backward_term(student, w_kd_con)
    student.zero_grad(set_to_none=True)

    cos = float(torch.nn.functional.cosine_similarity(
        ce_vec.unsqueeze(0), kd_vec.unsqueeze(0)).item())
    return {
        "tokens": {"ce_mask": int(ce_n), "kd_mask": int(kd_n),
                   "kd_prefix": int(n_pre), "kd_continuation": int(n_con),
                   "nonpadding": int(content.sum()),
                   "kd_per_ce_token": round(kd_n / max(1, ce_n), 4)},
        "loss": {
            "weighted_ce": round(v_ce, 6),
            "weighted_kd_total": round(v_kd, 6),
            "weighted_kd_prefix": round(v_pre, 6),
            "weighted_kd_continuation": round(v_con, 6),
            "kd_share_of_total_loss": round(v_kd / max(1e-12, v_ce + v_kd), 6),
            "prefix_kd_share_of_total_loss": round(
                v_pre / max(1e-12, v_ce + v_kd), 6),
        },
        "gradient": {
            "weighted_ce_norm": ce_norm,
            "weighted_kd_total_norm": kd_norm,
            "weighted_kd_prefix_norm": pre_norm,
            "weighted_kd_continuation_norm": con_norm,
            "kd_to_ce_norm_ratio": round(kd_norm / max(1e-12, ce_norm), 6),
            "prefix_kd_to_ce_norm_ratio": round(pre_norm / max(1e-12, ce_norm), 6),
            "ce_kd_cosine": round(cos, 6),
            "by_group": {"ce": ce_groups, "kd_total": kd_groups,
                         "kd_prefix": pre_groups, "kd_continuation": con_groups},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    # A teacher that is a copy of the student makes KL identically zero, which
    # silently turns every KD number into 0.0 and looks like a finding. Require
    # a real teacher unless the caller explicitly asks for the degenerate probe.
    ap.add_argument("--teacher", default=None,
                    help="HF id@revision, or a local path; required unless "
                         "--allow-degenerate-teacher is passed")
    ap.add_argument("--allow-degenerate-teacher", action="store_true",
                    help="use a copy of the student as teacher; KD will be "
                         "identically zero and proves nothing about magnitude")
    ap.add_argument("--examples", nargs="+", required=True,
                    help="one jsonl per arm, each with ids/mask fields")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--kd-weight", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trainable",
                    default=r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.,"
                            r"\.mlp\.(gate_proj|up_proj|down_proj)\.,"
                            r"input_layernorm,post_attention_layernorm,model\.norm\.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts/audit/e5_gradients.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32).to(device)
    student.config.use_cache = False
    select_trainable(student, args.trainable.split(","))

    if args.teacher:
        tid, _, rev = args.teacher.partition("@")
        teacher = AutoModelForCausalLM.from_pretrained(
            tid, revision=rev or None, dtype=torch.bfloat16).to(device).eval()
    elif args.allow_degenerate_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.float32).to(device).eval()
    else:
        raise SystemExit(
            "refusing to run without --teacher: a copy of the student gives "
            "KL == 0 and every KD number would read as a rounding error")
    for p in teacher.parameters():
        p.requires_grad_(False)

    labels = args.labels or [Path(p).stem for p in args.examples]
    arms = {}
    for label, path in zip(labels, args.examples):
        rows = [json.loads(line) for line in Path(path).open() if line.strip()]
        rows = rows[:args.max_batch]
        width = max(len(r["ids"]) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        ce_mask = torch.zeros(len(rows), width, dtype=torch.bool)
        content = torch.zeros(len(rows), width, dtype=torch.bool)
        for i, r in enumerate(rows):
            n = len(r["ids"])
            ids[i, :n] = torch.tensor(r["ids"], dtype=torch.long)
            ce_mask[i, :n] = torch.tensor(r["mask"], dtype=torch.bool)
            content[i, :n] = True
        arms[label] = analyse(
            student, teacher, ids.to(device), ce_mask.to(device),
            content.to(device), ce_w=args.ce_weight, kd_w=args.kd_weight,
            temperature=args.temperature)
        arms[label]["n_examples"] = len(rows)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": ("attribute the E5 objective's loss AND gradient to CE, "
                    "prefix KD and continuation KD on real batches"),
        "method": {
            "terms_backwarded_separately": True,
            "optimizer_step_taken": False,
            "kd_pieces_share_the_production_denominator": True,
            "why": ("re-normalizing prefix-only KD over its own token count "
                    "would measure a different objective; sharing kd_total means "
                    "the pieces sum to the term the trainer optimizes"),
        },
        "loss_weights": {"ce": args.ce_weight, "kd": args.kd_weight,
                         "temperature": args.temperature},
        "student": args.student,
        "teacher": args.teacher or "DEGENERATE copy-of-student (KL==0)",
        "degenerate_teacher": not bool(args.teacher),
        "arms": arms,
        "hardware": hardware_report(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"{'arm':10s}{'w*CE':>10s}{'w*KD':>10s}{'KD/loss':>10s}"
          f"{'|gCE|':>12s}{'|gKD|':>12s}{'|gKDpre|':>12s}{'gKD/gCE':>10s}{'cos':>9s}")
    for label, a in arms.items():
        L, G = a["loss"], a["gradient"]
        print(f"{label:10s}{L['weighted_ce']:10.4f}{L['weighted_kd_total']:10.4f}"
              f"{L['kd_share_of_total_loss']:10.5f}"
              f"{G['weighted_ce_norm']:12.5f}{G['weighted_kd_total_norm']:12.5f}"
              f"{G['weighted_kd_prefix_norm']:12.5f}"
              f"{G['kd_to_ce_norm_ratio']:10.4f}{G['ce_kd_cosine']:9.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

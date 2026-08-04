#!/usr/bin/env python
"""Compare the full-width and padding-truncated training paths numerically.

    PYTHONPATH=src python scripts/training/validate_padding_truncation.py \
        --pack artifacts/stage3/ladder_uniform_probe \
        --out artifacts/audit/padding_truncation_equivalence.json

What is being claimed, and what is not
--------------------------------------
Padding is a contiguous suffix and attention is causal, so no real token's
hidden state depends on a padded position, and every padded position is masked
out of CE and KD. Truncating the suffix therefore should not change what the run
computes.

That argument is about *mathematics*, not about floating point. Reduction shapes
change when the sequence is shorter, so kernels may accumulate in a different
order and results can differ in the last bits. **This validation does not claim
bitwise equality.** It measures the difference and checks it against explicit
tolerances, and it reports the observed numbers whether they pass or not.

Blocks are taken from a real pack in three padding regimes — dense, median, and
heavily padded (the tool-calling shape, which is >90% pad) — because the size of
any floating-point difference depends on how much is being dropped.

Token ids are remapped into a small vocabulary so the real 8,192-wide blocks and
their real fill fractions can be used on CPU: a `[1, 8192, 151936]` float32
logits tensor is ~5 GB and cannot be held twice. Remapping is sound here because
nothing in the truncation path reads token *identity* — the extent comes from the
content mask alone, and `kd_scope="all"` does not inspect ids. The block shapes,
masks and fill fractions are the real ones. A second pass at the true 151,936
vocabulary and a shorter block length cross-checks that vocabulary size does not
interact with the result.

Compared, per regime
--------------------
* hard-label CE, KD, total loss, and validation CE;
* per-parameter gradients: max absolute and relative difference, global norm
  under each path, and cosine similarity of the flattened gradient;
* one complete optimizer step from identical weights, optimizer state, RNG
  state, batch, scheduler state, gradient accumulation and clipping — then the
  updated parameters, Adam moments, learning rate, loss and grad norm.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.training.train import Trainer  # noqa: E402

# Explicit tolerances. float32 accumulation over sequences of different length
# reorders reductions, so exact equality is not the bar; these are set at the
# level where a real logic error would be obvious and float noise would not.
TOL = {
    "loss_atol": 1e-5,
    "loss_rtol": 1e-5,
    "grad_max_abs": 1e-5,
    "grad_cosine_min": 1.0 - 1e-9,
    "param_diff_max_fraction_of_update": 0.05,
    "adam_moment_max_abs": 1e-6,
    "grad_norm_rtol": 1e-5,
}


def tiny_cfg(block_len: int, seed: int = 7) -> dict:
    return {
        "stage": "validate", "run_name": "pad_trunc", "student_path": "-",
        "data_dir": "-", "out_dir": "-", "block_len": block_len, "seed": seed,
        "dtype": "float32", "device": "cpu", "groups": None, "teacher": None,
        "trainable_patterns": "all",
        "loss": {"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0,
                 "kd_scope": "all"},
        "optim": {"lr": 3e-4, "weight_decay": 0.01, "eps": 1e-8,
                  "grad_clip": 1.0, "betas": [0.9, 0.95]},
        "schedule": {"total_steps": 10, "warmup_steps": 2, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 2, "micro_blocks": 1},
        "checkpoint": {"save_every": 1000, "keep_last": 1},
        "intervals": {"log_every": 1, "eval_every": 1000, "eval_blocks": 2},
    }


def build_models(vocab: int, seed: int = 0):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen3", vocab_size=vocab, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
        max_position_embeddings=8192, tie_word_embeddings=True)
    student = AutoModelForCausalLM.from_config(cfg)
    torch.manual_seed(seed + 1)
    teacher = AutoModelForCausalLM.from_config(cfg).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


def pick_regimes(pack: Path, n_per: int = 2):
    """Two blocks each from the densest, median and most-padded parts of a pack."""
    a = np.load(pack / "blocks.npz")
    fill = a["content_mask"].sum(axis=1)
    order = np.argsort(fill)
    mid = len(order) // 2
    return {
        "heavy_pad": order[:n_per].tolist(),
        "median_pad": order[mid:mid + n_per].tolist(),
        "dense": order[-n_per:].tolist(),
    }, a


def run_path(cfg, blocks, val_blocks, truncate: bool, vocab: int, seed: int = 1234):
    """One optimizer step from a fixed starting state; returns everything."""
    cfg = copy.deepcopy(cfg)
    cfg["batch"]["truncate_padding"] = truncate
    student, teacher = build_models(vocab)
    tr = Trainer(cfg, student, blocks, val_blocks, teacher=teacher, device="cpu")

    before = {n: p.detach().clone() for n, p in student.named_parameters()}
    torch.manual_seed(seed)
    metrics = tr.step_once()
    grads = {n: p.grad.detach().clone() for n, p in student.named_parameters()
             if p.grad is not None}
    params = {n: p.detach().clone() for n, p in student.named_parameters()}
    update = {n: (params[n] - before[n]).abs().max().item() for n in params}
    adam = {n: {k: v.detach().clone() for k, v in st.items()
                if torch.is_tensor(v)}
            for n, st in zip([n for n, _ in student.named_parameters()],
                             tr.opt.state.values())} if tr.opt.state else {}
    val = tr.evaluate()
    return {"metrics": metrics, "grads": grads, "params": params,
            "adam": adam, "val": val, "update": update}


def compare(a, b) -> dict:
    out: dict = {}
    ma, mb = a["metrics"], b["metrics"]
    out["losses"] = {
        k: {"full": ma.get(k), "truncated": mb.get(k),
            "abs_diff": (abs(ma[k] - mb[k])
                         if isinstance(ma.get(k), (int, float))
                         and isinstance(mb.get(k), (int, float)) else None)}
        for k in ("loss", "ce", "kd", "grad_norm", "lr")
    }
    out["val_ce"] = {"full": a["val"].get("val_ce"),
                     "truncated": b["val"].get("val_ce"),
                     "abs_diff": abs(a["val"]["val_ce"] - b["val"]["val_ce"])
                     if a["val"].get("val_ce") is not None
                     and b["val"].get("val_ce") is not None else None}
    out["accounting"] = {
        k: {"full": ma.get(k), "truncated": mb.get(k)}
        for k in ("logical_block_tokens", "executed_positions",
                  "executed_nonpad_tokens", "supervised_tokens")
    }

    # --- gradients -------------------------------------------------------
    ga, gb = a["grads"], b["grads"]
    keys = sorted(set(ga) & set(gb))
    fa = torch.cat([ga[k].flatten() for k in keys])
    fb = torch.cat([gb[k].flatten() for k in keys])
    denom = fa.abs().clamp_min(1e-12)
    out["gradients"] = {
        "n_tensors": len(keys),
        "missing_in_full": sorted(set(gb) - set(ga)),
        "missing_in_truncated": sorted(set(ga) - set(gb)),
        "max_abs_diff": float((fa - fb).abs().max()),
        "max_rel_diff": float(((fa - fb).abs() / denom).max()),
        "global_norm_full": float(fa.norm()),
        "global_norm_truncated": float(fb.norm()),
        "global_norm_rel_diff": float(abs(fa.norm() - fb.norm())
                                      / max(float(fa.norm()), 1e-12)),
        # float64: a float32 dot product over ~1e6 elements accumulates enough
        # error to report a cosine slightly greater than 1, which is impossible.
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                fa.double()[None], fb.double()[None])),
        "worst_tensor": max(
            keys, key=lambda k: float((ga[k] - gb[k]).abs().max())),
    }

    # --- updated parameters and Adam state --------------------------------
    pa, pb = a["params"], b["params"]
    pk = sorted(set(pa) & set(pb))
    max_pd = max(float((pa[k] - pb[k]).abs().max()) for k in pk)
    max_update = max(a["update"].values()) if a["update"] else 0.0
    out["parameters_after_step"] = {
        "max_abs_diff": max_pd,
        "max_update_magnitude": max_update,
        # Adam divides by sqrt(v)+eps, so at eps=1e-8 a component whose gradient
        # is near zero gets an update of size ~lr regardless of the gradient's
        # magnitude. A 1e-8 gradient difference on such a component can therefore
        # move the parameter by a visible fraction of one step. The meaningful
        # quantity is that fraction, not the raw delta.
        "diff_as_fraction_of_update": (max_pd / max_update
                                       if max_update > 0 else None),
        "n_tensors": len(pk),
    }
    aa, ab = a["adam"], b["adam"]
    moms = []
    for k in sorted(set(aa) & set(ab)):
        for mk in sorted(set(aa[k]) & set(ab[k])):
            moms.append(float((aa[k][mk] - ab[k][mk]).abs().max()))
    out["adam_state_after_step"] = {
        "max_abs_diff": max(moms) if moms else None,
        "n_tensors": len(moms),
    }
    return out


def judge(cmp_: dict) -> tuple[bool, list[str]]:
    bad = []
    for k, v in cmp_["losses"].items():
        d = v["abs_diff"]
        if d is None:
            continue
        scale = max(abs(v["full"] or 0.0), 1.0)
        lim = TOL["loss_atol"] + TOL["loss_rtol"] * scale
        if k == "grad_norm":
            lim = TOL["grad_norm_rtol"] * max(abs(v["full"] or 0.0), 1.0)
        if d > lim:
            bad.append(f"loss::{k} diff {d:.3e} > {lim:.3e}")
    if cmp_["val_ce"]["abs_diff"] is not None and \
            cmp_["val_ce"]["abs_diff"] > TOL["loss_atol"] + TOL["loss_rtol"]:
        bad.append(f"val_ce diff {cmp_['val_ce']['abs_diff']:.3e}")
    g = cmp_["gradients"]
    if g["max_abs_diff"] > TOL["grad_max_abs"]:
        bad.append(f"grad max_abs {g['max_abs_diff']:.3e} > {TOL['grad_max_abs']:.3e}")
    if g["cosine_similarity"] < TOL["grad_cosine_min"]:
        bad.append(f"grad cosine {g['cosine_similarity']:.12f} < {TOL['grad_cosine_min']}")
    frac = cmp_["parameters_after_step"]["diff_as_fraction_of_update"]
    if frac is not None and frac > TOL["param_diff_max_fraction_of_update"]:
        bad.append(f"param delta is {frac:.4f} of one optimizer step "
                   f"(> {TOL['param_diff_max_fraction_of_update']})")
    am = cmp_["adam_state_after_step"]["max_abs_diff"]
    if am is not None and am > TOL["adam_moment_max_abs"]:
        bad.append(f"adam moment max_abs {am:.3e}")
    # The accounting must show the optimization actually happened, and must not
    # change the two quantities that define the run.
    acc = cmp_["accounting"]
    if acc["logical_block_tokens"]["full"] != acc["logical_block_tokens"]["truncated"]:
        bad.append("logical_block_tokens changed")
    if acc["supervised_tokens"]["full"] != acc["supervised_tokens"]["truncated"]:
        bad.append("supervised_tokens changed")
    if acc["executed_nonpad_tokens"]["full"] != acc["executed_nonpad_tokens"]["truncated"]:
        bad.append("executed_nonpad_tokens changed")
    if not (acc["executed_positions"]["truncated"] <= acc["executed_positions"]["full"]):
        bad.append("truncated path did not reduce executed positions")
    return not bad, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-len", type=int, default=0,
                    help="0 = use the pack's native block length")
    ap.add_argument("--vocab", type=int, default=4096,
                    help="remap ids into this vocab so real 8192 blocks fit in RAM")
    ap.add_argument("--label", default="remapped_vocab")
    args = ap.parse_args()

    regimes, arrays = pick_regimes(args.pack)
    T = arrays["input_ids"].shape[1] if not args.block_len else args.block_len
    results, all_ok = {}, True

    for name, idxs in regimes.items():
        raw_ids = arrays["input_ids"][idxs][:, :T].astype(np.int64)
        ids = torch.from_numpy(raw_ids % args.vocab)
        mask = torch.from_numpy(arrays["ce_mask"][idxs][:, :T]).bool()
        content = torch.from_numpy(arrays["content_mask"][idxs][:, :T]).bool()
        fill = float(content.float().mean())
        blocks = (ids, mask, content)
        cfg = tiny_cfg(T)
        a = run_path(cfg, blocks, blocks, truncate=False, vocab=args.vocab)
        b = run_path(cfg, blocks, blocks, truncate=True, vocab=args.vocab)
        cmp_ = compare(a, b)
        ok, bad = judge(cmp_)
        all_ok &= ok
        cmp_["fill_fraction"] = round(fill, 4)
        cmp_["block_indices"] = idxs
        cmp_["ok"] = ok
        cmp_["violations"] = bad
        results[name] = cmp_
        print(f"\n=== {name}  fill={fill:.3f}  blocks={idxs} -> "
              f"{'OK' if ok else 'VIOLATION'}")
        print(f"  loss full {cmp_['losses']['loss']['full']} "
              f"trunc {cmp_['losses']['loss']['truncated']} "
              f"diff {cmp_['losses']['loss']['abs_diff']:.3e}")
        print(f"  val_ce diff {cmp_['val_ce']['abs_diff']:.3e}")
        g = cmp_["gradients"]
        print(f"  grad max_abs {g['max_abs_diff']:.3e}  max_rel {g['max_rel_diff']:.3e}"
              f"  cos {g['cosine_similarity']:.12f}")
        print(f"  grad norm {g['global_norm_full']:.8f} vs "
              f"{g['global_norm_truncated']:.8f}")
        pas = cmp_["parameters_after_step"]
        print(f"  params after step max_abs {pas['max_abs_diff']:.3e}"
              f"  (update {pas['max_update_magnitude']:.3e}, "
              f"ratio {pas['diff_as_fraction_of_update']:.5f})")
        print(f"  adam max_abs {cmp_['adam_state_after_step']['max_abs_diff']}")
        a_ = cmp_["accounting"]
        print(f"  executed positions {a_['executed_positions']['full']} -> "
              f"{a_['executed_positions']['truncated']}  "
              f"(nonpad {a_['executed_nonpad_tokens']['full']}, "
              f"supervised {a_['supervised_tokens']['full']})")
        for v in bad:
            print(f"    ! {v}")

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "verdict": "pass" if all_ok else "fail",
        "pack": str(args.pack),
        "label": args.label,
        "block_len": T,
        "vocab": args.vocab,
        "ids_remapped": args.vocab != 151936,
        "tolerances": TOL,
        "note": ("Bitwise equality is NOT claimed: shorter sequences reorder "
                 "float32 reductions. Differences are measured against the "
                 "tolerances above and reported either way."),
        "regimes": results,
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nverdict: {out['verdict'].upper()}  -> {args.out}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare a reference model's architecture against our student, field by field.

    PYTHONPATH=src python scripts/evaluation/compare_geometry.py \
        --reference Qwen/Qwen3-0.6B --reference-revision <sha> \
        --student artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
        --out artifacts/audit/reference_geometry.json

"Same-geometry reference" is a claim that has to be checked rather than assumed.
If the released model differs in vocabulary, head layout, FFN width or embedding
tying, then it is a *near*-geometry reference and any comparison of capability
against our student carries that caveat. This records the answer so the caveat is
attached to the number instead of remembered.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402

FIELDS = ("hidden_size", "num_hidden_layers", "intermediate_size",
          "num_attention_heads", "num_key_value_heads", "head_dim",
          "vocab_size", "tie_word_embeddings", "max_position_embeddings",
          "rope_theta", "rms_norm_eps", "torch_dtype", "model_type")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-revision", default=None)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoConfig
    ref = AutoConfig.from_pretrained(args.reference,
                                     revision=args.reference_revision)
    stu = AutoConfig.from_pretrained(args.student)

    rows, same = {}, True
    for f in FIELDS:
        a, b = getattr(ref, f, None), getattr(stu, f, None)
        a = str(a) if not isinstance(a, (int, float, bool, type(None))) else a
        b = str(b) if not isinstance(b, (int, float, bool, type(None))) else b
        rows[f] = {"reference": a, "student": b, "equal": a == b}
        if a != b:
            same = False

    def n_params(cfg):
        """Parameter count from the config alone (no weights downloaded)."""
        h, L = cfg.hidden_size, cfg.num_hidden_layers
        i, v = cfg.intermediate_size, cfg.vocab_size
        hd = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
        q = h * cfg.num_attention_heads * hd
        kv = 2 * h * cfg.num_key_value_heads * hd
        o = cfg.num_attention_heads * hd * h
        mlp = 3 * h * i
        emb = v * h * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
        return emb + L * (q + kv + o + mlp)

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "reference": args.reference,
        "reference_revision": args.reference_revision,
        "student": args.student,
        "verdict": "same-geometry" if same else "near-geometry",
        "fields": rows,
        "differing_fields": [f for f, r in rows.items() if not r["equal"]],
        "approx_params": {"reference": n_params(ref), "student": n_params(stu)},
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"verdict: {out['verdict'].upper()}")
    print(f"{'field':28s} {'reference':>18s} {'student':>18s}")
    for f, r in rows.items():
        flag = "" if r["equal"] else "   <-- DIFFERS"
        print(f"{f:28s} {str(r['reference']):>18s} {str(r['student']):>18s}{flag}")
    p = out["approx_params"]
    print(f"\napprox params: reference {p['reference']:,}  student {p['student']:,}")


if __name__ == "__main__":
    main()

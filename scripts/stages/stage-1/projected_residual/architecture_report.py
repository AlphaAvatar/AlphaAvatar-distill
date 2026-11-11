#!/usr/bin/env python3
"""Parameters, FLOPs, KV cache and memory of the projected-residual students against their baselines ($0).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/architecture_report.py \\
        --write logs/stages/stage-1/projected_residual/analyses/architecture_report.json

Every count comes from `aadistill.models.cost_model`, and the two students'
totals are also counted from a meta-device build (no tensor allocated), which
must agree. Arithmetic over the geometry only: what a GPU measures (step time,
peak memory, throughput) is the engineering validation's to record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

COMMON = dict(vocab_size=151936, num_hidden_layers=28, intermediate_size=3072, num_attention_heads=16,
              num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
GEOMETRIES = {
    "teacher_thinking_2507": dict(vocab_size=151936, hidden_size=2560, num_hidden_layers=36,
                                  intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
                                  head_dim=128, tie_word_embeddings=True),
    "d4_top1_596m": dict(COMMON, hidden_size=1024),
    "pw_900m": dict(COMMON, hidden_size=2560, internal_size=1024),
    "pw_900m_folded": dict(COMMON, hidden_size=2560),
    "pw_600m_reserved": dict(COMMON, hidden_size=2560, internal_size=512, intermediate_size=1910),
}
SEQ_LEN = 8192


def build() -> dict:
    import torch

    from aadistill.models.cost_model import (forward_flops_per_token, kv_cache_bytes_per_token,
                                             parameter_counts, training_flops_per_token)
    from aadistill.initialization.adapters.projected_residual import ProjectedResidualConfig, ProjectedResidualForCausalLM

    mean_ctx = (SEQ_LEN + 1) / 2
    teacher_fwd = forward_flops_per_token(GEOMETRIES["teacher_thinking_2507"], mean_ctx)["total"]
    rows = {}
    for name, g in GEOMETRIES.items():
        pc = parameter_counts(g)
        row = {"geometry": g, "parameters": pc,
               "forward_gflops_per_token": {str(c): round(forward_flops_per_token(g, c)["total"] / 1e9, 4)
                                            for c in (1, 1024, 4096, SEQ_LEN)},
               "kv_cache_bytes_per_token_bf16": kv_cache_bytes_per_token(g),
               "weights_gb_bf16": round(pc["total"] * 2 / 1e9, 3)}
        if name != "teacher_thinking_2507" and name != "pw_900m_folded":
            tr = training_flops_per_token(g, mean_ctx, train_embedding=False)
            row["training_gflops_per_position"] = {"student": round(tr["total"] / 1e9, 4),
                                                   "with_teacher_forward": round((tr["total"] + teacher_fwd) / 1e9, 4)}
            row["training_state_gb"] = round((pc["non_embedding"] * 16 + pc["embedding"] * 4) / 1e9, 3)
        if "internal_size" in g:
            with torch.device("meta"):
                m = ProjectedResidualForCausalLM(ProjectedResidualConfig(**g))
            row["meta_build_parameters"] = sum(p.numel() for p in m.parameters())
            assert row["meta_build_parameters"] == pc["total"], name
        rows[name] = row
    base, pw = rows["d4_top1_596m"], rows["pw_900m"]
    ratio = lambda a, b: round(a / b, 4)  # noqa: E731
    return {
        "schema": "aadistill.stage1.projected_residual.architecture_report/v1",
        "conventions": "aadistill.models.cost_model docstring: 2 FLOPs per multiply-add, mean causal context "
                       f"(L+1)/2 at L = {SEQ_LEN}, elementwise ops excluded, frozen tied embedding in training "
                       "(E1's trainable patterns never match it)",
        "models": rows,
        "pw_900m_vs_d4": {
            "parameters": ratio(pw["parameters"]["total"], base["parameters"]["total"]),
            "trainable_parameters": ratio(pw["parameters"]["non_embedding"], base["parameters"]["non_embedding"]),
            "forward_flops_ctx_1": ratio(pw["forward_gflops_per_token"]["1"], base["forward_gflops_per_token"]["1"]),
            "forward_flops_ctx_4096": ratio(pw["forward_gflops_per_token"]["4096"], base["forward_gflops_per_token"]["4096"]),
            "forward_flops_ctx_8192": ratio(pw["forward_gflops_per_token"][str(SEQ_LEN)],
                                            base["forward_gflops_per_token"][str(SEQ_LEN)]),
            "training_student": ratio(pw["training_gflops_per_position"]["student"],
                                      base["training_gflops_per_position"]["student"]),
            "training_with_teacher": ratio(pw["training_gflops_per_position"]["with_teacher_forward"],
                                           base["training_gflops_per_position"]["with_teacher_forward"]),
            "kv_cache": ratio(pw["kv_cache_bytes_per_token_bf16"], base["kv_cache_bytes_per_token_bf16"]),
        },
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    print(json.dumps(doc["pw_900m_vs_d4"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

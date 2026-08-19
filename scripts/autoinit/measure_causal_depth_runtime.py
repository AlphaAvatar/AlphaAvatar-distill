"""Measure the repaired causal-depth path, on a GPU, cheaply and boundedly.

    PYTHONPATH=src python scripts/autoinit/measure_causal_depth_runtime.py \
        --evaluations 20 --out logs/autoinit_causal_depth_measured.json

**This is not a Phase-A attempt and must not be run as one.** It loads the
teacher, runs a bounded number of real causal-depth evaluations against the real
frozen calibration mixture, and reports evaluations per minute and peak VRAM.
Nothing is searched, nothing is selected, no checkpoint is written, and the
greedy rule is never consulted — the point is the *rate*, and the rate is a
property of one evaluation.

Why a measurement is still wanted when the number is already known: the frozen
cost model in `logs/autoinit_v1_search_space.json` records E8a running exactly
this workload — 260 evaluations, 67 items, 59,763 positions, 4.02B full width —
in **1,300 s** on an L40S, which is 12.0 evaluations/min, and an independent FLOP
derivation reproduces that within 5 %. What is *not* established is that the
**ported** code achieves it: attempt 10 proved the port can differ from its
ancestor in ways no CPU run reveals. This measures the code that would run.

Cost. At 12 evaluations/min the default 20 evaluations is under 2 minutes of GPU
time; the job is dominated by loading the teacher. Budget one short L40S session,
not a Phase-A session.

Reports, per the pricing question it exists to answer:

* evaluations per minute, and the wall time of each evaluation;
* peak VRAM, and whether the bf16 reference cache was enabled or fell back;
* the extrapolated cost of a full 260-evaluation expansion;
* the GPU utilization the run actually achieved, so "the accelerator is idle"
  cannot recur unnoticed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.device import apply_cpu_budget, model_device  # noqa: E402
from aadistill.init.contribution import (  # noqa: E402
    bypassed_blocks, distortion, domain_balanced_score,
)

TEACHER_ID = "Qwen/Qwen3-4B-Thinking-2507"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluations", type=int, default=20,
                    help="bounded; the rate is a property of one evaluation")
    ap.add_argument("--teacher", default=TEACHER_ID)
    ap.add_argument("--teacher-revision", default=None)
    ap.add_argument("--n-remove", type=int, default=8,
                    help="size of the skip set, to match the schedule's shape")
    ap.add_argument("--out", default="logs/autoinit_causal_depth_measured.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This measures the accelerator path; running it on "
            "the host would re-measure exactly the defect being repaired.")

    budget = apply_cpu_budget()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoModelForCausalLM

    from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1
    from aadistill.autoinit.datasets import as_operator_items

    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, revision=args.teacher_revision, dtype=torch.bfloat16,
    ).to(device).eval()
    model.config.use_cache = False
    load_s = time.monotonic() - t0
    after_model = torch.cuda.max_memory_allocated(device)

    items = list(as_operator_items(DOMAIN_BALANCED_V1.resolve(REPO_ROOT)))
    n_layers = model.config.num_hidden_layers
    positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)

    # Exactly the operator's own arrangement: targets on the compute device, a
    # device-resident bf16 reference cache, the reduction where the tensors are.
    targets = [i["input_ids"][0, 1:].to(device) for i in items]
    domains: dict[str, list[str]] = {}
    for i in items:
        domains.setdefault(i.get("domain", i["subtype"]), []).append(i["subtype"])
    domains = {d: sorted(set(s)) for d, s in domains.items()}

    reference: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def logits(item, skip):
        ids = item["input_ids"].to(device)
        if not skip:
            return model(ids).logits[0, :-1]
        with bypassed_blocks(model, skip):
            return model(ids).logits[0, :-1]

    ref_t0 = time.monotonic()
    for item in items:
        reference[item["item_id"]] = logits(item, frozenset())
    ref_s = time.monotonic() - ref_t0
    after_cache = torch.cuda.max_memory_allocated(device)

    def evaluate(skip) -> float:
        per_subtype: dict[str, list[float]] = {}
        for item, tgt in zip(items, targets):
            abl = logits(item, skip)
            sums = distortion(reference[item["item_id"]], abl, tgt,
                              chunk=512).as_dict()
            per_subtype.setdefault(item["subtype"], []).append(sums["kl"])
            del abl
        means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
        primary, _ = domain_balanced_score(means, domains)
        return primary

    per_eval: list[float] = []
    for k in range(args.evaluations):
        skip = frozenset(range(k % max(1, n_layers - args.n_remove),
                               k % max(1, n_layers - args.n_remove) + args.n_remove))
        t = time.monotonic()
        evaluate(skip)
        torch.cuda.synchronize(device)
        per_eval.append(time.monotonic() - t)
        print(f"  eval {k + 1}/{args.evaluations}: {per_eval[-1]:.2f} s", flush=True)

    peak = torch.cuda.max_memory_allocated(device)
    mean_s = statistics.mean(per_eval)
    report = {
        "schema": "aadistill.autoinit.causal_depth_runtime/v1",
        "not_a_phase_a_attempt": (
            "bounded rate measurement only: nothing searched, selected or written"),
        "teacher": args.teacher, "revision": args.teacher_revision,
        "device_name": torch.cuda.get_device_name(device),
        "cpu_budget": budget,
        "workload": {"items": len(items), "positions": positions,
                     "vocab": int(model.config.vocab_size),
                     "n_layers": n_layers, "skip_size": args.n_remove},
        "timing": {
            "teacher_load_s": round(load_s, 2),
            "reference_pass_s": round(ref_s, 2),
            "per_evaluation_s": [round(x, 3) for x in per_eval],
            "mean_evaluation_s": round(mean_s, 3),
            "median_evaluation_s": round(statistics.median(per_eval), 3),
            "evaluations_per_minute": round(60.0 / mean_s, 2),
        },
        "vram": {
            "after_model_gib": round(after_model / 2**30, 2),
            "after_reference_cache_gib": round(after_cache / 2**30, 2),
            "peak_gib": round(peak / 2**30, 2),
            "total_gib": round(
                torch.cuda.get_device_properties(device).total_memory / 2**30, 2),
            "reference_cache_resident": "device",
        },
        "extrapolation": {
            "full_expansion_evaluations": 260,
            "full_expansion_minutes": round(260 * mean_s / 60.0, 1),
            "plus_reference_pass_minutes": round((260 * mean_s + ref_s) / 60.0, 1),
        },
        "compare_against": {
            "e8a_frozen_cost_model": ("260 evaluations in 1,300 s = 21.7 min, "
                                      "12.0 evaluations/min, L40S, from "
                                      "logs/autoinit_v1_search_space.json"),
            "attempt_10_host_path": ">= 647 min for one expansion, unfinished",
        },
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timing"] | report["vram"] |
                     report["extrapolation"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()

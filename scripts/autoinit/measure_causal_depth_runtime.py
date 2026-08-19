"""Bounded runtime and backend validation for the repaired causal-depth path.

    PYTHONPATH=src python scripts/autoinit/measure_causal_depth_runtime.py \
        --samples-per-cardinality 3 \
        --out logs/autoinit_causal_depth_measured.json

**Not a Phase-A attempt, and not a search.** No greedy search runs, no depth map
is selected, no checkpoint is written. This measures a *rate* and validates a
*backend*, both of which are properties of individual evaluations.

Three questions, none of which a CPU box can answer:

1. does the repaired port achieve E8a's measured rate — 260 evaluations in
   1,300 s on an L40S, 12.0 evaluations/min — now that scoring is back on the
   accelerator?
2. what is the real peak VRAM, and **which way does the production cache gate
   actually decide** at the frozen mixture?
3. does the repaired port compute the same numbers as **E8a** on the same GPU?

The scientific reference is E8a — `scripts/training/search_depth_map.py` — which
is the frozen ancestor and has always run the reduction on the accelerator. The
failed CPU port is not a reference for anything.

Why the comparison is made per item, not on the final score
-----------------------------------------------------------
E8a merges raw `DistortionSums` across the items of a subtype and normalizes
once, which is a **position-weighted** mean. The operator normalizes each item
first and takes an **unweighted** mean over items — its own description says so:
"the unweighted mean over domains of the unweighted mean over each domain's
sub-types". On a mixture whose items differ in length these disagree by
construction: measured at ~0.027 on a two-item toy, which is ~300x the smallest
real decision margin (8.195e-05).

That difference is a **declared aggregation choice, not backend drift**, and a
naive score-level comparison would report it as catastrophic disagreement. So the
paired comparison is made where the two paths must agree exactly — the per-item
`DistortionSums` — and the aggregation difference is reported separately, marked
as expected.

Sampling
--------
The real 36->28 greedy search evaluates skip sets of cardinality 1..8, with
round weights 36,35,...,29 summing to 260. Measuring every sample at cardinality
8 — as the first version of this script did — would time only 28-layer forwards
and **overstate** evaluations/min while **understating** the 260-evaluation
runtime. The sample therefore spans all eight cardinalities and the
extrapolation is weighted by the real schedule.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from aadistill.autoinit.device import apply_cpu_budget  # noqa: E402
from aadistill.init.contribution import (  # noqa: E402
    DistortionSums, distortion, domain_balanced_score,
)

#: The frozen teacher. Pinned, because a paid measurement against an unpinned Hub
#: HEAD measures whatever was published that morning.
TEACHER_ID = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: The real schedule: round r evaluates |skip| = r+1 over 36-r candidates.
PARENT_LAYERS, N_REMOVE = 36, 8
SCHEDULE = {c: PARENT_LAYERS - (c - 1) for c in range(1, N_REMOVE + 1)}
assert sum(SCHEDULE.values()) == 260, SCHEDULE

#: Deterministic, RNG-free, and recomputable by hand. `gcd(5, 36) == 1`, so the
#: stride visits distinct layers and the sets spread across depth rather than
#: clustering — a block of eight adjacent layers is not a representative ablation.
STRIDE, OFFSET_STEP = 5, 3


def skip_set(cardinality: int, sample: int, n_layers: int = PARENT_LAYERS
             ) -> frozenset[int]:
    base = (sample * OFFSET_STEP) % n_layers
    return frozenset((base + k * STRIDE) % n_layers for k in range(cardinality))


class GpuSampler:
    """Sample utilization while the evaluations run.

    Attempt 10 sat at 0-1 % for eleven hours and nothing recorded it; the number
    was only discovered by hand, after the fact, on a live pod. A measurement job
    that reports a rate without reporting whether the accelerator was busy would
    leave the same gap.
    """

    def __init__(self, device, period_s: float = 0.5) -> None:
        self.device, self.period, self.samples = device, period_s, []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.method = "torch.cuda.utilization"

    def _read(self) -> int | None:
        try:
            return int(torch.cuda.utilization(self.device))
        except Exception:                                          # noqa: BLE001
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                self.method = "nvidia-smi"
                return int(out.splitlines()[0])
            except Exception:                                      # noqa: BLE001
                self.method = "unavailable"
                return None

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            v = self._read()
            if v is not None:
                self.samples.append(v)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def report(self) -> dict:
        if not self.samples:
            return {"method": self.method, "samples": 0,
                    "note": "no utilization samples; the claim is withdrawn "
                            "rather than estimated"}
        s = sorted(self.samples)
        return {
            "method": self.method, "samples": len(s),
            "mean_pct": round(statistics.mean(s), 1),
            "median_pct": s[len(s) // 2],
            "min_pct": s[0], "max_pct": s[-1],
            "fraction_below_10_pct": round(
                sum(1 for x in s if x < 10) / len(s), 3),
            "attempt_10_reference": "0-1 % for 11 h with the reduction on the host",
        }



def run_measurement(model, items, device, *, n_layers: int,
                    samples_per_cardinality: int, e8a_pairs: int,
                    n_remove: int = N_REMOVE) -> dict:
    """The whole measurement, minus the CUDA-only bookkeeping.

    A seam, so this executes for real on the dev box at toy scale. Four paid pods
    have died inside lines no $0 path had ever run; a measurement job whose body
    only ever runs on a GPU would be the fifth.
    """
    from aadistill.autoinit.operators.depth import _forward_logits, _ReferenceLogits

    if not 0 < n_remove < n_layers:
        raise ValueError(f"cannot remove {n_remove} of {n_layers} layers")
    schedule = {c: n_layers - (c - 1) for c in range(1, n_remove + 1)}
    targets = [i["input_ids"][0, 1:].to(device) for i in items]
    domains: dict[str, set] = {}
    for i in items:
        domains.setdefault(i.get("domain", i["subtype"]), set()).add(i["subtype"])
    domains = {d: sorted(s) for d, s in domains.items()}

    # The production cache path, imported rather than reimplemented: the point is
    # to exercise the gate the operator really uses, its 0.66-of-free decision and
    # its recompute fallback, and to report which way it went.
    reference = _ReferenceLogits(model, items, str(device))
    cache_decision = reference.decision()
    ref_t0 = time.monotonic()
    for item in items:
        reference.get(item)
    ref_s = time.monotonic() - ref_t0

    def evaluate(skip):
        per_subtype: dict[str, list[float]] = {}
        per_item: dict[str, dict] = {}
        for item, tgt in zip(items, targets):
            abl = _forward_logits(model, item, str(device), skip)
            sums = distortion(reference.get(item), abl, tgt, chunk=512).as_dict()
            per_subtype.setdefault(item["subtype"], []).append(sums["kl"])
            per_item[item["item_id"]] = sums
            del abl
        means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
        primary, _ = domain_balanced_score(means, domains)
        return primary, per_item

    timings: dict[int, list[float]] = {c: [] for c in schedule}
    port_scores: dict[str, float] = {}
    port_items: dict[str, dict] = {}
    with GpuSampler(device) as gpu:
        for c in schedule:
            for j in range(samples_per_cardinality):
                skip = skip_set(c, j, n_layers)
                t0 = time.monotonic()
                score, per_item = evaluate(skip)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                dt = time.monotonic() - t0
                timings[c].append(dt)
                key = ",".join(map(str, sorted(skip)))
                port_scores[key], port_items[key] = score, per_item
                print(f"  |skip|={c} sample {j}: {dt:.2f} s", flush=True)

    means_by_c = {c: statistics.mean(v) for c, v in timings.items()}
    weighted_s = sum(schedule[c] * means_by_c[c] for c in schedule)
    total = sum(schedule.values())
    flat_s = total * means_by_c[max(schedule)]

    # --- E8a, same process, same GPU, same teacher, same skip sets ---------
    from search_depth_map import Searcher, prepare

    e8a_items = [{"item_id": i["item_id"], "subtype": i["subtype"],
                  "ids": i["input_ids"][0].tolist(),
                  "n_prediction_positions": int(i["input_ids"].shape[1]) - 1,
                  "tags": {}} for i in items]
    prepared = prepare(e8a_items, device)
    by_id = {p["item_id"]: p for p in prepared}
    for p_ in prepared:
        src = next(i for i in items if i["item_id"] == p_["item_id"])
        assert torch.equal(p_["ids"].cpu(), src["input_ids"].cpu()), (
            f"{p_['item_id']}: the two paths were not given the same tokens")
    searcher = Searcher(model, prepared, domains, cache_reference=True, chunk=512)

    paired = []
    for key in list(port_scores)[:e8a_pairs]:
        skip = frozenset(int(x) for x in key.split(","))
        rec = searcher.evaluate(skip)
        deltas = []
        for item in items:
            p_ = by_id[item["item_id"]]
            e = distortion(searcher.reference_logits(p_),
                           searcher._logits(p_, skip), p_["targets"],
                           chunk=512).as_dict()
            deltas.append(abs(e["kl"] - port_items[key][item["item_id"]]["kl"]))
        paired.append({
            "skip": sorted(skip),
            "max_per_item_kl_delta": max(deltas),
            "mean_per_item_kl_delta": statistics.mean(deltas),
            "port_aggregated_score": port_scores[key],
            "e8a_aggregated_score": rec["primary_kl"],
            "aggregated_difference": abs(rec["primary_kl"] - port_scores[key]),
        })

    return {
        "schedule": schedule, "total_evaluations": total,
        "timings": timings, "means_by_c": means_by_c,
        "weighted_s": weighted_s, "flat_s": flat_s,
        "reference_pass_s": ref_s, "cache_decision": cache_decision,
        "gpu": gpu.report(), "paired": paired,
        "n_timed": sum(len(v) for v in timings.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-cardinality", type=int, default=3,
                    help="8 cardinalities x this = total evaluations timed")
    ap.add_argument("--teacher", default=TEACHER_ID)
    ap.add_argument("--teacher-revision", default=TEACHER_REVISION)
    ap.add_argument("--e8a-pairs", type=int, default=2,
                    help="skip sets scored by BOTH paths for the backend check")
    ap.add_argument("--out", default="logs/autoinit_causal_depth_measured.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This validates the accelerator path; running it on "
            "the host would re-measure exactly the defect being repaired.")
    if not args.teacher_revision:
        raise SystemExit(
            "refusing to measure against an unpinned Hub HEAD: pass the frozen "
            f"revision ({TEACHER_REVISION}).")

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
    assert model.config.num_hidden_layers == PARENT_LAYERS, (
        f"schedule assumes {PARENT_LAYERS} layers, teacher has "
        f"{model.config.num_hidden_layers}")

    profile = DOMAIN_BALANCED_V1
    calib_path = profile.resolve(REPO_ROOT)
    items = list(as_operator_items(calib_path))
    positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)

    core = run_measurement(model, items, device, n_layers=PARENT_LAYERS,
                           n_remove=N_REMOVE,
                           samples_per_cardinality=args.samples_per_cardinality,
                           e8a_pairs=args.e8a_pairs)
    peak = torch.cuda.max_memory_allocated(device)

    report = {
        "schema": "aadistill.autoinit.causal_depth_measured/v2",
        "not_a_phase_a_attempt": (
            "bounded rate and backend validation: no greedy search, no depth map, "
            "no checkpoint written"),
        "identities": {
            "teacher": args.teacher, "revision": args.teacher_revision,
            "revision_pinned": args.teacher_revision == TEACHER_REVISION,
            "calibration_profile": profile.qualified_id,
            "calibration_profile_hash": profile.profile_hash,
            "calibration_path": str(calib_path),
            "items": len(items), "positions": positions,
            "vocab": int(model.config.vocab_size),
            "layers": model.config.num_hidden_layers,
        },
        "device": {"name": torch.cuda.get_device_name(device),
                   "total_gib": round(torch.cuda.get_device_properties(device)
                                      .total_memory / 2**30, 2)},
        "cpu_budget": budget,
        "sampling": {
            "scheme": ("deterministic, RNG-free: skip(c, j) = {(3j + 5k) mod 36 : "
                       "k < c}. gcd(5,36)=1 so the layers are distinct and spread"),
            "cardinalities": sorted(core["schedule"]),
            "samples_per_cardinality": args.samples_per_cardinality,
            "total_evaluations_timed": core["n_timed"],
            "schedule_weights": core["schedule"],
            "weights_sum": sum(core["schedule"].values()),
        },
        "timing": {
            "teacher_load_s": round(load_s, 2),
            "reference_pass_s": round(core["reference_pass_s"], 2),
            "mean_seconds_by_cardinality": {c: round(v, 3)
                                            for c, v in core["means_by_c"].items()},
            "weighted_260_eval_minutes": round(core["weighted_s"] / 60, 2),
            "weighted_evaluations_per_minute": round(
                core["total_evaluations"] / (core["weighted_s"] / 60), 2),
            "flat_cardinality_8_minutes_WRONG": round(core["flat_s"] / 60, 2),
            "flat_would_have_understated_by_pct": round(
                100 * (core["weighted_s"] - core["flat_s"]) / core["weighted_s"], 1),
        },
        "vram": {"peak_gib": round(peak / 2**30, 2)},
        "reference_cache_decision": core["cache_decision"],
        "gpu_utilization": core["gpu"],
        "e8a_backend_comparison": {
            "reference_implementation": "scripts/training/search_depth_map.py",
            "paired": core["paired"],
            "per_item_is_the_comparison": (
                "E8a merges raw sums per subtype and normalizes once "
                "(position-weighted); the operator normalizes per item and takes "
                "an unweighted mean. That is a DECLARED aggregation difference, "
                "not drift - ~0.027 on a toy, ~300x the 8.195e-05 decision "
                "margin - so the backend check is the per-item delta above."),
        },
        "compare_against": {
            "e8a_frozen_cost_model": "260 evaluations in 1,300 s = 21.7 min, 12.0/min",
            "attempt_10_host_path": ">= 647 min for one expansion, unfinished",
            "cpu_equivalence_artifact": "logs/autoinit_depth_backend_equivalence.json",
        },
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("timing", "vram", "reference_cache_decision",
                       "gpu_utilization", "e8a_backend_comparison")}, indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()

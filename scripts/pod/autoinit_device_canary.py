#!/usr/bin/env python3
"""Operator device canary: every frozen operator, once, on a real GPU.

    /opt/train/bin/python scripts/pod/autoinit_device_canary.py \
        --out artifacts/audit/autoinit_device_canary/result.json

**This is not an experiment and its outputs may not enter scientific
selection.** It answers exactly one question: does every frozen operator, and
the materialize -> canonical reload -> validate -> measure lifecycle around it,
execute on CUDA without a device error? Three paid Phase-A sessions have now
been spent discovering that one operator at a time, because the dev box has a
single device and a cross-device use is unobservable on it. The $0 device-split
regressions model the split; this checks the model against the thing itself.

What it does NOT do, deliberately:

* no beam search — each operator is invoked once, directly, from one parent;
* no recovery training and no battery;
* no ranking, no selection, no leaves, nothing that could be mistaken for a
  result. `scientific_use: false` is written into the record and the checkpoints
  it produces are deleted with the pod.

The parent is the canonical student `qwen3_0p6b_init_v0` rather than the 4B
teacher: the defects are about *where tensors are*, not how large they are, and
a 0.6B parent exercises every line at a fraction of the load time. The
calibration is a handful of the frozen mixture's own items, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import torch  # noqa: E402

from aadistill.autoinit import device as device_contract  # noqa: E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1  # noqa: E402
from aadistill.autoinit.device import model_device, stats_bytes  # noqa: E402
from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem  # noqa: E402
from aadistill.autoinit.operators.base import (  # noqa: E402
    OperatorContext, get_implementation,
)
from aadistill.autoinit.stats import StatsCache  # noqa: E402

CANONICAL = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
#: Every frozen implementation, in registry order. Six, across five kinds.
OPERATORS = ("depth.positional_v0", "depth.causal_kl_greedy_v1",
             "width.global_pca_v0", "ffn.activation_importance_v0",
             "attention.weight_proxy_v0", "composite.stage1_sandwich_v0")


def canary_target(parent_spec: ArchSpec) -> ArchSpec:
    """A strictly smaller geometry, so every operator has work to do.

    Not the Phase-A target and not a recipe: the point is that each operator's
    code path runs, so each structural field is reduced by the smallest step
    that keeps the architecture legal.
    """
    return parent_spec.replace(
        num_hidden_layers=parent_spec["num_hidden_layers"] - 2,
        hidden_size=parent_spec["hidden_size"] // 2,
        intermediate_size=parent_spec["intermediate_size"] // 2,
        num_attention_heads=max(parent_spec["num_key_value_heads"],
                                parent_spec["num_attention_heads"] // 2))


def tiny_calibration(n_items: int, max_tokens: int, device):
    """A few real items from the frozen mixture, truncated.

    Real ids, because the point is the production loader's output shape; few and
    short, because the question is placement, not statistics quality.
    """
    from phase_a_search import as_operator_items

    items = as_operator_items(DOMAIN_BALANCED_V1.resolve(REPO))[:n_items]
    for item in items:
        item["input_ids"] = item["input_ids"][:, :max_tokens]
    return items


def tiny_suite(items, device):
    suite = StateEvalSuite(
        suite_id="device_canary", version=1, domains=("general",),
        subtypes={"general": ("general",)}, critical_tags=(),
        description="device canary only; not a state-evaluation suite")
    suite_items = [
        SuiteItem(item_id=f"canary-{i}", input_ids=it["input_ids"],
                  domain="general", subtype="general", tags={})
        for i, it in enumerate(items)]
    return suite, suite_items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/audit/autoinit_device_canary/result.json")
    ap.add_argument("--workdir", default="artifacts/autoinit/device_canary")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calibration-items", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("this canary exists to run on CUDA; refusing to "
                         "certify anything from a CPU run")

    from aadistill.autoinit.metrics import StateEvaluator
    from aadistill.autoinit.search import BeamSearch, SearchConfig
    from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1
    from aadistill.autoinit.state import make_root_state

    started = time.time()
    adapter = get_adapter("qwen3")
    parent = adapter.load(str(CANONICAL), device=args.device)
    parent_spec = adapter.spec_of(parent)
    target = canary_target(parent_spec)
    items = tiny_calibration(args.calibration_items, args.max_tokens, args.device)
    suite, suite_items = tiny_suite(items, args.device)

    evaluator = StateEvaluator(suite, suite_items, device=args.device)
    evaluator.prime_reference(parent)

    workdir = REPO / args.workdir
    config = SearchConfig(
        run_id="autoinit.device_canary", target_spec=target,
        schedule=SCHEDULE_V1, seed=20260817, workdir=workdir,
        profiles=(DOMAIN_BALANCED_V1,), policy=PARETO_V1, suite=suite,
        device=args.device,
        notes={"purpose": "device canary; NOT an experiment",
               "scientific_use": "false"})
    engine = BeamSearch(
        adapter=adapter, config=config, root_teacher_id="qwen3_0p6b_init_v0",
        root_teacher_sha256="0" * 64, root_loader=lambda: parent,
        calibration_loader=lambda profile: items,
        measurer=lambda model, digest: evaluator.evaluate(model, digest))

    results = []
    for impl_id in OPERATORS:
        impl = get_implementation(impl_id)
        record = {"impl_id": impl_id, "ok": False}
        t0 = time.time()
        try:
            # One shared cache across the run, exactly as the search uses it:
            # the host-resident entry and the per-invocation working copy are
            # the thing under test.
            cache = StatsCache(stats_spec=config.stats_spec, max_entries=1)
            ctx = OperatorContext(
                adapter=adapter, model=parent, parent_spec=parent_spec,
                target_spec=target, profile=DOMAIN_BALANCED_V1,
                calibration_items=items, seed=config.seed,
                device=str(model_device(parent)), workdir=workdir,
                stats_cache=cache, stats_cache_key=f"canary::{impl_id}")
            outcome = impl.execute(ctx)

            # The production lifecycle around it: materialize -> canonical
            # reload -> hash -> validate -> measure.
            state = make_root_state(
                root_teacher_id="qwen3_0p6b_init_v0", root_teacher_sha256="0" * 64,
                spec=adapter.spec_of(outcome.model), target_spec=target,
                num_parameters=adapter.param_count(adapter.spec_of(outcome.model)),
                seed=config.seed)
            engine._materialize_and_measure(state, outcome.model,
                                            adapter.spec_of(outcome.model))
            checks = state.notes["validation"]
            record.update(
                ok=True,
                child_device=str(model_device(outcome.model)),
                parent_device=str(model_device(parent)),
                validation_device=checks["validation_device"],
                measurement_device=checks["measurement_device"],
                reload_max_logit_diff=checks["reload_max_logit_diff"],
                cached_stats_entries=cache.report()["resident_entries"],
                cached_stats_device=sorted({
                    str(v.device) for e in cache._entries.values()
                    for v in e.values()}),
                cached_stats_bytes=sum(stats_bytes(e)
                                       for e in cache._entries.values()),
                peak_vram_gib=(torch.cuda.max_memory_allocated() / 2**30
                               if torch.cuda.is_available() else None))
        except Exception as exc:                                  # noqa: BLE001
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc()[-8000:])
        record["seconds"] = round(time.time() - t0, 2)
        results.append(record)
        print(f"{'OK  ' if record['ok'] else 'FAIL'} {impl_id} "
              f"({record['seconds']}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    report = {
        "schema": "aadistill.autoinit.device_canary/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_use": False,
        "not_an_experiment": (
            "one invocation of each frozen operator plus the materialize/"
            "reload/validate/measure lifecycle, to check device placement on "
            "real CUDA. No beam search, no recovery training, no battery, no "
            "ranking and no selection. Nothing here may enter scientific "
            "selection or a README record."),
        "device_contract": device_contract.as_dict(),
        "parent": {"checkpoint": str(CANONICAL.relative_to(REPO)),
                   "spec": parent_spec.as_dict()},
        "target": target.as_dict(),
        "calibration": {"profile": DOMAIN_BALANCED_V1.qualified_id,
                        "n_items": args.calibration_items,
                        "max_tokens": args.max_tokens,
                        "note": "truncated; placement is the question, not "
                                "statistics quality"},
        "operators": results,
        "all_passed": all(r["ok"] for r in results),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"all_passed": report["all_passed"],
                      "elapsed_minutes": report["elapsed_minutes"],
                      "failed": [r["impl_id"] for r in results if not r["ok"]]},
                     indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

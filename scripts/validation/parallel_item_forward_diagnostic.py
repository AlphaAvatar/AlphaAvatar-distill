"""Is a concurrently-scheduled B=1 forward the same computation as a serial one?

The batch-invariance investigation established that a `[B, T]` padded tensor
batch is deterministic but NOT numerically invariant, and that forbidding
split-K repairs the bare GEMMs without preserving the historical solo output.
What it did not establish is that item-forward CONCURRENCY is impossible.

Those are different things, and the difference is the whole point:

    padded tensor batching   one forward of shape [B, T_max], padding and all
    parallel-item execution  N forwards of shape [1, T_i], submitted to N CUDA
                             streams and allowed to overlap on the device

In the second, every item is presented to the model in exactly the shape it was
presented in historically. No padding, no batch dimension, the same attention
and mask path, the same numerical policy. Only the SCHEDULE changes. So the
standard here is bit-exactness, not a tolerance:

    parallel-B1 logits == sequential-B1 logits, bitwise

Phases, each gated on the previous one:

    1  logits         plain read-only forwards. Nothing else -- no hooks, no
                      bypass, no shared accumulators, because a race in those
                      would be indistinguishable from a numerical answer
    2  causal KL      B=1 forwards concurrently, assembled into [B, T_pred, V]
                      and reduced AFTER execution. The model forward stays B=1
    3  statistics     per-item sufficient statistics, merged in item order

Phase 2 runs only if phase 1 is exact; phase 3 only if phase 2 is.

    PYTHONPATH=src:scripts python scripts/validation/parallel_item_forward_diagnostic.py \
        --run-id <id> [--device cuda] [--concurrency 1,2,4]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

NOT_RUN = 4
FAILED = 3


def say(msg: str) -> None:
    print(f"[parallel-item] {msg}", flush=True)


def _shared():
    """The batch-invariance diagnostic's helpers, imported rather than copied.

    `compare`, `compare_logits`, `environment_identity`, `executable_identity`,
    `load_model`, `resolve_checkpoint` and `load_items` are already written,
    already exercised, and already the definitions the previous evidence used.
    A second copy would be a second set of metric definitions, and two
    definitions of `rel_l2` is exactly how two reports come to disagree.
    """
    path = REPO / "scripts/validation/batch_invariance_diagnostic.py"
    spec = importlib.util.spec_from_file_location("batch_invariance_diagnostic",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BI = _shared()


# --- execution --------------------------------------------------------------


def sequential_logits(model, items, device):
    """The ORACLE. One item at a time, exactly as the historical path runs.

    Returns logits per item at its own valid positions, on the host, so a later
    device-side wave cannot alias them.
    """
    import torch

    out = []
    with torch.no_grad():
        for item in items:
            ids = item["input_ids"].to(device)
            out.append(model(ids).logits[0].clone())
    return out


def _scheduler(kind: str, n: int):
    """(contexts, synchronize) for one wave.

    `streams` is the subject of the diagnostic. `serial` is the SAME code path
    with the scheduling primitive replaced by a null context -- it exists so
    that every other line here (the permutation, the submission loop, the
    result collection, the clone, the comparison) executes at `$0` on a box
    with no GPU. It answers nothing about concurrency and the report says so;
    what it buys is that the first execution of this function is not on a paid
    pod.
    """
    import contextlib

    import torch

    if kind == "streams":
        streams = [torch.cuda.Stream() for _ in range(n)]
        return ([torch.cuda.stream(s) for s in streams],
                torch.cuda.synchronize)
    if kind == "serial":
        return ([contextlib.nullcontext() for _ in range(n)], lambda: None)
    raise ValueError(f"unknown scheduler {kind!r}")


def parallel_logits(model, items, device, assignment=None, kind="streams"):
    """N independent `[1, T_i]` forwards, one per stream, then one sync.

    Every tensor an item touches is created and consumed inside that item's
    stream, and nothing is read until the global synchronize, so no
    cross-stream hazard exists. The model is READ-ONLY here -- no hooks, no
    bypass, no mutation -- which is what makes a difference in the result
    attributable to scheduling and nothing else.
    """
    import torch

    n = len(items)
    order = list(assignment) if assignment is not None else list(range(n))
    if sorted(order) != list(range(n)):
        raise ValueError(f"assignment must be a permutation of 0..{n-1}")

    contexts, sync = _scheduler(kind, n)
    results: list = [None] * n
    with torch.no_grad():
        for index, item in enumerate(items):
            with contexts[order[index]]:
                ids = item["input_ids"].to(device, non_blocking=True)
                results[index] = model(ids).logits[0]
    sync()
    return [r.clone() for r in results], [order[i] for i in range(n)]


def assemble_prediction_block(per_item, width: int, device):
    """`[1, T_i, V]` logits from independent forwards -> one `[B, width, V]`.

    AFTER execution, never before: the model forward stays `[1, T_i]`, and this
    only reshapes what those forwards returned so the already-validated batched
    reducer can consume it. Row `i` holds that item's `T_i - 1` prediction
    positions and zeros beyond, which the prediction mask excludes.
    """
    import torch

    block = torch.zeros(len(per_item), width, per_item[0].shape[-1],
                        dtype=per_item[0].dtype, device=device)
    for i, t in enumerate(per_item):
        block[i, : t.shape[0] - 1] = t[:-1]
    return block


def wave_time(fn, sync) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


# --- phase 1 ----------------------------------------------------------------


def stage_logits_invariance(cfg, ctx) -> dict:
    """Sequential B=1 versus concurrent B=1, per item, bitwise.

    Bit-exactness is the standard and not a strict one to ask for: the tensor
    shape, the operator path and the numerical policy are all unchanged. If
    these differ, concurrency itself moved the arithmetic.
    """
    import torch

    model, items, device = ctx["model"], ctx["items"], ctx["device"]
    concurrency = int(cfg["concurrency"])
    wave = items[:concurrency]

    kind = ctx["scheduler"]
    oracle = sequential_logits(model, wave, device)
    parallel, used = parallel_logits(model, wave, device, kind=kind)

    rows = []
    for index, (a, b) in enumerate(zip(oracle, parallel)):
        m = BI.compare_logits(a, b)
        rows.append({"item_index": index,
                     "item_id": wave[index].get("item_id", f"i{index}"),
                     "length": int(wave[index]["input_ids"].shape[1]),
                     "stream": used[index], **m})
    return {
        "concurrency": concurrency,
        "stream_assignment": used,
        "per_item": rows,
        "every_item_bitwise_identical": all(r["bitwise_identical"] for r in rows),
        "max_abs_over_items": max(r["max_abs"] for r in rows),
        "min_argmax_agreement": min(r["argmax_agreement"] for r in rows),
        "_standard": ("bitwise. The shape, the operator path and the numerical "
                      "policy are unchanged, so anything else means the "
                      "schedule moved the arithmetic."),
    }


def stage_repeatability(cfg, ctx) -> dict:
    """Three parallel waves. Is concurrent execution even self-consistent?"""
    import torch

    model, items, device = ctx["model"], ctx["items"], ctx["device"]
    wave = items[: int(cfg["concurrency"])]
    runs = [parallel_logits(model, wave, device, kind=ctx["scheduler"])[0]
            for _ in range(3)]
    rows = []
    for index in range(len(wave)):
        pairs = [BI.compare(runs[0][index], runs[r][index]) for r in (1, 2)]
        rows.append({
            "item_index": index,
            "bitwise_identical_across_repeats": all(
                p["bitwise_identical"] for p in pairs),
            "max_abs_between_repeats": max(p["max_abs"] for p in pairs),
        })
    return {
        "repeats": 3,
        "per_item": rows,
        "all_repeats_bitwise_identical": all(
            r["bitwise_identical_across_repeats"] for r in rows),
    }


def stage_stream_assignment(cfg, ctx) -> dict:
    """Does WHICH stream an item lands on change its result?

    If it does, the output depends on stream identity and no assignment is
    safe. Reversed and rotated, both against the same sequential oracle.
    """
    model, items, device = ctx["model"], ctx["items"], ctx["device"]
    n = int(cfg["concurrency"])
    wave = items[:n]
    oracle = sequential_logits(model, wave, device)

    out = {}
    for name, order in (("identity", list(range(n))),
                        ("reversed", list(reversed(range(n)))),
                        ("rotated", [(i + 1) % n for i in range(n)])):
        got, used = parallel_logits(model, wave, device, assignment=order,
                                    kind=ctx["scheduler"])
        rows = [BI.compare(oracle[i], got[i]) for i in range(n)]
        out[name] = {
            "assignment": used,
            "every_item_bitwise_identical": all(r["bitwise_identical"]
                                                for r in rows),
            "max_abs": max(r["max_abs"] for r in rows),
        }
    out["result_is_independent_of_stream_identity"] = all(
        v["every_item_bitwise_identical"] for v in out.values()
        if isinstance(v, dict))
    return out


def stage_throughput(cfg, ctx) -> dict:
    """Sequential wave versus parallel wave, warmed, across a concurrency sweep.

    Not a benchmark campaign. It answers one question: does concurrency buy
    anything here, given that the previous round measured padded batching at
    0.946x the speed of one item at a time.
    """
    import torch

    model, items, device = ctx["model"], ctx["items"], ctx["device"]
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    rows = []
    for c in cfg["concurrency_sweep"]:
        c = int(c)
        if c > len(items):
            continue
        wave = items[:c]
        torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
        sequential_logits(model, wave, device)          # warm
        seq = wave_time(lambda: sequential_logits(model, wave, device), sync)
        peak_seq = (torch.cuda.max_memory_allocated() / 2**30
                    if device.startswith("cuda") else None)

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        parallel_logits(model, wave, device, kind=ctx["scheduler"])   # warm
        par = wave_time(
            lambda: parallel_logits(model, wave, device, kind=ctx["scheduler"]),
            sync)
        peak_par = (torch.cuda.max_memory_allocated() / 2**30
                    if device.startswith("cuda") else None)

        rows.append({
            "concurrency": c,
            "tokens": sum(int(i["input_ids"].shape[1]) for i in wave),
            "sequential_seconds": seq,
            "parallel_seconds": par,
            "speedup": seq / par if par else None,
            "sequential_peak_vram_gib": peak_seq,
            "parallel_peak_vram_gib": peak_par,
            "extra_vram_gib": (peak_par - peak_seq
                               if peak_par is not None and peak_seq is not None
                               else None),
        })
    best = max((r for r in rows if r["speedup"]), key=lambda r: r["speedup"],
               default=None)
    return {
        "per_concurrency": rows,
        "best_speedup": best["speedup"] if best else None,
        "best_concurrency": best["concurrency"] if best else None,
        "concurrency_is_operationally_useful": bool(
            best and best["speedup"] > 1.10),
        "_threshold": ("10% is the bar for 'useful'. Stated before the "
                       "measurement, because a 2% speedup would not justify a "
                       "stream executor and it must not be argued into one."),
    }


# --- phase 2, gated ----------------------------------------------------------


def stage_causal_kl(cfg, ctx) -> dict:
    """The scored quantity, with the MODEL forward kept at B=1.

    Reference and ablated forwards are independent `[1, T_i]` runs submitted
    concurrently; the results are assembled into `[B, T_pred, V]` and the
    prediction mask AFTER execution, and reduced by the operator's own
    `forward_kl_mean_batch`. Only the reduction is tensor-batched -- which is
    the part already validated against the scalar oracle.

    `bypassed_blocks` mutates the model, so it is entered ONCE around a whole
    wave and left after the synchronize. No weight is touched while a stream
    may still be reading.
    """
    import torch
    from aadistill.initialization.calibration.batching import build_batch
    from aadistill.initialization.statistics.contribution import (
        bypassed_blocks, forward_kl_mean, forward_kl_mean_batch)

    model, device = ctx["model"], ctx["device"]
    wave = ctx["items"][: int(cfg["concurrency"])]
    skip = frozenset({int(cfg["bypass_layer"])})

    def forwards(parallel: bool):
        if parallel:
            kind = ctx["scheduler"]
            ref, _ = parallel_logits(model, wave, device, kind=kind)
            with bypassed_blocks(model, skip):
                abl, _ = parallel_logits(model, wave, device, kind=kind)
        else:
            ref = sequential_logits(model, wave, device)
            with bypassed_blocks(model, skip):
                abl = sequential_logits(model, wave, device)
        return ref, abl

    seq_ref, seq_abl = forwards(False)
    par_ref, par_abl = forwards(True)

    #: The scalar oracle, per item, from the sequential forwards.
    oracle = [float(forward_kl_mean(r[:-1], a[:-1]))
              for r, a in zip(seq_ref, seq_abl)]

    #: The batched reducer, over logits produced by CONCURRENT B=1 forwards and
    #: assembled afterwards. `build_batch` supplies the prediction mask.
    from aadistill.initialization.calibration.batching import resolve_pad_id
    batch = build_batch(wave, pad_id=resolve_pad_id(model), device=device)
    mask = batch.prediction_mask().to(device)
    width = mask.shape[1]

    batched = [float(v) for v in forward_kl_mean_batch(
        assemble_prediction_block(par_ref, width, device),
        assemble_prediction_block(par_abl, width, device), mask)]

    rows = [{"item_id": it.get("item_id", f"i{n}"),
             "length": int(it["input_ids"].shape[1]),
             "kl_sequential_scalar_oracle": o,
             "kl_parallel_batched_reducer": b,
             "abs_diff": abs(o - b),
             "rel_diff": abs(o - b) / o if o else None,
             "identical": o == b}
            for n, (it, o, b) in enumerate(zip(wave, oracle, batched))]
    rank_o = sorted(range(len(oracle)), key=lambda i: -oracle[i])
    rank_b = sorted(range(len(batched)), key=lambda i: -batched[i])
    max_rel = max((r["rel_diff"] or 0.0) for r in rows)
    bound = float(cfg["reducer_contract_rel_bound"])
    return {
        "bypassed_layer": int(cfg["bypass_layer"]),
        "per_item": rows,
        "every_item_bitwise_identical": all(r["identical"] for r in rows),
        "max_abs_diff": max(r["abs_diff"] for r in rows),
        "max_rel_diff": max_rel,
        "item_ranking_identical": rank_o == rank_b,
        #: THE acceptance test. Bitwise is the wrong bar here and the review
        #: says so: this compares the SCALAR oracle against the BATCHED
        #: reducer, two different reductions of the same logits, and that
        #: difference is the reducer's own already-validated contract (1.02e-07
        #: relative, measured on CUDA, bit-identical at B=1). The forwards are
        #: what must be bitwise, and phase 1 is where that is required.
        "reducer_contract_rel_bound": bound,
        "within_reducer_contract": max_rel <= bound,
        "_bound_was_declared_before_the_run": True,
        "_model_forward_stayed_b1": True,
    }


# --- configuration and the run ----------------------------------------------

DEFAULTS: dict = {
    "source": {"local_paths": [], "relay_repo": None, "relay_path": None,
               "hub_id": "Qwen/Qwen3-4B-Thinking-2507"},
    "calibration_profile": "calib.domain_balanced@v1",
    "n_items": 8,
    "concurrency": 4,
    "concurrency_sweep": [1, 2, 4],
    "bypass_layer": 17,
    #: Generous over the 1.02e-07 relative drift measured for
    #: `forward_kl_mean_batch` against the scalar oracle on CUDA. Declared here
    #: rather than chosen after seeing a number.
    "reducer_contract_rel_bound": 1e-6,
    "focal_length": 512,
    "neighbour_lengths": [384, 640, 256],
    "seed": 20260926,
}

STAGES = (
    ("logits_invariance", stage_logits_invariance),
    ("repeatability", stage_repeatability),
    ("stream_assignment", stage_stream_assignment),
    ("throughput", stage_throughput),
)
#: Runs ONLY if phase 1 is exact. A causal-KL number measured on top of a
#: forward that already differs would describe nothing.
GATED = (("causal_kl", stage_causal_kl),)


def derive_conclusion(report: dict) -> dict:
    """The verdict, computed from the stages."""
    s = report["stages"]

    def got(stage, *path, default=None):
        node = s.get(stage)
        if not isinstance(node, dict) or "error" in node:
            return default
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    exact = got("logits_invariance", "every_item_bitwise_identical")
    repeatable = got("repeatability", "all_repeats_bitwise_identical")
    stream_free = got("stream_assignment",
                      "result_is_independent_of_stream_identity")
    useful = got("throughput", "concurrency_is_operationally_useful")
    speedup = got("throughput", "best_speedup")

    invariant = bool(exact and repeatable and stream_free)
    if not invariant:
        verdict = "PARALLEL_B1_NOT_INVARIANT"
    elif useful:
        verdict = "PARALLEL_B1_EXACT_AND_FASTER"
    else:
        verdict = "PARALLEL_B1_EXACT_BUT_NO_SPEEDUP"

    return {
        "parallel_b1_verdict": verdict,
        "logits_bitwise_identical_to_sequential": exact,
        "parallel_waves_repeat_bitwise": repeatable,
        "independent_of_stream_identity": stream_free,
        "best_speedup": speedup,
        "best_concurrency": got("throughput", "best_concurrency"),
        "concurrency_is_operationally_useful": useful,
        "causal_kl_reached": "causal_kl" in s,
        "causal_kl_within_reducer_contract": got("causal_kl",
                                                 "within_reducer_contract"),
        "causal_kl_max_rel_diff": got("causal_kl", "max_rel_diff"),
        "causal_kl_ranking_identical": got("causal_kl",
                                           "item_ranking_identical"),
        "causal_kl_bitwise_identical": got("causal_kl",
                                           "every_item_bitwise_identical"),
        "_verdict_rule": ("invariant means exact AND repeatable AND "
                          "independent of stream identity; only then does "
                          "speed decide between the two passing outcomes"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--concurrency-sweep", default=None,
                    help="comma-separated, e.g. 1,2,4,8")
    ap.add_argument("--scheduler", default=None, choices=["streams", "serial"],
                    help="`streams` is the subject. `serial` is the same code "
                         "path with the scheduling primitive replaced by a "
                         "null context; it answers nothing about concurrency "
                         "and exists so every other line runs at $0. Defaults "
                         "to streams on CUDA and serial otherwise.")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without CUDA. The stream stages CANNOT run "
                         "there and are recorded as not-run; this exercises "
                         "the plumbing, never the question.")
    args = ap.parse_args(argv)

    import torch

    cfg = dict(DEFAULTS)
    if args.n_items is not None:
        cfg["n_items"] = args.n_items
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency
    if args.concurrency_sweep:
        cfg["concurrency_sweep"] = [int(x) for x in
                                    args.concurrency_sweep.split(",")]

    out_dir = Path(args.out) if args.out else (
        REPO / "artifacts/validation/parallel_item" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    have_cuda = torch.cuda.is_available()
    if device.startswith("cuda") and not have_cuda:
        if not args.allow_cpu:
            say("no CUDA. CUDA streams are the subject; pass --allow-cpu to "
                "exercise the plumbing and read the result as a different "
                "question.")
            return NOT_RUN
        device = "cpu"

    report: dict = {
        "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "question": ("does submitting N independent [1, T_i] forwards to N "
                     "CUDA streams give bit-identical logits to running them "
                     "one at a time"),
        "device": device,
        "dtype": args.dtype,
        "answers_the_concurrency_question": device.startswith("cuda"),
        "config": cfg,
        "executable": BI.executable_identity(),
        "environment": BI.environment_identity(device),
        "stages": {},
        "stage_status": {},
    }
    #: The HISTORICAL numerical policy, explicitly. The split-K intervention is
    #: deliberately NOT applied: the whole point is to preserve the accepted
    #: sequential B=1 path, so the only variable is the schedule.
    report["bf16_policy"] = BI.read_bf16_policy()
    report["blas_library"] = BI.read_blas_backend()

    say(f"device={device} dtype={args.dtype} torch={torch.__version__}")
    say(f"numerical policy left HISTORICAL: {report['bf16_policy']} "
        f"blas={report['blas_library']}")
    cfg["checkpoint"] = args.checkpoint or BI.resolve_checkpoint(cfg)
    report["checkpoint"] = cfg["checkpoint"]

    model, adapter = BI.load_model(cfg, device, args.dtype)
    report["model"] = {"hidden_size": int(model.config.hidden_size),
                       "num_hidden_layers": int(model.config.num_hidden_layers),
                       "vocab_size": int(model.config.vocab_size),
                       "param_dtype": str(next(model.parameters()).dtype)}
    items, provenance = BI.load_items(cfg, int(model.config.vocab_size))
    report["calibration"] = provenance
    say(f"calibration: {provenance['kind']} "
        f"({provenance.get('n_items')} items)")

    scheduler = args.scheduler or ("streams" if device.startswith("cuda")
                                   else "serial")
    report["scheduler"] = scheduler
    report["answers_the_concurrency_question"] = (
        device.startswith("cuda") and scheduler == "streams")
    say(f"scheduler={scheduler}"
        + ("" if report["answers_the_concurrency_question"]
           else "  (this run exercises the path, not the question)"))
    ctx = {"model": model, "adapter": adapter, "device": device,
           "items": items, "report": report, "scheduler": scheduler}

    if True:
        for name, fn in STAGES:
            say(f"stage {name}")
            try:
                report["stages"][name] = fn(cfg, ctx)
                report["stage_status"][name] = "ok"
            except Exception as exc:                        # noqa: BLE001
                import traceback
                report["stages"][name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
                report["stage_status"][name] = "error"
                say(f"  {name} FAILED: {type(exc).__name__}: {exc}")

        exact = (report["stages"].get("logits_invariance") or {}).get(
            "every_item_bitwise_identical")
        for name, fn in GATED:
            if not exact:
                report["stage_status"][name] = "skipped_phase_1_not_exact"
                say(f"stage {name} SKIPPED: phase 1 was not bit-exact, so a "
                    f"number here would describe nothing")
                continue
            say(f"stage {name}")
            try:
                report["stages"][name] = fn(cfg, ctx)
                report["stage_status"][name] = "ok"
            except Exception as exc:                        # noqa: BLE001
                import traceback
                report["stages"][name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
                report["stage_status"][name] = "error"
                say(f"  {name} FAILED: {type(exc).__name__}: {exc}")

        report["conclusion"] = derive_conclusion(report)
        if not report["answers_the_concurrency_question"]:
            #: A serial-scheduler run can produce a perfect-looking verdict
            #: while proving nothing about concurrency. Say so IN the verdict,
            #: not only in a field beside it.
            report["conclusion"]["parallel_b1_verdict"] = (
                "PATH_EXERCISED_ONLY_" +
                report["conclusion"]["parallel_b1_verdict"])
            report["conclusion"]["_not_a_concurrency_result"] = (
                f"scheduler={scheduler} on device={device}; the scheduling "
                "primitive under test was not used")

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    say(f"wrote {path}")
    say(f"verdict = {report['conclusion']['parallel_b1_verdict']}")
    errors = [n for n, v in report["stage_status"].items() if v == "error"]
    return FAILED if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

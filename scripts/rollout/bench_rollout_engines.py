"""Benchmark one rollout engine, from inside that engine's own image.

    python scripts/rollout/bench_rollout_engines.py --engine vllm_server \
        --base-url http://127.0.0.1:8000 --prompts prompts.json --out report.json

Why this exists separately from `bench_engines.py`
--------------------------------------------------
Engines are benchmarked in their **own official images** (AGENTS.md §4.6), so
this script runs where the project's package and dataset are not installed and
where `transformers` may be a different major version than the trainer's. It
therefore:

* takes prompts **pre-tokenized** as token ids, prepared on the dev box, so no
  tokenizer, chat template or dataset is needed here;
* never loads model weights — the engine already has them;
* speaks only stdlib HTTP through the engine adapters.

Running the client on the pod rather than over the internet keeps network
latency out of the throughput numbers entirely.

What it produces
----------------
One JSON report per arm carrying the environment identity a benchmark needs to
be reproducible (P4) — engine version, CUDA runtime, host driver, torch,
transformers, GPU architecture, attention backend — plus throughput at the
measured job shape, batch sensitivity, observed token/log-prob API behaviour,
and a rollout snapshot with per-token log-probs for the correction diagnostics.

It computes no importance statistics: those need the **trainer** policy, which
lives in the project environment, not in an engine image. The snapshot is
carried back and scored there.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.rollout.engines import SGLangServerEngine, VLLMServerEngine


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}>"


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}>"


def environment_identity() -> dict:
    """Everything an adoption decision has to be able to state (AGENTS.md §4.6).

    Collected rather than assumed: the whole reason this benchmark is being
    re-run is that an earlier session silently measured an engine on a host whose
    driver did not support it.
    """
    smi = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader"])
    identity = {
        "gpu_smi": smi,
        "nvidia_smi_cuda": _run(["bash", "-c",
                                 "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"]),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "vllm": _version("vllm"),
        "sglang": _version("sglang"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        identity["torch_cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            identity["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            identity["gpu_arch_sm"] = f"sm_{major}{minor}"
    except Exception as exc:  # noqa: BLE001
        identity["torch_error"] = f"{type(exc).__name__}: {exc}"
    return identity


def build_engine(name: str, base_url: str, model: str):
    if name == "vllm_server":
        return VLLMServerEngine(base_url, model)
    if name == "sglang_server":
        return SGLangServerEngine(base_url)
    raise SystemExit(f"unknown engine {name!r}")


def timed_run(engine, prompts, *, cap, stops, label, logprobs=False, concurrency=None):
    """One timed pass. Returns a stats dict; never raises — a failed cell is data."""
    subset = prompts if concurrency is None else prompts[:concurrency]
    start = time.perf_counter()
    try:
        completions = engine.generate(
            [p["ids"] for p in subset], max_new_tokens=cap, stop_ids=stops,
            greedy=True, logprobs=logprobs)
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300]}, None
    seconds = time.perf_counter() - start
    new_tokens = sum(c["n_new"] for c in completions)
    return {
        "label": label, "ok": True,
        "prompts": len(subset), "cap": cap,
        "seconds": round(seconds, 2),
        "new_tokens": new_tokens,
        "tokens_per_s": round(new_tokens / seconds, 1) if seconds else 0.0,
        "prompts_per_s": round(len(subset) / seconds, 5) if seconds else 0.0,
        "mean_new_tokens": round(new_tokens / len(completions), 1),
        "hit_cap": sum(c["hit_cap"] for c in completions),
        "finished": sum(c["finished"] for c in completions),
    }, completions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["vllm_server", "sglang_server"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--prompts", required=True, help="JSON from prepare_prompts")
    ap.add_argument("--caps", default="4096",
                    help="comma-separated caps for the long-context sweep")
    ap.add_argument("--concurrency", default="2,8",
                    help="prompt counts per request, for batch sensitivity")
    ap.add_argument("--logprob-cap", type=int, default=256,
                    help="short cap for the log-prob/correction snapshot; kept "
                         "small because the trainer scores these on CPU")
    ap.add_argument("--hourly-usd", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.prompts).read_text())
    prompts, stops = payload["prompts"], set(payload["stop_ids"])
    engine = build_engine(args.engine, args.base_url, args.model)

    report = {
        "engine": args.engine,
        "model": args.model,
        "base_url": args.base_url,
        "environment": environment_identity(),
        "prompt_set": {"n": len(prompts), "sha256": payload.get("sha256"),
                       "slices": payload.get("slices"),
                       "mean_prompt_tokens": payload.get("mean_prompt_tokens")},
        "throughput": [],
        "transport": {},
        "rollouts": [],
    }
    print(json.dumps(report["environment"], indent=2), flush=True)

    # --- throughput + long-context sweep + batch sensitivity ---------------
    for cap in [int(c) for c in args.caps.split(",") if c.strip()]:
        for conc in [int(c) for c in args.concurrency.split(",") if c.strip()]:
            if conc > len(prompts):
                continue
            label = f"cap={cap},concurrency={conc}"
            print(f"\n>>> {label}", flush=True)
            stats, _ = timed_run(engine, prompts, cap=cap, stops=stops, label=label,
                                 concurrency=conc)
            if stats.get("ok") and stats["prompts_per_s"]:
                stats["cost_usd_per_1k_prompts"] = round(
                    1000 / stats["prompts_per_s"] / 3600 * args.hourly_usd, 2)
            report["throughput"].append(stats)
            print(json.dumps(stats), flush=True)

    # --- token / log-prob transport + rollout snapshot ---------------------
    print("\n>>> log-prob transport", flush=True)
    stats, completions = timed_run(engine, prompts, cap=args.logprob_cap, stops=stops,
                                   label="logprobs", logprobs=True)
    report["transport"] = {
        "token_in_token_out": True,   # both adapters refuse a text round-trip
        "logprobs_ok": bool(stats.get("ok")),
        "detail": stats,
    }
    if completions:
        aligned = all(len(c.get("logprobs", [])) == len(c["tokens"]) for c in completions)
        masked = sum(sum(1 for v in c.get("logprobs", []) if v is None)
                     for c in completions)
        report["transport"].update(logprobs_aligned=aligned, masked_positions=masked)
        for prompt, c in zip(prompts, completions):
            report["rollouts"].append({
                "prompt_id": prompt["id"], "prompt_tokens": prompt["ids"],
                "tokens": c["tokens"], "logprobs": c.get("logprobs"),
                "finished": c["finished"], "hit_cap": c["hit_cap"],
            })
    print(json.dumps({k: v for k, v in report["transport"].items() if k != "detail"}),
          flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

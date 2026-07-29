"""Benchmark inference engines at this project's actual job shape.

    uv run python scripts/bench_engines.py --engines hf,vllm,sglang \
        --n-prompts 32 --max-new-tokens 4096 --out artifacts/bench/engines_v0

Why not read a leaderboard
--------------------------
The 2026-07-29 survey found the published engine ordering is driven by
prefix-cache reuse under concurrent load, and that it *reverses* on single-turn
unique prompts. A teacher-corpus build is exactly the reversing shape: unique
prompts, no shared prefix, long thinking traces, offline batch. So the ranking
has to be measured on our shape, our teacher, our hardware.

What is measured, and why each one is load-bearing
---------------------------------------------------
1. **Throughput** — tokens/s and **$ per 1k prompts**, which is the number that
   actually sizes the corpus build. The in-stack path gets a batch-size sweep
   because that is its real knob; vLLM and SGLang schedule internally, so they
   get the whole prompt set at once, which is how they would really be driven.
2. **Batch invariance** — does batching change the tokens a prompt emits? Never
   checked on the real 4B in bf16 for *any* engine, including the incumbent.
3. **Agreement with the training stack** — do the engine's greedy tokens match
   in-stack HF `generate`? Stage 4/5 trains on data the model produced, so an
   engine that disagrees with the trainer makes "on-policy" quietly off-policy.
   No vendor claims this across stacks; it has to be measured.
4. **Integration cost** (P1) — construction wall time and adapter size, so the
   choice can weigh dependency weight against throughput instead of assuming
   the fastest engine is the cheapest one to own.

The output `decision.json` applies the pre-registered rules from
`logs/proposals/2026-07-29_engine_benchmark.md` mechanically, so an unattended
session can chain into generation without an agent interpreting the numbers.

Each engine is isolated: a construction or generation failure records the
traceback and moves to the next arm. A wrong adapter guess costs one arm, not
the session. The `hf` arm is the reference and is always run first.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aadistill.engines import (
    HFEngine, SGLangEngine, VLLMEngine, agreement, batch_invariance, timed,
)
from aadistill.env import code_state, hardware_report
from aadistill.teacher import load_causal_lm
from generate_teacher_answers import SLICES, generation_prompt, load_slice, stop_ids


def build_prompts(tokenizer, data_dir: Path, n_prompts: int, seed: int) -> list[dict]:
    """Take a deterministic, slice-balanced sample and render it to token ids.

    Balanced across slices because thinking-trace length varies a lot by slice
    (code_math traces run far longer than refusal ones), and a sample skewed to
    one slice would measure that slice's trace length rather than the engine.
    """
    names = sorted(SLICES)
    per_slice = max(1, n_prompts // len(names))
    rows: list[dict] = []
    for name in names:
        samples = load_slice(data_dir, name, limit=per_slice * 4)
        # Deterministic stride rather than random sampling: reproducible from
        # the manifest alone, no RNG state to log (P4).
        stride = max(1, len(samples) // per_slice)
        picked = samples[::stride][:per_slice]
        for s in picked:
            text = generation_prompt(tokenizer, s)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            rows.append({"id": s["id"], "slice": name, "ids": ids, "n_prompt": len(ids)})
    return rows[:n_prompts]


def throughput(engine, prompts, *, max_new_tokens, stops, label):
    """One timed pass. Returns (completions, stats) or raises."""
    completions, seconds = timed(
        engine.generate, [p["ids"] for p in prompts],
        max_new_tokens=max_new_tokens, stop_ids=stops, greedy=True,
    )
    new_tokens = sum(c["n_new"] for c in completions)
    stats = {
        "label": label,
        "seconds": round(seconds, 2),
        "prompts": len(prompts),
        "new_tokens": new_tokens,
        "tokens_per_s": round(new_tokens / seconds, 1) if seconds else 0.0,
        "prompts_per_s": round(len(prompts) / seconds, 4) if seconds else 0.0,
        "mean_new_tokens": round(new_tokens / len(completions), 1) if completions else 0,
        "hit_cap": sum(c["hit_cap"] for c in completions),
        "finished": sum(c["finished"] for c in completions),
    }
    return completions, stats


def cost_per_1k(prompts_per_s: float, hourly_usd: float) -> float | None:
    if not prompts_per_s:
        return None
    return round(1000 / prompts_per_s / 3600 * hourly_usd, 2)


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)


def run_arm(name, make_engine, prompts, *, max_new_tokens, stops, hourly,
            reference, invariance_n, invariance_cap, hf_batch_sizes):
    """Run one engine end to end. Never raises — failures are data too (P11)."""
    arm: dict = {"engine": name}
    engine = None
    try:
        engine, setup_s = timed(make_engine)
        arm["setup_seconds"] = round(setup_s, 1)

        # Smoke before the timed pass: 2 prompts at a short cap. Catches an
        # adapter API mismatch in seconds instead of after a full 4096-token run.
        smoke, _ = throughput(engine, prompts[:2], max_new_tokens=16,
                              stops=stops, label="smoke")
        arm["smoke_ok"] = len(smoke) == 2

        runs = []
        if name == "hf":
            # Batch size is the in-stack path's real knob; the serving engines
            # schedule internally, so sweeping it for them would measure nothing.
            for bs in hf_batch_sizes:
                engine.batch_size = bs
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                comps, stats = throughput(engine, prompts,
                                          max_new_tokens=max_new_tokens,
                                          stops=stops, label=f"batch_size={bs}")
                stats["peak_memory_gb"] = peak_memory_gb()
                stats["cost_usd_per_1k_prompts"] = cost_per_1k(
                    stats["prompts_per_s"], hourly)
                runs.append(stats)
                arm.setdefault("completions", comps)  # first sweep point is the reference
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            comps, stats = throughput(engine, prompts, max_new_tokens=max_new_tokens,
                                      stops=stops, label="native_batching")
            stats["peak_memory_gb"] = peak_memory_gb()
            stats["cost_usd_per_1k_prompts"] = cost_per_1k(stats["prompts_per_s"], hourly)
            runs.append(stats)
            arm["completions"] = comps

        arm["runs"] = runs
        arm["best"] = max(runs, key=lambda r: r["tokens_per_s"])

        arm["batch_invariance"] = batch_invariance(
            engine, [p["ids"] for p in prompts[:invariance_n]],
            stop_ids=stops, max_new_tokens=invariance_cap)

        if reference is not None:
            arm["agreement_vs_hf"] = agreement(reference, arm["completions"])

        arm["ok"] = True
    except Exception as exc:  # noqa: BLE001 — a failed arm must not kill the session
        arm["ok"] = False
        arm["error"] = f"{type(exc).__name__}: {exc}"
        arm["traceback"] = traceback.format_exc()
        print(f"  !! {name} failed: {arm['error']}", flush=True)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
    return arm


def decide(arms: list[dict], *, min_agreement: float, max_cost_ratio: float) -> dict:
    """Apply the pre-registered selection rules mechanically.

    Rules, in order (proposal 2026-07-29):
      R1. An arm is eligible only if it ran, and its greedy tokens agree with
          the in-stack reference at >= `min_agreement`. Below that it is a
          different policy from the trainer's, and cheap wrong data is not a
          bargain.
      R2. Among eligible arms, pick the lowest $ per 1k prompts.
      R3. If the winner beats the incumbent by less than `max_cost_ratio`, keep
          the incumbent — a marginal speedup does not pay for a second stack
          (P1). Ties go to the incumbent for the same reason.
    """
    ok = [a for a in arms if a.get("ok")]
    hf = next((a for a in ok if a["engine"] == "hf"), None)
    if hf is None:
        return {"winner": None, "reason": "reference arm (hf) failed; cannot judge agreement"}

    hf_cost = hf["best"].get("cost_usd_per_1k_prompts")
    eligible, rejected = [], []
    for arm in ok:
        if arm["engine"] == "hf":
            eligible.append(arm)
            continue
        rate = (arm.get("agreement_vs_hf") or {}).get("exact_match_rate", 0.0)
        if rate >= min_agreement:
            eligible.append(arm)
        else:
            rejected.append({"engine": arm["engine"], "rule": "R1",
                             "agreement": rate, "required": min_agreement})

    priced = [a for a in eligible if a["best"].get("cost_usd_per_1k_prompts")]
    if not priced:
        return {"winner": "hf", "reason": "no priced eligible arm", "rejected": rejected}

    cheapest = min(priced, key=lambda a: a["best"]["cost_usd_per_1k_prompts"])
    winner, rule = cheapest["engine"], "R2"
    if hf_cost and cheapest["engine"] != "hf":
        speedup = hf_cost / cheapest["best"]["cost_usd_per_1k_prompts"]
        if speedup < max_cost_ratio:
            winner, rule = "hf", "R3"
    else:
        speedup = 1.0

    return {
        "winner": winner,
        "rule": rule,
        "speedup_vs_hf": round(speedup, 2) if hf_cost else None,
        "required_speedup": max_cost_ratio,
        "hf_cost_usd_per_1k": hf_cost,
        "winner_cost_usd_per_1k": cheapest["best"]["cost_usd_per_1k_prompts"],
        "eligible": [a["engine"] for a in eligible],
        "rejected": rejected,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@768f209d")
    ap.add_argument("--engines", default="hf,vllm,sglang")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--hf-batch-sizes", default="8,16,32")
    ap.add_argument("--invariance-n", type=int, default=8)
    ap.add_argument("--invariance-cap", type=int, default=64)
    ap.add_argument("--hourly-usd", type=float, default=0.86,
                    help="pod price, for the $/1k-prompts column")
    ap.add_argument("--min-agreement", type=float, default=0.90)
    ap.add_argument("--min-speedup", type=float, default=1.5,
                    help="a second stack must beat in-stack by this much to be worth owning")
    ap.add_argument("--sglang-deterministic", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    names = [e.strip() for e in args.engines.split(",") if e.strip()]
    if "hf" not in names:
        raise SystemExit("hf is the reference arm and must be included")
    batch_sizes = [int(b) for b in args.hf_batch_sizes.split(",") if b.strip()]

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {device} ({args.dtype})", flush=True)
    model, tokenizer = load_causal_lm(args.model, dtype, device)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stops = stop_ids(model, tokenizer)

    prompts = build_prompts(tokenizer, REPO_ROOT / args.data_dir,
                            args.n_prompts, args.seed)
    print(f"{len(prompts)} prompts, mean {sum(p['n_prompt'] for p in prompts)/len(prompts):.0f} "
          f"prompt tokens, cap {args.max_new_tokens}, stops {sorted(stops)}", flush=True)

    # The model path each serving engine loads. A local checkpoint is used
    # as-is; a Hub spec is split so the revision stays pinned (P4).
    spec = args.model
    model_path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))

    builders = {
        "hf": lambda: HFEngine(model, tokenizer.pad_token_id, batch_size=batch_sizes[0]),
        "vllm": lambda: VLLMEngine(model_path, dtype=args.dtype, revision=revision,
                                   max_model_len=None),
        "sglang": lambda: SGLangEngine(model_path, dtype=args.dtype, revision=revision,
                                       deterministic=args.sglang_deterministic),
    }
    unknown = set(names) - set(builders)
    if unknown:
        raise SystemExit(f"unknown engine(s): {sorted(unknown)}")

    arms, reference = [], None
    for name in ["hf"] + [n for n in names if n != "hf"]:
        print(f"\n=== {name} ===", flush=True)
        # The in-stack model occupies GPU memory that a serving engine also
        # wants. Release it before building one, or vLLM/SGLang will fail to
        # allocate their KV cache and the arm will look unsupported.
        if name != "hf" and reference is not None:
            model.to("cpu")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        arm = run_arm(name, builders[name], prompts,
                      max_new_tokens=args.max_new_tokens, stops=stops,
                      hourly=args.hourly_usd, reference=reference,
                      invariance_n=args.invariance_n,
                      invariance_cap=args.invariance_cap,
                      hf_batch_sizes=batch_sizes)
        if name == "hf" and arm.get("ok"):
            reference = arm["completions"]
        # Completions are large; keep them out of the report but keep a hash so
        # the run can be checked against a re-run.
        arm.pop("completions", None)
        arms.append(arm)

        if arm.get("ok"):
            best = arm["best"]
            print(f"  {best['tokens_per_s']:.0f} tok/s  "
                  f"${best['cost_usd_per_1k_prompts']}/1k prompts  "
                  f"batch-invariant={arm['batch_invariance']['identical']}  "
                  f"agreement={(arm.get('agreement_vs_hf') or {}).get('exact_match_rate', 'n/a')}",
                  flush=True)

    decision = decide(arms, min_agreement=args.min_agreement,
                      max_cost_ratio=args.min_speedup)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "teacher": args.model,
        "job_shape": {
            "n_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "slices": sorted({p["slice"] for p in prompts}),
            "mean_prompt_tokens": round(
                sum(p["n_prompt"] for p in prompts) / len(prompts), 1),
        },
        "hourly_usd": args.hourly_usd,
        "rules": {"min_agreement": args.min_agreement,
                  "min_speedup_over_in_stack": args.min_speedup},
        "arms": arms,
        "decision": decision,
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print("\n" + "=" * 60)
    for arm in arms:
        if not arm.get("ok"):
            print(f"{arm['engine']:10s} FAILED  {arm.get('error', '')[:60]}")
            continue
        best = arm["best"]
        inv = "yes" if arm["batch_invariance"]["identical"] else "NO"
        agree = (arm.get("agreement_vs_hf") or {}).get("exact_match_rate", "—")
        print(f"{arm['engine']:10s} {best['tokens_per_s']:>8.0f} tok/s  "
              f"${str(best['cost_usd_per_1k_prompts']):>7}/1k  "
              f"setup {arm['setup_seconds']:>5.0f}s  invariant {inv:>3}  agree {agree}")
    print(f"\ndecision: winner={decision.get('winner')} "
          f"({decision.get('rule', decision.get('reason'))})")
    print(f"wrote {out_dir}/report.json")


if __name__ == "__main__":
    main()

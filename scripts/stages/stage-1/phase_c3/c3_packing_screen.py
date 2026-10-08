#!/usr/bin/env python3
"""A short real-GPU throughput screen over packing protocols.

    python scripts/stages/stage-1/phase_c3/c3_packing_screen.py --parent DIR --out OUT

Four protocols, all 67 items, all 32 heads, but only three predeclared
layers — 0, 13, 27 — so the measurement spans early, middle and late depth
without paying for all 28. The model forward still traverses the whole model,
so what is measured is the kernel and packing throughput, which is the thing
the protocols differ in.

**It calls `score_heads`, the function the real scorer calls.** Not a copy.
Three paid failures in this project came from measuring a stage beside the
thing it was meant to describe.

A screened run produces an INCOMPLETE landscape by construction, and
`AttentionCausalKLV1.apply` refuses one. Nothing here can become a head map.

Each protocol is warmed identically and CUDA-synchronized at the timing
boundaries; the order is predeclared in the pilot record before any result is
read, because whichever protocol runs first pays for allocator growth the
others do not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PILOT_DIR = "logs/stages/stage-1/phase_c3/pilots/packing-optimization/v1"


def _say(msg: str) -> None:
    print(f"[screen {time.strftime('%H:%M:%S', time.gmtime())}] {msg}",
          flush=True)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                               ensure_ascii=False) + "\n")


def load_scope(repo: Path) -> dict:
    return json.loads((repo / PILOT_DIR / "scope.json").read_text())


def resolve_items(repo: Path, profile_id: str):
    """Delegates. The pilot module owns this; see the note there."""
    from stages.phase_c3.pilot import resolve_items as _resolve

    return _resolve(repo, profile_id)


def run(parent_dir: str, out_dir: Path, *, repo: Path, device: str,
        items=None, layers=None, model=None, deadline=None) -> dict:
    import torch

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    from aadistill.initialization.calibration.packing import (
        item_lengths, padding_profile)
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        attention_out_projection, score_heads, warm_up)
    from aadistill.initialization.operators.register import (
        register_builtin_operators)

    from shared.calibration import register_builtin_profiles

    register_builtin_adapters()
    register_builtin_operators()
    register_builtin_profiles()

    scope = load_scope(repo)
    #: `layers` is supplied only by a toy execution, whose geometry has fewer
    #: blocks than the record names. The REAL path reads the record.
    layers = list(scope["screen"]["layers"] if layers is None else layers)
    order = list(scope["screen"]["protocol_order"])
    protocols = {p["protocol"]: p for p in scope["protocols"]}
    if sorted(order) != sorted(protocols):
        raise RuntimeError(f"screen order {order} does not name the "
                           f"record's protocols {sorted(protocols)}")
    _say(f"predeclared layers {layers}, protocol order {order}")

    #: `items` is supplied only by a toy execution, which has no frozen
    #: mixture. The REAL path resolves and prepares it here, and that branch
    #: is what a pod takes.
    if items is None:
        items = resolve_items(repo, scope["frozen_inputs"]["calibration_profile"])
    lengths = item_lengths(items)
    _say(f"{len(items)} items, {sum(lengths):,} valid positions")

    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            parent_dir, dtype=torch.bfloat16).to(device).eval()
    n_q, n_kv, head_dim = QWEN3_ADAPTER.head_groups(
        QWEN3_ADAPTER.spec_of(model))
    projections = [attention_out_projection(QWEN3_ADAPTER, b)
                   for b in QWEN3_ADAPTER.blocks(model)]
    _say(f"parent loaded: {len(projections)} layers, {n_q} q heads, "
         f"{n_kv} kv heads")

    record = {
        "schema": "aadistill.phase_c3.packing_screen/v1",
        "device": device, "layers": layers, "protocol_order": order,
        "n_items": len(items), "valid_positions": sum(lengths),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": {},
    }
    path = out_dir / "screen.json"
    _write(path, record)

    for position, name in enumerate(order, start=1):
        spec = protocols[name]
        bs = spec["calibration_forward_batch_size"]
        packing = spec["calibration_batch_packing"]
        _say(f"{position}/{len(order)} {name}: B{bs} {packing}")

        #: IDENTICAL WARM-UP, at each protocol's own shape. Allocator growth
        #: and kernel autotuning happen on the first forwards of a shape, and
        #: they would otherwise land entirely on whichever protocol ran first
        #: — a difference between POSITIONS, not between protocols.
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
        warmed = warm_up(model, items, device, batch_size=bs, n=2)

        scored = score_heads(model, items, projections, n_q=n_q,
                             head_dim=head_dim, device=device, batch_size=bs,
                             packing=packing, layers=layers,
                             deadline=deadline, progress=False)
        prof = padding_profile(lengths, bs, packing=packing)
        seconds = scored["scorer_seconds"]
        peak = (int(torch.cuda.max_memory_allocated(device))
                if torch.device(device).type == "cuda" else None)
        record["results"][name] = {
            "protocol": name, "position": position,
            "calibration_forward_batch_size": bs,
            "calibration_batch_packing": packing,
            "wall_seconds": round(seconds, 4),
            "physical_forward_invocations":
                scored["physical_forward_invocations"],
            "invocations_per_minute":
                round(scored["physical_forward_invocations"] / (seconds / 60.0), 2),
            "executed_positions": scored["executed_positions"],
            "valid_positions_per_second":
                round(len(layers) * n_q * sum(lengths) / seconds, 2),
            "executed_padded_positions":
                scored["executed_positions"] - (len(layers) * n_q + 1)
                * sum(lengths),
            "padding_over_valid_full_corpus": prof["padding_over_valid"],
            "peak_vram_bytes": peak,
            "peak_vram_gib": round(peak / 2 ** 30, 3) if peak else None,
            "warmup": warmed,
            "n_groups": scored["n_groups"],
        }
        _write(path, record)
        r = record["results"][name]
        _say(f"  {seconds:.1f}s · {r['physical_forward_invocations']} inv · "
             f"{r['invocations_per_minute']:.0f} inv/min · "
             f"{r['valid_positions_per_second']:.0f} valid pos/s · "
             f"peak {r['peak_vram_gib']} GiB")
        del scored

    #: THE SCREEN GATE, from the record, applied to what was measured.
    ref = scope["screen"]["reference_protocol"]
    threshold = float(scope["screen"]["advance_threshold"])
    base = record["results"][ref]["wall_seconds"]
    ranked = sorted(((n, v["wall_seconds"]) for n, v in record["results"].items()
                     if n != ref), key=lambda kv: kv[1])
    candidates = [{"protocol": n, "wall_seconds": t,
                   "screen_speedup": round(base / t, 4)} for n, t in ranked]
    best = candidates[0] if candidates else None
    record["screen_gate"] = {
        "reference_protocol": ref, "reference_wall_seconds": base,
        "advance_threshold": threshold,
        "_selection": "by measured wall time ALONE; no head-map result is read",
        "ranked": candidates,
        "best": best,
        "advances": bool(best and best["screen_speedup"] >= threshold),
    }
    if not record["screen_gate"]["advances"]:
        record["verdict"] = "NO_BATCH_PACKING_CANDIDATE_WORTH_FULL_SCORER"
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(path, record)
    if best:
        _say(f"best non-reference: {best['protocol']} at "
             f"{best['screen_speedup']}x (threshold {threshold}x) -> "
             f"{'ADVANCE' if record['screen_gate']['advances'] else 'STOP'}")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", required=True,
                    help="the verified pre-ATTENTION parent checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args.parent, out, repo=Path(args.repo), device=device)
    except Exception as exc:      # noqa: BLE001 - a paid pod has died in one
        _write(out / "screen_failure.json", {
            "error": f"{type(exc).__name__}: {exc}",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _say(f"FAILED {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

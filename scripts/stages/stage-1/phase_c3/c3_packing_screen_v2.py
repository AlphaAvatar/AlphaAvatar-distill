#!/usr/bin/env python3
"""The counterbalanced packing screen: two rounds, pooled, with a tie rule.

v1 ran one ordered pass, `P0 → P1 → P2 → P3`, and P3 came last. That
ordering was correctly preregistered, but it separated the two leading
candidates by 0.76% — while their peak VRAM differed by 4.8 GiB. One ordered
pass cannot carry a sub-percent decision: whichever protocol runs last has a
warmer allocator and a hotter cache than the one that ran first, and that
advantage is worth more than 0.76%.

So v2 runs each protocol twice, in opposite orders:

    Round A   R0 → R2 → R3 → R4
    Round B   R4 → R3 → R2 → R0

and selects on the SUM of the two. A position advantage in one round is a
position disadvantage in the other, so pooling cancels what one pass could
not distinguish.

Three guards, all frozen before any timing:

* THE STABILITY GUARD. R0 is measured in both rounds. If the two differ by
  more than 5% of their mean, the environment is too noisy for a
  sub-percent decision and the screen stops. It does NOT add repeats
  opportunistically — that would be choosing the statistic after seeing it.
* THE NEAR-TIE RULE. Candidates within 2% of the fastest are operationally
  tied, and among tied candidates the lower peak VRAM wins. A 0.5% timing
  fluctuation must not buy an execution mode that nearly doubles memory.
* THE ADVANCE GATE. The selected candidate must still reach 1.10× pooled
  against R0, or nothing advances to a full scorer.

Like v1 it calls `score_heads` — the function the real scorer calls, not a
copy — and a screened landscape is incomplete by construction, so nothing
here can become a head map.
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

PILOT_DIR = "logs/stages/stage-1/phase_c3/pilots/packing-optimization/v2"


class ScreenUnstable(RuntimeError):
    """The timing environment cannot carry a sub-percent decision."""


def _say(msg: str) -> None:
    print(f"[screen2 {time.strftime('%H:%M:%S', time.gmtime())}] {msg}",
          flush=True)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True,
                               ensure_ascii=False) + "\n")


def load_scope(repo: Path) -> dict:
    return json.loads((repo / PILOT_DIR / "scope.json").read_text())


def stability(a: float, b: float, tolerance: float) -> dict:
    """R0 against itself, across the two rounds — and the verdict.

    The verdict is computed HERE, not by the caller. It used to be a
    comparison in `run`, where no test could reach it without a model and a
    GPU; a mutation that set it to `True` unconditionally survived. It is
    arithmetic, so it belongs where arithmetic can be tested.

    RELATIVE TO THE MEAN, not to either value: the two rounds are the same
    measurement twice and neither is the baseline.
    """
    mean = (a + b) / 2.0
    delta = abs(a - b)
    relative = round(delta / mean, 6) if mean else 0.0
    return {"round_A": round(a, 4), "round_B": round(b, 4),
            "mean": round(mean, 4), "abs_difference": round(delta, 4),
            "relative": relative, "tolerance": float(tolerance),
            "stable": relative <= float(tolerance)}


def check_rounds(rounds: dict, protocols) -> None:
    """The design is counterbalanced, or it is not a v2 screen.

    Also inside `run` originally, where a mutation removing it survived
    every test: nothing could reach the branch without a GPU.
    """
    names = sorted(protocols)
    for name, order in rounds.items():
        if sorted(order) != names:
            raise RuntimeError(
                f"round {name} order {order} does not name the record's "
                f"protocols {names}")
    if list(rounds["A"]) != list(reversed(rounds["B"])):
        raise RuntimeError(
            "round B is not the reverse of round A, so the design is not "
            "counterbalanced and position cannot cancel")


def select(pooled: dict, vram: dict, reference: str, *, near_tie: float,
           advance: float) -> dict:
    """The frozen selection rule, applied to pooled wall times.

    Ordering among tied candidates: lower peak VRAM, then smaller batch
    size, then lexical protocol id. All three are stated so a tie cannot be
    broken by whatever order a dict happened to iterate in.
    """
    candidates = {k: v for k, v in pooled.items() if k != reference}
    if not candidates:
        raise ValueError("no candidates besides the reference")
    fastest = min(candidates.values())
    tied = sorted(k for k, v in candidates.items()
                  if (v - fastest) / fastest <= near_tie)
    chosen = sorted(tied, key=lambda k: (vram[k]["peak_vram_bytes"] or 0,
                                         vram[k]["calibration_forward_batch_size"],
                                         k))[0]
    speedup = pooled[reference] / pooled[chosen]
    return {
        "reference": reference,
        "pooled_seconds": {k: round(v, 4) for k, v in pooled.items()},
        "speedups_vs_reference": {
            k: round(pooled[reference] / v, 4) for k, v in candidates.items()},
        "fastest_candidate": min(candidates, key=candidates.get),
        "near_tie_fraction": near_tie,
        "tied_with_fastest": tied,
        "_tie_break": "lower peak VRAM, then smaller batch size, then id",
        "selected": chosen,
        "selected_speedup": round(speedup, 4),
        "advance_threshold": advance,
        "advances": speedup >= advance,
        "_selection_is_speed_and_memory_only": (
            "no head-map or quality result is read to choose a protocol"),
    }


def run(parent_dir: str, out_dir: Path, *, repo: Path, device: str,
        items=None, layers=None, model=None, stability_tolerance=None,
        deadline=None) -> dict:
    import torch

    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    from aadistill.initialization.calibration.packing import (
        item_lengths, padding_profile)
    from aadistill.initialization.device import apply_cpu_budget
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        attention_out_projection, score_heads, warm_up)
    from aadistill.initialization.operators.register import (
        register_builtin_operators)

    from shared.calibration import register_builtin_profiles
    from stages.phase_c3 import pilot

    apply_cpu_budget()
    register_builtin_adapters()
    register_builtin_operators()
    register_builtin_profiles()

    scope = load_scope(repo)
    cfg = scope["screen"]
    layers = list(cfg["layers"] if layers is None else layers)
    rounds = {k: list(v) for k, v in cfg["rounds"].items()}
    protocols = {p["protocol"]: p for p in scope["protocols"]}
    reference = cfg["reference_protocol"]
    check_rounds(rounds, protocols)
    _say(f"layers {layers}; round A {rounds['A']}; round B {rounds['B']}")

    if items is None:
        items = pilot.resolve_items(
            repo, scope["frozen_inputs"]["calibration_profile"])
    lengths = item_lengths(items)

    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            parent_dir, dtype=torch.bfloat16).to(device).eval()
    n_q, n_kv, head_dim = QWEN3_ADAPTER.head_groups(QWEN3_ADAPTER.spec_of(model))
    projections = [attention_out_projection(QWEN3_ADAPTER, b)
                   for b in QWEN3_ADAPTER.blocks(model)]
    _say(f"{len(items)} items, {sum(lengths):,} valid positions, "
         f"{len(projections)} layers, {n_q} q heads")

    record = {
        "schema": "aadistill.phase_c3.packing_screen_v2/v1",
        "device": device, "layers": layers, "rounds": rounds,
        "reference_protocol": reference,
        "n_items": len(items), "valid_positions": sum(lengths),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurements": {},
    }
    path = out_dir / "screen_v2.json"
    _write(path, record)

    for round_name in ("A", "B"):
        for position, name in enumerate(rounds[round_name], start=1):
            spec = protocols[name]
            bs = spec["calibration_forward_batch_size"]
            packing = spec["calibration_batch_packing"]
            _say(f"round {round_name} {position}/{len(rounds[round_name])} "
                 f"{name}: B{bs} {packing}")

            #: IDENTICAL WARM-UP per protocol, then a clean peak counter.
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
            warmed = warm_up(model, items, device, batch_size=bs, n=2)

            scored = score_heads(model, items, projections, n_q=n_q,
                                 head_dim=head_dim, device=device,
                                 batch_size=bs, packing=packing,
                                 layers=layers, deadline=deadline,
                                 progress=False)
            prof = padding_profile(lengths, bs, packing=packing)
            peak = (int(torch.cuda.max_memory_allocated(device))
                    if torch.device(device).type == "cuda" else None)
            record["measurements"].setdefault(name, {})[round_name] = {
                "round": round_name, "position": position,
                "calibration_forward_batch_size": bs,
                "calibration_batch_packing": packing,
                "wall_seconds": round(scored["scorer_seconds"], 4),
                "physical_forward_invocations":
                    scored["physical_forward_invocations"],
                "executed_positions": scored["executed_positions"],
                "padded_positions_full_corpus": prof["padded_positions"],
                "padding_over_valid": prof["padding_over_valid"],
                "n_groups": prof["n_groups"],
                "peak_vram_bytes": peak,
                "peak_vram_gib": round(peak / 2 ** 30, 3) if peak else None,
                "warmup": warmed,
            }
            _write(path, record)
            r = record["measurements"][name][round_name]
            _say(f"  {r['wall_seconds']:.1f}s · {r['physical_forward_invocations']} inv "
                 f"· peak {r['peak_vram_gib']} GiB")
            del scored

    # --- the stability guard, before anything is selected -----------------
    ref = record["measurements"][reference]
    #: The record's tolerance, unless a toy execution supplies its own. A
    #: toy protocol runs for ~0.2 s, where 5% is scheduler noise — the guard
    #: would fire on it and prove nothing about the chain. The REAL path
    #: passes nothing and reads 0.05 from the record.
    tol = float(cfg["stability_tolerance"] if stability_tolerance is None
                else stability_tolerance)
    stab = stability(ref["A"]["wall_seconds"], ref["B"]["wall_seconds"], tol)
    stab["_tolerance_source"] = ("record" if stability_tolerance is None
                                 else "caller override (toy only)")
    stab["_predeclared"] = True
    record["stability"] = stab
    _write(path, record)
    _say(f"stability: {reference} A {stab['round_A']}s vs B {stab['round_B']}s "
         f"= {stab['relative'] * 100:.2f}% (tolerance "
         f"{stab['tolerance'] * 100:.0f}%) -> "
         f"{'stable' if stab['stable'] else 'UNSTABLE'}")
    if not stab["stable"]:
        record["verdict"] = "TIMING_SCREEN_UNSTABLE"
        record["_no_opportunistic_repeats"] = (
            "adding rounds after seeing the spread would be choosing the "
            "statistic from the data. A new protocol is needed.")
        record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())
        _write(path, record)
        return record

    # --- pooled selection --------------------------------------------------
    pooled = {k: v["A"]["wall_seconds"] + v["B"]["wall_seconds"]
              for k, v in record["measurements"].items()}
    vram = {k: {"peak_vram_bytes": max(
                    (v["A"]["peak_vram_bytes"] or 0),
                    (v["B"]["peak_vram_bytes"] or 0)),
                "calibration_forward_batch_size":
                    v["A"]["calibration_forward_batch_size"]}
            for k, v in record["measurements"].items()}
    record["round_variation"] = {
        k: {"abs_difference": round(abs(v["A"]["wall_seconds"]
                                        - v["B"]["wall_seconds"]), 4),
            "relative": round(abs(v["A"]["wall_seconds"] - v["B"]["wall_seconds"])
                              / ((v["A"]["wall_seconds"] + v["B"]["wall_seconds"]) / 2), 6)}
        for k, v in record["measurements"].items()}
    record["selection"] = select(
        pooled, vram, reference,
        near_tie=float(cfg["near_tie_fraction"]),
        advance=float(cfg["advance_threshold"]))
    if not record["selection"]["advances"]:
        record["verdict"] = "NO_PACKING_V2_CANDIDATE_WORTH_FULL_SCORER"
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(path, record)
    s = record["selection"]
    _say(f"pooled {s['pooled_seconds']}")
    _say(f"fastest {s['fastest_candidate']}, tied {s['tied_with_fastest']}, "
         f"SELECTED {s['selected']} at {s['selected_speedup']}x -> "
         f"{'ADVANCE' if s['advances'] else 'STOP'}")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", required=True)
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
    run(args.parent, out, repo=Path(args.repo), device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

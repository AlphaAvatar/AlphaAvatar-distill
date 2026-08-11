#!/usr/bin/env python3
"""Re-price E8b's depth-only sessions from the gate's MEASURED step time.

    PYTHONPATH=src python scripts/training/reprice_e8b_after_gate.py

The registered 20-step gate ran and **failed on memory, not speed** (S2 attempt 4,
2026-08-11, A100 SXM 80 GB, commit `ccba0fbf`). It is a stop-and-re-price event, so
nothing here is adopted — this script only states what the measurement implies.

What the gate measured before it died, on the real trainer and the real DP arm:

    step 1  5.39 s      step 2  4.87 s      step 3  4.80 s
    eval step 0: val_ce 1.798891  val_ppl 6.0429  val_kd 1.602439

Step 1 carries warmup, so the steady figure is ~4.83 s/step against a **derived**
7.86 s. The derivation was not wrong in structure — it modelled
`(8·N_student + 2·N_teacher)` FLOPs against E6b's measured 4.15 s/step and applied a
1.6× A100 ratio and a 1.15 safety factor — it was simply conservative. So the
depth-only sessions are materially cheaper than planned.

Then:

    torch.OutOfMemoryError: Tried to allocate 298.00 MiB.
    79.25 GiB capacity, 140.94 MiB free.
    72.44 GiB allocated by PyTorch, 6.16 GiB reserved but unallocated.
    at kd_forward_kl: torch.log_softmax(tp[i:i+chunk].float() / temperature)

The run missed by ~0.3 GB while 6.16 GB sat reserved-and-unusable. `chunk=512` at
vocab 151,936 makes each fp32 buffer 512·151936·4 B = 311 MB, which is the failing
allocation.

Two levers change the memory peak WITHOUT changing training semantics — not token
exposure, not the optimizer schedule, not seeds, not the loss value, not batch size:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — how the caching allocator
   maps segments, nothing else. Recovers fragmentation, which was 6.16 GB here. The
   torch error message recommends it by name.
2. KD `chunk` 512 -> 128 — `kd_forward_kl` sums row-independent log_softmax terms, so
   every partition computes the same quantity. It is **mathematically identical but
   not bitwise identical**: the loop accumulates one float32 scalar per chunk, so the
   chunk count changes the summation order and hence the last bits. Measured at ~7e-8
   relative (`tests/training/test_kd_chunk_invariance.py`, which also pins that the
   difference is real — an earlier check appeared to show bit-identity only because
   54 masked positions made 512/256/128/64 all a single chunk).

   Therefore it must be applied to **both arms of a pair or neither**, which keeps
   each registered contrast exact in the sense that matters. It is NOT proposed for
   S4: the compressed pair needs 23-27 GB on an L40S, has no memory problem, and its
   retained FP control trained at chunk 512. Costs 4x the loop iterations over the
   same tensors, so a small time increase, allowed for below.

Neither is adopted by running this script. Both are a maintainer decision, because a
registered gate failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.budget import Phase, StepTime, plan_session  # noqa: E402

A100_PRICE = 1.59
L40S_PRICE = 0.99
STEPS = 1761

# Measured, not modelled. Steps 2 and 3; step 1 carries warmup.
MEASURED_STEADY_S = 4.83
# The chunk reduction costs loop iterations over the same tensors. 5% is an
# allowance, not a measurement, and is labelled as such in the output.
CHUNK_ALLOWANCE = 1.05
PRICED_STEP_S = round(MEASURED_STEADY_S * CHUNK_ALLOWANCE, 2)

# Measured on attempt 4: setup complete at 15.6 min on a warm host.
SETUP_MIN = 20.0          # 15.6 measured, rounded up for a colder draw
GATE_MIN = 6.0            # throughput gate + pre-training gate
EVAL_MIN_PER_ARM = 30.0
GENERAL_TEXT_MIN = 6.0
MANIFEST_MIN = 8.0
TRANSFER_MB_PER_MIN = 720.0
DEPTH_TRANSFER_MB = 2 * 6430
RESERVE_MIN = 30.0
CONTINGENCY = 0.10

SPENT = {"S1 step-0": 5.21, "S2 attempts 1-3 (setup)": 3.10,
         "S2 attempt 4 (gate)": 0.55}
E8B_BACKSTOP = 47.18
S4_HARD_UNCHANGED = 6.4011


def depth_session(label: str) -> dict:
    phases = (
        Phase("setup_and_build_DP_DC", SETUP_MIN),
        Phase("throughput_and_pretraining_gates", GATE_MIN),
        Phase("train_two_depth_only_arms",
              round(2 * STEPS * PRICED_STEP_S / 60.0, 1)),
        Phase("evaluate_two_arms", 2 * EVAL_MIN_PER_ARM),
        Phase("general_text_diagnostics", GENERAL_TEXT_MIN),
        Phase("artifact_manifest_and_verify", MANIFEST_MIN),
        Phase("artifact_synchronization",
              round(DEPTH_TRANSFER_MB / TRANSFER_MB_PER_MIN, 1)),
    )
    plan = plan_session(
        price_per_hour=A100_PRICE, authorized_usd=1e9, arms=0, steps_per_arm=0,
        step_time=StepTime(PRICED_STEP_S,
                           f"MEASURED {MEASURED_STEADY_S}s/step on A100 SXM 80GB "
                           f"(S2 attempt 4, steps 2-3) x{CHUNK_ALLOWANCE} allowance "
                           "for a smaller KD chunk"),
        setup_minutes=0.0, other_phases=phases,
        contingency_fraction=CONTINGENCY,
        artifact_recovery_reserve_minutes=RESERVE_MIN)
    return {"label": label, "expected_minutes": plan.expected_minutes,
            "expected_usd": plan.expected_usd,
            "soft_stop_usd": plan.soft_stop_usd,
            "hard_terminate_usd": plan.hard_terminate_usd,
            "hard_terminate_minutes": plan.hard_terminate_minutes,
            "usd_per_step": round(A100_PRICE / 3600 * PRICED_STEP_S, 6),
            "breakdown": [(p.name, p.minutes) for p in phases]}


def main() -> int:
    s2 = depth_session("S2 depth-only DP+DC seed sa (re-priced)")
    s3 = depth_session("S3 depth-only DP+DC seed sb (re-priced)")
    spent = round(sum(SPENT.values()), 4)
    remaining = round(E8B_BACKSTOP - spent, 4)
    need = round(s2["hard_terminate_usd"] + s3["hard_terminate_usd"]
                 + S4_HARD_UNCHANGED, 4)

    print("=== what the gate measured ===")
    print(f"  step time            {MEASURED_STEADY_S} s   (derived was 7.86 s)")
    print(f"  priced step time     {PRICED_STEP_S} s   "
          f"(+{(CHUNK_ALLOWANCE-1)*100:.0f}% allowance for the KD chunk change)")
    print(f"  $/step               {s2['usd_per_step']}   "
          "(registered limit 0.003472 — PASSES)")
    print("  peak VRAM            NOT MEASURED — the run OOM'd at 79.10/79.25 GiB")
    print("  verdict              gate FAILED on memory, PASSED on speed and cost")

    for p in (s2, s3):
        print(f"\n=== {p['label']} — A100 SXM 80GB @ ${A100_PRICE}/h ===")
        for name, minutes in p["breakdown"]:
            print(f"  {name:42s} {minutes:7.1f} min")
        print(f"  {'expected':42s} {p['expected_minutes']:7.1f} min  "
              f"${p['expected_usd']:.2f}")
        print(f"  {'absolute termination':42s} "
              f"{p['hard_terminate_minutes']:7.1f} min  "
              f"${p['hard_terminate_usd']:.2f}   (was $18.76)")

    print("\n=== budget, against the EXISTING authorization ===")
    for k, v in SPENT.items():
        print(f"  {k:42s} ${v:7.2f}")
    print(f"  {'spent':42s} ${spent:7.2f}")
    print(f"  {'remaining of the $47.18 backstop':42s} ${remaining:7.2f}")
    print(f"  {'re-priced S2 + S3 + S4 (hard)':42s} ${need:7.2f}")
    print(f"  {'margin':42s} ${remaining - need:7.2f}")
    fits = need <= remaining
    print(f"\n  {'FITS within the existing authorization' if fits else 'SHORTFALL'}"
          f": {'no additional funds required' if fits else f'needs ${need-remaining:.2f} more'}")
    print("  The blocker is memory, not money. Nothing above is adopted; the two "
          "memory levers are a maintainer decision because a registered gate failed.")

    out = REPO_ROOT / "logs/e8b_reprice_after_gate.json"
    out.write_text(json.dumps({
        "created_from": "S2 attempt 4 gate measurement, commit ccba0fbf",
        "measured_steady_s_per_step": MEASURED_STEADY_S,
        "derived_s_per_step": 7.86,
        "priced_s_per_step": PRICED_STEP_S,
        "chunk_time_allowance": CHUNK_ALLOWANCE,
        "gate_verdict": {"speed": "pass", "usd_per_step": "pass",
                         "peak_vram": "not measured — OOM",
                         "blocking_failure": "CUDA OOM in kd_forward_kl"},
        "oom": {"capacity_gib": 79.25, "free_mib": 140.94,
                "allocated_gib": 72.44, "reserved_unallocated_gib": 6.16,
                "failed_allocation_mib": 298.0,
                "site": "kd_forward_kl log_softmax fp32 chunk",
                "chunk_buffer_mb": round(512 * 151936 * 4 / 1e6, 1)},
        "memory_levers_not_adopted": [
            {"lever": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
             "changes_numerics": False,
             "rationale": "allocator segment mapping only; recovers the 6.16 GiB "
                          "reserved-but-unallocated"},
            {"lever": "kd_forward_kl chunk 512 -> 128",
             "changes_numerics": "mathematically identical, NOT bitwise: float32 "
                                 "accumulation order changes with the chunk count",
             "measured_relative_difference": 7e-8,
             "verified": "identical position counts and agreement to <1e-6 relative "
                         "across chunk 512/256/128/64/17/1; the last-bit difference "
                         "is itself pinned by a test",
             "scope_constraint": "apply to BOTH arms of a pair or neither. Not "
                                 "proposed for S4, whose retained FP control trained "
                                 "at chunk 512 and which has no memory problem.",
             "rationale": "row-independent log_softmax summed; ~0.9 GB less "
                          "transient fp32 peak, more loop iterations"}],
        "sessions": {"s2": s2, "s3": s3,
                     "s4_unchanged_hard_usd": S4_HARD_UNCHANGED},
        "spent": SPENT, "spent_total": spent,
        "e8b_backstop_usd": E8B_BACKSTOP, "remaining_usd": remaining,
        "reprice_need_usd": need, "margin_usd": round(remaining - need, 4),
        "fits_existing_authorization": fits,
    }, indent=2) + "\n")
    print(f"\n-> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Price the Phase-C1 session. PRICING ONLY — this authorizes nothing.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/price_c1.py

Every line item is labelled `measured`, `derived` or `unmeasured`, and the
unmeasured ones carry the reasoning behind their bound. Two of them matter and
are stated rather than buried:

* **The 950-prompt evaluation is not free and is not the 190-prompt number.**
  The frozen battery is 5x the recovery-search battery, so the evaluation is
  scaled by 5x on the prompt-proportional part. Generation cost is dominated by
  decoded tokens under a per-sample unrestricted budget, and the C1 battery has
  the same per-stratum composition, so per-prompt cost is treated as unchanged
  and the count is what scales. That assumption is written into the artifact.

* **The attention-statistics pass has never been run on a GPU.** No timing
  exists, so it is bounded from the operators that *have* been measured on the
  same parent under the same profile rather than guessed: FFN and WIDTH each
  make one activation pass over the same 67-item mixture and cost 0.79 and 0.56
  minutes. The attention pass does the same forward work plus a per-head outer
  product, so it is bounded at 4x the slower of those and labelled unmeasured.

Three figures come out: a floor (nothing goes wrong), an expected case, and a
hard ceiling (contingency plus an artifact-recovery reserve). The ceiling is the
only number an authorization should ever be written against.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

OUT = REPO / "logs/phase_c1_pricing.json"

#: Secure-cloud L40S, the field `session_runner` prices on
#: (`gpuTypes[0].securePrice`). Was 0.99 when Phase A/B ran and when C1 was first
#: priced; the live secure rate is 1.09, and C1 attempt 3 refused to create a pod
#: twice because of it. REPRICED 2026-09-04 by maintainer decision.
#:
#: `communityPrice` was 0.79 at the same moment. It is a DIFFERENT product this
#: launcher does not provision, and reporting it as though it gated the launch is
#: a mistake this record now names so it is not repeated.
PRICE_PER_HOUR = 1.09
N_ARMS, N_SEEDS = 2, 3
N_PROBES = N_ARMS * N_SEEDS

# --- measured on the Phase-B pod, same GPU class, same operators -------------
DEPTH_MIN = 25.91              # depth.causal_kl_greedy_v1, 260 evaluations
FFN_MIN = 0.79                 # ffn.activation_importance_v0
WIDTH_MIN = 0.56               # width.global_pca_v0
ATTENTION_WEIGHT_MIN = 0.11    # attention.weight_proxy_v0
PROBE_TRAIN_MIN = 61.55        # 1023 steps at 4.15 s, x1.2 overhead
EVAL_190_MIN = 9.82            # 190-prompt recovery-search battery

# --- session overheads, from the committed continuation pricing artifact -----
SETUP_MIN = 22.0
STAGING_MIN = 8.0
CLOSEOUT_MIN = 12.0
ARTIFACT_RECOVERY_RESERVE_MIN = 20.0
CONTINGENCY_FRACTION = 0.10

# --- new to C1 ---------------------------------------------------------------
TEACHER_GIB = 8045591552 / 2**30
TEACHER_FETCH_MBPS = 100.0     # conservative for a datacentre pod pulling from HF
BATTERY_SCALE = 950 / 190      # 5.0x the recovery-search battery


def minutes_to_usd(m: float) -> float:
    return m / 60.0 * PRICE_PER_HOUR


def main() -> None:
    teacher_fetch_min = TEACHER_GIB * 1024 / TEACHER_FETCH_MBPS / 60
    attention_stats_min = 4 * max(FFN_MIN, WIDTH_MIN)
    eval_950_min = EVAL_190_MIN * BATTERY_SCALE

    items = [
        {"item": "session setup, pod, bundle, prechecks", "minutes": SETUP_MIN,
         "basis": "measured", "source": "continuation pricing fixed_minutes.setup"},
        {"item": "teacher fetch (8,045,591,552 B) and per-shard verification",
         "minutes": round(teacher_fetch_min + 2.0, 2), "basis": "derived",
         "source": (f"{TEACHER_GIB:.2f} GiB at a conservative "
                    f"{TEACHER_FETCH_MBPS:.0f} MB/s, plus 2 min to sha256 three "
                    "shards. NEW to C1: Phase A/B never re-fetched the teacher "
                    "inside a priced session")},
        {"item": "staging: calibration, battery, bundle", "minutes": STAGING_MIN,
         "basis": "measured", "source": "continuation pricing fixed_minutes.staging"},
        {"item": "fixed-parent replay: DEPTH -> FFN -> RESIDUAL_WIDTH",
         "minutes": round(DEPTH_MIN + FFN_MIN + WIDTH_MIN, 2), "basis": "measured",
         "source": ("wall_seconds recorded for exactly these three states in the "
                    "Phase-B search journal: 25.91 + 0.79 + 0.56")},
        {"item": "incumbent replay gate: attention.weight_proxy_v0",
         "minutes": ATTENTION_WEIGHT_MIN, "basis": "measured",
         "source": "fe9683's own ATTENTION step, 0.11 min"},
        {"item": "attention statistics pass + treatment materialization",
         "minutes": round(attention_stats_min + ATTENTION_WEIGHT_MIN, 2),
         "basis": "UNMEASURED",
         "source": ("NO GPU TIMING EXISTS for attention.activation_importance_v1. "
                    "Bounded at 4x the slower of the two measured activation passes "
                    "on the same parent under the same profile (FFN 0.79, WIDTH "
                    "0.56), plus the measured weight-slicing cost. The pass does the "
                    "same forward work plus a per-head outer product accumulated in "
                    "float64; 4x is a bound, not an estimate")},
        {"item": f"{N_PROBES} x 0.86M recovery training",
         "minutes": round(N_PROBES * PROBE_TRAIN_MIN, 2), "basis": "measured",
         "source": (f"{N_PROBES} probes x {PROBE_TRAIN_MIN} min; the same 1023-step "
                    "recipe at the same 4.15 s/step with the same 1.2 overhead "
                    "factor the continuation priced and then observed")},
        {"item": f"{N_PROBES} x 950-prompt evaluation",
         "minutes": round(N_PROBES * eval_950_min, 2), "basis": "derived",
         "source": (f"EXPLICIT SCALING: {EVAL_190_MIN} min measured on the "
                    f"190-prompt recovery-search battery x {BATTERY_SCALE:.1f} "
                    "(950/190). Per-prompt cost is treated as unchanged because the "
                    "C1 battery has the same per-stratum composition and the same "
                    "unrestricted per-sample generation budget; only the count "
                    "scales. NOT assumed free and NOT silently inherited")},
        {"item": "evidence collection, archive, transfer, teardown",
         "minutes": CLOSEOUT_MIN, "basis": "measured",
         "source": "continuation pricing fixed_minutes.closeout"},
    ]

    floor_min = sum(i["minutes"] for i in items)
    contingency_min = floor_min * CONTINGENCY_FRACTION
    expected_min = floor_min + contingency_min
    ceiling_min = expected_min + ARTIFACT_RECOVERY_RESERVE_MIN

    doc = {
        "schema": "aadistill.autoinit.c1_pricing/v1",
        "generated_utc": "2026-09-04T00:00:00Z",
        "_contract": ("A conservative cost bound for the Phase-C1 session. PRICING "
                      "ONLY: this is not an authorization, not a grant, and not "
                      "permission to launch. Nothing here approves spending."),
        "amendment": {
            "date": "2026-09-04",
            "what": "provider rate only, $0.99/h -> $1.09/h secure L40S",
            "why": ("the previous record priced secure L40S at $0.99/h, which is "
                    "what Phase A/B paid and what C1 attempt 2 was created at. The "
                    "live secure rate is $1.09/h, and C1 attempt 3 refused to "
                    "create a pod twice because of it -- correctly, since the "
                    "whole ceiling was derived at $0.99/h."),
            "not_changed": ("every minute assumption. Each line item's USD is "
                            "recomputed from its UNCHANGED minutes at the new rate, "
                            "and the floor/soft/hard minute envelope "
                            "(739.82 / 813.802 / 833.802) is identical."),
            "community_price_is_a_different_product": (
                "communityPrice was $0.79 at the same moment. `session_runner` "
                "prices on `gpuTypes[0].securePrice`, so the $0.79 figure never "
                "gated this session; it was reported as though it did, once, and "
                "that error is named here so the record cannot repeat it."),
            "supersedes": {
                "previous_price_per_hour_usd": 0.99,
                "previous_floor_usd": 12.207,
                "previous_expected_usd": 13.4277,
                "previous_hard_ceiling_usd": 13.7578,
            },
            "budget_effect": ("worst-case cumulative rises from $277.7974 to "
                              "$279.1871 against an unchanged $283.7600 cap, "
                              "leaving a $4.5729 reserve. Approved by maintainer "
                              "review; no cap increase, and no allowance for "
                              "another full C1 retry."),
        },
        "hardware": {"name": "L40S", "price_per_hour_usd": PRICE_PER_HOUR,
                     "basis": "measured; the instance class Phase A/B used"},
        "session_shape": {"arms": N_ARMS, "seeds": N_SEEDS, "probes": N_PROBES,
                          "battery_prompts": 950, "scorable_prompts": 850,
                          "elimination": False},
        "line_items": [dict(i, usd=round(minutes_to_usd(i["minutes"]), 4))
                       for i in items],
        "totals": {
            "floor_minutes": round(floor_min, 3),
            "floor_usd": round(minutes_to_usd(floor_min), 4),
            "contingency_fraction": CONTINGENCY_FRACTION,
            "expected_minutes": round(expected_min, 3),
            "expected_usd": round(minutes_to_usd(expected_min), 4),
            "artifact_recovery_reserve_minutes": ARTIFACT_RECOVERY_RESERVE_MIN,
            "hard_ceiling_minutes": round(ceiling_min, 3),
            # ROUNDED UP, deliberately. A ceiling is the one figure that must
            # never round down: at 4 dp the exact 13.757733 becomes 13.7577,
            # which is $0.000033 BELOW the plan it is meant to authorize, and a
            # grant written at that value fails closed with a $0.00 shortfall.
            # Discovered by building the real BudgetSpec against it.
            "hard_ceiling_usd": math.ceil(minutes_to_usd(ceiling_min) * 10000) / 10000,
            "hard_ceiling_usd_exact": minutes_to_usd(ceiling_min),
            "hard_ceiling_rounding": ("ceil to 4 dp; a ceiling rounds UP or it "
                                      "under-authorizes the plan it prices"),
        },
        "assumptions_that_could_move_the_number": [
            ("The attention-statistics pass is UNMEASURED. If it costs 10x the FFN "
             "pass rather than 4x, the session grows by about "
             f"{round(minutes_to_usd(6 * FFN_MIN), 2)} USD — small, because it runs "
             "once, not once per probe."),
            ("Teacher fetch throughput is assumed conservative at 100 MB/s. A slow "
             "pull is the largest single schedule risk and the watchdog bounds it."),
            ("Setup time has varied 30x across this project's sessions (5, 8.5 and "
             "150+ minutes for the same script and image). The 22-minute figure is "
             "the warm-image case; a cold image is what the reserve is for."),
            ("A replay-gate mismatch at stage D or E ENDS the session before any "
             "recovery training, which costs about "
             f"{round(minutes_to_usd(SETUP_MIN + teacher_fetch_min + 2 + STAGING_MIN + DEPTH_MIN + FFN_MIN + WIDTH_MIN + ATTENTION_WEIGHT_MIN + CLOSEOUT_MIN), 2)} "
             "USD and buys a real scientific finding. That is the cheap outcome, "
             "not the expensive one."),
        ],
        "what_is_not_priced_here": [
            "any authorization, which is a separate maintainer decision",
            "a second attempt after a failure",
            "Phase C2",
            "formal Stage-2/Stage-3 recovery training, which is out of scope",
        ],
        "budget_context": {
            "cumulative_spend_usd": 263.8597,
            "authorized_cap_usd": 283.76,
            "remaining_usd": 19.9003,
            "note": ("remaining headroom is NOT permission. Whether this session may "
                     "be funded is a maintainer decision against the ledger, not a "
                     "consequence of this file."),
        },
        "authorizes": "nothing",
    }
    doc["pricing_sha256"] = sha256_json(doc)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(REPO)}\n")
    print(f"{'line item':52s} {'basis':>11s} {'min':>8s} {'USD':>8s}")
    for i in doc["line_items"]:
        print(f"{i['item'][:52]:52s} {i['basis']:>11s} {i['minutes']:8.2f} "
              f"{i['usd']:8.4f}")
    t = doc["totals"]
    print(f"\n  floor        {t['floor_minutes']:8.1f} min   ${t['floor_usd']:.4f}")
    print(f"  expected     {t['expected_minutes']:8.1f} min   ${t['expected_usd']:.4f}"
          f"   (+{int(CONTINGENCY_FRACTION*100)}% contingency)")
    print(f"  HARD CEILING {t['hard_ceiling_minutes']:8.1f} min   "
          f"${t['hard_ceiling_usd']:.4f}   (+{ARTIFACT_RECOVERY_RESERVE_MIN:.0f} min "
          "artifact-recovery reserve)")
    print(f"\n  pricing_sha256 {doc['pricing_sha256']}")
    print("  AUTHORIZES NOTHING")


if __name__ == "__main__":
    main()

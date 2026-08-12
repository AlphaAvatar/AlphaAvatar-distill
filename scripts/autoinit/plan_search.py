"""Price the v1 AutoInitializer search space. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/plan_search.py \
        --out logs/autoinit_v1_search_space.json

Emits the v1 search-space manifest: what the space contains, how many states a
beam of each width materializes, what that costs in GPU-hours and dollars on each
candidate accelerator, how much working storage it needs, and what the downstream
successive-halving recovery probes add.

Every anchor is either measured and cited, or labelled ESTIMATED. In particular:

* 88.83 TFLOP/s on an L40S — measured, from E8a's 1,300 s depth search;
* 4.15 s/step for a 596M-student recovery step — measured, E6b;
* 1,023 optimizer steps at the 0.86M probe rung — from the frozen E1 config;
* $0.236 per 150-prompt battery evaluation — E6, $2.36 over 10 arms;
* the statistics-pass GPU/CPU split — **not** measured; reported as a range.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.calibration import V1_PROFILES, profile_summary  # noqa: E402
from aadistill.autoinit.cost import (  # noqa: E402
    A100_80GB_ESTIMATED,
    L40S_MEASURED,
    activation_stats_bytes,
    branching_estimate,
    checkpoint_bytes,
    price_search,
)
from aadistill.autoinit.operators import V1_IMPLEMENTATIONS, registry_ledger  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1  # noqa: E402
from aadistill.autoinit.recovery import E1_KD_HEAVY_0860K  # noqa: E402

ADAPTER = get_adapter("qwen3")

TEACHER = ArchSpec.of("qwen3", dict(
    hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))
TARGET = ArchSpec.of("qwen3", dict(
    hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))

# E8a's frozen mixture: 67 items, 59,763 prediction positions, ~892 tokens/item.
CALIBRATION_TOKENS = 59_763
CALIBRATION_SEQ_LEN = 892

# --- measured recovery anchors ---------------------------------------------
PROBE_STEPS_0860K = 1023          # configs/stage3/e1/e1_r0860k_sa_pca.json
SECONDS_PER_STEP_596M = 4.15      # E6b, measured
PROBE_EVAL_OVERHEAD = 1.20        # periodic eval + checkpointing, stated not measured
BATTERY_EVAL_USD = 0.236          # E6: $2.36 over 10 arms on 150 prompts

DECOMPOSED = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) == 1]
COMPOSITE = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) > 1]


def probe_cost(price_per_hour: float) -> dict[str, float]:
    seconds = PROBE_STEPS_0860K * SECONDS_PER_STEP_596M * PROBE_EVAL_OVERHEAD
    return {
        "steps": PROBE_STEPS_0860K,
        "seconds_per_step": SECONDS_PER_STEP_596M,
        "overhead_factor": PROBE_EVAL_OVERHEAD,
        "seconds": seconds,
        "hours": seconds / 3600,
        "train_usd": seconds / 3600 * price_per_hour,
        "battery_eval_usd": BATTERY_EVAL_USD,
        "total_usd": seconds / 3600 * price_per_hour + BATTERY_EVAL_USD,
    }


def halving_cost(top_n: int, survivors: int, price_per_hour: float) -> dict:
    unit = probe_cost(price_per_hour)
    probes = top_n + survivors
    return {
        "top_n": top_n, "survivors": survivors, "probes": probes,
        "per_probe_usd": round(unit["total_usd"], 4),
        "total_usd": round(probes * unit["total_usd"], 4),
        "total_hours": round(probes * unit["hours"], 3),
        "note": (f"{top_n} leaves on seed sa, then {survivors} survivors on seed sb; "
                 "two seeds because the behaviour metric's seed-only spread is 0.1290"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/autoinit_v1_search_space.json")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--profile-counts", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    sweep = []
    for hardware in (L40S_MEASURED, A100_80GB_ESTIMATED):
        for n_profiles in args.profile_counts:
            for beam_width in args.beam_widths:
                estimate = price_search(
                    TEACHER, TARGET, ADAPTER, DECOMPOSED,
                    calibration_tokens=CALIBRATION_TOKENS,
                    suite_tokens=CALIBRATION_TOKENS,
                    seq_len=CALIBRATION_SEQ_LEN,
                    n_profiles=n_profiles, beam_width=beam_width,
                    hardware=hardware, composite=COMPOSITE)
                row = estimate.as_dict()
                row["n_profiles"] = n_profiles
                row["beam_width"] = beam_width
                sweep.append(row)

    reference = branching_estimate(TEACHER, TARGET, ADAPTER, DECOMPOSED,
                                   n_profiles=1, beam_width=1,
                                   include_composite=COMPOSITE)

    manifest = {
        "schema": "aadistill.autoinit.search_space/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study": "4B -> 596M pilot",
        "teacher": {
            "model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "revision": "768f209d9ea81521153ed38c47d515654e938aea",
            "spec": TEACHER.as_dict(), "spec_hash": TEACHER.spec_hash,
            "num_parameters": ADAPTER.param_count(TEACHER),
        },
        "target": {
            "spec": TARGET.as_dict(), "spec_hash": TARGET.spec_hash,
            "num_parameters": ADAPTER.param_count(TARGET),
            "reference_checkpoint": "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
            "reference_sha256": ("86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc"
                                 "952cabd5df2633e54"),
        },
        "adapter": ADAPTER.identity(),
        "operator_registry": registry_ledger(),
        "calibration_profiles": profile_summary(V1_PROFILES),
        "beam_ranking_policy": PARETO_V1.as_dict(),
        "calibration": {"tokens": CALIBRATION_TOKENS, "seq_len": CALIBRATION_SEQ_LEN,
                        "source": "E8a frozen 67-item mixture"},
        "search_space": reference,
        "sweep": sweep,
        "sizes": {
            "teacher_ckpt_gib": checkpoint_bytes(TEACHER, ADAPTER) / 2**30,
            "target_ckpt_gib": checkpoint_bytes(TARGET, ADAPTER) / 2**30,
            "depth_only_intermediate_gib":
                checkpoint_bytes(TEACHER.replace(num_hidden_layers=28), ADAPTER) / 2**30,
            "activation_stats_gib": activation_stats_bytes(TEACHER) / 2**30,
        },
        "recovery": {
            "recipe": E1_KD_HEAVY_0860K.as_dict(),
            "anchors": {
                "probe_steps": PROBE_STEPS_0860K,
                "probe_steps_source": "configs/stage3/e1/e1_r0860k_sa_pca.json",
                "seconds_per_step": SECONDS_PER_STEP_596M,
                "seconds_per_step_source": "E6b measured, 596M student + 4B teacher",
                "battery_eval_usd": BATTERY_EVAL_USD,
                "battery_eval_source": "E6: $2.36 over 10 arms on the 150-prompt battery",
                "overhead_factor": PROBE_EVAL_OVERHEAD,
                "overhead_status": "STATED, not measured",
            },
            "halving_options": [
                halving_cost(n, s, L40S_MEASURED.price_per_hour_usd)
                for n, s in ((3, 2), (4, 2), (6, 3), (8, 3))
            ],
        },
        "caveats": [
            "the statistics-pass GPU/CPU split has never been measured; every cost "
            "here is reported as a low/high range because of it",
            "A100 throughput is scaled from the L40S measurement by peak bf16 ratio "
            "and is ESTIMATED",
            "pod setup time has varied 30x across this project's sessions (5 to 150+ "
            "minutes) and is not included in any figure below",
            "these are search costs only; the recovery probes are priced separately",
        ],
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(f"teacher {ADAPTER.param_count(TEACHER):,} -> target "
          f"{ADAPTER.param_count(TARGET):,} params")
    print(f"differing structural fields: {reference['differing_fields']}")
    print(f"kind orderings {reference['kind_orderings']}, "
          f"geometric paths {reference['complete_paths_geometry_only']}")
    print()
    header = (f"{'hw':<16}{'prof':>5}{'beam':>6}{'states':>14}{'leaves':>12}"
              f"{'hours':>16}{'usd':>16}{'work GiB':>11}")
    print(header)
    print("-" * len(header))
    for row in sweep:
        b = row["branching"]
        print(f"{row['hardware']['name']:<16}{row['n_profiles']:>5}{row['beam_width']:>6}"
              f"{b['states_materialized_min']:>6}-{b['states_materialized_max']:<7}"
              f"{b['leaves_min']:>5}-{b['leaves_max']:<6}"
              f"{row['hours_low']:>7.2f}-{row['hours_high']:<8.2f}"
              f"{row['usd_low']:>7.2f}-{row['usd_high']:<8.2f}"
              f"{row['peak_storage_gib_working']:>11.1f}")
    print()
    print("recovery probes (L40S, $0.99/h):")
    for option in manifest["recovery"]["halving_options"]:
        print(f"  Top-{option['top_n']} -> {option['survivors']} survivors: "
              f"{option['probes']} probes, {option['total_hours']:.1f} h, "
              f"${option['total_usd']:.2f}")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

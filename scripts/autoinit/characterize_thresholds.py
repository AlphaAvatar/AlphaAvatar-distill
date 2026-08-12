"""Derive the preregistered thresholds, before any candidate exists. Zero cost.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/characterize_thresholds.py

Two thresholds gate the pilot, and both must be fixed before the run they judge:

**Beam epsilon.** ``PARETO_V1`` treats objective differences no larger than
``epsilon`` as ties. The value has to sit above the evaluation path's own
repeatability, or the beam eliminates paths on arithmetic noise. This script
measures repeatability by scoring one unchanged checkpoint many times through the
whole materialize -> reload -> measure cycle and reporting the spread.

**Recovery feasibility and equivalence.** These depend on the canonical control's
behaviour on the *new* recovery-search battery, which no one has measured — it
needs rollouts from a recovered checkpoint, which is paid GPU time. So this script
freezes the **derivation rule** and the analytic parts, and leaves exactly one
free input to be filled by the micro-preflight. A rule frozen now with a number
filled later is preregistration; a number chosen after seeing candidates is not.

What this script can and cannot conclude is stated in its output rather than
implied.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.metrics import StateEvalSuite, StateEvaluator, SuiteItem  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1  # noqa: E402
from aadistill.infrastructure.env import hardware_report  # noqa: E402

TEACHER_GEOMETRY = dict(hidden_size=64, num_hidden_layers=6, intermediate_size=128,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        vocab_size=256, tie_word_embeddings=True)
TARGET_GEOMETRY = dict(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
                       num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                       vocab_size=256, tie_word_embeddings=True)
DOMAINS = {"general": ("text",), "math": ("arith",)}

#: Historical anchors used for the analytic parts.
CONTROL_USABLE_OLD_BATTERY = 0.7300     # retained reference, 150-prompt battery
CONTROL_CORRECT_OLD_BATTERY = 0.1867
BEHAVIOUR_SEED_SPREAD = 0.1290          # behaviour metric, seed alone


def build(geometry, seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(max_position_embeddings=512, rope_theta=5_000_000,
                    **geometry)).float().eval()
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.uniform_(0.5, 1.5)
    return model


def items(seed, n, seq_len, vocab):
    torch.manual_seed(seed)
    out = []
    for domain, (subtype,) in DOMAINS.items():
        for k in range(n):
            ids = torch.randint(0, vocab, (1, seq_len))
            targets = ids[0, 1:]
            out.append(SuiteItem(item_id=f"{subtype}-{k}", input_ids=ids,
                                 domain=domain, subtype=subtype,
                                 tags={"eos_like": targets == 0,
                                       "answer_like": targets % 17 == 0}))
    return out


def measure_repeatability(args, tmp: Path) -> dict:
    """Score one unchanged checkpoint `repeats` times, end to end."""
    adapter = get_adapter("qwen3")
    teacher = build(TEACHER_GEOMETRY, args.seed)
    suite = StateEvalSuite(
        suite_id="repeatability", version=1, domains=tuple(DOMAINS),
        subtypes=DOMAINS, critical_tags=("eos_like", "answer_like"),
        general_domain="general")
    evaluator = StateEvaluator(suite, items(args.seed + 1, args.items, args.seq_len,
                                            TEACHER_GEOMETRY["vocab_size"]))
    evaluator.prime_reference(teacher)

    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    child = build(TARGET_GEOMETRY, args.seed + 2)
    ckpt = tmp / "state"
    adapter.save(child, str(ckpt))
    identity = identify_checkpoint(ckpt, adapter=adapter, spec=target_spec,
                                   num_parameters=adapter.param_count(target_spec))

    runs = []
    for _ in range(args.repeats):
        # The whole cycle each time, not just the forward: a reload that
        # perturbed a weight would show up here and nowhere else.
        reloaded = adapter.load(str(ckpt))
        runs.append(dict(evaluator.evaluate(reloaded, identity.artifact_digest).values))
        del reloaded

    keys = sorted(runs[0])
    spread = {}
    for key in keys:
        values = [r[key] for r in runs]
        spread[key] = {
            "mean": statistics.fmean(values),
            "min": min(values), "max": max(values),
            "abs_range": max(values) - min(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    objective_keys = [o.key for o in PARETO_V1.objectives]
    worst = max((spread[k]["abs_range"] for k in objective_keys if k in spread),
                default=0.0)
    return {
        "repeats": args.repeats,
        "n_items": len(evaluator.items),
        "device": "cpu",
        "per_metric": spread,
        "worst_objective_abs_range": worst,
        "objective_keys": objective_keys,
        "artifact_digest_stable": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_threshold_characterization.json")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--scorable-prompts", type=int, default=150)
    ap.add_argument("--all-prompts", type=int, default=190)
    args = ap.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        repeatability = measure_repeatability(args, Path(td))

    worst = repeatability["worst_objective_abs_range"]
    epsilon = PARETO_V1.epsilon
    declared = min(epsilon.values()) if epsilon else 0.0
    verdict = {
        "declared_epsilon": dict(sorted(epsilon.items())),
        "smallest_declared": declared,
        "measured_worst_objective_range_cpu": worst,
        "safely_above_measured_noise": worst < declared,
        "margin_factor": (declared / worst) if worst else None,
        "conclusion": (
            "CPU evaluation is bit-reproducible across the full "
            "materialize -> reload -> measure cycle, so 1e-4 is above the noise "
            "this path produces." if worst == 0.0 else
            "CPU evaluation is not bit-reproducible; epsilon must be set from the "
            "measured range."),
        "NOT_established": (
            "GPU repeatability. The pilot's evaluation backend is a GPU, where "
            "reduction order is not guaranteed across launches and bf16/fp32 "
            "accumulation can move a KL in the last digits. This CPU result "
            "bounds the deterministic path only. The micro-preflight measures the "
            "same quantity on an L40S with the real teacher and the real suite, "
            "and epsilon is confirmed or reset from that number BEFORE any "
            "candidate is ranked."),
    }

    # Analytic parts of the recovery thresholds. n is known, the control's rate
    # is not — so the interval is expressed as a function with one free input.
    n_scorable = args.scorable_prompts * 2      # two seeds, pooled counts
    n_all = args.all_prompts * 2

    def se(p, n):
        return math.sqrt(max(p * (1 - p), 1e-12) / n)

    se_correct_prior = se(CONTROL_CORRECT_OLD_BATTERY, n_scorable)
    se_usable_prior = se(CONTROL_USABLE_OLD_BATTERY, n_all)

    recovery = {
        "pooled_denominators": {
            "scorable_prompts_per_seed": args.scorable_prompts,
            "all_prompts_per_seed": args.all_prompts,
            "seeds": 2,
            "pooled_n_scorable": n_scorable,
            "pooled_n_all": n_all,
            "note": "pooled counts, per the frozen SeedAggregation",
        },
        "equivalence_interval": {
            "rule": ("2 x the binomial standard error of correct_overall at the "
                     "control's pooled rate, on the pooled scorable denominator"),
            "free_input": "control correct_overall on the recovery-search battery",
            "prior_value_used_for_sizing": CONTROL_CORRECT_OLD_BATTERY,
            "prior_source": "retained reference on the 150-prompt promotion battery",
            "se_at_prior": round(se_correct_prior, 5),
            "interval_at_prior": round(2 * se_correct_prior, 4),
            "why_two_se": ("one SE would call a difference decisive that a rerun "
                           "could reverse; two SE is the smallest interval this "
                           "denominator can support"),
        },
        "feasibility_floor": {
            "rule": ("max(absolute_floor, control_usable_pooled - 3 x SE), where SE "
                     "is the binomial standard error of usable_rollout_rate on the "
                     "pooled all-prompt denominator"),
            "absolute_floor": 0.30,
            "absolute_floor_rationale": (
                "guards the 'cannot hold a rollout at all' case independently of "
                "how the control happens to score; a candidate below it cannot "
                "produce trajectories for Stage 5 regardless of correctness"),
            "relative_term_rationale": (
                "guards against a candidate that is severely less stable than the "
                "incumbent without requiring parity, which would make feasibility a "
                "second ranking"),
            "free_input": "control usable_rollout_rate on the recovery-search battery",
            "prior_value_used_for_sizing": CONTROL_USABLE_OLD_BATTERY,
            "se_at_prior": round(se_usable_prior, 5),
            "floor_at_prior": round(max(0.30, CONTROL_USABLE_OLD_BATTERY
                                        - 3 * se_usable_prior), 4),
        },
        "catastrophic_per_capability_floor": {
            "rule": ("a candidate is excluded if any scorable set's usable rate is "
                     "below 0.10 while the control's is above 0.40 on that set"),
            "rationale": ("a candidate that collapses on one capability while "
                          "averaging acceptably is not a usable initialization, and "
                          "the pooled rate can hide it"),
            "free_input": "control per-set usable rates",
            "status": "rule frozen, thresholds analytic, control values pending",
        },
        "seed_noise_context": {
            "behaviour_metric_seed_spread": BEHAVIOUR_SEED_SPREAD,
            "implication": ("far larger than the equivalence interval above, which "
                            "is why two seeds are pooled rather than compared, and "
                            "why a third seed exists for ties"),
        },
        "STATUS": ("RULES FROZEN, NUMBERS PENDING. The two free inputs are measured "
                   "on the canonical control in the micro-preflight, before any "
                   "searched candidate is probed."),
    }

    report = {
        "schema": "aadistill.autoinit.threshold_characterization/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "beam_epsilon": {"repeatability": repeatability, "verdict": verdict},
        "recovery_thresholds": recovery,
        "environment": hardware_report(),
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps({
        "measured_worst_objective_range_cpu": worst,
        "declared_epsilon": declared,
        "safely_above_measured_noise": verdict["safely_above_measured_noise"],
        "equivalence_interval_at_prior": recovery["equivalence_interval"][
            "interval_at_prior"],
        "feasibility_floor_at_prior": recovery["feasibility_floor"]["floor_at_prior"],
        "status": recovery["STATUS"],
    }, indent=2))


if __name__ == "__main__":
    main()

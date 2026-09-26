"""What the C3 pilot's two arms cost, derived rather than asserted.

    PYTHONPATH=src:scripts python scripts/experiments/phase_c3/pricing.py

**The pricing is CONSERVATIVE and IDENTICAL in both arms.** `forward_passes`
is corpus-equivalent *item* forwards — one ablated pass per (layer, head) over
the whole mixture, plus one intact reference pass. Batching changes how those
item-forwards are packed into kernel launches. It changes neither the
arithmetic that must happen nor the tokens it happens over, and if anything it
makes B4 do MORE work than B1, because a padded batch computes positions no
item owns.

So a B4 estimate of `B1 / 4` would not be an estimate at all: it would be the
speedup this pilot exists to measure, written down in advance and then
"confirmed". The observed wall clock is the measurement; this is the budget.

What is reported separately, because they are different quantities and
conflating them is how a plan starts predicting its own result:

    item-forward equivalents   the algorithmic work  (same in both arms)
    physical invocations       kernel launches       (B4 is ~1/4 of B1)
    valid tokens               real positions        (same in both arms)
    padded positions           positions no item owns (B1: zero)
    padding overhead           padded / valid        (B4's extra work)

Everything below is derived from the frozen 67-item mixture through the real
loader and the real grouper. Nothing is typed in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.batching import micro_batches  # noqa: E402
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items)
from aadistill.initialization.operators.attention.gqa.causal_kl import (  # noqa: E402
    ATTENTION_CAUSAL_KL_V1, BATCH_SIZE_CONFIG_KEY)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
from aadistill.runtime.cost import HardwareProfile, operator_cost  # noqa: E402

#: The frozen mixture `calib.domain_balanced@v1` resolves to. Gitignored bytes;
#: the identity is the profile, and the path is where a built tree keeps it.
MIXTURE = "artifacts/stage1/e8_calibration_v1/items.jsonl"

#: The ATTENTION target, verbatim from the committed C1 path record.
C1_PATH_RECORD = "logs/stages/stage-1/phase_c1/runs/attempt9/c1_replay_record.json"

#: The frozen arm identities, which record the parameter count of the actual
#: `eea90c91…` parent and of the ATTENTION arms built from it. The geometry
#: derived below is checked against these.
C1_ARM_IDENTITIES = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
                     "c1_arm_identities.json")

#: The teacher's query-head count. The pre-ATTENTION parent is the C1 target
#: with ATTENTION not yet applied, so it still carries all of them.
PARENT_Q_HEADS = 32

#: Both arms. The only difference between them.
ARMS = (1, 4)

#: The deployment dtype for the logits the scorer holds.
BF16_BYTES = 2


def target_spec(repo: str | Path) -> ArchSpec:
    doc = json.loads((Path(repo) / C1_PATH_RECORD).read_text())
    return ArchSpec.of(doc["path"]["family"], doc["path"]["target_spec"])


def parent_spec(repo: str | Path) -> ArchSpec:
    """The pre-ATTENTION parent: the target with ATTENTION not yet applied.

    CHECKED, not assumed. `PARENT_Q_HEADS` is the one number here that is not
    read out of a record, so the derived geometry's parameter count is
    compared against the count the frozen arm identities recorded for the real
    `eea90c91…` checkpoint -- and the target's against the ATTENTION arms'. A
    wrong head count changes both, and without this the pricing would simply
    be over the wrong model with nothing to notice.
    """
    parent = target_spec(repo).replace(num_attention_heads=PARENT_Q_HEADS)
    arms = json.loads((Path(repo) / C1_ARM_IDENTITIES).read_text())
    expected_parent = int(arms["parent"]["num_parameters"])
    got = QWEN3_ADAPTER.param_count(parent)
    if got != expected_parent:
        raise SystemExit(
            f"derived pre-ATTENTION geometry has {got:,} parameters but the "
            f"frozen parent {arms['parent']['artifact_digest'][:16]}… has "
            f"{expected_parent:,}; PARENT_Q_HEADS={PARENT_Q_HEADS} is wrong")
    expected_child = int(arms["incumbent"]["num_parameters"])
    child = QWEN3_ADAPTER.param_count(target_spec(repo))
    if child != expected_child:
        raise SystemExit(
            f"derived ATTENTION target has {child:,} parameters but the frozen "
            f"arms have {expected_child:,}")
    return parent


def mixture(repo: str | Path) -> list[dict]:
    path = Path(repo) / MIXTURE
    if not path.is_file():
        raise SystemExit(
            f"{MIXTURE} is absent. `artifacts/` is gitignored; build or fetch "
            "the frozen mixture before pricing against it.")
    raw = [json.loads(line) for line in path.read_text().splitlines() if line]
    return prepare_calibration_items(raw, profile_id="calib.domain_balanced@v1")


def grouping(items, batch_size: int) -> dict:
    """The five quantities, from the REAL grouper over the REAL lengths."""
    groups = list(micro_batches(items, batch_size, pad_id=0, device="cpu"))
    valid = sum(int(i["input_ids"].shape[-1]) for i in items)
    padded = sum(g.size * int(g.input_ids.shape[1]) - g.n_valid_tokens
                 for g in groups)
    return {
        "calibration_forward_batch_size": batch_size,
        "n_groups": len(groups),
        "valid_tokens": valid,
        "padded_positions": int(padded),
        "padding_overhead_ratio": (padded / valid) if valid else 0.0,
        "max_group_width": max(int(g.input_ids.shape[1]) for g in groups),
        "group_sizes": [g.size for g in groups],
    }


def derive(repo: str | Path, price_per_hour_usd: float,
           tflops: float) -> dict:
    parent, target = parent_spec(repo), target_spec(repo)
    items = mixture(repo)
    n_q = parent["num_attention_heads"]
    ablations = parent["num_hidden_layers"] * n_q

    hardware = HardwareProfile(
        name="pricing-estimate", price_per_hour_usd=price_per_hour_usd,
        effective_tflops=tflops, vram_gb=48.0, measured=False,
        source="estimate for this derivation; NOT a live provider quote")

    arms = {}
    for batch_size in ARMS:
        config = {"n_calibration_items": len(items),
                  BATCH_SIZE_CONFIG_KEY: batch_size}
        plan = ATTENTION_CAUSAL_KL_V1.plan(parent, target, QWEN3_ADAPTER, config)
        #: `operator_cost` plans at `n_calibration_items = 1` and multiplies
        #: by the token budget itself, so it never sees the batch size. That is
        #: the conservative property, structural rather than promised: the cost
        #: model CANNOT price the two arms differently. `plan.forward_passes`
        #: above is computed with each arm's real config, so the equality
        #: asserted below is a statement about the OPERATOR, not about this
        #: function's blindness.
        cost = operator_cost(ATTENTION_CAUSAL_KL_V1, parent, target, QWEN3_ADAPTER,
                             calibration_tokens=sum(
                                 int(i["input_ids"].shape[-1]) for i in items),
                             seq_len=max(int(i["input_ids"].shape[-1])
                                         for i in items),
                             hardware=hardware)
        g = grouping(items, batch_size)
        #: One reference invocation per group plus one ablated invocation per
        #: (group, layer, head) — exactly the loop the operator runs.
        physical = g["n_groups"] * (ablations + 1)
        #: PEAK LOGIT RESIDENCY, from the tensors the loop actually holds at
        #: once -- not from a per-step model. The reference block lives for
        #: the whole group and the ablated block is deleted each iteration, so
        #: two blocks are simultaneously live and no more. The reducer's fp32
        #: chunks are on top of this and are bounded by `chunk=512` positions.
        logit_bytes = (batch_size * g["max_group_width"]
                       * target["vocab_size"] * BF16_BYTES)
        arms[f"B{batch_size}"] = {
            **g,
            "peak_logit_bytes": 2 * logit_bytes,
            "peak_logit_gib": round(2 * logit_bytes / 2 ** 30, 3),
            "item_forward_equivalents": plan.forward_passes,
            "physical_forward_invocations": physical,
            "estimated_gpu_seconds": cost.gpu_seconds,
            "estimated_usd": cost.gpu_seconds / 3600.0 * price_per_hour_usd,
        }

    b1, b4 = arms["B1"], arms["B4"]
    return {
        "schema": "aadistill.phase_c3.batching_pilot_pricing/v1",
        "_contract": ("Conservative and arm-identical by construction. The "
                      "estimate must not encode the speedup the pilot measures."),
        "mixture": {"path": MIXTURE, "n_items": len(items),
                    "profile_id": "calib.domain_balanced@v1"},
        "geometry": {"parent_q_heads": n_q,
                     "target_q_heads": target["num_attention_heads"],
                     "kv_heads": parent["num_key_value_heads"],
                     "layers": parent["num_hidden_layers"],
                     "ablations": ablations},
        "hardware_estimate": {"price_per_hour_usd": price_per_hour_usd,
                              "effective_tflops": tflops,
                              "_not_a_quote": "re-query the provider before acquiring"},
        "arms": arms,
        "derived": {
            "arms_price_identically":
                b1["item_forward_equivalents"] == b4["item_forward_equivalents"],
            "physical_invocation_ratio_b1_over_b4":
                b1["physical_forward_invocations"] / b4["physical_forward_invocations"],
            "b4_extra_positions_vs_b1":
                b4["padded_positions"] - b1["padded_positions"],
            "both_arms_estimated_usd": b1["estimated_usd"] + b4["estimated_usd"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-per-hour-usd", type=float, default=1.09,
                    help="ESTIMATE only. The live quote is re-queried at launch.")
    ap.add_argument("--effective-tflops", type=float, default=90.0)
    ap.add_argument("--write", type=str, default="")
    args = ap.parse_args()

    doc = derive(REPO, args.price_per_hour_usd, args.effective_tflops)
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    print(f"mixture: {doc['mixture']['n_items']} items, "
          f"{b1['valid_tokens']:,} valid tokens")
    print(f"geometry: {doc['geometry']['layers']} layers x "
          f"{doc['geometry']['parent_q_heads']} heads = "
          f"{doc['geometry']['ablations']} ablations, + 1 reference")
    print(f"{'':>4} {'groups':>7} {'item-fwd':>10} {'physical':>9} "
          f"{'padded':>8} {'pad/valid':>10} {'logitGiB':>9} {'gpu_s':>9} {'usd':>7}")
    for name, arm in doc["arms"].items():
        print(f"{name:>4} {arm['n_groups']:>7} {arm['item_forward_equivalents']:>10,} "
              f"{arm['physical_forward_invocations']:>9,} "
              f"{arm['padded_positions']:>8,} "
              f"{arm['padding_overhead_ratio']:>10.4f} "
              f"{arm['peak_logit_gib']:>9.2f} "
              f"{arm['estimated_gpu_seconds']:>9.0f} "
              f"{arm['estimated_usd']:>7.2f}")
    d = doc["derived"]
    print(f"arms price identically: {d['arms_price_identically']}  |  "
          f"physical B1/B4 = {d['physical_invocation_ratio_b1_over_b4']:.2f}x  |  "
          f"both arms ~${d['both_arms_estimated_usd']:.2f} "
          f"(estimate at ${args.price_per_hour_usd}/h, not a quote)")
    if not d["arms_price_identically"]:
        print("REFUSING: the two arms do not price identically; the estimate "
              "would be asserting the pilot's result")
        return 2
    if args.write:
        out = REPO / args.write
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

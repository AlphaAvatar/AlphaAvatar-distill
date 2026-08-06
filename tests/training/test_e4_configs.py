"""Experiment 4 configs: P2-CE-heavy moved from the 0.86M rung to the 1.60M rung.

The experiment's whole claim is that **only the rung changed**. Everything a
rung legitimately drags with it — block count, supervised tokens, optimizer
steps, warmup, checkpoint and eval cadence — is taken from the tracked E1 config
for that same rung rather than invented, so "the schedule changed too" is a
consequence of the rung and not a second free variable.

Also asserted here, because each has a way of going wrong silently:

* the 1.60M rung is a **strict superset** of the 0.86M rung, block for block;
* both seeds keep the deterministic block order the earlier arms used;
* attention projections are full-rank trainable and embeddings/`lm_head` frozen
  (i.e. none of Experiment 3's freeze policy or LoRA leaked in);
* the objective is still CE 1.0 / KD 0.25 at `kd_scope: all`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.data.ladder import ladder_blocks, load_ladder_meta  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.training.train import (  # noqa: E402
    select_trainable, stream_block_indices, validate_train_config,
)

SEEDS = {"sa": 20260726, "sb": 20260801}
P2_CONFIG_SHA = {"sa": "42616c1921419d01", "sb": "b846fee7bcae670f"}
OBJECTIVE = {"ce_weight": 1.0, "kd_weight": 0.25,
             "kd_temperature": 1.0, "kd_scope": "all"}
PACK = REPO / "artifacts/stage3/ladder_uniform_probe"

# The rung drags these with it; every one is copied from the tracked E1 config
# for the 1.60M rung, so this set is exhaustive and closed.
RUNG_DERIVED = {"rung", "schedule", "checkpoint", "intervals"}
IDENTITY = {"run_name", "_purpose", "out_dir"}


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def p2(seed: str) -> dict:
    return load(f"configs/stage3/p2/p2_ceheavy_{seed}.json")


def e4(seed: str) -> dict:
    return load(f"configs/stage3/e4/e4_p2_r1600k_{seed}.json")


def e1_1600k(seed: str) -> dict:
    return load(f"configs/stage3/e1/e1_r1600k_{seed}_pca.json")


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_reference_is_still_the_p2_arm_we_think_it_is(seed):
    assert sha256_json(p2(seed))[:16] == P2_CONFIG_SHA[seed]
    assert p2(seed)["rung"] == 860000 and p2(seed)["loss"] == OBJECTIVE


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_e4_differs_from_p2_only_by_the_rung_and_what_the_rung_derives(seed):
    ref, arm = p2(seed), e4(seed)
    differing = {k for k in set(ref) | set(arm)
                 if ref.get(k) != arm.get(k)} - IDENTITY
    assert differing == RUNG_DERIVED, differing


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_rung_derived_fields_come_from_the_tracked_e1_config(seed):
    """Not invented for this experiment: copied from the same rung's E1 arm."""
    arm, e1 = e4(seed), e1_1600k(seed)
    for field in RUNG_DERIVED:
        assert arm[field] == e1[field], field
    # 3 passes over 1,174 blocks at 2 blocks/step = 1,761 steps, and warmup is
    # 5% of total as at every other rung (0.86M: 51/1023 = 4.99%).
    blocks, per_step = 1174, arm["batch"]["blocks_per_step"]
    assert arm["schedule"]["total_steps"] == blocks * 3 // per_step == 1761
    assert arm["schedule"]["warmup_steps"] == 88
    assert abs(88 / 1761 - 51 / 1023) < 0.002


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_objective_optimizer_and_data_source_are_unchanged(seed):
    ref, arm = p2(seed), e4(seed)
    for field in ("loss", "optim", "batch", "teacher", "student_path", "data_dir",
                  "packing", "block_len", "dtype", "autocast_bf16",
                  "gradient_checkpointing", "seed", "val_blocks",
                  "trainable_patterns"):
        assert arm[field] == ref[field], field
    assert arm["loss"] == OBJECTIVE
    assert "truncate_padding" not in arm["batch"]
    assert arm["student_path"] == "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_no_experiment3_freeze_policy_or_lora_leaked_in(seed):
    arm = e4(seed)
    assert "lora" not in arm
    for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
        assert field not in arm["optim"]
    # Attention projections must be back in the trainable set.
    joined = " ".join(arm["trainable_patterns"])
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert proj in joined, proj


def test_trainable_policy_on_the_real_geometry():
    """Full-rank attention + FFN + all norms; embeddings and lm_head frozen."""
    from transformers import AutoConfig, AutoModelForCausalLM

    init = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(init))
    report = select_trainable(model, e4("sa")["trainable_patterns"])
    assert report["lora_trainable_params"] == 0
    assert report["trainable_params"] == 440_467_456
    assert report["total_params"] == 596_049_920

    trainable = set(report["trainable_names"])
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert "model.embed_tokens.weight" in frozen
    assert not any("lm_head" in n for n in trainable)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert sum(f".self_attn.{proj}." in n for n in trainable) == 28, proj
    for norm in ("input_layernorm", "post_attention_layernorm",
                 "self_attn.q_norm", "self_attn.k_norm"):
        assert sum(norm in n for n in trainable) == 28, norm
    assert "model.norm.weight" in trainable


@pytest.mark.skipif(not (PACK / "blocks.npz").is_file(), reason="pack absent")
def test_1600k_is_a_strict_superset_of_860k_block_for_block():
    """Nesting is the reason this is a scale experiment and not a new dataset."""
    meta = load_ladder_meta(PACK)
    n860 = next(r["n_blocks"] for r in meta["rungs"]
                if r["target_supervised_tokens"] == 860000)
    n1600 = next(r["n_blocks"] for r in meta["rungs"]
                 if r["target_supervised_tokens"] == 1600000)
    assert (n860, n1600) == (682, 1174)

    small, _, s_stats = ladder_blocks(PACK, 860000, n_val=16)
    large, _, l_stats = ladder_blocks(PACK, 1600000, n_val=16)
    # A rung is a prefix of the pack, so the smaller must be the larger's prefix
    # -- asserted on the actual token ids, not on the block count.
    assert np.array_equal(small[0].numpy(), large[0][:n860].numpy())
    assert np.array_equal(small[1].numpy(), large[1][:n860].numpy())
    assert s_stats["train_supervised_tokens"] == 864_750
    assert l_stats["train_supervised_tokens"] == 1_600_353


@pytest.mark.skipif(not (PACK / "blocks.npz").is_file(), reason="pack absent")
def test_validation_blocks_are_identical_and_disjoint_from_both_rungs():
    """Both rungs must be scored on the same held-out tail, or CE is not comparable."""
    _, _, s = ladder_blocks(PACK, 860000, n_val=16)
    _, _, l = ladder_blocks(PACK, 1600000, n_val=16)
    assert s["val_block_indices"] == l["val_block_indices"]
    assert min(s["val_block_indices"]) >= 2941      # past the largest rung
    assert s["val_disjoint_from_all_rungs"] and l["val_disjoint_from_all_rungs"]


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_block_order_extends_the_earlier_arms_deterministically(seed):
    """Same seed, so epoch 0 of the 1.60M rung is a permutation of its own blocks.

    The 0.86M and 1.60M arms do NOT see the same order — the permutation is over
    a different number of blocks — which is why this experiment compares rungs
    and not orderings. What must hold is that the order is a pure function of
    (n_blocks, seed) and reproduces from the consumed-block counter alone.
    """
    s = SEEDS[seed]
    full = stream_block_indices(1174, s, 0, 1174 * 2)
    assert sorted(full[:1174]) == list(range(1174))
    assert sorted(full[1174:]) == list(range(1174))
    assert stream_block_indices(1174, s, 700, 50) == full[700:750]
    other = stream_block_indices(1174, SEEDS["sb" if seed == "sa" else "sa"], 0, 20)
    assert other != full[:20]


def test_configs_validate_and_are_distinct():
    hashes = set()
    for seed in SEEDS:
        cfg = e4(seed)
        validate_train_config(cfg)
        hashes.add(sha256_json(cfg))
        assert cfg["out_dir"] == f"artifacts/stage3/e4_p2_r1600k_{seed}"
    assert len(hashes) == 2
    assert not (hashes & {sha256_json(p2(s)) for s in SEEDS})
    assert not (hashes & {sha256_json(e1_1600k(s)) for s in SEEDS})

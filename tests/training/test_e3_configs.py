"""Experiment 3 configs: the single-variable guarantee, checked mechanically.

The whole experiment rests on A1 and A2 differing from the 0.86M P1 control in
exactly one thing each. Asserting that by eye across five nested config
dictionaries is how a silent second variable gets shipped, so it is asserted
here instead — and the same assertions run on the pod before any GPU time is
spent.

A0 = P1 = `e1_r0860k_{sa,sb}_pca`, config sha256 `08264ef1…` / `9048173d…`.
A1 = A0 minus the four attention projections from `trainable_patterns`.
A2 = A1 plus LoRA on those same four projections.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.training.lora import LoRAConfig  # noqa: E402
from aadistill.training.train import validate_train_config  # noqa: E402

SEEDS = {"sa": 20260726, "sb": 20260801}
# Recomputed from the tracked configs, and matching the registration in
# EXPERIMENTS.md §16.1.
P1_CONFIG_SHA = {"sa": "08264ef1225c119a", "sb": "9048173dc62cca84"}

ATTENTION_PROJECTIONS = r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\."
ATTENTION_NORMS_ONLY = r"\.self_attn\.(q_norm|k_norm)\."


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def p1(seed: str) -> dict:
    return load(f"configs/stage3/e1/e1_r0860k_{seed}_pca.json")


def a1(seed: str) -> dict:
    return load(f"configs/stage3/e3/e3_a1_frozen_attn_{seed}.json")


def a2(seed: str) -> dict:
    return load(f"configs/stage3/e3/e3_a2_lora_attn_{seed}.json")


# Fields that identify the run rather than define it, plus the one field each
# arm is allowed to change.
IDENTITY = {"run_name", "_purpose", "out_dir"}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_the_control_is_the_arm_we_think_it_is(seed):
    """A0 must still hash to the value the D0 registration recorded."""
    assert sha256_json(p1(seed))[:16] == P1_CONFIG_SHA[seed]
    assert p1(seed)["seed"] == SEEDS[seed]
    assert p1(seed)["rung"] == 860000
    assert p1(seed)["student_path"] == "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_a1_differs_from_the_control_only_by_the_freeze_set(seed):
    control, arm = p1(seed), a1(seed)
    differing = {k for k in set(control) | set(arm)
                 if control.get(k) != arm.get(k)} - IDENTITY
    assert differing == {"trainable_patterns"}, differing

    dropped = set(control["trainable_patterns"]) - set(arm["trainable_patterns"])
    added = set(arm["trainable_patterns"]) - set(control["trainable_patterns"])
    assert dropped == {ATTENTION_PROJECTIONS}
    assert added == {ATTENTION_NORMS_ONLY}
    # Everything the control trained outside attention projections still trains.
    for pattern in ("input_layernorm", "post_attention_layernorm", r"model\.norm\."):
        assert pattern in arm["trainable_patterns"]
    assert r"\.mlp\.(gate_proj|up_proj|down_proj)\." in arm["trainable_patterns"]
    assert "lora" not in arm


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_a2_differs_from_a1_only_by_the_adapter(seed):
    base, arm = a1(seed), a2(seed)
    differing = {k for k in set(base) | set(arm)
                 if base.get(k) != arm.get(k)} - IDENTITY
    assert differing == {"lora"}, differing

    cfg = LoRAConfig.from_dict(arm["lora"])
    assert (cfg.rank, cfg.alpha, cfg.dropout, cfg.bias) == (8, 16.0, 0.0, "none")
    assert cfg.scaling == 2.0
    assert cfg.target_patterns == (r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",)
    # No separate optimizer settings anywhere in the arm.
    assert arm["optim"] == base["optim"] == p1(seed)["optim"]
    for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
        assert field not in arm["optim"]


def test_both_a2_arms_share_one_pinned_adapter_start_point():
    """The run seed varies block order only, exactly as it does in A0 and A1."""
    sa, sb = a2("sa"), a2("sb")
    assert sa["lora"] == sb["lora"]
    assert sa["seed"] != sb["seed"]
    assert sa["lora"]["seed"] not in (sa["seed"], sb["seed"])


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_arms_fork_from_the_same_pinned_stage1_init_and_data(seed):
    control = p1(seed)
    for arm in (a1(seed), a2(seed)):
        for field in ("student_path", "data_dir", "packing", "rung", "block_len",
                      "val_blocks", "seed", "teacher", "loss", "optim",
                      "schedule", "batch", "dtype", "autocast_bf16",
                      "gradient_checkpointing"):
            assert arm[field] == control[field], field
        assert "truncate_padding" not in arm["batch"]


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_every_e3_config_validates_and_has_a_distinct_out_dir(seed):
    dirs = set()
    for arm in (a1(seed), a2(seed)):
        validate_train_config(arm)
        dirs.add(arm["out_dir"])
    assert len(dirs) == 2
    assert not any(d.startswith("artifacts/stage3/e1_") for d in dirs)


def test_the_four_arms_have_four_distinct_config_hashes():
    hashes = {sha256_json(cfg) for seed in SEEDS
              for cfg in (a1(seed), a2(seed))}
    assert len(hashes) == 4
    assert not (hashes & {sha256_json(p1(s)) for s in SEEDS})

"""LoRA tests for Experiment 3 arm A2 (CPU, tiny models, fast).

The load-bearing properties, in the order the experiment depends on them:

1. applying LoRA changes nothing (zero initial delta), so A2 forks from exactly
   the same Stage 1 model as A0 and A1;
2. the merged checkpoint is a plain Hugging Face checkpoint — no adapter is
   required at evaluation time, so all three arms share one inference
   architecture;
3. merged and unmerged agree in BF16, so the artifact that gets measured is the
   model that was trained;
4. resume is exact, including the next batch and every optimizer moment.

Everything else checks the freeze policy, the parameter-group construction and
the failure modes that would otherwise turn into a silently wrong arm.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.training.lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    apply_lora,
    lora_and_base_tensors,
    load_lora_and_base_,
    merged_state_dict,
)
from aadistill.training.train import (  # noqa: E402
    Trainer,
    validate_train_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train import VOCAB, tiny_model, toy_blocks, toy_cfg  # noqa: E402

A2_PATTERNS = [
    r"\.mlp\.(gate_proj|up_proj|down_proj)\.",
    "input_layernorm",
    "post_attention_layernorm",
    r"model\.norm\.",
    r"\.self_attn\.(q_norm|k_norm)\.",
]


def lora_cfg(**overrides) -> dict:
    return {"rank": 4, "alpha": 8, "dropout": 0.0, "bias": "none",
            "seed": 123, **overrides}


def a2_cfg(tmp_path, **overrides):
    overrides.setdefault("lora", lora_cfg())
    return toy_cfg(tmp_path, trainable_patterns=A2_PATTERNS, **overrides)


# --------------------------------------------------------------- zero delta

def test_initial_output_delta_is_exactly_zero():
    model = tiny_model(1)
    ids = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        before = model(ids).logits.clone()
    modules = apply_lora(model, LoRAConfig.from_dict(lora_cfg()))
    assert modules, "no modules wrapped"
    with torch.no_grad():
        after = model(ids).logits
    assert torch.equal(before, after)
    for m in modules.values():
        assert torch.count_nonzero(m.lora_B) == 0
        assert torch.count_nonzero(m.lora_A) > 0      # A is not degenerate


def test_merged_weight_at_init_is_the_base_weight():
    model = tiny_model(1)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    modules = apply_lora(model, LoRAConfig.from_dict(lora_cfg()))
    merged = merged_state_dict(model, modules)
    for name in modules:
        assert torch.equal(merged[f"{name}.weight"], base[f"{name}.weight"])


def test_wrapping_targets_exactly_qkvo_and_shares_the_base_weight():
    model = tiny_model(1)
    originals = {n: m.weight for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)}
    modules = apply_lora(model, LoRAConfig.from_dict(lora_cfg()))
    assert sorted(n.split(".")[-1] for n in modules) == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj"] * 2)      # 2 layers
    for name, m in modules.items():
        assert isinstance(m, LoRALinear) and isinstance(m, torch.nn.Linear)
        assert m.weight is originals[name]                 # shared, not copied
        assert m.weight.requires_grad is False
    assert "lm_head" not in modules and not any(".mlp." in n for n in modules)


def test_initialization_is_deterministic_and_seed_sensitive():
    a = apply_lora(tiny_model(1), LoRAConfig.from_dict(lora_cfg()))
    b = apply_lora(tiny_model(1), LoRAConfig.from_dict(lora_cfg()))
    c = apply_lora(tiny_model(1), LoRAConfig.from_dict(lora_cfg(seed=124)))
    for name in a:
        assert torch.equal(a[name].lora_A, b[name].lora_A)
        assert not torch.equal(a[name].lora_A, c[name].lora_A)
    # Applying LoRA must not perturb global RNG. It draws from its own seeded
    # generator, so an interposed apply_lora cannot shift the training stream.
    model = tiny_model(1)
    torch.manual_seed(7)
    first = torch.randn(4)
    torch.manual_seed(7)
    apply_lora(model, LoRAConfig.from_dict(lora_cfg()))
    assert torch.equal(first, torch.randn(4))


# ------------------------------------------------------------- merge fidelity

def _train_a_few_steps(tmp_path, steps=4):
    cfg = a2_cfg(tmp_path, schedule={"total_steps": steps})
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    for _ in range(steps):
        trainer.step_once()
    return trainer


def test_merged_and_unmerged_bf16_logits_match(tmp_path):
    trainer = _train_a_few_steps(tmp_path)
    for m in trainer.lora_modules.values():
        assert torch.count_nonzero(m.lora_B) > 0, "B never left zero; test is vacuous"

    ids = torch.randint(0, VOCAB, (2, 24))
    trainer.student.eval()
    with torch.no_grad():
        unmerged = trainer.student(ids).logits.to(torch.bfloat16).float()

    from transformers import Qwen3ForCausalLM

    merged_model = Qwen3ForCausalLM(trainer.student.config).float().eval()
    missing, unexpected = merged_model.load_state_dict(
        merged_state_dict(trainer.student, trainer.lora_modules), strict=False)
    assert not unexpected, unexpected
    assert all("lm_head" in k for k in missing), missing   # tied head only
    with torch.no_grad():
        merged = merged_model(ids).logits.to(torch.bfloat16).float()

    scale = unmerged.abs().max().clamp_min(1e-6)
    assert (merged - unmerged).abs().max() / scale < 5e-3
    assert torch.equal(merged.argmax(-1), unmerged.argmax(-1))


def test_saved_model_dir_is_a_plain_checkpoint_with_no_lora_keys(tmp_path):
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    trainer = _train_a_few_steps(tmp_path)
    ckpt = trainer.save_checkpoint()

    tensors = load_file(str(ckpt / "model" / "model.safetensors"))
    assert not [k for k in tensors if "lora" in k.lower()], "LoRA leaked into model/"

    # The normal evaluation path: no adapter code involved, nothing missing.
    reloaded = AutoModelForCausalLM.from_pretrained(ckpt / "model",
                                                    dtype=torch.float32).eval()
    assert not any(isinstance(m, LoRALinear) for m in reloaded.modules())
    ids = torch.randint(0, VOCAB, (1, 20))
    trainer.student.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(ids).logits,
                              trainer.student(ids).logits, atol=1e-5)

    meta = json.loads((ckpt / "checkpoint_meta.json").read_text())
    assert meta["model_dir_is_merged"] is True
    assert meta["lora_config"]["rank"] == 4 and meta["lora_config"]["alpha"] == 8
    assert meta["consumed_blocks"] == meta["step"] * 2
    assert meta["n_lora_modules"] == len(trainer.lora_modules)


def test_merged_checkpoint_carries_the_delta_not_the_base(tmp_path):
    """A merge that silently dropped the adapter would still load cleanly."""
    trainer = _train_a_few_steps(tmp_path)
    init = tiny_model(1)
    base = dict(init.named_parameters())
    merged = merged_state_dict(trainer.student, trainer.lora_modules)
    moved = [n for n in trainer.lora_modules
             if not torch.equal(merged[f"{n}.weight"], base[f"{n}.weight"])]
    assert len(moved) == len(trainer.lora_modules), "attention weights did not move"


# --------------------------------------------------------------- freeze policy

def test_trainable_counts_match_the_intended_policy(tmp_path):
    trainer = _train_a_few_steps(tmp_path, steps=1)
    rep = trainer.freeze_report
    model = trainer.student

    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # Attention projections and embeddings are frozen; adapters and FFN/norms train.
    assert any("self_attn.q_proj.weight" in n for n in frozen)
    assert all(".self_attn.q_proj.weight" not in n for n in trainable)
    assert "model.embed_tokens.weight" in frozen
    assert any(".mlp.gate_proj.weight" in n for n in trainable)
    assert any("self_attn.q_norm" in n for n in trainable)
    assert any(n.endswith(".lora_A") for n in trainable)

    assert rep["full_rank_trainable_params"] + rep["lora_trainable_params"] == \
        rep["trainable_params"]
    expected_lora = sum(m.lora_A.numel() + m.lora_B.numel()
                        for m in trainer.lora_modules.values())
    assert rep["lora_trainable_params"] == expected_lora
    # `total_params` counts the deployable (merged) model, so adapters are out.
    assert rep["total_params"] == sum(
        p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    assert rep["full_rank_trainable_params"] < rep["total_params"]


def test_frozen_attention_weights_do_not_move(tmp_path):
    """A1's guarantee, and A2's base-weight guarantee, in one check."""
    for use_lora in (False, True):
        cfg = (a2_cfg(tmp_path / f"l{use_lora}", schedule={"total_steps": 3})
               if use_lora else
               toy_cfg(tmp_path / f"l{use_lora}", trainable_patterns=A2_PATTERNS,
                       schedule={"total_steps": 3}))
        model = tiny_model(1)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        trainer = Trainer(cfg, model, toy_blocks(), toy_blocks(n=2), device="cpu")
        for _ in range(3):
            trainer.step_once()
        for name, param in trainer.student.named_parameters():
            if ".self_attn." in name and "_norm" not in name and "lora_" not in name:
                assert torch.equal(param, before[name]), f"{name} moved (lora={use_lora})"
            if ".mlp." in name:
                assert not torch.equal(param, before[name]), name


def test_lora_shares_one_optimizer_group_with_full_rank_parameters(tmp_path):
    """A2 tests low-rank parameterization, not adapter hyperparameters.

    LoRA tensors sit in the same single AdamW group at the same learning rate,
    schedule and weight decay as the FFN and norms, exactly as P1 optimized its
    trainable set. A second group would be a second variable.
    """
    cfg = a2_cfg(tmp_path, schedule={"total_steps": 2})
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    assert len(trainer.opt.param_groups) == 1
    group = trainer.opt.param_groups[0]
    assert group["weight_decay"] == cfg["optim"]["weight_decay"]
    assert group["lr"] == cfg["optim"]["lr"]

    lora_params = {id(p) for m in trainer.lora_modules.values()
                   for p in (m.lora_A, m.lora_B)}
    in_group = {id(p) for p in group["params"]}
    assert lora_params <= in_group
    assert len(in_group) == len(trainer.params)

    m = trainer.step_once()
    assert "lora_lr" not in m
    assert trainer.opt.param_groups[0]["lr"] == m["lr"]


def test_separate_lora_optimizer_settings_are_refused(tmp_path):
    for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
        cfg = a2_cfg(tmp_path)
        cfg["optim"] = {**cfg["optim"], field: 1e-4}
        with pytest.raises(ValueError, match=field):
            validate_train_config(cfg)


# -------------------------------------------------------------------- resume

def test_lora_resume_is_bitwise_exact(tmp_path):
    from transformers import AutoModelForCausalLM

    blocks, val = toy_blocks(), toy_blocks(n=2)
    cfg_a = a2_cfg(tmp_path / "a", schedule={"total_steps": 6})
    trainer_a = Trainer(cfg_a, tiny_model(1), blocks, val, device="cpu")
    for _ in range(6):
        trainer_a.step_once()

    cfg_b = a2_cfg(tmp_path / "b", schedule={"total_steps": 6})
    trainer_b = Trainer(cfg_b, tiny_model(1), blocks, val, device="cpu")
    for _ in range(3):
        trainer_b.step_once()
    ckpt = trainer_b.save_checkpoint()
    lora_b = {n: (m.lora_A.detach().clone(), m.lora_B.detach().clone())
              for n, m in trainer_b.lora_modules.items()}
    base_b = {n: m.weight.detach().clone() for n, m in trainer_b.lora_modules.items()}
    del trainer_b

    # `model/` is merged, so this student starts with the WRONG attention
    # weights; restore must put the frozen base back exactly.
    student = AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32)
    trainer_c = Trainer(cfg_b, student, blocks, val, device="cpu")
    trainer_c.restore(ckpt)
    assert trainer_c.step == 3 and trainer_c.consumed_blocks() == 6
    for name, m in trainer_c.lora_modules.items():
        assert torch.equal(m.lora_A, lora_b[name][0])
        assert torch.equal(m.lora_B, lora_b[name][1])
        assert torch.equal(m.weight, base_b[name])

    for _ in range(3):
        trainer_c.step_once()

    params_a = dict(trainer_a.student.named_parameters())
    for name, param in trainer_c.student.named_parameters():
        assert torch.equal(param, params_a[name]), name
    state_a = trainer_a.opt.state_dict()["state"]
    state_c = trainer_c.opt.state_dict()["state"]
    assert state_a.keys() == state_c.keys()
    for key in state_a:
        assert torch.equal(state_a[key]["exp_avg"], state_c[key]["exp_avg"])
        assert torch.equal(state_a[key]["exp_avg_sq"], state_c[key]["exp_avg_sq"])


def test_resume_consumes_the_same_next_batch(tmp_path):
    """The block stream position, not just the weights, must survive a restart."""
    from aadistill.training.train import stream_block_indices

    cfg = a2_cfg(tmp_path / "a", schedule={"total_steps": 6})
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    for _ in range(3):
        trainer.step_once()
    ckpt = trainer.save_checkpoint()

    from transformers import AutoModelForCausalLM

    student = AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32)
    resumed = Trainer(cfg, student, toy_blocks(), toy_blocks(n=2), device="cpu")
    resumed.restore(ckpt)
    n_blocks = trainer.train_ids.shape[0]
    assert (stream_block_indices(n_blocks, cfg["seed"], resumed.consumed_blocks(), 2)
            == stream_block_indices(n_blocks, cfg["seed"], 6, 2))


def test_resume_refuses_a_different_adapter(tmp_path):
    from transformers import AutoModelForCausalLM

    cfg = a2_cfg(tmp_path / "a", schedule={"total_steps": 2})
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    trainer.step_once()
    ckpt = trainer.save_checkpoint()

    other = a2_cfg(tmp_path / "a", schedule={"total_steps": 2}, lora=lora_cfg(rank=2))
    student = AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32)
    t2 = Trainer(other, student, toy_blocks(), toy_blocks(n=2), device="cpu")
    with pytest.raises(ValueError, match="different config"):
        t2.restore(ckpt)

    # A no-LoRA trainer must not accept a LoRA checkpoint's state either.
    plain = toy_cfg(tmp_path / "a", trainable_patterns=A2_PATTERNS,
                    schedule={"total_steps": 2})
    t3 = Trainer(plain, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    with pytest.raises(ValueError):
        t3.restore(ckpt)


def test_lora_state_roundtrip_is_exact_and_checked(tmp_path):
    model = tiny_model(1)
    modules = apply_lora(model, LoRAConfig.from_dict(lora_cfg()))
    for m in modules.values():
        with torch.no_grad():
            m.lora_B.normal_()
    saved = {k: v.clone() for k, v in lora_and_base_tensors(modules).items()}
    for m in modules.values():
        with torch.no_grad():
            m.lora_A.zero_(), m.lora_B.zero_(), m.weight.zero_()
    load_lora_and_base_(modules, saved)
    for key, tensor in lora_and_base_tensors(modules).items():
        assert torch.equal(tensor, saved[key]), key

    with pytest.raises(ValueError, match="lora state mismatch"):
        load_lora_and_base_(modules, {k: v for k, v in list(saved.items())[:-1]})


# ------------------------------------------------------------------ validation

def test_config_validation_rejects_bad_lora_specs(tmp_path):
    base = a2_cfg(tmp_path)
    for bad, match in (
        ({"rank": 0}, "rank"),
        ({"alpha": 0}, "alpha"),
        ({"dropout": 1.5}, "dropout"),
        ({"bias": "all"}, "bias"),
    ):
        cfg = {**base, "lora": lora_cfg(**bad)}
        with pytest.raises(ValueError, match=match):
            validate_train_config(cfg)

    cfg = {**base, "lora": {**lora_cfg(), "typo_field": 1}}
    with pytest.raises(ValueError, match="unknown lora config"):
        validate_train_config(cfg)


def test_trains_under_gradient_checkpointing_and_bf16_autocast(tmp_path):
    """The exact numerical/memory path the real arms run on the GPU.

    Gradient checkpointing recomputes each decoder block's forward, so an
    adapter that captured activations incorrectly would either lose its gradient
    or raise here rather than on a paid pod.
    """
    cfg = a2_cfg(tmp_path, schedule={"total_steps": 3},
                 gradient_checkpointing=True, autocast_bf16=True)
    model = tiny_model(1)
    trainer = Trainer(cfg, model, toy_blocks(), toy_blocks(n=2), device="cpu")
    before = {n: m.lora_B.detach().clone() for n, m in trainer.lora_modules.items()}

    metrics = [trainer.step_once() for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in metrics)
    for name, m in trainer.lora_modules.items():
        assert m.lora_A.grad is not None and torch.isfinite(m.lora_A.grad).all()
        assert m.lora_B.grad is not None and torch.isfinite(m.lora_B.grad).all()
        assert not torch.equal(m.lora_B, before[name]), f"{name} B never moved"
    assert model.config.use_cache is False


def test_manifest_fields_the_cli_writes_are_present(tmp_path):
    """`train_stage3.py` records the full-rank/LoRA split from this report."""
    trainer = _train_a_few_steps(tmp_path, steps=1)
    for key in ("trainable_params", "full_rank_trainable_params",
                "lora_trainable_params", "total_params", "lora_modules",
                "lora_config"):
        assert key in trainer.freeze_report, key
    assert trainer.lora_cfg.to_dict()["scaling"] == 2.0


def test_double_application_is_refused():
    model = tiny_model(1)
    cfg = LoRAConfig.from_dict(lora_cfg())
    apply_lora(model, cfg)
    with pytest.raises(ValueError, match="already has LoRA"):
        apply_lora(model, cfg)


def test_empty_target_match_fails_loudly():
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        apply_lora(tiny_model(1),
                   LoRAConfig.from_dict(lora_cfg(target_patterns=[r"\.nothing\."])))

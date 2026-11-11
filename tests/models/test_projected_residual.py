"""The projected-residual decoder: structure, exact folding, persistence, training.

Tiny CPU geometries throughout; the full-size count is checked on the meta
device, where no tensor is allocated.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from aadistill.models.cost_model import (
    forward_flops_per_token,
    kv_cache_bytes_per_token,
    parameter_counts,
    training_flops_per_token,
)
from aadistill.initialization.adapters.projected_residual import (
    GATES,
    MODEL_TYPE,
    ProjectedResidualConfig,
    ProjectedResidualForCausalLM,
    fold_to_qwen3,
    gate_statistics,
    projection_orthogonality,
    projections,
    register_auto_classes,
)
from aadistill.training.train import Trainer, select_trainable

TINY = dict(vocab_size=67, hidden_size=48, internal_size=16, intermediate_size=40,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            tie_word_embeddings=True, max_position_embeddings=256, rope_theta=10000.0)


def tiny(seed=0, **over):
    torch.manual_seed(seed)
    return ProjectedResidualForCausalLM(ProjectedResidualConfig(**{**TINY, **over})).double().eval()


def test_parameter_count_is_the_cost_models_count():
    m = tiny()
    geom = {k: v for k, v in TINY.items() if k in (
        "vocab_size", "hidden_size", "internal_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings")}
    assert sum(p.numel() for p in m.parameters()) == parameter_counts(geom)["total"]


def test_full_size_geometry_counts_on_meta():
    geom = dict(vocab_size=151936, hidden_size=2560, internal_size=1024, intermediate_size=3072,
                num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                tie_word_embeddings=True)
    with torch.device("meta"):
        m = ProjectedResidualForCausalLM(ProjectedResidualConfig(**geom))
    total = sum(p.numel() for p in m.parameters())
    frozen = m.model.embed_tokens.weight.numel()
    counts = parameter_counts(geom)
    assert total == counts["total"] == 902_911_488
    assert total - frozen == counts["non_embedding"] == 513_955_328
    # the standard geometries the repository has frozen
    std = {k: v for k, v in geom.items() if k != "internal_size"}
    assert parameter_counts({**std, "hidden_size": 1024})["total"] == 596_049_920
    assert parameter_counts(dict(std, hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
                                 num_attention_heads=32))["total"] == 4_022_468_096
    # the KV cache is the attention width's, the same as the 1024-wide student's
    assert kv_cache_bytes_per_token(geom) == kv_cache_bytes_per_token({**std, "hidden_size": 1024})


def test_flops_counter_matches_a_direct_count():
    geom = {k: v for k, v in TINY.items() if k not in ("max_position_embeddings", "rope_theta")}
    f = forward_flops_per_token(geom, context=10)
    d, r, q, kv, inter = 48, 16, 32, 16, 40
    per_layer = 8 * d * r + 2 * (r * q + 2 * r * kv + q * r) + 4 * q * 10 + 6 * r * inter
    assert f["per_layer"]["total"] == per_layer
    assert f["total"] == 3 * per_layer + 2 * d * 67
    t = training_flops_per_token(geom, 10, train_embedding=False)
    assert t["total"] == f["blocks"] * 4 + f["output_head"] * 2


def test_folded_qwen3_computes_the_same_function_for_a_non_orthogonal_projection():
    m = tiny(1)
    with torch.no_grad():
        for sites in projections(m):
            for p in sites.values():
                p.add_(0.3 * torch.randn_like(p))
    ids = torch.randint(0, 67, (2, 13))
    folded = fold_to_qwen3(m, dtype=torch.float64)
    assert folded.config.model_type == "qwen3"
    assert folded.model.layers[0].self_attn.q_proj.weight.shape == (32, 48)
    a, b = m(ids).logits, folded(ids).logits
    assert torch.allclose(a, b, atol=1e-10, rtol=0)
    assert a.abs().max() > 1e-3


def test_save_and_reload_through_the_auto_classes(tmp_path):
    register_auto_classes()
    m = tiny(2)
    m.save_pretrained(tmp_path)
    cfg = AutoConfig.from_pretrained(tmp_path)
    assert cfg.model_type == MODEL_TYPE and cfg.internal_size == 16
    again = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float64).eval()
    assert isinstance(again, ProjectedResidualForCausalLM)
    ids = torch.randint(0, 67, (1, 9))
    assert torch.equal(m(ids).logits, again(ids).logits)
    assert again.lm_head.weight.data_ptr() == again.model.embed_tokens.weight.data_ptr()


def test_cached_generation_equals_uncached():
    m = tiny(3).float()
    ids = torch.randint(0, 67, (1, 5))
    with_cache = m.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True)
    without = m.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False)
    assert torch.equal(with_cache, without)


def test_document_isolation_through_position_ids():
    m = tiny(4)
    a, b = torch.randint(0, 67, (1, 7)), torch.randint(0, 67, (1, 5))
    packed = torch.cat([a, b], dim=1)
    pos = torch.cat([torch.arange(7), torch.arange(5)]).unsqueeze(0)
    # transformers builds the block-diagonal mask from position ids only when no
    # cache and no 2-D mask are passed -- the trainer's configuration.
    out = m(packed, position_ids=pos, use_cache=False).logits
    assert torch.allclose(out[:, :7], m(a, use_cache=False).logits, atol=1e-10)
    assert torch.allclose(out[:, 7:], m(b, use_cache=False).logits, atol=1e-10)
    composed = m(packed, use_cache=False).logits      # cross_context: b conditioned on a
    assert not torch.allclose(composed[:, 7:], out[:, 7:], atol=1e-6)


def test_frozen_tied_embedding_and_trainable_projections(tmp_path):
    m = tiny(5).float()
    report = select_trainable(m, [r"^model\.layers\.", r"^model\.norm\."])
    assert "model.embed_tokens.weight" not in report["trainable_names"]
    assert report["trainable_params"] == parameter_counts(
        {k: v for k, v in TINY.items() if k not in ("max_position_embeddings", "rope_theta")})["non_embedding"]
    ids = torch.randint(0, 67, (2, 11))
    loss = m(ids).logits.float().logsumexp(-1).mean()
    loss.backward()
    assert m.model.embed_tokens.weight.grad is None
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for s in projections(m) for p in s.values())


def _cfg(tmp_path, total_steps):
    return {
        "stage": "stage3_recovery", "run_name": "projected", "student_path": "unused", "teacher": None,
        "data_dir": "unused", "groups": None, "block_len": 24, "dtype": "float32", "device": "cpu",
        "seed": 7, "trainable_patterns": [r"^model\.layers\.", r"^model\.norm\."],
        "loss": {"ce_weight": 1.0, "kd_weight": 0.0, "kd_temperature": 1.0, "kd_scope": "assistant"},
        "optim": {"lr": 5e-3, "weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": total_steps, "warmup_steps": 1, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 2, "micro_blocks": 1},
        "checkpoint": {"save_every": 2, "keep_last": 3},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(tmp_path / "run"),
    }


def _blocks():
    ids = (torch.arange(24).unsqueeze(0) * 5 + torch.arange(6).unsqueeze(1)) % 67
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 0] = False
    return ids.long(), mask


def test_trainer_resume_is_exact_for_the_projected_model(tmp_path):
    register_auto_classes()
    full = Trainer(_cfg(tmp_path / "a", 4), tiny(6).float(), _blocks())
    for _ in range(4):
        full.step_once()
    part = Trainer(_cfg(tmp_path / "b", 4), tiny(6).float(), _blocks())
    for _ in range(2):
        part.step_once()
    ckpt = part.save_checkpoint()
    resumed = Trainer(_cfg(tmp_path / "b", 4),
                      AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32), _blocks())
    resumed.restore(ckpt)
    assert resumed.step == 2
    for _ in range(2):
        resumed.step_once()
    ref = dict(full.student.named_parameters())
    for name, param in resumed.student.named_parameters():
        assert torch.equal(param, ref[name]), name
    # the frozen tied embedding never moved, and the projections did
    assert torch.equal(full.student.model.embed_tokens.weight, tiny(6).float().model.embed_tokens.weight)
    assert full.student.lm_head.weight.data_ptr() == full.student.model.embed_tokens.weight.data_ptr()
    assert projection_orthogonality(full.student)[0]["rel_fro"] > 1e-6


# --- four independent projections + the learnable SiLU residual gate -------------------------

FOUR = dict(projection_sharing="independent", projection_activation="learnable_silu_residual")


def _geom(**over):
    keys = ("vocab_size", "hidden_size", "internal_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings",
            "projection_sharing", "projection_activation")
    return {k: v for k, v in {**TINY, **over}.items() if k in keys}


def test_independent_gated_counts_and_full_size_geometry():
    m = tiny(0, **FOUR)
    assert sum(p.numel() for p in m.parameters()) == parameter_counts(_geom(**FOUR))["total"]
    layer = m.model.layers[0]
    assert layer.attn_down.weight.shape == (16, 48) and layer.attn_up.weight.shape == (48, 16)
    assert layer.ffn_down.weight.shape == (16, 48) and layer.ffn_up.weight.shape == (48, 16)
    assert all(getattr(layer, g).alpha.shape == (16,) for g in GATES)
    geom = dict(vocab_size=151936, hidden_size=2560, internal_size=1024, intermediate_size=3072,
                num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                tie_word_embeddings=True, **FOUR)
    with torch.device("meta"):
        big = ProjectedResidualForCausalLM(ProjectedResidualConfig(**geom))
    total = sum(p.numel() for p in big.parameters())
    gates = sum(p.numel() for n, p in big.named_parameters() if n.endswith("_gate.alpha"))
    frozen = big.model.embed_tokens.weight.numel()
    c = parameter_counts(geom)
    assert total == c["total"] == 1_123_227_136
    assert gates == 114_688 and total - gates == 1_123_112_448
    assert total - frozen == c["non_embedding"] == 734_270_976 and frozen == 388_956_160
    # FLOPs per token do not depend on how the four products' matrices are shared
    shared = {k: v for k, v in geom.items() if k not in FOUR}
    assert forward_flops_per_token(geom, 4096)["total"] == forward_flops_per_token(shared, 4096)["total"]


def test_gates_start_at_the_identity_and_still_receive_a_gradient():
    gated, linear = tiny(7, **FOUR), tiny(7, projection_sharing="independent")
    linear.load_state_dict({k: v for k, v in gated.state_dict().items() if "_gate." not in k})
    ids = torch.randint(0, 67, (2, 11))
    assert torch.equal(gated(ids).logits, linear(ids).logits)
    assert all(float(getattr(L, g).alpha.abs().max()) == 0.0 for L in gated.model.layers for g in GATES)
    gated.float().train()
    gated(ids).logits.float().logsumexp(-1).mean().backward()
    for L in gated.model.layers:
        for g in GATES:
            grad = getattr(L, g).alpha.grad
            assert grad is not None and float(grad.abs().sum()) > 0, g
    stats = gate_statistics(gated)
    assert len(stats) == 4 * TINY["num_hidden_layers"] and all(r["grad_norm"] > 0 for r in stats)


def test_a_trained_gate_makes_the_model_nonlinear_and_unfoldable():
    m = tiny(8, **FOUR)
    with torch.no_grad():
        for L in m.model.layers:
            for g in GATES:
                getattr(L, g).alpha.uniform_(-0.5, 0.5)
    ids = torch.randint(0, 67, (1, 9))
    linear = tiny(8, projection_sharing="independent")
    linear.load_state_dict({k: v for k, v in m.state_dict().items() if "_gate." not in k})
    assert not torch.allclose(m(ids).logits, linear(ids).logits, atol=1e-6)
    import pytest
    with pytest.raises(ValueError, match="does not fold"):
        fold_to_qwen3(m)


def test_independent_linear_model_folds_exactly():
    m = tiny(9, projection_sharing="independent")
    with torch.no_grad():
        for sites in projections(m):
            for p in sites.values():
                p.add_(0.3 * torch.randn_like(p))
    ids = torch.randint(0, 67, (2, 13))
    folded = fold_to_qwen3(m, dtype=torch.float64)
    assert torch.allclose(m(ids).logits, folded(ids).logits, atol=1e-10, rtol=0)


def test_independent_gated_save_reload_generate_and_isolate(tmp_path):
    register_auto_classes()
    m = tiny(10, **FOUR)
    with torch.no_grad():
        for L in m.model.layers:
            for g in GATES:
                getattr(L, g).alpha.uniform_(-0.3, 0.3)
    m.save_pretrained(tmp_path)
    cfg = AutoConfig.from_pretrained(tmp_path)
    assert (cfg.projection_sharing, cfg.projection_activation) == ("independent", "learnable_silu_residual")
    again = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float64).eval()
    ids = torch.randint(0, 67, (1, 9))
    assert torch.equal(m(ids).logits, again(ids).logits)
    f = again.float()
    assert torch.equal(f.generate(ids[:, :5], max_new_tokens=6, do_sample=False, use_cache=True),
                       f.generate(ids[:, :5], max_new_tokens=6, do_sample=False, use_cache=False))
    a, b = torch.randint(0, 67, (1, 7)), torch.randint(0, 67, (1, 5))
    pos = torch.cat([torch.arange(7), torch.arange(5)]).unsqueeze(0)
    out = m(torch.cat([a, b], 1), position_ids=pos, use_cache=False).logits
    assert torch.allclose(out[:, 7:], m(b, use_cache=False).logits, atol=1e-10)
    orth = projection_orthogonality(m)
    assert {r["site"] for r in orth} >= {"attn_down", "attn_up", "ffn_down", "ffn_up", "attn_up_vs_down"}


def test_trainer_resume_is_exact_for_the_gated_model(tmp_path):
    register_auto_classes()
    full = Trainer(_cfg(tmp_path / "a", 4), tiny(11, **FOUR).float(), _blocks())
    for _ in range(4):
        full.step_once()
    part = Trainer(_cfg(tmp_path / "b", 4), tiny(11, **FOUR).float(), _blocks())
    for _ in range(2):
        part.step_once()
    ckpt = part.save_checkpoint()
    resumed = Trainer(_cfg(tmp_path / "b", 4),
                      AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32), _blocks())
    resumed.restore(ckpt)
    for _ in range(2):
        resumed.step_once()
    ref = dict(full.student.named_parameters())
    for name, param in resumed.student.named_parameters():
        assert torch.equal(param, ref[name]), name
    moved = [r["mean_abs"] for r in gate_statistics(full.student)]
    assert max(moved) > 0          # the gates trained away from zero

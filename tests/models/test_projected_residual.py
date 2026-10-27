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
    MODEL_TYPE,
    ProjectedResidualConfig,
    ProjectedResidualForCausalLM,
    fold_to_qwen3,
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
        for p in projections(m):
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
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in projections(m))


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

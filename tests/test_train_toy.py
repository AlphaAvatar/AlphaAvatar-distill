"""Stage 3 trainer tests on tiny models (CPU, fast).

The load-bearing tests are loss correctness against hand computation and
bitwise resume equivalence: an interrupted run restored from its checkpoint
must produce exactly the weights and optimizer state of an uninterrupted
run. Everything else checks freeze policy, deterministic block streaming,
the LR schedule, and the run loop's logging/checkpoint side effects.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.train import (
    Trainer,
    epoch_permutation,
    kd_forward_kl,
    lr_factor,
    masked_ce,
    prediction_mask,
    select_trainable,
    stream_block_indices,
    validate_train_config,
)

VOCAB = 64


def tiny_model(seed: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).float()


def toy_blocks(n: int = 6, length: int = 32, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    # Predictable content (shifted arithmetic sequences) so a few optimizer
    # steps can visibly reduce the loss.
    ids = (torch.arange(length).unsqueeze(0) * 3 + torch.arange(n).unsqueeze(1)) % VOCAB
    mask = torch.randint(0, 2, (n, length), generator=g).bool()
    mask[:, 0] = False
    return ids.long(), mask


def toy_cfg(tmp_path, **overrides):
    cfg = {
        "stage": "stage3_recovery",
        "run_name": "toy",
        "student_path": "unused",
        "teacher": None,
        "data_dir": "unused",
        "groups": None,
        "block_len": 32,
        "dtype": "float32",
        "device": "cpu",
        "seed": 11,
        "trainable_patterns": "all",
        "loss": {"ce_weight": 1.0, "kd_weight": 0.0,
                 "kd_temperature": 1.0, "kd_scope": "assistant"},
        "optim": {"lr": 5e-3, "weight_decay": 0.0, "betas": [0.9, 0.95],
                  "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 6, "warmup_steps": 1, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 2, "micro_blocks": 1},
        "checkpoint": {"save_every": 0, "keep_last": 3},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(tmp_path / "run"),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    validate_train_config(cfg)
    return cfg


def test_masked_ce_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, False, True]])
    loss, count = masked_ce(logits, ids, mask)
    # Targets: token 2 predicted at position 0, token 4 predicted at position 2.
    logp = torch.log_softmax(logits[0], dim=-1)
    expected = -(logp[0, 2] + logp[2, 4])
    assert count == 2
    assert torch.allclose(loss, expected, atol=1e-6)

    zero_loss, zero_count = masked_ce(logits, ids, torch.zeros_like(mask))
    assert zero_count == 0 and float(zero_loss) == 0.0
    assert zero_loss.requires_grad is False  # plain constant, no targets


def test_kd_forward_kl_properties():
    torch.manual_seed(1)
    s = torch.randn(1, 5, 16, requires_grad=True)
    t = torch.randn(1, 5, 16)
    pos = torch.ones(1, 4, dtype=torch.bool)

    same, n = kd_forward_kl(s, s.detach().clone(), pos)
    assert n == 4 and abs(float(same.detach())) < 1e-5

    diff, _ = kd_forward_kl(s, t, pos)
    assert float(diff.detach()) > 0.0
    diff.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()

    # Chunking must not change the value.
    a, _ = kd_forward_kl(s.detach(), t, pos, chunk=1)
    b, _ = kd_forward_kl(s.detach(), t, pos, chunk=1024)
    assert torch.allclose(a, b, atol=1e-5)

    scoped = prediction_mask(torch.tensor([[False, True, False]]), "assistant")
    assert scoped.tolist() == [[True, False]]
    assert prediction_mask(torch.zeros(1, 3, dtype=torch.bool), "all").all()


def test_select_trainable_real_stage3_patterns():
    patterns = json.loads(
        (Path(__file__).resolve().parent.parent
         / "configs" / "stage3_s1_ffn_norm.json").read_text()
    )["trainable_patterns"]
    model = tiny_model(0)
    report = select_trainable(model, patterns)
    for name, param in model.named_parameters():
        expected = (
            ".mlp." in name
            or "input_layernorm" in name
            or "post_attention_layernorm" in name
            or name.startswith("model.norm.")
        )
        assert param.requires_grad == expected, name
    # Attention (incl. q_norm/k_norm) and the tied embedding stay frozen.
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any("self_attn.q_norm" in n for n in frozen)
    assert "model.embed_tokens.weight" in frozen
    assert 0 < report["trainable_params"] < report["total_params"]

    select_trainable(model, "all")
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="no parameters match"):
        select_trainable(model, ["does_not_exist"])


def test_lr_schedule():
    assert lr_factor(0, 100, 10, 0.1) == pytest.approx(0.1)
    assert lr_factor(9, 100, 10, 0.1) == pytest.approx(1.0)
    assert lr_factor(100, 100, 10, 0.1) == pytest.approx(0.1)
    factors = [lr_factor(s, 100, 10, 0.1) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))


def test_stream_block_indices_deterministic_resume():
    full = stream_block_indices(5, 7, 0, 12)
    assert sorted(full[:5]) == list(range(5))  # epoch 0 is a permutation
    assert sorted(full[5:10]) == list(range(5))  # epoch 1 too
    # Any restart position reproduces the same stream slice.
    assert stream_block_indices(5, 7, 3, 6) == full[3:9]
    assert stream_block_indices(5, 7, 10, 2) == full[10:12]
    assert torch.equal(epoch_permutation(5, 7, 2), epoch_permutation(5, 7, 2))


def test_training_reduces_loss(tmp_path):
    cfg = toy_cfg(tmp_path, schedule={"total_steps": 25})
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), device="cpu")
    before = trainer.evaluate()["val_ce"]
    metrics = [trainer.step_once() for _ in range(25)]
    after = trainer.evaluate()["val_ce"]
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in metrics)
    assert after < before * 0.9


def test_kd_training_step_and_teacher_requirement(tmp_path):
    cfg = toy_cfg(tmp_path, loss={"ce_weight": 0.5, "kd_weight": 1.0, "kd_scope": "all"})
    with pytest.raises(ValueError, match="requires a teacher"):
        Trainer(cfg, tiny_model(1), toy_blocks(), device="cpu")
    trainer = Trainer(
        cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2),
        teacher=tiny_model(2), device="cpu",
    )
    m = trainer.step_once()
    assert m["kd"] is not None and m["kd"] > 0
    assert m["ce"] is not None and m["kd_positions"] > m["ce_targets"] > 0
    ev = trainer.evaluate()
    assert "val_kd" in ev and ev["val_kd"] > 0


def test_run_writes_logs_and_checkpoint(tmp_path):
    cfg = toy_cfg(
        tmp_path,
        schedule={"total_steps": 3},
        checkpoint={"save_every": 2, "keep_last": 3},
        intervals={"log_every": 1, "eval_every": 2, "eval_blocks": 2},
    )
    trainer = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=3), device="cpu")
    summary = trainer.run()
    assert summary["steps"] == 3 and summary["final_eval"]["val_ce"] > 0

    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
    ]
    for expected in ("run_start", "eval_result", "train_step", "checkpoint_saved", "run_end"):
        assert expected in events, expected
    ckpts = tmp_path / "run" / "checkpoints"
    assert (ckpts / "latest.txt").read_text().strip() == "step_000003"
    assert (ckpts / "step_000003" / "model" / "config.json").is_file()
    assert (ckpts / "step_000002" / "trainer_state.pt").is_file()


def test_resume_is_bitwise_exact(tmp_path):
    from transformers import AutoModelForCausalLM

    blocks, val = toy_blocks(), toy_blocks(n=2)
    cfg_a = toy_cfg(tmp_path / "a", schedule={"total_steps": 6})
    trainer_a = Trainer(cfg_a, tiny_model(1), blocks, val, device="cpu")
    for _ in range(6):
        trainer_a.step_once()

    # Same run, interrupted after 3 steps and restored from the checkpoint.
    cfg_b = toy_cfg(tmp_path / "b", schedule={"total_steps": 6})
    trainer_b = Trainer(cfg_b, tiny_model(1), blocks, val, device="cpu")
    for _ in range(3):
        trainer_b.step_once()
    ckpt = trainer_b.save_checkpoint()
    del trainer_b

    student = AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32)
    trainer_c = Trainer(cfg_b, student, blocks, val, device="cpu")
    trainer_c.restore(ckpt)
    assert trainer_c.step == 3
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

    # A different config must be refused.
    cfg_other = toy_cfg(tmp_path / "b", optim={"lr": 1e-3})
    trainer_d = Trainer(cfg_other, tiny_model(1), blocks, val, device="cpu")
    with pytest.raises(ValueError, match="different config"):
        trainer_d.restore(ckpt)


def test_validate_config_rejects_bad_fields(tmp_path):
    cfg = toy_cfg(tmp_path)
    for corrupt in (
        {"dtype": "int8"},
        {"loss": {"kd_scope": "everything"}},
        {"batch": {"micro_blocks": 0}},
        {"trainable_patterns": "some"},
    ):
        bad = toy_cfg(tmp_path)
        key, value = next(iter(corrupt.items()))
        if isinstance(value, dict):
            bad[key] = {**bad[key], **value}
        else:
            bad[key] = value
        with pytest.raises(ValueError):
            validate_train_config(bad)

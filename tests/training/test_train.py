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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.training.train import (
    Trainer,
    build_blocks,
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
        (Path(__file__).resolve().parents[2]
         / "configs" / "stage3" / "recovery.json").read_text()
    )["trainable_patterns"]
    model = tiny_model(0)
    report = select_trainable(model, patterns)
    # configs/stage3/recovery.json uses the attention-unfrozen freeze set adopted
    # by the 2026-07-27 start-point ablation: attention (incl. q_norm/k_norm),
    # FFN and every norm train; the tied embedding does not.
    for name, param in model.named_parameters():
        expected = (
            ".self_attn." in name
            or ".mlp." in name
            or "input_layernorm" in name
            or "post_attention_layernorm" in name
            or name.startswith("model.norm.")
        )
        assert param.requires_grad == expected, name
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("self_attn.q_norm" in n for n in trainable)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
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


def test_extra_val_sets_logged_separately(tmp_path):
    cfg = toy_cfg(
        tmp_path,
        extra_val={"val_v0": "unused_dir"},
        schedule={"total_steps": 2},
        intervals={"log_every": 1, "eval_every": 2, "eval_blocks": 2},
    )
    trainer = Trainer(
        cfg, tiny_model(1), toy_blocks(), toy_blocks(n=3),
        extra_val_blocks={"val_v0": toy_blocks(n=4, seed=9)}, device="cpu",
    )
    summary = trainer.run()
    assert summary["final_eval"]["val_ce"] > 0

    evals = [
        json.loads(line)
        for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "eval_result"
    ]
    by_set = {}
    for ev in evals:
        by_set.setdefault(ev["val_set"], []).append(ev)
    # step-0 and final evals for both the primary and the extra set
    assert len(by_set["val"]) == len(by_set["val_v0"]) >= 2
    assert all("val_ce" in ev for ev in by_set["val_v0"])
    # different data => different metrics
    assert by_set["val"][0]["val_ce"] != by_set["val_v0"][0]["val_ce"]

    run_start = next(
        json.loads(line)
        for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "run_start"
    )
    assert run_start["extra_val_blocks"] == {"val_v0": 4}


def test_validate_config_extra_val_forms(tmp_path):
    toy_cfg(tmp_path, extra_val={"val_v0": "data/stage2"})  # valid
    for bad_extra in ({"val": "x"}, {"v": 3}, ["data/stage2"]):
        with pytest.raises(ValueError):
            toy_cfg(tmp_path, extra_val=bad_extra)
    with pytest.raises(ValueError, match="no primary val"):
        Trainer(
            toy_cfg(tmp_path, extra_val={"val_v0": "unused"}),
            tiny_model(1), toy_blocks(), None,
            extra_val_blocks={"val_v0": toy_blocks(n=2)}, device="cpu",
        )


# ---------- packing knob (build_blocks) ----------


def test_validate_config_packing_field(tmp_path):
    """`packing` is optional (the logged runs omit it) and value-checked."""
    validate_train_config(toy_cfg(tmp_path))  # absent -> concat, still valid
    validate_train_config(toy_cfg(tmp_path, packing="concat"))
    validate_train_config(toy_cfg(tmp_path, packing="best_fit"))
    with pytest.raises(ValueError):
        validate_train_config(toy_cfg(tmp_path, packing="bin_packing"))


@pytest.fixture(scope="module")
def teacher_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            revision="768f209d9ea81521153ed38c47d515654e938aea",
            local_files_only=True,
        )
    except Exception:
        pytest.skip("teacher tokenizer not in local HF cache")


def _toy_split(tmp_path, lengths):
    """Write a one-group train split whose samples have varied token lengths."""
    split = tmp_path / "train"
    split.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"t-{i:06d}", "group": "instruction", "source": "test",
         "format": "text", "text": " ".join(["word"] * n)}
        for i, n in enumerate(lengths)
    ]
    (split / "instruction.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return tmp_path


def test_build_blocks_packing_modes_differ_and_report_stats(
    tmp_path, teacher_tokenizer
):
    data_dir = _toy_split(tmp_path / "data", [7, 61, 13, 40, 25, 55, 31, 19] * 6)
    block_len = 64

    ids_c, mask_c, groups_c, stats_c, content_c = build_blocks(
        teacher_tokenizer, data_dir, "train", block_len, packing="concat"
    )
    ids_b, mask_b, groups_b, stats_b, content_b = build_blocks(
        teacher_tokenizer, data_dir, "train", block_len, packing="best_fit", seed=7
    )

    assert ids_c.shape[1] == block_len and ids_b.shape[1] == block_len
    assert len(groups_c) == ids_c.shape[0] and len(groups_b) == ids_b.shape[0]
    # concat wastes nothing but tears samples; best_fit pads instead.
    assert "dropped_tail_tokens" in stats_c["instruction"]
    assert stats_b["instruction"]["efficiency"] <= 1.0
    assert stats_b["instruction"]["padding_tokens"] > 0
    # Padding is never supervised.
    assert int(mask_b.sum()) < mask_b.numel()
    assert int(mask_c.sum()) > 0
    # Best-fit needs at least as many blocks for the same tokens (it pads).
    assert ids_b.shape[0] >= ids_c.shape[0]


def test_build_blocks_rejects_unknown_packing(tmp_path, teacher_tokenizer):
    data_dir = _toy_split(tmp_path / "data", [10, 20, 30])
    with pytest.raises(ValueError):
        build_blocks(teacher_tokenizer, data_dir, "train", 64, packing="nope")


def test_concat_has_no_content_mask_but_best_fit_marks_padding(
    tmp_path, teacher_tokenizer
):
    """The content mask exists only where padding does."""
    data_dir = _toy_split(tmp_path / "data", [7, 61, 13, 40, 25, 55] * 5)
    _, _, _, _, content_c = build_blocks(
        teacher_tokenizer, data_dir, "train", 64, packing="concat"
    )
    ids_b, _, _, stats_b, content_b = build_blocks(
        teacher_tokenizer, data_dir, "train", 64, packing="best_fit", seed=3
    )
    assert content_c is None, "concat packing never pads"
    assert content_b is not None and content_b.shape == ids_b.shape
    padding = int((~content_b).sum())
    assert padding == stats_b["instruction"]["padding_tokens"]
    # Padding is a suffix of each block: content is monotone non-increasing.
    assert bool((content_b[:, :-1].int() >= content_b[:, 1:].int()).all())


def test_kd_scope_all_excludes_padding_when_content_mask_given():
    """`all` means every real position, not every slot in a padded block."""
    loss_mask = torch.zeros(2, 5, dtype=torch.bool)
    content = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    # No content mask (concat packing): every position counts, as the four
    # logged runs trained.
    assert int(prediction_mask(loss_mask, "all").sum()) == 2 * 4
    # With padding marked, the pad run is excluded from both the loss and the
    # normalizer.
    got = prediction_mask(loss_mask, "all", content)
    assert int(got.sum()) == 2 + 4
    assert got.shape == (2, 4)
    # `assistant` is unaffected by the content mask (padding is never
    # supervised, so loss_mask already excludes it).
    assert int(prediction_mask(loss_mask, "assistant", content).sum()) == 0


def test_trainer_accepts_two_and_three_tuple_blocks(tmp_path):
    """Back-compat: (ids, mask) still works; (ids, mask, content) is the new form."""
    from aadistill.training.train import _unpack_blocks

    ids = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.bool)
    content = torch.ones(2, 4, dtype=torch.bool)
    assert _unpack_blocks((ids, mask))[2] is None
    assert _unpack_blocks((ids, mask, content))[2] is content
    with pytest.raises(ValueError):
        _unpack_blocks((ids,))


# ---------- kd_scope all_no_think (the CE/KD protocol conflict fix) ----------


def test_think_span_mask_marks_open_through_close():
    from aadistill.training.train import think_span_mask

    OPEN, CLOSE = 90, 91
    ids = torch.tensor([[5, OPEN, 7, 8, CLOSE, 9, 10]])
    m = think_span_mask(ids, OPEN, CLOSE)
    assert m.tolist() == [[False, True, True, True, True, False, False]], (
        "span must cover <think> through </think> inclusive")


def test_think_span_mask_handles_several_packed_samples():
    """A packed block holds many samples; every span must be marked."""
    from aadistill.training.train import think_span_mask

    OPEN, CLOSE = 90, 91
    ids = torch.tensor([[OPEN, 1, CLOSE, 2, 3, OPEN, 4, CLOSE, 5]])
    m = think_span_mask(ids, OPEN, CLOSE)
    assert m.tolist() == [[True, True, True, False, False, True, True, True, False]]


def test_think_span_mask_without_any_span_is_all_false():
    from aadistill.training.train import think_span_mask

    ids = torch.tensor([[1, 2, 3, 4]])
    assert not think_span_mask(ids, 90, 91).any()


def test_all_no_think_is_all_minus_the_span():
    """The contested positions leave the KD target set; nothing else changes."""
    OPEN, CLOSE = 90, 91
    ids = torch.tensor([[5, OPEN, 7, CLOSE, 9, 10]])
    loss_mask = torch.ones(1, 6, dtype=torch.bool)
    every = prediction_mask(loss_mask, "all")
    without = prediction_mask(loss_mask, "all_no_think", None,
                              input_ids=ids, think_ids=(OPEN, CLOSE))
    assert int(every.sum()) == 5
    # positions 1..3 are the span; shifted by one, entries 0..2 drop out.
    assert without.tolist() == [[False, False, False, True, True]]
    assert int(without.sum()) == 2


def test_all_no_think_composes_with_the_padding_content_mask():
    OPEN, CLOSE = 90, 91
    ids = torch.tensor([[OPEN, 7, CLOSE, 9, 0, 0]])
    loss_mask = torch.ones(1, 6, dtype=torch.bool)
    content = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    got = prediction_mask(loss_mask, "all_no_think", content,
                          input_ids=ids, think_ids=(OPEN, CLOSE))
    # Neither padding nor the think span may be a KD target.
    assert got.tolist() == [[False, False, True, False, False]]


def test_all_no_think_fails_loudly_without_think_ids():
    """A silent fallback to 'all' would look like the treatment arm."""
    loss_mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="all_no_think"):
        prediction_mask(loss_mask, "all_no_think", None)
    with pytest.raises(ValueError, match="all_no_think"):
        prediction_mask(loss_mask, "all_no_think", None,
                        input_ids=torch.zeros(1, 4, dtype=torch.long))


def test_trainer_rejects_all_no_think_without_think_ids(tmp_path):
    cfg = toy_cfg(tmp_path, loss={"ce_weight": 0.5, "kd_weight": 1.0,
                                  "kd_scope": "all_no_think"})
    with pytest.raises(ValueError, match="think_ids"):
        Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2),
                teacher=tiny_model(2), device="cpu")


def test_trainer_trains_with_all_no_think(tmp_path):
    cfg = toy_cfg(tmp_path, loss={"ce_weight": 0.5, "kd_weight": 1.0,
                                  "kd_scope": "all_no_think"})
    ids, mask = toy_blocks()
    ids[:, 3] = 60   # <think>   (must be inside the toy VOCAB of 64)
    ids[:, 5] = 61   # </think>
    trainer = Trainer(cfg, tiny_model(1), (ids, mask), toy_blocks(n=2),
                      teacher=tiny_model(2), device="cpu", think_ids=(60, 61))
    m = trainer.step_once()
    assert m["kd"] is not None and torch.isfinite(torch.tensor(m["loss"]))
    # Fewer KD positions than plain "all" — the span was removed.
    every = toy_cfg(tmp_path / "b", loss={"ce_weight": 0.5, "kd_weight": 1.0,
                                          "kd_scope": "all"})
    t2 = Trainer(every, tiny_model(1), (ids, mask), toy_blocks(n=2),
                 teacher=tiny_model(2), device="cpu")
    assert m["kd_positions"] < t2.step_once()["kd_positions"]


def test_real_tokenizer_think_tags_are_single_tokens(teacher_tokenizer):
    """The scope depends on this; if it ever stops holding, fail visibly."""
    assert len(teacher_tokenizer.encode("<think>", add_special_tokens=False)) == 1
    assert len(teacher_tokenizer.encode("</think>", add_special_tokens=False)) == 1


# --- padding-suffix truncation -------------------------------------------
# Padding is a contiguous suffix and attention is causal, so the pad run cannot
# affect a real token. Truncating it before the forward is therefore a pure
# saving. These guard the three ways that could go wrong: the contiguity
# precondition, the numbers the run reports, and the accounting.

def _padded_blocks(n=2, length=32, real=(11, 20)):
    """Blocks whose real tokens form a prefix and whose tail is pad."""
    ids = (torch.arange(length).unsqueeze(0) * 3 + torch.arange(n).unsqueeze(1)) % VOCAB
    content = torch.zeros(n, length, dtype=torch.bool)
    mask = torch.zeros(n, length, dtype=torch.bool)
    for i, r in enumerate(real[:n]):
        content[i, :r] = True
        mask[i, r // 2:r] = True          # supervise the back half of the real span
    ids = torch.where(content, ids, torch.full_like(ids, 0))
    return ids.long(), mask, content


def test_nonpad_extent_is_the_last_real_position_plus_one():
    from aadistill.training.train import nonpad_extent
    _, _, content = _padded_blocks(real=(11, 20))
    assert nonpad_extent(content) == 20            # max over the microbatch
    assert nonpad_extent(content[:1]) == 11


def test_nonpad_extent_rejects_non_contiguous_padding():
    """A packer that interleaved padding would silently break the optimization."""
    from aadistill.training.train import nonpad_extent
    holey = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous suffix"):
        nonpad_extent(holey)


def test_truncation_leaves_losses_and_normalizers_unchanged(tmp_path):
    """Same CE, KD and normalizers; only the executed positions shrink."""
    ids, mask, content = _padded_blocks()
    blocks = (ids, mask, content)
    out = {}
    for flag in (False, True):
        cfg = toy_cfg(tmp_path, batch={"truncate_padding": flag},
                      loss={"ce_weight": 1.0, "kd_weight": 1.0,
                            "kd_temperature": 1.0, "kd_scope": "all"})
        torch.manual_seed(0)
        tr = Trainer(cfg, tiny_model(1), blocks, blocks,
                     teacher=tiny_model(2).eval(), device="cpu")
        out[flag] = tr.step_once()
    full, trunc = out[False], out[True]
    # The quantities that define the run are untouched.
    for key in ("ce_targets", "kd_positions", "logical_block_tokens",
                "executed_nonpad_tokens", "supervised_tokens"):
        assert full[key] == trunc[key], key
    # Mathematically the same computation; float32 reductions are reordered, so
    # this is a tolerance, not bitwise equality.
    assert full["ce"] == pytest.approx(trunc["ce"], abs=1e-6)
    assert full["kd"] == pytest.approx(trunc["kd"], abs=1e-6)
    assert full["loss"] == pytest.approx(trunc["loss"], abs=1e-6)
    # And the saving is real. At micro_blocks=1 each block is forwarded at its
    # own extent, so the cost is the sum of the real lengths (11 + 20), not the
    # batch maximum -- mixed-length blocks cost nothing extra.
    assert trunc["executed_positions"] < full["executed_positions"]
    assert full["executed_positions"] == 2 * 32
    assert trunc["executed_positions"] == 11 + 20


def test_truncation_uses_the_microbatch_maximum_when_blocks_share_a_forward(tmp_path):
    """At micro_blocks>1 a short block rides along to the longest row's extent."""
    ids, mask, content = _padded_blocks(real=(11, 20))
    blocks = (ids, mask, content)
    cfg = toy_cfg(tmp_path, batch={"blocks_per_step": 2, "micro_blocks": 2,
                                   "truncate_padding": True})
    torch.manual_seed(0)
    tr = Trainer(cfg, tiny_model(1), blocks, device="cpu")
    m = tr.step_once()
    assert m["executed_positions"] == 2 * 20        # both rows padded to 20
    assert m["executed_nonpad_tokens"] == 11 + 20   # real tokens, unchanged
    assert m["logical_block_tokens"] == 2 * 32


def test_truncation_is_a_noop_on_dense_blocks(tmp_path):
    """With nothing to drop the two paths must agree exactly, not approximately."""
    ids, mask, content = _padded_blocks(real=(32, 32))
    blocks = (ids, mask, content)
    res = {}
    for flag in (False, True):
        cfg = toy_cfg(tmp_path, batch={"truncate_padding": flag})
        torch.manual_seed(0)
        tr = Trainer(cfg, tiny_model(1), blocks, blocks, device="cpu")
        res[flag] = tr.step_once()
    assert res[False]["loss"] == res[True]["loss"]
    assert res[False]["executed_positions"] == res[True]["executed_positions"]


def test_truncation_defaults_off_so_logged_configs_keep_their_path(tmp_path):
    """P4: enabling it must be an explicit, hash-visible config choice."""
    cfg = toy_cfg(tmp_path)
    assert "truncate_padding" not in cfg["batch"]
    ids, mask, content = _padded_blocks()
    tr = Trainer(cfg, tiny_model(1), (ids, mask, content), device="cpu")
    assert tr.truncate_padding is False
    with pytest.raises(ValueError, match="truncate_padding"):
        toy_cfg(tmp_path, batch={"truncate_padding": "yes"})

"""E7's dual-stream trainer: the extra stream must be additive and nothing else.

The whole experiment rests on one claim — that arm B is `e1_r1600k_*_pca` plus a
KD-only stream, and not a different run that happens to resemble it. That claim
is mechanical, so it is tested mechanically: same blocks, same order, same
learning rates, same optimizer steps, same normalizers, with the extra stream
present and absent.

The failure this guards against is not hypothetical. Merging the extra text into
the rollout pack — the obvious implementation — moves every block boundary and
every example's position against the LR schedule, and the resulting arm cannot be
compared to the retained baseline at all.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

from aadistill.data.extra_stream import (
    kd_positions, load_extra_stream, pack_dense, stream_budget, write_stream,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.training.train import (
    Trainer, gradient_share, lr_factor, stream_block_indices,
    validate_train_config,
)

from test_train import VOCAB, tiny_model, toy_blocks, toy_cfg


def dense_stream(tmp_path, name="extra", n=8, length=16, seed=5):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n, length), generator=g).numpy().astype("int32")
    import numpy as np
    write_stream(tmp_path / name, ids, np.ones_like(ids, dtype=bool), [],
                 {"stream": name, "kind": "general_text_kd", "purpose": "train",
                  "assistant_ce_positions": 0})
    return tmp_path / name


def extra_cfg(data_dir, **over):
    cfg = {"data_dir": str(data_dir), "kind": "general_text_kd",
           "lambda_extra": 0.25, "blocks_per_step": 1, "micro_blocks": 1,
           "every_n_steps": 1, "seed": 4242}
    cfg.update(over)
    return cfg


def make(tmp_path, *, with_extra, steps=6, **over):
    ids, mask = toy_blocks(n=6, length=16)
    stream = dense_stream(tmp_path) if with_extra else None
    cfg = toy_cfg(
        tmp_path,
        block_len=16,
        loss={"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0,
              "kd_scope": "all"},
        schedule={"total_steps": steps, "warmup_steps": 2, "min_lr_frac": 0.1},
        **over)
    if with_extra:
        cfg["extra_stream"] = extra_cfg(stream)
        validate_train_config(cfg)
    blocks = None
    if with_extra:
        e_ids, e_content, _ = load_extra_stream(stream)
        blocks = (e_ids, e_content)
    trainer = Trainer(cfg, tiny_model(0), (ids, mask), (ids, mask),
                      teacher=tiny_model(1), device="cpu",
                      out_dir=tmp_path / ("with" if with_extra else "without"),
                      extra_stream_blocks=blocks)
    return trainer, cfg


# --------------------------------------------------------------------------
# 1-2. the rollout trajectory is untouched
# --------------------------------------------------------------------------

def test_rollout_block_order_is_unchanged_by_the_extra_stream(tmp_path):
    with_extra, cfg_w = make(tmp_path / "a", with_extra=True)
    without, cfg_o = make(tmp_path / "b", with_extra=False)
    bps = cfg_w["batch"]["blocks_per_step"]

    for step in range(cfg_w["schedule"]["total_steps"]):
        a = stream_block_indices(with_extra.train_ids.shape[0], cfg_w["seed"],
                                 step * bps, bps)
        b = stream_block_indices(without.train_ids.shape[0], cfg_o["seed"],
                                 step * bps, bps)
        assert a == b, f"step {step}: rollout blocks differ"


def test_rollout_lr_positions_are_unchanged(tmp_path):
    _, cfg_w = make(tmp_path / "a", with_extra=True)
    _, cfg_o = make(tmp_path / "b", with_extra=False)
    sched_w, sched_o = cfg_w["schedule"], cfg_o["schedule"]
    assert sched_w == sched_o
    base = cfg_w["optim"]["lr"]
    for step in range(sched_w["total_steps"]):
        assert (base * lr_factor(step, sched_w["total_steps"],
                                 sched_w["warmup_steps"], sched_w["min_lr_frac"])
                == base * lr_factor(step, sched_o["total_steps"],
                                    sched_o["warmup_steps"],
                                    sched_o["min_lr_frac"]))


def test_the_optimizer_step_count_is_unchanged(tmp_path):
    with_extra, cfg = make(tmp_path / "a", with_extra=True, steps=4)
    without, _ = make(tmp_path / "b", with_extra=False, steps=4)
    for _ in range(4):
        with_extra.step_once()
        without.step_once()
    assert with_extra.step == without.step == 4


def test_the_rollout_loss_terms_are_identical_at_step_zero(tmp_path):
    """The extra stream adds a term; it must not perturb the existing ones."""
    a, _ = make(tmp_path / "a", with_extra=True)
    b, _ = make(tmp_path / "b", with_extra=False)
    ma, mb = a.step_once(), b.step_once()
    for key in ("ce", "kd", "ce_targets", "kd_positions", "lr",
                "logical_block_tokens", "supervised_tokens"):
        assert ma[key] == pytest.approx(mb[key]), f"{key} moved"
    assert ma["loss"] == pytest.approx(mb["loss"]), (
        "the reported loss is the rollout objective and must stay comparable "
        "with every single-stream run")


# --------------------------------------------------------------------------
# 3-4. the extra stream is KD-only and dense
# --------------------------------------------------------------------------

def test_the_extra_stream_contributes_zero_ce_positions(tmp_path):
    trainer, _ = make(tmp_path, with_extra=True)
    m = trainer.step_once()
    assert m["extra_kd_positions"] > 0
    # `ce_targets` counts rollout CE only; there is no CE path for the stream.
    assert m["ce_targets"] == m["supervised_tokens"]
    without, _ = make(tmp_path / "b", with_extra=False)
    assert m["ce_targets"] == without.step_once()["ce_targets"]


def test_a_config_with_extra_ce_weight_is_refused(tmp_path):
    stream = dense_stream(tmp_path)
    cfg = toy_cfg(tmp_path, block_len=16)
    cfg["extra_stream"] = extra_cfg(stream, ce_weight=1.0)
    with pytest.raises(ValueError, match="carries no CE by construction"):
        validate_train_config(cfg)


def test_a_padded_extra_stream_is_refused(tmp_path):
    """Padding would break the exact `n_blocks * (block_len - 1)` budget."""
    import numpy as np
    ids = np.zeros((4, 8), dtype="int32")
    content = np.ones_like(ids, dtype=bool)
    content[:, -2:] = False
    write_stream(tmp_path / "padded", ids, content, [],
                 {"stream": "padded", "kind": "general_text_kd"})
    with pytest.raises(ValueError, match="contains padding"):
        load_extra_stream(tmp_path / "padded")


def test_dense_packing_drops_the_partial_tail(tmp_path):
    docs = [("a", "h1", 0, list(range(10))), ("b", "h2", 1, list(range(7)))]
    ids, content, rows = pack_dense(docs, block_len=8, separator_id=99)
    assert ids.shape == (2, 8)          # 10+1+7+1 = 19 tokens -> 2 full blocks
    assert content.all()
    assert kd_positions(2, 8) == 14
    assert [r.source_index for r in rows] == [0, 1]
    assert 99 in ids.reshape(-1).tolist(), "the boundary must be an explicit token"


# --------------------------------------------------------------------------
# 5-6. B and C are matched
# --------------------------------------------------------------------------

def test_two_streams_of_equal_shape_have_identical_budgets():
    b = stream_budget(1761, 1024, total_steps=1761, blocks_per_step=1,
                      every_n_steps=1)
    c = stream_budget(1761, 1024, total_steps=1761, blocks_per_step=1,
                      every_n_steps=1)
    assert b == c
    assert b["kd_positions"] == 1761 * 1023 == 1_801_503
    assert b["forward_tokens"] == 1761 * 1024
    assert b["exposures"] == 1.0


def test_the_real_e7_arm_configs_are_budget_matched():
    """The shipped configs, not a toy: B and C must differ only in source."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    index = json.loads((root / "configs/stage3/e7/e7_configs.json").read_text())
    budgets, extras = {}, {}
    for run, meta in index.items():
        cfg = json.loads((root / meta["path"]).read_text())
        extras[run] = cfg["extra_stream"]
        budgets[run] = json.dumps(meta["extra_budget"], sort_keys=True)
    assert len(set(budgets.values())) == 1, budgets
    for seed in ("sa", "sb"):
        b = dict(extras[f"e7_fineweb_r1600k_{seed}"])
        c = dict(extras[f"e7_control_r1600k_{seed}"])
        differing = {k for k in set(b) | set(c) if b.get(k) != c.get(k)}
        assert differing == {"data_dir", "kind"}, differing


def test_the_cadence_is_a_pure_function_of_the_step(tmp_path):
    trainer, _ = make(tmp_path, with_extra=True, steps=8)
    trainer.extra_cfg = {**trainer.extra_cfg, "every_n_steps": 3}
    assert [trainer.extra_active(s) for s in range(7)] == [
        True, False, False, True, False, False, True]
    # The cursor advances only on active steps, so the stream is consumed as a
    # contiguous prefix rather than skipped through.
    assert [trainer.extra_cursor(s) for s in range(7)] == [0, 0, 0, 1, 1, 1, 2]


# --------------------------------------------------------------------------
# 7. independent normalizers
# --------------------------------------------------------------------------

def test_the_two_streams_have_independent_normalizers(tmp_path):
    """A pooled mean would make each stream's weight depend on the other's
    packing efficiency — and the rollout pack is 72% padding at this rung."""
    trainer, cfg = make(tmp_path, with_extra=True)
    m = trainer.step_once()
    n_extra_blocks = cfg["extra_stream"]["blocks_per_step"]
    block_len = trainer.extra_ids.shape[1]
    assert m["extra_kd_positions"] == n_extra_blocks * (block_len - 1)
    # The rollout KD normalizer is its own mask's count, untouched by the extra
    # stream's presence.
    without, _ = make(tmp_path / "b", with_extra=False)
    assert m["kd_positions"] == without.step_once()["kd_positions"]
    # And the reported extra mean is a mean over the extra positions alone.
    assert m["extra_weighted"] == pytest.approx(
        cfg["extra_stream"]["lambda_extra"] * m["extra_kd"], abs=1e-6)


def test_lambda_extra_scales_only_the_extra_gradient(tmp_path):
    small, _ = make(tmp_path / "a", with_extra=True)
    big, _ = make(tmp_path / "b", with_extra=True)
    big.extra_cfg = {**big.extra_cfg, "lambda_extra": 1.0}
    gs_small = gradient_share(small, n_steps=2)
    gs_big = gradient_share(big, n_steps=2)
    assert gs_big["ratio_mean"] > gs_small["ratio_mean"]
    # 4x the weight, 4x the extra-gradient norm; the rollout norm is unmoved.
    for a, b in zip(gs_small["per_step"], gs_big["per_step"]):
        assert b["rollout_grad_norm"] == pytest.approx(a["rollout_grad_norm"],
                                                       rel=1e-6)
        assert b["extra_grad_norm"] == pytest.approx(4 * a["extra_grad_norm"],
                                                     rel=1e-4)


def test_gradient_share_does_not_advance_the_run(tmp_path):
    """The preflight diagnostic must leave the run bit-identical."""
    trainer, _ = make(tmp_path, with_extra=True)
    before = [p.detach().clone() for p in trainer.params]
    gradient_share(trainer, n_steps=2)
    assert trainer.step == 0
    for a, b in zip(before, trainer.params):
        assert torch.equal(a, b), "the diagnostic took an optimizer step"


# --------------------------------------------------------------------------
# 8. resume
# --------------------------------------------------------------------------

def test_resume_reproduces_both_stream_cursors(tmp_path):
    trainer, cfg = make(tmp_path / "a", with_extra=True, steps=6)
    bps = cfg["batch"]["blocks_per_step"]
    seen_rollout, seen_extra = [], []
    for _ in range(3):
        step = trainer.step
        seen_rollout.append(stream_block_indices(
            trainer.train_ids.shape[0], cfg["seed"], step * bps, bps))
        seen_extra.append(stream_block_indices(
            trainer.extra_ids.shape[0], cfg["extra_stream"]["seed"],
            trainer.extra_cursor(step), cfg["extra_stream"]["blocks_per_step"]))
        trainer.step_once()
    ckpt = trainer.save_checkpoint()

    fresh, _ = make(tmp_path / "b", with_extra=True, steps=6)
    fresh.cfg = cfg
    fresh.config_sha = trainer.config_sha
    fresh.restore(ckpt)
    assert fresh.step == 3

    # Both cursors are pure functions of the restored step counter, so the
    # resumed run re-derives the same first three steps it already did and
    # continues from exactly where it stopped. No dataloader state is saved,
    # and none is needed.
    for k in range(3):
        assert stream_block_indices(fresh.train_ids.shape[0], cfg["seed"],
                                    k * bps, bps) == seen_rollout[k]
        assert stream_block_indices(
            fresh.extra_ids.shape[0], cfg["extra_stream"]["seed"],
            fresh.extra_cursor(k),
            cfg["extra_stream"]["blocks_per_step"]) == seen_extra[k]
    # And step 3 — the one about to run — advances both, together.
    assert fresh.extra_cursor(3) == 3
    assert fresh.extra_cursor(3) != fresh.extra_cursor(2)


# --------------------------------------------------------------------------
# the budget is knowable before the run, and matches what the run consumes
# --------------------------------------------------------------------------

def test_the_planned_budget_equals_what_the_run_consumes(tmp_path):
    trainer, cfg = make(tmp_path, with_extra=True, steps=5)
    planned = trainer.planned_extra_kd_positions()
    for _ in range(5):
        trainer.step_once()
    assert trainer._extra_kd_positions == planned, (
        "a budget that is only knowable afterwards is not preregistered")


def test_declaring_a_stream_without_passing_it_is_refused(tmp_path):
    stream = dense_stream(tmp_path)
    ids, mask = toy_blocks(n=6, length=16)
    cfg = toy_cfg(tmp_path, block_len=16)
    cfg["extra_stream"] = extra_cfg(stream)
    with pytest.raises(ValueError, match="no blocks were passed"):
        Trainer(cfg, tiny_model(0), (ids, mask), teacher=tiny_model(1),
                device="cpu", out_dir=tmp_path / "run")


def test_passing_a_stream_without_declaring_it_is_refused(tmp_path):
    """Otherwise the config hash would not record the treatment."""
    stream = dense_stream(tmp_path)
    e_ids, e_content, _ = load_extra_stream(stream)
    ids, mask = toy_blocks(n=6, length=16)
    cfg = toy_cfg(tmp_path, block_len=16)
    with pytest.raises(ValueError, match="config declares none"):
        Trainer(cfg, tiny_model(0), (ids, mask), teacher=tiny_model(1),
                device="cpu", out_dir=tmp_path / "run",
                extra_stream_blocks=(e_ids, e_content))


def test_an_extra_stream_without_a_teacher_is_refused(tmp_path):
    stream = dense_stream(tmp_path)
    e_ids, e_content, _ = load_extra_stream(stream)
    ids, mask = toy_blocks(n=6, length=16)
    cfg = toy_cfg(tmp_path, block_len=16,
                  loss={"ce_weight": 1.0, "kd_weight": 0.0,
                        "kd_temperature": 1.0, "kd_scope": "assistant"})
    cfg["extra_stream"] = extra_cfg(stream)
    with pytest.raises(ValueError, match="KD-only and needs a teacher"):
        Trainer(cfg, tiny_model(0), (ids, mask), device="cpu",
                out_dir=tmp_path / "run", extra_stream_blocks=(e_ids, e_content))

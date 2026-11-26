"""Per-logical-layer hidden-state distillation (`aadistill.training.hidden_state`) through the real trainer.

* the error and its hand-written backward equal autograd of the plain formula (float64);
* the rows are the KD prediction rows (``b * T + t``), the teacher's states are taken once per layer without a graph,
  the student's are handed over once per logical layer by its own loop -- never again by a checkpoint recomputation;
* the trainer adds ``weight x sum / (pairs x KD rows)`` to the step's objective, reports it beside CE and KD, and its
  gradient reaches every block; the teacher gets none;
* gradient checkpointing on or off gives the same hidden loss and gradients;
* without ``loss.hidden_state`` nothing of this runs (the CE / KD and the gradients are those of the unchanged path);
* configs that cannot be computed are refused at construction;
* a micro-batch's teacher targets do not outlive its backward (training) or its loss (validation): none is alive when
  the next micro-batch's forward starts or when the optimizer steps.
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train import VOCAB, toy_blocks, toy_cfg  # noqa: E402

from aadistill.initialization.adapters import recurrent_depth as RD  # noqa: E402
from aadistill.training import hidden_state as HS  # noqa: E402
from aadistill.training.train import Trainer  # noqa: E402

GEOM = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, tie_word_embeddings=True, max_position_embeddings=128)
SCHED = [0, 1, 1, 2]                    #: a 3-layer teacher's spans [0], [1, 2], [3]: 3 blocks, 4 logical layers


def teacher(seed=2, layers=4):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=layers, **GEOM)).float()


def student(seed=1, sched=SCHED):
    torch.manual_seed(seed)
    cfg = RD.RecurrentDepthConfig(num_hidden_layers=len(sched), layer_to_block=sched,
                                  num_parameter_blocks=max(sched) + 1, **GEOM)
    return RD.RecurrentDepthForCausalLM(cfg).float()


def hidden_cfg(weight=1.0, pairs=None):
    return {"weight": weight, "pairs": pairs or [[p, p] for p in range(len(SCHED))], "rows": "kd",
            "error": "relative_squared_error_v1"}


def cfg(tmp_path, hidden=None, **over):
    loss = {"ce_weight": 0.25, "kd_weight": 1.0, "kd_scope": "assistant",
            "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": 8}}
    if hidden is not None:
        loss["hidden_state"] = hidden
    return toy_cfg(tmp_path, loss=loss, **over)


# ---- the error ----------------------------------------------------------------------------------------------------

def test_flat_rows_index_the_prediction_positions():
    mask = torch.tensor([[True, False, True], [False, True, False]])           #: [B=2, T-1=3], T = 4
    assert HS.flat_rows(mask).tolist() == [0, 2, 5]


def test_the_error_and_its_backward_equal_the_plain_formula():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([0, 3, 6, 9])
    t = torch.randn(4, 7, dtype=torch.float64)
    inv = 1.0 / (t * t).sum(-1)
    got = HS.relative_squared_error_sum(h, rows, t, inv)
    got.backward()
    g_got = h.grad.clone()
    h.grad = None
    hr = h.reshape(-1, 7)[rows]
    want = (((hr - t) ** 2).sum(-1) / (t * t).sum(-1)).sum()
    want.backward()
    assert float(got) == pytest.approx(float(want), rel=1e-12)
    assert torch.allclose(g_got, h.grad, atol=1e-14)


def test_a_zero_energy_target_and_a_repeated_layer_are_refused():
    with pytest.raises(HS.HiddenStateError):
        HS.HiddenSupervision(torch.tensor([0]), {0: torch.zeros(1, 4)})
    sup = HS.HiddenSupervision(torch.tensor([0]), {0: torch.ones(1, 4)})
    sup.add(0, torch.zeros(1, 1, 4))
    with pytest.raises(HS.HiddenStateError):
        sup.add(0, torch.zeros(1, 1, 4))


def test_teacher_layers_are_captured_once_at_the_rows_without_a_graph():
    t = teacher().eval()
    x = toy_blocks()[0][:1]
    rows = torch.tensor([0, 5, 17])
    with torch.no_grad(), HS.capture_layer_outputs(t, [0, 3], rows) as got:
        out = t(x, output_hidden_states=True, use_cache=False)
    #: HF's hidden_states[l + 1] is layer l's output for every layer but the last (which is after the final norm)
    assert torch.equal(got[0], out.hidden_states[1].reshape(-1, 32)[rows])
    assert sorted(got) == [0, 3] and not got[3].requires_grad
    with pytest.raises(HS.HiddenStateError):
        with torch.no_grad(), HS.capture_layer_outputs(t, [7], rows):
            pass


def test_the_student_hands_over_each_logical_layer_once_even_under_checkpointing():
    s = student()
    s.train()
    s.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    x = toy_blocks()[0][:1]
    rows = torch.tensor([1, 4, 9])
    sup = HS.HiddenSupervision(rows, {p: torch.randn(3, 32) for p in range(len(SCHED))})
    h = s.get_decoder()(input_ids=x, use_cache=False, hidden_supervision=sup).last_hidden_state
    (h.sum() + sup.loss_sum).backward()                 #: the backward recomputes every block call
    sup.finish()
    assert sup.seen == list(range(len(SCHED)))          #: once each, in order, nothing re-registered


# ---- through the trainer ------------------------------------------------------------------------------------------

def one_step(tmp_path, hidden=None, gc=False, seed=1):
    c = cfg(tmp_path, hidden=hidden, gradient_checkpointing=gc)
    tr = Trainer(c, student(seed), toy_blocks(), teacher=teacher(), device="cpu")
    out = tr.step_once()
    grads = {n: p.grad.clone() for n, p in tr.student.named_parameters() if p.grad is not None}
    return tr, out, grads


def test_the_step_objective_adds_the_weighted_hidden_mean_and_reports_it(tmp_path):
    tr, out, grads = one_step(tmp_path / "h", hidden=hidden_cfg(0.5))
    assert out["hidden"] is not None and out["hidden"] > 0 and out["hidden_weight"] == 0.5
    assert out["loss"] == pytest.approx(0.25 * out["ce"] + out["kd"] + 0.5 * out["hidden"], rel=1e-5)
    assert sorted(out["hidden_by_layer"], key=int) == [str(p) for p in range(len(SCHED))]
    assert all(p.grad is None for p in tr.teacher.parameters())
    for b in range(3):                                   #: the hidden term's gradient reaches every block
        assert any(n.startswith(f"model.layers.{b}.") for n in grads)
    rec = tr.execution_record()["hidden_state_objective"]
    assert rec["objective"] == HS.OBJECTIVE_ID and rec["pairs"] == [[p, p] for p in range(len(SCHED))]


def test_the_hidden_term_moves_the_gradients_and_its_absence_is_the_unchanged_path(tmp_path):
    _, out0, g0 = one_step(tmp_path / "0")
    _, outh, gh = one_step(tmp_path / "h", hidden=hidden_cfg(1.0))
    assert "hidden" not in out0
    assert out0["ce"] == pytest.approx(outh["ce"], rel=1e-6) and out0["kd"] == pytest.approx(outh["kd"], rel=1e-6)
    assert any(not torch.allclose(g0[n], gh[n]) for n in g0)
    tr = Trainer(cfg(tmp_path / "p"), student(), toy_blocks(), teacher=teacher(), device="cpu")
    tr._micro_losses(toy_blocks()[0][:1], toy_blocks()[1][:1])
    assert tr._micro_hidden is None and not tr.hidden_cfg


def test_gradient_checkpointing_gives_the_same_hidden_loss_and_gradients(tmp_path):
    _, a, ga = one_step(tmp_path / "a", hidden=hidden_cfg(1.0), gc=False)
    _, b, gb = one_step(tmp_path / "b", hidden=hidden_cfg(1.0), gc=True)
    assert a["hidden"] == pytest.approx(b["hidden"], rel=1e-5)
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-6, rtol=1e-4), n


def test_validation_reports_the_hidden_error_per_layer(tmp_path):
    blocks = toy_blocks()
    tr = Trainer(cfg(tmp_path, hidden=hidden_cfg(1.0)), student(), blocks, val_blocks=blocks, teacher=teacher(),
                 device="cpu")
    ev = tr.evaluate(2)
    assert ev["val_hidden"] > 0 and sorted(ev["val_hidden_by_layer"], key=int) == ["0", "1", "2", "3"]


def test_configs_that_cannot_be_computed_are_refused(tmp_path):
    from aadistill.training.train import validate_train_config
    with pytest.raises(ValueError, match="kd_support"):
        validate_train_config(toy_cfg(tmp_path, loss={"ce_weight": 0.25, "kd_weight": 1.0, "kd_scope": "assistant",
                                                     "hidden_state": hidden_cfg()}))
    with pytest.raises(ValueError, match="weight"):
        validate_train_config(cfg(tmp_path, hidden=hidden_cfg(weight=0.0)))
    with pytest.raises(ValueError, match="twice"):
        validate_train_config(cfg(tmp_path, hidden=hidden_cfg(pairs=[[0, 0], [0, 1]])))
    with pytest.raises(ValueError, match="supports_hidden_supervision"):
        Trainer(cfg(tmp_path, hidden=hidden_cfg(pairs=[[0, 0]])), teacher(5, layers=2), toy_blocks(),
                teacher=teacher(), device="cpu")
    with pytest.raises(ValueError, match="outside"):
        Trainer(cfg(tmp_path, hidden=hidden_cfg(pairs=[[0, 9]])), student(), toy_blocks(), teacher=teacher(),
                device="cpu")


# ---- the targets' lifetime ------------------------------------------------------------------------------------------

def _watch_targets(monkeypatch):
    """Every HiddenSupervision the trainer builds, its teacher targets watched by weak reference, by micro-batch."""
    made: list[list[weakref.ref]] = []

    class Watched(HS.HiddenSupervision):
        def __init__(self, rows, targets):
            super().__init__(rows, targets)
            made.append([weakref.ref(t) for t in targets.values()])
    monkeypatch.setattr(HS, "HiddenSupervision", Watched)
    return made


def _alive(refs) -> int:
    gc.collect()
    return sum(r() is not None for r in refs)


def test_a_micro_batchs_targets_are_gone_before_the_next_forward_and_the_optimizer_step(tmp_path, monkeypatch):
    made = _watch_targets(monkeypatch)
    tr = Trainer(cfg(tmp_path, hidden=hidden_cfg(1.0)), student(), toy_blocks(), teacher=teacher(), device="cpu")
    assert tr.cfg["batch"]["blocks_per_step"] // tr.cfg["batch"]["micro_blocks"] >= 2
    seen = []
    real = tr._micro_losses

    def micro(*a, **k):
        seen.append(sum(_alive(m) for m in made))       #: every earlier micro-batch's targets, at this forward
        return real(*a, **k)
    monkeypatch.setattr(tr, "_micro_losses", micro)
    real_step = tr.opt.step

    def step(*a, **k):
        seen.append(("opt", sum(_alive(m) for m in made)))
        return real_step(*a, **k)
    monkeypatch.setattr(tr.opt, "step", step)
    out = tr.step_once()
    assert out["hidden"] > 0 and len(made) >= 2
    assert seen[0] == 0 and all(x == 0 for x in seen[1:-1]), seen
    assert seen[-1] == ("opt", 0), seen


def test_validation_releases_each_micro_batchs_targets_before_the_next(tmp_path, monkeypatch):
    made = _watch_targets(monkeypatch)
    blocks = toy_blocks()
    tr = Trainer(cfg(tmp_path, hidden=hidden_cfg(1.0)), student(), blocks, val_blocks=blocks, teacher=teacher(),
                 device="cpu")
    seen = []
    real = tr._micro_losses

    def micro(*a, **k):
        seen.append(sum(_alive(m) for m in made))
        return real(*a, **k)
    monkeypatch.setattr(tr, "_micro_losses", micro)
    ev = tr.evaluate(3)
    assert ev["val_hidden"] > 0 and len(made) == 3
    assert seen == [0, 0, 0] and sum(_alive(m) for m in made) == 0

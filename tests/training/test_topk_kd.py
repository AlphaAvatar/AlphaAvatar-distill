"""Head-row reference-Top-K + tail KD (`aadistill.training.topk_kd`) on tiny models.

The load-bearing tests: with ``top_k >= |V|`` the objective IS the historical
full-vocabulary KD (values and gradients, through the trainer); below it, the
value is the K+1-partition formula computed by hand and never exceeds the full
KL; the student's tail gradient is the analytic one; chunking and recompute do
not move the result; resume stays bitwise exact.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.training import topk_kd
from aadistill.training.train import Trainer, validate_train_config
from test_train import VOCAB, tiny_model, toy_blocks, toy_cfg


def _kd_cfg(tmp_path, top_k, **over):
    loss = {"ce_weight": 0.25, "kd_weight": 1.0, "kd_scope": "assistant",
            "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": top_k}}
    loss.update(over.pop("loss", {}))
    return toy_cfg(tmp_path, loss=loss, **over)


def _partition_kl(t_logits, s_logits, k):
    """The K+1 partition KL computed directly from full logits."""
    tp = torch.log_softmax(t_logits.double(), -1)
    sp = torch.log_softmax(s_logits.double(), -1)
    idx = torch.topk(t_logits.float(), k, -1).indices
    p_s, q_s = tp.gather(-1, idx), sp.gather(-1, idx)
    mask = torch.ones_like(tp, dtype=torch.bool).scatter(-1, idx, False)
    p_t = torch.logsumexp(tp.masked_fill(~mask, float("-inf")), -1)
    q_t = torch.logsumexp(sp.masked_fill(~mask, float("-inf")), -1)
    return ((p_s.exp() * (p_s - q_s)).sum(-1) + p_t.exp() * (p_t - q_t)).sum()


def test_full_support_equals_the_historical_full_vocab_kd(tmp_path):
    """top_k >= V: same CE, same KD, same gradients as the full-logit path."""
    blocks = toy_blocks()
    ids, mask = blocks[0][:2], blocks[1][:2]
    out = {}
    for name, cfg in (("full", toy_cfg(tmp_path / "f", loss={"ce_weight": 0.25, "kd_weight": 1.0,
                                                               "kd_scope": "assistant"})),
                      ("rows", _kd_cfg(tmp_path / "r", VOCAB + 7))):
        tr = Trainer(cfg, tiny_model(1), blocks, teacher=tiny_model(2), device="cpu")
        ce, ce_n, kd, kd_n = tr._micro_losses(ids, mask)
        (0.25 * ce / ce_n + kd / kd_n).backward()
        out[name] = (float(ce), ce_n, float(kd), kd_n,
                     {n: p.grad.clone() for n, p in tr.student.named_parameters() if p.grad is not None})
    f, r = out["full"], out["rows"]
    assert f[1] == r[1] and f[3] == r[3] and r[3] > 0
    assert r[0] == pytest.approx(f[0], rel=1e-5)
    assert r[2] == pytest.approx(f[2], rel=1e-5)
    assert f[4].keys() == r[4].keys()
    for n in f[4]:
        assert torch.allclose(f[4][n], r[4][n], atol=1e-6, rtol=1e-4), n


@pytest.mark.parametrize("k", [1, 5, 17])
def test_topk_tail_value_is_the_partition_formula_and_a_lower_bound(k):
    torch.manual_seed(k)
    n, v = 9, 40
    t_logits, s_logits = torch.randn(n, v) * 3, torch.randn(n, v) * 3
    ident = lambda h: h  # noqa: E731  (the "head" of logits is the identity)
    sketch = topk_kd.teacher_sketch(ident, t_logits, top_k=k, chunk=4)
    assert sketch.has_tail
    rows = torch.ones(n, dtype=torch.bool)
    targets = torch.randint(0, v, (n,))
    ce, kd = topk_kd.topk_tail_losses(ident, s_logits, sketch, targets, rows, rows, chunk=4)
    assert float(kd) == pytest.approx(float(_partition_kl(t_logits, s_logits, k)), rel=1e-5)
    tl, sl = torch.log_softmax(t_logits, -1), torch.log_softmax(s_logits, -1)
    full = (tl.exp() * (tl - sl)).sum()
    assert float(kd) <= float(full) + 1e-5
    expect_ce = torch.nn.functional.cross_entropy(s_logits, targets, reduction="sum")
    assert float(ce) == pytest.approx(float(expect_ce), rel=1e-6)


def test_student_gradient_is_the_analytic_one_inside_and_outside_the_support():
    torch.manual_seed(3)
    v, k = 30, 4
    t = torch.randn(1, v) * 2
    z = (torch.randn(1, v) * 2).requires_grad_(True)
    ident = lambda h: h  # noqa: E731
    sketch = topk_kd.teacher_sketch(ident, t, top_k=k)
    one = torch.ones(1, dtype=torch.bool)
    _, kd = topk_kd.topk_tail_losses(ident, z, sketch, torch.zeros(1, dtype=torch.long),
                                     torch.zeros(1, dtype=torch.bool), one)
    kd.backward()
    q = torch.softmax(z.detach().double(), -1)[0]
    p = torch.softmax(t.double(), -1)[0]
    s = sketch.support_indices[0]
    out = torch.ones(v, dtype=torch.bool)
    out[s] = False
    p_tail, q_tail = p[out].sum(), q[out].sum()
    expect = q.clone()
    expect[s] = q[s] - p[s]
    expect[out] = q[out] - p_tail * q[out] / q_tail
    assert torch.allclose(z.grad[0].double(), expect, atol=1e-6)
    # Gradient reaches entries outside the support (the tail is not detached).
    assert float(z.grad[0][out].abs().sum()) > 0


def test_zero_mass_buckets_give_finite_values_and_gradients():
    v = 8
    z = torch.randn(2, v, requires_grad=True)
    sk = topk_kd.TeacherSketch(
        support_indices=torch.tensor([[0, 1], [2, 3]]),
        support_log_probs=torch.log(torch.tensor([[1.0, 0.0], [0.5, 0.5]])),
        tail_log_prob=torch.full((2,), float("-inf")), has_tail=True)
    rows = torch.ones(2, dtype=torch.bool)
    _, kd = topk_kd.topk_tail_losses(lambda h: h, z, sk, torch.zeros(2, dtype=torch.long),
                                     ~rows, rows)
    kd.backward()
    assert torch.isfinite(kd) and torch.isfinite(z.grad).all()


def test_chunking_and_recompute_do_not_move_value_or_gradient():
    torch.manual_seed(5)
    head = torch.nn.Linear(6, 25, bias=False)
    h0 = torch.randn(11, 6)
    t = torch.randn(11, 25)
    sketch = topk_kd.teacher_sketch(lambda x: x, t, top_k=6)
    targets = torch.randint(0, 25, (11,))
    ce_rows = torch.rand(11) > 0.3
    kd_rows = torch.rand(11) > 0.3
    ref = None
    for chunk, recompute in ((11, False), (1, True), (4, True), (64, True)):
        h = h0.clone().requires_grad_(True)
        head.zero_grad()
        ce, kd = topk_kd.topk_tail_losses(head, h, sketch, targets, ce_rows, kd_rows,
                                          temperature=1.7, chunk=chunk, recompute=recompute)
        (ce + kd).backward()
        got = (float(ce), float(kd), h.grad.clone(), head.weight.grad.clone())
        if ref is None:
            ref = got
            continue
        assert got[0] == pytest.approx(ref[0], rel=1e-6) and got[1] == pytest.approx(ref[1], rel=1e-6)
        assert torch.allclose(got[2], ref[2], atol=1e-6) and torch.allclose(got[3], ref[3], atol=1e-6)


def test_bf16_autocast_rows_agree_with_full_logits_within_bf16(tmp_path):
    blocks = toy_blocks()
    ids, mask = blocks[0][:2], blocks[1][:2]
    vals = {}
    for name, cfg in (("full", toy_cfg(tmp_path / "f", autocast_bf16=True,
                                       loss={"ce_weight": 0.25, "kd_weight": 1.0, "kd_scope": "assistant"})),
                      ("rows", _kd_cfg(tmp_path / "r", VOCAB, autocast_bf16=True))):
        tr = Trainer(cfg, tiny_model(1), blocks, teacher=tiny_model(2), device="cpu")
        with torch.no_grad():
            ce, ce_n, kd, kd_n = tr._micro_losses(ids, mask)
        vals[name] = (float(ce) / ce_n, float(kd) / kd_n)
    assert vals["rows"][0] == pytest.approx(vals["full"][0], rel=2e-2)
    assert vals["rows"][1] == pytest.approx(vals["full"][1], rel=5e-2, abs=1e-3)


def test_trainer_trains_reports_and_records_the_objective(tmp_path):
    cfg = _kd_cfg(tmp_path, 8, schedule={"total_steps": 4})
    tr = Trainer(cfg, tiny_model(1), toy_blocks(), toy_blocks(n=2), teacher=tiny_model(2), device="cpu")
    m = [tr.step_once() for _ in range(4)]
    assert all(torch.isfinite(torch.tensor([x["loss"]])).all() for x in m)
    assert 0.0 <= m[0]["kd_target_in_support"] <= 1.0 and 0.0 < m[0]["kd_teacher_tail_mass"] < 1.0
    rec = tr.execution_record()["kd_objective"]
    assert rec["support"] == {"support_id": "reference_topk_tail_v1", "top_k": 8}
    assert rec["is_full_vocabulary_kl"] is False and rec["tail"] == topk_kd.TAIL_ALGORITHM_ID
    ev = tr.evaluate()
    assert ev["val_kd"] > 0
    # A historical config records no objective block: its manifest is unchanged.
    plain = Trainer(toy_cfg(tmp_path / "p"), tiny_model(1), toy_blocks(), device="cpu")
    assert "kd_objective" not in plain.execution_record()


def test_document_isolated_rows_use_the_position_ids(tmp_path):
    ids, mask = toy_blocks(n=2)
    content = torch.ones_like(mask)
    pos = torch.cat([torch.arange(16), torch.arange(16)]).repeat(2, 1)
    cfg = _kd_cfg(tmp_path, VOCAB, batch={"context_strategy": "document_isolated"})
    tr = Trainer(cfg, tiny_model(1), (ids, mask, content, pos), teacher=tiny_model(2), device="cpu")
    full = Trainer(toy_cfg(tmp_path / "f", batch={"context_strategy": "document_isolated"},
                           loss={"ce_weight": 0.25, "kd_weight": 1.0, "kd_scope": "assistant"}),
                   tiny_model(1), (ids, mask, content, pos), teacher=tiny_model(2), device="cpu")
    with torch.no_grad():
        a = tr._micro_losses(ids, mask, content, pos)
        b = full._micro_losses(ids, mask, content, pos)
    assert a[1] == b[1] and a[3] == b[3]
    assert float(a[0]) == pytest.approx(float(b[0]), rel=1e-5)
    assert float(a[2]) == pytest.approx(float(b[2]), rel=1e-5)


def test_resume_is_bitwise_exact_and_binds_the_resume_identity(tmp_path):
    from transformers import AutoModelForCausalLM

    blocks, ident = toy_blocks(), {"data": "d1", "initialization": "i1"}
    teacher = tiny_model(2)
    cfg = _kd_cfg(tmp_path / "a", 8, schedule={"total_steps": 4})
    a = Trainer(cfg, tiny_model(1), blocks, teacher=teacher, device="cpu", resume_identity=ident)
    for _ in range(4):
        a.step_once()
    cfg_b = _kd_cfg(tmp_path / "b", 8, schedule={"total_steps": 4})
    b = Trainer(cfg_b, tiny_model(1), blocks, teacher=teacher, device="cpu", resume_identity=ident)
    for _ in range(2):
        b.step_once()
    ckpt = b.save_checkpoint()
    student = AutoModelForCausalLM.from_pretrained(ckpt / "model", dtype=torch.float32)
    c = Trainer(cfg_b, student, blocks, teacher=teacher, device="cpu", resume_identity=ident)
    c.restore(ckpt)
    for _ in range(2):
        c.step_once()
    pa = dict(a.student.named_parameters())
    for n, p in c.student.named_parameters():
        assert torch.equal(p, pa[n]), n
    other = Trainer(cfg_b, tiny_model(1), blocks, teacher=teacher, device="cpu",
                    resume_identity={**ident, "data": "d2"})
    with pytest.raises(ValueError, match="different data, initialization"):
        other.restore(ckpt)


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="kd_weight is 0"):
        _kd_cfg(tmp_path, 8, loss={"kd_weight": 0.0})
    with pytest.raises(ValueError, match="not a training support"):
        toy_cfg(tmp_path, loss={"ce_weight": 1.0, "kd_weight": 1.0, "kd_scope": "assistant",
                                "kd_support": {"support_id": "full_vocab_v1"}})
    with pytest.raises(ValueError, match="integer top_k"):
        _kd_cfg(tmp_path, None)
    with pytest.raises(ValueError, match="unknown keys"):
        toy_cfg(tmp_path, loss={"ce_weight": 1.0, "kd_weight": 1.0, "kd_scope": "assistant",
                                "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": 4,
                                               "tail": "x"}})
    cfg = _kd_cfg(tmp_path, 4)
    validate_train_config(cfg)


def test_a_model_that_post_processes_its_logits_is_refused():
    m = tiny_model(1)
    m.config.final_logit_softcapping = 30.0
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        topk_kd.output_head(m)


def test_monitor_logs_group_statistics_and_changes_nothing(tmp_path):
    import json

    base = toy_cfg(tmp_path / "a", schedule={"total_steps": 2})
    mon = toy_cfg(tmp_path / "b", schedule={"total_steps": 2},
                  monitor={"every": 1, "groups": {"norms": "layernorm", "attn": r"self_attn\.q_proj"}})
    a, b = Trainer(base, tiny_model(1), toy_blocks(), device="cpu"), Trainer(mon, tiny_model(1), toy_blocks(),
                                                                          device="cpu")
    for _ in range(2):
        a.step_once()
        b.step_once()
    pa = dict(a.student.named_parameters())
    assert all(torch.equal(p, pa[n]) for n, p in b.student.named_parameters())
    rows = [json.loads(line) for line in (tmp_path / "b" / "run" / "train_log.jsonl").read_text().splitlines()]
    stats = [r for r in rows if r["event"] == "param_stats"]
    assert [r["step"] for r in stats] == [1, 2]
    assert stats[0]["groups"]["norms"]["grad_norm"] > 0 and stats[0]["groups"]["attn"]["parameters"] > 0
    with pytest.raises(ValueError, match="monitor must be"):
        toy_cfg(tmp_path / "c", monitor={"every": 0, "groups": {"x": "y"}})

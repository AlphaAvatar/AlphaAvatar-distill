"""Stage 1 initialization tests on a tiny random Qwen3 teacher (CPU, fast).

The load-bearing test is identity exactness: at equal geometry with an
identity projection, sandwich init (norm folding, head/neuron "selection" of
everything, final-norm solve) must reproduce the teacher's logits. That
verifies the algebra end to end; the compressed cases then only need shape,
finiteness, determinism, and selection-logic checks.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.collect import ActivationStatsCollector
from aadistill.sandwich import depth_span_map, init_student, select_q_heads
from aadistill.student import build_student, build_student_config


def tiny_teacher(seed: int = 7):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=32, num_hidden_layers=4,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=256,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    # Randomize all RMSNorm weights away from 1.0 so norm folding is actually
    # exercised (fresh models initialize every norm weight to ones).
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.uniform_(0.5, 1.5)
    return model


def collect_stats(model, seed: int = 11, n_seqs: int = 8, seq_len: int = 64):
    torch.manual_seed(seed)
    collector = ActivationStatsCollector(model)
    for _ in range(n_seqs):
        ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
        collector.process(ids)
    collector.close()
    return collector.state()


def geometry(cfg, **overrides):
    g = dict(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        tie_word_embeddings=True,
    )
    g.update(overrides)
    return g


def test_depth_span_map():
    spans = depth_span_map(36, 28)
    assert len(spans) == 28
    # Middle-band policy: first 4 layers 1:1, merge band t4..19, tail 1:1.
    assert spans[0] == {"student": 0, "teacher_span": [0, 1], "representative": 0}
    assert spans[3] == {"student": 3, "teacher_span": [3, 4], "representative": 3}
    assert spans[4] == {"student": 4, "teacher_span": [4, 6], "representative": 4}
    assert spans[11] == {"student": 11, "teacher_span": [18, 20], "representative": 18}
    assert spans[12] == {"student": 12, "teacher_span": [20, 21], "representative": 20}
    assert spans[27]["representative"] == 35
    # Every teacher layer is covered exactly once, in order.
    covered = [t for s in spans for t in range(*s["teacher_span"])]
    assert covered == list(range(36))
    assert depth_span_map(4, 4) == [
        {"student": s, "teacher_span": [s, s + 1], "representative": s} for s in range(4)
    ]
    spans43 = depth_span_map(4, 3)
    assert [t for s in spans43 for t in range(*s["teacher_span"])] == list(range(4))
    with pytest.raises(ValueError):
        depth_span_map(4, 5)
    with pytest.raises(ValueError):
        depth_span_map(4, 1)


def test_select_q_heads_gqa_structure():
    head_dim, t_q, t_kv = 4, 8, 2
    w_q = torch.zeros(t_q * head_dim, 16)
    w_o = torch.ones(16, t_q * head_dim)
    # Give heads strictly increasing W_q norms inside each group of 4.
    for h in range(t_q):
        w_q[h * head_dim:(h + 1) * head_dim, :] = (h % 4) + 1
    kept = select_q_heads(w_q, w_o, t_q, t_kv, s_q_heads=4, head_dim=head_dim)
    assert kept == [2, 3, 6, 7]  # top-2 per group, original order preserved
    with pytest.raises(ValueError):
        select_q_heads(w_q, w_o, t_q, t_kv, s_q_heads=3, head_dim=head_dim)


def test_identity_projection_reproduces_teacher():
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(teacher.config))
    student = build_student(cfg, torch.float32, seed=123)
    eye = torch.eye(teacher.config.hidden_size, dtype=torch.float64)
    init_student(teacher, student, state, proj_override=eye)

    # Final-norm least squares must recover the teacher weight exactly at P=I.
    assert torch.allclose(
        student.model.norm.weight.double(),
        teacher.model.norm.weight.double(), atol=1e-6,
    )
    torch.manual_seed(3)
    ids = torch.randint(0, 128, (1, 48))
    with torch.no_grad():
        lt = teacher(ids).logits
        ls = student(ids).logits
    assert (lt - ls).abs().max().item() < 5e-3


def test_compressed_init_forward_and_determinism():
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(
        teacher.config, hidden_size=16, num_hidden_layers=3,
        intermediate_size=24, num_attention_heads=2,
    ))
    student = build_student(cfg, torch.float32, seed=123)
    diag = init_student(teacher, student, state)

    assert diag["projection"]["orthonormality_error"] < 1e-9
    assert 0.0 < diag["projection"]["energy_captured_frac"] <= 1.0 + 1e-12
    assert len(diag["depth_map"]) == 3
    for kept in diag["kept_q_heads"].values():
        assert len(kept) == 2  # one Q head per KV group survives

    torch.manual_seed(5)
    ids = torch.randint(0, 128, (1, 40))
    with torch.no_grad():
        a = student(ids).logits
        b = student(ids).logits
    assert torch.isfinite(a).all()
    assert (a - b).abs().max().item() == 0.0

    # Same stats + same seed → bitwise-identical second init (P4/P5).
    student2 = build_student(cfg, torch.float32, seed=123)
    init_student(teacher, student2, state)
    for (n1, p1), (_, p2) in zip(student.named_parameters(), student2.named_parameters()):
        assert torch.equal(p1, p2), n1


def test_compressed_save_load_roundtrip(tmp_path):
    from transformers import Qwen3ForCausalLM

    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(
        teacher.config, hidden_size=16, num_hidden_layers=3,
        intermediate_size=24, num_attention_heads=2,
    ))
    student = build_student(cfg, torch.float32, seed=123)
    init_student(teacher, student, state)
    student.save_pretrained(tmp_path / "ckpt")
    reloaded = Qwen3ForCausalLM.from_pretrained(tmp_path / "ckpt").eval()

    torch.manual_seed(9)
    ids = torch.randint(0, 128, (1, 32))
    with torch.no_grad():
        assert torch.equal(student(ids).logits, reloaded(ids).logits)


def test_init_rejects_geometry_mismatch():
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(
        teacher.config, num_key_value_heads=1, num_attention_heads=2))
    student = build_student(cfg, torch.float32, seed=1)
    with pytest.raises(ValueError, match="KV head count"):
        init_student(teacher, student, state)

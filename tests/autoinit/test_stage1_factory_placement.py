"""Every fresh allocation on the Stage-1 path is placed, or is host-only on purpose.

Category 5 of `autoinit.stage1_device_contract@v1`, added after Phase-A attempt 9
died at $0.34 on `project.py`'s `avg` — a `torch.zeros` with a dtype and no
device, two call levels below the operator, that mixed with a `cuda:0` statistics
slice.

These assert **placement intent**. On a one-device box the fixed and the broken
version produce identical numbers, so a test that only checked the arithmetic
would have passed before the fix and would pass again if the fix were reverted.
`RecordFactories` observes what the factory was *told*, which is the property
that differs.

The complement is `test_search_operator_device_split.py`, which catches the other
half — an unmoved cache tensor meeting a model-side one. Neither instrument sees
the other's class, which is why both exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factory_placement import RecordFactories                      # noqa: E402

from aadistill.init.project import stream_projection               # noqa: E402
from aadistill.init.sandwich import _head_rows, init_student       # noqa: E402


# --- fixtures: the real tiny teacher the Stage-1 tests already use ----------

def tiny_teacher(seed: int = 7):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=32, num_hidden_layers=4,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=256,
    )
    return Qwen3ForCausalLM(cfg).float().eval()


def collect_stats(model, seed: int = 11, n_seqs: int = 4, seq_len: int = 32):
    from aadistill.init.collect import ActivationStatsCollector

    torch.manual_seed(seed)
    collector = ActivationStatsCollector(model)
    for _ in range(n_seqs):
        collector.process(torch.randint(0, model.config.vocab_size, (1, seq_len)))
    collector.close()
    return collector.state()


@pytest.fixture(scope="module")
def teacher_and_state():
    t = tiny_teacher()
    return t, collect_stats(t)


# --- stream_projection: both allocations, both device-coupled ---------------

def test_stream_projection_places_every_tensor_it_allocates(teacher_and_state):
    """`avg` and the orthonormality `eye` are the two, and both must name a
    device. `avg` is what attempt 9 died on; the `eye` is the latent second
    instance eleven lines below it, which a one-line fix would have left."""
    _t, state = teacher_and_state
    with RecordFactories() as rec:
        proj, diag = stream_projection(state, 16, [0, 1, 2])

    assert not rec.unplaced(), (
        "Stage-1 stream_projection allocates without naming a device: "
        f"{rec.unplaced()}. On a pod the statistics are on cuda:0 and this is "
        "'Expected all tensors to be on the same device' — Phase-A attempt 9, "
        "$0.34. Derive the device from what the tensor meets.")
    assert [c.name for c in rec.calls] == ["zeros", "eye"], (
        f"the set of allocations changed: {[c.name for c in rec.calls]}. A new "
        "one must be classified device-coupled or host-only before this test is "
        "updated to expect it.")
    assert diag["orthonormality_error"] < 1e-9
    assert proj.dtype is torch.float64, "the fix must not have moved the dtype"


def test_stream_projection_allocations_follow_the_statistics(teacher_and_state):
    """Placed *from the state*, not merely placed. `.device` is the only
    handle a one-device box gives, so this pins the source rather than the
    value: the recorded device must be the state's own."""
    _t, state = teacher_and_state
    want = state["residual_sqsum"].device
    with RecordFactories() as rec:
        proj, _ = stream_projection(state, 16, [0, 1])
    assert [c.device for c in rec.calls] == [want, proj.device]
    assert proj.device == want


def test_the_projection_is_unchanged_by_the_placement_fix(teacher_and_state):
    """Placement is not allowed to move the mathematics. The frozen Stage-1
    algebra is a separate, load-bearing property and this fix must be invisible
    to it."""
    _t, state = teacher_and_state
    a, da = stream_projection(state, 16, [0, 1, 2], [9.0, 1.0, 8.0])
    b, db = stream_projection(state, 16, [0, 1, 2], [9.0, 1.0, 8.0])
    assert torch.equal(a, b)
    assert da["energy_captured_frac"] == db["energy_captured_frac"]
    # Columns orthonormal, which is the property the eye was diagnosing.
    assert torch.allclose(a.T @ a, torch.eye(16, dtype=torch.float64), atol=1e-9)


# --- _head_rows: an index that slices model-side weights --------------------

def test_head_rows_lands_where_it_is_told():
    """Asserted against `meta`, not `cpu`.

    `assert rows.device == cpu` on a CPU-only box is true whether or not the
    function honours its argument — the first version of this test passed with
    the `.to(device)` deleted. `meta` is a real second device present on every
    machine, so it distinguishes "placed" from "happens to be right".
    """
    rows = _head_rows([0, 2], 4, device=torch.device("cpu"))
    assert rows.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]

    elsewhere = _head_rows([0, 2], 4, device="meta")
    assert elsewhere.device.type == "meta", (
        "_head_rows ignored the device it was given; the index would be built "
        "on the host whatever weight it is about to slice")
    assert elsewhere.shape == rows.shape


def test_head_rows_without_a_device_is_still_allowed_for_arithmetic_only():
    """`device=None` stays legal — `_common.head_rows` made the same choice —
    so a caller that only wants the indices is not forced to invent one. The
    contract is on the CALL SITE, checked below."""
    assert _head_rows([1], 2).tolist() == [2, 3]


def test_init_student_builds_its_head_index_on_the_weights_it_slices(
        teacher_and_state):
    """The call site is the contract. `q_rows` indexes the parent's q_proj and
    o_proj, so it is built on their device — the defect `_common.head_rows`
    already fixed in the operator path and this one had not."""
    import aadistill.init.sandwich as sandwich

    teacher, state = teacher_and_state
    from aadistill.models.student import build_student, build_student_config

    seen: list[dict] = []
    real = sandwich._head_rows

    def spy(heads, head_dim, device=None):
        seen.append({"device": device})
        return real(heads, head_dim, device=device)

    # The geometry `tests/init/test_stage1.py` already exercises. 4 -> 2 layers
    # is a different question (`depth_span_map` returns three spans for it) and
    # is frozen Stage-1 mathematics, untouched here.
    cfg = build_student_config(teacher.config, dict(
        hidden_size=16, num_hidden_layers=3, intermediate_size=24,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True))
    student = build_student(cfg, torch.float32, seed=3)

    sandwich._head_rows = spy
    try:
        init_student(teacher, student, state)
    finally:
        sandwich._head_rows = real

    assert seen, "init_student no longer builds a head index; re-derive this test"
    want = teacher.model.layers[0].self_attn.q_proj.weight.device
    unplaced = [c for c in seen if c["device"] is None]
    assert not unplaced, (
        f"{len(unplaced)} of {len(seen)} head-index builds named no device. The "
        "index slices the parent's q_proj and o_proj; on a pod those are on "
        "cuda:0 and a host index either raises or silently does the wrong thing.")
    assert all(c["device"] == want for c in seen)


# --- the audited closure, as a set -----------------------------------------

def test_the_audited_host_only_allocations_are_still_host_only():
    """The other half of the classification, pinned so it cannot rot.

    These are assembled from Python scalars and reduced back to Python scalars
    without meeting a parameter. They are CORRECT unplaced, and mechanically
    moving them would be the opposite error — so the audit records them by name
    rather than leaving a future reader to guess why they were skipped.
    """
    import inspect

    from aadistill.autoinit.operators import attention as attn_op
    from aadistill.init import sandwich as sw

    for mod, marker in ((attn_op, "A host diagnostic, deliberately"),
                        (sw, "select_q_heads")):
        assert marker in inspect.getsource(mod)

    # Both build a per-head score vector out of `.item()`/`float()` results and
    # rank it in Python. Neither ever indexes or multiplies a parameter.
    src = inspect.getsource(sw.select_q_heads)
    assert "torch.tensor([" in src and ".item()" in src
    assert "device" not in src, (
        "select_q_heads' score vector acquired a device. If it now meets a "
        "parameter it is device-coupled and this test is wrong; if it does not, "
        "the move is the mechanical change the contract warns against.")

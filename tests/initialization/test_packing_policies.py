"""Deterministic packing, and the separation that makes it safe to use.

Sorting items by length changes WHICH items share a forward. If it also
changed the order the per-item KLs were summed in, then switching packing
would move two things at once — batch composition and floating-point
aggregation order — and no measurement could tell them apart.

So the rule is: execution may reorder, aggregation may not. The scorer writes
each value into `per_item[layer][head][original_index]`, and the aggregation
walks the frozen mixture from 0. The tests below hold both halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.batching import micro_batches  # noqa: E402
from aadistill.initialization.calibration.packing import (  # noqa: E402
    LENGTH_SORTED_V1, ORIGINAL_ORDER_V1, PACKING_POLICIES, PackingError,
    item_lengths, pack, packed_batches, padding_profile,
)
from aadistill.initialization.calibration.profiles import NO_CALIBRATION  # noqa: E402
from aadistill.initialization.operators import get_implementation  # noqa: E402
from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext, OperatorError,
)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

GEOMETRY = dict(hidden_size=32, num_hidden_layers=3, intermediate_size=48,
                num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                vocab_size=128, tie_word_embeddings=True)
KEEP = 4


@pytest.fixture(autouse=True)
def registered():
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


def ragged(seed=13):
    """Deliberately out of length order, so sorting actually reorders."""
    g = torch.Generator().manual_seed(seed)
    lens = [16, 7, 14, 9, 11, 10]
    subtypes = ["general", "general", "alpha", "alpha", "beta", "beta"]
    return [{"item_id": f"i{k}",
             "domain": "general" if st == "general" else "task",
             "subtype": st,
             "input_ids": torch.randint(0, 128, (1, n), generator=g)}
            for k, (n, st) in enumerate(zip(lens, subtypes))]


# --- the policy is a pure function of lengths -----------------------------

def test_original_order_groups_consecutively():
    assert pack([5, 3, 9, 1, 7], 2) == [(0, 1), (2, 3), (4,)]


def test_length_sorted_is_ascending_with_index_tie_break():
    lengths = [5, 3, 9, 1, 7, 3]
    assert pack(lengths, 2, packing=LENGTH_SORTED_V1) == [(3, 1), (5, 0), (4, 2)]
    #: The tie between items 1 and 5 (both length 3) resolves to the LOWER
    #: original index. `sorted` is stable, so the explicit `i` in the key is
    #: belt-and-braces over that — but stability is an implementation detail
    #: of CPython's sort, and a key that ordered ties the other way would
    #: change which items share a forward and therefore the result.
    flat = [i for g in pack(lengths, 2, packing=LENGTH_SORTED_V1) for i in g]
    assert flat.index(1) < flat.index(5)
    #: The property, stated independently of how the key is written: among
    #: equal-length items, original order is preserved everywhere.
    lengths2 = [3, 3, 3, 3, 1]
    order = [i for g in pack(lengths2, 2, packing=LENGTH_SORTED_V1) for i in g]
    assert order == [4, 0, 1, 2, 3], order


def test_ascending_not_descending():
    """Stated, because the direction is a decision.

    With a short final partial group, descending leaves the largest remaining
    item grouped with much shorter ones — the case that pads worst.
    """
    lengths = [100, 90, 10, 9]
    asc = pack(lengths, 3, packing=LENGTH_SORTED_V1)
    assert asc[0] == (3, 2, 1), asc
    assert asc[-1] == (0,), "the longest item should end up alone"


def test_every_item_appears_exactly_once_under_every_policy():
    lengths = [5, 3, 9, 1, 7, 3, 12, 2]
    for policy in PACKING_POLICIES:
        for bs in (1, 2, 3, 4, 8, 16):
            flat = [i for g in pack(lengths, bs, packing=policy) for i in g]
            assert sorted(flat) == list(range(len(lengths))), (policy, bs)


def test_it_is_deterministic():
    lengths = [5, 3, 9, 1, 7, 3]
    for policy in PACKING_POLICIES:
        first = pack(lengths, 2, packing=policy)
        assert all(pack(lengths, 2, packing=policy) == first for _ in range(5))


def test_an_unknown_policy_refuses_rather_than_defaulting():
    with pytest.raises(PackingError, match="unknown packing"):
        pack([1, 2], 2, packing="length_sorted")
    with pytest.raises(PackingError, match="batch_size"):
        pack([1, 2], 0)


def test_b1_is_the_same_under_every_policy():
    """One item per forward has no composition to change."""
    lengths = [5, 3, 9, 1, 7]
    assert (pack(lengths, 1, packing=ORIGINAL_ORDER_V1)
            == [(0,), (1,), (2,), (3,), (4,)])
    assert sorted(pack(lengths, 1, packing=LENGTH_SORTED_V1)) == [
        (0,), (1,), (2,), (3,), (4,)]


# --- the reference policy must not become a second implementation --------

def test_original_order_reproduces_micro_batches_row_for_row():
    items = ragged()
    for bs in (1, 2, 3, 4):
        a = list(micro_batches(items, bs, pad_id=0, device="cpu"))
        b = list(packed_batches(items, bs, packing=ORIGINAL_ORDER_V1,
                                pad_id=0, device="cpu"))
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert torch.equal(x.input_ids, y.batch.input_ids)
            assert torch.equal(x.attention_mask, y.batch.attention_mask)
            assert x.lengths == y.batch.lengths


def test_the_original_indices_describe_the_rows():
    items = ragged()
    for policy in PACKING_POLICIES:
        for packed in packed_batches(items, 2, packing=policy, pad_id=0,
                                     device="cpu"):
            for row, index in enumerate(packed.original_indices):
                assert packed.batch.lengths[row] == int(
                    items[index]["input_ids"].shape[-1])
                assert packed.batch.items[row]["item_id"] == items[index][
                    "item_id"]


# --- padding is a $0 fact -------------------------------------------------

def test_sorting_reduces_padding_on_a_ragged_mixture():
    lengths = item_lengths(ragged())
    a = padding_profile(lengths, 2, packing=ORIGINAL_ORDER_V1)
    b = padding_profile(lengths, 2, packing=LENGTH_SORTED_V1)
    assert b["padded_positions"] < a["padded_positions"]
    assert a["valid_positions"] == b["valid_positions"]
    assert a["n_groups"] == b["n_groups"]


def test_b1_pads_nothing_under_either_policy():
    lengths = item_lengths(ragged())
    for policy in PACKING_POLICIES:
        p = padding_profile(lengths, 1, packing=policy)
        assert p["padded_positions"] == 0
        assert p["executed_positions"] == p["valid_positions"]


def test_the_profile_accounts_for_every_position():
    lengths = item_lengths(ragged())
    for bs in (1, 2, 4):
        for policy in PACKING_POLICIES:
            p = padding_profile(lengths, bs, packing=policy)
            expected = sum(w * n for w, n in zip(p["group_max_lengths"],
                                                 p["group_sizes"]))
            assert p["executed_positions"] == expected
            assert p["valid_positions"] + p["padded_positions"] == expected


# --- THE SEPARATION: execution order may move, aggregation may not -------

def context(model, calib, *, batch_size, packing):
    geo = ArchSpec.of("qwen3", GEOMETRY)
    return OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=geo,
        target_spec=geo.replace(num_attention_heads=KEEP),
        profile=NO_CALIBRATION, calibration_items=calib, seed=0, device="cpu",
        config={"n_calibration_items": len(calib),
                "calibration_forward_batch_size": batch_size,
                "calibration_batch_packing": packing})


def run(calib, *, batch_size, packing):
    impl = get_implementation("attention.causal_kl_v1")
    return impl.execute(context(build_tiny_model(GEOMETRY), calib,
                                batch_size=batch_size, packing=packing))


def test_the_landscape_is_indexed_by_the_frozen_mixture_not_by_execution():
    """The load-bearing one. A sorted run must attribute each value to the
    item that produced it, not to the row it happened to occupy."""
    calib = ragged()
    a = run(calib, batch_size=2, packing=ORIGINAL_ORDER_V1)
    b = run(calib, batch_size=2, packing=LENGTH_SORTED_V1)
    for out in (a, b):
        assert out.artifacts["causal_head_evidence"]["item_ids"] == [
            i["item_id"] for i in calib]
    #: On CPU fp32 the two agree per item, which they can only do if both are
    #: indexed the same way. A reordering bug shuffles these.
    ea = a.artifacts["causal_head_evidence"]["per_item_kl"]
    eb = b.artifacts["causal_head_evidence"]["per_item_kl"]
    for layer in range(GEOMETRY["num_hidden_layers"]):
        for head in range(GEOMETRY["num_attention_heads"]):
            for x, y in zip(ea[layer][head], eb[layer][head]):
                assert x == pytest.approx(y, abs=1e-6)


def test_rearranging_execution_does_not_change_the_aggregate(monkeypatch):
    """SYNTHETIC values, so only the ORDER varies.

    The same per-item numbers are fed through two different execution
    orders; the head scores must be bit-identical. This is what makes a
    packing change a one-variable change: if the aggregation followed
    execution order, switching packing would also change the float summation
    order and the two effects could not be separated.
    """
    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    calib = ragged()
    #: A value per (layer, head, ORIGINAL index) that does not depend on how
    #: the items were grouped.
    def fake(model, items, out_projections, *, n_q, head_dim, device,
             batch_size=1, packing=ORIGINAL_ORDER_V1, layers=None,
             deadline=None, progress=True):
        groups = pack(item_lengths(items), batch_size, packing=packing)
        per_item = [[[None] * len(items) for _ in range(n_q)]
                    for _ in out_projections]
        for layer in range(len(out_projections)):
            for head in range(n_q):
                #: Filled in EXECUTION order, exactly as the real scorer
                #: does, but the value depends only on the original index.
                for g in groups:
                    for index in g:
                        per_item[layer][head][index] = (
                            0.1 + 0.017 * index + 0.003 * layer + 0.011 * head)
        return {"per_item_kl": per_item, "scorer_seconds": 0.0,
                "physical_forward_invocations": len(groups),
                "executed_positions": 0, "valid_tokens": 0,
                "padded_positions": 0, "n_groups": len(groups),
                "layers_scored": list(range(len(out_projections))),
                "packing": packing,
                "calibration_forward_batch_size": batch_size}

    monkeypatch.setattr(m, "score_heads", fake)
    a = run(calib, batch_size=2, packing=ORIGINAL_ORDER_V1)
    b = run(calib, batch_size=2, packing=LENGTH_SORTED_V1)
    c = run(calib, batch_size=4, packing=LENGTH_SORTED_V1)
    ha = a.artifacts["causal_head_evidence"]["head_scores"]
    for other in (b, c):
        assert other.artifacts["causal_head_evidence"]["head_scores"] == ha, (
            "the aggregate moved with the execution order")
    assert a.artifacts["kept_heads"] == b.artifacts["kept_heads"]
    assert a.artifacts["kept_heads"] == c.artifacts["kept_heads"]


def test_the_aggregation_walks_the_items_not_the_groups():
    """Structural: the buckets are built from `enumerate(items)`."""
    import ast
    import inspect
    import textwrap

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    src = textwrap.dedent(inspect.getsource(m.AttentionCausalKLV1.apply))
    assert "for index, item in enumerate(items)" in src, (
        "the aggregation no longer walks the frozen mixture order")
    tree = ast.parse(src)
    #: And nothing in `apply` iterates groups any more.
    assert "packed_batches" not in ast.dump(tree)
    assert "micro_batches" not in ast.dump(tree)


# --- `layers` is a TIMING knob and must never produce a head map ---------

def test_the_layer_restriction_scores_only_those_layers():
    """The timing screen depends on it: three layers, full model forward."""
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        attention_out_projection, score_heads)

    model = build_tiny_model(GEOMETRY)
    projections = [attention_out_projection(QWEN3_ADAPTER, b)
                   for b in QWEN3_ADAPTER.blocks(model)]
    calib = ragged()
    full = score_heads(model, calib, projections, n_q=8, head_dim=8,
                       device="cpu", batch_size=1, progress=False)
    part = score_heads(model, calib, projections, n_q=8, head_dim=8,
                       device="cpu", batch_size=1, layers=[0, 2],
                       progress=False)
    assert full["layers_scored"] == [0, 1, 2]
    assert part["layers_scored"] == [0, 2]
    #: FEWER FORWARDS, which is the whole point of a screen.
    assert (part["physical_forward_invocations"]
            < full["physical_forward_invocations"])
    assert part["physical_forward_invocations"] == len(calib) * (2 * 8 + 1)
    #: The scored layers agree with the full run; layer 1 is left unscored.
    for layer in (0, 2):
        for head in range(8):
            for a, b in zip(full["per_item_kl"][layer][head],
                            part["per_item_kl"][layer][head]):
                assert a == pytest.approx(b, abs=1e-9)
    assert all(v is None for h in part["per_item_kl"][1] for v in h)


def test_a_restricted_run_cannot_become_a_head_map(monkeypatch):
    """A partial landscape must REFUSE at aggregation, not select heads from
    the layers that happen to be filled."""
    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    real = m.score_heads

    def partial(*a, **kw):
        kw["layers"] = [0]
        return real(*a, **kw)

    monkeypatch.setattr(m, "score_heads", partial)
    with pytest.raises(OperatorError, match="produced no causal KL"):
        run(ragged(), batch_size=1, packing=ORIGINAL_ORDER_V1)


def test_apply_never_restricts_the_layers():
    """`layers` exists only for timing; the operator must pass none."""
    import ast
    import inspect
    import textwrap

    from aadistill.initialization.operators.attention.gqa import causal_kl as m

    src = textwrap.dedent(inspect.getsource(m.AttentionCausalKLV1.apply))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "score_heads":
            assert "layers" not in {k.arg for k in node.keywords}, (
                "apply restricts the layers; that produces a partial landscape")
            return
    raise AssertionError("apply does not call score_heads")


@pytest.mark.parametrize("bad", [[-1], [99], [0, 3]])
def test_an_out_of_range_layer_refuses(bad):
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        attention_out_projection, score_heads)

    model = build_tiny_model(GEOMETRY)
    projections = [attention_out_projection(QWEN3_ADAPTER, b)
                   for b in QWEN3_ADAPTER.blocks(model)]
    with pytest.raises(OperatorError, match="outside"):
        score_heads(model, ragged(), projections, n_q=8, head_dim=8,
                    device="cpu", layers=bad, progress=False)


# --- the packing is identity-bearing --------------------------------------

def test_the_packing_reaches_the_trace_and_the_evidence():
    out = run(ragged(), batch_size=2, packing=LENGTH_SORTED_V1)
    assert out.trace["calibration_batch_packing"] == LENGTH_SORTED_V1
    assert out.artifacts["causal_head_evidence"][
        "calibration_batch_packing"] == LENGTH_SORTED_V1


def test_an_undeclared_packing_is_the_frozen_order():
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        resolve_packing)

    assert resolve_packing(None) == ORIGINAL_ORDER_V1
    assert resolve_packing({}) == ORIGINAL_ORDER_V1


@pytest.mark.parametrize("bad", ["length_sorted", "sorted", 4, None, True])
def test_an_unknown_packing_refuses_rather_than_defaulting(bad):
    from aadistill.initialization.operators.attention.gqa.causal_kl import (
        resolve_packing)

    with pytest.raises(OperatorError):
        resolve_packing({"calibration_batch_packing": bad})


def test_two_packings_are_two_states():
    """Same impl, same profile, same batch size — different identity."""
    from aadistill.infrastructure.manifest import sha256_json
    from aadistill.initialization.planning.fixed_path import FixedPathStep

    def step(packing):
        return FixedPathStep(
            impl_id="attention.causal_kl_v1", profile_id="p",
            config={"calibration_forward_batch_size": 2,
                    "calibration_batch_packing": packing})

    assert (sha256_json(step(ORIGINAL_ORDER_V1).as_dict())
            != sha256_json(step(LENGTH_SORTED_V1).as_dict()))


def test_micro_batches_default_is_unchanged():
    """The global default must not have moved: two runs of the same mixture
    have to batch identically, and everything else in the repository still
    calls `micro_batches`."""
    import inspect

    src = inspect.getsource(micro_batches)
    assert "never sorted by length" in src
    assert "sorted(" not in src.split('"""')[-1], (
        "micro_batches now sorts; that is a global behaviour change")

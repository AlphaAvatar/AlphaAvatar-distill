"""Micro-batched calibration forwards: same estimand, same decisions.

Three things are under test, and they are deliberately separated because they
fail for different reasons:

1. **the batcher** — padding, masks, order, and taking items back out again;
2. **the collectors and operators** — that ``batch_size > 1`` reaches the same
   *discrete* decision as the per-item reference path, and that padded positions
   reach no accumulator;
3. **the shape Phase C3 needs** — an intact forward once, then many ablated
   batched forwards, each reduced to one KL per original item.

Several tests here are **negative controls**: they reinstate the defect (pool
the padding, sort by length, drop the mask) and require the result to change.
A masking test that passes whether or not the mask is applied proves nothing,
and this repository has shipped exactly that kind of test before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.initialization.calibration.batching import (  # noqa: E402
    FALLBACK_PAD_ID,
    BatchingError,
    ItemBatch,
    build_batch,
    micro_batches,
    resolve_pad_id,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.attention.gqa._statistics import (  # noqa: E402
    AttentionHeadStatsCollector,
)
from aadistill.initialization.statistics.collect import (  # noqa: E402
    ActivationStatsCollector,
)
from aadistill.initialization.statistics.contribution import forward_kl_mean  # noqa: E402
from aadistill.initialization.operators.attention.gqa import activation_importance as attention_activation  # noqa: E402
from aadistill.initialization.operators._common import (  # noqa: E402
    collect_activation_stats,
)
from aadistill.initialization.operators.attention.gqa._common import head_rows  # noqa: E402
from aadistill.initialization.operators.attention.gqa._common import attention_out_projection  # noqa: E402
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.operators.depth.causal_kl_greedy import DEPTH_CAUSAL_KL_GREEDY_V1  # noqa: E402
from aadistill.initialization.operators.ffn.dense.activation_importance import FFN_ACTIVATION_IMPORTANCE_V0  # noqa: E402
from aadistill.initialization.operators.width.residual.global_pca import WIDTH_GLOBAL_PCA_V0  # noqa: E402
from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import NO_CALIBRATION  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from conftest import TEACHER_GEOMETRY, build_tiny_model  # noqa: E402

#: Deliberately RAGGED. Equal lengths would pad nothing, and every masking
#: defect in this refactor is invisible without padding.
LENGTHS = (23, 11, 17, 9, 20, 13)


def ragged_items(vocab: int = 128, seed: int = 404):
    torch.manual_seed(seed)
    items = []
    for index, length in enumerate(LENGTHS):
        domain, subtype = (("general", "text") if index % 2 == 0
                           else ("math", "arith"))
        items.append({
            "item_id": f"item-{index}",
            "input_ids": torch.randint(1, vocab, (1, length)),
            "domain": domain,
            "subtype": subtype,
        })
    return items


@pytest.fixture
def model():
    m = build_tiny_model(TEACHER_GEOMETRY)
    m.config.use_cache = False
    return m


@pytest.fixture
def items():
    return ragged_items()


@pytest.fixture(autouse=True)
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


def context(model, items, batch_size, target_spec=None):
    parent = ArchSpec.of("qwen3", TEACHER_GEOMETRY)
    execution = (ExecutionConfig() if batch_size is None
                 else ExecutionConfig(micro_batch_size=batch_size))
    return OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=parent,
        target_spec=target_spec if target_spec is not None else parent,
        profile=NO_CALIBRATION, calibration_items=items, seed=0, device="cpu",
        config={"n_calibration_items": len(items)}, execution=execution)


# --- 1. the batcher ---------------------------------------------------------


class TestTheBatcher:
    def test_padding_is_on_the_right_and_the_mask_marks_it(self, items):
        batch = build_batch(items[:3], pad_id=0)
        assert batch.input_ids.shape == (3, max(LENGTHS[:3]))
        for row, length in enumerate(batch.lengths):
            assert torch.equal(batch.input_ids[row, :length],
                               items[row]["input_ids"][0])
            assert bool(batch.attention_mask[row, :length].all())
            assert not bool(batch.attention_mask[row, length:].any())
            # Right padding, so every real token keeps the position index it
            # would occupy alone. Left padding would move all of them.
            assert bool((batch.input_ids[row, length:] == batch.pad_id).all())

    def test_split_predictions_returns_each_item_its_own_positions(self, items):
        batch = build_batch(items[:4], pad_id=0)
        fake = torch.arange(
            batch.size * (batch.input_ids.shape[1] - 1) * 2, dtype=torch.float32
        ).reshape(batch.size, batch.input_ids.shape[1] - 1, 2)
        parts = batch.split_predictions(fake)
        assert [p.shape[0] for p in parts] == [n - 1 for n in batch.lengths]
        for row, part in enumerate(parts):
            assert torch.equal(part, fake[row, :batch.lengths[row] - 1])

    def test_valid_tokens_drops_padding_and_keeps_order(self, items):
        batch = build_batch(items[:3], pad_id=0)
        width = batch.input_ids.shape[1]
        hidden = torch.arange(batch.size * width * 2,
                              dtype=torch.float32).reshape(batch.size, width, 2)
        kept = batch.valid_tokens(hidden)
        assert kept.shape[0] == batch.n_valid_tokens == sum(batch.lengths)
        expected = torch.cat([hidden[r, :n] for r, n in enumerate(batch.lengths)])
        assert torch.equal(kept, expected)

    def test_micro_batches_preserve_mixture_order_and_cover_every_item(self, items):
        seen = [it["item_id"]
                for b in micro_batches(items, 4, pad_id=0) for it in b.items]
        assert seen == [it["item_id"] for it in items]

    def test_a_ragged_batch_is_padded_and_a_uniform_one_is_not(self):
        same = [{"item_id": str(i), "input_ids": torch.ones(1, 5, dtype=torch.long)}
                for i in range(3)]
        assert build_batch(same, pad_id=0).is_padded is False
        assert build_batch(ragged_items()[:3], pad_id=0).is_padded is True

    def test_the_pad_id_comes_from_the_model_and_is_never_invented(self, model):
        model.config.pad_token_id = None
        model.config.eos_token_id = 7
        assert resolve_pad_id(model) == 7
        model.config.pad_token_id = 3
        assert resolve_pad_id(model) == 3
        # A declared id that does not fit the vocabulary is a real disagreement
        # and must be reported, never quietly replaced by the fallback.
        model.config.pad_token_id = 10**6
        with pytest.raises(BatchingError, match="vocabulary"):
            resolve_pad_id(model)
        # Nothing declared at all: the documented fallback, not a refusal.
        model.config.pad_token_id = None
        model.config.eos_token_id = None
        model.config.bos_token_id = None
        assert resolve_pad_id(model) == FALLBACK_PAD_ID

    def test_the_pad_id_is_numerically_inert(self, model, items):
        """The claim the fallback rests on, demonstrated rather than asserted.

        Right padding plus an `attention_mask` means a real token can only
        attend to real tokens, so the logits at real positions must not depend
        on which token sits in the pad slots. If this ever fails, the fallback
        is unsafe and so is every batched measurement.
        """
        rows = items[:4]
        out = []
        for pad in (0, 5, 99):
            batch = build_batch(rows, pad_id=pad)
            with torch.no_grad():
                logits = model(batch.input_ids,
                               attention_mask=batch.attention_mask).logits
            out.append([logits[r, :n] for r, n in enumerate(batch.lengths)])
        for row in range(len(rows)):
            for other in out[1:]:
                assert torch.equal(out[0][row], other[row]), (
                    f"item {row}: real-position logits changed with the pad id, "
                    "so padding is NOT inert and batching is unsound here")

    def test_the_real_positions_do_not_depend_on_how_much_padding_they_sit_beside(
            self, model, items):
        """The load-bearing property, pinned.

        An item's measurement must not depend on which other items happened to
        share its micro-batch. If this ever fails, every batched score becomes a
        function of the mixture's grouping rather than of the item.
        """
        alone = build_batch([items[1]], pad_id=0)
        with_a_longer_neighbour = build_batch([items[0], items[1]], pad_id=0)
        assert alone.input_ids.shape[1] < with_a_longer_neighbour.input_ids.shape[1]
        length = alone.lengths[0]
        with torch.no_grad():
            solo = model(alone.input_ids,
                         attention_mask=alone.attention_mask).logits[0, :length]
            padded = model(with_a_longer_neighbour.input_ids,
                           attention_mask=with_a_longer_neighbour.attention_mask
                           ).logits[1, :length]
        scale = solo.abs().max().clamp(min=1e-12)
        assert (solo - padded).abs().max() / scale < 1e-5

    def test_the_mask_is_defensive_under_right_padding_not_load_bearing(
            self, model, items):
        """Documents a measured fact, so nobody re-derives it under pressure.

        Under right padding the causal mask already prevents a real token from
        attending to a pad, so passing ``attention_mask`` changes the
        real-position logits by exactly zero. This is recorded as a test because
        the opposite belief is the natural one, and because it is the reason a
        mutation that deletes the mask from the batched DEPTH forward does not
        fail the suite — that is the code being redundant, not the suite being
        blind.

        It is emphatically **not** a licence to left-pad: there the mask cannot
        restore the position indices, and the RoPE phase of every real token
        moves.
        """
        batch = build_batch(items[:4], pad_id=0)
        with torch.no_grad():
            masked = model(batch.input_ids,
                           attention_mask=batch.attention_mask).logits
            unmasked = model(batch.input_ids).logits
        for row, length in enumerate(batch.lengths):
            assert torch.equal(masked[row, :length], unmasked[row, :length])

    def test_right_padding_matches_running_the_item_alone(self, model, items):
        """The other half of the same claim: a padded row equals a solo forward."""
        batch = build_batch(items[:4], pad_id=resolve_pad_id(model))
        with torch.no_grad():
            batched = model(batch.input_ids,
                            attention_mask=batch.attention_mask).logits
            for row, item in enumerate(items[:4]):
                solo = model(item["input_ids"]).logits[0]
                length = batch.lengths[row]
                scale = solo.abs().max().clamp(min=1e-12)
                drift = (batched[row, :length] - solo).abs().max() / scale
                assert drift < 1e-5, (
                    f"item {row}: padded row drifted {drift:.3e} from its own "
                    "forward")

    def test_an_empty_batch_and_a_wrong_shape_are_refused(self):
        with pytest.raises(BatchingError, match="empty batch"):
            build_batch([], pad_id=0)
        with pytest.raises(BatchingError, match=r"\[1, T\]"):
            build_batch([{"item_id": "x",
                          "input_ids": torch.ones(2, 4, dtype=torch.long)}],
                        pad_id=0)


# --- 2. collectors: padding must not reach an accumulator -------------------


class TestTheResidualAndFFNCollector:
    def _state(self, model, items, batch_size):
        return collect_activation_stats(QWEN3_ADAPTER, model, items, "cpu",
                                        batch_size=batch_size)

    def test_token_counts_are_real_tokens_only(self, model, items):
        expected = sum(LENGTHS)
        for size in (1, 2, 4, 64):
            state = self._state(model, items, size)
            assert int(state["residual_count"][0]) == expected
            assert int(state["token_counts"].sum()) == expected

    def test_batched_statistics_match_the_per_item_path(self, model, items):
        ref = self._state(model, items, 1)
        for size in (2, 4, 64):
            got = self._state(model, items, size)
            for key in ("residual_sum", "residual_sqsum", "ffn_abs_sum",
                        "ffn_sq_sum"):
                a, b = ref[key], got[key]
                scale = a.abs().max().clamp(min=1e-12)
                assert (a - b).abs().max() / scale < 1e-5, (
                    f"{key} drifted at batch_size={size}")
            # Exact, because these are counts rather than sums of floats.
            assert torch.equal(ref["token_counts"], got["token_counts"])
            assert torch.equal(ref["residual_count"], got["residual_count"])

    def test_a_single_item_batch_is_bit_identical_to_the_reference_path(
            self, model, items):
        """`batch_size=1` must not merely agree — it must be the same numbers.

        It is the path every frozen decision was produced by, so "close" is the
        wrong standard for it.
        """
        ref = self._state(model, items, 1)
        again = collect_activation_stats(
            QWEN3_ADAPTER, model, [i["input_ids"] for i in items], "cpu")
        for key in ref:
            assert torch.equal(ref[key], again[key]), key

    def test_pooling_the_padding_would_change_the_answer(self, model, items):
        """NEGATIVE CONTROL. Without the mask the statistics must differ.

        If this passes while the masking is removed, every equivalence test
        above is vacuous.
        """
        masked = self._state(model, items, 4)
        collector = ActivationStatsCollector(model)
        try:
            for batch in micro_batches(items, 4, pad_id=0):
                collector._accumulate(batch.input_ids, attention_mask=None)
        finally:
            collector.close()
        pooled = collector.state()
        assert int(pooled["residual_count"][0]) > int(masked["residual_count"][0])
        assert not torch.allclose(pooled["residual_sum"], masked["residual_sum"])


class TestTheAttentionCollector:
    def _state(self, model, items, batch_size):
        outs = [attention_out_projection(QWEN3_ADAPTER, b)
                for b in QWEN3_ADAPTER.blocks(model)]
        n_q, _, head_dim = QWEN3_ADAPTER.head_groups(
            ArchSpec.of("qwen3", TEACHER_GEOMETRY))
        c = AttentionHeadStatsCollector(model, outs, num_heads=n_q,
                                        head_dim=head_dim)
        try:
            if batch_size == 1:
                for item in items:
                    c.process(item["input_ids"])
            else:
                for batch in micro_batches(items, batch_size,
                                           pad_id=resolve_pad_id(model)):
                    c.process_batch(batch)
        finally:
            c.close()
        return c.state()

    def test_the_token_count_excludes_padding(self, model, items):
        for size in (1, 2, 4, 64):
            assert int(self._state(model, items, size)["attn_token_count"]) == sum(LENGTHS)

    def test_the_second_moment_matches_the_per_item_path(self, model, items):
        ref = self._state(model, items, 1)["attn_head_sqsum"]
        for size in (2, 4, 64):
            got = self._state(model, items, size)["attn_head_sqsum"]
            scale = ref.abs().max().clamp(min=1e-12)
            assert (ref - got).abs().max() / scale < 1e-5, f"batch_size={size}"

    def test_padding_would_inflate_the_second_moment(self, model, items):
        """NEGATIVE CONTROL for `_keep_valid` in the attention hook."""
        masked = self._state(model, items, 4)
        outs = [attention_out_projection(QWEN3_ADAPTER, b)
                for b in QWEN3_ADAPTER.blocks(model)]
        c = AttentionHeadStatsCollector(model, outs, num_heads=4, head_dim=8)
        try:
            for batch in micro_batches(items, 4, pad_id=0):
                c._run(batch.input_ids, attention_mask=None)
        finally:
            c.close()
        pooled = c.state()
        assert int(pooled["attn_token_count"]) > int(masked["attn_token_count"])
        assert not torch.allclose(pooled["attn_head_sqsum"],
                                  masked["attn_head_sqsum"])


# --- 3. operators: the discrete decision must not move ----------------------


class TestTheDiscreteDecisionsAreUnchanged:
    def test_attention_keeps_the_same_heads_in_every_layer(self, model, items):
        target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY,
                                       "num_attention_heads": 2})
        impl = attention_activation.ATTENTION_ACTIVATION_IMPORTANCE_V1
        ref = impl.apply(context(model, items, 1, target))
        for size in (2, 4, 64):
            got = impl.apply(context(model, items, size, target))
            assert (got.artifacts["kept_heads"] == ref.artifacts["kept_heads"]), (
                f"kept heads moved at batch_size={size}")
            assert got.trace["calibration_tokens"] == ref.trace["calibration_tokens"]

    def test_ffn_keeps_the_same_neurons(self, model, items):
        target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY,
                                       "intermediate_size": 24})
        ref = FFN_ACTIVATION_IMPORTANCE_V0.apply(context(model, items, 1, target))
        for size in (2, 4, 64):
            got = FFN_ACTIVATION_IMPORTANCE_V0.apply(
                context(model, items, size, target))
            assert got.artifacts["kept_neurons"] == ref.artifacts["kept_neurons"], (
                f"kept neurons moved at batch_size={size}")

    def test_width_projects_onto_the_same_subspace(self, model, items):
        """WIDTH has no discrete selection, so the invariant is its subspace.

        Compared through `P P^T`, which is invariant to the eigenvector sign and
        to any rotation *within* a degenerate eigenspace — the two things an
        eigendecomposition is entitled to change under a 1e-16 perturbation of
        its input, and neither of which is a scientific difference. The captured
        energy is compared beside it because it is basis-independent and is what
        the operator actually reports.
        """
        target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY, "hidden_size": 16})
        ref = WIDTH_GLOBAL_PCA_V0.apply(context(model, items, 1, target))
        ref_embed = QWEN3_ADAPTER.embedding(ref.model).weight.double()
        for size in (2, 4):
            got = WIDTH_GLOBAL_PCA_V0.apply(context(model, items, size, target))
            for key in ("op.width.energy_captured_frac",
                        "op.width.orthonormality_error",
                        "op.width.min_kept_eigenvalue"):
                a = ref.local_metrics.values[key]
                b = got.local_metrics.values[key]
                assert abs(a - b) <= 1e-9 + 1e-6 * abs(a), (
                    f"{key} moved at batch_size={size}: {a!r} vs {b!r}")
            # E @ P for both, so `Pr = E^+ ...` is not needed: comparing the
            # projected embeddings' Gram matrices compares the subspace the
            # stream was projected onto.
            got_embed = QWEN3_ADAPTER.embedding(got.model).weight.double()
            gram_ref = ref_embed.T @ ref_embed
            gram_got = got_embed.T @ got_embed
            scale = gram_ref.abs().max().clamp(min=1e-12)
            drift = (gram_ref - gram_got).abs().max() / scale
            assert drift < 1e-6, (
                f"the projected subspace moved at batch_size={size}: {drift:.3e}")

    def test_depth_removes_the_same_layers_in_the_same_order(self, model, items):
        target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY,
                                       "num_hidden_layers": 4})
        ref = DEPTH_CAUSAL_KL_GREEDY_V1.apply(context(model, items, 1, target))
        for size in (2, 3, 64):
            got = DEPTH_CAUSAL_KL_GREEDY_V1.apply(
                context(model, items, size, target))
            assert got.trace["removal_order"] == ref.trace["removal_order"], (
                f"DEPTH removal ORDER moved at batch_size={size}: "
                f"{got.trace['removal_order']} vs {ref.trace['removal_order']}")
            assert got.trace["kept_layers"] == ref.trace["kept_layers"]

    def test_depth_scores_agree_far_inside_the_decision_margin(self, model, items):
        """The number, not just the decision.

        A matching removal order could still hide a score that is drifting
        toward a boundary, so compare the scores directly and require the drift
        to be far smaller than the margin that separated the winner from the
        runner-up.
        """
        target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY,
                                       "num_hidden_layers": 4})
        ref = DEPTH_CAUSAL_KL_GREEDY_V1.apply(context(model, items, 1, target))
        got = DEPTH_CAUSAL_KL_GREEDY_V1.apply(context(model, items, 4, target))
        for a, b in zip(ref.artifacts["search_rounds"], got.artifacts["search_rounds"]):
            table_a = {r["candidate"]: r["score"] for r in a["table"]}
            table_b = {r["candidate"]: r["score"] for r in b["table"]}
            assert table_a.keys() == table_b.keys()
            drift = max(abs(table_a[k] - table_b[k]) for k in table_a)
            ranked = sorted(table_a.values())
            margin = ranked[1] - ranked[0] if len(ranked) > 1 else float("inf")
            assert drift < margin, (
                f"round {a['round']}: batching drift {drift:.3e} is not smaller "
                f"than the decision margin {margin:.3e}")


class TestTheReferenceCacheStaysWithinItsOwnByteBudget:
    def test_a_cached_reference_does_not_pin_the_padded_batch(self, model, items):
        """`_ReferenceLogits` sizes itself at `(T_i - 1) * V` per admitted item.

        `split_predictions` hands back views into the batch's `[B, T_max, V]`
        logits, so caching one unchanged would keep the whole padded block —
        every other item's logits and all the padding — resident for the
        lifetime of the search, while the admission budget went on describing
        the smaller number. That is the same shape as the 33.8 GiB overrun this
        cache was repaired for, so it is asserted rather than trusted.
        """
        from aadistill.initialization.operators.depth.causal_kl_greedy import _ReferenceLogits

        ref = _ReferenceLogits(model, items, "cpu")
        batch = build_batch(items[:4], pad_id=resolve_pad_id(model))
        got = ref.get_batch(batch)
        assert len(got) == 4
        for item in batch.items:
            cached = ref._cache.get(item["item_id"])
            if cached is None:                   # not admitted; nothing to pin
                continue
            assert cached._base is None, (
                f"{item['item_id']}: the cached reference is a VIEW into the "
                "padded batch, so it pins T_max*V rather than its own T_i*V")
            assert cached.shape[0] == int(item["input_ids"].shape[1]) - 1

    def test_a_fully_cached_batch_runs_no_forward(self, model, items):
        from aadistill.initialization.operators.depth import causal_kl_greedy as depth_module

        ref = depth_module._ReferenceLogits(model, items, "cpu")
        batch = build_batch(items[:4], pad_id=resolve_pad_id(model))
        first = ref.get_batch(batch)
        calls = []
        real = depth_module._forward_logits_batch

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        depth_module._forward_logits_batch = counting
        try:
            again = ref.get_batch(batch)
        finally:
            depth_module._forward_logits_batch = real
        assert not calls, "a fully cached batch performed a forward anyway"
        for a, b in zip(first, again):
            assert torch.equal(a, b)


# --- 4. the shape Phase C3 needs -------------------------------------------


def test_the_batching_api_supports_one_reference_and_many_head_interventions(
        model, items):
    """C3's execution shape, on the generic machinery only.

    intact forward once -> for each (layer, head): batched ablated forward ->
    one KL per ORIGINAL item. No C3 operator exists yet and none is implemented
    here; what is being proved is that the batching API can carry that shape —
    in particular that an intervention can be applied once and amortised over a
    whole micro-batch, rather than forcing the causal operator back to one item
    per forward.
    """
    from aadistill.initialization.operators.depth.causal_kl_greedy import _forward_logits_batch

    _, _, head_dim = QWEN3_ADAPTER.head_groups(
        ArchSpec.of("qwen3", TEACHER_GEOMETRY))
    n_heads = TEACHER_GEOMETRY["num_attention_heads"]
    batches = list(micro_batches(items, 4, pad_id=resolve_pad_id(model)))

    # ONE intact forward per micro-batch, reused by every intervention.
    reference = {id(b): _forward_logits_batch(model, b, "cpu") for b in batches}

    scores = {}
    interventions = [(layer, head)
                     for layer in range(TEACHER_GEOMETRY["num_hidden_layers"])
                     for head in range(n_heads)]
    for layer, head in interventions:
        proj = attention_out_projection(QWEN3_ADAPTER,
                                        QWEN3_ADAPTER.blocks(model)[layer])
        cols = head_rows([head], head_dim, device=proj.weight.device)
        saved = proj.weight[:, cols].clone()
        try:
            with torch.no_grad():
                proj.weight[:, cols] = 0
            per_item = []
            for batch in batches:
                ablated = _forward_logits_batch(model, batch, "cpu")
                per_item.extend(
                    forward_kl_mean(r, a, chunk=512)
                    for r, a in zip(reference[id(batch)], ablated))
        finally:
            with torch.no_grad():
                proj.weight[:, cols] = saved
        assert len(per_item) == len(items), (
            "a head intervention must yield exactly one KL per original item")
        scores[(layer, head)] = sum(per_item) / len(per_item)

    assert len(scores) == len(interventions)
    assert all(v >= 0.0 for v in scores.values()), "forward KL cannot be negative"
    # The interventions must be distinguishable, or the operator they are for
    # could not rank anything.
    assert len(set(round(v, 12) for v in scores.values())) > 1

    # The temporary mutation is undone exactly — a scoring pass must leave the
    # parent it measured untouched.
    for layer in range(TEACHER_GEOMETRY["num_hidden_layers"]):
        proj = attention_out_projection(QWEN3_ADAPTER,
                                        QWEN3_ADAPTER.blocks(model)[layer])
        assert not bool((proj.weight == 0).all(dim=0).any()), (
            f"layer {layer}: a head's o_proj columns were left zeroed")


def test_every_batcher_refusal_raises_batchingerror_not_nameerror():
    """A regression test for a real defect, not a formality.

    `BatchingError` was briefly deleted along with this module's configuration
    machinery while the batch size moved to `OperatorContext.execution`. Every
    refusal path still NAMED it, so each one raised `NameError: name
    'BatchingError' is not defined` instead of the refusal it was written to
    give — on exactly the paths that only run when something is already wrong,
    which is why no happy-path test noticed.

    So the error TYPE is asserted, on each refusal, rather than just "it
    raised".
    """
    good = {"item_id": "a", "input_ids": torch.ones(1, 4, dtype=torch.long)}
    cases = [
        lambda: build_batch([], pad_id=0),
        lambda: build_batch([{"item_id": "x", "input_ids": [1, 2]}], pad_id=0),
        lambda: build_batch(
            [{"item_id": "x", "input_ids": torch.ones(2, 4, dtype=torch.long)}],
            pad_id=0),
        lambda: list(micro_batches([good], 0, pad_id=0)),
        lambda: build_batch([good], pad_id=0).split_predictions(
            torch.zeros(9, 3, 2)),
        lambda: build_batch([good], pad_id=0).valid_tokens(torch.zeros(9, 3, 2)),
    ]
    for index, case in enumerate(cases):
        with pytest.raises(BatchingError):
            case()


def test_composite_records_whether_it_collected_statistics_or_was_handed_them(
        model, items):
    """A batch size is evidence that a statistics pass RAN.

    `composite.stage1_sandwich_v0` accepts a pre-computed `activation_state`.
    When it does, this invocation collected nothing, so reporting its configured
    micro-batch size would describe work it did not do — and a reader comparing
    traces would see two runs that look identically batched when only one of
    them measured anything.
    """
    from aadistill.initialization.operators.composite.stage1_sandwich import (
        COMPOSITE_STAGE1_SANDWICH_V0 as COMPOSITE)
    from aadistill.initialization.operators._common import collect_activation_stats

    target = ArchSpec.of("qwen3", {**TEACHER_GEOMETRY, "hidden_size": 16,
                                   "num_hidden_layers": 4,
                                   "intermediate_size": 24,
                                   "num_attention_heads": 2})

    collected = COMPOSITE.apply(context(model, items, 4, target))
    assert collected.trace["activation_stats"] == "collected_here"
    assert collected.trace["micro_batch_size"] == 4

    state = collect_activation_stats(QWEN3_ADAPTER, model, items, "cpu",
                                     batch_size=1)
    ctx = context(model, items, 4, target)
    ctx.config = {**ctx.config, "activation_state": state}
    supplied = COMPOSITE.apply(ctx)
    assert supplied.trace["activation_stats"] == "supplied_by_caller"
    assert supplied.trace["micro_batch_size"] is None, (
        "a supplied state was reported with this invocation's batch size, which "
        "claims a statistics pass that never ran")

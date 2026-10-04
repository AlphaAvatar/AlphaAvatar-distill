"""The two distribution supports, and that adopting the second cannot move the
identity of a measurement taken under the first.

Generic core behaviour: no `K` from any experiment's policy appears here except
as a test input, and the vocabulary is always taken from the tensors.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.scoring.protocol_identity import (  # noqa: E402
    ReductionSemantics, measurement_protocol_id,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, SUPPORT_FULL_VOCAB_V1, SUPPORT_REFERENCE_TOPK_TAIL_V1,
    DistributionSupport, ReferenceDistributionSketch,
    distortion_on_reference_support, log1mexp, reference_topk_tail,
    sketch_reference,
)
from aadistill.initialization.statistics.contribution import distortion  # noqa: E402


def _pair(T=48, V=400, seed=0, spread=2.0, noise=0.5):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(T, V, generator=g) * spread
    cand = ref + torch.randn(T, V, generator=g) * noise
    targets = torch.randint(0, V, (T,), generator=g)
    return ref, cand, targets


class TestTheSupportIsAnIdentityBearingValue:

    def test_full_vocab_takes_no_k(self):
        with pytest.raises(ValueError, match="takes\n?\\s*no top_k"):
            DistributionSupport(SUPPORT_FULL_VOCAB_V1, top_k=200)

    def test_topk_requires_an_explicit_integer_k(self):
        """There is no framework default for K; it is experiment policy."""
        with pytest.raises(ValueError, match="needs an integer top_k"):
            DistributionSupport(SUPPORT_REFERENCE_TOPK_TAIL_V1)
        with pytest.raises(ValueError, match="needs an integer top_k"):
            DistributionSupport(SUPPORT_REFERENCE_TOPK_TAIL_V1, top_k=True)
        with pytest.raises(ValueError, match="at least 1"):
            DistributionSupport(SUPPORT_REFERENCE_TOPK_TAIL_V1, top_k=0)

    def test_an_unknown_support_is_refused(self):
        with pytest.raises(ValueError, match="unknown distribution support"):
            DistributionSupport("candidate_topk_v1", top_k=200)

    def test_core_names_no_experiment_k(self):
        """`200` belongs to the D-series policy, not to this module."""
        from pathlib import Path

        import aadistill.initialization.scoring.support as mod

        source = Path(mod.__file__).read_text()
        #: Only inside the prose that explains it is NOT a default.
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith(("#", "*", '"', "'"))
                         and "``" not in line)
        assert "200" not in code, (
            "a core module must not carry an experiment's K")


class TestHistoricalIdentityDoesNotMove:
    """785 committed records cite ids computed before this field existed."""

    TERMS = dict(suite_structural_identity="struct",
                 suite_content_identity="content",
                 scoring_content_identity="scoring",
                 position_policy_hash="policy",
                 execution_fingerprint="fingerprint")

    def test_the_historical_serialization_is_byte_identical(self):
        before = {"chunk": 512, "reference_strategy": "RECOMPUTE",
                  "aggregation_rule": "equal_subtype_then_equal_domain/v1"}
        historical = ReductionSemantics(chunk=512,
                                       reference_strategy="RECOMPUTE")
        assert historical.as_dict() == before
        #: And the hash over it, which is what a record actually cites.
        assert sha256_json(historical.as_dict()) == sha256_json(before)

    def test_declaring_full_vocab_explicitly_changes_nothing(self):
        """A caller may be explicit without moving any id."""
        absent = ReductionSemantics(chunk=512, reference_strategy="RECOMPUTE")
        explicit = ReductionSemantics(chunk=512, reference_strategy="RECOMPUTE",
                                      distribution_support=FULL_VOCAB_V1)
        assert absent.as_dict() == explicit.as_dict()
        assert (measurement_protocol_id(reduction=absent, **self.TERMS)
                == measurement_protocol_id(reduction=explicit, **self.TERMS))

    def test_the_new_contract_does_move_the_id(self):
        """Top-K+tail is a different measurement and must not compare equal."""
        absent = ReductionSemantics(chunk=512, reference_strategy="RECOMPUTE")
        topk = ReductionSemantics(chunk=512, reference_strategy="RECOMPUTE",
                                  distribution_support=reference_topk_tail(200))
        assert (measurement_protocol_id(reduction=absent, **self.TERMS)
                != measurement_protocol_id(reduction=topk, **self.TERMS))

    def test_k_is_part_of_the_identity(self):
        ids = {k: measurement_protocol_id(
            reduction=ReductionSemantics(
                chunk=512, reference_strategy="RECOMPUTE",
                distribution_support=reference_topk_tail(k)), **self.TERMS)
            for k in (50, 100, 200)}
        assert len(set(ids.values())) == 3


class TestLog1mexp:

    def test_it_matches_the_naive_form_where_the_naive_form_is_accurate(self):
        x = torch.tensor([-0.1, -0.5, -1.0, -5.0, -20.0], dtype=torch.float64)
        naive = torch.log1p(-torch.exp(x))
        assert torch.allclose(log1mexp(x), naive, atol=1e-12)

    def test_it_survives_both_regimes_without_nan(self):
        x = torch.tensor([-1e-12, -1e-8, -0.69, -0.70, -1e3, -1e30],
                         dtype=torch.float64)
        got = log1mexp(x)
        assert torch.isfinite(got).all(), got

    def test_at_zero_the_tail_is_empty(self):
        got = log1mexp(torch.zeros(3, dtype=torch.float64))
        assert torch.isinf(got).all() and (got < 0).all()

    def test_near_zero_does_not_cancel(self):
        """`log(1 - exp(-1e-10))` is about `log(1e-10)`, not `-inf`."""
        got = float(log1mexp(torch.tensor([-1e-10], dtype=torch.float64))[0])
        assert math.isclose(got, math.log(1e-10), rel_tol=1e-6), got


class TestTheSketchHoldsOnlyWhatIsNeeded:

    def test_its_shapes_and_bytes_scale_with_k_not_v(self):
        ref, _, targets = _pair()
        small = sketch_reference(ref, targets, top_k=10, chunk=16)
        large = sketch_reference(ref, targets, top_k=100, chunk=16)
        assert small.support_indices.shape == (48, 10)
        assert small.tail_log_prob.shape == (48,)
        #: ~10x the support, so ~10x the support storage. Not exactly 10x
        #: because the per-position vectors do not scale with K.
        assert large.bytes_held() > 8 * small.bytes_held()
        #: And both are far below a full `[T, V]` float32 block.
        assert large.bytes_held() < 48 * 400 * 4

    def test_the_support_is_sorted_and_the_top1_is_the_argmax(self):
        ref, _, targets = _pair()
        sk = sketch_reference(ref, targets, top_k=16, chunk=16)
        assert torch.equal(sk.top1_token, ref.argmax(dim=-1))
        lp = sk.support_log_probs
        assert (lp[:, :-1] >= lp[:, 1:] - 1e-6).all(), "support not descending"

    def test_the_mass_on_support_plus_tail_is_one(self):
        ref, _, targets = _pair()
        sk = sketch_reference(ref, targets, top_k=37, chunk=16)
        total = sk.support_log_probs.exp().sum(dim=-1) + sk.tail_log_prob.exp()
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_the_target_log_prob_is_exact_even_outside_the_support(self):
        ref, _, targets = _pair()
        exact = torch.log_softmax(ref.float(), dim=-1).gather(
            1, targets.view(-1, 1)).squeeze(1)
        sk = sketch_reference(ref, targets, top_k=2, chunk=16)
        assert torch.allclose(sk.target_log_prob, exact, atol=1e-5)
        #: And with K=2 the gold token is usually NOT in the support, which is
        #: the case that makes this worth asserting.
        inside = (sk.support_indices == targets.view(-1, 1)).any(dim=-1)
        assert not inside.all(), "pick a K where some gold token falls outside"

    def test_it_refuses_shapes_that_describe_nothing(self):
        ref, _, targets = _pair()
        with pytest.raises(ValueError, match=r"must be \[T, V\]"):
            sketch_reference(ref[0], targets, top_k=4)
        with pytest.raises(ValueError, match="position count"):
            sketch_reference(ref, targets[:-1], top_k=4)
        with pytest.raises(ValueError, match="zero positions"):
            sketch_reference(ref[:0], targets[:0], top_k=4)

    def test_a_malformed_sketch_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="top_k"):
            ReferenceDistributionSketch(
                support_indices=torch.zeros(4, 3, dtype=torch.long),
                support_log_probs=torch.zeros(4, 3),
                tail_log_prob=torch.zeros(4), top1_token=torch.zeros(4, dtype=torch.long),
                target_log_prob=torch.zeros(4), top_k=5, vocab_size=10)
        with pytest.raises(ValueError, match="tail_log_prob"):
            ReferenceDistributionSketch(
                support_indices=torch.zeros(4, 3, dtype=torch.long),
                support_log_probs=torch.zeros(4, 3),
                tail_log_prob=torch.zeros(3), top1_token=torch.zeros(4, dtype=torch.long),
                target_log_prob=torch.zeros(4), top_k=3, vocab_size=10)


class TestTheReduction:

    def test_identical_distributions_give_zero_divergence(self):
        ref, _, targets = _pair()
        sk = sketch_reference(ref, targets, top_k=50, chunk=16)
        out = distortion_on_reference_support(sk, ref, targets, chunk=16)
        assert out.kl == pytest.approx(0.0, abs=1e-6)
        assert out.reverse_kl == pytest.approx(0.0, abs=1e-6)
        assert out.top1_agree == pytest.approx(float(ref.shape[0]))

    def test_k_at_least_v_agrees_with_the_full_vocabulary_oracle(self):
        """The coarsening is the identity when the support IS the vocabulary."""
        ref, cand, targets = _pair()
        V = int(ref.shape[1])
        full = distortion(ref, cand, targets, chunk=16)
        for k in (V, V + 1, V * 2):
            sk = sketch_reference(ref, targets, top_k=k, chunk=16)
            got = distortion_on_reference_support(sk, cand, targets, chunk=16)
            assert got.kl == pytest.approx(full.kl, rel=1e-5), k
            assert got.reverse_kl == pytest.approx(full.reverse_kl, rel=1e-5), k
            assert got.ref_ce == pytest.approx(full.ref_ce, rel=1e-5)
            assert got.abl_ce == pytest.approx(full.abl_ce, rel=1e-5)
            assert got.top1_agree == pytest.approx(full.top1_agree)

    def test_when_the_support_is_the_vocabulary_the_tail_is_ignored_entirely(self):
        """`top_k >= vocab_size` means there is no tail bucket at all.

        Checked by CORRUPTING the sketch's tail and requiring the answer not to
        move. A tolerance-based test cannot see this: at `K == V` the two tails
        both round to about 1e-8, and the spurious term they produce is ~1e-9 on
        a KL of ~7 — which a `rel=1e-5` assertion swallows. Mutating the guard
        away left all thirty tests passing, so this is the detector it needed.
        """
        import dataclasses

        ref, cand, targets = _pair(T=16, V=60)
        V = int(ref.shape[1])
        sk = sketch_reference(ref, targets, top_k=V, chunk=8)
        honest = distortion_on_reference_support(sk, cand, targets, chunk=8)

        corrupted = dataclasses.replace(
            sk, tail_log_prob=torch.full_like(sk.tail_log_prob, -0.5))
        got = distortion_on_reference_support(corrupted, cand, targets, chunk=8)

        assert got.kl == honest.kl, (
            "a sketch whose support IS the vocabulary has no tail bucket, so a "
            "tail value must be unreachable; it changed the answer")
        assert got.reverse_kl == honest.reverse_kl

    def test_the_coarse_kl_never_exceeds_the_full_kl(self):
        """Data processing: coarsening a partition cannot increase KL."""
        ref, cand, targets = _pair()
        full = distortion(ref, cand, targets, chunk=16)
        previous = -1.0
        for k in (1, 2, 8, 64, 256, int(ref.shape[1])):
            sk = sketch_reference(ref, targets, top_k=k, chunk=16)
            got = distortion_on_reference_support(sk, cand, targets, chunk=16)
            assert got.kl <= full.kl + 1e-5, (k, got.kl, full.kl)
            assert got.kl >= previous - 1e-9, "refining K should not lose KL"
            previous = got.kl

    def test_ce_and_top1_stay_exact_at_every_k(self):
        """They do not degrade with the KL partition."""
        ref, cand, targets = _pair()
        full = distortion(ref, cand, targets, chunk=16)
        for k in (1, 4, 64):
            sk = sketch_reference(ref, targets, top_k=k, chunk=16)
            got = distortion_on_reference_support(sk, cand, targets, chunk=16)
            assert got.ref_ce == pytest.approx(full.ref_ce, rel=1e-5), k
            assert got.abl_ce == pytest.approx(full.abl_ce, rel=1e-5), k
            assert got.top1_agree == pytest.approx(full.top1_agree), k

    def test_the_candidate_never_chooses_the_support(self):
        """The gathered entries are the SKETCH's, not the candidate's top-k.

        The semantic commitment, checked against two hand-built oracles: one that
        reduces on the reference support, which must match, and one that reduces
        on the CANDIDATE's own top-k, which must not. The first version of this
        test swapped the two arguments instead, which swaps the support AND the
        KL direction at once — and with a candidate that is the reference plus
        small noise the two forward KLs coincide to 0.06%, so it was asserting
        the absence of a difference it had cancelled out.
        """
        #: Noise large enough that the two top-k sets genuinely differ.
        ref, cand, targets = _pair(T=16, V=70, spread=1.0, noise=4.0)
        k = 6
        sk = sketch_reference(ref, targets, top_k=k, chunk=8)
        got = distortion_on_reference_support(sk, cand, targets, chunk=8)

        p_all = torch.log_softmax(ref.float(), dim=-1)
        q_all = torch.log_softmax(cand.float(), dim=-1)

        def oracle(support_idx):
            p = p_all.gather(1, support_idx).exp()
            q = q_all.gather(1, support_idx).exp()
            p_t = (1.0 - p.sum(-1)).clamp(min=0.0)
            q_t = (1.0 - q.sum(-1)).clamp(min=0.0)
            pf = torch.cat([p, p_t.view(-1, 1)], -1)
            qf = torch.cat([q, q_t.view(-1, 1)], -1)
            return float((pf * (pf.clamp_min(1e-30).log()
                                - qf.clamp_min(1e-30).log())).sum())

        ref_support = p_all.topk(k, dim=-1).indices
        cand_support = q_all.topk(k, dim=-1).indices
        #: The two supports really are different sets, or this proves nothing.
        assert not torch.equal(ref_support, cand_support)

        assert got.kl == pytest.approx(oracle(ref_support), rel=1e-3), (
            "the reduction did not use the reference support")
        assert got.kl != pytest.approx(oracle(cand_support), rel=1e-2), (
            "the reduction agrees with a CANDIDATE-defined support, which is "
            "the one support policy this protocol must not have")

    def test_both_directions_use_one_partition(self):
        """Reverse KL on the reference support, not on the candidate's.

        Computed here against a hand-built `K+1` oracle that uses the REFERENCE
        support for both directions.
        """
        ref, cand, targets = _pair(T=12, V=60)
        k = 7
        sk = sketch_reference(ref, targets, top_k=k, chunk=4)
        got = distortion_on_reference_support(sk, cand, targets, chunk=4)

        p_all = torch.log_softmax(ref.float(), dim=-1)
        q_all = torch.log_softmax(cand.float(), dim=-1)
        idx = p_all.topk(k, dim=-1).indices
        p = p_all.gather(1, idx).exp()
        q = q_all.gather(1, idx).exp()
        p_tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0)
        q_tail = (1.0 - q.sum(dim=-1)).clamp(min=0.0)
        p_full = torch.cat([p, p_tail.view(-1, 1)], dim=-1)
        q_full = torch.cat([q, q_tail.view(-1, 1)], dim=-1)
        fwd = (p_full * (p_full.clamp_min(1e-30).log()
                         - q_full.clamp_min(1e-30).log())).sum()
        rev = (q_full * (q_full.clamp_min(1e-30).log()
                         - p_full.clamp_min(1e-30).log())).sum()
        assert got.kl == pytest.approx(float(fwd), rel=1e-3)
        assert got.reverse_kl == pytest.approx(float(rev), rel=1e-3)

    def test_weights_and_tags_keep_their_existing_semantics(self):
        ref, cand, targets = _pair(T=20, V=80)
        sk = sketch_reference(ref, targets, top_k=9, chunk=8)
        tag = torch.zeros(20, dtype=torch.bool)
        tag[:7] = True
        w = torch.rand(20).double().float()

        unweighted = distortion_on_reference_support(
            sk, cand, targets, tags={"t": tag}, chunk=8)
        weighted = distortion_on_reference_support(
            sk, cand, targets, tags={"t": tag}, weights=w, chunk=8)

        assert unweighted.weight == pytest.approx(20.0)
        assert weighted.weight == pytest.approx(float(w.sum()), rel=1e-6)
        #: A 0/1 weight equals subsetting the positions exactly.
        binary = torch.zeros(20)
        binary[:7] = 1.0
        subset = distortion_on_reference_support(
            sk, cand, targets, weights=binary, chunk=8)
        head = distortion_on_reference_support(
            sketch_reference(ref[:7], targets[:7], top_k=9, chunk=8),
            cand[:7], targets[:7], chunk=8)
        assert subset.kl == pytest.approx(head.kl, rel=1e-5)
        #: The tag counts positions as well as weight, which a weighted tagged
        #: mean needs separately.
        assert unweighted.tagged["t"][2] == pytest.approx(7.0)
        assert weighted.tagged["t"][1] == pytest.approx(
            float(w[:7].sum()), rel=1e-6)

    def test_the_scalar_and_chunked_reductions_agree(self):
        ref, cand, targets = _pair(T=33, V=90)
        sk = sketch_reference(ref, targets, top_k=11, chunk=33)
        one = distortion_on_reference_support(sk, cand, targets, chunk=1)
        many = distortion_on_reference_support(sk, cand, targets, chunk=7)
        whole = distortion_on_reference_support(sk, cand, targets, chunk=1000)
        assert one.kl == pytest.approx(whole.kl, rel=1e-6)
        assert many.kl == pytest.approx(whole.kl, rel=1e-6)

    def test_a_chunked_sketch_equals_an_unchunked_one(self):
        ref, cand, targets = _pair(T=29, V=70)
        a = sketch_reference(ref, targets, top_k=6, chunk=1)
        b = sketch_reference(ref, targets, top_k=6, chunk=1000)
        assert torch.equal(a.support_indices, b.support_indices)
        assert torch.allclose(a.support_log_probs, b.support_log_probs, atol=1e-6)
        assert torch.allclose(a.tail_log_prob, b.tail_log_prob, atol=1e-5)

    def test_mismatched_candidates_are_refused(self):
        ref, cand, targets = _pair(T=10, V=50)
        sk = sketch_reference(ref, targets, top_k=5, chunk=4)
        with pytest.raises(ValueError, match="same positions"):
            distortion_on_reference_support(sk, cand[:-1], targets, chunk=4)
        with pytest.raises(ValueError, match="logit-comparable"):
            distortion_on_reference_support(sk, cand[:, :-1], targets, chunk=4)
        with pytest.raises(ValueError, match=r"must be \[T, V\]"):
            distortion_on_reference_support(sk, cand[0], targets, chunk=4)

    def test_an_extremely_peaked_reference_does_not_produce_nan(self):
        """The tail underflows; the term must vanish rather than become NaN."""
        T, V = 8, 120
        ref = torch.full((T, V), -40.0)
        ref[:, 0] = 60.0
        cand = torch.randn(T, V)
        targets = torch.zeros(T, dtype=torch.long)
        sk = sketch_reference(ref, targets, top_k=2, chunk=4)
        out = distortion_on_reference_support(sk, cand, targets, chunk=4)
        assert math.isfinite(out.kl), out.kl
        assert math.isfinite(out.reverse_kl), out.reverse_kl

    def test_an_extremely_peaked_candidate_does_not_produce_nan(self):
        T, V = 8, 120
        ref = torch.randn(T, V)
        cand = torch.full((T, V), -40.0)
        cand[:, 0] = 60.0
        targets = torch.zeros(T, dtype=torch.long)
        sk = sketch_reference(ref, targets, top_k=4, chunk=4)
        out = distortion_on_reference_support(sk, cand, targets, chunk=4)
        assert not math.isnan(out.kl), out.kl
        assert not math.isnan(out.reverse_kl), out.reverse_kl


class TestTheDepthReducers:
    """DEPTH reads forward KL only, so it has its own thinner reducers.

    They must agree with the full-vocabulary ones they stand in for when the
    support is the vocabulary, and the per-item and batched forms must agree with
    each other — they share one implementation precisely so they cannot drift.
    """

    def test_the_per_item_mean_matches_the_full_vocab_reducer_at_k_ge_v(self):
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean,
        )
        from aadistill.initialization.statistics.contribution import (
            forward_kl_mean,
        )

        ref, cand, targets = _pair(T=40, V=300, spread=1.5, noise=0.4)
        sk = sketch_reference(ref, targets, top_k=int(ref.shape[1]), chunk=8)
        assert sketch_forward_kl_mean(sk, cand) == pytest.approx(
            forward_kl_mean(ref, cand, chunk=8), rel=1e-4)
        w = torch.rand(40)
        assert sketch_forward_kl_mean(sk, cand, weights=w) == pytest.approx(
            forward_kl_mean(ref, cand, weights=w, chunk=8), rel=1e-4)

    def test_the_batched_mean_matches_the_full_vocab_reducer_at_k_ge_v(self):
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean_batch,
        )
        from aadistill.initialization.statistics.contribution import (
            forward_kl_mean_batch,
        )

        g = torch.Generator().manual_seed(5)
        B, T, V = 3, 20, 120
        rb = torch.randn(B, T, V, generator=g) * 1.5
        cb = rb + torch.randn(B, T, V, generator=g) * 0.4
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :20] = mask[1, :12] = mask[2, :7] = True

        flat = rb.reshape(-1, V)
        sb = sketch_reference(flat, torch.zeros(B * T, dtype=torch.long),
                              top_k=V, chunk=64)
        got = sketch_forward_kl_mean_batch(
            sb.support_indices.reshape(B, T, -1),
            sb.support_log_probs.reshape(B, T, -1),
            sb.tail_log_prob.reshape(B, T), cb, mask, has_tail=False)
        want = forward_kl_mean_batch(rb, cb, mask, chunk=8)
        assert got.dtype == want.dtype, (
            'the stand-in must return what its caller consumes')
        assert torch.allclose(got, want, rtol=1e-4)

    def test_the_batched_mean_is_per_item_not_pooled(self):
        """Rows of different lengths must get different means."""
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean_batch,
        )

        g = torch.Generator().manual_seed(7)
        B, T, V = 3, 24, 90
        rb = torch.randn(B, T, V, generator=g)
        cb = rb + torch.randn(B, T, V, generator=g) * 0.6
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :24] = mask[1, :10] = mask[2, :3] = True
        flat = rb.reshape(-1, V)
        sb = sketch_reference(flat, torch.zeros(B * T, dtype=torch.long),
                              top_k=9, chunk=64)
        got = sketch_forward_kl_mean_batch(
            sb.support_indices.reshape(B, T, -1),
            sb.support_log_probs.reshape(B, T, -1),
            sb.tail_log_prob.reshape(B, T), cb, mask, has_tail=True)
        #: A pooled reduction would give every row the same number.
        assert len({round(float(v), 5) for v in got}) == B
        #: And a row's mean must not depend on its neighbours. Score row 2 alone
        #: and require the same answer.
        alone = sketch_forward_kl_mean_batch(
            sb.support_indices.reshape(B, T, -1)[2:3],
            sb.support_log_probs.reshape(B, T, -1)[2:3],
            sb.tail_log_prob.reshape(B, T)[2:3], cb[2:3], mask[2:3],
            has_tail=True)
        assert float(alone[0]) == pytest.approx(float(got[2]), rel=1e-6)

    def test_has_tail_is_the_callers_and_changes_the_answer(self):
        """A sketch carries a tail value whether or not the partition has one."""
        from aadistill.initialization.scoring.support import sketch_forward_kl

        ref, cand, targets = _pair(T=8, V=50)
        sk = sketch_reference(ref, targets, top_k=5, chunk=4)
        with_tail = sketch_forward_kl(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            has_tail=True)
        without = sketch_forward_kl(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            has_tail=False)
        assert not torch.allclose(with_tail, without), (
            "if these agree the tail term is not being applied at all")
        #: NO ORDERING between them, and asserting one was my error. The tail
        #: term `p_tail * (log p_tail - log q_tail)` is NEGATIVE whenever the
        #: candidate puts more mass outside the support than the reference does,
        #: and the support-only sum is not a KL at all when `K < V` -- its `p_i`
        #: sum to `1 - p_tail`, not to 1. What DOES hold is that the complete
        #: `K+1` divergence is a real KL between two distributions over K+1
        #: buckets, so it is non-negative. That is the property worth pinning.
        assert float(with_tail.min()) >= -1e-6, (
            "the complete K+1 divergence is a KL and cannot be negative")

    def test_a_row_with_no_valid_position_is_refused(self):
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean_batch,
        )

        g = torch.Generator().manual_seed(9)
        B, T, V = 2, 6, 40
        rb = torch.randn(B, T, V, generator=g)
        cb = torch.randn(B, T, V, generator=g)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :4] = True          # row 1 has nothing
        flat = rb.reshape(-1, V)
        sb = sketch_reference(flat, torch.zeros(B * T, dtype=torch.long),
                              top_k=4, chunk=16)
        with pytest.raises(ValueError, match="no weighted valid position"):
            sketch_forward_kl_mean_batch(
                sb.support_indices.reshape(B, T, -1),
                sb.support_log_probs.reshape(B, T, -1),
                sb.tail_log_prob.reshape(B, T), cb, mask, has_tail=True)

    def test_an_all_zero_weight_item_is_refused(self):
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean,
        )

        ref, cand, targets = _pair(T=6, V=30)
        sk = sketch_reference(ref, targets, top_k=4, chunk=4)
        with pytest.raises(ValueError, match="every scoring weight is zero"):
            sketch_forward_kl_mean(sk, cand, weights=torch.zeros(6))

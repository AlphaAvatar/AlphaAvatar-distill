"""Prefix/continuation splitting — every registered guard, and the ways it lies.

The guards are not stylistic. A split that leaves an empty prefix trains nothing
about recovery; a split past a terminal token asks the model to continue after
its own `<|im_end|>`; two identical truncations halve the corpus while appearing
to double it; and a sample that overflows the block silently gets its
continuation cut by the packer, shortening supervision without any error.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.prefix_split import (  # noqa: E402
    MIN_CONTINUATION_TOKENS, Split, TruncationError, build_splits,
    prefix_length_profile, split_at, supervised_span, truncation_points,
)

STOP = frozenset({151645, 151643})


def sample(n_ctx: int = 20, n_sup: int = 60):
    ids = list(range(1000, 1000 + n_ctx + n_sup))
    mask = [False] * n_ctx + [True] * n_sup
    return ids, mask


# ------------------------------------------------------------------ span

def test_supervised_span_finds_the_single_run():
    _, mask = sample(5, 10)
    assert supervised_span(mask) == (5, 15)


def test_span_rejects_a_split_mask():
    with pytest.raises(TruncationError, match="not_contiguous"):
        supervised_span([True, False, True])
    with pytest.raises(TruncationError, match="no_supervised_tokens"):
        supervised_span([False, False])


# ------------------------------------------------------- truncation points

def test_points_are_deterministic_distinct_and_legal():
    a = truncation_points(60, seed_material="id-7", count=2)
    b = truncation_points(60, seed_material="id-7", count=2)
    c = truncation_points(60, seed_material="id-8", count=2)
    assert a == b and len(set(a)) == 2
    assert a != c
    for k in a:
        assert 1 <= k <= 60 - MIN_CONTINUATION_TOKENS


def test_points_never_repeat_even_on_the_shortest_legal_span():
    n = 1 + MIN_CONTINUATION_TOKENS + 1        # exactly two legal points
    pts = truncation_points(n, seed_material="x", count=2)
    assert len(set(pts)) == 2


def test_span_too_short_to_split_is_refused():
    with pytest.raises(TruncationError, match="too_short_to_split"):
        truncation_points(MIN_CONTINUATION_TOKENS, seed_material="x")
    with pytest.raises(TruncationError, match="too_few_distinct"):
        truncation_points(1 + MIN_CONTINUATION_TOKENS, seed_material="x", count=2)


# -------------------------------------------------------------- split_at

def test_split_moves_only_the_mask_never_the_tokens():
    ids, mask = sample()
    s = split_at(ids, mask, 10, stop_ids=STOP)
    assert s.ids == ids                                  # tokens untouched
    assert sum(s.mask) == 60 - 10 == s.n_continuation_tokens
    assert s.n_prefix_tokens == 20 + 10
    assert not any(s.mask[:30]) and all(s.mask[30:])


def test_empty_assistant_prefix_is_refused():
    ids, mask = sample()
    with pytest.raises(TruncationError, match="empty_assistant_prefix"):
        split_at(ids, mask, 0, stop_ids=STOP)


def test_truncation_at_or_past_the_answer_end_is_refused():
    ids, mask = sample(5, 20)
    with pytest.raises(TruncationError, match="continuation_too_short"):
        split_at(ids, mask, 20, stop_ids=STOP)
    with pytest.raises(TruncationError, match="continuation_too_short"):
        split_at(ids, mask, 20 - MIN_CONTINUATION_TOKENS + 1, stop_ids=STOP)


def test_truncation_immediately_after_a_stop_token_is_refused():
    ids, mask = sample(5, 40)
    ids[5 + 9] = 151645                       # a terminal token ends the prefix
    with pytest.raises(TruncationError, match="truncation_after_terminal"):
        split_at(ids, mask, 10, stop_ids=STOP)
    # One token later the prefix no longer ends on it, so the split is legal.
    assert split_at(ids, mask, 11, stop_ids=STOP).n_continuation_tokens == 29


def test_continuation_is_exactly_the_unsupervised_remainder():
    ids, mask = sample(7, 33)
    for k in (1, 5, 33 - MIN_CONTINUATION_TOKENS):
        s = split_at(ids, mask, k, stop_ids=STOP)
        assert s.n_continuation_tokens == 33 - k
        assert s.span_start == 7 and s.span_end == 40
        assert s.mask.index(True) == 7 + k


# ------------------------------------------------------------ build_splits

def test_build_splits_returns_two_distinct_legal_splits():
    ids, mask = sample()
    splits = build_splits(ids, mask, seed_material="s1", stop_ids=STOP)
    assert len(splits) == 2
    assert splits[0].k != splits[1].k
    for s in splits:
        assert s.ids == ids and sum(s.mask) == s.n_continuation_tokens


def test_build_splits_is_reproducible_and_sample_local():
    ids, mask = sample()
    a = [s.k for s in build_splits(ids, mask, seed_material="s1", stop_ids=STOP)]
    b = [s.k for s in build_splits(ids, mask, seed_material="s1", stop_ids=STOP)]
    c = [s.k for s in build_splits(ids, mask, seed_material="s2", stop_ids=STOP)]
    assert a == b and a != c


def test_oversized_sample_is_rejected_before_the_packer_can_cut_it():
    ids, mask = sample(10, 100)
    with pytest.raises(TruncationError, match="exceeds_context_budget"):
        build_splits(ids, mask, seed_material="x", stop_ids=STOP,
                     max_total_tokens=50)
    # Exactly at the budget is fine; the guard is on overflow, not on equality.
    ok = build_splits(ids, mask, seed_material="x", stop_ids=STOP,
                      max_total_tokens=110)
    assert len(ok) == 2


def test_reasons_are_machine_readable_for_the_rejection_census():
    ids, mask = sample(5, 3)
    try:
        build_splits(ids, mask, seed_material="x", stop_ids=STOP)
    except TruncationError as e:
        assert e.reason == "too_short_to_split"
    else:
        pytest.fail("expected TruncationError")


# ---------------------------------------------------------------- profile

def test_profile_summarises_both_distributions():
    ids, mask = sample(5, 60)
    splits = [split_at(ids, mask, k, stop_ids=STOP) for k in (1, 10, 30, 50)]
    p = prefix_length_profile(splits)
    assert p["n"] == 4
    assert p["prefix_tokens"]["min"] == 6 and p["prefix_tokens"]["max"] == 55
    assert p["continuation_tokens"]["total"] == sum(
        s.n_continuation_tokens for s in splits)
    assert p["continuation_tokens"]["max"] == 59


def test_empty_profile_is_not_a_crash():
    assert prefix_length_profile([]) == {"n": 0}


# ------------------------------------------------- fraction-matched cutting

def test_fractions_are_deterministic_distinct_and_open_interval():
    from aadistill.data.prefix_split import truncation_fractions
    a = truncation_fractions(seed_material="p1")
    assert a == truncation_fractions(seed_material="p1")
    assert a != truncation_fractions(seed_material="p2")
    assert len(set(a)) == 2 and all(0.0 < f < 1.0 for f in a)


def test_the_same_prompt_cuts_at_the_same_relative_depth_in_both_arms():
    """This is what lets C and R be compared as state distributions.

    A student rollout and a teacher target for one prompt have very different
    lengths; matching the fraction keeps 'how far into its own trajectory the
    model is' comparable, which absolute token counts cannot do.
    """
    from aadistill.data.prefix_split import k_from_fraction, truncation_fractions
    fracs = truncation_fractions(seed_material="prompt-42")
    teacher_span, student_span = 600, 4000
    ks_t = [k_from_fraction(teacher_span, f) for f in fracs]
    ks_s = [k_from_fraction(student_span, f) for f in fracs]
    for f, kt, ks in zip(fracs, ks_t, ks_s):
        assert abs(kt / teacher_span - f) < 0.01
        assert abs(ks / student_span - f) < 0.01
    assert ks_t != ks_s          # same depth, different absolute lengths


def test_fraction_clamping_never_emits_duplicate_truncations():
    """On a short span two nearby fractions can round together; that is refused."""
    ids, mask = sample(3, MIN_CONTINUATION_TOKENS + 3)
    splits = build_splits(ids, mask, seed_material="short", stop_ids=STOP)
    assert len({s.k for s in splits}) == 2


def test_fraction_path_still_honours_every_guard():
    ids, mask = sample(4, 500)
    for s in build_splits(ids, mask, seed_material="g", stop_ids=STOP):
        assert s.k >= 1
        assert s.n_continuation_tokens >= MIN_CONTINUATION_TOKENS
        assert ids[s.n_prefix_tokens - 1] not in STOP


def test_a_fraction_collision_falls_back_instead_of_discarding_the_sample():
    """A ~1-in-10,000 hash accident must not cost a usable rollout.

    Found in the real arm-C build: one session of 1,147 drew two identical
    fractions and was rejected outright, even though the integer fallback for
    post-clamp collisions was sitting right there.
    """
    import hashlib
    from aadistill.data.prefix_split import truncation_fractions

    colliding = None
    for i in range(200000):
        m = f"probe-{i}"
        d = hashlib.sha256(("frac:" + m).encode()).digest()
        a = int.from_bytes(d[0:4], "big") % 9999
        b = int.from_bytes(d[4:8], "big") % 9999
        if a == b:
            colliding = m
            break
    assert colliding, "no collision found in the probe range"
    with pytest.raises(TruncationError, match="duplicate_fractions"):
        truncation_fractions(seed_material=colliding)

    ids, mask = sample(5, 400)
    splits = build_splits(ids, mask, seed_material=colliding, stop_ids=STOP)
    assert len(splits) == 2 and splits[0].k != splits[1].k

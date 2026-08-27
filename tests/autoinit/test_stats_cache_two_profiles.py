"""At P=2 the statistics cache must hold two entries, and not one more thing.

`_candidate_expansions` orders by `impl_id`, so a P=2 parent is expanded as

    attention(none), composite(db), composite(rh), depth.causal(db),
    depth.causal(rh), depth.positional(none), ffn(db), ffn(rh),
    width(db), width(rh)

`width.global_pca_v0` and `ffn.activation_importance_v0` both consume
ACTIVATION_STATS. With a single-entry cache that sequence thrashes: ffn(db)
misses and fills, ffn(rh) misses and evicts it, width(db) misses and evicts that,
width(rh) misses again — **four** statistics passes for a parent that needs two.
At the 4B teacher a pass is ~1.8 GiB of float64 accumulation and the single
largest unmeasured term in the cost model.

Widening the cache to one entry per active profile fixes it **without touching
the reuse boundary**, which is the part that carries scientific weight: the key
still leads with the parent's artifact digest, so reuse across parents remains
impossible by construction rather than by discipline, and the search now clears
the cache at each parent boundary so residency is bounded by the parent being
expanded rather than by eviction order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.stats import (  # noqa: E402
    DEFAULT_STATS_SPEC,
    StatsCache,
    stats_cache_key,
)

PARENT_A = "a" * 64
PARENT_B = "b" * 64
DB = "profile-domain-balanced-hash"
RH = "profile-reasoning-heavy-hash"


def key(parent: str, profile: str) -> str:
    return stats_cache_key(
        parent_artifact_digest=parent, profile_hash=profile,
        stats_spec=DEFAULT_STATS_SPEC, adapter_version="qwen3/1",
        numerical_config={"device": "cpu", "accumulation": "float64"})


class Collector:
    """Deterministic per (parent, profile), and it counts how often it runs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def for_(self, parent: str, profile: str):
        def collect():
            self.calls.append((parent, profile))
            g = torch.Generator().manual_seed(
                abs(hash((parent, profile))) % (2**31))
            return {"residual_sqsum": torch.randn(4, generator=g, dtype=torch.float64)}
        return collect


#: The order `_candidate_expansions` yields for the two stats-consuming
#: operators at P=2, which is what makes the thrash reachable.
P2_SEQUENCE = ((PARENT_A, DB), (PARENT_A, RH), (PARENT_A, DB), (PARENT_A, RH))


def drive(cache: StatsCache, sequence, collector: Collector):
    out = []
    for parent, profile in sequence:
        out.append(cache.get_or_collect(key(parent, profile),
                                        collector.for_(parent, profile)))
    return out


def test_one_entry_thrashes_at_P2_and_two_entries_do_not():
    """The defect and the fix, measured in passes rather than described."""
    thrashing = Collector()
    drive(StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=1),
          P2_SEQUENCE, thrashing)
    assert len(thrashing.calls) == 4, (
        "the single-entry cache did not thrash; this test no longer reproduces "
        "the condition it exists to fix")

    fixed = Collector()
    drive(StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=2),
          P2_SEQUENCE, fixed)
    assert len(fixed.calls) == 2
    assert set(fixed.calls) == {(PARENT_A, DB), (PARENT_A, RH)}


def test_the_cached_values_are_what_a_fresh_collect_would_have_returned():
    """Equivalence: a hit must be indistinguishable from a recollection."""
    collector = Collector()
    cache = StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=2)
    got = drive(cache, P2_SEQUENCE, collector)

    fresh = Collector()
    expected = [fresh.for_(p, pr)() for p, pr in P2_SEQUENCE]
    assert len(fresh.calls) == 4
    for a, b in zip(got, expected):
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), k


def test_a_single_profile_search_is_unchanged():
    """P=1 must still be exactly one resident entry and one pass."""
    collector = Collector()
    cache = StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=1)
    drive(cache, ((PARENT_A, DB), (PARENT_A, DB)), collector)
    assert len(collector.calls) == 1
    assert cache.hits == 1 and cache.misses == 1


def test_reuse_across_parents_is_still_impossible():
    """The property the whole cache design exists to protect.

    After a depth or attention operator the residual second moments are no longer
    the parent's, and E8a's central negative result is that a statistic taken
    before composition mispredicts the composed model. Two entries must not
    become a way to carry one parent's statistics into another's expansion.
    """
    collector = Collector()
    cache = StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=2)
    drive(cache, ((PARENT_A, DB), (PARENT_B, DB)), collector)
    assert collector.calls == [(PARENT_A, DB), (PARENT_B, DB)]
    assert cache.hits == 0, "a different parent produced a cache hit"
    assert key(PARENT_A, DB) != key(PARENT_B, DB)


def test_every_keyed_dimension_still_separates_entries():
    """Parent, profile, spec, adapter and numerics each make a distinct entry."""
    base = dict(parent_artifact_digest=PARENT_A, profile_hash=DB,
                stats_spec=DEFAULT_STATS_SPEC, adapter_version="qwen3/1",
                numerical_config={"device": "cpu", "accumulation": "float64"})
    baseline = stats_cache_key(**base)
    assert stats_cache_key(**{**base, "parent_artifact_digest": PARENT_B}) != baseline
    assert stats_cache_key(**{**base, "profile_hash": RH}) != baseline
    assert stats_cache_key(**{**base, "adapter_version": "qwen3/2"}) != baseline
    assert stats_cache_key(**{**base, "numerical_config": {
        "device": "cuda", "accumulation": "float64"}}) != baseline


def test_residency_is_bounded_by_the_profiles_not_by_the_history():
    """Two entries, then a third profile, must not grow without limit."""
    collector = Collector()
    cache = StatsCache(stats_spec=DEFAULT_STATS_SPEC, max_entries=2)
    drive(cache, ((PARENT_A, DB), (PARENT_A, RH), (PARENT_A, "third")), collector)
    assert cache.report()["resident_entries"] == 2


def test_the_search_sizes_the_cache_from_the_ACTIVE_profiles(
        teacher, teacher_spec, target_spec, eval_suite, two_profiles, profile,
        tmp_path):
    """The wiring, not just the container: P=2 gets 2, P=1 still gets 1."""
    from aadistill.autoinit.arch import get_adapter
    from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1
    from aadistill.autoinit.search import BeamSearch, SearchConfig

    def build(profiles, name):
        config = SearchConfig(
            run_id=name, target_spec=target_spec, schedule=SCHEDULE_V1, seed=7,
            workdir=tmp_path / name, profiles=tuple(profiles), policy=PARETO_V1,
            suite=eval_suite)
        return BeamSearch(
            adapter=get_adapter("qwen3"), config=config,
            root_loader=lambda: teacher, root_spec=teacher_spec,
            calibration_loader=lambda p: [], measurer=lambda m, d: None,
            root_teacher_id="t", root_teacher_sha256="0" * 64)

    assert build(two_profiles, "p2").stats_cache.max_entries == 2
    assert build((profile,), "p1").stats_cache.max_entries == 1


def test_the_search_drops_the_previous_parents_statistics():
    """Clearing at the parent boundary is what bounds resident memory.

    Without it a P=2 run holds two ~1.8 GiB entries that can never be hit again,
    because the key carries the parent digest.
    """
    import inspect

    from aadistill.autoinit import search as search_module

    source = inspect.getsource(search_module.BeamSearch.run)
    assert "self.stats_cache.clear()" in source
    body = source[source.index("for parent in parents:"):]
    assert body.index("self.stats_cache.clear()") < body.index("_candidate_expansions"), (
        "the cache is cleared after the expansions are enumerated rather than "
        "before the parent is expanded")

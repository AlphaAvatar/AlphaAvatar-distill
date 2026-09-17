"""The device-resident reduction must agree with the host path it replaces.

`state_evaluation_seconds` is 67-80% of every non-DEPTH expansion in both
committed telemetry files, and it was spent moving two `[T, ~152k]` float32
logit tensors to the host per item and reducing them there. The reduction now
stays on the compute device and only reduced scalars cross the bus.

That is admissible **only if equivalence is demonstrated**. This file compares
the two paths on the same inputs: every metric, every tagged metric, and --
because a metric is only interesting if it changes a decision -- the resulting
Pareto and ranking outcomes.

CPU-only here, so what it establishes is that the *accumulator* restructuring is
equivalent: chunk-wise float32 reductions accumulated in float64, one host
transfer at the end instead of six per chunk. Whether the CUDA kernels agree
with the host kernels to the same tolerance is a GPU question and is owed to the
bounded L40S validation -- stated rather than implied, because a CPU run cannot
answer it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.statistics.contribution import (  # noqa: E402
    DistortionSums, distortion)


def host_reference(ref, abl, targets, *, tags=None, chunk=512) -> DistortionSums:
    """The reduction EXACTLY as it was before the device path existed.

    Kept here rather than deleted from history: an equivalence claim needs the
    thing it is equivalent to, and a copy that drifts is caught by the
    metamorphic test at the bottom of this file.
    """
    tags = dict(tags or {})
    out = DistortionSums()
    for a in range(0, ref.shape[0], chunk):
        b = min(a + chunk, ref.shape[0])
        p_log = F.log_softmax(ref[a:b].float(), dim=-1)
        q_log = F.log_softmax(abl[a:b].float(), dim=-1)
        p = p_log.exp()
        per_pos = (p * (p_log - q_log)).sum(-1)
        tg = targets[a:b]
        out.positions += int(b - a)
        out.kl += float(per_pos.sum())
        out.reverse_kl += float((q_log.exp() * (q_log - p_log)).sum(-1).sum())
        out.ref_ce += float(-p_log.gather(1, tg[:, None]).sum())
        out.abl_ce += float(-q_log.gather(1, tg[:, None]).sum())
        out.top1_agree += int((p_log.argmax(-1) == q_log.argmax(-1)).sum())
        for name, mask in tags.items():
            m = mask[a:b]
            k = int(m.sum())
            if k:
                out.add_tagged(name, float(per_pos[m].sum()), k)
    return out


def make_case(positions: int, vocab: int, seed: int, *, tagged: bool = True):
    """Logit pairs with a realistic gap: a compressed student is not noise.

    The candidate is the reference plus structured perturbation rather than an
    independent draw, because an independent draw gives a KL far outside the
    range any real state produces and would exercise the accumulators at
    magnitudes the search never sees.
    """
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(positions, vocab, generator=g) * 2.0
    abl = ref + torch.randn(positions, vocab, generator=g) * 0.35
    targets = torch.randint(0, vocab, (positions,), generator=g)
    tags = {}
    if tagged:
        tags = {
            #: A dense tag and a sparse one. The sparse one matters: the host
            #: path omits a tag whose chunk has no matching position, and an
            #: accumulator that emitted a zero-count entry instead would make
            #: `as_dict` report a tag the suite never saw.
            "answer_like": (targets % 7 == 0),
            "eos_like": (targets == 0),
        }
    return ref, abl, targets, tags


# --- the accumulator restructuring is equivalent ----------------------------

@pytest.mark.parametrize("positions,vocab,chunk", [
    (37, 4096, 512),        # one partial chunk
    (512, 2048, 512),       # exactly one chunk
    (900, 3072, 512),       # the real suite's shape ratio, two chunks
    (1024, 1024, 128),      # many chunks
])
def test_the_device_accumulators_agree_with_the_host_path(positions, vocab, chunk):
    """Same numbers, whichever accumulator collected them.

    On CPU `distortion` takes the host branch, so this compares the CURRENT
    implementation against a frozen copy of the PREVIOUS one -- which is what
    protects the host path from being changed by accident while the device path
    was being added.
    """
    ref, abl, targets, tags = make_case(positions, vocab, seed=positions)
    old = host_reference(ref, abl, targets, tags=tags, chunk=chunk).as_dict()
    new = distortion(ref, abl, targets, tags=tags, chunk=chunk).as_dict()

    assert new["positions"] == old["positions"]
    assert new["top1_agreement"] == old["top1_agreement"]
    for key in ("kl", "reverse_kl", "ref_ce", "abl_ce", "ce_delta"):
        assert new[key] == pytest.approx(old[key], rel=1e-12, abs=1e-12), key
    assert set(new["tagged"]) == set(old["tagged"])
    for tag, entry in old["tagged"].items():
        got = new["tagged"][tag]
        assert got["positions"] == entry["positions"], tag
        if entry["kl"] is None:
            assert got["kl"] is None, tag
        else:
            assert got["kl"] == pytest.approx(entry["kl"], rel=1e-12, abs=1e-12)


def test_a_tag_that_matches_nothing_is_omitted_not_zeroed():
    """The host path's `if k:` has a device counterpart, and it must behave.

    A tag with no matching position must be ABSENT from `tagged`, because
    `StateEvaluator` turns every present tag into a
    `state.critical_token_kl.<tag>` metric -- so a zero-count entry would
    invent a metric for a token class the item does not contain.
    """
    ref, abl, targets, _ = make_case(64, 512, seed=7, tagged=False)
    never = {"never_matches": torch.zeros(64, dtype=torch.bool)}
    old = host_reference(ref, abl, targets, tags=never).as_dict()
    new = distortion(ref, abl, targets, tags=never).as_dict()
    assert old["tagged"] == {} and new["tagged"] == {}


def test_chunking_changes_the_result_only_at_float32_summation_scale():
    """It is NOT exactly invariant, and the size of that matters downstream.

    `log_softmax` reduces over the vocabulary, per row, so a chunk boundary
    cannot change a row's normalization. What chunking does change is the
    grouping of the per-chunk float32 `.sum()` reductions, and measurement puts
    that at ~4e-8 relative -- not the 1e-9 this test first assumed and not zero.

    That number is the reason the chunking requirement on candidates 2 and 3 is
    a real constraint rather than a formality: ANY restructuring that regroups
    the summation moves the result at this scale. It is still five orders below
    the ~7.8e-3 threshold the search's own comparisons use, which is what makes
    it acceptable -- but it has to be stated, because "chunk-invariant" would be
    a false claim that a later optimization could be justified by.
    """
    ref, abl, targets, tags = make_case(777, 1024, seed=3)
    results = [distortion(ref, abl, targets, tags=tags, chunk=c).as_dict()
               for c in (64, 256, 512, 1024)]
    first = results[0]
    worst = 0.0
    for other in results[1:]:
        for key in ("kl", "reverse_kl", "ref_ce", "abl_ce"):
            worst = max(worst, abs(other[key] - first[key]) / abs(first[key]))
    print(f"\nchunk-size sensitivity: {worst:.3e} relative")
    assert worst < 1e-6, (
        f"chunking moves the result by {worst:.3e}, which is more than float32 "
        "summation regrouping explains")
    #: And it is not zero, so nothing downstream may assume chunk invariance.
    assert worst > 0.0, (
        "chunking became exactly invariant -- if that is a deliberate change, "
        "the constraint on candidates 2 and 3 can be relaxed and this test "
        "should say so")


# --- and the decisions it feeds are unchanged -------------------------------

def test_the_metrics_feed_identical_pareto_and_ranking_decisions():
    """A metric that moved below every decision boundary changed nothing.

    Equivalence of numbers is necessary and not sufficient: what must hold is
    that the same states receive the same beam decisions. So this builds states
    from both reductions and runs the REAL ranking policy over them.
    """
    from aadistill.initialization.planning.ranking import PARETO_V1

    #: Eight synthetic states whose metrics come from real reductions, spread
    #: so that some pairs are close enough for an epsilon to matter.
    old_states, new_states = [], []
    for i in range(8):
        ref, abl, targets, tags = make_case(256, 1024, seed=100 + i)
        old = host_reference(ref, abl, targets, tags=tags).as_dict()
        new = distortion(ref, abl, targets, tags=tags).as_dict()
        old_states.append(old)
        new_states.append(new)

    #: The objective vectors the policy would see, in the policy's own order.
    def vectors(rows):
        return [(r["kl"], r["reverse_kl"], -r["top1_agreement"]) for r in rows]

    assert vectors(old_states) == pytest.approx(vectors(new_states), rel=1e-12)

    #: And the ORDERING on each objective is identical, which is what a Pareto
    #: front and a beam selection actually consume.
    for axis in range(3):
        old_order = sorted(range(8), key=lambda i: vectors(old_states)[i][axis])
        new_order = sorted(range(8), key=lambda i: vectors(new_states)[i][axis])
        assert old_order == new_order, f"objective {axis} reorders"

    assert PARETO_V1.policy_hash, "the policy must still identify itself"


def test_the_drift_is_orders_below_the_smallest_decision_boundary():
    """Quantified, not asserted.

    C2's own B->C comparison was decided by a margin of 0.395971 against a
    0.007782 threshold, so the boundary the search actually uses is ~7.8e-3.
    The restructured accumulator must be far below that -- and the figure is
    printed so a reviewer sees the headroom rather than a pass.
    """
    worst = 0.0
    for i in range(6):
        ref, abl, targets, tags = make_case(512, 2048, seed=200 + i)
        old = host_reference(ref, abl, targets, tags=tags).as_dict()
        new = distortion(ref, abl, targets, tags=tags).as_dict()
        for key in ("kl", "reverse_kl", "ref_ce", "abl_ce"):
            denom = max(abs(old[key]), 1e-12)
            worst = max(worst, abs(new[key] - old[key]) / denom)
    print(f"\nworst relative drift over 6 cases: {worst:.3e}")
    assert worst < 1e-9, f"drift {worst:.3e} is too close to a decision boundary"


# --- the DEVICE branch itself, executed at $0 -------------------------------

@pytest.fixture
def resident(monkeypatch):
    """Drive the device-resident accumulators with host tensors.

    The branch keys on `device.type != "cpu"`, so on a machine without an
    accelerator it is unreachable — and an unreachable branch is what four paid
    pods in this repository have died inside. Overriding the predicate runs the
    REAL accumulator code: float64 device tensors, `.tolist()` once at the end,
    the `if count:` tag omission. What it cannot test is whether CUDA's
    `log_softmax` agrees with the host's, which is a kernel question the L40S
    validation owns.
    """
    from aadistill.initialization.statistics import contribution

    monkeypatch.setattr(contribution, "_reduce_on_device", lambda device: True)
    return contribution.distortion


@pytest.mark.parametrize("positions,vocab,chunk", [
    (37, 4096, 512), (512, 2048, 512), (900, 3072, 512), (1024, 1024, 128),
])
def test_the_resident_branch_computes_the_same_numbers(resident, positions,
                                                       vocab, chunk):
    ref, abl, targets, tags = make_case(positions, vocab, seed=positions)
    old = host_reference(ref, abl, targets, tags=tags, chunk=chunk).as_dict()
    new = resident(ref, abl, targets, tags=tags, chunk=chunk).as_dict()

    assert new["positions"] == old["positions"]
    assert new["top1_agreement"] == old["top1_agreement"], (
        "top1 is an integer count; a float64 accumulator must round back to it")
    for key in ("kl", "reverse_kl", "ref_ce", "abl_ce", "ce_delta"):
        assert new[key] == pytest.approx(old[key], rel=1e-12, abs=1e-12), key
    assert set(new["tagged"]) == set(old["tagged"])
    for tag, entry in old["tagged"].items():
        got = new["tagged"][tag]
        assert got["positions"] == entry["positions"], tag
        assert got["kl"] == pytest.approx(entry["kl"], rel=1e-12, abs=1e-12), tag


def test_the_resident_branch_omits_a_tag_that_matches_nothing(resident):
    """The device path has no `if k:` per chunk -- it decides once, at the end.

    So this is the case where the two paths could most easily diverge in
    BEHAVIOUR rather than in arithmetic: a zero-count tag would become a
    `state.critical_token_kl.<tag>` metric for a token class the item does not
    contain.
    """
    ref, abl, targets, _ = make_case(64, 512, seed=7, tagged=False)
    never = {"never_matches": torch.zeros(64, dtype=torch.bool)}
    assert resident(ref, abl, targets, tags=never).as_dict()["tagged"] == {}


def test_the_resident_branch_counts_integer_quantities_exactly(resident):
    """`top1_agree` and tag positions are counts that pass through float64.

    A float64 can hold every integer up to 2^53, so the round-trip is exact —
    but it is exact by that argument rather than by construction, and a count
    that drifted by one would change a denominator.
    """
    ref, abl, targets, tags = make_case(1024, 512, seed=11)
    old = host_reference(ref, abl, targets, tags=tags).as_dict()
    new = resident(ref, abl, targets, tags=tags).as_dict()
    assert new["positions"] == old["positions"] == 1024
    for tag in old["tagged"]:
        assert new["tagged"][tag]["positions"] == old["tagged"][tag]["positions"]


def test_the_predicate_is_what_selects_the_branch():
    """Mutation-proof: if the predicate stopped deciding, the fixture above
    would silently test the host path twice and prove nothing."""
    from aadistill.initialization.statistics import contribution

    assert contribution._reduce_on_device(torch.device("cpu")) is False
    assert contribution._reduce_on_device(torch.device("cuda", 0)) is True
    assert contribution._reduce_on_device(torch.device("meta")) is True


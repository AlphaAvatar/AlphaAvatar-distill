"""The forward-KL-only reduction must produce the same DEPTH decisions.

`depth.causal_kl_greedy_v1` read `sums["kl"]` out of a six-quantity reduction,
17,420 times per expansion, and discarded the other five. `forward_kl_mean`
computes that one quantity and skips the reverse-KL term, both cross-entropy
gathers and both argmaxes.

A score that is merely close is not enough here: the operator's output is a
LAYER REMOVAL ORDER chosen by argmin over candidate scores, so what has to hold
is that every round's table, every chosen layer and the full removal order are
identical. A drift smaller than any real score gap changes nothing; a drift that
crosses one changes the checkpoint.

The tolerance is predeclared below rather than discovered from the failure.
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
    distortion, forward_kl_mean, greedy_removal)

#: PREDECLARED. The reduction is the same float32 log-softmax over the same
#: chunks accumulated in the same float64, so the only admissible difference is
#: float32 summation regrouping -- measured at ~9e-8 relative for a chunk-size
#: change. Anything larger means the arithmetic diverged, not the rounding.
TOLERANCE = 1e-9


def make_case(positions: int, vocab: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(positions, vocab, generator=g) * 2.0
    abl = ref + torch.randn(positions, vocab, generator=g) * 0.35
    targets = torch.randint(0, vocab, (positions,), generator=g)
    return ref, abl, targets


# --- the scalar agrees ------------------------------------------------------

@pytest.mark.parametrize("positions,vocab,chunk", [
    (37, 4096, 512), (512, 2048, 512), (900, 3072, 512), (1024, 1024, 128),
    (1, 512, 512),
])
def test_it_equals_the_kl_the_full_reduction_reports(positions, vocab, chunk):
    ref, abl, targets = make_case(positions, vocab, seed=positions)
    full = distortion(ref, abl, targets, chunk=chunk).as_dict()["kl"]
    only = forward_kl_mean(ref, abl, chunk=chunk)
    assert only == pytest.approx(full, rel=TOLERANCE, abs=TOLERANCE)


def test_it_preserves_the_chunk_boundaries():
    """Chunking is a real constraint: it moves the result at ~9e-8.

    So the KL-only path must chunk the same way the full one does, and this
    checks the two agree AT EACH chunk size rather than only at the default.
    """
    ref, abl, targets = make_case(777, 1024, seed=3)
    for chunk in (64, 256, 512, 1024, 2048):
        full = distortion(ref, abl, targets, chunk=chunk).as_dict()["kl"]
        only = forward_kl_mean(ref, abl, chunk=chunk)
        assert only == pytest.approx(full, rel=TOLERANCE, abs=TOLERANCE), chunk


def test_it_refuses_mismatched_shapes_and_empty_input():
    ref, abl, _ = make_case(16, 128, seed=1)
    with pytest.raises(ValueError, match="shape mismatch"):
        forward_kl_mean(ref, abl[:8])
    with pytest.raises(ValueError, match="no positions"):
        forward_kl_mean(ref[:0], abl[:0])


def test_the_device_branch_of_the_kl_path_agrees(monkeypatch):
    """Executed at $0, like the full reduction's.

    A branch only reachable on an accelerator is a branch no cheap test runs.
    """
    from aadistill.initialization.statistics import contribution

    ref, abl, targets = make_case(900, 2048, seed=42)
    host = forward_kl_mean(ref, abl, chunk=512)
    monkeypatch.setattr(contribution, "_reduce_on_device", lambda device: True)
    resident = contribution.forward_kl_mean(ref, abl, chunk=512)
    assert resident == pytest.approx(host, rel=1e-12, abs=1e-12)


# --- and so does every decision it feeds ------------------------------------

def _score_table(reduce, layers, items, protect=()):
    """Run the REAL greedy removal with `reduce` as the objective.

    `greedy_removal` is the frozen selection rule -- its tie-breaking, its
    candidate order and its round structure are the scientific semantics -- so
    driving it with each reduction is what tests the decisions rather than the
    numbers.
    """
    rounds = []

    def score_fn(skip):
        per = []
        for ref, abl_by_skip in items:
            abl = abl_by_skip(skip)
            per.append(reduce(ref, abl))
        return sum(per) / len(per)

    result = greedy_removal(score_fn, layers, 2, protect=protect,
                            on_round=lambda r: rounds.append(r))
    return result, rounds


def test_identical_rounds_choices_and_removal_order():
    """The property that matters: same table, same layer, same order.

    Built so that scores are CLOSE -- candidate gaps of ~1e-3 -- because a
    synthetic case with large gaps would pass under any tolerance and prove
    nothing about a boundary.
    """
    torch.manual_seed(1234)
    n_layers, positions, vocab = 8, 256, 1024
    ref = torch.randn(positions, vocab) * 2.0
    #: One deterministic perturbation per layer, so a skip set maps to a fixed
    #: ablated tensor -- the real operator's structure, where a candidate subset
    #: determines the forward exactly.
    per_layer = [torch.randn(positions, vocab) * 0.05 for _ in range(n_layers)]

    def abl_by_skip(skip):
        out = ref.clone()
        for layer in sorted(skip):
            out = out + per_layer[layer]
        return out

    items = [(ref, abl_by_skip)]
    full_result, full_rounds = _score_table(
        lambda r, a: distortion(r, a, torch.zeros(positions, dtype=torch.long),
                                chunk=512).as_dict()["kl"], n_layers, items)
    only_result, only_rounds = _score_table(
        lambda r, a: forward_kl_mean(r, a, chunk=512), n_layers, items)

    #: The REMOVAL ORDER, which is the operator's actual output.
    assert full_result["removed"] == only_result["removed"], (
        f"removal order diverged: {full_result['removed']} vs "
        f"{only_result['removed']}")
    #: Every round: same number, same chosen layer, same candidate set.
    assert len(full_rounds) == len(only_rounds) == 2
    for a, b in zip(full_rounds, only_rounds):
        assert a["chosen"] == b["chosen"], (a["chosen"], b["chosen"])
        assert a["removed_before"] == b["removed_before"]
        assert a["n_candidates"] == b["n_candidates"]
        assert a["chosen_score"] == pytest.approx(
            b["chosen_score"], rel=TOLERANCE, abs=TOLERANCE)
        #: THE COMPLETE PER-ROUND TABLE, in order. `greedy_removal` breaks ties
        #: by `(score, candidate)`, so the candidate ORDER is load-bearing and
        #: comparing sets would miss a reordering that changes a tie.
        assert [r["candidate"] for r in a["table"]] == \
            [r["candidate"] for r in b["table"]], "candidate order differs"
        for ra, rb in zip(a["table"], b["table"]):
            assert ra["candidate"] == rb["candidate"]
            assert rb["score"] == pytest.approx(
                ra["score"], rel=TOLERANCE, abs=TOLERANCE), ra["candidate"]


def test_the_candidate_gaps_are_small_enough_for_the_test_to_mean_something():
    """A tolerance test on well-separated candidates proves nothing.

    So this measures the smallest gap the table above actually contains and
    requires it to be orders ABOVE the tolerance -- which is what makes
    "identical choices" evidence rather than luck.
    """
    torch.manual_seed(1234)
    n_layers, positions, vocab = 8, 256, 1024
    ref = torch.randn(positions, vocab) * 2.0
    per_layer = [torch.randn(positions, vocab) * 0.05 for _ in range(n_layers)]

    scores = {}
    for layer in range(n_layers):
        abl = ref + per_layer[layer]
        scores[layer] = forward_kl_mean(ref, abl, chunk=512)
    ordered = sorted(scores.values())
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    smallest = min(gaps)
    print(f"\nsmallest candidate gap: {smallest:.3e}  tolerance: {TOLERANCE:.0e}"
          f"  ratio: {smallest / TOLERANCE:.1e}x")
    assert smallest > TOLERANCE * 100, (
        f"the smallest candidate gap {smallest:.3e} is within 100x of the "
        f"tolerance {TOLERANCE:.0e}; the decision test cannot distinguish "
        "equivalence from coincidence")


def test_the_operator_no_longer_asks_for_targets():
    """With no cross-entropy there is nothing to gather, so nothing to move.

    Asserted on the source because the absence of a per-item host-to-device
    transfer is not observable from the operator's output -- and a target list
    quietly reintroduced would cost one transfer per item per candidate.
    """
    import ast

    body = (REPO / "src/aadistill/initialization/operators/depth.py").read_text()
    tree = ast.parse(body)
    #: THE CAUSAL-KL CLASS's apply. depth.py defines two operators and
    #: `depth.positional_v0` comes first, so taking the first `apply` in the
    #: module tested the wrong one -- and passed nothing meaningful.
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    causal = [n for n in classes if "causalkl" in n.name.lower()]
    assert len(causal) == 1, (
        f"expected one causal-KL class, found {[n.name for n in causal]} among "
        f"{[n.name for n in classes]}")
    causal = causal[0]
    apply_fn = next(n for n in causal.body
                    if isinstance(n, ast.FunctionDef) and n.name == "apply")
    src = ast.unparse(apply_fn)
    assert "forward_kl_mean" in src
    assert "distortion(" not in src, (
        "the operator calls the six-quantity reduction again; it needs one")
    assert "input_ids'][0, 1:]" not in src.replace('"', "'"), (
        "the operator builds a target list it has no consumer for")

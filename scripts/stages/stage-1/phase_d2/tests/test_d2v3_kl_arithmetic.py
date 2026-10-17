"""Stage W revision 2's KL arithmetic gate (`d2v3_gpu_validation.kl_arithmetic`)
at $0: a predicate table, mutants the table must catch, an independent
oracle, and the float32 reduction under extreme numerical conditions.

    pytest scripts/stages/stage-1/phase_d2/tests/test_d2v3_kl_arithmetic.py

Revision 1 failed on a slack that modelled one log-normalizer's rounding
while every float64 bound held; the table pins each predicate of its
replacement separately, so a mutant that drops a check, swaps a side or
brings that slack back changes at least one row's outcome.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from stages.phase_d2 import d2v3_gpu_validation as W  # noqa: E402

TOL = W.ARITHMETIC_TOLERANCES
ABS_T = TOL["teacher_topk_kl_abs_nats_vs_float64"]
REL_T = TOL["teacher_topk_kl_rel_vs_float64"]
ABS_U = TOL["union_kl_abs_nats_vs_float64"]
REL_U = TOL["union_kl_rel_vs_float64"]


def _row(topk64, union64, full64, kl32_topk, kl32_union, *, log_z=10.0,
         item="i0", layer=0):
    """One (item, candidate) row of one position. The float32 sides are
    float64 tensors here so the table states errors exactly; the gate only
    upcasts."""
    t = lambda x: torch.tensor([x], dtype=torch.float64)     # noqa: E731
    return {"item_id": item, "layer": layer,
            "kl32_topk": t(kl32_topk), "kl32_union": t(kl32_union),
            "oracle": {"topk": t(topk64), "union": t(union64),
                       "full": t(full64)},
            "log_z_ref": t(log_z)}


def _tol(kl, a, r):
    return a + r * abs(kl)


def _outcome(ka):
    return (ka["teacher_topk"]["within_tolerance"],
            ka["union"]["within_tolerance"], ka["bounds_hold"],
            ka["crossings_within_combined_bound"],
            ka["float32_crossings"]["lower"]["count"],
            ka["float32_crossings"]["upper"]["count"],
            ka["anomalies"]["total"])


#: (name, rows, expected outcome):
#: (topk ok, union ok, fp64 order, crossings ok, lower #, upper #, anomalies)
TABLE = [
    ("exact and ordered",
     [_row(1.0, 1.2, 1.5, 1.0, 1.2)],
     (True, True, True, True, 0, 0, 0)),
    ("topk error just inside its tolerance",
     [_row(1.0, 1.2, 1.5, 1.0 + 0.9 * _tol(1.0, ABS_T, REL_T), 1.2)],
     (True, True, True, True, 0, 0, 0)),
    ("topk error just outside, union exact",
     [_row(1.0, 1.2, 1.5, 1.0 + 1.1 * _tol(1.0, ABS_T, REL_T), 1.2)],
     (False, True, True, True, 0, 0, 0)),
    ("union error just outside, topk exact",
     [_row(1.0, 1.2, 1.5, 1.0, 1.2 - 1.1 * _tol(1.2, ABS_U, REL_U))],
     (True, False, True, True, 0, 0, 0)),
    ("large KL: the relative term carries the allowance",
     [_row(10.0, 10.0, 10.5, 10.0 + 5e-4, 10.0 - 5e-4)],
     (True, True, True, True, 1, 0, 0)),
    ("zero KL: the absolute floor carries the allowance",
     [_row(0.0, 0.0, 0.0, 5e-5, -5e-5)],
     (True, True, True, True, 1, 0, 0)),
    ("revision 1's case: an upper crossing inside the combined bound",
     [_row(1.0, 1.5, 1.5, 1.0, 1.5 + 0.5 * _tol(1.5, ABS_U, REL_U))],
     (True, True, True, True, 0, 1, 0)),
    ("a lower crossing only the SUM of both tolerances explains",
     [_row(1.0, 1.0, 1.5, 1.0 + 0.9 * _tol(1.0, ABS_T, REL_T),
           1.0 - 0.9 * _tol(1.0, ABS_U, REL_U))],
     (True, True, True, True, 1, 0, 0)),
    ("a float64 order violation is an anomaly twice over",
     [_row(1.3, 1.2, 1.5, 1.3, 1.2)],
     (True, True, False, False, 1, 0, 2)),
    ("an upper crossing beyond the bound needs an accuracy breach",
     [_row(1.0, 1.5, 1.5, 1.0, 1.5 + 2.0 * _tol(1.5, ABS_U, REL_U))],
     (True, False, True, False, 0, 1, 1)),
    ("matching infinities are not compared",
     [_row(math.inf, math.inf, math.inf, math.inf, math.inf),
      _row(1.0, 1.2, 1.5, 1.0, 1.2)],
     (True, True, True, True, 0, 0, 0)),
    ("a finite float32 KL against an infinite float64 one fails",
     [_row(1.0, math.inf, math.inf, 1.0, 3.0)],
     (True, False, True, True, 0, 0, 1)),
]


@pytest.mark.parametrize("name,rows,want", TABLE, ids=[t[0] for t in TABLE])
def test_the_predicate_table(name, rows, want):
    assert _outcome(W.kl_arithmetic(rows)) == want


def _table_outcomes():
    return [_outcome(W.kl_arithmetic(rows)) for _, rows, _ in TABLE]


MUTANTS = {
    "the teacher-Top-K accuracy check dropped": lambda mp: mp.setitem(
        W.ARITHMETIC_TOLERANCES, "teacher_topk_kl_abs_nats_vs_float64", 1e9),
    "absolute floor only": lambda mp: mp.setattr(
        W, "kl_tolerance", lambda kl, a, r: a + 0.0 * kl),
    "relative term only": lambda mp: mp.setattr(
        W, "kl_tolerance", lambda kl, a, r: r * kl.abs()),
    "lower and upper bounds swapped": lambda mp: mp.setattr(
        W, "combined_bounds", lambda t, u, s: (u + s, t + u + s)),
    "one tolerance for the lower crossing": lambda mp: mp.setattr(
        W, "combined_bounds", lambda t, u, s: (u + s, u + s)),
    "revision 1's 4 eps |log Z| slack as the bound": lambda mp: mp.setattr(
        W, "combined_bounds",
        lambda t, u, s: (torch.full_like(t, 4 * 1.1920929e-07 * 10.0),
                         torch.full_like(t, 4 * 1.1920929e-07 * 10.0))),
    "no float64 order check": lambda mp: mp.setattr(
        W, "ORACLE_BOUND_SLACK", 1e9),
}


@pytest.mark.parametrize("mutant", list(MUTANTS))
def test_every_mutant_changes_some_row(mutant, monkeypatch):
    """MUTATION: each plausible way to get the gate wrong -- including the
    defect revision 1 shipped -- moves at least one table row."""
    want = [w for _, _, w in TABLE]
    MUTANTS[mutant](monkeypatch)
    assert _table_outcomes() != want, mutant


def test_the_bounds_follow_from_the_gates():
    """The combined bounds are the triangle inequality, not a constant."""
    t, u, s = torch.tensor([3e-4]), torch.tensor([5e-4]), torch.tensor([1e-12])
    lo, up = W.combined_bounds(t, u, s)
    assert float(lo) == pytest.approx(8e-4 + 1e-12)
    assert float(up) == pytest.approx(5e-4 + 1e-12)


def test_every_crossing_is_recorded_with_its_float64_values():
    rows = [_row(1.0, 1.5, 1.5, 1.0, 1.5 + 0.5 * _tol(1.5, ABS_U, REL_U),
                 item="a", layer=3),
            _row(2.0, 2.0, 2.5, 2.0 + 1e-4, 2.0 - 1e-4, item="b", layer=5),
            _row(1.0, 1.2, 1.5, 1.0, 1.2, item="c", layer=7)]
    ka = W.kl_arithmetic(rows)
    rec = {(r["item_id"], r["layer"]): r for r in ka["positions_recorded"]}
    assert set(rec) == {("a", 3), ("b", 5)}
    assert rec[("a", 3)]["upper_crossing"] > 0
    assert rec[("b", 5)]["lower_crossing"] > 0
    up = ka["float32_crossings"]["upper"]["at_max"]
    assert (up["item_id"], up["layer"], up["kl64_full"]) == ("a", 3, 1.5)
    assert ka["positions_recorded_omitted"] == 0
    #: revision 1's slack would have flagged both
    assert ka["revision_1_slack_count"]["count"] == 2


def test_the_record_cap_is_stated_not_silent(monkeypatch):
    monkeypatch.setattr(W, "POSITION_RECORD_CAP", 1)
    rows = [_row(1.0, 1.5, 1.5, 1.0, 1.5 + 1e-4, item=str(i)) for i in range(3)]
    ka = W.kl_arithmetic(rows)
    assert len(ka["positions_recorded"]) == 1
    assert ka["positions_recorded_total"] == 3
    assert ka["positions_recorded_omitted"] == 2


# --- an independent oracle --------------------------------------------------------------

def _partition_kl(p_logits, q_logits, members):
    """Per row, in float64, by explicit loops over buckets: a second
    implementation sharing nothing with `oracle_kls` but the definition."""
    out = []
    for r in range(int(p_logits.shape[0])):
        lp = [float(x) for x in torch.log_softmax(p_logits[r].double(), -1)]
        lq = [float(x) for x in torch.log_softmax(q_logits[r].double(), -1)]
        kl, pt, qt = 0.0, 0.0, 0.0
        for i, (a, b) in enumerate(zip(lp, lq)):
            if i in members[r]:
                kl += math.exp(a) * (a - b) if a > -math.inf else 0.0
            else:
                pt += math.exp(a)
                qt += math.exp(b)
        if pt > 0:
            kl += pt * (math.log(pt) - math.log(qt))
        out.append(kl)
    return torch.tensor(out, dtype=torch.float64)


def test_the_oracle_agrees_with_an_independent_implementation():
    g = torch.Generator().manual_seed(11)
    pz = torch.randn(6, 50, generator=g) * 4
    qz = pz + torch.randn(6, 50, generator=g) * 2
    st, sc = torch.topk(pz, 8).indices, torch.topk(qz, 8).indices
    (o,) = W.oracle_kls(pz, [(qz, sc)], st)
    every = [set(range(50))] * 6
    assert torch.allclose(o["full"], _partition_kl(pz, qz, every), atol=1e-12)
    assert torch.allclose(o["topk"], _partition_kl(
        pz, qz, [set(x.tolist()) for x in st]), atol=1e-12)
    assert torch.allclose(o["union"], _partition_kl(
        pz, qz, [set(a.tolist()) | set(b.tolist()) for a, b in zip(st, sc)]),
        atol=1e-12)


# --- the float32 reduction under extreme conditions -------------------------------------

K = 8


def _scenario(name):
    """``(teacher float32 head logits [n, V], hidden, head, candidate bf16)``."""
    from aadistill.initialization.scoring.output_head import LinearOutputHead
    #: a STABLE seed per scenario (str hashes are salted per process)
    g = torch.Generator().manual_seed(1000 + SCENARIOS.index(name))
    n, v, h = 24, 320, 16
    w = torch.randn(v, h, generator=g)
    hidden = torch.randn(n, h, generator=g)
    noise = torch.randn(n, v, generator=g)
    if name == "peaked teacher":
        w = w * 6.0
    elif name == "flat teacher, tail near one":
        w = w * 0.01
    head = LinearOutputHead(weight=w)
    if name == "union swallows the teacher's tail":
        #: the teacher's mass sits on 2K entries; the rest is ~e^-40 down
        bias = torch.full((v,), -40.0)
        bias[:2 * K] = 0.0
        head = LinearOutputHead(weight=w * 0.3, bias=bias)
    z = head.logits(hidden)
    if name == "large logit magnitude":
        #: a common offset leaves every probability unchanged and puts the
        #: float32 log-normalizer at |z| ~ 100
        head = LinearOutputHead(weight=w, bias=torch.full((v,), 100.0))
        z = head.logits(hidden)
    if name == "student equals teacher":
        cand = z.bfloat16()
    elif name == "student far from teacher":
        cand = (-z).bfloat16()
    elif name == "union swallows the teacher's tail":
        #: the candidate's Top-K is the teacher's NEXT K entries, so the
        #: union's tail is the teacher's tail minus most of its mass
        order = z.argsort(-1, descending=True)
        cand = torch.full_like(z, -30.0)
        cand.scatter_(-1, order[:, K:2 * K], 5.0)
        cand = cand.bfloat16()
    elif name == "bf16 ties at the boundary":
        cand = (torch.round(z + noise) / 4).bfloat16() * 4
    else:
        cand = (z + noise).bfloat16()
    return z, hidden, head, cand


SCENARIOS = ["ordinary", "peaked teacher", "flat teacher, tail near one",
             "student equals teacher", "student far from teacher",
             "union swallows the teacher's tail", "bf16 ties at the boundary",
             "large logit magnitude"]


def _reduce(z, hidden, head, cand, *, store_path: bool, tamper=None):
    from aadistill.initialization.scoring.support import (
        TeacherRetrieval, deterministic_topk, sketch_reference_deterministic,
        union_forward_kl_terms,
    )
    n = int(z.shape[0])
    sk, log_z = sketch_reference_deterministic(
        z, torch.zeros(n, dtype=torch.long), top_k=K)
    retrieval = (TeacherRetrieval(hidden, log_z, head) if store_path
                 else W.HeadLogitsRetrieval(z, log_z))
    tail = sk.tail_log_prob if tamper is None else tamper(sk.tail_log_prob)
    terms = union_forward_kl_terms(sk.support_indices, sk.support_log_probs,
                                   tail, cand, retrieval, top_k=K,
                                   has_tail=True)
    _, sc = deterministic_topk(cand, K)
    (o,) = W.oracle_kls(z, [(cand, sc)], sk.support_indices)
    return {"item_id": "x", "layer": 0, "kl32_topk": terms.kl_reference_topk,
            "kl32_union": terms.kl, "oracle": o, "log_z_ref": log_z}


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("store_path", [False, True],
                         ids=["same-input", "store-path"])
def test_the_float32_reduction_holds_under_extremes(name, store_path):
    """Both KLs inside the budget, the float64 order exact, every crossing
    explained -- for a peaked, a flat, an identical, a distant and a
    tail-absorbing candidate, at bf16 ties and at |z| ~ 100."""
    z, hidden, head, cand = _scenario(name)
    ka = W.kl_arithmetic([_reduce(z, hidden, head, cand,
                                  store_path=store_path)])
    assert ka["anomalies"]["total"] == 0, (name, ka["anomalies"])
    assert ka["within_tolerance"], (name, ka["teacher_topk"], ka["union"])
    assert ka["bounds_hold"] and ka["crossings_within_combined_bound"]


@pytest.mark.parametrize("bump", [1e-2, -1e-2])
def test_a_wrong_teacher_tail_is_caught(bump):
    """A defect in the teacher-Top-K side alone -- the partition revision 1
    never compared -- fails the teacher-Top-K gate."""
    z, hidden, head, cand = _scenario("flat teacher, tail near one")
    ka = W.kl_arithmetic([_reduce(z, hidden, head, cand, store_path=False,
                                  tamper=lambda t: t + bump)])
    assert not ka["teacher_topk"]["within_tolerance"]

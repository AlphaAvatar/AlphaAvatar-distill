"""Paired C/R construction — the invariants that make E5 a controlled comparison.

Two failure modes this guards, both of which produce a corpus that looks fine:

* an R-side rejection leaving a C example whose counterpart never existed, so the
  arms differ in composition and not only in state distribution;
* a block-budget cut taken in emission order, which inherits the corpus's task
  sort and gives the arms different mixes at different budgets.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.paired_corpus import (  # noqa: E402
    comparability_report, intersect, length_profile, pair_key, prefix_bucket,
    select_paired, stratified_order, stratum,
)

TASKS = ("gsm8k", "openmath", "rag_evidence", "code")


def ex(sid, seed, ti, task="gsm8k", prefix=200, cont=100, frac=0.4):
    return {"id": f"{sid}#{seed}#{ti}", "source_session_id": sid,
            "source_seed": seed, "truncation_index": ti, "data_type": task,
            "n_prefix_tokens": prefix, "n_continuation_tokens": cont,
            "n_total_tokens": prefix + cont, "truncation_fraction": frac}


def corpus(n=40, seed="sa"):
    out = []
    for i in range(n):
        out.append(ex(f"s{i:03d}", seed, i % 2, TASKS[i % len(TASKS)],
                      prefix=100 * (1 + i % 6), cont=50 + i, frac=0.2 + 0.01 * (i % 50)))
    return out


# ------------------------------------------------------------- intersection

def test_an_r_rejection_removes_its_c_counterpart():
    c = corpus(10)
    r = [e for e in corpus(10) if e["source_session_id"] not in {"s003", "s007"}]
    ck, rk, census = intersect(c, r)
    assert len(ck) == len(rk) == 8
    assert census["paired_common"] == 8
    assert census["c_dropped_for_pairing"] == 2
    assert {e["source_session_id"] for e in ck} == {e["source_session_id"] for e in rk}


def test_pairing_is_on_prompt_seed_and_truncation_not_prompt_alone():
    """Dropping one truncation must not drop the other for the same prompt."""
    c = [ex("s1", "sa", 0), ex("s1", "sa", 1)]
    r = [ex("s1", "sa", 1)]
    ck, rk, _ = intersect(c, r)
    assert len(ck) == 1 and ck[0]["truncation_index"] == 1


def test_seeds_are_kept_distinct_when_pairing():
    c = [ex("s1", "sa", 0), ex("s1", "sb", 0)]
    r = [ex("s1", "sb", 0)]
    ck, _, _ = intersect(c, r)
    assert [e["source_seed"] for e in ck] == ["sb"]


def test_both_arms_come_back_in_one_canonical_order():
    c = corpus(12)
    r = list(reversed(corpus(12)))
    ck, rk, _ = intersect(c, r)
    assert [pair_key(a) for a in ck] == [pair_key(b) for b in rk]


def test_duplicate_pair_keys_fail_loudly():
    dup = [ex("s1", "sa", 0), ex("s1", "sa", 0)]
    with pytest.raises(ValueError, match="duplicate pair keys in arm C"):
        intersect(dup, corpus(2))


# --------------------------------------------------------------- selection

def test_selection_preserves_composition_far_better_than_taking_the_first_n():
    c = corpus(200)
    ck, rk, _ = intersect(c, corpus(200))
    sel_c, sel_r, rep = select_paired(ck, rk, 60)
    assert rep.kept == 60 and len(sel_c) == len(sel_r) == 60
    assert [pair_key(a) for a in sel_c] == [pair_key(b) for b in sel_r]

    def task_shares(xs):
        n = len(xs)
        return {t: sum(e["data_type"] == t for e in xs) / n for t in TASKS}

    full = task_shares(ck)
    strat = task_shares(sel_c)
    naive = task_shares(sorted(ck, key=lambda e: e["data_type"])[:60])
    strat_err = max(abs(strat[t] - full[t]) for t in TASKS)
    naive_err = max(abs(naive[t] - full[t]) for t in TASKS)
    assert strat_err < naive_err
    assert strat_err < 0.05


def test_selection_is_deterministic_and_salt_sensitive():
    ck, rk, _ = intersect(corpus(80), corpus(80))
    a, _, _ = select_paired(ck, rk, 30)
    b, _, _ = select_paired(ck, rk, 30)
    c2, _, _ = select_paired(ck, rk, 30, salt="other")
    assert [pair_key(x) for x in a] == [pair_key(x) for x in b]
    assert [pair_key(x) for x in a] != [pair_key(x) for x in c2]


def test_selection_reports_stratum_drift_and_keeps_it_small():
    ck, rk, _ = intersect(corpus(240), corpus(240))
    _, _, rep = select_paired(ck, rk, 120)
    assert rep.dropped == 120
    assert rep.max_share_drift < 0.02
    assert rep.strata_before and rep.strata_after


def test_requesting_more_than_available_keeps_everything():
    ck, rk, _ = intersect(corpus(10), corpus(10))
    sel_c, _, rep = select_paired(ck, rk, 999)
    assert rep.kept == 10 and rep.dropped == 0 and len(sel_c) == 10


def test_selecting_from_mismatched_arms_is_refused():
    with pytest.raises(ValueError, match="intersect"):
        select_paired(corpus(5), corpus(4), 3)


def test_stratified_order_is_a_permutation():
    c = corpus(37)
    order = stratified_order(c)
    assert sorted(order) == list(range(37))


# ------------------------------------------------------------------ strata

def test_prefix_buckets_are_monotone_and_cover_the_tail():
    assert prefix_bucket(0) == "0-128"
    assert prefix_bucket(127) == "0-128"
    assert prefix_bucket(128) == "128-256"
    assert prefix_bucket(99999) == "4096+"


def test_stratum_separates_the_four_registered_dimensions():
    base = ex("s1", "sa", 0, "gsm8k", prefix=200)
    assert stratum(base) != stratum(ex("s1", "sa", 0, "code", prefix=200))
    assert stratum(base) != stratum(ex("s1", "sb", 0, "gsm8k", prefix=200))
    assert stratum(base) != stratum(ex("s1", "sa", 1, "gsm8k", prefix=200))
    assert stratum(base) != stratum(ex("s1", "sa", 0, "gsm8k", prefix=3000))


# ----------------------------------------------------------------- reports

def test_length_profile_reports_every_required_quantity():
    p = length_profile(corpus(50))
    for key in ("n", "supervised_continuation_tokens", "total_nonpadding_tokens",
                "prefix_tokens", "continuation_tokens", "prefix_buckets",
                "by_task_continuation_tokens", "truncation_fraction"):
        assert key in p, key
    assert p["supervised_continuation_tokens"] == sum(
        e["n_continuation_tokens"] for e in corpus(50))


def test_comparability_gate_fires_only_outside_the_pre_registered_tolerance():
    c = corpus(50)
    r = [dict(e, n_continuation_tokens=e["n_continuation_tokens"] + 1) for e in c]
    close = comparability_report(c, r, supervised_tolerance=0.05)
    assert close["within_tolerance"] is True
    far = [dict(e, n_continuation_tokens=e["n_continuation_tokens"] * 3) for e in c]
    assert comparability_report(c, far, supervised_tolerance=0.05
                                )["within_tolerance"] is False


def test_comparability_report_states_the_relative_depth_caveat():
    rep = comparability_report(corpus(4), corpus(4), supervised_tolerance=0.05)
    assert "FRACTION" in rep["note"] and "absolute prefix tokens differ" in rep["note"]
    assert rep["arm_c"]["n"] == rep["arm_r"]["n"] == 4

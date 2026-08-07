"""Independent per-arm selection to one supervised-token budget, R nested in C.

Attempt 4 measured why a common bundle count cannot work: R's supervised
continuation runs 1.66-1.76x longer than C's on the same bundle at the same cut
depth. Selecting to a common *token* budget therefore forces different bundle
counts, and the design choice is to accept that and keep R's selection nested
inside C's so the corpora share as much material as the budget allows.

These tests use an R/C length ratio in the measured range, so a selector that
silently reverts to a common bundle count fails them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.paired_corpus import (  # noqa: E402
    bundle_key, common_token_target, composition_report, select_nested_to_target,
    supervised_tokens,
)

TASKS = ("gsm8k", "glaive", "code")


def make_arm(arm, n_bundles, *, ratio=1.0, seed_label="sa", base=700):
    """One arm's examples: `n_bundles` bundles of two truncations each.

    Lengths vary deterministically per bundle so the refinement has something to
    work with, and `ratio` scales R against C the way the pod measured.
    """
    rows = []
    for i in range(n_bundles):
        task = TASKS[i % len(TASKS)]
        for j in range(2):
            cont = int((base + 37 * (i % 11) + 90 * j) * ratio)
            prefix = 200 + 130 * (i % 7) + 60 * j
            rows.append({
                "id": f"{task}-{i:05d}#t{j}", "arm": arm,
                "source_session_id": f"{task}-{i:05d}", "source_seed": seed_label,
                "truncation_index": j, "truncation_fraction": 0.25 + 0.3 * j,
                "data_type": task, "n_prefix_tokens": prefix,
                "n_continuation_tokens": cont,
                "n_total_tokens": prefix + cont,
            })
    return rows


def test_common_target_is_bound_by_the_weakest_pool():
    from aadistill.data.paired_corpus import as_bundles
    pools = {"C_sa": as_bundles(make_arm("C", 60)),
             "C_sb": as_bundles(make_arm("C", 40)),       # the binding pool
             "R_sa": as_bundles(make_arm("R", 60, ratio=1.7)),
             "R_sb": as_bundles(make_arm("R", 40, ratio=1.7))}
    t = common_token_target(pools, original=10**9)
    assert t["binding_pool"] == "C_sb"
    assert t["common_target"] == t["pool_totals"]["C_sb"]
    # An original target below every pool total must win instead.
    small = common_token_target(pools, original=1000)
    assert small["common_target"] == 1000 and small["reduced_from_original"] == 0


def test_both_arms_land_on_the_target_despite_the_length_asymmetry():
    c, r = make_arm("C", 220), make_arm("R", 220, ratio=1.7)
    target = 60_000
    out = select_nested_to_target(c, r, target)
    rep = out["report"]
    assert rep["worst_relative_deviation_from_target"] < 0.01, rep
    assert rep["arm_to_arm_relative_delta"] < 0.01, rep
    # The asymmetry must show up as different bundle counts, not equal ones.
    assert rep["r_selected_bundles"] < rep["c_selected_bundles"], rep


def test_r_selection_is_nested_inside_c():
    c, r = make_arm("C", 200), make_arm("R", 200, ratio=1.7)
    out = select_nested_to_target(c, r, 50_000)
    c_sel, r_sel = out["examples"]
    c_keys = {bundle_key(e) for e in c_sel}
    r_keys = {bundle_key(e) for e in r_sel}
    assert r_keys <= c_keys, "R must be a subset of C"
    assert out["report"]["shared_bundles"] == len(r_keys)
    assert out["report"]["c_only_bundles"] == len(c_keys) - len(r_keys)


def test_selection_is_deterministic():
    c, r = make_arm("C", 150), make_arm("R", 150, ratio=1.7)
    a = select_nested_to_target(c, r, 40_000)["report"]
    b = select_nested_to_target(c, r, 40_000)["report"]
    assert a == b


def test_bundles_stay_atomic():
    c, r = make_arm("C", 160), make_arm("R", 160, ratio=1.7)
    c_sel, r_sel = select_nested_to_target(c, r, 45_000)["examples"]
    for rows in (c_sel, r_sel):
        per = {}
        for e in rows:
            per.setdefault(e["source_session_id"], set()).add(e["truncation_index"])
        assert all(v == {0, 1} for v in per.values()), "a cut was selected alone"


def test_every_registered_stratum_is_still_represented():
    c, r = make_arm("C", 240), make_arm("R", 240, ratio=1.7)
    c_sel, r_sel = select_nested_to_target(c, r, 55_000)["examples"]
    comp = composition_report(c_sel, r_sel)
    for arm in ("C", "R"):
        assert set(comp[arm]["task"]) == set(TASKS), f"{arm} lost a task"
        assert set(comp[arm]["truncation_index"]) == {"0", "1"}
        assert len(comp[arm]["cut_depth_bucket"]) >= 2, "cut depth collapsed"
    # Selecting to a token budget must not skew the task mix between arms.
    for task, delta in comp["task_share_delta"].items():
        assert abs(delta) < 0.08, f"task {task} share differs by {delta}"


def test_r_holding_a_bundle_c_lacks_is_refused():
    c, r = make_arm("C", 40), make_arm("R", 50, ratio=1.7)
    with pytest.raises(ValueError, match="absent from C"):
        select_nested_to_target(c, r, 20_000)


def test_composition_report_separates_ce_from_kd_mask():
    c, r = make_arm("C", 80), make_arm("R", 80, ratio=1.7)
    c_sel, r_sel = select_nested_to_target(c, r, 25_000)["examples"]
    comp = composition_report(c_sel, r_sel)
    for arm in ("C", "R"):
        a = comp[arm]
        # kd_scope=all: KD covers the whole context, CE only the continuation.
        assert a["kd_mask_tokens"] > a["ce_mask_tokens"]
        assert a["kd_mask_tokens"] == a["nonpadding_tokens"]


def test_a_target_above_every_pool_is_the_pools_own_ceiling():
    """The selector must not invent tokens to reach an impossible target."""
    c, r = make_arm("C", 30), make_arm("R", 30, ratio=1.7)
    total_c = sum(supervised_tokens(b) for b in
                  __import__("aadistill.data.paired_corpus", fromlist=["x"])
                  .as_bundles(c))
    out = select_nested_to_target(c, r, total_c * 10)
    assert out["report"]["arm_c_supervised"] <= total_c
    assert out["report"]["c_selected_bundles"] == 30

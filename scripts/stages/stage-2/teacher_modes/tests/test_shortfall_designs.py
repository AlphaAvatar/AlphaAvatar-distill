"""The two $0 designs of the review of a143b7ed (2026-11-04): the six-category common-support control (section 2) and
the non-thinking state evaluation's limited-coverage plan (section 3). Their arithmetic, not their records."""
from __future__ import annotations

import pytest

from stages.teacher_modes import six_category_common_support as CS
from stages.teacher_modes import state_eval_limited_coverage as LC


def test_the_common_support_shares_keep_every_other_proportion_and_sum_to_one():
    reg = CS.registered()["shares"]
    cs = CS.cs_shares(reg, 0.023915)
    assert sum(cs.values()) == pytest.approx(1.0)
    assert cs["schema_arguments"] == 0.023915
    others = [c for c in reg if c != "schema_arguments"]
    ratios = {round(cs[c] / reg[c], 9) for c in others}
    assert len(ratios) == 1 and ratios.pop() == pytest.approx((1 - 0.023915) / 0.85)


def test_the_quota_is_the_lower_projection_rounded_down_and_every_arm_shares_it():
    rec = CS.recompute()
    low = rec["quota"]["hn_projection"][0]
    assert rec["quota"]["quota_tokens"] <= low < rec["quota"]["quota_tokens"] + 100
    shares = {round(a["categories"]["schema_arguments"]["common_support"] / a["tool_type_tokens"], 6)
              for a in rec["per_arm"].values()}
    assert len(shares) == 1                                   #: one composition for A1, A2 and C1
    for arm in rec["per_arm"].values():                       #: each arm's tool type total unchanged
        assert sum(c["common_support"] for c in arm["categories"].values()) == pytest.approx(arm["tool_type_tokens"],
                                                                                              abs=1.0)


def test_kish_counts_effective_episodes():
    assert LC.kish([10, 10, 10, 10]) == 4.0
    assert LC.kish([388, 163, 161, 125, 56, 40, 30, 25, 25, 25, 25]) == pytest.approx(4.98, abs=0.01)


def test_the_cell_weight_follows_the_two_level_unweighted_mean():
    quotas = {"schema_arguments": 1800.0, **{f"c{i}": (12000.0 - 1800.0) / 6 for i in range(6)}}
    w = LC.weights(1800.0, 1063.0, quotas, n_supporting_types=6, n_domains=5)
    assert w["registered"]["in_equal_domain_mean"] == pytest.approx(0.15 / 12 / 5)
    assert w["limited"]["within_tool_subtype"] == pytest.approx(1063 / (1063 + 10200), abs=1e-4)

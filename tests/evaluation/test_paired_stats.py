"""Paired statistics tests — including the ways these numbers mislead.

The load-bearing checks are that the bootstrap is deterministic, that it covers
a known difference, and that it stays honest when there is nothing to find. The
rest guard the failure modes: unpaired inputs, all-concordant pairs, and the
conjunction metric hiding which of its two halves moved.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.evaluation.paired_stats import (  # noqa: E402
    joint_rate, mcnemar_counts, paired_bootstrap_ci,
)


def ids(bits: str) -> dict:
    return {f"p{i}": b == "1" for i, b in enumerate(bits)}


def test_mcnemar_counts_only_discordant_pairs_carry_information():
    a = ids("11110000")
    b = ids("11001100")
    c = mcnemar_counts(a, b)
    assert c["n_paired"] == 8
    assert c["both_true"] == 2 and c["both_false"] == 2
    assert c["b_gained"] == 2 and c["b_lost"] == 2
    assert c["net"] == 0 and c["discordant"] == 4
    assert c["rate_a"] == c["rate_b"] == 0.5 and c["delta"] == 0.0


def test_identical_arms_have_no_discordant_pairs_and_a_zero_delta():
    a = ids("10110")
    c = mcnemar_counts(a, dict(a))
    assert c["discordant"] == 0 and c["delta"] == 0.0
    ci = paired_bootstrap_ci(a, dict(a), iterations=500)
    # Every resample of an all-zero difference vector is zero.
    assert ci["delta"] == 0.0 and ci["ci_low"] == 0.0 == ci["ci_high"]
    assert ci["ci_excludes_zero"] is False


def test_bootstrap_is_deterministic_and_seed_sensitive():
    a, b = ids("1100110011"), ids("1111111100")
    x = paired_bootstrap_ci(a, b, iterations=800, seed=1)
    y = paired_bootstrap_ci(a, b, iterations=800, seed=1)
    z = paired_bootstrap_ci(a, b, iterations=800, seed=2)
    assert x == y
    assert x["delta"] == z["delta"]          # the point estimate is not resampled
    assert (x["ci_low"], x["ci_high"]) != (z["ci_low"], z["ci_high"])


def test_interval_brackets_the_point_estimate_and_covers_a_large_effect():
    a = ids("0" * 100)
    b = ids("1" * 80 + "0" * 20)
    ci = paired_bootstrap_ci(a, b, iterations=2000)
    assert ci["delta"] == 0.8
    assert ci["ci_low"] <= ci["delta"] <= ci["ci_high"]
    assert ci["ci_excludes_zero"] is True
    assert ci["ci_low"] > 0.6


def test_a_tiny_effect_on_a_small_battery_does_not_exclude_zero():
    """The honest outcome for one prompt of difference out of 150."""
    a = ids("0" * 150)
    b = ids("1" + "0" * 149)
    ci = paired_bootstrap_ci(a, b, iterations=2000)
    assert ci["delta"] == pytest.approx(1 / 150, abs=1e-4)
    assert ci["ci_excludes_zero"] is False


def test_unpaired_inputs_fail_loudly():
    with pytest.raises(ValueError, match="not paired"):
        mcnemar_counts({"a": True}, {"b": False})
    with pytest.raises(ValueError, match="not paired"):
        paired_bootstrap_ci({"a": True}, {"b": False})


def test_only_shared_ids_are_compared():
    a = {"p1": True, "p2": False, "extra": True}
    b = {"p1": False, "p2": False, "other": True}
    c = mcnemar_counts(a, b)
    assert c["n_paired"] == 2 and c["b_lost"] == 1 and c["b_gained"] == 0


def test_joint_rate_is_a_conjunction_that_reports_its_halves():
    correct = ids("11110000")
    terminated = ids("11001100")
    j = joint_rate(correct, terminated)
    assert j["correct_and_naturally_terminated"] == 0.25   # p0, p1
    assert j["correct"] == 0.5 and j["natural_termination"] == 0.5
    # The two prompts that were right but never stopped are named, not hidden.
    assert j["correct_but_unterminated"] == 2
    assert j["per_sample"]["p0"] is True and j["per_sample"]["p2"] is False


def test_joint_rate_never_exceeds_either_marginal():
    correct = ids("1110101010")
    terminated = ids("1010101110")
    j = joint_rate(correct, terminated)
    assert j["correct_and_naturally_terminated"] <= min(
        j["correct"], j["natural_termination"])


def test_joint_rate_composes_with_the_paired_machinery():
    """The conjunction is per-sample, so it feeds straight into the CI."""
    a = joint_rate(ids("11110000"), ids("11110000"))["per_sample"]
    b = joint_rate(ids("11111100"), ids("11111100"))["per_sample"]
    c = mcnemar_counts(a, b)
    assert c["b_gained"] == 2 and c["b_lost"] == 0
    assert paired_bootstrap_ci(a, b, iterations=500)["delta"] == 0.25

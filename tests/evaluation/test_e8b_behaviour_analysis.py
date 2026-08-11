"""E8b's level-3 analyzer: the double difference, and what may not be claimed.

The interaction `(DC-DP) - (FC-FP)` is the one genuinely new statistic in E8b, and
it is the easiest to get wrong. Two ways in particular:

* subtracting two independently-bootstrapped intervals, which is not a CI on the
  difference. The four cells answer the **same** frozen 150 prompts, so the
  resampling must be joint — one draw of prompt ids applied to all four arms.
* pooling across the compression regimes as though hardware were controlled. It is
  not: DP/DC ran on A100 and FP/FC on L40S, so the nesting has to travel with the
  number.

These tests use synthetic per-sample dictionaries, so they pin the arithmetic and
the refusals without needing any generations.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/evaluation"))

pytest.importorskip("aadistill.evaluation.paired_stats")
mod = pytest.importorskip("analyze_e8b_behaviour")

IDS = [f"p{i}" for i in range(100)]


def arm(usable_frac: float, correct_frac: float = 0.0) -> dict:
    n_u = int(round(usable_frac * len(IDS)))
    n_c = int(round(correct_frac * len(IDS)))
    return {"per_sample": {
        "usable": {i: (k < n_u) for k, i in enumerate(IDS)},
        "correct": {i: (k < n_c) for k, i in enumerate(IDS)},
        "natural_termination": {i: (k < n_u) for k, i in enumerate(IDS)},
        "context_limit": {i: False for i in IDS},
        "generated_tokens": {i: 100 for i in IDS}}}


def arms_for(dp, dc, fp, fc) -> dict:
    out = {}
    for cell, frac in (("DP", dp), ("DC", dc), ("FP", fp), ("FC", fc)):
        for seed in mod.SEEDS:
            out[f"{cell}-{seed}"] = arm(frac)
    return out


def test_interaction_is_the_difference_of_the_two_paired_differences():
    # depth-only: +0.20   fully compressed: -0.10   interaction: +0.30
    r = mod.interaction(arms_for(0.40, 0.60, 0.50, 0.40), "usable", iterations=200)
    for seed in mod.SEEDS:
        s = r["per_seed"][seed]
        assert s["depth_only_delta"] == pytest.approx(0.20, abs=1e-9)
        assert s["fully_compressed_delta"] == pytest.approx(-0.10, abs=1e-9)
        assert s["interaction"] == pytest.approx(0.30, abs=1e-9)
    assert r["pooled_interaction"] == pytest.approx(0.30, abs=1e-9)
    assert r["seed_consistent"] is True


def test_a_map_that_helps_equally_in_both_regimes_has_no_interaction():
    r = mod.interaction(arms_for(0.40, 0.60, 0.30, 0.50), "usable", iterations=200)
    assert r["pooled_interaction"] == pytest.approx(0.0, abs=1e-9)


def test_interaction_carries_the_hardware_nesting_caveat():
    r = mod.interaction(arms_for(0.40, 0.60, 0.50, 0.40), "usable", iterations=200)
    caveat = r["hardware_caveat"].lower()
    assert "nested" in caveat
    # It must name the confound it cannot exclude, not merely say "be careful".
    assert "hardware" in caveat and "bridge" in caveat


def test_interaction_refuses_on_a_missing_arm():
    a = arms_for(0.40, 0.60, 0.50, 0.40)
    del a["FC-sb"]
    r = mod.interaction(a, "usable", iterations=200)
    assert "incomplete" in r and "FC-sb" in r["incomplete"]
    assert "pooled_interaction" not in r


def test_bootstrap_ci_brackets_the_point_estimate():
    r = mod.interaction(arms_for(0.40, 0.60, 0.50, 0.40), "usable", iterations=500)
    s = r["per_seed"]["sa"]
    lo, hi = s["bootstrap_ci"]
    assert lo <= s["interaction"] <= hi


def test_bootstrap_is_seeded_and_reproducible():
    a = arms_for(0.40, 0.60, 0.50, 0.40)
    first = mod.interaction(a, "usable", iterations=300)
    second = mod.interaction(a, "usable", iterations=300)
    assert (first["per_seed"]["sa"]["bootstrap_ci"]
            == second["per_seed"]["sa"]["bootstrap_ci"])


def test_a_zero_effect_ci_includes_zero():
    r = mod.interaction(arms_for(0.50, 0.50, 0.50, 0.50), "usable", iterations=500)
    assert r["per_seed"]["sa"]["ci_excludes_zero"] is False


# --- the conditional bridge trigger ---------------------------------------

def bridge_with(depth_delta, full_delta, *, consistent=True, floor_ok=True):
    d = {"pooled_delta": depth_delta, "seed_consistent": consistent}
    f = {"pooled_delta": full_delta, "seed_consistent": consistent}
    return mod.bridge_trigger({"contrasts": {
        "depth_only": {"usable": d}, "fully_compressed": {"usable": f}}})


def test_bridge_fires_only_on_the_registered_reversal():
    r = bridge_with(+0.20, -0.15)
    assert r["fires"] is True and r["sign_reversal"] is True


def test_bridge_does_not_fire_when_both_directions_agree():
    assert bridge_with(+0.20, +0.15)["fires"] is False


def test_bridge_does_not_fire_on_the_opposite_reversal():
    # DC worse while FC better is a reversal, but not the registered trigger.
    r = bridge_with(-0.20, +0.15)
    assert r["sign_reversal"] is True
    assert r["fires"] is False


def test_bridge_does_not_fire_inside_the_floor():
    # Direction is right but both effects are noise-sized.
    r = bridge_with(+0.02, -0.02)
    assert r["both_exceed_floor"] is False
    assert r["fires"] is False


def test_bridge_does_not_fire_when_seeds_disagree():
    assert bridge_with(+0.20, -0.15, consistent=False)["fires"] is False


def test_bridge_is_not_evaluable_before_both_contrasts_exist():
    r = mod.bridge_trigger({"contrasts": {"depth_only": {}}})
    assert r["evaluable"] is False


def test_bridge_is_defined_on_recovered_behaviour_not_step0():
    r = bridge_with(+0.20, -0.15)
    assert "step-0" in r["note"].lower() or "step0" in r["note"].lower()


def test_floors_match_the_registered_values():
    assert mod.FLOORS == {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}


def test_the_binding_inclusion_mask_is_pinned():
    assert mod.EXPECTED_MASK == (
        "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba")


def test_fp_is_declared_retained_rather_than_retrained():
    stems, regime, depth_map, hardware = mod.CELLS["FP"]
    assert stems == {"sa": "E1-1.60M-sa", "sb": "E1-1.60M-sb"}
    assert (regime, depth_map, hardware) == (
        "fully_compressed", "positional", "L40S_48GB")


def test_each_contrast_is_within_one_hardware_class():
    for a, b, _label in mod.CONTRASTS:
        assert mod.CELLS[a][3] == mod.CELLS[b][3], (
            f"{a} vs {b} spans hardware classes; that is not a clean contrast")


def test_incomplete_message_names_the_absent_arms_not_neither():
    a = arms_for(0.40, 0.60, 0.50, 0.40)
    for seed in mod.SEEDS:
        del a[f"FC-{seed}"]
    out = mod.contrast("FP", "FC", a, iterations=100)
    # FP is fully present; the message must not claim we have "neither".
    assert "neither" not in out["incomplete"]
    assert "FC-sa" in out["incomplete"] and "FC-sb" in out["incomplete"]
    assert "FP-sa" not in out["incomplete"]

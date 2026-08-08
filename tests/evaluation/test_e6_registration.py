"""Experiment 6 is evaluation-only and its comparison is fragile in one place.

Two failure modes are worth a durable test rather than a comment.

**The evaluation rung.** The three-mode harness samples its 150 examples from
whatever rung it is handed, and the E1 rungs hold 1,174 / 1,944 / 2,941 blocks.
Handing each arm its own training rung would resample the battery per arm and
end the comparison — silently, because every arm would still report 150 prompts
and a mask hash. E4 pinned the rung to 860000 for exactly this reason; E6 must
keep it pinned, and the mask must be asserted after every arm.

**The reuse rule.** E6 re-scores two arms from artifacts a previous session
produced. That is only sound while the evaluator is byte-identical to the one
that produced them, and while those artifacts are actually present and complete.
Both are checked here rather than assumed, because the alternative is a
comparison between two different instruments wearing the same metric names.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRATION = REPO / "logs/e6_registration.json"
DRIVER = REPO / "scripts/pod/e6_driver.py"

pytestmark = pytest.mark.skipif(
    not REGISTRATION.is_file(), reason="E6 registration not written yet")

BINDING_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
# The floors carried unchanged from E3/E4/E5. They are measured seed spreads, not
# preferences, and E6 may not renegotiate them after seeing its own numbers.
REGISTERED_FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGISTRATION.read_text())


def test_registration_declares_evaluation_only(reg):
    assert reg["kind"] == "evaluation-only"
    assert reg["trains_anything"] is False
    assert reg["modifies_any_checkpoint"] is False


def test_driver_pins_the_evaluation_rung_to_the_shared_battery():
    src = DRIVER.read_text()
    assert re.search(r"^EVAL_RUNG\s*=\s*860000\b", src, re.M), \
        "the E6 driver must pin the evaluation rung to 860000"
    # The arm's own training rung must never reach the harness.
    assert '--rung", EVAL_RUNG' in src or "--rung\", EVAL_RUNG" in src, \
        "the harness must be invoked with the pinned rung, not a per-arm rung"
    assert 'arm["rung"]' not in src, \
        "the driver must not pass a per-arm training rung to the harness"


def test_driver_asserts_the_binding_mask_after_every_arm():
    src = DRIVER.read_text()
    assert BINDING_MASK in src, "the driver must carry the binding mask"
    assert "mask != EXPECTED_MASK" in src, \
        "the driver must fail an arm whose inclusion mask is not the binding one"


def test_every_arm_names_a_full_length_weight_hash(reg):
    for alias, arm in reg["arms"].items():
        sha = arm["weights_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", sha), \
            f"{alias}: weights_sha256 is {len(sha)} chars, not a full sha256"


def test_the_scale_curve_is_a_single_recipe_at_three_rungs(reg):
    """Only the rung, seed and derived schedule may differ across E1 arms."""
    e1 = {a: v for a, v in reg["arms"].items() if v["lineage"].startswith("E1")}
    assert len(e1) == 6, f"expected six E1 arms, got {sorted(e1)}"
    assert {v["rung"] for v in e1.values()} == {1600000, 2960000, 5500000}
    for alias, arm in e1.items():
        cfg = json.loads((REPO / arm["config"]).read_text())
        assert cfg["rung"] == arm["rung"], alias
        assert cfg["seed"] == arm["seed"], alias
        assert cfg["student_path"].endswith("qwen3_0p6b_init_v0/checkpoint"), alias
        assert cfg["loss"] == {"ce_weight": 0.25, "kd_weight": 1.0,
                               "kd_temperature": 1.0, "kd_scope": "all"}, alias
        assert cfg["optim"]["lr"] == 5e-05, alias
        assert cfg["block_len"] == 8192, alias


def test_the_anchor_is_a_different_objective_and_is_labelled_as_one(reg):
    """P2-1.60M shares the rung but not the recipe; it is not a curve point."""
    anchors = {a: v for a, v in reg["arms"].items() if v["lineage"].startswith("external")}
    assert len(anchors) == 2, sorted(anchors)
    for alias, arm in anchors.items():
        cfg = json.loads((REPO / arm["config"]).read_text())
        assert cfg["loss"]["ce_weight"] == 1.0 and cfg["loss"]["kd_weight"] == 0.25, \
            f"{alias}: the anchor is the CE-heavy objective"
        assert alias not in reg["scale_curve"]


def test_reused_arms_have_complete_retained_generations(reg):
    """An arm that is not regenerated must already have every artifact it needs."""
    for alias, arm in reg["arms"].items():
        if arm["generate"]:
            continue
        d = REPO / "artifacts/audit/three_mode" / arm["retained_three_mode"]
        for f in ("free.generations.jsonl", "oracle.generations.jsonl", "report.json"):
            assert (d / f).is_file(), f"{alias}: retained artifact {f} is missing"
        report = json.loads((d / "report.json").read_text())
        assert report["inclusion"]["mask_sha256"] == BINDING_MASK, alias
        assert report["rung"] == 860000, alias
        assert report["context"] == 8192, alias
        assert report["decoding"] == {"greedy": True, "temperature": 0.0}, alias
        rows = [json.loads(l) for l in (d / "free.generations.jsonl").open() if l.strip()]
        assert len(rows) == 150, f"{alias}: {len(rows)} free generations, expected 150"
        assert all("token_ids" in r and "raw" in r for r in rows), \
            f"{alias}: retained generations must carry raw text and token ids"


def test_reuse_is_only_sound_because_the_evaluator_is_unchanged(reg):
    v = reg["evaluator_unchanged_since_reused_artifacts"]
    assert v["verified"] is True, (
        "E6 reuses artifacts from an earlier commit; that is only valid while "
        f"the evaluator is byte-identical. diff: {v.get('diff_stat')!r}")


def test_floors_are_the_registered_ones_and_no_composite_is_invented(reg):
    assert reg["interpretation"]["floors"] == REGISTERED_FLOORS
    assert "no_new_composite" in reg["interpretation"]


def test_the_mask_is_model_independent_and_reproduces(reg):
    a = reg["frozen_assets"]
    assert a["inclusion_mask_sha256"] == BINDING_MASK
    assert a["inclusion_mask_matches_binding"] is True
    assert a["sampled"] == 150
    assert a["eval_rung"] == 860000
    assert sum(a["inclusion_mask_by_task"].values()) == 150


def test_cost_is_bounded_by_the_authorization(reg):
    auth = reg["authorization"]
    assert auth["fits"] is True
    assert auth["hard_backstop_usd"] <= auth["remaining_authorized_usd"]
    assert "any training" in auth["excludes"]

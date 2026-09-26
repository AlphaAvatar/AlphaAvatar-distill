"""The pilot driver, EXECUTED, not inspected.

`--toy` runs the identical seven-step sequence at a CPU geometry: the same
`replay_prefix`, `verified_parent` and `run_arm`, the same operator, the same
comparator, the same committed records. Nothing is stubbed.

It has already earned its keep. Three process-global registries — adapters,
operators, calibration profiles — are all EMPTY in a fresh interpreter and all
full in any pytest session, because some sibling test fills them first. The
driver was missing two of the three, and each failure surfaced only after the
stage before it had succeeded. On a pod that is after the root teacher has
been downloaded and placed on the GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/pod/c3_batching_pilot_driver.py"


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """One real toy execution, shared by the assertions below."""
    out = tmp_path_factory.mktemp("pilot")
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path_factory.mktemp("home")),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--toy", "--device", "cpu",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return out, json.loads((out / "pilot_result.json").read_text()), done


def test_the_whole_sequence_runs_in_a_bare_interpreter(toy_run):
    """A BARE one: no pytest, no conftest, no sibling test having filled a
    registry. That is the environment the pod has."""
    _out, record, done = toy_run
    assert record["verdict"]
    assert "Traceback" not in done.stderr, done.stderr[-2000:]


def test_the_prefix_runs_exactly_once(toy_run):
    _out, record, _ = toy_run
    prefix = record["stages"]["prefix"]
    assert prefix["replay_count"] == 1
    assert prefix["execution"]["micro_batch_size"] == 1, (
        "the frozen parent was built by the one-item path")
    assert len(prefix["steps"]) == 3


def test_both_arms_score_from_the_same_verified_parent(toy_run):
    _out, record, _ = toy_run
    a, b = record["stages"]["causal-B1"], record["stages"]["causal-B4"]
    parent = record["stages"]["prefix"]["parent_artifact_digest"]
    for arm in (a, b):
        ev = arm["suffix_evidence"]
        assert ev["premise"] == "verified"
        assert ev["parent_artifact_digest"] == parent
        assert ev["executed_step_indices"] == [3], (
            "an arm re-ran part of the prefix")
    assert (a["suffix_evidence"]["parent_checkpoint"]
            == b["suffix_evidence"]["parent_checkpoint"])


def test_the_arm_order_is_the_predeclared_one(toy_run):
    _out, record, _ = toy_run
    scope = json.loads((REPO / "logs/stages/stage-1/phase_c3/pilots/"
                        "batching-adoption/v1/scope.json").read_text())
    assert record["arm_order"] == scope["arm_order"]
    assert record["stages"][record["arm_order"][0]]["position"] == 1
    assert record["stages"][record["arm_order"][1]]["position"] == 2


def test_each_arm_records_the_fields_the_return_owes(toy_run):
    _out, record, _ = toy_run
    for arm_id in ("causal-B1", "causal-B4"):
        a = record["stages"][arm_id]
        for field in ("path_id", "path_hash", "scorer_seconds",
                      "physical_forward_invocations", "item_forward_equivalents",
                      "padded_positions", "child_artifact_digest",
                      "child_weights_digest", "child_checkpoint_path"):
            assert field in a, f"{arm_id} does not record {field}"
        assert a["step"]["selection"]["causal_head_evidence"]


def test_the_two_arms_have_different_path_hashes(toy_run):
    _out, record, _ = toy_run
    assert (record["stages"]["causal-B1"]["path_hash"]
            != record["stages"]["causal-B4"]["path_hash"])


def test_b4_issues_fewer_invocations_for_the_same_work(toy_run):
    _out, record, _ = toy_run
    a, b = record["stages"]["causal-B1"], record["stages"]["causal-B4"]
    assert b["physical_forward_invocations"] < a["physical_forward_invocations"]
    assert a["item_forward_equivalents"] == b["item_forward_equivalents"]
    assert a["padded_positions"] == 0 and b["padded_positions"] > 0


def test_the_gate_is_applied_to_the_measured_scorer_clock(toy_run):
    _out, record, _ = toy_run
    c = record["stages"]["comparison"]
    assert c["speed"]["b1_scorer_seconds"] == (
        record["stages"]["causal-B1"]["scorer_seconds"])
    assert c["speed"]["b4_scorer_seconds"] == (
        record["stages"]["causal-B4"]["scorer_seconds"])
    assert "measured" in c["speed"]["_basis"]
    assert record["speed_gate_threshold"] == 1.25


def test_the_verdict_is_one_of_the_three_predeclared(toy_run):
    _out, record, _ = toy_run
    assert record["verdict"] in {
        "B4_NOT_WORTH_ADOPTION_PILOT",
        "B4_STRUCTURALLY_EQUIVALENT_AND_FASTER",
        "B4_FASTER_AND_STRUCTURALLY_DIFFERENT_RECOVERY_TRIGGERED"}
    assert record["recovery_triggered"] is record["verdict"].endswith(
        "RECOVERY_TRIGGERED")


def test_the_structural_comparison_carries_its_denominators(toy_run):
    _out, record, _ = toy_run
    st = record["stages"]["comparison"]["structural"]
    for level in ("layers", "gqa_groups", "retained_slots"):
        assert st[level]["total"] > 0, f"{level} has no denominator"
    assert "rank_correlation" in st and "score_drift" in st


def test_every_stage_is_persisted_as_it_completes(toy_run):
    """A pilot that dies in stage 5 still measured stage 4."""
    out, record, _ = toy_run
    assert set(record["stages"]) == {"prefix", "causal-B1", "causal-B4",
                                     "comparison"}
    assert (out / "pilot_result.json").is_file()
    assert not (out / "pilot_failure.json").exists()


def test_the_summary_omits_the_raw_values_and_names_their_digest(toy_run):
    """`per_item_kl` is 60k floats at the real geometry: it belongs with the
    artifacts, and dropping it from a record must not make it unverifiable."""
    import hashlib

    out, _record, _ = toy_run
    summary = json.loads((out / "pilot_summary.json").read_text())
    ev = summary["stages"]["causal-B4"]["step"]["selection"][
        "causal_head_evidence"]
    assert "per_item_kl" not in ev
    assert "omitted here" in ev["_per_item_kl"]
    assert ev["head_scores"] and ev["gqa_decisions"], (
        "the summary dropped the aggregate evidence too")
    raw = (out / "pilot_result.json").read_bytes()
    assert summary["raw_evidence"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["raw_evidence"]["file"] == "pilot_result.json"


def test_the_summary_is_small_enough_to_review(toy_run):
    out, _record, _ = toy_run
    assert (out / "pilot_summary.json").stat().st_size < (
        out / "pilot_result.json").stat().st_size


def test_a_driver_failure_writes_why(tmp_path):
    """The broad `except` exists because a paid pod has died inside an
    exception this driver did not name, and the evidence went with it."""
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    out = tmp_path / "o"
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--toy", "--device", "cpu",
         "--out", str(out), "--repo", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 1
    failure = json.loads((out / "pilot_failure.json").read_text())
    assert "error" in failure and failure["error"]
    assert "FAILED" in done.stdout


def test_the_driver_registers_all_three_global_registries():
    """Named, because two of the three were missing and a pytest session
    cannot see it: a sibling test has always filled them first."""
    src = DRIVER.read_text()
    for call in ("register_builtin_adapters()", "register_builtin_operators()",
                 "register_builtin_profiles()", "causal_kl.register("):
        assert call in src, f"the driver never calls {call}"

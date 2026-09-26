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
    assert set(record["stages"]) == {"prefix_steps", "prefix", "causal-B1",
                                     "causal-B4", "comparison"}
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


# --- the four repairs attempt a1 bought -----------------------------------

def test_the_required_mixtures_are_derived_from_the_pilots_own_steps():
    """a1 staged ONE mixture and died at WIDTH. The prefix uses TWO, which is
    the exact non-uniformity a review had already corrected in the pilot
    module — so the list must be asked of the steps, not written beside them."""
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--required-inputs"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env={"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"})
    assert done.returncode == 0, done.stderr
    rows = [json.loads(l) for l in done.stdout.splitlines() if l.strip()]
    got = {r["profile_id"]: r for r in rows}
    assert set(got) == {"calib.domain_balanced@v1", "calib.reasoning_heavy@v2"}, (
        f"the derived mixture set is {sorted(got)}; the prefix uses two "
        "profiles and the causal step reuses the first")
    for r in rows:
        assert r["items_path"] and r["items_file_sha256"], r
        assert r["materialized"] is True


def test_each_entry_carries_the_profiles_own_pinned_hash():
    """So a stager verifies against the profile, not against a constant
    copied next to it."""
    from aadistill.initialization.calibration.profiles import get_profile

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    from experiments.calibration import register_builtin_profiles

    register_builtin_profiles()
    sys.path.insert(0, str(REPO / "scripts/pod"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("c3drv", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for entry in mod.required_profiles(REPO):
        profile = get_profile(entry["profile_id"])
        assert entry["items_file_sha256"] == profile.items_file_sha256
        assert entry["items_path"] == profile.items_path


def test_check_inputs_resolves_the_real_profiles(tmp_path):
    """The check a1 did not have. The toy preflight it DID have supplies
    `calibration_items` explicitly and never touches the profile registry,
    so it passed while the mixture the prefix needed was absent."""
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    done = subprocess.run([sys.executable, str(DRIVER), "--check-inputs"],
                          cwd=REPO, capture_output=True, text=True,
                          timeout=300, env=env)
    if not (REPO / "artifacts/stage1").is_dir():
        pytest.skip("artifacts/ is gitignored and not built in this tree")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "calib.reasoning_heavy@v2" in done.stdout, (
        "the check did not resolve the mixture whose absence cost a1")
    assert "calib.domain_balanced@v1" in done.stdout


def test_check_inputs_REFUSES_when_a_mixture_is_absent(tmp_path):
    """Against an empty repo root: the failure mode, reproduced at $0."""
    import shutil

    fake = tmp_path / "repo"
    (fake / "configs/calibration").mkdir(parents=True)
    shutil.copy(REPO / "configs/calibration/profiles.json",
                fake / "configs/calibration/profiles.json")
    (fake / "logs/stages/stage-1/phase_c3/pilots/batching-adoption/v1").mkdir(
        parents=True)
    shutil.copy(REPO / "logs/stages/stage-1/phase_c3/pilots/batching-adoption"
                       "/v1/scope.json",
                fake / "logs/stages/stage-1/phase_c3/pilots/batching-adoption"
                       "/v1/scope.json")
    env = {"PYTHONHASHSEED": "7", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--check-inputs", "--repo", str(fake)],
        cwd=REPO, capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode == 30, done.stdout + done.stderr
    assert "INPUTS UNAVAILABLE" in done.stdout
    assert "MISSING" in done.stdout


def test_each_prefix_step_is_recorded_as_it_lands(toy_run):
    """a1 ran DEPTH and FFN for 24 minutes and recorded neither, because the
    prefix stage was written only after all three returned."""
    _out, record, _ = toy_run
    steps = record["stages"]["prefix_steps"]
    assert len(steps) == 3
    assert [s["kind"] for s in steps] == ["DEPTH", "FFN", "RESIDUAL_WIDTH"]
    for s in steps:
        assert s["artifact_digest"]


def test_the_launcher_pushes_the_derived_mixtures_and_fetches_no_weights():
    """Both halves of the transport defect a1 exposed, asserted on the source.

    It cannot be executed without a pod, so it is checked structurally: the
    launcher must derive the list, verify it locally before creating anything,
    push it, and it must NOT `scp -r` the whole output tree — which pulled
    9.1 GiB of checkpoints over a 0.72 MB/s uplink while the pod billed.
    """
    src = (REPO / "scripts/pod/c3_batching_pilot_launch.sh").read_text()
    assert "--required-inputs" in src, "the launcher hardcodes its inputs"
    assert "items_file_sha256" in src, "the push is not verified against the profile"
    assert "/workspace/mixtures" in src, "the mixtures are never pushed"
    assert "scp -r" not in src, (
        "the launcher recursively fetches the output tree; that is 9.1 GiB of "
        "checkpoints nothing on the dev box consumes")
    assert "evidence.tgz" in src, "the evidence is not packed before fetching"
    #: And the pack must exclude the work tree explicitly.
    assert "-not -path './*/work/*'" in src


def test_the_remote_payload_checks_real_inputs_before_the_gpu_work():
    src = (REPO / "scripts/pod/c3_batching_pilot_remote.sh").read_text()
    assert "--check-inputs" in src
    #: BEFORE the toy preflight, and before the pilot.
    assert src.index("--check-inputs") < src.index("--toy")
    assert "/workspace/mixtures" in src
    assert "e8_inputs_20260810" not in src, (
        "the payload still fetches a mixture from the relay by name; the list "
        "is derived and pushed now")


# --- the create must not turn an error into a pod id ----------------------

def test_the_pod_id_regex_does_not_match_the_providers_error_text():
    """`specifications` is fourteen lowercase letters.

    The fallback was `\\b[a-z0-9]{13,16}\\b`, so "There are no longer any
    instances available with the requested specifications" yielded a pod id.
    A watchdog was started against it and the launcher polled a pod that had
    never existed. Every real RunPod id contains a digit.
    """
    import re

    src = (REPO / "scripts/pod/c3_batching_pilot_launch.sh").read_text()
    m = re.search(r"POD_ID=\$\(echo \"\$CREATE\" \| grep -oE '([^']+)'", src)
    assert m, "no fallback pod-id pattern found in the launcher"
    pattern = m.group(1).replace("\\\\", "\\")
    error = ("Error: There are no longer any instances available with the "
             "requested specifications. Please refresh and try again.")
    hits = [h for h in re.findall(pattern, error) if 13 <= len(h) <= 16]
    assert not hits, f"the fallback still matches the error text: {hits}"
    #: And it still finds a real id.
    ok = [h for h in re.findall(pattern, 'pod "7ucb8jox3buc29" created')
          if 13 <= len(h) <= 16]
    assert ok == ["7ucb8jox3buc29"], ok


def test_the_launcher_confirms_the_pod_with_the_provider():
    """Parsing is a guess; the provider is the authority. A pod id this
    script believes in but the provider has never heard of leaves the
    watchdog guarding nothing."""
    src = (REPO / "scripts/pod/c3_batching_pilot_launch.sh").read_text()
    create_at = src.index("runpodctl create pod")
    #: The START of the watchdog, not the string "watchdog.py" — that also
    #: appears in the teardown's pid check, which is EARLIER in the file and
    #: made this assertion compare the wrong two positions.
    watchdog_at = src.index("setsid nohup")
    confirm_at = src.index("the provider does not list a pod")
    assert create_at < confirm_at < watchdog_at, (
        "the pod must be confirmed with the provider BEFORE a watchdog is "
        "started against it")
    assert "no longer any instances" in src, (
        "the launcher does not recognise the provider's refusal text")

"""E6b trains, so its pod contract differs from E6's in ways worth pinning.

E6 asserted the training pack was *absent*; E6b needs it, plus the teacher for
KD. That inversion is exactly the kind of thing a copied script gets wrong, and
the failure — training against a pack that is not there, or computing KD against
a teacher that was never downloaded — costs a full setup cycle to discover.

Also pinned: the evaluation rung stays 860000 (the battery must not be resampled
from the training rung), the driver refuses to continue from a trained
checkpoint, and the launcher retrieves the produced weights before teardown,
since they are the only artifacts that cannot be regenerated without paying again.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"
SETUP = POD / "e6b_setup.sh"
DRIVER = POD / "e6b_driver.py"
LAUNCH = POD / "e6b_launch.sh"


def test_setup_stages_the_training_pack_under_both_names():
    """The trainer reads `ladder_uniform`; the battery reads `ladder_uniform_probe`."""
    s = SETUP.read_text()
    assert 'test -f "$REPO/artifacts/stage3/ladder_uniform/blocks.npz"' in s, \
        "E6b trains, so the training pack must be present — E6 asserted the opposite"
    assert 'test -f "$REPO/artifacts/stage3/ladder_uniform_probe/blocks.npz"' in s
    assert "ladder_uniform'" in s and "audit.jsonl" in s, \
        "ladder_blocks reads audit.jsonl; a pack without it fails only on a pod"


def test_setup_downloads_the_teacher_because_kd_needs_it():
    s = SETUP.read_text()
    assert "Qwen/Qwen3-4B-Thinking-2507" in s
    assert "TEACHER_REVISION" in s, "the teacher must be pinned to a revision"
    assert "mark TEACHER_READY" in s


def test_setup_installs_ninja_and_verifies_the_frozen_assets():
    s = SETUP.read_text()
    assert re.search(r"apt-get install[^\n]*\bninja-build\b", s)
    assert "command -v ninja" in s
    for sha in ("6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
                "2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd",
                "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"):
        assert sha in s, f"setup does not verify {sha[:12]}…"
    assert "MASK MISMATCH" in s


def test_setup_validates_the_arms_before_any_step_is_taken():
    s = SETUP.read_text()
    assert "validate_e6b_arms.py" in s
    assert s.index("validate_e6b_arms.py") < s.index("mark SETUP_DONE")


def test_driver_pins_the_evaluation_rung_to_the_shared_battery():
    src = DRIVER.read_text()
    assert re.search(r"^EVAL_RUNG\s*=\s*860000\b", src, re.M), \
        "the battery rung must stay pinned; the training rung would resample it"
    assert "2960000" in src, "the driver should assert the training rung too"
    assert 'cfg["rung"] == 2960000' in src
    assert "mask != EXPECTED_MASK" in src


def test_driver_refuses_to_continue_from_a_trained_checkpoint():
    src = DRIVER.read_text()
    assert 'cfg["student_path"].endswith("qwen3_0p6b_init_v0/checkpoint")' in src
    assert "must fork from the Stage 1 init" in src


def test_driver_asserts_the_registered_objective_from_the_file_that_trains():
    src = DRIVER.read_text()
    assert '"ce_weight": 1.0' in src and '"kd_weight": 0.25' in src
    assert 'cfg["loss"] == OBJECTIVE' in src


def test_validate_and_train_are_blocking_stages():
    src = DRIVER.read_text()
    blocking = src[src.index("BLOCKING = ("):src.index("def main()")]
    assert '"validate"' in blocking and '"train"' in blocking, \
        "a failed validation or training must stop the run, not fall through to eval"
    order = src[src.index("STAGES = {"):src.index("BLOCKING = (")]
    assert order.index('"validate"') < order.index('"train"') < order.index('"three_mode"')


def test_launcher_has_no_devbox_upload_machinery():
    """Every E6b input is on the relay; an upload path would be dead code."""
    s = LAUNCH.read_text()
    for token in ("UPLOAD_PID", "ckpt_upload_ok", "ckpt_local", "model.safetensors.zst"):
        assert token not in s, f"{token} is E6 machinery E6b does not need"


def test_launcher_fetches_the_produced_weights_before_teardown():
    s = LAUNCH.read_text()
    assert "e6b_ckpt_hashes.txt" in s, "weights must be hashed on the pod first"
    fetch = s.index("fetching checkpoints")
    assert fetch < s.index("\nteardown", fetch), \
        "weights are the only artifact that cannot be regenerated for free"


def test_launcher_budget_layers_and_backstop_cover_a_training_session():
    s = LAUNCH.read_text()
    assert "securePrice" in s
    assert "--terminate-after" in s
    backstop = int(re.search(r"BACKSTOP_MINUTES=\$\{BACKSTOP_MINUTES:-(\d+)\}", s).group(1))
    # 2 arms x 2916 steps x ~3.6 s is ~352 min of training alone.
    assert backstop >= 400, f"backstop {backstop} min cannot cover two 2.96M arms"
    poll = int(re.search(r"POLL_LIMIT_MIN=\$\{POLL_LIMIT_MIN:-(\d+)\}", s).group(1))
    assert poll < backstop, "the launcher must give up before RunPod kills the pod"


def test_status_file_is_consistent_across_the_three_components():
    names = set()
    for f in (SETUP, DRIVER, LAUNCH):
        names |= set(re.findall(r"/?workspace/(e6b\.status)", f.read_text()))
        names |= set(re.findall(r"STATUS=\$WS/(e6b\.status)", f.read_text()))
    assert names == {"e6b.status"}, f"components disagree on the status file: {names}"

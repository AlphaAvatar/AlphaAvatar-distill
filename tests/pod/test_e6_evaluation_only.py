"""E6 spends money to evaluate, never to train — and that must be mechanical.

"Evaluation-only" is a claim about what a pod does, and claims about pods have
been wrong before in this repository. So the E6 driver proves it by parsing the
scripts it will execute and refusing to continue if any of them contains an
executable optimizer step, and the setup script declines to stage the training
ladder pack at all. Both guarantees are asserted here, plus the ordering that
makes the proof blocking rather than advisory.

The staging script is also covered: it is the only place where a checkpoint from
the dev box and a checkpoint from the relay meet, and the whole comparison rests
on both routes ending in bytes that match the registration.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/pod/e6_driver.py"
SETUP = REPO / "scripts/pod/e6_setup.sh"
LAUNCH = REPO / "scripts/pod/e6_launch.sh"
STAGE = REPO / "scripts/pod/e6_stage_checkpoints.py"


def test_the_no_training_proof_runs_first_and_is_blocking():
    src = DRIVER.read_text()
    order = src[src.index("STAGES = {"):src.index("def main()")]
    assert order.index('"notrain"') < order.index('"three_mode"'), \
        "the no-training proof must run before anything generates"
    blocking = src[src.index("BLOCKING = ("):src.index("def main()")]
    assert '"notrain"' in blocking, \
        "a failed no-training proof must stop the run, not be logged and skipped"


def test_the_driver_actually_parses_the_scripts_it_executes():
    """The proof must cover every script the driver runs, not a fixed list."""
    src = DRIVER.read_text()
    tree = ast.parse(src)
    executed = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "EXECUTED" for t in n.targets))
    declared = {c.value for c in executed.value.elts}
    invoked = set(re.findall(r'"(scripts/[a-z_/]+\.py)"', src))
    assert invoked <= declared, (
        f"the driver runs {sorted(invoked - declared)} but does not parse them "
        "for optimizer steps")


def test_the_executed_scripts_are_free_of_optimizer_steps():
    """The same check the pod runs, run here so a regression fails on CPU."""
    src = DRIVER.read_text()
    tree = ast.parse(src)
    executed = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "EXECUTED" for t in n.targets))
    for rel in (c.value for c in executed.value.elts):
        path = REPO / rel
        assert path.is_file(), f"{rel} does not exist"
        for node in ast.walk(ast.parse(path.read_text(), filename=rel)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("step", "backward")):
                raise AssertionError(
                    f"{rel}:{node.lineno} calls .{node.func.attr}() — E6 would "
                    "no longer be evaluation-only")


def test_setup_refuses_to_stage_the_training_pack():
    src = SETUP.read_text()
    assert "test ! -d \"$REPO/artifacts/stage3/ladder_uniform\"" in src, \
        "setup must assert the training ladder pack is absent"
    assert "ladder_uniform_probe" in src, "the probe pack is what the battery needs"


def test_driver_fails_when_the_training_pack_is_present():
    src = DRIVER.read_text()
    assert "training_pack_present" in src
    assert 'raise AssertionError("the training ladder pack is present' in src


def test_setup_verifies_the_frozen_battery_assets_by_hash():
    src = SETUP.read_text()
    for sha in ("6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
                "2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd",
                "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"):
        assert sha in src, f"setup does not verify {sha[:12]}…"
    assert "MASK MISMATCH" in src, \
        "setup must rebuild the inclusion mask in the pod environment and fail on drift"


def test_staging_refuses_a_hash_mismatch_from_either_store():
    src = STAGE.read_text()
    assert 'sys.exit("CHECKPOINT HASH MISMATCH' in src
    # Both routes must land in the same verification, not just the relay one.
    assert '"relay"' in src and '"devbox"' in src
    body = src[src.index("for alias, arm in sorted"):src.index("args.out.parent")]
    assert "failures.append" in body and body.index("stage_relay") < body.index("got = sha256")
    assert body.index("stage_devbox") < body.index("got = sha256"), \
        "dev-box checkpoints must be hashed by the same code path as relay ones"


def test_staging_only_stages_arms_the_registration_marks_for_generation():
    src = STAGE.read_text()
    assert 'if not arm["generate"]:' in src, \
        "a reused arm must never be staged; its result comes from retained files"
    assert "registration declares" in src, \
        "the staged count must be reconciled against the registration"


def test_setup_stages_relay_only_so_it_cannot_race_the_devbox_upload():
    """A real race, caught before it cost anything.

    The launcher uploads the two dev-box checkpoints in the background so 4.8 GB
    overlaps `uv sync`. Setup therefore cannot verify them — a partial file is
    present but wrong, and staging would exit on the hash. Setup stages `relay`
    only; the launcher joins the upload and re-runs over both stores.
    """
    setup = SETUP.read_text()
    launch = LAUNCH.read_text()
    assert "--stores relay" in setup, \
        "setup must stage relay checkpoints only; the dev-box upload is in flight"
    # The launcher's own call must NOT restrict the stores, or the sb arms are
    # never staged at all and half the seeds silently vanish.
    call = launch[launch.index("e6_stage_checkpoints.py"):]
    call = call[:call.index("|| {")]
    assert "--stores" not in call, \
        "the launcher must stage both stores; restricting it drops the sb arms"
    assert "e6_checkpoint_manifest.json" in call, \
        "the driver reads e6_checkpoint_manifest.json; the launcher must write it"
    assert launch.index("wait \"$UPLOAD_PID\"") < launch.index("e6_stage_checkpoints.py"), \
        "the upload must be joined before the full staging pass runs"


def test_staging_reverifies_bytes_before_skipping_work():
    """The second pass may skip a staged arm only after re-hashing it."""
    src = STAGE.read_text()
    block = src[src.index("already = ("):src.index("got = sha256")]
    assert "sha256(dest" in block and 'arm["weights_sha256"]' in block, \
        "a skip must be justified by the bytes on disk, not by the file existing"


def test_launcher_budget_layers_are_all_present():
    src = LAUNCH.read_text()
    assert "securePrice" in src, "the price guard must read securePrice, not community"
    assert "--terminate-after" in src, "a RunPod-side deadline must exist"
    assert "teardown" in src and "ALL_DONE" in src, \
        "the launcher must delete the pod on completion; pods idle-bill"
    assert "pgrep -f \"[e]6_driver.py\"" in src, \
        "a dead driver with no terminal marker must end the run, not idle"
    assert "--ports \"22/tcp\"" in src, "readiness needs a real TCP 22 mapping"


def test_launcher_verifies_artifacts_before_deleting_the_pod():
    src = LAUNCH.read_text()
    fetch = src.index("bundling artifacts on the pod")
    assert src.index("POD_SHA=", fetch) < src.index("teardown\n", fetch), \
        "artifacts must be digest-verified while the pod still exists"


def test_launcher_does_not_fetch_weights_because_none_are_written():
    src = LAUNCH.read_text()
    assert "checkpoints/" not in src, \
        "E6 writes no checkpoints; a weight-fetch step would be dead code hiding intent"

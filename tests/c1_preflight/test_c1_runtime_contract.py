"""What a C1 pod must prove before it is allowed to spend money on science.

This directory is the ONLY thing the paid C1 pod runs. Until 2026-09-13 it ran
the entire repository — 3892 tests, 16 minutes of a billing L40S — and attempt
14 died in it on two tests that had nothing to do with C1: one asserted a
dev-box-only artifact store, the other a `.venv` no pod checkout has. Most of
the machinery that failure produced (skip-set digests, simulator/host parity,
readiness divergence analysis) existed only to make "run the whole repository on
a pod" work at all.

So the selection is now positive and small. A check earns a place here only if a
C1 pod could plausibly fail it in a way that costs money or invalidates the
result:

* **A — C1 runtime-critical.** The launcher/driver seam, the frozen scientific
  identities, the staged battery, the paths the driver writes to, the
  contracts the scoring path loads. Here.
* **B — generic infrastructure the run depends on.** Run layout and artifact
  specification. The genuinely necessary subset is here; the rest is not.
* **C — unrelated repository tests.** Docs, log organisation, budget
  arithmetic, other experiments' history. Development and CI, never a pod.
* **D — dev-box-only or historical.** Anything whose premise is a host-local
  store, a Phase-A/B retained checkpoint, or the simulator itself. A pod does
  not own those premises and must not be asked about them.

Everything in C and D still runs — in development, in convergence, and in the
launch-bound validation of this directory. What changed is that a paid
experimental environment is no longer responsible for regression-testing the
whole repository.

**Nothing here may skip.** A test that cannot answer on both the dev box and a
pod belongs in `test_c1_gpu_runtime.py`, which is declared GPU-only and is the
one module whose skip set legitimately differs between the two.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests/pod"))


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


# --- the launcher/driver seam ------------------------------------------------
#
# Attempt 7 was the first C1 attempt ever to clear the pod test gate. It then
# died for $0.4231 because the launcher emitted `--stage all` and the driver's
# parser has no such option: argparse exited 2 before a line of the driver ran.

def test_the_driver_and_launcher_both_import():
    """The two modules a session cannot start without."""
    import autoinit_c1_driver
    import autoinit_c1_launch

    assert hasattr(autoinit_c1_driver, "build_parser")
    assert hasattr(autoinit_c1_launch, "spec")


def test_the_driver_parser_accepts_exactly_what_the_launcher_emits():
    """The seam attempt 7 paid for, executed rather than described.

    The command is BUILT by the launcher and PARSED by the driver's real
    parser, so an option that exists on one side and not the other fails here
    instead of at argparse exit 2 on a billing pod.
    """
    import shlex
    import types

    import autoinit_c1_driver as driver
    import autoinit_c1_launch as launcher
    from session_specs import session_args

    args = session_args(launcher)
    ctx = types.SimpleNamespace(
        args=args, price=1.09, spent_usd=0.0, image_digest="preflight",
        auth=types.SimpleNamespace(hard_cap_usd=15.1475,
                                   authorization_id="preflight"),
        evidence={}, scr=Path("/tmp/c1pre"))
    #: The launcher reads exactly one field off the plan. Supplied rather than
    #: constructed, the same way `tests/pod/test_launcher_driver_cli_seam.py`
    #: does — that module is the thorough dev-side version of this check; this
    #: one exists because the pod is a different machine with a different
    #: interpreter and a different install, and the seam has to hold there.
    plan = types.SimpleNamespace(soft_stop_usd=14.78)
    command = launcher.driver_command(ctx, plan)
    assert "--stage" not in command, command

    #: Everything after the script path is the driver's own argv.
    tokens = shlex.split(command)
    script = next(i for i, t in enumerate(tokens)
                  if t.endswith("autoinit_c1_driver.py"))
    parsed = driver.build_parser().parse_args(tokens[script + 1:])
    assert parsed.authorized_usd > 0 and parsed.soft_stop_usd > 0


def test_the_session_spec_builds_and_declares_its_markers():
    """A spec that cannot be built cannot be launched."""
    import autoinit_c1_launch as launcher
    from session_specs import session_args

    spec = launcher.spec(session_args(launcher))
    assert spec.setup.required_env
    assert "SETUP_DONE" in spec.setup.setup_markers
    assert spec.markers.success == "ALL_DONE"
    assert spec.authorization_path.endswith("/governance/authorization.json")


# --- the frozen scientific identities ----------------------------------------
#
# These protect scientific validity directly: an arm, a seed, a replay digest or
# a battery that moved between the sweep and the pod would silently change what
# the run measures.

def test_the_arms_seeds_and_replay_digests_are_the_frozen_ones():
    """Two arms, three derived seeds, two replay digests. Nothing else.

    Checked through the launcher's own gate, which is the code the launch
    actually runs, rather than through a second transcription of the same four
    constants.
    """
    import autoinit_c1_launch as launcher
    from experiments.phase_c1.probe_results import ARMS

    assert ARMS == ("incumbent", "treatment"), ARMS
    ok, why = launcher.frozen_c1_science_gate(None)
    assert ok, why
    assert "950/850" in why and "seeds" in why, why


def test_the_staged_battery_is_the_frozen_950_over_850():
    """The battery's BYTES, which travel, not the store they were frozen in.

    `battery_staged_gate` compares the staged copy against the canonical
    out-of-tree store before a provider is created. That store is host-local by
    design and no pod has it, which is exactly what attempt 14 failed on here.
    What a pod can and must answer is whether the bytes it received are the
    frozen ones.
    """
    identity = load("logs/stages/stage-1/phase_c1/plans/battery.json")
    staged = REPO / "artifacts/stage3/c1_confirmation_v1"
    assert staged.is_dir(), f"{staged} was not staged; the session has no battery"
    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["content_sha256"] == identity["content_sha256"]
    assert (manifest["n_prompts"], manifest["n_scorable_prompts"]) == (950, 850)


def test_the_preregistration_still_describes_this_executable():
    """Self-hash and harness digest, checked with the launcher's own gate."""
    import autoinit_c1_launch as launcher

    ok, why = launcher.preregistration_gate(None)
    assert ok, why


# --- what the driver loads once it starts ------------------------------------

def test_the_scoring_and_behaviour_contracts_import():
    """The evaluation path, imported before six trainings rather than after.

    Attempt 8's driver died inside a stage; an import that only happens after
    70 minutes of training is an import that fails at the worst moment.
    """
    from aadistill.evaluation.usable_rollout import summarize, usable  # noqa: F401
    from experiments.phase_c1.scoring import (
        C1_METRIC_CONTRACT, c1_scoring_contract, validate_c1_battery,  # noqa: F401
    )

    contract = c1_scoring_contract(REPO)
    assert contract["contract"], contract
    for axis in ("correct_overall", "usable_rollout_rate", "correct_given_usable"):
        assert axis in C1_METRIC_CONTRACT, axis


def test_the_artifact_spec_never_requires_what_a_failed_run_cannot_produce():
    """A FAILED spec demanding post-training classes blocks teardown.

    Which is the single most expensive way to be wrong on a pod: the science
    already failed, and the collection step then refuses to let go of the GPU.
    """
    import autoinit_c1_launch as launcher

    failed = load(launcher.SPEC_FAILED)
    required = [p for p in failed.get("patterns", [])
                if p.get("required") and p.get("class") in launcher.POST_TRAINING_CLASSES]
    assert not required, required


def test_the_run_layout_resolves_where_the_driver_will_write():
    from experiments.run_layout import rel_run_dir

    d = rel_run_dir("phase_c1", "attempt15", "1")
    assert d == "logs/stages/stage-1/phase_c1/runs/attempt15", d


def test_no_check_here_can_skip():
    """The whole sweep/pod parity problem, reduced to one property.

    A repository-wide skip-predicate audit existed to prove that the launch-bound
    sweep and the paid pod would decide every skip the same way. It had to:
    3892 tests ran on both, and a predicate that answered differently produced a
    divergence nobody could explain — attempt 14's sweep certified 138 skips and
    its pod produced 115.

    Thirteen tests with no skip predicate in them make that question trivial.
    Checked by AST rather than by reading, because a comment mentioning `skipif`
    is not a `skipif` and a regex cannot tell them apart.
    """
    import ast

    offenders = []
    for f in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("skip", "skipif",
                                                                 "xfail"):
                offenders.append(f"{f.name}:{node.lineno} pytest.{node.attr}")
    assert not offenders, (
        "a C1 preflight check that can skip is a check the pod may silently not "
        f"run: {offenders}")


# --- the repository the pod checked out --------------------------------------

def test_the_checkout_is_the_commit_the_session_declares():
    """A bundle that unpacked to the wrong commit runs the wrong executable."""
    import os

    declared = os.environ.get("SESSION_COMMIT")
    if not declared:
        #: On the dev box there is no session; HEAD is trivially itself. The
        #: assertion that matters is that the two agree WHEN a session declares
        #: one, which is every pod.
        declared = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == declared, f"checkout is {head}, the session declares {declared}"

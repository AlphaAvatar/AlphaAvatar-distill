"""D1 owns a launch-bound sweep contract, and it is derived rather than copied.

AGENTS.md P8.3 requires one `launch_bound` sweep when implementation, metadata
and grant are final and an experiment is about to launch. D1 had no
`pod_environment.py` and no registry entry, so

    record_pod_environment.py --experiment phase_d1 --kind launch_bound

was refused as an unknown experiment: the sweep was not "not yet run", it was
**not expressible** through the canonical mechanism. The six-gate pre-provider
dry run is not a substitute — it validates the orchestration before a provider
is contacted, whereas the sweep validates the session's BLOCKING TEST GATE
under the pod-like environment the simulator produces.

Two classes of fact are asserted here, and the second is the one that cost
money elsewhere:

* the contract exists, dispatches, and is run-owned with no global pointer;
* every experiment fact in it is READ from the real `SessionSpec` or from the
  module that owns it, never transcribed — the session id, the pod test
  selection, the canonical bundle deriver and the executable closure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit", "tests"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from stages.phase_d1 import d1_authorization as A  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1 import pod_environment as PE  # noqa: E402

RUN_ID = "d1_sweep_contract_probe"


def _recorder():
    from shared.pod import record_pod_environment as R

    return R


class TestTheRegistryDispatchesPhaseD1:
    """The missing line was the whole blocker."""

    def test_phase_d1_is_declared(self):
        assert "phase_d1" in _recorder().EXPERIMENTS

    def test_it_resolves_to_this_module_and_factory(self):
        module, factory = _recorder().EXPERIMENTS["phase_d1"]
        assert module == "stages.phase_d1.pod_environment"
        assert factory == "sweep_contract"
        assert getattr(PE, factory) is PE.sweep_contract

    def test_the_cli_accepts_it_as_a_choice(self):
        """`--experiment` uses `choices=sorted(EXPERIMENTS)`, so an unregistered
        experiment is rejected by argparse before the tool runs at all."""
        out = subprocess.run(
            [sys.executable, str(REPO / "scripts/shared/pod/record_pod_environment.py"),
             "--experiment", "phase_d1", "--kind", "launch_bound", "--help"],
            capture_output=True, text=True, cwd=REPO, timeout=180,
            env={"PYTHONPATH": f"{REPO}/src:{REPO}/scripts", "HOME": str(Path.home()),
                 "PATH": "/usr/bin:/bin"})
        assert out.returncode == 0, out.stderr[-2000:]
        assert "phase_d1" in out.stdout

    def test_the_contract_builds_through_the_generic_entry_point(self):
        sweep = _recorder().sweep_contract("phase_d1", RUN_ID, "1",
                                           "launch_bound")
        assert sweep.experiment_id == "phase_d1"
        assert sweep.launcher_module == "autoinit_d1_launch"

    def test_an_unregistered_experiment_still_refuses_by_name(self):
        """The guard that made this blocker legible rather than mysterious."""
        with pytest.raises(SystemExit, match="not declared"):
            _recorder().sweep_contract("phase_zz", RUN_ID, "1", "launch_bound")


class TestNothingIsTranscribedFromTheLauncher:
    """Every experiment fact is read from the thing that owns it."""

    def test_the_session_id_is_the_real_specs(self):
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        assert PE.session_id() == spec.session_id
        assert PE.sweep_contract(RUN_ID, "1").session_id == spec.session_id

    def test_the_pod_selection_is_the_real_specs_test_paths(self):
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        assert PE.pod_test_selection() == tuple(spec.setup.test_paths)
        assert PE.readiness_groups().watched == tuple(spec.setup.test_paths)

    def test_the_bundle_deriver_is_d1s_own_not_c1s(self):
        """D1 has its own TransportSpec; the sweep builds the production setup
        environment from this, so borrowing C1's would put another
        experiment's bundle name into the environment pytest runs under."""
        commit = "b354d386191d85796c7c9bc00ab48498d1986e43"
        assert PE.bundle_name(commit) == D1S.canonical_bundle_name(commit)
        assert PE.bundle_name(commit) == f"aad_autoinit_{commit[:8]}.bundle"

    def test_the_harness_is_the_live_d1_closure(self):
        live = A.d1_current_executable(REPO)
        got = PE.harness(REPO)
        assert got["digest"] == live["digest"]
        assert got["n_files"] == live["n_files"]
        assert PE.harness_digest(REPO) == live["digest"]

    def test_the_harness_is_a_callable_not_a_captured_value(self):
        """C1's real launch path once refused with "no harness_digest provider
        was supplied" while every test passed one explicitly."""
        sweep = PE.sweep_contract(RUN_ID, "1")
        assert callable(sweep.harness)
        assert callable(sweep.record.harness_digest)
        assert callable(sweep.bundle_name)

    def test_the_experiment_and_stage_ids_come_from_the_session_module(self):
        assert PE.EXPERIMENT_ID == D1S.EXPERIMENT_ID == "phase_d1"
        assert PE.STAGE_ID == D1S.STAGE_ID


class TestTheRecordIsRunOwned:
    """No global pointer, and no borrowing of C1's."""

    def test_the_sweep_declares_no_pointer(self):
        assert PE.sweep_contract(RUN_ID, "1").pointer_path is None

    def test_the_record_path_is_inside_the_run(self):
        from shared.run_layout import rel_run_dir

        rel = PE.record_path_for(RUN_ID, "1")
        assert rel == f"{rel_run_dir('phase_d1', RUN_ID, '1')}/governance/readiness.json"

    def test_a_run_id_is_required(self):
        """A shared path is how one attempt's evidence comes to describe
        another's tree."""
        with pytest.raises(PE.D1ReadinessError, match="without a run id"):
            PE.record_path_for(None)

    def test_it_is_a_different_file_from_the_launch_readiness_record(self):
        """Two records, two questions: can the pod's test gate pass, versus is
        this launch the one that was prepared."""
        from stages.phase_d1 import autoinit_d1_launch as L

        assert PE.record_path_for(RUN_ID, "1") != L.readiness_path_for(RUN_ID)

    def test_no_c1_pointer_is_named_anywhere_in_the_contract(self):
        """`logs/state/current.json :: latest_verification` is derived from the
        Phase-C1 readiness pointer. D1 must not make itself answer to it."""
        src = (REPO / "scripts/stages/stage-1/phase_d1/pod_environment.py").read_text()
        for foreign in ("c1_pod_environment_verification",
                        "c1_readiness_pointer", "c3_pod_environment",
                        "a3_pod_environment"):
            assert foreign not in src, foreign

    def test_the_schema_is_d1s_own(self):
        """A record of one experiment must not satisfy another's verifier."""
        assert PE.SCHEMA == "aadistill.autoinit.d1_pod_environment/v1"
        for other in ("c1_pod_environment", "c3_pod_environment",
                      "a3_pod_environment", "c2_full_search"):
            assert other not in PE.SCHEMA


class TestTheEnvironmentDigestCoversWhatTheGateRuns:
    """D1's selection is OUTSIDE `tests/`, which the generic digest walks."""

    def test_d1s_own_tests_are_named(self):
        """Zero of them are in the executable closure and zero are under
        `tests/`, so without naming them a D1 test could change and the sweep
        that certified it would still verify."""
        named = set(PE.pod_test_environment_files())
        d1_tests = {str(p.relative_to(REPO)) for p in
                    (REPO / "scripts/stages/stage-1/phase_d1/tests").rglob("*.py")}
        assert d1_tests, "the selection has no tests; the probe is vacuous"
        assert d1_tests <= named, sorted(d1_tests - named)

    def test_none_of_them_is_in_the_executable_closure(self):
        """The premise of the test above. A file in BOTH digests is measured
        twice and could satisfy one check while failing the other."""
        closure = {f["path"] for f in A.d1_current_executable(REPO)["files"]}
        named = set(PE.pod_test_environment_files())
        assert not (closure & named), sorted(closure & named)

    def test_the_simulator_and_both_applicable_conftests_are_named(self):
        named = set(PE.pod_test_environment_files())
        assert "scripts/shared/pod/simulate_pod_env.sh" in named
        #: The conftests that apply to a selection outside `tests/`: the rootdir
        #: one and the experiments-tree one. `tests/conftest.py` does not apply
        #: and is covered anyway by the generic `tests/**` walk.
        assert "conftest.py" in named
        assert "scripts/conftest.py" in named

    def test_every_named_file_exists(self):
        for rel in PE.pod_test_environment_files():
            assert (REPO / rel).is_file(), rel

    def test_the_digest_moves_when_a_d1_test_moves(self, tmp_path):
        """Measured, not asserted: the whole point of naming them."""
        from aadistill.runtime.pod_environment import pod_test_environment_digest

        before = pod_test_environment_digest(
            REPO, named_files=PE.pod_test_environment_files())["digest"]
        #: A copy of the tree with one D1 test byte changed.
        import shutil

        clone = tmp_path / "repo"
        for rel in ("tests", "scripts/stages/stage-1/phase_d1/tests",
                    "scripts/pod", "scripts/experiments"):
            src = REPO / rel
            dst = clone / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(REPO / "conftest.py", clone / "conftest.py")
        victim = (clone / "scripts/stages/stage-1/phase_d1/tests"
                  / "test_d1_pod_environment.py")
        victim.write_text(victim.read_text() + "\n# one changed byte\n")
        after = pod_test_environment_digest(
            clone, named_files=PE.pod_test_environment_files())["digest"]
        assert before != after


class TestTheSweepCommandIsThePodsCommand:
    """The shared gap D1 is the third session to hit."""

    def test_the_declared_selection_uses_test_paths(self):
        from aadistill.runtime.staging_contract import (
            derive_contract, pod_pytest_command,
        )
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        contract = derive_contract(spec.setup, session_id=spec.session_id)
        selection = contract["pytest_selection"]
        #: The pod runs `pytest ${SESSION_TEST_PATHS:-tests/} -q <ignores>`.
        assert "scripts/stages/stage-1/phase_d1/tests" in selection
        assert selection == pod_pytest_command(spec.setup)

    def test_it_is_not_the_core_suite(self):
        """The defect: `pytest tests/ -q` swept the core suite while the pod
        gate runs only D1's preflight directory."""
        from aadistill.runtime.staging_contract import derive_contract
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        selection = derive_contract(spec.setup,
                                    session_id=spec.session_id)["pytest_selection"]
        assert " tests/ " not in f" {selection} "

    def test_test_paths_is_hashed_into_the_staging_contract(self):
        """It was absent, so a session could change which suite its pod gate
        runs without invalidating a single readiness record."""
        import dataclasses

        from aadistill.runtime.staging_contract import derive_contract
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        base = derive_contract(spec.setup, session_id=spec.session_id)
        assert base["test_paths"] == list(spec.setup.test_paths)
        moved = dataclasses.replace(spec.setup, test_paths=("tests/",))
        assert derive_contract(moved, session_id=spec.session_id)["digest"] \
            != base["digest"]

    def test_the_invocation_check_catches_a_path_mismatch(self):
        """`[] == []` on the ignore lists used to satisfy this check while the
        paths differed, which is exactly how the mismatch survived."""
        from aadistill.runtime.staging_contract import derive_contract
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        R = _recorder()
        spec = L.spec(session_args(L))
        contract = derive_contract(spec.setup, session_id=spec.session_id)
        setup_env = spec.setup_environment(session_commit="a" * 40,
                                           bundle="b.bundle")
        wrong = ".venv/bin/python -m pytest tests/ -q"
        out = R.check_invocation_matches(
            contract, setup_env, wrong,
            {**setup_env, "HIDDEN_PATHS": "x", "PODSIM_CMD": wrong})
        assert out["problems"], "a path mismatch was accepted"
        assert any("test selection mismatch" in p for p in out["problems"])

    def test_the_real_invocation_is_accepted(self):
        """The control: the command the recorder builds must pass its own
        check, or the test above would pass for the wrong reason."""
        from aadistill.runtime.staging_contract import (
            derive_contract, pod_pytest_command,
        )
        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        R = _recorder()
        spec = L.spec(session_args(L))
        contract = derive_contract(spec.setup, session_id=spec.session_id)
        setup_env = spec.setup_environment(session_commit="a" * 40,
                                           bundle="b.bundle")
        cmd = pod_pytest_command(spec.setup, python=".venv/bin/python")
        out = R.check_invocation_matches(
            contract, setup_env, cmd,
            {**setup_env, "HIDDEN_PATHS": "x", "PODSIM_CMD": cmd})
        assert out["problems"] == [], out["problems"]


class TestTheReadinessGroupsAreD1sAndNotInherited:

    def test_no_c1_battery_or_readiness_roles(self):
        groups = PE.readiness_groups()
        assert groups.expected_skips == {}
        assert groups.must_pass == {}
        assert groups.staged_role_nodeid is None
        assert tuple(groups.known_non_environment_skips) == ()

    def test_the_watched_prefix_is_d1s_selection(self):
        assert PE.readiness_groups().watched == (
            "scripts/stages/stage-1/phase_d1/tests",)

    def test_a_failure_fails_the_verdict(self):
        out = PE.evaluate_sweep({
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_a": "failed",
        })
        assert out["verdict"] == "FAIL"

    def test_an_unexpected_environment_skip_fails_the_verdict(self):
        """The claim D1 makes: its selection passes with NO unexpected
        environment skips."""
        out = PE.evaluate_sweep({
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_a": "passed",
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_b": "skipped",
        }, {"scripts/stages/stage-1/phase_d1/tests/test_x.py::test_b":
            "no HF token"})
        assert out["verdict"] == "FAIL"
        assert out["unexpected_environment_skips"] == [
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_b"]

    def test_a_clean_pass_passes(self):
        out = PE.evaluate_sweep({
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_a": "passed",
            "scripts/stages/stage-1/phase_d1/tests/test_y.py::test_b": "passed",
        })
        assert out["verdict"] == "PASS"
        assert out["problems"] == []
        assert out["unexpected_environment_skips"] == []

    def test_a_skip_outside_the_watched_selection_is_not_a_finding(self):
        """The pod does not run those modules at all."""
        out = PE.evaluate_sweep({
            "scripts/stages/stage-1/phase_d1/tests/test_x.py::test_a": "passed",
            "tests/data/test_other.py::test_z": "skipped",
        })
        assert out["verdict"] == "PASS", out["problems"]


class TestAnEmptySelectionIsRefused:
    """A readiness group over zero modules passes vacuously."""

    def test_a_session_with_no_test_paths_is_refused(self, monkeypatch):
        import dataclasses

        from stages.phase_d1 import autoinit_d1_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        stripped = dataclasses.replace(
            spec, setup=dataclasses.replace(spec.setup, test_paths=()))
        monkeypatch.setattr(PE, "_spec", lambda: stripped)
        with pytest.raises(PE.D1ReadinessError, match="no test_paths"):
            PE.pod_test_selection()

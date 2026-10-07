"""The D1 replay's readiness sweep must model the D1 replay's pod.

Split out of `tests/architecture/test_sweep_registry_models_its_own_session.py`,
which keeps the GENERIC property -- every sweep-registry entry binds the
launcher whose session id it declares, no two entries bind the same launcher,
and every declared session launcher is either registered or named. AGENTS.md
2.8a: a historical experiment that exposed a generic core defect leaves its
regression in core, and what belongs to the experiment is its own state. The
core suite's own boundary test refused the combined file, correctly.

WHAT THIS SESSION'S STATE IS. The replay was swept four times through
`--experiment phase_d1`, which binds the SEARCH's launcher, so each of those
readiness records derived the staged view of a different session:

    SEARCH   staged 11   hidden 1851
    REPLAY   staged 12   hidden 1850   + artifacts/stage1/d1_replay_plan.json

The missing file is the resolved plan the replay's driver is invoked with via
`--plan`. A MISSING registry entry would have been loud -- the recorder refuses
with "unknown experiment" -- but a WRONG one sweeps successfully and produces a
record indistinguishable from a correct one.

The second class here is a separate failure found in the same pass: the
provider-account gate that the 2026-10-06 exhaustion produced was never
declared by this session, so it did not run for any of those four subruns.

These run on CPU, create nothing, and contact no provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/pod", "scripts/autoinit",
               "scripts/experiments/stage-1", "tests"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))


def _registry() -> dict:
    import record_pod_environment as R

    return dict(R.EXPERIMENTS)


def _contract(experiment: str):
    import record_pod_environment as R

    return R.sweep_contract(experiment, run_id="registry-probe", stage_id="1")


def _launcher_spec(name: str):
    from support.session_specs import load_session_launcher, session_args

    mod = load_session_launcher(name)
    return mod.spec(session_args(mod))


def _staged_view(name: str) -> dict:
    from aadistill.runtime.staging_contract import derive_contract, describe

    spec = _launcher_spec(name)
    return describe(derive_contract(spec.setup, session_id=spec.session_id),
                    REPO)


class TestTheD1ReplayModelsItsOwnPod:
    """The specific thing four subruns got wrong, asserted specifically."""

    def test_it_has_its_own_entry(self):
        assert "phase_d1_replay" in _registry()

    def test_that_entry_binds_the_replay_launcher(self):
        assert _contract("phase_d1_replay").launcher_module == \
            "autoinit_d1_replay_launch"

    def test_the_search_entry_still_binds_the_search_launcher(self):
        assert _contract("phase_d1").launcher_module == "autoinit_d1_launch"

    def test_the_two_sessions_stage_different_things(self):
        """Non-vacuity: if these ever coincide, the entry above protects
        nothing and this file is testing a distinction that does not exist."""
        search = _staged_view("autoinit_d1_launch")
        replay = _staged_view("autoinit_d1_replay_launch")
        assert search["digest"] != replay["digest"]
        assert search["n_staged_files"] != replay["n_staged_files"]

    def test_the_replays_staged_view_contains_the_plan_its_driver_reads(self):
        replay = _staged_view("autoinit_d1_replay_launch")
        assert any(d.endswith("d1_replay_plan.json")
                   for d in replay["local_asset_destinations"])

    def test_the_searchs_does_not(self):
        search = _staged_view("autoinit_d1_launch")
        assert not any(d.endswith("d1_replay_plan.json")
                       for d in search["local_asset_destinations"])

    def test_the_replay_record_is_filed_under_its_own_experiment(self):
        """Not among the search's attempts. Its run directory, session record
        and stage attribution are all under `phase_d1_replay/`."""
        from experiments.phase_d1 import replay_pod_environment as RPE

        path = RPE.record_path_for("d1_replay_002", "1")
        assert path.startswith("logs/stages/stage-1/phase_d1_replay/runs/")

    def test_its_harness_is_the_one_the_authorization_binds(self):
        """One owner. A readiness record describing a different executable
        from the authorization is two identities for one session."""
        from issue_d1_replay_authorization import HARNESS_FILES
        from experiments.phase_d1 import replay_pod_environment as RPE

        harness = RPE.harness(REPO)
        assert harness["n_files"] == len(HARNESS_FILES)
        assert [f["path"] for f in harness["files"]] == sorted(HARNESS_FILES)


class TestTheReplaySessionArmsTheAccountGate:
    """Separate failure, same session, found in the same pass: the gate that
    the 2026-10-06 account exhaustion produced was never declared here, so it
    did not run for four paid subruns."""

    def test_the_requirement_is_declared(self):
        spec = _launcher_spec("autoinit_d1_replay_launch")
        assert spec.budget.account_balance_required_usd is not None, (
            "the replay creates a pod without asking whether the provider "
            "account can pay for it; that cost $8.1716 once")

    def test_it_is_derived_from_the_priced_plan_and_not_typed(self):
        spec = _launcher_spec("autoinit_d1_replay_launch")
        requirement = spec.budget.account_balance_required_usd
        assert callable(requirement), (
            "a constant here would not track a re-pricing, and this session "
            "has already been re-priced once against its campaign remainder")
        plan = spec.budget.plan(price_per_hour=1.09, authorized_usd=2.5856)
        import autoinit_d1_replay_launch as L

        assert requirement(plan) == round(
            plan.hard_terminate_usd + L.ACCOUNT_OPERATIONAL_RESERVE_USD, 4)

    def test_it_exceeds_the_sessions_own_ceiling(self):
        """Non-vacuity: a requirement at or below the ceiling would be
        satisfied by an account that cannot fund the run."""
        spec = _launcher_spec("autoinit_d1_replay_launch")
        plan = spec.budget.plan(price_per_hour=1.09, authorized_usd=2.5856)
        assert spec.budget.account_balance_required_usd(plan) > \
            plan.hard_terminate_usd

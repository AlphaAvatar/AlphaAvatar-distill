"""A readiness sweep must model the session that will actually launch.

THE FAILURE. The D1 replay was swept four times through
`--experiment phase_d1`, and that registry entry binds `autoinit_d1_launch` --
the SEARCH's launcher. So every one of those readiness records derived the
staged view of a *different session*: the search stages three frozen
calibration/eval assets, the replay stages those three PLUS the resolved plan
its driver is invoked with via `--plan`. Each record certified a pod that
lacked the one file the replay cannot start without.

    SEARCH   staged 11   hidden 1851
    REPLAY   staged 12   hidden 1850   + artifacts/stage1/d1_replay_plan.json

A MISSING registry entry is loud: the recorder refuses with "unknown
experiment", which is how five other sessions got theirs. A WRONG one is
silent, because it sweeps successfully and produces a record that looks exactly
like a correct one. It cost nothing for four subruns because no test in the
selection read the plan; the moment one did, the symptom was an unexpected
environment skip, which reads as "the test's guard is wrong" and sends the
repair a level too shallow.

So this asserts the property the registry is FOR, rather than its membership:

* every entry's contract loads, and its declared `session_id` is the session
  id of the launcher it binds -- a contract wired to another session's
  launcher fails here;
* no two entries bind the same launcher, because two contracts for one session
  are two things that can disagree about it;
* every session launcher whose staged view is DISTINCT has an entry of its
  own, with the launchers that legitimately have none named and reasoned
  about rather than silently tolerated.

These run on CPU, create nothing, and contact no provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
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


ALL = sorted(_registry())

#: Entries whose launcher CANNOT be constructed from the current tree, by that
#: session's own deliberate refusal rather than by breakage.
#:
#: `phase_c2_replay` is the one. Its module refuses to build a digest-pinned
#: replay once the files that decided attempt 3's artifact digests have moved,
#: and twenty of them have -- the operator reorganisation alone removed six.
#: That refusal is correct and is the behaviour its own tests assert: replaying
#: attempt 3's pins with today's bytes is not a replay. AGENTS.md 2.8a is
#: explicit that current `main` is not permanently constrained by a closed
#: experiment's state, so this is a fact about that session and not a defect in
#: the registry.
#:
#: Named rather than tolerated, because the whole point of a list like this is
#: that a LIVE session which stops constructing shows up as a new entry
#: somebody has to justify.
CANNOT_CONSTRUCT: frozenset[str] = frozenset({"phase_c2_replay"})

LIVE = [e for e in ALL if e not in CANNOT_CONSTRUCT]


class TestEveryEntryBindsTheLauncherItClaims:

    @pytest.mark.parametrize("experiment", ALL)
    def test_the_contract_loads(self, experiment):
        """Every entry, including the closed ones: a contract that cannot even
        be built is a registry defect regardless of its session's state."""
        contract = _contract(experiment)
        assert contract.launcher_module
        assert contract.session_id

    @pytest.mark.parametrize("experiment", LIVE)
    def test_its_session_id_is_its_launchers_session_id(self, experiment):
        """THE REGRESSION. A contract whose `session_id` came from one launcher
        while `launcher_module` named another would sweep a staged view that
        belongs to neither."""
        contract = _contract(experiment)
        assert contract.session_id == _launcher_spec(
            contract.launcher_module).session_id, (
            f"{experiment}'s sweep contract declares session "
            f"{contract.session_id!r} and binds launcher "
            f"{contract.launcher_module!r}, whose session is "
            f"{_launcher_spec(contract.launcher_module).session_id!r}")

    @pytest.mark.parametrize("experiment", sorted(CANNOT_CONSTRUCT))
    def test_a_closed_session_still_refuses_for_its_own_stated_reason(
            self, experiment):
        """Not an exemption taken on faith. If one of these starts
        constructing, or starts failing for a DIFFERENT reason, the list is
        stale and this says so.
        """
        contract = _contract(experiment)
        with pytest.raises(Exception) as exc:
            _launcher_spec(contract.launcher_module)
        assert "not a replay" in str(exc.value), (
            f"{experiment} no longer refuses for the reason this list records; "
            f"it raised {type(exc.value).__name__}: {str(exc.value)[:200]}")

    def test_no_two_entries_bind_the_same_launcher(self):
        seen: dict[str, str] = {}
        for experiment in ALL:
            launcher = _contract(experiment).launcher_module
            assert launcher not in seen, (
                f"{experiment} and {seen[launcher]} both sweep "
                f"{launcher}; two contracts for one session are two things "
                "that can disagree about it")
            seen[launcher] = experiment


class TestASessionWithItsOwnStagedViewHasItsOwnEntry:
    """The launchers with no sweep contract are NAMED, so adding a session and
    forgetting its contract fails here rather than being swept under another
    session's entry."""

    #: Sessions that legitimately have no sweep contract. Every one predates
    #: the pod-environment machinery; none is launched today. A NEW session
    #: belongs in a contract, not in here.
    NO_CONTRACT: frozenset[str] = frozenset({
        "autoinit_preflight_launch",
        "autoinit_phase_a_launch",
        "autoinit_continuation_launch",
        "autoinit_device_canary_launch",
        "autoinit_measurement_launch",
        "autoinit_recovery_continuation_launch",
    })

    def test_every_session_launcher_is_either_registered_or_named(self):
        from support.session_specs import SESSION_LAUNCHERS

        registered = {_contract(e).launcher_module for e in ALL}
        declared = {name for name, _ in SESSION_LAUNCHERS}
        unexplained = sorted(declared - registered - self.NO_CONTRACT)
        assert not unexplained, (
            f"{unexplained} are session launchers with no sweep contract and "
            "no entry in NO_CONTRACT. A session swept through ANOTHER "
            "session's registry entry gets a readiness record describing a "
            "pod it will not run on.")

    def test_the_named_set_does_not_quietly_absorb_a_live_session(self):
        """`NO_CONTRACT` is an exemption list, and an exemption list is how a
        live session gets excused. These two are named because they are the
        sessions this failure happened to."""
        assert "autoinit_d1_launch" not in self.NO_CONTRACT
        assert "autoinit_d1_replay_launch" not in self.NO_CONTRACT

    #: A deeper non-vacuity check -- "no two live entries derive the SAME
    #: staged view" -- was written and removed. It has to build every
    #: launcher's spec in one process, and the operator and cost-model
    #: registries are process-global: loading one session's launcher changes
    #: what the next one's space resolves against, so it failed with
    #: `CostModelError: no measured cost for 'attention.causal_kl_v1'` from a
    #: session whose own parametrized case passes in isolation. That is a real
    #: constraint on this kind of check and it needs a subprocess per launcher
    #: to do honestly. The three assertions above already catch the defect this
    #: file exists for -- all three fail when the replay entry is pointed back
    #: at the search's contract -- so the deeper one is not worth a
    #: subprocess harness written under launch pressure.

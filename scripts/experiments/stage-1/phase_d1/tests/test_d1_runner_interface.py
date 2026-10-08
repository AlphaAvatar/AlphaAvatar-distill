"""The REAL `SessionRunner`, far enough to exercise the authorization interface.

A dispatch test that only loads the artifact is not sufficient, and this round
proved it: `D1Authorization` implemented `require_plan` and nothing else, so the
launcher could not have reached a provider query — it would have failed locally on
`require_harness`, having reported itself ready. A test that loaded the JSON saw
none of that.

So this constructs the actual runner against the actual spec and drives the four
calls it makes before any provider work:

    require_plan           the design revision this pod runs
    require_harness        the derived closure, by MEMBERSHIP and digest
    require_within_cap     the planned hard threshold
    require_within_launch_limit   one launch is not the cumulative allowance

Provider creation is prevented structurally: the runner is built with a stub
provider and a fake key, and the only methods exercised are the ones that precede
acquisition. Nothing here can reach `create`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from experiments.phase_d1 import d1_authorization as A  # noqa: E402
from experiments.phase_d1 import d1_session as S  # noqa: E402
from support.design_blockers import (  # noqa: E402
    autouse_blocker_free_design,
)

GRANT = {"granted_by": "launch-review test", "covers": "one D1 formal search"}

#: THE RUN THIS FIXTURE'S ARTIFACT IS ISSUED FOR, and the one the runner is
#: built with. They must be the same string: `run_id` is a field of
#: `SearchConfig.as_dict()` and therefore of the `config_hash` the artifact
#: binds, so an authorization issued for one run id authorizes no other.
RUN_ID = "runner_iface"


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


#: THIS FILE'S SUBJECT IS THE RUNNER INTERFACE, not the blocker refusal.
#: `build_payload` refuses an open design blocker before anything else, so
#: while one is open the `issued` fixture below raises and every test here
#: errors out -- eleven errors and one failure, none of them about the runner.
reach_past_the_blocker_refusal = autouse_blocker_free_design(A)


@pytest.fixture(scope="module")
def issued(tmp_path_factory):
    """A real authorization, issued at a FIXED rate so the test is offline."""
    out = tmp_path_factory.mktemp("gov") / "authorization.json"
    payload = A.build_payload(run_id=RUN_ID, grant=GRANT, session_commit=_commit(),
                             granted_utc="2026-10-06T00:00:00Z",
                             live_rate=1.09)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return out, payload


@pytest.fixture(scope="module")
def runner(issued, tmp_path_factory):
    """The REAL `SessionRunner`, constructed. No provider call is reachable."""
    import autoinit_d1_launch as L
    from aadistill.infrastructure import session_runner as SR

    path, payload = issued
    scr = tmp_path_factory.mktemp("scr")
    args = L.build_parser().parse_args([
        "--scr", str(scr), "--session-commit", payload[
            "authorized_session_commit"],
        "--bundle", "b.bundle", "--run-id", RUN_ID])
    spec = L.spec(args)
    #: THE AUTHORIZATION PATH POINTS AT THE ISSUED ARTIFACT. The runner reads
    #: `repo_root / spec.authorization_path`, so the artifact is placed there.
    dest = REPO / spec.authorization_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(path.read_text())

    #: NOTHING MAY REACH A PROVIDER. The key is fake and the provider is a stub
    #: whose only methods raise, so an accidental acquisition call fails loudly
    #: rather than quietly costing money.
    class _NoProvider:
        def __getattr__(self, name):
            def _refuse(*a, **k):
                raise AssertionError(
                    f"a $0 test reached the provider: {name}(*{a}, **{k})")
            return _refuse

    import os
    os.environ["RUNPOD_API_KEY"] = "test-key-not-a-credential"
    real_provider = SR.RunPodProvider
    SR.RunPodProvider = lambda key: _NoProvider()
    try:
        r = SR.SessionRunner(spec, args, REPO)
    finally:
        SR.RunPodProvider = real_provider
    yield r
    dest.unlink(missing_ok=True)


class TestTheRunnerAcceptsTheAuthorizationInterface:
    """Construction alone exercises `require_plan` and `require_harness`."""

    def test_the_runner_constructs(self, runner):
        """It calls `require_plan` then `require_harness` in `__init__`, so
        reaching this line means both passed against the live tree."""
        assert runner.auth.authorization_id == "autoinit.v1.phase_d1"
        assert runner.auth.arm == A.FORMAL_ARM

    def test_the_harness_it_bound_is_the_derived_closure(self, runner):
        """Not a hand-maintained list: the runner stored what `require_harness`
        re-derived, and it must be the live closure."""
        live = A.d1_current_executable(REPO)
        assert runner.harness["digest"] == live["digest"]
        assert runner.harness["n_files"] == live["n_files"]
        assert runner.harness["n_files"] > 50, runner.harness["n_files"]
        #: And the entry points are D1's, not another experiment's.
        paths = {f["path"] for f in runner.harness["files"]}
        for owned in ("scripts/pod/autoinit_d1_launch.py",
                      "scripts/pod/autoinit_d1_driver.py",
                      "scripts/experiments/stage-1/phase_d1/d1_session.py"):
            assert owned in paths, owned

    def test_the_two_money_methods_the_runner_calls(self, runner):
        ceiling = runner.auth.hard_cap_usd
        #: Under the ceiling: accepted.
        runner.auth.require_within_cap(ceiling - 0.01, what="probe")
        runner.auth.require_within_launch_limit(ceiling - 0.01, what="probe")
        #: Over it: refused, by BOTH, because they bound different things -- the
        #: cumulative allowance and one launch's share of it.
        with pytest.raises(AuthorizationError):
            runner.auth.require_within_cap(ceiling + 1.0, what="probe")
        with pytest.raises(AuthorizationError):
            runner.auth.require_within_launch_limit(ceiling + 1.0, what="probe")

    def test_every_method_the_runner_source_calls_exists(self):
        """Read from the runner's own source, so a new call is caught here.

        The previous version of this authorization satisfied a hand-written list
        and not the runner.
        """
        import ast

        src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
        called = set()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "auth"):
                called.add(node.func.attr)
        assert called, "the probe found no self.auth.<method>() calls"
        missing = [m for m in called if not hasattr(A.D1Authorization, m)]
        assert not missing, (
            f"the runner calls {missing} on its authorization and "
            f"D1Authorization has no such method; the launcher would fail "
            f"locally having reported itself ready. Runner calls: {sorted(called)}")

    def test_the_budget_plan_the_runner_would_use_fits_the_authorization(
            self, runner):
        """The plan is phase-TIME; the ceiling is the artifact's. They must agree
        at the rate the ceiling was derived at."""
        plan = runner.spec.budget.plan(price_per_hour=1.09,
                                      authorized_usd=runner.auth.hard_cap_usd)
        runner.auth.require_within_cap(plan.hard_terminate_usd,
                                      what="planned hard threshold")
        runner.auth.require_within_launch_limit(plan.hard_terminate_usd,
                                              what="planned hard threshold")


class TestTheRunnerRefusesADriftedHarness:

    def test_an_edited_harness_member_is_refused(self, issued, tmp_path):
        """The closure's digest covers every member's bytes, so editing one is
        enough -- and the message must name the drift rather than a cap."""
        import dataclasses

        path, payload = issued
        auth = A.D1Authorization.load(path)
        drifted = dataclasses.replace(auth, harness_source_digest="0" * 64)
        with pytest.raises(AuthorizationError, match="digests to"):
            drifted.require_harness(REPO)

    def test_a_changed_membership_is_refused_even_at_the_same_digest(
            self, issued):
        """The failure a digest-only check cannot see.

        The base compares a digest over the authorization's OWN file list, so a
        module joining or leaving what runs digests as the same set. Membership is
        compared first for exactly that reason.
        """
        import dataclasses

        path, _ = issued
        auth = A.D1Authorization.load(path)
        short = dataclasses.replace(
            auth, harness_source_files=tuple(auth.harness_source_files[:-1]))
        with pytest.raises(AuthorizationError, match="MEMBERSHIP"):
            short.require_harness(REPO)

    def test_an_authorization_with_no_harness_digest_authorizes_nothing(self,
                                                                       issued):
        import dataclasses

        path, _ = issued
        auth = A.D1Authorization.load(path)
        bare = dataclasses.replace(auth, harness_source_digest=None)
        with pytest.raises(AuthorizationError, match="no harness_source_digest"):
            bare.require_harness(REPO)


class TestTheArtifactIsSelfVerified:

    def test_a_single_edited_byte_is_refused_on_load(self, issued, tmp_path):
        path, payload = issued
        for field, value in (("hard_cap_usd", 29.99),
                             ("arm", "all_positions"),
                             ("design_hash", "0" * 64),
                             ("harness_source_digest", "f" * 64)):
            doc = json.loads(path.read_text())
            doc[field] = value
            bad = tmp_path / f"{field}.json"
            bad.write_text(json.dumps(doc))
            with pytest.raises(AuthorizationError,
                               match="authorization_sha256"):
                A.D1Authorization.load(bad)

    def test_the_self_hash_covers_the_whole_payload(self, issued):
        """Including the fields this class added, not only the base's."""
        from aadistill.infrastructure.manifest import sha256_json

        path, payload = issued
        check = dict(payload)
        stated = check.pop("authorization_sha256")
        assert stated == sha256_json(check)
        for d1_field in ("arm", "design_hash", "measurement_protocol_id",
                         "config_hash", "suite_content_sha256", "harness"):
            assert d1_field in check, d1_field


class TestTheFormalSearchIsTreatmentOnly:

    def test_the_control_arm_cannot_be_issued(self):
        with pytest.raises(A.D1AuthorizationRefused, match="may not be issued"):
            A.build_payload(run_id=RUN_ID, grant=GRANT, session_commit=_commit(),
                            granted_utc="2026-10-06T00:00:00Z",
                            arm=S.CONTROL_ARM, live_rate=1.09)

    def test_an_issued_artifact_refuses_a_control_arm_at_the_boundary(self,
                                                                     issued):
        import dataclasses

        path, _ = issued
        auth = A.D1Authorization.load(path)
        auth.require_treatment_arm()
        control = dataclasses.replace(auth, arm=S.CONTROL_ARM)
        with pytest.raises(AuthorizationError, match="only"):
            control.require_treatment_arm()

    def test_the_control_arm_is_still_constructible_at_zero_dollars(self,
                                                                   tmp_path):
        """Useful for protocol-identity checks, and that must not be removed."""
        S._register_frozen_operators()
        control = S.build_session(arm=S.CONTROL_ARM, workdir=tmp_path,
                                  run_id="free", device="cpu")
        S.assert_session_contract(control)
        assert control.arm == S.CONTROL_ARM


class TestTheLivePriceIsQueriedBeforeIssuance:

    def test_the_same_minute_bound_reprices_at_a_different_rate(self):
        """Only the dollar consequence moves. No GPU timing is repeated."""
        accepted = A.session_ceiling(REPO)
        at_accepted = A.reprice_at(accepted["price_per_hour"], REPO)
        assert at_accepted["hard_ceiling_usd"] == pytest.approx(
            accepted["hard_ceiling_usd"], abs=5e-4)
        assert at_accepted["hard_ceiling_minutes"] == \
            accepted["hard_ceiling_minutes"]

    def test_a_lower_rate_lowers_the_authorized_ceiling(self):
        """The half an abort-if-higher launcher check cannot provide."""
        high = A.reprice_at(1.09, REPO)["hard_ceiling_usd"]
        low = A.reprice_at(0.80, REPO)["hard_ceiling_usd"]
        assert low < high
        assert A.reprice_at(1.40, REPO)["hard_ceiling_usd"] > high

    def test_a_rate_that_breaks_a_limit_stops_at_zero_dollars(self):
        """And the refusal says the science does not change to absorb it."""
        with pytest.raises(A.D1AuthorizationRefused,
                           match="does not change to absorb"):
            A.build_payload(run_id=RUN_ID, grant=GRANT, session_commit=_commit(),
                            granted_utc="2026-10-06T00:00:00Z",
                            live_rate=99.0)

    def test_the_issued_artifact_records_the_quote_it_was_issued_at(self, issued):
        _, payload = issued
        quote = payload["money"]["live_quote"]
        assert quote["usd_per_hour"] == 1.09
        derived = payload["money"]["derived_session"]
        assert derived["price_per_hour"] == quote["usd_per_hour"]
        assert "ONLY the rate" in derived["_what_moved"]

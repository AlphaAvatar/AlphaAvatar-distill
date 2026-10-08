"""D1's one-use authorization: four conditions, each refused on its own.

The 2026-10-01 note records that this experiment family once checked THREE money
conditions and that the package total binds separately — it looked implied only
while it equalled formal + engineering, which an amendment can change. So each of
the four is driven to failure here independently, and the refusal must name it.

Everything is `$0`: a refusal is the product, and the whole point of the gate is
that it is cheaper than the session it refuses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d1 import d1_authorization as A  # noqa: E402
from experiments.phase_d1 import d1_session as S  # noqa: E402
from support.design_blockers import (  # noqa: E402
    autouse_blocker_free_design,
)

GRANT = {"granted_by": "maintainer decision 2026-10-05, D1 phase envelope"}
UTC = "2026-10-05T12:00:00Z"
COMMIT = "a" * 40
#: A FIXED RATE, so these are offline and deterministic. The live quote is
#: exercised by `test_d1_runner_interface.py`, which is where that contract lives.
RATE = 1.09
#: THE RUN THE GRANT IS ISSUED FOR. Required since `config_hash` carries it:
#: a payload built without a run id binds an identity no launch reproduces.
RUN_ID = "d1_search_test"


#: EVERY TEST HERE IS ABOUT SOMETHING OTHER THAN AN OPEN DESIGN BLOCKER.
#: `build_payload` refuses one first -- correctly, it is the cheapest and most
#: categorical refusal -- so while a blocker is open all fourteen assertions
#: here reported the blocker's message instead of the refusal each was written
#: for. Asserting on whichever message came out would have retired fourteen
#: checks and left fourteen copies of the blocker test.
#:
#: `test_an_open_design_blocker` sets `DESIGN_REL` itself and so overrides
#: this: it is the one test whose subject IS the refusal.
reach_past_the_blocker_refusal = autouse_blocker_free_design(A)


@pytest.fixture(scope="module")
def money():
    return A.live_money()


@pytest.fixture(scope="module")
def priced():
    return A.session_ceiling()


class TestTheDerivedPriceIsTheOneTheDesignCarries:

    def test_the_ceiling_is_the_design_chains_search_session(self, priced):
        """One current search cost, and the authorization uses THAT one."""
        design = json.loads(
            (REPO / "logs/stages/stage-1/phase_d1/plans/d1_design.json").read_text())
        chain = design["budget"]["chain"]["sessions"]["search"]
        assert priced["hard_ceiling_usd"] == pytest.approx(
            chain["hard_ceiling_usd"])
        assert priced["hard_ceiling_minutes"] == pytest.approx(
            chain["hard_ceiling_minutes"])

    def test_it_says_which_basis_it_came_from(self, priced):
        assert "MEASURED" in priced["_basis"] or "FROZEN" in priced["_basis"]


class TestEachOfTheFourConditionsRefusesOnItsOwn:
    """One failing limit at a time, so none can be carried by the others."""

    @staticmethod
    def _money(**overrides):
        base = {
            "per_session_envelope_usd": 30.0,
            "project_cap_usd": 465.0,
            "formal_allowance_usd": 131.6523,
            "package_total_usd": 151.6523,
            "formal_remaining_usd": 62.2431,
            "package_remaining_usd": 67.2690,
            "project_remaining_usd": 61.2641,
            "cumulative_spend_usd": 403.7359,
            "funds_formal_sessions_of": ["phase_d1"],
        }
        return {**base, **overrides}

    def test_all_four_pass_at_the_live_position(self, money, priced):
        assert A.check_the_four_conditions(
            ceiling=priced["hard_ceiling_usd"], money=money) == []

    def test_the_per_session_envelope_alone(self):
        failed = A.check_the_four_conditions(
            ceiling=30.5, money=self._money())
        assert len(failed) == 1 and "per-session envelope" in failed[0]

    def test_the_project_cap_alone(self):
        """Plenty of formal and package left, no room under the cap."""
        failed = A.check_the_four_conditions(
            ceiling=25.0, money=self._money(cumulative_spend_usd=450.0))
        assert len(failed) == 1 and "project cap" in failed[0]

    def test_the_formal_allowance_alone(self):
        failed = A.check_the_four_conditions(
            ceiling=25.0, money=self._money(formal_remaining_usd=10.0))
        assert len(failed) == 1 and "FORMAL allowance" in failed[0]

    def test_the_package_total_alone(self):
        """The condition that looked implied. It is not: an amendment can raise
        the formal allowance without raising the package, and then only this
        check stands between the session and an over-authorized grant."""
        failed = A.check_the_four_conditions(
            ceiling=25.0, money=self._money(package_remaining_usd=10.0))
        assert len(failed) == 1 and "PACKAGE total" in failed[0]

    def test_several_failing_limits_are_all_reported(self):
        failed = A.check_the_four_conditions(
            ceiling=99.0, money=self._money())
        assert len(failed) == 4, failed

    def test_a_ceiling_exactly_at_a_limit_is_allowed(self):
        """A boundary is fundable; `>` not `>=`, with a tolerance for rounding."""
        assert A.check_the_four_conditions(
            ceiling=30.0, money=self._money(cumulative_spend_usd=400.0)) == []


class TestWhatTheIssuerRefuses:

    def test_a_grant_stating_a_derived_field(self):
        for field in ("authorization_id", "authorized_stages", "plan_hash",
                      "expected_usd", "design_hash", "config_hash"):
            with pytest.raises(A.D1AuthorizationRefused, match="DERIVES"):
                A.build_payload(run_id=RUN_ID, live_rate=RATE, grant={**GRANT, field: "anything"},
                                session_commit=COMMIT, granted_utc=UTC,
                                arm=S.TREATMENT_ARM)

    def test_a_grant_asking_for_the_envelope_rather_than_the_price(self):
        """The envelope is not the grant: asking for $30 when the session prices
        at $21.4897 would authorize nine dollars nothing measured."""
        with pytest.raises(A.D1AuthorizationRefused, match="DERIVED"):
            A.build_payload(run_id=RUN_ID, live_rate=RATE, grant={**GRANT, "hard_cap_usd": 30.0},
                            session_commit=COMMIT, granted_utc=UTC)
        #: And the figure the LIVE rate derives is accepted, so this is not
        #: refusing everything. Asked of `reprice_at`, because the grant must match
        #: what the rate produces -- which is the whole point of re-querying it.
        at_rate = A.reprice_at(RATE)["hard_ceiling_usd"]
        ok = A.build_payload(run_id=RUN_ID, live_rate=RATE,
                            grant={**GRANT, "hard_cap_usd": at_rate},
                            session_commit=COMMIT, granted_utc=UTC)
        assert ok["hard_cap_usd"] == pytest.approx(at_rate)

    def test_a_grant_naming_the_wrong_project_cap(self):
        with pytest.raises(A.D1AuthorizationRefused, match="names cap"):
            A.build_payload(run_id=RUN_ID, live_rate=RATE, grant={**GRANT, "cumulative_cap_usd": 410.0},
                            session_commit=COMMIT, granted_utc=UTC,
                            arm=S.TREATMENT_ARM)

    def test_an_unknown_arm(self):
        """Refused at ISSUANCE now, not by the session builder: a formal search is
        treatment-only, so any other arm -- known or not -- cannot be issued."""
        with pytest.raises(A.D1AuthorizationRefused, match="may not be issued"):
            A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT,
                            granted_utc=UTC, arm="whatever")

    def test_an_open_design_blocker(self, monkeypatch, tmp_path):
        """An authorization may not be issued over a blocker the design reports."""
        design = json.loads(
            (REPO / A.DESIGN_REL).read_text())
        design["open_blockers"] = ["funding authorization"]
        fake = tmp_path / "d1_design.json"
        fake.write_text(json.dumps(design))
        monkeypatch.setattr(A, "DESIGN_REL", str(fake))
        monkeypatch.setattr(A, "REPO", Path("/"))
        with pytest.raises(A.D1AuthorizationRefused, match="open blockers"):
            A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT, granted_utc=UTC,
                            arm=S.TREATMENT_ARM, repo_root="/")

    def test_an_experiment_outside_the_funded_list(self, monkeypatch):
        """Without `phase_d1` in `funds_formal_sessions_of`, the spend reaches the
        project cumulative while the formal book reports it as never happening."""
        real = A.live_money

        def unfunded(repo_root=A.REPO):
            m = dict(real(repo_root))
            m["funds_formal_sessions_of"] = ["phase_c1"]
            return m

        monkeypatch.setattr(A, "live_money", unfunded)
        with pytest.raises(A.D1AuthorizationRefused,
                           match="funds_formal_sessions_of"):
            A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT, granted_utc=UTC,
                            arm=S.TREATMENT_ARM)


class TestTheIssuedArtifactRoundTrips:
    """An artifact the issuer writes and the loader cannot read fails on the pod,
    after the money is committed."""

    def test_it_loads_back_with_every_identity(self, tmp_path):
        payload = A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT,
                                  granted_utc=UTC, arm=S.TREATMENT_ARM)
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        back = A.D1Authorization.load(path)
        assert back.authorized_session_commit == COMMIT
        assert back.arm == S.TREATMENT_ARM
        assert back.hard_cap_usd == payload["hard_cap_usd"]
        assert back.measurement_protocol_id == payload["measurement_protocol_id"]
        assert back.config_hash == payload["config_hash"]
        assert len(back.suite_content_sha256) == 64
        assert back.authorized_stages == tuple(A.AUTHORIZED_STAGES)

    def test_it_states_what_it_does_not_authorize(self):
        payload = A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT,
                                  granted_utc=UTC, arm=S.TREATMENT_ARM)
        for forbidden in ("recovery", "behavioural", "promotion", "D2", "D3",
                          "repetition"):
            assert forbidden in payload["authorizes"], forbidden
        assert "ONE grant, ONE issuance, ONE launcher session" in payload["one_use"]

    def test_the_stages_stop_at_commit_top_k(self):
        payload = A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT, session_commit=COMMIT,
                                  granted_utc=UTC, arm=S.TREATMENT_ARM)
        assert tuple(payload["authorized_stages"]) == ("A", "B", "C", "D")
        assert "commit_top_k" in payload["stage_conditions"]["D"]
        assert "NO recovery" in payload["stage_conditions"]["D"]

    @pytest.mark.parametrize("drop", [
        "authorization_id", "hard_cap_usd", "authorized_session_commit",
        "design_hash", "arm", "measurement_protocol_id", "config_hash",
        "suite_content_sha256", "authorized_stages", "harness_source_digest"])
    def test_a_missing_identity_is_refused_on_load(self, tmp_path, drop):
        """Dropping a field breaks the SELF-HASH first, which is the stronger
        refusal: an edited artifact is refused before anyone asks what is missing.
        Re-hashing after the edit then reaches the field check, so both fire."""
        from aadistill.infrastructure.manifest import sha256_json

        payload = A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT,
                                  session_commit=COMMIT, granted_utc=UTC)
        payload.pop(drop)
        path = tmp_path / "a.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused,
                           match="authorization_sha256"):
            A.D1Authorization.load(path)
        #: Re-hashed, so the omission is what is left to catch. The OLD hash is
        #: popped first: the loader pops it before hashing, so hashing a dict that
        #: still contains it can never agree.
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        rehashed = tmp_path / "b.json"
        rehashed.write_text(json.dumps(payload))
        with pytest.raises(A.D1AuthorizationRefused, match="omits"):
            A.D1Authorization.load(rehashed)

    def test_a_foreign_schema_is_refused(self, tmp_path):
        """A correctly-hashed artifact of another type. The hash check cannot
        catch this one -- it is internally consistent -- so the SCHEMA must."""
        from aadistill.infrastructure.manifest import sha256_json

        doc = {"schema": "aadistill.phase_c1.authorization/v1",
               "authorization_id": "autoinit.v1.phase_c1"}
        doc["authorization_sha256"] = sha256_json(doc)
        path = tmp_path / "a.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(A.D1AuthorizationRefused, match="not"):
            A.D1Authorization.load(path)

    def test_an_artifact_claiming_a_denied_permission_is_refused(self, tmp_path):
        """The ACTION POLICY refuses it, so the forbidden set lives in one
        declaration rather than a chain of `if`s a new permission could dodge."""
        from aadistill.infrastructure.manifest import sha256_json

        for claim in ("allows_recovery", "allows_behavioural",
                      "automatic_followon_start"):
            payload = A.build_payload(run_id=RUN_ID, live_rate=RATE, grant=GRANT,
                                      session_commit=COMMIT, granted_utc=UTC)
            payload[claim] = True
            payload.pop("authorization_sha256")
            payload["authorization_sha256"] = sha256_json(payload)
            path = tmp_path / f"{claim}.json"
            path.write_text(json.dumps(payload))
            #: The BASE error: `ActionPolicy.check_claims` is a generic
            #: governance primitive and knows nothing about D1, which is why it
            #: raises the base type. `D1AuthorizationRefused` is a subclass, so
            #: catching it here would NOT catch this -- the asymmetry that cost
            #: this round a debugging pass.
            from aadistill.governance.authorization import AuthorizationError

            with pytest.raises(AuthorizationError, match="claims"):
                A.D1Authorization.load(path)

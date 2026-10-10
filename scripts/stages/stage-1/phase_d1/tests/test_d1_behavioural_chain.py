"""The D1 behavioural EXECUTION CHAIN: preregistration, issuance, launcher.

What `test_d1_behavioural_contract.py` covers is the frozen science -- the
arms, seeds, battery and ranking. This file covers the chain that was built to
EXECUTE it: the committed preregistration and replay plan, the one-use
authorization's build/load round trip and its refusals, the launcher's
declaration, and the pod-side materialization module's own refusals. Everything
here runs at `$0` on a CPU host.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/pod", "scripts/autoinit",
               "scripts/stages/stage-1", "tests"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1 import behavioural_authorization as BA  # noqa: E402
from stages.phase_d1 import behavioural_materialize as M  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1BehaviouralAuthorization,
    D1BehaviouralRefused,
)

HEAD = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                      text=True, cwd=REPO).stdout.strip() or ("0" * 40)

GRANT = {
    "granted_by": "maintainer-test",
    "covers": "ONE formal D1 behavioural SCREENING session",
    "rung": "screening",
    "hard_cap_usd": 25.0053,
    "cumulative_cap_usd": 490.0,
    "granted_utc": "2026-10-10",
    "does_not_authorize": ["a beam", "D2", "D3"],
}

#: The rate the design's cells were priced at; supplying it makes issuance a
#: $0 operation with no network.
ACCEPTED_RATE = 1.09


def _payload(**over):
    kwargs = dict(grant=GRANT, session_commit=HEAD,
                  granted_utc="2026-10-10T00:00:00+00:00",
                  run_id="d1b_test", rung="screening",
                  live_rate=ACCEPTED_RATE, repo_root=REPO)
    kwargs.update(over)
    return BA.build_payload(**kwargs)


class TestThePreregistrationBindsTheFrozenValues:
    """C0: the seeds must be materialized and hash-bound in the execution
    preregistration BEFORE any candidate behavioural result exists."""

    @staticmethod
    def _doc():
        return json.loads((REPO / BA.PREREG_REL).read_text())

    @staticmethod
    def _writer():
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "write_d1_behavioural_preregistration.py")
        spec = importlib.util.spec_from_file_location("_prereg_w", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_it_exists_and_matches_its_own_hash(self):
        doc = self._doc()
        writer = self._writer()
        assert doc["preregistration_sha256"] == \
            writer.preregistration_identity(doc)

    def test_this_tree_regenerates_it_identically(self):
        """A preregistration an owner record has outrun names a session
        nobody would run. The wall clock is outside the identity, so this is
        a fixed point, not a race."""
        writer = self._writer()
        live = writer.preregistration_identity(writer.build(REPO))
        assert live == self._doc()["preregistration_sha256"]

    def test_it_binds_the_materialized_seeds(self):
        doc = self._doc()
        assert doc["seeds"]["screening"] == list(B.screening_seeds(REPO))
        assert doc["seeds"]["confirmation"] == list(B.confirmation_seeds(REPO))
        assert doc["seeds"]["excluded"] == list(B.excluded_seeds(REPO))

    def test_it_binds_the_design_and_the_five_arms(self):
        doc = self._doc()
        assert doc["design"]["design_hash"] == B.design(REPO)["design_hash"]
        members = {m["arm"]: m for m in doc["arms"]["members"]}
        assert sorted(members) == ["B", "q1", "q2", "q3", "q4"]
        for arm in B.arms(REPO):
            bound = members[arm.arm_id]["identities"]
            assert bound == dict(arm.identities)
            for value in bound.values():
                assert len(value) == 64
        assert members["B"]["identities"]["artifact_digest"].startswith(
            "53e30566")

    def test_it_binds_the_replay_plan_by_hash(self):
        doc = self._doc()
        assert doc["arm_materialization"]["plan_sha256"] == \
            M.plan_sha256(REPO)
        assert doc["arm_materialization"]["plan_path"] == M.PLAN_REL

    def test_it_binds_both_batteries_and_the_guardrails(self):
        doc = self._doc()
        for role in ("d1_screening", "d1_confirmation"):
            identity = B.battery_role(role, REPO)
            bound = doc["batteries"][role]
            assert bound["item_ids_sha256"] == identity["item_ids_sha256"]
            assert bound["files"] == identity["files"]
        vetoes = doc["screening"]["ranking"]["guardrail_vetoes"]
        assert vetoes["pooled_usable_delta_min"] == \
            B.GUARDRAIL_POOLED_MIN_DELTA
        assert vetoes["per_seed_usable_delta_min"] == \
            B.GUARDRAIL_PER_SEED_MIN_DELTA

    def test_the_confirmation_candidate_is_not_prenamed(self):
        doc = self._doc()
        text = doc["confirmation"]["_candidate_binding"]
        assert "CANNOT be named here" in text
        assert doc["confirmation"]["arms"] == "the advancing candidate and B"

    def test_it_authorizes_nothing(self):
        assert self._doc()["authorizes"] == "nothing"


class TestTheReplayPlanIsPinnedAndCoversTheFrozenField:

    @staticmethod
    def _plan():
        return json.loads((REPO / M.PLAN_REL).read_text())

    def test_four_leaves_matching_the_retention_identities(self):
        plan = self._plan()
        assert plan["n_leaves"] == 4
        decision = json.loads((REPO / B.RETENTION_REL).read_text())
        members = {m["state_id"]: m for m in
                   decision["the_frozen_behavioural_finalists"]["members"]}
        for leaf in plan["leaves"]:
            want = members[leaf["state_id"]]
            for key in ("artifact_digest", "weights_digest",
                        "single_shard_sha256", "arch_signature"):
                assert leaf["expected"][key] == want[key], key

    def test_every_step_is_digest_pinned(self):
        for leaf in self._plan()["leaves"]:
            assert len(leaf["steps"]) == 4
            for step in leaf["steps"]:
                assert len(step["expected_artifact_digest"]) == 64
                assert len(step["expected_config_hash"]) == 64

    def test_the_execution_agreement_is_the_searchs(self):
        agreement = self._plan()["execution_agreement"]
        assert agreement["using"] == {"micro_batch_size": 3,
                                      "calibration_batch_packing":
                                          "length_sorted_v1"}
        assert agreement["n_steps"] == 16

    def test_the_root_state_travels_and_is_rederivable(self):
        root = self._plan()["root_state"]
        assert root["chosen_candidate"] == "session_driver_use_cache_false"
        assert root["config_overrides"] == {"use_cache": False}

    def test_the_pod_loader_rejects_the_wrong_schema(self):
        doc = dict(self._plan())
        doc["schema"] = "aadistill.phase_d1.replay_plan/v1"
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as handle:
            json.dump(doc, handle)
        with pytest.raises(M.D1MaterializeError, match="different leaf sets"):
            M.load_plan(handle.name)


class TestIssuanceDerivesEverythingAndRefusesTheRest:

    def test_the_round_trip(self, tmp_path):
        payload = _payload()
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload, indent=1))
        auth = D1BehaviouralAuthorization.load(path)
        auth.require_rung("screening")
        auth.require_run_id("d1b_test")
        auth.require_plan(B.design(REPO)["design_hash"])
        auth.require_contract(sha256_json(B.session_contract("screening",
                                                             REPO)))
        assert auth.n_probes == 10
        assert auth.seeds == tuple(B.screening_seeds(REPO))
        assert auth.battery_role == "d1_screening"
        assert payload["preregistration_sha256"] == json.loads(
            (REPO / BA.PREREG_REL).read_text())["preregistration_sha256"]
        assert payload["replay_plan_sha256"] == M.plan_sha256(REPO)

    def test_the_accepted_rate_reproduces_the_accepted_ceiling(self):
        priced = BA.reprice_rung_at(ACCEPTED_RATE, "screening", REPO)
        assert priced["hard_ceiling_usd"] == pytest.approx(
            priced["accepted_hard_ceiling_usd"], abs=5e-4)

    def test_a_lower_live_rate_lowers_the_ceiling(self):
        """The half an abort-if-higher launcher check cannot provide."""
        lower = BA.reprice_rung_at(0.80, "screening", REPO)
        assert lower["hard_ceiling_usd"] < lower["accepted_hard_ceiling_usd"]
        #: And the grant naming the old figure is then refused.
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="carries the derived figure"):
            _payload(live_rate=0.80)

    def test_a_grant_stating_a_derived_field_is_refused(self):
        for field in ("contract_hash", "plan_hash", "seeds", "n_probes"):
            with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                               match="DERIVES"):
                _payload(grant={**GRANT, field: "x"})

    def test_a_grant_for_the_other_rung_is_refused(self):
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="neither grant covers the other"):
            _payload(grant={**GRANT, "rung": "confirmation"})

    def test_screening_refuses_an_advancing_candidate(self):
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="may not name"):
            _payload(advancing_candidate="q1")

    def test_confirmation_requires_one(self):
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="must name"):
            _payload(grant={**GRANT, "rung": "confirmation"},
                     rung="confirmation")

    def test_an_unfundable_rate_is_refused_on_the_four_conditions(self):
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="not fundable"):
            _payload(live_rate=2.50,
                     grant={**GRANT, "hard_cap_usd": None})

    def test_the_schedule_must_match_the_priced_cell(self, monkeypatch):
        real = BA.session_cell

        def fewer(rung, repo_root=REPO):
            cell = dict(real(rung, repo_root))
            cell["n_probes"] = 6
            return cell

        monkeypatch.setattr(BA, "session_cell", fewer)
        with pytest.raises(BA.D1BehaviouralIssuanceRefused,
                           match="different experiment at the same price"):
            _payload()

    def test_a_confirmation_payload_narrows_to_six_probes(self):
        grant = {**GRANT, "rung": "confirmation",
                 "hard_cap_usd": 15.2025,
                 "covers": "ONE formal D1 behavioural CONFIRMATION session"}
        payload = BA.build_payload(
            grant=grant, session_commit=HEAD, granted_utc="t",
            run_id="d1b_conf_test", rung="confirmation",
            advancing_candidate="q1", live_rate=ACCEPTED_RATE, repo_root=REPO)
        assert payload["n_probes"] == 6
        assert payload["advancing_candidate"] == "q1"
        assert payload["seeds"] == list(B.confirmation_seeds(REPO))
        assert payload["battery_role"] == "d1_confirmation"


class TestTheLoadedArtifactCannotBeSubstitutedOrEdited:

    def test_an_edited_artifact_fails_its_self_hash(self, tmp_path):
        payload = _payload()
        payload["hard_cap_usd"] = 999.0
        path = tmp_path / "edited.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(D1BehaviouralRefused, match="edited since"):
            D1BehaviouralAuthorization.load(path)

    def test_a_screening_artifact_naming_a_candidate_is_refused(self,
                                                                tmp_path):
        payload = _payload()
        body = {k: v for k, v in payload.items()
                if k != "authorization_sha256"}
        body["advancing_candidate"] = "q2"
        body["authorization_sha256"] = sha256_json(body)
        path = tmp_path / "pre_named.json"
        path.write_text(json.dumps(body))
        with pytest.raises(D1BehaviouralRefused, match="pre-named"):
            D1BehaviouralAuthorization.load(path)

    def test_the_search_schema_cannot_stand_in(self, tmp_path):
        payload = _payload()
        body = {k: v for k, v in payload.items()
                if k != "authorization_sha256"}
        body["schema"] = "aadistill.phase_d1.authorization/v1"
        body["authorization_sha256"] = sha256_json(body)
        path = tmp_path / "wrong_schema.json"
        path.write_text(json.dumps(body))
        with pytest.raises(D1BehaviouralRefused):
            D1BehaviouralAuthorization.load(path)

    def test_the_wrong_run_id_is_refused(self, tmp_path):
        payload = _payload()
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload))
        auth = D1BehaviouralAuthorization.load(path)
        with pytest.raises(D1BehaviouralRefused, match="one-use|One grant"):
            auth.require_run_id("some_other_run")


class TestTheLauncherDeclaresTheSessionCompletely:

    @staticmethod
    def _launcher():
        from support.session_specs import load_session_launcher

        return load_session_launcher("autoinit_d1_behavioural_launch")

    @staticmethod
    def _spec(launcher):
        from support.session_specs import session_args

        return launcher.spec(session_args(launcher))

    def test_the_spec_builds_and_names_its_own_kind(self):
        launcher = self._launcher()
        spec = self._spec(launcher)
        assert spec.setup.env["SESSION_KIND"] == "d1_behavioural"
        assert spec.session_id == "autoinit-d1-behavioural"
        assert spec.plan_hash == B.design(REPO)["design_hash"]

    def test_the_status_path_is_named_once(self):
        """A driver writing markers to a path its launcher does not poll made
        every marker invisible on a paid session."""
        launcher = self._launcher()
        spec = self._spec(launcher)

        class _Auth:
            rung = "screening"
            advancing_candidate = ""

        class _Args:
            run_id = "d1b_probe"

        class _Ctx:
            auth = _Auth()
            args = _Args()
            image_digest = "sha256:test"

        command = launcher.driver_command(_Ctx(), None)
        assert f"--status '{launcher.STATUS}'" in command
        assert spec.status_path == launcher.STATUS

    def test_the_driver_command_parses_with_the_drivers_own_parser(self):
        """C1 attempt 7 died at $0.4231 because its launcher emitted a flag
        the driver's parser has no option for."""
        import importlib.util
        import shlex

        launcher = self._launcher()

        class _Auth:
            rung = "confirmation"
            advancing_candidate = "q3"

        class _Args:
            run_id = "d1b_probe"

        class _Ctx:
            auth = _Auth()
            args = _Args()
            image_digest = "sha256:test"

        command = launcher.driver_command(_Ctx(), None)
        tokens = shlex.split(command)
        driver_index = next(i for i, t in enumerate(tokens)
                            if t.endswith("autoinit_d1_behavioural_driver.py"))
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py")
        module_spec = importlib.util.spec_from_file_location("_d1b_drv", src)
        driver = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(driver)
        parsed = driver.build_parser().parse_args(tokens[driver_index + 1:])
        assert parsed.rung == "confirmation"
        assert parsed.advancing_candidate == "q3"
        assert parsed.arm_root == launcher.M.POD_ARM_ROOT
        assert parsed.authorization.endswith("governance/authorization.json")

    def test_the_budget_decomposes_the_accepted_bound_exactly(self):
        """One total: expected x 1.1 + reserve + recovery == the accepted
        hard bound, so the plan lands on the priced cell rather than beside
        it."""
        launcher = self._launcher()
        for rung in ("screening", "confirmation"):
            cell = BA.session_cell(rung, REPO)
            budget = launcher.budget_spec(REPO, rung)
            reserve = sum(p.minutes for p in budget.soft_stop_reserves)
            total = (cell["expected_minutes"] * 1.1 + reserve
                     + budget.artifact_recovery_reserve_minutes)
            assert total == pytest.approx(cell["hard_ceiling_minutes"],
                                          abs=0.05), rung
            assert reserve >= 0

    def test_the_account_requirement_is_the_two_term_rule(self):
        launcher = self._launcher()
        spec = self._spec(launcher)
        cell = BA.session_cell("screening", REPO)
        assert spec.budget.account_balance_required_usd == pytest.approx(
            max(30.0, cell["hard_ceiling_usd"]
                + launcher.ACCOUNT_OPERATIONAL_RESERVE_USD), abs=1e-4)

    def test_both_batteries_travel_as_local_assets(self):
        """`roles_are_disjoint` reads BOTH roles' realized item ids, on the
        pod as on this host; a session that shipped one would refuse on the
        other's absence after setup was paid for."""
        spec = self._spec(self._launcher())
        names = {a.dest_name for a in spec.setup.local_assets}
        assert {"d1_screening", "d1_confirmation"} <= names

    def test_products_secured_refuses_unknown_and_missing(self, tmp_path):
        launcher = self._launcher()

        class _Args:
            scr = str(tmp_path)

        class _Ctx:
            args = _Args()
            evidence = {}

            @staticmethod
            def say(_msg):
                pass

        ctx = _Ctx()
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "UNKNOWN" in why

        relay = tmp_path / "relay"
        relay.mkdir()
        (relay / "d1_behavioural.json").write_text(json.dumps({
            "probes": [{"probe_id": "p1", "scored": True}]}))
        ctx.evidence = {}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "p1" in why

        ctx.evidence = {"probe_evidence_secured": {
            "p1": {"result.json": True, "per_sample.jsonl": True,
                   "admission.json": False, "generations": False}}}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "admission" in why

        ctx.evidence = {"probe_evidence_secured": {
            "p1": {"result.json": True, "per_sample.jsonl": True,
                   "admission.json": True, "generations": False}}}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "attested" in why

        store = tmp_path / "store"
        store.mkdir(exist_ok=True)
        (store / "d1_behavioural_attested_protocol.json").write_text("{}")
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "final archive" in why

        (relay / "d1_behavioural.json").write_text(json.dumps({"probes": []}))
        ctx.evidence = {}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "no scored evidence is owed" in why


class TestTheMaterializationModuleRefusesWhatItMust:

    def test_a_plan_missing_a_frozen_leaf_is_refused(self, tmp_path):
        plan = json.loads((REPO / M.PLAN_REL).read_text())
        plan["leaves"] = plan["leaves"][:3]
        path = tmp_path / "short_plan.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(M.D1MaterializeError,
                           match="does not describe the frozen field"):
            M.materialize_arms("screening", plan_path=path,
                               arm_root=tmp_path / "arms", device="cpu",
                               repo_root=REPO, say=lambda _s: None)

    def test_confirmation_requires_the_candidate(self, tmp_path):
        with pytest.raises(M.D1MaterializeError, match="two arms, not five"):
            M.materialize_arms("confirmation",
                               plan_path=REPO / M.PLAN_REL,
                               arm_root=tmp_path / "arms", device="cpu",
                               repo_root=REPO, say=lambda _s: None)

    def test_resolution_refuses_a_missing_row(self, tmp_path):
        with pytest.raises(M.D1MaterializeError,
                           match="no materialization row"):
            M.resolve_arm_checkpoints("screening", tmp_path, {"arms": []},
                                      repo_root=REPO)

    def test_resolution_refuses_a_row_whose_bytes_are_absent(self, tmp_path):
        with pytest.raises(M.D1MaterializeError,
                           match="no checkpoint at"):
            M.resolve_arm_checkpoints(
                "confirmation", tmp_path,
                {"arms": [
                    {"arm": "q1", "state_id": "x",
                     "checkpoint_path": str(tmp_path / "missing")},
                    {"arm": "B", "checkpoint_path": str(tmp_path / "b")},
                ]},
                advancing_candidate="q1", repo_root=REPO)


class TestTheConfirmationScorerIsItsOwnPinnedEntryPoint:

    def test_it_pins_its_role_and_schema(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "score_d1_confirmation.py").read_text()
        assert 'ROLE = "d1_confirmation"' in src
        assert "confirmation_result/v1" in src
        assert '"--battery"' not in src
        assert "THE BATTERY IS NOT AN ARGUMENT" in src

    def test_the_metric_is_c1s_imported_unchanged(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "score_d1_confirmation.py").read_text()
        assert ("from stages.phase_c1.score_c1_confirmation import "
                "build_result, score_battery") in src

    def test_the_driver_dispatches_by_rung(self):
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py")
        module_spec = importlib.util.spec_from_file_location("_d1b_drv2", src)
        driver = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(driver)
        assert driver.SCORER.name == "score_d1_screening.py"
        assert driver.CONFIRMATION_SCORER.name == "score_d1_confirmation.py"
        assert driver.CONFIRMATION_SCORER.is_file()

    def test_the_result_must_not_be_read_as_a_verdict(self):
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "score_d1_confirmation.py")
        module_spec = importlib.util.spec_from_file_location("_d1b_sc", src)
        mod = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mod)
        text = " ".join(mod.MAY_NOT)
        assert "verdict" in text and "off-pod" in text


class TestTheContractIsLocationFreeAndNarrows:

    def test_no_machine_local_path_enters_the_hash(self):
        for rung, kwargs in (("screening", {}),
                             ("confirmation",
                              {"advancing_candidate": "q2"})):
            contract = B.session_contract(rung, REPO, **kwargs)
            text = json.dumps(contract)
            assert "/home/" not in text, rung
            assert "aad-artifacts" not in text, rung

    def test_the_confirmation_contract_binds_its_candidate(self):
        c = B.session_contract("confirmation", REPO,
                               advancing_candidate="q3")
        assert c["advancing_candidate"] == "q3"
        assert c["n_probes"] == 6
        arms = {row["arm"] for row in c["arms"]["arms"]}
        assert arms == {"q1", "q2", "q3", "q4", "B"}
        scheduled = {p["arm"] for p in c["probes"]}
        assert scheduled == {"q3", "B"}

    def test_two_candidates_bind_two_different_contracts(self):
        one = sha256_json(B.session_contract("confirmation", REPO,
                                             advancing_candidate="q1"))
        two = sha256_json(B.session_contract("confirmation", REPO,
                                             advancing_candidate="q2"))
        assert one != two

    def test_an_arm_root_does_not_move_the_hash_fields(self):
        """Location is execution configuration. The full contract cannot be
        built under a fake arm root (the incumbent's bytes do not exist on
        this host), so the invariance is asserted where it lives: the bound
        arm rows derive from frozen records alone."""
        plain = B.session_contract("screening", REPO)["arms"]
        field = B.arms(REPO, arm_root="/nonexistent/pod/root")
        rooted = [{
            "arm": a.arm_id, "role": a.role,
            "quality_position": a.quality_position,
            "state_id": a.state_id, "identities": dict(a.identities),
        } for a in field]
        assert plain["arms"] == rooted


class TestTheIssuerCommandRefusesBeforeItQuotes:

    @staticmethod
    def _issuer():
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "issue_d1_behavioural_authorization.py")
        spec = importlib.util.spec_from_file_location("_d1b_issue", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_parser_builds(self):
        parser = self._issuer().build_parser()
        for action in parser._actions:
            if action.required:
                assert action.type in (None, str, Path), action.option_strings

    def test_a_bare_selection_record_is_no_longer_accepted(self, tmp_path):
        """The 2026-10-10 review's confirmation obligation: an arbitrary JSON
        file containing `advanced: true` and an arm label is not evidence.
        The full validation (schema, rung, contract hash, recomputed
        selection) lives in TestTheConfirmationCandidateIsRecomputedNotTrusted."""
        issuer = self._issuer()
        record = tmp_path / "selection.json"
        record.write_text(json.dumps({
            "selection": {"advanced": True, "arm": "q2",
                          "outcome": "ONE_CANDIDATE_ADVANCES"}}))
        with pytest.raises(SystemExit, match="schema"):
            issuer.advanced_arm_from(record)


class TestTheSweepContractIsExpressible:
    """AGENTS.md P8.3's launch-bound sweep must be EXPRESSIBLE through the
    canonical recorder; the search's own sweep was once refused as an unknown
    experiment."""

    def test_the_registry_resolves_the_behavioural_entry(self):
        from shared.pod.record_pod_environment import sweep_contract

        contract = sweep_contract("phase_d1_behavioural", "d1b_probe", None,
                                  "launch_bound")
        assert contract.launcher_module == "autoinit_d1_behavioural_launch"
        assert contract.session_id == "autoinit-d1-behavioural"
        assert contract.record.record_path.endswith(
            "runs/d1b_probe/governance/readiness.json")

    def test_the_record_is_run_owned_with_no_pointer(self):
        from stages.phase_d1 import behavioural_pod_environment as BP

        contract = BP.sweep_contract(run_id="d1b_probe")
        assert contract.pointer_path is None
        with pytest.raises(BP.D1BehaviouralReadinessError):
            BP.record_path_for(None)

    def test_the_shell_has_a_dispatch_branch_for_this_kind(self):
        """A missing dispatch entry is not a type error: it falls through to
        `spend` and costs a paid pod. The dedicated suite checks the loader
        semantics; this asserts the branch exists beside the experiment that
        needs it."""
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        assert '"$SESSION_KIND" = "d1_behavioural"' in src
        assert "D1BehaviouralAuthorization" in src
        branch = src.split('"$SESSION_KIND" = "d1_behavioural"', 1)[1]
        branch = branch.split("elif", 1)[0]
        assert "allows_recovery_training is True" in branch
        assert "allows_beam_search is False" in branch
        assert "automatic_followon_start is False" in branch


class TestTheProducersOwnCheckpointLayoutIsWhatConsumersRead:
    """The review finding: `materialize_fixed_path` writes a path's final
    checkpoint under `workdir/steps/{i:02d}_{kind}/`, and the first version of
    this chain verified `arm_root/<state_id>/config.json` -- a directory the
    producer never writes -- AFTER paying for the materialization. These
    exercise the actual layout end to end, not declarations about it."""

    def test_final_checkpoint_is_the_last_step_directory(self, tmp_path):
        dest = tmp_path / "state"
        for index, kind in enumerate(("ffn", "depth", "residual_width",
                                      "attention")):
            (dest / "steps" / f"{index:02d}_{kind}").mkdir(parents=True)
            (dest / "steps" / f"{index:02d}_{kind}" / "config.json"
             ).write_text("{}")
        found, complete = M.final_checkpoint_in(dest, n_steps=4)
        assert complete and found.name == "03_attention"

    def test_a_partial_path_is_not_adopted(self, tmp_path):
        dest = tmp_path / "state"
        for index, kind in enumerate(("ffn", "depth")):
            (dest / "steps" / f"{index:02d}_{kind}").mkdir(parents=True)
            (dest / "steps" / f"{index:02d}_{kind}" / "config.json"
             ).write_text("{}")
        found, complete = M.final_checkpoint_in(dest, n_steps=4)
        assert found is not None and not complete

    def test_a_flattened_restore_is_recognized(self, tmp_path):
        dest = tmp_path / "state"
        dest.mkdir()
        (dest / "config.json").write_text("{}")
        found, complete = M.final_checkpoint_in(dest, n_steps=4)
        assert complete and found == dest

    def test_nothing_there_is_nothing(self, tmp_path):
        assert M.final_checkpoint_in(tmp_path / "absent", n_steps=4) == \
            (None, False)

    def test_the_step_layout_verifies_through_the_real_consumer(self,
                                                                tmp_path):
        """PRODUCER -> CONSUMER on real bytes: a pod-shaped
        `arm_root/<state_id>/steps/03_attention` holding q1's actual secured
        checkpoint must be found by the layout resolver AND pass the identity
        gate at the FOUND path -- the exact seam the session would have died
        on."""
        q1 = B.arms(REPO)[0]
        dest = tmp_path / q1.state_id
        (dest / "steps").mkdir(parents=True)
        (dest / "steps" / "03_attention").symlink_to(q1.checkpoint_dir)
        found, complete = M.final_checkpoint_in(dest, n_steps=4)
        assert complete and found.name == "03_attention"
        identity = M._verify_bytes_at(found, dict(q1.identities),
                                      what="q1-layout-test")
        assert identity["artifact_digest"] == q1.identities["artifact_digest"]
        #: And the CONTRACT verification consumes the resolved path, not the
        #: guessed one, without moving the location-free hash.
        resolved = {a.arm_id: a.checkpoint_dir for a in B.arms(REPO)
                    if a.checkpoint_dir}
        resolved["q1"] = str(found)
        contract = B.session_contract("screening", REPO,
                                      resolved_paths=resolved)
        assert sha256_json(contract) == sha256_json(
            B.session_contract("screening", REPO))

    def test_resolution_consumes_the_rows_checkpoint_path(self, tmp_path):
        inner = tmp_path / "steps" / "03_attention"
        inner.mkdir(parents=True)
        (inner / "config.json").write_text("{}")
        rows = {"arms": [
            *[{"arm": a.arm_id, "state_id": a.state_id,
               "checkpoint_path": a.checkpoint_dir}
              for a in B.arms(REPO) if not a.is_incumbent],
            {"arm": "B", "checkpoint_path": str(inner)},
        ]}
        out = M.resolve_arm_checkpoints("screening", tmp_path, rows,
                                        repo_root=REPO)
        assert out["B"] == str(inner)
        assert Path(out["q1"]).is_dir()


class TestTheIncumbentRebuildsUnderItsHistoricalProtocol:
    """A3's terminal finding: canonical B (53e30566...) reproduces exactly
    only under A_bsz1. A rebuild under DEFAULT_EXECUTION (mbs=4) or the
    candidates' bsz=3 policy is a THIRD protocol nothing measured B under."""

    def test_the_execution_is_a_bsz1_read_from_its_owner(self):
        from stages.phase_a3.a_bsz3 import A_BSZ1

        execution = M.incumbent_execution()
        assert execution is A_BSZ1
        assert execution.as_fingerprint() == {
            "micro_batch_size": 1,
            "calibration_batch_packing": "original_order_v1"}

    def test_materialize_incumbent_passes_it_to_the_producer(self,
                                                             monkeypatch,
                                                             tmp_path):
        import aadistill.initialization.planning.fixed_path as FP

        captured = {}

        class _Stop(RuntimeError):
            pass

        def capture(spec, **kwargs):
            captured["execution"] = kwargs.get("execution")
            raise _Stop()

        monkeypatch.setattr(FP, "materialize_fixed_path", capture)
        with pytest.raises(_Stop):
            M.materialize_incumbent(tmp_path / "B", device="cuda",
                                    identities={"artifact_digest": "x" * 64},
                                    repo_root=REPO)
        from stages.phase_a3.a_bsz3 import A_BSZ1

        assert captured["execution"] is A_BSZ1


class TestFailureClassificationSeparatesHarnessFromScience:
    """The stop condition is narrower than 'the comparison failed': only an
    identity discrepancy AFTER the inputs were demonstrably reproduced is
    scientific. A missing path or plan leaf is P12.1 ordinary engineering."""

    def test_a_missing_plan_leaf_is_engineering_not_identity(self, tmp_path):
        plan = json.loads((REPO / M.PLAN_REL).read_text())
        plan["leaves"] = plan["leaves"][:3]
        path = tmp_path / "short.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(M.D1MaterializeError) as excinfo:
            M.materialize_arms("screening", plan_path=path,
                               arm_root=tmp_path / "arms", device="cpu",
                               repo_root=REPO, say=lambda _s: None)
        assert not isinstance(excinfo.value, M.D1ArmIdentityMismatch)

    def test_a_mid_path_digest_divergence_is_the_identity_class(
            self, monkeypatch, tmp_path):
        import aadistill.initialization.planning.fixed_path as FP
        from stages.phase_d1 import replay_specs as R

        def diverge(spec, **kwargs):
            raise FP.FixedPathDigestMismatch(0, "step 0 label", "y" * 64,
                                             "x" * 64, {})

        monkeypatch.setattr(FP, "materialize_fixed_path", diverge)
        plan = M.load_plan(REPO / M.PLAN_REL)
        leaf = R.leaves_from_plan(plan)[0]
        with pytest.raises(M.D1ArmIdentityMismatch, match="verified"):
            M.materialize_candidate(leaf, tmp_path / "x", device="cpu",
                                    repo_root=REPO)

    def test_the_driver_marks_mismatch_only_for_the_identity_class(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        assert "except M.D1ArmIdentityMismatch:" in src
        block = src.split("except M.D1ArmIdentityMismatch:", 1)[1]
        assert "mismatch = True" in block.split("record[", 1)[0]
        #: And the base class is NOT caught into the mismatch flag.
        assert "except M.D1MaterializeError:" not in src


class TestGenerationAdmissionGuardsTheScoringPath:

    @staticmethod
    def _driver():
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py")
        spec = importlib.util.spec_from_file_location("_d1b_drv3", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_scoring_contract_digests_the_rungs_own_scorer(self):
        driver = self._driver()
        screening = driver.d1_scoring_contract("screening", REPO)
        confirmation = driver.d1_scoring_contract("confirmation", REPO)
        assert screening["contract"].startswith("c1_confirmation_scoring@")
        assert screening["digest"] != confirmation["digest"], (
            "two rungs with two pinned scorers must not share one digest")
        names = {e["path"] for e in screening["files"]}
        assert ("scripts/stages/stage-1/phase_d1/score_d1_screening.py"
                in names)

    def test_the_battery_triple_is_the_familys_own_identity(self):
        driver = self._driver()
        contract = B.session_contract("screening", REPO)
        fields = driver.battery_protocol_fields(contract)
        battery = contract["battery"]
        assert fields["battery_artifact"] == \
            f"{battery['family_id']}:{battery['role']}"
        assert fields["battery_manifest_sha256"] == \
            battery["item_ids_sha256"]
        assert fields["battery_content_sha256"] == \
            battery["family_content_id"]

    def test_admission_refuses_a_drifted_protocol_before_scoring(
            self, tmp_path):
        """A probe whose summaries cannot establish the protocol is refused
        -- fail-closed, through the real observer."""
        driver = self._driver()
        gen_dir = tmp_path / "gen"
        gen_dir.mkdir()
        (gen_dir / "code.json").write_text(json.dumps({
            "label": "p", "prompts": "code"}))
        scoring = {"contract": "c1_confirmation_scoring@v1", "digest": "d"}
        fields = {"battery_artifact": "a", "battery_manifest_sha256": "m",
                  "battery_content_sha256": "c"}

        class _Attested:
            evaluation_protocol_hash = "x"

        with pytest.raises(Exception) as excinfo:
            driver.admit_generation("p", gen_dir, attested=_Attested(),
                                    scoring=scoring, battery_fields=fields,
                                    out=tmp_path)
        assert "material generation field" in str(excinfo.value) or \
            "not comparable" in str(excinfo.value)

    def test_the_driver_admits_before_it_scores_and_binds_provenance(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        assert src.index("build_attested_protocol(") < src.index(
            '"--prompts"'), "the attestation must exist before generation"
        assert src.index("admit_generation(\n") < src.index(
            '"--trained-run"'), "admission must precede scoring"
        assert '"--trained-run"' in src
        assert '"--generation-fingerprint"' in src


class TestTrainedButUnscoredProbesSurviveAndResume:
    """P8.2.1 + P8.4 state 2 together: a trained, not-validly-scored probe's
    weights are preserved at the moment the later stage fails, and a resumed
    session scores them rather than retraining the frozen unit."""

    @staticmethod
    def _model_root(tmp_path, *, sha=None, sidecar_sha=None):
        import hashlib

        root = tmp_path / "probe"
        model = root / "checkpoints" / "step100" / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        shard = b"not-a-real-shard"
        (model / "model.safetensors").write_bytes(shard)
        (root / "checkpoints" / "latest.txt").write_text("step100")
        (root / "run_completion.json").write_text(json.dumps(
            {"final_step": 100}))
        if sidecar_sha is not None:
            (root / "preserved_identity.json").write_text(json.dumps(
                {"trained_sha256": sidecar_sha}))
        return root, hashlib.sha256(shard).hexdigest()

    def test_a_completed_training_resumes_at_evaluation(self, tmp_path):
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        root, _sha = self._model_root(tmp_path)
        state = driver.trained_checkpoint_state(root)
        assert state is not None
        assert state["model_dir"].endswith("checkpoints/step100/model")

    def test_a_restored_probe_is_rehashed_against_its_sidecar(self, tmp_path):
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        root, sha = self._model_root(tmp_path)
        (root / "preserved_identity.json").write_text(json.dumps(
            {"trained_sha256": sha}))
        state = driver.trained_checkpoint_state(root)
        assert state and state.get("restored_sha256_verified")
        (root / "preserved_identity.json").write_text(json.dumps(
            {"trained_sha256": "f" * 64}))
        with pytest.raises(driver.D1BehaviouralDriverError,
                           match="nobody preserved"):
            driver.trained_checkpoint_state(root)

    def test_an_incomplete_training_retrains(self, tmp_path):
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        root, _sha = self._model_root(tmp_path)
        (root / "run_completion.json").unlink()
        assert driver.trained_checkpoint_state(root) is None

    def test_teardown_refuses_while_a_trained_unscored_probe_is_pod_only(
            self, tmp_path):
        from support.session_specs import load_session_launcher

        launcher = load_session_launcher("autoinit_d1_behavioural_launch")

        class _Args:
            scr = str(tmp_path)

        class _Ctx:
            args = _Args()
            evidence = {}

            @staticmethod
            def say(_msg):
                pass

        relay = tmp_path / "relay"
        relay.mkdir()
        (relay / "d1_behavioural.json").write_text(json.dumps({
            "probes": [{"probe_id": "p1", "trained": True, "scored": False,
                        "model_dir": "/workspace/x/checkpoints/t/model",
                        "trained_sha256": "a" * 64}]}))
        ctx = _Ctx()
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "P8.4" in why

        ctx.evidence = {"trained_unscored_preserved": {
            "p1": {"verified": True}}}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "preserved" in why

    def test_a_restore_without_a_sidecar_is_refused(self, tmp_path):
        from support.session_specs import load_session_launcher

        launcher = load_session_launcher("autoinit_d1_behavioural_launch")
        store = tmp_path / "preserved"
        (store / "p1").mkdir(parents=True)

        said = []

        class _Args:
            restore_trained = str(store)

        class _Ctx:
            args = _Args()
            evidence = {}

            @staticmethod
            def say(msg):
                said.append(msg)

        assert launcher.restore_trained_probes(_Ctx()) is False
        assert any("unverifiable" in s for s in said)

    def test_scored_probes_weights_are_never_fetched(self):
        """P8.4 state 1: completed + validly scored contributes evidence, not
        checkpoint bytes."""
        from support.session_specs import load_session_launcher

        launcher = load_session_launcher("autoinit_d1_behavioural_launch")
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_launch.py").read_text()
        assert "if p.get(\"trained\") and not p.get(\"scored\")" in src
        assert launcher is not None


class TestTheConfirmationCandidateIsRecomputedNotTrusted:

    @staticmethod
    def _issuer():
        import importlib.util

        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "issue_d1_behavioural_authorization.py")
        spec = importlib.util.spec_from_file_location("_d1b_issue2", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _screening_record(tmp_path, *, tamper_arm=None, tamper_contract=None):
        seeds = B.screening_seeds(REPO)
        table = {"q1": (0.40, 0.90), "q2": (0.45, 0.90), "q3": (0.42, 0.89),
                 "q4": (0.38, 0.91), "B": (0.40, 0.90)}
        probes = [{"probe_id": f"d1_screening_{arm}_s{seed}", "arm": arm,
                   "seed": seed, "scored": True,
                   "correct_overall": value[0],
                   "usable_rollout_rate": value[1]}
                  for arm, value in table.items() for seed in seeds]
        rows = [{"arm": p["arm"], "seed": p["seed"],
                 "correct_overall": p["correct_overall"],
                 "usable_rollout_rate": p["usable_rollout_rate"]}
                for p in probes]
        selection = B.advance_one(B.rank_screening(rows, REPO))
        if tamper_arm:
            selection = dict(selection)
            selection["arm"] = tamper_arm
        record = {
            "schema": "aadistill.phase_d1.behavioural_session/v1",
            "rung": "screening", "status": "COMPLETE",
            "contract_hash": tamper_contract or sha256_json(
                B.session_contract("screening", REPO)),
            "probes": probes, "selection": selection,
        }
        path = tmp_path / "d1_behavioural.json"
        path.write_text(json.dumps(record))
        return path

    def test_a_consistent_record_yields_the_recomputed_arm(self, tmp_path):
        issuer = self._issuer()
        path = self._screening_record(tmp_path)
        assert issuer.advanced_arm_from(path) == "q2"

    def test_a_tampered_selection_is_refused_not_resolved(self, tmp_path):
        issuer = self._issuer()
        path = self._screening_record(tmp_path, tamper_arm="q4")
        with pytest.raises(SystemExit, match="refused, not resolved"):
            issuer.advanced_arm_from(path)

    def test_a_record_from_another_field_is_refused(self, tmp_path):
        issuer = self._issuer()
        path = self._screening_record(tmp_path, tamper_contract="e" * 64)
        with pytest.raises(SystemExit, match="not the frozen field"):
            issuer.advanced_arm_from(path)

    def test_a_bare_verdict_file_is_refused(self, tmp_path):
        issuer = self._issuer()
        path = tmp_path / "claim.json"
        path.write_text(json.dumps({"selection": {"advanced": True,
                                                  "arm": "q1"}}))
        with pytest.raises(SystemExit, match="schema"):
            issuer.advanced_arm_from(path)


class TestDiskResidencyFollowsTheRealFreeCallSites:
    """The 2026-10-10 second review: C2 attempt3 died at ENOSPC with five full
    arm paths resident, and attempt5 at probe 11/12 under accumulated probe
    workdirs. The same program shapes exist here, so the same releases do --
    and a resource bound has to follow the real free() call sites."""

    @staticmethod
    def _steps(tmp_path, n=4):
        from types import SimpleNamespace

        results = []
        for index, kind in enumerate(["ffn", "depth", "width", "attn"][:n]):
            d = tmp_path / "steps" / f"{index:02d}_{kind}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"x" * 64)
            results.append(SimpleNamespace(checkpoint_path=str(d),
                                           impl_id=f"{kind}.v0"))
        return results

    def test_intermediates_are_released_and_the_final_is_kept(self, tmp_path):
        results = self._steps(tmp_path)
        out = M.release_intermediate_steps("q1", results, tmp_path)
        assert out["removed_steps"] == ["ffn.v0", "depth.v0", "width.v0"]
        assert not Path(results[0].checkpoint_path).exists()
        assert Path(results[-1].checkpoint_path).is_dir()
        assert out["failed"] == []

    def test_nothing_outside_the_arms_workdir_is_touched(self, tmp_path):
        from types import SimpleNamespace

        outside = tmp_path / "elsewhere" / "step"
        outside.mkdir(parents=True)
        (outside / "x").write_bytes(b"x")
        results = self._steps(tmp_path / "arm")
        results.insert(0, SimpleNamespace(checkpoint_path=str(outside),
                                          impl_id="outside.v0"))
        out = M.release_intermediate_steps("q1", results, tmp_path / "arm")
        assert outside.is_dir()
        assert "outside.v0" not in out["removed_steps"]

    def test_a_release_failure_is_reported_not_raised(self, tmp_path):
        from types import SimpleNamespace

        results = self._steps(tmp_path)
        results.insert(0, SimpleNamespace(
            checkpoint_path=str(tmp_path / "steps" / "99_gone"),
            impl_id="gone.v0"))
        out = M.release_intermediate_steps("q1", results, tmp_path)
        assert any("gone.v0" in f for f in out["failed"])

    def test_the_arm_loop_fails_closed_on_a_failed_release(self, monkeypatch,
                                                           tmp_path):
        """The 120 GB provision is derived on the assumption intermediates
        are freed; a falsified assumption stops the session with the verified
        arm preserved."""
        def fake_candidate(leaf, dest, **kwargs):
            return {"state_id": leaf.state_id, "spec_hash": "x",
                    "seconds": 0.0, "checkpoint_path": str(dest),
                    "identity": {}, "adopted": True,
                    "intermediates_released": {
                        "removed_steps": [], "freed_gib": 0.0,
                        "failed": ["ffn.v0: Permission denied"],
                        "kept": str(dest)}}

        monkeypatch.setattr(M, "materialize_candidate", fake_candidate)
        with pytest.raises(M.D1MaterializeError,
                           match="NO FURTHER ARM"):
            M.materialize_arms("screening", plan_path=REPO / M.PLAN_REL,
                               arm_root=tmp_path, device="cpu",
                               repo_root=REPO, say=lambda _s: None)

    def test_the_probe_footprint_is_derived_not_typed(self):
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        need = driver.probe_local_need_bytes(REPO)
        assert need["need_bytes"] == (need["transient_bytes"]
                                      + need["retained_bytes"]
                                      + need["package_bytes"])
        #: fp32 training of a ~0.6B model: the working set alone is > 8 GiB,
        #: so a derived value below that means the derivation broke.
        assert need["transient_bytes"] > 8 * 2**30
        assert need["checkpoint"]["num_parameters"] == int(
            B.design(REPO)["incumbent"]["num_parameters"])

    def test_headroom_refuses_before_training(self, monkeypatch, tmp_path):
        import aadistill.runtime.leaf_durability as LD

        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        monkeypatch.setattr(LD, "free_bytes_at", lambda _p: 1 * 2**30)
        record: dict = {}
        with pytest.raises(driver.D1BehaviouralDriverError,
                           match="REFUSING before training"):
            driver.require_probe_headroom("p1", record, lambda: None)
        assert record["probe_headroom"][0]["free_gib"] == 1.0

    def test_acked_scored_workdirs_release_and_the_bound_fails_closed(
            self, monkeypatch, tmp_path):
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
        ack_dir = (tmp_path
                   / "artifacts/audit/autoinit_d1_behavioural/release_acks")
        ack_dir.mkdir(parents=True)
        work = tmp_path / "artifacts/stages/stage-3/d1_behavioural"
        pkg = tmp_path / driver.EVAL_PACKAGES_REL
        for probe_id in ("p_acked", "p_unacked", "p_unscored"):
            (work / probe_id).mkdir(parents=True)
            (work / probe_id / "w").write_bytes(b"x" * 32)
            (pkg / probe_id).mkdir(parents=True)
        (ack_dir / "p_acked.json").write_text("{}")
        (ack_dir / "p_unscored.json").write_text("{}")
        record = {"probes": [
            {"probe_id": "p_acked", "scored": True},
            {"probe_id": "p_unacked", "scored": True},
            {"probe_id": "p_unscored", "scored": False},
        ]}
        out = driver.release_acked_probe_workdirs(record, lambda: None)
        assert out["released"] == ["p_acked"]
        assert not (work / "p_acked").exists()
        assert not (pkg / "p_acked").exists()
        #: Unacked stays; unscored stays even though acked -- an ack is not a
        #: score, and only a validly scored probe's weights lose their
        #: consumer.
        assert (work / "p_unacked").is_dir()
        assert (work / "p_unscored").is_dir()
        assert out["kept"] == ["p_unacked"]

    def test_packages_are_built_outside_the_audit_dir(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        assert 'out / "packages"' not in src
        assert "EVAL_PACKAGES_REL" in src
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        assert not driver.EVAL_PACKAGES_REL.startswith("artifacts/audit")

    def test_release_runs_before_each_probe(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_driver.py").read_text()
        assert src.index("release_acked_probe_workdirs(record, save)") < \
            src.index("require_probe_headroom(probe.probe_id")
        assert src.index("require_probe_headroom(probe.probe_id") < \
            src.index('"--config", str(config)')


class TestProtocolEvidenceTravelsInTheArtifactContracts:
    """Second-review finding 2: the raw protocol-identity evidence -- engine
    probe, attestation, per-probe admissions -- must be collected, not just
    the hashes results carry. And the specs must LOAD: an invented lifecycle
    word makes the whole document unloadable and the collector exits 1 --
    which is exactly what the committed failed specs would have done, on the
    exact path (a failed session) they exist for."""

    @staticmethod
    def _load(rel):
        import importlib.util

        src = REPO / "scripts/shared/pod/collect_artifacts.py"
        spec = importlib.util.spec_from_file_location("_collect", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_specs(str(REPO / rel))

    def test_all_four_d1_specs_load_through_the_real_loader(self):
        for rel in (
                "configs/stages/stage-1/phase_d1/d1_behavioural_artifacts.json",
                "configs/stages/stage-1/phase_d1/"
                "d1_behavioural_artifacts_failed.json",
                "configs/stages/stage-1/phase_d1/d1_search_artifacts.json",
                "configs/stages/stage-1/phase_d1/"
                "d1_search_artifacts_failed.json"):
            assert self._load(rel), rel

    def test_the_success_spec_requires_the_protocol_evidence(self):
        entries = {e.artifact_class: e for e in self._load(
            "configs/stages/stage-1/phase_d1/d1_behavioural_artifacts.json")}
        for cls in ("d1_behavioural_attested_protocol",
                    "d1_behavioural_engine_probe",
                    "d1_behavioural_generation_admissions"):
            assert entries[cls].required, cls
        assert "engine_probe.json" in entries[
            "d1_behavioural_engine_probe"].pattern

    def test_the_failed_spec_collects_them_optionally(self):
        entries = {e.artifact_class: e for e in self._load(
            "configs/stages/stage-1/phase_d1/"
            "d1_behavioural_artifacts_failed.json")}
        for cls in ("d1_behavioural_attested_protocol",
                    "d1_behavioural_engine_probe",
                    "d1_behavioural_generation_admissions"):
            assert cls in entries and not entries[cls].required, cls

    def test_the_poll_hook_pulls_the_admission_beside_the_result(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_launch.py").read_text()
        assert "_generation_admission.json" in src
        #: And the ack is written only when all three strict pieces arrived.
        ack_block = src.split('state["release_acked"] = ', 1)[0]
        tail = ack_block[ack_block.rfind("if ("):]
        for piece in ("result.json", "per_sample.jsonl", "admission.json"):
            assert piece in tail, piece


class TestTrainedUnscoredDurabilityRunsDuringTheSession:
    """Second-review finding 3: closeout-only preservation repeats the C1
    failure -- a provider disappearance between training and scoring loses a
    completed checkpoint. Preservation now runs on every poll, and a preserved
    copy is pruned the moment its probe is validly scored."""

    def test_the_poll_hook_preserves_and_prunes(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "autoinit_d1_behavioural_launch.py").read_text()
        body = src.split("def secure_probe_evidence(", 1)[1]
        body = body.split("\ndef ", 1)[0]
        assert "preserve_trained_unscored(ctx)" in body
        assert "prune_preserved_scored(ctx)" in body

    def test_pruning_requires_a_scored_probe_with_durable_evidence(
            self, tmp_path):
        from support.session_specs import load_session_launcher

        launcher = load_session_launcher("autoinit_d1_behavioural_launch")
        store = tmp_path / "products" / "trained_unscored"
        kept = store / "p_kept"
        kept.mkdir(parents=True)
        (kept / "w").write_bytes(b"x")
        pruned = store / "p_pruned"
        pruned.mkdir(parents=True)
        (pruned / "w").write_bytes(b"x")
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        class _Args:
            scr = str(tmp_path)
            ckpt_store = None

        class _Ctx:
            args = _Args()
            evidence = {
                "trained_unscored_preserved": {
                    "p_pruned": {"verified": True, "dest": str(pruned)},
                    "p_kept": {"verified": True, "dest": str(kept)},
                    "p_outside": {"verified": True, "dest": str(outside)},
                },
                "probe_evidence_secured": {
                    "p_pruned": {"result.json": True,
                                 "per_sample.jsonl": True,
                                 "admission.json": True},
                    "p_outside": {"result.json": True,
                                  "per_sample.jsonl": True,
                                  "admission.json": True},
                },
            }

            @staticmethod
            def say(_msg):
                pass

        launcher.prune_preserved_scored(_Ctx())
        assert not pruned.exists(), "scored + durable evidence -> pruned"
        assert kept.is_dir(), "no durable evidence yet -> kept"
        assert outside.is_dir(), "containment: never deletes outside the store"


class TestTheFirstProbesTrainedWindowIsPreserved:
    """Third review: `secure_probe_evidence` returned on zero scored probes,
    so trained-unscored preservation never ran during the FIRST probe's
    trained-but-unscored interval -- the exact provider-disappearance window
    the repair was written for. These drive the REAL poll hook (a source
    assertion cannot see a premature return), with only the transfers
    simulated at `_scp_from_pod`."""

    SHARD = b"trained-weights-of-probe-one"

    def _run_hook(self, tmp_path, monkeypatch, *, fail_remote=None):
        import hashlib

        from support.session_specs import load_session_launcher

        launcher = load_session_launcher("autoinit_d1_behavioural_launch")
        sha = hashlib.sha256(self.SHARD).hexdigest()
        model_dir = f"{launcher.POD_PROBE_ROOT}/p1/checkpoints/step100/model"
        relay = tmp_path / "relay"
        relay.mkdir()
        #: ZERO scored probes, ONE completed training: the first probe's
        #: window, verbatim.
        (relay / "d1_behavioural.json").write_text(json.dumps({
            "probes": [{"probe_id": "p1", "trained": True, "scored": False,
                        "model_dir": model_dir, "trained_sha256": sha}]}))
        remote_files = {
            f"{launcher.POD_PROBE_ROOT}/p1/run_completion.json":
                json.dumps({"final_step": 100}).encode(),
            f"{launcher.POD_PROBE_ROOT}/p1/checkpoints/latest.txt":
                b"step100",
        }
        shard = self.SHARD

        def fake_scp(ctx, remote, dest, *, recursive=False, limit_min=5):
            if fail_remote and fail_remote in remote:
                return 1
            dest = Path(dest)
            if recursive:
                #: scp -r of `.../model` into the pre-created tag dir nests
                #: it as `<tag>/model/`, which is what the real transfer does.
                target = dest / "model" if remote.endswith("/model") else dest
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}")
                (target / "model.safetensors").write_bytes(shard)
                return 0
            data = remote_files.get(remote)
            if data is None:
                return 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return 0

        monkeypatch.setattr(launcher, "_scp_from_pod", fake_scp)

        class _Args:
            scr = str(tmp_path)
            ckpt_store = None
            ckpt_fetch_limit_min = 45

        class _Ctx:
            args = _Args()
            evidence = {}

            @staticmethod
            def say(_msg):
                pass

        ctx = _Ctx()
        launcher.secure_probe_evidence(ctx)
        return launcher, ctx, sha

    def test_zero_scored_probes_still_preserve_the_trained_one(
            self, tmp_path, monkeypatch):
        launcher, ctx, sha = self._run_hook(tmp_path, monkeypatch)
        assert not ctx.evidence.get("on_poll_errors"), \
            ctx.evidence.get("on_poll_errors")
        row = ctx.evidence["trained_unscored_preserved"]["p1"]
        assert row["verified"] is True, row
        assert all(row["completeness"].values()), row["completeness"]
        assert row["arrived_sha256"] == sha
        assert (Path(row["dest"]) / "preserved_identity.json").is_file()

    def test_an_incomplete_transfer_is_not_durable(self, tmp_path,
                                                   monkeypatch):
        """A matching shard must not compensate for a failed transfer of the
        other resume inputs: a set without its training-completion record
        restores as 'never finished training' and RETRAINS the frozen unit."""
        launcher, ctx, _sha = self._run_hook(
            tmp_path, monkeypatch, fail_remote="run_completion.json")
        row = ctx.evidence["trained_unscored_preserved"]["p1"]
        assert row["verified"] is False
        assert row["completeness"]["weights_digest"] is True
        assert row["completeness"]["run_completion"] is False
        assert not (Path(row["dest"]) / "preserved_identity.json").exists()
        #: And the teardown gate therefore still refuses: unpreserved.
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert not ok and "p1" in why

    def test_a_complete_preservation_resumes_at_evaluation_only(
            self, tmp_path, monkeypatch):
        launcher, ctx, sha = self._run_hook(tmp_path, monkeypatch)
        row = ctx.evidence["trained_unscored_preserved"]["p1"]
        driver = TestGenerationAdmissionGuardsTheScoringPath._driver()
        state = driver.trained_checkpoint_state(Path(row["dest"]))
        assert state is not None
        assert state["model_dir"].endswith("checkpoints/step100/model")
        assert state.get("restored_sha256_verified") is True
        #: And the gate accepts teardown once the preservation is verified.
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "preserved" in why

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
                   "generations": False}}}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "final archive" in why

        (relay / "d1_behavioural.json").write_text(json.dumps({"probes": []}))
        ctx.evidence = {}
        ok, why = launcher.probes_evidence_secured(ctx, [])
        assert ok and "none is owed" in why


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

    def test_confirmation_reads_the_candidate_from_the_record(self, tmp_path):
        issuer = self._issuer()
        record = tmp_path / "selection.json"
        record.write_text(json.dumps({
            "selection": {"advanced": True, "arm": "q2",
                          "outcome": "ONE_CANDIDATE_ADVANCES"}}))
        assert issuer.advanced_arm_from(record) == "q2"
        record.write_text(json.dumps({
            "selection": {"advanced": None,
                          "outcome": "NO_CANDIDATE_ADVANCES"}}))
        with pytest.raises(SystemExit, match="advanced nobody"):
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

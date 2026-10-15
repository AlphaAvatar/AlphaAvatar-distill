"""D2's screening chain at `$0`: contract, ranking, grant, launcher, driver.

Each class pins what a later paid session rests on, and each refusal is shown
to fire on the input it claims to read (a check that passes the real tree
proves nothing until a mutation of its premise fails it).
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]

from stages.phase_d2 import d2_screening as SC  # noqa: E402


# --- the contract -------------------------------------------------------------

class TestTheContractBindsTheFrozenField:

    def test_it_derives_with_the_arms_bytes_verified(self):
        c = SC.session_contract(REPO)
        assert c["n_probes"] == len(c["probes"]) == 8
        assert c["shared_protocol"]["protocol_hash"].startswith("5ed198ae")
        assert c["preregistration_hash"].startswith("b52810921a4c")

    def test_the_hash_is_location_free(self):
        a = SC.session_contract(REPO, arm_root="/pod/a", verify_bytes=False)
        b = SC.session_contract(REPO, arm_root="/pod/b", verify_bytes=False)
        assert SC.contract_hash(a) == SC.contract_hash(b) == \
            SC.contract_hash(SC.session_contract(REPO))

    def test_the_field_is_the_committed_selection_in_its_order(self):
        sel = SC.selection(REPO)["selected"]
        arms = SC.arms(REPO)
        assert [a.state_id for a in arms] == [r["state_id"] for r in sel]
        assert [a.arm_id for a in arms] == ["c1", "c2", "c3", "c4"]
        for arm, row in zip(arms, sel):
            assert arm.identities == {k: row[k] for k in (
                "artifact_digest", "weights_digest", "single_shard_sha256",
                "arch_signature")}
        assert not any(a.is_incumbent for a in arms)

    def test_seeds_are_the_shared_rungs_and_d1s_screening_seeds(self):
        from stages.phase_d1 import behavioural as D1B

        assert list(SC.seeds(REPO)) == [516448251, 614923639]
        assert sorted(SC.seeds(REPO)) == sorted(D1B.screening_seeds(REPO))

    def test_the_comparator_is_bs_two_imported_units(self):
        comp = SC.comparator_units(REPO)
        assert [u["probe_id"] for u in comp["B"]] == [
            "d1_screening_B_s516448251", "d1_screening_B_s614923639"]
        assert len(comp["q2"]) == 2

    def test_a_drifted_recipe_is_refused(self, monkeypatch):
        from stages.d_series import shared_evaluation as SE

        monkeypatch.setattr(SE, "recipe_digest", lambda *a, **k: "0" * 64)
        with pytest.raises(SC.D2ScreeningError, match="recipe_digest"):
            SC.recovery(REPO)

    def test_a_drifted_trainer_is_refused(self, monkeypatch):
        from stages.d_series import shared_evaluation as SE

        monkeypatch.setattr(SE, "tree_matches", lambda *a, **k: {
            "trainer": ["trainer_source_digest"], "generation": [],
            "scoring": []})
        with pytest.raises(SC.D2ScreeningError, match="pins"):
            SC.session_contract(REPO, verify_bytes=False)

    def test_an_arm_whose_bytes_differ_is_refused(self, tmp_path):
        """Symlinked weights, an edited config: the artifact digest moves."""
        for arm in SC.arms(REPO):
            dest = tmp_path / arm.state_id
            dest.mkdir()
            for name in SC.ARM_FILES:
                (dest / name).symlink_to(Path(arm.checkpoint_dir) / name)
        victim = tmp_path / SC.arms(REPO)[2].state_id / "config.json"
        doc = json.loads(victim.read_text())
        victim.unlink()
        doc["rms_norm_eps"] = 1e-5 if doc.get("rms_norm_eps") != 1e-5 else 1e-6
        victim.write_text(json.dumps(doc))
        with pytest.raises(SC.D2ScreeningError, match="c3"):
            SC.require_arms_present(REPO, arm_root=tmp_path)


# --- the ranking ----------------------------------------------------------------

def _rows(values: dict[str, tuple[float, float]]) -> list[dict]:
    """arm -> (correct, usable) at both seeds."""
    return [{"arm": arm, "seed": seed, "correct_overall": c,
             "usable_rollout_rate": u}
            for arm, (c, u) in values.items() for seed in SC.seeds(REPO)]


class TestTheRankingIsTheFrozenRule:

    B_POOLED = (0.0259 + 0.0176) / 2

    def test_the_best_unvetoed_candidate_advances_against_imported_b(self):
        out = SC.rank(_rows({"c1": (0.03, 0.6), "c2": (0.05, 0.6),
                             "c3": (0.02, 0.6), "c4": (0.01, 0.6)}), REPO)
        assert out["selection"]["arm"] == "c2"
        top = out["ranking"]["ranked"][0]
        assert abs(top["delta_vs_b"] - (0.05 - self.B_POOLED)) < 1e-9
        assert out["comparator_units"] == ["d1_screening_B_s516448251",
                                           "d1_screening_B_s614923639"]

    def test_usable_rollout_vetoes_and_never_credits(self):
        out = SC.rank(_rows({"c1": (0.30, 0.10), "c2": (0.03, 0.6),
                             "c3": (0.02, 0.6), "c4": (0.01, 0.6)}), REPO)
        assert out["selection"]["arm"] == "c2"
        assert out["ranking"]["ranked"][-1]["arm"] == "c1"
        assert out["ranking"]["ranked"][-1]["vetoed"] is True

    def test_none_advances_when_every_candidate_is_vetoed(self):
        out = SC.rank(_rows({a: (0.5, 0.0) for a in ("c1", "c2", "c3", "c4")}),
                      REPO)
        assert out["selection"]["advanced"] is None
        assert out["selection"]["outcome"] == "NO_CANDIDATE_ADVANCES"

    def test_a_tie_breaks_by_the_frozen_quality_position(self):
        out = SC.rank(_rows({a: (0.03, 0.6) for a in ("c4", "c2", "c3", "c1")}),
                      REPO)
        assert out["selection"]["arm"] == "c1"
        assert out["selection"]["tie_broken"] is True

    def test_an_incomplete_field_is_refused(self):
        from stages.d_series.shared_evaluation import SharedEvaluationError

        rows = _rows({a: (0.03, 0.6) for a in ("c1", "c2", "c3", "c4")})[:-1]
        with pytest.raises(SharedEvaluationError):
            SC.rank(rows, REPO)

    def test_q2_is_descriptive_and_never_ranked(self):
        out = SC.rank(_rows({a: (0.03, 0.6) for a in ("c1", "c2", "c3", "c4")}),
                      REPO)
        assert "q2" not in {r["arm"] for r in out["ranking"]["ranked"]}
        assert "DESCRIPTIVE ONLY" in out["q2_descriptive"]["_role"]


class TestOffPodVerificationRefusesWhatIsNotPaired:

    def _unit(self, tmp_path, *, correct=(True, False)) -> tuple[Path, dict]:
        from aadistill.infrastructure.manifest import sha256_json

        probe = tmp_path / "probes" / "p"
        probe.mkdir(parents=True)
        rows = [{"id": f"x{i}", "set": "gsm8k", "seed": 1, "correct": c,
                 "usable": True, "scorable": True}
                for i, c in enumerate(correct)]
        (probe / "per_sample.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        import hashlib
        result = {"correct_overall": 0.5, "usable_rollout_rate": 1.0,
                  "per_sample_sha256": hashlib.sha256(
                      (probe / "per_sample.jsonl").read_bytes()).hexdigest()}
        result["result_sha256"] = sha256_json(result)
        (probe / "result.json").write_text(json.dumps(result))
        rung = SC.protocol(REPO)["rungs"]["shared_screening"]
        session = {"result_sha256": result["result_sha256"],
                   "generation_admission": {
                       "evaluation_protocol_hash":
                           rung["evaluation_protocol_hash"]}}
        return probe, session

    def test_an_unadmitted_or_unmanifested_unit_does_not_verify(self, tmp_path):
        from stages.phase_d2 import rank_d2_screening as R

        probe, session = self._unit(tmp_path)
        got = R.verify_unit(probe, session, repo_root=REPO)
        assert got["checks"]["admitted_under_the_shared_rung"] is True
        assert got["checks"]["training_identity_is_the_protocols"] is False
        session["generation_admission"]["evaluation_protocol_hash"] = "0" * 64
        assert R.verify_unit(probe, session, repo_root=REPO)["checks"][
            "admitted_under_the_shared_rung"] is False

    def test_alignment_and_paired_counts(self):
        from stages.phase_d2 import rank_d2_screening as R

        ref = [{"id": "a", "correct": True, "scorable": True},
               {"id": "b", "correct": False, "scorable": True}]
        cand = [{"id": "a", "correct": False, "scorable": True},
                {"id": "b", "correct": True, "scorable": True}]
        assert R.aligned(cand, ref) is True
        assert R.paired_counts(cand, ref) == {"both": 0, "candidate_only": 1,
                                              "reference_only": 1, "neither": 0}
        assert R.aligned(cand[:1], ref) is False


# --- the grant ------------------------------------------------------------------

GRANT = {"rung": "screening", "granted_by": "test",
         "cumulative_envelope_usd": 25.0}


def _grant(**extra):
    g = dict(GRANT)
    g["approves_preregistration_hash"] = SC.preregistration(REPO)[
        "preregistration_hash"]
    g.update(extra)
    return g


class TestTheGrantIsDerivedAndOneUse:

    def _payload(self, **kw):
        from stages.phase_d2 import d2_screening_authorization as A

        args = dict(grant=_grant(), session_commit="a" * 40,
                    granted_utc="2026-10-15T00:00:00+00:00",
                    run_id="d2_screening_test", repo_root=REPO,
                    live_rate=1.09, consumed_usd=0.0)
        args.update(kw)
        return A.build_payload(**args)

    def test_the_round_trip_binds_the_contract_and_the_price(self, tmp_path):
        from stages.phase_d2 import d2_screening_authorization as A

        payload = self._payload()
        path = tmp_path / "auth.json"
        path.write_text(json.dumps(payload))
        auth = A.D2ScreeningAuthorization.load(path)
        auth.require_contract(SC.contract_hash(SC.session_contract(REPO)))
        auth.require_run_id("d2_screening_test")
        assert auth.n_probes == 8 and list(auth.seeds) == list(SC.seeds(REPO))
        assert auth.hard_cap_usd == pytest.approx(
            A.reprice_at(1.09, REPO)["hard_ceiling_usd"])
        assert auth.hard_cap_usd <= 30.0
        assert not auth.authorizes_d2_confirmation

    @pytest.mark.parametrize("field", ["hard_cap_usd", "contract_hash", "seeds",
                                       "n_probes"])
    def test_a_grant_stating_a_derived_field_is_refused(self, field):
        from stages.phase_d2 import d2_screening_authorization as A

        with pytest.raises(A.D2ScreeningRefused, match=field):
            self._payload(grant=_grant(**{field: 1}))

    def test_an_envelope_narrowed_below_the_price_is_refused(self):
        from stages.phase_d2 import d2_screening_authorization as A

        with pytest.raises(A.D2ScreeningRefused, match="does not shrink"):
            self._payload(consumed_usd=20.0)

    def test_no_envelope_and_an_unapproved_preregistration_are_refused(self):
        from stages.phase_d2 import d2_screening_authorization as A

        g = _grant()
        g.pop("cumulative_envelope_usd")
        with pytest.raises(A.D2ScreeningRefused, match="envelope"):
            self._payload(grant=g)
        with pytest.raises(A.D2ScreeningRefused, match="preregistration"):
            self._payload(grant=_grant(approves_preregistration_hash="0" * 64))

    def test_the_run_id_must_carry_the_netted_prefix(self):
        from stages.phase_d2 import d2_screening_authorization as A

        with pytest.raises(A.D2ScreeningRefused, match="prefix"):
            self._payload(run_id="d2_other_run")

    def test_the_loader_refuses_edits_other_schemas_and_denied_claims(
            self, tmp_path):
        from aadistill.governance.authorization import AuthorizationError
        from aadistill.infrastructure.manifest import sha256_json
        from stages.phase_d2 import d2_screening_authorization as A

        payload = self._payload()
        for mutate in (lambda d: d.update(hard_cap_usd=99.0),
                       lambda d: d.update(schema="aadistill.phase_d2."
                                          "authorization/v1"),
                       lambda d: d.update(authorizes_d2_confirmation=True)):
            doc = json.loads(json.dumps(payload))
            mutate(doc)
            if doc["schema"] != payload["schema"] or doc.get(
                    "authorizes_d2_confirmation"):
                doc.pop("authorization_sha256")
                doc["authorization_sha256"] = sha256_json(doc)
            path = tmp_path / "a.json"
            path.write_text(json.dumps(doc))
            with pytest.raises(AuthorizationError):
                A.D2ScreeningAuthorization.load(path)

    def test_an_unknown_prior_subrun_cost_is_refused(self, tmp_path):
        from stages.phase_d2 import d2_screening_authorization as A

        run = tmp_path / A.RUNS_ROOT / "d2_screening_x" / "runtime"
        run.mkdir(parents=True)
        (run / "session.json").write_text("{}")
        with pytest.raises(A.D2ScreeningRefused, match="no booked cost"):
            A.consumed_by_prior_subruns(tmp_path)
        closeout = run.parent / "closeout"
        closeout.mkdir()
        (closeout / "outcome.json").write_text(json.dumps(
            {"money": {"all_in_usd": 2.5}}))
        assert A.consumed_by_prior_subruns(tmp_path)["consumed_usd"] == 2.5


# --- the launcher ---------------------------------------------------------------

class TestTheLauncherDeclaresTheSession:

    @staticmethod
    def _launcher():
        from stages.phase_d2 import autoinit_d2_screening_launch as L
        return L

    def test_host_admission_is_the_protocols_exact_image(self):
        L = self._launcher()
        want = SC.protocol(REPO)["recovery"]["training_image_digest"]
        assert L.host_admission(want)[0] is True
        for other in (want.replace("595.91.07", "580.126.09"),
                      want.replace("595.91.07", "595.91.08"),
                      "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"):
            assert L.host_admission(other)[0] is False

    def test_the_layout_shares_no_path_with_d1s(self):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L

        L = self._launcher()
        for field in ("evidence_dir", "evidence_filename", "attested_name",
                      "pod_probe_root", "release_ack_rel"):
            assert getattr(L.LAYOUT, field) != getattr(D1L.D1_LAYOUT, field)

    def test_the_layout_names_what_the_driver_writes(self):
        from stages.phase_d2 import autoinit_d2_screening_driver as D

        L = self._launcher()
        assert L.LAYOUT.attested_name == D.ATTESTED_NAME
        assert L.LAYOUT.evidence_filename == D.EVIDENCE_FILENAME
        assert L.LAYOUT.release_ack_rel == D.ACK_REL
        assert L.LAYOUT.pod_probe_root.endswith(D.PROBE_ROOT_REL)
        assert L.EVAL_PACKAGES_REL == D.EVAL_PACKAGES_REL

    def test_the_driver_command_parses_with_the_drivers_parser(self):
        import shlex

        from stages.phase_d2 import autoinit_d2_screening_driver as D

        L = self._launcher()
        ctx = SimpleNamespace(args=SimpleNamespace(run_id="d2_screening_x"),
                              image_digest="img@595.91.07")
        argv = shlex.split(L.driver_command(ctx, None))
        assert argv[1].endswith(L.DRIVER)
        args = D.build_parser().parse_args(argv[2:])
        assert args.arm_root.endswith(SC.POD_ARMS_REL)
        assert args.image_digest == "img@595.91.07"
        assert args.authorization.endswith(
            "d2_screening_x/governance/authorization.json")

    def test_the_budget_decomposes_the_priced_bound(self):
        from stages.phase_d2.d2_screening_authorization import session_ceiling

        L = self._launcher()
        spec = L.budget_spec(REPO)
        priced = session_ceiling(REPO)
        total = (spec.setup_minutes + spec.transfer_minutes
                 + sum(p.minutes for p in spec.other_phases))
        assert total == pytest.approx(priced["expected_minutes"], abs=0.05)
        with pytest.raises(SystemExit):
            L.budget_spec(REPO, authorized_minutes=priced["expected_minutes"])

    def test_the_arms_travel_as_pinned_relay_inputs(self):
        L = self._launcher()
        arms = [r for r in L.relay_inputs(REPO)
                if r.dest and r.dest.startswith(SC.POD_ARMS_REL)]
        assert len(arms) == 4 * len(SC.ARM_FILES)
        assert all(r.sha256 for r in arms)
        shards = {r.dest.rsplit("/", 1)[1]: r.sha256 for r in arms
                  if r.path.endswith("model.safetensors")}
        for arm in SC.arms(REPO):
            assert shards[arm.state_id] == arm.identities["single_shard_sha256"]

    def test_the_spec_builds_with_its_own_kind_and_loader(self):
        L = self._launcher()
        args = L.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b",
            "--run-id", "d2_screening_x"])
        session = L.spec(args)
        assert session.setup.env["SESSION_KIND"] == "d2_screening"
        assert session.authorization_loader.__self__.__name__ == \
            "D2ScreeningAuthorization"
        assert session.host_admission is L.host_admission
        assert "VLLM_READY" in session.setup.setup_markers


# --- the driver -----------------------------------------------------------------

class TestTheDriverReusesD1sPathAndIsolatesItsDiagnostics:

    def test_every_subprocess_passes_the_launcher_environment(self):
        src = (REPO / "scripts/stages/stage-1/phase_d2/"
                      "autoinit_d2_screening_driver.py").read_text()
        bare = [ast.unparse(n)[:80] for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "subprocess.run"
                and "env" not in {k.arg for k in n.keywords}]
        assert not bare, bare

    def test_the_probe_config_differs_only_in_the_allowed_overrides(
            self, tmp_path):
        from stages.phase_c1.autoinit_c1_driver import (
            C1_PROBE_OVERRIDES, FROZEN_RECIPE, PACK_DIR,
        )
        from stages.phase_d1 import autoinit_d1_behavioural_driver as D1D
        from stages.phase_d2 import autoinit_d2_screening_driver as D

        probe = SC.probes(REPO)[0]
        path = D1D.probe_config(probe, audit=tmp_path,
                                frozen_recipe=FROZEN_RECIPE, pack_dir=PACK_DIR,
                                overrides=C1_PROBE_OVERRIDES,
                                out_root=D.PROBE_ROOT_REL, experiment="D2")
        cfg = json.loads(path.read_text())
        frozen = json.loads(FROZEN_RECIPE.read_text())
        assert {k for k in set(cfg) | set(frozen)
                if cfg.get(k) != frozen.get(k)} <= set(C1_PROBE_OVERRIDES)
        assert cfg["out_dir"] == f"{D.PROBE_ROOT_REL}/{probe.probe_id}"
        assert cfg["seed"] == probe.seed

    def test_a_failed_wrapper_falls_back_to_the_plain_generator(
            self, tmp_path, monkeypatch):
        from stages.phase_d2 import autoinit_d2_screening_driver as D

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            assert "env" in kw
            gen = Path(cmd[cmd.index("--out-dir") + 1])
            gen.mkdir(parents=True, exist_ok=True)
            (gen / "partial.json").write_text("{}")
            rc = 3 if str(D.TOKEN_ID_WRAPPER) in cmd else 0
            return SimpleNamespace(returncode=rc, stdout="", stderr="")

        monkeypatch.setattr(D.subprocess, "run", fake_run)
        gen_dir = tmp_path / "generations" / "p"
        row = D.generate(tmp_path / "pkg", ["a.jsonl"], gen_dir, label="p",
                         image_digest=None, timeout_s=10, out=tmp_path)
        assert len(calls) == 2
        assert str(D.TOKEN_ID_WRAPPER) in calls[0]
        assert str(D.TOKEN_ID_WRAPPER) not in calls[1]
        assert calls[0][-7:] == calls[1][-7:], "the same argv, minus the flag"
        assert row["via"] == "plain_generator_after_wrapper_failure"
        assert row["plain_rc"] == 0
        assert Path(row["wrapper_partial_preserved_at"]).is_dir()

    def test_a_different_training_runtime_is_refused_before_training(
            self, monkeypatch):
        from aadistill.initialization.planning import recovery
        from stages.phase_d2 import autoinit_d2_screening_driver as D

        monkeypatch.setattr(recovery.RuntimeEnvironmentFingerprint, "observe",
                            classmethod(lambda cls, **kw: cls(
                                image_digest="other@1", python_version="3.12",
                                torch_version="x", transformers_version="y",
                                cuda_runtime="z", attention_backend="sdpa")))
        with pytest.raises(D.D2ScreeningDriverError, match="not be paired"):
            D.training_runtime("other@1")

    def test_check_only_derives_the_contract_and_trains_nothing(self, tmp_path):
        from stages.phase_d2 import autoinit_d2_screening_driver as D

        rc = D.main(["--out", str(tmp_path / "audit"), "--run-id",
                     "d2_screening_check", "--check-only",
                     "--status", str(tmp_path / "status")])
        assert rc == 0
        rec = json.loads((tmp_path / "audit" / D.EVIDENCE_FILENAME).read_text())
        assert rec["status"] == "CHECK_ONLY_OK"
        assert rec["contract_hash"] == SC.contract_hash(SC.session_contract(REPO))
        assert rec["probes"] == []
        assert "CHECK_ONLY_OK" in (tmp_path / "status").read_text()


# --- derived evidence, corrected ------------------------------------------------

class TestTheAlignmentCountsArtifactsNotRecords:

    def test_two_records_of_one_artifact_are_one_comparison(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_an", REPO / "scripts/stages/stage-1/phase_d2/analyze_d2_search.py")
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)

        def state(sid, digest, **v):
            return {"state_id": sid, "artifact_digest": digest,
                    "path_label": sid, "evaluation": {"values": v}}

        d2 = {"s1": state("s1", "d", **{A.TOPK: 0.5, A.RAW: 0.6, A.OBJ: 0.4}),
              "s2": state("s2", "d", **{A.TOPK: 0.5, A.RAW: 0.6, A.OBJ: 0.4})}
        d1 = tmp_path / "d1.jsonl"
        d1.write_text("".join(json.dumps(state(f"t{i}", "d", **{A.D1KL: 0.51}))
                              + "\n" for i in range(2)))
        out = A.d1_alignment(d2, d1)
        assert (out["n_d2_state_records"], out["n_unique_artifacts"],
                out["independent_comparisons"]) == (2, 1, 1)
        assert out["artifacts"][0]["abs_difference_root_topk_vs_d1"] == \
            pytest.approx(0.01)

    def test_the_v1_files_are_frozen_and_the_erratum_binds_them(self):
        import hashlib

        run = REPO / SC.SEARCH_RUN_REL / "evidence"
        erratum = json.loads((run / "derived_evidence_erratum_v1.json")
                             .read_text())
        for rel, sha in erratum["frozen_v1_files_unchanged"].items():
            assert hashlib.sha256((REPO / SC.SEARCH_RUN_REL / rel)
                                  .read_bytes()).hexdigest() == sha
        for rel, sha in erratum["superseded_by"].items():
            if rel.startswith("evidence/"):
                assert hashlib.sha256((REPO / SC.SEARCH_RUN_REL / rel)
                                      .read_bytes()).hexdigest() == sha
        leaves = json.loads((run / "complete_leaf_ranking_v2.json").read_text())
        assert all(x["state_teacher_kl_equals_raw_union_kl"]
                   for x in leaves["leaves"])
        assert not any("d1_metric_teacher_kl_equal_domain_mean" in x
                       for x in leaves["leaves"])


# --- descriptive candidate differences -----------------------------------------

class TestTheCandidateDifferencesAreDescriptive:

    def test_the_record_shows_two_ffn_masks_and_four_attention_masks(self):
        doc = json.loads((REPO / SC.SEARCH_RUN_REL / "evidence"
                          / "candidate_differences.json").read_text())
        assert doc["distinct_ffn_masks"] == 2
        assert doc["distinct_attention_masks"] == 4
        for pair in doc["weights"]["pairs"].values():
            for kind, counts in pair["shared_units"].items():
                for k, v in counts.items():
                    if k.startswith("shared"):
                        twin = k.replace("shared", "bitwise_identical")
                        twin = "bitwise_identical" if k == "shared" else twin
                        assert counts[twin] == v, (kind, k)

    def test_no_rule_reads_it(self):
        """No CODE path names the file (a docstring may describe it)."""
        for rel in ("scripts/stages/stage-1/phase_d2/d2_screening.py",
                    "scripts/stages/stage-1/phase_d2/rank_d2_screening.py",
                    "scripts/stages/stage-1/phase_d2/"
                    "autoinit_d2_screening_driver.py"):
            tree = ast.parse((REPO / rel).read_text())
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef,
                                      ast.ClassDef))
                    and n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)}
            code = [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in docs]
            assert not any("candidate_differences" in v for v in code), rel

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


OTHER_ARM_PATH = (f"/workspace/aad/{SC.POD_ARMS_REL}/"
                  "c2cb27efcf68b4f18bf7284aea84877a")


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    """The consistent candidate unit, built once, with the REAL re-score
    cached: a mutation that edits a score is re-checked against the scorer's
    actual output for these generations, never against a stub's opinion."""
    from stages.d_series import shared_evaluation as SE

    built = TestOffPodVerificationRefusesWhatIsNotPaired._build(
        tmp_path_factory.mktemp("unit"))
    cache: dict = {}

    def rescore(*args, **kwargs):
        key = (kwargs["label"], kwargs["seed"], kwargs["screening_arm"],
               tuple(kwargs.get("extra_args", ())))
        if key not in cache:
            cache[key] = SE.rescore_unit(*args, **kwargs)
        return cache[key]

    built["rescore"] = rescore
    return built


class TestOffPodVerificationRefusesWhatIsNotPaired:
    """The ranking gate, on a COMPLETE and CONSISTENT candidate unit built
    from real evidence: B's stored generations scored by the real pinned
    scorer AS c1 at its first seed, with training records showing the
    protocol's recovery of c1's committed initialization. Every check is then
    shown to fail on a mutation of the premise it reads, including records
    whose self-hash is re-sealed after the edit."""

    @staticmethod
    def _store():
        from stages.d_series import shared_evaluation as SE

        store = SE.durable_store(SC.SHARED_RUNG, REPO)
        if not store.is_dir():
            pytest.skip(f"B's durable store {store} is not on this host")
        return store

    @classmethod
    def _build(cls, tmp_path) -> dict:
        import hashlib
        import subprocess
        import sys

        from aadistill.infrastructure.manifest import sha256_json
        from stages.d_series import shared_evaluation as SE

        store = cls._store()
        contract = SC.session_contract(REPO, verify_bytes=False)
        probe = contract["probes"][0]
        arm = next(a for a in contract["field"]["arms"]
                   if a["arm"] == probe["arm"])
        pid, seed = probe["probe_id"], probe["seed"]
        b_id = f"d1_screening_B_s{seed}"
        b_dir = store / "store" / "probes" / b_id
        record = SC.protocol(REPO)
        block = record["recovery"]
        rung = record["rungs"][SC.SHARED_RUNG]

        probe_dir = tmp_path / "probes" / pid
        shutil.copytree(b_dir / "generations", probe_dir / "generations")
        b_config = json.loads((REPO / "logs/stages/stage-1/phase_d1/runs/"
                               f"{SE.RUNGS[SC.SHARED_RUNG]['source_run']}/"
                               f"evidence/configs/{b_id}.json").read_text())
        student = f"/workspace/aad/{SC.POD_ARMS_REL}/{arm['state_id']}"
        config = {**b_config, "run_name": pid, "seed": seed,
                  "student_path": student,
                  "out_dir": f"artifacts/stages/stage-3/d2_screening/{pid}",
                  "_purpose": "D2 screening probe"}
        training = probe_dir / "training"
        training.mkdir(parents=True)
        (training / "run_manifest.json").write_text(json.dumps({
            "config": config, "config_sha256": sha256_json(config),
            "student_source": student,
            "execution": {"trainer_source": {
                "digest": block["trainer_source_digest"]},
                "runtime_digest": block["training_runtime_digest"],
                "runtime": {"image_digest": block["training_image_digest"]}},
            "hardware": {"cuda_devices": [block["device_class"]]},
            "teacher": dict(zip(("model_id", "revision"),
                                block["teacher"].split("@"))),
            "ladder": {"blocks_sha256": block["data_digest"]}}))
        (training / "run_completion.json").write_text(json.dumps({
            "config_sha256": sha256_json(config), "final_step": 1023,
            "planned_total_steps": 1023, "completed_all_steps": True}))
        b_admission = json.loads((b_dir / "admission.json").read_text())
        fingerprint = b_admission["generation_fingerprint"]
        (probe_dir / "admission.json").write_text(json.dumps(
            {**b_admission, "probe_id": pid}))
        proc = subprocess.run(
            [sys.executable, str(REPO / SE.RUNGS[SC.SHARED_RUNG]["scorer"]),
             "--generations", str(probe_dir / "generations"), "--label", pid,
             "--seed", str(seed), "--out", str(probe_dir / "result.json"),
             "--per-sample", str(probe_dir / "per_sample.jsonl"),
             "--screening-arm", probe["arm"],
             "--init-digest", arm["identities"]["artifact_digest"],
             "--trained-run", str(training / "run_completion.json"),
             "--generation-fingerprint", fingerprint],
            capture_output=True, text=True, cwd=REPO,
            env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr[-800:]
        result = json.loads((probe_dir / "result.json").read_text())
        session = {
            "probe_id": pid, "arm": probe["arm"], "seed": seed,
            "state_id": arm["state_id"], "scored": True,
            "result_sha256": result["result_sha256"],
            "generation_admission": {
                "evaluation_protocol_hash": rung["evaluation_protocol_hash"],
                "generation_fingerprint": fingerprint},
            "training_records_sha256": {
                name: hashlib.sha256((training / name).read_bytes()
                                     ).hexdigest()
                for name in SC.TRAINING_RECORDS}}
        expected = {"probe_id": pid, "arm": probe["arm"], "seed": seed,
                    "state_id": arm["state_id"],
                    "artifact_digest": arm["identities"]["artifact_digest"]}
        return {"dir": probe_dir, "session": session, "expected": expected,
                "contract": contract}

    @staticmethod
    def _copy(unit, tmp_path) -> tuple[Path, dict, dict]:
        probe_dir = tmp_path / "p"
        shutil.copytree(unit["dir"], probe_dir)
        return (probe_dir, json.loads(json.dumps(unit["session"])),
                dict(unit["expected"]))

    @staticmethod
    def _reseal(path: Path, edit) -> dict:
        from aadistill.infrastructure.manifest import sha256_json

        doc = json.loads(path.read_text())
        edit(doc)
        doc.pop("result_sha256", None)
        doc["result_sha256"] = sha256_json(doc)
        path.write_text(json.dumps(doc))
        return doc

    def test_the_consistent_unit_verifies_with_the_real_scorer(self, unit):
        from stages.phase_d2 import rank_d2_screening as R

        got = R.verify_unit(unit["dir"], unit["session"], unit["expected"],
                            repo_root=REPO)
        assert got["verified"] is True, {k: v for k, v in
                                         got["checks"].items() if v is not True}
        assert got["training"]["training_runtime_digest"] == SC.protocol(
            REPO)["recovery"]["training_runtime_digest"]

    @staticmethod
    def _resealed_training(probe_dir, session, name, edit):
        """Edit a training record AND re-seal everything that binds it -- the
        session's digest of it, and for the completion record the result's
        trained_run -- so only the CONTENT check can refuse it."""
        import hashlib

        from aadistill.infrastructure.manifest import sha256_json

        path = probe_dir / "training" / name
        doc = json.loads(path.read_text())
        edit(doc)
        if name == "run_manifest.json" and "config" in doc:
            doc["config_sha256"] = sha256_json(doc["config"])
            comp = probe_dir / "training" / "run_completion.json"
            c = json.loads(comp.read_text())
            c["config_sha256"] = doc["config_sha256"]
            comp.write_text(json.dumps(c))
        path.write_text(json.dumps(doc))
        for n in SC.TRAINING_RECORDS:
            session["training_records_sha256"][n] = hashlib.sha256(
                (probe_dir / "training" / n).read_bytes()).hexdigest()
        comp_sha = session["training_records_sha256"]["run_completion.json"]
        comp_doc = json.loads((probe_dir / "training" /
                               "run_completion.json").read_text())

        def bind(r):
            r["trained_run"]["run_completion_sha256"] = comp_sha
            r["trained_run"]["config_sha256"] = comp_doc["config_sha256"]
        res = TestOffPodVerificationRefusesWhatIsNotPaired._reseal(
            probe_dir / "result.json", bind)
        session["result_sha256"] = res["result_sha256"]

    MUTATIONS = {
        "result_self_hash": lambda d, s, e, T: (
            (d / "result.json").write_text(json.dumps({
                **json.loads((d / "result.json").read_text()),
                "correct_overall": 0.99}))),
        "result_is_the_session_record": lambda d, s, e, T: s.update(
            result_sha256="0" * 64),
        "label_is_the_frozen_probe": lambda d, s, e, T: e.update(
            probe_id="d2_screening_c1_s614923639"),
        "seed_is_the_frozen_seed": lambda d, s, e, T: e.update(seed=614923639),
        "arm_is_the_frozen_arm": lambda d, s, e, T: e.update(arm="c2"),
        "initialization_is_the_committed_candidate": lambda d, s, e, T:
            e.update(artifact_digest="f" * 64),
        "session_entry_is_the_frozen_probe": lambda d, s, e, T: s.update(
            state_id="c2cb27efcf68b4f18bf7284aea84877a"),
        "admitted_under_the_shared_rung": lambda d, s, e, T: s[
            "generation_admission"].update(evaluation_protocol_hash="0" * 64),
        "admission_record_is_this_probes": lambda d, s, e, T: (
            d / "admission.json").write_text(json.dumps({
                **json.loads((d / "admission.json").read_text()),
                "generation_fingerprint": "0" * 64})),
        "generations_sha256": lambda d, s, e, T: (
            d / "generations" / "code.generations.jsonl").write_bytes(
            (d / "generations" / "code.generations.jsonl").read_bytes()
            + b"\n"),
        "per_sample_present": lambda d, s, e, T: (
            d / "per_sample.jsonl").unlink(),
        "per_sample_sha256": lambda d, s, e, T: (
            d / "per_sample.jsonl").write_text(
            (d / "per_sample.jsonl").read_text().replace(
                '"correct": false', '"correct": true', 1)),
        "rows_counts_are_the_results": lambda d, s, e, T: s.update(
            result_sha256=T._reseal(d / "result.json", lambda r: r.update(
                n=r["n"] + 1))["result_sha256"]),
        "rescore_reproduces_the_result": lambda d, s, e, T: s.update(
            result_sha256=T._reseal(d / "result.json", lambda r: r.update(
                correct_overall=r["correct_overall"] + 0.05))["result_sha256"]),
        "training_records_are_the_sessions": lambda d, s, e, T: s[
            "training_records_sha256"].update({"run_manifest.json": "0" * 64}),
        "completion_is_the_results": lambda d, s, e, T: s.update(
            result_sha256=T._reseal(d / "result.json", lambda r: r[
                "trained_run"].update(run_completion_sha256="0" * 64))[
                "result_sha256"]),
        "training_completed_every_step": lambda d, s, e, T:
            T._resealed_training(d, s, "run_completion.json",
                                 lambda c: c.update(final_step=511)),
        "training_identity_is_the_protocols": lambda d, s, e, T:
            T._resealed_training(d, s, "run_manifest.json", lambda m: m[
                "execution"].update(runtime_digest="0" * 64)),
        "recipe_digest_is_the_protocols": lambda d, s, e, T:
            T._resealed_training(d, s, "run_manifest.json", lambda m: m[
                "config"]["optim"].update(lr=1e-4)),
        #: Another arm's staged initialization, consistent everywhere else
        #: (student_source too), so only the state-id comparison can refuse.
        "trained_from_the_candidate_at_the_seed": lambda d, s, e, T:
            T._resealed_training(d, s, "run_manifest.json", lambda m: (
                m["config"].update(student_path=OTHER_ARM_PATH),
                m.update(student_source=OTHER_ARM_PATH))),
        "config_is_bound_end_to_end": lambda d, s, e, T: (
            d / "training" / "run_manifest.json").write_text(json.dumps({
                **json.loads((d / "training" / "run_manifest.json"
                              ).read_text()), "config_sha256": "0" * 64})),
    }

    @pytest.mark.parametrize("check", sorted(MUTATIONS))
    def test_each_check_refuses_a_mutation_of_its_premise(self, unit, tmp_path,
                                                          check):
        from stages.phase_d2 import rank_d2_screening as R

        probe_dir, session, expected = self._copy(unit, tmp_path)
        self.MUTATIONS[check](probe_dir, session, expected, type(self))
        got = R.verify_unit(probe_dir, session, expected, repo_root=REPO,
                            rescore=unit["rescore"])
        assert got["verified"] is False
        assert got["checks"][check] is False, got["checks"]

    def test_every_check_is_falsifiable(self, unit):
        """No check may be vacuous: each one is the target of a mutation
        above, except two whose premise the shared evidence checks pin
        jointly with another."""
        from stages.phase_d2 import rank_d2_screening as R

        got = R.verify_unit(unit["dir"], unit["session"], unit["expected"],
                            repo_root=REPO, rescore=unit["rescore"])
        covered = set(self.MUTATIONS) | {
            "per_sample_rows", "per_sample_pairable",
            "rows_carry_the_frozen_unit", "rescore_reproduces_per_sample"}
        assert set(got["checks"]) - covered == set()

    def test_a_relabelled_row_and_a_forged_rate_need_the_real_scorer(
            self, unit, tmp_path):
        """Rows edited AND the result re-sealed over them: every hash agrees,
        and only the pinned scorer, re-run on the stored generations, says the
        rows are not what those generations score to."""
        import hashlib

        from stages.phase_d2 import rank_d2_screening as R

        probe_dir, session, expected = self._copy(unit, tmp_path)
        rows = probe_dir / "per_sample.jsonl"
        rows.write_text(rows.read_text().replace('"correct": false',
                                                 '"correct": true', 3))
        sha = hashlib.sha256(rows.read_bytes()).hexdigest()
        res = self._reseal(probe_dir / "result.json",
                           lambda r: r.update(per_sample_sha256=sha))
        session["result_sha256"] = res["result_sha256"]
        got = R.verify_unit(probe_dir, session, expected, repo_root=REPO)
        assert got["checks"]["per_sample_sha256"] is True
        assert got["checks"]["rescore_reproduces_per_sample"] is False
        assert got["verified"] is False

    def test_a_config_file_that_is_not_the_manifests_is_refused(self, unit,
                                                                tmp_path):
        from stages.phase_d2 import rank_d2_screening as R

        probe_dir, session, expected = self._copy(unit, tmp_path)
        manifest = json.loads((probe_dir / "training" /
                               "run_manifest.json").read_text())
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps(manifest["config"]))
        ok = R.verify_unit(probe_dir, session, expected, repo_root=REPO,
                           config_path=cfg, rescore=unit["rescore"])
        assert ok["checks"]["config_file_is_the_manifests"] is True
        cfg.write_text(json.dumps({**manifest["config"], "seed": 1}))
        bad = R.verify_unit(probe_dir, session, expected, repo_root=REPO,
                            config_path=cfg, rescore=unit["rescore"])
        assert bad["checks"]["config_file_is_the_manifests"] is False

    def test_the_ranking_refuses_a_session_without_the_fields_arm_check(
            self, tmp_path):
        from stages.phase_d2 import rank_d2_screening as R

        contract = SC.session_contract(REPO, verify_bytes=False)
        session = {"status": "COMPLETE",
                   "contract_hash": SC.contract_hash(contract),
                   "probes": [], "arm_verification": {"arms": []}}
        with pytest.raises(R.RankingRefused, match="arm verification"):
            R.rank_run(session, tmp_path, REPO)

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
        (run / "session.json").write_text(json.dumps({"pod_id": "p1"}))
        with pytest.raises(A.D2ScreeningRefused, match="no booked cost"):
            A.consumed_by_prior_subruns(tmp_path)
        closeout = run.parent / "closeout"
        closeout.mkdir()
        (closeout / "outcome.json").write_text(json.dumps(
            {"money": {"all_in_usd": 2.5},
             "provider_teardown_verified": True}))
        assert A.consumed_by_prior_subruns(tmp_path)["consumed_usd"] == 2.5

    @pytest.mark.parametrize("files,cost", [
        #: closed: booked cost AND verified teardown
        ({"closeout/outcome.json": {"money": {"all_in_usd": 3.1},
                                    "provider_teardown_verified": True}}, 3.1),
        #: a booked cost while the resource may still bill is not closed
        ({"closeout/outcome.json": {"money": {"all_in_usd": 3.1}}}, None),
        ({"closeout/outcome.json": {"money": {"all_in_usd": 3.1},
                                    "provider_teardown_verified": False}},
         None),
        #: a closeout with no booked cost, unless nothing was ever created
        ({"closeout/outcome.json": {"provider_teardown_verified": True}}, None),
        ({"closeout/outcome.json": {"provider_resource_created": False}}, 0.0),
        ({"closeout/outcome.json": "{not json"}, None),
        #: no closeout: only a record that NOTHING was created is $0
        ({"runtime/session.json": {"provider_resource_created": False,
                                   "terminal": "DRY_RUN_GATES_PASSED"}}, 0.0),
        ({"runtime/session.json": {"provider_resource_created": True,
                                   "pod_id": "p"}}, None),
        ({"runtime/session.json": {"provider_resource_created": False,
                                   "draws": [{"pod_id": "p"}]}}, None),
        ({"runtime/session.json": {"terminal": "REFUSED"}}, None),
        ({"runtime/session.json": "{not json"}, None),
        #: an issued authorization with nothing recorded about its use
        ({"governance/authorization.json": {"run_id": "x"}}, None),
        ({}, 0.0),
    ])
    def test_only_a_closed_or_never_created_attempt_has_a_known_cost(
            self, tmp_path, files, cost):
        from stages.phase_d2 import d2_screening_authorization as A

        run = tmp_path / "d2_screening_x"
        run.mkdir()
        for rel, doc in files.items():
            (run / rel).parent.mkdir(parents=True, exist_ok=True)
            (run / rel).write_text(doc if isinstance(doc, str)
                                   else json.dumps(doc))
        got, basis = A._subrun_cost(run)
        assert got == cost, basis

    def test_the_envelope_must_hold_the_ceiling_and_the_teardown_reserve(
            self):
        from stages.d_series import d2_budget as BUD
        from stages.phase_d2 import d2_screening_authorization as A

        hard = A.reprice_at(1.09, REPO)["hard_ceiling_usd"]
        exact = round(25.0 - hard - BUD.TEARDOWN_RESERVE_USD, 4)
        payload = self._payload(consumed_usd=exact)
        assert payload["hard_cap_usd"] == pytest.approx(hard)
        assert payload["money"]["envelope_required_for_this_session_usd"] == \
            pytest.approx(hard + BUD.TEARDOWN_RESERVE_USD)
        #: One cent short of the reserve is refused, though the ceiling
        #: itself would still fit.
        with pytest.raises(A.D2ScreeningRefused, match="teardown reserve"):
            self._payload(consumed_usd=round(exact + 0.01, 4))
        assert 25.0 - (exact + 0.01) > hard


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


# --- durability of what the ranking reads ---------------------------------------

class TestTheRankingsTrainingRecordsAreDurable:
    """D1's durability functions under D2's layout, driven through the REAL
    poll and closeout hooks with only the transfers simulated. The run
    manifest and completion record the off-pod ranking reads must be secured
    against the driver's digests BEFORE the pod may release a probe, before a
    scored probe counts as evidence-complete, and before the preserved copy --
    possibly the only off-pod one -- is pruned."""

    SHARD = b"trained-weights-of-d2-probe"
    PID = "d2_screening_c1_s516448251"

    def _pod(self, L, *, manifest=b'{"config": {}}', scored=False,
             recorded_manifest=None):
        import hashlib

        root = f"{L.LAYOUT.pod_probe_root}/{self.PID}"
        files = {f"{root}/run_completion.json": b'{"final_step": 1023}',
                 f"{root}/checkpoints/latest.txt": b"step_001023",
                 f"{root}/train_log.jsonl": b'{"step": 1}\n'}
        if manifest is not None:
            files[f"{root}/run_manifest.json"] = manifest
        ev = f"/workspace/aad/{L.LAYOUT.evidence_dir}"
        if scored:
            files.update({f"{ev}/results/{self.PID}.json": b'{"r": 1}',
                          f"{ev}/per_sample/{self.PID}.jsonl": b'{"id": 1}\n',
                          f"{ev}/{self.PID}_generation_admission.json":
                              b'{"a": 1}'})
        digest = lambda b: hashlib.sha256(b).hexdigest()       # noqa: E731
        unit = {"probe_id": self.PID, "trained": True, "scored": scored,
                "model_dir": f"{root}/checkpoints/step_001023/model",
                "trained_sha256": digest(self.SHARD),
                "training_records_sha256": {
                    "run_manifest.json": digest(
                        recorded_manifest if recorded_manifest is not None
                        else (manifest or b"")),
                    "run_completion.json": digest(
                        files[f"{root}/run_completion.json"])}}
        return files, unit

    def _ctx(self, L, tmp_path, monkeypatch, files, unit, *, terminal=None):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L

        relay = tmp_path / "relay"
        relay.mkdir(exist_ok=True)
        (relay / L.EVIDENCE_FILENAME).write_text(json.dumps(
            {"probes": [unit]}))
        shard = self.SHARD

        def fake_scp(ctx, remote, dest, *, recursive=False, limit_min=5):
            dest = Path(dest)
            if recursive:
                if remote.endswith("/model"):
                    target = dest / "model"
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "config.json").write_text("{}")
                    (target / "model.safetensors").write_bytes(shard)
                    return 0
                return 1
            data = files.get(remote)
            if data is None:
                return 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return 0

        acks = []
        monkeypatch.setattr(D1L, "_scp_from_pod", fake_scp)
        monkeypatch.setattr(D1L, "_scp_to_pod",
                            lambda ctx, src, remote, **kw: acks.append(src)
                            or 0)
        ctx = SimpleNamespace(
            args=SimpleNamespace(scr=str(tmp_path), ckpt_store=None,
                                 ckpt_fetch_limit_min=45),
            evidence={"terminal": terminal} if terminal else {},
            say=lambda _m: None)
        return ctx, acks

    def test_the_records_are_secured_when_training_completes(
            self, tmp_path, monkeypatch):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        files, unit = self._pod(L)
        ctx, _ = self._ctx(L, tmp_path, monkeypatch, files, unit)
        L.secure_probe_evidence(ctx)
        assert not ctx.evidence.get("on_poll_errors")
        row = ctx.evidence["trained_unscored_preserved"][self.PID]
        assert row["verified"] is True
        assert row["completeness"]["training_records"] is True
        assert D1L.training_records_ok(ctx, self.PID, L.LAYOUT)
        kept = tmp_path / "store/probes" / self.PID / "training"
        assert sorted(p.name for p in kept.iterdir()) == sorted(
            SC.TRAINING_RECORDS)

    def test_bytes_that_are_not_the_drivers_are_not_secured(
            self, tmp_path, monkeypatch):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        files, unit = self._pod(L, manifest=b'{"tampered": 1}',
                                recorded_manifest=b'{"config": {}}')
        ctx, _ = self._ctx(L, tmp_path, monkeypatch, files, unit)
        L.secure_probe_evidence(ctx)
        row = ctx.evidence["trained_unscored_preserved"][self.PID]
        #: Part of the resume set for this layout: a replacement pod restoring
        #: this copy could never secure the record again.
        assert row["completeness"]["training_records"] is False
        assert row["verified"] is False
        assert D1L.training_record_status(ctx, self.PID, L.LAYOUT) == {
            "run_manifest.json": False, "run_completion.json": True}

    def test_the_release_and_evidence_complete_wait_for_the_records(
            self, tmp_path, monkeypatch):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        good = b'{"config": {"seed": 1}}'
        files, unit = self._pod(L, manifest=None, scored=True,
                                recorded_manifest=good)
        ctx, acks = self._ctx(L, tmp_path, monkeypatch, files, unit)
        L.secure_probe_evidence(ctx)
        state = ctx.evidence["probe_evidence_secured"][self.PID]
        assert state["result.json"] and state["per_sample.jsonl"]
        assert not state.get("release_acked") and acks == []
        fetched = L.fetch_probe_evidence(ctx)
        assert fetched[0]["training_records"]["run_manifest.json"] is False
        assert D1L.probe_fetch_ok(fetched[0]) is False
        #: Refused -- here first by the checkpoint gate, because for this
        #: layout the manifest is part of the preserved resume set too.
        assert L.probes_evidence_secured(ctx, fetched)[0] is False
        #: The manifest becomes obtainable: the next poll secures it, the ack
        #: goes, and the probe is evidence-complete.
        files[f"{L.LAYOUT.pod_probe_root}/{self.PID}/run_manifest.json"] = good
        L.secure_probe_evidence(ctx)
        assert ctx.evidence["probe_evidence_secured"][self.PID][
            "release_acked"] is True and len(acks) == 1
        fetched = L.fetch_probe_evidence(ctx)
        assert D1L.probe_fetch_ok(fetched[0]) is True

    def test_a_durable_checkpoint_alone_does_not_release_the_probe(
            self, tmp_path, monkeypatch):
        """The checkpoint is preserved and verified, but the records never
        reach the store beside the evidence: the pod's probe root still
        holds the only copy the ranking would read, so no ack."""
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        files, unit = self._pod(L, scored=True)
        ctx, acks = self._ctx(L, tmp_path, monkeypatch, files, unit)
        monkeypatch.setattr(D1L, "secure_training_records",
                            lambda *a, **k: None)
        L.secure_probe_evidence(ctx)
        assert ctx.evidence["trained_unscored_preserved"][self.PID][
            "verified"] is True
        assert not ctx.evidence["probe_evidence_secured"][self.PID].get(
            "release_acked") and acks == []

    def test_the_prune_keeps_the_only_verified_copy(self, tmp_path,
                                                    monkeypatch):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        files, unit = self._pod(L, scored=True)
        ctx, _ = self._ctx(L, tmp_path, monkeypatch, files, unit)
        L.secure_probe_evidence(ctx)
        dest = Path(ctx.evidence["trained_unscored_preserved"][self.PID][
            "dest"])
        kept = tmp_path / "store/probes" / self.PID / "training"
        manifest = kept / "run_manifest.json"
        before = manifest.read_bytes()
        #: The beside-the-evidence copy is damaged after it was verified: the
        #: preserved copy is now the only good one and must survive the prune.
        manifest.write_bytes(b"damaged")
        ok, why = L.probes_evidence_secured(ctx, [])
        assert ok is False and "training records" in why
        ctx.evidence["terminal"] = "ALL_DONE"
        D1L.prune_preserved_scored(ctx, L.LAYOUT)
        row = ctx.evidence["trained_unscored_preserved"][self.PID]
        assert dest.is_dir() and not row.get("pruned")
        assert "training records" in row["prune_withheld"]
        #: Restored and re-verified: the prune proceeds, and never writes over
        #: the verified copy beside the evidence.
        manifest.write_bytes(before)
        D1L.prune_preserved_scored(ctx, L.LAYOUT)
        assert row["pruned"] is True and not dest.exists()
        assert manifest.read_bytes() == before
        assert D1L.training_records_ok(ctx, self.PID, L.LAYOUT)

    def test_d1s_layout_requires_nothing_new(self, tmp_path):
        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L

        ctx = SimpleNamespace(args=SimpleNamespace(scr=str(tmp_path)),
                              evidence={})
        assert D1L.D1_LAYOUT.required_training_records == ()
        assert D1L.training_records_ok(ctx, "anything") is True
        assert D1L.probe_fetch_ok({"result.json": True,
                                   "per_sample.jsonl": True,
                                   "admission.json": True}) is True

    def test_the_driver_digests_the_records_the_layout_requires(self):
        src = (REPO / "scripts/stages/stage-1/phase_d2/"
                      "autoinit_d2_screening_driver.py").read_text()
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        assert L.LAYOUT.required_training_records == SC.TRAINING_RECORDS
        assert 'entry["training_records_sha256"]' in src
        assert "SC.TRAINING_RECORDS" in src


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

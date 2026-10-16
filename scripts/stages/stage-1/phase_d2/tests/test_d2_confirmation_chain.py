"""D2's shared-confirmation chain at `$0`: contract, verdict, unit identity,
grant, launcher, driver, durability.

Each class pins what the paid confirmation rests on, and each refusal is shown
to fire on the input it claims to read.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]

from stages.phase_d2 import d2_confirmation as CF  # noqa: E402

D1_CONFIRMATION_RUN = "d1_behavioural_20261012_064341"
SEEDS = [396880939, 766527341, 1004399678]


# --- the contract -------------------------------------------------------------

class TestTheContractBindsTheFrozenCandidate:

    def test_the_candidate_is_the_frozen_screenings_and_is_bound(self):
        c = CF.session_contract(REPO, verify_bytes=False)
        cand = c["candidate"]
        assert (cand["arm"], cand["state_id"]) == (
            "c4", "3a4173e4eff45b0d73fd82bd0b263f03")
        assert len(cand["ranking_sha256"]) == 64
        assert [p["probe_id"] for p in c["probes"]] == [
            f"d2_confirmation_c4_s{s}" for s in SEEDS]
        assert c["seeds"] == SEEDS and c["n_probes"] == 3

    def test_every_probe_starts_from_the_initial_search_product(self):
        from stages.phase_d2.d2_product_durability import DEFAULT_SCR

        (arm,) = CF.arms(REPO)
        assert Path(arm.checkpoint_dir) == DEFAULT_SCR / "products" / arm.state_id
        assert "trained" not in arm.checkpoint_dir
        got = CF.require_arms_present(REPO)
        assert got["n_arms"] == 1

    def test_the_evaluation_identity_is_the_shared_confirmation_rungs(self):
        c = CF.session_contract(REPO, verify_bytes=False)
        rung = CF.protocol(REPO)["rungs"]["shared_confirmation"]
        assert c["evaluation"]["evaluation_protocol_hash"] == \
            rung["evaluation_protocol_hash"]
        assert c["evaluation"]["scorer"]["path"].endswith(
            "score_d1_confirmation.py")
        assert c["battery"]["role"] == "d1_confirmation"
        assert sorted(u["seed"] for u in c["comparator"]["B"]) == sorted(SEEDS)

    @pytest.mark.parametrize("edit,match", [
        (lambda r: r["selection"].update(outcome="NONE_ADVANCES"), "outcome"),
        (lambda r: r["selection"].update(arm="c9"), "not a committed"),
        (lambda r: r.update(contract_hash="0" * 64), "another screening"),
        (lambda r: r.update(run_id="d2_screening_other"), "the ranking is run"),
    ])
    def test_a_ranking_that_does_not_name_a_candidate_is_refused(
            self, tmp_path, monkeypatch, edit, match):
        ranking = json.loads((REPO / CF.RANKING_REL).read_text())
        edit(ranking)
        path = tmp_path / "ranking.json"
        path.write_text(json.dumps(ranking))
        monkeypatch.setattr(CF, "RANKING_REL", str(path))
        with pytest.raises(CF.D2ConfirmationError, match=match):
            CF.advancing(REPO)

    @pytest.mark.parametrize("key,value,match", [
        ("alpha_one_sided", 0.05, "alpha"),
        ("bootstrap_seed", 1, "bootstrap seed"),
    ])
    def test_a_registration_off_the_protocol_is_refused(self, monkeypatch, key,
                                                        value, match):
        real = CF.behaviour(REPO)
        edited = json.loads(json.dumps(real))
        edited["confirmation"][key] = value
        monkeypatch.setattr(CF, "behaviour", lambda repo_root=REPO: edited)
        with pytest.raises(CF.D2ConfirmationError, match=match):
            CF.rule(REPO)

    def test_the_screening_contract_is_unchanged_by_the_refactor(self):
        from stages.phase_d2 import d2_screening as SC

        got = SC.contract_hash(SC.session_contract(REPO, verify_bytes=False))
        assert got.startswith("6d4283e7")


# --- the verdict --------------------------------------------------------------

def _reference(arm):
    from stages.d_series import shared_evaluation as SE

    return SE.load_reference_rows("shared_confirmation", arm, repo_root=REPO)


def _skip_without_store():
    from stages.d_series import shared_evaluation as SE

    store = SE.durable_store("shared_confirmation", REPO)
    if not store.is_dir():
        pytest.skip(f"B's durable store {store} is not on this host")


class TestTheFrozenVerdict:

    def test_it_reproduces_d1s_recorded_no_go_on_q2s_rows(self):
        _skip_without_store()
        got = CF.verdict(_reference("q2"), repo_root=REPO)
        d1 = json.loads((REPO / "logs/stages/stage-1/phase_d1/runs" /
                         D1_CONFIRMATION_RUN / "evidence" /
                         "confirmation_verdict.json").read_text())
        assert got["outcome"] == "NO_GO" and d1["decision"]["verdict"] == "NO-GO"
        assert got["primary_vs_B"]["decision"]["delta"] == pytest.approx(
            d1["decision"]["delta"])
        assert got["primary_vs_B"]["per_seed_delta"] == pytest.approx(
            d1["per_seed_delta"])
        assert got["rule"]["bootstrap_seed"] == 450870042

    def test_rows_identical_to_b_never_clear(self):
        _skip_without_store()
        got = CF.verdict(_reference("B"), repo_root=REPO)
        assert got["primary_vs_B"]["decision"]["delta"] == 0.0
        assert got["outcome"] != "SHARED_GO"

    @staticmethod
    def _rows(correct: bool, usable: bool):
        out = {}
        for seed, rows in _reference("B").items():
            out[seed] = [{**r, "correct": bool(r["scorable"]) and correct,
                          "usable": usable} for r in rows]
        return out

    def test_a_clearly_better_candidate_is_shared_go(self):
        _skip_without_store()
        got = CF.verdict(self._rows(correct=True, usable=True), repo_root=REPO)
        assert got["outcome"] == "SHARED_GO"
        assert "never promotes" in got["_shared_go_is_not_a_promotion"] or \
            "nothing is promoted" in got["_shared_go_is_not_a_promotion"]

    def test_the_usable_guardrail_vetoes_even_a_better_correctness(self):
        _skip_without_store()
        got = CF.verdict(self._rows(correct=True, usable=False), repo_root=REPO)
        assert got["outcome"] == "NO_GO"
        assert got["primary_vs_B"]["decision"]["vetoes"]

    def test_the_seeds_must_be_exactly_the_registered_three(self):
        _skip_without_store()
        from stages.d_series.shared_evaluation import SharedEvaluationError

        rows = _reference("q2")
        rows.pop(SEEDS[0])
        with pytest.raises(SharedEvaluationError, match="seeds"):
            CF.verdict(rows, repo_root=REPO)


# --- one unit's identity, off-pod ----------------------------------------------

@pytest.fixture(scope="module")
def conf_unit(tmp_path_factory):
    """A complete, consistent c4 CONFIRMATION unit from real evidence: B's
    stored confirmation generations scored by the real pinned confirmation
    scorer AS c4 at the first confirmation seed, with training records showing
    the protocol's recovery of c4's committed initialization."""
    import hashlib
    import subprocess
    import sys

    from aadistill.infrastructure.manifest import sha256_json
    from stages.d_series import shared_evaluation as SE

    store = SE.durable_store("shared_confirmation", REPO)
    if not store.is_dir():
        pytest.skip("B's confirmation store is not on this host")
    tmp = tmp_path_factory.mktemp("conf")
    contract = CF.session_contract(REPO, verify_bytes=False)
    probe, arm = contract["probes"][0], contract["field"]["arms"][0]
    pid, seed = probe["probe_id"], probe["seed"]
    b_id = f"d1_confirmation_B_s{seed}"
    b_dir = store / "store" / "probes" / b_id
    record = CF.protocol(REPO)
    block, rung = record["recovery"], record["rungs"][CF.SHARED_RUNG]
    probe_dir = tmp / "probes" / pid
    shutil.copytree(b_dir / "generations", probe_dir / "generations")
    b_config = json.loads((REPO / "logs/stages/stage-1/phase_d1/runs" /
                           D1_CONFIRMATION_RUN / "evidence/configs" /
                           f"{b_id}.json").read_text())
    student = f"/workspace/aad/{CF.POD_ARMS_REL}/{arm['state_id']}"
    config = {**b_config, "run_name": pid, "seed": seed,
              "student_path": student, "_purpose": "D2 confirmation probe",
              "out_dir": f"artifacts/stages/stage-3/d2_confirmation/{pid}"}
    training = probe_dir / "training"
    training.mkdir(parents=True)
    (training / "run_manifest.json").write_text(json.dumps({
        "config": config, "config_sha256": sha256_json(config),
        "student_source": student,
        "execution": {"trainer_source": {"digest": block["trainer_source_digest"]},
                      "runtime_digest": block["training_runtime_digest"],
                      "runtime": {"image_digest": block["training_image_digest"]}},
        "hardware": {"cuda_devices": [block["device_class"]]},
        "teacher": dict(zip(("model_id", "revision"),
                            block["teacher"].split("@"))),
        "ladder": {"blocks_sha256": block["data_digest"]}}))
    (training / "run_completion.json").write_text(json.dumps({
        "config_sha256": sha256_json(config), "final_step": 1023,
        "planned_total_steps": 1023, "completed_all_steps": True}))
    b_adm = json.loads((b_dir / "admission.json").read_text())
    fingerprint = b_adm["generation_fingerprint"]
    (probe_dir / "admission.json").write_text(json.dumps({**b_adm,
                                                          "probe_id": pid}))
    proc = subprocess.run(
        [sys.executable, str(REPO / SE.RUNGS[CF.SHARED_RUNG]["scorer"]),
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
    session = {"probe_id": pid, "arm": probe["arm"], "seed": seed,
               "state_id": arm["state_id"], "scored": True,
               "result_sha256": result["result_sha256"],
               "generation_admission": {
                   "evaluation_protocol_hash": rung["evaluation_protocol_hash"],
                   "generation_fingerprint": fingerprint},
               "training_records_sha256": {
                   n: hashlib.sha256((training / n).read_bytes()).hexdigest()
                   for n in CF.TRAINING_RECORDS}}
    expected = {"probe_id": pid, "arm": probe["arm"], "seed": seed,
                "state_id": arm["state_id"],
                "artifact_digest": arm["identities"]["artifact_digest"]}
    return {"dir": probe_dir, "session": session, "expected": expected}


class TestAConfirmationUnitIsVerifiedUnderItsOwnRung:

    def test_the_consistent_unit_verifies_under_the_confirmation_rung(
            self, conf_unit):
        from stages.phase_d2 import rank_d2_screening as R

        got = R.verify_unit(conf_unit["dir"], conf_unit["session"],
                            conf_unit["expected"], repo_root=REPO,
                            shared_rung=CF.SHARED_RUNG)
        assert got["verified"] is True, {k: v for k, v in got["checks"].items()
                                         if v is not True}

    def test_the_screening_rungs_admission_is_not_the_confirmations(
            self, conf_unit, tmp_path):
        """A unit admitted under the SCREENING rung's protocol -- or checked
        against it -- does not verify for the confirmation."""
        from stages.phase_d2 import rank_d2_screening as R
        from stages.phase_d2 import d2_screening as SC

        screening = SC.protocol(REPO)["rungs"]["shared_screening"][
            "evaluation_protocol_hash"]
        session = json.loads(json.dumps(conf_unit["session"]))
        session["generation_admission"]["evaluation_protocol_hash"] = screening
        got = R.verify_unit(conf_unit["dir"], session, conf_unit["expected"],
                            repo_root=REPO, shared_rung=CF.SHARED_RUNG)
        assert got["checks"]["admitted_under_the_shared_rung"] is False
        wrong = R.verify_unit(conf_unit["dir"], conf_unit["session"],
                              conf_unit["expected"], repo_root=REPO,
                              shared_rung="shared_screening")
        assert wrong["verified"] is False

    def test_the_verdict_refuses_an_incomplete_or_foreign_session(self, tmp_path):
        from stages.phase_d2 import verdict_d2_confirmation as V

        contract = CF.session_contract(REPO, verify_bytes=False)
        base = {"status": "COMPLETE", "run_id": "d2_confirmation_t",
                "contract_hash": CF.contract_hash(contract),
                "arm_verification": {"arms": [
                    {"state_id": contract["field"]["arms"][0]["state_id"]}]},
                "probes": []}
        with pytest.raises(V.VerdictRefused, match="not scored"):
            V.verdict_run(base, tmp_path, REPO)
        with pytest.raises(V.VerdictRefused, match="status"):
            V.verdict_run({**base, "status": "FAILED"}, tmp_path, REPO)
        with pytest.raises(V.VerdictRefused, match="different contract"):
            V.verdict_run({**base, "contract_hash": "0" * 64}, tmp_path, REPO)


# --- the grant ------------------------------------------------------------------

def _grant(**extra):
    g = {"rung": "confirmation", "granted_by": "test",
         "cumulative_envelope_usd": 12.0,
         "approves_preregistration_hash": CF.preregistration(REPO)[
             "preregistration_hash"]}
    g.update(extra)
    return g


def _programme(satisfied=True, left=96.97):
    return lambda: {"reservation_rule": {
        "left_after_d2_at_hard_usd": left,
        "needed_for_d3_and_fresh_at_hard_usd": 66.45,
        "satisfied": satisfied, "shortfall_usd": 0.0},
        "d2_confirmation_hard_with_reserve_usd": 8.51}


class TestTheConfirmationGrantIsItsOwnType:

    @staticmethod
    def _payload(**kw):
        from stages.phase_d2 import d2_screening_authorization as A

        args = dict(grant=_grant(), session_commit="a" * 40,
                    granted_utc="2026-10-16T00:00:00+00:00",
                    run_id="d2_confirmation_test", repo_root=REPO,
                    live_rate=1.09, consumed_usd=0.0, rung="confirmation",
                    programme_check=_programme())
        args.update(kw)
        return A.build_payload(**args)

    def test_the_round_trip_binds_the_confirmation_and_refuses_cross_loading(
            self, tmp_path):
        from stages.phase_d2 import d2_screening_authorization as A

        payload = self._payload()
        path = tmp_path / "auth.json"
        path.write_text(json.dumps(payload))
        auth = A.D2ConfirmationAuthorization.load(path)
        auth.require_contract(CF.contract_hash(CF.session_contract(REPO)))
        assert auth.n_probes == 3 and list(auth.seeds) == SEEDS
        assert auth.authorizes_d2_confirmation and not auth.authorizes_d2_screening
        assert not auth.authorizes_fresh_confirmation and not auth.authorizes_d3
        assert auth.hard_cap_usd == pytest.approx(
            A.reprice_at(1.09, REPO, "confirmation")["hard_ceiling_usd"])
        with pytest.raises(A.D2ScreeningRefused):
            A.D2ScreeningAuthorization.load(path)

    def test_a_screening_grant_or_prefix_is_refused(self):
        from stages.phase_d2 import d2_screening_authorization as A

        with pytest.raises(A.D2ScreeningRefused, match="rung"):
            self._payload(grant=_grant(rung="screening"))
        with pytest.raises(A.D2ScreeningRefused, match="prefix"):
            self._payload(run_id="d2_screening_x")

    def test_the_envelope_must_hold_the_ceiling_and_the_reserve(self):
        from stages.d_series import d2_budget as BUD
        from stages.phase_d2 import d2_screening_authorization as A

        hard = A.reprice_at(1.09, REPO, "confirmation")["hard_ceiling_usd"]
        exact = round(12.0 - hard - BUD.TEARDOWN_RESERVE_USD, 4)
        assert self._payload(consumed_usd=exact)["hard_cap_usd"] == \
            pytest.approx(hard)
        with pytest.raises(A.D2ScreeningRefused, match="teardown reserve"):
            self._payload(consumed_usd=round(exact + 0.01, 4))

    def test_the_d3_and_fresh_reservation_must_hold(self):
        from stages.phase_d2 import d2_screening_authorization as A

        with pytest.raises(A.D2ScreeningRefused, match="reservation"):
            self._payload(programme_check=_programme(satisfied=False))
        #: Holding the whole envelope's remainder (12 - 8.51 = 3.49 above the
        #: session) pushes 68.0 below the 66.45 needed... only when it does.
        with pytest.raises(A.D2ScreeningRefused, match="reservation"):
            self._payload(programme_check=_programme(left=69.0))
        assert self._payload(programme_check=_programme(left=70.0))

    def test_an_incomplete_screening_refuses_the_confirmation(self, monkeypatch):
        from stages.d_series import d2_budget as BUD
        from stages.phase_d2 import d2_screening_authorization as A

        monkeypatch.setattr(BUD, "completed_d2_screening",
                            lambda *a, **k: None)
        with pytest.raises(A.D2ScreeningRefused, match="not complete"):
            self._payload()

    def test_the_real_programme_reading_holds(self):
        from stages.phase_d2 import d2_screening_authorization as A

        view = A._programme(REPO)
        assert view["reservation_rule"]["satisfied"] is True


# --- the launcher and the driver -----------------------------------------------

class TestTheConfirmationLauncherAndDriver:

    @staticmethod
    def _launcher():
        from stages.phase_d2 import autoinit_d2_confirmation_launch as C
        return C

    def test_the_spec_builds_with_its_own_kind_loader_and_one_arm(self):
        C = self._launcher()
        args = C.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b",
            "--run-id", "d2_confirmation_x"])
        session = C.spec(args)
        assert session.setup.env["SESSION_KIND"] == C.SESSION_KIND == \
            "d2_confirmation"
        assert session.authorization_loader.__self__.__name__ == \
            "D2ConfirmationAuthorization"
        (arm,) = CF.arms(REPO)
        staged = [r for r in session.setup.relay_inputs
                  if r.dest and r.dest.endswith(arm.state_id)]
        assert len(staged) == 3
        shard = next(r for r in staged if r.path.endswith("model.safetensors"))
        assert shard.sha256 == arm.identities["single_shard_sha256"]
        assert session.setup.test_paths == C.TEST_PATHS
        assert all((REPO / p).is_dir() for p in C.TEST_PATHS)
        for rel in (session.artifacts.spec_success,
                    session.artifacts.spec_failed):
            assert (REPO / rel).is_file()

    def test_the_driver_command_parses_with_the_drivers_parser(self):
        import shlex

        from stages.phase_d2 import autoinit_d2_screening_driver as D

        C = self._launcher()
        args = C.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b",
            "--run-id", "d2_confirmation_x"])
        cmd = C.spec(args).driver_command(
            SimpleNamespace(args=SimpleNamespace(run_id="d2_confirmation_x"),
                            image_digest="img@595.91.07"), None)
        parsed = D.build_parser().parse_args(shlex.split(cmd)[2:])
        assert parsed.rung == "confirmation"
        assert parsed.out.endswith("autoinit_d2_confirmation")
        layout = D.rung_layout("confirmation")
        assert C.LAYOUT.pod_probe_root.endswith(layout["probe_root_rel"])
        assert C.LAYOUT.evidence_filename == layout["evidence_filename"]
        assert C.LAYOUT.release_ack_rel == layout["ack_rel"]
        assert layout["scorer"].name == "score_d1_confirmation.py"

    def test_the_budget_decomposes_the_confirmations_priced_bound(self):
        from stages.phase_d2 import autoinit_d2_screening_launch as L
        from stages.phase_d2.d2_screening_authorization import session_ceiling

        C = self._launcher()
        spec = L.budget_spec(REPO, rung=C.RUNG)
        priced = session_ceiling(REPO, "confirmation")
        total = (spec.setup_minutes + spec.transfer_minutes
                 + sum(p.minutes for p in spec.other_phases))
        assert total == pytest.approx(priced["expected_minutes"], abs=0.05)
        assert priced["n_probes"] == 3

    def test_the_screening_launcher_is_unchanged_in_kind_and_field(self):
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        args = L.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b",
            "--run-id", "d2_screening_x"])
        session = L.spec(args)
        assert session.setup.env["SESSION_KIND"] == "d2_screening"
        assert len([r for r in session.setup.relay_inputs
                    if r.dest and "/arms/" in r.dest]) == 12


class TestAConfirmationFailureResumesItsOwnTrainedProbe:
    """A trained confirmation probe is preserved under the confirmation's
    layout and restored where the driver resumes it (P8.4 state 2): the
    launcher's probe root and the driver's are one path."""

    SHARD = b"trained-confirmation-weights"

    def test_preserve_then_restore_lands_where_the_driver_resumes(
            self, tmp_path, monkeypatch):
        import hashlib

        from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L
        from stages.phase_d1 import autoinit_d1_behavioural_driver as D1D
        from stages.phase_d2 import autoinit_d2_screening_driver as D
        from stages.phase_d2 import autoinit_d2_screening_launch as L

        C = self._launcher()
        pid = "d2_confirmation_c4_s396880939"
        root = f"{C.LAYOUT.pod_probe_root}/{pid}"
        manifest, completion = b'{"config": {}}', b'{"final_step": 1023}'
        files = {f"{root}/run_completion.json": completion,
                 f"{root}/run_manifest.json": manifest,
                 f"{root}/checkpoints/latest.txt": b"step_001023"}
        sha = lambda b: hashlib.sha256(b).hexdigest()           # noqa: E731
        relay = tmp_path / "relay"
        relay.mkdir()
        (relay / C.LAYOUT.evidence_filename).write_text(json.dumps({"probes": [{
            "probe_id": pid, "trained": True, "scored": False,
            "model_dir": f"{root}/checkpoints/step_001023/model",
            "trained_sha256": sha(self.SHARD),
            "training_records_sha256": {"run_manifest.json": sha(manifest),
                                        "run_completion.json": sha(completion)}}]}))

        def fake_scp(ctx, remote, dest, *, recursive=False, limit_min=5):
            dest = Path(dest)
            if recursive:
                target = dest / "model"
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}")
                (target / "model.safetensors").write_bytes(self.SHARD)
                return 0
            if remote not in files:
                return 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[remote])
            return 0

        pushed = []
        monkeypatch.setattr(D1L, "_scp_from_pod", fake_scp)
        monkeypatch.setattr(D1L, "_scp_to_pod",
                            lambda ctx, src, remote, **kw: pushed.append(remote)
                            or 0)
        ctx = SimpleNamespace(args=SimpleNamespace(
            scr=str(tmp_path), ckpt_store=None, ckpt_fetch_limit_min=45,
            restore_trained=None), evidence={}, say=lambda _m: None)
        L.secure_probe_evidence(ctx, C.RUNG)
        row = ctx.evidence["trained_unscored_preserved"][pid]
        assert row["verified"] is True
        assert D1L.training_records_ok(ctx, pid, C.LAYOUT)
        #: The replacement pod: the preserved directory is pushed back to the
        #: confirmation's probe root, which is the driver's model root.
        ctx.args.restore_trained = str(Path(row["dest"]).parent)
        assert L.restore_trained_probes(ctx, C.RUNG) is True
        layout = D.rung_layout("confirmation")
        assert pushed == [f"/workspace/aad/{layout['probe_root_rel']}/{pid}"]
        #: And the driver resumes from that directory without retraining.
        local = Path(row["dest"])
        state = D1D.trained_checkpoint_state(local)
        assert state is not None and state["restored_sha256_verified"] is True

    @staticmethod
    def _launcher():
        from stages.phase_d2 import autoinit_d2_confirmation_launch as C
        return C


# --- the budget contract ---------------------------------------------------------

class TestTheConfirmationLeavesThePlanOnlyByContract:

    def test_no_confirmation_attempt_yet_keeps_it_reserved(self):
        from stages.d_series import d2_budget as BUD

        assert BUD.completed_d2_confirmation() is None
        budget = BUD.build()
        view = BUD.programme_scenarios(
            {"project_remaining_usd": 100.0, "project_cap_usd": 595.0,
             "formal_remaining_usd": 100.0, "engineering_remaining_usd": 3.0,
             "per_session_cap_usd": 30.0}, budget)
        block = view["search_pricing"][view["search_pricing_in_force"]][
            "d2_block"]
        assert block["sessions"] == ["d2_confirmation"]

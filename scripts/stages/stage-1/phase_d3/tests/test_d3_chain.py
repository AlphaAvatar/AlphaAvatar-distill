"""The new D3's $0 chain: design, records, governance, launchers, analysis.

Every check here runs without a GPU, a provider or the network (the provider
quote and the live limits are substituted where the issuer would read them).
The DEPTH arithmetic under the declared teacher reference is the core's and is
tested in `tests/initialization/test_depth_kl_reference.py`.
"""
from __future__ import annotations

import copy
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1", "tests"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d3 import d3_design as D  # noqa: E402


@pytest.fixture(scope="module")
def leaf():
    return D.control_leaf(REPO)


@pytest.fixture(scope="module")
def control_spec(leaf):
    return D.control_spec(leaf)


@pytest.fixture(scope="module")
def treatment(leaf):
    return D.registered_treatment_spec(REPO)


@pytest.fixture(scope="module")
def control_record():
    return json.loads((REPO / D.CONTROL_RECORD).read_text())


# --- 1. the registered design and its records -----------------------------------

class TestTheRecordsAreCurrent:

    def test_the_registration_regenerates_byte_identically(self):
        from stages.phase_d3 import write_d3_registration as W

        body = json.dumps(W.build(REPO), indent=1, sort_keys=True) + "\n"
        assert (REPO / D.REGISTRATION).read_text() == body

    def test_the_budget_regenerates_apart_from_its_dated_live_block(self):
        from stages.phase_d3 import d3_budget as B

        have = json.loads((REPO / B.RECORD).read_text())
        have.pop("live", None)
        assert have == json.loads(json.dumps(B.build()))

    def test_the_control_record_is_the_plans_q2(self, leaf, control_record):
        assert control_record["control_state_id"] == D.CONTROL_STATE_ID
        assert control_record["final"]["artifact_digest"] == \
            D.CONTROL_ARTIFACT_DIGEST
        assert [s["artifact_digest"] for s in control_record["steps"]] == \
            [s.expected_artifact_digest for s in leaf.steps]
        depth = control_record["depth"]
        assert [r["chosen"] for r in depth["rounds"]] == depth["removal_order"]
        assert len(depth["kept_layers"]) == 28

    def test_the_cancelled_d3_is_named_and_not_reused(self):
        reg = D.registration(REPO)
        assert reg["replaces"]["status"].startswith("CANCELLED")
        assert reg["identity_is_distinct_from_the_cancelled_d3"] is True
        assert "NONE" in reg["behavioural_rung"]["exploratory_classification"][
            "alpha"]

    def test_the_registered_seeds_and_battery_are_the_maintainers(self):
        rung = D.registration(REPO)["behavioural_rung"]
        assert rung["seeds"] == [396880939, 766527341, 1004399678]
        assert rung["battery"]["role"] == "d1_confirmation"
        assert (rung["battery"]["n_prompts"], rung["battery"]["n_scorable"]) \
            == (950, 850)
        assert rung["recovery"]["recipe_id"] == "e1_p1_kd_heavy@0.86M"
        assert rung["n_probes"] == 3


# --- 2. the single-factor treatment -------------------------------------------------

class TestTheTreatmentIsOneFactor:

    def test_only_the_declared_fields_differ(self, control_spec, treatment):
        contract = D.treatment_contract(control_spec, treatment)
        assert contract["treated_step"] == 1
        assert all(d["declared"] for d in contract["differences"])
        assert {(d["step"], d["field"]) for d in contract["differences"]} >= {
            (1, "config"), (1, "expected_config_hash")}

    def test_q2s_untreated_input_pins_are_kept(self, control_spec, treatment):
        for i in (0, 2, 3):
            assert treatment.steps[i].expected_config_hash == \
                control_spec.steps[i].expected_config_hash
        assert treatment.steps[0].expected_artifact_digest == \
            control_spec.steps[0].expected_artifact_digest
        assert all(treatment.steps[i].expected_artifact_digest is None
                   for i in (1, 2, 3))

    def _swap(self, spec, index, **fields):
        from aadistill.initialization.planning.fixed_path import (
            FixedPathSpec, FixedPathStep,
        )

        steps = list(spec.steps)
        s = steps[index]
        steps[index] = FixedPathStep(**{
            "impl_id": s.impl_id, "profile_id": s.profile_id,
            "expected_artifact_digest": s.expected_artifact_digest,
            "label": s.label, "config": s.config,
            "expected_config_hash": s.expected_config_hash, **fields})
        return FixedPathSpec(
            path_id=spec.path_id, family=spec.family,
            target_spec=spec.target_spec, steps=tuple(steps),
            root_repo_id=spec.root_repo_id, root_revision=spec.root_revision,
            stats_spec=spec.stats_spec, device=spec.device, seed=spec.seed,
            position_policy=spec.position_policy,
            distribution_support=spec.distribution_support)

    @pytest.mark.parametrize("index,fields", [
        (2, {"profile_id": "calib.domain_balanced@v1"}),
        (0, {"expected_artifact_digest": None}),
        (3, {"expected_config_hash": "0" * 64}),
    ])
    def test_any_other_difference_is_refused(self, control_spec, treatment,
                                             index, fields):
        with pytest.raises(D.D3DesignError, match="single-factor"):
            D.treatment_contract(control_spec,
                                 self._swap(treatment, index, **fields))

    def test_a_second_declared_reference_cannot_even_be_built(self, treatment):
        from aadistill.initialization.planning.fixed_path import FixedPathError

        with pytest.raises(FixedPathError, match="does not accept"):
            self._swap(treatment, 2, config=dict(treatment.steps[1].config))

    def test_the_treated_hash_is_not_q2s(self, control_spec, treatment):
        assert treatment.steps[1].expected_config_hash != \
            control_spec.steps[1].expected_config_hash


# --- 3. the structural outcome --------------------------------------------------------

def _identities(digests):
    return [{"artifact_digest": d, "weights_digest": f"w{d}",
             "single_shard_sha256": f"s{d}", "arch_signature": "a"}
            for d in digests]


class TestTheStructuralOutcome:

    def _control(self):
        return {"depth": {"step": 1, "kept_layers": list(range(28))},
                "steps": [{"kind": k, "artifact_digest": d} for k, d in
                          zip(("FFN", "DEPTH", "WIDTH", "ATTN"), "abcd")],
                "final": {"artifact_digest": "d", "weights_digest": "wd",
                          "single_shard_sha256": "sd", "arch_signature": "a"}}

    def test_same_layers_same_bytes_is_no_structural_change(self):
        got = D.classify_structure(control=self._control(),
                                   depth_kept=list(range(28)),
                                   step_identities=_identities("abcd"))
        assert got["outcome"] == D.NO_STRUCTURAL_CHANGE
        assert "NO recovery training" in got["next"]

    def test_different_layers_is_a_structural_change(self):
        got = D.classify_structure(control=self._control(),
                                   depth_kept=[*range(27), 30],
                                   step_identities=_identities("axyz"))
        assert got["outcome"] == D.STRUCTURAL_CHANGE
        assert got["layers_only_in_treatment"] == [30]

    @pytest.mark.parametrize("digests", ["abcz", "abyd", "axcd"])
    def test_same_layers_different_bytes_anywhere_downstream_is_diagnosed(
            self, digests):
        got = D.classify_structure(control=self._control(),
                                   depth_kept=list(range(28)),
                                   step_identities=_identities(digests))
        assert got["outcome"] == D.UNEXPLAINED_BYTE_DIFFERENCE

    def test_different_layers_with_q2s_final_bytes_is_diagnosed(self):
        got = D.classify_structure(control=self._control(),
                                   depth_kept=[*range(27), 30],
                                   step_identities=_identities("axyd"))
        assert got["outcome"] == D.UNEXPLAINED_BYTE_DIFFERENCE


def _depth_step(control_record, *, removal=None, shift=0.0):
    """A synthetic treated DEPTH step result in the persisted shape."""
    removal = removal or list(control_record["depth"]["removal_order"])
    rounds, cands, removed = [], {}, []
    for r, chosen in enumerate(removal):
        table = []
        for c in range(36):
            if c in removed:
                continue
            key = ",".join(str(x) for x in sorted([*removed, c]))
            root = 1.0 + c * 0.01 if c != chosen else 0.1
            parent = (control_record["depth"]["rounds"][0]["table"][c]["score"]
                      + shift) if r == 0 else 1.0 + (35 - c) * 0.01
            cands[key] = {"round": r, "root_reference_kl": root,
                          "parent_reference_kl": parent}
            table.append({"candidate": c, "score": root})
        rounds.append({"round": r, "removed_before": list(removed),
                       "chosen": chosen, "chosen_score": 0.1,
                       "n_candidates": len(table), "table": table})
        removed.append(chosen)
    kept = sorted(set(range(36)) - set(removal))
    return {"selection": {"search_rounds": rounds,
                          "reference_candidates": cands},
            "trace": {"removal_order": removal, "kept_layers": kept,
                      "removed_layers": sorted(removal),
                      "kl_reference": "root_reference/v1"}}


class TestTheDepthEvidence:

    def test_every_round_reports_what_the_parent_would_have_chosen(
            self, control_record):
        ev = D.depth_evidence(_depth_step(control_record), control_record)
        assert len(ev["rounds"]) == 8
        assert all("parent_reference_would_choose" in r for r in ev["rounds"])
        assert ev["rounds_where_the_parent_reference_would_choose_differently"] \
            >= 1

    def test_a_record_without_the_reference_tables_is_refused(
            self, control_record):
        with pytest.raises(D.D3DesignError, match="declared reference"):
            D.depth_evidence({"selection": {}, "trace": {}}, control_record)

    def test_the_closeout_refuses_a_pod_outcome_it_cannot_rederive(
            self, control_record, leaf):
        from stages.phase_d3 import closeout_d3_structural as C

        steps = [{"index": i, **ident} for i, ident in enumerate(
            _identities([s["artifact_digest"]
                         for s in control_record["steps"]]))]
        steps[1].update(_depth_step(control_record))
        record = {"status": "COMPLETE", "steps": steps,
                  "structural": {"outcome": D.STRUCTURAL_CHANGE}}
        with pytest.raises(C.CloseoutRefused, match="re-derivation"):
            C.rederive(record)


# --- 3b. the parent-reference comparability gate -----------------------------------

@pytest.fixture(scope="module")
def admission():
    return D.admission_record(REPO)


def _r0_candidates(step):
    return {k: v for k, v in step["selection"]["reference_candidates"].items()
            if v["round"] == 0}


class TestTheAdmissionRecord:
    """The tolerance is fixed from the evidence the record carries."""

    def test_it_binds_the_registration_and_the_control(self, admission):
        assert admission["registration_hash"] == D.registration_hash(REPO)
        assert admission["fixed_before_execution"] is True

    def test_the_tolerance_is_the_evidence_rounded_up(self, admission):
        from stages.phase_d3.write_d3_admission import ceil_one_significant

        rows = [r for p in admission["evidence"]["cross_session"]
                for r in p["rows"]]
        max_abs = max(abs(r["other"] - r["base"]) for r in rows)
        max_rel = max(abs(r["other"] - r["base"]) / abs(r["base"])
                      for r in rows)
        tol = admission["tolerance"]
        assert tol["atol"] == ceil_one_significant(max_abs) == 5e-05
        assert tol["rtol"] == ceil_one_significant(max_rel) == 5e-04
        assert all(abs(r["other"] - r["base"])
                   <= tol["atol"] + tol["rtol"] * abs(r["base"]) for r in rows)

    @pytest.mark.parametrize("x,want", [(4.949e-05, 5e-05), (4.432e-04, 5e-04),
                                        (1e-05, 1e-05), (1.01e-05, 2e-05)])
    def test_rounding_is_up_at_one_significant_figure(self, x, want):
        from stages.phase_d3.write_d3_admission import ceil_one_significant

        assert ceil_one_significant(x) == want

    def test_the_same_execution_repeat_is_exact(self, admission):
        rep = admission["evidence"]["same_execution_repeat"]
        assert rep["candidates"] == 260 and rep["max_abs_difference"] == 0.0

    def test_every_other_parents_table_was_refused(self, admission):
        disc = admission["discrimination"]
        assert disc["tables_checked"] == disc["tables_refused"] > 0
        assert disc["min_violations"] > 0
        #: q2's own parent under the other profile: argmin ALSO 26, so only
        #: the numerical check refuses it.
        assert [o["argmin"] for o in disc["same_parent_other_profile"]] == [26]

    def test_the_driver_change_is_stated_not_measured(self, admission):
        drv = admission["evidence"]["host_drivers"]
        assert drv["d1_search"]["drivers"] == drv["d2_search"]["drivers"]
        assert drv["d1_replay_reproduced"]["differ"] == []

    def test_an_edited_record_is_refused(self, tmp_path):
        for rel in (D.REGISTRATION, D.CONTROL_RECORD, D.ADMISSION):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
        D.admission_record(tmp_path)
        doc = json.loads((tmp_path / D.ADMISSION).read_text())
        doc["tolerance"]["rtol"] = 0.5
        (tmp_path / D.ADMISSION).write_text(json.dumps(doc))
        with pytest.raises(D.D3DesignError, match="own admission_hash"):
            D.admission_record(tmp_path)


class TestTheAdmissionGate:
    """The predicate, tabled: each mutation must be refused on its own."""

    def test_q2s_own_table_is_admitted(self, control_record, admission):
        v = D.parent_reference_admission(_depth_step(control_record),
                                         control_record, admission)
        assert v["admissible"] is True and v["failures"] == []
        assert v["n_bit_identical"] == 36 and v["parent_reference_argmin"] == 26

    def test_the_measured_cross_session_variation_is_admitted(
            self, control_record, admission):
        pair = next(p for p in admission["evidence"]["cross_session"]
                    if p["profile_id"] == "calib.reasoning_heavy@v2")
        step = _depth_step(control_record)
        for r in pair["rows"]:
            _r0_candidates(step)[str(r["candidate"])]["parent_reference_kl"] \
                += r["other"] - r["base"]
        v = D.parent_reference_admission(step, control_record, admission)
        assert v["admissible"] is True and v["n_bit_identical"] < 36

    @pytest.mark.parametrize("factor,admitted", [(0.99, True), (1.01, False)])
    def test_the_boundary_is_the_tolerance(self, control_record, admission,
                                           factor, admitted):
        step = _depth_step(control_record)
        rec = _r0_candidates(step)["0"]
        tol = admission["tolerance"]
        rec["parent_reference_kl"] += factor * (
            tol["atol"] + tol["rtol"] * abs(rec["parent_reference_kl"]))
        v = D.parent_reference_admission(step, control_record, admission)
        assert v["admissible"] is admitted

    def _mutate(self, control_record, how):
        step = _depth_step(control_record)
        r0 = step["selection"]["search_rounds"][0]
        cands = step["selection"]["reference_candidates"]
        if how == "no_diagnostic":
            step["selection"]["reference_candidates"] = {}
        elif how == "no_rounds":
            step["selection"]["search_rounds"] = []
        elif how == "partial":
            del cands["35"]
        elif how == "nan":
            cands["3"]["parent_reference_kl"] = float("nan")
        elif how == "another_round":
            cands["3"]["round"] = 1
        elif how == "missing_candidate":
            r0["table"] = [e for e in r0["table"] if e["candidate"] != 35]
        elif how == "duplicate_candidate":
            r0["table"].append(dict(r0["table"][0]))
        elif how == "shifted":
            for k in _r0_candidates(step):
                cands[k]["parent_reference_kl"] += 1e-3
        return step

    @pytest.mark.parametrize("how,match", [
        ("no_diagnostic", "no finite"), ("no_rounds", "no round-0 table"),
        ("partial", "1 of 36"), ("nan", "1 of 36"), ("another_round", "1 of 36"),
        ("missing_candidate", r"\[35\] unlike q2"),
        ("duplicate_candidate", "twice"), ("shifted", "beyond tolerance")])
    def test_each_failure_is_refused(self, control_record, admission, how,
                                     match):
        v = D.parent_reference_admission(self._mutate(control_record, how),
                                         control_record, admission)
        assert v["admissible"] is False
        assert any(re.search(match, f) for f in v["failures"]), v["failures"]

    def test_a_moved_argmin_is_refused_on_its_own(self, control_record,
                                                  admission):
        """With a tolerance wide enough to pass every number, the argmin
        check alone must still refuse a parent that chooses another layer."""
        loose = json.loads(json.dumps(admission))
        loose["tolerance"]["rtol"] = 1.0
        step = self._mutate(control_record, "none")
        _r0_candidates(step)["29"]["parent_reference_kl"] = 0.01
        v = D.parent_reference_admission(step, control_record, loose)
        assert v["failures"] == ["the parent-reference argmin is 29, q2 "
                                 "removed 26 first"]

    def test_records_that_disagree_are_a_harness_error(self, control_record,
                                                       admission):
        bad = json.loads(json.dumps(admission))
        bad["expected"]["first_removed_layer"] = 29
        with pytest.raises(D.D3DesignError, match="different round-0"):
            D.parent_reference_admission(_depth_step(control_record),
                                         control_record, bad)


class _Row:
    def __init__(self, row):
        self._row = row

    def as_dict(self):
        return dict(self._row)


class TestTheDriverStopsAnInadmissibleRun:

    @pytest.fixture
    def run(self, tmp_path, monkeypatch, control_record):
        import aadistill.initialization.planning.fixed_path as FP
        from stages.phase_d1 import autoinit_d1_driver as D1D
        from stages.phase_d1 import d1_session as D1S
        from stages.phase_d1 import replay_specs as R
        from stages.phase_d3 import autoinit_d3_structural_driver as DR

        monkeypatch.setattr(DR, "probe_environment", lambda: {})
        monkeypatch.setattr(D1S, "verify_staged_teacher", lambda path: {})
        monkeypatch.setattr(R, "teacher_config", lambda *a: {})
        monkeypatch.setattr(R, "root_state_from_plan", lambda *a, **k: {
            "chosen_candidate": "x", "config_overrides": {}})
        monkeypatch.setattr(R, "verify_execution_config", lambda *a, **k: {})
        monkeypatch.setattr(D1D, "resolve_root_teacher_path",
                            lambda: {"path": "/nowhere"})
        seen = []

        def go(depth_step):
            def fake(spec, *, on_step, **kw):
                for i, kind in enumerate(("ffn", "depth", "width", "attn")):
                    row = {"index": i, "impl_id": kind, "seconds": 1.0,
                           "artifact_digest": f"{i}" * 64}
                    if i == 1:
                        row.update(depth_step)
                    seen.append(i)
                    on_step(_Row(row))
                raise RuntimeError("the fake stops after the path")

            monkeypatch.setattr(FP, "materialize_fixed_path", fake)
            auth = tmp_path / "auth.json"
            auth.write_text(json.dumps(_issue()))
            out = tmp_path / "out"
            rc = DR.main(["--out", str(out), "--run-id", "d3_structural_test",
                          "--authorization", str(auth),
                          "--status", str(tmp_path / "status")])
            return rc, json.loads((out / DR.EVIDENCE_FILENAME).read_text())
        return go, seen

    def test_an_inadmissible_depth_step_stops_before_width(
            self, run, control_record, tmp_path):
        go, seen = run
        step = TestTheAdmissionGate()._mutate(control_record, "shifted")
        rc, doc = go(step)
        assert rc == 1 and doc["status"] == D.INADMISSIBLE
        assert seen == [0, 1] and len(doc["steps"]) == 2
        assert doc["admission"]["admissible"] is False
        assert "structural" not in doc
        assert "RUN_FAILED" in (tmp_path / "status").read_text()

    def test_an_admitted_step_continues(self, run, control_record):
        go, seen = run
        rc, doc = go(_depth_step(control_record))
        assert doc["admission"]["admissible"] is True
        assert seen == [0, 1, 2, 3] and doc["status"] == "FAILED"


class TestTheCloseoutRederivesTheAdmission:

    def _record(self, control_record, step, *, pod_admissible=None):
        steps = [{"index": i, **ident} for i, ident in enumerate(
            _identities([s["artifact_digest"]
                         for s in control_record["steps"]]))][:2]
        steps[1].update(step)
        verdict = D.parent_reference_admission(
            steps[1], control_record, D.admission_record(REPO))
        if pod_admissible is not None:
            verdict = {**verdict, "admissible": pod_admissible, "failures": []}
        return {"status": D.INADMISSIBLE, "steps": steps, "admission": verdict}

    def test_an_inadmissible_run_is_rederived(self, control_record):
        from stages.phase_d3 import closeout_d3_structural as C

        step = TestTheAdmissionGate()._mutate(control_record, "shifted")
        v = C.rederive_admission(self._record(control_record, step))
        assert v["admissible"] is False

    def test_a_pod_verdict_it_cannot_rederive_is_refused(self, control_record):
        from stages.phase_d3 import closeout_d3_structural as C

        step = TestTheAdmissionGate()._mutate(control_record, "shifted")
        with pytest.raises(C.CloseoutRefused, match="admission"):
            C.rederive_admission(self._record(control_record, step,
                                              pod_admissible=True))

    def test_no_depth_step_means_nothing_to_admit(self, control_record):
        from stages.phase_d3 import closeout_d3_structural as C

        assert C.rederive_admission({"steps": [{"index": 0}]}) is None


def _closeout_repo(tmp_path, **over):
    for rel in (D.REGISTRATION,):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    reg = D.registration(REPO)
    doc = {"structural_outcome": D.STRUCTURAL_CHANGE,
           "registration_hash": reg["registration_hash"],
           "treatment_path_hash": reg["treatment"]["treatment_path_hash"],
           "scientifically_admissible": True,
           "admission": {"admissible": True},
           "provider_teardown_verified": True,
           "product": {"relay_verified": True, "identities": {}, "files": {},
                       "relay": {}}, **over}
    out = (tmp_path / "logs/stages/stage-1/phase_d3/runs/d3_structural_x"
           / "closeout" / "outcome.json")
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(doc))
    return tmp_path


class TestTheProbesNeedAnAdmittedRun:

    def test_an_admitted_structural_change_names_the_product(self, tmp_path):
        from stages.phase_d3 import d3_probes as P

        assert P.structural_product(_closeout_repo(tmp_path))["run_id"] == \
            "d3_structural_x"

    @pytest.mark.parametrize("over", [{"scientifically_admissible": False},
                                      {"admission": None},
                                      {"admission": {"admissible": False}}])
    def test_an_unadmitted_run_names_nothing(self, tmp_path, over):
        from stages.phase_d3 import d3_probes as P

        with pytest.raises(P.D3ProbesError, match="comparability"):
            P.structural_product(_closeout_repo(tmp_path, **over))


# --- 4. the structural driver, launcher and grant ---------------------------------

class TestTheStructuralSession:

    def test_the_driver_check_only_passes_at_zero_cost(self, tmp_path):
        from stages.phase_d3 import autoinit_d3_structural_driver as DR

        rc = DR.main(["--out", str(tmp_path / "out"), "--run-id",
                      "d3_structural_check", "--check-only",
                      "--status", str(tmp_path / "status")])
        doc = json.loads((tmp_path / "out" / DR.EVIDENCE_FILENAME).read_text())
        assert rc == 0 and doc["status"] == "CHECK_ONLY_OK"
        assert doc["treatment_path_hash"] == \
            D.registration(REPO)["treatment"]["treatment_path_hash"]
        assert "CHECK_ONLY_OK" in (tmp_path / "status").read_text()

    def _launcher(self):
        from stages.phase_d3 import autoinit_d3_structural_launch as L
        from support.session_specs import session_args

        return L, session_args(L)

    def test_the_spec_builds_and_the_driver_command_parses(self):
        from stages.phase_d3 import autoinit_d3_structural_driver as DR

        L, args = self._launcher()
        spec = L.spec(args)
        assert spec.setup.env["SESSION_KIND"] == L.SESSION_KIND
        assert spec.authorization_loader.__self__.__name__ == \
            "D3StructuralAuthorization"
        assert spec.plan_hash == D.registration_hash(REPO)
        cmd = spec.driver_command(
            SimpleNamespace(args=SimpleNamespace(run_id="d3_structural_x"),
                            elapsed_minutes=10.0),
            SimpleNamespace(soft_stop_minutes=100.0))
        argv = shlex.split(cmd)
        assert argv[1].endswith("autoinit_d3_structural_driver.py")
        assert "--deadline-s" in argv and "--authorization" in argv
        assert argv[argv.index("--deadline-s") + 1] == str(90 * 60)
        assert "VLLM_READY" not in spec.setup.setup_markers

    def test_host_admission_is_the_protocols_image_identity(self):
        from stages.d_series import shared_evaluation as SE

        L, _ = self._launcher()
        want = SE._read(SE.RECORD, REPO)["recovery"]["training_image_digest"]
        assert L.host_admission(want)[0] is True
        assert L.host_admission(want.replace("595.91.07", "580.126.09"))[0] \
            is False

    def test_the_budget_decomposes_the_priced_bound(self):
        from stages.phase_d3.d3_authorization import session_ceiling

        L, _ = self._launcher()
        priced = session_ceiling(REPO, "structural")
        assert L.budget_spec(REPO) is not None
        assert priced["hard_ceiling_minutes"] > priced["expected_minutes"]

    @pytest.mark.parametrize("outcome,fetched,ok", [
        (D.NO_STRUCTURAL_CHANGE, [], True),
        (D.STRUCTURAL_CHANGE, [], False),
        (D.UNEXPLAINED_BYTE_DIFFERENCE, [], False),
        (D.STRUCTURAL_CHANGE, [SimpleNamespace(ok=True)], True),
        (D.STRUCTURAL_CHANGE, [SimpleNamespace(ok=False)], False),
    ])
    def test_the_product_gate_follows_the_outcome(self, tmp_path, outcome,
                                                   fetched, ok):
        L, _ = self._launcher()
        store = tmp_path / "store"
        store.mkdir()
        (store / "d3_structural.json").write_text(json.dumps({
            "structural": {"outcome": outcome},
            "stages": [{"stage": "E"}]}))
        (store / "product.json").write_text(json.dumps({"identities": {}}))
        ctx = SimpleNamespace(args=SimpleNamespace(scr=str(tmp_path)))
        assert L.product_secured(ctx, fetched)[0] is ok

    def test_unreadable_evidence_never_reads_as_nothing_to_secure(self,
                                                                   tmp_path):
        L, _ = self._launcher()
        ctx = SimpleNamespace(args=SimpleNamespace(scr=str(tmp_path)))
        ok, why = L.product_secured(ctx, [])
        assert ok is False and "GATE FAILURE" in why


def _money(funded: bool = True):
    from stages.phase_d1.d1_authorization import live_money

    money = dict(live_money(REPO))
    others = [x for x in money["funds_formal_sessions_of"]
              if x != D.EXPERIMENT_ID]
    money["funds_formal_sessions_of"] = (
        [*others, D.EXPERIMENT_ID] if funded else others)
    return money


def _grant(**over):
    return {"rung": "structural",
            "approves_registration_hash": D.registration_hash(REPO),
            "cumulative_envelope_usd": 7.0, "granted_by": "test",
            "covers": "test", **over}


def _issue(grant=None, **kw):
    from stages.phase_d3 import d3_authorization as A

    args = dict(grant=grant or _grant(), session_commit="d" * 40,
                granted_utc="2026-10-16T00:00:00+00:00",
                run_id="d3_structural_test", rung="structural",
                repo_root=REPO, live_rate=1.09, consumed_usd=0.0,
                money=_money())
    args.update(kw)
    return A.build_payload(**args)


class TestTheStructuralGrant:

    def test_the_issued_artifact_round_trips_through_its_type(self, tmp_path):
        from stages.phase_d3.d3_authorization import D3StructuralAuthorization

        payload = _issue()
        path = tmp_path / "a.json"
        path.write_text(json.dumps(payload))
        auth = D3StructuralAuthorization.load(path)
        assert auth.contract_hash == \
            D.registration(REPO)["treatment"]["treatment_path_hash"]
        assert auth.authorizes_d3_structural and \
            auth.allows_fixed_path_materialization
        for denied in ("authorizes_recovery_training", "authorizes_d3_probes",
                       "authorizes_beam_search", "authorizes_reopen_d1_d2",
                       "authorizes_cancelled_d3", "authorizes_promotion",
                       "authorizes_automatic_followon_start"):
            assert getattr(auth, denied) is False, denied
        auth.require_harness(REPO)

    @pytest.mark.parametrize("change,match", [
        ({"hard_cap_usd": 1.0}, "DERIVES"),
        ({"rung": "probes"}, "funds rung"),
        ({"approves_registration_hash": "0" * 64}, "different registration"),
        ({"cumulative_envelope_usd": 3.0}, "envelope"),
    ])
    def test_the_grant_cannot_state_or_underfund(self, change, match):
        from stages.phase_d3.d3_authorization import D3AuthorizationRefused

        with pytest.raises(D3AuthorizationRefused, match=match):
            _issue(_grant(**change))

    def test_earlier_subruns_net_the_envelope(self):
        from stages.phase_d3.d3_authorization import D3AuthorizationRefused

        with pytest.raises(D3AuthorizationRefused, match="envelope"):
            _issue(consumed_usd=4.0)

    def test_a_run_id_outside_the_rung_prefix_is_refused(self):
        from stages.phase_d3.d3_authorization import D3AuthorizationRefused

        with pytest.raises(D3AuthorizationRefused, match="prefix|start with"):
            _issue(run_id="d3_probes_x")

    def test_an_unfunded_phase_is_refused(self):
        from stages.phase_d3.d3_authorization import D3AuthorizationRefused

        with pytest.raises(D3AuthorizationRefused, match="funds_formal"):
            _issue(money=_money(funded=False))

    def test_the_amended_scope_reaches_the_deriver_and_the_issuer(self):
        """The 2026-10-17 amendment: phase_d3 is on the formal book, read by
        both consumers from the one config, and nothing else moved."""
        sys.path.insert(0, str(REPO / "scripts/maintenance/consolidation"))
        import derive_budget as B
        from stages.phase_d1.d1_authorization import live_money

        cfg = json.loads((REPO / "configs/stages/stage-1/phase_c1/"
                          "authorization.json").read_text())
        assert D.EXPERIMENT_ID in B.funded_experiment_ids(
            cfg["execution_package"])
        assert D.EXPERIMENT_ID in live_money(REPO)["funds_formal_sessions_of"]
        amendment = cfg["_amendment_2026_10_17_d3_structural"]
        assert amendment["added_usd"] == 0.0 and amendment["raised"] == []
        kept = amendment["deliberately_unchanged"]
        assert kept["cumulative_cap_usd"] == \
            cfg["accepted_pricing"]["cumulative_cap_usd"]
        for key in ("formal_allowance_usd", "gpu_engineering_allowance_usd",
                    "package_total_usd", "per_attempt_hard_ceiling_usd"):
            assert kept[key] == cfg["execution_package"][key], key
        assert amendment["before"]["funds_formal_sessions_of"] + \
            [D.EXPERIMENT_ID] == \
            cfg["execution_package"]["funds_formal_sessions_of"][
                "experiment_ids"]

    def test_an_edited_artifact_is_refused(self, tmp_path):
        from stages.phase_d3.d3_authorization import (
            D3AuthorizationRefused, D3StructuralAuthorization,
        )

        payload = _issue()
        payload["hard_cap_usd"] = 30.0
        path = tmp_path / "a.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(D3AuthorizationRefused, match="own authorization"):
            D3StructuralAuthorization.load(path)


# --- 5. the conditional probes rung -------------------------------------------------

@pytest.fixture
def product(monkeypatch):
    """A STRUCTURAL_CHANGE product, as the probes contract would read it."""
    from stages.phase_d3 import d3_probes as P

    fake = {"run_id": "d3_structural_fixture",
            "identities": {"artifact_digest": "e" * 64,
                           "weights_digest": "f" * 64,
                           "single_shard_sha256": "a" * 64,
                           "arch_signature": "b" * 64,
                           "num_parameters": 596049920},
            "files": {n: {"sha256": "c" * 64, "bytes": 1}
                      for n in ("model.safetensors", "config.json",
                                "generation_config.json")},
            "relay": {"repo": "relay/repo", "prefix": "p",
                      "files": {n: {"path": f"p/{n}", "sha256": "c" * 64}
                                for n in ("model.safetensors", "config.json",
                                          "generation_config.json")}},
            "closeout_rel": "x"}
    monkeypatch.setattr(P, "structural_product", lambda repo_root=REPO: fake)
    return fake


class TestTheProbesRung:

    def test_without_a_structural_change_there_is_no_arm(self):
        from stages.phase_d3 import d3_probes as P

        with pytest.raises(P.D3ProbesError, match="STRUCTURAL_CHANGE"):
            P.arms(REPO)

    def test_three_probes_from_the_d3_initial_checkpoint(self, product):
        from stages.phase_d3 import d3_probes as P

        probes = P.probes(REPO, arm_root="/pod/arms")
        assert [p.seed for p in probes] == [396880939, 766527341, 1004399678]
        assert {p.checkpoint_dir for p in probes} == {f"/pod/arms/{'e' * 64}"}
        arm = P.arms(REPO)[0]
        assert arm.state_id == arm.artifact_digest == "e" * 64

    def test_the_contract_binds_the_registration_and_the_comparators(
            self, product):
        from stages.phase_d3 import d3_probes as P

        contract = P.session_contract(REPO, verify_bytes=False)
        assert contract["preregistration_hash"] == D.registration_hash(REPO)
        assert sorted(contract["comparator"]) == ["B", "q2"]
        assert contract["initialization"]["identities"]["artifact_digest"] \
            == "e" * 64
        moved = copy.deepcopy(product)
        moved["identities"]["artifact_digest"] = "9" * 64
        assert P.contract_hash(contract) != P.contract_hash(
            {**contract, "initialization": {**contract["initialization"],
                                            "identities": moved["identities"]}})

    def test_the_shared_driver_runs_the_rung_under_its_own_names(self):
        from stages.phase_d2 import autoinit_d2_screening_driver as DR

        layout = DR.rung_layout("d3_probes")
        assert layout["battery_rung"] == "confirmation"
        assert layout["scorer"].name == "score_d1_confirmation.py"
        assert DR.authorization_type(layout).__name__ == \
            "D3ProbesAuthorization"
        assert DR.rung_layout("confirmation")["experiment"] == "D2"
        assert DR.authorization_type(DR.rung_layout("confirmation")).__name__ \
            == "D2ConfirmationAuthorization"

    def test_the_launcher_builds_a_d3_session_on_the_shared_driver(
            self, product):
        from stages.phase_d2 import autoinit_d2_screening_driver as DR
        from stages.phase_d3 import autoinit_d3_probes_launch as L
        from support.session_specs import session_args

        spec = L.spec(session_args(L))
        assert spec.setup.env["SESSION_KIND"] == "d3_probes"
        cmd = spec.driver_command(
            SimpleNamespace(args=SimpleNamespace(run_id="d3_probes_x"),
                            image_digest="img@595.91.07"), None)
        parsed = DR.build_parser().parse_args(shlex.split(cmd)[2:])
        assert parsed.rung == "d3_probes"
        assert parsed.out.endswith("autoinit_d3_probes")
        assert "phase_d3" in parsed.authorization
        paths = {r.path for r in spec.setup.relay_inputs}
        assert {f"p/{n}" for n in ("model.safetensors", "config.json",
                                   "generation_config.json")} <= paths

    def test_the_dispatch_branch_loads_the_probes_type(self):
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        branch = src.split('"$SESSION_KIND" = "d3_probes"', 1)[1]
        branch = branch.split("\nelif", 1)[0]
        assert "D3ProbesAuthorization.load(" in branch
        assert "require_plan(os.environ['SESSION_PLAN_HASH'])" in branch
        assert 'mark "AUTHORIZATION_MISMATCH"; exit 98' in branch


# --- 6. the exploratory comparison -------------------------------------------------------

class TestTheExploratoryComparison:

    @pytest.mark.parametrize("interval,seeds,want", [
        ((0.001, 0.02), [0.01, 0.02, -0.01], "BEHAVIOURAL_IMPROVEMENT"),
        ((0.001, 0.02), [0.01, -0.02, -0.01], "INCONCLUSIVE"),
        ((-0.02, -0.001), [-0.01, -0.02, 0.0], "BEHAVIOURAL_REGRESSION"),
        ((-0.02, 0.0), [-0.01, -0.02, -0.03], "INCONCLUSIVE"),
        ((-0.01, 0.01), [0.01, 0.02, 0.03], "INCONCLUSIVE"),
    ])
    def test_the_registered_label_table(self, interval, seeds, want):
        from stages.phase_d3.analyze_d3_probes import label

        assert label(interval, seeds) == want

    def test_it_reproduces_d2s_recorded_comparisons_on_d2s_rows(self):
        """EVIDENCE-REUSE VALIDATION: fed D2's c4 rows as the candidate, the
        D3 analysis reproduces the D2 confirmation's recorded point deltas
        against the imported q2 and B units (hash-checked by the shared
        protocol)."""
        from stages.phase_d3 import analyze_d3_probes as A

        store = Path("/home/ecs-user/aad-artifacts/phase_d2/"
                     "d2_confirmation_20261016_105802/store/probes")
        if not store.is_dir():
            pytest.skip("the D2 confirmation's durable store is not on this host")
        seeds = [396880939, 766527341, 1004399678]
        cand = {s: [json.loads(x) for x in (
            store / f"d2_confirmation_c4_s{s}" / "per_sample.jsonl"
        ).read_text().splitlines() if x.strip()] for s in seeds}
        verdict = json.loads((REPO / "logs/stages/stage-1/phase_d2/runs/"
                              "d2_confirmation_20261016_105802/evidence/"
                              "d2_confirmation_verdict.json").read_text())
        reg = copy.deepcopy(D.registration(REPO))
        reg["behavioural_rung"]["exploratory_classification"][
            "bootstrap_iterations"] = 200
        out = A.analyze(cand, registration=reg)
        q2 = out["comparisons"]["q2"]["correct_overall"]
        b = out["comparisons"]["B"]["correct_overall"]
        recorded_q2 = verdict["verdict"]["secondary_vs_q2_descriptive"][
            "decision"]
        assert q2["delta"] == pytest.approx(recorded_q2["delta"], abs=1e-12)
        assert q2["per_seed_delta"] == pytest.approx(
            recorded_q2["per_seed_delta"], abs=1e-12)
        assert b["delta"] == pytest.approx(
            verdict["verdict"]["primary_vs_B"]["decision"]["delta"], abs=1e-12)
        assert out["comparisons"]["q2"]["usable_rollout_descriptive"][
            "delta"] == pytest.approx(recorded_q2["usable"]["pooled_delta"],
                                      abs=1e-12)

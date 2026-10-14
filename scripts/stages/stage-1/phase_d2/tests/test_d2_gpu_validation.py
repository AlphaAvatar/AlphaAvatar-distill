"""The D2 GPU validation (stage V) at $0: protocol, verdicts, records, a toy run.

    pytest scripts/stages/stage-1/phase_d2/tests/test_d2_gpu_validation.py

The stage runs END TO END here on a CPU toy through the same function the pod
calls, with the models, mixtures and suite injected: every step, every check
and every verdict path executes. What only the GPU can answer -- the real
teacher's entropy, BF16 drift, timings -- is what the session is for.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]

from support.toy import (  # noqa: E402
    TARGET_GEOMETRY, TEACHER_GEOMETRY, build_tiny_model, make_items,
    make_profile,
)
from stages.phase_d2 import d2_gpu_validation as V  # noqa: E402
from stages.phase_d2 import d2_gpu_validation_plan as P  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    register_builtin_adapters()
    register_builtin_operators()
    activation_importance.register(replace=True)
    yield


# --- class 1: arithmetic ----------------------------------------------------------

class TestArithmetic:

    def test_float32_chunked_entropy_holds_against_float64(self):
        logits = torch.randn(40, 3000, generator=torch.Generator().manual_seed(1)) * 4
        out = V.entropy_arithmetic(logits)
        assert out["within_tolerance"], out
        assert out["positions"] == 40 and out["vocab"] == 3000

    def test_a_half_precision_path_would_fail_the_gate(self):
        """The tolerance is tight enough to catch a precision regression."""
        from aadistill.initialization.scoring.objective import (
            full_vocabulary_entropy,
        )
        logits = torch.randn(40, 3000, generator=torch.Generator().manual_seed(2)) * 4
        half = full_vocabulary_entropy(logits.half().float()).double()
        ref = V.float64_entropy(logits)
        assert float((half - ref).abs().max()) > \
            V.ARITHMETIC_TOLERANCES["entropy_abs_nats_vs_float64"]

    def test_the_reduction_matches_float64_on_identical_inputs(self):
        from stages.d_series import objectives as O
        g = torch.Generator().manual_seed(3)
        per_item = [{"kl": torch.rand(30, generator=g) * 5,
                     "confidence": torch.rand(30, generator=g),
                     "scored": torch.arange(30) >= 7} for _ in range(5)]
        out = V.reduction_arithmetic(O.D2_OBJECTIVE, per_item)
        assert out["items"] == 5 and out["within_tolerance"], out

    def test_the_raw_kl_control_is_relative_and_strict(self):
        kl = {"state.teacher_kl.equal_domain_mean": 8.0,
              "state.teacher_kl.worst_domain": 9.0,
              "state.critical_token_kl": 7.0,
              "state.critical_token_kl.eos": 3.0}
        same = {"state.objective.raw_kl.equal_domain_mean": 8.0,
                "state.objective.raw_kl.worst_domain": 9.0,
                "state.objective.raw_kl.critical_token": 7.0,
                "state.objective.raw_kl.critical_token.eos": 3.0}
        out = V.raw_kl_control_relative(kl, same)
        assert out["metrics_compared"] == 4 and out["within_tolerance"]
        off = dict(same, **{"state.objective.raw_kl.worst_domain": 9.0 * (1 + 1e-5)})
        assert not V.raw_kl_control_relative(kl, off)["within_tolerance"]


# --- the diagnostics ----------------------------------------------------------------

class TestDiagnostics:

    def test_jaccard_per_layer(self):
        out = V.jaccard_per_layer([[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 6]])
        assert out["identical_layers"] == 1
        assert out["min"] == pytest.approx(1 / 3)

    def test_projection_overlap_recovers_one_subspace_and_separates_two(self):
        g = torch.Generator().manual_seed(4)
        et = torch.randn(500, 24, generator=g, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(24, 8, generator=g,
                                           dtype=torch.float64))
        r, _ = torch.linalg.qr(torch.randn(24, 8, generator=g,
                                           dtype=torch.float64))
        same = V.projection_overlap(et, et @ q, et @ (q @ torch.diag(
            torch.tensor([1., -1, 1, 1, -1, 1, 1, 1], dtype=torch.float64))),
            rows=97)
        assert same["subspace_overlap"] == pytest.approx(1.0, abs=1e-9)
        other = V.projection_overlap(et, et @ q, et @ r)
        assert other["subspace_overlap"] < 0.9

    def test_three_way_rankings_report_agreement_and_choice(self):
        cands = {str(k): {"parent_kl": float(k), "raw_kl": float(k),
                          "objective": float(-k)} for k in range(6)}
        out = V.three_way_rankings(cands, [str(k) for k in range(6)])
        p = out["pairs"]
        assert p["parent_kl__vs__root_kl_unweighted"]["identical_order"]
        assert p["parent_kl__vs__root_kl_teacher_weighted"]["spearman"] == \
            pytest.approx(-1.0)
        assert out["choice"]["root_kl_teacher_weighted"] == "5"
        assert len(out["table"]) == 6

    def test_the_depth_table_must_be_the_objective(self):
        art = {"search_rounds": [{"table": [{"candidate": 1, "score": 2.0}]}],
               "objective_candidates": {"1": {"objective": 2.0,
                                              "parent_kl": 1.0}}}
        assert V.depth_table_consistency(art)["consistent"]
        art["objective_candidates"]["1"]["objective"] = 2.5
        assert not V.depth_table_consistency(art)["consistent"]
        del art["objective_candidates"]["1"]["parent_kl"]
        assert V.depth_table_consistency(art)["missing_records"] == ["1"]


# --- the verdicts: the predicate, table-tested -----------------------------------

def _record(*, entropy=True, reduction=True, raw=True, b=True, stores=True,
            ref="root_reference", consistent=True, se=True, dp=True,
            v5=True):
    rec = {"premises": {"teacher_revision": "r"},
           "V1_root_stores_and_validity": {
               "profiles": {"p": {"arithmetic": {"within_tolerance": entropy}}},
               "stores_match_declaration": stores},
           "V2_b_rebuild_and_kl_relation": {
               "b_identity_verified": b,
               "profiles": {"p": {"reduction_arithmetic":
                                  {"within_tolerance": reduction}}}},
           "V6_state_evaluation": {"raw_kl_control": {"within_tolerance": raw},
                                   "scored_rows": {"within_tolerance": se}},
           "V4_depth_root_vs_parent": {
               "reference": ref,
               "table_consistency": {"consistent": consistent}}}
    if v5:
        rec["V5_depth_scored_rows"] = {"agreement": {"within_tolerance": dp}}
    return rec


class TestVerdicts:

    def test_all_holding_is_pass(self):
        v = V.verdicts(_record())
        assert (v["arithmetic"]["verdict"], v["integrity"]["verdict"],
                v["scored_rows_equivalence"]["verdict"], v["validation"]) == \
            ("PASS", "PASS", "PASS", "PASS")

    @pytest.mark.parametrize("breach", ["entropy", "reduction", "raw"])
    def test_any_arithmetic_breach_fails_the_validation(self, breach):
        v = V.verdicts(_record(**{breach: False}))
        assert v["arithmetic"]["verdict"] == "FAIL"
        assert v["validation"] == "FAIL"

    @pytest.mark.parametrize("breach,value", [("b", False), ("stores", False),
                                              ("ref", "parent_state"),
                                              ("consistent", False)])
    def test_any_integrity_breach_fails_the_validation(self, breach, value):
        v = V.verdicts(_record(**{breach: value}))
        assert v["integrity"]["verdict"] == "FAIL" and v["validation"] == "FAIL"

    def test_an_unmeasured_check_is_incomplete_not_pass(self):
        rec = _record()
        del rec["V2_b_rebuild_and_kl_relation"]
        assert V.verdicts(rec)["arithmetic"]["verdict"] == "INCOMPLETE"

    def test_scored_rows_is_its_own_verdict(self):
        v = V.verdicts(_record(dp=False))
        assert v["scored_rows_equivalence"]["verdict"] == "FAIL"
        assert v["validation"] == "PASS"
        assert V.verdicts(_record(v5=False))[
            "scored_rows_equivalence"]["verdict"] == "INCOMPLETE"

    def test_drift_and_diagnostics_never_gate(self):
        v = V.verdicts(_record())
        assert "bsz1_vs_bsz3_root_store" in v["not_gated"]
        assert "root_vs_parent_vs_weighted_rankings" in v["not_gated"]


# --- the stage, end to end on a CPU toy -----------------------------------------------

def _calib(profile_id: str, seed: int):
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    raw = make_items(n_per_subtype=2, seq_len=20, seed=seed)
    for k, item in enumerate(raw):
        n = int(item["input_ids"].shape[1]) - 1 - k
        item["input_ids"] = item["input_ids"][:, :n + 1]
        t = torch.arange(n)
        item["tags"] = {"assistant": t >= 4, "reasoning": (t >= 4) & (t < 11),
                        "final_answer": t >= 11, "eos": t == n - 1}
    return prepare_calibration_items(raw, profile_id=profile_id)


def _suite():
    from aadistill.initialization.specs.metrics import StateEvalSuite, SuiteItem
    items = []
    for k, raw in enumerate(make_items(seed=202)):
        keep = 10 + 4 * k
        n = keep - 1
        tags = {name: mask[:n] for name, mask in raw["tags"].items()}
        tags["eos_like"] = tags["eos_like"] | (torch.arange(n) == n - 1)
        tags["assistant"] = torch.arange(n) >= 4
        items.append(SuiteItem(item_id=raw["item_id"],
                               input_ids=raw["input_ids"][:, :keep],
                               domain=raw["domain"], subtype=raw["subtype"],
                               tags=tags))
    suite = StateEvalSuite(
        suite_id="test.v", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    return suite, items, "toy"


def toy_environment() -> V.Environment:
    from aadistill.initialization.scoring.positions import SUPERVISED_TARGET_V1
    from aadistill.initialization.scoring.support import reference_topk_tail
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.materialization import (
        NumericalEnvironment,
    )

    ident = {"artifact_digest": "a" * 64, "weights_digest": "w" * 64,
             "single_shard_sha256": "s" * 64, "arch_signature": "g" * 64}

    def teacher():
        m = build_tiny_model(TEACHER_GEOMETRY)
        m.config.use_cache = False
        return m

    def b(_workdir):
        m = build_tiny_model(TARGET_GEOMETRY, seed=5)
        m.config.use_cache = False
        return m, {"identity": dict(ident), "expected": dict(ident),
                   "seconds": 0.0}

    profiles = [(make_profile("balanced"), _calib("test.balanced@v1", 101)),
                (make_profile("reasoning", seed=2),
                 _calib("test.reasoning@v1", 303))]
    return V.Environment(
        repo=REPO, device="cpu", adapter=get_adapter("qwen3"),
        support=reference_topk_tail(8), policy=SUPERVISED_TARGET_V1,
        numerics=NumericalEnvironment(device_type="cpu",
                                      compute_dtype="float32"),
        target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY), seed=0,
        micro_batch_size=3, packing="length_sorted_v1",
        root_identity={"teacher_id": "toy", "teacher_sha256": "ab" * 32},
        teacher_revision="toyrev", load_teacher=teacher, materialize_b=b,
        profiles=lambda: profiles, load_suite=_suite,
        decode=lambda t: str(t))


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    from stages.phase_d1.topk_adoption_driver import Journal

    out = tmp_path_factory.mktemp("v")
    flushed = []
    record = V.stage_V(env=toy_environment(), journal=Journal(out / "j.jsonl"),
                       out=out, flush=lambda r: flushed.append(dict(r)))
    return record, flushed


class TestTheStageEndToEnd:

    def test_every_step_runs_and_none_fails(self, toy_run):
        record, flushed = toy_run
        for name in V.STEP_ORDER:
            assert name in record, name
            assert "failed" not in record[name], record[name].get("traceback")
            assert "skipped" not in record[name]
        assert len(flushed) == len(V.STEP_ORDER)

    def test_the_verdicts_pass_on_the_toy(self, toy_run):
        v = toy_run[0]["verdicts"]
        assert v["arithmetic"]["verdict"] == "PASS", v["arithmetic"]
        assert v["integrity"]["verdict"] == "PASS", v["integrity"]
        assert v["scored_rows_equivalence"]["verdict"] == "PASS"
        assert v["validation"] == "PASS"

    def test_v1_records_stores_drift_and_validity_per_profile(self, toy_run):
        v1 = toy_run[0]["V1_root_stores_and_validity"]
        assert set(v1["profiles"]) == {"test.balanced@v1", "test.reasoning@v1"}
        for prof in v1["profiles"].values():
            assert prof["identity_matches"]
            assert set(prof["builds"]) == {"bsz3_formal", "bsz1"}
            assert prof["bsz1_vs_bsz3"]["_class"].startswith("EXECUTION DRIFT")
            assert "iqr" in prof["confidence"]
            assert any(k.startswith("token:") for k in prof["validity"]["groups"])

    def test_v2_relates_confidence_to_bs_root_kl(self, toy_run):
        v2 = toy_run[0]["V2_b_rebuild_and_kl_relation"]
        assert v2["b_identity_verified"]
        for prof in v2["profiles"].values():
            assert "relation_to_kl" in prof["validity_with_b_kl"]

    def test_v6_measures_four_arms(self, toy_run):
        v6 = toy_run[0]["V6_state_evaluation"]
        assert set(v6["runs"]) == {"kl_only", "d2", "d2_scored_v1", "d2_bsz1"}
        assert v6["bsz1_vs_bsz3"]["_class"].startswith("EXECUTION DRIFT")

    def test_v3_compares_full_and_narrow_scope_per_operator(self, toy_run):
        ops = toy_run[0]["V3_weighted_statistics"]["operators"]
        assert set(ops) == {"ffn.activation_importance_v0",
                            "attention.activation_importance_v1",
                            "width.global_pca_v0"}
        for rec in ops.values():
            assert rec["arms"]["full_scope"]["weighted"] is True
            assert rec["arms"]["narrow_scope"]["weighted"] is False
            assert rec["arms"]["full_scope"]["config_declares_objective"]
            assert not rec["arms"]["narrow_scope"]["config_declares_objective"]
        assert "subspace_overlap" in ops["width.global_pca_v0"][
            "selection_overlap"]

    def test_v4_reports_three_metrics_on_a_non_root_parent(self, toy_run):
        v4 = toy_run[0]["V4_depth_root_vs_parent"]
        assert v4["reference"] == "root_reference"
        r = v4["rankings"]
        assert r["candidates"] == TEACHER_GEOMETRY["num_hidden_layers"]
        assert set(r["choice"]) == {"parent_kl", "root_kl_unweighted",
                                    "root_kl_teacher_weighted"}
        assert "effective_sample_size" in v4["teacher_confidence"]
        assert "parent_kl_seconds" in v4["timing"]
        #: Below the root the two references are different models.
        rows = r["table"]
        assert any(abs(x["parent_kl"] - x["root_kl_unweighted"]) > 1e-6
                   for x in rows)

    def test_a_missing_b_fails_its_step_and_the_rest_still_runs(self, tmp_path):
        from stages.phase_d1.topk_adoption_driver import Journal
        env = toy_environment()

        def broken(_w):
            raise RuntimeError("B could not be built")
        env.materialize_b = broken
        rec = V.stage_V(env=env, journal=Journal(tmp_path / "j.jsonl"),
                        out=tmp_path)
        assert "failed" in rec["V2_b_rebuild_and_kl_relation"]
        assert "failed" in rec["V6_state_evaluation"]
        assert "failed" not in rec["V4_depth_root_vs_parent"]
        assert rec["verdicts"]["validation"] != "PASS"


# --- the records: plan, PROPOSED grant, campaign ------------------------------------

class TestTheRecords:

    def test_they_are_what_the_builder_derives(self):
        assert P.main([]) == 0, "run d2_gpu_validation_plan.py --write"

    def test_the_plan_registers_the_codes_tolerances(self):
        plan = json.loads((REPO / V.PLAN).read_text())
        acc = plan["acceptance"]
        assert acc["arithmetic"]["tolerances"] == V.ARITHMETIC_TOLERANCES
        assert acc["row_equivalence"]["tolerances"] == \
            V.ROW_EQUIVALENCE_TOLERANCES
        assert acc["execution_drift"]["gate"] is False
        assert acc["diagnostics"]["gate"] is False
        assert set(plan["step_bounds_s"]) == set(V.STEP_ORDER)
        assert plan["formal_topology"]["micro_batch_size"] == 3

    def test_the_proposal_is_not_a_grant(self):
        assert not (REPO / V.AUTHORIZATION).exists()
        auth = json.loads((REPO / V.AUTHORIZATION_PROPOSED).read_text())
        assert auth["status"].startswith("PROPOSED")
        assert auth["plan_sha256"] == hashlib.sha256(
            (REPO / V.PLAN).read_bytes()).hexdigest()
        plan = json.loads((REPO / V.PLAN).read_text())
        t = plan["totals"]
        assert auth["session_cap_usd"] == t["proposed_session_cap_usd"]
        assert auth["all_in_ceiling_usd"] == pytest.approx(
            t["proposed_session_cap_usd"] + t["teardown_reserve_usd"])
        assert "ADOPT_STAGES=V" in auth["command"]
        assert f"SESSION_CAP={t['proposed_session_cap_usd']:.2f}" in \
            auth["command"]

    def test_the_launcher_cannot_derive_a_budget_without_the_grant(self):
        """THE PAID BOUNDARY: no authorization.json, no ceiling, no pod."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ecb", REPO / "scripts/shared/pod/engineering_campaign_budget.py")
        ecb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ecb)
        with pytest.raises(ecb.BudgetError, match="authorization.json"):
            ecb.derive(REPO / V.CAMPAIGN, session_cap_usd=1.5,
                       rate_usd_per_hour=1.09)

    def test_the_cap_covers_the_core_and_v5_is_extended(self):
        plan = json.loads((REPO / V.PLAN).read_text())
        classes = {s["id"]: s["class"] for s in plan["steps"]}
        assert classes["V5_depth_scored_rows"] == "extended"
        assert V.CORE_STEPS == tuple(n for n in V.STEP_ORDER
                                     if classes[n] == "core")
        core = sum(s["bound_minutes"] for s in plan["steps"]
                   if s["class"] == "core")
        assert plan["totals"]["core_bound"]["minutes"] == pytest.approx(
            round(core, 1))

    def test_the_premises_resolve_on_this_host(self):
        out = V.premises(REPO)
        from stages.d_series import objectives as O
        assert out["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert out["contract_revision"] == 7
        assert out["plan_sha256"] == hashlib.sha256(
            (REPO / V.PLAN).read_bytes()).hexdigest()


# --- booking and the budget's reader ---------------------------------------------------

def _booked_tree(tmp_path, *, objective_hash=None, status="COMPLETE"):
    import importlib.util
    from stages.d_series import objectives as O
    from stages.d_series import d2_budget as BUD

    run = tmp_path / "artifacts" / "d2val_x"
    (run / "out" / "adoption_all_v1" / "work").mkdir(parents=True)
    (run / "cost.json").write_text(json.dumps(
        {"all_in_usd": 0.91, "pod_id": "pod1", "elapsed_minutes": 50.0}))
    (run / "teardown_confirmed").write_text("confirmed")
    (run / "out" / "adoption_all_v1" / "work" / "big.json").write_text("{}")
    (run / "out" / "adoption_all_v1" / "adoption.json").write_text(json.dumps(
        {"status": status, "V_d2_validation": {
            "premises": {"objective_hash":
                         objective_hash or O.D2_OBJECTIVE.objective_hash},
            "verdicts": {"validation": "PASS",
                         "scored_rows_equivalence": {"verdict": "PASS"}}}}))
    vdir = tmp_path / Path(BUD.D2_VALIDATION_CAMPAIGN).parent
    vdir.mkdir(parents=True)
    (vdir / "campaign.json").write_text(json.dumps(P.campaign()))
    spec = importlib.util.spec_from_file_location(
        "book", REPO / "scripts/shared/pod/book_engineering_subrun.py")
    book = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(book)
    return run, vdir, book


class TestBooking:

    def test_a_session_is_booked_with_cost_and_evidence_hash(self, tmp_path):
        from stages.d_series import d2_budget as BUD
        run, vdir, book = _booked_tree(tmp_path)
        row = book.book(run_dir=run, validation_dir=vdir, subrun_id="d2val_s1",
                        outcome="ok",
                        evidence_session="out/adoption_all_v1/adoption.json",
                        repo=tmp_path)
        assert row["cost_usd"] == 0.91 and row["teardown_confirmed"]
        assert not any("work" in k for k in row["evidence_sha256"])
        camp = json.loads((vdir / "campaign.json").read_text())
        assert camp["booked_usd"] == 0.91
        ev = BUD.d2_validation_evidence(tmp_path)
        assert ev["usable"] and ev["scored_rows_equivalence"] == "PASS"
        assert ev["validation"] == "PASS"
        book.book(run_dir=run, validation_dir=vdir, subrun_id="d2val_s1",
                  outcome="rebooked", repo=tmp_path)
        assert len(json.loads((vdir / "campaign.json").read_text())[
            "subruns"]) == 1

    def test_an_unpriced_run_is_refused(self, tmp_path):
        run, vdir, book = _booked_tree(tmp_path)
        (run / "cost.json").unlink()
        with pytest.raises(book.BookingError, match="unpriced"):
            book.book(run_dir=run, validation_dir=vdir, subrun_id="s",
                      outcome="x", repo=tmp_path)

    @pytest.mark.parametrize("bad", ["objective", "incomplete"])
    def test_the_budget_reads_only_a_complete_run_of_this_objective(
            self, tmp_path, bad):
        from stages.d_series import d2_budget as BUD
        run, vdir, book = _booked_tree(
            tmp_path, objective_hash="0" * 64 if bad == "objective" else None,
            status="FAILED" if bad == "incomplete" else "COMPLETE")
        book.book(run_dir=run, validation_dir=vdir, subrun_id="s",
                  outcome="x",
                  evidence_session="out/adoption_all_v1/adoption.json",
                  repo=tmp_path)
        ev = BUD.d2_validation_evidence(tmp_path)
        assert not ev["usable"] and ev["scored_rows_equivalence"] is None


def test_the_driver_knows_stage_v_and_runs_it_under_all_v1_only():
    """Read from the driver's source: stage V is a known letter and refuses a
    scored_v1 base mode before anything resident."""
    src = (REPO / "scripts/stages/stage-1/phase_d1/topk_adoption_driver.py"
           ).read_text()
    assert '{"C", "D", "P", "L", "V"}' in src
    assert "stage V compares all_v1 against scored_v1 internally" in src
    assert "D2V.premises(repo)" in src and "D2V.stage_V(" in src

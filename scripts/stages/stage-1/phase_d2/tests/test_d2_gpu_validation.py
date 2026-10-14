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

def _v3(ok=True, failed_ops=(), missing_ops=(), arm_fails=()):
    ops = {}
    for impl_id in V.REQUIRED_STATISTICS_OPERATORS:
        if impl_id in missing_ops:
            continue
        if impl_id in failed_ops:
            ops[impl_id] = {"arms": {}, "failed": "RuntimeError: boom"}
            continue
        ops[impl_id] = {"arms": {
            arm: {"integrity": {"passed": ok and (impl_id, arm) not in arm_fails}}
            for arm in ("full_scope", "narrow_scope")}}
    return {"operators": ops}


def _record(*, entropy=True, reduction=True, raw=True, b=True, stores=True,
            ref="root_reference", consistent=True, se=True, dp=True,
            v5=True, v3="ok"):
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
    if v3 == "ok":
        rec["V3_weighted_statistics"] = _v3()
    elif v3 is not None:
        rec["V3_weighted_statistics"] = v3
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

    @pytest.mark.parametrize("v3", [
        _v3(failed_ops=("attention.activation_importance_v1",)),
        _v3(failed_ops=("width.global_pca_v0",)),
        _v3(missing_ops=("width.global_pca_v0",)),
        _v3(arm_fails=(("attention.activation_importance_v1", "full_scope"),)),
        _v3(arm_fails=(("ffn.activation_importance_v0", "narrow_scope"),)),
        {"failed": "RuntimeError: V3 died"}],
        ids=["attention_failed", "width_failed", "width_missing",
             "attention_full_arm_wrong", "ffn_narrow_arm_wrong",
             "step_failed"])
    def test_a_partial_full_scope_is_never_a_pass(self, v3):
        """FFN's child exists and V4 ran on it -- and still no PASS."""
        v = V.verdicts(_record(v3=v3))
        assert v["integrity"]["checks"]["full_scope_statistics_operators"] \
            is False
        assert v["integrity"]["verdict"] == "FAIL"
        assert v["validation"] == "FAIL"

    def test_an_unmeasured_v3_is_incomplete(self):
        for v3 in (None, {"skipped": "deadline"}):
            v = V.verdicts(_record(v3=v3))
            assert v["integrity"]["verdict"] == "INCOMPLETE"
            assert v["validation"] == "INCOMPLETE"

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

    def test_v3_verifies_every_operators_scope_and_denominator(self, toy_run):
        v3 = toy_run[0]["V3_weighted_statistics"]
        assert v3["full_scope_integrity"] is True
        for impl_id in V.REQUIRED_STATISTICS_OPERATORS:
            full = v3["operators"][impl_id]["arms"]["full_scope"]["integrity"]
            narrow = v3["operators"][impl_id]["arms"]["narrow_scope"][
                "integrity"]
            assert full["passed"] and narrow["passed"], (impl_id, full, narrow)
            assert full["denominator"] == pytest.approx(full["weight_total"],
                                                        rel=1e-12)
            assert 0 < full["weight_total"]

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

    def test_a_failing_width_cannot_pass_though_v4_runs(self, tmp_path,
                                                       monkeypatch):
        """The reviewed failure mode, for real: FFN verifies and keeps its
        child, WIDTH raises, V4 runs on FFN's child -- validation FAILS."""
        from aadistill.initialization.operators.width.residual import (
            global_pca,
        )
        from stages.phase_d1.topk_adoption_driver import Journal

        def boom(self, ctx):
            raise RuntimeError("WIDTH broke")
        monkeypatch.setattr(global_pca.WidthGlobalPCAV0, "apply", boom)
        rec = V.stage_V(env=toy_environment(),
                        journal=Journal(tmp_path / "j.jsonl"), out=tmp_path)
        ops = rec["V3_weighted_statistics"]["operators"]
        assert "failed" in ops["width.global_pca_v0"]
        assert "failed" not in ops["ffn.activation_importance_v0"]
        assert "failed" not in rec["V4_depth_root_vs_parent"]
        v = rec["verdicts"]
        assert v["integrity"]["checks"]["full_scope_statistics_operators"] \
            is False
        assert v["integrity"]["verdict"] == "FAIL"
        assert v["validation"] == "FAIL"

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

    def test_the_executable_commit_carries_this_trees_session(self):
        """The commit a grant runs executes exactly what was reviewed: every
        executed path and record is identical between it and HEAD."""
        import subprocess
        commit = P.EXECUTABLE_COMMIT

        def git(*a):
            return subprocess.run(["git", "-C", str(REPO), *a],
                                  capture_output=True, text=True)
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
            pytest.skip("the executable commit is not in this checkout")
        assert git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
        changed = [p for p in git("diff", "--name-only", commit, "HEAD", "--",
                                  *P.EXECUTED_PATHS).stdout.split()
                   if p not in P.EXECUTED_PATH_EXCEPTIONS
                   and "/tests/" not in p and not p.startswith("tests/")]
        assert changed == [], (
            f"{changed} differ between the executable commit {commit[:12]} and "
            "this tree: re-pin EXECUTABLE_COMMIT before any grant runs it")
        auth = json.loads((REPO / V.AUTHORIZATION_PROPOSED).read_text())
        assert auth["executable_commit"] == commit
        assert f"ADOPT_COMMIT={commit}" in auth["command"]

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

def _adoptable(*, objective_hash=None, tamper=None, **record_kw):
    """A stage-V record that recomputes to PASS, with this tree's identities."""
    rec = _record(**record_kw)
    rec["premises"] = {**V.expected_identities(REPO), "teacher_revision": "r"}
    if objective_hash:
        rec["premises"]["objective_hash"] = objective_hash
    rec["verdicts"] = V.verdicts(rec)
    if tamper:
        tamper(rec)
    return rec


def _booked_tree(tmp_path, *, objective_hash=None, status="COMPLETE",
                 record=None, teardown=True):
    import importlib.util
    from stages.d_series import objectives as O
    from stages.d_series import d2_budget as BUD

    run = tmp_path / "artifacts" / "d2val_x"
    (run / "out" / "adoption_all_v1" / "work").mkdir(parents=True)
    (run / "cost.json").write_text(json.dumps(
        {"all_in_usd": 0.91, "pod_id": "pod1", "elapsed_minutes": 50.0}))
    if teardown:
        (run / "teardown_confirmed").write_text("confirmed")
    (run / "out" / "adoption_all_v1" / "work" / "big.json").write_text("{}")
    (run / "out" / "adoption_all_v1" / "adoption.json").write_text(json.dumps(
        {"status": status, "V_d2_validation": record or _adoptable(
            objective_hash=objective_hash)}))
    vdir = tmp_path / Path(BUD.D2_VALIDATION_CAMPAIGN).parent
    vdir.mkdir(parents=True)
    (vdir / "campaign.json").write_text(json.dumps(P.campaign()))
    spec = importlib.util.spec_from_file_location(
        "book", REPO / "scripts/shared/pod/book_engineering_subrun.py")
    book = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(book)
    return run, vdir, book


def _adopted(tmp_path, **kw):
    from stages.d_series import d2_budget as BUD
    run, vdir, book = _booked_tree(tmp_path, **kw)
    book.book(run_dir=run, validation_dir=vdir, subrun_id="s", outcome="x",
              evidence_session="out/adoption_all_v1/adoption.json",
              repo=tmp_path)
    return BUD.d2_validation_evidence(tmp_path)


class TestAdoption:
    """No wrong evidence may enter pricing or the scored_v1 gate."""

    def test_a_passing_run_is_adopted(self, tmp_path):
        ev = _adopted(tmp_path)
        assert ev["usable"] and ev["validation"] == "PASS"
        assert ev["scored_rows_equivalence"] == "PASS"

    def test_complete_but_validation_fail_is_not_adopted(self, tmp_path):
        ev = _adopted(tmp_path, record=_adoptable(entropy=False))
        assert not ev["usable"] and ev["scored_rows_equivalence"] is None
        assert any("validation is FAIL" in r for r in ev["reasons"])

    def test_complete_but_validation_incomplete_is_not_adopted(self, tmp_path):
        ev = _adopted(tmp_path, record=_adoptable(v3=None))
        assert not ev["usable"]

    def test_validation_pass_with_rows_fail_is_adopted_without_scored_v1(
            self, tmp_path):
        ev = _adopted(tmp_path, record=_adoptable(dp=False))
        assert ev["usable"] and ev["validation"] == "PASS"
        assert ev["scored_rows_equivalence"] == "FAIL"

    @pytest.mark.parametrize("field", ["preregistration_hash", "plan_sha256",
                                       "contract_hash", "objective_hash"])
    def test_an_identity_mismatch_is_not_adopted(self, tmp_path, field):
        def tamper(rec):
            rec["premises"][field] = "0" * 64
        ev = _adopted(tmp_path, record=_adoptable(tamper=tamper))
        assert not ev["usable"]
        assert any(field in r for r in ev["reasons"])

    def test_a_recorded_pass_the_measurements_contradict_is_refused(
            self, tmp_path):
        def tamper(rec):
            rec["V1_root_stores_and_validity"]["profiles"]["p"][
                "arithmetic"]["within_tolerance"] = False
        ev = _adopted(tmp_path, record=_adoptable(tamper=tamper))
        assert not ev["usable"]
        assert any("not what the record" in r for r in ev["reasons"])

    def test_the_authorization_gate_refuses_scored_v1_on_unadopted_evidence(
            self, tmp_path, monkeypatch):
        from stages.d_series import d2_budget as BUD
        from stages.phase_d2 import d2_authorization as A
        from stages.phase_d2 import d2_session as S
        ev = _adopted(tmp_path, record=_adoptable(entropy=False))
        monkeypatch.setattr(BUD, "d2_validation_evidence", lambda *a, **k: ev)
        with pytest.raises(A.D2AuthorizationRefused, match="scored-rows"):
            A.build_payload(
                grant={"approves_preregistration_hash":
                       S.preregistration_hash(), "granted_by": "t"},
                session_commit="a" * 40, granted_utc="t", run_id="r1",
                prediction_rows="scored_v1", live_rate=1.09)


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
        #: IDEMPOTENT: the same run, cost and evidence leave one row.
        again = book.book(run_dir=run, validation_dir=vdir,
                          subrun_id="d2val_s1", outcome="rebooked",
                          evidence_session="out/adoption_all_v1/adoption.json",
                          repo=tmp_path)
        assert again["outcome"] == "ok"
        assert len(json.loads((vdir / "campaign.json").read_text())[
            "subruns"]) == 1

    def test_a_booked_subrun_id_is_never_reused_for_another_run(self, tmp_path):
        run, vdir, book = _booked_tree(tmp_path)
        book.book(run_dir=run, validation_dir=vdir, subrun_id="s1",
                  outcome="first", repo=tmp_path)
        other = run.parent / "d2val_y"
        import shutil
        shutil.copytree(run, other)
        (other / "cost.json").write_text(json.dumps(
            {"all_in_usd": 0.40, "pod_id": "pod2"}))
        with pytest.raises(book.BookingError, match="already booked"):
            book.book(run_dir=other, validation_dir=vdir, subrun_id="s1",
                      outcome="second", repo=tmp_path)
        camp = json.loads((vdir / "campaign.json").read_text())
        assert [r["run_id"] for r in camp["subruns"]] == ["d2val_x"]
        assert camp["booked_usd"] == 0.91
        book.book(run_dir=other, validation_dir=vdir, subrun_id="s2",
                  outcome="second", repo=tmp_path)
        camp = json.loads((vdir / "campaign.json").read_text())
        assert camp["booked_usd"] == pytest.approx(1.31)

    def test_teardown_confirmation_is_persisted_and_hashed(self, tmp_path):
        run, vdir, book = _booked_tree(tmp_path)
        row = book.book(run_dir=run, validation_dir=vdir, subrun_id="s",
                        outcome="x", repo=tmp_path)
        assert row["teardown_confirmed"] is True
        assert "teardown_confirmed" in row["evidence_sha256"]
        assert (tmp_path / row["teardown_evidence"]["path"]).read_text() == \
            "confirmed"
        assert "resource_risk" not in row

    def test_a_run_without_teardown_is_booked_at_risk_and_blocks_closing(
            self, tmp_path):
        run, vdir, book = _booked_tree(tmp_path, teardown=False)
        with pytest.raises(book.BookingError, match="teardown"):
            book.book(run_dir=run, validation_dir=vdir, subrun_id="s",
                      outcome="x", close="COMPLETE and CLOSED -- x",
                      repo=tmp_path)
        camp = json.loads((vdir / "campaign.json").read_text())
        (row,) = camp["subruns"]
        assert row["cost_usd"] == 0.91 and row["teardown_confirmed"] is False
        assert "resource_risk" in row
        assert not camp["status"].startswith("COMPLETE and CLOSED")

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

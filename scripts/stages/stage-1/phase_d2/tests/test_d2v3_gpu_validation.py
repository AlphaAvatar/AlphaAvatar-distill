"""D2 version 3's GPU validation (stage W) at $0: protocol, verdicts, records,
and a CPU rehearsal through the driver itself.

    pytest scripts/stages/stage-1/phase_d2/tests/test_d2v3_gpu_validation.py

The rehearsal runs `topk_adoption_driver.main(["--stages", "W", ...])` -- the
function the pod runs -- with only the three host-specific pieces injected
(the CUDA probe, the teacher fetch and the stage environment): the real
premises, both protocol bindings, the journal, the deadline, every flush and
the `adoption.json` the launcher fetches all execute. What only the GPU can
answer -- the real teacher's confidence, bf16 ties, timings -- is what the
session is for.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]

from support.toy import (  # noqa: E402
    TARGET_GEOMETRY, TEACHER_GEOMETRY, build_tiny_model, make_items,
    make_profile,
)
from stages.phase_d2 import d2v3_gpu_validation as W  # noqa: E402
from stages.phase_d2 import d2v3_gpu_validation_plan as P  # noqa: E402

K = 8


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


def _objectives():
    from aadistill.initialization.scoring.objective import (
        FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL, KL_REFERENCE_ROOT,
        ScoringObjective,
    )
    full = ScoringObjective("objective.toy_union", 1,
                            FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
                            entropy_top_k=K, kl_reference=KL_REFERENCE_ROOT,
                            weights_activation_statistics=True)
    narrow = ScoringObjective("objective.toy_union_narrow", 1,
                              FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
                              entropy_top_k=K, kl_reference=KL_REFERENCE_ROOT)
    return full, narrow


# --- the float64 oracle and the arithmetic gates -------------------------------------

def _brute(p_z, q_z, members):
    p = torch.log_softmax(p_z.double(), -1)
    q = torch.log_softmax(q_z.double(), -1)
    out = []
    for r in range(int(p.shape[0])):
        m = torch.zeros(int(p.shape[1]), dtype=torch.bool)
        m[sorted(members[r])] = True
        pe, qe = p[r].exp(), q[r].exp()
        kl = float((pe[m] * (p[r][m] - q[r][m])).sum())
        pr, qr = float(pe[~m].sum()), float(qe[~m].sum())
        if pr > 0:
            kl += pr * (math.log(pr) - math.log(qr))
        out.append(kl)
    return torch.tensor(out, dtype=torch.float64)


class TestTheOracle:

    def test_it_is_the_brute_force_partition_kl(self):
        g = torch.Generator().manual_seed(1)
        pz = torch.randn(10, 70, generator=g) * 3
        qz = pz + torch.randn(10, 70, generator=g)
        st = torch.topk(pz, K).indices
        sc = torch.topk(qz, K).indices
        (o,) = W.oracle_kls(pz, [(qz, sc)], st, chunk=3)
        full = (torch.log_softmax(pz.double(), -1).exp()
                * (torch.log_softmax(pz.double(), -1)
                   - torch.log_softmax(qz.double(), -1))).sum(-1)
        assert torch.allclose(o["full"], full, atol=1e-12)
        mem_u = [set(a.tolist()) | set(b.tolist()) for a, b in zip(st, sc)]
        mem_t = [set(a.tolist()) for a in st]
        assert torch.allclose(o["union"], _brute(pz, qz, mem_u), atol=1e-12)
        assert torch.allclose(o["topk"], _brute(pz, qz, mem_t), atol=1e-12)
        assert bool((o["topk"] <= o["union"] + 1e-12).all())
        assert bool((o["union"] <= o["full"] + 1e-12).all())

    def test_the_teacher_pass_is_shared_by_every_candidate(self):
        g = torch.Generator().manual_seed(2)
        pz = torch.randn(6, 40, generator=g)
        qs = [pz + torch.randn(6, 40, generator=g) for _ in range(3)]
        st = torch.topk(pz, K).indices
        many = W.oracle_kls(pz, [(q, torch.topk(q, K).indices) for q in qs], st)
        for q, o in zip(qs, many):
            (one,) = W.oracle_kls(pz, [(q, torch.topk(q, K).indices)], st)
            assert torch.equal(one["union"], o["union"])

    def test_the_float32_union_kl_passes_against_it(self):
        from aadistill.initialization.scoring.output_head import (
            LinearOutputHead,
        )
        from aadistill.initialization.scoring.support import (
            TeacherRetrieval, deterministic_topk,
            sketch_reference_deterministic, union_forward_kl_terms,
        )
        g = torch.Generator().manual_seed(3)
        head = LinearOutputHead(weight=torch.randn(300, 12, generator=g))
        hidden = torch.randn(20, 12, generator=g) * 2
        z = head.logits(hidden)
        cand = (z + torch.randn(z.shape, generator=g)).bfloat16()
        sk, log_z = sketch_reference_deterministic(
            z, torch.zeros(20, dtype=torch.long), top_k=K)
        terms = union_forward_kl_terms(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            TeacherRetrieval(hidden, log_z, head), top_k=K, has_tail=True)
        _, sc = deterministic_topk(cand, K)
        (o,) = W.oracle_kls(z, [(cand, sc)], sk.support_indices)
        rows = [{"kl32": terms.kl, "kl_ref32": terms.kl_reference_topk,
                 "oracle": o, "log_z_ref": log_z}]
        ua = W.union_arithmetic(rows)
        assert ua["within_tolerance"] and ua["bounds_hold"], ua

    def test_comparing_against_the_native_bf16_teacher_is_caught(self):
        """THE NUMERICAL-PROTOCOL GUARD: the oracle must use the float32 HEAD
        distribution the store defines. A teacher whose native logits are
        rounded differently is another distribution, and treating it as the
        same one breaks the arithmetic gate."""
        from aadistill.initialization.scoring.output_head import (
            LinearOutputHead,
        )
        from aadistill.initialization.scoring.support import (
            TeacherRetrieval, deterministic_topk,
            sketch_reference_deterministic, union_forward_kl_terms,
        )
        g = torch.Generator().manual_seed(4)
        head = LinearOutputHead(weight=torch.randn(300, 12, generator=g) * 8)
        hidden = torch.randn(30, 12, generator=g)
        z = head.logits(hidden)
        native = z.bfloat16().float() + 0.05 * torch.randn(z.shape, generator=g)
        cand = (z + 0.3 * torch.randn(z.shape, generator=g)).bfloat16()
        sk, log_z = sketch_reference_deterministic(
            z, torch.zeros(30, dtype=torch.long), top_k=K)
        terms = union_forward_kl_terms(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            TeacherRetrieval(hidden, log_z, head), top_k=K, has_tail=True)
        _, sc = deterministic_topk(cand, K)
        (wrong,) = W.oracle_kls(native, [(cand, sc)], sk.support_indices)
        ua = W.union_arithmetic([{"kl32": terms.kl,
                                  "kl_ref32": terms.kl_reference_topk,
                                  "oracle": wrong, "log_z_ref": log_z}])
        assert not ua["within_tolerance"]

    def test_the_retrieval_gate_is_the_registered_tolerance(self):
        tol = W.ARITHMETIC_TOLERANCES["retrieval_abs_nats_vs_full_head"]
        assert W.retrieval_arithmetic(tol)["within_tolerance"]
        assert not W.retrieval_arithmetic(tol * 1.01)["within_tolerance"]
        assert not W.retrieval_arithmetic(float("nan"))["within_tolerance"]

    def test_the_objective_reduction_gate(self):
        obj, _ = _objectives()
        g = torch.Generator().manual_seed(5)
        per_item = [{"kl": torch.rand(30, generator=g) * 5,
                     "confidence": torch.rand(30, generator=g),
                     "scored": torch.arange(30) >= 7} for _ in range(4)]
        out = W.objective_reduction_arithmetic(obj, per_item)
        assert out["items"] == 4 and out["within_tolerance"], out


class TestTheRawKLControl:

    def _run(self, values, pid="p", support="reference_student_union_topk_tail_v1",
             objective="objective.x@v1"):
        return {"values": values, "measurement_protocol_id": pid,
                "distribution_support": {"support_id": support},
                "objective": objective}

    def test_within_one_evaluation(self):
        v = {"state.objective.raw_kl.equal_domain_mean": 2.0,
             "state.teacher_kl.equal_domain_mean": 2.0,
             "state.objective.raw_kl.worst_domain": 3.0,
             "state.teacher_kl.worst_domain": 3.0,
             "state.objective.raw_kl.critical_token": 1.0,
             "state.critical_token_kl": 1.0}
        run = self._run(v)
        out = W.raw_kl_control(run, run)
        assert out["defined"] and out["within_tolerance"]
        assert out["metrics_compared"] == 3

    def test_another_support_is_refused_not_compared(self):
        """MUTATION: the historical reference-Top-K KL wired in as the
        control. It is a different partition, so the control refuses it --
        and its numbers would not have matched anyway."""
        union = self._run({"state.objective.raw_kl.equal_domain_mean": 2.0,
                           "state.teacher_kl.equal_domain_mean": 2.0})
        hist = self._run({"state.teacher_kl.equal_domain_mean": 1.9},
                         pid="q", support="reference_topk_tail_v1",
                         objective=None)
        out = W.raw_kl_control(union, hist)
        assert out["defined"] is False and out["within_tolerance"] is False
        assert any("different supports" in r for r in out["refused"])
        assert any("different evaluations" in r for r in out["refused"])

    def test_a_breached_control_fails(self):
        v = {"state.objective.raw_kl.equal_domain_mean": 2.0,
             "state.teacher_kl.equal_domain_mean": 2.0 * (1 + 1e-4)}
        run = self._run(v)
        assert not W.raw_kl_control(run, run)["within_tolerance"]


# --- verdicts ------------------------------------------------------------------------

def _passing() -> dict:
    ok = {"within_tolerance": True}
    return {
        "premises": {"teacher_revision": "r"},
        "W1_union_stores_and_confidence": {
            "stores_match_declaration": True,
            "profiles": {"p": {"confidence_arithmetic": dict(ok)}}},
        "W2_ffn_child_weighted": {"arms": {
            a: {"integrity": {"passed": True}}
            for a in ("full_scope", "narrow_scope")}},
        "W3_union_numerics": {
            "union_arithmetic": {"within_tolerance": True,
                                 "bounds_hold": True},
            "retrieval": dict(ok), "objective_reduction": dict(ok)},
        "W4_depth_union_all_v1": {
            "reference": "root_reference",
            "reference_cache": {"mode": "root_reference_store",
                                "retrieval_bytes": 10},
            "table_consistency": {"consistent": True}},
        "W5_depth_union_scored_v1": {"agreement": dict(ok)},
        "W6_state_evaluation": {
            "raw_kl_control": dict(ok), "scored_rows": dict(ok),
            "runs": {"d2v3": {"objective": "o", "union_reference": "h",
                              "teacher_forwards_during_evaluate": 0},
                     "historical_reference_topk_kl": {
                         "objective": None,
                         "teacher_forwards_during_evaluate": 5}}}}


class TestVerdicts:

    def test_all_holding_is_pass(self):
        v = W.verdicts(_passing())
        assert v["validation"] == "PASS"
        assert v["scored_rows_equivalence"]["verdict"] == "PASS"

    @pytest.mark.parametrize("path", [
        ("W3_union_numerics", "union_arithmetic", "within_tolerance"),
        ("W3_union_numerics", "union_arithmetic", "bounds_hold"),
        ("W3_union_numerics", "retrieval", "within_tolerance"),
        ("W6_state_evaluation", "raw_kl_control", "within_tolerance")])
    def test_an_arithmetic_breach_fails(self, path):
        rec = _passing()
        rec[path[0]][path[1]][path[2]] = False
        assert W.verdicts(rec)["validation"] == "FAIL"

    def test_a_teacher_forward_during_evaluation_fails_integrity(self):
        rec = _passing()
        rec["W6_state_evaluation"]["runs"]["d2v3"][
            "teacher_forwards_during_evaluate"] = 1
        v = W.verdicts(rec)
        assert v["integrity"]["verdict"] == "FAIL"

    def test_depth_without_retrieval_fails_integrity(self):
        rec = _passing()
        rec["W4_depth_union_all_v1"]["reference_cache"].pop("retrieval_bytes")
        assert W.verdicts(rec)["integrity"]["verdict"] == "FAIL"

    def test_an_unmeasured_step_is_incomplete_and_a_failed_one_fails(self):
        rec = _passing()
        rec["W3_union_numerics"] = {"skipped": "deadline"}
        assert W.verdicts(rec)["validation"] == "INCOMPLETE"
        rec["W3_union_numerics"] = {"failed": "boom", "items_done": 3}
        assert W.verdicts(rec)["validation"] == "FAIL"

    def test_scored_rows_is_its_own_verdict(self):
        rec = _passing()
        rec["W5_depth_union_scored_v1"]["agreement"]["within_tolerance"] = False
        v = W.verdicts(rec)
        assert v["validation"] == "PASS"
        assert v["scored_rows_equivalence"]["verdict"] == "FAIL"


# --- the rehearsal: the driver's main(), end to end on a CPU toy ---------------------

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
        suite_id="test.w", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    return suite, items, "toy"


def toy_environment(repo=REPO, teacher_path=None, numerics=None,
                    step_bounds_s=None) -> W.Environment:
    from aadistill.initialization.scoring.positions import SUPERVISED_TARGET_V1
    from aadistill.initialization.scoring.support import (
        reference_student_union_topk_tail, reference_topk_tail,
    )
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter
    from aadistill.initialization.specs.materialization import (
        NumericalEnvironment,
    )

    def teacher():
        #: bf16, as on the pod: the native-bf16-vs-float32-head comparison
        #: and the bf16 Top-K ties only exist in this dtype.
        m = build_tiny_model(TEACHER_GEOMETRY).to(torch.bfloat16)
        m.config.use_cache = False
        return m

    full, narrow = _objectives()
    profiles = [(make_profile("balanced"), _calib("test.balanced@v1", 101)),
                (make_profile("reasoning", seed=2),
                 _calib("test.reasoning@v1", 303))]
    return W.Environment(
        repo=REPO, device="cpu", adapter=get_adapter("qwen3"),
        support=reference_student_union_topk_tail(K),
        historical_support=reference_topk_tail(K),
        policy=SUPERVISED_TARGET_V1,
        numerics=NumericalEnvironment(device_type="cpu",
                                      compute_dtype="float32"),
        target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY), seed=0,
        micro_batch_size=3, packing="length_sorted_v1",
        root_identity={"teacher_id": "toy", "teacher_sha256": "ab" * 32},
        teacher_revision="toyrev", load_teacher=teacher,
        profiles=lambda: profiles, load_suite=_suite,
        decode=lambda t: str(t),
        step_bounds_s=dict(step_bounds_s or {}),
        objective=full, narrow_objective=narrow)


def _hooks(**env_kw):
    return {"environment": lambda repo: {"device": "cpu",
                                         "gpu_name": None,
                                         "_rehearsal": "CPU toy"},
            "fetch_teacher": lambda repo: "toy://teacher",
            "W": lambda repo, path, numerics: toy_environment(**env_kw)}


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    from stages.phase_d1 import topk_adoption_driver as D
    out = tmp_path_factory.mktemp("w") / "adoption_all_v1"
    rc = D.main(["--out", str(out), "--stages", "W",
                 "--repo", str(REPO), "--deadline-s", "3600"],
                hooks=_hooks())
    doc = json.loads((out / "adoption.json").read_text())
    return rc, doc, out


class TestTheRehearsalThroughTheDriver:

    def test_the_driver_completes_and_writes_its_record(self, rehearsal):
        rc, doc, out = rehearsal
        assert rc == 0, doc.get("failure")
        assert doc["status"] == "COMPLETE"
        assert doc["stages_requested"] == ["W"]
        #: the real premises and both real protocol bindings ran
        assert doc["W_premises"]["objective_hash"] == \
            doc[W.RECORD_KEY]["premises"]["objective_hash"]
        assert doc["bound_reference_topk_tail"]["measurement_protocol_id"]
        stages = [s["stage"] for s in doc["stages"]]
        assert "d2v3_validation_premises" in stages
        assert "W_d2v3_validation" in stages
        assert (out / "journal.jsonl").is_file()

    def test_every_step_runs_and_none_fails(self, rehearsal):
        _, doc, _ = rehearsal
        rec = doc[W.RECORD_KEY]
        for name in W.STEP_ORDER:
            assert name in rec, name
            assert "failed" not in rec[name] and "skipped" not in rec[name], \
                (name, rec[name].get("failed"))
            assert rec[name]["wall_seconds"] >= 0

    def test_the_verdicts_pass_on_the_toy(self, rehearsal):
        _, doc, _ = rehearsal
        v = doc[W.RECORD_KEY]["verdicts"]
        assert v["arithmetic"]["verdict"] == "PASS", v["arithmetic"]
        assert v["integrity"]["verdict"] == "PASS", v["integrity"]
        assert v["validation"] == "PASS"
        assert v["scored_rows_equivalence"]["verdict"] == "PASS"

    def test_w1_measures_stores_confidence_and_drift(self, rehearsal):
        _, doc, _ = rehearsal
        w1 = doc[W.RECORD_KEY]["W1_union_stores_and_confidence"]
        assert w1["stores_match_declaration"] is True
        for prof in w1["profiles"].values():
            assert prof["identity"]["schema"] == \
                "aadistill.root_reference_store/v2"
            assert prof["builds"]["bsz1"]["identity_hash"] != \
                prof["builds"]["bsz3_formal"]["identity_hash"]
            assert prof["confidence_arithmetic"]["categories"] == K + 1
            assert prof["confidence"]["effective_fraction"] > 0
            assert "bsz1_vs_bsz3" in prof and "validity" in prof

    def test_w3_measures_the_union_against_the_oracle(self, rehearsal):
        _, doc, _ = rehearsal
        w3 = doc[W.RECORD_KEY]["W3_union_numerics"]
        assert w3["items"] == 2 * W.NUMERICS_SAMPLE["items_per_profile"] or \
            w3["items"] > 0
        ua = w3["union_arithmetic"]
        assert ua["position_violations"] == 0 and ua["bounds_hold"]
        assert ua["oracle_bound_violations"] == 0
        for key in ("union_size", "union_overlap", "ratio_union_full",
                    "p_tail_union"):
            assert key in w3["union_statistics"]
        assert set(w3["float32_head_vs_native_bf16"]) >= {
            "logit_max_abs_per_row", "topk_jaccard", "kl_head_vs_native",
            "teacher_prob_abs_on_head_topk", "confidence_abs"}
        #: the bf16 paths actually ran
        assert w3["bf16_ties"]["boundary_tie_rows"] > 0
        assert w3["float32_head_vs_native_bf16"]["kl_head_vs_native"][
            "max"] > 0
        comp = w3["component_timing"]
        for key in ("deterministic_topk_s", "retrieval_s",
                    "complement_tail_s", "union_reduction_s",
                    "k_plus_1_reduction_s"):
            assert comp[key] >= 0
        assert "share_of_weighted_objective" in json.dumps(
            w3["validity_with_child_union_kl"])

    def test_w4_ranks_four_metrics_on_a_non_root_parent(self, rehearsal):
        _, doc, _ = rehearsal
        w4 = doc[W.RECORD_KEY]["W4_depth_union_all_v1"]
        assert w4["reference"] == "root_reference"
        assert w4["reference_cache"]["retrieval_bytes"] > 0
        assert w4["synchronized_split"] is True
        assert w4["store_hits_during_round"] > 0
        assert set(w4["rankings"]["choice"]) == {
            "parent_kl", "root_teacher_topk_kl", "root_union_kl_unweighted",
            "root_union_kl_teacher_weighted"}
        assert "union_size" in w4["union_candidates"]

    def test_w6_controls_raw_kl_inside_one_evaluation(self, rehearsal):
        _, doc, _ = rehearsal
        w6 = doc[W.RECORD_KEY]["W6_state_evaluation"]
        assert w6["raw_kl_control"]["defined"] is True
        assert w6["raw_kl_control"]["support"] == \
            "reference_student_union_topk_tail_v1"
        assert w6["historical_reference_topk_kl"]["control_refuses_it"]
        for label, run in w6["runs"].items():
            if run["objective"]:
                assert run["teacher_forwards_during_evaluate"] == 0, label
                assert run["union_reference"]
        assert w6["runs"]["historical_reference_topk_kl"][
            "teacher_forwards_during_evaluate"] > 0

    def test_the_launchers_report_renders_it(self, rehearsal):
        from stages.phase_d1 import topk_adoption_report as R
        _, doc, _ = rehearsal
        text = "\n".join(R.lines(doc))
        assert "W  D2 v3 GPU validation" in text and "validation=PASS" in text
        assert "W4 choices" in text and "W3 union vs float64" in text

    def test_its_record_is_adoptable_only_under_its_identities(
            self, rehearsal, tmp_path):
        _, doc, _ = rehearsal
        ev = W.adopt(doc, REPO)
        assert ev["usable"], ev["reasons"]
        ev = W.adopt(doc, REPO, expected_commit="f" * 40)
        assert not ev["usable"]
        assert any("executable commit" in r for r in ev["reasons"])
        bad = json.loads(json.dumps(doc))
        bad[W.RECORD_KEY]["W3_union_numerics"]["union_arithmetic"][
            "within_tolerance"] = False
        ev = W.adopt(bad, REPO)
        assert not ev["usable"]
        assert any("not what the record" in r for r in ev["reasons"])


class TestDeadlineAndFailureEvidence:

    def test_a_short_deadline_skips_and_is_incomplete(self, tmp_path):
        from stages.phase_d1 import topk_adoption_driver as D
        out = tmp_path / "adoption_all_v1"
        bounds = {name: 10_000 for name in W.STEP_ORDER}
        D.main(["--out", str(out), "--stages", "W", "--repo", str(REPO),
                "--deadline-s", "120"],
               hooks=_hooks(step_bounds_s=bounds))
        doc = json.loads((out / "adoption.json").read_text())
        rec = doc[W.RECORD_KEY]
        assert all("skipped" in rec[n] for n in W.STEP_ORDER)
        assert rec["verdicts"]["validation"] == "INCOMPLETE"

    def test_a_failing_step_keeps_what_it_measured(self, tmp_path,
                                                    monkeypatch):
        """W3 dies mid-sample: the items it finished stay in the record next
        to the traceback, and the steps after it still run."""
        from stages.phase_d1 import topk_adoption_driver as D
        calls = {"n": 0}
        real = W.oracle_kls

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("injected oracle failure")
            return real(*a, **k)
        monkeypatch.setattr(W, "oracle_kls", flaky)
        out = tmp_path / "adoption_all_v1"
        D.main(["--out", str(out), "--stages", "W", "--repo", str(REPO),
                "--deadline-s", "3600"], hooks=_hooks())
        rec = json.loads((out / "adoption.json").read_text())[W.RECORD_KEY]
        w3 = rec["W3_union_numerics"]
        assert "injected oracle failure" in w3["failed"]
        assert w3["items_done"] == 2 and "traceback" in w3
        assert "failed" not in rec["W4_depth_union_all_v1"]
        assert rec["verdicts"]["validation"] == "FAIL"

    def test_a_scored_v1_base_mode_is_refused(self, tmp_path):
        from stages.phase_d1 import topk_adoption_driver as D
        out = tmp_path / "x"
        rc = D.main(["--out", str(out), "--stages", "W", "--repo", str(REPO),
                     "--prediction-rows", "scored_v1"], hooks=_hooks())
        doc = json.loads((out / "adoption.json").read_text())
        assert rc == 1 and "stage W compares all_v1" in doc["failure"][
            "message"]


# --- the records ------------------------------------------------------------------------

class TestTheRecords:

    def test_they_are_what_the_builder_derives(self):
        assert P.main([]) == 0, "run d2v3_gpu_validation_plan.py --write"

    def test_the_plan_registers_the_codes_tolerances_and_steps(self):
        plan = json.loads((REPO / W.PLAN).read_text())
        acc = plan["acceptance"]
        assert acc["arithmetic"]["tolerances"] == W.ARITHMETIC_TOLERANCES
        assert acc["integrity"]["tolerances"] == W.INTEGRITY_TOLERANCES
        assert acc["row_equivalence"]["tolerances"] == \
            W.ROW_EQUIVALENCE_TOLERANCES
        assert acc["arithmetic"]["teacher_distribution"].startswith(
            "the float32 head")
        assert set(plan["step_bounds_s"]) == set(W.STEP_ORDER)
        assert plan["execution_order"] == list(W.STEP_ORDER)
        assert plan["numerics_sample"]["items_per_profile"] == 8
        assert plan["numerics_sample"]["bypass_candidates"] == 3

    def test_the_premises_resolve_on_this_host(self):
        from stages.d_series import objectives as O
        from stages.phase_d2 import d2_session as S
        out = W.premises(REPO)
        assert out["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert out["preregistration_hash"] == S.preregistration_hash(REPO)
        assert out["preregistration_status"].startswith("DRAFT")
        assert out["plan_sha256"] == hashlib.sha256(
            (REPO / W.PLAN).read_bytes()).hexdigest()

    def test_the_proposal_prices_the_plan_and_grants_nothing(self):
        import importlib.util
        auth = json.loads((REPO / W.AUTHORIZATION_PROPOSED).read_text())
        plan = json.loads((REPO / W.PLAN).read_text())
        t = plan["totals"]
        assert auth["status"].startswith("PROPOSED")
        assert auth["plan_sha256"] == hashlib.sha256(
            (REPO / W.PLAN).read_bytes()).hexdigest()
        assert auth["session_cap_usd"] == t["proposed_session_cap_usd"]
        assert auth["all_in_ceiling_usd"] == pytest.approx(
            t["proposed_session_cap_usd"] + t["teardown_reserve_usd"])
        assert t["bound"]["all_in_usd"] <= auth["session_cap_usd"]
        assert "ADOPT_STAGES=W" in auth["launcher_command"]
        assert auth["executable_commit"] == P.EXECUTABLE_COMMIT
        assert not (REPO / W.AUTHORIZATION).exists()
        spec = importlib.util.spec_from_file_location(
            "ecb", REPO / "scripts/shared/pod/engineering_campaign_budget.py")
        ecb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ecb)
        with pytest.raises(ecb.BudgetError, match="authorization.json"):
            ecb.derive(REPO / W.CAMPAIGN, session_cap_usd=1.0,
                       rate_usd_per_hour=1.09)

    def test_every_executed_path_is_the_executable_commits(self):
        """What the pod checks out is what this tree carries, outside the
        declared exceptions (this builder, tests, dev-box tools)."""
        commit = P.EXECUTABLE_COMMIT
        have = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                              cwd=REPO, capture_output=True)
        if have.returncode != 0:
            pytest.skip(f"{commit[:12]} is not in this clone (shallow?)")
        changed = subprocess.run(
            ["git", "diff", "--name-only", commit, "--", *P.EXECUTED_PATHS],
            cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--",
             *P.EXECUTED_PATHS], cwd=REPO, capture_output=True, text=True,
            check=True).stdout.split()
        offending = [f for f in changed + untracked
                     if not any(x in f if x.startswith("/") else
                                f.startswith(x)
                                for x in P.EXECUTED_PATH_EXCEPTIONS)]
        assert offending == [], offending


def test_the_driver_knows_stage_w_and_runs_it_under_all_v1_only():
    src = (REPO / "scripts/stages/stage-1/phase_d1/topk_adoption_driver.py"
           ).read_text()
    assert '{"C", "D", "P", "L", "V", "W"}' in src
    assert "D2W.premises(repo)" in src and "D2W.stage_W(" in src


# --- the price contract: measured terms, and the scored_v1 gate ---------------------

class TestThePriceContract:

    def test_measured_union_terms_come_from_the_adopted_record(self,
                                                               rehearsal):
        from stages.d_series import d2_budget as BUD
        _, doc, _ = rehearsal
        w = {"record": doc[W.RECORD_KEY], "evidence_session": "toy"}
        base = {"operator_cell_factor": {"expected": 1.02, "bound": 1.05},
                "_geometry": {"depth_rows_per_candidate": {
                    "all_v1": 100.0, "scored_v1": 80.0}}}
        ov = BUD.measured_union_overhead(w, base)
        rec = doc[W.RECORD_KEY]
        per = (rec["W4_depth_union_all_v1"]["objective_seconds_per_candidate"]
               + max(0.0, rec["W3_union_numerics"]["component_timing"][
                   "union_minus_k_plus_1_per_row_s"]) * 100.0)
        assert ov["depth_seconds_per_candidate"]["expected"] == \
            pytest.approx(per, abs=1e-4)
        f = BUD.MEASURED_OVERHEAD_BOUND_FACTOR
        assert ov["depth_seconds_per_candidate"]["bound"] == pytest.approx(
            per * f, abs=1e-4)
        assert ov["state_eval_seconds_per_evaluation"]["expected"] >= 0.0
        assert ov["operator_cell_factor"]["expected"] >= 1.02
        assert ov["source"].startswith("measured once")
        scored = BUD.measured_union_overhead(w, base, rows_mode="scored_v1")
        assert scored["union_terms"]["rows_mode"] == "scored_v1"

    def test_the_registered_plan_stays_provisional(self):
        plan = json.loads((REPO / W.PLAN).read_text())
        terms = plan["cost_model"]["union_terms"]
        assert set(terms) == {"depth_seconds_per_candidate",
                              "state_eval_seconds_per_evaluation",
                              "one_time_minutes"}

    def test_without_an_adopted_w_the_price_is_provisional(self):
        from stages.d_series import d2_budget as BUD
        if BUD.adopted_v3_validation() is not None:
            pytest.skip("an adopted stage W exists in this tree")
        record = json.loads((REPO / BUD.RECORD).read_text())
        ov = record["steps"]["d2_search"]["objective_overhead"]
        assert ov["union_terms"]["status"] == "PROVISIONAL -- not measured"

    def test_scored_v1_needs_version_3s_own_equivalence(self, monkeypatch):
        """REGRESSION: the D2 search grant read Stage V's -- version 2's --
        scored-rows PASS, which would have let version 3 run scored_v1 on
        another objective's evidence."""
        from stages.d_series import d2_budget as BUD
        from stages.phase_d2 import d2_authorization as A
        from stages.phase_d2 import d2_session as S
        assert BUD.d2_validation_evidence()["scored_rows_equivalence"] == "PASS"
        monkeypatch.setattr(BUD, "d2v3_validation_evidence",
                            lambda *a, **k: {"scored_rows_equivalence": None})
        with pytest.raises(A.D2AuthorizationRefused, match="stage W"):
            A.build_payload(
                grant={"approves_preregistration_hash":
                       S.preregistration_hash(), "granted_by": "t"},
                session_commit="a" * 40, granted_utc="t", run_id="r1",
                prediction_rows="scored_v1", live_rate=1.09)

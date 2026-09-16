"""The baseline-completion package, checked where it can be checked for nothing.

Attempt 4's beam search completed and committed a ranking; its conditional
baseline rebuild then hit a reserve that could not fund the work. So the
candidate side of the B->C comparison is measured and frozen and the baseline
side does not exist, and a narrow completion session is being prepared to supply
only the missing half.

Everything here is `$0`. What it protects:

* the frozen candidate record really is a deterministic extraction from a
  journal the selection committed to -- and the extraction REFUSES when it is
  not, which is checked by mutating each input in turn;
* the baseline's frozen identity still reproduces;
* the completion path cannot reach a beam search;
* the comparison consumes one measured B and the five frozen C measurements,
  and no candidate is measured again;
* the pricing arithmetic is the arithmetic of its own line items, and none of it
  is the reserve that failed.
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in ("scripts", "scripts/pod", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.initialization.planning.ranking import PARETO_V1  # noqa: E402
from experiments.phase_c2 import baseline as B  # noqa: E402
from experiments.phase_c2.frozen_inputs import (  # noqa: E402
    FrozenInputError, load_frozen_candidates, load_record,
    numerically_sensitive_pairs)
from freeze_c2_comparison_inputs import FreezeError, freeze, verify  # noqa: E402

RUN = REPO / "logs/stages/stage-1/phase_c2/runs/attempt4"
FROZEN = RUN / "evidence/c2_frozen_comparison_inputs.json"
PROTOCOL = (REPO / "logs/stages/stage-1/phase_c2/plans"
            / "phase_c2_baseline_completion_protocol.json")
PRICING = (REPO / "logs/stages/stage-1/phase_c2/plans"
           / "phase_c2_baseline_completion_pricing.json")
DEFECT = (REPO / "logs/stages/stage-1/phase_c2/analyses"
          / "c2_baseline_reserve_defect.json")

def _synthetic_store(tmp_path: Path, *, validity: str = "measured",
                     tamper_objective: bool = False) -> Path:
    """A store the freeze tool can verify, built from nothing.

    The real search evidence is a 28.9 MB journal that lives outside git and
    outside a pod, so a test that reads it can only SKIP on the machines where
    it matters most -- and `audit_skip_predicates.py` flagged exactly that. A
    synthetic store removes the host dependency and reaches more refusal paths
    than the real one can: the real evidence is, correctly, not tamperable.

    The document is built with the selection's own commitment rule, so a change
    to that rule breaks this fixture rather than silently passing it.
    """
    root = tmp_path / "store"
    workdir = root / "autoinit/phase_c2_search"
    workdir.mkdir(parents=True)

    suite_hash = "a" * 64
    values = {objective.key: 1.0 + index
              for index, objective in enumerate(PARETO_V1.objectives)}
    digest = "d" * 64
    evaluation = {
        "artifact_digest": digest, "suite_id": "state_eval@v1",
        "suite_hash": suite_hash, "reference": "root_teacher",
        "values": dict(values), "positions": 10, "detail": {},
        "measured_utc": "2026-09-16T00:00:00+00:00", "runtime": {},
    }
    #: Append-only, and the FIRST record is deliberately a stale one: the
    #: canonical-record rule is `latest_by_state_id`, so a fixture with one
    #: record per state would not exercise it.
    records = [
        {"state_id": "s1", "validity": "materialized", "artifact_digest": digest,
         "evaluation": None},
        {"state_id": "s1", "validity": validity, "artifact_digest": digest,
         "evaluation": evaluation},
    ]
    journal = workdir / "states.jsonl"
    journal.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))

    decision_values = dict(values)
    if tamper_objective:
        first = PARETO_V1.objectives[0].key
        decision_values[first] = decision_values[first] + 1.0

    body = {
        "generated_utc": "2026-09-16T00:00:17+00:00",
        "search": {"run_id": "synthetic", "config_hash": "c" * 64, "seed": 1,
                   "target_spec_hash": "t" * 64,
                   "workdir": str(workdir)},
        "journal": {"path": str(journal), "sha256": sha256_file(journal)},
        "policy": {"qualified_id": PARETO_V1.qualified_id,
                   "hash": PARETO_V1.policy_hash},
        "suite": {"qualified_id": "state_eval@v1", "hash": suite_hash},
        "profiles": [],
        "selected": [{"state_id": "s1", "path": "DEPTH(x)->FFN(x)",
                      "artifact_digest": digest, "num_parameters": 1,
                      "weights_digest": "w" * 64, "single_shard_sha256": "s" * 64,
                      "arch_signature": "g" * 64, "impl_ids": [], 
                      "calibration_profiles": [], "checkpoint_path": "/gone"}],
        "n_selected": 1,
        "decisions": [{"state_id": "s1", "path": "DEPTH(x)->FFN(x)", "front": 0,
                       "position_in_front": 0, "lineage": "", "selected": True,
                       "reason": "", "objectives": decision_values,
                       "diagnostics": {}}],
    }
    body["selection_sha256"] = sha256_json(
        {k: v for k, v in body.items()
         if k not in ("selection_sha256", "generated_utc")})
    (workdir / "stage1_selection.json").write_text(json.dumps(body, indent=1))
    (root / "manifest.json").write_text(json.dumps({"entries": []}, indent=1))
    return root


def test_the_synthetic_store_verifies(tmp_path):
    """The fixture is valid, so the refusals below mean what they say."""
    context = verify(_synthetic_store(tmp_path))
    assert len(context["verified"]) == 1
    assert context["n_journal_records"] == 2, "the stale record must be present"
    assert context["n_distinct_states"] == 1
    assert context["verified"][0]["record"]["validity"] == "measured", (
        "latest_by_state_id must have chosen the LAST record, not the first")


def test_a_journal_that_is_not_the_committed_one_is_refused(tmp_path):
    """The check the whole freeze rests on, mutated.

    A selection binds its journal by sha256. Append one byte and the file is no
    longer the journal that selection describes, so neither may be used.
    """
    root = _synthetic_store(tmp_path)
    with (root / "autoinit/phase_c2_search/states.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(FreezeError, match="not the journal that selection describes"):
        verify(root)


def test_an_edited_selection_is_refused(tmp_path):
    """Its own commitment hash, through the reader that owns that rule."""
    root = _synthetic_store(tmp_path)
    path = root / "autoinit/phase_c2_search/stage1_selection.json"
    body = json.loads(path.read_text())
    body["n_selected"] = 99
    path.write_text(json.dumps(body, indent=1))
    with pytest.raises(FreezeError, match="selection_sha256"):
        verify(root)


def test_an_unmeasured_canonical_record_is_refused(tmp_path):
    """A selected state whose latest record never reached `measured`."""
    root = _synthetic_store(tmp_path, validity="validated")
    with pytest.raises(FreezeError, match="not 'measured'"):
        verify(root)


def test_a_decision_row_that_disagrees_with_the_journal_is_refused(tmp_path):
    """The cross-check: extraction and ranking must describe one measurement."""
    root = _synthetic_store(tmp_path, tamper_objective=True)
    with pytest.raises(FreezeError, match="decision row says"):
        verify(root)


def test_the_committed_frozen_record_matches_the_committed_selection():
    """The real evidence, with no external dependency.

    The 28.9 MB journal is outside git, but both documents this compares are
    committed -- so the freeze's agreement with the ranking is checkable on any
    machine, including a pod.
    """
    record = load_record(FROZEN)
    selection = json.loads((RUN / "evidence/stage1_selection.json").read_text())
    assert record["sources"]["selection_commitment_sha256"] == selection["selection_sha256"]
    assert record["sources"]["journal_sha256"] == selection["journal"]["sha256"]
    assert record["search"]["config_hash"] == selection["search"]["config_hash"]
    assert record["suite"] == selection["suite"]
    assert record["policy"] == selection["policy"]


# --- 3. the baseline's frozen identity ---------------------------------------


def test_the_frozen_b_identity_still_reproduces():
    from experiments.phase_c2.search_space import register_c2_operators
    register_c2_operators()
    spec = B.frozen_baseline_spec(device="cuda")
    assert spec.spec_hash == B.B_SPEC_HASH
    assert B.path_identity_of_spec(spec) == B.B_PATH
    assert spec.device == "cuda", "device is part of the frozen spec hash"
    kinds = [kind for kind, _impl, _profile in B.B_PATH]
    assert kinds == ["DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION"]


def test_the_protocol_binds_the_identities_it_claims_to():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["protocol_sha256"] == sha256_json(
        {k: v for k, v in protocol.items() if k != "protocol_sha256"})
    assert protocol["authorizes"] == "nothing"

    b = protocol["the_baseline_B"]
    assert b["spec_hash"] == B.B_SPEC_HASH
    assert b["expected_artifact_digest"] == B.B_ARTIFACT_DIGEST
    assert b["_spec_hash_reproduced_live"] is True

    forbidden = protocol["explicitly_forbidden"]
    for key in ("rerun_the_attempt_4_beam", "generate_any_new_C_candidate",
                "repeat_or_replace_any_C_measurement", "search_2",
                "behavioural_confirmation"):
        assert forbidden[key] is False, key

    bound = protocol["cross_session_comparability_contract"]["bound"]
    assert bound["reference_strategy"] == "RECOMPUTE"
    assert bound["pareto_policy_hash"] == PARETO_V1.policy_hash
    frozen = protocol["frozen_identities_that_must_not_move"]
    assert {k: float(v) for k, v in frozen["pareto_epsilon"].items()} == {
        k: float(v) for k, v in PARETO_V1.epsilon.items()}
    #: The contract must name the evaluator it is binding, by content, and those
    #: hashes must describe the tree as it is now.
    for path, expected in bound["evaluator_implementation_sha256"].items():
        assert sha256_file(REPO / path) == expected, path


# --- 4. the completion path cannot reach a beam search ----------------------


DRIVER = REPO / "scripts/pod/autoinit_phase_c2_baseline_driver.py"

#: `Deadline` lives in the same module as `BeamSearch`, and the baseline rebuild
#: legitimately imports it for its own clock. So importability of that module is
#: NOT the property under test. What must be unreachable is the beam ENTRY POINT
#: and any construction of a search.
BEAM_ENTRY_NAMES = ("run_phase_a_search", "BeamSearch", "SearchConfig",
                    "SCHEDULE_V1", "phase_a_search")


def test_the_driver_never_names_a_beam_entry_point():
    tree = ast.parse(DRIVER.read_text())
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            named.add(node.module.split(".")[-1])
            named |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            named |= {alias.name.split(".")[-1] for alias in node.names}
    leaked = sorted(set(BEAM_ENTRY_NAMES) & named)
    assert not leaked, (
        f"the baseline-completion driver names {leaked}. A grant authorizing "
        "baseline completion must not be able to run another Search-1 beam, and "
        "that has to be true by construction rather than by instruction.")


def test_the_driver_calls_the_rebuild_not_the_search_hook():
    """`BaselineFallback.__call__` is the hook a BEAM calls; `rebuild` is not."""
    source = DRIVER.read_text()
    assert ".rebuild(" in source
    assert "conditional_candidates" not in source, (
        "conditional_candidates is the search's hook parameter; naming it here "
        "means this path is being wired into a search")


def test_no_module_the_driver_imports_reaches_the_beam_runner():
    """Transitively, over in-repo modules, by import graph.

    `scripts/autoinit/phase_a_search.py` is the only thing in this repository
    that runs a beam. Nothing the completion driver imports may reach it.
    """
    roots = {"src/aadistill": REPO / "src", "scripts": REPO / "scripts"}

    def resolve(module: str) -> Path | None:
        for base in (REPO / "src", REPO / "scripts", REPO / "scripts/pod",
                     REPO / "scripts/autoinit"):
            candidate = base / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                return candidate
            package = base / module.replace(".", "/") / "__init__.py"
            if package.is_file():
                return package
        return None

    seen, queue, offenders = set(), [DRIVER], []
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules += [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
        for module in modules:
            if module.split(".")[-1] == "phase_a_search":
                offenders.append((str(path.relative_to(REPO)), module))
            resolved = resolve(module)
            if resolved and resolved not in seen:
                queue.append(resolved)
    assert not offenders, (
        f"the beam runner is reachable from the completion driver: {offenders}")
    assert len(seen) > 5, "the import walk did not actually traverse anything"


# --- 5. the comparison consumes frozen C and one measured B -----------------


def test_the_comparison_reads_five_frozen_candidates_and_measures_none():
    candidates = load_frozen_candidates(
        FROZEN,
        expect_suite_hash=json.loads(PROTOCOL.read_text())[
            "cross_session_comparability_contract"]["bound"]["state_eval_suite_hash"])
    assert len(candidates) == 5
    for candidate in candidates:
        #: Each arrives already measured. There is no method on this object that
        #: could measure it again -- that is why it is not an
        #: `InitializationState`.
        assert candidate.evaluation.values
        assert not hasattr(candidate, "attach_evaluation")
        assert not hasattr(candidate, "materialize")
        for objective in PARETO_V1.objectives:
            assert objective.key in candidate.evaluation.values


def test_the_disclosure_rule_fires_only_inside_the_registered_threshold():
    candidates = load_frozen_candidates(FROZEN)
    objectives = tuple(o.key for o in PARETO_V1.objectives)
    protocol = json.loads(PROTOCOL.read_text())
    disclosure = protocol["cross_session_comparability_contract"][
        "preregistered_disclosure_RULE_not_a_rule_change"]
    threshold = float(disclosure["threshold"])

    #: A B that sits exactly on a candidate's value is maximally sensitive.
    on_top = dict(candidates[0].evaluation.values)
    flagged = numerically_sensitive_pairs(on_top, candidates,
                                          objectives=objectives, threshold=threshold)
    assert any(f["state_id"] == candidates[0].state_id for f in flagged)

    #: A B far from every candidate is flagged nowhere.
    far = {key: 1000.0 for key in objectives}
    assert numerically_sensitive_pairs(far, candidates, objectives=objectives,
                                       threshold=threshold) == []
    assert threshold > float(PARETO_V1.epsilon[objectives[0]]), (
        "the disclosure threshold is the candidate side's own tightest observed "
        "gap and must be wider than epsilon, or it discloses nothing epsilon "
        "does not already decide")


# --- 6. the pricing is its own arithmetic, and not the failed reserve -------


def test_the_pricing_totals_are_the_sum_of_their_line_items():
    pricing = json.loads(PRICING.read_text())
    assert pricing["pricing_sha256"] == sha256_json(
        {k: v for k, v in pricing.items() if k != "pricing_sha256"})
    totals = pricing["totals"]

    bounding = sum(item["minutes"] for item in pricing["line_items_bounding"])
    assert bounding == pytest.approx(totals["bounding_path_minutes"], abs=5e-4)
    expected = sum(item["minutes"] for item in pricing["line_items_expected"])
    assert expected == pytest.approx(totals["expected_minutes"], abs=5e-4)

    hard = (totals["bounding_path_minutes"] + totals["contingency_minutes"]
            + totals["artifact_recovery_reserve_minutes"])
    assert hard == pytest.approx(totals["hard_ceiling_minutes"], abs=5e-4)
    assert totals["contingency_minutes"] == pytest.approx(
        totals["bounding_path_minutes"] * totals["contingency_fraction"], abs=5e-4)

    #: A ceiling rounds UP, and it must be at least the exact product.
    assert totals["hard_ceiling_usd"] >= (
        totals["hard_ceiling_minutes"] / 60.0 * totals["price_per_hour"])
    assert totals["expected_minutes"] < totals["hard_ceiling_minutes"]


def test_the_pricing_bounds_depth_by_eval_count_not_by_a_past_wall_clock():
    pricing = json.loads(PRICING.read_text())
    depth = next(item for item in pricing["line_items_bounding"]
                 if item["item"] == "depth_full_derivation")
    observed = pricing["observations"]["depth"]
    assert observed["evals_per_complete_derivation"] == 260
    assert observed["complete_derivations"] == 7
    assert depth["minutes"] == pytest.approx(
        260 / observed["rate_min_eval_per_min"], abs=1e-6)
    #: The bound must exceed the whole failed reserve, which is the point.
    assert depth["minutes"] > 27.665


def test_the_failed_reserve_is_not_an_input_to_the_new_pricing():
    pricing = json.loads(PRICING.read_text())
    for item in pricing["line_items_bounding"]:
        assert item["minutes"] != 27.665
        assert item["minutes"] != 24.185
    #: It is named in prose -- deliberately, to record what is NOT being reused.
    assert "27.665" in pricing["_does_not_reuse_the_failed_reserve"]


def test_the_defect_record_classifies_and_does_not_edit_history():
    defect = json.loads(DEFECT.read_text())
    assert defect["classification"] == "RUNTIME/PRICING ENVELOPE DEFECT"
    assert defect["authorizes"] == "nothing"
    assert defect["the_discrepancy"]["reserve_minutes"] == 27.665
    assert defect["the_discrepancy"]["shortfall_factor"] > 1.5

    #: The historical pricing document is untouched and still carries the
    #: reserve that failed, because it is the authorization basis attempts 2, 3
    #: and 4 ran under.
    historical = json.loads((REPO / "logs/stages/stage-1/phase_c2/plans"
                             / "phase_c2_pricing.json").read_text())
    reserve = next(r for r in historical["reserves"]
                   if r["reserve"] == "baseline_rebuild_reserve")
    assert reserve["minutes"] == 27.665
    assert historical["pricing_sha256"] == sha256_json(
        {k: v for k, v in historical.items() if k != "pricing_sha256"})


# --- 7. governance scope: nothing here authorizes anything ------------------


def test_nothing_in_this_package_authorizes_a_paid_session():
    for path in (PROTOCOL, PRICING, DEFECT, FROZEN):
        assert json.loads(path.read_text())["authorizes"] == "nothing", path
    runs = REPO / "logs/stages/stage-1/phase_c2/runs"
    completion = sorted(runs.glob("*baseline*"))
    assert not completion, (
        f"a baseline-completion run directory already exists: {completion}. "
        "This round is zero-cost and no such run is authorized.")


# --- 8. the stage-B seam, orchestrated for real ----------------------------
#
# The checks above are all static: they read documents and source. None of them
# executes `rebuild_measure_compare`, and every defect this section exists to
# catch lived inside it -- the suite root, the unprimed evaluator, the second
# teacher, the caller-supplied search identity, the wrong citation, the nested
# verdict, and a record mutated after its own hash. Each would have surfaced
# only after B had been rebuilt and measured, which is the most expensive
# possible moment.
#
# So this runs the real orchestration and stubs ONLY model work: the teacher
# object, the checkpoint identity, the adapter's config/param/load calls, the
# rebuild itself, and the single evaluation. Everything that decides CORRECTNESS
# stays real -- the suite is resolved and loaded from its declaration, the
# evaluator is the real one, the candidates are the real frozen five, the policy
# is PARETO_V1, and the record is built and hashed by the real builder.


class _FakeTeacher:
    """A sentinel. Identity is the point: exactly one must exist."""

    def __call__(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the stubbed evaluation must not run the teacher")


class _FakeIdentity:
    """What the driver and the comparison read off a checkpoint identity.

    Not a real `CheckpointIdentity`: that type DERIVES `artifact_digest` from
    the bytes it hashed, so it cannot be told to report B's digest without real
    weights. `identify_checkpoint` is the stub boundary here, and this is the
    shape it returns.
    """

    def __init__(self, digest: str) -> None:
        self.artifact_digest = digest
        self.weights_digest = "w" * 64
        self.config_sha256 = "c" * 64
        self.arch_signature = "g" * 64
        self.tokenizer_sha256 = "t" * 64
        self.index_sha256 = None
        self.single_shard_sha256 = "s" * 64
        self.num_parameters = 596049920


def _fake_identity(digest: str):
    return _FakeIdentity(digest)


def _run_stage_b(tmp_path, monkeypatch, *, baseline_values):
    """Drive the real stage B with the expensive work stubbed. Returns (driver, calls)."""
    import autoinit_phase_c2_baseline_driver as D
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.specs.metrics import StateEvaluation
    from experiments.phase_c2 import baseline as BL

    calls = {"teachers": [], "evaluate": [], "rebuild_loader_gave": [],
             "primed_with": []}

    monkeypatch.setattr(D, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(D, "WORK", tmp_path / "work")
    monkeypatch.setattr(D, "STATUS", tmp_path / "status")
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)

    #: ONE teacher. The driver's loader is stubbed, and the test counts how many
    #: times it is called: a second load would mean two 4B models and two
    #: different objects behind B's construction and B's reference.
    def fake_loader(self):
        teacher = _FakeTeacher()
        calls["teachers"].append(teacher)
        return teacher
    monkeypatch.setattr(D.BaselineCompletionDriver, "load_original_teacher", fake_loader)

    real_prime = StateEvaluator.prime_reference

    def spy_prime(self, teacher):
        calls["primed_with"].append(teacher)
        return real_prime(self, teacher)
    monkeypatch.setattr(StateEvaluator, "prime_reference", spy_prime)

    def fake_evaluate(self, model, artifact_digest, **kwargs):
        #: The priming semantic, enforced where it actually matters: the real
        #: `evaluate` REFUSES an unprimed evaluator, so a driver that skipped
        #: `prime_reference` would fail here rather than produce a comparison.
        assert self._ref_ready and self._teacher is not None, (
            "the evaluator was never primed with the original teacher")
        calls["evaluate"].append(artifact_digest)
        return StateEvaluation(
            artifact_digest=artifact_digest, suite_id=self.suite.qualified_id,
            suite_hash=self.suite.suite_hash, reference="root_teacher",
            values=dict(baseline_values), positions=74022,
            detail={"reference_strategy": "recompute"},
            measured_utc="2026-09-17T00:00:00+00:00")
    monkeypatch.setattr(StateEvaluator, "evaluate", fake_evaluate)

    def fake_rebuild(self, root_loader):
        calls["rebuild_loader_gave"].append(root_loader())
        directory = tmp_path / "b"
        directory.mkdir(exist_ok=True)
        self.outcome = {"resolution": "rebuilt", "rebuilt": True, "rebuilds": 1,
                        "state_id": BL.REBUILT_STATE_ID,
                        "identity_matches": {"artifact_digest": True},
                        "construction": {"verified": True}}
        return {"candidate_id": BL.REBUILT_STATE_ID,
                "checkpoint_dir": str(directory),
                "expected_artifact_digest": BL.B_ARTIFACT_DIGEST,
                "provenance": "c2_baseline_rebuilt_from_frozen_fixed_path",
                "description": "stubbed rebuild"}
    monkeypatch.setattr(BL.BaselineFallback, "rebuild", fake_rebuild)

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        staticmethod(lambda *a, **k: object()))
    from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
    monkeypatch.setattr(QWEN3_ADAPTER, "spec_from_config", lambda config: None)
    monkeypatch.setattr(QWEN3_ADAPTER, "param_count", lambda spec: 596049920)
    monkeypatch.setattr(QWEN3_ADAPTER, "load", lambda *a, **k: object())
    monkeypatch.setattr("aadistill.initialization.specs.artifact.identify_checkpoint",
                        lambda *a, **k: _fake_identity(BL.B_ARTIFACT_DIGEST))
    monkeypatch.setattr(D, "make_retained_state", _fake_retained)

    evidence = RUN / "evidence"
    #: REAL argument parsing, through the driver's own parser.
    args = D.build_parser().parse_args([
        "--protocol", str(PROTOCOL),
        "--frozen-inputs", str(evidence / "c2_frozen_comparison_inputs.json"),
        "--selection-record", str(evidence / "stage1_selection.json"),
        "--rebuild-minutes", "45", "--rate", "1.09", "--soft-stop-usd", "1.10",
        #: `cuda`, because the frozen B spec hash INCLUDES the device and
        #: `cuda` is the value C1 froze -- stage A correctly refuses anything
        #: else. No CUDA call happens: every model operation is stubbed, and the
        #: device is a string in a spec until one of them runs.
        "--device", "cuda"])
    driver = D.BaselineCompletionDriver(args)
    assert driver.bind_identities() is True
    assert driver.rebuild_measure_compare() is True
    return driver, calls


def _fake_retained(*, state_id, artifact, spec, target_spec, num_parameters,
                   root_teacher_id, root_teacher_sha256, description, provenance,
                   expected_artifact_digest=None):
    """The retained-state shape the comparison reads, without the lifecycle."""
    from aadistill.initialization.specs.state import StateValidity

    class _State:
        """Only what the comparison and the ranking read of a measured state."""

        def __init__(self):
            self.state_id = state_id
            self.artifact = artifact
            self.artifact_digest = expected_artifact_digest
            self.num_parameters = num_parameters
            self.path_label = "DEPTH(db)->FFN(db)->RESIDUAL_WIDTH(rh)->ATTENTION(db)"
            self.provenance = provenance
            self.evaluation = None
            self.impl_ids = ("depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
                             "width.global_pca_v0", "attention.activation_importance_v1")

        @property
        def validity(self):
            return (StateValidity.MEASURED if self.evaluation is not None
                    else StateValidity.VALIDATED)

        def ready_for_ranking(self, required):
            assert self.evaluation is not None
            self.evaluation.require(list(required))

        def attach_evaluation(self, evaluation):
            self.evaluation = evaluation
    return _State()


def _far_from_every_candidate() -> dict:
    return {objective.key: 1000.0 for objective in PARETO_V1.objectives}


def test_stage_b_orchestration_is_correct_end_to_end(tmp_path, monkeypatch):
    """One pass over the real seam, asserting each repaired defect."""
    driver, calls = _run_stage_b(tmp_path, monkeypatch,
                                 baseline_values=_far_from_every_candidate())
    record = json.loads((tmp_path / "audit" / "c2_baseline_comparison.json").read_text())

    #: 1. the suite came from its declaration, not the repository root
    detail = driver.ev["stages"]["rebuild_measure_compare"]["detail"]
    from experiments.phase_c2.frozen_assets import state_eval_root
    assert detail["suite_root"] == str(state_eval_root(REPO))
    assert record["suite"]["hash"] == json.loads(FROZEN.read_text())["suite"]["hash"]

    #: 2. exactly ONE teacher, primed AND handed to the rebuild
    assert len(calls["teachers"]) == 1, "the teacher must be loaded once"
    assert calls["primed_with"] == calls["teachers"], "the evaluator was not primed"
    assert calls["rebuild_loader_gave"] == calls["teachers"], (
        "the rebuild received a different teacher object than the evaluator was "
        "primed with")
    assert detail["teacher_instances_loaded"] == 1

    #: 3. the Search-1 identities are the frozen ones
    frozen = json.loads(FROZEN.read_text())
    assert record["run_id"] == frozen["search"]["run_id"] == "autoinit.v1.phase_c2.search1"
    assert record["search_config_hash"] == frozen["search"]["config_hash"]

    #: 4. the beam ranking is cited as the ranking; the extraction separately
    assert record["cites"]["beam_ranking"].endswith("stage1_selection.json")
    assert record["cites"]["frozen_candidate_inputs"].endswith(
        "c2_frozen_comparison_inputs.json")

    #: 5. the verdict is read from where the builder puts it
    assert record["comparison"]["verdict"] == detail["verdict"]

    #: 6. the disclosure is INSIDE the hash
    assert "numerical_sensitivity" in record
    recomputed = sha256_json({k: v for k, v in record.items() if k != "record_sha256"})
    assert record["record_sha256"] == recomputed, (
        "the final record does not match its own self-hash; a field was added "
        "after finalization")

    #: 7. one B measurement, five frozen C, none remeasured
    assert calls["evaluate"] == [B.B_ARTIFACT_DIGEST]
    assert len(record["candidates"]) == 5
    assert detail["candidates_remeasured"] == 0

    #: and the disclosure is honest when nothing is flagged
    sensitivity = record["numerical_sensitivity"]
    assert sensitivity["status"] == "NOT_FLAGGED"
    assert sensitivity["cross_session_variance"] == "NOT DIRECTLY MEASURED"
    assert "NOT an estimated noise bound" in sensitivity["_what_the_threshold_is_not"]
    assert "not a determinism claim" in sensitivity["interpretation"].lower()


def test_a_close_baseline_is_flagged_without_changing_the_verdict(tmp_path, monkeypatch):
    """A B that lands on a candidate's values discloses, and the verdict stands."""
    candidates = load_frozen_candidates(FROZEN)
    on_top = dict(candidates[0].evaluation.values)
    driver, _ = _run_stage_b(tmp_path, monkeypatch, baseline_values=on_top)
    record = json.loads((tmp_path / "audit" / "c2_baseline_comparison.json").read_text())

    sensitivity = record["numerical_sensitivity"]
    assert sensitivity["status"] == "FLAGGED"
    assert sensitivity["n_flagged_pairs"] >= 1
    assert "SENSITIVITY-LIMITED" in sensitivity["interpretation"]

    #: The PARETO result is computed, not adjusted by the disclosure.
    assert record["comparison"]["verdict"] in (
        "CANDIDATE_IN_A_BETTER_FRONT_THAN_BASELINE",
        "BASELINE_AND_CANDIDATES_SHARE_A_FRONT",
        "BASELINE_IN_A_BETTER_FRONT_THAN_EVERY_CANDIDATE")
    assert record["policy"]["epsilon"] == {
        objective.key: 1e-4 for objective in PARETO_V1.objectives}
    #: still inside the hash
    assert record["record_sha256"] == sha256_json(
        {k: v for k, v in record.items() if k != "record_sha256"})


def test_the_builder_refuses_an_interpretation_that_would_shadow_the_result():
    """Interpretation is added beside the computed result, never over it."""
    from experiments.phase_c2.comparison import ComparisonError, build
    with pytest.raises(ComparisonError, match="already computes"):
        build(baseline=_fake_baseline_for_build(), baseline_outcome={},
              candidates=[], suite=_FakeSuite(), policy=PARETO_V1,
              run_id="r", config_hash="h",
              interpretation={"comparison": {"verdict": "FABRICATED"}})


class _FakeSuite:
    qualified_id = "state_eval@v1"
    suite_hash = "a" * 64


def _fake_baseline_for_build():
    from aadistill.initialization.specs.metrics import StateEvaluation
    from aadistill.initialization.specs.state import StateValidity

    class _B:
        state_id = "b"
        artifact_digest = "d" * 64
        num_parameters = 1
        path_label = "x"
        impl_ids = ()
        validity = StateValidity.MEASURED

        def ready_for_ranking(self, required):
            self.evaluation.require(list(required))

        evaluation = StateEvaluation(
            artifact_digest="d" * 64, suite_id="state_eval@v1",
            suite_hash="a" * 64, reference="root_teacher",
            values={o.key: 1.0 for o in PARETO_V1.objectives}, positions=1)
    return _B()


# --- 9. the thin formal path, and the scope it cannot exceed ----------------

LAUNCHER = REPO / "scripts/pod/autoinit_phase_c2_baseline_launch.py"
SETUP_SCRIPT = REPO / "scripts/pod/autoinit_preflight_setup.sh"


def _completion_spec():
    import autoinit_phase_c2_baseline_launch as L
    args = L.build_parser().parse_args([
        "--scr", "/tmp/unused", "--session-commit", "0" * 40,
        "--bundle", "b.bundle", "--run-id", "attempt5"])
    return L, args, L.spec(args)


def test_the_setup_script_has_a_branch_for_this_session_kind():
    """A missing SESSION_KIND branch is not a type error -- it is $0.23.

    `SESSION_KIND` falls through to `spend`, which loads a
    `PreflightAuthorization` and exits 98 AFTER setup has run on a billing pod.
    Phase-B attempt 2 established the price. So a new session kind owes a branch
    in the shell dispatch table, and this is the check that says so.
    """
    _, _, spec = _completion_spec()
    kind = spec.setup.env["SESSION_KIND"]
    script = SETUP_SCRIPT.read_text()
    assert f'"$SESSION_KIND" = "{kind}"' in script, (
        f"the shared setup script has no branch for SESSION_KIND={kind!r}; it "
        "would fall through to the spend branch and exit 98 on a paid pod")
    #: And the branch must carry the governance boundary into the pod.
    branch = script.split(f'"$SESSION_KIND" = "{kind}"')[1].split("elif [")[0]
    assert "BaselineCompletionAuthorization" in branch
    assert "authorizes_c2_search1 is False" in branch, (
        "the pod-side branch must refuse an artifact that could authorize a beam")


def test_the_completion_session_stages_only_what_it_needs():
    """No vLLM, no canonical control, both mixtures, the frozen suite."""
    _, _, spec = _completion_spec()
    setup = spec.setup

    assert "VLLM_READY" not in setup.setup_markers, (
        "this session serves nothing; declaring the marker is what would build "
        "the inference environment")
    for required in ("ENV_READY", "REPO_READY", "TRAIN_ENV", "ASSETS_READY",
                     "TEACHER_READY", "ROPE_OK", "TESTS_OK", "AUTHORIZATION_OK"):
        assert required in setup.setup_markers, required

    staged = {asset.dest_name for asset in setup.local_assets}
    assert staged == {"reasoning_heavy_v2", "state_eval_v1"}, staged
    relayed = {inp.path for inp in setup.relay_inputs}
    assert any("calibration_v1" in path for path in relayed), (
        "calib.domain_balanced@v1's items must be staged")
    #: Search-1 staged the canonical 0.6B init as its measured CONTROL. This
    #: session has no control and must not stage one.
    assert not any("qwen3_0p6b_init_v0" in path for path in relayed), (
        "the canonical control is Search-1's, not this session's")
    assert setup.teacher_revision, "the pinned teacher revision must be declared"


def test_the_completion_session_names_its_own_driver_and_markers():
    import inspect
    _, _, spec = _completion_spec()
    assert spec.driver_job_id == "autoinit_phase_c2_baseline_driver"
    #: `driver_command` is a callable the runner invokes with the live context,
    #: so the question is what it WOULD invoke.
    command_source = inspect.getsource(spec.driver_command)
    assert "autoinit_phase_c2_baseline_driver.py" in command_source
    assert "autoinit_phase_c2_driver.py" not in command_source, (
        "the completion launcher would start the Search-1 driver")
    #: And it passes no scientific identity: those are bound from the frozen
    #: record inside the driver.
    assert "--config-hash" not in command_source
    assert spec.markers.success == "BASELINE_COMPLETION_ALL_DONE"
    assert spec.markers.failure == ("BASELINE_COMPLETION_FAILED",)
    #: Nothing this session produces is fetched as a product: the rebuilt B is
    #: re-derivable and what must survive is its measurement.
    assert spec.markers.products_eligible("ANY", {}) is False


def test_the_completion_authorization_can_never_authorize_a_beam():
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2.session import C2Authorization

    #: The property, on the type.
    assert BC.BaselineCompletionAuthorization.authorizes_c2_search1.fget(
        None) is False
    assert BC.BaselineCompletionAuthorization.authorizes_c2_baseline_completion.fget(
        None) is True
    #: And Search-1's type still says the opposite, so the two are distinct
    #: permissions rather than one relabelled.
    assert C2Authorization.authorizes_c2_search1.fget(None) is True


def test_the_two_authorization_types_refuse_each_others_artifacts(tmp_path):
    """Symmetry. Neither schema may stand in for the other."""
    from aadistill.governance.authorization import AuthorizationError
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2.session import SCHEMA as C2_SCHEMA, C2Authorization

    def written(schema: str, **extra) -> Path:
        payload = {"schema": schema, "authorization_id": "x", **extra}
        payload["authorization_sha256"] = sha256_json(payload)
        path = tmp_path / f"{abs(hash(schema)) % 10000}.json"
        path.write_text(json.dumps(payload, indent=1))
        return path

    with pytest.raises(AuthorizationError, match="schema"):
        C2Authorization.load(written(BC.SCHEMA))
    with pytest.raises(AuthorizationError, match="schema"):
        BC.BaselineCompletionAuthorization.load(written(C2_SCHEMA))
    #: And an artifact of the right schema that claims the beam is refused too.
    with pytest.raises(AuthorizationError, match="authorizes_c2_search1"):
        BC.BaselineCompletionAuthorization.load(
            written(BC.SCHEMA, authorizes_c2_search1=True))


def test_the_launcher_cannot_reach_the_beam_runner():
    """The same import-graph question, asked of the launcher."""
    tree = ast.parse(LAUNCHER.read_text())
    named = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            named.add(node.module.split(".")[-1])
            named |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            named |= {alias.name.split(".")[-1] for alias in node.names}
    leaked = sorted(set(BEAM_ENTRY_NAMES) & named)
    assert not leaked, f"the completion launcher names {leaked}"

    #: And the DERIVED closure -- what would actually run -- contains no
    #: Search-1 module at all.
    from experiments.phase_c2 import baseline_completion as BC
    paths = [row["path"] for row in BC.current_executable(REPO)["files"]]
    for forbidden in ("scripts/autoinit/phase_a_search.py",
                      "scripts/pod/autoinit_phase_c2_driver.py",
                      "scripts/pod/autoinit_phase_c2_launch.py"):
        assert forbidden not in paths, (
            f"{forbidden} is in the completion executable set")


def test_the_completion_budget_reproduces_the_accepted_pricing():
    from experiments.phase_c2 import baseline_completion as BC
    assert BC.hard_ceiling_usd(REPO) == 1.1950
    assert BC.price_per_hour_usd(REPO) == 1.09
    #: The rebuild allowance must exceed the reserve that failed, or the repair
    #: is cosmetic.
    assert BC.rebuild_minutes(REPO) > 27.665
    budget = BC.budget_spec(REPO)
    assert budget.arms == 0, "this session trains nothing"
    assert budget.soft_stop_reserves == (), (
        "no beam-composition risk and no conditional rebuild reserve: the "
        "rebuild IS the session and is priced in the path")


def test_no_completion_run_or_authorization_exists_yet():
    """This round is zero-cost and prepares tooling only."""
    import autoinit_phase_c2_baseline_launch as L
    runs = REPO / "logs/stages/stage-1" / L.RUN_EXPERIMENT_ID / "runs"
    assert not runs.exists() or not any(runs.iterdir()), (
        f"a completion run directory already exists at {runs}; no completion "
        "session is authorized")
    assert not list(REPO.glob(
        "logs/stages/**/phase_c2_baseline_completion/**/authorization.json"))

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
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

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
    #: Model work: it reads a real config and instantiates a rotary module. The
    #: guard itself is driven for real by
    #: `test_the_rope_guard_refuses_a_disagreeing_base`.
    monkeypatch.setattr(
        D.BaselineCompletionDriver, "assert_rope_base_of_rebuilt_b",
        lambda self, directory: {"stored": 5_000_000.0, "runtime": 5_000_000.0,
                                 "transformers": "stub"})

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
                     "TEACHER_READY", "TESTS_OK", "AUTHORIZATION_OK"):
        assert required in setup.setup_markers, required
    #: ROPE_OK is deliberately absent and the guard it names moved into the
    #: driver: see `test_the_completion_does_not_declare_rope_ok` and
    #: `test_undeclaring_rope_ok_did_not_remove_the_guard`.
    assert "ROPE_OK" not in setup.setup_markers

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


def test_the_governance_chain_is_never_out_of_order():
    """The chronology, asserted as an invariant that holds at every stage.

    This used to assert that NO completion run existed, which was true while
    the package was being prepared and false the moment a maintainer granted
    one. What must hold forever is the ORDER: a readiness record is evidence
    about a grant-containing tree, and an authorization is issued on a tree that
    already carries the record. So each artifact implies its predecessor, and a
    chain that skipped a step would be visible here rather than at a pod.
    """
    import autoinit_phase_c2_baseline_launch as L

    runs = REPO / "logs/stages/stage-1" / L.RUN_EXPERIMENT_ID / "runs"
    for run in sorted(runs.glob("*")) if runs.exists() else []:
        governance = run / "governance"
        grant = (governance / "grant.json").is_file()
        readiness = (governance / "readiness.json").is_file()
        authorization = (governance / "authorization.json").is_file()
        if readiness:
            assert grant, f"{run.name}: a readiness record with no grant"
        if authorization:
            assert readiness, (
                f"{run.name}: an authorization issued with no readiness record "
                "-- the record is evidence about the tree the authorization was "
                "issued on")
            assert grant, f"{run.name}: an authorization with no grant"


def test_any_completion_authorization_permits_only_completion():
    """Whatever exists, it cannot authorize a beam."""
    import autoinit_phase_c2_baseline_launch as L
    from experiments.phase_c2 import baseline_completion as BC

    runs = REPO / "logs/stages/stage-1" / L.RUN_EXPERIMENT_ID / "runs"
    for found in sorted(runs.glob("*/governance/authorization.json")) if runs.exists() else []:
        auth = BC.BaselineCompletionAuthorization.load(found)
        assert auth.authorizes_c2_baseline_completion is True, found
        assert auth.authorizes_c2_search1 is False, found
        assert auth.hard_cap_usd == 1.1950, found


# --- 10. the formal launch plumbing ----------------------------------------
#
# The seams between the completion session and the generic machinery: the
# runner's argument contract, run ownership of the session record, a readiness
# instance that is this experiment's rather than Search-1's, the recorder's
# registry, an issuer that can produce a loadable artifact, and the bundle
# round-trip. Each was a deterministic refusal or a wrong-record acceptance
# before this round.


def test_the_completion_parser_satisfies_the_runner_argument_contract():
    """`SessionRunner.__init__` reads eighteen attributes off the namespace.

    It refuses a namespace missing any of them -- before provider creation, so
    cheaply, but the invocation is still wasted. Device-canary attempt 1 died at
    $0.0603 on an attribute a hand-written namespace had and the parser did not,
    after the pod was billing.
    """
    from aadistill.infrastructure.session import (
        RUNNER_ARGUMENT_CONTRACT, missing_arguments)

    _, args, _ = _completion_spec()
    assert missing_arguments(args) == [], (
        "the completion parser does not supply every runner argument")
    for name in RUNNER_ARGUMENT_CONTRACT:
        assert hasattr(args, name), name
    #: The accepted operational identities, from their owners.
    from experiments.phase_c2 import baseline_completion as BC
    assert args.gpu == BC.gpu_class(REPO)
    assert args.image == BC.image_name(REPO)
    assert args.max_price == BC.price_per_hour_usd(REPO) == 1.09


def test_the_session_record_path_is_owned_by_the_run():
    """No `--out`. `run_id` owns the run, and the record lives inside it."""
    import autoinit_phase_c2_baseline_launch as L

    with pytest.raises(SystemExit):
        L.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b",
            "--run-id", "attempt5", "--out", "/tmp/elsewhere.json"])

    def out_for(run_id: str) -> str:
        return L.build_parser().parse_args([
            "--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b",
            "--run-id", run_id]).out

    five, six = out_for("attempt5"), out_for("attempt6")
    assert five != six, "two runs must not share a session-record path"
    for run_id, path in (("attempt5", five), ("attempt6", six)):
        assert path == L.session_record_path(run_id)
        assert f"/{L.RUN_EXPERIMENT_ID}/runs/{run_id}/" in path, path
        assert path.endswith("runtime/session.json")


def test_storage_has_exactly_one_owner():
    """The `$0` gate and provider creation must read the same attribute."""
    import inspect

    import autoinit_phase_c2_baseline_launch as L

    _, args, _ = _completion_spec()
    assert args.disk_gb == L.COMPLETION_PROVISION_GIB == 60
    source = inspect.getsource(L.storage_gate)
    assert "disk_gb" in source
    assert "volume_gib" not in source, (
        "the gate reads a flag of its own; it could pass while the pod is "
        "provisioned from a different number")
    assert "--volume-gib" not in L.build_parser().format_help()


# --- readiness: the completion's own, not Search-1's ------------------------


def _completion_readiness(run_id="attempt5", stage_id="1", **overrides) -> dict:
    """A record shaped like the one the recorder would write, for refusal tests."""
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE

    record = {
        "schema": CPE.SCHEMA,
        "record_kind": CPE.LAUNCH_BOUND,
        "verdict": "PASS",
        "problems": [],
        "completion_harness_digest": CPE.harness_digest(REPO),
        "completion_harness_n_files": CPE.harness(REPO)["n_files"],
        "staging_contract_digest": "s" * 64,
        "swept_base_commit": "0" * 40,
        "self_sha256": "r" * 64,
    }
    record.update(overrides)
    return record


def test_the_completion_readiness_contract_is_its_own():
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE
    from experiments.phase_c2 import pod_environment as SPE

    assert CPE.SCHEMA != SPE.SCHEMA, (
        "sharing Search-1's schema would let one experiment's readiness record "
        "satisfy the other's verifier")
    assert CPE.EXPERIMENT_ID == "phase_c2_baseline_completion"

    sweep = CPE.sweep_contract("attempt5", "1")
    assert sweep.experiment_id == CPE.EXPERIMENT_ID
    assert sweep.launcher_module == "autoinit_phase_c2_baseline_launch"
    assert sweep.session_id == "autoinit-phase-c2-baseline-completion"
    assert sweep.harness_n_files_field == "completion_harness_n_files"
    assert sweep.record.harness_field == "completion_harness_digest"

    #: The harness is the LIVE completion closure, not Search-1's.
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2.session import c2_harness_digest
    assert sweep.harness(REPO)["digest"] == BC.executable_digest(REPO)
    assert sweep.harness(REPO)["digest"] != c2_harness_digest(REPO)["digest"]

    #: Run-owned, and there is no phase-level path to fall back to.
    path = CPE.record_path_for("attempt5", "1")
    assert f"/{CPE.EXPERIMENT_ID}/runs/attempt5/" in path
    with pytest.raises(CPE.ReadinessError):
        CPE.record_path_for(None)


def test_a_search_1_readiness_record_cannot_satisfy_the_completion():
    """The schema is the refusal, and it is checked before anything else."""
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE
    from experiments.phase_c2 import pod_environment as SPE

    foreign = _completion_readiness(schema=SPE.SCHEMA)
    ok, reason = CPE.verify_record(
        foreign, REPO, run_id="attempt5", stage_id="1",
        required_kind=CPE.LAUNCH_BOUND)
    assert ok is False
    assert "schema" in reason.lower(), reason


def test_the_generic_recorder_resolves_the_completion_experiment():
    """One registry entry, and everything it needs comes through it."""
    import record_pod_environment as R

    assert "phase_c2_baseline_completion" in R.EXPERIMENTS
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE
    sweep = R.sweep_contract("phase_c2_baseline_completion", "attempt5", "1",
                             CPE.LAUNCH_BOUND)
    assert sweep.experiment_id == "phase_c2_baseline_completion"
    #: It can LOAD the completion launcher and derive the completion staging
    #: contract from that launcher's own manifest -- which is what the sweep
    #: does, and what would otherwise have been derived from Search-1's.
    launcher = R.launcher_module(sweep.launcher_module)
    args = launcher.build_parser().parse_args([
        "--scr", "/tmp/x", "--session-commit", "0" * 40,
        "--bundle", "aad_autoinit_00000000.bundle", "--run-id", "attempt5"])
    from aadistill.runtime.staging_contract import derive_contract
    contract = derive_contract(launcher.spec(args).setup,
                               session_id=sweep.session_id)
    assert contract["digest"]
    from experiments.phase_c2 import baseline_completion as BC
    assert sweep.harness(REPO)["digest"] == BC.executable_digest(REPO)
    assert "/phase_c2_baseline_completion/runs/attempt5/" in sweep.record.record_path


# --- the issuer -------------------------------------------------------------


def _synthetic_grant(**overrides) -> dict:
    """A grant shaped as the contract requires, with identities DERIVED.

    Synthetic and tmp-path only: no real grant is written this round. The
    identities are taken from the live derivation deliberately -- a grant that
    asserted stale ones is a separate refusal with its own test.
    """
    from experiments.phase_c2 import baseline_completion_authorization as BCA

    live = BCA.live_identities(REPO)
    grant = {
        "granted_by": "a synthetic maintainer decision, for tests only",
        "covers": "one baseline completion",
        "explicitly_not_authorized": ["the Search-1 beam", "Search-2"],
        #: The full block the money boundary requires, from the pricing record
        #: the issuer re-derives it from.
        "approved_money": {"expected_usd": live["_expected_usd"],
                           "hard_cap_usd": live["_hard_ceiling_usd"],
                           "price_basis_usd_per_hour": live["_price_per_hour_usd"]},
        "one_use": {"issuances_permitted": 1, "launch_attempts_permitted": 1,
                    "provider_resources_permitted": 2,
                    "one_billing_resource_at_a_time": True},
        "budget_context_at_approval": {"cumulative_spend_usd": 296.8185,
                                       "authorized_cap_usd": 320.0},
        "bound_identities_the_issuer_must_reproduce": {
            k: v for k, v in live.items() if not k.startswith("_")},
    }
    grant.update(overrides)
    return grant


def _issue(monkeypatch, tmp_path, grant=None, run_id="attempt5"):
    from experiments.phase_c2 import baseline_completion_authorization as BCA
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE

    #: The readiness record is STUBBED, not written: creating one at the real
    #: path would create attempt5 as a run, and no completion run is authorized.
    #: `verify_record` is stubbed to PASS as well, because these cases are about
    #: identity and money derivation -- the production verifier is driven for
    #: real by `test_issuance_runs_the_production_readiness_verifier` and
    #: `test_a_tampered_readiness_record_is_refused_by_the_real_verifier`.
    monkeypatch.setattr(CPE, "load_record",
                        lambda *a, **k: _completion_readiness(run_id))
    monkeypatch.setattr(CPE, "verify_record",
                        lambda *a, **k: (True, "stubbed PASS for an identity test"))
    return BCA.build_payload(
        grant=grant if grant is not None else _synthetic_grant(),
        session_commit="1" * 40, granted_utc="2026-09-17T00:00:00+00:00",
        run_id=run_id, stage_id="1", repo_root=REPO,
        grant_path=str(tmp_path / "grant.json"))


def test_the_issuer_builds_an_artifact_the_completion_type_loads(monkeypatch, tmp_path):
    from experiments.phase_c2 import baseline_completion as BC

    payload = _issue(monkeypatch, tmp_path)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload, indent=1))

    auth = BC.BaselineCompletionAuthorization.load(path)
    assert auth.authorizes_c2_baseline_completion is True
    assert auth.authorizes_c2_search1 is False
    assert auth.allows_phase_a is False
    assert auth.allows_recovery_training is False
    assert auth.automatic_followon_start is False
    assert auth.hard_cap_usd == 1.1950
    assert auth.plan_hash == BC.plan_hash(REPO)
    assert auth.harness_source_digest == BC.executable_digest(REPO)
    assert tuple(auth.harness_source_files) == tuple(
        row["path"] for row in BC.current_executable(REPO)["files"])
    assert auth.resource_scope.run_id == "attempt5"


def test_the_issued_artifact_reproduces_every_derived_identity(monkeypatch, tmp_path):
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2 import baseline_completion_authorization as BCA

    payload = _issue(monkeypatch, tmp_path)
    bound, live = payload["bound"], BCA.live_identities(REPO)
    for key in ("completion_harness_digest", "completion_harness_n_files",
                "completion_plan_hash", "completion_pricing_sha256",
                "baseline_spec_hash", "baseline_artifact_digest",
                "frozen_candidates_self_sha256",
                "search1_selection_commitment_sha256"):
        assert bound[key] == live[key], key
    assert bound["expected_usd"] == 0.7212
    assert bound["price_per_hour_usd"] == 1.09
    assert payload["hard_cap_usd"] == 1.1950
    #: And the artifact's own hash covers all of it.
    assert payload["authorization_sha256"] == sha256_json(
        {k: v for k, v in payload.items() if k != "authorization_sha256"})


def test_the_issuer_refuses_a_grant_that_asserts_a_stale_identity(monkeypatch, tmp_path):
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused)

    grant = _synthetic_grant()
    grant["bound_identities_the_issuer_must_reproduce"][
        "completion_harness_digest"] = "0" * 64
    with pytest.raises(CompletionAuthorizationRefused, match="disagree"):
        _issue(monkeypatch, tmp_path, grant)


def test_the_issuer_refuses_an_identity_it_cannot_derive(monkeypatch, tmp_path):
    """A grant may not introduce a binding nobody checks."""
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused)

    grant = _synthetic_grant()
    grant["bound_identities_the_issuer_must_reproduce"]["invented_identity"] = "x"
    with pytest.raises(CompletionAuthorizationRefused, match="cannot derive"):
        _issue(monkeypatch, tmp_path, grant)


def test_the_issuer_refuses_a_search_1_readiness_record(monkeypatch, tmp_path):
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE
    from experiments.phase_c2 import pod_environment as SPE
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused, build_payload)

    monkeypatch.setattr(CPE, "load_record",
                        lambda *a, **k: _completion_readiness(schema=SPE.SCHEMA))
    #: The REAL verifier decides. A foreign schema cannot describe this session.
    with pytest.raises(CompletionAuthorizationRefused, match="does not verify"):
        build_payload(grant=_synthetic_grant(), session_commit="1" * 40,
                      granted_utc="2026-09-17T00:00:00+00:00",
                      run_id="attempt5", stage_id="1", repo_root=REPO)


def test_the_issuer_refuses_a_diagnostic_readiness_record(monkeypatch, tmp_path):
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused, build_payload)

    monkeypatch.setattr(CPE, "load_record",
                        lambda *a, **k: _completion_readiness(record_kind="diagnostic"))
    with pytest.raises(CompletionAuthorizationRefused, match="does not verify"):
        build_payload(grant=_synthetic_grant(), session_commit="1" * 40,
                      granted_utc="2026-09-17T00:00:00+00:00",
                      run_id="attempt5", stage_id="1", repo_root=REPO)


def test_the_completion_config_states_only_what_the_mechanism_needs():
    from experiments.phase_c2.baseline_completion_authorization import load_config

    cfg = load_config(REPO)
    assert cfg["authorizes"] == "nothing"
    #: The project cap the arithmetic needs, and NOT this session's own prices:
    #: those have canonical owners the issuer derives from.
    assert set(cfg["accepted_pricing"]) == {"_why_only_the_cap",
                                            "cumulative_cap_usd"}
    assert cfg["accepted_pricing"]["cumulative_cap_usd"] == 320.0
    #: On the VALUES, not on the document's text: the prose deliberately names
    #: $15.0446 in order to say it is NOT reused, and a substring check would
    #: fail on its own explanation. This is the third time that trap has been
    #: hit in this package; asserting on parsed values is the fix.
    def strip_prose(node):
        if isinstance(node, dict):
            return {k: strip_prose(v) for k, v in node.items()
                    if not k.startswith("_")}
        if isinstance(node, list):
            return [strip_prose(v) for v in node]
        return node
    values = json.dumps(strip_prose(cfg))
    for figure in ("15.0446", "1.1950", "0.7212"):
        assert figure not in values, (
            f"{figure} has a canonical owner; restating it here creates a "
            "second place to edit")
    #: Same rule: the `_no_reviewed_commit` key explains why there is no such
    #: verified identity, so the phrase legitimately appears in prose.
    def without_prose(node):
        if isinstance(node, dict):
            return {k: without_prose(v) for k, v in node.items()
                    if not k.startswith("_")}
        if isinstance(node, list):
            return [without_prose(v) for v in node]
        return node
    assert "reviewed_commit" not in json.dumps(without_prose(cfg))


# --- the bundle round-trip --------------------------------------------------


def test_the_completion_transport_binds_the_completion_authorization_and_closure():
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2 import baseline_completion_bundle as BCT
    from experiments.phase_c2 import bundle as SEARCH1_BUNDLE

    digest, files = BCT.completion_executable_set(REPO)
    assert digest == BC.executable_digest(REPO)
    assert tuple(files) == tuple(row["path"] for row in
                                 BC.current_executable(REPO)["files"])
    #: A DIFFERENT set from Search-1's, which is the point: a bundle verified
    #: against the wrong closure would pass while carrying the wrong code.
    search1_digest, _ = SEARCH1_BUNDLE.c2_executable_set(REPO)
    assert digest != search1_digest

    #: The relay and prefix are shared; the label is not.
    assert BCT.COMPLETION_TRANSPORT.relay_repo == "AlphaAvatar/aadistill-artifacts"
    assert BCT.COMPLETION_TRANSPORT.transfer_prefix == "transfer"
    assert BCT.COMPLETION_TRANSPORT.label != SEARCH1_BUNDLE.C2_TRANSPORT.label
    assert BCT.canonical_bundle_name("a" * 40) == "aad_autoinit_aaaaaaaa.bundle"


def test_the_launcher_gates_include_the_bundle_round_trip():
    _, _, spec = _completion_spec()
    names = [getattr(g, "__name__", type(g).__name__) for g in spec.precheck]
    assert "bundle_staged_gate" in names, (
        "the pod downloads the relay object before any scientific stage; a "
        "launch that never asked whether it exists is C1 attempt 1")
    #: LAST, because it is the only gate that touches the network.
    assert names[-1] == "bundle_staged_gate"
    for required in ("session_commit_and_lineage", "completion_executable_gate",
                     "completion_scope_gate", "frozen_inputs_gate",
                     "frozen_assets_gate", "pricing_and_plan_gate",
                     "storage_gate", "readiness_gate"):
        assert required in names, required


def test_the_bundle_gate_uses_the_completion_transport():
    import inspect

    import autoinit_phase_c2_baseline_launch as L

    source = inspect.getsource(L.bundle_staged_gate)
    assert "BCT." in source, "the gate must use the completion transport"
    assert "BUNDLE." not in source, (
        "that is Search-1's transport: it re-derives the Search-1 closure and "
        "resolves the Search-1 authorization path")


# --- attempt-4's evidence must not have moved ------------------------------


def test_attempt_4_frozen_evidence_bytes_are_unchanged():
    """The accepted science is closed. Nothing this round may touch it."""
    import subprocess

    for path in ("logs/stages/stage-1/phase_c2/runs/attempt4/evidence/"
                 "stage1_selection.json",
                 "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/"
                 "c2_frozen_comparison_inputs.json",
                 "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/"
                 "telemetry.jsonl"):
        out = subprocess.run(["git", "diff", "--name-only",
                              "6c6c699cef62b1190b441bc2e51ea6ca99f5d15e", "--",
                              path], cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert not out.stdout.strip(), f"{path} changed since the reviewed HEAD"

    #: And the values themselves still reproduce their own commitments.
    record = load_record(FROZEN)
    selection = json.loads((RUN / "evidence/stage1_selection.json").read_text())
    assert record["sources"]["selection_commitment_sha256"] == selection[
        "selection_sha256"]
    assert len(record["candidates_in_committed_order"]) == 5


#: The six identities the 2026-09-17 decision froze by value. They are written
#: out here, once, because they are the ONLY mechanical line between a permitted
#: engineering repair and a prohibited change to closed science: the same
#: decision says the completion executable closure is EXPECTED to move, so a
#: moved closure proves nothing either way. Every repair round since has moved
#: it -- the RoPE relocation moved it from `89b9a860...` to `bb3aa38a...` over
#: the same 92 files -- and nothing in the tree checked that the protocol,
#: pricing, baseline recipe, rebuilt-artifact digest, frozen candidate set and
#: Search-1 commitment came through untouched.
FROZEN_SCIENTIFIC_IDENTITIES = {
    "completion_plan_hash":
        "9f566eb6f8d71d570a27c82a945d5e33dd87002338ed8463162e2c0ac595835c",
    "completion_pricing_sha256":
        "dcc64bcf9b3dc9fb0085dab39e31604fd53d7d515ecdfb1d465504e95f212cb8",
    "baseline_spec_hash":
        "3a233a9017b3b8a717ff18fc1aaa171dad84f36adc97920765d181ca98c53612",
    "baseline_artifact_digest":
        "53e30566c5f795f1870d76c1fa6a970ddc507fa5459047f3010ffab8aa890342",
    "frozen_candidates_self_sha256":
        "55f6677067392fd072e31bfe8144d4aa969f547187a89804be3e311e916b9b8c",
    "search1_selection_commitment_sha256":
        "d5c0ce372aba926c54e31a86edffbef95c4110530a394f145bdac30329e3b4bf",
}


def test_the_six_frozen_scientific_identities_are_unchanged():
    """Derived through the PRODUCTION assembler, not recomputed here.

    Recomputing them in the test would check the maintainer's arithmetic
    against my own and leave the path an authorization actually takes unchecked.
    `live_identities` is what the issuer binds, so this asserts the values a
    real chain would carry.
    """
    from experiments.phase_c2.baseline_completion_authorization import (
        live_identities)

    live = live_identities(REPO)
    moved = {key: (expected, live.get(key))
             for key, expected in FROZEN_SCIENTIFIC_IDENTITIES.items()
             if live.get(key) != expected}
    assert not moved, (
        "the 2026-09-17 decision froze these by value and a repair may not "
        f"move them: {json.dumps(moved, indent=1)}")


# --- 11. the CURRENT snapshot must not tell two histories ------------------


SNAPSHOT = REPO / "logs/state/current.json"


def test_the_latest_run_outcome_is_derived_from_that_runs_own_closeout():
    """The field that says WHAT HAPPENED must come from the run it names.

    It was the one hand-maintained field inside a derived block, so the block
    advanced its run id to attempt 4 while carrying attempt 2's outcome: the
    snapshot said "search complete, $6.0785" in one place and "PRE-SCIENCE
    ABORT at setup, $0.0552" in another, about the same run.

    **The newest run legitimately has no closeout yet.** A grant is committed
    into a fresh run before its launch-bound sweep, so "prepared but not
    executed" is a reachable state of the newest run and `_run_outcome` already
    has a branch for it. This read the closeout unconditionally, so it could
    only pass while the newest run happened to have one -- a consumer narrower
    than its producer, which failed the moment a chain was prepared. What is
    asserted is the derivation being FAITHFUL in whichever state the run is in;
    the inherited-outcome defect is caught in both branches, because an
    unexecuted run must not be wearing an earlier attempt's classification or
    cost either.
    """
    snapshot = json.loads(SNAPSHOT.read_text())
    latest = snapshot["latest_run"]
    path = REPO / latest["root"] / "closeout/outcome.json"

    assert latest["run_id"] in latest["root"]
    if path.is_file():
        closeout = json.loads(path.read_text())
        assert closeout["attempt"] == latest["run_id"], (
            "the closeout read is not the named run's")
        #: The outcome must be THIS run's classification and cost.
        assert closeout["classification"].rstrip(". ") in latest["outcome"]
        cost = closeout["budget"]["this_attempt"]
        assert f"${float(cost):.4f}" in latest["outcome"]
    else:
        #: No closeout: the outcome must SAY so rather than borrow one, and the
        #: index must agree the run has not executed.
        assert "no closeout has been written for this run" in latest["outcome"], (
            f"{latest['run_id']} has no closeout, so the outcome must say so "
            f"rather than state {latest['outcome']!r}")
        assert "$" not in latest["outcome"], (
            "a run that never executed is carrying a cost")
        index = json.loads((REPO / "logs/index.json").read_text())
        unrecorded = {e["run_id"] for e in index["unrecorded"]
                      if e["experiment_id"] == latest["experiment_id"]}
        assert latest["run_id"] in unrecorded, (
            "the snapshot says this run wrote no closeout while the index "
            "reports it as recorded")
    #: And specifically not an earlier attempt's.
    assert "0.0552" not in latest["outcome"], (
        "attempt 2's cost is attached to a later run")


def test_the_snapshot_does_not_contradict_itself_about_attempt_4():
    snapshot = json.loads(SNAPSHOT.read_text())
    c2 = snapshot["phase_c"]["c2"]
    latest = snapshot["latest_run"]

    #: One history: the search completed and the comparison did not. Asserted
    #: where that history is OWNED -- the phase status, and attempt 4's own
    #: closeout -- not on `latest_run`, which names whichever run executed most
    #: recently and now legitimately names a baseline-completion attempt.
    #: Content, not the phrase "SEARCH COMPLETE": the snapshot now says
    #: "Search-1 DONE and FROZEN; B->C COMPUTED", which states the same history
    #: and more of it. A guard that only accepts one wording of a fact fails as
    #: soon as the fact is stated better.
    assert re.search(r"search[- ]1 (is )?(done|complete)|search complete",
                     c2["status"], re.I), c2["status"]
    attempt4 = json.loads((REPO / "logs/stages/stage-1/phase_c2/runs/attempt4"
                           "/closeout/outcome.json").read_text())
    assert "SEARCH COMPLETE" in attempt4["classification"].upper()
    for stale in ("PRE-SCIENCE INFRASTRUCTURE ABORT at setup",
                  "nothing measured"):
        assert stale not in latest["outcome"], stale

    #: The accepted ruling is recorded as accepted, not as owed.
    nxt = json.dumps(snapshot["next_starting_point"])
    assert "ACCEPTED" in nxt
    assert "ruling on the not-yet-quantified" not in nxt, (
        "the cross-session ruling is accepted and recorded in the protocol")
    #: And the snapshot's own prepared-launch claim must match the tree.
    #: Asserting a fixed value here made this test a statement about one moment
    #: rather than about consistency; asserting merely that a grant EXISTS made
    #: it wrong in the other direction once chains started being consumed. A
    #: grant is prepared while its chain is open, and a closeout closes it, so
    #: the tree's answer is: a grant whose run carries no closeout.
    import autoinit_phase_c2_baseline_launch as L
    runs = REPO / "logs/stages/stage-1" / L.RUN_EXPERIMENT_ID / "runs"
    open_chains = sorted(
        grant.parents[1].name
        for grant in (runs.glob("*/governance/grant.json") if runs.exists() else ())
        if not (grant.parents[1] / "closeout/outcome.json").is_file())
    assert snapshot["prepared_launch"]["any"] is bool(open_chains), (
        "the snapshot and the tree disagree about whether a completion session "
        f"is prepared; open chains on disk: {open_chains}")
    #: A named run must be one of them, so the snapshot cannot point a reader at
    #: a chain that has already been closed out.
    #:
    #: Compared by RUN ID. `prepared_launch.run` carries the run's path, the way
    #: every other location field in the snapshot does, and this compared it
    #: against `open_chains`, which holds bare run ids -- so the positive branch
    #: could never hold. It had never run: the field was null when this was
    #: written, so the `else` side passed vacuously until a chain was prepared.
    named = snapshot["prepared_launch"].get("run")
    if named:
        run_id = PurePosixPath(str(named)).name
        assert run_id in open_chains, (
            f"prepared_launch names {named!r}, whose run id {run_id!r} is not "
            f"an open chain; open chains on disk: {open_chains}")
    else:
        assert not open_chains


def test_the_renderer_reports_a_missing_closeout_rather_than_inheriting_one(tmp_path):
    """A run with no closeout says so. Nothing is carried over."""
    sys.path.insert(0, str(REPO / "scripts"))
    from consolidate.render_log_navigation import _run_outcome

    assert "no closeout" in _run_outcome(tmp_path, "runs/attempt9", recorded=True)
    assert "no run root" in _run_outcome(tmp_path, None, recorded=True)

    #: And a real closeout is read through, classification and cost together.
    outcome = _run_outcome(
        REPO, "logs/stages/stage-1/phase_c2/runs/attempt4", recorded=True)
    assert "SEARCH COMPLETE" in outcome and "$6.0785" in outcome


# --- 12. the six enforcement repairs ---------------------------------------
#
# Each of these was a partial check or an unenforced default: the kind of gap
# that passes every test written against the happy path and lets a wrong
# artifact through on the one invocation that matters.


def _tampered(record: dict, **changes) -> dict:
    out = dict(record)
    out.update(changes)
    return out


def test_issuance_runs_the_production_readiness_verifier(monkeypatch, tmp_path):
    """Not three of eight checks by hand.

    The assembler used to accept a record on schema, kind and verdict alone,
    which passes a record whose self-hash is wrong, whose harness digest
    describes other code, whose staged view is another session's, or whose swept
    base cannot be an ancestor of the issuance HEAD.
    """
    import inspect

    from experiments.phase_c2 import baseline_completion_authorization as BCA
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE

    source = inspect.getsource(BCA._readiness)
    assert "CPE.verify_record(" in source, (
        "the issuer must call the production verifier")
    for binding in ("required_kind", "session_commit", "authorization_path",
                    "staging_contract_digest"):
        assert binding in source, binding

    #: And it must actually refuse what the verifier refuses. The verifier is
    #: driven here rather than stubbed: a synthetic record with a fake self hash
    #: cannot verify, whatever else it says.
    calls = {}

    def spy(record, repo_root=".", **kwargs):
        calls.update(kwargs)
        return False, "self_sha256 does not match the record"

    monkeypatch.setattr(CPE, "load_record",
                        lambda *a, **k: _completion_readiness())
    monkeypatch.setattr(CPE, "verify_record", spy)
    #: `build_payload` directly, not through `_issue`: that helper stubs the
    #: verifier to pass, which would silently replace this spy and make the
    #: assertion below vacuous.
    with pytest.raises(BCA.CompletionAuthorizationRefused,
                       match="does not verify"):
        BCA.build_payload(grant=_synthetic_grant(), session_commit="1" * 40,
                          granted_utc="2026-09-17T00:00:00+00:00",
                          run_id="attempt5", stage_id="1", repo_root=REPO)
    assert calls["required_kind"] == CPE.LAUNCH_BOUND
    assert calls["session_commit"] == "1" * 40
    assert calls["authorization_path"].endswith("governance/authorization.json")
    assert calls["staging_contract_digest"], (
        "the LIVE staging contract must be passed, not omitted")


def test_the_issuer_derives_the_staging_contract_from_the_real_session_spec():
    """From the completion launcher's own manifest, under its own session id."""
    from aadistill.runtime.staging_contract import derive_contract
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2 import baseline_completion_authorization as BCA

    _, args, spec = _completion_spec()
    expected = derive_contract(spec.setup, session_id=BC.SESSION_ID)["digest"]
    assert BCA._staging_contract_digest("attempt5") == expected


def test_a_tampered_readiness_record_is_refused_by_the_real_verifier():
    """Self hash, harness digest and staging contract, each on its own."""
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE

    good = _completion_readiness()
    for changes, why in (
            ({"self_sha256": "0" * 64}, "a fake self hash"),
            ({"completion_harness_digest": "0" * 64}, "another closure"),
            ({"staging_contract_digest": "0" * 64}, "another staged view"),
            ({"record_kind": "diagnostic"}, "a diagnostic sweep"),
            ({"verdict": "FAIL"}, "a failing sweep")):
        ok, reason = CPE.verify_record(
            _tampered(good, **changes), REPO, run_id="attempt5", stage_id="1",
            required_kind=CPE.LAUNCH_BOUND,
            staging_contract_digest=good["staging_contract_digest"])
        assert ok is False, f"{why} was accepted: {reason}"


def test_the_identity_block_must_be_complete(monkeypatch, tmp_path):
    """Absence of a disagreement is not agreement."""
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused)

    grant = _synthetic_grant()
    block = grant["bound_identities_the_issuer_must_reproduce"]
    dropped = "baseline_spec_hash"
    block.pop(dropped)
    with pytest.raises(CompletionAuthorizationRefused, match="does not state"):
        _issue(monkeypatch, tmp_path, grant)

    empty = _synthetic_grant()
    empty["bound_identities_the_issuer_must_reproduce"] = {}
    with pytest.raises(CompletionAuthorizationRefused, match="absent or empty"):
        _issue(monkeypatch, tmp_path, empty)

    predictive = _synthetic_grant()
    predictive["bound_identities_the_issuer_must_reproduce"][
        "reviewed_commit"] = "a" * 40
    with pytest.raises(CompletionAuthorizationRefused, match="reviewed_commit"):
        _issue(monkeypatch, tmp_path, predictive)


def test_the_identity_block_must_name_all_eight(monkeypatch, tmp_path):
    from experiments.phase_c2 import baseline_completion_authorization as BCA

    derivable = {k for k in BCA.live_identities(REPO) if not k.startswith("_")}
    assert derivable == {
        "completion_harness_digest", "completion_harness_n_files",
        "completion_plan_hash", "completion_pricing_sha256",
        "baseline_spec_hash", "baseline_artifact_digest",
        "frozen_candidates_self_sha256", "search1_selection_commitment_sha256"}
    payload = _issue(monkeypatch, tmp_path)
    assert derivable <= set(payload["bound"])


def test_approved_money_is_a_real_boundary(monkeypatch, tmp_path):
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused, check_approved_money)

    assert check_approved_money(_synthetic_grant(), REPO) == {
        "expected_usd": 0.7212, "hard_cap_usd": 1.1950,
        "price_basis_usd_per_hour": 1.09}

    for money, why in (
            (None, "missing"),
            ({}, "empty"),
            ({"expected_usd": 0.7212, "hard_cap_usd": 15.0446,
              "price_basis_usd_per_hour": 1.09}, "Search-1's ceiling"),
            ({"expected_usd": 7.1787, "hard_cap_usd": 1.1950,
              "price_basis_usd_per_hour": 1.09}, "Search-1's expected cost"),
            ({"expected_usd": 0.7212, "hard_cap_usd": 1.1950,
              "price_basis_usd_per_hour": 1.49}, "a higher rate basis"),
            ({"hard_cap_usd": 1.1950,
              "price_basis_usd_per_hour": 1.09}, "a missing figure")):
        grant = _synthetic_grant()
        if money is None:
            grant.pop("approved_money")
        else:
            grant["approved_money"] = money
        with pytest.raises(CompletionAuthorizationRefused):
            _issue(monkeypatch, tmp_path, grant)


def test_the_cap_arithmetic_still_gates_the_ceiling(monkeypatch, tmp_path):
    """296.8185 + 1.1950 <= 320.0, and a grant that does not fit is refused."""
    from experiments.phase_c2.baseline_completion_authorization import (
        CompletionAuthorizationRefused)

    payload = _issue(monkeypatch, tmp_path)
    headroom = payload["budget_headroom"]
    assert headroom["remaining_usd"] == round(320.0 - 296.8185, 4)

    over = _synthetic_grant()
    over["budget_context_at_approval"]["cumulative_spend_usd"] = 319.9
    with pytest.raises(CompletionAuthorizationRefused, match="exceeds"):
        _issue(monkeypatch, tmp_path, over)


# --- the launcher's enforcement --------------------------------------------


class _Ctx:
    """The fields these gates read off a live SessionContext."""

    def __init__(self, args, auth=None):
        self.args = args
        self.auth = auth
        self.evidence = {}


def _scoped_auth(*, run_id="attempt5", draws=2, one_billing=True,
                 completion=True, search1=False):
    from experiments.phase_c2.session import C2ResourceScope

    class _Auth:
        authorizes_c2_baseline_completion = completion
        authorizes_c2_search1 = search1
        resource_scope = C2ResourceScope(
            run_id=run_id, issuances_permitted=1, launch_attempts_permitted=1,
            provider_resources_permitted=draws,
            one_billing_resource_at_a_time=one_billing)
    return _Auth()


def test_the_scope_gate_uses_the_scopes_own_methods():
    import inspect

    import autoinit_phase_c2_baseline_launch as L

    source = inspect.getsource(L.completion_scope_gate)
    assert "permits_run(" in source and "permits_draws(" in source, (
        "the gate must ask the scope, not re-implement it")

    _, args, _ = _completion_spec()
    args.host_draws = 2
    ok, reason = L.completion_scope_gate(_Ctx(args, _scoped_auth(draws=2)))
    assert ok, reason

    #: More draws than the grant permits.
    args.host_draws = 8
    ok, reason = L.completion_scope_gate(_Ctx(args, _scoped_auth(draws=2)))
    assert ok is False and "host-draws" in reason

    #: Another run's authorization.
    args.host_draws = 2
    ok, reason = L.completion_scope_gate(
        _Ctx(args, _scoped_auth(run_id="attempt9")))
    assert ok is False and "attempt9" in reason

    #: And an artifact that could authorize a beam.
    ok, reason = L.completion_scope_gate(
        _Ctx(args, _scoped_auth(search1=True)))
    assert ok is False and "Search-1" in reason


def test_the_frozen_runtime_is_enforced_not_merely_defaulted():
    import autoinit_phase_c2_baseline_launch as L
    from experiments.phase_c2 import baseline_completion as BC

    _, args, _ = _completion_spec()
    ok, reason = L.frozen_runtime_gate(_Ctx(args))
    assert ok, reason
    assert args.gpu == "NVIDIA L40S"
    assert args.image == "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"

    for attr, value, needle in (("gpu", "NVIDIA H100 PCIe", "--gpu"),
                                ("image", "runpod/pytorch:2.0.0", "--image"),
                                ("max_price", 1.49, "max-price")):
        _, bad, _ = _completion_spec()
        setattr(bad, attr, value)
        ok, reason = L.frozen_runtime_gate(_Ctx(bad))
        assert ok is False, f"{attr}={value!r} was accepted"
        assert needle in reason, reason

    #: A LOWER max price is safe: it can only refuse a launch.
    _, cheap, _ = _completion_spec()
    cheap.max_price = 0.99
    ok, reason = L.frozen_runtime_gate(_Ctx(cheap))
    assert ok, reason
    assert BC.price_per_hour_usd(REPO) == 1.09


def test_the_frozen_runtime_gate_is_in_the_pre_provider_set():
    _, _, spec = _completion_spec()
    names = [getattr(g, "__name__", type(g).__name__) for g in spec.precheck]
    assert "frozen_runtime_gate" in names
    assert names.index("frozen_runtime_gate") < names.index("bundle_staged_gate")


# --- the durability boundary ------------------------------------------------


def test_the_b_measurement_survives_a_post_measurement_failure(tmp_path, monkeypatch):
    """The repair that matters: a comparison failure must not discard B.

    The fake evaluation SUCCEEDS and the comparison is then made to fail
    immediately afterwards. The session's final evidence must still carry the
    complete B measurement, and the measurement count must still be one.
    """
    import autoinit_phase_c2_baseline_driver as D
    from experiments.phase_c2 import comparison as C

    def explode(**kwargs):
        raise RuntimeError("serialisation failed after the measurement")

    monkeypatch.setattr(C, "build", explode)

    with pytest.raises(RuntimeError, match="serialisation failed"):
        _run_stage_b(tmp_path, monkeypatch,
                     baseline_values=_far_from_every_candidate())

    #: The evidence document on disk, written by `save()` inside the boundary.
    evidence = json.loads(
        (tmp_path / "audit" / "c2_baseline_completion_evidence.json").read_text())
    measurement = evidence["baseline_measurement"]
    assert measurement["status"] == "MEASURED"
    assert measurement["state_eval_measurements_performed"] == 1
    assert measurement["expected_artifact_digest"] == B.B_ARTIFACT_DIGEST
    assert measurement["actual_artifact_digest"] == B.B_ARTIFACT_DIGEST
    #: COMPLETE, so the comparison can be rebuilt from it at $0.
    for key in ("artifact_digest", "suite_id", "suite_hash", "reference",
                "values", "positions", "detail", "measured_utc"):
        assert key in measurement["evaluation"], key
    for objective in PARETO_V1.objectives:
        assert objective.key in measurement["evaluation"]["values"]
    assert measurement["suite"]["hash"] == json.loads(
        FROZEN.read_text())["suite"]["hash"]
    assert measurement["reference_strategy"] == "recompute"
    assert measurement["teacher"]["revision"]
    assert measurement["bound_search1_identity"]["search_run_id"] == (
        "autoinit.v1.phase_c2.search1")
    assert measurement["frozen_candidate_inputs"].endswith(
        "c2_frozen_comparison_inputs.json")
    assert "no_candidate_was_remeasured" in measurement


def test_the_durable_block_is_written_before_any_post_processing():
    """Ordering, in the source: measure, persist, THEN compare."""
    import inspect

    import autoinit_phase_c2_baseline_driver as D

    source = inspect.getsource(D.BaselineCompletionDriver.rebuild_measure_compare)
    measured = source.index("attach_evaluation(")
    persisted = source.index("persist_baseline_measurement(")
    disclosure = source.index("numerical_sensitivity_disclosure(")
    built = source.index("C.build(")
    assert measured < persisted < disclosure < built, (
        "the measurement must be durable before any post-processing")


def test_exactly_one_state_eval_is_performed(tmp_path, monkeypatch):
    driver, calls = _run_stage_b(tmp_path, monkeypatch,
                                 baseline_values=_far_from_every_candidate())
    assert len(calls["evaluate"]) == 1
    evidence = json.loads(
        (tmp_path / "audit" / "c2_baseline_completion_evidence.json").read_text())
    assert evidence["baseline_measurement"][
        "state_eval_measurements_performed"] == 1
    assert driver.ev["stages"]["rebuild_measure_compare"]["detail"][
        "candidates_remeasured"] == 0


# --- 13. the RoPE guard, moved to where the artifact exists ----------------


def test_the_completion_does_not_declare_rope_ok():
    """The shared step cannot do its job for a session that stages no checkpoint.

    It globs `artifacts/stage1/*/checkpoint/config.json` and exits 1 on an empty
    match. C1 attempt 2 paid $0.1013 there and this session's attempt 6 paid
    $0.0412 -- both correct refusals about a staged checkpoint neither had.
    """
    _, _, spec = _completion_spec()
    assert "ROPE_OK" not in spec.setup.setup_markers
    #: Everything else it does need is still declared.
    for required in ("ENV_READY", "REPO_READY", "TRAIN_ENV", "ASSETS_READY",
                     "TEACHER_READY", "TESTS_OK", "AUTHORIZATION_OK"):
        assert required in spec.setup.setup_markers, required


def test_undeclaring_rope_ok_did_not_remove_the_guard():
    """Undeclaring the step is only legitimate because the check moved."""
    import autoinit_phase_c2_baseline_driver as D

    assert hasattr(D.BaselineCompletionDriver, "assert_rope_base_of_rebuilt_b")
    import inspect
    source = inspect.getsource(
        D.BaselineCompletionDriver.rebuild_measure_compare)
    checked = source.index("assert_rope_base_of_rebuilt_b(")
    measured = source.index("attach_evaluation(")
    assert checked < measured, (
        "the RoPE base must be checked BEFORE the measurement it would "
        "invalidate")


class _Config:
    """Only the fields the two helpers read."""

    model_type = "qwen3"

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_the_rope_guard_refuses_a_disagreeing_base(tmp_path, monkeypatch):
    """A 500x disagreement is the transformers 4.x/5.x field-layout bug.

    Driven through the driver's real method, with only `AutoConfig` and the
    runtime resolution stubbed -- the comparison and the refusal are the real
    ones.
    """
    import autoinit_phase_c2_baseline_driver as D

    driver = D.BaselineCompletionDriver.__new__(D.BaselineCompletionDriver)

    def run(stored, runtime):
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                            staticmethod(lambda *a, **k: _Config(rope_theta=stored)))
        monkeypatch.setattr(
            "aadistill.models.student.assert_rope_from_config",
            lambda config, path="": runtime)
        return driver.assert_rope_base_of_rebuilt_b(tmp_path)

    #: Agreeing: the guard passes and reports both readings.
    evidence = run(5_000_000.0, 5_000_000.0)
    assert evidence["stored"] == evidence["runtime"] == 5_000_000.0
    assert evidence["checked_on"] == str(tmp_path)

    #: The 500x bug: 10,000 where 5,000,000 is recorded.
    with pytest.raises(D.CompletionError, match="500x|positional basis"):
        run(5_000_000.0, 10_000.0)

    #: And a config recording no base at all.
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        staticmethod(lambda *a, **k: _Config()))
    monkeypatch.setattr("aadistill.models.student.assert_rope_from_config",
                        lambda config, path="": float("nan"))
    with pytest.raises(D.CompletionError, match="no RoPE base"):
        driver.assert_rope_base_of_rebuilt_b(tmp_path)


def test_the_rope_reading_is_carried_into_the_durable_measurement(tmp_path, monkeypatch):
    """Whatever the guard read is part of the evidence B is defended by."""
    _run_stage_b(tmp_path, monkeypatch, baseline_values=_far_from_every_candidate())
    evidence = json.loads(
        (tmp_path / "audit" / "c2_baseline_completion_evidence.json").read_text())
    rope = evidence["baseline_measurement"]["rope_base"]
    assert rope["stored"] == rope["runtime"]

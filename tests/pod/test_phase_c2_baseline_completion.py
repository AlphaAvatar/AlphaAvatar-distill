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

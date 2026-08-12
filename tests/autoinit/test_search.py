"""End-to-end beam search on real (tiny) checkpoints, and family-agnosticism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.manifest import build_manifest, verify_manifest  # noqa: E402
from aadistill.autoinit.metrics import StateEvaluation, StateEvaluator  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1  # noqa: E402
from aadistill.autoinit.search import BeamSearch, SearchConfig  # noqa: E402
from aadistill.autoinit.state import StateValidity  # noqa: E402
from conftest import TARGET_GEOMETRY, TEACHER_GEOMETRY, build_tiny_model  # noqa: E402

ADAPTER = get_adapter("qwen3")


def make_search(tmp_path, teacher, target_spec, eval_suite, suite_items, profiles,
                beam_width=2, run_id="dryrun", **overrides):
    evaluator = StateEvaluator(eval_suite, suite_items)
    evaluator.prime_reference(teacher)

    from conftest import make_items
    items = make_items()

    config = SearchConfig(
        run_id=run_id, target_spec=target_spec, beam_width=beam_width, seed=4242,
        workdir=tmp_path / run_id, profiles=tuple(profiles), policy=PARETO_V1,
        suite=eval_suite, **overrides)
    return BeamSearch(
        adapter=ADAPTER, config=config,
        root_teacher_id="tiny-teacher", root_teacher_sha256="a" * 64,
        root_loader=lambda: teacher,
        calibration_loader=lambda profile: items,
        measurer=lambda model, sha: evaluator.evaluate(model, sha)), config


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    """One real search, reused by the assertions below (it takes a few seconds)."""
    from conftest import make_items, make_profile
    from aadistill.autoinit.metrics import StateEvalSuite, SuiteItem

    tmp_path = tmp_path_factory.mktemp("dryrun")
    teacher = build_tiny_model(TEACHER_GEOMETRY)
    target_spec = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    suite = StateEvalSuite(
        suite_id="test.state_eval", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=4)
    suite_items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                             domain=i["domain"], subtype=i["subtype"], tags=i["tags"])
                   for i in make_items(seed=202)]
    search, config = make_search(tmp_path, teacher, target_spec, suite, suite_items,
                                 [make_profile("balanced")])
    result = search.run()
    return {"search": search, "result": result, "teacher": teacher,
            "target_spec": target_spec, "suite": suite, "tmp_path": tmp_path,
            "config": config}


# --- the invariants ---------------------------------------------------------


def test_the_search_produces_complete_leaves(dry_run):
    result = dry_run["result"]
    assert result.complete_leaves, "the search found no complete path"
    assert len(result.states) > len(result.complete_leaves)


def test_every_leaf_matches_the_target_architecture_exactly(dry_run):
    target = dry_run["target_spec"]
    for leaf in dry_run["result"].complete_leaves:
        assert leaf.spec.matches(target), leaf.path_label
        assert leaf.remaining_differences() == frozenset()
        assert leaf.num_parameters == ADAPTER.param_count(target)
        leaf.require_recovery_admissible()


def test_intermediate_states_exist_and_are_not_leaves(dry_run):
    intermediates = [s for s in dry_run["result"].states.values()
                     if s.depth > 0 and not s.is_complete_leaf()]
    assert intermediates, "a search with no intermediate states is not a search"
    for state in intermediates:
        assert state.remaining_differences()
        with pytest.raises(Exception, match="intermediate search state"):
            state.require_recovery_admissible()


def test_operator_order_is_searched_not_fixed(dry_run):
    orders = {tuple(leaf.applied_kinds) for leaf in dry_run["result"].complete_leaves}
    decomposed = {o for o in orders if len(o) > 1}
    assert decomposed, "no decomposed path completed"
    # Distinct orders reached the same target architecture from the same teacher.
    assert all(len(set(o)) == len(o) for o in decomposed), "a kind repeated on a path"


def test_the_incumbent_composite_is_a_leaf_of_the_same_search(dry_run):
    composite = [leaf for leaf in dry_run["result"].complete_leaves
                 if leaf.impl_ids == ("composite.stage1_sandwich_v0",)]
    assert len(composite) >= 1, "the incumbent recipe must be inside the search space"
    assert composite[0].depth == 1


def test_every_measured_state_binds_its_metrics_to_its_own_weights(dry_run):
    measured = [s for s in dry_run["result"].states.values() if s.evaluation]
    assert len(measured) > 5
    for state in measured:
        assert state.evaluation.checkpoint_sha256 == state.checkpoint_sha256
        assert state.evaluation.suite_hash == dry_run["suite"].suite_hash
        assert state.evaluation.reference == "root_teacher"
    # No two distinct states share a weight hash, so no metric can have been
    # reused across states even by accident.
    hashes = [s.checkpoint_sha256 for s in measured]
    assert len(set(hashes)) == len(hashes)


def test_every_state_was_reloaded_and_validated_before_being_measured(dry_run):
    for state in dry_run["result"].states.values():
        if state.evaluation is None:
            continue
        checks = state.notes.get("validation")
        if checks is None:      # restored from the journal on resume
            continue
        assert checks["finite"]
        assert checks["num_parameters"] == ADAPTER.param_count(state.spec)
        assert checks["reload_max_logit_diff"] == 0.0


def test_pruned_states_keep_their_hash_metrics_and_reason(dry_run):
    pruned = [s for s in dry_run["result"].states.values()
              if s.validity is StateValidity.PRUNED]
    assert pruned, "a beam of 2 must have pruned something"
    for state in pruned:
        assert state.prune_reason and state.prune_reason.startswith("pruned:")
        assert state.checkpoint_sha256
        assert state.evaluation is not None
        # Weights released, evidence retained.
        assert state.checkpoint_path is None
        assert state.notes["weights_released"]["sha256"] == state.checkpoint_sha256


def test_the_manifest_records_the_whole_search(dry_run):
    result = dry_run["result"]
    manifest = build_manifest(
        result, adapter=ADAPTER, profiles=list(dry_run["config"].profiles),
        policy=PARETO_V1,
        teacher={"model_id": "tiny-teacher", "sha256": "a" * 64},
        cost={"usd": 0.0, "note": "CPU dry run"})
    report = verify_manifest(manifest)
    assert report["manifest_hash_matches"]
    assert report["leaves_match_target"]

    assert manifest["beam_ranking_policy"]["policy_hash"] == PARETO_V1.policy_hash
    assert manifest["adapter"]["adapter_version"] == ADAPTER.adapter_version
    assert manifest["operator_registry"]["implementations"]["depth.positional_v0"]
    assert manifest["state_eval_suite"]["suite_hash"] == dry_run["suite"].suite_hash
    assert manifest["target_architecture"]["num_parameters"] == ADAPTER.param_count(
        dry_run["target_spec"])
    # Pruned states are in the manifest with their reasons, not omitted.
    assert manifest["state_index"]["pruned"]
    pruned_records = [s for s in manifest["states"] if s["validity"] == "pruned"]
    assert all(r["prune_reason"] for r in pruned_records)
    # And every leaf carries its checkpoint hash and metrics.
    assert all(leaf["checkpoint_sha256"] and leaf["metrics"]
               for leaf in manifest["leaf_set"])
    assert json.dumps(manifest, default=str)


def test_top_n_refuses_intermediate_states(dry_run):
    result = dry_run["result"]
    top = result.top_n(PARETO_V1, n=2)
    assert len(top.selected) <= 2
    for state in top.selected:
        assert state.is_complete_leaf()

    intermediate = next(s for s in result.states.values()
                        if s.depth > 0 and not s.is_complete_leaf())
    result.leaves.append(intermediate)
    try:
        with pytest.raises(Exception, match="intermediate search state"):
            result.top_n(PARETO_V1, n=2)
    finally:
        result.leaves.remove(intermediate)


# --- resume -----------------------------------------------------------------


def test_resume_is_deterministic_and_recomputes_nothing(tmp_path, teacher,
                                                        target_spec, eval_suite,
                                                        suite_items, profile):
    search_a, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                              [profile], run_id="resume")
    first = search_a.run()

    search_b, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                              [profile], run_id="resume")
    second = search_b.run()

    assert sorted(first.states) == sorted(second.states)
    assert [leaf.state_id for leaf in first.complete_leaves] == \
           [leaf.state_id for leaf in second.complete_leaves]
    for level_a, level_b in zip(first.levels, second.levels):
        assert level_a.generated == level_b.generated
        if level_a.ranking:
            assert level_a.ranking.selected_ids == level_b.ranking.selected_ids

    # The second run restored rather than recomputed: everything whose weights
    # survived the first run came back from the journal.
    assert second.resumed, "resume recomputed every state"
    for sid in second.resumed:
        assert second.states[sid].notes.get("resumed") is True
    assert first.config.config_hash == second.config.config_hash


def test_a_different_target_does_not_resume_from_another_targets_journal(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile):
    search_a, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                              [profile], run_id="shared")
    search_a.run()
    other_target = target_spec.replace(intermediate_size=32)
    search_b, _ = make_search(tmp_path, teacher, other_target, eval_suite, suite_items,
                              [profile], run_id="shared")
    result = search_b.run()
    assert not result.resumed


# --- family-agnosticism -----------------------------------------------------


def test_a_new_family_and_a_new_operator_kind_need_no_core_edit(tmp_path):
    """A non-transformers MoE family with fields the core has never seen."""
    import inspect

    import aadistill.autoinit.search as search_module
    from aadistill.autoinit.arch import register_adapter, unregister_adapter
    from aadistill.autoinit.operators.base import unregister_implementation
    from fake_family import TOY_FAMILY, ToyAdapter, ToyConfig, register_all

    # Scan the executable core, not the prose: the module docstring names DEPTH
    # as an example of what the engine does *not* branch on.
    core_source = inspect.getsource(search_module).replace(
        search_module.__doc__ or "", "")
    for forbidden in ("qwen3", "hidden_size", "num_hidden_layers", "self_attn",
                      "DEPTH", "n_experts"):
        assert forbidden not in core_source, (
            f"the search core mentions {forbidden!r}; it is not family-agnostic")

    adapter = register_adapter(ToyAdapter())
    register_all()
    try:
        teacher_config = ToyConfig(d_model=8, n_experts=6, expert_width=10,
                                   vocab_size=32)
        teacher = adapter.build_model(teacher_config, torch.float32, seed=5)
        root_spec = adapter.spec_of(teacher)
        target = root_spec.replace(n_experts=3, expert_width=5)

        def measurer(model, sha):
            with torch.no_grad():
                logits = model(torch.arange(8).reshape(1, 8)).logits
            return StateEvaluation(
                checkpoint_sha256=sha, suite_id="toy@v1", suite_hash="toy",
                reference="root_teacher", positions=8,
                values={"state.teacher_kl.equal_domain_mean": float(logits.abs().mean()),
                        "state.critical_token_kl": float(logits.std()),
                        "state.nll.general": float(-logits.log_softmax(-1).mean())})

        config = SearchConfig(
            run_id="toy", target_spec=target, beam_width=2, seed=7,
            workdir=tmp_path / "toy",
            profiles=(_toy_profile(),), policy=PARETO_V1, suite=_toy_suite(),
            allowed_impls=("moe.expert_set_topk_v1", "moe.expert_width_topk_v1"))
        search = BeamSearch(
            adapter=adapter, config=config, root_teacher_id="toy",
            root_teacher_sha256="b" * 64, root_loader=lambda: teacher,
            calibration_loader=lambda p: [{"input_ids": torch.arange(8).reshape(1, 8)}],
            measurer=measurer)
        result = search.run()

        assert result.complete_leaves
        for leaf in result.complete_leaves:
            assert leaf.spec.matches(target)
            assert leaf.spec.family == TOY_FAMILY
            leaf.require_recovery_admissible()
        # Both operator orders were explored, and neither kind is one of the four.
        orders = {leaf.applied_kinds for leaf in result.complete_leaves}
        assert orders == {("MOE_EXPERT_SET", "MOE_EXPERT_WIDTH"),
                          ("MOE_EXPERT_WIDTH", "MOE_EXPERT_SET")}
    finally:
        unregister_adapter(TOY_FAMILY)
        for impl_id in ("moe.expert_set_topk_v1", "moe.expert_width_topk_v1",
                        "attention.toy_mla_v1"):
            unregister_implementation(impl_id)


def test_an_mla_attention_implementation_is_not_offered_to_a_gqa_adapter(
        teacher_spec, target_spec):
    from aadistill.autoinit.operators.base import (
        applicable_implementations, unregister_implementation)
    from fake_family import register_all

    register_all()
    try:
        offered = {i.impl_id for i, _ in applicable_implementations(
            ADAPTER, teacher_spec, target_spec)}
        assert "attention.weight_proxy_v0" in offered
        assert "attention.toy_mla_v1" not in offered, (
            "an MLA implementation was offered to a GQA adapter; dispatch is not "
            "capability-based")
    finally:
        for impl_id in ("moe.expert_set_topk_v1", "moe.expert_width_topk_v1",
                        "attention.toy_mla_v1"):
            unregister_implementation(impl_id)


def _toy_profile():
    from aadistill.autoinit.calibration import CalibrationProfile, CalibrationSource
    return CalibrationProfile(
        profile_id="toy.calib", version=1, description="toy",
        sources=(CalibrationSource("toy", "local", "general", 1),),
        domain_weights={"general": 1.0}, token_budget=8, sample_rule="fixed", seed=1)


def _toy_suite():
    from aadistill.autoinit.metrics import StateEvalSuite
    return StateEvalSuite(suite_id="toy", version=1, domains=("general",),
                          subtypes={"general": ("text",)}, critical_tags=())

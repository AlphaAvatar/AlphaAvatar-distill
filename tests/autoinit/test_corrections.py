"""The pre-GPU correction pass, each item pinned by a test.

Every test here exists because the tiny dry run would not have exposed the defect
at 4B: a checkpoint that never shards, a profile set of one, a suite small enough
that caching 34 GiB of reference logits looks free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.artifact import (  # noqa: E402
    ArtifactError,
    CheckpointIdentity,
    ShardRecord,
    identify_checkpoint,
    verify_frozen_single_file_hash,
)
from aadistill.autoinit.calibration import (  # noqa: E402
    NO_CALIBRATION,
    consumes_calibration,
    profile_for,
)
from aadistill.autoinit.metrics import (  # noqa: E402
    MeasurementError,
    ReferenceStrategy,
    StateEvalSuite,
    StateEvaluator,
    reference_cache_bytes,
)
from aadistill.autoinit.operators.base import get_implementation  # noqa: E402
from aadistill.autoinit.ranking import SCHEDULE_V1, BeamSchedule  # noqa: E402
from aadistill.autoinit.stats import (  # noqa: E402
    DEFAULT_STATS_SPEC,
    StatsCache,
    stats_cache_key,
)
from conftest import TARGET_GEOMETRY, TEACHER_GEOMETRY, build_tiny_model, make_items

ADAPTER = get_adapter("qwen3")


# --- 1. calibration branch identity -----------------------------------------


def test_no_calibration_operators_use_the_canonical_sentinel():
    positional = get_implementation("depth.positional_v0")
    attention = get_implementation("attention.weight_proxy_v0")
    causal = get_implementation("depth.causal_kl_greedy_v1")

    assert not consumes_calibration(positional)
    assert not consumes_calibration(attention)
    assert consumes_calibration(causal)

    from conftest import make_profile
    a, b = make_profile("a", seed=1), make_profile("b", seed=2)
    assert a.profile_hash != b.profile_hash
    # Whatever profile is offered, a NONE-calibration operator sees the sentinel,
    # so its state identity cannot depend on the mixture.
    assert profile_for(positional, a) is NO_CALIBRATION
    assert profile_for(positional, b) is NO_CALIBRATION
    assert profile_for(causal, a) is a


def test_the_sentinel_is_a_single_object_that_never_resolves():
    from aadistill.autoinit.calibration import CalibrationError

    assert NO_CALIBRATION.is_no_calibration
    assert NO_CALIBRATION.qualified_id == "calib.none@v1"
    with pytest.raises(CalibrationError, match="not built"):
        NO_CALIBRATION.resolve(REPO)


def test_two_profiles_do_not_duplicate_the_weight_proxy_expansion(
        tmp_path, teacher, target_spec, eval_suite, suite_items, two_profiles):
    """The whole point: P profiles must not multiply a weight-only operator."""
    from test_search import make_search

    search, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                            list(two_profiles), run_id="branch")
    root = search.root_state()
    expansions = list(search._candidate_expansions(root))
    by_impl: dict[str, list[str]] = {}
    for impl, profile in expansions:
        by_impl.setdefault(impl.impl_id, []).append(profile.qualified_id)

    assert by_impl["depth.positional_v0"] == ["calib.none@v1"]
    assert by_impl["attention.weight_proxy_v0"] == ["calib.none@v1"]
    assert len(by_impl["depth.causal_kl_greedy_v1"]) == 2
    assert len(by_impl["width.global_pca_v0"]) == 2
    assert len(by_impl["ffn.activation_importance_v0"]) == 2
    # 2 (no-calibration) + 4x2 (calibrated, incl. the composite) = 10, not 6x2.
    assert len(expansions) == 10


# --- 2. shard-safe artifact identity ----------------------------------------


def test_a_forced_shard_split_is_identified_correctly(tmp_path):
    """Sharding is exercised without building a multi-GiB checkpoint."""
    model = build_tiny_model(TEACHER_GEOMETRY)
    spec = ADAPTER.spec_of(model)

    single = tmp_path / "single"
    ADAPTER.save(model, str(single))
    one = identify_checkpoint(single, adapter=ADAPTER, spec=spec,
                              num_parameters=ADAPTER.param_count(spec))
    assert not one.is_sharded and len(one.shards) == 1
    assert one.single_shard_sha256 is not None
    assert one.index_sha256 is None

    sharded = tmp_path / "sharded"
    ADAPTER.save(model, str(sharded), max_shard_size="16KB")
    many = identify_checkpoint(sharded, adapter=ADAPTER, spec=spec,
                               num_parameters=ADAPTER.param_count(spec))
    assert many.is_sharded, "max_shard_size did not split the checkpoint"
    assert len(many.shards) > 1
    assert many.index_sha256 is not None, "a sharded checkpoint must have an index"
    assert many.single_shard_sha256 is None
    # Deterministic order, and every shard hashed individually.
    names = [s.filename for s in many.shards]
    assert names == sorted(names)
    assert len({s.sha256 for s in many.shards}) == len(many.shards)
    assert many.total_bytes > 0

    # Same weights, different layout -> different artifact identity, because a
    # runtime reads them differently and the index is part of what it reads.
    assert one.artifact_digest != many.artifact_digest
    # And both reload to the same model.
    torch.manual_seed(0)
    ids = torch.randint(0, TEACHER_GEOMETRY["vocab_size"], (1, 8))
    with torch.no_grad():
        a = ADAPTER.load(str(single))(ids).logits
        b = ADAPTER.load(str(sharded))(ids).logits
    assert torch.equal(a, b)


def test_a_frozen_single_file_hash_stays_checkable(tmp_path):
    model = build_tiny_model(TARGET_GEOMETRY)
    spec = ADAPTER.spec_of(model)
    path = tmp_path / "ckpt"
    ADAPTER.save(model, str(path))
    identity = identify_checkpoint(path, adapter=ADAPTER, spec=spec,
                                   num_parameters=ADAPTER.param_count(spec))
    frozen = identity.single_shard_sha256
    assert verify_frozen_single_file_hash(identity, frozen)
    assert not verify_frozen_single_file_hash(identity, "0" * 64)

    # A sharded rebuild is not a hash mismatch; it is a different layout, and the
    # check says so rather than crying corruption.
    sharded_path = tmp_path / "sharded"
    ADAPTER.save(model, str(sharded_path), max_shard_size="16KB")
    sharded = identify_checkpoint(sharded_path, adapter=ADAPTER, spec=spec,
                                  num_parameters=ADAPTER.param_count(spec))
    assert not verify_frozen_single_file_hash(sharded, frozen)


def test_an_empty_checkpoint_directory_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(ArtifactError, match="no weight shards"):
        identify_checkpoint(empty, adapter=ADAPTER,
                            spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
                            num_parameters=1)


def test_shard_order_is_enforced_not_assumed():
    with pytest.raises(ArtifactError, match="sorted order"):
        CheckpointIdentity(
            path="/tmp/x",
            shards=(ShardRecord("b.safetensors", "h2", 1),
                    ShardRecord("a.safetensors", "h1", 1)),
            config_sha256="c", arch_signature="a", num_parameters=1)


def test_the_search_measures_a_sharded_checkpoint_end_to_end(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile):
    """The full cycle with sharding on, at a size a CPU test can afford."""
    from test_search import make_search

    # A depth-only target: one level, and the child keeps the teacher's full
    # width, so it is the largest artifact the search produces — the same shape
    # that makes a real depth-only 4B intermediate cross the shard threshold.
    depth_only_target = ADAPTER.spec_of(teacher).replace(
        num_hidden_layers=target_spec["num_hidden_layers"])
    search, _ = make_search(tmp_path, teacher, depth_only_target, eval_suite,
                            suite_items, [profile], run_id="shardrun",
                            max_shard_size="8KB")
    result = search.run()
    produced = [s for s in result.states.values() if s.artifact]
    assert produced
    assert any(s.artifact.is_sharded for s in produced), "nothing sharded"
    for state in produced:
        assert state.evaluation.artifact_digest == state.artifact_digest
        assert state.notes["validation"]["n_shards"] >= 1
        if state.artifact.is_sharded:
            assert state.checkpoint_sha256 is None
            assert state.artifact.index_sha256


# --- 3. control vs re-executed composite ------------------------------------


def test_a_control_is_injected_by_artifact_not_regenerated(tmp_path, target_spec):
    from aadistill.autoinit.state import StateError, make_control_state

    model = build_tiny_model(TARGET_GEOMETRY)
    path = tmp_path / "canonical"
    ADAPTER.save(model, str(path))
    spec = ADAPTER.spec_of(model)
    identity = identify_checkpoint(path, adapter=ADAPTER, spec=spec,
                                   num_parameters=ADAPTER.param_count(spec))

    control = make_control_state(
        control_id="qwen3_0p6b_init_v0", artifact=identity, spec=spec,
        target_spec=target_spec, num_parameters=ADAPTER.param_count(spec),
        root_teacher_id="teacher", root_teacher_sha256="r",
        description="the retained canonical Stage-1 init",
        expected_single_file_sha256=identity.single_shard_sha256)

    assert control.provenance == "retained_canonical"
    assert control.steps == (), "a control was not produced by this search"
    assert control.is_complete_leaf()
    assert control.path_label == "ROOT"

    with pytest.raises(StateError, match="not the retained checkpoint"):
        make_control_state(
            control_id="wrong", artifact=identity, spec=spec, target_spec=target_spec,
            num_parameters=1, root_teacher_id="t", root_teacher_sha256="r",
            description="", expected_single_file_sha256="0" * 64)


def test_a_control_must_already_be_at_the_target(tmp_path, teacher_spec, target_spec):
    from aadistill.autoinit.state import StateError, make_control_state

    model = build_tiny_model(TEACHER_GEOMETRY)
    path = tmp_path / "wrongsize"
    ADAPTER.save(model, str(path))
    spec = ADAPTER.spec_of(model)
    identity = identify_checkpoint(path, adapter=ADAPTER, spec=spec,
                                   num_parameters=ADAPTER.param_count(spec))
    with pytest.raises(StateError, match="must already be at the target"):
        make_control_state(control_id="c", artifact=identity, spec=spec,
                           target_spec=target_spec, num_parameters=1,
                           root_teacher_id="t", root_teacher_sha256="r",
                           description="")


# --- 4. NLL semantics -------------------------------------------------------


def test_general_nll_comes_from_the_general_domain_alone(teacher, eval_suite,
                                                         suite_items):
    evaluator = StateEvaluator(eval_suite, suite_items)
    evaluator.prime_reference(teacher)
    student = build_tiny_model(TARGET_GEOMETRY, seed=3)
    evaluation = evaluator.evaluate(student, "digest")

    values = evaluation.values
    assert "state.nll.general" in values
    assert "state.nll.math" in values
    assert "state.nll.pooled_all_domains" in values
    per_domain = evaluation.detail["per_domain_nll"]
    assert values["state.nll.general"] == pytest.approx(per_domain["general"])
    # The general number is genuinely not the pooled one.
    assert values["state.nll.general"] != pytest.approx(
        values["state.nll.pooled_all_domains"])
    # And the pooled figure sits between the domains it pools.
    lo, hi = min(per_domain.values()), max(per_domain.values())
    assert lo <= values["state.nll.pooled_all_domains"] <= hi


def test_a_suite_with_no_general_domain_emits_no_general_nll(teacher):
    suite = StateEvalSuite(
        suite_id="mathonly", version=1, domains=("math",),
        subtypes={"math": ("arith",)}, critical_tags=(), general_domain=None)
    from aadistill.autoinit.metrics import SuiteItem

    items = [SuiteItem(item_id=i["item_id"], input_ids=i["input_ids"],
                       domain="math", subtype="arith", tags={})
             for i in make_items() if i["domain"] == "math"]
    evaluator = StateEvaluator(suite, items)
    evaluator.prime_reference(teacher)
    values = evaluator.evaluate(build_tiny_model(TARGET_GEOMETRY, seed=3),
                                "digest").values
    assert "state.nll.general" not in values, (
        "a pooled average over non-general text must not be named general NLL")
    assert "state.nll.pooled_all_domains" in values


def test_nll_is_not_a_required_ranking_metric(eval_suite):
    required = eval_suite.required_metrics()
    assert not any(k.startswith("state.nll") for k in required)
    assert "state.teacher_kl.worst_domain" in required


# --- 5. beam schedule -------------------------------------------------------


def test_the_v1_schedule_retains_everything_at_level_zero():
    assert SCHEDULE_V1.warmup_levels == 1
    assert SCHEDULE_V1.width == 6
    assert SCHEDULE_V1.width_at(0) is None, "level 0 must not prune"
    assert SCHEDULE_V1.width_at(1) == 6
    assert SCHEDULE_V1.width_at(3) == 6
    assert len(SCHEDULE_V1.schedule_hash) == 64
    moved = BeamSchedule("beam.delayed_prune", 1, "", warmup_levels=0, width=6)
    assert moved.schedule_hash != SCHEDULE_V1.schedule_hash


def test_a_warmup_level_prunes_nothing_and_says_so(teacher_spec, target_spec):
    from aadistill.autoinit.ranking import PARETO_V1
    from test_ranking import make_state

    states = [make_state(teacher_spec, target_spec, n, kl, 0.5, 3.0)
              for n, kl in [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4)]]
    warm = PARETO_V1.rank(states, beam_width=None)
    assert len(warm.selected) == 4
    assert warm.pruned == ()
    assert all("warmup" in d["reason"] for d in warm.decisions)

    pruned = PARETO_V1.rank(states, beam_width=2)
    assert len(pruned.selected) == 2


def test_epsilon_dominance_protects_a_practically_equivalent_state(teacher_spec,
                                                                   target_spec):
    """A 1e-9 edge is a floating-point accident, not a reason to kill a path."""
    from aadistill.autoinit.ranking import PARETO_V1, BeamRankingPolicy
    from test_ranking import make_state

    a = make_state(teacher_spec, target_spec, "a", 0.500000000, 0.5, 3.0)
    b = make_state(teacher_spec, target_spec, "b", 0.500000001, 0.5, 3.0)

    strict = BeamRankingPolicy(
        policy_id="strict", version=1, description="",
        objectives=PARETO_V1.objectives, tie_break=PARETO_V1.tie_break,
        guardrails=PARETO_V1.guardrails, epsilon={}, diversity_key="none")
    assert len(strict.rank([a, b], beam_width=2).fronts) == 2, "strict Pareto separates them"
    # The shipped policy treats the difference as noise, so both stay in front 0.
    assert len(PARETO_V1.rank([a, b], beam_width=2).fronts) == 1


def test_diversity_keeps_distinct_lineages_in_the_beam(teacher_spec, target_spec):
    """A beam must not fill with variants of one hypothesis."""
    from aadistill.autoinit.ranking import PARETO_V1
    from test_ranking import make_state

    # Three refinements of lineage A, all better than the single B; a plain
    # tie-break beam of 2 would take two As and extinguish B.
    states = [
        make_state(teacher_spec, target_spec, "a1", 0.10, 0.10, 1.0,
                   parent_impl="depth.positional_v0"),
        make_state(teacher_spec, target_spec, "a2", 0.11, 0.11, 1.0,
                   parent_impl="depth.positional_v0"),
        make_state(teacher_spec, target_spec, "b1", 0.30, 0.30, 1.0,
                   parent_impl="width.global_pca_v0"),
    ]
    result = PARETO_V1.rank(states, beam_width=2)
    assert len(set(result.lineages_kept)) == 2, (
        "the beam kept two variants of one lineage and dropped the other")
    assert set(result.selected_ids) == {states[0].state_id, states[2].state_id}


# --- 6/7. statistics cache boundary -----------------------------------------


def test_the_stats_key_requires_the_parent_artifact_digest():
    with pytest.raises(ValueError, match="parent artifact digest"):
        stats_cache_key(parent_artifact_digest="", profile_hash="p",
                        stats_spec=DEFAULT_STATS_SPEC, adapter_version="v",
                        numerical_config={})


def test_the_stats_key_separates_parents_profiles_specs_and_numerics():
    base = dict(parent_artifact_digest="parentA", profile_hash="p1",
                stats_spec=DEFAULT_STATS_SPEC, adapter_version="qwen3.dense_v1",
                numerical_config={"device": "cpu", "accumulation": "float64"})
    key = stats_cache_key(**base)
    assert key == stats_cache_key(**base)
    assert key != stats_cache_key(**{**base, "parent_artifact_digest": "parentB"})
    assert key != stats_cache_key(**{**base, "profile_hash": "p2"})
    assert key != stats_cache_key(**{**base, "adapter_version": "qwen3.dense_v2"})
    assert key != stats_cache_key(
        **{**base, "numerical_config": {"device": "cuda", "accumulation": "float64"}})


def test_width_and_ffn_share_one_pass_on_the_same_parent_and_never_across_parents(
        tmp_path, teacher, teacher_spec, target_spec, calibration_items, profile):
    """The reuse boundary, exercised through the real operators."""
    from aadistill.autoinit.operators.base import OperatorContext

    cache = StatsCache()
    key_a = stats_cache_key(parent_artifact_digest="parentA",
                            profile_hash=profile.profile_hash,
                            stats_spec=DEFAULT_STATS_SPEC, adapter_version="v",
                            numerical_config={})
    key_b = stats_cache_key(parent_artifact_digest="parentB",
                            profile_hash=profile.profile_hash,
                            stats_spec=DEFAULT_STATS_SPEC, adapter_version="v",
                            numerical_config={})

    def run(impl_id, key):
        return get_implementation(impl_id).execute(OperatorContext(
            adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
            target_spec=target_spec, profile=profile,
            calibration_items=calibration_items, seed=1,
            stats_cache=cache, stats_cache_key=key))

    run("width.global_pca_v0", key_a)
    assert cache.misses == 1 and cache.hits == 0
    run("ffn.activation_importance_v0", key_a)     # same parent -> shared
    assert cache.hits == 1 and cache.passes_saved == 1

    run("width.global_pca_v0", key_b)              # different parent -> never shared
    assert cache.misses == 2

    # And the cache holds one entry, so a 4B-class run does not accumulate
    # gigabytes of stale statistics.
    assert cache.report()["resident_entries"] == 1


def test_an_operator_without_a_cache_still_collects(teacher, teacher_spec,
                                                    target_spec, calibration_items,
                                                    profile):
    from aadistill.autoinit.operators.base import OperatorContext

    outcome = get_implementation("ffn.activation_importance_v0").execute(
        OperatorContext(adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
                        target_spec=target_spec, profile=profile,
                        calibration_items=calibration_items, seed=1))
    assert outcome.model is not teacher


def test_the_root_state_does_not_share_statistics(tmp_path, teacher, target_spec,
                                                  eval_suite, suite_items, profile):
    """The root has no artifact digest of this search's making, so it pays per pass."""
    from test_search import make_search

    search, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                            [profile], run_id="rootstats")
    assert search._stats_key(search.root_state(), profile) is None


# --- 8. reference logit memory ----------------------------------------------


def test_caching_the_reference_is_refused_when_it_would_not_fit(teacher, eval_suite,
                                                                suite_items):
    """The real suite at Qwen's vocabulary is 33.8 GiB. Refuse before allocating."""
    qwen_vocab = 151_936
    # A realistic suite: 67 items of ~892 positions.
    assert reference_cache_bytes(suite_items, qwen_vocab) > 0
    with pytest.raises(MeasurementError, match="RECOMPUTE"):
        StateEvaluator(eval_suite, suite_items,
                       reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
                       vocab_size=qwen_vocab, cache_budget_bytes=1024)


def test_caching_without_a_vocab_size_is_refused(eval_suite, suite_items):
    with pytest.raises(MeasurementError, match="before allocating"):
        StateEvaluator(eval_suite, suite_items,
                       reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY)


def test_recompute_is_the_default_and_agrees_with_caching(teacher, eval_suite,
                                                          suite_items):
    """Recomputing must not change a single number; it only changes the memory."""
    student = build_tiny_model(TARGET_GEOMETRY, seed=3)

    recompute = StateEvaluator(eval_suite, suite_items)
    assert recompute.reference_strategy is ReferenceStrategy.RECOMPUTE
    recompute.prime_reference(teacher)
    a = recompute.evaluate(student, "digest")

    cached = StateEvaluator(eval_suite, suite_items,
                            reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
                            vocab_size=TEACHER_GEOMETRY["vocab_size"])
    cached.prime_reference(teacher)
    b = cached.evaluate(student, "digest")

    assert a.values == b.values
    assert a.detail["reference_strategy"] == "recompute"
    assert b.detail["reference_strategy"] == "cache_in_memory"


def test_the_reference_cache_estimate_is_the_number_in_the_proposal():
    """59,763 positions x 151,936 vocab x 4 bytes = 33.8 GiB."""
    class _Item:
        def __init__(self, positions):
            self.input_ids = torch.zeros(1, positions + 1, dtype=torch.long)

    total = reference_cache_bytes([_Item(59_763)], 151_936)
    assert total / 2**30 == pytest.approx(33.8, rel=0.01)


# --- resume fails closed on a changed suite ---------------------------------


def test_resume_refuses_a_journal_measured_under_a_different_suite(
        tmp_path, teacher, target_spec, suite_items, profile):
    """State identity is the path and does not include the eval suite.

    Without this check the beam would rank this run's states on last run's
    questions, silently, and the manifest would report them as measured.
    """
    from aadistill.autoinit.metrics import StateEvalSuite
    from test_search import make_search

    suite_a = StateEvalSuite(
        suite_id="suite", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like",), general_domain="general")
    suite_b = StateEvalSuite(
        suite_id="suite", version=2, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like",), general_domain="general")
    assert suite_a.suite_hash != suite_b.suite_hash

    first, _ = make_search(tmp_path, teacher, target_spec, suite_a, suite_items,
                           [profile], run_id="suitechange")
    first.run()

    same, _ = make_search(tmp_path, teacher, target_spec, suite_a, suite_items,
                          [profile], run_id="suitechange")
    assert same.run().resumed, "an unchanged suite should resume"

    changed, _ = make_search(tmp_path, teacher, target_spec, suite_b, suite_items,
                             [profile], run_id="suitechange")
    result = changed.run()
    assert result.resumed == [], (
        "a journal measured under a different suite must not be adopted")
    for state in result.states.values():
        if state.evaluation:
            assert state.evaluation.suite_hash == suite_b.suite_hash


def test_resume_refuses_a_journal_whose_artifact_no_longer_matches(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile):
    """Weights edited on disk must not be adopted along with their old metrics."""
    from test_search import make_search

    first, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                           [profile], run_id="tamper")
    result = first.run()
    victim = next(s for s in result.states.values()
                  if s.checkpoint_path and s.evaluation)
    shard = Path(victim.checkpoint_path) / victim.artifact.shards[0].filename
    shard.write_bytes(shard.read_bytes() + b"tampered")

    again, _ = make_search(tmp_path, teacher, target_spec, eval_suite, suite_items,
                           [profile], run_id="tamper")
    second = again.run()
    assert victim.state_id not in second.resumed, (
        "a checkpoint whose bytes changed was resumed with its stale metrics")

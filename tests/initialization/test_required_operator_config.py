"""A config an operator cannot run without is refused BEFORE anything expensive, and a declared one reaches it.

`width.global_pca_v0` and `composite.stage1_sandwich_v0` read a stream weighting that the core no longer holds
(`OperatorImplementation.required_config`). Each layer refuses an undeclared one before the work it would follow:

* the operators themselves -- before the statistics pass (and a malformed one is refused there too);
* a search -- when its root is known and before the first forward: an allowed implementation the beam can expand on
  some path toward the target (its own `applicable`, walked over the planned specs at $0) and that is not given its
  config by `SearchConfig.operator_configs`; one that never applies to the target asks nothing;
* a fixed path -- before its root is loaded, for every step that would run (a spec is still constructed without one:
  deriving a historical step's config hash needs no execution).

The declaration itself enters the operator's hashed config, where the search and the fixed path put it.
"""
from __future__ import annotations

import pytest

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
from aadistill.initialization.calibration.profiles import register_profile, unregister_profile
from aadistill.initialization.operators.base import OperatorContext, OperatorError, get_implementation, missing_config
from aadistill.initialization.planning.fixed_path import (FixedPathError, FixedPathSpec, FixedPathStep,
                                                         materialize_fixed_path, step_operator_config)
from aadistill.initialization.planning.metrics import StateEvaluator
from aadistill.initialization.planning.ranking import PARETO_V1, BeamSchedule
from aadistill.initialization.planning.search import BeamSearch, SearchConfig, SearchError
from aadistill.initialization.specs.arch import ArchSpec
from support.recipes import STREAM_WEIGHTING_V0, WEIGHTED_OPERATOR_CONFIGS
from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile

WIDTH, COMPOSITE = "width.global_pca_v0", "composite.stage1_sandwich_v0"


def test_the_two_projections_declare_what_they_cannot_run_without():
    for impl_id in (WIDTH, COMPOSITE):
        impl = get_implementation(impl_id)
        assert impl.required_config == {"stream_weighting"}
        assert missing_config(impl, None) == ["stream_weighting"]
        assert missing_config(impl, {"stream_weighting": STREAM_WEIGHTING_V0}) == []
        assert "required_config" not in impl.signature()          #: no frozen signature hash moves
    assert missing_config(get_implementation("ffn.activation_importance_v0"), None) == []


@pytest.mark.parametrize("impl_id,module", [
    (WIDTH, "aadistill.initialization.operators.width.residual.global_pca"),
    (COMPOSITE, "aadistill.initialization.operators.composite.stage1_sandwich")])
@pytest.mark.parametrize("weighting", [None, {"endpoint_weights": {"first": 9.0, "interior": 1.0, "last": -1.0},
                                              "source": "a malformed declaration"}])
def test_an_undeclared_or_malformed_weighting_is_refused_before_the_statistics_pass(
        impl_id, module, weighting, teacher, teacher_spec, target_spec, calibration_items, profile, monkeypatch):
    import importlib
    mod = importlib.import_module(module)

    def collected(*_a, **_k):
        raise AssertionError("the statistics pass ran before the declaration was checked")
    monkeypatch.setattr(mod, "collect_activation_stats", collected)
    ctx = OperatorContext(adapter=QWEN3_ADAPTER, model=teacher, parent_spec=teacher_spec, target_spec=target_spec,
                          profile=profile, calibration_items=calibration_items, seed=1,
                          config={} if weighting is None else {"stream_weighting": weighting})
    with pytest.raises((OperatorError, ValueError)) as e:
        get_implementation(impl_id).apply(ctx)
    assert "stream_weighting" in str(e.value) or "weight" in str(e.value)


def _search(tmp_path, teacher, target_spec, eval_suite, suite_items, profile, **cfg):
    evaluator = StateEvaluator(eval_suite, suite_items)
    measured = []

    def measure(model, sha):
        measured.append(sha)
        return evaluator.evaluate(model, sha)

    def calibration(_profile):
        raise AssertionError("calibration was read before the declaration was checked")
    config = SearchConfig(run_id="required_config", target_spec=target_spec,
                          schedule=BeamSchedule("t.beam", 1, "t", warmup_levels=0, width=2), seed=1,
                          workdir=tmp_path / "s", profiles=(profile,), policy=PARETO_V1, suite=eval_suite, **cfg)
    return BeamSearch(adapter=QWEN3_ADAPTER, config=config, root_teacher_id="tiny", root_teacher_sha256="a" * 64,
                      root_loader=lambda: teacher, calibration_loader=calibration, measurer=measure), measured


def test_a_search_that_could_apply_width_without_its_weighting_stops_before_the_first_forward(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile):
    assert "hidden_size" in QWEN3_ADAPTER.spec_of(teacher).diff(target_spec)
    search, measured = _search(tmp_path, teacher, target_spec, eval_suite, suite_items, profile)
    with pytest.raises(SearchError, match="width.global_pca_v0.*stream_weighting"):
        search.run()
    assert measured == []
    #: declared, the same search passes the check (its root state is built)
    ok, _ = _search(tmp_path / "ok", teacher, target_spec, eval_suite, suite_items, profile,
                    operator_configs=WEIGHTED_OPERATOR_CONFIGS)
    assert ok.root_state().remaining_differences()
    #: an implementation that cannot act on the target needs nothing declared
    no_width = target_spec.replace(hidden_size=QWEN3_ADAPTER.spec_of(teacher)["hidden_size"])
    s, _ = _search(tmp_path / "nw", teacher, no_width, eval_suite, suite_items, profile,
                   allowed_impls=("width.global_pca_v0", "ffn.activation_importance_v0", "depth.positional_v0",
                                  "attention.weight_proxy_v0"))
    assert s.root_state() is not None


@pytest.mark.parametrize("compressed", [("num_hidden_layers",), ("intermediate_size",), ("num_attention_heads",),
                                        ("num_hidden_layers", "intermediate_size", "num_attention_heads")])
def test_the_default_operator_set_asks_nothing_of_a_target_that_keeps_the_width(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile, compressed):
    """Every registered implementation allowed (the default), a target that compresses depth, the FFN or the heads
    but not the width: neither projection can be expanded on any path toward it -- the width is already at target,
    and the all-at-once recipe applies only when every structural field differs -- so nothing is asked of the
    search. The root state is built."""
    root = QWEN3_ADAPTER.spec_of(teacher)
    keep = {f: root[f] for f in ("hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads")
            if f not in compressed}
    target = target_spec.replace(**keep)
    assert target.diff(root) == set(compressed)
    s, measured = _search(tmp_path, teacher, target, eval_suite, suite_items, profile)
    assert s.config.allowed_impls is None
    assert s.root_state().remaining_differences()
    assert not {WIDTH, COMPOSITE} & s._expandable_implementations(root)
    assert measured == []


def test_the_all_at_once_recipe_is_asked_for_its_weighting_where_it_applies(
        tmp_path, teacher, target_spec, eval_suite, suite_items, profile):
    """The composite recipe, allowed without the width projection, toward a target differing in every structural
    field (where it applies, from the root): refused before the first forward, naming it."""
    root = QWEN3_ADAPTER.spec_of(teacher)
    assert root.diff(target_spec) >= get_implementation(COMPOSITE).modifies
    s, measured = _search(tmp_path, teacher, target_spec, eval_suite, suite_items, profile,
                          allowed_impls=(COMPOSITE, "ffn.activation_importance_v0", "depth.positional_v0",
                                         "attention.weight_proxy_v0"))
    with pytest.raises(SearchError, match="composite.stage1_sandwich_v0.*stream_weighting") as e:
        s.run()
    assert WIDTH not in str(e.value) and measured == []


def test_a_fixed_path_step_without_its_weighting_stops_before_the_root_is_loaded(tmp_path):
    p = make_profile("balanced")
    register_profile(p, replace=True)
    try:
        target = ArchSpec.of("qwen3", dict(TEACHER_GEOMETRY, hidden_size=TEACHER_GEOMETRY["hidden_size"] // 2))
        step = FixedPathStep(WIDTH, p.qualified_id)
        spec = FixedPathSpec(path_id="t.width", family="qwen3", target_spec=target, steps=(step,),
                             root_repo_id="t/teacher", root_revision="deadbeef")   #: constructed: hashes derive

        def loaded():
            raise AssertionError("the root was loaded before the declaration was checked")
        with pytest.raises(FixedPathError, match="stream_weighting"):
            materialize_fixed_path(spec, adapter=QWEN3_ADAPTER, root_loader=loaded, workdir=tmp_path,
                                   calibration_items={p.qualified_id: make_items()})
        good = FixedPathSpec(path_id="t.width", family="qwen3", target_spec=target,
                             steps=(FixedPathStep(WIDTH, p.qualified_id, config={"stream_weighting": STREAM_WEIGHTING_V0}),),
                             root_repo_id="t/teacher", root_revision="deadbeef")
        out = materialize_fixed_path(good, adapter=QWEN3_ADAPTER, root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
                                     workdir=tmp_path / "ok", calibration_items={p.qualified_id: make_items()})
        assert len(out) == 1 and out[0].kind == "RESIDUAL_WIDTH"
        #: the declaration is what the step is applied with: in its hashed config
        assert step_operator_config(good.steps[0], 4)["stream_weighting"] == STREAM_WEIGHTING_V0
    finally:
        unregister_profile(p.profile_id)

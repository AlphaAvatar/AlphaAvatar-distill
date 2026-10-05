"""A non-full distribution support, through a REAL `BeamSearch`, end to end.

The gap this closes was invisible to everything that existed. The D-series Top-K
protocol was implemented in the operator, in the state evaluator, in the protocol
identity and in a driver that injected it by hand — and `SearchConfig` had no
field for it, so the only path a formal search could actually take fell back to
`FULL_VOCAB_V1`. A paid 40-minute measurement timed the intended operator path
faithfully while that path was unreachable through the beam.

So this test runs the thing nothing else ran: a search configured with a coarsened
partition, on a tiny real Qwen3 model, and asserts the support reaches every place
it has to reach and nowhere it must not.

What it pins:

* the support arrives in `OperatorContext` and in the hashed operator config, and
  the operator's own artifacts say which partition they reduced over;
* an otherwise identical full-vocabulary run has a DIFFERENT `config_hash` and
  different state ids — a coarsened estimand is a different hypothesis;
* a full-vocabulary run's identity is BYTE-IDENTICAL to what it was before the
  field existed, because the field is absent at its default;
* the operators and the beam metric cannot reduce over different partitions, in
  both directions, which is the hazard a run-level field exists to prevent.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from support.toy import (  # noqa: E402
    TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1, SCHEDULE_V1,
)
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig, SearchError,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, reference_topk_tail,
)
from aadistill.initialization.specs.arch import (  # noqa: E402
    ArchSpec, adapter_for_config,
)
from aadistill.initialization.specs.materialization import (  # noqa: E402
    NumericalEnvironment,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    ReferenceStrategy, StateEvalSuite, SuiteItem,
)

CPU_NUMERICS = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
B3_SORTED = ExecutionConfig(micro_batch_size=3,
                            calibration_batch_packing="length_sorted_v1")
#: DEPTH-ONLY, because the space is DEPTH-only: `depth.causal_kl_greedy_v1` is the
#: operator whose objective IS a KL over the vocabulary, so it is the one a
#: partition can move, and a target that also differs in width or FFN would be
#: unreachable from a one-kind space -- which the search refuses, correctly,
#: before it loads anything.
DEPTH_ONLY_TARGET = {**TEACHER_GEOMETRY, "num_hidden_layers": 4}
#: A GENUINE COARSENING on this geometry: 8 of a 128-entry vocabulary, so the tail
#: bucket holds real mass and the partition is not the identity. No campaign's K
#: appears here or in `src/aadistill`; a support is a parameter.
TOY_TOPK = reference_topk_tail(8)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    register_builtin_adapters()
    register_builtin_operators()
    yield


@pytest.fixture(scope="module")
def pieces():
    teacher = build_tiny_model(TEACHER_GEOMETRY)
    teacher.config.use_cache = False
    raw = make_items()
    calib = prepare_calibration_items(raw, profile_id="test.balanced@v1")
    #: `tags` is a MAPPING of tag name to a boolean mask over prediction
    #: positions, which is the form `normalized_prediction_tags` reads. Passing
    #: the names as a tuple makes `dict(...)` raise on an 8-character string, and
    #: a tag that silently selected nothing would make a target-aware objective
    #: quietly full-sequence.
    suite_items = tuple(
        SuiteItem(item_id=i["item_id"], domain=i["domain"], subtype=i["subtype"],
                  input_ids=i["input_ids"], tags=i.get("tags") or {})
        for i in calib)
    suite = StateEvalSuite(
        suite_id="test.support", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(suite_items),
        general_domain="general")
    return teacher, calib, suite, suite_items


def _evaluator(teacher, suite, suite_items, support):
    ev = StateEvaluator(
        suite, suite_items, device="cpu",
        reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
        vocab_size=TEACHER_GEOMETRY["vocab_size"],
        distribution_support=support)
    ev.prime_reference(teacher)
    return ev


def _search(workdir, pieces, *, support, declare_protocol=True,
            evaluator_support=None, run_id="support-regression"):
    """A real search over the DEPTH operator, under one distribution support.

    `evaluator_support` defaults to the search's own, which is the only coherent
    configuration; the tests that drive them apart pass it explicitly.
    """
    teacher, calib, suite, suite_items = pieces
    evaluator = _evaluator(teacher, suite, suite_items,
                           evaluator_support if evaluator_support is not None
                           else support)
    config = SearchConfig(
        run_id=run_id, target_spec=ArchSpec.of("qwen3", DEPTH_ONLY_TARGET),
        schedule=SCHEDULE_V1, seed=7, workdir=workdir,
        profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
        distribution_support=support,
        measurement_protocol_id=(evaluator.measurement_protocol_id
                                 if declare_protocol else None),
        #: DEPTH alone: it is the operator whose objective IS a KL over the
        #: vocabulary, so it is the one a partition can move. A one-kind space
        #: also keeps the toy search to a handful of expansions.
        allowed_impls=("depth.causal_kl_greedy_v1",),
        max_depth=1, device="cpu")
    search = BeamSearch(
        adapter=adapter_for_config(teacher.config), config=config,
        root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
        root_loader=lambda: teacher,
        calibration_loader=lambda profile: calib,
        measurer=lambda model, digest: evaluator.evaluate(model, digest),
        execution=B3_SORTED, numerics=CPU_NUMERICS)
    return search, config, evaluator


def _spy_on(search, seen: dict):
    """Record the context every expansion hands its operator.

    On the INSTANCE the registry hands out, because an instance attribute wins
    over the class and a class-level patch does not: a singleton can arrive
    carrying a leaked instance `execute` from an earlier module, which shadows the
    class and makes a class patch invisible. The spy is installed for the life of
    the test; the module-boundary fixture in the root `conftest.py` removes it.
    """
    from aadistill.initialization.operators.base import get_implementation

    for impl_id in (search.config.allowed_impls or ()):
        impl = get_implementation(impl_id)
        real = type(impl).execute

        def spy(ctx, _real=real, _impl=impl):
            seen["config"] = dict(ctx.config)
            seen["declared"] = dict(ctx.config).get("distribution_support")
            seen["support"] = ctx.distribution_support
            return _real(_impl, ctx)

        vars(impl)["execute"] = spy
    return seen


class TestTheSupportReachesTheOperatorThroughTheBeam:
    """The path that did not exist: `SearchConfig` -> `_expand_one` -> operator."""

    def test_the_context_and_the_config_both_carry_it(self, pieces):
        """Both sides, because `execute` refuses a disagreement and because a
        declaration that never reaches the object is a lie in the state id."""
        seen = {}
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=TOY_TOPK)
        impl = search._allowed_impl_ids()
        assert "depth.causal_kl_greedy_v1" in (impl or ())

        #: PATCHED ON THE INSTANCE the registry actually hands out, not on the
        #: class. An instance attribute wins over the class either way, and a
        #: class-level patch does NOT: a registry singleton can arrive carrying a
        #: leaked instance `execute` from an earlier module -- monkeypatch's undo
        #: writes an inherited attribute back as an instance one -- and then a
        #: class patch is silently ignored. That cost a debugging round, and the
        #: root conftest now strips such leaks at the module boundary; this is
        #: simply the form that cannot be shadowed.
        seen.update(_spy_on(search, seen))
        search.run()

        assert seen, "no expansion ran, so nothing was observed"
        assert seen["support"].support_id == TOY_TOPK.support_id
        assert seen["support"].top_k == 8
        #: The DECLARATION, which is what `config_hash` sees.
        assert seen["declared"] == TOY_TOPK.as_dict()

    def test_the_depth_artifacts_name_the_partition_they_reduced_over(self, pieces):
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=TOY_TOPK)
        result = search.run()
        states = list(result.states.values())
        assert states, "the search produced no state"
        caches = []
        for state in states:
            for step in state.steps:
                cache = (step.artifacts or {}).get("reference_cache")
                if cache:
                    caches.append(cache)
        assert caches, "no DEPTH expansion recorded a reference cache"
        for cache in caches:
            assert cache["distribution_support"] == "reference_topk_tail_v1"
            assert cache["top_k"] == 8

    def test_a_full_vocab_search_leaves_the_operator_config_without_the_key(
            self, pieces):
        """Absent, not `full_vocab_v1`: the historical identity is preserved by
        the key not existing, and 785 committed records depend on it."""
        seen = {}
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=FULL_VOCAB_V1)

        _spy_on(search, seen)
        search.run()
        assert "distribution_support" not in seen["config"], seen["config"]
        assert seen["support"].is_full_vocab


class TestACoarsenedPartitionIsADifferentHypothesis:

    def test_the_config_hash_moves_and_the_full_vocab_one_does_not(self, pieces):
        teacher, calib, suite, suite_items = pieces
        base = dict(
            run_id="hash", target_spec=ArchSpec.of("qwen3", DEPTH_ONLY_TARGET),
            schedule=SCHEDULE_V1, seed=7, workdir=Path("/tmp/x"),
            profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
            allowed_impls=("depth.causal_kl_greedy_v1",), device="cpu")
        full = SearchConfig(**base)
        topk = SearchConfig(**base, distribution_support=TOY_TOPK)
        other_k = SearchConfig(**base,
                               distribution_support=reference_topk_tail(16))

        assert full.config_hash != topk.config_hash, (
            "a coarsened partition is a different estimand and must not share a "
            "config_hash with the full-vocabulary run")
        assert topk.config_hash != other_k.config_hash, (
            "two different K are two different partitions")
        #: And the full-vocabulary dict is exactly what it was: the field is
        #: absent, so a committed search's hash is unchanged by this field
        #: existing at all.
        assert "distribution_support" not in full.as_dict()
        assert topk.as_dict()["distribution_support"] == TOY_TOPK.as_dict()

    def test_every_expanded_state_id_differs_and_the_root_is_shared(self, pieces):
        """The hash is the claim; the state ids are the consequence.

        The ROOT is shared and must be: it is the same teacher with no operator
        applied, so a partition cannot reach it. Everything with a step differs,
        which is the thing worth asserting — and asserting total disjointness
        instead would have been asserting that the root is not the root.
        """
        states = {}
        for label, support in (("full", FULL_VOCAB_V1), ("topk", TOY_TOPK)):
            tmp = Path(tempfile.mkdtemp())
            search, _, _ = _search(tmp, pieces, support=support,
                                   run_id=f"ids-{label}")
            states[label] = search.run().states
        roots = {k: {i for i, s in v.items() if not s.steps}
                 for k, v in states.items()}
        expanded = {k: {i for i, s in v.items() if s.steps}
                    for k, v in states.items()}
        assert roots["full"] == roots["topk"], (
            "the root moved, which no distribution support can do: it is the "
            "teacher with no operator applied")
        assert expanded["full"] and expanded["topk"], "nothing was expanded"
        assert expanded["full"].isdisjoint(expanded["topk"]), (
            "an expanded state has the same id under both partitions, so one "
            "run's record could be adopted by the other")


class TestTheOperatorsAndTheBeamCannotDisagree:
    """Both directions. This is why the field is on the RUN and not per operator."""

    def test_a_topk_search_with_a_full_vocab_measurer_is_refused(self, pieces):
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=TOY_TOPK,
                               evaluator_support=FULL_VOCAB_V1,
                               declare_protocol=False)
        with pytest.raises(SearchError, match="ONE partition|reduce over"):
            search.run()

    def test_a_full_vocab_search_with_a_topk_measurer_is_refused(self, pieces):
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=FULL_VOCAB_V1,
                               evaluator_support=TOY_TOPK,
                               declare_protocol=False)
        with pytest.raises(SearchError, match="ONE partition|reduce over"):
            search.run()

    def test_the_agreeing_configuration_runs(self, pieces):
        """So the two refusals above are not refusing everything."""
        tmp = Path(tempfile.mkdtemp())
        search, _, _ = _search(tmp, pieces, support=TOY_TOPK)
        assert search.run()

    def test_the_declared_protocol_binds_the_partition(self, pieces):
        """A driver that declares a protocol id taken under one partition and
        configures another is refused at construction, before anything is paid
        for."""
        teacher, calib, suite, suite_items = pieces
        full_ev = _evaluator(teacher, suite, suite_items, FULL_VOCAB_V1)
        topk_ev = _evaluator(teacher, suite, suite_items, TOY_TOPK)
        assert full_ev.measurement_protocol_id != topk_ev.measurement_protocol_id, (
            "the two partitions produced the same protocol id, so the identity "
            "does not distinguish them")
        tmp = Path(tempfile.mkdtemp())
        config = SearchConfig(
            run_id="declared", target_spec=ArchSpec.of("qwen3", DEPTH_ONLY_TARGET),
            schedule=SCHEDULE_V1, seed=7, workdir=tmp,
            profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
            distribution_support=TOY_TOPK,
            measurement_protocol_id=full_ev.measurement_protocol_id,
            allowed_impls=("depth.causal_kl_greedy_v1",), device="cpu")
        with pytest.raises(SearchError, match="protocol"):
            BeamSearch(
                adapter=adapter_for_config(teacher.config), config=config,
                root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
                root_loader=lambda: teacher,
                calibration_loader=lambda profile: calib,
                measurer=topk_ev, execution=B3_SORTED, numerics=CPU_NUMERICS)


class TestCoreHoldsNoCampaignsK:
    """The support is a parameter; a campaign's value belongs to the experiment."""

    def test_no_k_literal_reaches_the_search_module(self):
        src = (REPO / "src/aadistill/initialization/planning/search.py").read_text()
        assert "reference_topk_tail(" not in src, (
            "core constructs a Top-K support, which means it chose a K")
        assert "top_k=" not in src
        assert "200" not in src.replace("2000", "").replace("20000", "")

    def test_the_default_is_the_historical_contract(self):
        from aadistill.initialization.planning.search import SearchConfig as SC

        default = SC.__dataclass_fields__["distribution_support"].default
        assert default is FULL_VOCAB_V1

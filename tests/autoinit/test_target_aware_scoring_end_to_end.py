"""The target-aware path, through real operators, a real search and a real suite.

Four claims, each on a tiny but genuine Qwen3 model rather than a mock:

1. **The incumbent policy changes nothing.** Every operator takes its untouched
   branch and produces the artifact it produced before the policy existed.
2. **The target-aware policy changes the structural decisions.** DEPTH's kept
   layers, FFN's kept neurons and ATTENTION's kept heads all move, which is the
   hypothesis — an abstraction that threaded through without moving anything
   would not be testing it.
3. **The operators and the beam metric consume one policy.** A driver that
   passes the policy to the search and forgets the evaluator is refused.
4. **Resume refuses across protocols and across policies.** The A3 collision,
   and its scoring-side twin.

Written against the operators FIRST rather than through a driver, because the
question is what the operators compute, and a driver test would also be testing
the driver.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from conftest import (  # noqa: E402  type: ignore
    TARGET_GEOMETRY, TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, get_implementation,
)
from aadistill.initialization.planning.metrics import (  # noqa: E402
    REFERENCE_EXECUTION, StateEvaluator,
)
from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1  # noqa: E402
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig, SearchError,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    ALL_POSITIONS_V1, SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.specs.arch import ArchSpec, adapter_for_config  # noqa: E402
from aadistill.initialization.specs.artifact import identify_checkpoint  # noqa: E402
from aadistill.initialization.specs.materialization import (  # noqa: E402
    NumericalEnvironment,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    MeasurementError, ReferenceStrategy, StateEvalSuite, SuiteItem,
)

#: The incumbent chain, in the incumbent order. These four implementations are
#: the frozen current-best per structural kind.
CHAIN = ("depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
         "width.global_pca_v0", "attention.activation_importance_v1")

B3_SORTED = ExecutionConfig(micro_batch_size=3,
                            calibration_batch_packing="length_sorted_v1")
B1_ORIGINAL = ExecutionConfig(micro_batch_size=1,
                              calibration_batch_packing="original_order_v1")
CPU_NUMERICS = NumericalEnvironment(device_type="cpu", compute_dtype="float32")


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    register_builtin_adapters()
    register_builtin_operators()
    #: Registered by its own module — the frozen-set escape route its docstring
    #: describes — so importing it is inert and a test must ask.
    activation_importance.register()
    yield


def tagged_items(*, seed: int = 101, n: int = 3, seq_len: int = 24):
    """Calibration items where HALF carry an `assistant` tag.

    Half, deliberately: it is the shape of both frozen mixtures (51 of 67, 50 of
    62), it makes the mixture's combined form `select` rather than `all`, and it
    exercises the untagged fallback in the same pass.
    """
    raw = make_items(n_per_subtype=n, seq_len=seq_len, seed=seed)
    for index, row in enumerate(raw):
        if index % 2 == 0:
            row["tags"] = dict(row["tags"])
            row["tags"]["assistant"] = list(
                range(10, int(row["input_ids"].shape[1]) - 1))
    return raw


def _truncated_tags(tags, n_predictions: int):
    """Tags restricted to an item that has been shortened.

    Both stored forms, because a tag is an index LIST on a mixture item and a
    boolean MASK on a suite item, and truncating a list by slicing its elements
    keeps out-of-range indices — which is a real mistake and not a hypothetical
    one: it is what this helper exists because of.
    """
    out = {}
    for name, value in tags.items():
        if hasattr(value, "dtype"):                    # a mask: slice positions
            out[name] = value[:n_predictions]
        else:                                          # indices: filter by range
            out[name] = [int(i) for i in value if int(i) < n_predictions]
    return out


def run_chain(policy, execution, *, items, teacher):
    """Apply the four incumbent operators in order; return the decisions."""
    adapter = adapter_for_config(teacher.config)
    target = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    model, spec = teacher, ArchSpec.of("qwen3", TEACHER_GEOMETRY)
    profile = make_profile("balanced")
    decisions: list[dict] = []
    for impl_id in CHAIN:
        impl = get_implementation(impl_id)
        config = {"n_calibration_items": len(items), **policy_config(policy)}
        ctx = OperatorContext(
            adapter=adapter, model=model, parent_spec=spec, target_spec=target,
            profile=profile, calibration_items=items, seed=7, device="cpu",
            config=config, execution=execution, position_policy=policy)
        outcome = impl.execute(ctx)
        model, spec = outcome.model, adapter.spec_of(outcome.model)
        decisions.append({
            "impl_id": impl_id,
            "trace": dict(outcome.trace),
            "artifacts": dict(outcome.artifacts),
            "metrics": dict(outcome.local_metrics.values),
        })
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "final"
        adapter.save(model, str(out))
        digest = identify_checkpoint(
            out, adapter=adapter, spec=spec,
            num_parameters=adapter.param_count(spec)).artifact_digest
    return digest, decisions


class TestTheIncumbentPolicyChangesNothing:
    def test_naming_the_incumbent_policy_is_the_same_as_not_naming_one(self, teacher):
        """The compatibility claim, as an experiment rather than an argument.

        A run that passes `ALL_POSITIONS_V1` explicitly and a run that leaves the
        context's default must produce the SAME BYTES, because the incumbent
        policy's job is to be the behaviour that was already there.
        """
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        adapter = adapter_for_config(teacher.config)
        target = ArchSpec.of("qwen3", TARGET_GEOMETRY)
        spec = ArchSpec.of("qwen3", TEACHER_GEOMETRY)
        impl = get_implementation("ffn.activation_importance_v0")

        def once(**kwargs):
            ctx = OperatorContext(
                adapter=adapter, model=teacher, parent_spec=spec,
                target_spec=target, profile=make_profile("balanced"),
                calibration_items=items, seed=7, device="cpu",
                config={"n_calibration_items": len(items)}, **kwargs)
            return impl.execute(ctx).artifacts["kept_neurons"]

        assert once() == once(position_policy=ALL_POSITIONS_V1)

    def test_the_incumbent_policy_contributes_no_config_key(self, teacher):
        """So a historical state id stays derivable from live code."""
        assert policy_config(ALL_POSITIONS_V1) == {}

    def test_every_operator_reports_an_unrestricted_pass(self, teacher):
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        _, decisions = run_chain(ALL_POSITIONS_V1, B1_ORIGINAL,
                                 items=items, teacher=teacher)
        for step in decisions:
            report = step["trace"]["scoring_positions"]
            assert report["position_policy"] == ALL_POSITIONS_V1.qualified_id
            assert report["token_axis"]["form"] == "all"
            assert report["prediction_axis"]["form"] == "all"
            assert report["token_axis"]["active"] == \
                report["token_axis"]["positions"]


class TestTheTargetAwarePolicyMovesTheDecisions:
    def test_the_artifact_differs(self, teacher):
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        all_digest, _ = run_chain(ALL_POSITIONS_V1, B1_ORIGINAL,
                                  items=items, teacher=teacher)
        sup_digest, _ = run_chain(SUPERVISED_TARGET_V1, B1_ORIGINAL,
                                  items=items, teacher=teacher)
        assert all_digest != sup_digest

    def test_each_structural_kind_moves(self, teacher):
        """If the abstraction threaded through without moving DEPTH, FFN and
        ATTENTION, D1 would be measuring nothing. One assertion per kind, so a
        regression says WHICH operator stopped reading the policy."""
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        _, base = run_chain(ALL_POSITIONS_V1, B1_ORIGINAL,
                            items=items, teacher=teacher)
        _, sup = run_chain(SUPERVISED_TARGET_V1, B1_ORIGINAL,
                           items=items, teacher=teacher)
        moved = {}
        moved["DEPTH"] = base[0]["trace"]["kept_layers"] != \
            sup[0]["trace"]["kept_layers"]
        moved["FFN"] = base[1]["artifacts"]["kept_neurons"] != \
            sup[1]["artifacts"]["kept_neurons"]
        #: WIDTH produces a projection rather than a selection, so its movement
        #: shows in the captured-energy metric rather than in an index list.
        moved["RESIDUAL_WIDTH"] = (
            base[2]["metrics"]["op.width.energy_captured_frac"]
            != sup[2]["metrics"]["op.width.energy_captured_frac"])
        moved["ATTENTION"] = base[3]["trace"]["kept_q_heads_per_layer"] != \
            sup[3]["trace"]["kept_q_heads_per_layer"]
        assert all(moved.values()), f"did not move: {sorted(k for k, v in moved.items() if not v)}"

    def test_every_operator_reports_the_restriction_it_applied(self, teacher):
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        _, decisions = run_chain(SUPERVISED_TARGET_V1, B1_ORIGINAL,
                                 items=items, teacher=teacher)
        for step in decisions:
            report = step["trace"]["scoring_positions"]
            assert report["position_policy"] == SUPERVISED_TARGET_V1.qualified_id
            assert report["token_axis"]["form"] == "select"
            assert report["token_axis"]["active"] < \
                report["token_axis"]["positions"]

    def test_the_attention_trace_separates_valid_from_admitted(self, teacher):
        """`executed - padded == valid` and `admitted == calibration_tokens` are
        two different identities, and conflating them is how a reader would read
        a target-aware statistic as a masking bug."""
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        _, decisions = run_chain(SUPERVISED_TARGET_V1, B3_SORTED,
                                 items=items, teacher=teacher)
        trace = decisions[3]["trace"]
        assert trace["executed_positions"] - trace["padded_positions"] == \
            trace["valid_positions"]
        assert trace["admitted_positions"] == trace["calibration_tokens"]
        assert trace["admitted_positions"] < trace["valid_positions"]


class TestTheHashedConfigAndThePolicyMustAgree:
    def test_a_config_that_names_another_policy_is_refused(self, teacher):
        """The declared hash is what the state id is derived from, so a mismatch
        would record a scoring rule that did not run."""
        items = prepare_calibration_items(tagged_items(), profile_id="toy")
        adapter = adapter_for_config(teacher.config)
        ctx = OperatorContext(
            adapter=adapter, model=teacher,
            parent_spec=ArchSpec.of("qwen3", TEACHER_GEOMETRY),
            target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
            profile=make_profile("balanced"), calibration_items=items, seed=7,
            device="cpu",
            config={"n_calibration_items": len(items),
                    **policy_config(SUPERVISED_TARGET_V1)},
            position_policy=ALL_POSITIONS_V1)
        with pytest.raises(ContractViolation, match="did not run"):
            get_implementation("ffn.activation_importance_v0").execute(ctx)


def _suite(items):
    return StateEvalSuite(
        suite_id="test.target_aware", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")


def _suite_items(rows):
    return [SuiteItem(item_id=r["item_id"], input_ids=r["input_ids"],
                      domain=r["domain"], subtype=r["subtype"],
                      tags=r["tags"]) for r in rows]


class TestTheStateMetricIsTargetAwareToo:
    def test_the_policy_moves_the_teacher_kl_and_states_itself(self, teacher):
        """A candidate chosen on supervised positions must not be pruned by a
        full-sequence beam metric. So the metric reads the same policy, and the
        evaluation records which."""
        rows = tagged_items(seed=303)
        #: Prediction-position INDEX lists become boolean masks for a SuiteItem,
        #: which is the other stored form the policy has to read.
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        suite = _suite(items)
        candidate = build_tiny_model(TEACHER_GEOMETRY, seed=9)

        def measure(policy):
            ev = StateEvaluator(
                suite, items, device="cpu",
                reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
                vocab_size=TEACHER_GEOMETRY["vocab_size"],
                position_policy=policy)
            ev.prime_reference(teacher)
            return ev.evaluate(candidate, "digest-under-test")

        base, sup = measure(ALL_POSITIONS_V1), measure(SUPERVISED_TARGET_V1)
        assert base.values["state.teacher_kl.equal_domain_mean"] != \
            sup.values["state.teacher_kl.equal_domain_mean"]
        assert sup.detail["position_policy"] == SUPERVISED_TARGET_V1.qualified_id
        #: The count and the denominator are different numbers, and only one of
        #: them is the estimand's. Under the incumbent they coincide, which is
        #: exactly why a test has to drive them apart.
        assert base.positions == sup.positions
        assert sup.detail["scored_weight"] < base.detail["scored_weight"]
        assert base.detail["scored_weight"] == float(base.positions)

    @pytest.mark.parametrize("strategy", [ReferenceStrategy.RECOMPUTE,
                                          ReferenceStrategy.CACHE_IN_MEMORY])
    @pytest.mark.parametrize("policy", [ALL_POSITIONS_V1, SUPERVISED_TARGET_V1])
    def test_batching_the_forwards_does_not_move_the_metric(self, teacher,
                                                            strategy, policy):
        """Batching moves the FORWARD; the reduction stays the same `distortion`
        call at the same per-item shape with the same chunk boundaries. On CPU
        float32 that makes the two paths numerically identical, which is the
        strongest form the claim can take at `$0` — the batch-shape question
        belongs to the GPU validation that owns it."""
        rows = tagged_items(seed=404)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        #: Varied lengths, so the batched path actually pads. Equal lengths would
        #: make this test pass without exercising a single mask.
        for index, r in enumerate(rows):
            length = 24 - 3 * index
            r["input_ids"] = r["input_ids"][:, :length]
            r["tags"] = normalized_prediction_tags(
                _truncated_tags(r["tags"], length - 1), length - 1)
        items = _suite_items(rows)
        suite = _suite(items)
        candidate = build_tiny_model(TEACHER_GEOMETRY, seed=9)

        def measure(execution):
            ev = StateEvaluator(
                suite, items, device="cpu", reference_strategy=strategy,
                vocab_size=TEACHER_GEOMETRY["vocab_size"],
                position_policy=policy, execution=execution)
            ev.prime_reference(teacher)
            return ev.evaluate(candidate, "digest-under-test")

        one, three = measure(REFERENCE_EXECUTION), measure(B3_SORTED)
        assert one.positions == three.positions
        assert one.detail["scored_weight"] == three.detail["scored_weight"]
        for key in sorted(one.values):
            assert one.values[key] == pytest.approx(three.values[key],
                                                    rel=1e-12, abs=1e-15), key
        assert one.detail["reference_path"] and not three.detail["reference_path"]

    def test_an_over_budget_batch_is_refused_before_the_forward(self, teacher):
        """Not discovered as an OOM mid-search. The causal-depth rehearsal was
        killed by the OOM killer for the absence of exactly this check."""
        rows = tagged_items(seed=505)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        ev = StateEvaluator(
            _suite(items), items, device="cpu",
            reference_strategy=ReferenceStrategy.RECOMPUTE,
            position_policy=ALL_POSITIONS_V1, execution=B3_SORTED,
            batch_budget_bytes=1024)
        ev.prime_reference(teacher)
        with pytest.raises(MeasurementError, match="over the"):
            ev.evaluate(build_tiny_model(TEACHER_GEOMETRY, seed=9), "d")

    def test_handing_distortion_bf16_rows_is_numerically_free(self):
        """Which is what lets the batched path skip a 2.4 GiB eager upcast.

        `_logit_pairs` yields each row in the model's own dtype instead of
        calling `.float()` on it, because `distortion` upcasts per chunk and
        bfloat16 -> float32 is exact: same exponent width, more mantissa. The
        memory bound in `batch_plan` counts the logit blocks and NOT two full
        float32 row copies per item, so that claim has to be true rather than
        plausible.
        """
        from aadistill.initialization.statistics.contribution import distortion

        torch.manual_seed(3)
        positions, vocab = 200, 501
        ref = torch.randn(positions, vocab).bfloat16()
        abl = torch.randn(positions, vocab).bfloat16()
        targets = torch.randint(0, vocab, (positions,))
        lazy = distortion(ref, abl, targets).as_dict()
        eager = distortion(ref.float(), abl.float(), targets).as_dict()
        for key in ("kl", "reverse_kl", "ref_ce", "abl_ce", "ce_delta",
                    "top1_agreement", "positions", "weight"):
            assert lazy[key] == eager[key], key

    def test_the_plan_is_derivable_before_any_model_exists(self, teacher):
        """So a preflight can bound the protocol from the frozen suite."""
        rows = tagged_items(seed=606)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        ev = StateEvaluator(_suite(items), items, device="cpu",
                            execution=B3_SORTED)
        plan = ev.batch_plan(151936, 2)
        assert plan["peak_logit_bytes"] > 0 and plan["within_budget"]
        assert plan["reference_path"] is False
        #: The widest group, not the mean. Averages are not bounds: the
        #: allocation that fails is the largest one.
        assert plan["max_group_width"] == max(
            int(i.input_ids.shape[1]) for i in items)


def _search(workdir, *, policy, execution, teacher, calib, suite, suite_items):
    adapter = adapter_for_config(teacher.config)
    evaluator = StateEvaluator(
        suite, suite_items, device="cpu",
        reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
        vocab_size=TEACHER_GEOMETRY["vocab_size"], position_policy=policy)
    evaluator.prime_reference(teacher)
    config = SearchConfig(
        run_id="regression", target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
        schedule=SCHEDULE_V1, seed=7, workdir=workdir,
        profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
        position_policy=policy,
        #: DECLARED, which is the path a driver should take: the protocol enters
        #: `config_hash`, it is checked before the first expensive measurement,
        #: and the first restore of a resumed run is answered correctly instead
        #: of being declined while the search waits to learn its own protocol.
        measurement_protocol_id=evaluator.measurement_protocol_id,
        allowed_impls=("depth.positional_v0", "ffn.activation_importance_v0",
                       "width.global_pca_v0", "attention.weight_proxy_v0"),
        device="cpu")
    return BeamSearch(
        adapter=adapter, config=config, root_teacher_id="toy/teacher",
        root_teacher_sha256="de" * 32, root_loader=lambda: teacher,
        calibration_loader=lambda profile: calib,
        measurer=lambda model, digest: evaluator.evaluate(model, digest),
        execution=execution, numerics=CPU_NUMERICS), config


class TestDevicePlacementAtZeroCost:
    """Placement, asserted rather than hoped for, without renting a GPU.

    Four paid pods in this project have died inside a cross-device line that
    every CPU test passed, because on a CPU-only box every device coincides and
    a defaulted `torch.zeros(...)` is indistinguishable from a placed one. The
    `meta` device breaks that coincidence at `$0`: it is not the host, tensors
    carry it through `.to()` and shape-only ops, and a tensor that silently
    defaulted to CPU is immediately visible.

    What this does NOT replace is the real-CUDA validation: meta performs no
    arithmetic, so it says nothing about a kernel, a bf16 reduction or real
    memory. It is the cheapest step that can answer the placement question,
    which is the only question it is asked.
    """

    def _items(self):
        return [
            {"item_id": "a", "domain": "general", "subtype": "text",
             "input_ids": torch.arange(20, dtype=torch.long)[None, :],
             "tags": {"assistant": list(range(12, 19))}},
            {"item_id": "b", "domain": "math", "subtype": "arith",
             "input_ids": torch.arange(16, dtype=torch.long)[None, :],
             "tags": {"assistant": list(range(8, 15))}},
        ]

    def test_every_mask_and_weight_is_placed_from_the_batch(self):
        from aadistill.initialization.calibration.batching import build_batch
        from aadistill.initialization.scoring.batches import active_positions

        items = self._items()
        active = active_positions(items, SUPERVISED_TARGET_V1)
        batch = build_batch(items, pad_id=0, device="meta")
        assert batch.input_ids.device.type == "meta"

        mask = active.token_mask_for(batch, (0, 1))
        weights = active.prediction_weights_for(batch, (0, 1))
        assert mask.device == batch.input_ids.device
        assert weights.device == batch.input_ids.device
        #: And the shapes follow the two axes, not one of them twice.
        assert tuple(mask.shape) == tuple(batch.input_ids.shape)
        assert tuple(weights.shape) == (batch.size,
                                        batch.input_ids.shape[1] - 1)
        assert mask.dtype == torch.bool

    def test_active_rows_does_not_drag_a_mask_back_to_the_host(self):
        from aadistill.initialization.calibration.batching import (
            active_rows, build_batch,
        )

        items = self._items()
        batch = build_batch(items, pad_id=0, device="meta")
        host_mask = torch.ones(batch.input_ids.shape, dtype=torch.bool)
        placed = active_rows(host_mask, batch.input_ids.shape,
                             batch.input_ids.device)
        assert placed.device == batch.input_ids.device

    def test_the_reducers_move_the_weights_to_the_logits(self):
        """A host float vector meeting a device logit tensor raises, so this is
        not an optimization — it is what makes the weighted path run at all."""
        from aadistill.initialization.statistics.contribution import (
            _position_weights,
        )

        host = torch.ones(8, dtype=torch.float64)
        moved = _position_weights(host, 8, torch.device("meta"), "probe")
        assert moved.device.type == "meta"
        rows = _position_weights(torch.ones(2, 8, dtype=torch.float64), 8,
                                 torch.device("meta"), "probe", rows=2)
        assert rows.device.type == "meta"


class TestResumeRefusals:
    @pytest.fixture
    def pieces(self, teacher):
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=707)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        return calib, _suite(items), items

    def test_the_same_protocol_resumes(self, teacher, pieces):
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            first, _ = _search(tmp, policy=ALL_POSITIONS_V1,
                               execution=B1_ORIGINAL, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            first.run()
            again, _ = _search(tmp, policy=ALL_POSITIONS_V1,
                               execution=B1_ORIGINAL, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            again.run()
            assert len(again.resumed_ids) > 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_different_execution_protocol_does_not(self, teacher, pieces):
        """The A3 collision. A journal entry's `state_id` says the same path was
        walked; it does not say the same bytes were built."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            first, _ = _search(tmp, policy=ALL_POSITIONS_V1,
                               execution=B1_ORIGINAL, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            first.run()
            other, _ = _search(tmp, policy=ALL_POSITIONS_V1,
                               execution=B3_SORTED, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            other.run()
            assert other.resumed_ids == set()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_different_scoring_policy_does_not(self, teacher, pieces):
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            first, _ = _search(tmp, policy=ALL_POSITIONS_V1,
                               execution=B1_ORIGINAL, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            first.run()
            other, _ = _search(tmp, policy=SUPERVISED_TARGET_V1,
                               execution=B1_ORIGINAL, teacher=teacher,
                               calib=calib, suite=suite, suite_items=items)
            other.run()
            assert other.resumed_ids == set()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_search_records_all_four_identities(self, teacher, pieces):
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            search, _ = _search(tmp, policy=SUPERVISED_TARGET_V1,
                                execution=B3_SORTED, teacher=teacher,
                                calib=calib, suite=suite, suite_items=items)
            result = search.run()
            leaf = result.leaves[0]
            identity = leaf.materialization
            assert identity is not None
            assert identity.semantic_state_id == leaf.state_id
            assert identity.artifact_digest == leaf.artifact_digest
            assert identity.materialization_id != identity.semantic_state_id
            assert "materialization" in leaf.as_dict()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_search_with_no_declared_numerics_records_none(self, teacher,
                                                             pieces):
        """Which is the historical behaviour and is why it is not required: a
        caller that has not stated a device class has nothing honest to
        fingerprint, and inventing one would be worse than having none."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            search, config = _search(tmp, policy=ALL_POSITIONS_V1,
                                     execution=B1_ORIGINAL, teacher=teacher,
                                     calib=calib, suite=suite,
                                     suite_items=items)
            search.numerics = None
            result = search.run()
            assert all(s.materialization is None for s in result.leaves)
            assert "materialization" not in result.leaves[0].as_dict()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTheMeasurementProtocolBinder:
    """One run, one measurement protocol — however the search comes to know it.

    `_restore` used to ask two independent questions (the suite hash, then the
    policy hash) and the next three terms would have been three more questions.
    One `measurement_protocol_id` replaced them, which moves the risk: a search
    that did not know its own protocol would decline every restore silently.
    """

    @pytest.fixture
    def pieces(self, teacher):
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=606)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        return calib, _suite(items), items

    def _evaluator(self, teacher, suite, items, policy=ALL_POSITIONS_V1):
        evaluator = StateEvaluator(
            suite, items, device="cpu",
            reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
            vocab_size=TEACHER_GEOMETRY["vocab_size"], position_policy=policy)
        evaluator.prime_reference(teacher)
        return evaluator

    def _build(self, tmp, teacher, calib, suite, items, *, evaluator,
               declared, run_id="binder"):
        adapter = adapter_for_config(teacher.config)
        config = SearchConfig(
            run_id=run_id, target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
            schedule=SCHEDULE_V1, seed=7, workdir=tmp,
            profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
            measurement_protocol_id=declared,
            allowed_impls=("depth.positional_v0", "ffn.activation_importance_v0",
                           "width.global_pca_v0", "attention.weight_proxy_v0"),
            device="cpu")
        return BeamSearch(
            adapter=adapter, config=config, root_teacher_id="toy/teacher",
            root_teacher_sha256="de" * 32, root_loader=lambda: teacher,
            calibration_loader=lambda profile: calib,
            measurer=lambda m, d: evaluator.evaluate(m, d),
            execution=B1_ORIGINAL, numerics=CPU_NUMERICS)

    def test_an_undeclared_search_adopts_the_protocol_it_measured_under(
            self, teacher, pieces):
        """Every driver wraps its evaluator in a lambda, so the attribute the
        constructor looks for is invisible. A search that could not learn its
        own protocol would stamp records it could never match again."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            evaluator = self._evaluator(teacher, suite, items)
            search = self._build(tmp, teacher, calib, suite, items,
                                 evaluator=evaluator, declared=None)
            assert search.measurement_protocol_id is None
            search.run()
            assert search.measurement_protocol_id == \
                evaluator.measurement_protocol_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_declared_protocol_is_known_before_the_first_measurement(
            self, teacher, pieces):
        """Which is the better path, and why a driver should declare: the first
        restore of a resumed run is answered correctly instead of being
        declined while the search waits to learn its own protocol."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            evaluator = self._evaluator(teacher, suite, items)
            search = self._build(
                tmp, teacher, calib, suite, items, evaluator=evaluator,
                declared=evaluator.measurement_protocol_id)
            assert search.measurement_protocol_id == \
                evaluator.measurement_protocol_id
            result = search.run()
            #: And it is in the run's own identity, not only in its records.
            assert result.config.as_dict()["measurement_protocol_id"] == \
                evaluator.measurement_protocol_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_declaration_disagreeing_with_the_measurer_is_refused(
            self, teacher, pieces):
        """At the FIRST measurement, not silently: a search that declared one
        protocol and measured under another has no comparable numbers at all."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            evaluator = self._evaluator(teacher, suite, items)
            search = self._build(tmp, teacher, calib, suite, items,
                                 evaluator=evaluator, declared="f" * 32)
            with pytest.raises(SearchError, match="another protocol"):
                search.run()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_declared_run_does_not_resume_an_undeclared_journal(
            self, teacher, pieces):
        """A record written before measurement-protocol identity existed has no
        recorded reduction or execution, so nothing can show it comparable. The
        refusal is explicit — the old record is not reinterpreted."""
        calib, suite, items = pieces
        tmp = Path(tempfile.mkdtemp())
        try:
            evaluator = self._evaluator(teacher, suite, items)
            first = self._build(tmp, teacher, calib, suite, items,
                                evaluator=evaluator, declared=None)
            first.run()
            #: Strip the protocol from every journal record, which is exactly
            #: what a pre-identity journal looks like.
            journal = tmp / "states.jsonl"
            rows = [json.loads(line) for line in
                    journal.read_text().splitlines() if line.strip()]
            for row in rows:
                detail = ((row.get("evaluation") or {}).get("detail") or {})
                detail.pop("measurement_protocol_id", None)
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))

            again = self._build(
                tmp, teacher, calib, suite, items, evaluator=evaluator,
                declared=evaluator.measurement_protocol_id)
            again.run()
            assert again.resumed_ids == set()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestOnePolicyForTheOperatorsAndTheBeam:
    def test_a_measurer_scoring_other_positions_is_refused(self, teacher):
        """`measurer` is an opaque callable, so this is the only place the search
        can check it. A driver that passed the policy to `SearchConfig` and
        forgot the evaluator would run target-aware operators and prune them on a
        full-sequence KL — two experiments reported as one."""
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=808)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        suite = _suite(items)
        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = adapter_for_config(teacher.config)
            #: The evaluator left at the INCUMBENT while the search is
            #: target-aware. The mistake, exactly.
            evaluator = StateEvaluator(
                suite, items, device="cpu",
                reference_strategy=ReferenceStrategy.CACHE_IN_MEMORY,
                vocab_size=TEACHER_GEOMETRY["vocab_size"],
                position_policy=ALL_POSITIONS_V1)
            evaluator.prime_reference(teacher)
            config = SearchConfig(
                run_id="mismatch",
                target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
                schedule=SCHEDULE_V1, seed=7, workdir=tmp,
                profiles=(make_profile("balanced"),), policy=PARETO_V1,
                suite=suite, position_policy=SUPERVISED_TARGET_V1,
                allowed_impls=("depth.positional_v0",
                               "ffn.activation_importance_v0",
                               "width.global_pca_v0",
                               "attention.weight_proxy_v0"),
                device="cpu")
            search = BeamSearch(
                adapter=adapter, config=config, root_teacher_id="toy/teacher",
                root_teacher_sha256="de" * 32, root_loader=lambda: teacher,
                calibration_loader=lambda profile: calib,
                measurer=lambda m, d: evaluator.evaluate(m, d),
                execution=B1_ORIGINAL, numerics=CPU_NUMERICS)
            with pytest.raises(SearchError, match="ONE policy"):
                search.run()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_policy_forks_the_search_config_hash(self, teacher):
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=909)
        items = _suite_items(rows)
        suite = _suite(items)
        tmp = Path(tempfile.mkdtemp())
        try:
            _, base = _search(tmp, policy=ALL_POSITIONS_V1,
                              execution=B1_ORIGINAL, teacher=teacher,
                              calib=calib, suite=suite, suite_items=items)
            _, sup = _search(tmp, policy=SUPERVISED_TARGET_V1,
                             execution=B1_ORIGINAL, teacher=teacher,
                             calib=calib, suite=suite, suite_items=items)
            assert base.config_hash != sup.config_hash
            #: And the incumbent omits the field entirely, so a search recorded
            #: before it existed still hashes to the value its own record
            #: carries.
            assert "position_policy" not in base.as_dict()
            assert sup.as_dict()["position_policy"] == \
                SUPERVISED_TARGET_V1.qualified_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_stats_cache_key_separates_policies_and_protocols(self, teacher):
        """Two operators share one statistics pass when they expand the same
        parent under the same profile. A pass taken under one policy or one batch
        rule is not the pass another would produce, so the key has to carry
        both — the key's own docstring promised the batch rule and did not."""
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=1010)
        items = _suite_items(rows)
        suite = _suite(items)
        tmp = Path(tempfile.mkdtemp())
        try:
            from aadistill.initialization.specs.state import make_root_state
            from aadistill.initialization.specs.artifact import (
                CheckpointIdentity, ShardRecord,
            )
            keys = set()
            for policy in (ALL_POSITIONS_V1, SUPERVISED_TARGET_V1):
                for execution in (B1_ORIGINAL, B3_SORTED):
                    search, _ = _search(tmp, policy=policy,
                                        execution=execution, teacher=teacher,
                                        calib=calib, suite=suite,
                                        suite_items=items)
                    parent = make_root_state(
                        root_teacher_id="t", root_teacher_sha256="de" * 32,
                        spec=ArchSpec.of("qwen3", TEACHER_GEOMETRY),
                        target_spec=ArchSpec.of("qwen3", TARGET_GEOMETRY),
                        num_parameters=1, seed=0)
                    #: ONE fixed parent identity across all four, so the only
                    #: thing that can separate the keys is the policy and the
                    #: execution protocol.
                    parent.artifact = CheckpointIdentity(
                        path="/nowhere",
                        shards=(ShardRecord(filename="m.safetensors",
                                            sha256="ab" * 32, size_bytes=1),),
                        config_sha256="c", arch_signature="a",
                        num_parameters=1)
                    keys.add(search._stats_key(parent, make_profile("balanced")))
            assert len(keys) == 4, (
                "two policies x two execution protocols must be four distinct "
                f"statistics-cache keys, got {len(keys)}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMaterializationOwnership:
    """Ownership, not only checking: paths and the resume lookup.

    Refusing a mismatched record was necessary and not sufficient. Two further
    things had to become materialization-keyed, and each is driven here against
    a real search rather than argued:

    * the checkpoint destination, because two materializations of one semantic
      state writing to one directory means the second overwrites the first;
    * the journal lookup, because keyed on the semantic state the journal can
      only offer the NEWEST record for a path — so a run whose own
      materialization was journalled first would be told "not yours" and would
      rebuild work it already had.
    """

    @pytest.fixture
    def pieces(self, teacher):
        calib = prepare_calibration_items(tagged_items(), profile_id="toy")
        rows = tagged_items(seed=1111)
        from aadistill.initialization.scoring.positions import (
            normalized_prediction_tags,
        )
        for r in rows:
            r["tags"] = normalized_prediction_tags(
                r["tags"], int(r["input_ids"].shape[1]) - 1)
        items = _suite_items(rows)
        return calib, _suite(items), items

    def _run(self, tmp, execution, teacher, pieces):
        calib, suite, items = pieces
        search, _ = _search(tmp, policy=ALL_POSITIONS_V1, execution=execution,
                            teacher=teacher, calib=calib, suite=suite,
                            suite_items=items)
        return search, search.run()

    def test_checkpoint_directories_cannot_collide(self, teacher, pieces):
        """Two protocols, one semantic state, two destinations."""
        tmp = Path(tempfile.mkdtemp())
        try:
            one, first = self._run(tmp, B1_ORIGINAL, teacher, pieces)
            three, second = self._run(tmp, B3_SORTED, teacher, pieces)

            shared = ({s.state_id for s in first.leaves}
                      & {s.state_id for s in second.leaves})
            assert shared, "the two runs must share at least one semantic state"
            by_id_one = {s.state_id: s for s in first.leaves}
            by_id_three = {s.state_id: s for s in second.leaves}
            for state_id in shared:
                a, b = by_id_one[state_id], by_id_three[state_id]
                assert a.materialization.materialization_id != \
                    b.materialization.materialization_id
                assert a.checkpoint_path != b.checkpoint_path, (
                    f"{state_id} has one checkpoint path for two "
                    "materializations; the second overwrote the first")
                #: And the layout is the stated one: the semantic id is still
                #: the outer level, so one hypothesis's materializations sit
                #: together rather than scattered by protocol.
                assert Path(a.checkpoint_path).parent.name == state_id
                assert Path(a.checkpoint_path).name == \
                    a.materialization.materialization_id
            #: Both sets of bytes survive on disk at once.
            for state_id in shared:
                assert Path(by_id_one[state_id].checkpoint_path).is_dir()
                assert Path(by_id_three[state_id].checkpoint_path).is_dir()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_one_semantic_state_retains_and_resolves_two_materializations(
            self, teacher, pieces):
        tmp = Path(tempfile.mkdtemp())
        try:
            one, _ = self._run(tmp, B1_ORIGINAL, teacher, pieces)
            three, _ = self._run(tmp, B3_SORTED, teacher, pieces)
            journal = one.store.latest_by_materialization_id()
            semantic = one.store.latest_by_state_id()
            #: The materialization view keeps BOTH; the semantic view keeps one
            #: record per path and therefore strictly fewer. That difference is
            #: the whole reason the second view exists.
            assert len(journal) > len(semantic)
            ids = {r["state_id"] for r in journal.values()}
            per_state = {sid: sum(1 for r in journal.values()
                                  if r["state_id"] == sid) for sid in ids}
            assert max(per_state.values()) >= 2, (
                "no semantic state retained two materializations, so this test "
                "did not exercise what it claims")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resume_finds_ITS_materialization_not_the_latest(self, teacher,
                                                             pieces):
        """THE BUG THE REFUSAL ALONE LEFT OPEN.

        Run protocol A, then protocol B — so B's records are the newest for
        every shared semantic state — then run A again. A must RESUME. Keyed on
        the semantic state it would find B's record, correctly refuse it as not
        its own, and rebuild everything it already had.
        """
        tmp = Path(tempfile.mkdtemp())
        try:
            self._run(tmp, B1_ORIGINAL, teacher, pieces)
            self._run(tmp, B3_SORTED, teacher, pieces)
            again, _ = self._run(tmp, B1_ORIGINAL, teacher, pieces)
            assert again.resumed_ids, (
                "protocol A did not resume its own earlier materialization "
                "after protocol B wrote newer records for the same semantic "
                "states — the lookup is still semantic-state keyed")

            #: And the resumed records really are A's, not B's.
            #:
            #: ASSERTED, not skipped over. This used to end in a `pytest.skip`
            #: for "the toy search did not reach the interesting case", which
            #: was both unreachable — protocol B visits the same semantic states
            #: and therefore writes the newest record for every one of them —
            #: and a skip predicate that no contract said would decide the same
            #: way on a pod. A test that cannot show it exercised its own case
            #: should fail, not pass quietly.
            journal = again.store.latest_by_state_id()
            foreign_newest = 0
            for state_id in again.resumed_ids:
                newest = journal[state_id]
                restored = again.states[state_id]
                assert restored.materialization is not None
                if newest.get("materialization", {}).get("materialization_id") \
                        != restored.materialization.materialization_id:
                    foreign_newest += 1
            assert foreign_newest, (
                "every resumed state's newest semantic record was its own, so "
                "this run never faced a foreign newer record and did not "
                "exercise the bug it exists for")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_root_carries_the_teachers_pinned_identity(self, teacher,
                                                           pieces):
        tmp = Path(tempfile.mkdtemp())
        try:
            search, _ = self._run(tmp, B1_ORIGINAL, teacher, pieces)
            root = search.root_state()
            assert root.materialization is not None
            assert root.materialization.parent_materialization_id is None
            #: Derived from the teacher, so it does not move with the protocol.
            other, _ = self._run(tmp, B3_SORTED, teacher, pieces)
            assert other.root_state().materialization.materialization_id == \
                root.materialization.materialization_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_child_binds_the_parent_it_actually_consumed(self, teacher,
                                                           pieces):
        tmp = Path(tempfile.mkdtemp())
        try:
            search, result = self._run(tmp, B1_ORIGINAL, teacher, pieces)
            root_id = search.root_state().materialization.materialization_id
            depth1 = [s for s in search.states.values() if s.depth == 1]
            assert depth1, "the search produced no level-1 state"
            for s in depth1:
                assert s.materialization.parent_materialization_id == root_id
            deeper = [s for s in search.states.values() if s.depth == 2]
            for s in deeper:
                parent = search.states[s.parent_id]
                assert s.materialization.parent_materialization_id == \
                    parent.materialization.materialization_id
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

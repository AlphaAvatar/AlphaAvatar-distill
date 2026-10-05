"""Stage P times the PRODUCTION operator invocation, and these are why.

The stage this replaced timed a shadow loop: it reimplemented the candidate inner
loop in the driver and therefore omitted the position weights, the
`values.tolist()` host transfer that is the production synchronization, the
per-subtype collection, `domain_balanced_score`, and the reference-cache fill. It
then priced `260 x per-candidate`, which is not the quantity
`BeamSearch._expand_one` records as `operator_seconds` at all.

So the three things worth testing at `$0` are:

* the builder passes every context keyword the production expansion passes, so a
  field added to core cannot be missed here in silence;
* the real operator actually runs through that builder, on a toy model, with the
  observer firing and the artifacts the stage reads present;
* the pricing gate REFUSES every record shape that cannot price -- which includes
  a7's, and that refusal is what returned the envelope to UNRESOLVED.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit", "tests"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

import topk_adoption_driver as drv  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

#: `tie_word_embeddings` because depth surgery does not assign an untied
#: `lm_head`; the operator refuses rather than shipping random weights.
GEOMETRY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=64, tie_word_embeddings=True)

#: Keywords the timing builder passes that `_expand_one` does not, each with the
#: reason. ANYTHING ELSE IS A DEFECT, and in the direction that actually bit: the
#: timing builder injected `distribution_support` while `SearchConfig` had no such
#: field, so a paid 40-minute measurement timed an operator path that no search
#: could reach. The one-directional check could not see it -- it asked only whether
#: the builder was a SUPERSET of core's keywords, and a superset it was.
JUSTIFIED_EXTRAS = {
    #: An execution-only hook. The search has no use for it and must not: it
    #: exists so a validation can reduce a second time from forwards already paid
    #: for, its return value is ignored, and a test asserts watching cannot change
    #: a decision. It is not scientific context.
    "score_observer",
}

#: Keywords `_expand_one` passes that ONE operator invocation has no use for,
#: each with the reason. Anything else it passes must be passed here too.
JUSTIFIED_OMISSIONS = {
    #: An activation-statistics cache. `depth.causal_kl_greedy_v1` reduces KL over
    #: logits and consumes no activation statistics, so there is nothing to cache
    #: and nothing to share -- and `cost.py` records that `operator_seconds`
    #: EXCLUDES the statistics pass in any case.
    "stats_cache",
    "stats_cache_key",
    #: The search's wall-clock deadline. A timing measurement that aborted
    #: part-way would produce a number for a partial invocation, which is worse
    #: than no number; the pod's own watchdog bounds the subrun instead.
    "deadline",
}


def context_keywords_of(source: str, function: str) -> set[str]:
    """The keywords `function` passes to `OperatorContext(...)`, by AST.

    By AST rather than by hand: a hand-copied list is a second declaration of
    somebody else's contract and goes stale the moment core gains a field. Taking
    a SOURCE STRING rather than a path is what lets the predicate itself be
    table-tested below, against sources that differ in exactly one keyword.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and getattr(call.func, "id", None) == "OperatorContext"):
                    return {kw.arg for kw in call.keywords if kw.arg}
    raise AssertionError(f"no OperatorContext(...) call found in {function}")


def missing_context_keywords(production: set[str], mine: set[str]) -> set[str]:
    """What production sets and a timing builder does not, less the excused."""
    assert mine, "the builder does not construct an OperatorContext at all"
    return production - mine - JUSTIFIED_OMISSIONS


def extra_context_keywords(production: set[str], mine: set[str]) -> set[str]:
    """What a timing builder sets and production does NOT, less the excused.

    The other direction, and the one that cost a round. A field the timing path
    injects and the production path cannot express means the measurement describes
    a path nothing can execute -- which is worse than a missing field, because the
    number looks right.
    """
    assert mine, "the builder does not construct an OperatorContext at all"
    return mine - production - JUSTIFIED_EXTRAS


def _expand_one_context_keywords() -> set[str]:
    return context_keywords_of(
        (REPO / "src/aadistill/initialization/planning/search.py").read_text(),
        "_expand_one")


def _toy_items(n=4, vocab=64):
    g = torch.Generator().manual_seed(17)
    return [{"item_id": f"i{k}",
             "input_ids": torch.randint(1, vocab, (1, 10 + k), generator=g),
             "domain": "general" if k % 2 else "math",
             "subtype": "general" if k % 2 else "openmath"}
            for k in range(n)]


class TestTheBuilderMatchesTheProductionExpansion:

    @staticmethod
    def _both() -> tuple[set[str], set[str]]:
        return (_expand_one_context_keywords(),
                context_keywords_of(
                    (REPO / "scripts/pod/topk_adoption_driver.py").read_text(),
                    "production_operator_context"))

    def test_it_passes_every_keyword_production_passes(self):
        production, mine = self._both()
        assert not missing_context_keywords(production, mine), (
            "the production expansion passes a context field the timing builder "
            "does not. A timing run that omits a field production sets is not "
            "timing production.")

    def test_it_passes_no_scientific_field_production_cannot_express(self):
        """The direction that cost a round.

        The builder injected `distribution_support` while `SearchConfig` had no
        such field, so the measured operator path was unreachable through
        `BeamSearch`. A field here that core cannot carry means the timing
        describes a path nothing can run — and the number looks right.
        """
        production, mine = self._both()
        extra = extra_context_keywords(production, mine)
        assert not extra, (
            f"the timing builder passes {sorted(extra)} and the production "
            "expansion does not. Either core must carry it or the builder must "
            "stop injecting it; a measurement of an unreachable path prices "
            "nothing.")

    def test_every_omission_is_one_core_actually_passes(self):
        """So the exclusion list cannot quietly accumulate dead entries."""
        production = _expand_one_context_keywords()
        stale = JUSTIFIED_OMISSIONS - production
        assert not stale, (
            f"{sorted(stale)} is excused but production does not pass it; an "
            "exclusion list that outlives its reason hides the next real gap")

    def test_the_hashed_config_carries_the_policy_and_the_support(self, tmp_path):
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.scoring.positions import (
            SUPERVISED_TARGET_V1, policy_config,
        )
        from aadistill.initialization.specs.arch import ArchSpec, get_adapter
        from experiments.phase_d_series import scoring_protocol as SP

        register_builtin_adapters()
        model = build_tiny_model(GEOMETRY)
        items = _toy_items()
        ctx = drv.production_operator_context(
            adapter=get_adapter("qwen3"), model=model,
            target_spec=ArchSpec.of("qwen3", {**GEOMETRY,
                                             "num_hidden_layers": 4}),
            profile=None, items=items, seed=0, device="cpu",
            workdir=tmp_path / "w", policy=SUPERVISED_TARGET_V1)
        #: The declaration the state id will hash, and the object the operator
        #: will reduce over. `execute` refuses a disagreement, so both sides.
        assert ctx.config["distribution_support"] == \
            SP.D_SERIES_SUPPORT.as_dict()
        assert ctx.distribution_support is SP.D_SERIES_SUPPORT
        for key, value in policy_config(SUPERVISED_TARGET_V1).items():
            assert ctx.config[key] == value
        #: And what the policy READS, which `profile_hash` does not cover.
        assert any(k.startswith("scoring_content") for k in ctx.config), \
            sorted(ctx.config)
        assert ctx.execution.micro_batch_size == SP.D_SERIES_MICRO_BATCH_SIZE
        assert ctx.execution.calibration_batch_packing == \
            SP.D_SERIES_BATCH_PACKING


class TestTheParityPredicateItself:
    """The gate above is only worth its line if it bites. Table-tested.

    A previous round shipped a gate that passed on the first tree it saw and would
    have passed on a broken one too, so a new predicate gets mutated here rather
    than trusted.
    """

    CORE = """
def _expand_one(self):
    ctx = OperatorContext(adapter=a, model=m, parent_spec=p, target_spec=t,
                          profile=f, calibration_items=i, seed=s, device=d,
                          workdir=w, config=c, stats_cache=sc,
                          stats_cache_key=sk, execution=e,
                          position_policy=pp, deadline=dl)
"""

    def test_an_identical_builder_is_clean(self):
        mine = context_keywords_of(self.CORE, "_expand_one")
        assert missing_context_keywords(
            context_keywords_of(self.CORE, "_expand_one"), mine) == set()

    @pytest.mark.parametrize("dropped", [
        "config", "position_policy", "execution", "profile",
        "calibration_items", "seed", "target_spec",
    ])
    def test_dropping_any_load_bearing_keyword_is_caught(self, dropped):
        mine = context_keywords_of(self.CORE, "_expand_one") - {dropped}
        assert missing_context_keywords(
            context_keywords_of(self.CORE, "_expand_one"), mine) == {dropped}

    @pytest.mark.parametrize("excused", sorted(JUSTIFIED_OMISSIONS))
    def test_dropping_an_excused_keyword_is_allowed(self, excused):
        mine = context_keywords_of(self.CORE, "_expand_one") - {excused}
        assert missing_context_keywords(
            context_keywords_of(self.CORE, "_expand_one"), mine) == set()

    @pytest.mark.parametrize("injected", [
        "distribution_support", "position_policy", "numerics", "reduction",
    ])
    def test_a_field_the_timing_path_invents_is_caught(self, injected):
        """The real defect, simulated: core lacks it, the timing builder has it."""
        core = context_keywords_of(self.CORE, "_expand_one") - {injected}
        mine = context_keywords_of(self.CORE, "_expand_one") | {injected}
        assert extra_context_keywords(core, mine) == {injected}

    @pytest.mark.parametrize("excused", sorted(JUSTIFIED_EXTRAS))
    def test_an_excused_extra_is_allowed(self, excused):
        core = context_keywords_of(self.CORE, "_expand_one")
        assert extra_context_keywords(core, core | {excused}) == set()

    def test_every_excused_extra_is_one_the_builder_actually_passes(self):
        """So the extras list cannot rot into a blanket exemption."""
        mine = context_keywords_of(
            (REPO / "scripts/pod/topk_adoption_driver.py").read_text(),
            "production_operator_context")
        stale = JUSTIFIED_EXTRAS - mine
        assert not stale, (
            f"{sorted(stale)} is excused as an extra and the builder does not "
            "pass it; an exemption that outlives its reason hides the next gap")

    def test_an_identical_pair_has_neither_missing_nor_extra(self):
        core = context_keywords_of(self.CORE, "_expand_one")
        assert missing_context_keywords(core, core) == set()
        assert extra_context_keywords(core, core) == set()

    def test_a_new_core_field_is_caught(self):
        """The failure mode this exists for: core grows a field, nobody notices."""
        core = self.CORE.replace("deadline=dl)", "deadline=dl, numerics=n)")
        mine = context_keywords_of(self.CORE, "_expand_one")
        assert missing_context_keywords(
            context_keywords_of(core, "_expand_one"), mine) == {"numerics"}

    def test_a_builder_with_no_context_at_all_is_refused(self):
        with pytest.raises(AssertionError):
            missing_context_keywords({"config"}, set())


class TestTheRealOperatorRunsThroughThatBuilder:
    """On a toy model, for real: `execute`, not a stub.

    A stage whose only caller is a paid pod has its contract tested at $1.09/h.
    This runs the production path the stage runs -- registry lookup, `execute`,
    the observer, the artifacts the stage reads -- for nothing.
    """

    @staticmethod
    def _run(tmp_path):
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.operators.base import get_implementation
        from aadistill.initialization.operators.register import (
            register_builtin_operators,
        )
        from aadistill.initialization.scoring.positions import (
            SUPERVISED_TARGET_V1,
        )
        from aadistill.initialization.specs.arch import ArchSpec, get_adapter

        register_builtin_adapters()
        register_builtin_operators()
        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _toy_items()
        stamps: list[tuple[str, float]] = []
        import time as _t

        def stamp(**kw):
            stamps.append((repr(sorted(kw["skip"])), _t.perf_counter()))

        ctx = drv.production_operator_context(
            adapter=get_adapter("qwen3"), model=model,
            target_spec=ArchSpec.of("qwen3", {**GEOMETRY,
                                             "num_hidden_layers": 4}),
            profile=None, items=items, seed=0, device="cpu",
            workdir=tmp_path / "w", policy=SUPERVISED_TARGET_V1,
            observer=stamp)
        impl = get_implementation("depth.causal_kl_greedy_v1")
        #: `perf_counter`, the clock the observer stamps with. The stage takes the
        #: invocation total from `time.time()` -- production's definition -- and
        #: the span baseline from this one, and conflating them is the defect
        #: `TestTheSpanArithmetic` now refuses.
        span_base = _t.perf_counter()
        outcome = impl.execute(ctx)
        return outcome, stamps, span_base

    def test_the_artifacts_the_stage_reads_are_all_present(self, tmp_path):
        outcome, stamps, _ = self._run(tmp_path)
        timing = outcome.artifacts["timing"]
        assert outcome.artifacts["search_rounds"]
        assert outcome.artifacts["reference_cache"]["distribution_support"] == \
            "reference_topk_tail_v1"
        #: The attribution the record reports is derived from these two.
        assert timing["item_seconds"] > 0
        assert timing["candidate_subsets"] > 0
        assert stamps, "the observer never fired, so there are no spans"

    def test_the_spans_account_for_every_candidate_the_operator_scored(
            self, tmp_path):
        """The diagnostic distribution must not silently cover a subset."""
        outcome, stamps, started = self._run(tmp_path)
        spans = drv._candidate_spans(stamps, started)
        assert len(spans) == outcome.artifacts["timing"]["candidate_subsets"], (
            "the per-candidate spans and the operator's own candidate count "
            "disagree, so the distribution describes something else")
        #: Two rounds on this geometry: 6 layers to 4.
        assert len(outcome.artifacts["search_rounds"]) == 2

    def test_the_operator_passes_exactly_what_the_policy_returns(self, tmp_path):
        """The call a7 omitted, checked where it happens.

        On this toy mixture the treatment policy supervises every position and
        `prediction_weights_for` therefore returns `None` -- its "uniform" answer.
        That is a property of the DATA, not of the operator, so the assertion is
        that the operator hands the reducer the policy's answer rather than
        skipping the policy. The real mixture's answer is a tensor, and
        `TestTheCeilingArm` below measures that separately.
        """
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.operators.base import get_implementation
        from aadistill.initialization.operators.register import (
            register_builtin_operators,
        )
        from aadistill.initialization.scoring.batches import active_positions
        from aadistill.initialization.scoring.positions import (
            SUPERVISED_TARGET_V1,
        )
        from aadistill.initialization.specs.arch import ArchSpec, get_adapter

        register_builtin_adapters()
        register_builtin_operators()
        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _toy_items()
        seen: list[dict[str, Any]] = []
        ctx = drv.production_operator_context(
            adapter=get_adapter("qwen3"), model=model,
            target_spec=ArchSpec.of("qwen3", {**GEOMETRY,
                                             "num_hidden_layers": 5}),
            profile=None, items=items, seed=0, device="cpu",
            workdir=tmp_path / "w", policy=SUPERVISED_TARGET_V1,
            observer=lambda **kw: seen.append(kw))
        get_implementation("depth.causal_kl_greedy_v1").execute(ctx)
        assert seen, "no reduction was observed"
        active = active_positions(items, SUPERVISED_TARGET_V1)
        for call in seen:
            expected = active.prediction_weights_for(call["group"],
                                                     call["indices"])
            got = call["weights"]
            if expected is None:
                assert got is None, (
                    "the operator invented weights the policy did not give it")
            else:
                assert got is not None and torch.equal(got, expected), (
                    "the reducer was handed weights the policy did not produce")


class TestTheCeilingArm:
    """Which D1 arm the priced invocation must run, measured on the REAL items.

    Both arms reach the same weighted reducer, so the question is which one hands
    it more work. `$0`, on the host, because this is a property of the frozen
    calibration mixture and the policy -- no model and no GPU is involved.
    """

    @staticmethod
    def _groups_and_items(profile_id):
        from aadistill.initialization.calibration.items import (
            prepare_calibration_items,
        )
        from aadistill.initialization.calibration.packing import packed_batches
        from aadistill.initialization.calibration.profiles import get_profile
        from experiments.calibration import register_builtin_profiles
        from experiments.phase_d_series import scoring_protocol as SP

        register_builtin_profiles()
        profile = get_profile(profile_id)
        items = prepare_calibration_items(profile.resolve(REPO),
                                          profile_id=profile.qualified_id)
        groups = [(p.batch, p.original_indices) for p in packed_batches(
            items, SP.D_SERIES_MICRO_BATCH_SIZE,
            packing=SP.D_SERIES_BATCH_PACKING, pad_id=0, device="cpu")]
        return items, groups

    @pytest.mark.parametrize("profile_id", drv.D1_DEPTH_PROFILES)
    def test_the_treatment_weights_the_real_items_and_the_control_does_not(
            self, profile_id):
        from aadistill.initialization.scoring.batches import active_positions
        from aadistill.initialization.scoring.positions import (
            ALL_POSITIONS_V1, SUPERVISED_TARGET_V1,
        )

        items, groups = self._groups_and_items(profile_id)
        group, indices = groups[0]
        treatment = active_positions(items, SUPERVISED_TARGET_V1
                                     ).prediction_weights_for(group, indices)
        control = active_positions(items, ALL_POSITIONS_V1
                                   ).prediction_weights_for(group, indices)
        assert treatment is not None, (
            "the treatment does not weight the real mixture, so the priced "
            "invocation would not exercise the weighted path at all")
        assert int((treatment > 0).sum()) < treatment.numel(), (
            "every position is supervised, so the treatment is the control in "
            "disguise and the arm choice below means nothing")
        assert control is None, (
            "the control now produces weights too; the treatment is no longer "
            "known to be the more expensive arm and the ceiling must be "
            "re-measured on whichever is")

class TestTheSpanArithmetic:
    """Table-tested, because an off-by-one here mis-describes the distribution."""

    def test_one_candidate_one_group(self):
        spans = drv._candidate_spans([("[0]", 10.0)], started=9.0)
        assert spans == [{"skip": "[0]", "groups": 1, "seconds": 1.0}]

    def test_groups_of_one_candidate_collapse_to_its_last_stamp(self):
        spans = drv._candidate_spans(
            [("[0]", 1.0), ("[0]", 2.0), ("[0]", 4.0)], started=0.0)
        assert spans == [{"skip": "[0]", "groups": 3, "seconds": 4.0}]

    def test_each_candidate_starts_where_the_previous_one_ended(self):
        spans = drv._candidate_spans(
            [("[0]", 1.0), ("[0]", 2.0), ("[1]", 5.0), ("[1]", 6.0)],
            started=0.0)
        assert [s["seconds"] for s in spans] == [2.0, 4.0]
        assert [s["groups"] for s in spans] == [2, 2]
        #: The sum is the whole measured stretch, which is what makes it exact in
        #: aggregate even though a single span carries a neighbour's aggregation.
        assert sum(s["seconds"] for s in spans) == 6.0

    def test_a_repeated_skip_in_a_later_round_is_its_own_span(self):
        """Round 1 re-evaluates nothing round 0 did, but the key can repeat
        across rounds in principle and must not merge."""
        spans = drv._candidate_spans(
            [("[0]", 1.0), ("[1]", 2.0), ("[0]", 3.0)], started=0.0)
        assert [s["skip"] for s in spans] == ["[0]", "[1]", "[0]"]

    def test_a_baseline_from_the_other_clock_is_refused(self):
        """The defect this guard exists for, and it already shipped once.

        The stage took its baseline from `time.time()` while the observer stamped
        `perf_counter()`. The first span then read about -1.76e9 seconds and the
        distribution's min, p1, p5 and mean were all that number in a plausible
        shape. The invocation total was unaffected -- it is a `time.time()`
        interval, which is exactly what `search.py` records -- so the price was
        right and only the diagnostic lied, which is the worst version of this bug.
        """
        import time

        with pytest.raises(ValueError, match="different"):
            drv._candidate_spans([("[0]", time.perf_counter())],
                                 started=time.time())

    def test_a_single_impossible_stamp_is_enough_to_refuse(self):
        """Not only the first one: any stamp before the baseline is impossible."""
        with pytest.raises(ValueError, match="different"):
            drv._candidate_spans([("[0]", 10.0), ("[1]", 4.0)], started=5.0)

    def test_no_stamps_is_no_spans_not_a_crash(self):
        assert drv._candidate_spans([], started=0.0) == []


class TestThePricingGateRefusesEveryRecordThatCannotPrice:

    @staticmethod
    def _valid_record():
        return {"P_production_timing": {
            "measurement_path": "operator_execute_v1",
            "operator_invocation_seconds_max": 1500.0,
            "_valid_for_pricing": True,
            "sync_split_enabled": False,
            "per_profile": {
                "calib.domain_balanced@v1": {
                    "operator_invocation_seconds": 1400.0,
                    "candidate_subsets": 260},
                "calib.reasoning_heavy@v2": {
                    "operator_invocation_seconds": 1500.0,
                    "candidate_subsets": 260}}}}

    def _basis(self, tmp_path, record, monkeypatch):
        import json

        import write_d1_design as w

        path = tmp_path / "adoption.json"
        path.write_text(json.dumps(record))
        #: An ABSOLUTE override, so `REPO / TOPK_PRODUCTION` resolves to it.
        #: Repointing `REPO` instead would also repoint the committed telemetry
        #: the non-operator overhead is derived from, and the gate would refuse
        #: for a reason this test is not about.
        monkeypatch.setattr(w, "TOPK_PRODUCTION", str(path))
        return w._topk_production_basis()

    def test_the_valid_shape_prices_one_measured_invocation(self, tmp_path,
                                                           monkeypatch):
        basis = self._basis(tmp_path, self._valid_record(), monkeypatch)
        assert basis is not None
        #: 1500 s = 25 min, and NOT multiplied by any candidate count.
        assert basis["operator_minutes_at_max"] == pytest.approx(25.0)
        assert basis["candidate_subsets_measured"] == 260
        assert basis["root_max_minutes"] > basis["operator_minutes_at_max"], (
            "the cell must add the committed non-operator overhead")

    @pytest.mark.parametrize("mutate,why", [
        (lambda r: r["P_production_timing"].pop("measurement_path"),
         "a7's shape: no production-path marker"),
        (lambda r: r["P_production_timing"].__setitem__(
            "measurement_path", "candidate_loop_v1"),
         "a different measurement path"),
        (lambda r: r["P_production_timing"].pop(
            "operator_invocation_seconds_max"),
         "a per-candidate record with no invocation total"),
        (lambda r: r["P_production_timing"].__setitem__(
            "_valid_for_pricing", False),
         "explicitly not valid for pricing"),
        (lambda r: r["P_production_timing"].__setitem__(
            "sync_split_enabled", True),
         "a synced run"),
        (lambda r: r["P_production_timing"].pop("_valid_for_pricing"),
         "a record that does not say whether it synced"),
        (lambda r: r["P_production_timing"]["per_profile"].pop(
            "calib.reasoning_heavy@v2"),
         "only one of the two DEPTH profiles"),
        (lambda r: [v.__setitem__("candidate_subsets", 0)
                    for v in r["P_production_timing"]["per_profile"].values()],
         "no candidate subsets, so nothing was actually scored"),
    ])
    def test_it_refuses(self, tmp_path, monkeypatch, mutate, why):
        record = self._valid_record()
        mutate(record)
        assert self._basis(tmp_path, record, monkeypatch) is None, (
            f"the gate accepted a record that {why}")

    @pytest.mark.parametrize("subrun", ["a5", "a6", "a7"])
    def test_no_superseded_committed_record_can_price(self, subrun, monkeypatch):
        """The real files, not synthetics. Each of these was once the pointer.

        Named by path rather than read through the live pointer: this used to
        assert `_topk_production_basis() is None` while the pointer happened to be
        a7, so it silently became an assertion that the CURRENT basis is absent
        and failed the moment a valid one existed.
        """
        import write_d1_design as w

        path = (REPO / "logs/stages/stage-1/phase_d1/validations/topk-adoption/v1"
                / "runs" / subrun / "adoption.json")
        if not path.is_file():
            pytest.skip(f"{subrun} has no committed record")
        monkeypatch.setattr(w, "TOPK_PRODUCTION", str(path))
        assert w._topk_production_basis() is None, (
            f"{subrun} prices, so the envelope could resolve on a measurement of "
            "the shadow loop")

    def test_the_live_pointer_names_a_record_that_can_price(self):
        """The complement, and the thing the resolved envelope rests on."""
        import write_d1_design as w

        path = REPO / w.TOPK_PRODUCTION
        assert path.is_file(), f"the pointer names {w.TOPK_PRODUCTION}, absent"
        basis = w._topk_production_basis()
        if basis is None:
            #: A legitimate state -- it is what UNRESOLVED means -- so this
            #: asserts the PAIR agrees rather than demanding a basis exist.
            import json
            design = json.loads(
                (REPO / "logs/stages/stage-1/phase_d1/plans/d1_design.json"
                 ).read_text())
            assert design["budget"]["per_session_envelope_compatibility"] == \
                "UNRESOLVED"
            return
        assert basis["operator_invocation_seconds_max"] > 0
        assert basis["candidate_subsets_measured"] > 0
        #: And it is a WHOLE invocation, not a per-candidate product: the minutes
        #: must equal the seconds over sixty, with nothing multiplied in. The
        #: tolerance bounds exactly one thing -- the basis rounds minutes to four
        #: decimals, so half of the last digit is the whole permitted difference.
        assert basis["operator_minutes_at_max"] == pytest.approx(
            basis["operator_invocation_seconds_max"] / 60.0, abs=5e-5)



class TestTheCloseoutAndTheDesignCannotDisagreeAboutThePrice:
    """One cross-record invariant, because keeping them aligned by hand failed.

    The closeout stated a production cost and a `$23.0738` repricing while the
    design's own gate refused that record as a pricing basis. Nothing checked the
    pair, so the contradiction lived in two committed files at once and had to be
    found by reading them.

    What must hold, in both directions: a closeout answer that presents a live
    production cost requires the design to have a basis, and a design with a basis
    requires the closeout's cited price to be the design's.
    """

    @staticmethod
    def _records():
        import json

        import write_d1_design as w

        closeout = json.loads(
            (REPO / "logs/stages/stage-1/phase_d1/validations/topk-adoption/v1"
                    "/closeout.json").read_text())
        design = json.loads(
            (REPO / "logs/stages/stage-1/phase_d1/plans/d1_design.json").read_text())
        return closeout, design, w._topk_production_basis()

    def test_a_withdrawn_cost_answer_says_so_where_a_reader_lands(self):
        closeout, design, basis = self._records()
        answers = closeout.get("answers") or {}
        cost = answers.get("E_production_cost")
        if basis is not None:
            return  # covered by the next test
        assert "_REOPENED_BY_REVIEW" in closeout, (
            "the design has no valid production basis, so the closeout's cost "
            "answer is withdrawn -- and the closeout must say that at its top "
            "level, not leave a reader to infer it from the design")
        withdrawn = closeout["_REOPENED_BY_REVIEW"]
        assert "what_was_withdrawn" in withdrawn and withdrawn["what_was_withdrawn"]
        assert "why" in withdrawn and len(withdrawn["why"]) > 80, (
            "a withdrawn answer must carry the reason; 'superseded' is not one")
        #: And the figures it withdrew must be named, not just alluded to.
        assert cost is None or any(
            str(k).startswith("_") or k in ("from", "operator_seconds_max",
                                            "operator_invocation_seconds_max")
            for k in cost), sorted(cost)

    def test_when_a_basis_exists_the_two_records_cite_the_same_price(self):
        closeout, design, basis = self._records()
        if basis is None:
            pytest.skip("no production basis; the previous test owns this state")
        priced = design["budget"]["topk_production_basis"]["search_session"][
            "hard_ceiling_usd"]
        repricing = (closeout.get("answers") or {}).get("D1_repricing") or {}
        cited = repricing.get("search_session_hard_ceiling_usd")
        assert cited is not None, (
            "a production basis exists and the closeout's D1_repricing cites no "
            "session price under `search_session_hard_ceiling_usd`; a reader "
            "would have to go to the design to learn what the round concluded")
        assert float(cited) == pytest.approx(float(priced)), (
            f"the closeout cites ${float(cited):.4f} and the design prices "
            f"${float(priced):.4f}; one of them is stale")
        #: And the operator term, which is the thing this round re-measured.
        cited_op = repricing.get("operator_invocation_seconds_max")
        if cited_op is not None:
            assert float(cited_op) == pytest.approx(
                float(basis["operator_invocation_seconds_max"]))

    def test_the_closeouts_scientific_answers_survive_a_withdrawn_cost(self):
        """Withdrawing the price must not withdraw the science."""
        closeout, _, _ = self._records()
        answers = closeout.get("answers") or {}
        for key in ("A_reference_top200_mass", "B_full_vs_k_plus_1",
                    "C_depth_decisions", "D_state_evaluation", "lower_bound"):
            assert key in answers, f"{key} went missing from the closeout"
        assert answers["lower_bound"], "the lower-bound answer is the verdict"

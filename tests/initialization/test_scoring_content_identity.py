"""The two identities that close the scoring-metadata collisions.

`scoring_content_identity` answers *what will be scored*; `measurement_protocol_id`
answers *under what protocol a state measurement was taken*. Both exist because a
chain of hashes that looked complete was not:

* a calibration mixture's `content_sha256` covers item ids and token ids, so two
  assets with identical tokens and DIFFERENT supervised masks shared every hash
  in the operator path and could produce different DEPTH/FFN/WIDTH/ATTENTION
  decisions under one state id;
* `scripts/shared/evaluation/load_state_eval.py` built its suite without the manifest's
  `content_sha256`, so `suite_hash` described the suite's declared SHAPE and not
  its prompts — and the resume path compares exactly that field.

Each test below is a collision that must not be possible, not a property that
happens to hold.
"""

from __future__ import annotations

import json

import pytest
import torch

from aadistill.initialization.scoring.content import (
    CONTENT_CONFIG_KEY,
    scoring_content_config,
    scoring_content_identity,
    scoring_content_report,
)
from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1,
    SUPERVISED_TARGET_V1,
    ScoringPositionError,
)
from aadistill.initialization.scoring.protocol_identity import (
    AGGREGATION_RULE,
    PROTOCOL_FIELD,
    UNBOUND_SUITE_CONTENT,
    UNDECLARED_EXECUTION,
    ReductionSemantics,
    measurement_is_comparable,
    measurement_protocol_id,
)

SUPERVISION_TAG = "assistant"


def _item(item_id="a-0", *, ids=(5, 6, 7, 8, 9), supervised=(2, 3),
          domain="general", subtype="text", extra_tags=None):
    """One calibration-shaped item with an explicit supervised mask."""
    tokens = torch.tensor([list(ids)], dtype=torch.long)
    mask = torch.zeros(tokens.shape[1] - 1, dtype=torch.bool)
    for index in supervised:
        mask[index] = True
    tags = {SUPERVISION_TAG: mask}
    tags.update(extra_tags or {})
    return {"item_id": item_id, "ids": list(ids), "input_ids": tokens,
            "domain": domain, "subtype": subtype, "tags": tags}


# --- the collision that motivated the whole mechanism -----------------------


class TestTheMaskCollision:
    """Same ids, same tokens, different supervised positions."""

    def test_changing_one_supervised_position_forks_the_identity(self):
        before = scoring_content_identity([_item(supervised=(2, 3))],
                                          SUPERVISED_TARGET_V1)
        after = scoring_content_identity([_item(supervised=(2,))],
                                         SUPERVISED_TARGET_V1)
        assert before != after, (
            "two assets with identical item ids and identical token ids but "
            "different supervised masks share every other hash in the operator "
            "path; if they share this one too, one state id covers two "
            "different scientific objectives")

    def test_the_tokens_are_identical_so_nothing_else_could_see_it(self):
        """The premise of the collision, asserted rather than assumed."""
        from aadistill.initialization.calibration.profiles import (
            mixture_content_sha256,
        )

        a, b = [_item(supervised=(2, 3))], [_item(supervised=(2,))]
        assert mixture_content_sha256(a) == mixture_content_sha256(b), (
            "if the mixture identity already separated these, this module "
            "would be closing a gap that does not exist")
        assert a[0]["item_id"] == b[0]["item_id"]
        assert ALL_POSITIONS_V1.policy_hash == ALL_POSITIONS_V1.policy_hash

    def test_moving_a_supervised_position_forks_it_too(self):
        """Not just the COUNT: which positions, specifically."""
        two_early = scoring_content_identity([_item(supervised=(0, 1))],
                                             SUPERVISED_TARGET_V1)
        two_late = scoring_content_identity([_item(supervised=(2, 3))],
                                            SUPERVISED_TARGET_V1)
        assert two_early != two_late

    def test_an_absent_tag_and_an_empty_tag_differ(self):
        """`positions.supervised_target_v1` treats them differently, so the
        identity has to tell them apart."""
        empty = _item(supervised=())
        absent = _item(supervised=())
        del absent["tags"][SUPERVISION_TAG]
        assert scoring_content_identity([empty], SUPERVISED_TARGET_V1) != \
               scoring_content_identity([absent], SUPERVISED_TARGET_V1)

    def test_the_incumbent_policy_is_blind_to_the_mask_by_construction(self):
        """And that is correct, not an oversight: it reads no tags, so it has
        nothing to collide on — which is why historical state ids stay
        derivable."""
        assert ALL_POSITIONS_V1.reads_tags == ()
        assert scoring_content_identity([_item(supervised=(2, 3))],
                                        ALL_POSITIONS_V1) == \
               scoring_content_identity([_item(supervised=(2,))],
                                        ALL_POSITIONS_V1)

    def test_a_token_change_still_forks_both_policies(self):
        """The mask is an ADDITIONAL term, not a replacement for the tokens."""
        for policy in (ALL_POSITIONS_V1, SUPERVISED_TARGET_V1):
            assert scoring_content_identity([_item(ids=(5, 6, 7, 8, 9))], policy) \
                != scoring_content_identity([_item(ids=(5, 6, 7, 8, 10))], policy)


class TestWhatElseIsBound:

    def test_the_subtype_label_is_bound(self):
        """Labels decide the aggregation, so they are part of what is measured."""
        assert scoring_content_identity([_item(subtype="text")],
                                        SUPERVISED_TARGET_V1) != \
               scoring_content_identity([_item(subtype="arith")],
                                        SUPERVISED_TARGET_V1)

    def test_item_order_is_content(self):
        """`original_order_v1` groups consecutive items, so order is execution."""
        a, b = _item("a-0"), _item("a-1", supervised=(1,))
        assert scoring_content_identity([a, b], SUPERVISED_TARGET_V1) != \
               scoring_content_identity([b, a], SUPERVISED_TARGET_V1)

    def test_a_policy_reading_another_tag_binds_that_tag_instead(self):
        """Generic for D2/D3: `assertion` appears nowhere in the identity code.

        The identity asks the POLICY what it reads. A policy declaring a
        different tag binds a different mask with no edit to `content.py`.
        """
        class ReadsFinalAnswer(type(SUPERVISED_TARGET_V1)):
            policy_id = "positions.reads_final_answer_test"
            version = 1

            @property
            def reads_tags(self):
                return ("final_answer",)

        policy = ReadsFinalAnswer()
        answer_a = torch.tensor([True, False, False, False])
        answer_b = torch.tensor([False, True, False, False])
        a = _item(extra_tags={"final_answer": answer_a})
        b = _item(extra_tags={"final_answer": answer_b})
        assert scoring_content_identity([a], policy) != \
               scoring_content_identity([b], policy)
        #: And the supervised policy cannot see that difference, because it
        #: does not read that tag — which is the point of asking the policy.
        assert scoring_content_identity([a], SUPERVISED_TARGET_V1) == \
               scoring_content_identity([b], SUPERVISED_TARGET_V1)

    def test_an_unidentified_item_is_refused(self):
        item = _item()
        del item["item_id"]
        with pytest.raises(ScoringPositionError, match="item_id"):
            scoring_content_identity([item], SUPERVISED_TARGET_V1)

    def test_an_empty_asset_is_refused(self):
        with pytest.raises(ScoringPositionError, match="empty"):
            scoring_content_identity([], SUPERVISED_TARGET_V1)

    def test_reserialization_cannot_move_it(self):
        """Nothing arbitrary is hashed: not JSON, not key order, not whitespace."""
        item = _item()
        reordered = {k: item[k] for k in reversed(list(item))}
        assert scoring_content_identity([item], SUPERVISED_TARGET_V1) == \
               scoring_content_identity([reordered], SUPERVISED_TARGET_V1)


class TestTheConfigAndTheReport:

    def test_the_config_key_is_absent_under_the_incumbent(self):
        """So every committed state id stays derivable from live code."""
        assert scoring_content_config([_item()], ALL_POSITIONS_V1) == {}

    def test_the_config_key_is_present_under_a_target_aware_policy(self):
        config = scoring_content_config([_item()], SUPERVISED_TARGET_V1)
        assert set(config) == {CONTENT_CONFIG_KEY}
        assert config[CONTENT_CONFIG_KEY] == \
               scoring_content_identity([_item()], SUPERVISED_TARGET_V1)

    def test_the_report_gives_a_reader_something_to_check_the_hash_against(self):
        report = scoring_content_report([_item(supervised=(2, 3))],
                                        SUPERVISED_TARGET_V1)
        assert report["n_items"] == 1
        assert report["prediction_positions"] == 4
        assert report["active_prediction_positions"] == 2
        assert report["reads_tags"] == [SUPERVISION_TAG]
        assert report["position_policy"] == SUPERVISED_TARGET_V1.qualified_id


# --- the measurement protocol ----------------------------------------------


REDUCTION = ReductionSemantics(chunk=64, reference_strategy="cache_in_memory")


def _protocol(**overrides):
    terms = {"suite_structural_identity": "suite-shape",
             "suite_content_identity": "suite-abc",
             "scoring_content_identity": "scoring-abc",
             "position_policy_hash": "policy-abc",
             "reduction": REDUCTION,
             "execution_fingerprint": "fp-abc"}
    terms.update(overrides)
    return measurement_protocol_id(**terms)


class TestTheMeasurementProtocolId:

    def test_every_term_moves_it(self):
        """A term that could not move the id would be decoration."""
        base = _protocol()
        assert _protocol(suite_structural_identity="other") != base
        assert _protocol(suite_content_identity="other") != base
        assert _protocol(scoring_content_identity="other") != base
        assert _protocol(position_policy_hash="other") != base
        assert _protocol(execution_fingerprint="other") != base
        assert _protocol(reduction=ReductionSemantics(
            chunk=128, reference_strategy="cache_in_memory")) != base
        assert _protocol(reduction=ReductionSemantics(
            chunk=64, reference_strategy="recompute")) != base
        assert _protocol(reduction=ReductionSemantics(
            chunk=64, reference_strategy="cache_in_memory",
            aggregation_rule="something_else/v1")) != base

    def test_it_is_stable_and_deterministic(self):
        assert _protocol() == _protocol()
        assert len(_protocol()) == 32

    @pytest.mark.parametrize("term", ["suite_structural_identity",
                                      "suite_content_identity",
                                      "scoring_content_identity",
                                      "position_policy_hash",
                                      "execution_fingerprint"])
    def test_an_absent_term_is_refused_rather_than_defaulted(self, term):
        with pytest.raises(ValueError, match=term):
            _protocol(**{term: ""})

    def test_the_aggregation_rule_is_part_of_the_default(self):
        """The two-level unweighted mean is a semantic, not an implementation
        detail: a different rule computes a different quantity."""
        assert REDUCTION.as_dict()["aggregation_rule"] == AGGREGATION_RULE


class TestComparability:
    """`measurement_is_comparable` replaced two independent `_restore` checks.
    These are the cases that used to be spread across them, plus the three the
    list did not have yet."""

    def _record(self, *, protocol=None, suite="suite-abc", policy="policy-abc"):
        detail = {"position_policy_hash": policy}
        if protocol:
            detail[PROTOCOL_FIELD] = protocol
        return {"detail": detail, "suite_hash": suite}

    def test_the_same_protocol_is_comparable(self):
        ok, why = measurement_is_comparable(
            self._record(protocol=_protocol()), protocol_id=_protocol())
        assert ok and "same measurement protocol" in why

    def test_a_different_protocol_is_not(self):
        ok, why = measurement_is_comparable(
            self._record(protocol=_protocol(suite_content_identity="other")),
            protocol_id=_protocol())
        assert not ok and "protocol" in why

    def test_a_declaring_run_does_not_adopt_an_undeclared_record(self):
        """The record predates the identity; its reduction and execution were
        never written down, so nothing can show it comparable."""
        ok, why = measurement_is_comparable(self._record(),
                                            protocol_id=_protocol())
        assert not ok
        assert "predates" in why and "reconstructing" in why

    def test_an_undeclared_run_does_not_adopt_a_protocol_record(self):
        ok, why = measurement_is_comparable(self._record(protocol=_protocol()),
                                            protocol_id=None)
        assert not ok and "declares none" in why

    def test_a_historical_record_keeps_its_historical_semantics(self):
        """Explicitly, not by silent reinterpretation: the two fields such a
        record DOES carry are the two it is judged on."""
        ok, why = measurement_is_comparable(
            self._record(), protocol_id=None,
            historical_suite_hash="suite-abc", historical_policy_hash="policy-abc")
        assert ok and "historical" in why

        ok, why = measurement_is_comparable(
            self._record(suite="other"), protocol_id=None,
            historical_suite_hash="suite-abc", historical_policy_hash="policy-abc")
        assert not ok and "suite" in why

        ok, why = measurement_is_comparable(
            self._record(policy="other"), protocol_id=None,
            historical_suite_hash="suite-abc", historical_policy_hash="policy-abc")
        assert not ok and "position policy" in why

    def test_a_pre_policy_record_omitting_the_field_is_still_readable(self):
        """Every record written before the policy existed was measured over all
        positions; reading the omission as anything else would refuse every
        historical resume."""
        record = {"detail": {}, "suite_hash": "suite-abc"}
        ok, _ = measurement_is_comparable(
            record, protocol_id=None, historical_suite_hash="suite-abc",
            historical_policy_hash=ALL_POSITIONS_V1.policy_hash)
        assert ok

    def test_no_record_is_not_comparable(self):
        ok, why = measurement_is_comparable(None, protocol_id=None)
        assert not ok and why == "no record"

    def test_it_returns_a_reason_in_every_branch(self):
        """A resume that declines silently is indistinguishable from one that
        found nothing, and this project has spent rounds on that difference."""
        for record, protocol in ((None, None),
                                 (self._record(), _protocol()),
                                 (self._record(protocol=_protocol()), None),
                                 (self._record(protocol="x" * 32), _protocol()),
                                 (self._record(), None)):
            _, why = measurement_is_comparable(
                record, protocol_id=protocol,
                historical_suite_hash="suite-abc",
                historical_policy_hash="policy-abc")
            assert why and len(why) > 5


class TestTheEvaluatorStampsIt:

    def _evaluator(self, **overrides):
        from aadistill.initialization.planning.metrics import StateEvaluator
        from aadistill.initialization.specs.metrics import (
            StateEvalSuite,
            SuiteItem,
        )

        suite = overrides.pop("suite", None) or StateEvalSuite(
            suite_id="t.suite", version=1, domains=("general",),
            subtypes={"general": ("text",)}, critical_tags=("eos_like",),
            content_sha256="c" * 64, n_items=1)
        items = overrides.pop("items", None) or [SuiteItem(
            item_id="text-0", input_ids=torch.tensor([[5, 6, 7, 8, 9]]),
            domain="general", subtype="text",
            tags={"eos_like": torch.tensor([False, False, True, False]),
                  SUPERVISION_TAG: torch.tensor([False, True, True, True])})]
        return StateEvaluator(suite, items, device="cpu", vocab_size=16,
                              **overrides), suite, items

    def test_every_evaluation_carries_the_protocol_it_was_taken_under(self):
        evaluator, _, _ = self._evaluator()
        assert evaluator.measurement_protocol_id
        assert len(evaluator.measurement_protocol_id) == 32

    def test_an_undeclared_numerical_environment_is_stated_not_omitted(self):
        """"Nobody wrote down the numerics" is itself part of the identity."""
        evaluator, suite, items = self._evaluator()
        assert evaluator.numerics is None
        expected = measurement_protocol_id(
            suite_structural_identity=suite.suite_hash,
            suite_content_identity=suite.content_sha256,
            scoring_content_identity=scoring_content_identity(
                items, ALL_POSITIONS_V1),
            position_policy_hash=ALL_POSITIONS_V1.policy_hash,
            reduction=evaluator.reduction,
            execution_fingerprint=UNDECLARED_EXECUTION)
        assert evaluator.measurement_protocol_id == expected

    def test_the_mask_content_moves_the_measurement_identity(self):
        """The whole point of item 3 reaching item 4: a suite whose tokens are
        unchanged and whose supervised mask moved is a different measurement."""
        from aadistill.initialization.specs.metrics import SuiteItem

        def items_with(mask):
            return [SuiteItem(
                item_id="text-0", input_ids=torch.tensor([[5, 6, 7, 8, 9]]),
                domain="general", subtype="text",
                tags={"eos_like": torch.tensor([False, False, True, False]),
                      SUPERVISION_TAG: torch.tensor(mask)})]

        a, _, _ = self._evaluator(items=items_with([False, True, True, True]),
                                  position_policy=SUPERVISED_TARGET_V1)
        b, _, _ = self._evaluator(items=items_with([False, False, True, True]),
                                  position_policy=SUPERVISED_TARGET_V1)
        assert a.measurement_protocol_id != b.measurement_protocol_id

        #: And the incumbent policy, which reads no tags, is unmoved — so no
        #: historical measurement identity shifts underneath a closed result.
        c, _, _ = self._evaluator(items=items_with([False, True, True, True]))
        d, _, _ = self._evaluator(items=items_with([False, False, True, True]))
        assert c.measurement_protocol_id == d.measurement_protocol_id

    def test_the_suite_content_moves_it_without_moving_the_suite_hash(self):
        """The whole shape of the 2026-10-03 repair. `suite_hash` is the
        STRUCTURAL identity and 45 committed records pin it, so the content is
        bound in the protocol id — which is new and pins nothing historical —
        rather than folded into the structural hash."""
        a, suite_a, _ = self._evaluator(suite_content_sha256="c" * 64)
        b, suite_b, _ = self._evaluator(suite_content_sha256="d" * 64)
        assert a.measurement_protocol_id != b.measurement_protocol_id
        assert suite_a.suite_hash == suite_b.suite_hash, (
            "the structural identity must not move with the content, or every "
            "record that pinned it has been reinterpreted")

    def test_the_structural_identity_moves_it_too(self):
        from aadistill.initialization.specs.metrics import StateEvalSuite

        def suite_with(critical):
            return StateEvalSuite(
                suite_id="t.suite", version=1, domains=("general",),
                subtypes={"general": ("text",)}, critical_tags=critical,
                content_sha256="c" * 64, n_items=1)

        a, _, _ = self._evaluator(suite=suite_with(("eos_like",)))
        b, _, _ = self._evaluator(suite=suite_with(("eos_like", "answer_like")))
        assert a.measurement_protocol_id != b.measurement_protocol_id

    def test_unbound_content_is_recorded_rather_than_refused(self):
        """An in-memory suite has no frozen content record. The literal keeps
        the omission visible instead of making it indistinguishable from a
        content-bound measurement."""
        from aadistill.initialization.specs.metrics import StateEvalSuite

        bare = StateEvalSuite(
            suite_id="t.suite", version=1, domains=("general",),
            subtypes={"general": ("text",)}, critical_tags=("eos_like",),
            n_items=1)
        evaluator, _, items = self._evaluator(suite=bare)
        assert evaluator.suite_content_sha256 is None
        assert evaluator.measurement_protocol_id == measurement_protocol_id(
            suite_structural_identity=bare.suite_hash,
            suite_content_identity=UNBOUND_SUITE_CONTENT,
            scoring_content_identity=scoring_content_identity(
                items, ALL_POSITIONS_V1),
            position_policy_hash=ALL_POSITIONS_V1.policy_hash,
            reduction=evaluator.reduction,
            execution_fingerprint=UNDECLARED_EXECUTION)

    def test_the_chunk_size_moves_it(self):
        """Chunking is not a formality: this project measured it mattering at
        ~9e-8 relative, the same order as the drift budget."""
        a, _, _ = self._evaluator(chunk=64)
        b, _, _ = self._evaluator(chunk=128)
        assert a.measurement_protocol_id != b.measurement_protocol_id
        assert a.reduction.chunk == 64 and b.reduction.chunk == 128

    def test_the_execution_moves_it_when_the_numerics_are_declared(self):
        from aadistill.initialization.planning.metrics import REFERENCE_EXECUTION
        from aadistill.initialization.calibration.packing import LENGTH_SORTED_V1
        from aadistill.initialization.execution import ExecutionConfig
        from aadistill.initialization.specs.materialization import (
            NumericalEnvironment,
        )
        numerics = NumericalEnvironment(
            device_type="cpu", compute_dtype="float32",
            accumulation_dtype="float32")
        a, _, _ = self._evaluator(execution=REFERENCE_EXECUTION,
                                  numerics=numerics)
        b, _, _ = self._evaluator(
            execution=ExecutionConfig(micro_batch_size=3,
                                      calibration_batch_packing=LENGTH_SORTED_V1),
            numerics=numerics)
        assert a.measurement_protocol_id != b.measurement_protocol_id
        #: And neither equals the undeclared one: a run that states its numerics
        #: and one that does not are measuring under different stated conditions.
        undeclared, _, _ = self._evaluator(execution=REFERENCE_EXECUTION)
        assert undeclared.measurement_protocol_id not in {
            a.measurement_protocol_id, b.measurement_protocol_id}


class TestTheStateEvalLoaderRequiresTheContentHash:
    """It must EXIST, and it must not be folded into the structural hash.

    The first version of this repair passed the manifest's `content_sha256`
    into `StateEvalSuite`, which moved `suite_hash` for an unchanged asset —
    and `suite_hash` is the structural identity that 45 committed records pin
    and that the C2 baseline driver refuses a measurement against. The suite's
    content belongs in the measurement protocol identity, which is new.
    """

    def _asset(self, tmp_path, *, content="a" * 64, omit=False):
        root = tmp_path / "state_eval_v1"
        root.mkdir(parents=True)
        manifest = {"suite_id": "t.suite", "version": 1,
                    "domains": {"general": ["text"]},
                    "critical_tags": ["eos_like"], "general_domain": "general"}
        if not omit:
            manifest["content_sha256"] = content
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "items.jsonl").write_text(json.dumps({
            "item_id": "text-0", "ids": [5, 6, 7, 8, 9], "domain": "general",
            "subtype": "text", "n_prediction_positions": 4,
            "tags": {"eos_like": [2]}}) + "\n")
        return root

    def _load(self, root):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "load_state_eval_under_test",
            "scripts/shared/evaluation/load_state_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load(root)

    def test_the_content_hash_is_returned_for_the_caller_to_bind(self, tmp_path):
        suite, items, manifest = self._load(self._asset(tmp_path))
        assert manifest["content_sha256"] == "a" * 64
        assert len(items) == 1

    def test_it_is_NOT_folded_into_the_structural_suite_hash(self, tmp_path):
        """Two assets differing only in content must keep ONE structural
        identity, because that is what the structural identity means and what
        every record pinning it recorded."""
        a, _, _ = self._load(self._asset(tmp_path / "a", content="a" * 64))
        b, _, _ = self._load(self._asset(tmp_path / "b", content="b" * 64))
        assert a.content_sha256 is None and b.content_sha256 is None
        assert a.suite_hash == b.suite_hash

    def test_the_frozen_asset_still_hashes_to_its_pinned_value(self):
        """The regression that would have caught the first repair. The live
        loader must reproduce the structural hash the Phase-A driver pins."""
        from pathlib import Path

        driver_pin = "6421fa4cf12ee2a16f452557c486aa95beb37e4aac4f7c7fd72d380993b39833"
        asset = Path(__file__).resolve().parents[2] / "artifacts/stages/stage-1/state_eval_v1"
        if not (asset / "manifest.json").is_file():
            pytest.skip("the frozen state_eval asset is not staged here")
        suite, _, manifest = self._load(asset)
        assert suite.suite_hash == driver_pin
        #: And the content is still available to bind — it is simply not in the
        #: structural hash.
        assert manifest["content_sha256"]

    def test_a_manifest_without_a_content_hash_is_refused(self, tmp_path):
        """The loader's own obligation: the value must EXIST, so a caller
        cannot end up binding `None` into a measurement identity."""
        with pytest.raises(ValueError, match="content"):
            self._load(self._asset(tmp_path, omit=True))


class TestHistoricalIdentitiesStayDerivable:
    """Every committed state id must still be computable from live code.

    This is the compatibility requirement all three new keys are omitted for.
    A state id is a hash of the operator configs along its path, so one extra
    key at the incumbent policy would move every id this project has recorded —
    and the C2 replay specs, which pin attempt 3's artifact digests step by
    step, would be replaying paths that no longer exist.

    `BeamSearch._position_policy_config` composes exactly the two functions
    below and returns `{}` as soon as the first is empty, so the composition is
    what these tests exercise; the search-level consequence — that the
    incumbent's `config_hash` omits the fields entirely — is asserted in
    `test_target_aware_scoring_end_to_end.py`.
    """

    def test_the_incumbent_contributes_no_key_from_either_function(self):
        from aadistill.initialization.scoring.positions import policy_config

        assert policy_config(ALL_POSITIONS_V1) == {}
        assert scoring_content_config([_item()], ALL_POSITIONS_V1) == {}

    def test_a_target_aware_policy_contributes_all_three(self):
        """And the treatment is visible rather than silent: the policy id, its
        hash and the scoring content all enter the hashed config."""
        from aadistill.initialization.scoring.positions import policy_config

        merged = {**policy_config(SUPERVISED_TARGET_V1),
                  **scoring_content_config([_item()], SUPERVISED_TARGET_V1)}
        assert set(merged) == {"position_policy", "position_policy_hash",
                               CONTENT_CONFIG_KEY}

    def test_the_incumbent_policy_reads_no_position_metadata_at_all(self):
        """Which is WHY omitting the keys is sound and not merely convenient:
        there is no mask for the incumbent to collide on."""
        assert ALL_POSITIONS_V1.reads_tags == ()

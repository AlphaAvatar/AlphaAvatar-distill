"""The position policy: what it computes, what it refuses, and what it hashes.

The policy decides which predictions a structural decision is allowed to care
about, so three properties are load-bearing and each is asserted rather than
described:

* the incumbent policy is **numerically inert** — it reports `form == "all"` on
  both axes, which is the flag every reducer reads to take its untouched path;
* the supervised-target policy reads the **mixture's own** `assistant` tag and
  falls back to every prediction position for an untemplated item, so it changes
  the scoring and not the data;
* an id is permanently bound to a declaration, and a config that names one is
  resolved from the registry rather than from a caller's object.

The real frozen mixtures are exercised directly, not imitated. Their tags are
the artifact this policy is defined over, and a toy fixture that invented tags
would prove the code runs rather than that it reads the right field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.scoring.batches import (  # noqa: E402
    ActivePositions, active_positions, consecutive_indices,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    ALL_POSITIONS_V1,
    FORM_ALL,
    FORM_SELECT,
    FORM_WEIGHTED,
    PREDICTION_AXIS,
    SUPERVISED_TARGET_V1,
    TOKEN_AXIS,
    ScoringPositionError,
    ScoringPositionPolicy,
    WEIGHT_DTYPE,
    get_position_policy,
    normalized_prediction_tags,
    policy_config,
    register_position_policy,
    registered_position_policies,
    resolve_position_policy,
    unregister_position_policy,
    weights_for_items,
)

#: The two mixtures a search actually calibrates on. Skipped rather than faked
#: when absent: the point of reading them is that they are the frozen artifact.
MIXTURES = ("artifacts/stage1/e8_calibration_v1/items.jsonl",
            "artifacts/stage1/reasoning_heavy_v2/items.jsonl")

#: THE PATHS ARE LITERAL IN THE CONDITION, deliberately. `audit_skip_predicates`
#: resolves a filesystem premise by reading the path out of the predicate's
#: source and asking whether the git index or the setup manifest puts it on a
#: pod. A premise built from a loop variable is one it cannot read, and it
#: reports UNRESOLVED — "classified, but nothing says the two machines must
#: decide it the same way". Writing the literals here is what makes the parity
#: derivable instead of owed.
requires_the_frozen_mixtures = pytest.mark.skipif(
    not (REPO / "artifacts/stage1/e8_calibration_v1/items.jsonl").is_file()
    or not (REPO / "artifacts/stage1/reasoning_heavy_v2/items.jsonl").is_file(),
    reason=("the frozen calibration mixtures are gitignored out-of-tree "
            "assets; a checkout without them has no tags to read"))


def load_mixture(rel: str):
    path = REPO / rel
    raw = [json.loads(line) for line in path.read_text().splitlines()
           if line.strip()]
    return prepare_calibration_items(raw, profile_id=rel)


def item(tokens: int, *, assistant_from: int | None = None, item_id="i",
         domain="general", subtype="text"):
    """One operator-shaped item, optionally with an `assistant` tag.

    Tags are stored as PREDICTION-POSITION INDICES, which is the frozen
    mixtures' convention (`tag_positions` in the builder) and the one a policy
    has to read correctly for any of this to mean anything.
    """
    out = {"item_id": item_id, "domain": domain, "subtype": subtype,
           "input_ids": torch.arange(tokens, dtype=torch.long)[None, :],
           "tags": {}}
    if assistant_from is not None:
        out["tags"]["assistant"] = list(range(assistant_from, tokens - 1))
    return out


class TestTheIncumbentPolicyIsInert:
    def test_both_axes_are_all_ones_at_the_real_length(self):
        w_tok = ALL_POSITIONS_V1.weights(item(10), axis=TOKEN_AXIS)
        w_pred = ALL_POSITIONS_V1.weights(item(10), axis=PREDICTION_AXIS)
        assert w_tok.n_positions == 10 and w_pred.n_positions == 9
        assert w_tok.form == FORM_ALL and w_pred.form == FORM_ALL
        assert w_tok.total == 10.0 and w_pred.total == 9.0

    def test_the_token_axis_keeps_the_final_token(self):
        """Which is the whole reason the two axes are separate.

        An activation statistic has always summed over all `L` token positions.
        The incumbent policy must keep saying so, or the refactor that introduced
        this abstraction would silently drop one token per item from every
        committed statistic.
        """
        w = ALL_POSITIONS_V1.weights(item(5), axis=TOKEN_AXIS)
        assert bool(w.active_mask().all())
        assert w.n_active == 5

    def test_form_all_is_what_the_reducers_branch_on(self):
        """Not a cosmetic label: `form` is read to decide whether to multiply."""
        active = ActivePositions([item(6), item(8)], ALL_POSITIONS_V1)
        assert active.all_token_positions_active
        assert active.all_prediction_positions_active


class TestTheSupervisedTargetPolicy:
    def test_it_reads_the_assistant_tag(self):
        w = SUPERVISED_TARGET_V1.weights(item(20, assistant_from=12),
                                         axis=PREDICTION_AXIS)
        assert w.form == FORM_SELECT
        assert w.n_active == 7                       # positions 12..18
        assert [int(i) for i in w.active_mask().nonzero().flatten()] == \
            list(range(12, 19))

    def test_an_untagged_item_keeps_every_prediction_position(self):
        """An untemplated document has no assistant turn and every one of its
        next-token predictions is a legitimate supervised position. Zeroing it
        would drop the general-language domain out of the mixture, which is a
        DATA change this policy must not make."""
        w = SUPERVISED_TARGET_V1.weights(item(20), axis=PREDICTION_AXIS)
        assert w.form == FORM_ALL
        assert w.n_active == 19

    def test_a_present_but_empty_tag_is_REFUSED_not_treated_as_untagged(self):
        """The distinction the content identity already claimed to make.

        `scoring/content.py::_position_component` binds `assistant=absent` and
        `assistant=0:<digest>` to different identities, on the stated grounds
        that this policy treats them differently. **It did not** — one `or`
        collapsed both into the all-positions fallback — so an asset whose
        `assistant` tag named nothing was scored FULL-SEQUENCE under a
        target-aware policy id. This test asserts the fail-closed semantics and
        is the reason this file no longer contains
        `test_an_empty_tag_is_treated_as_untagged`, which encoded the defect.
        """
        raw = item(12)
        raw["tags"]["assistant"] = []
        with pytest.raises(ScoringPositionError, match="selects no prediction"):
            SUPERVISED_TARGET_V1.weights(raw, axis=PREDICTION_AXIS)

    def test_the_two_empty_cases_are_not_the_same_case(self):
        """Absent falls back; present-and-empty refuses. Asserted together so a
        future change cannot quietly re-merge them."""
        absent = item(12)
        assert absent["tags"] == {}, "the helper's untagged item must carry no tag"
        assert SUPERVISED_TARGET_V1.weights(
            absent, axis=PREDICTION_AXIS).form == FORM_ALL

        present_empty = item(12)
        present_empty["tags"]["assistant"] = []
        with pytest.raises(ScoringPositionError):
            SUPERVISED_TARGET_V1.weights(present_empty, axis=PREDICTION_AXIS)

    def test_an_all_false_mask_is_refused_as_well_as_an_empty_index_list(self):
        """The two STORED forms of "present but empty" must behave alike.

        `normalized_prediction_tags` accepts a tag either as prediction-position
        indices (the frozen mixtures' form) or as a boolean mask (a loaded
        `SuiteItem`'s form), and turns an empty index list into a correct-length
        all-False mask. So the length check cannot see this case, and a policy
        that only refused one form would refuse the mixture but not the loaded
        suite — or the reverse.
        """
        as_mask = item(12)
        as_mask["tags"]["assistant"] = torch.zeros(11, dtype=torch.bool)
        with pytest.raises(ScoringPositionError, match="selects no prediction"):
            SUPERVISED_TARGET_V1.weights(as_mask, axis=PREDICTION_AXIS)

    def test_the_token_axis_refuses_it_too(self):
        """Not only the prediction axis: the token-axis vector is derived from
        the prediction mask, so a refusal there must not be bypassable by
        asking for activations instead."""
        raw = item(12)
        raw["tags"]["assistant"] = []
        with pytest.raises(ScoringPositionError, match="selects no prediction"):
            SUPERVISED_TARGET_V1.weights(raw, axis=TOKEN_AXIS)

    def test_the_token_axis_drops_the_final_token(self):
        """The documented consequence of defining the statistic over positions
        whose PREDICTION is supervised: the last token of an item feeds no
        prediction inside the item."""
        w = SUPERVISED_TARGET_V1.weights(item(20), axis=TOKEN_AXIS)
        assert w.n_positions == 20
        assert w.n_active == 19
        assert not bool(w.active_mask()[-1])

    def test_the_two_axes_agree_about_which_prediction_is_active(self):
        raw = item(20, assistant_from=12)
        tok = SUPERVISED_TARGET_V1.weights(raw, axis=TOKEN_AXIS)
        pred = SUPERVISED_TARGET_V1.weights(raw, axis=PREDICTION_AXIS)
        assert torch.equal(tok.active_mask()[:-1], pred.active_mask())
        assert tok.total == pred.total

    def test_a_tag_longer_than_the_item_raises(self):
        raw = item(10)
        raw["tags"]["assistant"] = list(range(40))
        with pytest.raises(ScoringPositionError, match="outside"):
            SUPERVISED_TARGET_V1.weights(raw, axis=PREDICTION_AXIS)


@requires_the_frozen_mixtures
class TestAgainstTheFrozenMixtures:
    @pytest.mark.parametrize("rel", MIXTURES)
    def test_the_incumbent_policy_reproduces_the_recorded_totals(self, rel):
        """`n_tokens` and `n_prediction_positions` are fields of the frozen
        mixture. The incumbent policy's two totals must equal them exactly, item
        for item, or this abstraction does not describe the statistic it
        replaced."""
        items = load_mixture(rel)
        tok = weights_for_items(items, ALL_POSITIONS_V1, axis=TOKEN_AXIS)
        pred = weights_for_items(items, ALL_POSITIONS_V1, axis=PREDICTION_AXIS)
        assert sum(w.total for w in tok) == sum(int(i["n_tokens"]) for i in items)
        assert sum(w.total for w in pred) == sum(
            int(i["n_prediction_positions"]) for i in items)

    @pytest.mark.parametrize("rel", MIXTURES)
    def test_the_supervised_policy_restricts_and_keeps_every_item_scorable(self, rel):
        items = load_mixture(rel)
        pred = weights_for_items(items, SUPERVISED_TARGET_V1,
                                 axis=PREDICTION_AXIS)
        total_all = sum(int(i["n_prediction_positions"]) for i in items)
        total_sup = sum(w.total for w in pred)
        #: A real restriction, and not a near-total one: both frozen mixtures sit
        #: around three quarters. A policy that restricted to a handful of
        #: positions, or to none, would be a different experiment.
        assert 0.5 * total_all < total_sup < total_all
        #: Every item still carries weight. `weights_for_items` enforces it, and
        #: this asserts the frozen mixtures actually satisfy it rather than that
        #: the enforcement exists.
        assert all(w.total > 0 for w in pred)

    @pytest.mark.parametrize("rel", MIXTURES)
    def test_no_frozen_item_carries_a_present_but_empty_assistant_tag(self, rel):
        """What makes the fail-closed policy safe for the frozen assets.

        `tag_positions` ends with `{k: v for k, v in tags.items() if v}` and
        returns `{}` outright for raw prose, so it never serializes an empty
        tag — but that is the BUILDER, and these are the ARTIFACTS, which were
        written by whatever the builder was at the time. Measured here instead:
        every row is `absent` or non-empty, so refusing present-but-empty
        refuses nothing that exists. If this ever fails, the policy change and
        the asset have to be reconciled before either is used.
        """
        offenders = [raw.get("item_id")
                     for raw in load_mixture(rel)
                     if "assistant" in (raw.get("tags") or {})
                     and not (raw["tags"]["assistant"])]
        assert not offenders, (
            f"{rel} carries a present-but-empty `assistant` tag on "
            f"{offenders}; SupervisedTargetV1 refuses those")

    @pytest.mark.parametrize("rel", MIXTURES)
    def test_the_restriction_is_exactly_the_assistant_tag_plus_untagged_items(
            self, rel):
        """Derived independently from the mixture's own fields, so the policy is
        checked against the artifact rather than against itself.

        The two cases are spelled out rather than collapsed into `if tag`: a
        falsy tag used to mean both "absent" and "present but empty", and those
        are now different outcomes. Only `absent` contributes the item's full
        prediction count; a present-but-empty tag would raise, which the test
        above establishes does not occur in these assets.
        """
        items = load_mixture(rel)
        expected = 0
        for raw in items:
            tags = raw.get("tags") or {}
            if "assistant" in tags:
                expected += len(tags["assistant"])
            else:
                expected += int(raw["n_prediction_positions"])
        got = sum(w.total for w in weights_for_items(
            items, SUPERVISED_TARGET_V1, axis=PREDICTION_AXIS))
        assert got == expected


class TestTagNormalization:
    def test_index_lists_and_boolean_masks_reconcile(self):
        """The two stored forms — a mixture's index list and a SuiteItem's mask —
        must read identically. The state-eval loader's own comment warns that
        reading one as the other "would silently reweight the critical-token
        metric"."""
        indices = normalized_prediction_tags({"assistant": [1, 3, 5]}, 8)
        mask = torch.zeros(8, dtype=torch.bool)
        mask[[1, 3, 5]] = True
        same = normalized_prediction_tags({"assistant": mask}, 8)
        assert torch.equal(indices["assistant"], same["assistant"])

    def test_a_wrong_length_mask_raises(self):
        with pytest.raises(ScoringPositionError, match="mask over"):
            normalized_prediction_tags(
                {"assistant": torch.ones(3, dtype=torch.bool)}, 8)

    def test_a_negative_index_raises(self):
        with pytest.raises(ScoringPositionError, match="outside"):
            normalized_prediction_tags({"assistant": [-1]}, 8)


class TestIdentityAndRegistry:
    def test_the_hash_covers_the_declaration(self):
        assert ALL_POSITIONS_V1.policy_hash != SUPERVISED_TARGET_V1.policy_hash
        declared = ALL_POSITIONS_V1.declare()
        assert declared["policy_hash"] == ALL_POSITIONS_V1.policy_hash
        assert "untagged_behaviour" in declared

    def test_both_shipped_policies_are_registered(self):
        assert ALL_POSITIONS_V1.qualified_id in registered_position_policies()
        assert SUPERVISED_TARGET_V1.qualified_id in registered_position_policies()
        assert get_position_policy(
            SUPERVISED_TARGET_V1.qualified_id) is SUPERVISED_TARGET_V1

    def test_rebinding_an_id_to_a_different_declaration_raises(self):
        class Impostor(ScoringPositionPolicy):
            policy_id = SUPERVISED_TARGET_V1.policy_id
            version = SUPERVISED_TARGET_V1.version
            reads_tags = ("final_answer",)
            untagged_behaviour = "something else"

            def prediction_weights(self, *, n_predictions, tags):
                return torch.ones(n_predictions, dtype=WEIGHT_DTYPE)

            def token_weights(self, *, n_tokens, tags):
                return torch.ones(n_tokens, dtype=WEIGHT_DTYPE)

        with pytest.raises(ScoringPositionError, match="new version"):
            register_position_policy(Impostor())

    def test_the_incumbent_policy_names_nothing_in_a_config(self):
        """The compatibility guarantee, as an assertion. Every committed state
        hashed an operator config without a policy key, so the incumbent must
        contribute none — otherwise a historical state id stops being derivable
        from live code."""
        assert policy_config(ALL_POSITIONS_V1) == {}

    def test_a_named_policy_round_trips_through_a_config(self):
        cfg = policy_config(SUPERVISED_TARGET_V1)
        assert cfg and resolve_position_policy(cfg) is SUPERVISED_TARGET_V1

    def test_an_absent_policy_resolves_to_the_incumbent(self):
        assert resolve_position_policy({}) is ALL_POSITIONS_V1
        assert resolve_position_policy(None) is ALL_POSITIONS_V1

    def test_a_hash_that_disagrees_with_the_registry_raises(self):
        with pytest.raises(ScoringPositionError, match="rebound"):
            resolve_position_policy({
                "position_policy": SUPERVISED_TARGET_V1.qualified_id,
                "position_policy_hash": "0" * 64})

    def test_a_hash_with_no_id_raises(self):
        with pytest.raises(ScoringPositionError, match="names nothing"):
            resolve_position_policy({"position_policy_hash": "abc"})

    def test_an_unregistered_id_raises_rather_than_defaulting(self):
        unregister_position_policy("positions.never@v1")
        with pytest.raises(KeyError):
            resolve_position_policy({"position_policy": "positions.never@v1"})


class TestWeightValidation:
    def test_a_negative_weight_is_refused(self):
        class Negative(ScoringPositionPolicy):
            policy_id, version = "positions.negative_test", 1
            untagged_behaviour = "n/a"

            def prediction_weights(self, *, n_predictions, tags):
                return torch.full((n_predictions,), -1.0, dtype=WEIGHT_DTYPE)

            def token_weights(self, *, n_tokens, tags):
                return torch.ones(n_tokens, dtype=WEIGHT_DTYPE)

        with pytest.raises(ScoringPositionError, match="sign flip"):
            Negative().weights(item(6), axis=PREDICTION_AXIS)

    def test_a_wrong_length_return_is_refused(self):
        class WrongLength(ScoringPositionPolicy):
            policy_id, version = "positions.wrong_length_test", 1
            untagged_behaviour = "n/a"

            def prediction_weights(self, *, n_predictions, tags):
                return torch.ones(n_predictions + 1, dtype=WEIGHT_DTYPE)

            def token_weights(self, *, n_tokens, tags):
                return torch.ones(n_tokens, dtype=WEIGHT_DTYPE)

        with pytest.raises(ScoringPositionError, match="weights for"):
            WrongLength().weights(item(6), axis=PREDICTION_AXIS)

    def test_an_all_zero_item_is_refused_rather_than_divided_by(self):
        class Nothing(ScoringPositionPolicy):
            policy_id, version = "positions.nothing_test", 1
            untagged_behaviour = "n/a"

            def prediction_weights(self, *, n_predictions, tags):
                return torch.zeros(n_predictions, dtype=WEIGHT_DTYPE)

            def token_weights(self, *, n_tokens, tags):
                return torch.zeros(n_tokens, dtype=WEIGHT_DTYPE)

        with pytest.raises(ScoringPositionError, match="weight zero"):
            weights_for_items([item(6)], Nothing(), axis=PREDICTION_AXIS)

    def test_a_one_token_item_predicts_nothing_and_raises(self):
        with pytest.raises(ScoringPositionError, match="predicts nothing"):
            ALL_POSITIONS_V1.weights(item(1), axis=PREDICTION_AXIS)


class TestContinuousWeightsAreRefusedWhereOnlyAMaskWorks:
    """The boundary the activation collectors declare, asserted by name.

    A confidence-weighted policy is the D2 direction and the API expresses it.
    The collectors do not implement it — their divisor is an integer token count
    — and the refusal has to be explicit, because rounding a confidence weight to
    a mask would run a different experiment and report it as the right one.
    """

    class Confidence(ScoringPositionPolicy):
        policy_id, version = "positions.confidence_test", 1
        untagged_behaviour = "n/a"

        def prediction_weights(self, *, n_predictions, tags):
            return torch.linspace(0.1, 0.9, n_predictions, dtype=WEIGHT_DTYPE)

        def token_weights(self, *, n_tokens, tags):
            return torch.linspace(0.1, 0.9, n_tokens, dtype=WEIGHT_DTYPE)

    def test_the_form_is_classified_as_weighted(self):
        w = self.Confidence().weights(item(8), axis=TOKEN_AXIS)
        assert w.form == FORM_WEIGHTED

    def test_the_token_mask_refuses_with_what_is_owed(self):
        active = ActivePositions([item(8)], self.Confidence())
        with pytest.raises(ScoringPositionError, match="StatsSpec"):
            active.require_binary_token_weights()

    def test_but_the_prediction_weights_are_available(self):
        """A weighted next-token objective needs no schema change, because its
        denominator is formed per row. So D2's beam metric is reachable even
        though its activation statistics are not."""
        active = ActivePositions([item(8)], self.Confidence())
        assert not active.all_prediction_positions_active
        assert active.prediction_weights_for_item(0) is not None


class TestRowOrderIsNotMixtureOrder:
    """`original_indices` is the only thing that says which item a row holds.

    A consumer that assumed row `r` was item `r` would score one item's
    supervised positions against another's activations, and the result would
    still look like a ranking. So the lookup is validated.
    """

    def _batch(self, items, indices):
        from aadistill.initialization.calibration.batching import build_batch
        return build_batch([items[i] for i in indices], pad_id=0)

    def test_the_mask_follows_the_permutation(self):
        items = [item(20, assistant_from=15, item_id="a"),
                 item(20, assistant_from=5, item_id="b")]
        active = active_positions(items, SUPERVISED_TARGET_V1)
        swapped = self._batch(items, (1, 0))
        mask = active.token_mask_for(swapped, (1, 0))
        #: Row 0 now holds item "b", supervised from prediction position 5 to 18
        #: inclusive — 14 positions. Row 1 holds "a", supervised from 15: four.
        assert int(mask[0].sum()) == 14
        assert int(mask[1].sum()) == 4
        #: And the other way round, to show the assertion is about the
        #: permutation rather than about which row happens to be longer.
        unswapped = active.token_mask_for(self._batch(items, (0, 1)), (0, 1))
        assert int(unswapped[0].sum()) == 4
        assert int(unswapped[1].sum()) == 14

    def test_a_mismatched_permutation_raises(self):
        items = [item(20, assistant_from=15), item(20, assistant_from=5)]
        active = active_positions(items, SUPERVISED_TARGET_V1)
        batch = self._batch(items, (0, 1))
        with pytest.raises(ScoringPositionError, match="mismatched permutation"):
            active.token_mask_for(batch, (0,))

    def test_an_out_of_range_index_raises(self):
        items = [item(20, assistant_from=15), item(20, assistant_from=5)]
        active = active_positions(items, SUPERVISED_TARGET_V1)
        batch = self._batch(items, (0, 1))
        with pytest.raises(ScoringPositionError, match="outside"):
            active.token_mask_for(batch, (0, 7))

    def test_consecutive_indices_describes_an_unpermuted_batch(self):
        items = [item(20, assistant_from=15), item(20, assistant_from=5)]
        batch = self._batch(items, (0, 1))
        assert consecutive_indices(batch) == (0, 1)


class TestMixedForms:
    def test_a_mixture_is_as_general_as_its_most_general_item(self):
        """The frozen mixtures make this concrete: under the supervised policy
        the untemplated items are `all` and the templated ones are `select`. A
        caller reading the first item's form would have skipped the mask over
        two thirds of the corpus."""
        items = [item(20, item_id="untagged"),
                 item(20, assistant_from=10, item_id="tagged")]
        active = active_positions(items, SUPERVISED_TARGET_V1)
        assert active.token_form == FORM_SELECT
        assert not active.all_token_positions_active

    def test_active_positions_of_nothing_is_none(self):
        assert active_positions([], SUPERVISED_TARGET_V1) is None

    def test_the_report_states_both_axes_and_the_policy(self):
        items = [item(20, item_id="untagged"),
                 item(20, assistant_from=10, item_id="tagged")]
        report = active_positions(items, SUPERVISED_TARGET_V1).report()
        assert report["position_policy"] == SUPERVISED_TARGET_V1.qualified_id
        assert report["position_policy_hash"] == SUPERVISED_TARGET_V1.policy_hash
        #: 19 untagged predictions + 9 tagged ones.
        assert report["prediction_axis"]["active"] == 28
        assert report["prediction_axis"]["positions"] == 38
        assert report["token_axis"]["active"] == 28
        assert report["token_axis"]["positions"] == 40

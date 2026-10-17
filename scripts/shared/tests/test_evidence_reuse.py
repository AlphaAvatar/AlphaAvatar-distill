"""Reuse is decided by identity, never by a name, a path or a seed.

`shared.evaluation.evidence_reuse.classify` decides whether completed
behavioural work can stand in for a requested unit. Each outcome has a test,
and so does each way a partial match could be mistaken for a full one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for extra in ("src", "scripts"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

from shared.evaluation import evidence_reuse as R  # noqa: E402

TRAINING = ("initialization_digest", "recipe_digest", "data_digest", "seed",
            "trainer_source_digest")
EVALUATION = ("evaluation_protocol_hash",)


def _training(**change):
    base = {"initialization_digest": "init-B", "recipe_digest": "recipe",
            "data_digest": "pack", "seed": 7, "trainer_source_digest": "trainer"}
    return {**base, **change}


def _evaluation(**change):
    return {"evaluation_protocol_hash": "eval-A", **change}


def _unit(uid="u1", *, result=True, checkpoint=False, training=None,
          evaluation=None):
    return R.UnitEvidence(unit_id=uid, training=training or _training(),
                          evaluation=evaluation or _evaluation(),
                          trained_sha256="bytes-" + uid,
                          result_verified=result,
                          checkpoint_verified=checkpoint)


def classify(training, evaluation, units):
    return R.classify(training, evaluation, units, training_keys=TRAINING,
                      evaluation_keys=EVALUATION)


def test_identical_identity_reuses_the_result_and_names_its_bytes():
    got = classify(_training(), _evaluation(), [_unit()])
    assert got["outcome"] == R.REUSE_RESULT
    assert got["trained_sha256"] == "bytes-u1"


def test_a_retired_checkpoint_does_not_stop_a_result_being_reused():
    got = classify(_training(), _evaluation(), [_unit(checkpoint=False)])
    assert got["outcome"] == R.REUSE_RESULT


def test_a_new_evaluation_protocol_reuses_bytes_that_still_exist():
    got = classify(_training(), _evaluation(evaluation_protocol_hash="eval-B"),
                   [_unit(checkpoint=True)])
    assert got["outcome"] == R.REUSE_CHECKPOINT_REEVALUATE
    assert got["differs"] == ["evaluation_protocol_hash"]


def test_a_new_evaluation_protocol_with_retired_bytes_must_retrain():
    """Retraining is a new draw, so it is TRAIN_FROM_INIT, never a reuse."""
    got = classify(_training(), _evaluation(evaluation_protocol_hash="eval-B"),
                   [_unit(checkpoint=False)])
    assert got["outcome"] == R.TRAIN_FROM_INIT
    assert "not bit-reproducible" in got["why"]


@pytest.mark.parametrize("field", ["recipe_digest", "data_digest",
                                   "trainer_source_digest"])
def test_the_same_seed_with_any_other_difference_is_not_the_same_unit(field):
    got = classify(_training(**{field: "changed"}), _evaluation(), [_unit()])
    assert got["outcome"] == R.TRAIN_FROM_INIT
    assert got["differs"] == [field]


def test_the_nearest_unit_names_the_difference_not_a_union():
    units = [_unit("s7"), _unit("s9", training=_training(seed=9))]
    got = classify(_training(seed=11), _evaluation(), units)
    assert got["outcome"] == R.TRAIN_FROM_INIT
    assert got["differs"] == ["seed"]


def test_a_different_initialization_recomputes_everything():
    got = classify(_training(initialization_digest="init-new"), _evaluation(),
                   [_unit()])
    assert got["outcome"] == R.RECOMPUTE_ALL


def test_a_partial_request_is_refused_not_matched():
    request = _training()
    del request["trainer_source_digest"]
    got = classify(request, _evaluation(), [_unit()])
    assert got["outcome"] == R.REFUSED


def test_evidence_that_does_not_record_a_field_cannot_match():
    partial = _training()
    partial["trainer_source_digest"] = None
    got = classify(_training(), _evaluation(), [_unit(training=partial)])
    assert got["outcome"] == R.REFUSED


def test_unverified_result_evidence_is_refused():
    got = classify(_training(), _evaluation(), [_unit(result=False)])
    assert got["outcome"] == R.REFUSED


def test_the_initialization_field_is_mandatory():
    with pytest.raises(ValueError):
        R.classify(_training(), _evaluation(), [], training_keys=("seed",),
                   evaluation_keys=EVALUATION)

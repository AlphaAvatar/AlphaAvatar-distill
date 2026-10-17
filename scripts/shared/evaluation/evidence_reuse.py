"""Which completed behavioural work a requested unit may reuse, by identity.

A behavioural unit is two computations with two identities:

    training    (initialization, recovery recipe, recovery data, seed,
                 trainer implementation, training runtime)  ->  trained bytes
    evaluation  (trained bytes, evaluation protocol)        ->  generations,
                                                                per-sample rows

so whether earlier work can stand in for a requested unit is decided by
comparing those identities -- never by experiment name, file path or seed
alone. Two facts make the comparison asymmetric:

* **Training is not bit-reproducible on CUDA.** Retraining a unit with an
  identical training identity yields a NEW draw, not the artifact that was
  evaluated. So an evaluation result belongs to the specific trained bytes it
  measured (`trained_sha256`), and a result whose checkpoint has been retired
  can still be reused -- but nothing can re-evaluate it under a different
  protocol, because the bytes it measured no longer exist.
* **A matching seed is not a matching unit.** Seed is one field of the
  training identity among several; it is compared like the others.

Five outcomes, in the order a requested unit falls through them:

    REUSE_RESULT                  training and evaluation identity both match
                                  and the result evidence is verified: consume
                                  the rows, compute nothing
    REUSE_CHECKPOINT_REEVALUATE   training identity matches, the evaluation
                                  protocol differs, and the trained bytes are
                                  present and hash-verified: generate and score
                                  only
    TRAIN_FROM_INIT               the initialization matches but the training
                                  identity does not, or it matches and the
                                  evaluation differs while the checkpoint is
                                  retired: train (a new draw), then evaluate
    RECOMPUTE_ALL                 no evidence shares the initialization: the
                                  model weights differ, so nothing downstream
                                  of them is reusable
    REFUSED                       an identity is incomplete or the evidence
                                  fails verification; reuse is never inferred
                                  from a partial match

Everything instance-specific -- which fields a protocol pins and what their
values are -- is passed in. Nothing here names an experiment, a model or a
store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

REUSE_RESULT = "REUSE_RESULT"
REUSE_CHECKPOINT_REEVALUATE = "REUSE_CHECKPOINT_REEVALUATE"
TRAIN_FROM_INIT = "TRAIN_FROM_INIT"
RECOMPUTE_ALL = "RECOMPUTE_ALL"
REFUSED = "REFUSED"

OUTCOMES = (REUSE_RESULT, REUSE_CHECKPOINT_REEVALUATE, TRAIN_FROM_INIT,
            RECOMPUTE_ALL, REFUSED)

#: The field that ties a training identity to model weights. Kept separate from
#: the rest of the training identity because it decides a different outcome: a
#: different initialization invalidates everything, a different recipe only the
#: training.
INIT_FIELD = "initialization_digest"


@dataclass(frozen=True)
class UnitEvidence:
    """What a completed unit left behind, as identities and verifications.

    `training` and `evaluation` are flat field -> value maps. The verification
    flags are the CALLER's measurements (hashes recomputed against the bytes),
    not claims copied from a record.
    """

    unit_id: str
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    trained_sha256: str | None
    result_verified: bool
    checkpoint_verified: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _missing(identity: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [k for k in keys if identity.get(k) in (None, "")]


def _differ(a: Mapping[str, Any], b: Mapping[str, Any],
            keys: Sequence[str]) -> list[str]:
    return [k for k in keys if a.get(k) != b.get(k)]


def classify(requested_training: Mapping[str, Any],
             requested_evaluation: Mapping[str, Any],
             candidates: Sequence[UnitEvidence], *,
             training_keys: Sequence[str],
             evaluation_keys: Sequence[str]) -> dict[str, Any]:
    """The cheapest admissible way to obtain one requested unit.

    `training_keys` must include :data:`INIT_FIELD`. Every key the protocol
    pins must be present on the request: a request that leaves a field unset
    could match evidence produced under any value of it, which is the partial
    match this refuses.
    """
    if INIT_FIELD not in training_keys:
        raise ValueError(f"training_keys must include {INIT_FIELD!r}")
    gaps = (_missing(requested_training, training_keys)
            + _missing(requested_evaluation, evaluation_keys))
    if gaps:
        return {"outcome": REFUSED, "unit": None,
                "why": f"the request leaves {sorted(gaps)} unset; a partial "
                       "identity cannot be matched"}

    same_init = [c for c in candidates
                 if c.training.get(INIT_FIELD) == requested_training[INIT_FIELD]]
    if not same_init:
        return {"outcome": RECOMPUTE_ALL, "unit": None,
                "why": "no completed unit shares the initialization digest; "
                       "the model weights differ, so training and evaluation "
                       "must both run"}

    best_reject: dict[str, Any] | None = None
    for c in same_init:
        incomplete = (_missing(c.training, training_keys)
                      + _missing(c.evaluation, evaluation_keys))
        if incomplete:
            best_reject = best_reject or {
                "outcome": REFUSED, "unit": c.unit_id,
                "why": f"evidence {c.unit_id} does not record "
                       f"{sorted(incomplete)}; it cannot be shown to match"}
            continue
        t_diff = _differ(requested_training, c.training, training_keys)
        if t_diff:
            continue
        e_diff = _differ(requested_evaluation, c.evaluation, evaluation_keys)
        if not e_diff:
            if c.result_verified:
                return {"outcome": REUSE_RESULT, "unit": c.unit_id,
                        "trained_sha256": c.trained_sha256,
                        "why": "training and evaluation identity match and "
                               "the result evidence re-hashes to its record"}
            best_reject = {"outcome": REFUSED, "unit": c.unit_id,
                           "why": f"identity matches {c.unit_id} but its "
                                  "result evidence did not verify"}
            continue
        if c.checkpoint_verified:
            return {"outcome": REUSE_CHECKPOINT_REEVALUATE, "unit": c.unit_id,
                    "trained_sha256": c.trained_sha256,
                    "differs": e_diff,
                    "why": "the trained bytes exist and verify; only "
                           f"{sorted(e_diff)} differ, so generate and score"}
        return {"outcome": TRAIN_FROM_INIT, "unit": c.unit_id,
                "differs": e_diff,
                "why": f"{c.unit_id} was trained under this identity but "
                       f"evaluated under a different {sorted(e_diff)}, and its "
                       "checkpoint is retired. Retraining yields a new draw "
                       "(training is not bit-reproducible), not the bytes "
                       f"{c.unit_id} measured"}
    if best_reject is not None:
        return best_reject
    #: The NEAREST unit names what actually differs; a union over every unit
    #: sharing the initialization would also list the seeds of all the others.
    nearest = min(same_init, key=lambda c: (
        len(_differ(requested_training, c.training, training_keys)),
        c.unit_id))
    differs = _differ(requested_training, nearest.training, training_keys)
    return {"outcome": TRAIN_FROM_INIT, "unit": None, "nearest": nearest.unit_id,
            "differs": differs,
            "why": "the initialization is reusable but no completed unit "
                   f"matches the training identity (nearest, "
                   f"{nearest.unit_id}, differs in {differs})"}


__all__ = ["INIT_FIELD", "OUTCOMES", "RECOMPUTE_ALL", "REFUSED",
           "REUSE_CHECKPOINT_REEVALUATE", "REUSE_RESULT", "TRAIN_FROM_INIT",
           "UnitEvidence", "classify"]

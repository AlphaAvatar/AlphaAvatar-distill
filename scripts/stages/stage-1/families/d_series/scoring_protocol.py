"""The D-series scoring protocol. One place, shared by D1, D2 and D3.

Maintainer decision 2026-10-04: the D-series reduces KL over the Top-K entries of
the REFERENCE distribution plus one aggregate tail bucket, at ``K = 200``.

``K = 200`` IS A MAINTAINER-SELECTED PROTOCOL PARAMETER. There is no sweep, and it
must never be tuned against D1/D2/D3 results — a K chosen from outcomes would make
every downstream comparison a selection effect. It lives here, in the experiment
layer, because it is this campaign's policy and not a framework default;
``src/aadistill`` contains no value for it and a core test asserts that.

D2 and D3 consume this same constant and this same core primitive. Their declared
changes are confidence weighting, not a different divergence, so a second Top-K
implementation anywhere in the D-series would be a defect.

What this does NOT change: the behavioural battery family. Top-K alters structural
scoring and evaluation, not prompt content, so ``family_content_id``
``1e3445f1b676…74cd58`` stands. Changing it is a separate maintainer decision.
"""
from __future__ import annotations

from typing import Any

from aadistill.initialization.scoring.support import (
    DistributionSupport, reference_topk_tail,
)

#: The D-series' K. Maintainer-selected 2026-10-04; never tuned against outcomes.
D_SERIES_TOP_K = 200

#: The protocol every D-series round scores under.
D_SERIES_SUPPORT: DistributionSupport = reference_topk_tail(D_SERIES_TOP_K)

#: The execution protocol adopted alongside it, carried here so a round cannot
#: pick up one and miss the other. These are EXECUTION knobs, not hashed science
#: — but the full-vocab qualification measured that the calibration batch size
#: moves three of four fixed-path operator selections, so a round that changed it
#: silently would change its initialization.
D_SERIES_MICRO_BATCH_SIZE = 3
D_SERIES_BATCH_PACKING = "length_sorted_v1"


def operator_config(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``config`` fragment an operator needs to DECLARE this support.

    The declaration and the handed object are two sides of one contract and
    ``OperatorImplementation.execute`` refuses a mismatch — so this returns the
    declaration rather than leaving each caller to spell it, which is how the two
    sides drift apart.
    """
    out = dict(extra or {})
    out["distribution_support"] = D_SERIES_SUPPORT.as_dict()
    return out


def describe() -> dict[str, Any]:
    """The protocol as a record, for a design or a closeout to carry."""
    return {
        "protocol_id": D_SERIES_SUPPORT.support_id,
        "top_k": D_SERIES_TOP_K,
        "support": ("the Top-200 entries of the REFERENCE distribution at each "
                    "prediction position"),
        "tail": ("every vocabulary entry outside that support, aggregated into "
                 "one K+1 bucket"),
        "reference_for_depth": ("the intact current parent; the candidate is that "
                                "parent with the candidate block bypassed. NOT "
                                "the original teacher -- DEPTH's objective is "
                                "local and parent-relative, and forcing the "
                                "teacher's support onto it would score a "
                                "different question"),
        "reference_for_state_evaluation": "the original teacher",
        "exact_under_this_protocol": ["target-token CE/NLL", "top-1 agreement"],
        "coarsened_under_this_protocol": ["forward KL", "reverse KL",
                                          "critical-token KL"],
        "reverse_kl_partition": ("the SAME reference-defined K+1 partition as the "
                                 "forward direction. One state evaluation must "
                                 "not carry two incompatible partitions"),
        "critical_tokens": ("critical tags identify prediction POSITIONS, not "
                            "vocabulary entries, so those positions use this same "
                            "KL and need no separate support rule"),
        "k_is_not_outcome_selected": (
            "maintainer-selected 2026-10-04. No sweep was run and K must not be "
            "tuned against D1/D2/D3 results."),
        "execution": {"micro_batch_size": D_SERIES_MICRO_BATCH_SIZE,
                      "calibration_batch_packing": D_SERIES_BATCH_PACKING},
        "owner": "scripts/experiments/stage-1/phase_d_series/scoring_protocol.py",
        "decision_record": "logs/budget/decisions.md 2026-10-04",
        "_does_not_move_the_batteries": (
            "Top-K changes structural scoring and evaluation, not prompt "
            "content. family_content_id 1e3445f1b6769169287f6d091e50086e3cf9b663"
            "98d1138af8f137b31e74cd58 stands."),
    }

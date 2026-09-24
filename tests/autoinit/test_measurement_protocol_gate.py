"""One confirmation field must be one measurement protocol.

C2 is why this exists, and C2's own archive is the fixture that matters: a
uniform battery, scoring contract and metric contract, and THREE distinct
generation protocol fingerprints across six probes — two of them inside a
single seed's pair, on the seed with the largest magnitude. Six valid row files
measured under different protocols are six measurements of different things,
and the paired interval computed over them had no estimand. Nothing refused it.

The gate reuses `generation_compat` v2 rather than inventing a C2-specific
interpretation of "same protocol": that rule already owns the question, and it
deliberately demotes the NVIDIA driver patch to recorded-not-material, because
the field named `image_digest` is really `imageName@driver` and the provider
assigns whatever host is free. So differing fingerprints are not automatically
incomparable — they are *unjudgeable* without the expanded blocks, and
unjudgeable fails closed.

Both directions are fixtured. A gate that refused everything would satisfy the
refusal tests while making every future field unusable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.phase_c2 import behavioural_decision as BD  # noqa: E402

SEEDS = (1936324010, 1916380711, 1523147638)
BATTERY = {"artifact": "c1_confirmation_v1", "content_sha256": "a" * 64}
SCORING = {"contract": "c1_confirmation_scoring@v1", "digest": "b" * 64}
METRIC = {"contract": "c1_confirmation_scoring@v1", "schema": "CAPABILITY_V1"}
FP = "c" * 64


def _field(*, fingerprints=None, battery=None, drop=None):
    """Six probes' recorded protocol identities, keyed as `confirm` keys them."""
    out = {}
    for i, seed in enumerate(SEEDS):
        for arm in (BD.INCUMBENT_ARM, BD.TREATMENT_ARM):
            rec = {"battery": dict(battery or BATTERY),
                   "scoring_contract": dict(SCORING),
                   "metric_contract": dict(METRIC),
                   "generation_protocol_fingerprint": FP}
            if fingerprints:
                rec["generation_protocol_fingerprint"] = fingerprints.get(
                    (arm, seed), FP)
            for k in (drop or ()):
                rec.pop(k, None)
            out[(arm, seed)] = rec
    return out


def test_one_uniform_protocol_passes():
    """The gate must not be a refuse-everything gate."""
    report = BD.assert_one_measurement_protocol(_field())
    assert report["uniform"]["battery"] is True
    assert report["uniform"]["generation_protocol_fingerprint"] is True
    assert report["compared_by"]["generation_protocol_fingerprint"] == "exact identity"


def test_the_c2_field_is_refused():
    """C2's actual shape: uniform everything except the generation protocol.

    Three fingerprints — attempt5's four probes on one, incumbent B's third
    seed on a second, the candidate's third seed on a third. The pair at that
    seed therefore spans two protocols, which is a confound inside the pair.
    """
    field = _field(fingerprints={
        (BD.INCUMBENT_ARM, 1523147638): "a" * 64,
        (BD.TREATMENT_ARM, 1523147638): "b" * 64,
    })
    with pytest.raises(BD.BehaviouralDecisionError) as e:
        BD.assert_one_measurement_protocol(field)
    msg = str(e.value)
    assert "3 distinct generation protocol fingerprints" in msg
    assert "Unjudgeable is refused" in msg
    #: The fingerprints are NAMED, so a reader can see which field was mixed.
    assert "a" * 64 in msg and "b" * 64 in msg


def test_a_differing_battery_is_refused_by_identity_not_by_the_v2_rule():
    """A different battery is a different measurement, full stop.

    The v2 rule's demotion applies to the driver patch, never to the battery:
    two probes scored against different prompt sets are not comparable however
    the runtimes compare.
    """
    field = _field()
    victim = (BD.TREATMENT_ARM, SEEDS[0])
    field[victim]["battery"] = {"artifact": "c1_confirmation_v1",
                                "content_sha256": "z" * 64}
    with pytest.raises(BD.BehaviouralDecisionError, match="distinct battery"):
        BD.assert_one_measurement_protocol(field)


def test_an_absent_identity_is_refused_because_absent_is_not_equal():
    """The failure mode that would otherwise pass silently.

    A probe that recorded nothing about how it was measured cannot be shown to
    match, and a gate that treated missing as matching would certify exactly
    the fields nobody captured.
    """
    field = _field(drop=("generation_protocol_fingerprint",))
    with pytest.raises(BD.BehaviouralDecisionError) as e:
        BD.assert_one_measurement_protocol(field)
    assert "Absent is not equal" in str(e.value)

    with pytest.raises(BD.BehaviouralDecisionError, match="no measurement protocol"):
        BD.assert_one_measurement_protocol({})


def test_differing_fingerprints_ARE_comparable_when_only_the_driver_patch_moved():
    """The whole reason to reuse v2 instead of comparing fingerprints.

    `image_digest` fuses the image with the host NVIDIA driver, and the
    provider assigns whatever host is free. Two probes whose every
    generation-semantic field matches but which landed on different driver
    patches within one branch MUST remain comparable, or the gate makes the
    experiment a host lottery — which is the defect v2 was written to fix.
    """
    proto = {"generation": {"max_new_tokens": None, "temperature": 0.0,
                            "top_p": 1.0, "top_k": -1,
                            "runtime_digest": "irrelevant",
                            "generation_protocol_fingerprint": "derived"},
             "scoring_contract": SCORING["contract"],
             "scoring_digest": SCORING["digest"], "battery": BATTERY}
    base_rt = {"image_digest": "runpod/pytorch:1.1.0@580.126.20",
               "python_version": "3.12.3", "torch_version": "2.11.0+cu128",
               "transformers_version": "5.13.1", "cuda_runtime": "12.8",
               "attention_backend": "FLASH_ATTN"}
    other_rt = dict(base_rt, image_digest="runpod/pytorch:1.1.0@580.159.03")

    field = _field(fingerprints={
        (BD.TREATMENT_ARM, SEEDS[0]): "d" * 64,
    })
    for key, rt in zip(field, [base_rt] * len(field)):
        field[key]["protocol"] = proto
        field[key]["runtime"] = rt
    field[(BD.TREATMENT_ARM, SEEDS[0])]["runtime"] = other_rt

    report = BD.assert_one_measurement_protocol(field)
    assert report["uniform"]["generation_protocol_fingerprint"] == "comparable under v2"
    assert report["compared_by"]["generation_protocol_fingerprint"] == (
        "generation_runtime_comparability@v2")


def test_a_moved_driver_BRANCH_is_still_refused():
    """v2 demotes the patch, not the branch. A branch change is a real event."""
    proto = {"generation": {"temperature": 0.0, "runtime_digest": "x"},
             "scoring_contract": SCORING["contract"],
             "scoring_digest": SCORING["digest"], "battery": BATTERY}
    base_rt = {"image_digest": "runpod/pytorch:1.1.0@580.126.20",
               "python_version": "3.12.3", "torch_version": "2.11.0+cu128",
               "transformers_version": "5.13.1", "cuda_runtime": "12.8",
               "attention_backend": "FLASH_ATTN"}
    field = _field(fingerprints={(BD.TREATMENT_ARM, SEEDS[0]): "e" * 64})
    for key in field:
        field[key]["protocol"] = proto
        field[key]["runtime"] = base_rt
    field[(BD.TREATMENT_ARM, SEEDS[0])]["runtime"] = dict(
        base_rt, image_digest="runpod/pytorch:1.1.0@535.104.05")

    with pytest.raises(BD.BehaviouralDecisionError, match="driver branch"):
        BD.assert_one_measurement_protocol(field)


def test_confirm_requires_protocols_and_refuses_before_any_arithmetic():
    """`protocols` is required, so a caller cannot omit it and get a number."""
    import inspect

    sig = inspect.signature(BD.confirm)
    p = sig.parameters["protocols"]
    assert p.default is inspect.Parameter.empty, (
        "protocols has a default, so a caller can reach a verdict without "
        "declaring how the probes were measured")
    assert p.kind is inspect.Parameter.KEYWORD_ONLY

    #: And the gate runs before the arithmetic. Matched on the CALL, not the
    #: name: `paired_differences` is imported at the top of the function body,
    #: so a bare name search finds the import and reports the gate as late
    #: when it is not.
    src = inspect.getsource(BD.confirm)
    assert (src.index("assert_one_measurement_protocol(")
            < src.index("paired_differences(")), (
        "the protocol gate runs after the arithmetic it is meant to prevent")

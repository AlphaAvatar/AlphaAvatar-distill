"""The measurement-field admission rule, tested where it LIVES.

`tests/autoinit/test_measurement_protocol_gate.py` drives this rule through C2,
which is the caller that needed it. This module drives it as core: no C2
battery, no C2 arms, no C2 seeds, no experiment at all — because the claim being
made about it is that C3 and C4 inherit it, and a rule only tested through one
experiment's wrapper has not been shown to be inheritable.

It also pins the boundary that makes the wrapper safe: core raises
`ProtocolFieldError`, and the caller is free to translate that into its own
error type.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.evaluation import protocol_field as PF  # noqa: E402

BATTERY = {"artifact": "some_battery_v1", "content_sha256": "1" * 64}
SCORING = {"contract": "some_scoring@v1", "digest": "2" * 64}
METRIC = {"contract": "some_scoring@v1"}
FP = "3" * 64


def _entries(n=4, **overrides):
    """N measurements keyed by opaque keys — tuples, ints and strings alike.

    The keys are deliberately heterogeneous: the rule must not assume an
    experiment's key shape, and C2's are `(arm, seed)` tuples while a future
    stage's may be anything hashable.
    """
    keys = [("arm_a", 1), ("arm_b", 1), "run-3", 4][:n]
    out = {}
    for k in keys:
        out[k] = {"battery": dict(BATTERY), "scoring_contract": dict(SCORING),
                  "metric_contract": dict(METRIC),
                  PF.GENERATION_IDENTITY_FIELD: FP}
    for k, patch in overrides.items():
        target = next(x for x in out if str(x) == k)
        out[target].update(patch)
    return out


def test_core_names_no_experiment():
    """The genericity claim, checked against the WHOLE source.

    An earlier version of this test read only the source below the exception
    class, which excluded the module docstring — and the docstring was exactly
    where four C2 labels were. `tests/architecture/test_core_boundaries.py`
    caught what this one had been written to catch and missed, so the scope is
    now the entire file, prose included: core does not carry this project's run
    labels, and a docstring is part of the file core ships.
    """
    src = (REPO / "src/aadistill/evaluation/protocol_field.py").read_text()
    for token in ("c1_confirmation", "attempt", "1a2b5b03", "834816710",
                  "phase_c", "gsm8k", "SESOI", "incumbent", "C2", "C3",
                  "probe"):
        assert token not in src, (
            f"core's admission rule names {token!r}, so it encodes one "
            "experiment and the inheritance claim is false")

    #: And the FILENAME must not collide with an experiment's name either: the
    #: architecture inventory classifies a core module by its basename, and
    #: this repository has had an experiment called `measurement`, so
    #: `measurement_field.py` was flagged as an experiment instance. That was
    #: a real ambiguity in the name, not a false positive to suppress.
    stem = Path("src/aadistill/evaluation/protocol_field.py").stem.lower()
    for token in ("phase_a", "phase_b", "c1_", "continuation",
                  "recovery_continuation", "measurement"):
        assert token not in stem, (
            f"the module basename contains {token!r}, which the architecture "
            "inventory reads as naming one experiment")


def test_a_uniform_field_is_admitted():
    report = PF.assert_one_measurement_protocol(_entries())
    assert all(report["uniform"][f] is True for f in PF.PROTOCOL_IDENTITY_FIELDS)
    assert report["uniform"][PF.GENERATION_IDENTITY_FIELD] is True
    assert len(report["members"]) == 4


@pytest.mark.parametrize("field", PF.PROTOCOL_IDENTITY_FIELDS)
def test_each_identity_field_is_load_bearing(field):
    """Every named field must be able to refuse on its own.

    Parametrized because a rule that checked only the first of three would pass
    a single-field test and admit a field that differed in either other one.
    """
    entries = _entries(**{"run-3": {field: {"contract": "something_else"}}})
    with pytest.raises(PF.ProtocolFieldError, match=f"distinct {field}"):
        PF.assert_one_measurement_protocol(entries)


def test_absent_is_not_equal():
    entries = _entries()
    del entries[("arm_b", 1)][PF.GENERATION_IDENTITY_FIELD]
    with pytest.raises(PF.ProtocolFieldError, match="Absent is not equal"):
        PF.assert_one_measurement_protocol(entries)


def test_an_empty_field_is_refused_not_vacuously_admitted():
    """Nothing to compare is the most dangerous thing to admit."""
    with pytest.raises(PF.ProtocolFieldError, match="no measurement protocol"):
        PF.assert_one_measurement_protocol({})


def test_one_entry_is_admitted():
    """A field of one is trivially uniform, and must not be refused.

    The refusal for an EMPTY mapping must not be implemented as "fewer than two
    entries", or a single-measurement field — legitimate in a later stage —
    would be rejected for having nothing to disagree with.
    """
    report = PF.assert_one_measurement_protocol(_entries(n=1))
    assert report["uniform"][PF.GENERATION_IDENTITY_FIELD] is True


def test_the_context_appears_in_every_refusal():
    """The caller names the field; the message must carry that name.

    Without it a C3 operator reading a refusal cannot tell which of several
    fields was mixed.
    """
    entries = _entries()
    del entries[("arm_b", 1)]["battery"]
    with pytest.raises(PF.ProtocolFieldError, match="the C3 recovery field"):
        PF.assert_one_measurement_protocol(entries, context="the C3 recovery field")


def test_differing_fingerprints_are_judged_by_generation_compat_not_by_equality():
    """The reuse requirement: core must defer to the rule that owns runtimes.

    Two entries whose every generation-semantic field matches but which landed
    on different NVIDIA driver PATCHES stay comparable, because `image_digest`
    fuses the image with whatever host the provider had free. Refusing them
    would make a measurement a host lottery.
    """
    proto = {"generation": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                            "runtime_digest": "not material"},
             "scoring_contract": SCORING["contract"],
             "scoring_digest": SCORING["digest"], "battery": BATTERY}
    rt = {"image_digest": "vendor/image:1.1.0@580.126.20",
          "python_version": "3.12.3", "torch_version": "2.11.0+cu128",
          "transformers_version": "5.13.1", "cuda_runtime": "12.8",
          "attention_backend": "FLASH_ATTN"}

    entries = _entries(**{"run-3": {PF.GENERATION_IDENTITY_FIELD: "9" * 64}})
    for k in entries:
        entries[k]["protocol"] = proto
        entries[k]["runtime"] = dict(rt)
    #: same driver BRANCH, different patch
    next(entries[k] for k in entries if str(k) == "run-3")["runtime"][
        "image_digest"] = "vendor/image:1.1.0@580.159.03"

    report = PF.assert_one_measurement_protocol(entries)
    assert report["compared_by"][PF.GENERATION_IDENTITY_FIELD] == (
        "generation_runtime_comparability@v2")
    assert report["uniform"][PF.GENERATION_IDENTITY_FIELD] == "comparable under v2"

    #: A different driver BRANCH is a real event and is still refused.
    next(entries[k] for k in entries if str(k) == "run-3")["runtime"][
        "image_digest"] = "vendor/image:1.1.0@535.104.05"
    with pytest.raises(PF.ProtocolFieldError, match="driver branch"):
        PF.assert_one_measurement_protocol(entries)


def test_unjudgeable_fingerprints_fail_CLOSED():
    """Differing fingerprints with no expanded blocks: refused, not assumed.

    This is the state every C2 probe was in, and the reason the stage produced
    no canonical verdict. "Not shown to be comparable" must never resolve to
    "comparable".
    """
    entries = _entries(**{"run-3": {PF.GENERATION_IDENTITY_FIELD: "9" * 64}})
    with pytest.raises(PF.ProtocolFieldError) as e:
        PF.assert_one_measurement_protocol(entries)
    msg = str(e.value)
    assert "Unjudgeable is refused" in msg
    assert "2 distinct generation protocol fingerprints" in msg
    #: Partially expanded is still unjudgeable: the rule is pairwise and needs
    #: both sides, so one entry carrying blocks must not license the field.
    entries[("arm_a", 1)]["protocol"] = {"generation": {}}
    entries[("arm_a", 1)]["runtime"] = {"image_digest": "x@1"}
    with pytest.raises(PF.ProtocolFieldError, match="Unjudgeable is refused"):
        PF.assert_one_measurement_protocol(entries)

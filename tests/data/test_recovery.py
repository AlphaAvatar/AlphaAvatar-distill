"""Arm R construction and the ten registered quality gates.

The gates exist because each failure produces a corpus that trains something
other than what the registration says. The one worth naming: `exact_prefix_echo`
catches a serving path that silently re-tokenized or dropped the supplied prefix,
which would leave R training on a state the student never visited — an arm that
looks like on-policy recovery and is not.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.recovery import (  # noqa: E402
    GATES, GateFailure, build_example, check_gates, kd_decomposition,
    loss_attribution, roundtrip_ok,
)

STOP = frozenset({151645, 151643})


def make(prompt=5, prefix=10, cont=20, stop=True, **kw):
    p = list(range(100, 100 + prompt))
    s = list(range(200, 200 + prefix))
    c = list(range(300, 300 + cont - 1)) + [151645 if stop else 999]
    kwargs = dict(source_session_id="gsm8k-1", source_seed="sa",
                  truncation_index=0, truncation_fraction=0.4,
                  data_type="gsm8k", system_key="k0", n_system_tokens=3)
    kwargs.update(kw)
    return build_example(prompt_ids=p, student_prefix_ids=s,
                         teacher_continuation_ids=c, **kwargs), p, s, c


def gate(ex, s, **over):
    kw = dict(echoed_prefix_ids=s, student_prefix_ids=s, stop_ids=STOP,
              context_limit_hit=False, block_len=8192,
              held_out_ids=set(), seen_targets=set(), answer_ok=True)
    kw.update(over)
    check_gates(ex, **kw)


# ------------------------------------------------------------- construction

def test_only_the_teacher_recovery_is_supervised():
    ex, p, s, c = make(prompt=5, prefix=10, cont=20)
    assert ex.n_total_tokens == 35
    assert sum(ex.mask) == 20 == ex.n_continuation_tokens
    assert not any(ex.mask[:15])          # prompt + student prefix are context
    assert all(ex.mask[15:])
    assert ex.ids == p + s + c


def test_record_separates_context_length_from_student_prefix_length():
    ex, _, _, _ = make(prompt=5, prefix=10, cont=20)
    r = ex.to_record()
    assert r["n_prefix_tokens"] == 15          # what the continuation conditions on
    assert r["n_student_prefix_tokens"] == 10  # student tokens only
    assert r["arm"] == "R" and r["prefix_source"] == "student_generated"
    assert r["truncation_index"] == 0


def test_an_empty_student_prefix_is_refused_at_construction():
    with pytest.raises(GateFailure, match="exact_prefix_echo"):
        build_example(prompt_ids=[1], student_prefix_ids=[],
                      teacher_continuation_ids=[2], source_session_id="x",
                      source_seed="sa", truncation_index=0,
                      truncation_fraction=0.5, data_type="gsm8k",
                      system_key="k0", n_system_tokens=1)


def test_an_empty_continuation_is_refused_at_construction():
    with pytest.raises(GateFailure, match="non_empty_continuation"):
        build_example(prompt_ids=[1], student_prefix_ids=[2],
                      teacher_continuation_ids=[], source_session_id="x",
                      source_seed="sa", truncation_index=0,
                      truncation_fraction=0.5, data_type="gsm8k",
                      system_key="k0", n_system_tokens=1)


# -------------------------------------------------------------------- gates

def test_a_clean_sample_passes_every_gate():
    ex, _, s, _ = make()
    gate(ex, s)


def test_every_gate_reason_is_registered_and_countable():
    for reason in GATES:
        assert GateFailure(reason).reason == reason
    with pytest.raises(ValueError, match="unregistered gate reason"):
        GateFailure("something_went_wrong")


def test_a_rewritten_or_dropped_prefix_is_caught():
    """The gate that stops R silently becoming off-policy."""
    ex, _, s, _ = make()
    with pytest.raises(GateFailure, match="exact_prefix_echo"):
        gate(ex, s, echoed_prefix_ids=s[:-1])
    with pytest.raises(GateFailure, match="exact_prefix_echo"):
        gate(ex, s, echoed_prefix_ids=[v + 1 for v in s])


def test_unterminated_and_context_limited_samples_are_refused():
    ex, _, s, _ = make(stop=False)
    with pytest.raises(GateFailure, match="natural_termination"):
        gate(ex, s)
    ok, _, s2, _ = make()
    with pytest.raises(GateFailure, match="no_context_limit"):
        gate(ok, s2, context_limit_hit=True)


def test_held_out_prompts_are_refused():
    ex, _, s, _ = make()
    with pytest.raises(GateFailure, match="not_held_out"):
        gate(ex, s, held_out_ids={"gsm8k-1"})


def test_the_context_budget_counts_the_system_block_exactly_once():
    """`prompt_ids` are rendered from the FULL message list, so the system block
    is already inside `ids`. Charging `n_system_tokens` again on top made the
    gate reject valid long samples for a budget they never exceeded."""
    ex, _, s, _ = make(prompt=5, prefix=10, cont=20)   # 35 tokens INCLUDING system
    assert ex.n_total_tokens == 35
    gate(ex, s, block_len=35)                          # exactly fits
    with pytest.raises(GateFailure, match="within_context_budget"):
        gate(ex, s, block_len=34)


def test_a_mask_that_disagrees_with_the_supervision_count_is_refused():
    ex, _, s, _ = make()
    ex.mask[-1] = False
    with pytest.raises(GateFailure, match="mask_matches_supervision"):
        gate(ex, s)


def test_duplicate_recovery_targets_are_caught_across_samples():
    ex1, _, s1, _ = make()
    ex2, _, s2, _ = make()
    seen: set = set()
    gate(ex1, s1, seen_targets=seen)
    with pytest.raises(GateFailure, match="not_duplicate_target"):
        gate(ex2, s2, seen_targets=seen)


def test_a_failing_task_verifier_rejects_but_absence_of_one_does_not():
    ex, _, s, _ = make()
    with pytest.raises(GateFailure, match="answer_valid"):
        gate(ex, s, answer_ok=False)
    gate(ex, s, answer_ok=None)          # no verifier for this task: not a failure


def test_roundtrip_must_be_exact_in_ids_and_mask():
    ex, _, _, _ = make()
    roundtrip_ok(ex, ex.ids, ex.mask)
    with pytest.raises(GateFailure, match="roundtrip_stable"):
        roundtrip_ok(ex, ex.ids[:-1], ex.mask[:-1])
    bad = list(ex.mask); bad[0] = True
    with pytest.raises(GateFailure, match="roundtrip_stable"):
        roundtrip_ok(ex, ex.ids, bad)


# --------------------------------------------------- KD decomposition/attribution

def test_kd_decomposition_splits_prefix_from_continuation():
    ex, _, _, _ = make(prompt=2, prefix=3, cont=5)     # 10 tokens, 9 predictions
    kd = [1.0] * 4 + [0.2] * 5                          # 4 context, 5 supervised
    d = kd_decomposition(kd, ex.mask)
    assert d["prefix_kd_tokens"] == 4
    assert d["continuation_kd_tokens"] == 5
    assert d["prefix_kd_mean"] == 1.0
    assert d["continuation_kd_mean"] == 0.2
    assert d["total_kd_mean"] == pytest.approx((4 * 1.0 + 5 * 0.2) / 9)
    assert d["prefix_share_of_kd_mass"] == pytest.approx(4 / 5)


def test_kd_decomposition_rejects_a_misaligned_vector():
    ex, _, _, _ = make()
    with pytest.raises(ValueError, match="prediction positions"):
        kd_decomposition([0.1] * 3, ex.mask)


def test_loss_attribution_surfaces_a_negligible_prefix_signal():
    """Correct normalization does not imply a meaningful signal; say which."""
    d = {"prefix_share_of_kd_mass": 0.8}
    a = loss_attribution(ce_mean=7.6, kd_mean=0.0255, ce_weight=1.0,
                         kd_weight=0.25, decomposition=d)
    assert a["weighted_ce_contribution"] == pytest.approx(7.6)
    assert a["weighted_kd_contribution"] == pytest.approx(0.006375)
    assert a["kd_share_of_total_loss"] < 0.001
    assert a["prefix_kd_share_of_total_loss"] < 0.001


def test_loss_attribution_handles_a_missing_decomposition():
    a = loss_attribution(ce_mean=1.0, kd_mean=1.0, ce_weight=1.0,
                         kd_weight=1.0, decomposition={})
    assert a["prefix_kd_contribution"] is None
    assert a["kd_share_of_total_loss"] == pytest.approx(0.5)


# ------------------------------------------------- the on-disk packing contract
# 2026-08-07: the E5 pod generated, gated and wrote 4,196 R examples, then died
# at the pairing gate because `to_record()` emitted only the summary counts --
# the token payload it had just validated was dropped on the way to disk. Arm C
# was unaffected, so every offline check passed. These tests close that gap by
# asserting the record against the packer's own contract rather than against a
# hand-written field list.

def test_recovery_record_is_packable():
    """The exact call that failed on the pod: record -> packer input."""
    from aadistill.data.e5_pack import REQUIRED_FIELDS, example_to_rendered

    ex, p, s, c = make(prompt=5, prefix=10, cont=20)
    rec = ex.to_record()
    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    assert not missing, f"to_record() omits packer-required fields: {missing}"

    rendered = example_to_rendered(rec)
    # The system block is stripped for packing and re-emitted per block.
    assert rendered.n_system_tokens == ex.n_system_tokens
    assert rendered.body_ids == ex.ids[ex.n_system_tokens:]
    assert rendered.body_mask == [bool(m) for m in ex.mask[ex.n_system_tokens:]]
    assert sum(rendered.body_mask) == ex.n_continuation_tokens
    # `source_id` is the trajectory, which is what keeps bundle siblings apart.
    assert rendered.source_id == ex.source_session_id


def test_the_record_carries_the_tokens_not_just_their_lengths():
    ex, p, s, c = make(prompt=5, prefix=10, cont=20)
    rec = ex.to_record()
    assert rec["ids"] == ex.ids and rec["mask"] == [bool(m) for m in ex.mask]
    # A record that survives JSON must still be packable -- this is how it lands.
    import json

    from aadistill.data.e5_pack import example_to_rendered
    assert example_to_rendered(json.loads(json.dumps(rec))).body_ids \
        == ex.ids[ex.n_system_tokens:]


def test_both_arms_satisfy_one_packing_contract():
    """C and R are packed by the same function, so they answer to one contract.

    Arm C's record shape is asserted here from its builder's own field list so
    the two arms cannot drift apart silently again."""
    from aadistill.data.e5_pack import REQUIRED_FIELDS

    ex, *_ = make()
    r_record = set(ex.to_record())
    c_record = {"id", "ids", "mask", "system_key", "n_system_tokens",
                "source_session_id", "truncation_index", "truncation_fraction",
                "data_type", "arm", "prefix_source", "source_seed"}
    for field_name in REQUIRED_FIELDS:
        assert field_name in r_record, f"arm R record missing {field_name}"
        assert field_name in c_record, f"arm C record missing {field_name}"

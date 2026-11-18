"""The registered A - B analysis (`analysis.py`) and B's protocol check (`b_evidence.py`): pure functions and one
synthetic end to end. Seconds, CPU, no GPU and no network.

    pytest scripts/stages/stage-1/teacher_parent_initialization/tests
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stages.teacher_parent_initialization import analysis as TA
from stages.teacher_parent_initialization import b_evidence as BE

SEEDS = [20260726, 20260801]


def _y(values: dict) -> dict:
    return {s: dict(v) for s, v in values.items()}


def test_an_items_two_seeds_land_in_one_group():
    groups = {"g0": ["a", "b"], "c": ["c"]}
    y_a = _y({1: {"a": 1, "b": 0, "c": 1}, 2: {"a": 1, "b": 1, "c": 0}})
    y_b = _y({1: {"a": 0, "b": 0, "c": 0}, 2: {"a": 0, "b": 1, "c": 1}})
    keys, d, n = TA.pooled_group_differences(y_a, y_b, groups, [1, 2])
    assert keys == ["c", "g0"]
    assert d.tolist() == [0, 2]            # c: +1 - 1; g0: +1 + 0 + 1 + 0
    assert n.tolist() == [2, 4]            # n_g = 2 seeds x |g|


def test_a_missing_item_in_one_seed_is_descriptive_only():
    groups = {f"i{k}": [f"i{k}"] for k in range(6)}
    full = {i: 1 for g in groups.values() for i in g}
    y_b = {m: _y({s: dict.fromkeys(full, 0) for s in SEEDS}) for m in ("H-T", "H-N")}
    y_a = {m: _y({s: dict(full) for s in SEEDS}) for m in ("H-T", "H-N")}
    del y_a["H-T"][SEEDS[1]]["i3"]
    with pytest.raises(ValueError):
        TA.pooled_group_differences(y_a["H-T"], y_b["H-T"], groups, SEEDS)
    out = TA.decide(y_a, y_b, groups, SEEDS, primary="H-T", secondary="H-N")
    assert out["H-T"]["status"] == "descriptive_only" and "test" not in out["H-T"]
    assert out["H-T"]["paired_records_present"] == 2 * 6 - 1
    #: no partial pooling: the secondary is not tested either
    assert out["H-N"]["status"] == "descriptive_only" and "p_two_sided" not in out["H-N"]


def test_identical_records_are_not_distinguished():
    groups = {f"i{k}": [f"i{k}"] for k in range(30)}
    y = {s: {f"i{k}": k % 3 == 0 for k in range(30)} for s in SEEDS}
    out = TA.decide({"H-T": copy.deepcopy(y), "H-N": copy.deepcopy(y)}, {"H-T": y, "H-N": y}, groups, SEEDS,
                    primary="H-T", secondary="H-N")
    assert out["H-T"]["status"] == "formal"
    assert out["H-T"]["delta"] == 0.0 and out["H-T"]["p_two_sided"] == 1.0
    assert out["H-T"]["verdict"] == "NOT_DISTINGUISHED"


def test_the_secondary_is_untested_unless_the_primary_rejects():
    groups = {f"i{k}": [f"i{k}"] for k in range(25)}
    zero = {s: dict.fromkeys((f"i{k}" for k in range(25)), 0) for s in SEEDS}
    one = {s: dict.fromkeys((f"i{k}" for k in range(25)), 1) for s in SEEDS}
    #: primary A == B; secondary A far better: the secondary still gets no test (fixed sequence)
    out = TA.decide({"H-T": copy.deepcopy(zero), "H-N": one}, {"H-T": zero, "H-N": zero}, groups, SEEDS,
                    primary="H-T", secondary="H-N")
    assert out["H-T"]["verdict"] == "NOT_DISTINGUISHED"
    assert out["H-N"]["status"] == "descriptive_only" and "p_two_sided" not in out["H-N"]
    assert out["H-N"]["delta"] == 1.0
    #: primary rejects (every group +2): the secondary is tested
    out = TA.decide({"H-T": one, "H-N": one}, {"H-T": zero, "H-N": zero}, groups, SEEDS,
                    primary="H-T", secondary="H-N")
    assert out["H-T"]["verdict"] == "A_BETTER" and out["H-T"]["p_two_sided"] <= TA.ALPHA
    assert out["H-N"]["status"] == "formal" and out["H-N"]["verdict"] == "A_BETTER"


def test_a_protocol_that_is_not_bs_makes_the_mode_descriptive():
    groups = {f"i{k}": [f"i{k}"] for k in range(25)}
    zero = {s: dict.fromkeys((f"i{k}" for k in range(25)), 0) for s in SEEDS}
    one = {s: dict.fromkeys((f"i{k}" for k in range(25)), 1) for s in SEEDS}
    out = TA.decide({"H-T": one, "H-N": one}, {"H-T": zero, "H-N": zero}, groups, SEEDS, primary="H-T",
                    secondary="H-N", protocol_ok={"H-T": False, "H-N": True})
    assert out["H-T"]["status"] == "descriptive_only" and "test" not in out["H-T"]
    assert out["H-N"]["status"] == "descriptive_only"
    assert "not formal" in out["H-N"]["why"]


def _b_fluid_summary() -> dict:
    d = json.loads(BE.DESIGN.read_text())
    p = BE.b_gens(d, SEEDS[0], "H-T") / f"{BE.FLUID}.summary.json"
    if not p.is_file():
        pytest.skip(f"B's sealed evidence is not on this machine ({p})")
    return json.loads(p.read_text())


def test_the_protocol_check_reads_compositions_and_flags_a_changed_decoding():
    s = _b_fluid_summary()
    want = BE.mode_protocol({BE.FLUID: BE.protocol_fields(s, BE.FLUID)})
    assert BE.protocol_problems({BE.FLUID: s}, want) == []
    #: A's streamed fluid unit: a composition over the runner summaries, the registered task file on top
    tasks = s["identity"]["tasks_sha256"]
    run = copy.deepcopy(s)
    run["identity"]["tasks_sha256"] = "0" * 64      #: a resumed run submits only the missing tasks
    comp = {"composition": "the union of the sealed parts", "runs": [{"summary": run}], "tasks_sha256": tasks}
    assert BE.protocol_problems({BE.FLUID: comp}, want) == []
    bad = copy.deepcopy(comp)
    bad["runs"][0]["summary"]["identity"]["decoding_protocol"]["sha256"] = "f" * 64
    probs = BE.protocol_problems({BE.FLUID: bad}, want)
    assert probs and all("decoding_sha256" in p for p in probs)
    #: a unit whose protocol was not recorded is not B's -- nor one checked against an empty registered protocol
    probs = BE.protocol_problems({"stage3_capability_846__gsm8k": {"composition": "x", "run_summary": None}}, want)
    assert "stage3_capability_846__gsm8k: no runner summary recorded" in probs
    assert "stage3_capability_846__gsm8k: the registered protocol has no decoding_sha256" in probs


def test_complete_generations_with_a_missing_or_empty_protocol_are_never_consistent():
    s = _b_fluid_summary()
    want = BE.mode_protocol({BE.FLUID: BE.protocol_fields(s, BE.FLUID)})
    assert BE.protocol_check({BE.FLUID: s}, want, generated=(BE.FLUID,)) == {"differs": [], "missing": []}
    #: the records are there, the summary is not
    chk = BE.protocol_check({}, want, generated=(BE.FLUID,))
    assert chk["missing"] == [f"{BE.FLUID}: generations present, no summary recorded"] and not chk["differs"]
    #: a key field recorded empty on both sides is missing, not equal
    blank = copy.deepcopy(s)
    blank["identity"]["chat_template_sha256"] = ""
    w2 = copy.deepcopy(want)
    w2["fluid"]["chat_template_sha256"] = ""
    chk = BE.protocol_check({BE.FLUID: blank}, w2)
    assert f"{BE.FLUID}: the registered protocol has no chat_template_sha256" in chk["missing"]
    assert f"{BE.FLUID}[run 0]: chat_template_sha256 not recorded" in chk["missing"]
    #: no registered protocol at all
    chk = BE.protocol_check({BE.FLUID: s}, {"fluid": {}, "capability": {}})
    assert len(chk["missing"]) == len(BE.KEY_FIELDS["fluid"]) and not chk["differs"]
    #: an empty template-kwargs dict is a value
    assert not BE.empty({}, "chat_template_kwargs") and BE.empty({}, "sampling") and BE.empty(None, "stop_ids")


def test_a_missing_summary_field_is_not_recorded_never_a_pass():
    s = _b_fluid_summary()
    want = BE.mode_protocol({BE.FLUID: BE.protocol_fields(s, BE.FLUID)})
    stripped = copy.deepcopy(s)
    del stripped["identity"]["context_resolution"]
    got = BE.protocol_fields(stripped, BE.FLUID)
    assert got["resolved_context"] == BE.NOT_RECORDED
    assert any("resolved_context" in p for p in BE.protocol_problems({BE.FLUID: stripped}, want))

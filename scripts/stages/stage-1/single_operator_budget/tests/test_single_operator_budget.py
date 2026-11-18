"""SOB1's preparation: the three structures are the real implementation's counts and legal for their operators, the
design's gates repeat committed records, the generated plans reproduce, and the registered decision rule does what
the design says. CPU, seconds, no network (the teacher's config is read from the local HF cache when present).

    pytest scripts/stages/stage-1/single_operator_budget/tests
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stages.single_operator_budget import analysis as SA
from stages.single_operator_budget import proposal as PR
from stages.single_operator_budget import structures as ST

REPO = Path(__file__).resolve().parents[5]
DESIGN = json.loads(ST.DESIGN.read_text())


def _teacher_or_skip() -> dict:
    try:
        return ST.teacher_config(DESIGN)
    except ST.StructureError as exc:
        pytest.skip(str(exc))


def test_the_committed_structures_and_proposal_reproduce():
    _teacher_or_skip()
    assert ST.render(ST.build()) == ST.OUT.read_text()
    assert PR.render(PR.build()) == PR.OUT.read_text()


def test_the_counts_are_the_implementations_with_the_tied_embedding_once():
    base = _teacher_or_skip()
    s = json.loads(ST.OUT.read_text())
    for name, arm in s["arms"].items():
        m = ST.instantiate(base, DESIGN["arms"][name]["geometry"])
        assert m.lm_head.weight is m.model.embed_tokens.weight
        assert sum(p.numel() for p in m.parameters()) == arm["num_parameters"] == sum(arm["modules"].values())
        assert arm["trainable_parameters"] == arm["num_parameters"] - arm["modules"]["embedding (tied head)"]
    #: the method reproduces A's recorded 596M count, and the teacher's
    assert s["a_596m_reference"]["num_parameters"] == 596049920 and s["teacher"]["num_parameters"] == 4022468096
    assert abs(s["spread"]["relative_to_target"]) < 0.01 and all(abs(a["n"] - 2) < 0.01 for a in s["arms"].values())


def test_each_arm_changes_only_what_its_operators_own_and_illegal_geometries_are_refused():
    base = _teacher_or_skip()
    t = ST.geometry(base)
    for name, arm in DESIGN["arms"].items():
        g = ST.geometry({**base, **arm["geometry"]})
        assert ST.legality(g, t, arm["build"]) == []
        assert {k for k in ST.GEOMETRY_KEYS if g[k] != t[k]} == set(arm["geometry"])
    assert any("multiple of 8 KV" in p for p in ST.legality({**t, "num_attention_heads": 12}, t, ["ATTENTION"]))
    assert any("preserved" in p for p in ST.legality({**t, "num_key_value_heads": 4}, t, ["ATTENTION"]))
    assert any("not in the arm's build" in p for p in ST.legality({**t, "num_hidden_layers": 16}, t, ["FFN"]))
    assert any("multiple of 128" in p for p in ST.legality({**t, "hidden_size": 1275}, t, ["RESIDUAL_WIDTH"]))


def test_the_design_gates_repeat_tpi1s_committed_teacher_level_selections():
    sel = json.loads((REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json").read_text())
    arms = DESIGN["arms"]
    assert str(sel["removal_order"]) in arms["depth_only"]["selection"]["gate"]
    assert sel["states"]["DEPTH"]["state_id"] in arms["depth_only"]["selection"]["gate"]
    assert sel["states"]["FFN"]["state_id"] in arms["ffn_attention"]["selection"]["FFN"]
    att = sel["states"]["ATTENTION"]
    assert att["state_id"] in arms["ffn_attention"]["selection"]["ATTENTION"]
    #: the reused attention selection is exactly this arm's head count
    assert att["arch_spec"]["num_attention_heads"] == arms["ffn_attention"]["geometry"]["num_attention_heads"]
    assert sel["states"]["RESIDUAL_WIDTH"]["state_id"] in arms["residual_width"]["selection"]["RESIDUAL_WIDTH"]
    assert DESIGN["analysis"]["primary_items"] == json.loads(
        (REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json").read_text())["analysis"]["primary_items"]


def test_holm_steps_down_and_stops_at_the_first_acceptance():
    h = SA.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["rejected"] and h["c"]["rejected"] is False and h["b"]["rejected"] is False   #: 0.03 > 0.05 / 2
    assert h["a"]["p_adjusted"] == pytest.approx(0.03) and h["c"]["p_adjusted"] == pytest.approx(0.06)
    assert h["b"]["p_adjusted"] == pytest.approx(0.06)                    #: monotone
    assert all(v["rejected"] for v in SA.holm({"a": 0.001, "b": 0.002, "c": 0.003}).values())


def _records(acc: dict[str, float], items: list[str], seeds: list[int], rng) -> dict:
    return {arm: {m: {s: {i: int(rng.random() < p) for i in items} for s in seeds} for m in ("H-T", "H-N")}
            for arm, p in acc.items()}


def test_the_second_family_is_tested_only_when_the_first_rejects_every_pair():
    rng = np.random.default_rng(0)
    items = [f"i{k}" for k in range(300)]
    groups, seeds = {i: [i] for i in items}, [1, 2]
    pairs = [tuple(p) for p in DESIGN["analysis"]["comparisons"]]
    far = _records({"depth_only": 0.9, "ffn_attention": 0.5, "residual_width": 0.1}, items, seeds, rng)
    out = SA.decide(far, pairs, groups, seeds)
    assert all(v["status"] == "formal" and v["rejected"] for v in out["H-T"].values())
    assert out["H-T"]["depth_only - ffn_attention"]["verdict"] == "depth_only_BETTER"
    assert out["H-T"]["ffn_attention - residual_width"]["verdict"] == "ffn_attention_BETTER"
    assert all(v["status"] == "formal" for v in out["H-N"].values())
    near = _records({"depth_only": 0.5, "ffn_attention": 0.5, "residual_width": 0.1}, items, seeds, rng)
    out = SA.decide(near, pairs, groups, seeds)
    assert out["H-T"]["depth_only - ffn_attention"]["verdict"] == "NOT_DISTINGUISHED"
    assert all(v["status"] == "descriptive_only" and "serial gatekeeping" in v["why"] for v in out["H-N"].values())
    #: an arm missing an item in one seed is never pooled partially
    del near["residual_width"]["H-T"][2][items[0]]
    out = SA.decide(near, pairs, groups, seeds)
    assert out["H-T"]["depth_only - residual_width"]["status"] == "descriptive_only"
    assert "p_two_sided" in out["H-T"]["depth_only - ffn_attention"]


def test_the_budget_phases_are_the_components_they_name():
    p = json.loads(PR.OUT.read_text())
    c, o = p["components"], p["options"]
    full = o["full_two_seeds"]
    assert full["usd_expected"] == pytest.approx(o["phase_1_initialization_and_damage"]["usd_expected"]
                                                 + o["phase_2_recovery_two_seeds"]["usd_expected"], abs=0.02)
    assert c["training"]["units"] == 6 and p["status"].startswith("proposed")
    assert p["authorizes"] == "nothing" and DESIGN["authorizes"] == "nothing" and DESIGN["budget"]["grant"] is None

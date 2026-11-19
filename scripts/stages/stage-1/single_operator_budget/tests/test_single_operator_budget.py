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


def test_the_gates_repeat_tpi1s_committed_teacher_level_selections():
    tsel = json.loads((REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json").read_text())
    sel = json.loads((REPO / "configs/stages/stage-1/single_operator_budget/selection.json").read_text())
    arms = DESIGN["arms"]
    assert str(tsel["removal_order"]) in arms["depth_only"]["selection"]["gate"]
    assert tsel["states"]["DEPTH"]["state_id"] == sel["depth_only"]["state_id"] in arms["depth_only"]["selection"]["DEPTH"]
    assert sel["depth_only"]["step"]["config_hash"] == tsel["steps"]["DEPTH"]["config_hash"]
    assert tsel["states"]["FFN"]["state_id"] in arms["ffn_attention"]["selection"]["FFN"]
    assert sel["ffn_attention"]["ffn"]["gate"]["artifact_digest"] == tsel["states"]["FFN"]["artifact_digest"]
    att = tsel["states"]["ATTENTION"]
    assert att["state_id"] in arms["ffn_attention"]["selection"]["ATTENTION"]
    #: the reused attention selection is exactly this arm's head count, block by block
    assert att["arch_spec"]["num_attention_heads"] == arms["ffn_attention"]["geometry"]["num_attention_heads"]
    assert sel["ffn_attention"]["attention"]["query_heads"] == tsel["query_heads"]
    assert tsel["states"]["RESIDUAL_WIDTH"]["state_id"] in arms["residual_width"]["selection"]["RESIDUAL_WIDTH"]
    assert sel["residual_width"]["points"] == list(range(37)) and sel["residual_width"]["hidden_size"] == 1280
    assert DESIGN["analysis"]["primary_items"] == json.loads(
        (REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json").read_text())["analysis"]["primary_items"]


def test_holm_steps_down_and_stops_at_the_first_acceptance():
    h = SA.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["rejected"] and h["c"]["rejected"] is False and h["b"]["rejected"] is False   #: 0.03 > 0.05 / 2
    assert h["a"]["p_adjusted"] == pytest.approx(0.03) and h["c"]["p_adjusted"] == pytest.approx(0.06)
    assert h["b"]["p_adjusted"] == pytest.approx(0.06)                    #: monotone
    assert all(v["rejected"] for v in SA.holm({"a": 0.001, "b": 0.002, "c": 0.003}).values())


def test_holm_keeps_m_at_the_registered_three_when_a_pair_is_absent():
    """The defect fixed 2026-11-19: with one pair untestable, Holm over the two left used m = 2 and adjusted
    0.02 / 0.024 to 0.04 / 0.04 (both rejected). With m = 3 they are 0.06 / 0.06: neither is."""
    two = {"a": 0.02, "b": 0.024}
    assert all(v["rejected"] for v in SA.holm(two).values())              #: m = len(p): the old behaviour
    h = SA.holm(two, m=3)
    assert h["a"]["p_adjusted"] == pytest.approx(0.06) and h["b"]["p_adjusted"] == pytest.approx(0.06)
    assert not any(v["rejected"] for v in h.values()) and {v["m"] for v in h.values()} == {3}
    with pytest.raises(ValueError):
        SA.holm({"a": 0.1, "b": 0.2, "c": 0.3}, m=2)


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
    assert all(v["status"] == "formal" and v["rejected"] for k, v in out["H-T"].items() if not k.startswith("_"))
    assert out["H-T"]["depth_only - ffn_attention"]["verdict"] == "depth_only_BETTER"
    assert out["H-T"]["ffn_attention - residual_width"]["verdict"] == "ffn_attention_BETTER"
    assert all(v["status"] == "formal" for k, v in out["H-N"].items() if not k.startswith("_"))
    near = _records({"depth_only": 0.5, "ffn_attention": 0.5, "residual_width": 0.1}, items, seeds, rng)
    out = SA.decide(near, pairs, groups, seeds)
    assert out["H-T"]["depth_only - ffn_attention"]["verdict"] == "NOT_DISTINGUISHED"
    assert all(v["status"] == "descriptive_only" and "serial gatekeeping" in v["why"]
               for k, v in out["H-N"].items() if not k.startswith("_"))
    assert out["_rule"]["m"] == 3 and all(v["holm_m"] == 3 for k, v in out["H-T"].items() if not k.startswith("_"))
    #: an arm missing an item in one seed: the mode's primary units are incomplete, so NO pair of it is tested
    del near["residual_width"]["H-T"][2][items[0]]
    out = SA.decide(near, pairs, groups, seeds)
    assert out["H-T"]["_completeness"] == {"complete": False, "missing": {"residual_width": [2]}}
    assert out["H-T"]["depth_only - residual_width"]["status"] == "descriptive_only"
    pair = out["H-T"]["depth_only - ffn_attention"]          #: both its arms complete: a delta, never a test
    assert pair["status"] == "descriptive_only" and "delta" in pair and "p_two_sided" not in pair
    assert "verdict" not in pair and "incomplete" in pair["why"]


def test_a_missing_arm_leaves_every_pair_descriptive_and_never_shrinks_m():
    """Regression (2026-11-19): two of three arms complete in both seeds, the third absent. Before the fix the two
    remaining pairs were tested with Holm over m = 2 and could be declared; now the mode is incomplete: every pair is
    descriptive, nothing is tested, and the non-thinking family stays closed."""
    rng = np.random.default_rng(1)
    items = [f"i{k}" for k in range(300)]
    groups, seeds = {i: [i] for i in items}, [1, 2]
    pairs = [tuple(p) for p in DESIGN["analysis"]["comparisons"]]
    y = _records({"depth_only": 0.9, "ffn_attention": 0.1, "residual_width": 0.5}, items, seeds, rng)
    complete = SA.decide(y, pairs, groups, seeds)
    assert complete["H-T"]["depth_only - ffn_attention"]["verdict"] == "depth_only_BETTER"
    del y["residual_width"]                                   #: one arm's units never ran
    out = SA.decide(y, pairs, groups, seeds)
    assert out["H-T"]["_completeness"]["missing"] == {"residual_width": [1, 2]}
    for key in ("depth_only - ffn_attention", "depth_only - residual_width", "ffn_attention - residual_width"):
        v = out["H-T"][key]
        assert v["status"] == "descriptive_only" and "p_holm" not in v and "verdict" not in v, key
    assert out["H-T"]["depth_only - ffn_attention"]["delta"] > 0.5      #: still described
    assert all(v["status"] == "descriptive_only" for k, v in out["H-N"].items() if not k.startswith("_"))
    #: one seed of one arm missing is the same: incomplete, descriptive
    y2 = _records({"depth_only": 0.9, "ffn_attention": 0.1, "residual_width": 0.5}, items, seeds, rng)
    del y2["ffn_attention"]["H-T"][1]
    assert all(v["status"] == "descriptive_only"
               for k, v in SA.decide(y2, pairs, groups, seeds)["H-T"].items() if not k.startswith("_"))


def test_the_arm_report_reads_damage_recovery_and_the_gap_at_one_context():
    items = [f"i{k}" for k in range(40)]
    groups, seeds = {f"g{k}": items[2 * k:2 * k + 2] for k in range(20)}, [1, 2]
    teacher = {i: 1 for i in items}
    pre = {i: int(k < 4) for k, i in enumerate(items)}                       #: 10%
    post = {1: {i: int(k < 20) for k, i in enumerate(items)}, 2: {i: int(k < 16) for k, i in enumerate(items)}}
    r = SA.arm_report(pre, post, teacher, groups, seeds)
    assert r["initialization_damage"]["delta"] == pytest.approx(0.1 - 1.0)
    assert r["recovery"]["pooled"]["delta"] == pytest.approx((0.5 + 0.4) / 2 - 0.1)
    assert r["recovery"]["per_seed"]["1"]["delta"] == pytest.approx(0.4)
    assert r["remaining_gap"]["delta"] == pytest.approx((0.5 + 0.4) / 2 - 1.0)
    assert "eliminate nothing" in r["role"]
    r = SA.arm_report(pre, None, teacher, groups, seeds)                   #: phase 1 only
    assert r["recovery"] == r["remaining_gap"] == "not_available" and "delta" in r["initialization_damage"]


def test_the_budget_proposes_phase_1_only_and_lists_phase_2_apart():
    p = json.loads(PR.OUT.read_text())
    ph = p["phase_1"]
    assert ph["driver_hours_expected"] == pytest.approx(sum(c["hours_expected"] for c in ph["components"].values()),
                                                       abs=0.02)
    assert set(ph["components"]) == {"builds", "pre_recovery_evaluation", "teacher_at_8192"}
    assert ph["components"]["builds"]["depth"]["candidate_subsets_new"] == 270
    assert p["ceiling_usd"] == ph["hard_cap"]["usd"] and p["teardown_reserve_usd"] == 1.0
    assert p["study_cumulative_cap_usd"] == pytest.approx(p["ceiling_usd"] + 1.0)
    assert p["expected_total_usd"] < p["ceiling_usd"] <= p["position_at_proposal"]["project_remaining_usd"] - 1.0
    two = p["phase_2_draft"]
    assert two["sessions"] == 3 and set(two["per_arm_session"]) == set(DESIGN["arms"])
    assert "no cap here is proposed" in two["status"]
    assert all(v["units"] == 2 for v in two["per_arm_session"].values())
    assert p["status"] == "proposed" and DESIGN["authorizes"] == "nothing" and DESIGN["budget"]["grant"] is None


# ---- the fixed protocol (maintainer 2026-11-19) --------------------------------------------------------------------

def test_every_model_is_evaluated_under_one_protocol_at_8192():
    from stages.single_operator_budget import common as C
    ev = DESIGN["evaluation"]
    assert ev["protocol"]["context"]["tokens"] == 8192 == C.protocol()["context"]["tokens"]
    assert set(ev["models"]) == {"pre_recovery", "teacher", "post_recovery"}
    assert "none identity-matched" in ev["teacher_reference"]["status"]
    assert DESIGN["phase_1"]["evaluation_order"][-1] == "teacher"
    assert set(DESIGN["phase_1"]["build_order"]) == set(DESIGN["arms"])
    assert DESIGN["training"]["unit_order"]["arms"][0] == "residual_width"
    assert C.protocol()["package"] == "teacher_sidecars_v1"
    assert "eliminate" in DESIGN["analysis"]["no_elimination"] or "dropped" in DESIGN["analysis"]["no_elimination"]


def test_the_depth_resume_record_reproduces_and_pins_270_new_candidates():
    from stages.single_operator_budget import selection as SSEL
    sel = json.loads(SSEL.OUT.read_text())
    assert SSEL.render(SSEL.extract()) == SSEL.OUT.read_text()
    d = sel["depth_only"]
    assert d["resume_checks"]["all_ok"] and d["resume_checks"]["missing"] == []
    assert [r["chosen"] for r in d["rounds"]] == [2, 32, 3, 26, 30, 8, 20, 28]
    assert d["resume"]["new_candidate_subsets"] == 270 and d["resume"]["first_new_round"] == 9
    assert all(len(r["table"]) == 36 - k for k, r in enumerate(d["rounds"]))   #: every candidate's score is kept


def _fake_init_record(tmp: Path, arm: str) -> Path:
    """A COMPLETE init record of ``arm`` with the shape phase1_driver writes (for the plan's derivation only)."""
    st = json.loads(ST.OUT.read_text())["arms"][arm]
    sha = "ab" * 32
    rec = {"schema": "aadistill.stage1.single_operator_budget.init_record/v1", "status": "COMPLETE", "arm": arm,
           "root_teacher": {"root_teacher_id": DESIGN["teacher"]["root_teacher_id"],
                            "root_teacher_sha256": DESIGN["teacher"]["root_teacher_sha256"]},
           "commit": {"selected_rows": [{"state_id": f"{arm}/x", "artifact_digest": "cd" * 32, "single_shard_sha256": sha}]},
           "checkpoint": {"artifact_digest": "cd" * 32, "shards": [{"filename": "model.safetensors", "sha256": sha}],
                          "num_parameters_counted": st["num_parameters"]},
           "relay": {"init": {"relay": "fake", "prefix": f"durable/stage1_single_operator_budget/{arm}/init/x",
                              "files_sha256": {"init/model.safetensors": sha, "init/config.json": "ef" * 32},
                              "bytes": {"init/model.safetensors": st["checkpoint_bytes"]["init_bf16"]},
                              "verified": True},
                     "mapping": {"verified": True}}}
    p = tmp / f"{arm}_init_record.json"
    p.write_text(json.dumps(rec))
    return p


def test_each_arms_plan_is_a_mix_and_tpi1s_recipe_with_only_the_initialization_changed(tmp_path, monkeypatch):
    from stages.single_operator_budget import plan as PL
    #: the init record must live under the repository (the plan binds it by repo-relative path)
    fx = REPO / "logs/stages/stage-1/single_operator_budget/.test_fixture"
    fx.mkdir(exist_ok=True)
    try:
        for arm in DESIGN["arms"]:
            doc = PL.derive(_fake_init_record(fx, arm), arm)
            assert [u["unit"] for u in doc["units"]] == [f"{arm}-s20260726", f"{arm}-s20260801"]
            assert doc["unit_order"] == [f"{arm}-s20260726", f"{arm}-s20260801"]
            for u in doc["units"]:
                c = u["config"]
                assert c["stage"] == "stage3_recovery" and c["batch"]["blocks_per_step"] == 2
                assert c["batch"]["micro_blocks"] == 1 and c["batch"]["context_strategy"] == "cross_context"
                assert c["student_path"] == f"/workspace/single_operator/init/{arm}"
                assert not doc["derivation_checks"][u["unit"]]["outside_declared"]
                assert not doc["derivation_checks"][u["unit"]]["outside_declared_tpi1"]
                assert u["bytes"]["save_point"] > 2e10 and u["bytes"]["final"] > 8e9
            assert doc["evaluation_protocol"]["context"]["tokens"] == 8192
            assert doc["durability"]["download_floor_bytes_per_second"] == 2.6e6
            assert doc["relay_prefix_root"] == "durable/stage1_single_operator_budget"
        with pytest.raises(PL.PlanError, match="not a COMPLETE"):
            PL.derive(_fake_init_record(fx, "depth_only"), "ffn_attention")
    finally:
        import shutil
        shutil.rmtree(fx, ignore_errors=True)


def test_the_launcher_binds_the_kinds_records_and_funds_phase_2_with_nothing():
    from stages.single_operator_budget import launch as L
    c = L.kind_conditions(L.PHASE1_KIND)
    assert set(c) == {"design_sha256", "selection_sha256", "structures_sha256"}
    with pytest.raises(L.SE.AuthorizationError, match="names its arm"):
        L.kind_conditions(L.TRAIN_KIND)
    p = json.loads(PR.OUT.read_text())
    assert p["status"] == "proposed" and p["kinds"][L.PHASE1_KIND]["per_session_hard_cap_usd"] == p["ceiling_usd"]
    assert p["kinds"][L.TRAIN_KIND]["per_session_hard_cap_usd"] is None
    sizes = L.relay_object_sizes(L.PHASE1_KIND)
    assert sum(sizes[:3]) == p["storage"]["phase_1_relay_peak_bytes"]
    assert L.relay_object_sizes(L.TRAIN_KIND, "residual_width")[0] == p["storage"]["per_arm_bytes"]["residual_width"]["save_point"]
    assert L.Authorization.STUDY.key == "SOB1" and L.package.__name__ == "package"
    #: setup's branch for this study's session kind loads exactly its type
    setup = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    body = setup.split('"$SESSION_KIND" = "stage1_single_operator"', 1)[1].split("elif [", 1)[0]
    assert "from stages.single_operator_budget.launch import Authorization, kind_conditions" in body
    assert 'mark "AUTHORIZATION_MISMATCH"; exit 98' in body


def test_the_hard_cap_is_the_runners_all_in():
    """The 2026-11-19 dry run refused a $22.53 cap the runner priced at $22.5485: the proposal's cap is now the
    runner's all-in (`study_engineering.all_in` over the plan's hard deadline) at the highest admitted price."""
    import types
    from shared import study_engineering as SE
    cm = DESIGN["cost_model"]
    for driver in (60.0, 1089.0, 3155.2):
        f = 1 + PR.SESSION["contingency_fraction"]
        plan = types.SimpleNamespace(hard_terminate_minutes=(PR.SESSION["setup"] + PR.SESSION["transfer"] + driver) * f
                                     + PR.SESSION["closeout"] * f + PR.SESSION["artifact_recovery"])
        runner = SE.all_in(plan, PR.MAX_PRICE, cm["container_disk"]["gb"])["all_in_usd"]
        cap = PR.hard_cap(driver, cm)["usd"]
        assert runner <= cap < runner + 0.01, (driver, runner, cap)

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


# ---- completion on real artifacts (maintainer 2026-11-19, second brief) ---------------------------------------------

def _seed_of(base: int, key: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"{base}|{key}".encode()).digest()[:8], "big") % 2 ** 63


def _complete_set(d: Path, mode: str, weights: dict | None, attempt: str = "a1") -> None:
    """A synthetic set every check accepts: every registered item once, each record under the registered protocol
    (seed rule, 8,192 context, the mode's think state, a stop id), each unit's summary with its runner fields."""
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import b_evidence as BE
    w = CP.expected(mode)
    cap, flu = w["capability"], w["fluid"]
    base, ctx, ts = cap["seed_rule_base"], cap["resolved_context"], w["think_state"]
    gens = d / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    turn = {"raw": "<think>x</think>\n\nx", "stop_reason": "eos", "natural_termination": True,
            "degeneration_triggered": False, "context_limit_reached": False, "generated_tokens": 1,
            "output_token_ids": [151645], "prompt_tokens": 10, "generation_allowance": ctx - 10, "think_state": ts}
    dp = {"protocol_id": cap["decoding_protocol_id"], "sha256": cap["decoding_sha256"], "sampling": cap["sampling"],
          "samples_per_item": 1, "seed_rule": {"base": base, "key": cap["seed_rule_key"],
                                               "derivation": cap["seed_rule_derivation"]}}
    common = {"decoding_protocol": dp, "context_resolution": {"resolved_context": ctx, "context_source": "trained_block_len",
                                                              "trained_context": ctx, "architectural_context": 40960},
              "system_message": cap["system_message"], "chat_template_sha256": cap["chat_template_sha256"],
              "chat_template_kwargs": cap["chat_template_kwargs"], "tokenizer_sha256": cap["tokenizer_sha256"],
              "stop_ids": [151645]}
    for key, (_field, ids) in BE.registered_units().items():
        if key == BE.FLUID:
            rows = [{"task_id": i, "label": "synthetic", "end": "answered", "messages": [{"role": "system", "content": flu["system_message"]}],
                     "turns": [{**turn, "turn": 0, "answer": "x", "seed": _seed_of(base, f"{key}|{i}|0|0")}]} for i in ids]
            summ = {"composition": "test", "tasks_sha256": flu["tasks_sha256"], "runs": [{"attempt": attempt, "summary": {
                "identity": {**common, "degeneration_check_every": 256, "action_protocol": flu["action_protocol"],
                             "seed_namespace": key, "tasks_sha256": flu["tasks_sha256"]}}}]}
        else:
            rows = [{**turn, "id": i, "sampling_seed": _seed_of(base, f"{key}|{i}|0|0"), "context_len": ctx,
                     "system_source": "default"} for i in ids]
            summ = {**common, "sampling": {"degeneration_check_every": 256, "max_tokens_rule": "per sample"},
                    "protocol": "project", "degeneration_stop": True}
        (gens / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (gens / f"{key}.summary.json").write_text(json.dumps(summ))
    if weights is not None:
        (d / CP.PACKAGES).write_text(json.dumps([{"attempt": attempt, "weights_sha256": weights}]))


def _eval_set_or_skip():
    from shared import eval_sets as ES
    if ES.unit_path("eval-set-1", "stage3_capability_846", "knowledge") is None:
        pytest.skip("eval-set-1's withheld unit has no verified local copy here")


def test_an_empty_report_is_rescored_from_the_sealed_generations_and_nothing_is_generated(tmp_path, monkeypatch):
    """Counter-example 1: a set an earlier attempt left with a ``{}`` formal report is not complete; the driver takes
    its sealed generations over, rescored, and generates nothing."""
    _eval_set_or_skip()
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import phase1_driver as P1
    out = tmp_path / "out"
    d = CP.ev_dir(out, "teacher", "H-T")
    _complete_set(d, "H-T", CP.expected_weights("teacher", out))
    (d / "formal_report.json").write_text("{}\n")
    st = CP.check(d, "teacher", "H-T", out, label="teacher")
    assert not st["complete"] and st["problems"] == {"formal_report": ["empty ({})"]}
    called = []
    for name in ("fluid", "capability"):
        monkeypatch.setattr(RID, name, lambda *a, _n=name, **k: called.append(_n))
    monkeypatch.setattr(P1, "model_for", lambda *a, **k: called.append("model_for"))
    row = P1.evaluate_set("teacher", "H-T", out, tmp_path / "w", "vllm", "a2", RID.Status(tmp_path / "st"))
    assert row["status"] == "complete" and row["generated"] is False and called == []
    rep = json.loads((d / "formal_report.json").read_text())
    assert rep["schema"] == "aadistill.eval_set_report/v2" and all(v["status"] == "measured" for v in rep["units"].values())
    assert json.loads((d / CP.COMPLETION).read_text())["complete"] is True


def test_missing_items_and_wrong_weights_are_never_complete_and_repair_takes_only_what_is_missing(tmp_path):
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import b_evidence as BE
    out = tmp_path / "out"
    d = CP.ev_dir(out, "teacher", "H-N")
    _complete_set(d, "H-N", CP.expected_weights("teacher", out))
    key = "stage3_capability_846__gsm8k"
    rows = [json.loads(x) for x in (d / "generations" / f"{key}.jsonl").read_text().splitlines()]
    (d / "generations" / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:-3] + rows[:1]))
    st = CP.check(d, "teacher", "H-N", out)
    assert not st["complete"] and st["units"][key]["action"] == "recompose"
    assert any("3 missing" in p and "1 repeated" in p for p in st["problems"]["coverage"])
    done = CP.repair(d, st, "a2")
    #: the file's 97 distinct valid records became a sealed part; the 3 missing items are what remains to generate
    assert {"unit": key, "records_given_to_parts": 97, "attempt": "a2"} in done
    assert CP.needs_generation(d) and not (d / "generations" / f"{key}.jsonl").exists()
    assert (d / CP.SUPERSEDED / "a2" / "generations" / f"{key}.jsonl").is_file()
    #: a package naming other weights: every unit is generated again (the earlier records stay superseded)
    _complete_set(d, "H-N", {"model-00001-of-00003.safetensors": "0" * 64}, attempt="a3")
    st = CP.check(d, "teacher", "H-N", out)
    assert any("are not" in p for p in st["problems"]["model_identity"])
    CP.repair(d, st, "a3")
    assert not any((d / "generations").glob("*.jsonl")) and not (d / CP.PACKAGES).exists()
    assert len(list((d / CP.SUPERSEDED / "a3" / "generations").glob("*.jsonl"))) == len(BE.registered_units())


def test_two_different_records_for_one_item_are_refused(tmp_path):
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    out = tmp_path / "out"
    d = CP.ev_dir(out, "teacher", "H-T")
    _complete_set(d, "H-T", CP.expected_weights("teacher", out))
    key = "stage3_capability_846__rag"
    rows = [json.loads(x) for x in (d / "generations" / f"{key}.jsonl").read_text().splitlines()]
    (d / "generations" / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows
                                                            + [{**rows[0], "raw": "another answer"}]))
    st = CP.check(d, "teacher", "H-T", out)
    with pytest.raises(CP.CompletionError, match="two different records"):
        CP.repair(d, st, "a2")


class _Shell:
    """A pod's shell that fails its first ``fail`` listings (a timeout, then a truncated answer), then answers."""

    def __init__(self, root: Path, fail: int):
        from aadistill.infrastructure.remote import LocalShellTarget
        self.real, self.fail, self.calls, self.root = LocalShellTarget(root), fail, 0, root
        (root / "workspace").mkdir(parents=True, exist_ok=True)

    def run(self, command, *, timeout):
        from aadistill.infrastructure.remote import CommandResult
        command = command.replace("/workspace/", f"{self.root}/workspace/")      #: the pod's scratch, here
        if "find ." in command:
            self.calls += 1
            if self.calls <= self.fail:
                return (CommandResult(124, "", "", timed_out=True) if self.calls % 2 else
                        CommandResult(0, "a.json\t12\n", ""))           #: no completion marker: truncated
        return self.real.run(command, timeout=timeout)


def _poll_ctx(tmp_path, fail):
    import types
    from stages.single_operator_budget import launch as L
    launch = L.launch_spec(L.PHASE1_KIND, 60.0)
    pod = tmp_path / "pod"
    ev = Path(L.SE.evidence_dir(launch, "r1").replace(L.SE.POD_REPO, str(pod)))
    (ev / "evaluation" / "live").mkdir(parents=True)
    (ev / "report.json").write_text("{}\n")
    (ev / "evaluation" / "live" / "teacher.a1.stream.jsonl").write_text("partial\n")
    ctx = types.SimpleNamespace(target=_Shell(pod, fail), args=types.SimpleNamespace(run_id="r1", scr=str(tmp_path / "scr")),
                                evidence={}, say=lambda *a: None)
    return L, launch, ctx, ev


def test_an_unreadable_remote_listing_is_retried_and_never_reads_as_nothing_pending(tmp_path, monkeypatch):
    """Counter-example 2: the pod's listing fails (a timeout, a truncated answer). It is retried; if it never reads,
    ``pending`` is unknown (None), never 0, so the products are not secured; a stream under ``evaluation/live/`` is
    never pulled; the final listing's hashes are checked file by file."""
    import shutil
    import time
    from stages.reasoning_inheritance import launch as RIL
    monkeypatch.setattr(time, "sleep", lambda s: None)
    L, launch, ctx, ev = _poll_ctx(tmp_path, fail=3)
    monkeypatch.setattr(L.SE, "evidence_dir", lambda la, run: str(ev))

    def scp(c, remote, dest, limit_s=0):
        src = Path(remote.replace("/workspace", str(tmp_path / "pod" / "workspace")))
        if not src.exists():
            src = Path(remote)
        if src.is_file():
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return 0
        return 1
    monkeypatch.setattr(L.SE, "_scp", scp)
    monkeypatch.setattr(RIL.SE, "_scp", scp)
    (tmp_path / "pod" / "workspace").mkdir(parents=True, exist_ok=True)
    lst = RIL.list_remote(ctx, str(ev), sleep=lambda s: None)
    assert lst is None and "after 3 tries" in ctx.evidence["listing_errors"][-1]
    RIL.secure_incremental(launch)(ctx)                       #: the shell now answers (3 failures spent)
    assert ctx.evidence["secured_files"]["pending"] == 0
    import types
    ctx2 = types.SimpleNamespace(target=_Shell(tmp_path / "pod", fail=99), evidence={}, say=lambda *a: None,
                                 args=types.SimpleNamespace(run_id="r1", scr=str(tmp_path / "scr2")))
    RIL.secure_incremental(launch)(ctx2)
    assert ctx2.evidence["secured_files"]["pending"] is None
    ok, why = RIL.products_secured(launch)(ctx2)
    assert not ok and "pending: None" in why
    #: what the first context pulled: the report, never the live stream; the final listing verifies by sha256
    store = Path(ctx.args.scr) / "store" / "out"
    assert (store / "report.json").is_file() and not any(store.rglob("*.stream.jsonl"))
    fin = RIL.final_hash_check(launch, ctx)
    assert fin["verified"] and fin["files"] == 1
    (store / "report.json").write_text("tampered\n")
    fin = RIL.final_hash_check(launch, ctx)                   #: re-pulled, then equal again
    assert fin["verified"] and (store / "report.json").read_text() == "{}\n"


def test_retain_records_complete_only_when_every_arm_is_on_the_disk_with_its_hashes(tmp_path):
    """Counter-example 3: one arm's weights never arrive. Local retention stays INCOMPLETE, naming it; the arms that
    did arrive are recorded verified; a later retain fetches only the missing arm and then records COMPLETE."""
    import hashlib
    from stages.single_operator_budget import launch as L
    scr, store, record = tmp_path / "scr", tmp_path / "store", tmp_path / "local_retention.json"
    files = {}
    for a in DESIGN["phase_1"]["build_order"]:
        recs = {}
        for name, rel in (("init", "init/model.safetensors"), ("mapping", "mapping/mapping.json")):
            body = f"{a}-{name}".encode()
            files[(a, name)] = body
            recs[name] = {"relay": "r", "prefix": f"durable/x/{a}/init/{a[:4]}", "verified": True,
                          "files_sha256": {rel: hashlib.sha256(body).hexdigest()}, "bytes": {rel: len(body)}}
        p = scr / "store" / "out" / a / "init_record.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"status": "COMPLETE", "arm": a, "relay": recs}))
    lost = DESIGN["phase_1"]["build_order"][1]
    fetched = []

    def getter(rec, dest):
        a = rec["prefix"].split("/")[2]
        name = "init" if "init/model.safetensors" in rec["files_sha256"] else "mapping"
        fetched.append((a, name))
        if a == lost and name == "init":
            raise RuntimeError("the relay download stalled")
        for rel in rec["files_sha256"]:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes(files[(a, name)])
        return {"files": list(rec["files_sha256"])}
    doc = L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)
    assert doc["status"] == "INCOMPLETE" and list(doc["missing"]) == [f"{lost}/init"]
    assert "stalled" in doc["missing"][f"{lost}/init"]
    assert all(v["verified"] for k, v in doc["products"].items() if k != f"{lost}/init")
    assert doc["products"][f"{lost}/init"]["verified"] is False
    #: an arm whose init record never reached the dev box
    (scr / "store" / "out" / lost / "init_record.json").rename(tmp_path / "held")
    assert set(L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)["missing"]) == \
        {f"{lost}/init", f"{lost}/mapping"}
    (tmp_path / "held").rename(scr / "store" / "out" / lost / "init_record.json")
    lost = None
    fetched.clear()
    doc = L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)
    assert doc["status"] == "COMPLETE" and not doc["missing"]
    assert fetched == [(DESIGN["phase_1"]["build_order"][1], "init")]       #: only what was missing is fetched


def test_live_streams_are_excluded_at_any_depth():
    from stages.teacher_parent_initialization import train_driver as TD
    from stages.reasoning_inheritance import launch as RIL
    assert TD.is_live(Path("evaluation/live/x.stream.jsonl")) and TD.is_live(Path("live/x"))
    assert not TD.is_live(Path("evaluation/teacher/evaluation_thinking/parts/part_0000.jsonl"))
    assert "'*/live/*'" in RIL.FIND


def test_the_width_arm_reads_the_sealed_statistics_and_refuses_other_bytes(tmp_path):
    """The residual-width build loads TPI1's sealed statistics instead of a second teacher pass: their sha256, tensor
    layout and statistics positions are the ones selection.json derives from TPI1's committed init record."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from stages.single_operator_budget import phase1_driver as P1
    sel = json.loads((REPO / "configs/stages/stage-1/single_operator_budget/selection.json").read_text())
    st = sel["residual_width"]["statistics"]
    rec = json.loads((REPO / st["tpi1_init_record"]["path"]).read_text())
    assert st["sha256"] == rec["statistics"]["sha256"] == rec["relay"]["statistics"]["files_sha256"][st["file"]]
    assert rec["gates"]["RESIDUAL_WIDTH"]["equal"] and rec["statistics_reproduction"]["equal"]
    tsel = json.loads((REPO / "configs/stages/stage-1/teacher_parent_initialization/selection.json").read_text())
    assert rec["gates"]["RESIDUAL_WIDTH"]["journal_state"] == tsel["states"]["RESIDUAL_WIDTH"]["state_id"]
    assert st["tensors"]["residual_sum"]["shape"] == [37, 2560] and st["stats_tokens"] == 236939
    f = tmp_path / st["file"]
    save_file({"residual_count": torch.tensor([236939]), "residual_sum": torch.zeros(37, 4, dtype=torch.float64)}, str(f))
    with pytest.raises(P1.DriverError, match="hashes to"):
        P1.load_statistics(f, st)
    small = {**st, "sha256": P1.sha256_file(f)}
    with pytest.raises(P1.DriverError, match="layout and positions"):
        P1.load_statistics(f, small)
    ok = {**small, "tensors": {"residual_count": {"shape": [1], "dtype": "torch.int64"},
                               "residual_sum": {"shape": [37, 4], "dtype": "torch.float64"}}}
    state, checks = P1.load_statistics(f, ok)
    assert checks["positions_equal"] and checks["points"] == 37
    #: the pod input is staged by setup at the sealed sha256
    from stages.single_operator_budget import launch as L
    ri = L.launch_spec(L.PHASE1_KIND, 60.0).relay_inputs[0]
    assert (ri.path, ri.sha256) == (st["relay"]["path"], st["sha256"])


def test_each_depth_round_is_persisted_and_resumed_from(tmp_path):
    from stages.single_operator_budget import phase1_driver as P1
    sel = json.loads((REPO / "configs/stages/stage-1/single_operator_budget/selection.json").read_text())["depth_only"]
    out = tmp_path / "out"
    rec = {"round": 8, "removed_before": [r["chosen"] for r in sel["rounds"]], "chosen": 5, "chosen_score": 0.6,
           "n_candidates": 28, "table": []}
    P1.persist_round(out, sel["step"], "a1", rec)
    P1.persist_round(out, sel["step"], "a1", {**rec, "round": 9})
    assert [r["round"] for r in P1.persisted_rounds(out, sel["step"], sel["rounds"])] == [8, 9]
    with pytest.raises(P1.DriverError, match="another step"):
        P1.persisted_rounds(out, {**sel["step"], "config_hash": "x"}, sel["rounds"])
    P1.persist_round(out, sel["step"], "a2", {**rec, "round": 11})
    with pytest.raises(P1.DriverError, match="not \\[8, 9, 10\\]"):
        P1.persisted_rounds(out, sel["step"], sel["rounds"])


def test_the_dev_box_disk_plan_reproduces_and_the_launch_checks_the_real_filesystem(tmp_path, monkeypatch):
    from stages.single_operator_budget import disk as DK
    assert DK.render(DK.plan()) == DK.OUT.read_text()
    pl = DK.plan()
    assert pl["peak_phase"] == "retain"
    inits = sum(i["bytes_bound"] for i in pl["items"] if "initialization" in i["item"])
    assert inits == sum(v["checkpoint_bytes"]["init_bf16"] for v in json.loads(ST.OUT.read_text())["arms"].values())
    assert pl["peak_bytes"]["bound"] == sum(i["bytes_bound"] for i in pl["items"] if "retain" in i["present_during"])
    room = DK.check_room("sob1_phase1", None, str(tmp_path / "scr"))
    assert len(room["filesystems"]) == 1 and room["filesystems"][0]["read_at"] == str(tmp_path)
    monkeypatch.setattr(DK.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 * DK.GIB})())
    assert not DK.check_room("sob1_phase1", None, str(tmp_path / "scr"))["fits"]


def test_phase_1_completes_only_with_eight_sets_and_three_inits_and_its_table_reads_them(tmp_path):
    """The phase check over the real layout (three init records, eight sets), and the comparison it unlocks: per
    mode and arm, the teacher's and the arm's accuracy and the item-paired teacher-minus-arm difference."""
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    out = tmp_path / "out"
    for a in DESIGN["phase_1"]["build_order"]:
        p = out / a / "init_record.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"status": "COMPLETE", "arm": a, "relay": {"init": {"verified": True}},
                                 "checkpoint": {"shards": [{"sha256": f"{a[:2]}" * 32}]}}))
    for t in CP.targets():
        for m in DESIGN["evaluation"]["protocol"]["modes"]:
            if (t, m) == ("depth_only", "H-N"):
                continue                                  #: one set missing
            d = CP.ev_dir(out, t, m)
            _complete_set(d, m, CP.expected_weights(t, out))
            CP.rescore(d, CP.check(d, t, m, out, label=t if t == "teacher" else f"{t}_untrained"))
    done = CP.phase(out)
    assert not done["complete"] and done["n_complete"] == 7 and done["inits"]["durable"]
    with pytest.raises(ValueError, match="not complete on its files"):
        SA.phase1_table(out, tmp_path / "t")
    d = CP.ev_dir(out, "depth_only", "H-N")
    _complete_set(d, "H-N", CP.expected_weights("depth_only", out))
    #: this set's fluid unit was generated in two runs (a takeover): the second submitted only the missing tasks,
    #: so its task file differs; it is still one protocol, and the mode's four sets share one digest
    f = d / "generations" / "fluid_reasoning_720__dev_v1.summary.json"
    summ = json.loads(f.read_text())
    second = json.loads(json.dumps(summ["runs"][0]))
    second["attempt"], second["summary"]["identity"]["tasks_sha256"] = "a2", "0" * 64
    summ["runs"].append(second)
    f.write_text(json.dumps(summ))
    CP.rescore(d, CP.check(d, "depth_only", "H-N", out, label="depth_only_untrained"))
    assert CP.phase(out)["complete"]
    doc = SA.phase1_table(out, tmp_path / "t")
    assert set(doc["modes"]) == {"H-T", "H-N"} and set(doc["modes"]["H-T"]["arms"]) == {
        "residual_width", "ffn_attention", "depth_only"}
    p = doc["modes"]["H-T"]["arms"]["residual_width"]["primary_static_fluid_core"]
    assert p["n_items"] == DESIGN["analysis"]["primary_items"]["n"] == 570 and p["n_groups"] == 558
    assert p["teacher_minus_arm_pp"] == 0.0 and p["teacher_accuracy_pct"] == p["arm_accuracy_pct"]
    md = SA.render_table(doc)
    assert "思考模式" in md and "非思考模式" in md and "teacher − arm, pp" in md
    #: the init record names other weights: the arm's sets are no longer complete
    p = out / "ffn_attention" / "init_record.json"
    p.write_text(p.read_text().replace("ff" * 32, "00" * 32))
    assert not CP.phase(out)["complete"]

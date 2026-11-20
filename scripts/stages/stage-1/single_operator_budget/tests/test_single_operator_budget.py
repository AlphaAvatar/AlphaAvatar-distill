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


def test_the_committed_structures_and_proposals_reproduce():
    """structures.json is the derivation; phase 1's proposal is frozen as granted (its grant binds its sha256);
    phase 2's proposal is the derivation from its own resource snapshot and disk plan."""
    _teacher_or_skip()
    assert ST.render(ST.build()) == ST.OUT.read_text()
    assert PR.phase1_frozen()
    assert PR.render(PR.build_phase2()) == PR.OUT2.read_text()


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
    #: a training session asks the relay for its real rotation peak: one final and two save points
    for a in DESIGN["arms"]:
        b = p["storage"]["per_arm_bytes"][a]
        assert sorted(L.relay_object_sizes(L.TRAIN_KIND, a)) == sorted([b["save_point"], b["save_point"], b["final_fp32"]])
        assert sum(L.relay_object_sizes(L.TRAIN_KIND, a)) == max(r["bytes"] for r in L.rotation_peaks(a))
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


def _proto(mode: str) -> dict:
    """The registered protocol's pieces the synthetic runners write: per-record fields and both runners' summaries."""
    from stages.single_operator_budget import completion as CP
    w = CP.expected(mode)
    cap, flu = w["capability"], w["fluid"]
    ctx, ts = cap["resolved_context"], w["think_state"]
    dp = {"protocol_id": cap["decoding_protocol_id"], "sha256": cap["decoding_sha256"], "sampling": cap["sampling"],
          "samples_per_item": 1, "seed_rule": {"base": cap["seed_rule_base"], "key": cap["seed_rule_key"],
                                               "derivation": cap["seed_rule_derivation"]}}
    common = {"decoding_protocol": dp, "context_resolution": {"resolved_context": ctx, "context_source": "trained_block_len",
                                                              "trained_context": ctx, "architectural_context": 40960},
              "system_message": cap["system_message"], "chat_template_sha256": cap["chat_template_sha256"],
              "chat_template_kwargs": cap["chat_template_kwargs"], "tokenizer_sha256": cap["tokenizer_sha256"],
              "stop_ids": [151645]}
    return {"w": w, "base": cap["seed_rule_base"], "ctx": ctx, "ts": ts, "flu": flu, "common": common}


def _row(key: str, i: str, pr: dict, raw: str = "<think>x</think>\n\nx", label: str = "teacher") -> dict:
    """One record a runner seals for item ``i`` of unit ``key`` under the registered protocol, with the runner's
    ``label``."""
    from stages.teacher_parent_initialization import b_evidence as BE
    turn = {"raw": raw, "stop_reason": "eos", "natural_termination": True, "degeneration_triggered": False,
            "context_limit_reached": False, "generated_tokens": 1, "output_token_ids": [151645], "prompt_tokens": 10,
            "generation_allowance": pr["ctx"] - 10, "think_state": pr["ts"]}
    if key == BE.FLUID:
        return {"task_id": i, "label": label, "end": "answered",
                "messages": [{"role": "system", "content": pr["flu"]["system_message"]}],
                "turns": [{**turn, "turn": 0, "answer": "x", "seed": _seed_of(pr["base"], f"{key}|{i}|0|0")}]}
    return {**turn, "id": i, "label": label, "sampling_seed": _seed_of(pr["base"], f"{key}|{i}|0|0"),
            "context_len": pr["ctx"], "system_source": "default"}


def _fluid_run_summary(pr: dict, key: str) -> dict:
    return {"identity": {**pr["common"], "degeneration_check_every": 256, "action_protocol": pr["flu"]["action_protocol"],
                         "seed_namespace": key, "tasks_sha256": pr["flu"]["tasks_sha256"]}}


def _cap_run_summary(pr: dict) -> dict:
    return {**pr["common"], "sampling": {"degeneration_check_every": 256, "max_tokens_rule": "per sample"},
            "protocol": "project", "degeneration_stop": True}


def _part(d: Path, key: str, gen: str | None, rows: list[dict], n: int = 0) -> Path:
    """A sealed part as the runners' sealer names it (``part_NNNN.<generation>.jsonl``; untagged: before SOB1's second correction)."""
    from stages.teacher_parent_initialization import b_evidence as BE
    root = d / "parts" if key == BE.FLUID else d / "parts" / key
    root.mkdir(parents=True, exist_ok=True)
    f = root / (f"part_{n:04d}" + (f".{gen}" if gen else "") + ".jsonl")
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def _entry(gen: str, target: str, mode: str, weights) -> dict:
    from stages.single_operator_budget import completion as CP
    return {"generation": gen, "attempt": gen.split("-")[0], "target": target, "mode": mode,
            "protocol_sha256": CP.protocol_sha256(mode), "weights_sha256": weights, "parts_at_start": {}}


def _complete_set(d: Path, mode: str, weights: dict | None, attempt: str = "a1", target: str = "teacher") -> None:
    """A synthetic set every check accepts: every registered item once, each record under the registered protocol
    (seed rule, 8,192 context, the mode's think state, a stop id) and sealed in a part of generation ``<attempt>-g1``,
    whose package record names ``weights``, the target, the mode and the registered protocol; each unit's summary
    with its runner fields."""
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import b_evidence as BE
    pr, gen = _proto(mode), CP.generation_id(attempt, target, mode, 1)
    gens = d / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    for key, (_field, ids) in BE.registered_units().items():
        rows = [_row(key, i, pr, label=CP.label_of(target)) for i in ids]
        if key == BE.FLUID:
            summ = {"composition": "test", "tasks_sha256": pr["flu"]["tasks_sha256"],
                    "runs": [{"attempt": gen, "summary": _fluid_run_summary(pr, key)}]}
        else:
            summ = _cap_run_summary(pr)
        _part(d, key, gen, rows)
        (gens / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (gens / f"{key}.summary.json").write_text(json.dumps(summ))
    if weights is not None:
        (d / CP.PACKAGES).write_text(json.dumps([_entry(gen, target, mode, weights)]))


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
    done = CP.repair(d, st, "a2-g1")
    #: the file is superseded; every proven part stays, so the runners compose the unit again from them
    assert any(x.get("unit") == key and x["superseded"] for x in done)
    assert CP.needs_generation(d) and not (d / "generations" / f"{key}.jsonl").exists()
    assert (d / CP.SUPERSEDED / "a2-g1" / "generations" / f"{key}.jsonl").is_file()
    assert all(v["proven"] for v in CP.provenance(d, "teacher", "H-N", out)["parts"].values())
    #: a generation whose package names other weights: its parts are isolated, none is taken
    other = tmp_path / "other"
    od = CP.ev_dir(other, "teacher", "H-N")
    _complete_set(od, "H-N", {"model-00001-of-00003.safetensors": "0" * 64}, attempt="a3")
    st = CP.check(od, "teacher", "H-N", other)
    assert any("other weights" in p for p in st["problems"]["model_identity"])
    assert all(v["action"] == "recompose" for v in st["units"].values())   #: no record a proven part holds
    CP.repair(od, st, "a4-g1")
    assert not any((od / "generations").glob("*.jsonl")) and not CP.part_inventory(od)
    assert (od / CP.PACKAGES).is_file()                                    #: the ledger stays whole
    iso = json.loads((od / CP.SUPERSEDED / "a4-g1" / "isolated_parts.json").read_text())
    assert len(iso) == len(BE.registered_units()) and all("other weights" in v["why"] for v in iso.values())


def test_two_different_records_for_one_item_are_refused(tmp_path):
    """Two proven parts holding one item twice are refused; a generations file's record no proven part holds is
    never taken (the file is composed again from the parts)."""
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    out = tmp_path / "out"
    d = CP.ev_dir(out, "teacher", "H-T")
    _complete_set(d, "H-T", CP.expected_weights("teacher", out))
    key = "stage3_capability_846__rag"
    rows = [json.loads(x) for x in (d / "generations" / f"{key}.jsonl").read_text().splitlines()]
    (d / "generations" / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [{**rows[0], "raw": "b"}]
                                                            + rows[1:]))
    st = CP.check(d, "teacher", "H-T", out)
    assert st["units"][key]["records_no_proven_part_holds"] == 1 and st["units"][key]["action"] == "recompose"
    _part(d, key, CP.generation_id("a1", "teacher", "H-T", 1), [{**rows[0], "raw": "another answer"}], n=1)
    st = CP.check(d, "teacher", "H-T", out)
    assert st["sealed_twice"] == [f"{key}/{rows[0]['id']}"]
    with pytest.raises(CP.CompletionError, match="sealed twice"):
        CP.repair(d, st, "a2-g1")


def _engine(pr: dict, asked: dict):
    """A synthetic engine behind the runners' real merge (`reasoning_inheritance.driver._run_bounded`): run_battery /
    uncapped_eval answer every item asked for, with the ``--label`` given, streamed as it finishes, then the run's
    files."""
    from stages.teacher_parent_initialization import b_evidence as BE
    fl = BE.FLUID

    def engine(cmd, log, env, time_box_s, poll=None):
        log.write_text("synthetic\n")
        arg = lambda f: cmd[cmd.index(f) + 1]  # noqa: E731
        if any("run_battery.py" in c for c in cmd):
            ids = [json.loads(x)["task_id"] for x in Path(arg("--tasks")).read_text().splitlines() if x.strip()]
            asked[fl] = ids
            Path(arg("--stream-completed")).write_text("".join(json.dumps(_row(fl, i, pr, label=arg("--label"))) + "\n"
                                                               for i in ids))
            o = Path(arg("--out"))
            o.write_text("")
            o.with_suffix(".summary.json").write_text(json.dumps(_fluid_run_summary(pr, fl)))
            return 0, False
        files = cmd[cmd.index("--prompts") + 1:cmd.index("--out-dir")]
        raw, stream = Path(arg("--out-dir")), Path(arg("--stream-completed"))
        raw.mkdir(parents=True, exist_ok=True)
        stream.mkdir(parents=True, exist_ok=True)
        for f in files:
            k = Path(f).stem
            ids = [json.loads(x)["id"] for x in Path(f).read_text().splitlines() if x.strip()]
            asked[k] = ids
            body = "".join(json.dumps(_row(k, i, pr, label=arg("--label"))) + "\n" for i in ids)
            (stream / f"{k}.stream.jsonl").write_text(body)
            (raw / f"{k}.generations.jsonl").write_text(body)
            (raw / f"{k}.json").write_text(json.dumps(_cap_run_summary(pr)))
        return 0, False
    return engine


def test_an_old_models_partial_answers_are_isolated_and_the_new_attempt_generates_only_the_rest(tmp_path, monkeypatch):
    """Counter-example (maintainer 2026-11-19, second brief): an attempt that evaluated the WRONG weights stopped with only
    ``parts/`` (no generations file yet); an earlier generation of the right weights sealed a few items; one part
    predates generation tags. The new attempt loads the right model: before anything is merged, every part's
    generation (weights, mode, protocol) and records are checked; the wrong and the unverifiable parts are isolated,
    the proven ones reused, and only the items no proven part holds are generated (the real runners' merge, with a
    synthetic engine)."""
    _eval_set_or_skip()
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import phase1_driver as P1
    from stages.teacher_parent_initialization import b_evidence as BE
    out, mode = tmp_path / "out", "H-T"
    d = CP.ev_dir(out, "teacher", mode)
    pr, units = _proto(mode), BE.registered_units()
    right = CP.expected_weights("teacher", out)
    fl, gsm, rag = BE.FLUID, "stage3_capability_846__gsm8k", "stage3_capability_846__rag"
    G = lambda a: CP.generation_id(a, "teacher", mode, 1)  # noqa: E731
    kept = {fl: units[fl][1][:3], gsm: units[gsm][1][:4]}
    _part(d, fl, G("a0"), [_row(fl, i, pr) for i in kept[fl]])
    _part(d, gsm, G("a0"), [_row(gsm, i, pr) for i in kept[gsm]])
    _part(d, fl, G("a1"), [_row(fl, i, pr, raw="OLD MODEL") for i in units[fl][1][3:8]], n=1)
    _part(d, gsm, G("a1"), [_row(gsm, i, pr, raw="OLD MODEL") for i in units[gsm][1][4:14]], n=1)
    _part(d, rag, None, [_row(rag, i, pr, raw="OLD MODEL") for i in units[rag][1][:2]])
    (d / CP.PACKAGES).write_text(json.dumps([_entry(G("a0"), "teacher", mode, right),
                                             _entry(G("a1"), "teacher", mode, {"model.safetensors": "1" * 64})]))
    st = CP.check(d, "teacher", mode, out, label="teacher")
    why = {rel: v["why"] for rel, v in st["parts"].items() if not v["proven"]}
    assert len(why) == 3 and sum("other weights" in w for w in why.values()) == 2
    assert any("names no generation" in w for w in why.values())
    asked = {}
    engine = _engine(pr, asked)
    monkeypatch.setattr(RID, "_run_bounded", engine)
    monkeypatch.setattr(P1, "model_for", lambda *a, **k: tmp_path / "model")
    monkeypatch.setattr(P1.C, "package", lambda model_dir, dest: {"weights_sha256": right, "linked": []})
    row = P1.evaluate_set("teacher", mode, out, tmp_path / "w", "vllm", "a2", RID.Status(tmp_path / "st"))
    assert row["status"] == "complete" and row["generated"] and len(row["runs"]) == 1
    #: only what no proven part held was asked for; the proven a0 records stand in the units unchanged
    assert asked[fl] == [i for i in units[fl][1] if i not in kept[fl]]
    assert asked[gsm] == [i for i in units[gsm][1] if i not in kept[gsm]] and len(asked[rag]) == len(units[rag][1])
    for key in units:
        text = (d / "generations" / f"{key}.jsonl").read_text()
        assert "OLD MODEL" not in text
    iso = json.loads((d / CP.SUPERSEDED / G("a2") / "isolated_parts.json").read_text())
    assert sorted(iso) == sorted(why)
    assert all((d / CP.SUPERSEDED / G("a2") / rel).is_file() for rel in iso)
    fin = CP.check(d, "teacher", mode, out, label="teacher")
    assert fin["complete"] and {v["generation"] for v in fin["parts"].values()} == {G("a0"), G("a2")}
    assert [e["generation"] for e in CP.packages(d)] == [G("a0"), G("a1"), G("a2")]
    assert CP.packages(d)[-1]["parts_at_start"] and all(f".{G('a0')}." in r for r in CP.packages(d)[-1]["parts_at_start"])


def _real_parts_or_skip() -> Path:
    """TPI1's sealed thinking-mode parts of its A arm at 8,192 (real runner records), if this machine holds them."""
    import glob
    import os
    hits = sorted(glob.glob(os.path.expanduser(
        "~/aad-artifacts/stage1_teacher_parent_init/*/extracted/A_teacher_parent-s*/evaluation_thinking/parts")))
    if not hits:
        pytest.skip("TPI1's sealed evaluation parts (real runner records) are an artifact outside git")
    return Path(hits[0])


def test_a_residual_width_part_filed_in_the_depth_only_set_is_isolated_and_only_missing_items_are_generated(
        tmp_path, monkeypatch):
    """Item 1 (maintainer 2026-11-19, third brief), on real runner records: the residual-width arm's parts (its
    generation id, its label, even its package record) land in the depth-only arm's thinking-mode set; a part named
    for depth only holds residual-width records; a depth-only part of the other mode is there too. Each is isolated
    by what it says about itself -- the generation id's target and mode, each record's runner ``label`` -- while the
    depth-only parts are reused and only the items none holds are generated."""
    _eval_set_or_skip()
    real = _real_parts_or_skip()
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import phase1_driver as P1
    from stages.teacher_parent_initialization import b_evidence as BE
    out, mode, fl, gsm = tmp_path / "out", "H-T", BE.FLUID, "stage3_capability_846__gsm8k"
    for a in ("depth_only", "residual_width"):
        _durable_init_record(out, a)
    d = CP.ev_dir(out, "depth_only", mode)
    R = [json.loads(x) for x in (real / "part_0000.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    Q = [json.loads(x) for x in (real / gsm / "part_0000.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    lab = lambda rows, a: [{**r, "label": CP.label_of(a)} for r in rows]  # noqa: E731
    gd, gr = CP.generation_id("a1", "depth_only", mode, 1), CP.generation_id("a1", "residual_width", mode, 1)
    _part(d, fl, gd, lab(R[:4], "depth_only"))
    _part(d, gsm, gd, lab(Q[:5], "depth_only"))
    _part(d, fl, gr, lab(R[4:9], "residual_width"), n=1)                     #: the misfiled residual-width parts
    _part(d, gsm, gr, lab(Q[5:12], "residual_width"), n=1)
    _part(d, gsm, gd, lab(Q[12:14], "residual_width"), n=2)                  #: right name, another arm's records
    _part(d, fl, CP.generation_id("a1", "depth_only", "H-N", 1), lab(R[9:10], "depth_only"), n=2)
    w = {a: CP.expected_weights(a, out) for a in ("depth_only", "residual_width")}
    (d / CP.PACKAGES).write_text(json.dumps([_entry(gd, "depth_only", mode, w["depth_only"]),
                                             _entry(gr, "residual_width", mode, w["residual_width"])]))
    st = CP.check(d, "depth_only", mode, out, label=CP.label_of("depth_only"))
    why = {rel: v["why"] for rel, v in st["parts"].items() if not v["proven"]}
    assert len(why) == 4 and sum("residual_width/H-T's" in x for x in why.values()) == 2
    assert sum("depth_only/H-N's" in x for x in why.values()) == 1
    assert sum("runner label ['residual_width_untrained']" in x for x in why.values()) == 1
    assert sorted(v["records"] for v in st["parts"].values() if v["proven"]) == [4, 5]
    asked = {}
    monkeypatch.setattr(RID, "_run_bounded", _engine(_proto(mode), asked))
    monkeypatch.setattr(P1, "model_for", lambda *a, **k: tmp_path / "model")
    monkeypatch.setattr(P1.C, "package", lambda model_dir, dest: {"weights_sha256": w["depth_only"], "linked": []})
    row = P1.evaluate_set("depth_only", mode, out, tmp_path / "w", "vllm", "a2", RID.Status(tmp_path / "st"))
    assert row["status"] == "complete"
    units = BE.registered_units()
    assert asked[fl] == [i for i in units[fl][1] if i not in {r["task_id"] for r in R[:4]}]
    assert asked[gsm] == [i for i in units[gsm][1] if i not in {r["id"] for r in Q[:5]}]
    rows = {k: [json.loads(x) for x in (d / "generations" / f"{k}.jsonl").read_text(encoding="utf-8").split("\n")
                if x.strip()] for k in (fl, gsm)}
    assert {r["label"] for k in rows for r in rows[k]} == {"depth_only_untrained"}
    kept = {r["task_id"]: r for r in rows[fl]}
    assert all(kept[r["task_id"]] == {**r, "label": "depth_only_untrained"} for r in R[:4])   #: reused unchanged
    fin = CP.check(d, "depth_only", mode, out, label=CP.label_of("depth_only"))
    assert fin["complete"] and gr not in {v["generation"] for v in fin["parts"].values()}
    assert len(json.loads((d / CP.SUPERSEDED / CP.generation_id("a2", "depth_only", mode, 1)
                           / "isolated_parts.json").read_text())) == 4


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


def _local_scp(pod: Path):
    import shutil

    def scp(c, remote, dest, limit_s=0):
        src = Path(remote.replace("/workspace", str(pod / "workspace")))
        if not src.exists():
            src = Path(remote)
        if src.is_file():
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return 0
        return 1
    return scp


def test_a_part_pulled_then_isolated_on_the_pod_leaves_the_dev_boxs_active_tree_and_every_resume(tmp_path, monkeypatch):
    """Item 2 (maintainer 2026-11-19, third brief): pull first, isolate on the pod, then take over. The dev box pulled
    a part the pod then isolated into ``superseded/``; after the final listing is read, the dev box's active copy is
    withdrawn (its bytes kept where the pod keeps them), so it is in neither the dev box's completion check nor the
    resume archive nor the next attempt's takeover. The poll's ``pending`` also stays honest when the pod has moved a
    pulled file (it read -1 before, so the products could never read as secured)."""
    _eval_set_or_skip()
    import tarfile
    import time
    from stages.reasoning_inheritance import launch as RIL
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import phase1_driver as P1
    from stages.teacher_parent_initialization import b_evidence as BE
    monkeypatch.setattr(time, "sleep", lambda s: None)
    L, launch, ctx, ev = _poll_ctx(tmp_path, fail=0)
    monkeypatch.setattr(L.SE, "evidence_dir", lambda la, run: str(ev))
    monkeypatch.setattr(L.SE, "_scp", _local_scp(tmp_path / "pod"))
    monkeypatch.setattr(RIL.SE, "_scp", _local_scp(tmp_path / "pod"))
    mode, fl, gsm = "H-T", BE.FLUID, "stage3_capability_846__gsm8k"
    pr, units = _proto(mode), BE.registered_units()
    G = lambda a: CP.generation_id(a, "teacher", mode, 1)  # noqa: E731
    d = CP.ev_dir(ev, "teacher", mode)
    _part(d, fl, G("a1"), [_row(fl, i, pr) for i in units[fl][1][:3]])
    _part(d, gsm, G("a1"), [_row(gsm, i, pr) for i in units[gsm][1][:4]])
    _part(d, fl, G("a0"), [_row(fl, i, pr, raw="OLD") for i in units[fl][1][3:6]], n=1)
    _part(d, gsm, G("a0"), [_row(gsm, i, pr, raw="OLD") for i in units[gsm][1][4:9]], n=1)
    (d / CP.PACKAGES).write_text(json.dumps([_entry(G("a1"), "teacher", mode, CP.expected_weights("teacher", ev)),
                                             _entry(G("a0"), "teacher", mode, {"model.safetensors": "1" * 64})]))
    RIL.secure_incremental(launch)(ctx)                       #: 1. pulled: both parts are on the dev box
    store = Path(ctx.args.scr) / "store" / "out"
    rel = lambda p: str(p.relative_to(ev))  # noqa: E731
    wrong = sorted(rel(p) for p in d.rglob(f"part_*.{G('a0')}.jsonl"))
    assert len(wrong) == 2 and all((store / w).is_file() for w in wrong)
    st = CP.check(d, "teacher", mode, ev)                     #: 2. the next attempt isolates them on the pod
    CP.repair(d, st, G("a2"))
    assert not any((ev / w).exists() for w in wrong)
    RIL.secure_incremental(launch)(ctx)                       #: the superseded copies are pulled; nothing pending
    assert ctx.evidence["secured_files"]["pending"] == 0
    moved = [rel(p) for p in (d / CP.SUPERSEDED).rglob("part_*.jsonl")]
    (store / [m for m in moved if gsm in m][0]).unlink()     #: one superseded copy not on the dev box yet
    fin = RIL.final_hash_check(launch, ctx)
    assert fin["verified"] and sorted(x["path"] for x in fin["withdrawn"]) == wrong
    assert {x["action"].split(":")[0] for x in fin["withdrawn"]} == {"moved", "removed"}
    assert not any((store / w).exists() for w in wrong) and all((store / m).is_file() for m in moved)
    dev = CP.ev_dir(store, "teacher", mode)                   #: 3. the dev box's completion check never sees them
    prov = CP.provenance(dev, "teacher", mode, store)
    assert prov["parts"] and all(v["proven"] for v in prov["parts"].values())
    assert sorted(prov["proven_records"][fl]) == sorted(units[fl][1][:3])
    tgz = L.resume_archive(Path(ctx.args.scr), tmp_path / "resume.tgz")        #: 4. nor a resume
    with tarfile.open(tgz) as t:
        names = set(t.getnames())
    assert not names & set(wrong) and set(moved) <= names
    out2 = tmp_path / "next" / "out"
    P1.take_over(tgz, out2)
    nd = CP.ev_dir(out2, "teacher", mode)
    assert all(G("a1") in r for r in CP.part_inventory(nd)) and len(CP.part_inventory(nd)) == 2
    assert all(v["proven"] for v in CP.provenance(nd, "teacher", mode, out2)["parts"].values())


def _product_bodies(a: str) -> dict[str, bytes]:
    """An arm's product files as the manifest names them (the tensor file only for the width arm)."""
    from stages.single_operator_budget import common as C
    return {rel: f"{a}:{rel}".encode() for g in C.required_products(a).values() for rel in sorted(g)}


def _durable_init_record(out: Path, a: str, relay: Path | None = None, bodies: dict | None = None) -> dict:
    """A COMPLETE init record of ``a`` with the shape phase1_driver writes: the checkpoint's and the mapping's
    manifests, each relay group verified with exactly them (the files under ``relay``, when given)."""
    import hashlib
    from stages.single_operator_budget import common as C
    bodies = bodies or _product_bodies(a)
    sha = {rel: hashlib.sha256(b).hexdigest() for rel, b in bodies.items()}
    prefix = f"durable/x/{a}/init/{a[:4]}"
    rec = {"status": "COMPLETE", "arm": a,
           "checkpoint": {"shards": [{"filename": "model.safetensors", "sha256": sha["init/model.safetensors"]}],
                          "files_sha256": {r.removeprefix("init/"): h for r, h in sha.items() if r.startswith("init/")}},
           "mapping": {"content_digest": "cd" * 32, "width_projection": C.width_declared(a),
                       "files_sha256": {r.removeprefix("mapping/"): h for r, h in sha.items() if r.startswith("mapping/")}}}
    rec["relay"] = {g: {"relay": "r", "prefix": prefix, "verified": True, "files_sha256": files,
                        "bytes": {r: len(bodies[r]) for r in files}} for g, files in C.products(rec).items()}
    if relay is not None:
        for rel, b in bodies.items():
            (relay / prefix / rel).parent.mkdir(parents=True, exist_ok=True)
            (relay / prefix / rel).write_bytes(b)
    p = out / a / "init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))
    return rec


def _relay_getter(relay: Path, fetched: list, fail=lambda rec: False):
    def getter(rec, dest):
        fetched.append(rec["prefix"].split("/")[2] + "/" + next(iter(rec["files_sha256"])).split("/")[0])
        if fail(rec):
            raise RuntimeError("the relay download stalled")
        for rel in rec["files_sha256"]:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes((relay / rec["prefix"] / rel).read_bytes())
        return {"files": list(rec["files_sha256"])}
    return getter


def test_retain_records_complete_only_when_every_arm_is_on_the_disk_with_its_hashes(tmp_path):
    """Counter-example 3: one arm's weights never arrive. Local retention stays INCOMPLETE, naming it; the arms that
    did arrive are recorded verified; a later retain fetches only the missing arm and then records COMPLETE. An empty
    relay list or an empty manifest proves nothing."""
    from stages.single_operator_budget import launch as L
    scr, store, record, relay = tmp_path / "scr", tmp_path / "store", tmp_path / "local_retention.json", tmp_path / "relay"
    out = scr / "store" / "out"
    for a in DESIGN["phase_1"]["build_order"]:
        _durable_init_record(out, a, relay)
    lost = DESIGN["phase_1"]["build_order"][1]
    fetched = []
    getter = _relay_getter(relay, fetched, fail=lambda rec: lost is not None and f"/{lost}/" in rec["prefix"]
                           and "init/model.safetensors" in rec["files_sha256"])
    doc = L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)
    assert doc["status"] == "INCOMPLETE" and list(doc["missing"]) == [f"{lost}/init"]
    assert "stalled" in doc["missing"][f"{lost}/init"]
    assert all(v["verified"] for k, v in doc["products"].items() if k != f"{lost}/init")
    assert doc["products"][f"{lost}/init"]["verified"] is False
    width = [a for a in DESIGN["phase_1"]["build_order"] if L.C.width_declared(a)]
    assert width == ["residual_width"] and all(
        set(doc["products"][f"{a}/mapping"]["files_sha256"]) == ({"mapping/mapping.json", "mapping/tensors.safetensors"}
                                                                  if a in width else {"mapping/mapping.json"})
        for a in DESIGN["phase_1"]["build_order"])
    #: an arm whose init record never reached the dev box
    (out / lost / "init_record.json").rename(tmp_path / "held")
    assert set(L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)["missing"]) == \
        {f"{lost}/init", f"{lost}/mapping"}
    (tmp_path / "held").rename(out / lost / "init_record.json")
    lost = None
    fetched.clear()
    doc = L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)
    assert doc["status"] == "COMPLETE" and not doc["missing"]
    assert fetched == [f"{DESIGN['phase_1']['build_order'][1]}/init"]       #: only what was missing is fetched
    #: an empty relay list, then an empty manifest: neither shows the weights saved
    a = DESIGN["phase_1"]["build_order"][0]
    ir = json.loads((out / a / "init_record.json").read_text())
    ir["relay"]["init"]["files_sha256"] = {}
    (out / a / "init_record.json").write_text(json.dumps(ir))
    doc = L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record, getter=getter)
    assert doc["status"] == "INCOMPLETE" and "relay copy holds []" in doc["missing"][f"{a}/init"]
    assert doc["products"][f"{a}/init"]["verified"] is False
    ir["checkpoint"]["files_sha256"], ir["relay"]["init"]["files_sha256"] = {}, {}
    (out / a / "init_record.json").write_text(json.dumps(ir))
    assert "the manifest names []" in L.retain(L.PHASE1_KIND, "r1", scr, store=store, record=record,
                                                getter=getter)["missing"][f"{a}/init"]


def test_each_arms_mapping_manifest_is_what_save_wrote_and_one_list_serves_upload_takeover_restore_and_retain(
        tmp_path, monkeypatch):
    """Three small mappings, one per arm (a width projection; FFN neurons and query heads; kept blocks only), through
    the real save, reload and manifest (`phase1_driver.save_mapping`), the upload (`persist`), the takeover of the
    secured evidence, the restore of the checkpoint (`model_for`) and the local retention: the two arms without a
    width projection need and carry ``mapping.json`` only."""
    torch = pytest.importorskip("torch")
    import shutil
    import types
    from aadistill.initialization.planning import frozen_mapping as FM
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import launch as L
    from stages.single_operator_budget import phase1_driver as P1
    from stages.teacher_parent_initialization import relay as R
    root = {"model": "m", "revision": "r", "root_teacher_sha256": "ab" * 32}
    q, _ = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64))
    maps = {"residual_width": FM.FrozenMapping(root=root, n_root_blocks=2, kept_blocks=(0, 1), width=FM.WidthTransform(
                projection=q.to(torch.float32), scale=1.0, final_norm=torch.ones(4))),
            "ffn_attention": FM.FrozenMapping(root=root, n_root_blocks=2, kept_blocks=(0, 1),
                                              ffn_neurons={0: (0, 2), 1: (1, 3)}, query_heads={0: (0, 1), 1: (2, 3)}),
            "depth_only": FM.FrozenMapping(root=root, n_root_blocks=2, kept_blocks=(1,))}
    work, out, relay = tmp_path / "work", tmp_path / "pod_out", tmp_path / "relay"
    status = RID.Status(tmp_path / "status")

    def put(payload, timeout_s):
        files = {k: Path(v) for k, v in payload["files"].items()}
        for k, v in files.items():
            (relay / payload["prefix"] / k).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(v, relay / payload["prefix"] / k)
        return {"relay": "fake", "prefix": payload["prefix"], "verified": bool(files),
                "files_sha256": {k: P1.sha256_file(relay / payload["prefix"] / k) for k in files},
                "bytes": {k: v.stat().st_size for k, v in files.items()}}
    for arm, m in maps.items():
        prod = work / arm / "products"
        back, man = P1.save_mapping(m, prod / "mapping", arm)
        assert back.content_digest == m.content_digest == man["content_digest"]
        on_disk = sorted(p.name for p in (prod / "mapping").iterdir())
        assert man["files"] == on_disk == (["mapping.json", "tensors.safetensors"] if arm == "residual_width"
                                            else ["mapping.json"])
        (prod / "init").mkdir(parents=True)
        (prod / "init" / "model.safetensors").write_bytes(f"weights of {arm}".encode())
        (prod / "init" / "config.json").write_text("{}")
        ident = types.SimpleNamespace(shards=[types.SimpleNamespace(
            filename="model.safetensors", sha256=P1.sha256_file(prod / "init" / "model.safetensors"))])
        ck = {"artifact_digest": "cd" * 32, "shards": [{"filename": "model.safetensors",
                                                        "sha256": ident.shards[0].sha256}],
              "files_sha256": P1.checkpoint_files(ident, prod / "init")}
        rec = {"run_id": "r1", "arm": arm, "root": root, "checkpoint": ck, "mapping": man, "gates": {},
               "design_sha256": "d", "selection_sha256": "s"}
        P1.write_json(out / arm / "records" / "build.json", rec)
        doc = P1.persist(arm, out, work, "r1", status, put=put)
        assert set(C.products(doc)["mapping"]) == C.required_products(arm)["mapping"]
        assert P1.durable_init(out, arm) == doc
    #: an empty answer from the relay never verifies a group
    with pytest.raises(P1.DriverError, match="not its manifest"):
        P1.persist("depth_only", out, work, "r1", status, put=lambda payload, t: {"verified": True, "files_sha256": {}})
    #: the takeover: a later attempt reads the secured copy, takes every arm as durable, and restores a checkpoint
    secured = tmp_path / "scr" / "store" / "out"
    shutil.copytree(out, secured)
    out2 = tmp_path / "pod2_out"
    P1.take_over(secured, out2)
    assert all(P1.durable_init(out2, a) for a in maps)

    def get(op, payload, timeout_s=0):
        assert op == "get"
        recd = payload["record"]
        for k in recd["files_sha256"]:
            (Path(payload["dest"]) / k).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(relay / recd["prefix"] / k, Path(payload["dest"]) / k)
        return {"verified": True}
    monkeypatch.setattr(R, "run_online", get)
    dest = P1.model_for("depth_only", out2, tmp_path / "work2")
    assert sorted(p.name for p in dest.iterdir()) == ["config.json", "model.safetensors"]
    #: the local retention reads the same manifest
    fetched = []
    doc = L.retain(L.PHASE1_KIND, "r1", tmp_path / "scr", store=tmp_path / "store", record=tmp_path / "local.json",
                   getter=_relay_getter(relay, fetched))
    assert doc["status"] == "COMPLETE" and len(fetched) == 6
    assert set(doc["products"]["depth_only/mapping"]["files_sha256"]) == {"mapping/mapping.json"}
    #: counter-examples: the width arm's relay record without its tensor file; a mapping claiming a tensor file
    #: where the design declares no width projection
    ir = json.loads((out2 / "residual_width" / "init_record.json").read_text())
    ir["relay"]["mapping"]["files_sha256"].pop("mapping/tensors.safetensors")
    (out2 / "residual_width" / "init_record.json").write_text(json.dumps(ir))
    assert P1.durable_init(out2, "residual_width") is None
    assert "residual_width/mapping" in C.product_problems(ir, "residual_width")
    saved = FM.save(maps["ffn_attention"], tmp_path / "m2")
    with pytest.raises(C.SOB1Error, match="declares none"):
        C.mapping_manifest({**saved, "width": {"d_child": 4}, "tensor_file": {"name": "tensors.safetensors",
                                                                              "sha256": "0" * 64}},
                           tmp_path / "m2", "ffn_attention")
    with pytest.raises(C.SOB1Error, match="declares one"):
        C.mapping_manifest(saved, tmp_path / "m2", "residual_width")


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
    _eight_sets(out, skip=("depth_only", "H-N"))
    done = CP.phase(out)
    assert not done["complete"] and done["n_complete"] == 7 and done["inits"]["durable"]
    with pytest.raises(ValueError, match="not complete on its files"):
        SA.phase1_table(out, tmp_path / "t")
    d = CP.ev_dir(out, "depth_only", "H-N")
    _complete_set(d, "H-N", CP.expected_weights("depth_only", out), target="depth_only")
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
    assert "思考模式" in md and "非思考模式" in md and "teacher − arm, pp" in md and "degeneration stop" in md
    term = doc["modes"]["H-T"]["terminations"]
    assert set(term) == {"teacher", "residual_width", "ffn_attention", "depth_only"}
    assert term["teacher"]["primary_static_fluid_core"] == {"turns": 570, "natural_termination": 570,
                                                             "degeneration_stop": 0, "context_limit_hit": 0,
                                                             "other_stop": 0}
    #: the init record names other weights: the arm's sets are no longer complete
    a = "ffn_attention"
    _durable_init_record(out, a, bodies={**_product_bodies(a), "init/model.safetensors": b"other weights"})
    assert not CP.phase(out)["complete"]
    assert any("other weights" in p for p in CP.phase(out)["sets"][f"{a}/H-T"]["problems"]["model_identity"])


def _eight_sets(out: Path, skip=()) -> None:
    """Three durable init records and the eight sets, each complete on its files, its report rescored."""
    from stages.single_operator_budget import completion as CP
    for a in DESIGN["phase_1"]["build_order"]:
        _durable_init_record(out, a)
    for t in CP.targets():
        for m in DESIGN["evaluation"]["protocol"]["modes"]:
            if (t, m) == tuple(skip):
                continue
            d = CP.ev_dir(out, t, m)
            _complete_set(d, m, CP.expected_weights(t, out), target=t)
            CP.rescore(d, CP.check(d, t, m, out, label=t if t == "teacher" else f"{t}_untrained"))


def test_every_closeout_records_phase_1_complete_only_on_the_eight_sets_and_the_local_manifests(tmp_path, monkeypatch):
    """The formal launch's closeout and a manual one run the same check (`Launch.completion_check`): phase 1 is
    COMPLETE only when the eight sets are complete on the secured copy AND every arm's initialization and mapping is
    on the dev box as its manifest; otherwise INCOMPLETE with what is unfinished, the outcome's money record intact."""
    _eval_set_or_skip()
    import functools
    from stages.single_operator_budget import launch as L
    scr, relay = tmp_path / "scr", tmp_path / "relay"
    out = scr / "store" / "out"
    _eight_sets(out)
    for a in DESIGN["phase_1"]["build_order"]:
        _durable_init_record(out, a, relay)
    stall = {"on": True}
    fetched = []
    monkeypatch.setattr(L, "phase1_completion", functools.partial(
        L.phase1_completion, store=tmp_path / "store", record=tmp_path / "local.json", root=tmp_path,
        getter=_relay_getter(relay, fetched, fail=lambda rec: stall["on"] and "/depth_only/" in rec["prefix"])))
    launch = L.launch_spec(L.PHASE1_KIND, 60.0)
    assert launch.completion_check is L.phase1_completion
    assert L.launch_spec(L.TRAIN_KIND, 60.0, arm="depth_only").completion_check.keywords["arm"] == "depth_only"
    close = lambda: L.SE.closeout(L.STUDY, launch, "r1", scr, root=tmp_path, reconcile=lambda: {"ok": True})  # noqa: E731
    res = close()
    rec = json.loads((tmp_path / L.STUDY.run_dir("r1") / L.PHASE1_COMPLETION).read_text())
    assert res["study_completion"]["status"] == rec["status"] == "INCOMPLETE"
    assert set(rec["unfinished"]) == {"local_retention"} and set(rec["unfinished"]["local_retention"]) == {
        "depth_only/init", "depth_only/mapping"}
    assert rec["sets"]["n_complete"] == 8 and rec["inits_durable"]
    assert json.loads((tmp_path / L.STUDY.run_dir("r1") / "closeout/outcome.json").read_text())["money"]["all_in_usd"] == 0.0
    stall["on"] = False
    assert close()["study_completion"]["status"] == "COMPLETE"
    #: a set that is not complete keeps phase 1 INCOMPLETE whatever the local store holds
    from stages.single_operator_budget import completion as CP
    (CP.ev_dir(out, "teacher", "H-N") / "formal_report.json").write_text("{}\n")
    rec = close()["study_completion"]
    assert rec["status"] == "INCOMPLETE" and rec["unfinished"] == ["sets"]


# ---- phase 2 (maintainer 2026-11-20): takeover, completion and the study, through the real entry points --------------

ARM = "residual_width"


def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _mode_engine(asked: dict):
    """`_engine` for either mode: the runners' own ``--chat-template-kwargs`` say which protocol to answer under."""
    engines = {m: _engine(_proto(m), asked.setdefault(m, {})) for m in ("H-T", "H-N")}

    def engine(cmd, log, env, time_box_s, poll=None):
        kw = json.loads(cmd[cmd.index("--chat-template-kwargs") + 1])
        return engines["H-T" if kw.get("enable_thinking") else "H-N"](cmd, log, env, time_box_s, poll)
    return engine


def _final_bytes(seed: int) -> dict[str, bytes]:
    from stages.teacher_modes import a1_units as AU
    return {n: f"{ARM} seed {seed} final {n}".encode() for n in AU.FINAL_FILES}


def _trainer(trained: list):
    """A trainer that finishes the config it is given: the final weights and the completion record of its steps."""
    from aadistill.infrastructure.manifest import sha256_json

    def train(cfg_path, *, resume, deadline_s, log):
        cfg = json.loads(Path(cfg_path).read_text())
        out_dir, steps = Path(cfg["out_dir"]), int(cfg["schedule"]["total_steps"])
        m = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
        m.mkdir(parents=True, exist_ok=True)
        for n, b in _final_bytes(cfg["seed"]).items():
            (m / n).write_bytes(b)
        (out_dir / "run_completion.json").write_text(json.dumps({
            "config_sha256": sha256_json(cfg), "final_step": steps, "planned_total_steps": steps,
            "completed_all_steps": True}))
        trained.append(cfg["seed"])
        return 0
    return train


def _phase2_plan(tmp: Path) -> dict:
    from stages.single_operator_budget import launch as L
    plan = json.loads((REPO / L.plan_path(ARM)).read_text())
    for u in plan["units"]:
        u["config"]["out_dir"] = str(tmp / "pod" / u["unit"] / "train")
        u["config"]["student_path"] = str(tmp / "pod" / "init")
    (tmp / "plan.json").write_text(json.dumps(plan))
    return plan


def _run_phase2(tmp: Path, monkeypatch, *, out: Path, resume: Path | None = None) -> dict:
    """One `train_driver.main` attempt: everything but the state machine, the completion checks, the runners' merge
    and the relay's contract stood in for (the trainer, the engine, the package, the pod's environment)."""
    from shared import study_engineering as SE
    import shared.pod.teacher_cache as TC
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import launch as L
    from stages.single_operator_budget import train_driver as SD
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_modes import packed_training_plan as PTP
    from stages.teacher_modes import training_plan as TP
    from stages.teacher_parent_initialization import train_driver as TD
    trained, asked, fetched = [], {}, []
    monkeypatch.setattr(L.Authorization, "load", classmethod(lambda cls, p: object()))
    monkeypatch.setattr(SE, "require_on_pod", lambda *a, **k: {"authorization_id": "test"})
    monkeypatch.setattr(SE, "identity_hash", lambda *a, **k: "h")
    monkeypatch.setattr(TC, "local_snapshot", lambda *a, **k: str(tmp))
    monkeypatch.setattr(TP, "stage_tokenizers", lambda snap: {})
    monkeypatch.setattr(TP, "admit", lambda *a, **k: None)
    monkeypatch.setattr(PTP, "attention_problems", lambda cfg, repo: [])
    monkeypatch.setattr(AU, "train_telemetry", lambda d: {"context_strategies": [PTP.CONTEXT_STRATEGY]})
    monkeypatch.setattr(TD, "fetch_init", lambda u, d=None: fetched.append(u["unit"]) or {"stub": True})
    monkeypatch.setattr(TD, "_sleep", lambda s: None)
    monkeypatch.setattr(AU, "train", _trainer(trained))
    monkeypatch.setattr(RID, "_run_bounded", _mode_engine(asked))
    monkeypatch.setattr(C, "package", lambda model_dir, dest: {
        "weights_sha256": {"model.safetensors": _sha((Path(model_dir) / "model.safetensors").read_bytes())},
        "linked": ["model.safetensors"]})
    argv = ["--authorization", "a.json", "--out", str(out), "--status", str(tmp / "status.log"), "--plan",
            str(tmp / "plan.json"), "--unit-deadline-hours", "1", "--attempt", "sob1_train_test2"]
    rc = SD.main(argv + (["--resume", str(resume)] if resume else []))
    return {"rc": rc, "units": json.loads((out / "units.json").read_text()), "trained": trained, "asked": asked,
            "fetched": fetched}


def _sealed_final(out: Path, relay, seed: int, tmp: Path) -> dict:
    """A seed's final made durable as the driver does it: on the (fake) relay, handed off, recorded in the evidence."""
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import train_driver as TD
    unit = CP.unit_name(ARM, seed)
    src = out.parent / "lost_pod" / unit
    src.mkdir(parents=True, exist_ok=True)
    files = {}
    for n, b in _final_bytes(seed).items():
        (src / n).write_bytes(b)
        files[n] = str(src / n)
    plan = json.loads((tmp / "plan.json").read_text())         #: the plan this test's driver reads (`_phase2_plan`)
    cfg = next(u for u in plan["units"] if u["seed"] == seed)["config"]
    from aadistill.infrastructure.manifest import sha256_json
    (src / "run_completion.json").write_text(json.dumps({"config_sha256": sha256_json(cfg),
                                                         "final_step": cfg["schedule"]["total_steps"],
                                                         "planned_total_steps": cfg["schedule"]["total_steps"],
                                                         "completed_all_steps": True}))
    files["run_completion.json"] = str(src / "run_completion.json")
    prefix = f"durable/stage1_single_operator_budget/{ARM}/{unit}"
    rec = relay.put({"files": files, "prefix": f"{prefix}/final/{_sha(_final_bytes(seed)['model.safetensors'])[:16]}"})
    rec["handoff"] = TD.handoff(rec, src, f"{prefix}/final", "test", newest=False)
    (out / unit / "train").mkdir(parents=True, exist_ok=True)
    (out / unit / "final.json").write_text(json.dumps(rec))
    (out / unit / "train" / "run_completion.json").write_text((src / "run_completion.json").read_text())
    return rec


def _trained_set(out: Path, seed: int, mode: str, *, attempt: str = "sob1_train_test1", of_seed: int | None = None):
    """A complete sealed set of ``seed``'s target, generated from ``of_seed``'s final weights (default: its own)."""
    from stages.single_operator_budget import completion as CP
    t = CP.train_target(ARM, seed)
    src_t = CP.train_target(ARM, of_seed or seed)
    d = CP.ev_dir(out, t, mode)
    _complete_set(d, mode, None, attempt=attempt, target=src_t)
    gen = CP.generation_id(attempt, src_t, mode, 1)
    w = {"model.safetensors": _sha(_final_bytes(of_seed or seed)["model.safetensors"])}
    pt = CP.parse_train_target(src_t)
    (d / CP.PACKAGES).write_text(json.dumps([{**_entry(gen, src_t, mode, w), "arm": pt["arm"], "seed": pt["seed"],
                                              "unit": pt["unit"]}]))
    return d


def test_empty_reports_without_final_weights_are_never_done_and_the_unit_is_trained(tmp_path, monkeypatch):
    """The 84c48bea reproduction: a previous attempt left a ``{}`` formal_report.json in both modes of both seeds and
    no final weights anywhere. TPI1's state function called that unit ``done``; this driver trains each seed, makes
    its final durable and generates every set from it, each part's package naming the arm, seed, unit and weights."""
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    from stages.teacher_parent_initialization import relay as R
    relay = FakeRelay()
    monkeypatch.setattr(R, "run_online", relay)
    _phase2_plan(tmp_path)
    prev = tmp_path / "devbox/store/out"
    dirs = []
    for s in DESIGN["training"]["seeds"]:
        for m in DESIGN["evaluation"]["protocol"]["modes"]:
            d = CP.ev_dir(prev, CP.train_target(ARM, s), m)
            d.mkdir(parents=True)
            (d / "formal_report.json").write_text("{}\n")
            dirs.append(d)
    #: the defect: the old state function read the report files alone
    assert AU.state(tmp_path / "pod/none", dirs) == "done"
    assert CP.session(prev, ARM)["complete"] is False
    got = _run_phase2(tmp_path, monkeypatch, out=tmp_path / "attempt2/out", resume=prev)
    assert got["rc"] == 0 and got["units"]["terminal"] == "ALL_DONE", got["units"].get("error")
    assert got["trained"] == DESIGN["training"]["seeds"]
    assert all(got["units"]["units"][CP.unit_name(ARM, s)]["state_at_start"] == "train" for s in DESIGN["training"]["seeds"])
    out = tmp_path / "attempt2/out"
    ses = CP.session(out, ARM)
    assert ses["complete"] and ses["n_complete"] == 4
    for k, v in ses["sets"].items():
        t = k.rsplit("/", 1)[0]
        seed = CP.parse_train_target(t)["seed"]
        assert v["binding"]["weights_sha256"] == {"model.safetensors": _sha(_final_bytes(seed)["model.safetensors"])}
        assert (v["binding"]["arm"], v["binding"]["seed"]) == (ARM, seed)
    assert [c for c in relay.calls if c[0] == "put" and "/final/" in c[1] and not c[1].endswith("final")]


def test_an_empty_report_beside_a_durable_final_is_rescored_without_training_or_loading_a_model(tmp_path, monkeypatch):
    """Both seeds' finals durable, every set's generations sealed from them, the formal reports ``{}``: the driver
    rescores the reports from the sealed generations; nothing is trained, restored or generated."""
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    from stages.teacher_parent_initialization import relay as R
    relay = FakeRelay()
    monkeypatch.setattr(R, "run_online", relay)
    _phase2_plan(tmp_path)
    prev = tmp_path / "devbox/store/out"
    for s in DESIGN["training"]["seeds"]:
        _sealed_final(prev, relay, s, tmp_path)
        for m in DESIGN["evaluation"]["protocol"]["modes"]:
            (_trained_set(prev, s, m) / "formal_report.json").write_text("{}\n")
    before = CP.session(prev, ARM)
    assert not before["complete"] and all(v["problems"] == {"formal_report": ["empty ({})"]}
                                          for v in before["sets"].values())
    relay.calls.clear()
    got = _run_phase2(tmp_path, monkeypatch, out=tmp_path / "attempt2/out", resume=prev)
    assert got["rc"] == 0 and got["units"]["terminal"] == "ALL_DONE", got["units"].get("error")
    assert got["trained"] == [] and got["fetched"] == [] and not any(got["asked"].values())
    assert not [c for c in relay.calls if c[0] in ("get", "put") and "/final/" in str(c[1])]
    assert CP.session(tmp_path / "attempt2/out", ARM)["complete"]
    for s in DESIGN["training"]["seeds"]:
        row = got["units"]["units"][CP.unit_name(ARM, s)]
        assert row["state_at_start"] == "final_durable" and "restored_final" not in row
        assert all(e["generated"] is False for e in row["evaluations"].values())


def test_another_seeds_parts_are_isolated_and_only_the_missing_items_come_from_this_seeds_restored_final(
        tmp_path, monkeypatch):
    """Seed 20260726's thinking set holds parts sealed by seed 20260801's generation (its target, package and
    weights) and a few of its own; its final is durable on the relay only. The other seed's parts are isolated, the
    final is restored (never retrained), and only the items no proven part of THIS seed holds are generated."""
    _eval_set_or_skip()
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import b_evidence as BE
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    from stages.teacher_parent_initialization import relay as R
    relay = FakeRelay()
    monkeypatch.setattr(R, "run_online", relay)
    _phase2_plan(tmp_path)
    s1, s2 = DESIGN["training"]["seeds"]
    prev = tmp_path / "devbox/store/out"
    for s in (s1, s2):
        _sealed_final(prev, relay, s, tmp_path)
    for s, m in ((s1, "H-N"), (s2, "H-T"), (s2, "H-N")):
        _trained_set(prev, s, m)
        CP.rescore(CP.ev_dir(prev, CP.train_target(ARM, s), m),
                   CP.check(CP.ev_dir(prev, CP.train_target(ARM, s), m), CP.train_target(ARM, s), m, prev))
    #: the wrong seed's shards filed in seed 1's thinking set, plus three of seed 1's own
    d = _trained_set(prev, s1, "H-T", of_seed=s2)
    t1, pr, units = CP.train_target(ARM, s1), _proto("H-T"), BE.registered_units()
    own = CP.generation_id("sob1_train_test0", t1, "H-T", 1)
    keep = units[BE.FLUID][1][:3]
    _part(d, BE.FLUID, own, [_row(BE.FLUID, i, pr, label=t1) for i in keep], n=1)
    pk = json.loads((d / CP.PACKAGES).read_text())
    pk.append({**_entry(own, t1, "H-T", {"model.safetensors": _sha(_final_bytes(s1)["model.safetensors"])}),
               "arm": ARM, "seed": s1, "unit": CP.unit_name(ARM, s1)})
    (d / CP.PACKAGES).write_text(json.dumps(pk))
    st = CP.check(d, t1, "H-T", prev, label=t1)
    wrong = [v for v in st["parts"].values() if not v["proven"]]
    assert not st["complete"] and len(wrong) == len(units) and all(f"{CP.train_target(ARM, s2)}/H-T's" in v["why"]
                                                                   for v in wrong)
    got = _run_phase2(tmp_path, monkeypatch, out=tmp_path / "attempt2/out", resume=prev)
    assert got["rc"] == 0 and got["units"]["terminal"] == "ALL_DONE", got["units"].get("error")
    assert got["trained"] == []
    row = got["units"]["units"][CP.unit_name(ARM, s1)]
    assert row["restored_final"]["model_safetensors_sha256"] == _sha(_final_bytes(s1)["model.safetensors"])
    assert got["asked"]["H-T"][BE.FLUID] == [i for i in units[BE.FLUID][1] if i not in keep]
    assert all(len(got["asked"]["H-T"][k]) == len(units[k][1]) for k in units if k != BE.FLUID)
    assert "H-N" not in got["asked"] or not got["asked"]["H-N"]     #: the complete sets were taken as they were
    out = tmp_path / "attempt2/out"
    d2 = CP.ev_dir(out, t1, "H-T")
    fin = CP.check(d2, t1, "H-T", out, label=t1)
    assert fin["complete"] and {v["generation"] for v in fin["parts"].values()} == {
        own, CP.generation_id("sob1_train_test2", t1, "H-T", 1)}
    iso = json.loads((d2 / CP.SUPERSEDED / CP.generation_id("sob1_train_test2", t1, "H-T", 1)
                      / "isolated_parts.json").read_text())
    assert len(iso) == len(units) and CP.session(out, ARM)["complete"]


def _arm_session_evidence(out: Path, arm: str) -> dict[str, dict]:
    """One arm's secured copy as a finished session leaves it: both finals durable and published, four sets
    complete against them, reports rescored. Returns each final's file bodies by unit."""
    from stages.single_operator_budget import completion as CP
    bodies = {}
    for s in DESIGN["training"]["seeds"]:
        unit, t = CP.unit_name(arm, s), CP.train_target(arm, s)
        b = {n: f"{arm} {s} {n}".encode() for n in ("model.safetensors", "config.json", "run_completion.json")}
        bodies[unit] = b
        (out / unit).mkdir(parents=True, exist_ok=True)
        (out / unit / "final.json").write_text(json.dumps({
            "relay": "fake", "prefix": f"durable/stage1_single_operator_budget/{arm}/{unit}/final/x", "verified": True,
            "files_sha256": {n: _sha(v) for n, v in b.items()}, "bytes": {n: len(v) for n, v in b.items()},
            "handoff": {"verified": True}}))
        w = {"model.safetensors": _sha(b["model.safetensors"])}
        for m in DESIGN["evaluation"]["protocol"]["modes"]:
            d = CP.ev_dir(out, t, m)
            _complete_set(d, m, None, target=t, attempt="r1")
            gen = CP.generation_id("r1", t, m, 1)
            (d / CP.PACKAGES).write_text(json.dumps([{**_entry(gen, t, m, w), "arm": arm, "seed": s, "unit": unit}]))
            CP.rescore(d, CP.check(d, t, m, out))
    return bodies


def test_every_training_closeout_checks_its_arm_and_the_study_completes_only_with_six_finals_and_twelve_sets(
        tmp_path, monkeypatch):
    """Each sob1_train closeout (the launch's and a manual one: `Launch.completion_check`) records its arm COMPLETE only
    with both finals durable, four sets complete against them and both finals re-hashed on the dev box; the study is
    COMPLETE only when all three arms are, and its registered comparison is recomputed from the sealed records."""
    _eval_set_or_skip()
    import functools
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import launch as L
    bodies = {}
    for a in DESIGN["training"]["unit_order"]["arms"]:
        bodies.update(_arm_session_evidence(tmp_path / f"scr_{a}" / "store" / "out", a))

    def getter(rec, dest):
        unit = rec["prefix"].split("/")[3]
        for n in rec["files_sha256"]:
            (Path(dest) / n).write_bytes(bodies[unit][n])
        return {"verified": True}
    monkeypatch.setattr(L, "phase2_completion", functools.partial(
        L.phase2_completion, store=tmp_path / "store", record=tmp_path / "local2.json", getter=getter, root=tmp_path))
    first = DESIGN["training"]["unit_order"]["arms"][0]
    launch = L.launch_spec(L.TRAIN_KIND, 60.0, arm=first)
    assert launch.completion_check.keywords["arm"] == first
    #: one arm's {} report: that arm is INCOMPLETE on the formal closeout, and its money record stands
    scr = tmp_path / f"scr_{first}"
    d = CP.ev_dir(scr / "store/out", CP.train_target(first, 20260726), "H-T")
    good = (d / "formal_report.json").read_text()
    (d / "formal_report.json").write_text("{}\n")
    res = L.SE.closeout(L.STUDY, launch, "t1", scr, root=tmp_path, reconcile=lambda: {"ok": True})
    assert res["study_completion"]["status"] == "INCOMPLETE" and res["study_completion"]["unfinished"] == ["sets"]
    assert json.loads((tmp_path / L.STUDY.run_dir("t1") / "closeout/outcome.json").read_text())["money"]["all_in_usd"] == 0.0
    (d / "formal_report.json").write_text(good)
    res = L.SE.closeout(L.STUDY, launch, "t1", scr, root=tmp_path, reconcile=lambda: {"ok": True})
    assert res["study_completion"]["status"] == "COMPLETE"
    assert res["study_completion"]["study"]["status"] == "INCOMPLETE"        #: two arms have no session yet
    for i, a in enumerate(DESIGN["training"]["unit_order"]["arms"][1:], start=2):
        out = L.phase2_completion(f"t{i}", tmp_path / f"scr_{a}", arm=a)
        assert out["status"] == "COMPLETE"
    assert out["study"]["status"] == "COMPLETE"
    doc = json.loads((tmp_path / L.STUDY_RESULT).read_text())
    assert doc["completion"]["finals"] == {"n_verified": 6, "n_expected": 6}
    assert doc["completion"]["sets"] == {"n_complete": 12, "n_expected": 12}
    dec = doc["readout"]["decision"]
    assert dec["_rule"]["m"] == 3 and dec["_rule"]["seeds"] == DESIGN["training"]["seeds"]
    pairs = [k for k in dec["H-T"] if not k.startswith("_")]
    assert len(pairs) == 3 and all(dec["H-T"][k]["status"] == "formal" and dec["H-T"][k]["holm_m"] == 3 for k in pairs)
    assert all(dec["H-T"][k]["verdict"] == "NOT_DISTINGUISHED" for k in pairs)   #: identical records: no difference
    assert all(v["status"] == "descriptive_only" for k, v in dec["H-N"].items() if not k.startswith("_"))
    #: recomputed from the same sealed records, the readout is the same
    again = L.study_completion(root=tmp_path, write=False)
    assert again["readout"]["decision"] == doc["readout"]["decision"]
    #: the six finals are on the dev box as their records
    rec = json.loads((tmp_path / "local2.json").read_text())
    assert sum(v["verified"] for k, v in rec["products"].items() if k.endswith("/final")) == 6   #: every arm kept
    assert len(list((tmp_path / "store").glob("*-s*/final/*/model.safetensors"))) == 6


def test_phase_1_is_closed_in_code_and_phase_2_is_funded_once_by_its_own_grant(tmp_path, monkeypatch, capsys):
    """Item 2 of the maintainer's 2026-11-20 brief: issue and launch refuse a new phase-1 session (the remainder is
    void) while closeout, retain and the completion check stay; phase 2 -- the Stage-3 recovery training and its
    evaluation -- has its own proposal / grant, per-arm caps, and an envelope that counts only sob1_train runs."""
    import math
    from stages.single_operator_budget import launch as L
    pos = {"fetched_utc": "t", "provider_billed_usd": 974.1887, "committed_usd": 974.1887, "project_cap_usd": 1000.0,
           "project_remaining_usd": 25.8113, "account_pods": [], "account_volumes": [], "balance_usd": 35.8108}
    monkeypatch.setattr(L.SE, "live_position", lambda root=None, snapshot=None: dict(pos))
    b = L.budget(L.PHASE1_KIND)
    assert not b["fundable"] and b["remaining_usd"] == 0.0 and "CLOSED" in b["problems"][0]
    assert L.main(["issue", "--kind", L.PHASE1_KIND, "--run-id", "x", "--expected-usd", "1", "--hard-cap-usd", "1",
                   "--dry-run", "--out", str(tmp_path / "a.json")]) == 11
    assert L.main(["launch", "--kind", L.PHASE1_KIND, "--run-id", "x", "--scr", str(tmp_path), "--max-price", "1.1",
                   "--dry-run", "--authorization", str(tmp_path / "a.json")]) == 11
    assert capsys.readouterr().out.count("sob1_phase1 kind is CLOSED") == 2 and not (tmp_path / "a.json").exists()
    by = {c.__name__: c for c in L.SE.prechecks(L.STUDY, L.launch_spec(L.PHASE1_KIND, 60.0), L.REPO)}
    assert not by["the_study_is_open"](None)[0]
    L.SE.require_open(L.STUDY)                       #: the study is not closed: phase 2 can still be granted under it
    L.SE.require_kind_open(L.STUDY, L.TRAIN_KIND)
    #: phase 2's money: its own proposal until a grant exists, its own envelope (no phase-1 run in it)
    g = L.kind_grant(L.TRAIN_KIND, dry_run=True)
    assert g["_source"] == str(L.PHASE2_PROPOSAL) and g["status"] == "proposed"
    env = L.SE.envelope(L.STUDY, g, L.REPO, L.TRAIN_KIND)
    assert env["spent_usd"] == 0.0 and env["runs"] == [] and env["remaining_usd"] == g["ceiling_usd"]
    caps = {a: v["per_session_hard_cap_usd"] for a, v in g["kinds"][L.TRAIN_KIND]["per_arm"].items()}
    assert g["ceiling_usd"] == round(sum(caps.values()), 2) and g["teardown_reserve_usd"] == 1.0
    assert g["per_session_hard_cap_usd"] == max(caps.values()) and g["kinds"][L.TRAIN_KIND]["order"][0] == "residual_width"
    basis = g["project_cap_basis"]
    assert g["project_cumulative_cap_usd"] == math.ceil(round((basis["committed_usd"] + g["ceiling_usd"] + 1.0) * 100, 6)) / 100
    assert "no separate Stage-3 allowance" in g["money_rules"]["once"]
    b2 = L.budget(L.TRAIN_KIND, "residual_width")
    assert b2["hard_cap_usd"] == caps["residual_width"] and not b2["fundable"]
    assert any("project cap" in x for x in b2["problems"])     #: the cap a new maintainer decision sets, never inferred
    with pytest.raises(L.SE.AuthorizationError, match="no grant at .*phase2_grant"):
        L.kind_grant(L.TRAIN_KIND)                          #: a real issue reads only a grant
    #: each committed plan is the derivation the proposal binds
    for a, v in g["plans"].items():
        assert v["sha256"] == L.SE.sha256_file(REPO / v["path"])


def test_phase_2_disk_plan_counts_finals_evidence_and_archives_on_the_one_filesystem():
    from stages.single_operator_budget import disk as DK
    st = json.loads(ST.OUT.read_text())["arms"]
    pl = DK.plan("sob1_train", ARM, "/s")
    names = " | ".join(i["item"] for i in pl["items"])
    for w in ("evidence", "pull archive", "teardown collector", "resume archive", "s20260726 final", "s20260801 final"):
        assert w in names
    assert pl["peak_phase"] == "retain"
    finals = sum(st[a]["checkpoint_bytes"]["final_fp32"] for a in st) * len(DESIGN["training"]["seeds"])
    doc = json.loads(DK.OUT_PHASE2.read_text())
    again = DK.phase2_total(free_bytes=doc["free_bytes"], archive_bytes=doc["phase1_archive_bytes"])
    assert DK.render(again) == DK.OUT_PHASE2.read_text()
    assert doc["peak_bytes"]["expected"] >= finals and doc["sessions"][-1]["arm"] == DESIGN["training"]["unit_order"]["arms"][-1]
    assert doc["shortfall_with_swing_bytes"]["bound"] == max(0, doc["peak_bytes"]["bound"]
                                                            + doc["shared_filesystem_swing_bytes"] - doc["free_bytes"])

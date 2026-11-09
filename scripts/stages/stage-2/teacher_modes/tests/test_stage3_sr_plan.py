"""Stage 3 of six_category_schema_restored@v1 (maintainer 2026-11-07): the six units are bound to their arm's frozen
pack and their own teacher lineage's initialization under the frozen recipe; save points come about every hour; the
three kinds refuse until the grant's Stage-3 block approves the plan as it is, attach only the granted volume, relay
their packs; and a later attempt resumes each unit from the volume -- the latest save point, or only the evaluation of
a trained unit -- and never trains past what the volume can hold."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stages.teacher_modes import a1_units as AU
from stages.teacher_modes import packed_training_plan as PTP
from stages.teacher_modes import token_matched_phase as TM
from stages.teacher_modes import training_plan as TP

PLAN = json.loads(PTP.BOUND.read_text())
BUDGET = json.loads(PTP.BUDGET.read_text())
#: the six units' kinds (the mixed arm's kind has its own plan and grant block: test_mixed_arm.py)
SR_ARMS = {k: a for k, a in TM.PACKED_TRAIN_ARMS.items() if TM.STAGE3_BLOCKS[k] == TM.STAGE3_BLOCK}


def test_the_six_units_are_bound_to_their_frozen_packs_and_their_own_lineage():
    fr = PTP.frozen()["arms"]
    assert [(u["arm"], u["seed"]) for u in PLAN["units"]] == [(a, s) for a in PTP.ARMS for s in PTP.SEEDS]
    top = json.loads((PTP.SEARCH_EVIDENCE / "products.json").read_text())["top1"]
    for u in PLAN["units"]:
        cfg, f = u["config"], fr[u["arm"]]
        data = cfg["lineage"]["data"][0]
        assert data["files_sha256"] == f["identity"]["pack_files_sha256"]
        assert data["packed_content_sha256"] == f["identity"]["packed_content_sha256"]
        assert data["data_manifest_content_sha256"] == f["identity"]["content_sha256"]
        assert cfg["rung"] == f["before_packing"]["training"]["unique_supervised_tokens"]
        assert cfg["data_dir"] == f"{PTP.PACKS}/{u['arm']}" and cfg["val_blocks"] == f["packing"]["validation_sequences"]
        lineage = PTP.LINEAGE[u["arm"]][0]
        t = TP.teachers()[lineage]
        assert cfg["teacher"] == {"model_id": t["model_id"], "revision": t["revision"], "dtype": "bfloat16"}
        assert cfg["tokenizer_source"] == TP.TOKENIZERS[lineage]
        assert cfg["lineage"]["tokenizer"]["files_sha256"] == TP.tokenizer_files()[lineage]
        init = cfg["lineage"]["initialization"]
        if u["arm"] == "C1-6cat-sr":            #: Thinking-2507's own D4 Top-1, never the hybrid's search
            assert init["model_safetensors_sha256"].startswith("fc28e0dd") and init["provenance"]["record"] == str(
                TP.D4_SEARCH.relative_to(TP.REPO))
        else:                                   #: the packed search's Top-1, by its committed search record
            assert cfg["student_path"] == top["path"]
            assert init["model_safetensors_sha256"] == top["files_sha256"]["model.safetensors"] \
                == "b74dede833fba34478a3319fb0b9c98b415e2f5b307983e2441599101cf83b6d"
            assert init["provenance"]["record"].endswith("a_packed_search_20261105_195011/evidence/search_record.json")
        #: the frozen recipe: the objective, 3 epochs of batch 2 x micro-batch 1, E1's optimizer and schedule rule
        assert cfg["loss"] == TP.OBJECTIVE
        assert cfg["batch"] == {"blocks_per_step": 2, "micro_blocks": 1, "context_strategy": "cross_context"}
        sch = TP.schedule(f["packing"]["packed_sequences"])
        assert cfg["schedule"] == {k: sch[k] for k in ("total_steps", "warmup_steps", "min_lr_frac")}
        assert cfg["optim"] == json.loads(TP.E1.read_text())["optim"] and cfg["intervals"]["eval_every"] == sch["eval_every"]
        assert u["effective_schedule"]["optimizer_steps"] == f["compute"]["optimizer_steps"]
    assert all(v["problems"] == [] for v in PTP.config_problems(PLAN["units"]).values())
    assert PLAN["comparisons"]["A2_over_A1_optimizer_steps"] == pytest.approx(3.53, abs=0.01)


def test_the_committed_plan_is_what_the_frozen_records_give_now():
    got = PTP.units()
    assert [u["config"] for u in got] == [u["config"] for u in PLAN["units"]]


def test_save_points_come_about_every_hour_and_only_the_newest_is_kept():
    for u in PLAN["units"]:
        ck, es = u["config"]["checkpoint"], u["effective_schedule"]
        s = PTP.seconds_per_step(es["kd_positions_per_step"], "expected")
        assert abs(ck["save_every"] * s - PTP.SAVE_TARGET_S) <= s and ck["keep_last"] == 1 and ck["bind_identity"]
        assert es["save_points"][-1] == es["optimizer_steps"]          #: the final save point always exists
        assert ck["save_every"] < es["optimizer_steps"] // 2           #: no longer T // 2


def test_the_budget_prices_the_plans_steps_and_the_kinds_read_their_deadlines():
    for u in PLAN["units"]:
        row = BUDGET["per_unit"][u["unit"]]
        assert row["lower"]["unit_s"] < row["expected"]["unit_s"] < row["upper"]["unit_s"]
        assert row["upper"]["evaluation_s"] == PTP.EVAL_S["upper"]
    for kind, arm in SR_ARMS.items():
        a = BUDGET["arms"][arm]
        assert a["kind"] == kind and a["kind_cumulative_cap_usd"] > a["session_hard_cap"]["hard_cap_usd"]
        assert TM.unit_deadline_hours(kind) == a["unit_deadline_hours"]
        upper_train_h = max(BUDGET["per_unit"][f"{arm}-s{s}"]["upper"]["train_s"] for s in PTP.SEEDS) / 3600
        assert upper_train_h <= a["unit_deadline_hours"] < upper_train_h + 0.25
    assert BUDGET["stage3_cumulative_cap_usd"] == pytest.approx(
        sum(a["kind_cumulative_cap_usd"] for a in BUDGET["arms"].values()), abs=0.01)
    #: the runner's own formula, on the packed search's 2,584 driver minutes, gives its granted $55.72
    assert PTP.runner_cap(2584)["hard_cap_usd"] == pytest.approx(55.72, abs=0.02)


def grant_root(tmp_path, block: bool, spent_usd: float | None = None, kind: str = "train_sr_a1"):
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    if block:
        caps = {k: BUDGET["arms"][a]["kind_cumulative_cap_usd"] for k, a in SR_ARMS.items()}
        g["session_hard_caps_usd"] = {**g["session_hard_caps_usd"], **caps}
        g[TM.STAGE3_BLOCK] = {"approves": TM.STAGE3_APPROVAL, "granted_by": "maintainer", "decision": "test",
                              "date": "2026-11-07", "cap_is_cumulative": True, "gpu": TM.GPU,
                              "plan_sha256": TM.P1._sha(TM.KIND_RECORDS[kind][0], TM.REPO_ROOT),
                              "budget_sha256": TM.P1._sha(TM.KIND_RECORDS[kind][1], TM.REPO_ROOT),
                              "kinds": {k: {"arm": a, "hard_cap_usd": caps[k]} for k, a in SR_ARMS.items()},
                              "network_volume": {"id": "thprcgh7s9"}, "protected_volumes": ["4rhubprswo", "am5m373kuq"]}
    (tmp_path / TM.STUDY.grant).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / TM.STUDY.grant).write_text(json.dumps(g))
    for rel in TM.KIND_RECORDS[kind]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((TM.REPO_ROOT / rel).read_bytes())
    if spent_usd is not None:
        run = tmp_path / TM.STUDY.logs / "runs" / f"a_{kind}_1"
        (run / "governance").mkdir(parents=True)
        (run / "closeout").mkdir()
        (run / "governance/authorization.json").write_text(json.dumps(
            {"plan_id": TM.STUDY.plan_id, "stage_conditions": {"kind": kind}}))
        (run / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": spent_usd}}))
    return tmp_path, g


def test_the_committed_grant_admits_each_kind_up_to_its_cumulative_cap_only():
    """maintainer 2026-11-07, 'Stage 3 六单元正式授权', section IV"""
    for kind, arm in SR_ARMS.items():
        left = round(BUDGET["arms"][arm]["kind_cumulative_cap_usd"] - TM.kind_spent(kind)["usd"], 4)
        assert TM.kind_cap_problems(kind, left) == []                     #: what every earlier attempt left
        assert TM.kind_cap_problems(kind, round(left + 0.01, 4))
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    assert g["ceiling_usd"] == 158.3762 and g["project_cumulative_cap_usd"] == 1000.0
    assert g["research_a_cumulative"]["cap_usd"] <= 263 and g[TM.STAGE3_BLOCK]["stage_cumulative_cap_usd"] == 73.24
    assert TM.kind_cap_problems("packed_search", 1.0)                                 #: closed for new attempts


def test_the_stage3_kinds_need_the_grants_block_for_the_plan_as_it_is(tmp_path):
    root, g = grant_root(tmp_path / "z", block=False)
    del g[TM.STAGE3_BLOCK]
    (root / TM.STUDY.grant).write_text(json.dumps(g))
    assert TM.kind_cap_problems("train_sr_a1", 1.0, root)                             #: no block, no Stage 3
    root, g = grant_root(tmp_path / "a", block=True)
    cap = g["session_hard_caps_usd"]["train_sr_a1"]
    assert TM.kind_cap_problems("train_sr_a1", cap, root) == []
    root, _ = grant_root(tmp_path / "b", block=True, spent_usd=5.0)                   #: cumulative over attempts
    assert TM.kind_cap_problems("train_sr_a1", round(cap - 5.0, 2), root) == []
    bad = TM.kind_cap_problems("train_sr_a1", round(cap - 4.99, 2), root)
    assert bad and "cumulative" in bad[0]
    root, _ = grant_root(tmp_path / "c", block=True)
    plan = root / TM.KIND_RECORDS["train_sr_a1"][0]
    plan.write_text(plan.read_text().replace("20260726", "20260727", 1))              #: not the approved plan
    assert any("plan_sha256" in b for b in TM.kind_cap_problems("train_sr_a1", cap, root))


def test_a_stage3_session_attaches_only_the_granted_volume(tmp_path, monkeypatch):
    root, _ = grant_root(tmp_path, block=True)
    monkeypatch.setattr(TM, "REPO_ROOT", root)
    ctx = lambda vid: SimpleNamespace(args=SimpleNamespace(network_volume_id=vid))  # noqa: E731
    assert TM.the_training_volume_is_the_granted_one(ctx("thprcgh7s9"))[0]
    assert not TM.the_training_volume_is_the_granted_one(ctx("4rhubprswo"))[0]
    assert not TM.the_training_volume_is_the_granted_one(ctx("othervol"))[0]


def test_each_kind_relays_its_frozen_pack_and_c1_its_own_initialization():
    fr = PTP.frozen()["arms"]
    for kind, arm in SR_ARMS.items():
        ri = TM.stage3_relay_inputs(kind)
        packs = {Path(r.path).name: r.sha256 for r in ri if r.path.startswith("transfer/stage3_sr/")}
        assert packs == fr[arm]["identity"]["pack_files_sha256"]
        assert all(r.dest == f"{PTP.PACKS}/{arm}" for r in ri if r.path.startswith("transfer/"))
        init = [r for r in ri if not r.path.startswith("transfer/")]
        if arm == "C1-6cat-sr":
            assert {Path(r.path).name: r.sha256[:8] for r in init}["model.safetensors"] == "fc28e0dd"
            assert all(r.dest == PLAN["units"][-1]["config"]["student_path"] for r in init)
        else:
            assert init == []               #: the Top-1 is read on the search volume, never moved
        assert TM.DRIVERS[kind][2] == ()    #: no pack is pushed from the dev box


# ---- the driver across attempts --------------------------------------------------------------------------------

def driver_fixture(tmp_path, monkeypatch, states: dict[str, str]):
    """Two C1-6cat-sr units in a scratch durable root, each left in ``states[unit]`` by an earlier attempt; the
    trainer, the evaluation and the pod-side checks are recorded instead of run."""
    from aadistill.infrastructure import manifest as M
    durable = tmp_path / "vol" / "teacher_modes"
    us = [json.loads(json.dumps(u)) for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    calls = {"train": [], "evaluate": [], "attention": [], "copy": []}
    for u in us:
        out = durable / "stage3_sr" / u["unit"] / "train"
        u["config"]["out_dir"] = str(out)
        T = u["config"]["schedule"]["total_steps"]
        st = states[u["unit"]]
        (out / "checkpoints").mkdir(parents=True)
        if st in ("resume", "trained", "done"):
            (out / "checkpoints" / "latest.txt").write_text("step_000858\n")
        if st in ("trained", "done"):
            (out / "checkpoints" / f"step_{T:06d}" / "model").mkdir(parents=True)
            (out / "checkpoints" / f"step_{T:06d}" / "model" / "model.safetensors").write_bytes(b"w")
            (out / "run_completion.json").write_text("{}")
        if st == "done":
            ev = durable / "stage3_sr" / u["unit"] / "evaluation"
            ev.mkdir(parents=True)
            (ev / "formal_report.json").write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({**PLAN, "units": us}))

    strategy = ["cross_context"]

    def train(cfg_path, *, resume, deadline_s, log):
        cfg = json.loads(Path(cfg_path).read_text())
        calls["train"].append((cfg["run_name"], resume))
        d = Path(cfg["out_dir"]) / "checkpoints" / f"step_{cfg['schedule']['total_steps']:06d}" / "model"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"w")
        with (Path(cfg["out_dir"]) / "train_log.jsonl").open("a") as fh:     #: what the trainer logs per step
            fh.write(json.dumps({"event": "train_step", "step": 1, "seconds": 3.2, "kd_positions": 8000,
                                 "context_strategy": strategy[0]}) + "\n")
        return 0

    def evaluate(job):
        calls["evaluate"].append(job["label"])
        calls.setdefault("package", []).append(job.get("package"))
        (job["out"]).mkdir(parents=True, exist_ok=True)
        (job["out"] / "formal_report.json").write_text("{}")
        return {}, {}
    from shared.pod import teacher_cache
    from stages.reasoning_data import h1pa_driver as HD
    monkeypatch.setattr(TM, "require_on_pod", lambda *a, **k: {"authorization_id": "test"})
    monkeypatch.setattr(teacher_cache, "local_snapshot", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(TP, "admit", lambda *a, **k: None)
    monkeypatch.setattr(AU, "train", train)
    monkeypatch.setattr(HD, "evaluate", evaluate)
    monkeypatch.setattr(M, "sha256_file", lambda p: "0" * 64)
    monkeypatch.setattr(PTP, "attention_problems", lambda cfg, root=None: calls["attention"].append(cfg["run_name"]) or [])

    def persist(unit_dir, unit, model_dir, sha256):
        calls["copy"].append(unit)
        out = {"verified": False, "refused": "quota"}
        (Path(unit_dir) / AU.OFF_VOLUME).write_text(json.dumps(out))
        return out
    monkeypatch.setattr(AU, "persist_off_volume", persist)
    calls["strategy"] = strategy
    argv = ["--authorization", "x", "--durable-root", str(durable), "--status", str(tmp_path / "status"),
            "--unit-deadline-hours", "1", "--kind", "train_sr_c1", "--plan", str(plan), "--arm", "C1-6cat-sr",
            "--units-dir", "stage3_sr"]
    return argv, calls, durable, [u["unit"] for u in us]


@pytest.mark.parametrize("first,second,trained,evaluated", [
    ("fresh", "fresh", [False, False], 2),         #: a first attempt trains both seeds from the initialization
    ("resume", "fresh", [True, False], 2),         #: a later one resumes the first from its latest save point
    ("trained", "resume", [True], 2),              #: a trained unit is only evaluated; never trained again
    ("done", "trained", [], 1),                    #: a scored unit is skipped; nothing is measured twice
])
def test_a_later_attempt_resumes_each_unit_where_the_volume_left_it(tmp_path, monkeypatch, first, second, trained,
                                                                     evaluated):
    names = [u["unit"] for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    argv, calls, durable, _ = driver_fixture(tmp_path, monkeypatch, {names[0]: first, names[1]: second})
    assert AU.main(argv) == 0
    assert [r for _, r in calls["train"]] == trained and len(calls["evaluate"]) == evaluated
    rec = json.loads((durable / "C1-6cat-sr_units.json").read_text())
    assert [rec["units"][n]["state_at_start"] for n in names] == [first, second] and rec["terminal"] == "ALL_DONE"
    if first == "resume":
        assert rec["units"][names[0]]["resume_from"] == "step_000858"


def test_a_session_whose_volume_cannot_hold_its_units_trains_nothing(tmp_path, monkeypatch):
    names = [u["unit"] for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    argv, calls, durable, _ = driver_fixture(tmp_path, monkeypatch, {names[0]: "fresh", names[1]: "fresh"})
    assert AU.main(argv + ["--volume-gb", "10"]) == 3                 #: 2 x 8.3 GB + a save point's 5.9 GB > 10 GB
    rec = json.loads((durable / "C1-6cat-sr_units.json").read_text())
    assert calls["train"] == [] and "volume" in rec["error"]["error"] and rec["terminal"] == "RUN_FAILED"
    argv, calls, durable, _ = driver_fixture(tmp_path / "b", monkeypatch, {names[0]: "done", names[1]: "trained"})
    assert AU.main(argv + ["--volume-gb", "10"]) == 0                 #: one evaluation package left to write
    rec = json.loads((durable / "C1-6cat-sr_units.json").read_text())
    assert rec["volume"]["need_bytes"] == PTP.MODEL_BYTES


def test_a_trained_unit_is_copied_off_the_volume_before_its_evaluation_and_a_refusal_never_fails(tmp_path,
                                                                                                 monkeypatch):
    names = [u["unit"] for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    argv, calls, durable, _ = driver_fixture(tmp_path, monkeypatch, {names[0]: "done", names[1]: "fresh"})
    assert AU.main(argv) == 0
    #: the scored unit's copy is still owed (retried at the start), the fresh one is copied once it is trained
    assert calls["copy"] == names and calls["attention"] == [f"research_a_stage3_C1-6cat-sr_s{PTP.SEEDS[1]}"]
    rec = json.loads((durable / "C1-6cat-sr_units.json").read_text())
    assert rec["units"][names[1]]["off_volume_copy"]["refused"] == "quota" and rec["terminal"] == "ALL_DONE"
    status = (tmp_path / "status").read_text()
    assert status.index(f"OFF_VOLUME_NOT_COPIED:{names[1]}") < status.index(f"EVAL_START:{names[1]}")


def test_a_unit_the_trainer_ran_under_another_attention_fails(tmp_path, monkeypatch):
    names = [u["unit"] for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    argv, calls, durable, _ = driver_fixture(tmp_path, monkeypatch, {names[0]: "fresh", names[1]: "fresh"})
    calls["strategy"][0] = "document_isolated"
    assert AU.main(argv) == 3
    rec = json.loads((durable / "C1-6cat-sr_units.json").read_text())
    assert "document_isolated" in rec["error"]["error"] and calls["evaluate"] == []


def test_the_attention_check_reads_the_config_the_pack_and_the_trainers_reader(tmp_path):
    u = PLAN["units"][-1]                       #: C1-6cat-sr, the smallest pack
    assert PTP.attention_problems(u["config"]) == []
    cfg = json.loads(json.dumps(u["config"]))
    del cfg["batch"]["context_strategy"]        #: the trainer's default is not a declaration
    assert any("context_strategy" in b for b in PTP.attention_problems(cfg))
    import numpy as np
    src = TP.REPO / cfg["data_dir"]
    dst = tmp_path / "pack"
    dst.mkdir()
    for n in ("ladder.json", "audit.jsonl"):
        (dst / n).write_bytes((src / n).read_bytes())
    with np.load(src / "blocks.npz") as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(dst / "blocks.npz", **arrays, position_ids=np.zeros_like(arrays["input_ids"]))
    cfg = {**u["config"], "data_dir": str(dst)}
    bad = PTP.attention_problems(cfg, root=Path("/"))
    assert any("position ids" in b for b in bad) and any("reader" in b for b in bad)


def test_the_copy_uploads_nothing_the_relays_quota_refuses(tmp_path, monkeypatch):
    import huggingface_hub
    from aadistill.runtime import hub_capacity
    from aadistill.infrastructure.manifest import sha256_file
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "config.json").write_text("{}")
    (tmp_path / "token").write_text("t")
    sha, uploads = sha256_file(d / "model.safetensors"), []

    class Api:
        def __init__(self, token=None):
            pass

        def get_paths_info(self, *a, **k):
            return []

        def upload_file(self, *, path_or_fileobj, path_in_repo, **k):
            uploads.append(path_in_repo)
    verdict = {"ok": False}
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, **k: str(d / Path(path).name))
    monkeypatch.setattr(hub_capacity, "would_accept", lambda repo, sizes, token: SimpleNamespace(
        accepted=verdict["ok"], __str__=lambda self: "probe"))
    out = AU.copy_final_off_volume(d, "C1-6cat-sr-s20260726", sha, token_file=tmp_path / "token", relay="r/x")
    assert out["verified"] is False and "quota" in out["refused"] and uploads == []
    verdict["ok"] = True
    out = AU.copy_final_off_volume(d, "C1-6cat-sr-s20260726", sha, token_file=tmp_path / "token", relay="r/x")
    assert out["verified"] and sorted(uploads) == [f"durable/stage3_sr/C1-6cat-sr-s20260726/{sha[:16]}/{n}"
                                                    for n in ("config.json", "model.safetensors")]
    assert AU.copy_final_off_volume(d, "u", "0" * 64, token_file=tmp_path / "token")["verified"] is False


def test_c1_is_evaluated_with_its_lineages_pinned_tokenizer_and_template(tmp_path, monkeypatch):
    """train_sr_c1 attempt 1 (2026-11-08): the package carried tokenizer.json and tokenizer_config.json only, and
    Thinking-2507's template lives in chat_template.jinja -- the engine started with no template and the first
    prompt raised. The package now names the lineage's pinned files, the ones the historical C1 package used."""
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256
    from transformers import AutoTokenizer
    names = [u["unit"] for u in PLAN["units"] if u["arm"] == "C1-6cat-sr"]
    argv, calls, durable, _ = driver_fixture(tmp_path, monkeypatch, {names[0]: "trained", names[1]: "done"})
    assert AU.main(argv) == 0 and len(calls["package"]) == 1
    pkg = calls["package"][0]
    assert pkg["sidecar_sha256"] == TP.T2507_TOKENIZER_FILES == TOKENIZER_SIDECAR_SHA256
    src = Path(pkg["tokenizer_source"])
    for keep, has_template in ((("tokenizer.json", "tokenizer_config.json"), False), (tuple(pkg["sidecar_sha256"]), True)):
        d = tmp_path / f"tok{len(keep)}"
        d.mkdir()
        for n in keep:
            (d / n).write_bytes((src / n).read_bytes())
        assert bool(AutoTokenizer.from_pretrained(str(d)).chat_template) is has_template

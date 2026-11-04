"""Thinking-2507's formal tool generation (maintainer 2026-11-04, review of a143b7ed): the formal source admission
both sides apply, the in-session L40S device admission that runs before any formal invocation, and the kind's
cumulative $2.78 grant. Nothing here generates or spends."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stages.teacher_modes import token_matched_phase as TM
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_session as TS
from stages.teacher_modes import tool_source_admission as TSA

PROTO = json.loads(TGP.PROTOCOL.read_text())
CFG = json.loads(TS.DEVICE_ADMISSION["thinking_2507"].read_text())
MODE = "T2507"


def _evidence(**over):
    """What a passing admission invocation records: the runner's identity, one engine row, the GPU sampler."""
    li = PROTO["lineages"]["thinking_2507"]
    ident = {"model": li["model"], "revision": li["revision"], "chat_template_sha256": CFG["teacher"]["chat_template_sha256"],
             "chat_template_kwargs": {}, "sampling": dict(li["modes"][MODE]["sampling"]),
             "stop_ids": sorted(PROTO["execution"]["stop_ids"]), "session_limit": 8184, "effective_limit": 8184,
             "engine": {"max_model_len": 8184, "gpu_memory_utilization": 0.8, "dtype": "bfloat16", "vllm_version": "0.27.1"}}
    ev = {"identity": ident, "engine": [{"engine_ready_seconds": [104.9], "kv_cache_tokens": [195616.0]}],
          "gpu": {"samples": 40, "gpus": {"0": {"name": "NVIDIA L40S", "memory_total_mib": 46068,
                                                "peak_memory_used_mib": 37749}}},
          "complete": True, "turns": 12, "log_text": "", "files": {"differ": []}, "free_wanted_gib": 4.0,
          "vllm_pin": "0.27.1"}
    ev.update(over)
    return ev


def _verdict(ev):
    return TS.device_verdict(CFG, PROTO, "thinking_2507", MODE, **ev)


def test_the_measured_teacher_engine_and_margin_admit():
    v = _verdict(_evidence())
    assert v["admitted"] and v["failed"] == []


def _with(path, value):
    ev = _evidence()
    d = ev
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return ev


@pytest.mark.parametrize("ev,failed", [
    (_with(("identity", "revision"), "0" * 40), "teacher_revision"),
    (_with(("identity", "model"), "Qwen/Qwen3-4B"), "teacher_model"),
    (_with(("identity", "chat_template_sha256"), "a55ee1b1"), "chat_template"),
    (_with(("identity", "chat_template_kwargs"), {"enable_thinking": False}), "template_kwargs"),
    (_with(("identity", "engine", "vllm_version"), "0.26.0"), "vllm_lock"),
    (_with(("identity", "engine", "max_model_len"), 40960), "context"),
    (_with(("identity", "effective_limit"), 4096), "context"),
    (_with(("identity", "engine", "gpu_memory_utilization"), 0.9), "gpu_memory_utilization"),
    (_with(("files",), {"differ": ["tokenizer.json"]}), "teacher_files"),
    (_with(("engine",), [{"engine_ready_seconds": None, "kv_cache_tokens": None}]), "engine_started_with_a_kv_cache"),
    (_with(("engine",), [{"engine_ready_seconds": [9.0], "kv_cache_tokens": [4000.0]}]), "engine_started_with_a_kv_cache"),
    (_with(("log_text",), "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"), "no_out_of_memory"),
    (_with(("complete",), False), "generation_complete"),
    (_with(("gpu", "gpus", "0", "peak_memory_used_mib"), 43000), "memory_margin"),
    (_with(("gpu",), {"samples": 0, "gpus": {}}), "memory_margin"),
    (_with(("vllm_pin",), "0.28.0"), "vllm_lock"),
])
def test_each_criterion_refuses_on_its_own(ev, failed):
    v = _verdict(ev)
    assert not v["admitted"] and failed in v["failed"]


def test_the_admission_record_names_the_protocol_teacher_and_the_pin():
    assert TM.device_admission_config_problems("tool_generation_t2507") == []
    assert CFG["engine"]["vllm_version"] == TS.pinned_vllm()
    assert (CFG["teacher"]["model"], CFG["teacher"]["revision"]) == (PROTO["lineages"]["thinking_2507"]["model"],
                                                                     PROTO["lineages"]["thinking_2507"]["revision"])


def test_teacher_files_hashes_every_registered_file(tmp_path):
    import hashlib
    (tmp_path / "a.json").write_text("x")
    teacher = {"files_sha256": {"a.json": hashlib.sha256(b"x").hexdigest()},
               "weight_shards_sha256": {"w.safetensors": hashlib.sha256(b"w").hexdigest()},
               "weight_shards_bytes": {"w.safetensors": 1}}
    assert TS.teacher_files(teacher, tmp_path)["differ"] == ["w.safetensors"]
    (tmp_path / "w.safetensors").write_bytes(b"w")
    assert TS.teacher_files(teacher, tmp_path)["differ"] == []
    assert TS.teacher_files(teacher, None)["differ"]


def _plan_and_tasks():
    plan = json.loads((TM.REPO_ROOT / "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json").read_text())
    cells = plan["lineages"]["thinking_2507"]["modes"][MODE]["cells"]
    by_id = {c["task_ids"][0]: {"task_id": c["task_ids"][0], "split": c["split"], "category": c["category"],
                                "source": {"dataset": "d", "revision": "r"}} for c in cells}
    return plan, by_id, [c["task_ids"][0] for c in cells]


def _fake_group(tmp_path, *, revision=None, raise_=None):
    def run_group(c, out, inputs, f, k, ids, run, mark, *, label=None, extra_args=()):
        if raise_:
            raise raise_
        rows = [json.loads(x) for x in (inputs / f["file"]).read_text().splitlines()]
        ident = _evidence()["identity"]
        if revision:
            ident["revision"] = revision
        (out / f"{label}.seg0.raw.summary.json").write_text(json.dumps({"identity": ident}))
        (out / f"{label}.seg0.log").write_text("Engine ready in 104.9 seconds\n")
        assert k == 0 and "--engine-metrics" in extra_args and label.startswith("admission_")
        return {"complete": True, "segments": [{"segment": 0}],
                "records": {r["task_id"]: {"turns": [{"generated_tokens": 5}]} for r in rows}}
    return run_group


@pytest.fixture
def admission_env(tmp_path, monkeypatch):
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import tool_pilot as TPL
    monkeypatch.setattr(TS, "snapshot_dir", lambda m, r: tmp_path)
    monkeypatch.setattr(TS, "teacher_files", lambda t, s: {"snapshot": str(s), "differ": []})
    monkeypatch.setattr(TPL, "engine_telemetry",
                        lambda log, summary=None: {"engine_ready_seconds": [104.9], "kv_cache_tokens": [195616.0]})

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def as_dict(self):
            return _evidence()["gpu"]
    monkeypatch.setattr(TPL, "GpuSampler", Sampler)
    return GD


def test_one_invocation_on_the_first_task_of_every_cell_then_the_verdict(tmp_path, monkeypatch, admission_env):
    plan, by_id, first = _plan_and_tasks()
    monkeypatch.setattr(admission_env, "run_group", _fake_group(tmp_path))
    marks = []
    rec = TS.run_device_admission(PROTO, plan, "thinking_2507", by_id, tmp_path, "id", None, marks.append)
    assert rec["admitted"] and rec["tasks"] == first and len(first) == 8
    assert json.loads((tmp_path / "device_admission.json").read_text())["admitted"]
    assert marks[-1] == "DEVICE_ADMISSION_OK"
    assert (tmp_path / "device_admission" / "inputs" / f"admission_{MODE}.jsonl").is_file()
    assert not list(tmp_path.glob("tool_*"))                       #: nothing under a formal label


@pytest.mark.parametrize("how", ["another revision", "the engine raised"])
def test_a_failed_admission_raises_with_its_evidence_written(tmp_path, monkeypatch, admission_env, how):
    plan, by_id, _ = _plan_and_tasks()
    fake = (_fake_group(tmp_path, revision="0" * 40) if how == "another revision"
            else _fake_group(tmp_path, raise_=RuntimeError("CUDA out of memory")))
    monkeypatch.setattr(admission_env, "run_group", fake)
    marks = []
    with pytest.raises(TS.DeviceAdmissionRefused):
        TS.run_device_admission(PROTO, plan, "thinking_2507", by_id, tmp_path, "id", None, marks.append)
    rec = json.loads((tmp_path / "device_admission.json").read_text())
    assert not rec["admitted"] and marks[-1] == "DEVICE_ADMISSION_FAILED"


@pytest.mark.parametrize("lineage,admits", [("thinking_2507", True), ("thinking_2507", False), ("hybrid", None)])
def test_the_formal_session_admits_the_device_before_any_formal_invocation(tmp_path, monkeypatch, lineage, admits):
    order = []
    monkeypatch.setattr(TM, "require_on_pod", lambda p, kind, root: {"identity": "id", "authorization_id": "a"})
    monkeypatch.setattr(TS, "checked_plan", lambda proto, li: ({"pool": {"content_sha256": "c"}}, None, {}))

    def admission(*a, **k):
        order.append("admission")
        if not admits:
            raise TS.DeviceAdmissionRefused("refused")
        return {"admitted": True, "failed": [], "tasks": []}
    monkeypatch.setattr(TS, "run_device_admission", admission)

    def generate(proto, plan, li, mode, *a, telemetry=None, extra_args=(), **k):
        order.append(("generate", mode, telemetry is not None, "--engine-metrics" in extra_args))
        return {"complete": True, "groups": [], "cells": []}
    monkeypatch.setattr(TS, "generate_mode", generate)
    code = TS.main(["--authorization", str(tmp_path / "a.json"), "--lineage", lineage, "--durable-root", str(tmp_path),
                    "--status", str(tmp_path / "s")], run=lambda *a, **k: None)
    if lineage == "hybrid":
        assert code == 0 and "admission" not in order and all(o[2] and o[3] for o in order)
    elif admits:
        assert code == 0 and order[0] == "admission" and order[1] == ("generate", MODE, True, True)
    else:
        assert code == 3 and order == ["admission"]
        rec = json.loads((tmp_path / "tool_generation_thinking_2507" / "tool_session.json").read_text())
        assert rec["terminal"] == "RUN_FAILED" and "DeviceAdmissionRefused" in rec["error"]["error"]


def test_only_the_hybrid_takes_the_generation_scoped_gate(monkeypatch):
    """The hybrid's generation is the one the L40S check measured; Thinking-2507's is not, so its gate is the
    in-session admission's record, and the scoped gate is never asked about it."""
    from stages.teacher_modes import tool_pilot as TPL
    seen = []
    monkeypatch.setattr(TPL, "l40s_generation_problems",
                        lambda root, kind, lineage: seen.append((kind, lineage)) or ["unmeasured"])
    monkeypatch.setattr(TM, "device_admission_config_problems", lambda kind, root: ["no record"])
    with pytest.raises(TM.SE.AuthorizationError, match="unmeasured"):
        TM.kind_conditions("tool_generation")
    with pytest.raises(TM.SE.AuthorizationError, match="no record"):
        TM.kind_conditions("tool_generation_t2507")
    assert seen == [("tool_generation", "hybrid")]


# --- the formal source admission ------------------------------------------------------------------------------

REC = TSA.load()


def _task(ds="Agent-Ark/Toucan-1.5M", rev="0df3cf37f2abefb380370cfb02eabea2a35ae782"):
    return {"source": {"dataset": ds, "revision": rev}}


def test_the_signed_record_admits_the_reviewed_content_and_its_two_sources():
    assert TSA.problems(REC, pool_content_sha256=TSA.REVIEWED_CONTENT, tasks=[_task()]) == []
    assert REC["sources"] == ["Agent-Ark/Toucan-1.5M@0df3cf37f2abefb380370cfb02eabea2a35ae782",
                              "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1@"
                              "bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e"]


@pytest.mark.parametrize("change,match", [
    (lambda r: r, "pool content"),
    (lambda r: r["signature"].update(approves="everything"), "approves"),
    (lambda r: r["signature"].update(dict.fromkeys(TSA.SIGNATURE_FIELDS)), "not signed"),
    (lambda r: r.update(does_not=r["does_not"][:-1]), "scope"),
    (lambda r: r["signature"].update(sources=r["sources"][:1]), "sources is not the record's"),
])
def test_the_admission_refuses(change, match):
    rec = json.loads(json.dumps(REC))
    content = "0" * 64 if match == "pool content" else TSA.REVIEWED_CONTENT
    change(rec)
    assert any(match in b for b in TSA.problems(rec, pool_content_sha256=content, tasks=[_task()]))


def test_a_task_from_an_unadmitted_source_refuses():
    bad = TSA.problems(REC, pool_content_sha256=TSA.REVIEWED_CONTENT,
                       tasks=[_task("nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1", "0ac34ec8")])
    assert any("does not admit" in b for b in bad)


def test_the_pod_refuses_a_plan_whose_content_the_admission_does_not_admit(monkeypatch):
    monkeypatch.setattr(TGP, "verified_bound", lambda proto, li: ({"pool": {"content_sha256": "0" * 64}}, {}))
    with pytest.raises(TS.PlanError, match="formal source admission"):
        TS.checked_plan(PROTO, "thinking_2507")


# --- the grant ----------------------------------------------------------------------------------------------------

def test_the_t2507_cap_is_cumulative_and_its_block_required(tmp_path, monkeypatch):
    kind = "tool_generation_t2507"
    assert TM.kind_cap_problems(kind, 2.78) == [] and TM.kind_cap_problems(kind, 2.79)
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    assert TM.t2507_grant_problems(g, kind) == []
    assert TM.t2507_grant_problems({k: v for k, v in g.items() if k != kind}, kind)
    monkeypatch.setattr(TM, "kind_spent", lambda k, root: {"runs": [{"run_id": "x", "usd": 1.5}], "unknown": [], "usd": 1.5})
    assert any("cumulative" in b for b in TM.kind_cap_problems(kind, 1.5))
    assert TM.kind_cap_problems(kind, 1.28) == []
    assert TM.kind_cap_problems("tool_generation", 1.0)          #: the hybrid is not authorized

"""The L40S tool pilot's $0 execution preparation (maintainer review of 2468c1aa, item 5): its own kind and refusals,
the funding and provider checks, the L40S-only card, the runner's hard cap, the frozen cells and task file, the
session's one round with second candidates for refusals only, resumption without regeneration, generation
failure, and the telemetry. Nothing here creates a resource."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aadistill.data.tool_calls import canonical_call
from aadistill.governance.authorization import AuthorizationError
from shared import study_engineering as SE
from stages.teacher_modes import token_matched_phase as TM
from stages.teacher_modes import tool_episodes as TE
from stages.teacher_modes import tool_pilot as TPL
from stages.teacher_modes import tool_pool as TP
from stages.teacher_modes import tool_session as TS

THINK, OPEN, CLOSE, EOS = 151668, 151657, 151658, 151645
SPEC = {"lineage": "hybrid", "training_per_category": 2, "single_step_training": 2,
        "whole_cells": (("calibration", "schema_arguments"), ("state_eval", "schema_arguments")), "rounds": 1}


# --- the kind, its funding and its card ------------------------------------------------------------------------------

def test_the_pilot_kind_claims_generation_only_and_refuses_without_its_records_and_cap():
    assert "tool_pilot" in TM.STUDY.kinds and "tool_pilot" in TM.DRIVERS and "tool_pilot" in TM.KIND_RECORDS
    assert TM.STUDY.claims_of("tool_pilot") == frozenset({"teacher_generation"})       #: never formal or training
    assert TM.kind_cap_problems("tool_pilot", 1.0) == ["the grant names no hard cap for kind 'tool_pilot'"]
    assert any("PREPARED, not signed" in p for p in TM.pilot_problems())               #: the admission is unsigned
    assert TM.main(["issue", "--kind", "tool_pilot", "--run-id", "x", "--expected-usd", "1", "--hard-cap-usd",
                    "1.8", "--dry-run"]) == 11


def test_a_grant_cap_admits_the_pilot_only_up_to_it(tmp_path):
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    g["session_hard_caps_usd"]["tool_pilot"] = 1.80
    (tmp_path / TM.STUDY.grant).parent.mkdir(parents=True)
    (tmp_path / TM.STUDY.grant).write_text(json.dumps(g))
    assert TM.kind_cap_problems("tool_pilot", 1.80, root=tmp_path) == []
    assert TM.kind_cap_problems("tool_pilot", 1.81, root=tmp_path)


@pytest.mark.parametrize("change,match", [
    ({"project_remaining_usd": 1.0}, "does not fit"), ({"balance_usd": 1.0}, "account holds"),
    ({"project_cap_usd": 900.0}, "a new maintainer decision"), ("envelope", "the grant's envelope leaves")])
def test_funding_refuses_on_each_live_condition(change, match):
    g = {"per_session_hard_cap_usd": 41.41, "project_cumulative_cap_usd": 868.0}
    env = {"remaining_usd": 29.92}
    pos = {"project_cap_usd": 868.0, "project_remaining_usd": 30.05, "balance_usd": 92.84}
    assert SE.funding_problems(1.80, g, env, pos) == []
    if change == "envelope":
        env = {"remaining_usd": 1.0}
    else:
        pos = {**pos, **change}
    assert any(match in p for p in SE.funding_problems(1.80, g, env, pos))


def test_an_unreadable_provider_funds_nothing(monkeypatch):
    from stages.calibration_volume import mixed_governance as MG

    def down():
        raise TimeoutError("provider API timed out")
    monkeypatch.setattr(MG, "provider_snapshot", down)
    with pytest.raises(AuthorizationError, match="an unknown position funds nothing"):
        SE.provider_snapshot()


def test_the_pilot_runs_on_one_l40s_and_never_falls_back_to_an_a100():
    ctx = lambda gpu, n=1: SimpleNamespace(args=SimpleNamespace(gpu=gpu, gpu_count=n))  # noqa: E731
    assert TM.the_pilot_runs_on_an_l40s(ctx("NVIDIA L40S"))[0]
    ok, why = TM.the_pilot_runs_on_an_l40s(ctx("NVIDIA A100 80GB PCIe"))
    assert not ok and "measured L40S OOM and a separate authorization" in why
    assert not TM.the_pilot_runs_on_an_l40s(ctx("NVIDIA L40S", 2))[0]
    assert TM.the_pilot_runs_on_an_l40s in TM.launch_spec("tool_pilot", 27).extra_prechecks
    assert TM.the_pilot_runs_on_an_l40s not in TM.launch_spec("tool_generation", 27).extra_prechecks


def test_the_hard_cap_check_prices_setup_execution_teardown_and_disk_at_the_provider_rate():
    a = TPL.runner_cap(27.0, 1.19, cap=1.80)
    assert a["hard_terminate_minutes"] == round((20 + 5 + 27.0) * 1.25 + 12.5 + 10, 1) and a["fits_cap"]
    b = TPL.runner_cap(55.0, 1.19, cap=1.80)                                #: the plan's upper driver time
    assert not b["fits_cap"] and TPL.runner_cap(55.0, 1.19, cap=2.50)["fits_cap"]
    assert {k: v for k, v in TPL.CAP_OPTIONS.items()} == {"A": (1.80, 27.0), "B": (2.50, 55.0)}
    disk = TPL.DISK_GB * SE.DISK_USD_PER_GB_MONTH / (24 * 30.4375)
    assert abs(a["all_in_usd"] - (a["hard_terminate_minutes"] + SE.TERMINATION_MINUTES) / 60 * (1.19 + disk)) < 1e-3
    launch = TM.launch_spec("tool_pilot", 55.0)
    check = next(f for f in SE.prechecks(TM.STUDY, launch) if f.__name__ == "all_in_within_cap")
    sess = SE.session_spec(TM.STUDY, launch, SimpleNamespace(run_id="t", dry_run=False))
    plan = sess.budget.plan(price_per_hour=1.19, authorized_usd=10 ** 6)
    ctx = lambda cap: SimpleNamespace(plan=plan, args=SimpleNamespace(max_price=1.19, disk_gb=TPL.DISK_GB),  # noqa: E731
                                      auth=SimpleNamespace(hard_cap_usd=cap), evidence={})
    ok, why = check(ctx(1.80))
    assert not ok and "The cap is not raised and the run not shrunk" in why
    assert check(ctx(2.50))[0]


# --- the frozen cells ------------------------------------------------------------------------------------------------

def _server(split: str) -> str:
    k = 0
    while TP.split_of(f"server:{split[:2]}{k}") != split:
        k += 1
    return f"{split[:2]}{k}"


def _tool(srv: str) -> list[dict]:
    return [{"type": "function", "function": {"name": f"{srv}-book_{srv}", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "nights": {"type": "integer"}},
        "required": ["city", "nights"]}}}]


def pool_tasks() -> list[dict]:
    out = []
    for split, n in (("training", 6), ("calibration", 3), ("state_eval", 3)):
        srv = _server(split)
        for j in range(n):
            call = {"name": f"{srv}-book_{srv}", "arguments": {"city": f"c{j}", "nights": j + 1}}
            base = {"source": {"dataset": "toy/tools", "revision": "r1", "recorded": {"steps": 1}}, "family": None,
                    "split": None, "servers": [srv], "tools": _tool(srv), "later_user_turns": []}
            for cat in (("schema_arguments", "tool_selection") if split == "training" else ("schema_arguments",)):
                out.append({**base, "task_id": f"tc1-{split}-{cat}-{j}", "category": cat,
                            "messages": [{"role": "user", "content": f"{split} {cat} full {j}"}],
                            "backend": {"kind": "replay", "table": {canonical_call(call): "booked"}},
                            "gold_steps": [[[call]], None], "final_check": {"kind": "none"}, "max_turns": 2})
            out.append({**base, "task_id": f"tc1-{split}-single-{j}", "category": "schema_arguments",
                        "episode_shape": TE.SINGLE_STEP,
                        "messages": [{"role": "user", "content": f"{split} single {j}"}], "backend": {"kind": "none"},
                        "gold_steps": [[[call]]], "final_check": {"kind": "none"}, "max_turns": 1})
    return out


@pytest.fixture()
def frozen(tmp_path):
    pool = TP.assemble(pool_tasks())
    TP.write_pool(pool, tmp_path / "pool", sources={"toy": {"converted_sha256": "0" * 64, "tasks": 1}}, extra={})
    proto = json.loads(TS.PROTOCOL.read_text())
    proto["lineages"]["hybrid"]["splits"] = ["training", "calibration", "state_eval"]
    doc, tasks = TPL.build(tmp_path / "pool", protocol=proto, spec=SPEC)
    doc = TPL.write(doc, tasks, tmp_path / "pilot_plan.json", tmp_path / "pilot_tasks.jsonl")
    return {"tmp": tmp_path, "doc": doc, "plan": tmp_path / "pilot_plan.json", "tasks": tmp_path / "pilot_tasks.jsonl",
            "proto": proto}


def test_the_frozen_cells_are_the_plans_pilot_sample_by_exact_task_ids(frozen):
    doc, by_id = TPL.load(frozen["plan"], frozen["tasks"])
    cells = {f"{c['split']}/{c['category']}": c for c in doc["modes"]["H-N"]}
    tr = cells["training/schema_arguments"]["task_ids"]
    assert [TE.episode_shape(by_id[t]) for t in tr] == [TE.FULL_EPISODE] * 2 + [TE.SINGLE_STEP] * 2
    assert len(cells["state_eval/schema_arguments"]["task_ids"]) == 6                   #: every task, both shapes
    assert len(cells["calibration/schema_arguments"]["task_ids"]) == 6
    assert len(cells["training/tool_selection"]["task_ids"]) == 2
    cs, orders = TPL.cells(doc, "H-N", by_id)
    assert [[t["task_id"] for t in o] for o in orders] == [c["task_ids"] for c in doc["modes"]["H-N"]]
    assert all(c["bound"] == c["tasks"]["expected"] == len(o) for c, o in zip(cs, orders))


@pytest.mark.parametrize("tamper,match", [
    ("edit a task", "not the one the plan froze"), ("drop a task", "not the one the plan froze"),
    ("edit and re-seal", "differ from their frozen sha256"), ("another protocol", "another tool_generation_protocol"),
    ("another verifier", "this verifier is")])
def test_any_change_to_the_frozen_pilot_refuses(frozen, tamper, match, monkeypatch):
    lines = frozen["tasks"].read_text().splitlines()
    proto = TS.PROTOCOL
    if tamper == "edit a task":
        frozen["tasks"].write_text(lines[0].replace("booked", "cancelled") + "\n" + "\n".join(lines[1:]) + "\n")
    elif tamper == "drop a task":
        frozen["tasks"].write_text("\n".join(lines[1:]) + "\n")
    elif tamper == "edit and re-seal":
        t = json.loads(lines[0])
        t["messages"][0]["content"] += " changed"
        frozen["tasks"].write_text("\n".join([json.dumps(t, sort_keys=True)] + lines[1:]) + "\n")
        doc = json.loads(frozen["plan"].read_text())
        doc["tasks_file"]["sha256"] = TPL.sha(frozen["tasks"])
        frozen["plan"].write_text(json.dumps(doc))
    elif tamper == "another protocol":
        proto = frozen["tmp"] / "proto.json"
        proto.write_text(json.dumps(frozen["proto"]))
    else:
        monkeypatch.setattr(TE, "VERIFICATION_VERSION", "tool_verification@v1")
    with pytest.raises(TPL.PilotError, match=match):
        TPL.load(frozen["plan"], frozen["tasks"], protocol_path=proto)


# --- the session -----------------------------------------------------------------------------------------------------

def fails_first(tid: str) -> bool:
    return tid.endswith(("-0", "-2"))


def episode(task: dict, k: int, thinking: bool) -> dict:
    call = task["gold_steps"][0][0][0]
    bad = k == 0 and fails_first(task["task_id"])
    head = ("plan" if bad else "plan</think>\n") if thinking else ("x</think>\n" if bad else "")
    out = ([7] * 20 + [THINK] if thinking else []) + [OPEN] + [8] * 10 + [CLOSE, EOS]
    turn = {"raw": head + f"<tool_call>{json.dumps(call)}</tool_call><|im_end|>", "stop_reason": "eos",
            "prompt_token_ids": [1] * 40, "output_token_ids": out, "generated_tokens": len(out),
            "ttft_seconds": 0.2, "seconds": 1.0, "prompt_tokens": 40}
    if TE.episode_shape(task) == TE.SINGLE_STEP:
        return {"turns": [{**turn, "responses": []}]}
    answer = {"raw": ("done</think>\nBooked." if thinking else "Booked.") + "<|im_end|>", "stop_reason": "eos",
              "responses": [], "prompt_token_ids": [1] * 60,
              "output_token_ids": ([7] * 5 + [THINK] if thinking else []) + [9] * 6 + [EOS]}
    return {"turns": [{**turn, "responses": ["booked"]}, answer]}


def signed(frozen, monkeypatch, *, plan_sha=None):
    rec = {"schema": TPL.ADMISSION_SCHEMA, "scope": "the tool_pilot kind's measurement only (test)",
           "pilot_plan": {"sha256": plan_sha or TPL.sha(frozen["plan"])},
           "signature": {"signed_by": "maintainer (test)"}}
    (frozen["tmp"] / "admission.json").write_text(json.dumps(rec))
    monkeypatch.setattr(TPL, "ADMISSION", frozen["tmp"] / "admission.json")


def test_one_round_of_the_frozen_cells_second_candidates_only_for_refusals(frozen, monkeypatch):
    from stages.reasoning_data import generation_driver as GD
    signed(frozen, monkeypatch)
    _, by_id = TPL.load(frozen["plan"], frozen["tasks"])
    calls = []

    def run_group(c, out, inputs, f, k, ids, run, mark, *, label=None, extra_args=()):
        rows = [json.loads(x) for x in (inputs / f["file"]).read_text().splitlines()]
        want = [r["task_id"] for r in rows if ids is None or r["task_id"] in ids]
        thinking = json.loads(extra_args[extra_args.index("--chat-template-kwargs") + 1]).get("enable_thinking")
        calls.append((label, len(want), "--engine-metrics" in extra_args))
        return {"label": label, "complete": True, "segments": [],
                "records": {t: episode(by_id[t], k, thinking) for t in want}}
    monkeypatch.setattr(GD, "run_group", run_group)
    rec, out = {}, frozen["tmp"] / "out"
    out.mkdir()
    assert TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], out, "id", None, lambda m: None, rec)
    n = sum(len(c["task_ids"]) for c in frozen["doc"]["modes"]["H-N"])
    first = {lab: k for lab, k, _ in calls if lab.endswith("pass0")}
    second = {lab: k for lab, k, _ in calls if lab.endswith("pass1")}
    assert first == {"pilot_H-N_r1_pass0": n, "pilot_H-T_r1_pass0": n}                 #: one round, every task
    assert all(0 < v < n for v in second.values()) and len(second) == 2                 #: refused tasks only
    assert all(m for _, _, m in calls)                                                  #: engine metrics asked for
    tel = [json.loads(x) for x in (out / "telemetry.jsonl").read_text().splitlines()]
    assert [t["label"] for t in tel] == [lab for lab, _, _ in calls] and all("gpu" in t for t in tel)
    assert rec["pilot_plan_sha256"] == TPL.sha(frozen["plan"]) and rec["verification_version"] == TE.VERIFICATION_VERSION


@pytest.mark.parametrize("which", ["unsigned", "another plan"])
def test_the_session_refuses_without_the_signed_admission_of_this_plan(frozen, monkeypatch, which):
    if which == "unsigned":
        signed(frozen, monkeypatch)
        rec = json.loads(TPL.ADMISSION.read_text())
        rec["signature"]["signed_by"] = None
        TPL.ADMISSION.write_text(json.dumps(rec))
    else:
        signed(frozen, monkeypatch, plan_sha="0" * 64)
    with pytest.raises(TPL.PilotError, match="not signed|another pilot plan"):
        TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], frozen["tmp"], "id", None, lambda m: None, {})


def test_an_interrupted_generation_fails_then_resumes_without_regenerating(frozen, monkeypatch):
    """The real run_group and sealing: a generation that dies after three records leaves them sealed and the session
    RUN_FAILED; the next session (a new run) submits only the unsealed tasks and completes."""
    signed(frozen, monkeypatch)
    _, by_id = TPL.load(frozen["plan"], frozen["tasks"])
    submitted = []

    def runner(stop_after):
        def run(cmd, log, tick):
            tasks = [json.loads(x) for x in Path(cmd[cmd.index("--tasks") + 1]).read_text().splitlines()]
            stream = Path(cmd[cmd.index("--stream-completed") + 1])
            k = int(cmd[cmd.index("--seed-base") + 1]) - TS.protocol()["execution"]["seed_base"]
            thinking = json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]).get("enable_thinking")
            submitted.append((Path(cmd[cmd.index("--tasks") + 1]).name, len(tasks)))
            with stream.open("a") as fh:
                for i, t in enumerate(tasks):
                    if stop_after is not None and i >= stop_after:
                        return 1                                          #: the generator died mid-group
                    fh.write(json.dumps({"task_id": t["task_id"], **episode(by_id[t["task_id"]], k, thinking)}) + "\n")
            Path(log).write_text("[x] engine ready in 42.0s\n")
            return 0
        return run
    out = frozen["tmp"] / "out"
    out.mkdir()
    assert not TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], out, "id", runner(3), lambda m: None, {})
    assert submitted[-1][1] > 3
    total = submitted[-1][1]
    submitted.clear()
    assert TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], out, "id", runner(None), lambda m: None, {})
    assert submitted[0][1] == total - 3                                     #: the three sealed ones never rerun
    tel = [json.loads(x) for x in (out / "telemetry.jsonl").read_text().splitlines()]
    assert any(e["engine_ready_seconds"] == [42.0] for t in tel for e in t.get("engine") or [])


# --- telemetry -------------------------------------------------------------------------------------------------------

def test_gpu_and_engine_telemetry_are_read_and_never_raise(tmp_path):
    rows = iter(["0, NVIDIA L40S, 30000, 46068, 80\n", "0, NVIDIA L40S, 38500, 46068, 97\n"])
    g = TPL.GpuSampler(query=lambda: next(rows))
    g.sample()
    g.sample()
    assert g.as_dict()["gpus"]["0"]["peak_memory_used_mib"] == 38500 and g.samples == 2
    g.sample()                                                              #: an exhausted / failing query
    assert g.errors and g.samples == 2
    log = tmp_path / "x.seg0.log"
    log.write_text("[x] engine ready in 95.3s\nModel loading took 7.6 GiB and 31.2 seconds\n"
                   "GPU KV cache size: 254,880 tokens\nMaximum concurrency for 8,184 tokens per request: 31.14x\n"
                   "WARNING sequence group 3 is preempted by PreemptionMode.RECOMPUTE\n")
    (tmp_path / "x.seg0.raw.summary.json").write_text(json.dumps({
        "identity": {"engine": {"init_seconds": 95.3, "vllm_version": "0.x"}}, "output_tokens": 1000,
        "generation_seconds": 10.0, "output_tokens_per_second": 100.0,
        "engine_metrics": {"samples": 3, "max": {"vllm:kv_cache_usage_perc": 0.81}, "final": {}, "errors": []}}))
    t = TPL.engine_telemetry(log, tmp_path / "x.seg0.raw.summary.json")
    assert t["engine_ready_seconds"] == [95.3] and t["model_loading_gib_seconds"] == [7.6, 31.2]
    assert t["kv_cache_tokens"] == [254880.0] and t["max_concurrency"] == [8184.0, 31.14]
    assert t["preemption_lines"] == 1 and t["engine_metrics"]["max"]["vllm:kv_cache_usage_perc"] == 0.81
    assert TPL.engine_telemetry(tmp_path / "missing.log")["log_found"] is False


def test_the_runner_records_engine_metrics_only_when_asked():
    from stages.calibration_volume import run_battery as RB
    into = {"samples": 0, "max": {}, "final": {}, "errors": []}
    m = lambda n, v: SimpleNamespace(name=n, value=v)  # noqa: E731
    llm = SimpleNamespace(get_metrics=lambda: [m("vllm:num_preemptions", 2), m("vllm:kv_cache_usage_perc", 0.4),
                                               m("vllm:unrelated", 9)])
    assert RB.sample_engine_metrics(llm, into) == {"vllm:num_preemptions": 2, "vllm:kv_cache_usage_perc": 0.4}
    llm.get_metrics = lambda: [m("vllm:kv_cache_usage_perc", 0.2)]
    RB.sample_engine_metrics(llm, into)
    assert into["max"]["vllm:kv_cache_usage_perc"] == 0.4 and into["samples"] == 2
    RB.sample_engine_metrics(SimpleNamespace(), into)                          #: an engine without the API
    assert into["errors"] and into["samples"] == 2


def test_the_committed_pilot_records_hold_together():
    doc, by_id = TPL.load(TPL.PLAN, TPL.TASKS) if TPL.TASKS.is_file() else (None, None)
    if doc is None:
        pytest.skip("the pilot's task file is an artifact outside git")
    adm = json.loads(TPL.ADMISSION.read_text())
    assert adm["pilot_plan"]["sha256"] == TPL.sha(TPL.PLAN) and adm["tasks"] == len(by_id) == len(doc["tasks"])
    assert adm["signature"]["signed_by"] is None                             #: prepared, not signed
    assert sum(len(c["task_ids"]) for cs in doc["modes"].values() for c in cs) == 794

"""The L40S tool pilot's $0 execution preparation (maintainer review of 2468c1aa, item 5): its own kind and refusals,
the funding and provider checks, the L40S-only card, the runner's hard cap, the frozen cells and task file, the
session's one round with second candidates for refusals only, resumption without regeneration, generation
failure, and the telemetry. Nothing here creates a resource."""
from __future__ import annotations

import json
from collections import Counter
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


def _grant_root(tmp_path, *, block=True, **over):
    """A temporary root holding the study grant amended as `tool_pilot.grant_amendment` proposes (decision and date
    filled for the test), the bound pilot records copied beside it; ``over`` edits the tool_pilot block."""
    import shutil
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    amend = TPL.grant_amendment()
    g["session_hard_caps_usd"]["tool_pilot"] = amend["edits"]["session_hard_caps_usd.tool_pilot"]
    if block:
        g["tool_pilot"] = {**amend["edits"]["tool_pilot"], "decision": "approved for the test", "date": "2026-11-04",
                           **over}
    (tmp_path / TM.STUDY.grant).parent.mkdir(parents=True)
    (tmp_path / TM.STUDY.grant).write_text(json.dumps(g))
    for rel in TM.KIND_RECORDS["tool_pilot"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(TM.REPO_ROOT / rel, tmp_path / rel)
    return tmp_path


def test_the_grant_must_carry_the_pilot_only_scope_not_just_a_cap(tmp_path):
    assert TM.kind_cap_problems("tool_pilot", 2.50, root=_grant_root(tmp_path / "a")) == []
    assert TM.kind_cap_problems("tool_pilot", 2.51, root=_grant_root(tmp_path / "b"))
    bad = TM.kind_cap_problems("tool_pilot", 2.50, root=_grant_root(tmp_path / "c", block=False))
    assert bad == ["the grant carries no tool_pilot scope: the maintainer's pilot-only amendment is required"]


@pytest.mark.parametrize("over,match", [
    ({"approves": "tool generation"}, "tool_pilot.approves"), ({"granted_by": "claude"}, "tool_pilot.granted_by"),
    ({"decision": None}, "quotes no decision"), ({"date": "soon"}, "not an ISO date"),
    ({"pilot_plan_sha256": "0" * 64}, "pilot_plan_sha256"), ({"tasks_sha256": "0" * 64}, "tasks_sha256"),
    ({"hard_cap_usd": 5.0}, "hard_cap_usd"), ({"gpu": "NVIDIA A100 80GB PCIe"}, "tool_pilot.gpu"),
    ({"network_volume": "4rhubprswo"}, "network_volume"), ({"ceiling_usd_at_amendment": 99.0}, "ceiling_usd_at_amendment")])
def test_a_pilot_grant_block_that_is_not_exactly_the_pilots_refuses(tmp_path, over, match):
    assert any(match in p for p in TM.kind_cap_problems("tool_pilot", 2.50, root=_grant_root(tmp_path, **over)))


def test_the_envelope_covers_the_pilot_and_the_amendment_raises_nothing():
    env = TPL.envelope_reconciliation()
    assert env["envelope"]["unknown_runs"] == [] and env["envelope"]["covers_the_pilot"]
    assert abs(env["envelope"]["spent_usd"] + env["envelope"]["remaining_usd"] - env["envelope"]["ceiling_usd"]) < 1e-6
    rc = env["research_a_cumulative"]
    assert abs(rc["cross_check"]["cap_less_the_first_ceiling_plus_the_supplement_usd"] - rc["before_the_a1_phase_usd"]) < 0.01
    assert rc["headroom_after_usd"] > 0
    a = TPL.grant_amendment()
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    assert a["unchanged"]["ceiling_usd"] == g["ceiling_usd"] and a["unchanged"]["project_cumulative_cap_usd"] == 868.0
    assert "tool_pilot" not in g and "tool_pilot" not in g["session_hard_caps_usd"]          #: the grant is untouched
    assert a["edits"]["tool_pilot"]["decision"] is None and a["edits"]["tool_pilot"]["date"] is None


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


def sign(rec: dict, **over) -> dict:
    """A test's stand-in for the maintainer's signature on a temporary copy (the real record stays unsigned)."""
    import datetime as dt
    rec = json.loads(json.dumps(rec))
    rec["signature"] = {"signed_by": TPL.SIGNER, "approves": TPL.APPROVAL, "decision": "approved for the test",
                        "date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
                        "pilot_plan_sha256": rec["pilot_plan"]["sha256"], "tasks_sha256": rec["tasks_sha256"],
                        "tasks": rec["tasks"], **over}
    return rec


def signed(frozen, monkeypatch, **over):
    rec = sign(TPL.source_admission(frozen["doc"], frozen["plan"]), **over)
    (frozen["tmp"] / "admission.json").write_text(json.dumps(rec))
    monkeypatch.setattr(TPL, "ADMISSION", frozen["tmp"] / "admission.json")
    return rec


def pilot(frozen, out, run=None, rec=None, volume_gb=55):
    return TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], out, "id", run, lambda m: None,
                        {} if rec is None else rec, mount=frozen["tmp"], volume_gb=volume_gb)


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
    assert pilot(frozen, out, rec=rec)
    assert rec["storage"]["free_bytes"] > TPL.MIN_FREE_BYTES
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
    rec = signed(frozen, monkeypatch)
    if which == "unsigned":
        rec["signature"] = dict.fromkeys(TPL.SIGNATURE_FIELDS)
    else:
        rec["pilot_plan"]["sha256"] = rec["signature"]["pilot_plan_sha256"] = "0" * 64
    TPL.ADMISSION.write_text(json.dumps(rec))
    with pytest.raises(TPL.PilotError, match="not signed|another pilot plan"):
        pilot(frozen, frozen["tmp"] / "out")


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
    assert not pilot(frozen, out, runner(3))
    assert submitted[-1][1] > 3
    total = submitted[-1][1]
    submitted.clear()
    assert pilot(frozen, out, runner(None))
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
    assert set(adm["signature"]) == set(TPL.SIGNATURE_FIELDS) and not any(adm["signature"].values())   #: unsigned
    assert TPL.admission_problems(adm, TPL.PLAN, TPL.TASKS) == ["PREPARED, not signed by the maintainer"]
    assert sum(len(c["task_ids"]) for cs in doc["modes"].values() for c in cs) == 794


# --- the pilot-only admission's signature (one check, issue and pod) -------------------------------------------------

def _bad(rec, path, value):
    rec = json.loads(json.dumps(rec))
    *head, last = path
    tgt = rec
    for k in head:
        tgt = tgt[k]
    tgt[last] = value
    return rec


@pytest.mark.parametrize("path,value,match", [
    (("signature", "approves"), "tool_generation and training", "not exactly 'tool_pilot measurement only'"),
    (("signature", "decision"), None, "carries no decision"), (("signature", "decision"), "  ", "carries no decision"),
    (("signature", "date"), None, "not an ISO date"), (("signature", "date"), "2099-01-01", "outside"),
    (("signature", "signed_by"), "claude", "not the maintainer"),
    (("signature", "tasks_sha256"), "0" * 64, "signature's tasks_sha256"),
    (("tasks_sha256",), "0" * 64, "another task file"), (("pilot_plan", "sha256"), "0" * 64, "another pilot plan"),
    (("tasks",), 1, "task count"), (("scope",), "the bounded tool generation and formal training", "scope is not"),
    (("does_not",), [], "scope is not")])
def test_a_signature_that_is_not_exactly_the_pilots_refuses(frozen, path, value, match):
    good = sign(TPL.source_admission(frozen["doc"], frozen["plan"]))
    assert TPL.admission_problems(good, frozen["plan"], frozen["tasks"]) == []
    assert any(match in p for p in TPL.admission_problems(_bad(good, path, value), frozen["plan"], frozen["tasks"]))


def test_signed_by_alone_is_not_a_signature(frozen):
    rec = TPL.source_admission(frozen["doc"], frozen["plan"])
    rec["signature"]["signed_by"] = "maintainer"
    bad = TPL.admission_problems(rec, frozen["plan"], frozen["tasks"])
    assert any("approves None" in p for p in bad) and any("no decision" in p for p in bad)


def test_another_task_file_or_source_refuses(frozen):
    good = sign(TPL.source_admission(frozen["doc"], frozen["plan"]))
    other = frozen["tmp"] / "other.jsonl"
    other.write_text(frozen["tasks"].read_text() + "\n")
    assert any("another task file" in p for p in TPL.admission_problems(good, frozen["plan"], other))
    good["sources"][0]["tasks"] += 1
    assert any("sources are not" in p for p in TPL.admission_problems(good, frozen["plan"], frozen["tasks"]))


def test_issue_and_the_pod_apply_the_same_admission_check(frozen, tmp_path, monkeypatch):
    """`token_matched_phase.pilot_problems` (issue) reads the bound records under a root and refuses on exactly what
    `tool_session.run_pilot` (the pod) refuses on: both call `tool_pilot.admission_problems`."""
    import shutil
    rec = TM.KIND_RECORDS["tool_pilot"]
    root = tmp_path / "root"
    for rel in rec[:2]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(TM.REPO_ROOT / rel, root / rel)
    doc = json.loads(frozen["plan"].read_text())
    doc["tasks_file"]["path"] = "artifacts/pilot_tasks.jsonl"
    (root / "artifacts").mkdir()
    shutil.copy(frozen["tasks"], root / doc["tasks_file"]["path"])
    (root / rec[2]).parent.mkdir(parents=True, exist_ok=True)
    (root / rec[2]).write_text(json.dumps(doc))
    seen = []
    real = TPL.admission_problems
    monkeypatch.setattr(TPL, "admission_problems", lambda *a, **k: (seen.append("called"), real(*a, **k))[1])
    adm = sign(TPL.source_admission(doc, root / rec[2], root / doc["tasks_file"]["path"]))
    (root / rec[3]).parent.mkdir(parents=True, exist_ok=True)
    (root / rec[3]).write_text(json.dumps(adm))
    assert TM.pilot_problems(root) == [] and seen
    (root / rec[3]).write_text(json.dumps(_bad(adm, ("signature", "approves"), "anything else")))
    assert any("source admission: the signature approves" in p for p in TM.pilot_problems(root))


# --- the volume ------------------------------------------------------------------------------------------------------

def test_the_pod_refuses_a_full_volume_or_a_directory_it_did_not_write(frozen, monkeypatch):
    signed(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    with pytest.raises(TPL.PilotError, match="below the 2 GB the pilot requires"):
        pilot(frozen, out, volume_gb=1)
    (out / "checkpoint").mkdir()                                    #: someone else's work in the pilot's directory
    with pytest.raises(TPL.PilotError, match="files the pilot did not write"):
        pilot(frozen, out)
    with pytest.raises(TPL.PilotError, match="provider size is unknown"):
        pilot(frozen, frozen["tmp"] / "out2", volume_gb=None)


def test_the_pilot_volume_is_am5m373kuq_and_never_4rhubprswo():
    args = lambda **o: SimpleNamespace(args=SimpleNamespace(**{"network_volume_id": "am5m373kuq", "volume_gb": 55,  # noqa: E731
                                                               "data_center_ids": "EUR-IS-2",
                                                               "volume_mount_path": "/runpod-volume", **o}))
    assert TM.the_pilot_volume_is_the_planned_one(args())[0]
    for bad in ({"network_volume_id": "4rhubprswo", "volume_gb": 350, "data_center_ids": "US-TX-3"},
                {"volume_gb": 30}, {"data_center_ids": "US-TX-3"}):
        assert not TM.the_pilot_volume_is_the_planned_one(args(**bad))[0]
    assert TM.the_pilot_volume_is_the_planned_one in TM.launch_spec("tool_pilot", 55).extra_prechecks


# --- the report on partial, resumed and forcibly ended sessions --------------------------------------------------------

def _interrupt_then_resume(frozen, monkeypatch, stop_after=3):
    signed(frozen, monkeypatch)
    _, by_id = TPL.load(frozen["plan"], frozen["tasks"])

    def runner(stop):
        def run(cmd, log, tick):
            tasks = [json.loads(x) for x in Path(cmd[cmd.index("--tasks") + 1]).read_text().splitlines()]
            stream = Path(cmd[cmd.index("--stream-completed") + 1])
            k = int(cmd[cmd.index("--seed-base") + 1]) - TS.protocol()["execution"]["seed_base"]
            thinking = json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]).get("enable_thinking")
            with stream.open("a") as fh:
                for i, t in enumerate(tasks):
                    if stop is not None and i >= stop:
                        return 1
                    fh.write(json.dumps({"task_id": t["task_id"], **episode(by_id[t["task_id"]], k, thinking)}) + "\n")
            return 0
        return run
    return runner


def test_a_partial_session_is_reported_from_its_sealed_shards(frozen, monkeypatch):
    """A generation killed after three records: the real sealer left them in .part shards with .done markers and no
    .complete; the report reads them, calls the run PARTIAL, and never counts an unattempted task as a failure."""
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    rec = {}
    assert not pilot(frozen, out, runner(3), rec=rec)
    rec.update(terminal="RUN_FAILED")
    (out / "tool_session.json").write_text(json.dumps(rec))
    assert not list(out.glob("*.complete")) and list(out.glob("pilot_H-N_r1_pass0.part*.done"))
    r = TPL.report(out, plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    assert r["state"] == "PARTIAL"
    hn = r["modes"]["H-N"]
    c0 = Counter()
    for row in hn["cells"].values():
        c0.update(row["c0"])
        assert row["final"].get("rejected", 0) <= row["c0"].get("rejected", 0)
    n = sum(len(c["task_ids"]) for c in frozen["doc"]["modes"]["H-N"])
    assert c0["admitted"] + c0["rejected"] == 3 == hn["sealed_records"]                #: exactly what was sealed
    assert c0["attempted_unsealed"] == n - 3 and c0.get("unattempted", 0) == 0       #: submitted, never sealed
    assert r["modes"]["H-T"]["sealed_records"] == 0                                   #: the session stopped at H-N
    ht = Counter()
    for row in r["modes"]["H-T"]["cells"].values():
        ht.update(row["c0"])
        assert row["first_candidate_pass_rate"] == TPL.UNMEASURED and "rejected" not in row["final"]
    assert ht["unattempted"] == n
    assert r["telemetry"]["peak_vram_mib"]["value"] == TPL.UNMEASURED                #: no nvidia-smi here: not 0


def test_without_session_evidence_the_identity_is_required_and_the_gap_recorded(frozen, monkeypatch):
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    pilot(frozen, out, runner(3))
    with pytest.raises(TPL.PilotError, match="identity is unknown"):
        TPL.report(out, plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    r = TPL.report(out, study_identity="id", plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    assert r["state"] == "PARTIAL" and str(r["session_evidence"][str(out)]).startswith("MISSING")
    from stages.reasoning_data import generation_driver as GD
    with pytest.raises(GD.ShardError, match="another launch identity"):
        TPL.report(out, study_identity="another", plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    (out / "tool_session.json").write_text(json.dumps({"identity": TPL.pilot_identity("id", frozen["plan"])["sha256"],
                                                       "terminal": "RUN_FAILED"}))
    with pytest.raises(TPL.PilotError, match="identity is ambiguous"):          #: the caller's and the session's differ
        TPL.report(out, study_identity="another", plan_path=frozen["plan"], tasks_path=frozen["tasks"])


def test_a_resumed_session_completes_and_merges_without_changing_what_was_sealed(frozen, monkeypatch):
    import shutil
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    pilot(frozen, out, runner(3))
    copy1 = frozen["tmp"] / "secured_run1"
    shutil.copytree(out, copy1)                                       #: the first run's secured evidence
    before = {p.name: p.read_bytes() for p in out.glob("*.part*.jsonl")}
    rec = {}
    assert pilot(frozen, out, runner(None), rec=rec)
    assert all(out.joinpath(k).read_bytes() == v for k, v in before.items())   #: sealed shards untouched
    rec.update(terminal="ALL_DONE")
    (out / "tool_session.json").write_text(json.dumps(rec))
    r = TPL.report([copy1, out], plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    assert r["state"] == "COMPLETE"
    for mode, m in r["modes"].items():
        n = sum(len(c["task_ids"]) for c in frozen["doc"]["modes"][mode])
        tot = Counter()
        for row in m["cells"].values():
            tot.update(row["final"])
            assert row["c1"].get("required", 0) == row["c0"].get("rejected", 0)
            assert row["c1"].get("required", 0) == sum(row["c1"].get(k, 0) for k in ("admitted", "rejected"))
        assert tot["admitted"] + tot["rejected"] == n and not tot["undetermined"]       #: each task once
    (out / "tool_session.json").unlink()
    r2 = TPL.report([copy1, out], study_identity="id", plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    assert r2["state"] == "INCOMPLETE" and r2["modes"] == r["modes"]                  #: data complete, evidence not


def test_a_task_sealed_differently_in_two_copies_or_nothing_sealed(frozen, monkeypatch):
    import shutil
    runner = _interrupt_then_resume(frozen, monkeypatch)
    empty = frozen["tmp"] / "empty"
    empty.mkdir()
    (empty / "tool_session.json").write_text(json.dumps({"identity": TPL.pilot_identity("id", frozen["plan"])["sha256"],
                                                         "terminal": "RUN_FAILED"}))
    assert TPL.report(empty, plan_path=frozen["plan"], tasks_path=frozen["tasks"])["state"] == "FAILED"
    out = frozen["tmp"] / "out"
    out.mkdir()
    pilot(frozen, out, runner(3))
    other = frozen["tmp"] / "other"
    shutil.copytree(out, other)
    part = sorted(other.glob("pilot_H-N_r1_pass0.part*.jsonl"))[0]
    rows = [json.loads(x) for x in part.read_text().splitlines()]
    rows[0]["turns"][0]["raw"] += " "
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    part.with_suffix(".done").write_text(TPL.sha(part) + "\n")                #: a re-sealed, different record
    with pytest.raises(TPL.PilotError, match="sealed differently in two session copies"):
        TPL.report([out, other], study_identity="id", plan_path=frozen["plan"], tasks_path=frozen["tasks"])



# --- resumption identity -------------------------------------------------------------------------------------------

def test_an_earlier_plan_protocol_or_implementation_never_resumes(frozen, monkeypatch, tmp_path):
    """The same study identity and the same task ids, but sealed under another pilot plan, another protocol or another
    generation implementation: the resumed session refuses those shards instead of reusing them."""
    import shutil
    from stages.reasoning_data import generation_driver as GD
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    pilot(frozen, out, runner(3))                                     #: three episodes sealed under plan A
    sealed = sorted(out.glob("pilot_H-N_r1_pass0.part*.jsonl"))
    assert sealed
    for case in ("plan", "protocol", "implementation"):
        work = tmp_path / case
        shutil.copytree(out, work / "out")
        f = dict(frozen)
        if case == "plan":                                            #: plan B: same tasks, another frozen document
            doc = json.loads(frozen["plan"].read_text())
            doc["written_utc"] = "2026-11-05T00:00:00Z"
            f["plan"] = work / "plan_b.json"
            f["plan"].write_text(json.dumps(doc))
            signed(f, monkeypatch)
        elif case == "protocol":                                      #: another tool generation protocol file
            other = work / "tool_generation_protocol_b.json"
            proto = json.loads(TS.PROTOCOL.read_text())
            proto["status"] += " (another revision)"
            other.write_text(json.dumps(proto))
            from stages.teacher_modes import tool_generation_plan as TGP
            monkeypatch.setattr(TGP, "PROTOCOL", other)
        else:
            changed = work / "run_battery_b.py"
            changed.write_text("# another runner\n")
            monkeypatch.setattr(TPL, "IMPLEMENTATION", (str(changed),))
        with pytest.raises(GD.ShardError, match="another launch identity"):
            TS.run_pilot(f["proto"], f["plan"], f["tasks"], work / "out", "id", runner(None), lambda m: None, {},
                         mount=f["tmp"], volume_gb=55)
        monkeypatch.undo()
        signed(frozen, monkeypatch)
    assert pilot(frozen, out, runner(None))                           #: the same frozen plan resumes normally


def test_a_sealed_episode_from_another_round_input_refuses(frozen, monkeypatch):
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    pilot(frozen, out, runner(3))
    part = sorted(out.glob("pilot_H-N_r1_pass0.part*.jsonl"))[0]
    rows = [json.loads(x) for x in part.read_text().splitlines()]
    rows[0]["generation_pass"]["group_input_sha256"] = "0" * 64      #: sealed from another input, re-marked
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    part.with_suffix(".done").write_text(TPL.sha(part) + "\n")
    with pytest.raises(TPL.PilotError, match="sealed from another input, protocol or runner"):
        pilot(frozen, out, runner(None))

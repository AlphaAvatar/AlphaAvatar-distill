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

def test_the_pilot_kind_claims_generation_only_and_its_committed_issue_path_passes():
    """The maintainer's 2026-11-04 authorization applied: the grant names the pilot's cap and scope, the admission is
    signed, and the WHOLE issue path's conditions pass on the committed tree -- asserted on what it returns, not on an
    exit code; a cap above the grant's still refuses, for that reason."""
    assert "tool_pilot" in TM.STUDY.kinds and "tool_pilot" in TM.DRIVERS and "tool_pilot" in TM.KIND_RECORDS
    assert TM.STUDY.claims_of("tool_pilot") == frozenset({"teacher_generation"})       #: never formal or training
    if not TPL.TASKS.is_file():
        pytest.skip("the pilot's task file is an artifact outside git")
    left = round(2.50 - TM.kind_spent("tool_pilot")["usd"], 4)          #: the cap is cumulative over attempts
    assert TM.kind_cap_problems("tool_pilot", left) == []
    if left < 2.50:
        assert any("cumulative $2.5" in p for p in TM.kind_cap_problems("tool_pilot", 2.50))
    assert TM.kind_cap_problems("tool_pilot", 2.51) == ["$2.51 exceeds the grant's $2.5 for kind 'tool_pilot'"]
    got = TM.kind_conditions("tool_pilot")
    from stages.teacher_modes import l40s_check as LC
    assert got["l40s_generation_scope_sha256"] == TPL.sha(TM.REPO_ROOT / LC.GENERATION_SCOPE)
    assert {f"record:{r}" for r in TM.KIND_RECORDS["tool_pilot"]} <= set(got)
    assert got["record:" + TM.KIND_RECORDS["tool_pilot"][2]] == "0a1c831fdfe42361d1229e39a474281cd073ce15f29b9075c6f44963ad6d5b83"
    assert TM.main(["issue", "--kind", "tool_pilot", "--run-id", "x", "--expected-usd", "1", "--hard-cap-usd",
                    "2.51", "--dry-run"]) == 11


def _grant_root(tmp_path, *, block=True, **over):
    """A temporary root holding the study grant amended as `tool_pilot.grant_amendment` proposes (decision and date
    filled for the test), the bound pilot records copied beside it; ``over`` edits the tool_pilot block."""
    import shutil
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    g.pop("tool_pilot", None)                   #: the committed grant carries the maintainer's block since 2026-11-04
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
    ({"network_volume": "4rhubprswo"}, "network_volume"), ({"network_volume": "am5m373kuq"}, "network_volume"),
    ({"cap_is_cumulative": False}, "cap_is_cumulative"), ({"placement": {}}, "tool_pilot.placement"),
    ({"ceiling_usd_at_amendment": 99.0}, "ceiling_usd_at_amendment")])
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
    #: the maintainer's amendment (2026-11-04) applied as proposed, with the 2026-11-04 (16Z) placement amendment
    #: (PLACEMENT_AMENDMENT: no network volume, the kind cap cumulative): the pilot's cap and block, nothing else
    assert g["session_hard_caps_usd"]["tool_pilot"] == a["edits"]["session_hard_caps_usd.tool_pilot"] == 2.5
    assert {k: v for k, v in g["tool_pilot"].items() if k not in ("decision", "date", "amended")} == {
        k: v for k, v in a["edits"]["tool_pilot"].items() if k not in ("decision", "date")}
    assert g["tool_pilot"]["decision"] and g["tool_pilot"]["date"] == "2026-11-04"
    assert g["tool_pilot"]["amended"]["date"] == "2026-11-04" and g["tool_pilot"]["network_volume"] is None
    assert next(x for x in g["amendments"] if "previous_tool_pilot" in x)["previous_tool_pilot"] == {
        "network_volume": "am5m373kuq"}
    assert (g["per_session_hard_cap_usd"], g["gpus"], g["gpu_count_max"]) == (
        a["unchanged"]["per_session_hard_cap_usd"], a["unchanged"]["gpus"], a["unchanged"]["gpu_count_max"])
    applied = json.loads(TPL.AMENDMENT.read_text())                  #: the proposal the maintainer approved
    later = {"tool_pilot", "tool_generation_t2507", "tool_generation"}   #: kinds later amendments added
    assert {k: v for k, v in g["session_hard_caps_usd"].items() if k not in later} == \
        applied["unchanged"]["session_hard_caps_usd (the A1 kinds)"]
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
    assert TM.the_pilot_runs_on_an_l40s in TM.launch_spec("tool_generation", 27).extra_prechecks   #: L40S-only too
    assert TM.the_pilot_runs_on_an_l40s not in TM.launch_spec("search", 27).extra_prechecks


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
    doc, by_id = TPL.load(TPL.PLAN, TPL.TASKS, protocol_path=TPL.frozen_protocol()) if TPL.TASKS.is_file() else (None, None)
    if doc is None:
        pytest.skip("the pilot's task file is an artifact outside git")
    adm = json.loads(TPL.ADMISSION.read_text())
    assert adm["pilot_plan"]["sha256"] == TPL.sha(TPL.PLAN) and adm["tasks"] == len(by_id) == len(doc["tasks"])
    #: signed on the maintainer's written decision of 2026-11-04, every field, nothing broader than the pilot
    assert set(adm["signature"]) == set(TPL.SIGNATURE_FIELDS) and all(adm["signature"].values())
    assert adm["signature"]["approves"] == TPL.APPROVAL and adm["signature"]["date"] == "2026-11-04"
    assert TPL.admission_problems(adm, TPL.PLAN, TPL.TASKS) == []
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


# --- storage and placement (maintainer 2026-11-04: any Secure Cloud L40S, no network volume) ---------------------

def test_the_pod_refuses_a_full_disk_or_a_directory_it_did_not_write(frozen, monkeypatch):
    signed(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    with pytest.raises(TPL.PilotError, match="below the 2 GB the pilot requires"):
        pilot(frozen, out, volume_gb=1)
    bad, got = TPL.storage_problems(out, frozen["tmp"], None, free=lambda p: 10 ** 9)     #: the container disk
    assert bad == ["1.00 GB free on the disk, below the 2 GB the pilot requires"]
    bad, got = TPL.storage_problems(out, frozen["tmp"], None, free=lambda p: 10 ** 11)
    assert bad == [] and got["storage"] == "container_disk" and got["used_bytes"] is None
    (out / "checkpoint").mkdir()                                    #: someone else's work in the pilot's directory
    with pytest.raises(TPL.PilotError, match="files the pilot did not write"):
        pilot(frozen, out)


def _placement_ctx(**o):
    return SimpleNamespace(evidence={}, args=SimpleNamespace(**{"network_volume_id": "", "volume_gb": None,
                                                                "data_center_ids": "", "volume_mount_path": "",
                                                                "resume_from": None, **o}))


def test_the_pilot_attaches_no_volume_and_may_be_drawn_in_any_datacenter(tmp_path):
    ok, why = TM.the_pilot_placement_is_the_planned_one(_placement_ctx())
    assert ok and "any Secure Cloud datacenter" in why
    ok, why = TM.the_pilot_placement_is_the_planned_one(_placement_ctx(data_center_ids="EU-NL-1,US-TX-3;US-CA-2"))
    assert ok and "EU-NL-1,US-TX-3 then US-CA-2" in why
    for bad in ({"network_volume_id": "4rhubprswo", "volume_gb": 350, "volume_mount_path": "/runpod-volume"},
                {"network_volume_id": "am5m373kuq", "volume_gb": 55, "volume_mount_path": "/runpod-volume"},
                {"volume_gb": 55}, {"volume_mount_path": "/runpod-volume"}):
        assert not TM.the_pilot_placement_is_the_planned_one(_placement_ctx(**bad))[0]
    spec = TM.launch_spec("tool_pilot", 55)
    assert TM.the_pilot_placement_is_the_planned_one in spec.extra_prechecks
    from stages.teacher_modes import phase as P1
    assert P1.the_volume_is_the_planned_one not in spec.extra_prechecks       #: the pilot reads nothing a volume holds
    assert P1.the_volume_is_the_planned_one in TM.launch_spec("inputs", 55).extra_prechecks       #: a volume kind
    assert (spec.on_poll, spec.products_secured) == (TM.secure_pilot, TM.pilot_secured)
    a = SimpleNamespace(volume_mount_path="", volume_gb=None, run_id="r", resume_from=None)
    assert TM.durable(a) == "/workspace/teacher_modes"
    assert TM.durable(SimpleNamespace(volume_mount_path="/runpod-volume")) == "/runpod-volume/teacher_modes"
    cmd = TM.tool_pilot_command(SimpleNamespace(args=a), None)
    assert "--volume-gb" not in cmd and "--resume-from" not in cmd and "--durable-root /workspace/teacher_modes" in cmd
    a.resume_from = str(tmp_path / "secured_a")
    assert f"--resume-from {TM.SE.POD_REPO}/{TM.PILOT_RESUME_DIR}/secured_a" in TM.tool_pilot_command(
        SimpleNamespace(args=a), None)
    assert TM.SHARDS["tool_pilot"][0] == TPL.PILOT_DIR and TM.SHARDS["tool_pilot"][1].match("pilot_H-N_r1_pass0.part0000.done")


def _pilot_run(root, run_id, usd, kind="tool_pilot"):
    d = root / TM.STUDY.run_dir(run_id)
    (d / "governance").mkdir(parents=True)
    (d / "governance/authorization.json").write_text(json.dumps({"plan_id": TM.STUDY.plan_id,
                                                                 "stage_conditions": {"kind": kind}}))
    (d / "closeout").mkdir()
    (d / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": usd},
                                                         "provider_resource_created": usd is None or usd > 0}))


def test_the_pilot_cap_is_cumulative_over_every_attempt(tmp_path):
    root = _grant_root(tmp_path / "g")
    assert TM.kind_cap_problems("tool_pilot", 2.50, root=root) == []
    _pilot_run(root, "a_tool_pilot_1", 0.0)                         #: a $0 no-stock abort spends nothing
    _pilot_run(root, "a_tool_pilot_2", 1.10)
    _pilot_run(root, "a_inputs_x", 2.00, kind="inputs")             #: another kind's spend is not the pilot's
    assert TM.kind_spent("tool_pilot", root)["usd"] == 1.10
    assert TM.kind_cap_problems("tool_pilot", 1.40, root=root) == []
    bad = TM.kind_cap_problems("tool_pilot", 1.50, root=root)
    assert len(bad) == 1 and "cumulative $2.5" in bad[0] and "at most $1.4 is left" in bad[0]
    _pilot_run(root, "a_tool_pilot_3", None)                        #: created a resource, no closeout cost
    assert any("reconcile them first" in b for b in TM.kind_cap_problems("tool_pilot", 0.10, root=root))


def test_a_replacement_pod_resumes_from_the_dev_box_copy_without_regenerating(frozen, monkeypatch):
    """The pilot on the container disk: pod A dies after three sealed episodes; the dev box holds what the poll pulled
    (here, a copy of the sealed shards and markers only); pod B starts EMPTY, receives that copy (--resume-from) and
    submits only the unsealed tasks. A copy holding anything else, or a shard that disagrees with the pod's own,
    refuses."""
    import shutil
    runner = _interrupt_then_resume(frozen, monkeypatch)
    pod_a = frozen["tmp"] / "pod_a"
    pod_a.mkdir()
    assert not pilot(frozen, pod_a, runner(3))
    secured = frozen["tmp"] / "secured" / TPL.PILOT_DIR
    secured.mkdir(parents=True)
    for f in pod_a.glob("pilot_*.part*.*"):
        shutil.copy2(f, secured / f.name)
    bad, _ = TPL.resume_problems(secured)
    assert bad and all("no receipt" in b for b in bad)                  #: copied, but never confirmed received
    for d in secured.glob("*.done"):                                     #: what token_matched_phase.secure records
        TM.receipt(secured, d.with_suffix(".jsonl").name, d.read_text().strip())
    assert TPL.resume_problems(secured)[0] == [] and TPL.resume_problems(secured)[1]["records"] == 3
    submitted = []

    def counting(stop):
        inner = runner(stop)

        def run(cmd, log, tick):
            submitted.append(len(Path(cmd[cmd.index("--tasks") + 1]).read_text().splitlines()))
            return inner(cmd, log, tick)
        return run
    pod_b, rec = frozen["tmp"] / "pod_b", {}
    pod_b.mkdir()
    assert TS.run_pilot(frozen["proto"], frozen["plan"], frozen["tasks"], pod_b, "id", counting(None), lambda m: None,
                        rec, mount=frozen["tmp"], volume_gb=55, resume_from=secured)
    total = sum(len(c["task_ids"]) for c in frozen["doc"]["modes"]["H-N"])
    assert submitted[0] == total - 3 and rec["resumed_from"]["copied"] == 2 * len(list(secured.glob("*.done")))
    (secured / "notes.txt").write_text("x")
    with pytest.raises(TPL.PilotError, match="did not write"):
        TS.bring_sealed(secured, frozen["tmp"] / "pod_c")
    (secured / "notes.txt").unlink()
    part = sorted(secured.glob("*.part*.jsonl"))[0]
    part.write_text(part.read_text() + "\n")
    with pytest.raises(TPL.PilotError, match="does not hash to its marker"):
        TS.bring_sealed(secured, frozen["tmp"] / "pod_d")
    part.with_suffix(".done").write_text(TPL.sha(part) + "\n")       #: re-marked: its receipt still names the old bytes
    with pytest.raises(TPL.PilotError, match="no receipt of its sha256"):
        TS.bring_sealed(secured, frozen["tmp"] / "pod_e")
    (secured / TM.RECEIPTS).unlink()                                     #: even freshly receipted, not what pod B holds
    for d in secured.glob("*.done"):
        TM.receipt(secured, d.with_suffix(".jsonl").name, d.read_text().strip())
    with pytest.raises(TPL.PilotError, match="differs from the shard already"):
        TS.bring_sealed(secured, pod_b)


def test_the_poll_secures_every_sealed_pilot_shard(tmp_path, monkeypatch):
    """`secure` on the tool pilot's kind: each marked shard pulled by its sha256 into the store's pilot directory, the
    session record and the completion marks too; a shard already secured is not pulled again."""
    from stages.teacher_modes import phase as P1
    pulled = []
    store = tmp_path / "store"
    listing = ["pilot_H-N_r1_pass0.part0000.done " + "a" * 64, "pilot_H-N_r1_pass0.complete",
               "s01t_x.part0000.done " + "b" * 64]

    def pull(ctx, remote, dest, digest):
        pulled.append((remote, digest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x")
        return True
    monkeypatch.setattr(P1, "_listing", lambda ctx, cmd: list(listing))
    monkeypatch.setattr(P1, "_pull", pull)
    monkeypatch.setattr(TM.SE, "store", lambda ctx: store)
    monkeypatch.setattr(TM.SE, "_scp", lambda *a, **k: 0)
    ctx = SimpleNamespace(target=object(), evidence={}, args=SimpleNamespace(volume_mount_path="", run_id="r"),
                          auth=SimpleNamespace(stage_conditions={"kind": "tool_pilot"}))
    TM.secure(ctx)
    root = "/workspace/teacher_modes/tool_pilot_hybrid"
    assert (f"{root}/pilot_H-N_r1_pass0.part0000.jsonl", "a" * 64) in pulled
    assert (f"{root}/pilot_H-N_r1_pass0.complete", None) in pulled
    assert ("/workspace/teacher_modes/tool_pilot_hybrid/tool_session.json", None) in pulled
    assert not any("s01t_" in r for r, _ in pulled)                   #: another study's stage label is not the pilot's
    assert ctx.evidence["secured"] == {"pilot_H-N_r1_pass0.part0000.jsonl": {"ok": True, "sha256": "a" * 64}}
    rec = json.loads((store / TPL.PILOT_DIR / TM.RECEIPTS).read_text())   #: the durable receipt of each pulled shard
    assert rec["pilot_H-N_r1_pass0.part0000.jsonl"]["sha256"] == "a" * 64 and rec["pilot_H-N_r1_pass0.part0000.jsonl"]["received_utc"]
    pulled.clear()
    TM.secure(ctx)
    assert not any(r.endswith(".part0000.jsonl") for r, _ in pulled)


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
        assert row["first_candidate_admission_rate"] == TPL.UNMEASURED and row["final"]["rejected"] == 0
        assert row["admission_yield_up_to_two_candidates"] == TPL.UNMEASURED
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


# --- the metric names (maintainer 2026-11-04: data admission, never a benchmark score) ------------------------------

def test_the_report_names_admission_metrics_with_their_counts_levels_and_coverage(frozen, monkeypatch):
    """A complete pilot through the real sealer: every rate is an admission rate with its numerator and denominator in
    the row, the second candidate is counted only where the first was rejected (a bounded retry, not pass@2), each
    verifier level's passed / failed / unverified counts are per candidate, and coverage counts admitted episodes by
    the levels they were verified at."""
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    rec = {}
    assert pilot(frozen, out, runner(None), rec=rec)
    rec.update(terminal="ALL_DONE")
    (out / "tool_session.json").write_text(json.dumps(rec))
    r = TPL.report(out, plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    assert r["state"] == "COMPLETE" and r["schema"].endswith("tool_pilot_report/v4")
    assert set(r["metrics"]) >= {"first_candidate_admission_rate", "admission_yield_up_to_two_candidates",
                                 "gold_tool_call_match", "recorded_execution", "target_tools_called"}
    assert "not pass@2" in r["metrics"]["admission_yield_up_to_two_candidates"]
    assert not any("pass_rate" in k for row in r["modes"]["H-N"]["cells"].values() for k in row)
    for row in r["modes"]["H-N"]["cells"].values():
        c0, c1, fin = row["candidate_0"], row["candidate_1"], row["final"]
        assert c1["required"] == c0["rejected"] and fin["admitted"] == c0["admitted"] + c1["admitted"]
        assert row["first_candidate_admission_rate"] == round(c0["admitted"] / c0["sealed"], 4)
        assert row["admission_yield_up_to_two_candidates"] == round(
            fin["admitted"] / (fin["admitted"] + fin["rejected"]), 4)
    single = "toy/tools|single_step_gold_tool_call@v1"
    lv = r["modes"]["H-N"]["levels"][single]["candidate_0"]
    assert lv["recorded_execution"] == {"passed": 0, "failed": 0, "unverified": lv["protocol"]["passed"]
                                        + lv["protocol"]["failed"], "pass_rate": TPL.UNMEASURED}
    assert lv["step_gold"]["failed"] == 0 and lv["task_success"]["pass_rate"] == TPL.UNMEASURED
    cov = r["modes"]["H-N"]["verification_coverage"]
    assert set(cov[single]) == {"protocol+schema_validity+step_gold"}
    assert set(cov["toy/tools|full_episode@v1"]) == {"protocol+schema_validity+recorded_execution+step_gold"}
    tot = r["modes"]["H-N"]["totals"]
    assert tot["offered"] == sum(len(c["task_ids"]) for c in frozen["doc"]["modes"]["H-N"])
    sa = r["schema_arguments_non_thinking_supply"]
    assert sa["training|single_step_gold_tool_call@v1"]["supervised_tokens"] > 0
    assert sa["state_eval|single_step_gold_tool_call@v1"]["scored_positions"] > 0
    assert all(v["status"] in ("REACHED", "SHORTFALL_EXHAUSTED", "REACHABLE", "SHORTFALL_PROJECTED", TPL.UNMEASURED)
               for v in r["formal_plan_projection"].values())


# --- the L40S generation-scoped gate, on the pilot's real generation path --------------------------------------------

def test_the_pilots_generation_path_never_reads_the_forward_baseline_selection(frozen, monkeypatch):
    """The maintainer's condition for the generation-scoped gate, OBSERVED: the pilot's real session path
    (`tool_session.run_pilot` -> `generate_mode` -> `generation_driver.run_group`) up to the engine command opens no
    forward-baseline file, and the engine command names none -- it carries exactly the engine the gate compares
    (`tool_pilot.generation_engine`). The engine process's own code is the gate's static scan."""
    import os
    import sys
    opened, on = [], [False]

    def hook(event, args):
        if on[0] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))
    sys.addaudithook(hook)
    signed(frozen, monkeypatch)
    cmds = []

    def run(cmd, log, tick):
        cmds.append([str(x) for x in cmd])
        return 1
    out = frozen["tmp"] / "out"
    out.mkdir()
    on[0] = True
    try:
        assert not pilot(frozen, out, run)
    finally:
        on[0] = False
    assert cmds and any(p.endswith("pilot_tasks.jsonl") for p in opened)               #: the trace sees the reads
    assert not [p for p in opened if "baselines" in p or p.endswith("selection.json")]
    cmd = cmds[0]
    assert not any("selection" in x or "baseline" in x for x in cmd)
    e = TPL.generation_engine()
    arg = lambda k: cmd[cmd.index(k) + 1]
    assert (arg("--model"), arg("--revision"), int(arg("--session-limit"))) == (e["model"], e["revision"],
                                                                                 e["max_model_len"])
    assert float(arg("--gpu-mem-util")) == e["gpu_memory_utilization"] and "--engine-metrics" in cmd
    assert json.loads(arg("--generate-sampling")) == e["modes"]["H-N"]["sampling"]
    assert json.loads(arg("--chat-template-kwargs")) == e["modes"]["H-N"]["template_kwargs"]


def test_the_committed_pilot_passes_the_generation_scope_and_the_whole_phase_gate_still_refuses():
    """On the committed tree: the pilot's generation is admitted by the check's generation measurement; the teacher
    gate on the whole phase identity still refuses (the forward baseline changed), so no other kind is widened."""
    from stages.teacher_modes import l40s_check as LC
    if not TPL.TASKS.is_file():
        pytest.skip("the pilot's task file is an artifact outside git")
    assert TPL.l40s_generation_problems() == []
    whole = LC.gate(LC.load_config()["gates"]["teacher"])
    assert whole == ["the check ran under another phase identity than the files hold now"]
    with pytest.raises(SE.AuthorizationError, match="another phase identity"):
        TM.kind_conditions("inputs")
    #: the hybrid's formal generation reuses the generation-scoped gate (maintainer 2026-11-04) and then needs the plan
    #: verified against the ADMITTED canonical pool (gitignored bytes under data/); Thinking-2507 is another teacher,
    #: so its gate is the in-session device admission's record instead (review of a143b7ed, section 4)
    pool = TM.REPO_ROOT / json.loads(TPL.TGP.PROTOCOL.read_text())["pool"]["dir"] / "pool.json"
    if not pool.is_file():
        assert any("admitted pool" in b for b in TM.issue_problems("tool_generation"))
        return
    assert TM.issue_problems("tool_generation") == [] == TM.issue_problems("tool_generation_t2507")
    hybrid, t2507 = TM.kind_conditions("tool_generation"), TM.kind_conditions("tool_generation_t2507")
    assert "l40s_generation_scope_sha256" in hybrid and "l40s_generation_scope_sha256" not in t2507
    assert f"record:{TM.IN_SESSION_DEVICE_ADMISSION['tool_generation_t2507']}" in t2507


# --- the revision on the accepted pilot (maintainer 2026-11-04): priors, import, diagnosis, projection by shape --------

def _complete_pilot(frozen, monkeypatch):
    runner = _interrupt_then_resume(frozen, monkeypatch)
    out = frozen["tmp"] / "out"
    out.mkdir()
    rec = {}
    assert pilot(frozen, out, runner(None), rec=rec)
    rec.update(terminal="ALL_DONE", identity=TPL.pilot_identity("id", frozen["plan"])["sha256"])
    (out / "tool_session.json").write_text(json.dumps(rec))
    return out


def test_the_priors_are_the_reports_own_counts_by_stratum(frozen, monkeypatch):
    from stages.teacher_modes import tool_generation_priors as TGPR
    out = _complete_pilot(frozen, monkeypatch)
    doc = TGPR.measure(out, plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    rep = TPL.report(out, plan_path=frozen["plan"], tasks_path=frozen["tasks"])
    for mode, m in rep["modes"].items():
        for key, row in m["cells"].items():
            split, cat, shape = key.split("|")
            st = doc["strata"][f"hybrid|{mode}|{split}|{cat}|{shape}"]
            assert st["attempted"] == row["c0"].get("admitted", 0) + row["c0"].get("rejected", 0)
            assert st["admitted"] == row["final"]["admitted"]
            assert st["units"] == row["supervised_tokens"] + row["scored_positions"]
            lo, mid, hi = st["units_per_attempted_task"]
            assert lo <= mid <= hi and mid == pytest.approx(st["units"] / st["attempted"], abs=1e-3)
    assert TGPR.wilson(0, 10)[0] == 0.0 and TGPR.wilson(10, 10)[1] == 1.0


def test_the_projection_reads_the_whole_supply_or_projects_by_shape():
    doc = {"modes": {"H-N": [{"split": "state_eval", "category": "c", "unit": "scored_positions", "target": 100, "supply": 4},
                             {"split": "training", "category": "c", "unit": "supervised_tokens", "target": 1000, "supply": 30}]}}
    row = lambda a, r, u, unit="scored_positions": {"final": {"admitted": a, "rejected": r},  # noqa: E731
                                                     "supervised_tokens": u if unit != "scored_positions" else 0,
                                                     "scored_positions": u if unit == "scored_positions" else 0}
    modes = {"H-N": {"cells": {"state_eval|c|full_episode@v1": row(2, 2, 60),
                               "training|c|full_episode@v1": row(1, 1, 50, "t"),
                               "training|c|single_step_gold_tool_call@v1": row(2, 8, 40, "t")}}}
    sup = {"H-N|training|c": {"full_episode@v1": 10, "single_step_gold_tool_call@v1": 20}}
    out = TPL._projection(doc, modes, sup)
    assert out["H-N|state_eval|c"]["status"] == "SHORTFALL_EXHAUSTED" and out["H-N|state_eval|c"]["sealed_units"] == 60
    t = out["H-N|training|c"]
    assert t["projected_units_from_whole_supply"] == pytest.approx(50 + 8 * 25 + 40 + 10 * 4)    #: 330, by shape
    assert t["status"] == "SHORTFALL_PROJECTED"
    assert TPL._projection(doc, modes)["H-N|training|c"]["status"] == TPL.UNMEASURED            #: no per-shape supply


def test_the_diagnosis_separates_replay_misses_from_their_consequences_and_gold_kinds():
    from stages.teacher_modes import tool_pilot_diagnosis as D
    v = SimpleNamespace(problems=["turn 0: the backend did not answer every call", "the episode ends on a call, not on an answer"],
                        checks={"schema_validity": True, "step_gold": None, "target_tools_called": False})
    assert D.causes({}, [], v) == ["replay_consequence", "replay_no_response"]
    v = SimpleNamespace(problems=["the episode ends on a call, not on an answer"], checks={"schema_validity": True})
    assert D.causes({}, [], v) == ["ends_on_answered_call"]
    v = SimpleNamespace(problems=["turn 0: stopped by 'length', not its end of turn",
                                  "the task's target_tools_called check failed"],
                        checks={"schema_validity": True, "target_tools_called": False})
    assert D.causes({}, [], v) == ["model_protocol", "target_tools_not_called"]
    task = {"gold_steps": [[[{"name": "f", "arguments": {"summary": "x" * 40, "n": 1}}]]]}
    turn = lambda args: [{"raw": f"<tool_call>{json.dumps({'name': 'f', 'arguments': args})}</tool_call>"}]  # noqa: E731
    assert D.gold_kind(task, turn({"summary": "y" * 40, "n": 1}))[0] == "free_text"
    assert D.gold_kind(task, turn({"summary": "x" * 40, "n": 2}))[0] == "value"
    assert D.gold_kind(task, [{"raw": "<tool_call>{\"name\": \"g\", \"arguments\": {}}</tool_call>"}])[0] == "other_tool"
    assert D.gold_kind(task, [{"raw": "I will look it up."}])[0] == "no_call"


def test_the_committed_import_record_and_its_shards_agree():
    """The import of the accepted pilot: every check equal, both modes' 397 tasks DONE, each candidate's shard by its
    sha256; the dev box's admission reads exactly those candidates and refuses a changed shard."""
    from stages.teacher_modes import tool_admission as TA
    from stages.teacher_modes import tool_generation_import as TGI
    rec = TGI.load()
    assert rec["compatibility"]["ok"] and all(c["equal"] for c in rec["compatibility"]["checks"].values())
    assert {m: v["totals"]["tasks"] for m, v in rec["modes"].items()} == {"H-N": 397, "H-T": 397}
    assert all(t["final"]["outcome"] == "admitted" or len(t["candidates"]) == 2 for v in rec["modes"].values()
               for t in v["tasks"].values())                                   #: a refused task spent both candidates
    store = TGI.REPO / rec["store"]
    if not store.is_dir():
        pytest.skip("the pilot's secured shards are an artifact outside git")
    eps = TA.episodes_from_import(rec, store, "H-N")
    assert len(eps) == rec["modes"]["H-N"]["totals"]["candidates"]
    bad = json.loads(json.dumps(rec))
    t = next(iter(bad["modes"]["H-N"]["tasks"].values()))
    t["candidates"][0]["shard_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="is not the import record's"):
        TA.episodes_from_import(bad, store, "H-N")


def test_an_import_refuses_when_a_generation_term_differs(monkeypatch):
    from stages.teacher_modes import tool_generation_import as TGI
    rec = TGI.load()
    real = TGI._at

    def other(commit, rel):
        b = real(commit, rel)
        if rel == TGI.PROTOCOL_REL:
            d = json.loads(b)
            d["execution"]["seed_base"] += 1
            return json.dumps(d).encode()
        return b
    monkeypatch.setattr(TGI, "_at", other)
    doc, by_id = TPL.load(TPL.PLAN, TPL.TASKS, protocol_path=TPL.frozen_protocol()) if TPL.TASKS.is_file() else (None, None)
    if doc is None:
        pytest.skip("the pilot's task file is an artifact outside git")
    session = {"identity_parts": {"implementation_sha256": {f: TPL.sha(TGI.REPO / f) for f in TGI.GENERATION_CODE},
                                  "verification_version": TE.VERIFICATION_VERSION}}
    comp = TGI.compatibility(rec["run_id"], session, by_id, doc, by_id)
    assert not comp["ok"] and comp["checks"]["execution"]["equal"] is False and comp["checks"]["inputs"]["equal"]

"""Research A on the L40S by default ($0): the card, the check before the teacher session, the initialization's declared
memory fallback, the check's verdict and its formal paths on diagnostic inputs, and one budget across its records."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-2"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import initialization as IN  # noqa: E402
from stages.teacher_modes import l40s_check as L  # noqa: E402
from stages.teacher_modes import phase as PH  # noqa: E402
from stages.teacher_modes import teacher_session as TS  # noqa: E402

APP = REPO / "logs/budget/applications/research_a_2026-10-29"
IDENT = "e" * 64


def test_every_kind_runs_on_the_l40s_and_the_proposal_lists_no_other_card():
    prop = json.loads((APP / "proposal.json").read_text())
    design = json.loads((REPO / PH.DESIGN).read_text())
    assert PH.GPU == "NVIDIA L40S" and set(prop["gpus"]) == {PH.GPU}
    assert PH.build_parser().parse_args(["--run-id", "r", "--scr", "s", "--session-commit", "0" * 40, "--bundle", "b",
                                         "--max-price", "1"]).gpu == PH.GPU
    assert "L40S" in design["execution"]["hardware"] and "A100" not in json.dumps(design["execution"]["sessions"])
    assert PH.STUDY.kinds == ("check", "teacher", "recovery")
    assert PH.STUDY.claims_of("check") == {"engineering_validation"}
    assert PH.STUDY.claims_of("teacher") == PH.STUDY.claims_of("recovery") == PH.STUDY.claims
    assert {PH.CHECK_CONFIG, PH.DIAGNOSTIC} <= set(PH.STUDY.identity_files)


def test_an_out_of_memory_initialization_is_rerun_once_at_the_declared_fallback(tmp_path, monkeypatch):
    base = IN.execution().micro_batch_size
    fallback = json.loads((REPO / PH.DESIGN).read_text())["initialization"]["execution"]["on_out_of_memory"]
    assert (base, fallback) == (3, 1) and IN.execution(1).micro_batch_size == 1
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        if "--micro-batch-size" in cmd:
            (tmp_path / "initialization" / "initialization.json").write_text("{}")
            return SimpleNamespace(returncode=0)
        (tmp_path / "initialization").mkdir(exist_ok=True)
        (tmp_path / "initialization" / f"oom_mb{base}.json").write_text("{}")
        return SimpleNamespace(returncode=IN.OOM_EXIT)
    monkeypatch.setattr(TS.subprocess, "run", fake)
    got = TS.initialize({"launch_identity_sha256": IDENT}, tmp_path, {}, None)
    assert got["status"] == "persisted" and [a["micro_batch_size"] for a in got["attempts"]] == [base, fallback]
    assert "--micro-batch-size" not in calls[0] and calls[1][-2:] == ["--micro-batch-size", "1"]
    # a size that already ran out of memory is never repeated: the next attempt starts at the fallback
    (tmp_path / "initialization" / "initialization.json").unlink()
    assert TS.initialization_sizes(tmp_path) == [fallback]
    (tmp_path / "initialization" / f"oom_mb{fallback}.json").write_text("{}")
    assert TS.initialize({"launch_identity_sha256": IDENT}, tmp_path, {}, None)["status"] == "failed"


def test_an_out_of_memory_error_is_seen_through_an_operators_wrapper():
    class OutOfMemoryError(RuntimeError):
        pass
    try:
        try:
            raise OutOfMemoryError("Tried to allocate 11.68 GiB")
        except OutOfMemoryError as inner:
            raise ValueError("DEPTH failed") from inner
    except ValueError as exc:
        assert IN.is_out_of_memory(exc)
    assert IN.is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not IN.is_out_of_memory(ValueError("shape mismatch"))


def _evidence(out: Path, *, init=("oom", "ok"), reserved=30.0, gen_used=41000) -> None:
    L._write(out, "device", {"generation": {"used_mib_peak": gen_used, "total_mib": 46068, "samples": 9},
                             "evaluation_engine": {"used_mib_peak": 41100, "total_mib": 46068, "samples": 4}})
    L._write(out, "generation", {"tasks": 64, "modes": {m: {"complete": True, "tokens_per_second": 900.0, "engine": {}}
                                                        for m in L.MODES}})
    L._write(out, "evaluation_engine", {"started": True, "context": 40960, "engine": {"max_concurrency": 4.9}})
    for m in L.MODES:
        L._write(out, f"train_{m}", {"memory": {"total_gib": 44.39, "reserved_gib": reserved, "allocated_gib": 24.0},
                                     "seconds_per_step_mean_after_warmup": 5.5, "nonpad_tokens_per_second": 2900.0,
                                     "documents": 60})
    for k, state in zip((3, 1), init):
        steps = [{"impl_id": "depth.causal_kl_greedy_v1", "seconds": 900.0,
                  "device_memory": {"allocated_gib": 30.0, "reserved_gib": 34.0, "total_gib": 44.39}}]
        L._write(out, f"initialization_mb{k}", {"execution": {"micro_batch_size": k}, "out_of_memory": state == "oom",
                                                 "steps": steps if state == "ok" else [], "seconds": 60.0,
                                                 **({"final": {"num_parameters": 596049920}} if state == "ok" else {})})


def test_the_check_verdict_derives_each_margin_and_names_only_memory_bound_operations(tmp_path):
    cfg = L.load_config()
    _evidence(tmp_path)
    v = L.verdict(tmp_path, cfg)
    assert v["complete"] and v["fit_with_margin"] and not v["a100_evidence"] and not v["formal"]
    ops = v["operations"]
    assert ops["train_H-T"]["memory"] == {"total_gib": 44.39, "peak_gib": 30.0, "free_gib": 14.39, "fits_with_margin": True}
    assert ops["generation"]["memory"]["free_gib"] == round(46068 / 1024 - 41000 / 1024, 2)
    assert [(a["micro_batch_size"], a["out_of_memory"], a["completed"]) for a in ops["initialization"]["attempts"]] == [
        (1, False, True), (3, True, False)]
    assert ops["initialization"]["memory"]["free_gib"] == 10.39
    tight = tmp_path / "tight"
    _evidence(tight, reserved=41.0)
    assert L.verdict(tight, cfg)["below_margin"] == ["train_H-N", "train_H-T"]
    both = tmp_path / "both"
    _evidence(both, init=("oom", "oom"))
    v = L.verdict(both, cfg)
    assert v["failed"] == ["initialization"] and v["a100_evidence"] == ["initialization"] and not v["complete"]


RUN = "check_r1"


def _chain(tmp: Path, monkeypatch, *, pod_report=None, **evidence) -> tuple[Path, Path]:
    """A closed check on a repository holding this phase's identity files: the pod's evidence and report, the
    shared closeout's verified archive and outcome, the committed check authorization; returns (repo, scr)."""
    root, ev, scr = tmp / "repo", tmp / "evidence", tmp / "scr"
    for rel in PH.STUDY.identity_files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).symlink_to(REPO / rel)
    _evidence(ev, **evidence)
    cfg = L.load_config()
    L._write(ev, "run", {"config_sha256": L._sha(REPO / L.CONFIG), "contract": {
        "status": "ok", "identity": PH.SE.identity_hash(PH.STUDY, root), "harness": "h", "authorization_id": "auth-1"}})
    L._write(ev, "report", pod_report if pod_report is not None else L._plain(L.verdict(ev, cfg)))
    (scr / "store").mkdir(parents=True)
    tgz = scr / "store" / PH.SE.EVIDENCE_ARCHIVE
    with tarfile.open(tgz, "w:gz") as t:
        t.add(ev, arcname=".")
    run = root / PH.STUDY.run_dir(RUN)
    L._write(run / "closeout", "outcome", {"kind": "check", "evidence": {"sha256": L._sha(tgz), "verified": True},
                                           "verdict_rederived": {"agrees_with_the_pod": pod_report is None}})
    auth = root / PH.STUDY.authorization_path(RUN)
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"stage_conditions": {"kind": "check"}, "authorization_id": "auth-1"}))
    #: the artifact loader's own checks are the shared tests'; here, what the gate reads from it
    monkeypatch.setattr(PH.PhaseAuthorization, "load", lambda path: SimpleNamespace(**json.loads(Path(path).read_text())))
    return root, scr


TEACHER, RECOVERY = (L.load_config()["gates"][k] for k in ("teacher", "recovery"))


def test_the_check_record_admits_a_formal_session_only_fail_closed(tmp_path, monkeypatch):
    root, scr = _chain(tmp_path / "ok", monkeypatch)
    with pytest.raises(PH.SE.AuthorizationError, match="L40S check"):
        PH.kind_conditions("teacher", root)                                    # no record
    (root / PH.CHECK_RECORD).parent.mkdir(parents=True)
    (root / PH.CHECK_RECORD).write_text('{"verdict": {}}\n')
    with pytest.raises(PH.SE.AuthorizationError, match="not a check record"):
        PH.kind_conditions("teacher", root)                                    # an empty verdict
    rec = L.record(RUN, scr, root)
    assert rec["agrees_with_the_pod"] and rec["evidence"].endswith(f"runs/{RUN}/check_evidence")
    assert PH.kind_conditions("teacher", root) == {"l40s_check_record_sha256": L._sha(root / PH.CHECK_RECORD)}
    assert L.gate(RECOVERY, root) == []
    # identities: the phase's, the run's
    for key, value, why in (("identity", "0" * 64, "another phase identity"),
                            ("authorization_id", "auth-2", "not the check the record names")):
        bad = {**rec, key: value}
        (root / PH.CHECK_RECORD).write_text(json.dumps(bad))
        assert any(why in b for b in L.gate(TEACHER, root))
    (root / PH.CHECK_RECORD).write_text(json.dumps(rec))
    # the committed evidence is the recorded one, and re-derives the recorded verdict
    t = root / rec["evidence"] / "train_H-T.json"
    t.write_text(t.read_text().replace('"reserved_gib": 30.0', '"reserved_gib": 29.0'))
    assert any("committed evidence" in b for b in L.gate(RECOVERY, root))
    # a failed initialization refuses the teacher session (and is the one operation an A100 may be asked for)
    root, scr = _chain(tmp_path / "oom", monkeypatch, init=("oom", "oom"))
    rec = L.record(RUN, scr, root)
    assert rec["verdict"]["a100_evidence"] == ["initialization"]
    with pytest.raises(PH.SE.AuthorizationError, match="initialization: FAILED"):
        PH.kind_conditions("teacher", root)
    assert L.gate(RECOVERY, root) == []
    # the pod's report and the dev box's re-derivation disagree
    root, scr = _chain(tmp_path / "pod", monkeypatch, pod_report={"passed": True, "checks": {}})
    assert not L.record(RUN, scr, root)["agrees_with_the_pod"]
    assert any("disagree" in b for b in L.gate(TEACHER, root)) and any("closeout" in b for b in L.gate(TEACHER, root))


def test_a_margin_below_the_wanted_one_always_refuses_and_never_names_an_a100(tmp_path, monkeypatch):
    root, scr = _chain(tmp_path, monkeypatch, gen_used=43000, reserved=41.0)
    rec = L.record(RUN, scr, root)
    v = rec["verdict"]
    assert v["complete"] and v["below_margin"] == ["generation", "train_H-N", "train_H-T"] and not v["a100_evidence"]
    assert [b.split(":")[0] for b in L.gate(TEACHER, root)] == ["generation"]
    assert [b.split(":")[0] for b in L.gate(RECOVERY, root)] == ["train_H-T", "train_H-N"]
    # no exception can be declared from the dev box: neither the record command nor a hand-edited record admits it
    assert "--accept-margin" not in L.build_parser().format_help() + L.build_parser()._subparsers._group_actions[0]  \
        .choices["record"].format_help()
    forged = {**rec, "margin_accepted": {"generation": {"decided_by": "maintainer", "decision": "accept"}}}
    (root / PH.CHECK_RECORD).write_text(json.dumps(forged))
    assert [b.split(":")[0] for b in L.gate(TEACHER, root)] == ["generation"]


class FakeEngine:
    """run_battery's contract in both modes: generate streams one record per task; eval writes --out at the end."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, log, tick):
        arg = lambda k: cmd[cmd.index(k) + 1]  # noqa: E731
        self.calls.append(cmd)
        Path(log).write_text("GPU KV cache size: 218,112 tokens\nMaximum concurrency for 40,960 tokens per request: 5.32x\n")
        mode = "H-T" if json.loads(arg("--chat-template-kwargs"))["enable_thinking"] else "H-N"
        proto = json.loads((REPO / L.PROTOCOL).read_text())
        rows = []
        for line in Path(arg("--tasks")).read_text().splitlines():
            t = json.loads(line)
            n = random.Random(t["task_id"] + mode).randint(300, 900)
            p, o = [11] * 30 + proto["modes"][mode]["generation_suffix_ids"], [7] * n + [151645]
            rows.append({"task_id": t["task_id"], "end": "answered", "messages": t["messages"],
                         "turns": [{"turn": 0, "prompt_tokens": len(p), "generated_tokens": len(o), "stop_reason": "eos",
                                    "think_state": proto["modes"][mode]["think_state"], "raw": "x", "answer": "x",
                                    "prompt_token_ids": p, "output_token_ids": o}]})
        if "--stream-completed" in cmd:
            with open(arg("--stream-completed"), "a") as out:
                out.write("".join(json.dumps(r) + "\n" for r in rows))
        else:
            Path(arg("--out")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return 0


def test_the_check_runs_the_teacher_sessions_runner_and_the_40960_engine_on_diagnostic_tasks_only(tmp_path):
    cfg = L.load_config()
    cfg.update(order=["generation", "evaluation_engine"], sampler_seconds=3600)
    cfg["generation"]["tasks"], cfg["evaluation_engine"]["tasks"] = 4, 2
    conf = tmp_path / "check.json"
    conf.write_text(json.dumps(cfg))
    args = SimpleNamespace(config=str(conf), out=str(tmp_path / "out"), work=str(tmp_path / "work"),
                           status=str(tmp_path / "status"), student="s", authorization="a.json", identity="")
    engine = FakeEngine()
    authorize = lambda path, repo: {"authorization_id": "a", "identity": IDENT, "harness": "h", "kind": "check"}  # noqa: E731
    sampler = L.Sampler(3600, query=lambda: (20000, 46068))
    report = L.run(args, runner=engine, authorize=authorize, sampler=sampler)
    gen = L._read(tmp_path / "out", "generation")
    ids = {t["task_id"] for t in L.diagnostic_tasks(cfg)}
    seen = {json.loads(l)["task_id"] for c in engine.calls for l in Path(c[c.index("--tasks") + 1]).read_text().splitlines()}
    assert seen <= ids and len(seen) == 4
    assert all(m["complete"] and m["tokens_per_second"] and m["engine"]["kv_cache_tokens"] == 218112
               for m in gen["modes"].values())
    eng = L._read(tmp_path / "out", "evaluation_engine")
    cmd = next(c for c in engine.calls if "--mode" in c and c[c.index("--mode") + 1] == "eval")
    assert eng["started"] and eng["context"] == 40960 and cmd[cmd.index("--trained-context") + 1] == "40960"
    assert "--session-limit" not in cmd and eng["engine"]["max_concurrency"] == 5.32
    assert report["operations"]["train_H-T"]["status"] == "NOT_RUN" and not report["complete"]
    assert "MARKER:RUN_FAILED" in Path(args.status).read_text()


def test_the_diagnostic_tasks_are_bound_disjoint_and_isolated():
    from stages.reasoning_data import assets as A
    cfg = L.load_config()
    rows = L.diagnostic_tasks(cfg)
    man = json.loads((REPO / PH.DIAGNOSTIC).with_name("diagnostic_v1.manifest.json").read_text())
    order = list(A.jsonl(REPO / PH.ORDER))
    proto = json.loads((REPO / L.PROTOCOL).read_text())
    t0 = {json.loads(l)["task_id"] for lst in proto["inputs"]["t0"] for l in (REPO / lst["path"]).read_text().splitlines()
          if l.strip()}
    known_ids, known_content = t0 | {r["task_id"] for r in order}, {r["content_sha256"] for r in order}
    assert len(rows) == man["n_tasks"] == 64 and man["sha256"] == cfg["diagnostic_tasks"]["sha256"]
    assert man["isolation"]["pass"] and not man["isolation"]["violations"]
    assert not {r["task_id"] for r in rows} & known_ids and not {r["content_sha256"] for r in rows} & known_content
    assert all(r["slice"] == "diagnostic" for r in rows)


def test_the_check_trains_the_formal_recipe_and_nothing_else():
    cfg = L.load_config()
    got, changed = L.train_config(cfg, "H-T", "student", Path("/tmp/x"))
    assert got["loss"]["kd_support"] == {"support_id": "reference_topk_tail_v1", "top_k": 200}
    assert got["loss"]["kd_scope"] == "assistant" and got["batch"]["context_strategy"] == "document_isolated"
    assert got["batch"]["micro_blocks"] == 1 and got["block_len"] == 8192
    declared = {"loss.kd_scope", "loss.kd_support.support_id", "loss.kd_support.top_k", "batch.context_strategy",
                "checkpoint.bind_identity", "teacher.model_id", "teacher.revision", "run_name", "_purpose",
                "student_path", "out_dir", "device", "schedule.total_steps", "schedule.warmup_steps",
                "checkpoint.save_every", "intervals.log_every", "intervals.eval_every", "intervals.eval_blocks"}
    assert set(changed) <= declared and {"loss.kd_support.top_k", "batch.context_strategy"} <= set(changed)


def test_the_forecast_application_and_proposal_state_one_budget():
    from stages.teacher_modes import phase_forecast as PF
    f = json.loads((APP / "forecast.json").read_text())
    assert PF.forecast() == f
    app, prop = (json.loads((APP / n).read_text()) for n in ("application.json", "proposal.json"))
    hc = f["hard_caps_usd"]
    assert app["phase_hard_cap_usd"] == prop["ceiling_usd"] == hc["phase_total"]
    assert app["expected_usd"] == f["expected_usd"] <= app["conservative_usd"] == f["conservative_usd"] <= hc["phase_total"]
    assert app["requested_project_cap_usd"] == prop["project_cumulative_cap_usd"] == math.ceil(
        app["project_position"]["committed_usd"] + hc["phase_total"])
    assert prop["session_hard_caps_usd"] == {"check": hc["check"], "teacher": hc["teacher_session"],
                                             "recovery": hc["recovery"]}
    assert prop["per_session_hard_cap_usd"] == max(prop["session_hard_caps_usd"].values())
    assert {e["id"]: e["hard_cap_usd"] for e in app["envelopes"]} == {
        "check": hc["check"], "teacher": hc["teacher_session"], "recovery": hc["recovery"], "storage": hc["storage"],
        "engineering_reserve": hc["engineering_reserve"]}
    assert "a100" in app["not_requested"] and set(app["hardware_range"]["cards"]) == {"NVIDIA L40S"}
    # the live price and the ceiling told apart: the proposal's ceiling, the launcher's planning, one set of bounds
    b = f["bounds"]
    assert PF.PRICE_CEILING == prop["gpus"]["NVIDIA L40S"] == b["ceiling_usd_per_gpu_hour"] > b["live_usd_per_gpu_hour"]
    assert PF.CONTINGENCY == PH.launch_spec("teacher").contingency_fraction
    assert app["price"]["phase"]["as_run"] == {k: b["phase"][k] for k in ("expected", "conservative")}
    assert b["phase"]["conservative"]["at_ceiling"] <= hc["phase_total"]
    assert not b["teacher"]["fits_the_cap_as_run"]["conservative"]["at_ceiling"]          # stated, handled, not raised
    assert all(v < b["teacher"]["longest_session_work_hours"]["at_ceiling"]
               for v in b["teacher"]["bounded_sessions"]["through_w00"].values())
    two = app["envelopes"][[e["id"] for e in app["envelopes"]].index("recovery")]
    assert "ESTIMATE" in two["two_gpus_estimate"]["_"] and "wall_two_gpus_expected" not in two["hours"]


def test_longest_session_is_the_shared_planners_refusal_point():
    from aadistill.infrastructure.budget import BudgetError, Phase, StepTime, plan_session
    from stages.teacher_modes import phase_forecast as PF
    h = PF.longest_session_hours(63.76, 1.19)

    def plan(hours):
        return plan_session(price_per_hour=1.19, authorized_usd=63.76, arms=0, steps_per_arm=0,
                            step_time=StepTime(1.0, "unused"), setup_minutes=20.0, transfer_minutes=5.0,
                            other_phases=(Phase("work", hours * 60),), contingency_fraction=0.25,
                            soft_stop_reserves=(Phase("closeout_and_teardown", 12.5),),
                            artifact_recovery_reserve_minutes=10.0)
    assert plan(h).hard_terminate_usd <= 63.76
    with pytest.raises(BudgetError):
        plan(h + 0.02)


def test_a_phase_that_raised_or_crashed_is_failed_not_unrun(tmp_path):
    cfg = L.load_config()
    L._write(tmp_path, "generation_error", {"error": "RuntimeError('engine')"})
    L._write(tmp_path, "initialization_exits", {"3": -9})
    v = L.verdict(tmp_path, cfg)["operations"]
    assert v["generation"]["status"] == "FAILED" and v["initialization"]["status"] == "FAILED"
    assert v["evaluation_engine"]["status"] == "NOT_RUN" and v["train_H-T"]["status"] == "NOT_RUN"


def test_the_check_launch_uses_the_shared_relay_and_its_own_driver():
    launch = PH.launch_spec("check", 250.0)
    assert launch.kind == "check" and launch.driver.endswith("l40s_check.py") and launch.verdict is PH.check_verdict
    assert launch.on_poll is None and launch.products_secured is None          # the shared evidence relay secures it
    assert PH.check_evidence_dir("r7") == PH.SE.evidence_dir(launch, "r7")
    assert [a.dest_name for a in launch.local_assets] == ["profiles_formal_v1"] and launch.relay_inputs
    args = SimpleNamespace(run_id="r7", volume_mount_path="/runpod-volume")
    cmd = PH.check_command(SimpleNamespace(args=args), None)
    assert f"--out {PH.check_evidence_dir('r7')}" in cmd and "--student /workspace/aad/" in cmd

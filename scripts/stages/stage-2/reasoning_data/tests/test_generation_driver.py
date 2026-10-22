"""The generation driver's control flow, with the generator injected.

The injected runner writes REAL Stage-0 teacher records for the tasks it is
given, so the second-pass decision is the adapter's real verdict; nothing
here imitates the engine beyond copying its records.
"""
from __future__ import annotations

import json

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_driver as D
from stages.reasoning_data import generation_plan as GP
from stages.reasoning_data.generation_adapter import candidate


@pytest.fixture(scope="module")
def stage0():
    tasks = A.calibration_tasks()
    recs = {r["task_id"]: r for r in A.teacher_cal_records()}
    good, bad = [], []
    for tid, r in recs.items():
        t = tasks[tid]
        if t["interactive"]:
            continue
        (good if candidate(t, r, 0).accepted_for_training else bad).append(tid)
    return tasks, recs, good[:2] + bad[:2], set(bad[:2])


def _setup(tmp_path, stage0):
    tasks, recs, ids, _ = stage0
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(tasks[i]) + "\n" for i in ids))
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    files = GP.write_inputs([tasks[i] for i in ids], spec, tmp_path / "inputs")
    plan = tmp_path / "generation_plan.json"
    plan.write_text(json.dumps({"inputs": files, "session_limit": GP.SESSION_LIMIT,
                                "pools": [{"path": str(pool), "sha256": A.sha256_file(pool)}]}))
    calls = []

    def run(cmd, log):
        calls.append(cmd)
        arg = dict(zip(cmd, cmd[1:]))
        with open(arg["--out"], "w") as f:
            for row in A.jsonl(arg["--tasks"]):
                f.write(json.dumps(recs[row["task_id"]]) + "\n")
        return 0

    argv = ["--plan", str(plan), "--inputs-dir", str(tmp_path / "inputs"), "--authorization", "a.json",
            "--status", str(tmp_path / "status"), "--out", str(tmp_path / "out")]
    return argv, run, calls


def _ok(path, sha):
    return {"authorization_id": "test", "hard_cap_usd": 0.0}


def test_two_passes_tag_each_candidate_and_retry_only_the_rejected(tmp_path, stage0):
    argv, run, calls = _setup(tmp_path, stage0)
    assert D.main(argv, run=run, authorize=_ok) == 0
    out = tmp_path / "out"
    p0, p1 = out / "pass0_generation_input_g0.jsonl", out / "pass1_generation_input_g0.jsonl"
    r0, r1 = list(A.jsonl(p0)), list(A.jsonl(p1))
    assert {r["candidate_index"] for r in r0} == {0} and {r["candidate_index"] for r in r1} == {1}
    assert {r["task_id"] for r in r1} == stage0[3]           # exactly the first pass's rejections
    assert r0[0]["generation_pass"]["seed_base"] == D.SEED_BASE
    assert r1[0]["generation_pass"]["seed_base"] == D.SEED_BASE + 1
    assert (out / "pass0_generation_input_g0.done").read_text().strip() == A.sha256_file(p0)
    assert [c[c.index("--seed-base") + 1] for c in calls] == [str(D.SEED_BASE), str(D.SEED_BASE + 1)]
    ev = json.loads((out / "generation_session.json").read_text())
    assert ev["terminal"] == "ALL_DONE"
    assert (tmp_path / "status").read_text().splitlines()[-1].endswith("MARKER:ALL_DONE")


def test_a_rerun_regenerates_nothing_already_done(tmp_path, stage0):
    argv, run, calls = _setup(tmp_path, stage0)
    D.main(argv, run=run, authorize=_ok)
    calls.clear()
    assert D.main(argv, run=run, authorize=_ok) == 0
    assert calls == []


def test_an_input_that_differs_from_the_plan_runs_nothing(tmp_path, stage0):
    argv, run, calls = _setup(tmp_path, stage0)
    f = tmp_path / "inputs" / "generation_input_g0.jsonl"
    f.write_text(f.read_text() + "\n")
    assert D.main(argv, run=run, authorize=_ok) == 2
    assert calls == [] and "RUN_FAILED" in (tmp_path / "status").read_text()


def test_a_failed_group_is_recorded_and_gets_no_second_pass(tmp_path, stage0):
    argv, _, calls = _setup(tmp_path, stage0)

    def fail(cmd, log):
        calls.append(cmd)
        return 1
    assert D.main(argv, run=fail, authorize=_ok) == 3
    ev = json.loads((tmp_path / "out" / "generation_session.json").read_text())
    assert ev["terminal"] == "RUNS_FAILED" and len(calls) == 1
    assert not (tmp_path / "out" / "pass0_generation_input_g0.done").exists()


def test_a_verdict_only_the_sandbox_can_give_gets_no_second_candidate(tmp_path, stage0, monkeypatch):
    argv, run, calls = _setup(tmp_path, stage0)
    monkeypatch.setattr(GP, "verifier_of", lambda t: ("code_asserts", None))
    assert D.main(argv, run=run, authorize=_ok) == 0
    assert len(calls) == 1


def test_an_authorization_for_another_plan_runs_nothing(tmp_path, stage0):
    argv, run, calls = _setup(tmp_path, stage0)

    def other(path, sha):
        raise D.AuthorizationError("bound to another plan")
    assert D.main(argv, run=run, authorize=other) == 2
    assert calls == []


def test_arm_b_places_every_session_in_its_cell_and_refuses_the_rest():
    from stages.reasoning_data import build_arm_b as AB
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    shares = AB.cell_shares(spec, "stage2.reasoning_master@v1")
    assert abs(sum(shares.values()) - 1.0) < 1e-9 and "supporting.code_algorithmic/code_contests_train" not in shares
    s = {"id": "u#t0", "unit_id": "u", "stratum": "supporting.math_new", "attributes": {"source": "math_train"}}
    assert AB.place([s], shares)[0]["stratum"] == "supporting.math_new/math_train"
    with pytest.raises(SystemExit, match="does not allocate"):
        AB.place([{**s, "attributes": {"source": "numinamath_1p5"}}], shares)
    with pytest.raises(SystemExit, match="does not allocate"):
        AB.place([{**s, "attributes": {}}], shares)

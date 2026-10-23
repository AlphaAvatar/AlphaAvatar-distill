"""The generation driver: bound contract, completion shards, resume, terminal.

The generator is injected: it writes REAL Stage-0 teacher records for the tasks
it is given into the per-episode stream, as `run_battery --stream-completed`
does, and can be made to die part way. The second-pass decision is the
adapter's real verdict. The repository the driver reads is a small copy in
tmp_path (plan, protocol, inputs, pool), so nothing here touches the real one.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_driver as D
from stages.reasoning_data import generation_plan as GP
from stages.reasoning_data import governance as GOV
from stages.reasoning_data.generation_adapter import candidate

PROTO = A.read_json(A.REPO / GOV.PROTOCOL)
LID = "f" * 64


@pytest.fixture(scope="module")
def stage0():
    tasks = A.calibration_tasks()
    recs = {r["task_id"]: r for r in A.teacher_cal_records()}
    good, bad = [], []
    for tid, r in recs.items():
        if tasks[tid]["interactive"]:
            continue
        (good if candidate(tasks[tid], r, 0).accepted_for_training else bad).append(tid)
    return tasks, recs, good[:2] + bad[:2], set(bad[:2])


def _repo(tmp_path, stage0, *, protocol=PROTO, plan=GOV.PLAN, inputs=GOV.INPUTS_DIR):
    """A minimal repository: plan, protocol, inputs and pool where governance names them."""
    tasks, _, ids, _ = stage0
    repo = tmp_path / "repo"
    pool = repo / "pool.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("".join(json.dumps(tasks[i]) + "\n" for i in ids))
    files = GP.write_inputs([tasks[i] for i in ids], A.read_json(A.CONFIG_DIR / "corpus_spec.json"),
                            repo / inputs)
    (repo / plan).parent.mkdir(parents=True, exist_ok=True)
    (repo / plan).write_text(json.dumps({"inputs": files, "inputs_dir": str(inputs),
                                         "pools": [{"path": "pool.jsonl", "sha256": A.sha256_file(pool)}]}))
    (repo / GOV.PROTOCOL).parent.mkdir(parents=True)
    (repo / GOV.PROTOCOL).write_text(json.dumps(protocol))
    return repo


def _ok(path, repo):
    return {"authorization_id": "test", "hard_cap_usd": 0.0, "launch_identity_sha256": LID, "harness_digest": "h",
            "plan_path": str(GOV.PLAN)}


class Gen:
    """The injected generator: writes the stream as run_battery would, and can die after `die_after` records."""

    def __init__(self, recs, die_after=None, tick_every=1):
        self.recs, self.die_after, self.tick_every, self.calls = recs, die_after, tick_every, []

    def __call__(self, cmd, log, tick):
        arg = dict(zip(cmd, cmd[1:]))
        self.calls.append({"cmd": cmd, "tasks": [r["task_id"] for r in A.jsonl(arg["--tasks"])]})
        with open(arg["--stream-completed"], "a") as f:
            for n, row in enumerate(A.jsonl(arg["--tasks"])):
                if self.die_after is not None and n >= self.die_after:
                    return 137                                         # killed part way
                f.write(json.dumps(self.recs[row["task_id"]]) + "\n")
                f.flush()
                if (n + 1) % self.tick_every == 0:
                    tick()
        Path(arg["--out"]).write_text("")
        return 0


def _argv(tmp_path, *extra):
    return ["--authorization", "a.json", "--status", str(tmp_path / "status"), "--out", str(tmp_path / "out"),
            *extra]


def _records(out: Path, label: str) -> dict:
    return D.sealed(out, label, LID)


def test_two_passes_seal_tagged_shards_and_bind_the_protocol(tmp_path, stage0):
    repo, gen = _repo(tmp_path, stage0), Gen(stage0[1])
    assert D.main(_argv(tmp_path), run=gen, authorize=_ok, repo=repo) == 0
    out = tmp_path / "out"
    r0, r1 = _records(out, "pass0_generation_input_g0"), _records(out, "pass1_generation_input_g0")
    assert len(r0) == 4 and set(r1) == stage0[3]                   # exactly the first pass's rejections
    assert {r["candidate_index"] for r in r0.values()} == {0} and {r["candidate_index"] for r in r1.values()} == {1}
    ex = PROTO["execution"]
    for k, call in enumerate(gen.calls):                            # every value comes from the bound protocol
        arg = dict(zip(call["cmd"], call["cmd"][1:]))
        assert arg["--action-protocol"] == ex["action_protocol"] == "v2"
        assert arg["--session-limit"] == str(ex["session_limit"]) and arg["--seed-base"] == str(ex["seed_base"] + k)
        assert (arg["--model"], arg["--revision"]) == (PROTO["teacher"]["model"], PROTO["teacher"]["revision"])
    assert (out / "pass0_generation_input_g0.complete").is_file()
    assert (tmp_path / "status").read_text().splitlines()[-1].endswith("MARKER:ALL_DONE")


def test_finished_candidates_are_sealed_while_the_generator_runs(tmp_path, stage0):
    repo = _repo(tmp_path, stage0, protocol={**PROTO, "execution": {
        **PROTO["execution"], "completion_shards": {"seal_every_records": 1, "seal_every_seconds": 1e9}}})
    seen = []

    def gen(cmd, log, tick):
        arg = dict(zip(cmd, cmd[1:]))
        with open(arg["--stream-completed"], "a") as f:
            for row in A.jsonl(arg["--tasks"]):
                f.write(json.dumps(stage0[1][row["task_id"]]) + "\n")
                f.flush()
                tick()
                seen.append(len(list((tmp_path / "out").glob("pass0_*.part*.done"))))
        return 0
    D.main(_argv(tmp_path), run=gen, authorize=_ok, repo=repo)
    assert seen[:4] == [1, 2, 3, 4]                                  # each record sealed before the next exists


def test_a_generator_killed_part_way_keeps_its_finished_candidates_and_resumes_only_the_rest(tmp_path, stage0):
    repo = _repo(tmp_path, stage0)
    dying = Gen(stage0[1], die_after=2)
    assert D.main(_argv(tmp_path), run=dying, authorize=_ok, repo=repo) == 3
    out = tmp_path / "out"
    kept = _records(out, "pass0_generation_input_g0")
    assert len(kept) == 2                                             # finished results survive the failure
    for tid, rec in kept.items():                                     # with a verifiable identity
        assert rec == {**stage0[1][tid], "candidate_index": 0, "generation_pass": rec["generation_pass"]}
        assert rec["generation_pass"]["launch_identity_sha256"] == LID
    assert not (out / "pass0_generation_input_g0.complete").exists()
    assert "ALL_DONE" not in (tmp_path / "status").read_text()         # never marked done
    assert json.loads((out / "generation_session.json").read_text())["terminal"] == "RUNS_INCOMPLETE"
    healthy = Gen(stage0[1])
    assert D.main(_argv(tmp_path), run=healthy, authorize=_ok, repo=repo) == 0
    assert set(healthy.calls[0]["tasks"]) == set(stage0[2]) - set(kept)   # only the missing candidates
    assert len(_records(out, "pass0_generation_input_g0")) == 4            # each exactly once


def test_a_stream_left_by_a_killed_driver_is_sealed_first_and_a_torn_line_ignored(tmp_path, stage0):
    repo, out = _repo(tmp_path, stage0), tmp_path / "out"
    out.mkdir()
    first, second = stage0[2][:2]
    (out / "pass0_generation_input_g0.seg0.stream.jsonl").write_text(
        json.dumps(stage0[1][first]) + "\n" + json.dumps(stage0[1][second])[:500])      # a torn last line
    gen = Gen(stage0[1])
    assert D.main(_argv(tmp_path), run=gen, authorize=_ok, repo=repo) == 0
    assert first not in gen.calls[0]["tasks"] and second in gen.calls[0]["tasks"]
    assert len(_records(out, "pass0_generation_input_g0")) == 4


def test_resume_from_secured_shards_and_refuse_a_tampered_one(tmp_path, stage0):
    repo = _repo(tmp_path, stage0)
    D.main(_argv(tmp_path), run=Gen(stage0[1], die_after=2), authorize=_ok, repo=repo)
    secured = tmp_path / "secured"
    shutil.copytree(tmp_path / "out", secured)
    fresh, gen = tmp_path / "fresh", Gen(stage0[1])
    argv = ["--authorization", "a.json", "--status", str(tmp_path / "s2"), "--out", str(fresh),
            "--resume-from", str(secured)]
    assert D.main(argv, run=gen, authorize=_ok, repo=repo) == 0 and len(gen.calls[0]["tasks"]) == 2
    shard = next(secured.glob("pass0_*.part*.jsonl"))
    shard.write_text(shard.read_text().replace("think", "thinK", 1))
    gen2 = Gen(stage0[1])
    argv[5] = str(tmp_path / "fresh2")
    assert D.main(argv, run=gen2, authorize=_ok, repo=repo) == 3 and gen2.calls == []


def test_a_p1t_authorization_runs_the_top_up_plan_and_its_inputs_only(tmp_path, stage0):
    """The plan comes from the authorization's phase: a P1t artifact reads
    plans/p1t/ and its inputs, and P1's plan path is never opened."""
    top_inputs = GOV.TOPUP_PLAN.parent / "generation_inputs"
    repo = _repo(tmp_path, stage0, plan=GOV.TOPUP_PLAN, inputs=top_inputs)
    assert not (repo / GOV.PLAN).exists() and not (repo / GOV.INPUTS_DIR).exists()
    gen = Gen(stage0[1])
    p1t = lambda p, r: {**_ok(p, r), "plan_path": str(GOV.TOPUP_PLAN)}   # noqa: E731
    assert D.main(_argv(tmp_path), run=gen, authorize=p1t, repo=repo) == 0
    assert all(str(repo / top_inputs) not in c["cmd"] for c in gen.calls)    # segments are written under --out
    rec = json.loads((tmp_path / "out/generation_session.json").read_text())
    assert rec["contract"]["plan_path"] == str(GOV.TOPUP_PLAN) and rec["terminal"] == "ALL_DONE"


def test_the_plan_is_the_one_the_authorizations_phase_names():
    auth = lambda phase: type("Auth", (), {"stage_conditions": {"phase": phase}, "authorization_id": "t"})()  # noqa: E731
    assert GOV.plan_of(auth("P1")) == GOV.PLAN and GOV.plan_of(auth("P1t")) == GOV.TOPUP_PLAN
    with pytest.raises(D.AuthorizationError, match="names no generation plan"):
        GOV.plan_of(auth("P2"))


def test_a_shard_from_another_launch_identity_is_refused(tmp_path, stage0):
    repo = _repo(tmp_path, stage0)
    D.main(_argv(tmp_path), run=Gen(stage0[1], die_after=2), authorize=_ok, repo=repo)
    other = lambda p, r: {**_ok(p, r), "launch_identity_sha256": "e" * 64}   # noqa: E731
    gen = Gen(stage0[1])
    assert D.main(_argv(tmp_path), run=gen, authorize=other, repo=repo) == 3 and gen.calls == []


def test_the_contract_refuses_before_anything_runs(tmp_path, stage0):
    repo, gen = _repo(tmp_path, stage0), Gen(stage0[1])

    def refuse(path, r):
        raise D.AuthorizationError("bound to another launch identity")
    assert D.main(_argv(tmp_path), run=gen, authorize=refuse, repo=repo) == 2 and gen.calls == []
    bad = _repo(tmp_path / "b", stage0, protocol={**PROTO, "decoding": {**PROTO["decoding"], "temperature": 1.0}})
    assert D.main(_argv(tmp_path / "b"), run=gen, authorize=_ok, repo=bad) == 2 and gen.calls == []


def test_a_verifier_the_protocol_names_gets_no_second_candidate(tmp_path, stage0):
    tasks, recs, ids, _ = stage0
    sandbox = {i: {**tasks[i], "verifier": "code_asserts"} for i in ids}
    c = {"protocol": PROTO, "tasks": sandbox}
    assert D.second_pass(c, {i: {**recs[i], "candidate_index": 0} for i in ids}) == []


def test_arm_b_places_every_session_in_its_cell_and_refuses_the_rest():
    from stages.reasoning_data import build_arm_b as AB
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    shares = AB.cell_shares(spec, "stage2.reasoning_master@v1")
    assert abs(sum(shares.values()) - 1.0) < 1e-9 and "supporting.code_algorithmic/code_contests_train" not in shares
    s = {"id": "u#t0", "unit_id": "u", "stratum": "supporting.math_new", "attributes": {"source": "math_train"}}
    assert AB.place([s], shares)[0]["stratum"] == "supporting.math_new/math_train"
    with pytest.raises(SystemExit, match="does not allocate"):
        AB.place([{**s, "attributes": {"source": "numinamath_1p5"}}], shares)

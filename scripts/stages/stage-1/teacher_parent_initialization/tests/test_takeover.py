"""A replacement attempt takes over what the lost one made durable, and nothing is retrained or re-measured: the final
weights under the completion record the trainer actually wrote (this round's resumed seed wrote
``run_completion_resume_step009701.json``), and the save-point rotation interrupted at every step. CPU, no network:
the relay is an in-memory stand-in with the real one's contract (put, get re-hashed, remove, the pointer listing).

    pytest scripts/stages/stage-1/teacher_parent_initialization/tests/test_takeover.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from stages.teacher_modes import a1_units as AU
from stages.teacher_parent_initialization import relay as R
from stages.teacher_parent_initialization import train_driver as TD

REPO = Path(__file__).resolve().parents[5]
TPI1 = REPO / "logs/stages/stage-1/teacher_parent_initialization"
EVIDENCE = TPI1 / "runs/tpi_train_20261117_180111/evidence/A_teacher_parent-s20260726"
RESUMED = "run_completion_resume_step009701.json"        #: the name this round's resumed seed actually wrote
UNIT = "A_teacher_parent-s20260726"
PREFIX = f"{R.ROOT_PREFIX}/A_teacher_parent/{UNIT}"


class PodLost(BaseException):
    """The pod disappears: nothing after this point of the attempt runs (not an Exception the driver records)."""


class FakeRelay:
    """The relay's contract in memory. ``crash(op, payload, phase)`` -> True loses the pod before (``phase ==
    'before'``) or after (``'after'``) that call takes effect."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.crash = lambda op, payload, phase: False
        self.refuse_pointer = False

    def __call__(self, op, payload, timeout_s=3600):
        where = payload.get("prefix") or (payload.get("record") or {}).get("prefix")
        self.calls.append((op, where))
        if self.crash(op, payload, "before"):
            raise PodLost(f"before {op} {where}")
        out = getattr(self, op)(payload)
        if self.crash(op, payload, "after"):
            raise PodLost(f"after {op} {where}")
        return out

    def put(self, p):
        if self.refuse_pointer and set(p["files"]) == {R.POINTER}:
            return {"verified": False, "error": "a transport failure"}
        sha = {}
        for k, path in p["files"].items():
            b = Path(path).read_bytes()
            self.files[f"{p['prefix']}/{k}"] = b
            sha[k] = hashlib.sha256(b).hexdigest()
        return {"relay": "fake", "prefix": p["prefix"], "files_sha256": sha, "verified": True, "utc": "t"}

    def get(self, p):
        rec, only = p["record"], p.get("only")
        for k, want in rec["files_sha256"].items():
            if only is not None and k not in only:
                continue
            b = self.files.get(f"{rec['prefix']}/{k}")
            if b is None or hashlib.sha256(b).hexdigest() != want:
                return {"verified": False, "error": f"{k}: absent or not the record's bytes"}
            (Path(p["dest"]) / k).parent.mkdir(parents=True, exist_ok=True)
            (Path(p["dest"]) / k).write_bytes(b)
        return {"verified": True, "from": f"fake:{rec['prefix']}"}

    def remove(self, p):
        for k in p["record"]["files_sha256"]:
            self.files.pop(f"{p['record']['prefix']}/{k}", None)
        return {"verified": True, "verified_absent": True}

    def records(self, p):
        def present(rec):
            return bool(rec.get("files_sha256")) and all(f"{rec['prefix']}/{k}" in self.files for k in rec["files_sha256"])
        recs = [{"pointer": k, "record": json.loads(v), "present": present(json.loads(v))}
                for k, v in sorted(self.files.items()) if k.startswith(p["prefix"] + "/") and k.endswith("/" + R.POINTER)]
        return {"verified": True, "records": recs}


@pytest.fixture
def relay(monkeypatch):
    fake = FakeRelay()
    monkeypatch.setattr(R, "run_online", fake)
    return fake


def plan_unit(tmp: Path) -> dict:
    plan = json.loads((TPI1 / "plans/training_plan.json").read_text())
    u = copy.deepcopy(next(x for x in plan["units"] if x["unit"] == UNIT))
    u["config"]["out_dir"] = str(tmp / "pod/train")
    u["config"]["student_path"] = str(tmp / "pod/init")
    return u


def completion_of(cfg: dict, name: str = RESUMED, **over) -> dict:
    """This round's resumed completion record, written for ``cfg`` (the test's paths change its hash)."""
    from aadistill.infrastructure.manifest import sha256_json
    rec = json.loads((EVIDENCE / "train" / RESUMED).read_text())
    return {**rec, "config_sha256": sha256_json(cfg), **over}


def save_point(out_dir: Path, step: int) -> None:
    ck = out_dir / "checkpoints" / f"step_{step:06d}"
    (ck / "model").mkdir(parents=True, exist_ok=True)
    (ck / "model" / "model.safetensors").write_bytes(f"weights at {step}".encode())
    (ck / "trainer_state.pt").write_bytes(f"adamw at {step}".encode())
    (out_dir / "checkpoints" / "latest.txt").write_text(f"step_{step:06d}\n")


def final_model(out_dir: Path, cfg: dict, name: str = RESUMED) -> Path:
    steps = cfg["schedule"]["total_steps"]
    m = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
    m.mkdir(parents=True, exist_ok=True)
    for n in AU.FINAL_FILES:
        (m / n).write_bytes(f"final {n}".encode())
    (out_dir / name).write_text(json.dumps(completion_of(cfg, name)))
    return m


# ---- the completion record, under the name the trainer wrote ----------------------------------------------------------

def test_this_rounds_resumed_completion_record_is_its_configs_finished_run():
    plan = json.loads((TPI1 / "plans/training_plan.json").read_text())
    cfg = next(x for x in plan["units"] if x["unit"] == UNIT)["config"]
    assert [p.name for p in AU.completion_records(EVIDENCE / "train")] == [RESUMED]
    path, rec = AU.completion(EVIDENCE / "train", config=cfg, steps=9701)
    assert path.name == RESUMED and rec["final_step"] == 9701
    other = next(x for x in plan["units"] if x["unit"] != UNIT)["config"]
    with pytest.raises(AU.CompletionError, match="config_sha256"):
        AU.completion(EVIDENCE / "train", config=other, steps=9701)
    with pytest.raises(AU.CompletionError, match="final_step"):
        AU.completion(EVIDENCE / "train", config=cfg, steps=9702)


def test_restore_final_takes_the_resumed_record_and_refuses_another_units(tmp_path, relay):
    u = plan_unit(tmp_path)
    cfg = u["config"]
    src = tmp_path / "old_pod"
    m = final_model(src, cfg)
    rec = relay.put({"files": {n: str(m / n) for n in AU.FINAL_FILES} | {RESUMED: str(src / RESUMED)},
                     "prefix": f"{PREFIX}/final/abc"})
    ev = tmp_path / "ev"
    (ev / "train").mkdir(parents=True)
    shutil.copyfile(src / RESUMED, ev / "train" / RESUMED)
    (ev / "final.json").write_text(json.dumps(rec))
    out = tmp_path / "pod/train"
    got = TD.restore_final(ev, out, cfg, PREFIX)
    assert got["completion"] == RESUMED and got["final_step"] == 9701 and got["found"] == "evidence"
    assert AU.state(out, [ev / "evaluation_thinking"]) == "trained"
    assert (out / "checkpoints/step_009701/model/model.safetensors").read_bytes() == b"final model.safetensors"
    #: another unit's completion is never adopted, nor silently retrained
    (ev / "train" / RESUMED).write_text(json.dumps(completion_of({**cfg, "seed": 1})))
    with pytest.raises(TD.DriverError, match="no completion of this unit's training checks out"):
        TD.restore_final(ev, tmp_path / "pod2/train", cfg, PREFIX)
    #: weights that do not hash to the record are refused
    (ev / "train" / RESUMED).write_text(json.dumps(completion_of(cfg)))
    relay.files[f"{PREFIX}/final/abc/model.safetensors"] = b"other bytes"
    with pytest.raises(TD.DriverError, match="did not restore"):
        TD.restore_final(ev, tmp_path / "pod3/train", cfg, PREFIX)


# ---- the whole attempt: final durable, save points retired, one evaluation missing -------------------------------------

def run_attempt(tmp: Path, monkeypatch, u: dict, *, resume: Path | None, out: Path, trainer=None) -> dict:
    """One driver attempt with everything but the state machine stood in for; returns the evaluations it ran."""
    from shared import study_engineering as SE
    import shared.pod.teacher_cache as TC
    from stages.teacher_modes import packed_training_plan as PTP
    from stages.teacher_modes import training_plan as TP
    from stages.teacher_parent_initialization import launch as L
    plan = {"seed_order": [u["seed"]], "units": [u]}
    (tmp / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(TD, "PLAN", tmp / "plan.json")
    monkeypatch.setattr(L.TPAuthorization, "load", classmethod(lambda cls, p: object()))
    monkeypatch.setattr(SE, "require_on_pod", lambda *a, **k: {"authorization_id": "test"})
    monkeypatch.setattr(SE, "identity_hash", lambda *a, **k: "h")
    monkeypatch.setattr(TC, "local_snapshot", lambda *a, **k: str(tmp))
    monkeypatch.setattr(TP, "stage_tokenizers", lambda snap: {})
    monkeypatch.setattr(TP, "admit", lambda *a, **k: None)
    monkeypatch.setattr(PTP, "attention_problems", lambda cfg, repo: [])
    monkeypatch.setattr(AU, "train_telemetry", lambda d: {"context_strategies": [PTP.CONTEXT_STRATEGY]})
    monkeypatch.setattr(TD, "fetch_init", lambda unit, d=None: {"stub": True})
    ran: list[str] = []

    def no_training(*a, **k):
        raise AssertionError("a unit whose final weights are durable was trained again")
    monkeypatch.setattr(AU, "train", trainer or no_training)

    def evaluate(model_dir, ev_dir, label, mode, vllm_py, attempt, work, ev=None, package_fn=None):
        assert (Path(model_dir) / "model.safetensors").is_file()
        ran.append(mode)
        Path(ev_dir).mkdir(parents=True, exist_ok=True)
        (Path(ev_dir) / "formal_report.json").write_text("{}")
        return {}
    monkeypatch.setattr(TD, "evaluate", evaluate)
    argv = ["--authorization", "a.json", "--out", str(out), "--status", str(tmp / "status.log"),
            "--unit-deadline-hours", "1"] + (["--resume", str(resume)] if resume else [])
    rc = TD.main(argv)
    doc = json.loads((out / "units.json").read_text())
    return {"rc": rc, "ran": ran, "units": doc}


def test_a_new_attempt_loads_the_durable_final_and_runs_only_the_missing_evaluation(tmp_path, monkeypatch, relay):
    """This round's seed-20260726 shape: resumed training (its completion is ``run_completion_resume_step009701``),
    final weights verified on the relay, every save point retired, thinking-mode evaluation formal, the non-thinking
    one not yet."""
    u = plan_unit(tmp_path)
    cfg = u["config"]
    src = tmp_path / "lost_pod"
    m = final_model(src, cfg)
    rec = relay.put({"files": {n: str(m / n) for n in AU.FINAL_FILES} | {RESUMED: str(src / RESUMED)},
                     "prefix": f"{PREFIX}/final/1a02ba42945d8a82"})
    secured = tmp_path / "devbox/store/out" / UNIT
    (secured / "train").mkdir(parents=True)
    shutil.copyfile(src / RESUMED, secured / "train" / RESUMED)
    (secured / "final.json").write_text(json.dumps(rec))
    shutil.copyfile(EVIDENCE / "savepoints.json", secured / "savepoints.json")       #: all eight retired
    (secured / "evaluation_thinking").mkdir()
    (secured / "evaluation_thinking" / "formal_report.json").write_text("{}")
    got = run_attempt(tmp_path, monkeypatch, u, resume=tmp_path / "devbox/store/out", out=tmp_path / "attempt2/out")
    row = got["units"]["units"][UNIT]
    assert got["rc"] == 0 and got["units"]["terminal"] == "ALL_DONE"
    assert row["state_at_start"] == "trained" and row["restored_final"]["completion"] == RESUMED
    assert got["ran"] == ["H-N"]                                          #: only the missing mode
    #: no weights move to the relay again: the one upload is the final's pointer, before anything could be retired
    assert [c for c in relay.calls if c[0] == "put"] == [("put", f"{PREFIX}/final/1a02ba42945d8a82")]
    assert not any(c[0] == "remove" for c in relay.calls)                 #: nothing left to retire


def test_a_pod_lost_after_the_final_retired_the_save_points_hands_over_through_the_relay(tmp_path, monkeypatch, relay):
    """The final weights hand off, the save points are deleted, and the pod is lost before the dev box ever pulls
    ``final.json`` or the completion record: the replacement finds both on the relay and trains nothing."""
    u = plan_unit(tmp_path)
    cfg = u["config"]
    out1 = tmp_path / "attempt1/out"

    def trainer(cfg_path, *, resume, deadline_s, log):
        final_model(Path(cfg["out_dir"]), cfg, name="run_completion.json")
        return 0
    #: the dev box's copy: a save point handed off before the run, nothing of the final
    od = Path(cfg["out_dir"])
    save_point(od, 8832)
    m = TD.SavepointMirror(UNIT, od, out1 / UNIT, 9701, PREFIX)
    assert m.mirror_once()["handoff"]["verified"]
    devbox = tmp_path / "devbox/out"
    shutil.copytree(out1, devbox)
    shutil.rmtree(od / "checkpoints")
    relay.crash = lambda op, p, phase: op == "remove" and phase == "after"
    lost = run_attempt(tmp_path, monkeypatch, u, resume=None, out=out1, trainer=trainer)
    assert lost["rc"] == 3 and "PodLost" in lost["units"]["error"]["error"]   #: nothing ran after the removal
    assert not any(k.startswith(f"{PREFIX}/savepoints/step_008832/model") for k in relay.files)   #: retired
    shutil.rmtree(tmp_path / "pod")                                       #: the pod's disk is gone
    relay.crash = lambda op, p, phase: False
    got = run_attempt(tmp_path, monkeypatch, u, resume=devbox, out=tmp_path / "attempt2/out")
    row = got["units"]["units"][UNIT]
    assert got["rc"] == 0, got["units"].get("error")
    assert row["restored_final"]["found"] == "relay listing" and row["restored_final"]["completion"] == "run_completion.json"
    assert sorted(got["ran"]) == ["H-N", "H-T"]


# ---- the save-point rotation, interrupted at every step ---------------------------------------------------------------

def _is_pointer(p):
    return set(p.get("files") or {}) == {R.POINTER}


#: (what the relay was doing when the pod was lost, before / after that call took effect)
INTERRUPTIONS = {
    "uploading the new save point": lambda op, p, ph: op == "put" and not _is_pointer(p) and ph == "before",
    "new save point uploaded, no pointer yet": lambda op, p, ph: op == "put" and not _is_pointer(p) and ph == "after",
    "pointer written, listing not confirmed": lambda op, p, ph: op == "put" and _is_pointer(p) and ph == "after",
    "handed off, old not yet removed": lambda op, p, ph: op == "remove" and ph == "before",
    "old removed, records not yet written": lambda op, p, ph: op == "remove" and ph == "after",
}


@pytest.mark.parametrize("when", sorted(INTERRUPTIONS))
def test_a_pod_lost_mid_rotation_leaves_a_save_point_the_next_attempt_restores(tmp_path, relay, when):
    out, ev = tmp_path / "pod/train", tmp_path / "pod/ev"
    m = TD.SavepointMirror(UNIT, out, ev, total_steps=9701, prefix=PREFIX)
    save_point(out, 1104)
    assert m.mirror_once()["handoff"]["verified"]
    devbox = tmp_path / "devbox/ev"                                       #: the poll's last copy: step 1104 only
    shutil.copytree(ev, devbox)
    save_point(out, 2208)
    relay.crash = INTERRUPTIONS[when]
    with pytest.raises(PodLost):
        m.mirror_once()
    relay.crash = lambda op, p, ph: False
    shutil.rmtree(tmp_path / "pod")
    got = TD.restore_savepoint(devbox, tmp_path / "pod2/train", PREFIX)
    on_relay = {t for t in ("step_001104", "step_002208")
                if f"{PREFIX}/savepoints/{t}/{R.POINTER}" in relay.files
                and f"{PREFIX}/savepoints/{t}/model/model.safetensors" in relay.files}
    assert got is not None and got["tag"] == max(on_relay)
    assert (tmp_path / "pod2/train/checkpoints" / got["tag"] / "model/model.safetensors").read_bytes() == \
        f"weights at {got['step']}".encode()
    if when == "old removed, records not yet written":
        assert got["tag"] == "step_002208" and on_relay == {"step_002208"}   #: the dev box never saw 2208's record


def test_a_save_point_that_does_not_hand_off_keeps_its_predecessor(tmp_path, relay):
    out, ev = tmp_path / "train", tmp_path / "ev"
    m = TD.SavepointMirror(UNIT, out, ev, total_steps=9701, prefix=PREFIX)
    save_point(out, 1104)
    m.mirror_once()
    save_point(out, 2208)
    relay.refuse_pointer = True
    rec = m.mirror_once()
    assert rec["verified"] and not rec["handoff"]["verified"]
    assert not any(op == "remove" for op, _ in relay.calls)
    assert f"{PREFIX}/savepoints/step_001104/model/model.safetensors" in relay.files
    relay.refuse_pointer = False
    save_point(out, 3312)
    m.mirror_once()                                                       #: the next one hands off and retires both
    assert {r["tag"] for r in m.records() if r.get("removed")} == {"step_001104", "step_002208"}
    assert not [k for k in relay.files if "/step_001104/" in k or "/step_002208/" in k]


# ---- the listing times out; the weights verified but their record did not publish (both reproduced) -------------------

FAST = {**TD.DURABILITY, "listing_attempts": 3, "listing_wait_seconds": 0, "handoff_attempts": 3,
        "handoff_wait_seconds": 0}


def _flaky_listing(relay, fails: int):
    """The relay listing times out ``fails`` times, then answers (the run_online timeout returns unverified)."""
    real, n = relay.records, {"left": fails}

    def records(p):
        if n["left"] > 0:
            n["left"] -= 1
            return {"verified": False, "error": "TimeoutExpired: list_repo_tree"}
        return real(p)
    relay.records = records


def test_a_listing_timeout_is_retried_and_never_sends_a_unit_back_to_step_0(tmp_path, relay, monkeypatch):
    monkeypatch.setattr(TD, "_sleep", lambda s: None)
    out_old, ev = tmp_path / "old/train", tmp_path / "ev"
    save_point(out_old, 2208)
    rec = relay.put({"files": {k: str(v) for k, v in TD.savepoint_files(out_old / "checkpoints/step_002208").items()},
                     "prefix": f"{PREFIX}/savepoints/step_002208"})
    rec.update(tag="step_002208", step=2208)
    TD.handoff(rec, tmp_path, f"{PREFIX}/savepoints", "t", newest=True, d=FAST)
    ev.mkdir()                                 #: the lost pod's records never reached the dev box
    _flaky_listing(relay, 2)                   #: two timeouts, then the listing answers
    got = TD.restore_savepoint(ev, tmp_path / "pod/train", PREFIX, FAST)
    assert got["step"] == 2208 and "3 attempt" in got["found"]
    #: every attempt times out and the evidence holds nothing: refused, not "no save point"
    _flaky_listing(relay, 99)
    with pytest.raises(TD.ListingUnreadable, match="refusing to train from step 0"):
        TD.restore_savepoint(ev, tmp_path / "pod2/train", PREFIX, FAST)
    #: ... and the same for the final weights
    with pytest.raises(TD.ListingUnreadable, match="no final.json"):
        TD.final_record(ev, PREFIX, FAST)
    #: a listing that WAS read and holds nothing is the only "none": a fresh unit starts at step 0
    relay.records = FakeRelay.records.__get__(relay)
    assert TD.restore_savepoint(tmp_path / "ev3", tmp_path / "pod3/train", f"{PREFIX}-other", FAST) is None
    #: with the listing down for good but a verified record in the evidence, that record is restored
    (ev / "savepoints.json").write_text(json.dumps([{**rec, "verified": True}]))
    _flaky_listing(relay, 99)
    got = TD.restore_savepoint(ev, tmp_path / "pod4/train", PREFIX, FAST)
    assert got["step"] == 2208 and "unreadable" in got["found"]


def test_an_unpublished_record_is_retried_alone_and_the_weights_are_sent_once(tmp_path, relay):
    out, ev = tmp_path / "pod/train", tmp_path / "ev"
    ev.mkdir()
    m = TD.SavepointMirror(UNIT, out, ev, 9701, PREFIX, d=FAST)
    save_point(out, 1104)
    assert m.mirror_once()["handoff"]["verified"]
    save_point(out, 2208)
    relay.refuse_pointer = True                #: the weights upload and verify; the record's own copy fails
    rec = m.mirror_once()
    assert rec["verified"] and not rec["handoff"]["verified"]
    assert f"{PREFIX}/savepoints/step_001104/model/model.safetensors" in relay.files   #: the predecessor stays
    weight_puts = lambda: [c for c in relay.calls if c == ("put", f"{PREFIX}/savepoints/step_002208")]
    sent = len(weight_puts())
    relay.refuse_pointer = False
    rec = m.mirror_once()                      #: the next tick: only the record is published, then the old one goes
    assert rec["handoff"]["verified"] and rec["handoff_tries"] == 2
    puts = [c for c in relay.calls if c[0] == "put" and c[1] == f"{PREFIX}/savepoints/step_002208"]
    assert len(puts) == sent + 1               #: one more put: the pointer, never the weights again
    assert not any(k.startswith(f"{PREFIX}/savepoints/step_001104/") for k in relay.files)
    assert m.mirror_once() is None             #: handed off: nothing left to do
    #: the final: verified weights, a record that publishes only on the third try; uploaded once
    fin = relay.put({"files": {"model.safetensors": str(out / "checkpoints/step_002208/model/model.safetensors")},
                     "prefix": f"{PREFIX}/final/abc"})
    n = {"left": 2}
    real_put = relay.put

    def put(p):
        if set(p["files"]) == {R.POINTER} and n["left"] > 0:
            n["left"] -= 1
            return {"verified": False, "error": "a transport failure"}
        return real_put(p)
    relay.put = put
    h = TD.hand_off_with_retries(fin, tmp_path, f"{PREFIX}/final", "t", newest=False, d=FAST)
    assert h["verified"] and len(h["attempts"]) == 3


def test_the_training_deadline_leaves_room_to_secure_a_2b_save_point():
    """A 2B unit's save point is about 22.6 GB and its final 8.0 GB (SOB1's structures): at the floor rates the
    reserve is hours, so a session near its end never starts a trainer it cannot secure."""
    d = TD.DURABILITY
    sp, fin = 22.6e9, 8.07e9
    reserve = TD.transfer_seconds(sp, d) + TD.transfer_seconds(fin, d) + d["closeout_reserve_seconds"]
    now = 1_800_000_000
    end = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(now + 30 * 3600))
    got = TD.training_deadline_s(20.0, end, sp, fin, d, now=now)
    assert got["seconds"] == 20 * 3600                    #: plenty of session: the unit's own bound
    end = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(now + 10 * 3600))
    got = TD.training_deadline_s(20.0, end, sp, fin, d, now=now)
    assert got["seconds"] == pytest.approx(10 * 3600 - reserve) and got["reserve_seconds"] == pytest.approx(reserve, abs=0.1)
    end = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(now + 3600))
    assert TD.training_deadline_s(20.0, end, sp, fin, d, now=now)["seconds"] <= 0
    assert TD.training_deadline_s(20.0, None, sp, fin, d, now=now)["seconds"] == 20 * 3600

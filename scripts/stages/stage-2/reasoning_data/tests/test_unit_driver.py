"""The H1 unit driver's control flow: order, durable-root state, refusals, terminal path.

The real plan is loaded (arms, seeds, schedule, frozen recipe, initialization
identity) and only what a launch would supply is replaced: a pack directory
whose identity is recorded, and the three operations that need a GPU or the
relay (restore, train, evaluate), injected as small functions that write a
synthetic checkpoint and synthetic outputs. Persistence is the real
`persist_unit`, so the durable records here are the ones a pod would leave.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stages.reasoning_data import persist_unit as PU
from stages.reasoning_data import unit_driver as U

UNIT = "A-s20260726"


@pytest.fixture(scope="module")
def real_plan():
    return U.load_plan()


def _pack(tmp_path, arm="A", expected=None):
    d = tmp_path / "packs" / arm
    d.mkdir(parents=True)
    (d / "blocks.npz").write_bytes(b"blocks-" + arm.encode())
    exp = expected or {"n_units": 3150, "n_sessions": 3312, "actual_supervised_tokens": 2960182, "n_blocks": 1459}
    (d / "ladder.json").write_text(json.dumps({"rungs": [{"target_supervised_tokens": 2_960_000, **exp}]}))
    return {"status": "recorded", "blocks_sha256": PU._sha256(d / "blocks.npz"),
            "ladder_sha256": PU._sha256(d / "ladder.json"), "expected": exp}


def _plan(real_plan, tmp_path, *, frozen=True, provisional=False):
    p = copy.deepcopy(real_plan)
    p["config"]["frozen"] = frozen
    p["schedule"]["status"] = "PROVISIONAL (test)" if provisional else "FROZEN (test)"
    p["packs"]["A"] = _pack(tmp_path)
    return p


class Calls:
    def __init__(self, tmp_path, *, fail_eval=0, fail_train=0, forbid=(), crash_after_save=0):
        self.tmp, self.log, self.fail_eval, self.fail_train, self.forbid = tmp_path, [], fail_eval, fail_train, forbid
        self.crash_after_save = crash_after_save
        self.root = tmp_path / "durable"
        self.eval_model_dirs = []
        self.jobs = []

    def restore_init(self, init_dir, ident):
        self.log.append("restore")
        init_dir.mkdir(parents=True, exist_ok=True)
        return init_dir

    def train(self, job):
        assert "train" not in self.forbid, "retrained a unit that was already trained"
        self.log.append("resume" if job.get("resume") else "train")
        self.jobs.append(job)
        assert job["recipe"]["seed"] == 20260726 and job["recipe"]["batch"]["blocks_per_step"] == 7
        if self.fail_train:
            self.fail_train -= 1
            raise RuntimeError("the trainer exited 1; see recovery.log")
        (job["attempt_dir"] / "recovery_config.json").write_text(json.dumps(job["recipe"]))
        out = Path(job["recipe"]["out_dir"])          # the trainer writes where its config says
        if self.crash_after_save:                     # the mid-run save point lands, then the pod dies
            self.crash_after_save -= 1
            save = out / "checkpoints/step_000307"
            (save / "model").mkdir(parents=True)
            (save / "trainer_state.pt").write_bytes(b"state")
            (out / "checkpoints/latest.txt").write_text("step_000307\n")
            raise RuntimeError("the pod was lost")
        if job.get("resume"):
            assert (out / "checkpoints/latest.txt").read_text().strip() == "step_000307"
        model = out / "checkpoints/step_000614/model"
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(b"\x05" * 4096)
        (model / "config.json").write_text(json.dumps({"hidden_size": 8}))
        (out / "run_manifest.json").write_text("{}")
        (out / "checkpoints/latest.txt").write_text("step_000614\n")
        return model

    def evaluate(self, job):
        assert "evaluate" not in self.forbid, "re-evaluated a scored unit"
        self.log.append("evaluate")
        # the checkpoint is durable BEFORE evaluation starts
        assert PU.completed(job["progress"].parents[1], job["unit"], "checkpoint")["persisted"] is True
        self.eval_model_dirs.append(job["model_dir"])
        if self.fail_eval:
            self.fail_eval -= 1
            raise RuntimeError("run_battery exited 1; see eval_fluid.log")
        out = job["out"]
        for name in ("generations/fluid.jsonl", "generations/fluid.summary.json", "rows/fluid.json"):
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_text(name)
        (out / "formal_report.json").write_text(json.dumps({"fluid": 0.0}))
        return {"fluid": 0.0}, {n: out / n for n in ("generations/fluid.jsonl", "generations/fluid.summary.json",
                                                      "rows/fluid.json", "formal_report.json")}

    def ops(self):
        return U.Ops(restore_init=self.restore_init, train=self.train, evaluate=self.evaluate,
                     adapter=U._qwen3_adapter, commit=lambda: "test-commit")


def _run(plan, tmp_path, calls, *extra):
    return U.run(["--run-id", "t1", "--durable-root", str(tmp_path / "durable"),
                  "--pack-root", str(tmp_path / "packs"), "--init-dir", str(tmp_path / "init"),
                  "--work", str(tmp_path / "work"), "--unit", UNIT, *extra], ops=calls.ops(), plan=plan)


def _status(tmp_path):
    return json.loads((tmp_path / "work/status.json").read_text())


def _snapshot(d):
    return {str(p.relative_to(d)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(d.rglob("*"))
            if p.is_file() and p.name != "attempts.jsonl"}


def test_a_fresh_unit_is_persisted_before_it_is_evaluated_and_reported_last(real_plan, tmp_path):
    calls = Calls(tmp_path)
    assert _run(_plan(real_plan, tmp_path), tmp_path, calls) == 0
    assert calls.log == ["restore", "train", "evaluate"]
    root = tmp_path / "durable"
    for kind in ("checkpoint", "training", "evaluation", "report"):
        assert PU.completed(root, UNIT, kind)["persisted"] is True, kind
    report = json.loads((root / UNIT / "report/report.json").read_text())
    assert report["formal"] is True and report["inputs"]["seed"] == 20260726
    assert {r["name"] for r in PU.completed(root, UNIT, "training")["files"]} >= {"recovery_config.json",
                                                                                   "trainer/run_manifest.json"}
    st = _status(tmp_path)
    assert st["terminal"] == "ALL_DONE" and st["exit_code"] == 0 and st["observers_open"] == 0


def test_an_evaluation_failure_leaves_the_checkpoint_durable_and_the_terminal_record_written(real_plan, tmp_path):
    calls = Calls(tmp_path, fail_eval=1)
    assert _run(_plan(real_plan, tmp_path), tmp_path, calls) == 1
    root = tmp_path / "durable"
    assert PU.completed(root, UNIT, "checkpoint")["persisted"] is True
    assert PU.completed(root, UNIT, "report") is None
    st = _status(tmp_path)
    assert st["terminal"] == "FAILED" and st["exit_code"] == 1 and st["units"][UNIT]["outcome"] == "failed"
    end = [a for a in U.attempts(root, UNIT) if a["event"] == "end"][-1]
    assert end["outcome"] == "failed" and end["stage"] == "evaluate" and "run_battery exited 1" in end["error"]


def test_a_restart_after_the_checkpoint_was_persisted_resumes_at_scoring_without_retraining(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path, fail_eval=1)) == 1
    again = Calls(tmp_path, forbid=("train",))
    assert _run(plan, tmp_path, again) == 0
    assert again.log == ["evaluate"]
    assert again.eval_model_dirs == [tmp_path / "durable" / UNIT / "checkpoint"]   # the re-verified durable copy
    assert PU.completed(tmp_path / "durable", UNIT, "report")["persisted"] is True


def test_a_completed_unit_is_skipped_and_only_its_evidence_comes_home(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path)) == 0
    before = _snapshot(tmp_path / "durable")
    third = Calls(tmp_path, forbid=("train", "evaluate"))
    assert _run(plan, tmp_path, third) == 0 and third.log == []
    assert _status(tmp_path)["units"][UNIT]["action"] == "skipped"
    home = tmp_path / "work" / UNIT / "home"
    assert (home / "report/report.json").is_file() and (home / "checkpoint.descriptor.json").is_file()
    assert not (home / "checkpoint").exists()            # weights stay where they are (P8.4)
    assert _snapshot(tmp_path / "durable") == before


def test_a_durable_record_of_other_inputs_stops_the_unit_and_consumes_nothing(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path, fail_eval=1)) == 1
    before = _snapshot(tmp_path / "durable")
    other = copy.deepcopy(plan)
    other["recipe"]["optim"]["lr"] = 1e-4               # another recipe -> another recipe identity
    calls = Calls(tmp_path, forbid=("train", "evaluate"))
    assert _run(other, tmp_path, calls) == 1 and calls.log == []
    assert _status(tmp_path)["units"][UNIT]["outcome"] == "identity_mismatch"
    assert _snapshot(tmp_path / "durable") == before
    assert U.attempts(tmp_path / "durable", UNIT)[-1]["outcome"] == "identity_mismatch"


def test_a_failed_training_attempt_is_retrained_from_the_initialization_as_its_replacement(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path, fail_train=1)) == 1
    calls = Calls(tmp_path)
    assert _run(plan, tmp_path, calls) == 0 and calls.log == ["restore", "train", "evaluate"]
    starts = [a for a in U.attempts(tmp_path / "durable", UNIT) if a["event"] == "start"]
    assert [a["action"] for a in starts] == ["train", "train"] and starts[1]["replacement_of"] == 1
    assert (tmp_path / "work" / UNIT / "attempt-1").is_dir()          # the failed attempt is kept


def test_a_model_trained_on_this_machine_is_persisted_rather_than_retrained(real_plan, tmp_path, monkeypatch):
    plan = _plan(real_plan, tmp_path)
    real = PU.persist_checkpoint

    def no_room(*a, **kw):
        return real(*a, **{**kw, "capacity": 1})
    monkeypatch.setattr(U.PU, "persist_checkpoint", no_room)
    assert _run(plan, tmp_path, Calls(tmp_path)) == 1                  # trained, then no room to persist
    monkeypatch.setattr(U.PU, "persist_checkpoint", real)
    calls = Calls(tmp_path, forbid=("train",))
    assert _run(plan, tmp_path, calls) == 0 and calls.log == ["evaluate"]
    assert [a["action"] for a in U.attempts(tmp_path / "durable", UNIT) if a["event"] == "start"] == \
        ["train", "persist_local"]


def test_a_provisional_schedule_is_refused_without_validation(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path, provisional=True)
    calls = Calls(tmp_path, forbid=("train", "evaluate"))
    assert _run(plan, tmp_path, calls) == 2 and calls.log == []
    st = _status(tmp_path)
    assert st["terminal"] == "REFUSED" and any("PROVISIONAL" in r for r in st["refusals"])
    assert not (tmp_path / "durable").exists()


def test_a_validation_run_may_use_the_provisional_schedule_and_never_writes_a_formal_result(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path, frozen=False, provisional=True)
    calls = Calls(tmp_path)
    assert _run(plan, tmp_path, calls, "--validation") == 0
    assert not (tmp_path / "durable" / UNIT).exists()
    report = json.loads((tmp_path / "durable/validation" / UNIT / "report/report.json").read_text())
    assert report["formal"] is False and report["inputs"]["formal"] is False


def test_the_unfrozen_config_is_refused_for_a_formal_launch(real_plan, tmp_path):
    calls = Calls(tmp_path, forbid=("train", "evaluate"))
    assert _run(_plan(real_plan, tmp_path, frozen=False), tmp_path, calls) == 2
    assert any("not frozen" in r for r in _status(tmp_path)["refusals"])


@pytest.mark.parametrize("validation", [False, True])
def test_a_pending_pack_is_refused_in_every_mode(real_plan, tmp_path, validation):
    plan = _plan(real_plan, tmp_path)
    plan["packs"]["A"] = {"status": "pending", "what": "pending: a pack not built yet", "expected": None,
                          "blocks_sha256": None, "ladder_sha256": None}
    calls = Calls(tmp_path, forbid=("train", "evaluate"))
    assert _run(plan, tmp_path, calls, *(["--validation"] if validation else [])) == 2
    assert any("pack identity pending" in r for r in _status(tmp_path)["refusals"])


def test_the_committed_config_admits_the_formal_units_only_with_their_real_packs(real_plan, tmp_path):
    """Frozen with both packs recorded (2026-10-23): a formal launch is refused
    only where the recorded packs are not, and admitted on the real ones."""
    assert real_plan["config"]["frozen"] is True
    assert {a: p["status"] for a, p in real_plan["packs"].items() if a in "AB"} == {"A": "recorded", "B": "recorded"}
    refusals = U.launch_refusals(real_plan, real_plan["units"], validation=False, pack_root=tmp_path)
    assert sorted(r.split(":")[0] for r in refusals) == ["arm A", "arm B"] and all("lacks" in r for r in refusals)
    real_root = U.REPO / "artifacts/stages/stage-2/reasoning_data/packs"
    if (real_root / "A/blocks.npz").is_file() and (real_root / "B/blocks.npz").is_file():
        assert U.launch_refusals(real_plan, real_plan["units"], validation=False, pack_root=real_root) == []


@pytest.mark.parametrize("break_it", ["blocks", "composition"])
def test_a_pack_that_is_not_its_recorded_identity_is_refused(real_plan, tmp_path, break_it):
    plan = _plan(real_plan, tmp_path)
    if break_it == "blocks":
        (tmp_path / "packs/A/blocks.npz").write_bytes(b"other")
    else:
        plan["packs"]["A"]["expected"] = {**plan["packs"]["A"]["expected"], "n_blocks": 1460}
    assert U.pack_refusals(plan["packs"]["A"], "A", tmp_path / "packs")


def test_the_derived_recipe_differs_from_the_frozen_one_only_in_the_declared_overrides(real_plan):
    sched = U.schedule_for(real_plan, "B")
    kw = dict(run_name="r", purpose="p", out_dir="o", data_dir="d", student_path="s", seed=1, sched=sched)
    d = U.derive_recipe(real_plan["recipe"], overrides=real_plan["config"]["recipe"]["overrides"], **kw)
    assert d["batch"]["blocks_per_step"] == 2 and d["schedule"]["total_steps"] == real_plan["schedule"]["steps"]
    assert d["rung"] == 2_960_000
    assert d["loss"] == real_plan["recipe"]["loss"] and d["optim"] == real_plan["recipe"]["optim"]
    # A's 7 blocks per step differ from the frozen 2: undeclared, that is refused
    kw["sched"] = U.schedule_for(real_plan, "A")
    with pytest.raises(U.DriverRefusal, match=r"outside the declared overrides: \['batch.blocks_per_step'\]"):
        U.derive_recipe(real_plan["recipe"], overrides=[o for o in real_plan["config"]["recipe"]["overrides"]
                                                        if o != "batch.blocks_per_step"], **kw)


def test_the_schedule_record_must_agree_with_its_rule(real_plan):
    assert U.lr_rule(625) == {"warmup_steps": 31, "eval_every": 78, "save_every": 312}
    bad = copy.deepcopy(real_plan)
    bad["schedule"]["warmup_steps"] = 30
    with pytest.raises(U.DriverRefusal, match="disagrees with its rule"):
        U.schedule_for(bad, "A")


def test_an_interrupted_attempt_resumes_from_its_durable_save_point(real_plan, tmp_path):
    """The pod dies after the mid-run save: the next attempt continues THAT
    attempt's run -- its config, its output directory, --resume -- instead of
    retraining from the initialization, and the save points go once the final
    checkpoint is durable."""
    plan = _plan(real_plan, tmp_path)
    first = Calls(tmp_path, crash_after_save=1)
    assert _run(plan, tmp_path, first) == 1 and first.log == ["restore", "train"]
    train_dir = tmp_path / "durable" / UNIT / "train" / "a1"
    assert (train_dir / "checkpoints/step_000307/trainer_state.pt").is_file()
    second = Calls(tmp_path)
    assert _run(plan, tmp_path, second) == 0
    assert second.log == ["resume", "evaluate"]                       # no restore, no fresh training
    job = second.jobs[0]
    assert job["resume"] is True and job["recipe"] == json.loads((train_dir / "recovery_config.json").read_text())
    assert job["recipe"]["out_dir"] == str(train_dir)
    starts = [a for a in U.attempts(tmp_path / "durable", UNIT) if a["event"] == "start"]
    assert [a["action"] for a in starts] == ["train", "resume"]
    assert starts[1]["replacement_of"] == 1 and starts[1]["resumed_from"] == str(train_dir)
    assert not (train_dir / "checkpoints").exists() and (train_dir / "train_inputs.json").is_file()
    assert PU.completed(tmp_path / "durable", UNIT, "report")["persisted"] is True


def test_a_save_point_for_other_inputs_is_never_resumed(real_plan, tmp_path):
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path, crash_after_save=1)) == 1
    meta = tmp_path / "durable" / UNIT / "train/a1/train_inputs.json"
    doc = json.loads(meta.read_text())
    doc["inputs"]["seed"] = 1
    meta.write_text(json.dumps(doc))
    calls = Calls(tmp_path)
    assert _run(plan, tmp_path, calls) == 0 and calls.log == ["restore", "train", "evaluate"]


def test_the_token_is_found_where_a_detached_pod_driver_can_read_it(tmp_path):
    """Setup writes the token to the pod's workspace and exports HF_TOKEN only in
    its own shell; the detached driver must still find it (P1g attempt 1)."""
    ws, home = tmp_path / "ws_token", tmp_path / "home_token"
    ws.write_text("from-workspace\n")
    assert U.hf_token(env={}, files=(ws, home)) == "from-workspace"
    assert U.hf_token(env={"HF_TOKEN": "from-env"}, files=(ws, home)) == "from-env"
    home.write_text("from-home\n")
    assert U.hf_token(env={}, files=(tmp_path / "absent", home)) == "from-home"
    with pytest.raises(FileNotFoundError, match="no HF token"):
        U.hf_token(env={}, files=(tmp_path / "absent",))
    assert Path("/workspace/hf/token") in U.TOKEN_FILES


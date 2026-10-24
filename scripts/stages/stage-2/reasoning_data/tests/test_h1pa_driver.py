"""H1-PA's driver: E1's recipe kept, nucleus_v1 bound, PCA-Original never retrained, durable-root state.

The real study and preregistration are loaded; only what a launch supplies is
replaced -- synthetic packs whose identities the plan then records, a staged
PCA checkpoint, and the GPU operations (restore, train, evaluate), injected as
small functions. Persistence is the real `persist_unit`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from stages.reasoning_data import h1pa_driver as D
from stages.reasoning_data import persist_unit as PU
from stages.reasoning_data import unit_driver as UD


@pytest.fixture(scope="module")
def real_plan():
    return D.load_plan()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _plan(real_plan, tmp_path, *, frozen=True):
    p = copy.deepcopy(real_plan)
    p["study"]["frozen"] = frozen
    for arm, (n, sup) in {"D4-Original": (1944, 2960507), "D4-Augmented": (2468, 5923021)}.items():
        key = p["study"]["arms"][arm]["dataset"]
        d = tmp_path / "packs" / key
        d.mkdir(parents=True)
        (d / "blocks.npz").write_bytes(f"blocks-{key}".encode())
        (d / "audit.jsonl").write_text("{}\n")
        (d / "ladder.json").write_text(json.dumps({"rungs": [{"target_supervised_tokens": sup, "n_blocks": n}]}))
        p["study"]["datasets"][key].update(blocks_sha256=_sha(d / "blocks.npz"), ladder_sha256=_sha(d / "ladder.json"))
    for seed in p["study"]["seeds"]:
        c = tmp_path / "pca" / str(seed)
        c.mkdir(parents=True)
        (c / "model.safetensors").write_bytes(f"pca-{seed}".encode())
        p["study"]["arms"]["PCA-Original"]["checkpoints"][str(seed)]["model_safetensors_sha256"] = \
            _sha(c / "model.safetensors")
    return p


class Calls:
    def __init__(self, tmp_path, *, forbid=(), crash_after_save=0, no_decoding=False):
        self.tmp, self.log, self.forbid, self.jobs, self.evals = tmp_path, [], set(forbid), [], []
        self.crash_after_save, self.no_decoding = crash_after_save, no_decoding

    def restore_init(self, init_dir, ident):
        assert "restore" not in self.forbid
        self.log.append("restore")
        init_dir.mkdir(parents=True, exist_ok=True)
        return init_dir

    def stage_checkpoint(self, root, seed, sha):
        self.log.append(f"stage-{seed}")
        return D.stage_checkpoint(root, seed, sha)

    def train(self, job):
        assert "train" not in self.forbid, "trained a unit that must not be trained"
        self.log.append("resume" if job.get("resume") else "train")
        self.jobs.append(job)
        out = Path(job["recipe"]["out_dir"])
        if self.crash_after_save:
            self.crash_after_save -= 1
            save = out / "checkpoints/step_001851"
            (save / "model").mkdir(parents=True)
            (save / "trainer_state.pt").write_bytes(b"state")
            (out / "checkpoints/latest.txt").write_text("step_001851\n")
            raise RuntimeError("the pod was lost")
        model = out / "checkpoints/final/model"
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(b"\x05" * 4096)
        (model / "config.json").write_text(json.dumps({"hidden_size": 8}))
        (out / "run_manifest.json").write_text("{}")
        return model

    def evaluate(self, job):
        assert "evaluate" not in self.forbid, "re-evaluated a scored unit"
        self.log.append("evaluate")
        self.evals.append(job)
        dec = job["evaluation"]["decoding"]
        assert dec["path"].endswith("nucleus_v1.json") and len(dec["sha256"]) == 64
        out = job["out"]
        (out / "generations").mkdir(parents=True, exist_ok=True)
        (out / "generations/fluid.jsonl").write_text("{}\n")
        summary = {} if self.no_decoding else {"identity": {"decoding_protocol": {"sha256": dec["sha256"]}}}
        (out / "generations/fluid.summary.json").write_text(json.dumps(summary))
        bad = D.generation_refusals(out / "generations", dec["sha256"])
        if bad:
            raise D.IdentityMismatch(f"generations not under the preregistered decoding protocol: {bad}")
        (out / "formal_report.json").write_text(json.dumps({"fluid": 0.0}))
        return {"fluid": 0.0}, {n: out / n for n in ("generations/fluid.jsonl", "generations/fluid.summary.json",
                                                      "formal_report.json")}

    def ops(self):
        return D.Ops(restore_init=self.restore_init, stage_checkpoint=self.stage_checkpoint, train=self.train,
                     evaluate=self.evaluate, adapter=UD._qwen3_adapter, commit=lambda: "test-commit")


def _run(plan, tmp_path, calls, *units, volume=None):
    vol = ["--volume-mount", str(tmp_path / "durable"), "--volume-bytes", str(volume)] if volume else []
    return D.run(["--run-id", "t1", "--durable-root", str(tmp_path / "durable"), "--pack-root", str(tmp_path / "packs"),
                  "--checkpoint-root", str(tmp_path / "pca"), "--init-dir", str(tmp_path / "init"),
                  "--work", str(tmp_path / "work"), *[a for u in units for a in ("--unit", u)], *vol],
                 ops=calls.ops(), plan=plan)


def _status(tmp_path):
    return json.loads((tmp_path / "work/status.json").read_text())


def test_e1s_builder_rule_reproduces_e1s_own_schedule(real_plan):
    e1 = real_plan["recipes"][20260726]
    assert D.e1_schedule(1944, epochs=3, batch_size=2) == {
        "total_steps": e1["schedule"]["total_steps"], "warmup_steps": e1["schedule"]["warmup_steps"],
        "eval_every": e1["intervals"]["eval_every"], "save_every": e1["checkpoint"]["save_every"]}
    assert D.e1_schedule(2468, epochs=3, batch_size=2)["total_steps"] == 3702


def test_d4_original_trains_e1s_config_unchanged(real_plan, tmp_path):
    calls = Calls(tmp_path)
    assert _run(_plan(real_plan, tmp_path), tmp_path, calls, "D4-Original-s20260801") == 0
    r, e1 = calls.jobs[0]["recipe"], real_plan["recipes"][20260801]
    a, b = UD._flat(e1), UD._flat(r)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= set(D.WHERE) | set(D.PINS)
    assert r["schedule"]["total_steps"] == 2916 and r["batch"] == {"blocks_per_step": 2, "micro_blocks": 1}
    assert r["tokenizer_source"] == D.TOKENIZER_SOURCE and r["seed"] == 20260801
    assert r["tokenizer_sha256"] == D.PINS["tokenizer_sha256"]
    assert calls.log == ["restore", "train", "evaluate"]
    for kind in ("checkpoint", "training", "evaluation", "report"):
        assert PU.completed(tmp_path / "durable", "D4-Original-s20260801", kind)["persisted"] is True


def test_d4_augmented_changes_only_the_builder_schedule(real_plan, tmp_path):
    calls = Calls(tmp_path)
    assert _run(_plan(real_plan, tmp_path), tmp_path, calls, "D4-Augmented-s20260726") == 0
    r, e1 = calls.jobs[0]["recipe"], real_plan["recipes"][20260726]
    a, b = UD._flat(e1), UD._flat(r)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= set(D.WHERE) | set(D.SCHEDULE_KEYS) | set(D.PINS)
    assert (r["schedule"]["total_steps"], r["schedule"]["warmup_steps"], r["rung"]) == (3702, 185, 5923021)
    assert r["optim"] == e1["optim"] and r["loss"] == e1["loss"] and r["batch"] == e1["batch"]


def test_a_schedule_that_disagrees_with_the_study_refuses(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    p["study"]["arms"]["D4-Augmented"]["schedule"]["optimizer_steps"] = 2916
    assert _run(p, tmp_path, Calls(tmp_path), "D4-Augmented-s20260726") == 1
    assert "not the study's record" in _status(tmp_path)["units"]["D4-Augmented-s20260726"]["error"]


def test_pca_original_is_evaluated_never_trained_and_its_rerun_is_skipped(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    calls = Calls(tmp_path, forbid=("train", "restore"))
    assert _run(p, tmp_path, calls, "PCA-Original-s20260801") == 0
    assert calls.log == ["stage-20260801", "evaluate"]
    root = tmp_path / "durable"
    assert PU.completed(root, "PCA-Original-s20260801", "checkpoint") is None      # weights never persisted again
    rep = json.loads((root / "PCA-Original-s20260801/report/report.json").read_text())
    assert rep["execution"] == "evaluate" and rep["checkpoint"]["persisted_again"] is False
    assert json.loads((root / "PCA-Original-s20260801/evaluation/unit_inputs.json").read_text()) == rep["inputs"]
    again = Calls(tmp_path, forbid=("train", "restore", "evaluate"))
    assert _run(p, tmp_path, again, "PCA-Original-s20260801") == 0
    assert _status(tmp_path)["units"]["PCA-Original-s20260801"]["action"] == "skipped"


def test_a_pca_checkpoint_that_is_not_e1s_stops_the_unit(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    (tmp_path / "pca/20260726/model.safetensors").write_bytes(b"something else")
    assert _run(p, tmp_path, Calls(tmp_path), "PCA-Original-s20260726") == 1
    assert _status(tmp_path)["units"]["PCA-Original-s20260726"]["outcome"] == "identity_mismatch"


def test_an_interrupted_training_resumes_from_its_save_point(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    assert _run(p, tmp_path, Calls(tmp_path, crash_after_save=1), "D4-Augmented-s20260801") == 1
    calls = Calls(tmp_path, forbid=("restore",))
    assert _run(p, tmp_path, calls, "D4-Augmented-s20260801") == 0
    assert calls.log == ["resume", "evaluate"]
    rows = UD.attempts(tmp_path / "durable", "D4-Augmented-s20260801")
    assert [r["action"] for r in rows if r["event"] == "start"] == ["train", "resume"]


def test_no_decoding_protocol_refuses_before_any_unit(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    p["prereg"]["evaluation"]["decoding"] = None
    assert _run(p, tmp_path, Calls(tmp_path, forbid=("train", "restore", "evaluate")), "D4-Original-s20260726") == 2
    st = _status(tmp_path)
    assert st["terminal"] == "REFUSED" and any("never evaluates greedily" in r for r in st["refusals"])


def test_a_generation_without_the_protocol_is_never_persisted(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    assert _run(p, tmp_path, Calls(tmp_path, no_decoding=True), "PCA-Original-s20260726") == 1
    assert PU.completed(tmp_path / "durable", "PCA-Original-s20260726", "evaluation") is None


def test_a_formal_launch_of_an_unfrozen_study_refuses(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path, frozen=False)
    assert _run(p, tmp_path, Calls(tmp_path, forbid=("train", "restore", "evaluate")), "D4-Original-s20260726") == 2


def test_generations_already_on_the_volume_under_another_protocol_refuse(tmp_path):
    g = tmp_path / "progress"
    g.mkdir()
    (g / "fluid_reasoning_720__dev_v1.summary.json").write_text(json.dumps({"identity": {"sampling": {"temperature": 0}}}))
    assert D.generation_refusals(g, "a" * 64) and not D.generation_refusals(tmp_path / "empty", "a" * 64)


def test_a_training_that_does_not_fit_the_volume_is_refused_before_it_starts(real_plan, tmp_path):
    p = _plan(real_plan, tmp_path)
    need = D.training_need(p)
    calls = Calls(tmp_path, forbid=("train", "restore"))
    assert _run(p, tmp_path, calls, "D4-Original-s20260726", volume=need // 2) == 1
    st = _status(tmp_path)
    assert st["units"]["D4-Original-s20260726"]["outcome"] == "capacity_refused"
    assert st["volume_probe"]["read_back_ok"] is True


def test_room_is_counted_from_the_provider_size_less_the_bytes_on_the_mount(real_plan, tmp_path):
    m = tmp_path / "mount"
    m.mkdir()
    assert D.free_bytes(m, 1000) == 1000
    (m / "x.bin").write_bytes(b"\0" * 300)
    assert D.free_bytes(m, 1000) == 700 and D.free_bytes(None, None) is None
    p = _plan(real_plan, tmp_path)
    calls = Calls(tmp_path)
    assert _run(p, tmp_path, calls, "D4-Original-s20260726", volume=10 * D.training_need(p)) == 0
    rep = json.loads((tmp_path / "durable/D4-Original-s20260726/report/report.json").read_text())
    assert set(rep["seconds"]) == {"train", "evaluate"}


def test_the_driver_is_formal_only(real_plan, tmp_path):
    with pytest.raises(SystemExit):
        D.run(["--run-id", "t", "--durable-root", str(tmp_path), "--validation"], ops=Calls(tmp_path).ops(),
              plan=_plan(real_plan, tmp_path))


def test_every_derived_recipe_meets_the_trainers_tokenizer_contract(real_plan, tmp_path, monkeypatch):
    # h1pav_20261024_191336: E1's 2.96M configs pin no tokenizer identity and the trainer refused X1;
    # run the trainer's own contract on the fields it reads (cfg.tokenizer_source, cfg.tokenizer_sha256)
    from aadistill.models import tokenizer_contract as TC
    from stages.reasoning_data import h1pa_validation as V
    src = tmp_path / "tok"
    src.mkdir()
    (src / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(TC, "tokenizer_hash", lambda _t: D.PINS["tokenizer_sha256"])
    recipes = []
    for u in (u for u in real_plan["units"] if u["execution"] == "train"):
        sc = real_plan["study"]["arms"][u["arm"]]["schedule"]
        sched = None if u["arm"] == "D4-Original" else {
            "rung": 1, "total_steps": sc["optimizer_steps"], "warmup_steps": sc["warmup_steps"],
            "eval_every": sc["eval_every"], "save_every": sc["save_every"]}
        recipes.append(D.derive_recipe(real_plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-",
                                       student_path="-"))
    recipes += [V.validation_recipe(real_plan, arm, steps=24, save_every=12, out_dir=tmp_path, data_dir=tmp_path,
                                    init=tmp_path, name="n") for arm in D.TRAINED]
    assert len(recipes) == 6
    for cfg in recipes:
        assert cfg["tokenizer_source"] == D.TOKENIZER_SOURCE
        TC.resolve_training_tokenizer(student_path=tmp_path / "student", tokenizer_source=src,
                                      expected_sha256=cfg.get("tokenizer_sha256"),
                                      loader=lambda _p: type("Tok", (), {"get_vocab": lambda self: {}})())
    e1 = copy.deepcopy(real_plan)
    e1["recipes"][20260726]["tokenizer_sha256"] = "0" * 64
    with pytest.raises(D.DriverRefusal, match="pins"):
        V.validation_recipe(e1, "D4-Augmented", steps=24, save_every=12, out_dir=tmp_path, data_dir=tmp_path,
                            init=tmp_path, name="n")


@pytest.mark.parametrize("unit", ["PCA-Original-s20260726", "D4-Original-s20260801", "D4-Augmented-s20260726"])
def test_the_analysis_expects_exactly_the_inputs_the_driver_binds(real_plan, tmp_path, unit):
    # h1pa_analysis refuses a unit whose report inputs are not what it derives from the frozen study alone
    from stages.reasoning_data import h1pa_analysis as AN
    plan = _plan(real_plan, tmp_path)
    assert _run(plan, tmp_path, Calls(tmp_path), unit) == 0
    rep = json.loads((tmp_path / "durable" / unit / "report/report.json").read_text())
    u = next(x for x in plan["units"] if x["unit"] == unit)
    assert rep["inputs"] == json.loads(json.dumps(AN.expected_inputs(plan, u)))

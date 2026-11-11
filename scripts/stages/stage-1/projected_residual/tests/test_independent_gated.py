"""The independent-gated study at $0: its plan, its recovery driver's hooks, its validation's host rehearsal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.projected_residual import independent_gated_validation as IGV  # noqa: E402
from stages.projected_residual import recovery_driver as RD  # noqa: E402
from stages.reasoning_data import h1pa_driver as HD  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402
from shared.validation import engine_admission as EA  # noqa: E402

DESIGN = json.loads((REPO / "configs/stages/stage-1/projected_residual/independent_gated_design.json").read_text())


def test_the_design_is_the_real_model_and_binds_its_inputs():
    assert DESIGN["architecture"]["num_parameters"] == 1_123_227_136
    assert DESIGN["architecture"]["trainable_parameters"] == 734_270_976
    assert UD._sha(REPO / DESIGN["base_study"]["path"]) == DESIGN["base_study"]["sha256"]
    sel = REPO / DESIGN["selection"]["path"]
    assert UD._sha(sel) == DESIGN["selection"]["file_sha256"]
    ctl = DESIGN["initializations"]["D4-Augmented-topk200"]
    assert UD._sha(REPO / ctl["record"]) == ctl["sha256"]


@pytest.mark.parametrize("arm", ["D4-Augmented-topk200", "Projected-1123M-topk200"])
def test_each_arm_differs_from_d4_augmented_only_by_the_declared_protocol(arm):
    plan = RD.load_plan(arm, require_initialization=False)
    got = RD.derived_recipe(arm, 20260726, run_name="r", purpose="p", out_dir="o", data_dir="d",
                            student_path="s", pack_root=None, plan=plan)
    h1 = HD.load_plan()
    u = {"unit": "D4-Augmented-s20260726", "arm": "D4-Augmented", "seed": 20260726, "execution": "train"}
    rec = h1["study"]["arms"]["D4-Augmented"]["schedule"]
    sched = {"rung": 5923021, **HD.e1_schedule(rec["packed_sequences"], epochs=3, batch_size=2)}
    ref = HD.derive_recipe(h1, u, sched, run_name="r", purpose="p", out_dir="o", data_dir="d", student_path="s")
    a, b = UD._flat(ref), UD._flat(got)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    declared = {"loss.kd_scope", "loss.kd_support.support_id", "loss.kd_support.top_k",
                "loss.exclude_template_prefill.generation_prompt", "loss.exclude_template_prefill.prefill",
                "checkpoint.bind_identity"}
    if arm.startswith("Projected"):
        declared |= {"trainable_patterns", "monitor.every", "monitor.groups.gates", "monitor.groups.projections"}
    assert set(diff) == declared
    # K is resolved from the forward baseline, and the schedule is D4-Augmented's
    assert got["loss"]["kd_support"] == {"support_id": "reference_topk_tail_v1", "top_k": 200}
    assert got["schedule"]["total_steps"] == 3702 == ref["schedule"]["total_steps"]


def _variant(tmp_path, *, teacher=None, design_seeds=None, prereg_seeds=None):
    d, pr = json.loads(json.dumps(DESIGN)), json.loads(RD.PREREG.read_text())
    if teacher is not None:
        d["teacher"] = teacher
    if design_seeds is not None:
        d["seeds"] = design_seeds
    if prereg_seeds is not None:
        pr["seeds"] = prereg_seeds
    (tmp_path / "design.json").write_text(json.dumps(d))
    (tmp_path / "prereg.json").write_text(json.dumps(pr))
    return tmp_path / "design.json", tmp_path / "prereg.json"


DECIDED = {"status": "decided", "model": "Qwen/Qwen3-4B-Thinking-2507",
           "revision": "768f209d9ea81521153ed38c47d515654e938aea", "dtype": "bfloat16"}


def test_no_formal_unit_while_the_teacher_is_pending():
    assert DESIGN["teacher"]["status"] == "pending" and "model" not in DESIGN["teacher"]
    for arm in DESIGN["arms"]:
        with pytest.raises(RD.TeacherPending, match="pending"):
            RD.load_plan(arm)
    with pytest.raises(RD.TeacherPending):
        RD.formal_teacher(DESIGN)
    #: teacher-independent engineering runs on the declared stand-in, and builds no unit
    assert RD.engineering_teacher(DESIGN) is DESIGN["engineering_teacher"]
    assert RD.load_plan("D4-Augmented-topk200", require_initialization=False)["units"] == []


def test_a_decided_teacher_builds_the_control_and_the_projected_arm_still_needs_its_init(tmp_path):
    design, prereg = _variant(tmp_path, teacher=DECIDED)
    with pytest.raises(HD.DriverRefusal, match="initialization record"):
        RD.load_plan("Projected-1123M-topk200", design, prereg)
    plan = RD.load_plan("D4-Augmented-topk200", design, prereg)
    assert [u["unit"] for u in plan["units"]] == ["D4-Augmented-topk200-s20260726", "D4-Augmented-topk200-s20260801"]
    with pytest.raises(HD.DriverRefusal, match="not an arm"):
        RD.load_plan("D4-Augmented", design, prereg)


@pytest.mark.parametrize("design_seeds,prereg_seeds", [
    (None, [20260726, 20260802]),          # design == base, the preregistration differs
    ([20260726, 20260802], None),          # base == preregistration, the design differs
    ([20260801, 20260726], [20260801, 20260726])])   # design == preregistration, the base differs
def test_the_three_seed_lists_must_be_identical(tmp_path, design_seeds, prereg_seeds):
    design, prereg = _variant(tmp_path, teacher=DECIDED, design_seeds=design_seeds, prereg_seeds=prereg_seeds)
    with pytest.raises(HD.DriverRefusal, match="seeds differ"):
        RD.load_plan("D4-Augmented-topk200", design, prereg)


def test_the_engine_probe_is_engineering_text_isolated_from_the_evaluation_set():
    from shared import eval_sets as ES
    probe = json.loads(RD.ADMISSION_CONFIG.read_text())["probe"]
    assert "items" not in probe and "eval_set_id" not in probe and len(probe["texts"]) >= 4
    ES.require_disjoint([{"id": t["id"], "prompt_text": t["text"]} for t in probe["texts"]])
    from huggingface_hub.constants import HF_HUB_CACHE
    t = probe["tokenizer"]
    #: the tokenizer files only (a host's snapshot may lack the weights `local_snapshot` also asks for)
    tok = Path(HF_HUB_CACHE) / f"models--{t['model'].replace('/', '--')}" / "snapshots" / t["revision"]
    if not (tok / "tokenizer.json").is_file():
        pytest.skip("the probe tokenizer is not cached on this host")
    rows, ids = EA.probe_ids(probe, tok)
    assert all(r["token_ids_sha256"] == r["expected_sha256"] for r in rows)
    assert sum(map(len, ids)) == sum(t["tokens"] for t in probe["texts"])


def test_the_projected_evaluation_runs_only_after_the_engine_is_admitted(tmp_path):
    job = {"unit": "u", "model_dir": tmp_path, "out": tmp_path / "attempt" / "evaluation"}
    job["out"].parent.mkdir(parents=True)
    ran = []
    fail = lambda m, w: {"status": "FAIL", "problems": ["p99 too large"]}  # noqa: E731
    with pytest.raises(UD.IdentityMismatch, match="not admitted"):
        RD.admitted_evaluate(job, admit=fail, evaluate=lambda j: ran.append(j) or ({}, {}))
    assert ran == [] and json.loads((job["out"].parent / "engine_admission.json").read_text())["status"] == "FAIL"
    ok = lambda m, w: {"status": "PASS", "problems": []}  # noqa: E731
    rep, files = RD.admitted_evaluate(job, admit=ok, evaluate=lambda j: ({"r": 1}, {"a": tmp_path / "a"}))
    assert rep == {"r": 1} and "engine_admission.json" in files


def _ref_and_result(engine_noise, base_noise=0.05, n=200):
    import random
    rnd = random.Random(0)
    ids = list(range(n + 1))
    fp32 = [-rnd.random() * 3 for _ in range(n)]
    bf16 = [x + rnd.uniform(-base_noise, base_noise) for x in fp32]
    ref = {"prompts": [{"id": "p0", "positions": n, "fp32_realized": fp32, "bf16_realized": bf16,
                        "fp32_argmax": ids[1:], "fp32_margin": [2.0] * n}]}
    rows = [None] + [[[ids[i], bf16[i - 1] + rnd.uniform(-engine_noise, engine_noise), 1]] for i in range(1, n + 1)]
    return ref, {"requests": [{"id": "p0", "prompt_token_ids": ids, "prompt_logprobs": rows, "outputs": []}]}


def test_engine_admission_is_judged_against_the_measured_bf16_baseline():
    crit = {"factor": 2.0, "confident_margin_nats": 1.0, "confident_top1_agreement_min": 0.98,
            "confident_positions_min": 50}
    ref, res = _ref_and_result(engine_noise=0.05)
    v = EA.compare(ref, res, crit)
    assert v["status"] == "PASS" and v["confident_top1_agreement"] == 1.0
    ref, res = _ref_and_result(engine_noise=0.5)
    v = EA.compare(ref, res, crit)
    assert v["status"] == "FAIL" and any("exceeds 2.0 x the bf16 baseline" in p for p in v["problems"])
    assert EA.compare(ref, None, crit)["status"] == "FAIL"


def test_the_validation_host_rehearsal_runs_the_real_entry_point(tmp_path):
    """The driver as the session runs it: a parent process that never imports torch."""
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}:{REPO / 'scripts/stages/stage-1'}"}
    r = subprocess.run([sys.executable, str(Path(IGV.__file__)), "--run-id", "host", "--out", str(tmp_path / "ev"),
                        "--work", str(tmp_path / "work"), "--tiny"], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=900)
    rep = json.loads((tmp_path / "ev" / "report.json").read_text())
    st = {k: v["status"] for k, v in rep["checks"].items()}
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert rep["status"] == "PASS", st
    assert all(st[k] == "NOT_RUN" for k in IGV.GPU_ONLY)
    t = json.loads((tmp_path / "ev" / "torch.json").read_text())["checks"]
    assert t["resume"]["killed_after_save"] is not None and t["resume"]["bit_identical"]
    assert t["smoke_projected"]["target_policy"]["train"]["removed_positions"] > 0
    assert t["smoke_projected"]["first_param_stats"]["gates"]["grad_norm"] > 0
    assert json.loads((tmp_path / "ev" / "isolation.json").read_text())["parent_imported_torch"] is False
    assert not (tmp_path / "work" / "init").exists()       # weights live only under --work and are removed


def test_a_resume_on_a_new_resource_restores_the_initialization_first(tmp_path):
    calls = []
    init = tmp_path / "init"
    restore = lambda d, ident: calls.append(("restore", d, ident)) or d.mkdir(parents=True, exist_ok=True)  # noqa: E731
    train = lambda job: calls.append(("train", job.get("resume"))) or tmp_path  # noqa: E731
    run = RD.train_restoring_on_resume(restore, {"artifact_digest": "x"}, train=train)
    run({"recipe": {"student_path": str(init)}, "resume": True})
    assert calls[0][0] == "restore" and calls[-1] == ("train", True)
    calls.clear()
    (init / "model.safetensors").write_bytes(b"0")
    run({"recipe": {"student_path": str(init)}, "resume": True})
    run({"recipe": {"student_path": str(init)}})
    assert [c[0] for c in calls] == ["train", "train"]

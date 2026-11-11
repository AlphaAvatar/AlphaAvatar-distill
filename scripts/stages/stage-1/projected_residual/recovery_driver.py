#!/usr/bin/env python3
"""On-pod driver of the independent-gated study's recovery units: one arm, both seeds, then eval-set-1.

    cd /workspace/aad && PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-1/projected_residual/recovery_driver.py --arm <arm> \\
        --run-id <run> --durable-root <volume>/independent_gated --pack-root <dir> --init-dir <dir> \\
        --volume-mount <volume> --volume-bytes <the provider's size> --status <file> [--unit ...]

H1-PA's unit driver, unchanged (`h1pa_driver.run`): state, resume, retry,
persistence, capacity accounting, the nucleus_v1 evaluation of every item of
eval-set-1 and the terminal path are its own. A unit whose checkpoint is
durable and scored is skipped; a trained one resumes at scoring; an interrupted
training continues from its last save point; a record of other inputs stops the
unit, preserved. What this module supplies is the PLAN (`load_plan`), the
initialization's restore and, for the projected arm, the engine admission
before its evaluation:

* the study is H1-PA's frozen record with ONE arm, a copy of its D4-Augmented
  arm (dataset, recipe, schedule) under this study's arm name, the design's
  initialization for that arm, and the design's declared protocol change
  (`recipe_overrides`, `arm_overrides`) as the only further recipe keys allowed
  to differ (`h1pa_driver.derive_recipe`);
* the KD support is RESOLVED from the forward baseline
  (`shared.forward_baseline.selected().distribution_support`), never retyped;
* the projected student's evaluation runs through vLLM's out-of-tree model,
  whose general plugin this process puts on every child's PYTHONPATH, and only
  after `engine_admission` passes on that checkpoint.

The TEACHER is the maintainer's decision after Research A (design :: teacher):
while it is pending, `load_plan` builds no formal unit and `formal_teacher`
refuses, so no formal initialization, recovery or matched control runs on a
provisional teacher. Teacher-independent engineering reads `engineering_teacher`
and the recipe side only (``require_initialization=False``).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import h1pa_driver as HD  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402

DESIGN = REPO / "configs/stages/stage-1/projected_residual/independent_gated_design.json"
PREREG = REPO / "logs/stages/stage-1/projected_residual/plans/independent_gated_preregistration.json"
PROJECTED = "Projected-1123M-topk200"
#: the engine admission's probe and criteria have one owner, the validation's check config
ADMISSION_CONFIG = REPO / "configs/stages/stage-1/projected_residual/independent_gated_validation.json"


class TeacherPending(HD.DriverRefusal):
    """Research B's teacher is not decided: nothing formal may run (design :: teacher)."""


def formal_teacher(design: dict) -> dict:
    """The decided teacher (model, revision, dtype, template kwargs); refuses while it is pending."""
    t = design["teacher"]
    if t.get("status") != "decided" or not t.get("model") or not t.get("revision"):
        raise TeacherPending(f"{design['study_id']}: the teacher is {t.get('status')!r}, decided by the maintainer "
                             "after Research A; no formal initialization, recovery or matched control runs before it")
    return t


def engineering_teacher(design: dict) -> dict:
    """The teacher a teacher-independent engineering check runs: the decided one, else the declared stand-in."""
    t = design["teacher"]
    return t if t.get("status") == "decided" else design["engineering_teacher"]


def resolved_kd_support() -> dict:
    from shared.forward_baseline import selected
    return selected(REPO).distribution_support.as_dict()


def load_plan(arm: str, design_path: Path = DESIGN, prereg_path: Path = PREREG, *,
              require_initialization: bool = True) -> dict:
    """`h1pa_driver`'s plan shape for one arm of this study.

    ``require_initialization=False`` builds the recipe side only (what
    `h1pa_driver.derive_recipe` reads), for the engineering validation, whose
    student is its own materialization: the plan then carries no ``init`` and no
    unit can run from it. A formal plan (``require_initialization=True``) also
    needs the decided teacher (`formal_teacher`)."""
    design, prereg = UD._read(design_path), UD._read(prereg_path)
    if require_initialization:
        formal_teacher(design)
    if arm not in design["arms"]:
        raise HD.DriverRefusal(f"{arm!r} is not an arm of {design['study_id']}; arms: {sorted(design['arms'])}")
    base_path = REPO / design["base_study"]["path"]
    if UD._sha(base_path) != design["base_study"]["sha256"]:
        raise HD.DriverRefusal(f"{design['base_study']['path']} is not the H1-PA record the design names")
    base = UD._read(base_path)
    ctl = design["base_study"]["control_arm"]
    seed_lists = [[int(s) for s in doc["seeds"]] for doc in (design, base, prereg)]
    if any(x != seed_lists[0] for x in seed_lists[1:]):
        raise HD.DriverRefusal(f"the design's, the base study's and the preregistration's seeds differ: {seed_lists}")
    init = design["initializations"][arm]
    rec = REPO / init["record"]
    bound = rec.is_file() and bool(init.get("sha256")) and UD._sha(rec) == init["sha256"]
    if require_initialization and not bound:
        raise HD.DriverRefusal(f"{arm}: its initialization record {init['record']} is not the one the design binds "
                               "(it is written by the formal initialization, then bound)")
    study = copy.deepcopy(base)
    study["study_id"] = design["study_id"]
    study["arms"] = {arm: {**copy.deepcopy(base["arms"][ctl]), "initialization": arm,
                           "execution": f"{ctl}'s training on {ctl}'s set under this study's protocol"}}
    study["initializations"] = {arm: {"record": init["record"]}}
    study["derived_schedule_arms"] = [arm]
    recipes = {}
    for seed, c in base["recipe"]["e1_configs"].items():
        cfg = UD._read(REPO / c["path"])
        if HD.canonical_sha(cfg) != c["canonical_sha256"]:
            raise HD.DriverRefusal(f"{c['path']}: not the E1 config the study records")
        recipes[int(seed)] = cfg
    overrides = {k: v for k, v in design["recipe_overrides"].items() if k != "loss.kd_support"}
    overrides["loss.kd_support"] = resolved_kd_support()
    arm_over = {}
    for k, v in (design.get("arm_overrides") or {}).get(arm, {}).items():
        if k == "trainable_patterns_added":
            first = recipes[int(design["seeds"][0])]["trainable_patterns"]
            if any(recipes[int(s)]["trainable_patterns"] != first for s in design["seeds"]):
                raise HD.DriverRefusal("the seeds' E1 configs disagree on trainable_patterns")
            arm_over["trainable_patterns"] = list(first) + list(v)
        else:
            arm_over[k] = v
    units = [{"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed), "execution": "train"}
             for seed in design["seeds"]]
    bound = bound and require_initialization        #: the recipe side builds no unit, bound or not
    return {"study": study, "study_sha256": UD._sha(design_path), "prereg": prereg,
            "prereg_sha256": UD._sha(prereg_path), "recipes": recipes,
            "init": UD._read(rec)["materialization_identity"] if bound else None,
            "init_record_sha256": UD._sha(rec) if bound else None,
            "units": units if bound else [], "recipe_overrides": overrides, "arm_overrides": {arm: arm_over},
            "design": design}


def derived_recipe(arm: str, seed: int, *, run_name: str, purpose: str, out_dir: str, data_dir: str,
                   student_path: str, pack_root: Path | None, plan: dict | None = None) -> dict:
    """One arm's derived recipe for one seed, exactly as its formal unit would train it.

    With ``pack_root`` the schedule is derived from the staged pack and checked
    against the study's record (`h1pa_driver.schedule_for`); without it, it is the
    record's own (a host rehearsal that has no real pack)."""
    plan = plan or load_plan(arm, require_initialization=False)
    u = {"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed), "execution": "train"}
    if pack_root is not None:
        sched = HD.schedule_for(plan, arm, pack_root)
    else:
        rec = plan["study"]["arms"][arm]["schedule"]
        _, ds = HD.dataset_of(plan, arm)
        e1 = plan["recipes"][int(plan["study"]["seeds"][0])]
        sched = {"rung": int(ds["rung"]["target_supervised_tokens"]),
                 **HD.e1_schedule(int(rec["packed_sequences"]), epochs=3,
                                  batch_size=int(e1["batch"]["blocks_per_step"]))}
    return HD.derive_recipe(plan, u, sched, run_name=run_name, purpose=purpose, out_dir=out_dir,
                            data_dir=data_dir, student_path=student_path)


def restore_staged(init_dir: Path, ident: dict) -> Path:
    """An initialization as setup staged it from the relay, re-identified from its bytes."""
    if not (Path(init_dir).is_dir() and UD._verified(Path(init_dir), ident, UD._qwen3_adapter())):
        raise UD.IdentityMismatch(f"{init_dir}: the staged initialization does not re-identify as "
                                  f"{ident['artifact_digest'][:12]}; preserved, nothing trained")
    return Path(init_dir)


def admitted_evaluate(job: dict, *, admit=None, evaluate=HD.evaluate) -> tuple[dict, dict]:
    """The projected checkpoint's evaluation, only after the engine admits it on this card."""
    verdict = (admit or engine_admission)(job["model_dir"], job["out"].parent / "engine_admission")
    (job["out"].parent / "engine_admission.json").write_text(json.dumps(verdict, indent=1) + "\n")
    if verdict["status"] != "PASS":
        raise UD.IdentityMismatch(f"{job['unit']}: the vLLM model is not admitted on this checkpoint "
                                  f"({verdict['problems'][:3]}); nothing was evaluated, preserved")
    rep, files = evaluate(job)
    return rep, {**files, "engine_admission.json": job["out"].parent / "engine_admission.json"}


def engine_admission(model_dir: Path, work: Path) -> dict:
    """HF reference in a child that exits, then vLLM on the clear GPU (`engine_admission`, `vllm_check`)."""
    import subprocess

    from shared.pod.teacher_cache import local_snapshot
    from shared.validation import engine_admission as EA
    from shared.validation import vllm_check as VC

    work.mkdir(parents=True, exist_ok=True)
    cfg = UD._read(ADMISSION_CONFIG)
    tok = local_snapshot(cfg["probe"]["tokenizer"]["model"], cfg["probe"]["tokenizer"]["revision"])
    rows, ids = EA.probe_ids(cfg["probe"], tok)
    bad = [r["id"] for r in rows if r["token_ids_sha256"] != r["expected_sha256"]]
    if bad:
        return {"status": "FAIL", "problems": [f"the probe's token ids are not the configured ones: {bad}"]}
    probes = [{"ids": x} for x in ids]
    ref_path = work / "hf_reference.json"
    code = ("import json,sys; sys.path[:0]=['src','scripts']; from shared.validation import engine_admission as EA; "
            f"json.dump(EA.hf_reference({str(model_dir)!r}, json.load(open({str(work / 'probes.json')!r})), "
            f"device='cuda:0'), open({str(ref_path)!r}, 'w'))")
    (work / "probes.json").write_text(json.dumps([p["ids"] for p in probes]))
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)
    ref = json.loads(ref_path.read_text())
    res = VC.run_isolated(os.environ.get("AAD_VLLM_PYTHON", "/opt/vllm/bin/python"),
                          EA.vllm_request(model_dir, [p["ids"] for p in probes], 8192, tokenizer=tok), work / "vllm")
    crit = {k: cfg["engine_admission"][k] for k in EA.CRITERIA_KEYS}
    return {**EA.compare(ref, res.get("result"), crit), "gpu_before": res.get("gpu_before"),
            "engine_status": res.get("status"), "probe": rows}


def train_restoring_on_resume(restore, ident: dict | None, train=UD.train):
    """`unit_driver.train`, with the initialization restored first when a resume finds it absent.

    A unit resumed on a NEW resource (the earlier pod gone, the save point on the
    volume) finds its recipe's ``student_path`` unstaged; the trainer does not read
    those weights on a resume, but its tokenizer contract and identity binding read
    the directory, so it is restored and re-identified rather than assumed."""
    def run(job: dict):
        init = Path(job["recipe"]["student_path"])
        if job.get("resume") and ident is not None and not any(init.glob("*.safetensors")):
            restore(init, ident)
        return train(job)
    return run


def ops_for(arm: str, plan: dict | None = None):
    from shared.rollout.vllm_plugin_path import vllm_plugin_pythonpath
    env = {**HD.PRODUCTION.env, "PYTHONPATH": vllm_plugin_pythonpath(REPO, HD.PRODUCTION.env.get("PYTHONPATH"))}
    restore = UD.restore_init if arm == "D4-Augmented-topk200" else restore_staged
    evaluate = admitted_evaluate if arm == PROJECTED else HD.evaluate
    train = train_restoring_on_resume(restore, (plan or {}).get("init"))
    return dataclasses.replace(HD.PRODUCTION, restore_init=restore, evaluate=evaluate, env=env, train=train)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", required=True)
    args, rest = ap.parse_known_args(argv)
    from shared.rollout.vllm_plugin_path import vllm_plugin_pythonpath
    #: every child -- the trainer, the vLLM evaluation, the admission's engine -- inherits it
    os.environ["PYTHONPATH"] = vllm_plugin_pythonpath(REPO, os.environ.get("PYTHONPATH"))
    plan = load_plan(args.arm)
    return HD.run(rest, ops=ops_for(args.arm, plan), plan=plan)


if __name__ == "__main__":
    raise SystemExit(main())

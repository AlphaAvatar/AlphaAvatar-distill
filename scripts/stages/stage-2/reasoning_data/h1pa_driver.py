#!/usr/bin/env python3
"""On-pod driver of H1-PA's six units: (restore -> train -> persist) -> evaluate under nucleus_v1 -> persist -> report.

    cd /workspace/aad && PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \
        scripts/stages/stage-2/reasoning_data/h1pa_driver.py \
        --run-id <run> --durable-root <volume>/h1pa --pack-root <dir> --checkpoint-root <dir> \
        --status <file> [--unit D4-Original-s20260726 ...] [--capacity-bytes N] \
        [--validation [--validation-steps N] [--validation-items N]]

Everything a unit is comes from its owner: the arms, seeds, datasets, recipes
and checkpoint identities from ``configs/stages/stage-2/reasoning_data/h1pa.json``;
the evaluation set, the decoding protocol and the decision rule from
``plans/h1pa_preregistration.json``; D4 Top-1's identity from its
materialization record. A unit is ``<arm>-s<seed>``.

* **PCA-Original** is EVALUATED ONLY: E1's checkpoint for that seed, staged
  under ``--checkpoint-root/<seed>/`` and re-identified by its
  ``model.safetensors`` sha256, is never retrained and its weights are never
  persisted again (they are durable on the relay).
* **D4-Original** trains D4 Top-1 under the seed's E1 config on E1's own pack:
  E1's batch size 2, 3 epochs, 2,916 optimizer steps, every key unchanged
  except where the run lives (`WHERE`) and the tokenizer source.
* **D4-Augmented** does the same on the merged set; its schedule is E1's
  builder rule on the merged set's real training-sequence count, checked
  against the study's recorded schedule. Nothing else changes: not the loss,
  optimizer, learning rate, micro-batch size or any other key.

Evaluation is `dose_driver.evaluate_all` with the preregistered decoding
protocol passed explicitly. A unit refuses -- and never falls back to greedy
decoding -- when the configuration names no protocol or another one, when a
generation already on the volume was recorded under another protocol, or
when a generation summary does not carry the protocol.

STATE, RESUME, RETRY and the TERMINAL PATH are `unit_driver`'s, reused, not
re-implemented: state is read from the durable root on every (re)start (a
scored unit is skipped and only its evidence is brought home; a trained and
durable one resumes at scoring; an interrupted training attempt continues from
its last save point with ``--resume latest``; a record whose inputs are not
this unit's is preserved and stops the unit), one complete endpoint per unit,
and one ``finally`` that writes the status and closes observers. An
evaluation-only unit binds its evaluation to its inputs by persisting
``unit_inputs.json`` inside the evaluation file set.

``--validation`` (the GPU validation) writes only under ``<durable
root>/validation`` and is never formal: each named training unit trains
``--validation-steps`` steps, and each named unit evaluates the first
``--validation-items`` fluid items and the first gsm8k prompts under the same
protocol; its report measures seconds per step, peak memory and generation
throughput, the formal session's pricing inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from stages.reasoning_data import persist_unit as PU
from stages.reasoning_data import unit_driver as UD

REPO = UD.REPO
STUDY = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
PREREG = UD.LOG / "plans/h1pa_preregistration.json"
REPORT_SCHEMA = "aadistill.stage2.reasoning_data.h1pa_unit_report/v1"
TOKENIZER_SOURCE = "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"
#: keys that name WHERE a run lives, never WHAT it trains: outside the recipe identity
WHERE = ("run_name", "_purpose", "out_dir", "data_dir", "student_path", "tokenizer_source")
#: what E1's builder derives from the sequence count: D4-Augmented's only recipe overrides
SCHEDULE_KEYS = ("rung", "schedule.total_steps", "schedule.warmup_steps", "checkpoint.save_every",
                 "intervals.eval_every")
TRAINED = ("D4-Original", "D4-Augmented")
VALIDATION_CAPABILITY = "stage3_capability_846__gsm8k"


DriverRefusal, IdentityMismatch = UD.DriverRefusal, UD.IdentityMismatch


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_sha(doc) -> str:
    return UD._json_sha(doc)


# ---- the plan --------------------------------------------------------------------------------------

def e1_schedule(n_sequences: int, *, epochs: int, batch_size: int) -> dict:
    """E1's builder rule: ceil(epochs x sequences / batch size) optimizer steps
    and `unit_driver.lr_rule`'s warmup, evaluation and save intervals."""
    steps = math.ceil(epochs * n_sequences / batch_size)
    return {"total_steps": steps, **UD.lr_rule(steps)}


def load_plan(study_path: Path = STUDY, prereg_path: Path = PREREG) -> dict:
    st, prereg = UD._read(study_path), UD._read(prereg_path)
    recipes = {}
    for seed, c in st["recipe"]["e1_configs"].items():
        cfg = UD._read(REPO / c["path"])
        if canonical_sha(cfg) != c["canonical_sha256"]:
            raise DriverRefusal(f"{c['path']}: not the E1 config the study records")
        recipes[int(seed)] = cfg
    init_rec = REPO / st["initializations"]["d4_top1"]["record"]
    units = [{"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed),
              "execution": "train" if arm in TRAINED else "evaluate"}
             for arm in st["arms"] for seed in st["seeds"]]
    return {"study": st, "study_sha256": UD._sha(study_path), "prereg": prereg,
            "prereg_sha256": UD._sha(prereg_path), "recipes": recipes,
            "init": UD._read(init_rec)["materialization_identity"], "init_record_sha256": UD._sha(init_rec),
            "units": units}


def dataset_of(plan: dict, arm: str) -> tuple[str, dict]:
    key = plan["study"]["arms"][arm]["dataset"]
    return key, plan["study"]["datasets"][key]


def pack_refusals(plan: dict, arm: str, pack_root: Path) -> list[str]:
    key, ds = dataset_of(plan, arm)
    d = Path(pack_root) / key
    missing = [n for n in ("blocks.npz", "ladder.json", "audit.jsonl") if not (d / n).is_file()]
    if missing:
        return [f"{arm}: {d} lacks {missing}"]
    return [f"{arm}: {d / n} sha256 {UD._sha(d / n)[:12]} is not the recorded {ds[k][:12]}"
            for n, k in (("blocks.npz", "blocks_sha256"), ("ladder.json", "ladder_sha256"))
            if UD._sha(d / n) != ds[k]]


def schedule_for(plan: dict, arm: str, pack_root: Path, validation_steps: int | None = None) -> dict | None:
    """None for D4-Original (E1's own schedule, unchanged); D4-Augmented's from
    the merged set's real training-sequence count, which must equal the
    study's record; a validation run's from its step count."""
    e1 = plan["recipes"][int(plan["study"]["seeds"][0])]
    batch, epochs = int(e1["batch"]["blocks_per_step"]), 3
    if validation_steps:
        return {"rung": None, "total_steps": int(validation_steps), **UD.lr_rule(int(validation_steps))}
    if arm != "D4-Augmented":
        return None
    _, ds = dataset_of(plan, arm)
    lad = UD._read(Path(pack_root) / plan["study"]["arms"][arm]["dataset"] / "ladder.json")
    rung = lad["rungs"][0]
    sched = {"rung": int(rung["target_supervised_tokens"]),
             **e1_schedule(int(rung["n_blocks"]), epochs=epochs, batch_size=batch)}
    want = plan["study"]["arms"][arm]["schedule"]
    got = {"packed_sequences": int(rung["n_blocks"]), "optimizer_steps": sched["total_steps"],
           "warmup_steps": sched["warmup_steps"], "eval_every": sched["eval_every"],
           "save_every": sched["save_every"]}
    bad = {k: (v, want.get(k)) for k, v in got.items() if want.get(k) != v}
    if bad:
        raise DriverRefusal(f"D4-Augmented's schedule from the merged set {bad} is not the study's record")
    return sched


def derive_recipe(plan: dict, u: dict, sched: dict | None, *, run_name: str, purpose: str, out_dir: str,
                  data_dir: str, student_path: str) -> dict:
    """The seed's E1 config with WHERE replaced and, for D4-Augmented (or a
    validation run), the builder-derived schedule; every other key, nested
    keys included, is checked unchanged."""
    e1 = plan["recipes"][u["seed"]]
    d = copy.deepcopy(e1)
    d.update(run_name=run_name, _purpose=purpose, out_dir=out_dir, data_dir=data_dir, student_path=student_path,
             tokenizer_source=TOKENIZER_SOURCE)
    allowed = set(WHERE)
    if sched is not None:
        if sched["rung"] is not None:
            d["rung"] = sched["rung"]
        d["schedule"].update(total_steps=sched["total_steps"], warmup_steps=sched["warmup_steps"])
        d["checkpoint"]["save_every"] = sched["save_every"]
        d["intervals"]["eval_every"] = sched["eval_every"]
        allowed |= set(SCHEDULE_KEYS)
    a, b = UD._flat(e1), UD._flat(d)
    outside = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in allowed)
    if outside:
        raise DriverRefusal(f"the derived recipe differs from E1's outside the declared overrides: {outside}")
    return d


def recipe_identity(derived: dict) -> str:
    return canonical_sha({k: v for k, v in derived.items() if k not in WHERE})


def evaluation_spec(plan: dict) -> dict:
    """What every evaluation binds: the preregistered set and decoding protocol.
    A study or preregistration without a decoding protocol refuses here."""
    ev, pr = plan["study"]["evaluation"], plan["prereg"]["evaluation"]
    dec = pr.get("decoding") or {}
    if not dec.get("path") or not dec.get("sha256"):
        raise DriverRefusal("the preregistration names no decoding protocol; H1-PA never evaluates greedily")
    if ev.get("decoding") != dec["path"]:
        raise DriverRefusal(f"the study's decoding {ev.get('decoding')!r} is not the preregistered {dec['path']!r}")
    return {"set": pr["set"]["eval_set_id"], "manifest_sha256": pr["set"]["manifest_sha256"],
            "trained_context": int(ev["trained_context"]), "action_protocol": ev["action_protocol"],
            "decoding": {"path": dec["path"], "sha256": dec["sha256"]}}


def unit_inputs(plan: dict, u: dict, sched: dict | None, recipe_sha: str | None, validation: bool) -> dict:
    out = {"arm": u["arm"], "seed": u["seed"], "formal": not validation, "execution": u["execution"],
           "evaluation": evaluation_spec(plan), "preregistration_sha256": plan["prereg_sha256"]}
    if u["execution"] == "evaluate":
        out["checkpoint_model_safetensors_sha256"] = \
            plan["study"]["arms"][u["arm"]]["checkpoints"][str(u["seed"])]["model_safetensors_sha256"]
        return out
    _, ds = dataset_of(plan, u["arm"])
    return {**out, "initialization_artifact_digest": plan["init"]["artifact_digest"],
            "initialization_weights_digest": plan["init"]["weights_digest"],
            "pack_blocks_sha256": ds["blocks_sha256"], "recipe_sha256": recipe_sha, "schedule": sched}


def launch_refusals(plan: dict, units: list[dict], *, validation: bool, pack_root: Path,
                    checkpoint_root: Path) -> list[str]:
    out = []
    if not validation and plan["study"].get("frozen") is not True:
        out.append("h1pa.json is not frozen: a formal launch needs the frozen study and its authorization")
    try:
        evaluation_spec(plan)
    except DriverRefusal as exc:
        out.append(str(exc))
    for arm in sorted({u["arm"] for u in units if u["execution"] == "train"}):
        out += pack_refusals(plan, arm, pack_root)
    for u in units:
        if u["execution"] == "evaluate":
            f = Path(checkpoint_root) / str(u["seed"]) / "model.safetensors"
            if not f.is_file():
                out.append(f"{u['unit']}: {f} is not staged")
    return out


# ---- operations (injected in tests) ----------------------------------------------------------------

def stage_checkpoint(checkpoint_root: Path, seed: int, sha256: str) -> Path:
    """E1's checkpoint for `seed`, re-identified by its weights' sha256."""
    d = Path(checkpoint_root) / str(seed)
    if UD._sha(d / "model.safetensors") != sha256:
        raise IdentityMismatch(f"{d}: model.safetensors is not the recorded {sha256[:12]}")
    return d


def _decoding_of(summary: dict) -> str | None:
    dp = summary.get("decoding_protocol") or (summary.get("identity") or {}).get("decoding_protocol") or {}
    return dp.get("sha256")


def generation_refusals(gens: Path, sha256: str) -> list[str]:
    """Every generation summary in `gens` must carry the decoding protocol."""
    return [f"{p.name}: decoding {(_decoding_of(json.loads(p.read_text())) or 'none')[:12]}, not {sha256[:12]}"
            for p in sorted(Path(gens).glob("*.summary.json")) if _decoding_of(json.loads(p.read_text())) != sha256]


def evaluate(job: dict) -> tuple[dict, dict]:
    from stages.calibration_volume.dose_driver import evaluate_all
    ev, out, progress = job["evaluation"], job["out"], job["progress"]
    if UD._sha(REPO / ev["decoding"]["path"]) != ev["decoding"]["sha256"]:
        raise DriverRefusal(f"{ev['decoding']['path']} is not the preregistered decoding protocol")
    out.mkdir(parents=True, exist_ok=True)
    progress.mkdir(parents=True, exist_ok=True)
    bad = generation_refusals(progress, ev["decoding"]["sha256"])
    if bad:
        raise IdentityMismatch(f"generations on the volume under another decoding protocol: {bad[:3]}; preserved")
    prereg = {"evaluation": {"trained_context": ev["trained_context"], "action_protocol": ev["action_protocol"],
                             "decoding": ev["decoding"]["path"]}}
    rep, _ = evaluate_all(job["model_dir"], out, progress, job["label"], prereg, job["env"])
    bad = generation_refusals(out / "generations", ev["decoding"]["sha256"])
    if bad:
        raise IdentityMismatch(f"generations not under the preregistered decoding protocol: {bad[:3]}")
    (out / "formal_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    files = {p.relative_to(out).as_posix(): p for sub in ("generations", "rows")
             for p in sorted((out / sub).rglob("*")) if p.is_file()}
    files |= {n: out / n for n in ("formal_report.json", "eval_fluid.log", "eval_capability.log")
              if (out / n).is_file()}
    return rep, files


def evaluate_subset(job: dict) -> tuple[dict, dict]:
    """The GPU validation's evaluation: the first `items` fluid items through
    run_battery and the first gsm8k prompts through uncapped_eval, both under
    the decoding protocol; generation seconds and tokens are measured."""
    from shared import eval_sets as ES
    from stages.calibration_volume.dose_driver import VLLM_PY
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256
    from stages.phase_c1.packaging import build_evaluation_package
    ev, out, n = job["evaluation"], job["out"], int(job["items"])
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / "package"
    shutil.rmtree(pkg, ignore_errors=True)
    build_evaluation_package(job["model_dir"], tokenizer_source=REPO / TOKENIZER_SOURCE, dest=pkg,
                             expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
    dec, ctx, gens = str(REPO / ev["decoding"]["path"]), str(ev["trained_context"]), out / "generations"
    gens.mkdir(exist_ok=True)
    t0 = time.time()
    with (out / "eval_fluid.log").open("w") as lf:
        rc = subprocess.run([VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py", "--model", str(pkg),
                             "--label", job["label"], "--tasks", str(ES.unit_path(ev["set"], "fluid_reasoning_720", "dev_v1")),
                             "--mode", "eval", "--out", str(gens / "fluid_reasoning_720__dev_v1.jsonl"),
                             "--trained-context", ctx, "--action-protocol", ev["action_protocol"], "--limit", str(n),
                             "--decoding", dec, "--seed-namespace", "fluid_reasoning_720__dev_v1"],
                            cwd=REPO, env=job["env"], stdout=lf, stderr=subprocess.STDOUT).returncode
    t1 = time.time()
    if rc != 0:
        raise RuntimeError(f"run_battery exited {rc}; see eval_fluid.log")
    src = ES.unit_path(ev["set"], "stage3_capability_846", "gsm8k")
    staged = out / "uncapped_in" / f"{VALIDATION_CAPABILITY}.jsonl"
    staged.parent.mkdir(exist_ok=True)
    staged.write_text("".join(src.read_text().splitlines(keepends=True)[:n]))
    with (out / "eval_capability.log").open("w") as lf:
        rc = subprocess.run([VLLM_PY, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg),
                             "--label", job["label"], "--prompts", str(staged), "--out-dir", str(gens),
                             "--trained-context", ctx, "--decoding", dec],
                            cwd=REPO, env=job["env"], stdout=lf, stderr=subprocess.STDOUT).returncode
    t2 = time.time()
    if rc != 0:
        raise RuntimeError(f"uncapped_eval exited {rc}; see eval_capability.log")
    if (gens / f"{VALIDATION_CAPABILITY}.json").is_file():
        os.replace(gens / f"{VALIDATION_CAPABILITY}.json", gens / f"{VALIDATION_CAPABILITY}.summary.json")
    fluid_sum = gens / "fluid_reasoning_720__dev_v1.summary.json"
    bad = generation_refusals(gens, ev["decoding"]["sha256"])
    rep = {"formal": False, "items_per_unit": n, "decoding_refusals": bad,
           "seconds": {"fluid": round(t1 - t0, 1), "gsm8k": round(t2 - t1, 1)},
           "fluid_summary": json.loads(fluid_sum.read_text()) if fluid_sum.is_file() else None}
    (out / "validation_eval.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    if bad:
        raise IdentityMismatch(f"validation generations not under the decoding protocol: {bad[:3]}")
    files = {p.relative_to(out).as_posix(): p for p in sorted(gens.rglob("*")) if p.is_file()}
    files |= {n: out / n for n in ("validation_eval.json", "eval_fluid.log", "eval_capability.log")}
    return rep, files


@dataclass
class Ops:
    restore_init: Callable[[Path, dict], Path]
    stage_checkpoint: Callable[[Path, int, str], Path]
    train: Callable[[dict], Path]
    evaluate: Callable[[dict], tuple[dict, dict]]
    evaluate_subset: Callable[[dict], tuple[dict, dict]]
    adapter: Callable[[], object]
    commit: Callable[[], str]
    env: dict = field(default_factory=dict)


PRODUCTION = Ops(restore_init=UD.restore_init, stage_checkpoint=stage_checkpoint, train=UD.train,
                 evaluate=evaluate, evaluate_subset=evaluate_subset, adapter=UD._qwen3_adapter,
                 commit=UD._git_commit, env={**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"})


# ---- state of an evaluation-only unit, from the durable root -----------------------------------------

def eval_unit_state(root: Path, unit: str, inputs: dict) -> tuple[str, str]:
    try:
        evaluation = PU.completed(root, unit, "evaluation")
        report = PU.completed(root, unit, "report")
    except Exception as exc:          # a record this driver did not write completes nothing
        return "mismatch", str(exc)
    if evaluation is not None:
        got = Path(root) / unit / "evaluation" / "unit_inputs.json"
        if not got.is_file() or json.loads(got.read_text()) != json.loads(json.dumps(inputs)):
            return "mismatch", "the persisted evaluation was made for other inputs"
    if report is not None:
        return ("scored", "") if evaluation is not None else ("mismatch", "a report without its evaluation")
    return ("evaluated", "") if evaluation is not None else ("unevaluated", "")


# ---- one unit ----------------------------------------------------------------------------------------

def run_unit(plan: dict, u: dict, *, root: Path, work: Path, pack_root: Path, checkpoint_root: Path,
             init_dir: Path, run_id: str, validation: bool, validation_steps: int | None,
             validation_items: int, capacity: int | None, ops: Ops) -> dict:
    unit, train_unit = u["unit"], u["execution"] == "train"
    sched = schedule_for(plan, u["arm"], pack_root, validation_steps) if train_unit else None
    recipe_sha = None
    if train_unit:
        probe = derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-")
        recipe_sha = recipe_identity(probe)
    inputs = unit_inputs(plan, u, sched, recipe_sha, validation)
    if validation:
        inputs["validation"] = {"steps": validation_steps, "items": validation_items}
    state, why = UD.unit_state(root, unit, inputs) if train_unit else eval_unit_state(root, unit, inputs)
    history = UD.attempts(root, unit)
    n = 1 + max((a["attempt"] for a in history), default=0)
    base = {"attempt": n, "run_id": run_id, "commit": ops.commit(), "state_at_start": state, "formal": not validation}
    if state == "mismatch":
        UD._append(root, unit, {**base, "event": "end", "action": "stop", "outcome": "identity_mismatch",
                                "error": why, "utc": _now()})
        raise IdentityMismatch(f"{unit}: {why}; preserved, nothing consumed")
    if state == "scored":
        home = UD.bring_home(root, unit, work / unit / "home")
        UD._append(root, unit, {**base, "event": "end", "action": "skip", "outcome": "complete", "utc": _now()})
        return {"unit": unit, "outcome": "complete", "action": "skipped", "brought_home": home}

    attempt_dir = work / unit / f"attempt-{n}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    unit_dir = Path(root) / unit
    started = {a["attempt"] for a in history if a.get("event") == "start" and a.get("action") in ("train", "resume")}
    failed_training = sorted(started - {a["attempt"] for a in history
                                        if a.get("event") == "end" and a.get("outcome") == "complete"})
    local = UD.local_trained(work, unit, inputs) if state == "untrained" else None
    saved = UD.resumable(root, unit, inputs) if state == "untrained" and local is None else None
    action = ({"trained": "evaluate", "evaluated": "report", "unevaluated": "evaluate"}.get(state)
              or ("persist_local" if local else "resume" if saved else "train"))
    UD._append(root, unit, {**base, "event": "start", "action": action, "utc": _now(),
                            "replacement_of": failed_training[-1] if action in ("train", "resume") and failed_training
                            else None, "resumed_from": str(saved) if saved else None})
    stage, model_dir, checkpoint = action, None, None
    try:
        if action in ("train", "resume", "persist_local"):
            train_dir = saved
            if action == "train":
                stage = "restore_initialization"
                init = ops.restore_init(init_dir, plan["init"])
                stage = "train"
                train_dir = unit_dir / "train" / f"a{n}"
                key, _ = dataset_of(plan, u["arm"])
                recipe = derive_recipe(
                    plan, u, sched, run_name=f"h1pa_{unit}_{run_id}_a{n}",
                    purpose=f"H1-PA {'GPU VALIDATION (not formal)' if validation else 'formal'} unit {unit}, attempt {n}",
                    out_dir=str(train_dir), data_dir=str(Path(pack_root) / key), student_path=str(init))
                train_dir.mkdir(parents=True, exist_ok=True)
                (train_dir / "recovery_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
                (train_dir / "train_inputs.json").write_text(json.dumps({"inputs": inputs, "attempt": n}, indent=1) + "\n")
                model_dir = ops.train({"unit": unit, "attempt_dir": attempt_dir, "recipe": recipe, "env": ops.env})
            elif action == "resume":
                stage = "resume"
                recipe = UD._read(saved / "recovery_config.json")
                model_dir = ops.train({"unit": unit, "attempt_dir": attempt_dir, "recipe": recipe, "env": ops.env,
                                       "resume": True})
            else:
                model_dir = local
            if action != "persist_local":
                (attempt_dir / "trained.json").write_text(json.dumps(
                    {"inputs": inputs, "model_dir": str(model_dir), "finished_utc": _now()}, indent=1) + "\n")
            stage = "persist_checkpoint"
            PU.persist_checkpoint(unit, model_dir, root, adapter=ops.adapter(),
                                  arch_signature=plan["init"]["arch_signature"],
                                  num_parameters=plan["init"]["num_parameters"], inputs=inputs, capacity=capacity)
            UD.persist_training_records(unit, model_dir, root, work, capacity, train_dir=train_dir,
                                        attempt_dir=attempt_dir)
            if not UD._verified(unit_dir / "checkpoint", PU.completed(root, unit, "checkpoint"), ops.adapter()):
                raise IdentityMismatch(f"{unit_dir / 'checkpoint'}: the persisted checkpoint does not re-identify")
            stage = "release_save_points"
            UD._append(root, unit, {**base, "event": "released", "save_points": UD.release_save_points(root, unit),
                                    "utc": _now()})
            model_dir = unit_dir / "checkpoint"
        elif state == "trained" and PU.completed(root, unit, "training") is None:
            stage = "persist_training_records"
            mine = UD.local_trained(work, unit, inputs)
            if mine is not None:
                UD.persist_training_records(unit, mine, root, work, capacity)
        if action != "report":
            if train_unit:
                if model_dir is None:
                    stage = "verify_checkpoint"
                    model_dir = unit_dir / "checkpoint"
                    if not UD._verified(model_dir, PU.completed(root, unit, "checkpoint"), ops.adapter()):
                        raise IdentityMismatch(f"{model_dir}: the persisted checkpoint no longer re-identifies")
            else:
                stage = "stage_checkpoint"
                model_dir = ops.stage_checkpoint(checkpoint_root, u["seed"], inputs["checkpoint_model_safetensors_sha256"])
            stage = "evaluate"
            job = {"unit": unit, "model_dir": model_dir, "out": attempt_dir / "evaluation",
                   "progress": unit_dir / "eval_progress", "label": f"h1pa_{unit}_{run_id}",
                   "evaluation": inputs["evaluation"], "env": ops.env, "items": validation_items}
            rep, files = (ops.evaluate_subset if validation else ops.evaluate)(job)
            ui = attempt_dir / "unit_inputs.json"
            ui.write_text(json.dumps(inputs, indent=1, sort_keys=True) + "\n")
            stage = "persist_evaluation"
            PU.persist_files(unit, {**files, "unit_inputs.json": ui}, root, label="evaluation", capacity=capacity)
        else:
            rep = UD._read(unit_dir / "evaluation" / ("validation_eval.json" if validation else "formal_report.json"))
        stage = "report"
        ev = PU.completed(root, unit, "evaluation")
        ckpt = PU.completed(root, unit, "checkpoint") if train_unit else None
        checkpoint = ({k: ckpt[k] for k in ("artifact_digest", "weights_digest", "destination")} if ckpt else
                      {"model_safetensors_sha256": inputs["checkpoint_model_safetensors_sha256"],
                       "durable": plan["study"]["arms"][u["arm"]]["checkpoints"][str(u["seed"])].get("relay"),
                       "persisted_again": False})
        report = {"schema": REPORT_SCHEMA, "unit": unit, "arm": u["arm"], "seed": u["seed"],
                  "execution": u["execution"], "formal": not validation, "run_id": run_id, "attempt": n,
                  "written_utc": _now(), "inputs": inputs, "study_sha256": plan["study_sha256"],
                  "initialization_record_sha256": plan["init_record_sha256"] if train_unit else None,
                  "checkpoint": checkpoint, "evaluation_files": len(ev["files"]),
                  "evaluation_descriptor_sha256": UD._sha(unit_dir / "evaluation.descriptor.json"),
                  "attempts": [a for a in UD.attempts(root, unit) if a.get("event") == "start"],
                  ("validation_report" if validation else "formal_report"): rep,
                  "not": None if not validation else "a GPU validation: never a formal H1-PA result"}
        (attempt_dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        PU.persist_files(unit, {"report.json": attempt_dir / "report.json"}, root, label="report", capacity=capacity)
    except BaseException as exc:
        import traceback
        from aadistill.infrastructure.failure_signature import signature_of
        UD._append(root, unit, {**base, "event": "end", "action": action, "outcome": "failed", "stage": stage,
                                "error": f"{type(exc).__name__}: {exc}"[:800],
                                "signature": signature_of(traceback.format_exc()), "utc": _now()})
        raise
    UD._append(root, unit, {**base, "event": "end", "action": action, "outcome": "complete", "utc": _now()})
    return {"unit": unit, "outcome": "complete", "action": action, "attempt": n}


# ---- the run -----------------------------------------------------------------------------------------

def run(argv=None, *, ops: Ops = PRODUCTION, plan: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--pack-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/packs"))
    ap.add_argument("--checkpoint-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/pca_original"))
    ap.add_argument("--init-dir", default=str(REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/"
                                                     "1301deb71685aa1ca5c7b6cf2c491ac4"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--unit", action="append", default=None)
    ap.add_argument("--capacity-bytes", type=int, default=None)
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--validation-steps", type=int, default=None)
    ap.add_argument("--validation-items", type=int, default=24)
    args = ap.parse_args(argv)
    if args.validation_steps and not args.validation:
        ap.error("--validation-steps is a GPU-validation knob; it needs --validation")
    work = Path(args.work or REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/units" / args.run_id)
    status = Path(args.status or work / "status.json")
    root = Path(args.durable_root) / ("validation" if args.validation else "")
    doc = {"schema": "aadistill.stage2.reasoning_data.h1pa_driver_status/v1", "run_id": args.run_id,
           "formal": not args.validation, "started_utc": _now(), "units": {}, "terminal": None}
    code = 1
    try:
        plan = plan or load_plan()
        known = {u["unit"]: u for u in plan["units"]}
        unknown = sorted(set(args.unit or []) - set(known))
        if unknown:
            raise DriverRefusal(f"unknown units {unknown}; the plan has {sorted(known)}")
        units = [known[k] for k in (args.unit or known)]
        refusals = launch_refusals(plan, units, validation=args.validation, pack_root=Path(args.pack_root),
                                   checkpoint_root=Path(args.checkpoint_root))
        if refusals:
            doc.update(terminal="REFUSED", refusals=refusals)
            code = 2
            return code
        for u in units:
            doc["units"][u["unit"]] = {"outcome": "running"}
            try:
                doc["units"][u["unit"]] = run_unit(
                    plan, u, root=root, work=work, pack_root=Path(args.pack_root),
                    checkpoint_root=Path(args.checkpoint_root), init_dir=Path(args.init_dir), run_id=args.run_id,
                    validation=args.validation, validation_steps=args.validation_steps if u["execution"] == "train"
                    and args.validation else None, validation_items=args.validation_items,
                    capacity=args.capacity_bytes, ops=ops)
            except Exception as exc:
                doc["units"][u["unit"]] = {"outcome": "identity_mismatch" if isinstance(exc, IdentityMismatch)
                                           else "failed", "error": f"{type(exc).__name__}: {exc}"[:800]}
                doc["terminal"] = "FAILED"
                return code
        doc["terminal"], code = "ALL_DONE", 0
        return code
    except DriverRefusal as exc:
        doc.update(terminal="REFUSED", refusals=[str(exc)])
        code = 2
        return code
    finally:
        doc.update(finished_utc=_now(), exit_code=code, observers_open=UD.close_observers(),
                   teardown="the launcher's: this driver creates and deletes no provider resource",
                   terminal=doc["terminal"] or "FAILED")
        status.parent.mkdir(parents=True, exist_ok=True)
        tmp = status.with_name(f".{status.name}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        os.replace(tmp, status)


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
"""On-pod driver of H1's training units: restore -> train -> persist -> evaluate -> persist -> report.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-2/reasoning_data/unit_driver.py \
        --run-id <run> --durable-root <mounted backend>/h1 --pack-root <dir with A/ and B/> \
        --status <file> [--unit A-s20260726 ...] [--capacity-bytes <provider figure>] \
        [--validation [--validation-steps N]]

What a unit is lives in `configs/stages/stage-2/reasoning_data/h1_units.json`,
and every fact it does not own is read where it is owned: the arms and seeds
from `plans/comparison_design.json`, the schedule from `plans/h1_accounting.json
:: schedule`, the initialization's identity from `assets/d4_top1_materialization.json`,
the recipe from the frozen recovery recipe. A FORMAL launch is refused at $0
while the config is not frozen, while the schedule is provisional, or while an
arm's pack identity is pending or its files do not match it. ``--validation``
(the separately authorized GPU validation) may run a provisional schedule or
``--validation-steps``, never a pending pack, and writes only under
``<durable root>/validation``; its report says ``formal: false``.

Per unit, in this order:

1. the D4 Top-1 initialization is restored from the relay (or taken from
   ``--init-dir``) and its identity re-proved from the bytes;
2. the frozen recipe is trained with H1's declared overrides only (the
   derived config is checked key by key against the frozen file);
3. the moment training ends, `persist_unit` persists the checkpoint (and the
   training records), so no later failure can take it;
4. eval-set-1, every item, by `dose_driver.evaluate_all`;
5. the evaluation outputs are persisted under their logical paths, and
   ``report.json`` last: its completed record is what "this unit is done" means.

STATE IS READ FROM THE DURABLE ROOT on every (re)start (AGENTS.md P8.4):
completed and scored -> skipped, its evidence (never its weights) brought home;
trained and durable -> re-verified and resumed at scoring, never retrained;
evaluated, report not persisted -> the report is written from the persisted
evaluation; not trained -> trained (or, on the same machine, a model a previous
attempt finished training is persisted instead of retrained); a record whose
inputs are not this unit's -> preserved, nothing consumed, the unit stopped.

RESUME (validated on a GPU by P1g, `p1g_driver.py`): the trainer writes its
save points (model, AdamW state, step, RNG) into ``<durable root>/<unit>/train/
a<attempt>/`` on the backend itself, with the attempt's config and inputs beside
them. A unit whose training attempt ended without completing, and whose save
point is there for THIS unit's inputs, is continued from it with
``train_stage3.py --resume latest`` under the attempt's own config -- the same
data order, learning rate and step count -- never retrained from the
initialization. The save points are deleted once the final checkpoint is
persisted and re-identified.

RETRY (P12.1): one complete endpoint per unit -- a completed checkpoint is never
overwritten and a scored unit never rerun. Once formal training began, an
engineering failure is repaired and the same unit and seed continued from its
last save point (or, with none, trained again from the initialization) as ONE
replacement endpoint; the failed attempt keeps its own directory and its row in
``attempts.jsonl`` (with its failure signature for the launcher's same-failure
gate); partial measurements are never pooled. The driver stops at the first
failed unit; every earlier unit is already durable.

TERMINAL PATH (P12.3): one ``finally`` writes the status (ALL_DONE / FAILED /
REFUSED, per-unit outcomes), closes any registered observer -- the driver
starts none, its children run to completion -- and returns the exit code.
Tearing the provider resource down is the launcher's job; this driver creates
and deletes nothing at the provider.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import persist_unit as PU  # noqa: E402

CONFIG = REPO / "configs/stages/stage-2/reasoning_data/h1_units.json"
LOG = REPO / "logs/stages/stage-2/reasoning_data"
TRAIN_PY = "/opt/train/bin/python"
REPORT_SCHEMA = "aadistill.stage2.reasoning_data.h1_unit_report/v1"
#: keys of the derived recipe that name WHERE, not WHAT: left out of its identity
WHERE = ("run_name", "_purpose", "out_dir", "student_path", "data_dir")
#: background processes started by this driver; the terminal path closes them
_OBSERVERS: list[subprocess.Popen] = []


class DriverRefusal(RuntimeError):
    """A $0 refusal before any unit work."""


class IdentityMismatch(RuntimeError):
    """A durable record that is not this unit's: preserved, nothing consumed."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _sha(path: Path) -> str:
    return PU._sha256(Path(path))


def _json_sha(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


# ---- the plan: everything read from its owner ---------------------------------------------------

def load_plan(config: Path = CONFIG) -> dict:
    cfg = _read(config)
    design = _read(LOG / "plans/comparison_design.json")
    schedule = _read(LOG / "plans/h1_accounting.json")["schedule"]
    init_record = REPO / cfg["initialization"]["record"]
    recipe_path = REPO / cfg["recipe"]["frozen"]
    if _sha(recipe_path) != cfg["recipe"]["frozen_sha256"]:
        raise DriverRefusal(f"{recipe_path}: not the frozen recipe the config names")
    return {"config": cfg, "config_sha256": _sha(config), "schedule": schedule,
            "init": _read(init_record)["materialization_identity"], "init_record_sha256": _sha(init_record),
            "recipe": _read(recipe_path), "packs": cfg["packs"],
            "units": [{"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed)}
                      for arm in design["arms"] for seed in design["fixed"]["seeds"]]}


def lr_rule(steps: int) -> dict:
    """The E1 builder's schedule rule (`h1_units.json :: schedule.lr_rule`)."""
    return {"warmup_steps": max(10, round(0.05 * steps)), "eval_every": max(25, steps // 8),
            "save_every": max(200, steps // 2)}


def schedule_for(plan: dict, arm: str, validation_steps: int | None = None) -> dict:
    s = plan["schedule"]
    steps = int(validation_steps or s["steps"])
    rule = lr_rule(steps)
    if validation_steps is None and {k: s[k] for k in rule} != rule:
        raise DriverRefusal(f"the schedule record {({k: s[k] for k in rule})} disagrees with its rule {rule}")
    return {"total_steps": steps, "blocks_per_step": int(s["blocks_per_step"][arm]), **rule,
            "rung": int(plan["config"]["schedule"]["rung"])}


def pack_refusals(pack: dict, arm: str, pack_root: Path) -> list[str]:
    if pack.get("status") != "recorded" or not (pack.get("blocks_sha256") and pack.get("ladder_sha256")
                                                and pack.get("expected")):
        return [f"arm {arm}: pack identity pending -- {pack.get('what', '')[:120]}"]
    d = Path(pack_root) / arm
    missing = [n for n in ("blocks.npz", "ladder.json") if not (d / n).is_file()]
    if missing:
        return [f"arm {arm}: {d} lacks {missing}"]
    out = [f"arm {arm}: {n} sha256 {_sha(d / n)[:12]} is not the recorded {pack[k][:12]}"
           for n, k in (("blocks.npz", "blocks_sha256"), ("ladder.json", "ladder_sha256"))
           if _sha(d / n) != pack[k]]
    if out:
        return out
    rung = next((r for r in _read(d / "ladder.json")["rungs"] if r["target_supervised_tokens"] == 2_960_000), None)
    if rung is None:
        return [f"arm {arm}: the pack has no 2.96M rung"]
    return [f"arm {arm}: rung {k} {rung.get(k)} != expected {v}" for k, v in pack["expected"].items()
            if rung.get(k) != v]


def launch_refusals(plan: dict, units: list[dict], *, validation: bool, pack_root: Path) -> list[str]:
    out = []
    if not validation:
        if plan["config"].get("frozen") is not True:
            out.append("h1_units.json is not frozen: a formal launch needs the frozen protocol and its authorization")
        if str(plan["schedule"].get("status", "")).upper().startswith("PROVISIONAL"):
            out.append("the schedule is PROVISIONAL (h1_accounting.json :: schedule); only --validation may run it")
    for arm in sorted({u["arm"] for u in units}):
        out += pack_refusals(plan["packs"][arm], arm, pack_root)
    return out


def derive_recipe(frozen: dict, *, run_name: str, purpose: str, out_dir: str, data_dir: str,
                  student_path: str, seed: int, sched: dict, overrides: list[str]) -> dict:
    d = copy.deepcopy(frozen)
    d.update(run_name=run_name, _purpose=purpose, out_dir=out_dir, data_dir=data_dir,
             student_path=student_path, seed=int(seed), rung=sched["rung"])
    d["schedule"].update(total_steps=sched["total_steps"], warmup_steps=sched["warmup_steps"])
    d["batch"]["blocks_per_step"] = sched["blocks_per_step"]
    d["checkpoint"]["save_every"] = sched["save_every"]
    d["intervals"]["eval_every"] = sched["eval_every"]
    a, b = _flat(frozen), _flat(d)
    outside = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in overrides)
    if outside:
        raise DriverRefusal(f"the derived recipe differs from the frozen one outside the declared overrides: {outside}")
    return d


def recipe_identity(derived: dict) -> str:
    return _json_sha({k: v for k, v in derived.items() if k not in WHERE})


def unit_inputs(plan: dict, u: dict, sched: dict, recipe_sha: str, validation: bool) -> dict:
    ev = plan["config"]["evaluation"]
    return {"arm": u["arm"], "seed": u["seed"], "formal": not validation,
            "initialization_artifact_digest": plan["init"]["artifact_digest"],
            "initialization_weights_digest": plan["init"]["weights_digest"],
            "pack_blocks_sha256": plan["packs"][u["arm"]]["blocks_sha256"],
            "recipe_sha256": recipe_sha, "schedule": sched,
            "evaluation": {k: ev[k] for k in ("set", "trained_context", "action_protocol")}}


# ---- state, from the durable root only ------------------------------------------------------------

def unit_state(root: Path, unit: str, inputs: dict) -> tuple[str, str]:
    try:
        ckpt = PU.completed(root, unit, "checkpoint")
        evaluation = PU.completed(root, unit, "evaluation")
        report = PU.completed(root, unit, "report")
    except Exception as exc:          # a record this driver did not write completes nothing
        return "mismatch", str(exc)
    if ckpt is not None and ckpt.get("inputs") != json.loads(json.dumps(inputs)):
        return "mismatch", "the persisted checkpoint was trained from other inputs"
    if ckpt is None and (evaluation or report):
        return "mismatch", "evaluation or report records without a persisted checkpoint"
    if report is not None:
        return ("scored", "") if evaluation is not None else ("mismatch", "a report without its evaluation")
    if evaluation is not None:
        return "evaluated", ""
    return ("trained", "") if ckpt is not None else ("untrained", "")


def attempts(root: Path, unit: str) -> list[dict]:
    p = Path(root) / unit / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def _append(root: Path, unit: str, row: dict) -> None:
    p = Path(root) / unit / "attempts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def bring_home(root: Path, unit: str, dest: Path) -> list[str]:
    """A scored unit's evidence -- never its weights (P8.4)."""
    src = Path(root) / unit
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for p in sorted(src.iterdir()):
        if p.name in ("checkpoint", "eval_progress") or p.name.endswith((".partial", ".inflight.json")):
            continue
        (shutil.copytree(p, dest / p.name, dirs_exist_ok=True) if p.is_dir() else shutil.copy2(p, dest / p.name))
        got.append(p.name)
    return got


def local_trained(work: Path, unit: str, inputs: dict) -> Path | None:
    """A model a previous attempt on this machine finished training, for these inputs."""
    for t in sorted((work / unit).glob("attempt-*/trained.json"), reverse=True):
        doc = _read(t)
        if doc.get("inputs") == json.loads(json.dumps(inputs)) and Path(doc["model_dir"]).is_dir():
            return Path(doc["model_dir"])
    return None


def resumable(root: Path, unit: str, inputs: dict) -> Path | None:
    """The latest durable training attempt of `unit` with a save point, for these inputs."""
    for d in sorted((Path(root) / unit / "train").glob("a*"), key=lambda p: int(p.name[1:]), reverse=True):
        meta, latest = d / "train_inputs.json", d / "checkpoints/latest.txt"
        if meta.is_file() and latest.is_file() and (d / "recovery_config.json").is_file() \
                and _read(meta).get("inputs") == json.loads(json.dumps(inputs)) \
                and (d / "checkpoints" / latest.read_text().strip() / "trainer_state.pt").is_file():
            return d
    return None


def release_save_points(root: Path, unit: str) -> list[dict]:
    """The resume state, once the final checkpoint is durable: nothing reads it again."""
    out = []
    for d in sorted((Path(root) / unit / "train").glob("a*/checkpoints")):
        out.append({"path": str(d), "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())})
        shutil.rmtree(d)
    return out


# ---- the operations a unit performs (injected in tests) ----------------------------------------------

@dataclass
class Ops:
    restore_init: Callable[[Path, dict], Path]
    train: Callable[[dict], Path]
    evaluate: Callable[[dict], tuple[dict, dict]]
    adapter: Callable[[], object]
    commit: Callable[[], str]
    env: dict = field(default_factory=dict)


def _qwen3_adapter():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    register_builtin_adapters()
    return get_adapter("qwen3")


def _verified(model_dir: Path, ident: dict, adapter) -> bool:
    from aadistill.runtime.leaf_durability import verify_transferred_leaf
    v = verify_transferred_leaf(model_dir, ident, adapter=adapter)
    return bool(v["matched"] and v["weights_digest_matched"] and v["config_matched"] is not False)


def restore_init(init_dir: Path, ident: dict) -> Path:
    """The fixed initialization, its identity re-proved from the bytes."""
    from stages.reasoning_data import restore_initialization as RI
    adapter = _qwen3_adapter()
    if init_dir.is_dir() and _verified(init_dir, ident, adapter):
        return init_dir
    token = os.environ.get("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    files = RI.download(init_dir, token)
    if not all(f["matched"] for f in files) or not _verified(init_dir, ident, adapter):
        raise IdentityMismatch(f"{init_dir}: the restored initialization does not re-identify")
    return init_dir


def train(job: dict) -> Path:
    from stages.phase_c1.autoinit_c1_driver import TRAINER, trained_model_dir
    cfg = job["attempt_dir"] / "recovery_config.json"
    cfg.write_text(json.dumps(job["recipe"], indent=2) + "\n")
    with (job["attempt_dir"] / "recovery.log").open("a") as lf:
        rc = subprocess.run([TRAIN_PY, str(TRAINER), "--config", str(cfg)]
                            + (["--resume", "latest"] if job.get("resume") else []), cwd=REPO, env=job["env"],
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"the trainer exited {rc}; see recovery.log")
    return trained_model_dir(Path(job["recipe"]["out_dir"]) if Path(job["recipe"]["out_dir"]).is_absolute()
                             else REPO / job["recipe"]["out_dir"])


def evaluate(job: dict) -> tuple[dict, dict]:
    from stages.calibration_volume.dose_driver import evaluate_all
    out, ev = job["out"], job["evaluation"]
    out.mkdir(parents=True, exist_ok=True)
    job["progress"].mkdir(parents=True, exist_ok=True)
    prereg = {"evaluation": {"trained_context": ev["trained_context"], "action_protocol": ev["action_protocol"]}}
    rep, _ = evaluate_all(job["model_dir"], out, job["progress"], job["label"], prereg, job["env"])
    (out / "formal_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    files = {p.relative_to(out).as_posix(): p for sub in ("generations", "rows")
             for p in sorted((out / sub).rglob("*")) if p.is_file()}
    files |= {n: out / n for n in ("formal_report.json", "eval_fluid.log", "eval_capability.log")
              if (out / n).is_file()}
    return rep, files


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip())
        return f"{sha}{'+dirty' if dirty else ''}"
    except OSError:
        return "unknown"


PRODUCTION = Ops(restore_init=restore_init, train=train, evaluate=evaluate, adapter=_qwen3_adapter,
                 commit=_git_commit,
                 env={**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"})


# ---- one unit --------------------------------------------------------------------------------------

def run_unit(plan: dict, u: dict, *, root: Path, work: Path, pack_root: Path, init_dir: Path,
             run_id: str, validation: bool, validation_steps: int | None, capacity: int | None,
             ops: Ops) -> dict:
    unit = u["unit"]
    sched = schedule_for(plan, u["arm"], validation_steps)
    overrides = plan["config"]["recipe"]["overrides"]
    probe = derive_recipe(plan["recipe"], run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-",
                          seed=u["seed"], sched=sched, overrides=overrides)
    inputs = unit_inputs(plan, u, sched, recipe_identity(probe), validation)
    state, why = unit_state(root, unit, inputs)
    history = attempts(root, unit)
    n = 1 + max((a["attempt"] for a in history), default=0)
    base = {"attempt": n, "run_id": run_id, "commit": ops.commit(), "state_at_start": state, "formal": not validation}
    if state == "mismatch":
        _append(root, unit, {**base, "event": "end", "action": "stop", "outcome": "identity_mismatch",
                             "error": why, "utc": _now()})
        raise IdentityMismatch(f"{unit}: {why}; preserved, nothing consumed")
    if state == "scored":
        home = bring_home(root, unit, work / unit / "home")
        _append(root, unit, {**base, "event": "end", "action": "skip", "outcome": "complete", "utc": _now()})
        return {"unit": unit, "outcome": "complete", "action": "skipped", "brought_home": home}

    attempt_dir = work / unit / f"attempt-{n}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    unit_dir = Path(root) / unit
    #: a training attempt with no "complete" end row failed -- raised, or died with its machine
    trained_starts = {a["attempt"] for a in history if a.get("event") == "start" and a.get("action") == "train"}
    failed_training = sorted(trained_starts - {a["attempt"] for a in history
                                               if a.get("event") == "end" and a.get("outcome") == "complete"})
    local = local_trained(work, unit, inputs) if state == "untrained" else None
    saved = resumable(root, unit, inputs) if state == "untrained" and local is None else None
    action = {"trained": "evaluate", "evaluated": "report"}.get(
        state, "persist_local" if local else "resume" if saved else "train")
    _append(root, unit, {**base, "event": "start", "action": action, "utc": _now(),
                         "replacement_of": failed_training[-1] if action in ("train", "resume") and failed_training
                         else None, "resumed_from": str(saved) if saved else None})
    stage = action
    try:
        model_dir = None
        if action in ("train", "resume", "persist_local"):
            train_dir = saved
            if action == "train":
                stage = "restore_initialization"
                init = ops.restore_init(init_dir, plan["init"])
                stage = "train"
                train_dir = unit_dir / "train" / f"a{n}"
                recipe = derive_recipe(
                    plan["recipe"], run_name=f"h1_{unit}_{run_id}_a{n}",
                    purpose=f"H1 {'GPU VALIDATION (not formal)' if validation else 'formal'} unit {unit}, attempt {n}",
                    out_dir=str(train_dir), data_dir=str(Path(pack_root) / u["arm"]),
                    student_path=str(init), seed=u["seed"], sched=sched, overrides=overrides)
                train_dir.mkdir(parents=True, exist_ok=True)
                (train_dir / "recovery_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
                (train_dir / "train_inputs.json").write_text(json.dumps({"inputs": inputs, "attempt": n}, indent=1) + "\n")
                model_dir = ops.train({"unit": unit, "attempt_dir": attempt_dir, "recipe": recipe, "env": ops.env})
            elif action == "resume":
                stage = "resume"
                recipe = _read(saved / "recovery_config.json")
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
            persist_training_records(unit, model_dir, root, work, capacity, train_dir=train_dir,
                                     attempt_dir=attempt_dir)
            if not _verified(unit_dir / "checkpoint", PU.completed(root, unit, "checkpoint"), ops.adapter()):
                raise IdentityMismatch(f"{unit_dir / 'checkpoint'}: the persisted checkpoint does not re-identify")
            stage = "release_save_points"
            _append(root, unit, {**base, "event": "released", "save_points": release_save_points(root, unit),
                                 "utc": _now()})
        elif state == "trained" and PU.completed(root, unit, "training") is None:
            stage = "persist_training_records"     # a previous attempt persisted the weights, not its records
            mine = local_trained(work, unit, inputs)
            if mine is not None:
                persist_training_records(unit, mine, root, work, capacity)
        if action != "report":
            if model_dir is None:
                stage = "verify_checkpoint"
                model_dir = unit_dir / "checkpoint"
                if not _verified(model_dir, PU.completed(root, unit, "checkpoint"), ops.adapter()):
                    raise IdentityMismatch(f"{model_dir}: the persisted checkpoint no longer re-identifies")
            stage = "evaluate"
            rep, files = ops.evaluate({"unit": unit, "model_dir": model_dir, "out": attempt_dir / "evaluation",
                                       "progress": unit_dir / "eval_progress", "label": f"h1_{unit}_{run_id}",
                                       "evaluation": plan["config"]["evaluation"], "env": ops.env})
            stage = "persist_evaluation"
            PU.persist_files(unit, files, root, label="evaluation", capacity=capacity)
        else:
            rep = _read(unit_dir / "evaluation/formal_report.json")
        stage = "report"
        ckpt, ev = PU.completed(root, unit, "checkpoint"), PU.completed(root, unit, "evaluation")
        report = {"schema": REPORT_SCHEMA, "unit": unit, "arm": u["arm"], "seed": u["seed"],
                  "formal": not validation, "run_id": run_id, "attempt": n, "written_utc": _now(),
                  "inputs": inputs, "config_sha256": plan["config_sha256"],
                  "initialization_record_sha256": plan["init_record_sha256"],
                  "checkpoint": {k: ckpt[k] for k in ("artifact_digest", "weights_digest", "destination")},
                  "evaluation_files": len(ev["files"]),
                  "evaluation_descriptor_sha256": _sha(unit_dir / "evaluation.descriptor.json"),
                  "attempts": [a for a in attempts(root, unit) if a.get("event") == "start"],
                  "formal_report": rep,
                  "not": None if not validation else "a GPU validation: never a formal H1 result"}
        (attempt_dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        PU.persist_files(unit, {"report.json": attempt_dir / "report.json"}, root, label="report", capacity=capacity)
    except BaseException as exc:
        from aadistill.infrastructure.failure_signature import signature_of
        text = f"{type(exc).__name__}: {exc}"
        _append(root, unit, {**base, "event": "end", "action": action, "outcome": "failed", "stage": stage,
                             "error": text[:800], "signature": signature_of(traceback.format_exc()), "utc": _now()})
        raise
    _append(root, unit, {**base, "event": "end", "action": action, "outcome": "complete", "utc": _now()})
    return {"unit": unit, "outcome": "complete", "action": action, "attempt": n}


def persist_training_records(unit: str, model_dir: Path, root: Path, work: Path, capacity: int | None, *,
                             train_dir: Path | None = None, attempt_dir: Path | None = None) -> None:
    """The derived config, the trainer log and the trainer's own records of the
    attempt that trained `model_dir`, under logical names: the attempt's local
    directory (config, the trainer's console log) and the trainer's output
    directory on the backend (its jsonl, manifests and completion records)."""
    if attempt_dir is None:
        attempt_dir = next((a for a in (work / unit).glob("attempt-*") if a in Path(model_dir).parents), None)
    if train_dir is None and attempt_dir is not None and (attempt_dir / "recovery").is_dir():
        train_dir = attempt_dir / "recovery"
    if train_dir is None and len(Path(model_dir).parents) > 2 \
            and (Path(model_dir).parents[2] / "recovery_config.json").is_file():
        train_dir = Path(model_dir).parents[2]          # <train>/checkpoints/<tag>/model
    records = {} if attempt_dir is None else {
        p.name: p for p in sorted(attempt_dir.glob("*")) if p.is_file() and p.name != "trained.json"}
    if train_dir is not None:
        records |= {f"trainer/{p.name}": p for p in sorted(Path(train_dir).glob("*")) if p.is_file()}
    if records:
        PU.persist_files(unit, records, root, label="training", capacity=capacity)


# ---- the run -------------------------------------------------------------------------------------

def close_observers() -> int:
    for p in _OBSERVERS:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    _OBSERVERS.clear()
    return len(_OBSERVERS)


def run(argv=None, *, ops: Ops = PRODUCTION, plan: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--pack-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/packs"))
    ap.add_argument("--init-dir", default=str(REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/"
                                                     "1301deb71685aa1ca5c7b6cf2c491ac4"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--unit", action="append", default=None)
    ap.add_argument("--capacity-bytes", type=int, default=None)
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--validation-steps", type=int, default=None)
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args(argv)
    if args.validation_steps and not args.validation:
        ap.error("--validation-steps is a GPU-validation knob; it needs --validation")
    work = Path(args.work or REPO / "artifacts/stages/stage-2/reasoning_data/units" / args.run_id)
    status = Path(args.status or work / "status.json")
    root = Path(args.durable_root) / ("validation" if args.validation else "")
    doc = {"schema": "aadistill.stage2.reasoning_data.h1_driver_status/v1", "run_id": args.run_id,
           "formal": not args.validation, "started_utc": _now(), "units": {}, "terminal": None}
    code = 1
    try:
        plan = plan or load_plan(Path(args.config))
        known = {u["unit"]: u for u in plan["units"]}
        unknown = sorted(set(args.unit or []) - set(known))
        if unknown:
            raise DriverRefusal(f"unknown units {unknown}; the plan has {sorted(known)}")
        units = [known[k] for k in (args.unit or known)]
        refusals = launch_refusals(plan, units, validation=args.validation, pack_root=Path(args.pack_root))
        if refusals:
            doc.update(terminal="REFUSED", refusals=refusals)
            code = 2
            return code
        for u in units:
            doc["units"][u["unit"]] = {"outcome": "running"}
            try:
                doc["units"][u["unit"]] = run_unit(
                    plan, u, root=root, work=work, pack_root=Path(args.pack_root), init_dir=Path(args.init_dir),
                    run_id=args.run_id, validation=args.validation, validation_steps=args.validation_steps,
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
        doc.update(finished_utc=_now(), exit_code=code, observers_open=close_observers(),
                   teardown="the launcher's: this driver creates and deletes no provider resource",
                   terminal=doc["terminal"] or "FAILED")
        status.parent.mkdir(parents=True, exist_ok=True)
        tmp = status.with_name(f".{status.name}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        os.replace(tmp, status)


if __name__ == "__main__":
    raise SystemExit(run())

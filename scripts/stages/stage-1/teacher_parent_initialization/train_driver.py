#!/usr/bin/env python3
"""On-pod driver of A_teacher_parent's two Stage-3 units (session 2 of the round): each seed trained from A's
initialization with the A-MIX-6cat-sr recipe, then evaluated on eval-set-1 in both modes, seed 20260726 first.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/teacher_parent_initialization/train_driver.py \\
        --authorization <a.json> --out <evidence dir> --status <file> --unit-deadline-hours <h> \\
        --vllm-python /opt/vllm/bin/python [--resume <secured evidence dir or archive>]

The units are `plan.py`'s (`logs/stages/stage-1/teacher_parent_initialization/plans/training_plan.json`): A-MIX's
configs with A's initialization. The state machine is A-MIX's (`teacher_modes.a1_units`: `state`, `train`,
`train_telemetry`, `final_model`; the launch admission `training_plan.admit` before any step; the declared
``cross_context`` attention checked on the pod's bytes and in the trainer's own log), on the container disk, with
three durability duties a network volume used to cover:

* A's initialization is fetched from its verified relay copy and re-hashed before admission;
* every save point the trainer completes (``checkpoints/latest.txt`` names it last) is copied to the relay by a
  watcher thread -- weights, trainer state (AdamW, step, consumed sequences, RNG) and resume identity -- downloaded
  again and re-hashed; the unit's previous save point is then permanently deleted from the relay. The records are
  ``<out>/<unit>/savepoints.json``, pulled by the poll, so a replacement session (``--resume``) restores the latest
  one and the trainer resumes it (``--resume latest``): never a retrain of finished steps;
* the final weights go to the relay (`final.json`) before the unit's evaluation starts.

Evaluation is B's protocol exactly (`design.json :: evaluation`): the package built as A-MIX built it (the hybrid's
tokenizer files), then `reasoning_inheritance.driver.fluid` / `capability` with this design's protocol passed in --
the same runners and flags as `dose_driver.evaluate_all` (their defaults are the system message and check interval
written here), plus the streaming that seals each finished sample. A later attempt generates only the samples with no
sealed record. The formal report is the registered scorer's (`eval_set_scoring.report`), as A-MIX's.

Evidence (small) goes under ``--out``: ``<unit>/train/`` (the trainer's log, config, completion record and stdout
tail), ``<unit>/evaluation_<mode>/`` (sealed parts, generations, rows, formal report), ``<unit>/savepoints.json``,
``<unit>/final.json`` and ``units.json``. Weights stay under `plan.POD_ROOT` and on the relay.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/teacher_parent_initialization/design.json"
PLAN = REPO / "logs/stages/stage-1/teacher_parent_initialization/plans/training_plan.json"
SCHEMA = "aadistill.stage1.teacher_parent_initialization.units/v1"
WATCH_EVERY_S = 60
SMALL_TRAIN_FILES = ("train_log.jsonl", "config.json", "run_completion.json")


class DriverError(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.partial")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def protocol() -> dict:
    """The design's evaluation block in the shape `reasoning_inheritance.driver.fluid` / `capability` read."""
    ev = read_json(DESIGN)["evaluation"]
    return {"eval_set": ev["eval_set"], "decoding": ev["decoding"]["path"], "context": {"tokens": ev["context"]["tokens"]},
            "action_protocol": ev["action_protocol"], "system_message": ev["system_message"],
            "degeneration_check_every": ev["degeneration_check_every"],
            "modes": {m: {"template_kwargs": v["template_kwargs"]} for m, v in ev["modes"].items()}}


def mode_of(evaluation: dict) -> str:
    """The plan row's evaluation ('thinking' / 'non_thinking') as the design's mode key."""
    return {"thinking": "H-T", "non_thinking": "H-N"}[evaluation["mode"]]


# ---- the durable save points ------------------------------------------------------------------------------------------

def savepoint_files(ckpt: Path) -> dict[str, Path]:
    files = {f"model/{p.name}": p for p in sorted((ckpt / "model").iterdir()) if p.is_file()}
    for n in ("trainer_state.pt", "resume_identity.json"):
        if (ckpt / n).is_file():
            files[n] = ckpt / n
    return files


class SavepointMirror(threading.Thread):
    """Copies each save point the trainer completes to the relay, then removes the unit's previous one there."""

    def __init__(self, unit: str, out_dir: Path, ev_unit: Path, total_steps: int, prefix: str, every: float = WATCH_EVERY_S):
        super().__init__(daemon=True)
        self.unit, self.out_dir, self.ev_unit, self.total = unit, Path(out_dir), Path(ev_unit), int(total_steps)
        self.prefix, self.every = prefix, every
        self.stop = threading.Event()
        self.path = self.ev_unit / "savepoints.json"

    def records(self) -> list[dict]:
        return read_json(self.path) if self.path.is_file() else []

    def mirror_once(self) -> dict | None:
        from stages.teacher_parent_initialization import relay as R
        latest = self.out_dir / "checkpoints" / "latest.txt"
        if not latest.is_file():
            return None
        tag = latest.read_text().strip()
        step = int(tag.removeprefix("step_"))
        recs = self.records()
        if step >= self.total or any(r["tag"] == tag and r.get("verified") for r in recs):
            return None
        ckpt = self.out_dir / "checkpoints" / tag
        res = R.run_online("put", {"files": {k: str(v) for k, v in savepoint_files(ckpt).items()},
                                   "prefix": f"{self.prefix}/savepoints/{tag}", "scratch": str(self.out_dir),
                                   "message": f"{self.unit}: save point {tag}"})
        rec = {"tag": tag, "step": step, **res, "latest_txt": tag}
        recs.append(rec)
        if rec.get("verified"):
            for old in recs[:-1]:
                if old.get("verified") and not old.get("removed"):
                    old["removed"] = R.run_online("remove", {"record": old, "message": f"{self.unit}: {old['tag']} "
                                                             f"superseded by {tag}"})
        write_json(self.path, recs)
        return rec

    def copy_small(self) -> None:
        dest = self.ev_unit / "train"
        dest.mkdir(parents=True, exist_ok=True)
        for n in SMALL_TRAIN_FILES:
            if (self.out_dir / n).is_file():
                shutil.copyfile(self.out_dir / n, dest / n)
        for p in self.out_dir.glob("run_completion_resume_step*.json"):
            shutil.copyfile(p, dest / p.name)
        log = self.out_dir / "train_stdout.log"
        if log.is_file():
            (dest / "train_stdout.tail.log").write_bytes(log.read_bytes()[-200_000:])

    def run(self):
        while not self.stop.wait(self.every):
            try:
                self.copy_small()
                self.mirror_once()
            except Exception as exc:  # noqa: BLE001 -- the watcher never stops training; the error is recorded
                with (self.ev_unit / "savepoint_errors.log").open("a") as fh:
                    fh.write(f"{utc()} {type(exc).__name__}: {str(exc)[:500]}\n")

    def close(self):
        self.stop.set()
        self.join(timeout=60)
        self.copy_small()


def restore_savepoint(ev_unit: Path, out_dir: Path) -> dict | None:
    """The latest verified, unremoved save point of the unit's records into ``out_dir/checkpoints``; None if none."""
    from stages.teacher_parent_initialization import relay as R
    p = ev_unit / "savepoints.json"
    recs = [r for r in (read_json(p) if p.is_file() else []) if r.get("verified") and not r.get("removed")]
    if not recs:
        return None
    rec = max(recs, key=lambda r: r["step"])
    ckpt = out_dir / "checkpoints" / rec["tag"]
    got = R.run_online("get", {"record": rec, "dest": str(ckpt)}, timeout_s=3600)
    if not got.get("verified"):
        raise DriverError(f"save point {rec['tag']} did not restore: {str(got)[:500]}")
    (out_dir / "checkpoints" / "latest.txt").write_text(rec["tag"] + "\n")
    return {"tag": rec["tag"], "from": got.get("from"), "utc": utc()}


def restore_final(ev_unit: Path, out_dir: Path, steps: int) -> dict | None:
    """A trained unit's final weights from their verified relay copy, and its completion record from the evidence."""
    from stages.teacher_parent_initialization import relay as R
    p = ev_unit / "final.json"
    rec = read_json(p) if p.is_file() else None
    done = ev_unit / "train" / "run_completion.json"
    if not (rec and rec.get("verified") and done.is_file()):
        return None
    model = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
    got = R.run_online("get", {"record": rec, "dest": str(model)}, timeout_s=3600)
    if not got.get("verified"):
        raise DriverError(f"the final weights did not restore: {str(got)[:500]}")
    shutil.copyfile(done, out_dir / "run_completion.json")
    if (ev_unit / "train" / "train_log.jsonl").is_file() and not (out_dir / "train_log.jsonl").is_file():
        shutil.copyfile(ev_unit / "train" / "train_log.jsonl", out_dir / "train_log.jsonl")
    return {"from": got.get("from"), "utc": utc()}


def fetch_init(u: dict) -> dict:
    """A's initialization at the plan's student path, from its verified relay copy, every file re-hashed."""
    from stages.teacher_parent_initialization import relay as R
    rec = u["initialization"]["relay"]
    dest = Path(u["config"]["student_path"])
    sub = {k: v for k, v in rec["files_sha256"].items() if k.startswith("init/")}
    flat = {**rec, "files_sha256": {k.removeprefix("init/"): v for k, v in sub.items()},
            "prefix": f"{rec['prefix']}/init"}
    got = R.run_online("get", {"record": flat, "dest": str(dest)}, timeout_s=3600)
    if not got.get("verified"):
        raise DriverError(f"A's initialization did not restore: {str(got)[:500]}")
    want = u["config"]["lineage"]["initialization"]["model_safetensors_sha256"]
    if R.sha256_file(dest / "model.safetensors") != want:
        raise DriverError("A's initialization on the pod is not the plan's")
    return {"path": str(dest), "from": got.get("from"), "utc": utc()}


# ---- evaluation ---------------------------------------------------------------------------------------------------------

def package(model_dir: Path, dest: Path) -> dict:
    """The evaluation package as A-MIX built it (`dose_driver.evaluate_all`'s dict branch, hybrid tokenizer)."""
    from stages.phase_c1.packaging import build_evaluation_package
    from stages.teacher_modes import training_plan as TP
    ident = read_json(REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json")
    side = {n: ident["files_sha256"][n] for n in ("tokenizer.json", "tokenizer_config.json")}
    shutil.rmtree(dest, ignore_errors=True)
    build_evaluation_package(model_dir, tokenizer_source=str(REPO / TP.TOKENIZERS["hybrid"]), dest=dest,
                             expected_sidecar_sha256=side, sidecars=tuple(side))
    return {"model_dir": str(model_dir), "tokenizer_source": TP.TOKENIZERS["hybrid"], "sidecar_sha256": side}


def evaluate(model_dir: Path, ev_dir: Path, label: str, mode: str, vllm_py: str, attempt: str, work: Path) -> dict:
    """Every eval-set-1 unit in ``mode``, resumable by sample, then the formal report (`eval_set_scoring.report`)."""
    from stages.reasoning_inheritance import driver as RID
    import shared.evaluation.eval_set_scoring as S
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}
    ev = protocol()
    pkg = work / "packages" / f"{label}_{mode}"
    rec = {"package": package(model_dir, pkg)}
    write_json(ev_dir / "package.json", rec["package"])
    rec["fluid"] = RID.fluid(pkg, ev_dir, label, mode, vllm_py, env, attempt, None, ev)
    rec["capability"] = RID.capability(pkg, ev_dir, label, mode, vllm_py, env, attempt, None, ev)
    rep = S.report(ev_dir / "generations", label, {}, ev["eval_set"], rows_out=ev_dir / "rows")
    write_json(ev_dir / "formal_report.json", rep)
    return rec


# ---- the state machine ----------------------------------------------------------------------------------------------------

def take_over(resume: Path, out: Path) -> dict:
    """A previous attempt's secured evidence into this attempt's ``out``: save-point and final records, the trainer's
    small files, every sealed evaluation file (never a live stream); returns what was taken."""
    import tarfile
    if resume.is_file():
        unpacked = out.parent / ".resume"
        shutil.rmtree(unpacked, ignore_errors=True)
        with tarfile.open(resume) as t:
            t.extractall(unpacked, filter="data")
        resume = unpacked
    taken = []
    for p in sorted(resume.rglob("*")):
        rel = p.relative_to(resume)
        if not p.is_file() or p.name.startswith(".") or rel.parts[0] == "live":
            continue
        dest = out / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        taken.append(str(rel))
    return {"from": str(resume), "files": len(taken), "utc": utc()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--attempt", default=None)
    a = ap.parse_args(argv)
    out, status = Path(a.out), Path(a.status)
    out.mkdir(parents=True, exist_ok=True)
    attempt = a.attempt or time.strftime("a%Y%m%d_%H%M%S", time.gmtime())
    record = {"schema": SCHEMA, "started_utc": utc(), "attempt": attempt, "units": {}}
    evidence = out / "units.json"

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{utc()} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        write_json(evidence, record)
        write_json(out / "report.json", record)        #: what the session's closeout and products check read

    ok = True
    try:
        from stages.teacher_modes import a1_units as AU
        from stages.teacher_modes import packed_training_plan as PTP
        from stages.teacher_modes import training_plan as TP
        from stages.teacher_parent_initialization import launch as L
        from stages.teacher_parent_initialization import plan as PL
        from stages.teacher_parent_initialization import relay as R
        from shared import study_engineering as SE
        from shared.pod.teacher_cache import local_snapshot
        auth = L.TPAuthorization.load(a.authorization)
        record["authorization"] = SE.require_on_pod(auth, L.STUDY, SE.identity_hash(L.STUDY, REPO), str(REPO),
                                                    kind=L.TRAIN_KIND).get("authorization_id")
        if a.resume:
            record["resumed_from"] = take_over(Path(a.resume), out)
        plan = read_json(PLAN)
        us = sorted(plan["units"], key=lambda u: plan["seed_order"].index(u["seed"]))
        t = us[0]["config"]["teacher"]
        snap = Path(local_snapshot(t["model_id"], t["revision"]))
        record["tokenizer"] = TP.stage_tokenizers(snap)
        save()
        mark("CONTRACT_OK")
        for u in us:
            name, cfg = u["unit"], u["config"]
            row = record["units"].setdefault(name, {})
            ev_unit = out / name
            out_dir = Path(cfg["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            steps = int(cfg["schedule"]["total_steps"])
            evals = [{**e, "path": ev_unit / e["dir"]} for e in u["evaluations"]]
            st = AU.state(out_dir, [e["path"] for e in evals])
            if st == "fresh":                   #: a replacement pod: the final weights, else the latest save point
                got = restore_final(ev_unit, out_dir, steps)
                if got:
                    row["restored_final"] = got
                else:
                    got = restore_savepoint(ev_unit, out_dir)
                    if got:
                        row["restored_savepoint"] = got
                st = AU.state(out_dir, [e["path"] for e in evals])
            row["state_at_start"] = st
            save()
            if st == "done":
                continue
            if st in ("fresh", "resume"):
                if not Path(cfg["student_path"], "model.safetensors").is_file():
                    row["init"] = fetch_init(u)
                bad = PTP.attention_problems(cfg, REPO)
                if bad:
                    raise DriverError(f"{name}: the attention is not the declared one: {bad}")
                TP.admit([{"unit": name, "config": cfg}], snapshots={t["model_id"]: snap})
                row["admitted_utc"] = utc()
                cfg_path = out_dir / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
                if st == "resume":
                    row["resume_from"] = (out_dir / "checkpoints" / "latest.txt").read_text().strip()
                save()
                mirror = SavepointMirror(name, out_dir, ev_unit, steps, f"{R.ROOT_PREFIX}/{PL.ARM}/{name}")
                mirror.start()
                mark(f"TRAIN_START:{name}")
                t0 = time.time()
                try:
                    rc = AU.train(cfg_path, resume=st == "resume", deadline_s=a.unit_deadline_hours * 3600,
                                  log=out_dir / "train_stdout.log")
                finally:
                    mirror.close()
                row.update(train_rc=rc, train_seconds=round(time.time() - t0, 1), telemetry=AU.train_telemetry(out_dir))
                save()
                if rc != 0:
                    mirror.mirror_once()        #: the last completed save point is durable before the session stops
                    raise DriverError(f"{name}: training exited {rc} ({'deadline' if rc == 124 else 'failure'})")
                if row["telemetry"].get("context_strategies") != [PTP.CONTEXT_STRATEGY]:
                    raise DriverError(f"{name}: the trainer logged {row['telemetry'].get('context_strategies')}")
                mark(f"TRAINED:{name}")
            model_dir = AU.final_model(out_dir, steps)
            row["final_model_safetensors_sha256"] = R.sha256_file(model_dir / "model.safetensors")
            fin = ev_unit / "final.json"
            prev = read_json(fin) if fin.is_file() else {}
            if not (prev.get("verified") and prev["files_sha256"].get("model.safetensors")
                    == row["final_model_safetensors_sha256"]):
                files = {n: str(model_dir / n) for n in AU.FINAL_FILES if (model_dir / n).is_file()}
                prev = R.run_online("put", {"files": files, "scratch": str(out_dir),
                                            "prefix": f"{R.ROOT_PREFIX}/{PL.ARM}/{name}/final/"
                                                      f"{row['final_model_safetensors_sha256'][:16]}",
                                            "message": f"{name}: final weights"})
                write_json(fin, prev)
                if not prev.get("verified"):
                    raise DriverError(f"{name}: the final weights' relay copy did not verify: {str(prev)[:500]}")
                for sp in [r for r in SavepointMirror(name, out_dir, ev_unit, steps, "").records()
                           if r.get("verified") and not r.get("removed")]:
                    sp["removed"] = R.run_online("remove", {"record": sp, "message": f"{name}: {sp['tag']} superseded "
                                                            "by the verified final weights"})
                    recs = [x if x["tag"] != sp["tag"] else sp for x in read_json(ev_unit / "savepoints.json")]
                    write_json(ev_unit / "savepoints.json", recs)
            row["final_relay"] = {k: prev.get(k) for k in ("relay", "prefix", "verified")}
            save()
            mark(f"FINAL_DURABLE:{name}")
            for e in evals:
                if (e["path"] / "formal_report.json").is_file():
                    continue                    #: a mode an earlier session scored is never measured again
                mode = mode_of(e)
                mark(f"EVAL_START:{name}:{e['mode']}")
                t0 = time.time()
                row.setdefault("evaluations", {})[e["mode"]] = {
                    **evaluate(model_dir, e["path"], name, mode, a.vllm_python, attempt, out_dir.parent / "work"),
                    "eval_seconds": round(time.time() - t0, 1), "template": e["template"]}
                save()
                mark(f"EVAL_DONE:{name}:{e['mode']}")
            mark(f"UNIT_DONE:{name}")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    save()
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

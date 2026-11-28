#!/usr/bin/env python3
"""On-pod driver of SOB1's phase 2 (session kind ``sob1_train``): one arm's two seeds trained with A-MIX's recipe from
the arm's initialization (`plan.py`), each seed's final weights made durable, then each evaluated at 8,192 tokens in
both modes, every set bound to that seed's final weights.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/single_operator_budget/train_driver.py --authorization <a.json> \\
        --out <evidence> --status <file> --plan logs/stages/stage-1/single_operator_budget/plans/training_plan_<arm>.json \\
        --prefix-root durable/stage1_single_operator_budget --unit-deadline-hours <h> [--session-deadline-utc <t>] \\
        --vllm-python /opt/vllm/bin/python --attempt <run id> [--resume <secured evidence>]

Per unit (one arm x one seed), in the plan's ``unit_order`` -- never deciding by the existence of a file:

1. **The final weights.** The unit's verified final record: ``<unit>/final.json`` in the evidence, else the one
   pointer the relay lists under ``<prefix>/final`` (`teacher_parent_initialization.train_driver.final_record`; an
   unreadable listing with no record in the evidence refuses, never reads as "no final").
2. **Done** only when that final is durable and published (`completion.final_problems`) AND both sets are complete
   against ITS ``model.safetensors`` sha256 (`completion.check`: coverage, records, protocol, the formal report
   rescored, every sealed part's generation naming this arm, seed, unit, mode, the registered protocol and these
   weights). A ``{}`` report, a part another seed's generation sealed, or a missing final is never done.
3. **Training** only when no final exists anywhere: the latest save point restored (or none: step 0), the
   initialization fetched and re-hashed, the launch admission, the declared ``cross_context`` attention, the trainer
   with its first ``durability.memory_probe_steps`` optimizer steps memory-probed (they are ordinary training steps:
   a successful step is kept), the save-point rotation on the relay (`SavepointMirror`). A final already on this pod
   (its completion record and weights) is secured without training again.
4. **The final secured**: uploaded with the trainer's completion record, re-hashed, published (handoff), and only then
   the save points retired; a unit never holds two finals.
5. **Evaluation** per mode (`completion.complete_set`): a complete set is taken as it is; a ``{}`` or stale report is
   rescored from the sealed generations with no model loaded; parts whose generation, weights, arm, seed, mode or
   protocol cannot be shown are isolated; only the items no proven part holds are generated, from the final weights
   (on this pod, else restored from their relay copy -- never retrained).

The session ends ``SESSION_COMPLETE`` only when `completion.session` holds on the evidence: both finals durable and the
four sets complete. Evidence (small) goes under ``--out``: ``<unit>/train/``, ``<unit>/savepoints.json``,
``<unit>/final.json``, ``<unit>/evaluation_<mode>/`` and ``units.json`` / ``report.json``. Weights stay under the
plan's pod root and on the relay.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_parent_initialization import train_driver as TD  # noqa: E402

SCHEMA = "aadistill.stage1.single_operator_budget.train_units/v1"
read_json, write_json, utc = TD.read_json, TD.write_json, TD.utc


class DriverError(RuntimeError):
    pass


def modes_of(u: dict) -> list[str]:
    """The plan's evaluations of a unit as the design's modes (by directory), refusing any other."""
    from stages.single_operator_budget import common as C
    by_dir = {v: k for k, v in C.mode_dirs().items()}
    got = [by_dir.get(e["dir"]) for e in u["evaluations"]]
    if None in got or sorted(got) != sorted(C.mode_dirs()):
        raise DriverError(f"{u['unit']}: evaluations {[e['dir'] for e in u['evaluations']]} are not the design's "
                          f"{sorted(C.mode_dirs().values())}")
    return got


def authorized_plan_problems(sc: dict, plan: dict, plan_path: Path, kind: str) -> list[str]:
    """The artifact funds ONE arm's plan: its kind, arm, plan sha256 and init record sha256 must be the plan this driver
    loaded (`launch.kind_conditions` wrote them at issue). Checked before anything is fetched or trained."""
    from stages.teacher_parent_initialization import relay as R
    got = {"kind": kind, "arm": plan.get("arm"), "plan_sha256": R.sha256_file(plan_path),
           "init_record_sha256": R.sha256_file(REPO / plan["init_record"]["path"])}
    return [f"{k}: the authorization binds {sc.get(k)!r}, the driver loaded {v!r}" for k, v in got.items()
            if sc.get(k) != v]


def final_here(out_dir: Path, cfg: dict, steps: int) -> Path | None:
    """The final weights on this pod: the trainer's completion record of this config's steps and the model file."""
    from stages.teacher_modes import a1_units as AU
    try:
        AU.completion(out_dir, config=cfg, steps=steps)
        return AU.final_model(out_dir, steps)
    except (AU.CompletionError, RuntimeError):
        return None


def train_unit(u: dict, row: dict, ev_unit: Path, out_dir: Path, unit_prefix: str, dur: dict, a, snap: Path,
               mark, save, stage_init=None) -> Path:
    """Step 3 of the module docstring; returns the final model directory. ``stage_init(u, dur)`` places the unit's
    initialization at its student path when it is not there (`teacher_parent_initialization.train_driver.fetch_init`
    by default; another study binding this chain passes its own, e.g. one that also writes an execution config)."""
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_modes import packed_training_plan as PTP
    from stages.teacher_modes import training_plan as TP
    name, cfg = u["unit"], u["config"]
    steps = int(cfg["schedule"]["total_steps"])
    if not (out_dir / "checkpoints" / "latest.txt").is_file():
        got = TD.restore_savepoint(ev_unit, out_dir, unit_prefix, dur)
        if got:
            row["restored_savepoint"] = got
    resume = (out_dir / "checkpoints" / "latest.txt").is_file()
    if not Path(cfg["student_path"], "model.safetensors").is_file():
        row["init"] = (stage_init or TD.fetch_init)(u, dur)
    bad = PTP.attention_problems(cfg, REPO)
    if bad:
        raise DriverError(f"{name}: the attention is not the declared one: {bad}")
    t = cfg["teacher"]
    TP.admit([{"unit": name, "config": cfg}], snapshots={t["model_id"]: snap})
    row["admitted_utc"] = utc()
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    if resume:
        row["resume_from"] = (out_dir / "checkpoints" / "latest.txt").read_text().strip()
    sizes = u.get("bytes") or {}
    row["training_deadline"] = TD.training_deadline_s(a.unit_deadline_hours, a.session_deadline_utc,
                                                      float(sizes.get("save_point", 0)), float(sizes.get("final", 0)),
                                                      dur)
    save()
    if row["training_deadline"]["seconds"] <= 0:
        raise DriverError(f"{name}: the session cannot secure a save point before its hard deadline "
                          f"({row['training_deadline']}); not starting the trainer")
    mirror = TD.SavepointMirror(name, out_dir, ev_unit, steps, unit_prefix, d=dur)
    mirror.start()
    mark(f"TRAIN_START:{name}")
    t0 = time.time()
    #: execution only: the trainer logs each phase's memory for its first steps (`memory_probe`) and the phase of an
    #: out-of-memory error (`memory_oom`); the probed steps are formal training steps and stay in the run
    os.environ["AADISTILL_MEMORY_PROBE_STEPS"] = str(int(dur.get("memory_probe_steps") or 0))
    try:
        rc = AU.train(cfg_path, resume=resume, deadline_s=row["training_deadline"]["seconds"],
                      log=out_dir / "train_stdout.log")
    finally:
        mirror.close()
    row.update(train_rc=rc, train_seconds=round(time.time() - t0, 1), telemetry=AU.train_telemetry(out_dir),
               memory=TD.memory_telemetry(out_dir))
    save()
    if rc != 0:
        mirror.mirror_once()                    #: the last completed save point is durable before the session stops
        oom = row["memory"].get("oom_phase")
        raise DriverError(f"{name}: training exited {rc} ("
                          f"{'deadline' if rc == 124 else f'out of memory in {oom}' if oom else 'failure'})")
    if row["telemetry"].get("context_strategies") != [PTP.CONTEXT_STRATEGY]:
        raise DriverError(f"{name}: the trainer logged {row['telemetry'].get('context_strategies')}")
    mark(f"TRAINED:{name}")
    model = final_here(out_dir, cfg, steps)
    if model is None:
        raise DriverError(f"{name}: training exited 0 without this config's completion record and final weights")
    return model


def secure_final(name: str, model_dir: Path, out_dir: Path, cfg: dict, ev_unit: Path, unit_prefix: str,
                 dur: dict, row: dict) -> dict:
    """Step 4: the final weights on the relay with the trainer's completion record, re-hashed and published."""
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_parent_initialization import relay as R
    steps = int(cfg["schedule"]["total_steps"])
    try:
        done, _ = AU.completion(out_dir, config=cfg, steps=steps)
    except AU.CompletionError as exc:
        raise DriverError(f"{name}: {exc}") from exc
    sha = R.sha256_file(model_dir / "model.safetensors")
    row["completion_record"], row["final_model_safetensors_sha256"] = done.name, sha
    fin = ev_unit / "final.json"
    prev = read_json(fin) if fin.is_file() else {}
    if prev.get("verified") and prev["files_sha256"].get("model.safetensors") != sha:
        raise DriverError(f"{name}: a verified final {prev['files_sha256']['model.safetensors'][:12]} exists and this "
                          f"pod holds other final weights {sha[:12]}; a unit never holds two finals")
    if not prev.get("verified"):
        row["removals_before_final"] = TD.retry_pending_removals(ev_unit, name)     #: room first, then the upload
        if TD.savepoint_copies(dur) == 1:
            row["single_copy_removed_before_final"] = TD.remove_before_upload(ev_unit, name, "final")
        files = {n: str(model_dir / n) for n in AU.FINAL_FILES if (model_dir / n).is_file()}
        files[done.name] = str(done)        #: the training's completion travels with its weights
        nbytes = sum(Path(f).stat().st_size for f in files.values())
        prev = R.run_online("put", {"files": files, "scratch": str(out_dir), "prefix": f"{unit_prefix}/final/{sha[:16]}",
                                    "message": f"{name}: final weights"},
                            timeout_s=TD.transfer_seconds(nbytes, dur) + 600)
        write_json(fin, prev)
        if not prev.get("verified"):
            raise DriverError(f"{name}: the final weights' relay copy did not verify: {str(prev)[:500]}")
    publish(name, prev, out_dir, ev_unit, unit_prefix, dur, row)
    return prev


def publish(name: str, rec: dict, scratch: Path, ev_unit: Path, unit_prefix: str, dur: dict, row: dict) -> None:
    """The final's record published where a replacement attempt reads it (only the pointer is retried, the weights
    are never sent again); the save points retired only once it is."""
    if not (rec.get("handoff") or {}).get("verified"):
        rec["handoff"] = TD.hand_off_with_retries(rec, scratch, f"{unit_prefix}/final", f"{name}: final weights",
                                                  newest=False, d=dur)
        write_json(ev_unit / "final.json", rec)
    if rec["handoff"]["verified"]:
        TD.retire_save_points(ev_unit, name)
        row.pop("save_points_kept", None)
    else:
        row["save_points_kept"] = rec["handoff"].get("why")
    row["final_relay"] = {k: rec.get(k) for k in ("relay", "prefix", "verified")}


def final_identity(name: str, out: Path, ev_unit: Path, plan: dict, row: dict) -> None:
    """Step 1's check: the final found is THIS unit's (`completion.final_identity_problems`: arm, seed, relay path,
    the completion record archived with it). A record the evidence lacks is fetched from the final's own relay copy
    and re-hashed first; any remaining problem stops the unit -- a final of another seed is never adopted, and never
    silently replaced by a retrain."""
    from stages.single_operator_budget import completion as CP
    from stages.teacher_parent_initialization import relay as R
    rec = read_json(ev_unit / "final.json")
    bad = CP.final_identity_problems(out, name, plan=plan)
    names = [k for k in rec.get("files_sha256", {}) if k.startswith("run_completion")]
    if bad and len(names) == 1 and not (ev_unit / "train" / names[0]).is_file():
        got = R.run_online("get", {"record": rec, "dest": str(ev_unit / "train"), "only": names}, timeout_s=600)
        row["completion_record_fetched"] = {k: got.get(k) for k in ("verified", "from", "error")}
        bad = CP.final_identity_problems(out, name, plan=plan)
    row["final_identity"] = {"problems": bad, "checked_utc": utc()}
    if bad:
        raise DriverError(f"{name}: the final found is not this unit's: {bad}")


def run_unit(u: dict, *, out: Path, record: dict, dur: dict, prefix_root: str, snap: Path, a, attempt: str,
             mark, save, plan: dict | None = None, stage_init=None) -> None:
    from stages.single_operator_budget import completion as CP
    name, cfg, arm, seed = u["unit"], u["config"], u["arm"], int(u["seed"])
    if CP.unit_name(arm, seed) != name:
        raise DriverError(f"{name} is not the unit of arm {arm} and seed {seed}")
    target, modes = CP.train_target(arm, seed), modes_of(u)
    row = record["units"].setdefault(name, {})
    ev_unit, out_dir = out / name, Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_unit.mkdir(parents=True, exist_ok=True)
    steps = int(cfg["schedule"]["total_steps"])
    unit_prefix = f"{prefix_root}/{arm}/{name}"
    #: a removal that failed in an earlier attempt is retried first (only of a superseded save point: the latest
    #: resumable checkpoint stays until a later one is published)
    row["pending_removals_at_takeover"] = TD.retry_pending_removals(ev_unit, name)
    # 1. the final weights, wherever they were made durable -- and only if they are THIS unit's
    hit = TD.final_record(ev_unit, unit_prefix, dur)
    if hit is not None and hit[1] != "evidence":
        rec = {**hit[0], "handoff": {"verified": True, "why": None, "utc": utc(),
                                     "found": f"listed under {unit_prefix}/final"}}
        write_json(ev_unit / "final.json", rec)
    if hit is not None:
        final_identity(name, out, ev_unit, plan, row)
    # 2. done only on the files, against this seed's final
    sets = {m: CP.check(CP.ev_dir(out, target, m), target, m, out, label=CP.label_of(target)) for m in modes}
    row["at_start"] = {"final": None if hit is None else hit[1], "final_problems": CP.final_problems(out, name,
                                                                                                    plan=plan),
                       "sets": {m: {"complete": v["complete"], "problems": v["problems"]} for m, v in sets.items()}}
    save()
    if not row["at_start"]["final_problems"] and all(v["complete"] for v in sets.values()):
        for m, v in sets.items():
            CP.write(CP.ev_dir(out, target, m), v)
        row["state_at_start"] = "done"
        save()
        mark(f"UNIT_DONE:{name}")
        return
    # 3-4. the final weights: secured if they exist, trained only if they exist nowhere
    if hit is None:
        model = final_here(out_dir, cfg, steps)
        row["state_at_start"] = "final_on_pod" if model is not None else "train"
        save()
        if model is None:
            model = train_unit(u, row, ev_unit, out_dir, unit_prefix, dur, a, snap, mark, save, stage_init)
        secure_final(name, model, out_dir, cfg, ev_unit, unit_prefix, dur, row)
    else:
        row["state_at_start"] = "final_durable"
        publish(name, read_json(ev_unit / "final.json"), out_dir, ev_unit, unit_prefix, dur, row)
    save()
    mark(f"FINAL_DURABLE:{name}")
    want = read_json(ev_unit / "final.json")["files_sha256"]["model.safetensors"]

    def weights() -> Path:
        """The final weights, only when a set must generate: this pod's, else restored (never retrained)."""
        from stages.teacher_parent_initialization import relay as R
        model = final_here(out_dir, cfg, steps)
        if model is None:
            got = TD.restore_final(ev_unit, out_dir, cfg, unit_prefix, dur)
            if not got:
                raise DriverError(f"{name}: the final record names weights the relay does not give back")
            row["restored_final"] = got
            model = final_here(out_dir, cfg, steps)
        if model is None or R.sha256_file(model / "model.safetensors") != want:
            raise DriverError(f"{name}: the weights on this pod are not the final record's {want[:12]}")
        return model
    # 5. each set to completion against the final
    entry = {"arm": arm, "seed": seed, "unit": name}
    for m in modes:
        mark(f"EVAL_START:{name}:{m}")
        t0 = time.time()
        try:
            res = CP.complete_set(CP.ev_dir(out, target, m), target, m, out, attempt=attempt, model_fn=weights,
                                  work=out_dir.parent / "work", vllm_py=a.vllm_python, mark=mark, entry=entry, utc=utc)
        except CP.CompletionError as exc:
            raise DriverError(str(exc)) from exc
        row.setdefault("evaluations", {})[m] = {**res, "eval_seconds": round(time.time() - t0, 1)}
        save()
        mark(f"EVAL_DONE:{name}:{m}")
    fin = read_json(ev_unit / "final.json")
    if not (fin.get("handoff") or {}).get("verified"):     #: one more try after the evaluations
        publish(name, fin, out_dir, ev_unit, unit_prefix, dur, row)
        save()
    mark(f"UNIT_DONE:{name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--plan", required=True, help="the arm's training plan (repo-relative or absolute)")
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--attempt", default=None, help="this session's run id (names every generation it seals)")
    ap.add_argument("--prefix-root", default=None, help="the relay prefix every unit's durable copies live under")
    ap.add_argument("--units", default=None, help="a comma-separated subset of the plan's units, in plan order")
    ap.add_argument("--session-deadline-utc", default=None,
                    help="the session's hard deadline: training stops early enough to secure its last save point")
    a = ap.parse_args(argv)
    out, status = Path(a.out), Path(a.status)
    out.mkdir(parents=True, exist_ok=True)
    attempt = a.attempt or time.strftime("a%Y%m%d_%H%M%S", time.gmtime())
    record = {"schema": SCHEMA, "started_utc": utc(), "attempt": attempt, "units": {}}

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{utc()} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        write_json(out / "units.json", record)
        write_json(out / "report.json", record)        #: what the session's closeout and products check read

    ok = True
    try:
        from shared import study_engineering as SE
        from shared.pod.teacher_cache import local_snapshot
        from stages.single_operator_budget import completion as CP
        from stages.single_operator_budget import launch as L
        from stages.teacher_modes import training_plan as TP
        from stages.teacher_parent_initialization import relay as R
        auth = L.Authorization.load(a.authorization)
        record["authorization"] = SE.require_on_pod(auth, L.STUDY, SE.identity_hash(L.STUDY, REPO), str(REPO),
                                                    kind=L.TRAIN_KIND).get("authorization_id")
        if a.resume:
            record["resumed_from"] = TD.take_over(Path(a.resume), out)
        plan_path = Path(a.plan) if Path(a.plan).is_absolute() else REPO / a.plan
        plan = read_json(plan_path)
        bad = authorized_plan_problems(getattr(auth, "stage_conditions", None) or {}, plan, plan_path, L.TRAIN_KIND)
        record["authorized_plan"] = {"problems": bad, "plan": str(plan_path), "arm": plan.get("arm")}
        if bad:
            raise DriverError(f"the authorization does not fund the plan this driver loaded: {bad}")
        dur = TD.durability(plan)
        prefix_root = a.prefix_root or plan["relay_prefix_root"]
        record.update(plan=str(plan_path), plan_sha256=R.sha256_file(plan_path), arm=plan["arm"],
                      prefix_root=prefix_root, durability=dur)
        us = [next(u for u in plan["units"] if u["unit"] == n) for n in plan["unit_order"]]
        if any(u["arm"] != plan["arm"] for u in us):
            raise DriverError(f"{plan_path}: a unit of another arm than {plan['arm']}")
        if a.units:
            want = [x for x in a.units.split(",") if x]
            unknown = sorted(set(want) - {u["unit"] for u in us})
            if unknown:
                raise DriverError(f"--units names {unknown}, not units of {plan_path}")
            us = [u for u in us if u["unit"] in want]
        t = us[0]["config"]["teacher"]
        snap = Path(local_snapshot(t["model_id"], t["revision"]))
        record["tokenizer"] = TP.stage_tokenizers(snap)
        save()
        mark("CONTRACT_OK")
        for u in us:
            run_unit(u, out=out, record=record, dur=dur, prefix_root=prefix_root, snap=snap, a=a, attempt=attempt,
                     mark=mark, save=save, plan=plan)
        record["completion"] = CP.session(out, plan["arm"], write_records=True, plan=plan)
        save()
        if not a.units and not record["completion"]["complete"]:
            raise DriverError(f"{plan['arm']}'s session is not complete on its files: "
                              f"{record['completion']['n_complete']} of {record['completion']['n_sets']} sets, "
                              f"finals {record['completion']['finals']}, modes {record['completion']['modes']}")
        mark("SESSION_COMPLETE")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    save()
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

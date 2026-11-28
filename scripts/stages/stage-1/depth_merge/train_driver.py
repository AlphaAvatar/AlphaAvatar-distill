#!/usr/bin/env python3
"""On-pod driver of DMR1's training sessions (every variant): one arm's two seeds trained with SOB1's depth recipe from
the arm's training-pre initialization, each seed's final weights made durable, then each evaluated at 8,192 tokens in
both modes, every set bound to that seed's final weights.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/depth_merge/train_driver.py --authorization <a.json> \\
        --out <evidence> --status <file> --plan <the arm's training_plan_<arm>.json> --prefix-root <relay root> \\
        --unit-deadline-hours <h> [--session-deadline-utc <t>] --vllm-python /opt/vllm/bin/python --attempt <run id> \\
        [--resume <secured evidence>]

It is SOB1's phase-2 state machine (`single_operator_budget.train_driver.run_unit`, on TPI1's durability helpers),
bound to the arm's design (`depth_merge.common.bind_for`), with two additions:

* **a recurrent arm's initialization** (depth_loop_kept, depth_loop_merged): `stage_init` fetches the BASE's verified relay
  copy (its weights and config, re-hashed), then writes the plan's recurrent config text over the config and checks its
  sha256 -- so the trainer loads `RecurrentDepthForCausalLM` (36 logical layers over the 16 blocks) and the weights are
  the base's byte for byte. Training, its save points and its final carry that config; nothing else changes;
* **the engine validation** (a plan with ``engine_validation.required``: the first V2 session): after the contract and
  before any training, `engine_validation` stages the first unit's init and runs `depth_recurrence.engine_check`; a
  FAIL ends the session (RUN_FAILED) with nothing trained, a PASS already in the taken-over evidence is not repeated;
* **the evaluation engine** loads the recurrent architecture through the out-of-tree vLLM model: for a recurrent
  arm this driver puts ``src`` and the plugin directory on the runners' ``PYTHONPATH``
  (`shared.rollout.vllm_plugin_path`) before any of them start; a stock Qwen3 arm (depth_merge) keeps SOB1's runner
  environment exactly.

Per unit, as SOB1's (`single_operator_budget.train_driver` module docstring): a final that exists anywhere is
restored, never retrained; training happens only when no final exists; a set is complete only on its files, and only
the items no proven part holds are generated. The session ends ``SESSION_COMPLETE`` only when the arm's two finals
are durable and its four sets complete (`completion.session`).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

SCHEMA = "aadistill.dmr1.train_units/v1"


class DriverError(RuntimeError):
    pass


def launcher_of(arm: str):
    """The variant's launch module (its STUDY, Authorization and training kind of ``arm``)."""
    if DC.is_recurrent(arm):
        from stages.depth_recurrence import launch as L
    else:
        from stages.depth_merge import launch as L
    kind = next(k for k, arms in L.VARIANT.train_kinds.items() if arm in arms)
    return L, kind


def write_recurrent_config(u: dict, student_path: Path) -> dict:
    """The plan's recurrent config text over the base's ``config.json`` (atomically), checked by sha256."""
    from stages.teacher_parent_initialization import relay as R
    rc = u["initialization"]["recurrent_config"]
    dest = Path(student_path) / "config.json"
    tmp = dest.with_name(".config.json.partial")
    tmp.write_text(rc["text"])
    os.replace(tmp, dest)
    got = R.sha256_file(dest)
    if got != rc["sha256"]:
        raise DriverError(f"{u['unit']}: the recurrent config written hashes to {got[:12]}, the plan's {rc['sha256'][:12]}")
    return {"config": str(dest), "sha256": got, "from": rc["path"]}


def stage_init(u: dict, dur: dict) -> dict:
    """A unit's initialization at its student path: the base's relay copy (`TD.fetch_init`) unless the path already
    holds the plan's weights (staged by `engine_validation`), then, for a recurrent unit, its config
    (`write_recurrent_config`)."""
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    w = Path(u["config"]["student_path"]) / "model.safetensors"
    want = u["config"]["lineage"]["initialization"]["model_safetensors_sha256"]
    if w.is_file() and R.sha256_file(w) == want:
        got = {"path": str(w.parent), "from": "already staged on this pod, re-hashed", "utc": TD.utc()}
    else:
        got = TD.fetch_init(u, dur)
    if "recurrent_config" in u["initialization"]:
        got["recurrent_config"] = write_recurrent_config(u, Path(u["config"]["student_path"]))
    return got


def engine_validation(u: dict, plan: dict, out: Path, dur: dict, vllm_python: str, mark) -> dict:
    """The plan's engine validation before any training step (`depth_recurrence.engine_check`), checked on this pod
    against this plan and this pod's environment (`record_problems`, which re-derives the verdict from the metrics):

    * ``mode: reuse`` (an arm after the first): the canonical record the plan names must stand; otherwise nothing
      trains -- the check is not re-run here;
    * ``mode: run``: a record already in the taken-over evidence is kept only if it stands; otherwise the first unit's
      init is staged and the check runs, and a result that does not stand trains nothing."""
    from stages.depth_recurrence import engine_check as EC
    from stages.teacher_parent_initialization import train_driver as TD
    ev = plan["engine_validation"]
    env = EC.environment(vllm_python)
    if ev.get("mode") == "reuse":
        p = REPO / ev["record"]
        bad = (EC.record_problems(TD.read_json(p), plan, env=env, verify_inputs=True) if p.is_file()
               else [f"{ev['record']} is not in this checkout"])
        if bad:
            mark("ENGINE_CHECK_NOT_APPLICABLE")
            raise DriverError(f"no engine validation stands for this plan, nothing was trained: {bad}")
        mark("ENGINE_CHECK_REUSED")
        return {"status": "PASS", "record": ev["record"], "reused": True, "from_arm": ev.get("from_arm")}
    p = Path(out) / "engine_check" / "engine_check.json"
    if p.is_file() and not EC.record_problems(TD.read_json(p), plan, env=env, verify_inputs=True):
        return {"status": "PASS", "record": str(p), "reused": True}
    mark("ENGINE_CHECK_START")
    stage_init(u, dur)
    doc = EC.run(plan, Path(u["config"]["student_path"]), Path(out) / "engine_check", vllm_python)
    bad = EC.record_problems(doc, plan, env=env, verify_inputs=True)
    if bad:
        mark("ENGINE_CHECK_FAILED")
        raise DriverError(f"the engine validation failed, nothing was trained: {bad}")
    mark("ENGINE_CHECK_PASSED")
    return {"status": "PASS", "record": str(p), "checks": doc["checks"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--attempt", default=None)
    ap.add_argument("--prefix-root", default=None)
    ap.add_argument("--units", default=None)
    ap.add_argument("--session-deadline-utc", default=None)
    a = ap.parse_args(argv)
    from stages.teacher_parent_initialization import train_driver as TD
    out, status = Path(a.out), Path(a.status)
    out.mkdir(parents=True, exist_ok=True)
    attempt = a.attempt or time.strftime("a%Y%m%d_%H%M%S", time.gmtime())
    record = {"schema": SCHEMA, "started_utc": TD.utc(), "attempt": attempt, "units": {}}

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{TD.utc()} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        TD.write_json(out / "units.json", record)
        TD.write_json(out / "report.json", record)

    ok = True
    try:
        from shared import study_engineering as SE
        from shared.pod.teacher_cache import local_snapshot
        from shared.rollout.vllm_plugin_path import vllm_plugin_pythonpath
        from stages.single_operator_budget import completion as CP
        from stages.single_operator_budget import train_driver as SD
        from stages.teacher_modes import training_plan as TP
        from stages.teacher_parent_initialization import relay as R
        plan_path = Path(a.plan) if Path(a.plan).is_absolute() else REPO / a.plan
        plan = TD.read_json(plan_path)
        arm = plan["arm"]
        DC.bind_for(arm)                         #: SOB1's chain reads THIS arm's design and plans from here on
        if DC.is_recurrent(arm):
            #: every runner process the evaluation starts reads the recurrent model through the plugin; a stock Qwen3
            #: arm's runners keep SOB1's environment exactly (no plugin imported into vLLM's processes)
            os.environ["PYTHONPATH"] = vllm_plugin_pythonpath(REPO, os.environ.get("PYTHONPATH"))
        record["vllm_plugin_on_pythonpath"] = DC.is_recurrent(arm)
        L, kind = launcher_of(arm)
        auth = L.Authorization.load(a.authorization)
        record["authorization"] = SE.require_on_pod(auth, L.STUDY, SE.identity_hash(L.STUDY, REPO), str(REPO),
                                                    kind=kind).get("authorization_id")
        if a.resume:
            record["resumed_from"] = TD.take_over(Path(a.resume), out)
        bad = SD.authorized_plan_problems(getattr(auth, "stage_conditions", None) or {}, plan, plan_path, kind)
        record["authorized_plan"] = {"problems": bad, "plan": str(plan_path), "arm": arm}
        if bad:
            raise DriverError(f"the authorization does not fund the plan this driver loaded: {bad}")
        dur = TD.durability(plan)
        prefix_root = a.prefix_root or plan["relay_prefix_root"]
        record.update(plan=str(plan_path), plan_sha256=R.sha256_file(plan_path), arm=arm, prefix_root=prefix_root,
                      durability=dur, recurrent=DC.is_recurrent(arm))
        us = [next(u for u in plan["units"] if u["unit"] == n) for n in plan["unit_order"]]
        if any(u["arm"] != arm for u in us):
            raise DriverError(f"{plan_path}: a unit of another arm than {arm}")
        if a.units:
            want = [x for x in a.units.split(",") if x]
            us = [u for u in us if u["unit"] in want]
        t = us[0]["config"]["teacher"]
        snap = Path(local_snapshot(t["model_id"], t["revision"]))
        record["tokenizer"] = TP.stage_tokenizers(snap)
        save()
        mark("CONTRACT_OK")
        if (plan.get("engine_validation") or {}).get("required"):
            record["engine_validation"] = engine_validation(us[0], plan, out, dur, a.vllm_python, mark)
            save()
        for u in us:
            SD.run_unit(u, out=out, record=record, dur=dur, prefix_root=prefix_root, snap=snap, a=a, attempt=attempt,
                        mark=mark, save=save, plan=plan, stage_init=stage_init)
        record["completion"] = CP.session(out, arm, write_records=True, plan=plan)
        save()
        if not a.units and not record["completion"]["complete"]:
            raise DriverError(f"{arm}'s session is not complete on its files: {record['completion']['n_complete']} of "
                              f"{record['completion']['n_sets']} sets, finals {record['completion']['finals']}")
        mark("SESSION_COMPLETE")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=TD.utc(), terminal="ALL_DONE" if ok else "RUN_FAILED")
    save()
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

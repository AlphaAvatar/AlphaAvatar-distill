#!/usr/bin/env python3
"""On-pod driver of Research A's teacher session: calibration, initialization, waves, X, teacher evaluation.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/teacher_session.py --authorization <authorization.json> \\
        --durable-root <volume>/teacher_modes --status <file> [--phases calibration,initialization,waves,x,teacher_eval]

Every value the runner receives is read from ``generation_protocol.json``
after the authorization's study identity and harness are re-checked here
(`phase.require_on_pod`). Everything finished is written under
``--durable-root`` on the network volume as it completes, and a later attempt
resumes from there (AGENTS.md P8.4): a sealed shard is never regenerated, an
initialization that re-identifies is never re-materialized, an evaluated unit
is rescored, never regenerated.

PHASES, in order; each skips what its durable records say is complete:

1. ``calibration`` -- the hybrid on the 160 calibration prompts
   (`hybrid_calibration.py`), each mode, one candidate.
2. ``initialization`` -- `initialization.py` in a child process (the GPU is
   clear of the engine): the mode-balanced hybrid profiles, the fixed path,
   the checkpoint persisted with its identity.
3. ``waves`` -- W00 (T0 and V), then E's waves, both modes; per wave and mode
   pass 0, then pass 1 for each task whose first candidate the pod rejects
   (H1's `generation_driver.second_pass`; none for MBPP). After each wave the
   pod's BOUND of the stopping rule (generation_protocol.json :: stopping):
   E stops at the first wave whose LOWER H-T-S count reaches the target with
   the margin. The exact rule is the dev box's (`admission.py`).
4. ``x`` -- X's waves, H-N only, until H-N-C's LOWER count reaches the
   highest UPPER H-T-S count with the margin, or X is exhausted.
5. ``teacher_eval`` -- eval-set-1 in each mode at the teacher's supported
   context (`h1pa_driver.evaluate`, the snapshot as is, the mode's kwargs).

Generation writes TEXT and runs no teacher program (MBPP's sandbox is the dev
box's). The status file gets ``MARKER:ALL_DONE`` only when every requested
phase is complete, else ``MARKER:RUN_FAILED`` with everything finished kept.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import generation_driver as GD  # noqa: E402
from stages.reasoning_data import generation_plan as GP  # noqa: E402
from stages.reasoning_data import source_plan as SP  # noqa: E402

PROTOCOL = Path("configs/stages/stage-2/teacher_modes/generation_protocol.json")
MODES = ("H-T", "H-N")
PHASES = ("calibration", "initialization", "waves", "x", "teacher_eval")
#: T0's own pools, the order's source (build_tasks.POOLS): every task row the pod's verdict reads
POOLS = ("train_tasks_v2", "train_tasks_v1", "external_tasks_v2", "math_train_v1", "mbpp_train_v1",
         "squad_v2_train_unanswerable_v1")
SCHEMA = "aadistill.stage2.teacher_modes.teacher_session/v1"


def _sha(p: Path) -> str:
    return GD.sha(p)


def authorize(path: Path, repo: Path) -> dict:
    from stages.teacher_modes import phase as PH
    return PH.require_on_pod(path, "teacher", repo)


# ---- the contract ----------------------------------------------------------------------------------------

def contract(repo: Path, authorization: Path, authorize=authorize) -> dict:
    bound = authorize(authorization, repo)
    proto = json.loads((repo / PROTOCOL).read_text())
    inp = proto["inputs"]
    for lst in [*inp["t0"], inp["order"], inp["calibration"]]:
        if not lst.get("sha256") or _sha(repo / lst["path"]) != lst["sha256"]:
            raise GD.ShardError(f"{lst['path']}: not the input the protocol binds")
    by = {}
    for name in POOLS:                      #: T0's own pools (build_tasks.POOLS), v2 before v1
        for t in A.jsonl(repo / "data/stages/stage-2/reasoning_data/tasks" / f"{name}.jsonl"):
            by.setdefault(t["task_id"], t)
    t0 = [json.loads(l)["task_id"] for lst in inp["t0"] for l in (repo / lst["path"]).read_text().splitlines()
          if l.strip()]
    order = list(A.jsonl(repo / inp["order"]["path"]))
    cal = list(A.jsonl(repo / inp["calibration"]["path"]))
    tasks = {**{t: by[t] for t in t0}, **{t["task_id"]: t for t in order}, **{t["task_id"]: t for t in cal}}
    return {**bound, "protocol": proto, "protocol_path": PROTOCOL, "tasks": tasks, "t0": t0, "order": order,
            "calibration": cal, "launch_identity_sha256": bound["identity"]}


# ---- stages: which tasks, in which runner inputs ------------------------------------------------------------

def stages(c: dict) -> list[dict]:
    """W00 (T0 + V), E's waves, X's waves, in generation order."""
    out = [{"stage": "w00", "slice": "T0+V", "modes": MODES,
            "ids": c["t0"] + [t["task_id"] for t in c["order"] if t["slice"] == "V"]}]
    waves = defaultdict(list)
    for t in c["order"]:
        if t["slice"] in ("E", "X"):
            waves[(t["slice"], t["wave"])].append(t["task_id"])
    for (s, w), ids in sorted(waves.items(), key=lambda kv: ("EX".index(kv[0][0]), kv[0][1])):
        out.append({"stage": f"{'w' if s == 'E' else 'x'}{w:02d}", "slice": s, "modes": MODES if s == "E" else ("H-N",),
                    "ids": ids})
    return out


def write_inputs(c: dict, out: Path, stage: str, ids: list[str]) -> list[dict]:
    """The stage's run_battery inputs (one file per system message), written once, deterministic."""
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    by_system = defaultdict(list)
    for tid in ids:
        t = c["tasks"][tid]
        if "calibration" in t:
            row, system = t, c["protocol"].get("system_message", "You are a helpful Assistant.")
        else:
            m = t["messages"]
            system = m[0]["content"] if m and m[0]["role"] == "system" else GP.SYSTEM_DEFAULT
            row = GP.run_input({**t, "stratum": t.get("stratum") or SP.stratum_of(t, spec)}, spec)
        by_system[system].append(row)
    d = out / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (system, rows) in enumerate(sorted(by_system.items())):
        path = d / f"{stage}_g{k}.jsonl"
        body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        if path.is_file() and path.read_text() != body:
            raise GD.ShardError(f"{path.name} on the volume is not this stage's input")
        path.write_text(body)
        files.append({"file": path.name, "system": system, "tasks": len(rows), "sha256": _sha(path)})
    return files


def mode_args(c: dict, mode: str) -> tuple[str, ...]:
    m = c["protocol"]["modes"][mode]
    return ("--chat-template-kwargs", json.dumps(m["template_kwargs"], sort_keys=True),
            "--generate-sampling", json.dumps(m["sampling"], sort_keys=True))


def run_stage(c: dict, out: Path, stage: dict, run, mark, *, candidates: int) -> dict:
    """Every mode's passes of one stage; returns per mode the records by task and pass, and completeness."""
    files = write_inputs(c, out, stage["stage"], stage["ids"])
    got = {"stage": stage["stage"], "groups": [], "complete": True, "records": {}}
    for mode in stage["modes"]:
        recs: dict[str, list[dict]] = defaultdict(list)
        for f in files:
            ids = None
            for k in range(candidates):
                if k and not ids:
                    break
                label = f"{stage['stage']}_{mode}_pass{k}_{Path(f['file']).stem}"
                g = GD.run_group(c, out, out / "inputs", f, k, ids, run, mark, label=label,
                                 extra_args=mode_args(c, mode))
                g_recs = g.pop("records")
                for tid, r in g_recs.items():
                    recs[tid].append(r)
                got["groups"].append(g)
                if not g["complete"]:
                    got["complete"] = False
                    mark(f"{GD.UNIT_FAILED}:{label}")
                    break
                mark(f"{GD.UNIT_OK}:{label}")
                if k == 0 and candidates > 1:
                    ids = GD.second_pass(c, g_recs)        #: the pod's judgement of each first candidate
        got["records"][mode] = {tid: sorted(rs, key=lambda r: r["candidate_index"]) for tid, rs in recs.items()}
    return got


# ---- the pod's bound of the stopping rule ----------------------------------------------------------------

def verdict(c: dict, tid: str, mode: str, recs: list[dict]) -> dict:
    """admitted (with its documents), pending (MBPP: naturally terminated, judged on the dev box) or rejected."""
    from aadistill.data.token_documents import DocumentError, document_from_turn, prompt_mode_problems
    from stages.reasoning_data.generation_adapter import candidate
    task = c["tasks"][tid]
    dev_box = set(c["protocol"]["execution"]["no_second_candidate_verifiers"])
    stop = c["protocol"]["teacher"]["stop_ids"]
    suffix = c["protocol"]["modes"][mode]["generation_suffix_ids"]
    for rec in recs:
        try:
            docs = [document_from_turn(t, doc_id=f"{tid}|{mode}|t{k}", stop_ids=stop) for k, t in enumerate(rec["turns"])]
        except DocumentError:
            continue
        if any(prompt_mode_problems(d, suffix) for d in docs):
            continue
        state = ("pending" if task.get("verifier") in dev_box else
                 "admitted" if candidate(task, rec, rec["candidate_index"]).accepted_for_training else None)
        if state:
            return {"state": state, "docs": {d.doc_id: len(d) for d in docs}}
    return {"state": "rejected", "docs": {}}


def bound(c: dict, verdicts: dict, stages_done: list[dict]) -> dict:
    """LOWER and UPPER H-T-S counts after each generated E stage, and H-N-C's LOWER count with X."""
    from aadistill.data.token_documents import isolated_sequence_count
    st = c["protocol"]["stopping"]
    seq, seed = int(st["sequence_length"]), int(st["pack_order_seed"])

    def count(ids: list[str], mode: str, pending_in: bool) -> int:
        docs = {}
        for tid in ids:
            v = verdicts.get((tid, mode))
            if v and (v["state"] == "admitted" or (pending_in and v["state"] == "pending")):
                docs.update(v["docs"])
        return isolated_sequence_count(dict(sorted(docs.items())), seq, order_seed=seed) if docs else 0

    def in_both(ids, pending_in):
        ok = ("admitted", "pending") if pending_in else ("admitted",)
        return [t for t in ids if all((verdicts.get((t, m)) or {}).get("state") in ok for m in MODES)]

    t0 = set(c["t0"])
    rows, ids = [], []
    for s in stages_done:
        if s["slice"] not in ("T0+V", "E"):
            continue
        ids += [t for t in s["ids"] if s["slice"] == "E" or t in t0]
        lo, up = in_both(ids, False), in_both(ids, True)
        rows.append({"stage": s["stage"], "I_lower": len(lo), "I_upper": len(up),
                     "H-T-S_lower": count(lo, "H-T", False), "H-T-S_upper": count(up, "H-T", True), "ids": list(ids)})
    out = {"stages": [{k: v for k, v in r.items() if k != "ids"} for r in rows]}
    target = int(st["target_sequences"]) * (1 + float(st["pod_margin"]))
    out["e_reached"] = bool(rows) and rows[-1]["H-T-S_lower"] >= target
    if rows:
        first_up = next((r for r in rows if r["H-T-S_upper"] >= int(st["target_sequences"])), rows[-1])
        base = in_both(first_up["ids"], False)
        x_ids = [t for s in stages_done if s["slice"] == "X" for t in s["ids"]]
        out["x_target"] = max(r["H-T-S_upper"] for r in rows) * (1 + float(st["pod_margin"]))
        out["H-N-C_lower"] = count(base + x_ids, "H-N", False)
        out["x_reached"] = out["H-N-C_lower"] >= out["x_target"]
    return out


# ---- initialization and teacher evaluation ----------------------------------------------------------------

def initialize(c: dict, root: Path, cal: dict, status) -> dict:
    """`initialization.py` in its own process, on the calibration records; skipped when durable."""
    rec = root / "initialization" / "initialization.json"
    if rec.is_file():
        return {"status": "already_persisted", "record": str(rec)}
    shards = root / "generation"
    cmd = [sys.executable, str(REPO / "scripts/stages/stage-2/teacher_modes/initialization.py"),
           "--records", str(shards), "--out", str(root / "initialization"), "--work", "/workspace/a_init_work",
           "--device", "cuda:0", "--identity", c["launch_identity_sha256"]]
    with (root / "initialization.log").open("a") as lf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}).returncode
    return {"status": "persisted" if rc == 0 and rec.is_file() else "failed", "returncode": rc}


def teacher_eval(c: dict, root: Path, mode: str) -> dict:
    """eval-set-1 on the teacher in `mode` at its supported context; durable generations are rescored."""
    from shared.pod.teacher_cache import local_snapshot
    from stages.reasoning_data import h1pa_driver as HD
    from stages.reasoning_data import persist_unit as PU
    unit = f"teacher_{mode}"
    if PU.completed(root / "teacher_eval", unit, "evaluation") is not None:
        return {"status": "already_persisted"}
    t, te = c["protocol"]["teacher"], c["protocol"]["teacher_evaluation"]
    design = json.loads((REPO / "configs/stages/stage-2/teacher_modes/d4_scale_design.json").read_text())
    pre = json.loads((REPO / "logs/stages/stage-2/teacher_modes/plans/d4_scale_preregistration.json").read_text())
    dec = pre["evaluation"]["decoding"]
    work = Path("/workspace/a_teacher_eval") / mode
    job = {"evaluation": {"decoding": {"path": "configs/shared/evaluation/decoding/nucleus_v1.json",
                                       "sha256": dec["sha256"]},
                          "trained_context": int(te["context"]), "action_protocol": c["protocol"]["execution"]["action_protocol"]},
           "out": work / "evaluation", "progress": root / "teacher_eval" / unit / "progress",
           "model_dir": Path(local_snapshot(t["model"], t["revision"])), "label": f"teacher_{mode}",
           "env": {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}, "package": "as_is",
           "template_kwargs": c["protocol"]["modes"][mode]["template_kwargs"]}
    assert design["evaluation"]["set"].startswith("eval-set-1")
    rep, files = HD.evaluate(job)
    PU.persist_files(unit, files, root / "teacher_eval", label="evaluation")
    return {"status": "persisted", "formal": {k: rep.get(k) for k in ("schema", "eval_set_id") if k in rep}}


# ---- the driver -----------------------------------------------------------------------------------------------

def main(argv=None, *, run=GD.run_battery, authorize=authorize, repo: Path | None = None,
         initialize=initialize, teacher_eval=teacher_eval) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--phases", default=",".join(PHASES))
    args = ap.parse_args(argv)
    repo = Path(repo or REPO)
    root, status = Path(args.durable_root), Path(args.status)
    out = root / "generation"
    out.mkdir(parents=True, exist_ok=True)
    phases = [p for p in args.phases.split(",") if p]
    if set(phases) - set(PHASES):
        raise SystemExit(f"unknown phases {sorted(set(phases) - set(PHASES))}")

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phases": {}}
    evidence = root / "teacher_session.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1, default=str) + "\n")

    try:
        c = contract(repo, Path(args.authorization), authorize)
        record["contract"] = {k: c[k] for k in ("authorization_id", "launch_identity_sha256", "harness_digest")
                              if k in c} | {"status": "ok"}
    except BaseException as exc:  # noqa: BLE001
        record["contract"] = {"status": "failed", "error": repr(exc)}
        save()
        mark("RUN_FAILED")
        return 2
    save()
    mark("CONTRACT_OK")
    complete = True
    n_cand = int(c["protocol"]["execution"]["candidates"])
    verdicts: dict = {}
    done_stages: list[dict] = []

    def judge(got: dict, s: dict) -> None:
        for mode, recs in got["records"].items():
            for tid in s["ids"]:
                verdicts[(tid, mode)] = verdict(c, tid, mode, recs.get(tid, []))

    try:
        if "calibration" in phases:
            s = {"stage": "cal", "slice": "calibration", "modes": MODES, "ids": [t["task_id"] for t in c["calibration"]]}
            got = run_stage(c, out, s, run, mark, candidates=1)
            record["phases"]["calibration"] = {"complete": got["complete"], "groups": got["groups"]}
            complete &= got["complete"]
            save()
        if "initialization" in phases:
            record["phases"]["initialization"] = initialize(c, root, record["phases"].get("calibration"), status)
            complete &= record["phases"]["initialization"]["status"] != "failed"
            save()
        all_stages = stages(c)
        for phase, slices in (("waves", ("T0+V", "E")), ("x", ("X",))):
            if phase not in phases:
                continue
            rows = record["phases"].setdefault(phase, {"stages": []})
            for s in all_stages:
                if s["slice"] not in slices:
                    continue
                b = bound(c, verdicts, done_stages) if done_stages else {}
                if phase == "waves" and b.get("e_reached"):
                    rows["stopped"] = {"after": done_stages[-1]["stage"], "bound": b}
                    break
                if phase == "x" and b.get("x_reached"):
                    rows["stopped"] = {"after": done_stages[-1]["stage"], "bound": b}
                    break
                got = run_stage(c, out, s, run, mark, candidates=n_cand)
                judge(got, s)
                done_stages.append(s)
                rows["stages"].append({"stage": s["stage"], "complete": got["complete"], "groups": got["groups"]})
                rows["bound"] = bound(c, verdicts, done_stages)
                save()
                if not got["complete"]:
                    complete = False
                    break
            else:
                rows["stopped"] = {"after": done_stages[-1]["stage"] if done_stages else None,
                                   "reason": f"{phase}: every stage of the slice generated"}
            save()
            if not complete:
                break
        if "teacher_eval" in phases and complete:
            record["phases"]["teacher_eval"] = {m: teacher_eval(c, root, m) for m in MODES}
            save()
    except BaseException as exc:  # noqa: BLE001
        complete = False
        record["error"] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if complete else "RUNS_INCOMPLETE")
    save()
    mark("ALL_DONE" if complete else "RUN_FAILED")
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())

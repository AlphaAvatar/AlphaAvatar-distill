#!/usr/bin/env python3
"""The tool pilot's sealed episodes, imported into the bounded tool generation by an explicit record. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_import \\
        --run a_tool_pilot_20261104_164531 --store artifacts/stages/stage-2/teacher_modes/tool_pilot_sessions/<run>/polled \\
        --pool <tool pool> --write logs/stages/stage-2/teacher_modes/plans/tool_generation_import.json

The pilot generated its tasks with the formal generation's own teacher, template, sampling, seeds, session limit,
engine and two-candidate rule, under `tool_verification@v2`. Those episodes are reused -- never regenerated, never
re-sampled -- when every one of these holds, each checked here and recorded with both values:

* the SAME INPUT: each task's canonical sha256 in the pilot's frozen plan equals the formal pool's task record;
* the SAME TEACHER LINEAGE AND TEMPLATE: the protocol's lineage (model, revision, per-mode template switch and
  sampling) and execution terms (seed base, candidates, session limit, action protocol, stop ids, system default,
  environment factory, memory fraction) at the pilot's session commit equal today's; `teacher_identity.json` (the
  tokenizer files' and chat template's sha256) and `requirements-vllm.txt` too;
* the SAME GENERATION CODE: the runner, the generation driver, the episode environment and verifier, and the core
  parser by sha256 (`GENERATION_CODE`: `tool_pilot.IMPLEMENTATION` without `tool_session.py`, which composes rounds
  -- which tasks, in which order -- and passes the protocol's mode terms checked above, but decides nothing a task
  generates), and every sealed record's runner sha256;
* the SAME SEEDS: every sealed record's pass seed base is the protocol's seed base + its candidate index.

The record lists, per mode and task, every candidate the pilot sealed (label, shard, the shard's sha256, the pilot's
identity, the v2 verdict) and the task's final outcome and units. A refused task is DONE (both candidates spent: the
two-candidate rule is never extended); an admitted one contributes its units. The session reads the outcomes only
(`tool_session.generate_mode`'s ``imported``); the dev box's admission reads the shards themselves
(`tool_admission.episodes_from_import`), verified again against this record.

A FORMAL session is imported the same way (`build_session`, ``--session``; maintainer 2026-11-05, review of f06ccd4,
section 1): a completed `tool_session.py` store of an earlier plan -- whose launch identity a rebuilt plan's session
cannot resume, because `generation_driver.sealed` binds every shard to it -- keeps its shards, hashes and identity
untouched. Its rounds' labels (``tool_<mode>_r<r>_pass<k>``) are read under that identity, each label's ``.complete``
listing exactly the shards present and each round's input hashing to what its records name; the same compatibility
checks run against the session commit's files; and the imported cells must reconcile EXACTLY with the session's own
record (tasks attempted, admitted, units). A rebuilt plan binds every import record (`tool_generation_plan.build`), so
its cells start from the imported units and walk the frozen orders without the imported tasks: nothing imported is
generated again, and the dev box admits the old shards under their own identity beside the new ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from stages.teacher_modes import tool_episodes as TE  # noqa: E402
from stages.teacher_modes import tool_pilot as TPL  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.tool_generation_import/v1"
IMPORT = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import.json"
PROTOCOL_REL = "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
TEACHER_IDENTITY = "configs/stages/stage-2/teacher_modes/teacher_identity.json"
VLLM_PINS = "requirements-vllm.txt"
#: the code that decides what one task's episode is (module docstring)
GENERATION_CODE = tuple(f for f in TPL.IMPLEMENTATION if not f.endswith("/tool_session.py"))
#: the execution terms that decide what a generation produces; the seal cadence decides only when shards close
EXECUTION_FIELDS = ("seed_base", "session_limit", "candidates", "action_protocol", "system_message_default",
                    "env_factory", "gpu_memory_utilization", "stop_ids")


class ImportRefused(RuntimeError):
    """A pilot artifact that is not the formal generation's: nothing is imported."""


def _at(commit: str, rel: str) -> bytes:
    r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True)
    if r.returncode != 0:
        raise ImportRefused(f"{rel} is not readable at {commit[:12]}")
    return r.stdout


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compatibility(run_id: str, session: dict, pool_tasks: dict[str, dict], doc: dict, by_id: dict) -> dict:
    """Every check with the pilot's and today's value; ``ok`` is their conjunction."""
    rd = REPO / "logs/stages/stage-2/teacher_modes/runs" / run_id
    commit = json.loads((rd / "runtime/session.json").read_text())["setup_environment"]["SESSION_COMMIT"]
    then = json.loads(_at(commit, PROTOCOL_REL))
    now = json.loads((REPO / PROTOCOL_REL).read_text())
    lineage = doc["lineage"]
    checks = {}

    def check(name, a, b):
        checks[name] = {"pilot": a, "formal": b, "equal": a == b}
    check("lineage", then["lineages"][lineage], now["lineages"][lineage])
    check("execution", {k: then["execution"].get(k) for k in EXECUTION_FIELDS},
          {k: now["execution"].get(k) for k in EXECUTION_FIELDS})
    for rel in (TEACHER_IDENTITY, VLLM_PINS):
        check(rel, _sha(_at(commit, rel)), _sha((REPO / rel).read_bytes()))
    impl = (session.get("identity_parts") or {}).get("implementation_sha256") or {}
    check("generation_code", {f: impl.get(f) for f in GENERATION_CODE}, {f: TPL.sha(REPO / f) for f in GENERATION_CODE})
    check("verification_version", (session.get("identity_parts") or {}).get("verification_version"),
          TE.VERIFICATION_VERSION)
    missing = sorted(t for t in by_id if t not in pool_tasks)
    differ = sorted(t for t in by_id if t in pool_tasks and TPL.task_sha256(pool_tasks[t]) != TPL.task_sha256(by_id[t]))
    checks["inputs"] = {"tasks": len(by_id), "absent_from_the_pool": missing[:10], "differ_from_the_pool": differ[:10],
                        "equal": not missing and not differ}
    return {"session_commit": commit, "checks": checks, "ok": all(c["equal"] for c in checks.values())}


def build(run_id: str, store: Path, pool_tasks: dict[str, dict], *, plan_path: Path = TPL.PLAN,
          tasks_path: Path = TPL.TASKS) -> dict:
    from stages.reasoning_data import generation_driver as GD
    doc, by_id = TPL.load(plan_path, tasks_path, protocol_path=TPL.frozen_protocol(plan_path))
    store = Path(store)
    session = json.loads((store / "tool_session.json").read_text()) if (store / "tool_session.json").is_file() else None
    if session is None:
        tgz = store.parent / "tool_pilot_hybrid.tgz"
        raise ImportRefused(f"{store} holds no tool_session.json (the archive {tgz.name} does)")
    identity = session["identity"]
    comp = compatibility(run_id, session, pool_tasks, doc, by_id)
    if not comp["ok"]:
        bad = [k for k, v in comp["checks"].items() if not v["equal"]]
        raise ImportRefused(f"the pilot's episodes are not the formal generation's: {bad}")
    proto = json.loads((REPO / PROTOCOL_REL).read_text())
    stop_ids = [int(x) for x in proto["execution"]["stop_ids"]]
    seed_base = int(proto["execution"]["seed_base"])
    modes = {}
    for mode in sorted(doc["modes"]):
        thinking = bool(proto["lineages"][doc["lineage"]]["modes"][mode]["template_kwargs"].get("enable_thinking", True))
        tasks = {}
        for k in (0, 1):
            label = f"pilot_{mode}_r1_pass{k}"
            shards = {p.name: GD.sha(p) for p in sorted(store.glob(f"{label}.part*.jsonl"))}
            for tid, rec in GD.sealed(store, label, identity).items():
                gp = rec.get("generation_pass") or {}
                if int(gp.get("seed_base", -1)) != seed_base + k or int(rec.get("candidate_index", -1)) != k:
                    raise ImportRefused(f"{tid} {label}: sealed under seed base {gp.get('seed_base')}, not {seed_base + k}")
                if gp.get("runner_sha256") != TPL.sha(REPO / GD.RUNNER):
                    raise ImportRefused(f"{tid} {label}: generated by another runner")
                shard = next(n for n in shards if any(json.loads(x)["task_id"] == tid for x in
                                                      (store / n).read_text().splitlines()))
                out, units, _ = TPL._outcome(by_id[tid], rec, set(), mode=mode, thinking=thinking,
                                                   stop_ids=stop_ids)
                t = tasks.setdefault(tid, {"split": by_id[tid]["split"], "category": by_id[tid]["category"],
                                           "shape": TE.episode_shape(by_id[tid]),
                                           "task_sha256": TPL.task_sha256(by_id[tid]), "candidates": []})
                #: the verdict's problems are re-derivable from the shard (tool_pilot_diagnosis); only the outcome is kept
                t["candidates"].append({"candidate": k, "label": label, "shard": shard, "shard_sha256": shards[shard],
                                        "outcome": out, "units": units})
        for tid, t in tasks.items():
            t["candidates"].sort(key=lambda c: c["candidate"])
            c0 = t["candidates"][0]
            if c0["candidate"] != 0:
                raise ImportRefused(f"{tid}: no sealed first candidate")
            if c0["outcome"] == "admitted" and len(t["candidates"]) > 1:
                raise ImportRefused(f"{tid}: a second candidate after an admitted first")
            final = next((c for c in t["candidates"] if c["outcome"] == "admitted"), None)
            if final is None and len(t["candidates"]) < int(proto["execution"]["candidates"]):
                raise ImportRefused(f"{tid}: refused at candidate 0 without its second candidate: not done")
            t["final"] = {"outcome": "admitted" if final else "rejected", "candidate": final["candidate"] if final else None,
                          "units": float(sum((final or {"units": {}})["units"].values()))}
        modes[mode] = {"tasks": tasks, "totals": {
            "tasks": len(tasks), "admitted": sum(t["final"]["outcome"] == "admitted" for t in tasks.values()),
            "candidates": sum(len(t["candidates"]) for t in tasks.values())}}
    return {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": run_id, "store": str(store.relative_to(REPO)) if REPO in store.resolve().parents else str(store),
            "pilot_identity": identity, "pilot_plan_sha256": TPL.sha(plan_path), "lineage": doc["lineage"],
            "pool_content_sha256": doc["source"]["pool_content_sha256"],
            "verification_version": TE.VERIFICATION_VERSION, "compatibility": comp, "modes": modes,
            "rule": "a task listed here is DONE in its mode: never attempted again, its second candidate never "
                    "repeated; a formal round takes only tasks this record does not list",
            "authorizes": "nothing"}


SESSION_LABEL = "tool_{mode}_r{round}_pass{k}"        #: tool_session.generate_mode's round labels


def session_commit(run_id: str) -> str:
    rd = REPO / "logs/stages/stage-2/teacher_modes/runs" / run_id
    return json.loads((rd / "runtime/session.json").read_text())["setup_environment"]["SESSION_COMMIT"]


def _session_compatibility(at, commit: str, lineage: str, pool_tasks: dict[str, dict], tasks: dict[str, dict]) -> dict:
    """`compatibility` for a formal session: every term read at its session commit (``at(rel)``) against today's (the
    generation code by the files' sha256 at that commit), and every imported task's input against the pool's."""
    then = json.loads(at(PROTOCOL_REL))
    now = json.loads((REPO / PROTOCOL_REL).read_text())
    checks = {}

    def check(name, a, b):
        checks[name] = {"session": a, "formal": b, "equal": a == b}
    check("lineage", then["lineages"][lineage], now["lineages"][lineage])
    check("execution", {k: then["execution"].get(k) for k in EXECUTION_FIELDS},
          {k: now["execution"].get(k) for k in EXECUTION_FIELDS})
    for rel in (TEACHER_IDENTITY, VLLM_PINS):
        check(rel, _sha(at(rel)), _sha((REPO / rel).read_bytes()))
    check("generation_code", {f: _sha(at(f)) for f in GENERATION_CODE},
          {f: TPL.sha(REPO / f) for f in GENERATION_CODE})
    missing = sorted(t for t in tasks if t not in pool_tasks)
    differ = sorted(t for t in tasks if t in pool_tasks and TPL.task_sha256(pool_tasks[t]) != TPL.task_sha256(tasks[t]))
    checks["inputs"] = {"tasks": len(tasks), "absent_from_the_pool": missing[:10], "differ_from_the_pool": differ[:10],
                        "equal": not missing and not differ}
    return {"session_commit": commit, "checks": checks, "ok": all(c["equal"] for c in checks.values())}


def _complete_labels(store: Path, mode: str) -> list[tuple[int, int, str]]:
    """``[(round, k, label)]`` of every completed round label of ``mode``; a label's ``.complete`` must list exactly
    the shards present, by sha256."""
    import re
    from stages.reasoning_data import generation_driver as GD
    pat = re.compile(rf"^tool_{re.escape(mode)}_r(\d+)_pass(\d+)$")
    out = []
    for done in sorted(store.glob(f"tool_{mode}_r*_pass*.complete")):
        label = done.name[:-len(".complete")]
        m = pat.match(label)
        if not m:
            continue
        listed = {x["name"]: x["sha256"] for x in json.loads(done.read_text())["shards"]}
        have = {p.name: GD.sha(p) for p in store.glob(f"{label}.part*.jsonl")}
        if have != listed:
            raise ImportRefused(f"{label}: the shards present are not the ones its .complete lists")
        out.append((int(m.group(1)), int(m.group(2)), label))
    return sorted(out)


PLAN_REL = "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json"


def build_session(run_id: str, store: Path, pool_tasks: dict[str, dict], *, at=None) -> dict:
    """The import record of a COMPLETED formal `tool_session.py` store (module docstring): per mode and task, every
    candidate it sealed -- label, shard, the shard's sha256 -- with its outcome and units re-derived under today's
    verifier, and the task's final outcome. Refuses a session that did not end ALL_DONE, a label whose shards or
    round input do not verify, a task sealed in two rounds, a refused first candidate without its second, a seed or
    runner that is not the protocol's, and cells that do not reconcile with the session's own record."""
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes.tool_generation_priors import POOL_V1_SOURCES
    store = Path(store)
    sess = json.loads((store / "tool_session.json").read_text())
    if sess.get("terminal") != "ALL_DONE":
        raise ImportRefused(f"{run_id} ended {sess.get('terminal')!r}: only a completed session is imported (an "
                            "interrupted one resumes under its own identity)")
    identity, lineage = sess["identity"], sess["lineage"]
    commit = session_commit(run_id) if at is None else "given"
    at = at or (lambda rel: _at(commit, rel))
    plan_then = json.loads(at(PLAN_REL))
    proto = json.loads((REPO / PROTOCOL_REL).read_text())
    stop_ids = [int(x) for x in proto["execution"]["stop_ids"]]
    seed_base, cand = int(proto["execution"]["seed_base"]), int(proto["execution"]["candidates"])
    runner = TPL.sha(REPO / GD.RUNNER)
    modes, inputs, shards_all = {}, {}, set()
    for mode in sorted(plan_then["lineages"][lineage]["modes"]):
        thinking = bool(proto["lineages"][lineage]["modes"][mode]["template_kwargs"].get("enable_thinking", True))
        tasks, round_of = {}, {}
        for r, k, label in _complete_labels(store, mode):
            shard_of = {}
            for p in sorted(store.glob(f"{label}.part*.jsonl")):
                shards_all.add(p.name)
                for line in p.read_text().split("\n"):
                    if line.strip():
                        shard_of[json.loads(line)["task_id"]] = (p.name, GD.sha(p))
            for tid, rec in GD.sealed(store, label, identity).items():
                gp = rec.get("generation_pass") or {}
                if int(gp.get("seed_base", -1)) != seed_base + k or int(rec.get("candidate_index", -1)) != k:
                    raise ImportRefused(f"{tid} {label}: sealed under seed base {gp.get('seed_base')}, not {seed_base + k}")
                if gp.get("runner_sha256") != runner:
                    raise ImportRefused(f"{tid} {label}: generated by another runner")
                inp = store / "inputs" / str(gp.get("group_input"))
                if gp.get("group_input") not in inputs:
                    if not inp.is_file() or GD.sha(inp) != gp.get("group_input_sha256"):
                        raise ImportRefused(f"{label}: its round input {gp.get('group_input')} is missing or not the "
                                            "one its records name")
                    inputs[gp["group_input"]] = {json.loads(x)["task_id"]: json.loads(x)
                                                 for x in inp.read_text().split("\n") if x.strip()}
                task = inputs[gp["group_input"]].get(tid)
                if task is None:
                    raise ImportRefused(f"{tid} {label}: not in its round's input")
                if round_of.setdefault(tid, r) != r:
                    raise ImportRefused(f"{tid}: sealed in rounds {round_of[tid]} and {r}")
                out, units, _ = TPL._outcome(task, rec, set(), mode=mode, thinking=thinking, stop_ids=stop_ids)
                ds = (task.get("source") or {}).get("dataset")
                t = tasks.setdefault(tid, {"split": task["split"], "category": task["category"],
                                           "shape": TE.episode_shape(task), "task_sha256": TPL.task_sha256(task),
                                           **({"dataset": ds} if ds not in POOL_V1_SOURCES else {}),
                                           "round": r, "candidates": []})
                t["candidates"].append({"candidate": k, "label": label, "shard": shard_of[tid][0],
                                        "shard_sha256": shard_of[tid][1], "outcome": out, "units": units})
        for tid, t in tasks.items():
            t["candidates"].sort(key=lambda c: c["candidate"])
            if t["candidates"][0]["candidate"] != 0:
                raise ImportRefused(f"{tid}: no sealed first candidate")
            if t["candidates"][0]["outcome"] == "admitted" and len(t["candidates"]) > 1:
                raise ImportRefused(f"{tid}: a second candidate after an admitted first")
            final = next((c for c in t["candidates"] if c["outcome"] == "admitted"), None)
            if final is None and len(t["candidates"]) < cand:
                raise ImportRefused(f"{tid}: refused at candidate 0 without its second candidate: not done")
            t["final"] = {"outcome": "admitted" if final else "rejected", "candidate": final["candidate"] if final else None,
                          "units": float(sum((final or {"units": {}})["units"].values()))}
        modes[mode] = {"tasks": tasks, "totals": {
            "tasks": len(tasks), "admitted": sum(t["final"]["outcome"] == "admitted" for t in tasks.values()),
            "candidates": sum(len(t["candidates"]) for t in tasks.values())},
            "reconciliation": reconcile(plan_then, lineage, mode, tasks, (sess.get("modes") or {}).get(mode) or {})}
    every = {tid: inputs_task for inp in inputs.values() for tid, inputs_task in inp.items()}
    comp = _session_compatibility(at, commit, lineage, pool_tasks,
                                  {t: every[t] for m in modes.values() for t in m["tasks"]})
    if not comp["ok"]:
        raise ImportRefused(f"the session's episodes are not the formal generation's: "
                            f"{[k for k, v in comp['checks'].items() if not v['equal']]}")
    bad = [f"{m}: {x}" for m, v in modes.items() for x in v["reconciliation"]["differ"]]
    if bad:
        raise ImportRefused("the imported cells do not reconcile with the session's record: " + "; ".join(bad[:5]))
    return {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "tool_session", "run_id": run_id, "kind": sess.get("kind"),
            "store": str(store.relative_to(REPO)) if REPO in store.resolve().parents else str(store),
            "identity": identity, "session_plan_sha256": _sha(at(PLAN_REL)),
            "lineage": lineage, "pool_content_sha256": plan_then["pool"]["content_sha256"],
            "verification_version": TE.VERIFICATION_VERSION, "compatibility": comp,
            "shards": {"count": len(shards_all), "sha256": {n: GD.sha(store / n) for n in sorted(shards_all)}},
            "modes": modes,
            "rule": "a task listed here is DONE in its mode: never attempted again, its second candidate never "
                    "repeated; a formal round takes only tasks this record does not list; its shards stay under "
                    "their own identity and are admitted beside the new ones",
            "authorizes": "nothing"}


def reconcile(plan: dict, lineage: str, mode: str, tasks: dict, recorded: dict) -> dict:
    """Per cell of the plan the session ran: the imported tasks, admissions and units -- plus what that plan's cell
    had itself imported (``sealed``) -- against the session's own record (`tool_session._report`, whose ``attempted``
    counts its new tasks only); ``differ`` lists every disagreement -- an import reconciles EXACTLY."""
    rows, differ = [], []
    cells = plan["lineages"][lineage]["modes"][mode]["cells"]
    rec_cells = {(c["split"], c["category"], c.get("group")): c for c in recorded.get("cells") or []}
    seen = set()
    for c in cells:
        mine = [tasks[t] for t in c["task_ids"] if t in tasks]
        seen |= {t for t in c["task_ids"] if t in tasks}
        base = c.get("sealed") or {"tasks": 0, "admitted": 0, "units": 0.0}
        adm = sum(t["final"]["outcome"] == "admitted" for t in mine)
        units = float(adm) if c["unit"] == "tasks" else float(sum(t["final"]["units"] for t in mine))
        got = rec_cells.get((c["split"], c["category"], c.get("group"))) or {}
        row = {"split": c["split"], "category": c["category"], **({"group": c["group"]} if c.get("group") else {}),
               "unit": c["unit"],
               "imported": {"tasks": len(mine), "admitted": adm, "units": units},
               "already_imported_by_that_plan": base,
               "session": {"attempted": got.get("attempted"), "admitted": got.get("admitted"),
                           "units": got.get("units"), "status": got.get("status")}}
        rows.append(row)
        want = (got.get("attempted"), got.get("admitted"), got.get("units"))
        if (len(mine), adm + int(base["admitted"]), round(units + float(base["units"]), 1)) != \
                (want[0], want[1], None if want[2] is None else round(float(want[2]), 1)):
            differ.append(f"{c['split']}/{c['category']}: imported {row['imported']} vs session {row['session']}")
    stray = sorted(set(tasks) - seen)
    if stray:
        differ.append(f"{len(stray)} sealed task(s) in no cell of the session's plan, e.g. {stray[:3]}")
    return {"cells": rows, "differ": differ}


def merged(records, lineage: str, mode: str) -> dict[str, dict]:
    """`outcomes` of every import record of ``lineage`` for ``mode``; a task two records list refuses."""
    out: dict[str, dict] = {}
    for r in records or []:
        if r.get("lineage") != lineage:
            continue
        for tid, o in outcomes(r, mode).items():
            if tid in out:
                raise ImportRefused(f"{tid} ({lineage} {mode}) is listed by two import records")
            out[tid] = o
    return out


def outcomes(record: dict, mode: str) -> dict[str, dict]:
    """``{task_id: {"split", "category", "shape", "admitted", "units"}}`` of one mode: what the session needs."""
    return {tid: {"split": t["split"], "category": t["category"], "shape": t["shape"],
                  "admitted": t["final"]["outcome"] == "admitted", "units": t["final"]["units"]}
            for tid, t in ((record.get("modes") or {}).get(mode, {}).get("tasks") or {}).items()}


def load(path: Path = IMPORT) -> dict:
    return json.loads(Path(path).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--session", action="store_true", help="the store is a completed formal tool_session.py store "
                    "(build_session), not the pilot's")
    ap.add_argument("--write", default=str(IMPORT))
    a = ap.parse_args(argv)
    from stages.teacher_modes import tool_pool as TP
    _, splits = TP.load_pool(Path(a.pool))
    pool_tasks = {t["task_id"]: t for rows in splits.values() for t in rows}
    doc = (build_session if a.session else build)(a.run, Path(a.store).resolve(), pool_tasks)
    Path(a.write).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({m: v["totals"] for m, v in doc["modes"].items()}), doc["compatibility"]["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

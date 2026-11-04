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
    doc, by_id = TPL.load(plan_path, tasks_path)
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
    ap.add_argument("--write", default=str(IMPORT))
    a = ap.parse_args(argv)
    from stages.teacher_modes import tool_pool as TP
    _, splits = TP.load_pool(Path(a.pool))
    pool_tasks = {t["task_id"]: t for rows in splits.values() for t in rows}
    doc = build(a.run, Path(a.store).resolve(), pool_tasks)
    Path(a.write).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({m: v["totals"] for m, v in doc["modes"].items()}), doc["compatibility"]["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

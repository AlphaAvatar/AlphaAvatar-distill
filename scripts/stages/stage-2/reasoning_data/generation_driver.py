#!/usr/bin/env python3
"""On-pod driver of H1's teacher generation (P1): bound inputs, two candidate passes, finished work sealed as it completes.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        -m stages.reasoning_data.generation_driver --authorization <authorization.json> \
        --status <file> --out <dir> [--resume-from <secured shards dir>]

Generation writes TEXT: teacher programs are never executed here (the
verifying sandbox is on the dev box, `code_sandbox.py`), so this pod's
credentials, network and volume never meet model-written code.

1. CONTRACT. The authorization is loaded under `governance.POLICY`; its phase
   names the plan it funds (`governance.plan_of`: P1's list or P1t's top-up);
   the launch identity is recomputed HERE from that plan and the protocol,
   inputs and pools on this machine and must equal the one the authorization
   binds; the harness on disk
   must digest to the authorized one; the protocol's sampling must equal the
   generator's. Only then is anything read from the protocol, and every value
   passed to the generator (teacher, revision, action protocol, session limit,
   seed base, system message) comes from these bound files. Nothing runs
   otherwise.
2. PASSES. Pass 0 runs each input group through the Stage-0 generator,
   `run_battery.py --mode generate` (one engine for the group: the teacher is
   loaded once), with ``--stream-completed``: every episode's record is
   appended the moment it ends. While the generator runs, the driver SEALS the
   stream's complete lines into shards (``<label>.partNNNN.jsonl``, each record
   tagged with its candidate identity) with a ``.done`` marker holding the
   shard's sha256, every ``seal_every_records`` records or
   ``seal_every_seconds`` (`generation_protocol.json :: execution`). The
   launcher pulls every marked shard on each poll, so finished candidates leave
   the pod before the group ends. A group is COMPLETE only when every task of
   its input has exactly one sealed record (``<label>.complete``).
3. PASS 1, the second candidate, only for tasks whose first failed, judged here
   by the adapter's own `candidate()` (pure Python), and none for the verifiers
   the protocol lists in ``no_second_candidate_verifiers`` (MBPP).
4. RESUME. ``--resume-from`` brings shards secured earlier into ``--out``; a
   stream left by a killed generator on this machine is sealed first. Each
   sealed shard is re-verified against its marker, and each record's launch
   identity must be this run's, before it counts. Only candidates without a
   sealed record are generated, under the same pass seed base, so a candidate's
   identity (task, pass, seed) never changes.
5. TERMINAL: one evidence record, ``MARKER:ALL_DONE`` only when every group of
   every pass is complete, else ``MARKER:RUN_FAILED`` with every sealed shard
   kept; the exit code; the generator process never outlives the driver.
   Teardown and provider reconciliation are the launcher's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import governance as GOV

from aadistill.governance.authorization import AuthorizationError

VLLM_PY = "/opt/vllm/bin/python"
RUNNER = "scripts/stages/stage-0/calibration_volume/run_battery.py"
UNIT_OK, UNIT_FAILED = "UNIT_OK", "UNIT_FAILED"      # contain neither session marker


class ShardError(RuntimeError):
    """A sealed shard that cannot be trusted: stop, preserve, count nothing from it."""


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize(path: Path, repo: Path) -> dict:
    """The contract: the authorization binds this machine's launch identity, of
    the plan its phase names, and harness."""
    auth = GOV.load(path)
    plan = GOV.plan_of(auth)
    lid = GOV.launch_hash(repo, plan)
    auth.require_plan(lid)
    harness = auth.require_harness(repo)
    return {"authorization_id": auth.authorization_id, "hard_cap_usd": auth.hard_cap_usd,
            "launch_identity_sha256": lid, "harness_digest": harness["digest"], "plan_path": str(plan)}


def contract(repo: Path, authorization: Path, authorize=authorize) -> dict:
    bound = authorize(authorization, repo)
    from stages.calibration_volume.run_battery import GENERATE_SAMPLING
    proto = json.loads((repo / GOV.PROTOCOL).read_text())
    sampling = {k: proto["decoding"][k] for k in GENERATE_SAMPLING}
    if sampling != GENERATE_SAMPLING:
        raise AuthorizationError(f"protocol sampling {sampling} is not the generator's {GENERATE_SAMPLING}")
    plan = json.loads((repo / bound["plan_path"]).read_text())
    tasks = {t["task_id"]: t for p in plan["pools"] for t in A.jsonl(repo / p["path"])}
    return {**bound, "protocol": proto, "plan": plan, "tasks": tasks}


def tag(rec: dict, meta: dict) -> dict:
    return {**rec, "candidate_index": meta["index"], "generation_pass": meta}


def sealed(out: Path, label: str, lid: str) -> dict[str, dict]:
    """task id -> its sealed record, from every marked shard of `label`; a shard
    whose bytes or identity do not verify refuses."""
    got: dict[str, dict] = {}
    for done in sorted(out.glob(f"{label}.part*.done")):
        shard = done.with_suffix(".jsonl")
        if not shard.is_file() or sha(shard) != done.read_text().strip():
            raise ShardError(f"{shard.name} does not hash to its marker")
        for rec in A.jsonl(shard):
            if rec["generation_pass"]["launch_identity_sha256"] != lid:
                raise ShardError(f"{shard.name}: {rec['task_id']} was generated under another launch identity")
            if rec["task_id"] in got:
                raise ShardError(f"{shard.name}: {rec['task_id']} sealed twice")
            got[rec["task_id"]] = rec
    return got


class Sealer:
    """Seals one stream's complete lines into marked shards of `label`."""

    def __init__(self, out: Path, label: str, stream: Path, meta: dict, want: set[str], have: set[str],
                 *, every_records: int, every_seconds: float):
        self.out, self.label, self.stream, self.meta = out, label, stream, meta
        self.want, self.have = want, set(have)
        self.every_records, self.every_seconds = every_records, every_seconds
        self.offset, self.last = 0, time.time()
        self.anomalies: list[str] = []

    def _next_part(self) -> Path:
        n = len(list(self.out.glob(f"{self.label}.part*.done")))
        return self.out / f"{self.label}.part{n:04d}.jsonl"

    def seal(self, *, force: bool = False) -> int:
        if not self.stream.is_file():
            return 0
        with open(self.stream, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        lines = [l for l in chunk[:end].decode().splitlines() if l.strip()]
        if not lines or (not force and len(lines) < self.every_records
                         and time.time() - self.last < self.every_seconds):
            return 0
        rows = []
        for line in lines:
            rec = json.loads(line)
            tid = rec.get("task_id")
            if tid not in self.want:
                self.anomalies.append(f"{tid}: not in this segment's input")
            elif tid in self.have:
                self.anomalies.append(f"{tid}: already sealed, second record discarded")
            else:
                self.have.add(tid)
                rows.append(tag(rec, self.meta))
        if rows:
            part = self._next_part()
            part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            part.with_suffix(".done").write_text(sha(part) + "\n")
        self.offset += end
        self.last = time.time()
        return len(rows)


def run_battery(cmd: list[str], log: Path, tick, *, every: float = 30.0) -> int:
    """The generator, with `tick` called while it runs; it never outlives this
    call. It executes no model output, so it keeps the session's environment
    (hub cache and access), as battery_driver's runs do."""
    with log.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=A.REPO, stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": "src:scripts"})
        try:
            while True:
                try:
                    rc = proc.wait(timeout=every)
                    break
                except subprocess.TimeoutExpired:
                    tick()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return rc


def second_pass(c: dict, records: dict[str, dict]) -> list[str]:
    """Task ids whose first candidate the adapter rejects, where the protocol
    allows a second (the verdict is computable here)."""
    from stages.reasoning_data.generation_adapter import candidate
    none_for = set(c["protocol"]["execution"]["no_second_candidate_verifiers"])
    retry = []
    for tid, rec in sorted(records.items()):
        task = c["tasks"][tid]
        if task.get("verifier") in none_for:
            continue
        if not candidate(task, rec, rec["candidate_index"]).accepted_for_training:
            retry.append(tid)
    return retry


def run_group(c: dict, out: Path, inputs: Path, f: dict, k: int, ids: list[str] | None, run, mark, *,
              label: str | None = None, extra_args: tuple[str, ...] = ()) -> dict:
    """One pass of one input group to completion, or as far as it gets. ``label`` and
    ``extra_args`` (runner flags appended to the command, such as a mode's template kwargs
    and sampling) are a caller's own group naming; H1 passes neither."""
    ex = c["protocol"]["execution"]
    label = label or f"pass{k}_{Path(f['file']).stem}"
    rows = [r for r in A.jsonl(inputs / f["file"]) if ids is None or r["task_id"] in ids]
    want = {r["task_id"] for r in rows}
    entry = {"label": label, "pass": k, "seed_base": ex["seed_base"] + k, "tasks": len(want), "segments": []}
    have = sealed(out, label, c["launch_identity_sha256"])
    if set(have) - want:
        raise ShardError(f"{label}: sealed records for tasks outside its input: {sorted(set(have) - want)[:3]}")
    meta = {"index": k, "seed_base": ex["seed_base"] + k, "group_input": f["file"], "group_input_sha256": f["sha256"],
            "launch_identity_sha256": c["launch_identity_sha256"],
            "protocol_sha256": sha(A.REPO / c.get("protocol_path", GOV.PROTOCOL)), "runner_sha256": sha(A.REPO / RUNNER)}
    # a stream a killed generator left on this machine: its finished lines count first
    for old in sorted(out.glob(f"{label}.seg*.stream.jsonl")):
        s = Sealer(out, label, old, {**meta, "segment": old.name}, want, have, every_records=1, every_seconds=0)
        s.seal(force=True)
        entry["segments"].append({"stream": old.name, "recovered": len(s.have) - len(have), "anomalies": s.anomalies})
        old.rename(old.with_name(old.name + ".recovered"))     # never read, never appended to again
        have = sealed(out, label, c["launch_identity_sha256"])
    todo = [r for r in rows if r["task_id"] not in have]
    if todo:
        used = [int(p.name[len(label) + 4:].split(".")[0]) for p in out.glob(f"{label}.seg*")]
        seg = max(used) + 1 if used else 0          # a new segment never reuses an old file
        seg_in = out / f"{label}.seg{seg}.input.jsonl"
        seg_in.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in todo))
        stream = out / f"{label}.seg{seg}.stream.jsonl"
        seg_meta = {**meta, "segment": seg, "segment_input_sha256": sha(seg_in)}
        sealer = Sealer(out, label, stream, seg_meta, {r["task_id"] for r in todo}, set(),
                        every_records=int(ex["completion_shards"]["seal_every_records"]),
                        every_seconds=float(ex["completion_shards"]["seal_every_seconds"]))
        proto = c["protocol"]["teacher"]
        cmd = [VLLM_PY, RUNNER, "--model", proto["model"], "--revision", proto["revision"],
               "--label", f"{label}.seg{seg}", "--tasks", str(seg_in), "--mode", "generate",
               "--session-limit", str(ex["session_limit"]), "--seed-base", str(ex["seed_base"] + k),
               "--system", f["system"], "--action-protocol", ex["action_protocol"],
               "--stream-completed", str(stream), "--out", str(out / f"{label}.seg{seg}.raw.jsonl"),
               *extra_args]
        try:
            rc = run(cmd, out / f"{label}.seg{seg}.log", sealer.seal)
        finally:
            sealer.seal(force=True)                     # whatever finished is sealed on every path
        entry["segments"].append({"segment": seg, "rc": rc, "todo": len(todo), "sealed": len(sealer.have),
                                  "anomalies": sealer.anomalies})
        have = sealed(out, label, c["launch_identity_sha256"])
    entry["sealed"] = len(have)
    entry["complete"] = set(have) == want
    if entry["complete"]:
        shards = sorted(out.glob(f"{label}.part*.jsonl"))
        (out / f"{label}.complete").write_text(json.dumps(
            {"label": label, "tasks": len(want), "shards": [{"name": s.name, "sha256": sha(s)} for s in shards]},
            indent=1) + "\n")
    entry["records"] = have
    return entry


def main(argv=None, *, run=run_battery, authorize=authorize, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default=None, help="shards secured by an earlier session")
    args = ap.parse_args(argv)
    repo = Path(repo or A.REPO)
    out, status = Path(args.out), Path(args.status)
    out.mkdir(parents=True, exist_ok=True)

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": "aadistill.stage2.reasoning_data.generation_session/v2",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "groups": []}
    evidence = out / "generation_session.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1) + "\n")

    try:
        c = contract(repo, Path(args.authorization), authorize)
        if args.resume_from:
            for p in sorted(Path(args.resume_from).glob("pass*.part*.*")):
                if p.suffix in (".jsonl", ".done") and not (out / p.name).exists():
                    shutil.copy2(p, out / p.name)
        record["contract"] = {k: c[k] for k in ("authorization_id", "hard_cap_usd", "launch_identity_sha256",
                                                "harness_digest", "plan_path")} | {"status": "ok"}
    except BaseException as exc:  # noqa: BLE001
        record["contract"] = {"status": "failed", "error": repr(exc)}
        save()
        mark("RUN_FAILED")
        return 2
    save()
    mark("CONTRACT_OK")
    inputs = repo / GOV.inputs_dir(c["plan"])
    complete = True
    for f in c["plan"]["inputs"]:
        ids = None
        for k in (0, 1):
            try:
                if k == 1:
                    if ids is None or not ids:
                        break
                g = run_group(c, out, inputs, f, k, ids, run, mark)
                records = g.pop("records")
                record["groups"].append(g)
                if not g["complete"]:
                    complete = False
                    mark(f"{UNIT_FAILED}:{g['label']}")
                    break                       # no second pass on an incomplete first
                mark(f"{UNIT_OK}:{g['label']}")
                if k == 0:
                    ids = second_pass(c, records)
                    g["second_candidate_tasks"] = len(ids)
            except BaseException as exc:  # noqa: BLE001
                complete = False
                record["groups"].append({"input": f["file"], "pass": k, "status": "failed", "error": repr(exc),
                                         "traceback": traceback.format_exc()[-3000:]})
                mark(f"{UNIT_FAILED}:pass{k}_{Path(f['file']).stem}")
                break
            save()
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if complete else "RUNS_INCOMPLETE")
    save()
    mark("ALL_DONE" if complete else "RUN_FAILED")
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())

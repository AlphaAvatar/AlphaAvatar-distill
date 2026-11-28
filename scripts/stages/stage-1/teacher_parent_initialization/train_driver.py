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
  again and re-hashed. The records are ``<out>/<unit>/savepoints.json``, pulled by the poll, so a replacement
  session (``--resume``) restores the latest one and the trainer resumes it (``--resume latest``): never a retrain of
  finished steps. The unit's previous save point is permanently deleted from the relay only after the new one HANDS
  OFF (`handoff`): its record is put beside its files (`relay.POINTER`) and the relay listing a replacement reads is
  confirmed to name it as the latest. A pod lost anywhere in the rotation leaves at least one listed save point, and
  `restore_savepoint` takes the latest one the relay still holds, whether or not the dev box ever saw its record;
* the final weights go to the relay (`final.json`), with the trainer's completion record, before the unit's
  evaluation starts, and hand off the same way before any save point is retired. A replacement pod restores them
  (`restore_final`) only when that completion record -- under either name the trainer writes -- is this unit's
  config finished to its last step, then runs only the evaluations without a formal report.

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
#: How the durable copies behave when the hub is slow or a read fails. A plan's ``durability`` block replaces each
#: value (SOB1's design carries its own); these are TPI1's measured rates, its plan having none: uploads 121-161 MB/s
#: and verify downloads 2.6-386 MB/s over its 20 save points and finals (the private-repo download path fell to
#: 2.6-4.6 MB/s for about two hours on 2026-11-17/18).
DURABILITY = {"listing_attempts": 5, "listing_wait_seconds": 60, "listing_timeout_seconds": 600,
              "handoff_attempts": 4, "handoff_wait_seconds": 120,
              "upload_floor_bytes_per_second": 120e6, "download_floor_bytes_per_second": 2.6e6,
              "transfer_margin": 1.25, "closeout_reserve_seconds": 1800, "memory_probe_steps": 0}
_sleep = time.sleep                     #: the waits between retries (a test replaces it)


class DriverError(RuntimeError):
    pass


class ListingUnreadable(DriverError):
    """The relay listing a replacement attempt restores from could not be read: what it holds is UNKNOWN."""


def durability(plan: dict | None = None) -> dict:
    return {**DURABILITY, **((plan or {}).get("durability") or {})}


def savepoint_copies(d: dict | None) -> int:
    """How many save points of a unit the relay may hold at once (``durability.savepoint_copies``): 2, the registered
    rotation (a new one is uploaded while the previous is still there, which goes only after the new one hands off);
    or 1, a single copy (the previous is removed before the next upload, and the last before the final's upload),
    for a relay without room for two. A single copy has a window per save point -- its upload -- in which the unit
    holds no durable save point."""
    n = int((d or {}).get("savepoint_copies", 2))
    if n not in (1, 2):
        raise DriverError(f"durability.savepoint_copies must be 1 or 2, not {n}")
    return n


def remove_before_upload(ev_unit: Path, name: str, before: str) -> bool:
    """Single-copy rotation: every verified, unretired save point of the unit removed from the relay before the
    upload of ``before`` (a later save point, or the final). True only when the relay confirms each one absent."""
    sp_path = Path(ev_unit) / "savepoints.json"
    recs = read_json(sp_path) if sp_path.is_file() else []
    ok = True
    for sp in recs:
        if sp.get("verified") and not retired(sp) and sp.get("tag") != before:
            ok = remove_now(sp, f"{name}: {sp['tag']} removed before {before}'s upload (single copy)") and ok
            write_json(sp_path, recs)
    return ok


def transfer_seconds(nbytes: float, d: dict, *, up: bool = True, down: bool = True) -> float:
    """The longest a copy of ``nbytes`` takes at the floor rates: upload, the verifying download, or both."""
    t = (nbytes / d["upload_floor_bytes_per_second"] if up else 0.0) + \
        (nbytes / d["download_floor_bytes_per_second"] if down else 0.0)
    return t * d["transfer_margin"]


def list_records(prefix: str, d: dict | None = None) -> dict:
    """The relay's records under ``prefix`` (`relay.records`), read up to ``listing_attempts`` times; raises
    `ListingUnreadable` when no attempt reads it. An unreadable listing is never read as an empty one."""
    from stages.teacher_parent_initialization import relay as R
    d = d or DURABILITY
    errors = []
    for k in range(int(d["listing_attempts"])):
        got = R.run_online("records", {"prefix": prefix}, timeout_s=float(d["listing_timeout_seconds"]))
        if got.get("verified"):
            return {**got, "attempts": k + 1}
        errors.append(str(got.get("error"))[:200])
        if k + 1 < int(d["listing_attempts"]):
            _sleep(float(d["listing_wait_seconds"]))
    raise ListingUnreadable(f"the relay listing under {prefix} was unreadable {len(errors)} times: {errors[-1]}")


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


def protocol(plan: dict | None = None) -> dict:
    """The evaluation protocol in the shape `reasoning_inheritance.driver.fluid` / `capability` read: the plan's own
    (``evaluation_protocol``, already in that shape) or TPI1's design's evaluation block."""
    if plan and plan.get("evaluation_protocol"):
        return plan["evaluation_protocol"]
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


def _pointer_body(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("handoff", "removed", "downloaded_sha256")}


def handoff(rec: dict, scratch: Path, listing: str, message: str, *, newest: bool, d: dict | None = None) -> dict:
    """Make ``rec`` findable off the pod before anything it supersedes is deleted: ``rec`` itself is put beside its
    files as `relay.POINTER` (uploaded, downloaded again, re-hashed), then the relay listing a replacement attempt
    restores from (``listing``) is read back and must name ``rec`` with its files present -- and, for a save point
    (``newest``), as the latest one. Only a verified handoff lets the caller remove the previous record."""
    from stages.teacher_parent_initialization import relay as R
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / f".pointer_{os.getpid()}_{time.time_ns()}.json"
    tmp.write_text(json.dumps(_pointer_body(rec), indent=1, sort_keys=True, default=str) + "\n")
    try:
        ptr = R.run_online("put", {"files": {R.POINTER: str(tmp)}, "prefix": rec["prefix"], "scratch": str(scratch),
                                   "message": f"{message}: its record"})
    finally:
        tmp.unlink(missing_ok=True)
    out = {"pointer": {k: ptr.get(k) for k in ("prefix", "files_sha256", "verified", "utc", "error", "refused")}}
    if not ptr.get("verified"):
        return {**out, "verified": False, "why": "the record's own copy on the relay did not verify"}
    try:
        listed = list_records(listing, d)
    except ListingUnreadable as exc:
        return {**out, "listing": listing, "verified": False, "utc": utc(), "why": f"the relay listing: {exc}"}
    present = [r["record"] for r in listed.get("records", []) if r.get("present")]
    mine = [r for r in present if r.get("prefix") == rec["prefix"] and r.get("files_sha256") == rec["files_sha256"]]
    later = [r for r in present if newest and int(r.get("step", -1)) > int(rec.get("step", -1))]
    ok = bool(mine) and not later
    return {**out, "listing": listing, "listed_present": len(present), "verified": ok, "utc": utc(),
            "why": None if ok else ("the listing does not name this record with its files" if not mine else
                                    "the listing names a later save point")}


def hand_off_with_retries(rec: dict, scratch: Path, listing: str, message: str, *, newest: bool,
                          d: dict | None = None) -> dict:
    """`handoff` until it verifies or ``handoff_attempts`` run out. Only the record's own pointer is (re)written and
    the listing read: the weights the record names were uploaded and verified once and are never sent again."""
    d = d or DURABILITY
    tries = []
    for k in range(int(d["handoff_attempts"])):
        h = handoff(rec, scratch, listing, message, newest=newest, d=d)
        tries.append({k2: h.get(k2) for k2 in ("verified", "why", "utc")})
        if h["verified"]:
            return {**h, "attempts": tries}
        if k + 1 < int(d["handoff_attempts"]):
            _sleep(float(d["handoff_wait_seconds"]))
    return {**h, "attempts": tries}


def retired(rec: dict) -> bool:
    """A record is retired only when the relay CONFIRMED its objects absent (`relay.remove`'s ``verified_absent``).
    A removal that failed or timed out (``{"verified": False, ...}``) leaves the record PENDING: still on the relay,
    still counted, retried before the next large upload. Records written before this rule carry the same field."""
    return (rec.get("removed") or {}).get("verified_absent") is True


def remove_now(rec: dict, message: str) -> bool:
    """One removal attempt of ``rec`` (`relay.remove`, permanent): its answer is kept as ``removed`` and appended to
    ``remove_attempts``; True only when the relay confirms the objects absent."""
    from stages.teacher_parent_initialization import relay as R
    got = R.run_online("remove", {"record": with_pointer(rec), "message": message})
    rec.setdefault("remove_attempts", []).append({k: got.get(k) for k in ("verified_absent", "error", "utc")})
    rec["removed"] = got
    return retired(rec)


def superseded(rec: dict, recs: list[dict], *, final_handed_off: bool = False) -> bool:
    """Whether ``rec`` may be removed: a LATER save point of the unit has handed off, or the unit's final has. The
    latest resumable checkpoint is never superseded by anything that is not itself published."""
    return final_handed_off or any(o is not rec and int(o.get("step", -1)) > int(rec.get("step", -1))
                                   and o.get("verified") and (o.get("handoff") or {}).get("verified") for o in recs)


def retry_pending_removals(ev_unit: Path, name: str, *, final_handed_off: bool = False) -> dict:
    """Every PENDING removal of the unit (verified, not confirmed absent, superseded): tried again, before the next
    large upload and at a takeover. Returns what was freed (only confirmed removals count) and what is still pending."""
    p = Path(ev_unit) / "savepoints.json"
    recs = read_json(p) if p.is_file() else []
    freed, pending = [], []
    for r in recs:
        if not r.get("verified") or retired(r) or "removed" not in r:
            continue                         #: never attempted: the rotation removes it when its time comes
        if not superseded(r, recs, final_handed_off=final_handed_off):
            continue
        ok = remove_now(r, f"{name}: {r['tag']} superseded (retry of a pending removal)")
        write_json(p, recs)
        (freed if ok else pending).append(r["tag"])
    return {"freed": freed, "freed_bytes": sum(sum((r.get("bytes") or {}).values()) for r in recs if r["tag"] in freed),
            "pending": pending, "utc": utc()}


def with_pointer(rec: dict) -> dict:
    """The record whose removal also deletes its pointer (a record written before pointers existed has none)."""
    from stages.teacher_parent_initialization import relay as R
    ptr = ((rec.get("handoff") or {}).get("pointer") or {}).get("files_sha256") or {}
    return {**rec, "files_sha256": {**rec["files_sha256"], **({R.POINTER: ptr[R.POINTER]} if R.POINTER in ptr else {})}}


class SavepointMirror(threading.Thread):
    """Copies each save point the trainer completes to the relay, then removes the unit's previous one there."""

    def __init__(self, unit: str, out_dir: Path, ev_unit: Path, total_steps: int, prefix: str, every: float = WATCH_EVERY_S,
                 d: dict | None = None):
        super().__init__(daemon=True)
        self.unit, self.out_dir, self.ev_unit, self.total = unit, Path(out_dir), Path(ev_unit), int(total_steps)
        self.prefix, self.every, self.d = prefix, every, d or DURABILITY
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
        if step >= self.total:
            return None
        rec = next((r for r in recs if r["tag"] == tag and r.get("verified")), None)
        if rec is not None and (rec.get("handoff") or {}).get("verified"):
            return None
        if rec is None:
            retry_pending_removals(self.ev_unit, self.unit)     #: a failed removal frees its bytes before the upload
            if savepoint_copies(self.d) == 1 and not remove_before_upload(self.ev_unit, self.unit, tag):
                return None                     #: the relay holds no room for a second copy: retried next round
            recs = self.records()
            ckpt = self.out_dir / "checkpoints" / tag
            files = savepoint_files(ckpt)
            nbytes = sum(f.stat().st_size for f in files.values())
            #: the time a copy of these bytes can take at the floor rates, never a fixed hour: a 2B save point is
            #: about 22 GB, and a verifying download at 2.6 MB/s takes about 2.4 h
            res = R.run_online("put", {"files": {k: str(v) for k, v in files.items()},
                                       "prefix": f"{self.prefix}/savepoints/{tag}", "scratch": str(self.out_dir),
                                       "message": f"{self.unit}: save point {tag}"},
                               timeout_s=transfer_seconds(nbytes, self.d) + 600)
            rec = {"tag": tag, "step": step, **res, "latest_txt": tag}
            recs.append(rec)
            write_json(self.path, recs)         #: what the poll pulls, before anything on the relay is deleted
            if not rec.get("verified"):
                return rec
        #: uploaded and verified; only its record's publication is (re)tried, never the weights' upload
        h = handoff(rec, self.out_dir, f"{self.prefix}/savepoints", f"{self.unit}: {tag}", newest=True, d=self.d)
        rec["handoff_tries"] = int(rec.get("handoff_tries", 0)) + 1
        rec["handoff"] = h
        write_json(self.path, recs)
        if not rec["handoff"]["verified"]:
            return rec                          #: the previous save point stays until one hands off
        for old in recs:
            if old is not rec and old["step"] < rec["step"] and old.get("verified") and not retired(old):
                remove_now(old, f"{self.unit}: {old['tag']} superseded by {tag}")
                write_json(self.path, recs)     #: a failure stays PENDING (`retired`): retried before the next upload
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


def restore_savepoint(ev_unit: Path, out_dir: Path, prefix: str | None = None, d: dict | None = None) -> dict | None:
    """The latest save point into ``out_dir/checkpoints``; None only when there is none to restore. Candidates are the
    unit's verified, unremoved records in the evidence and, with ``prefix``, the pointers the relay lists under
    ``<prefix>/savepoints`` with their files present (a rotation whose records the lost pod never handed to the dev
    box). A save point is removed only after a later one handed off (`handoff`), so the latest candidate is one the
    relay holds.

    The listing is read with retries (`list_records`). If it still cannot be read, the evidence's own records are
    used when there are any (a later one the relay might hold is then given up, and the steps after it are trained
    again); with none, the restore REFUSES (`ListingUnreadable`): an unreadable listing never sends a unit back to
    step 0, because only a listing that was read can show there is no save point to resume."""
    from stages.teacher_parent_initialization import relay as R
    d = d or DURABILITY
    p = ev_unit / "savepoints.json"
    held = read_json(p) if p.is_file() else []
    local = [r for r in held if r.get("verified") and not retired(r)]
    found = "evidence"
    cands = {r["tag"]: r for r in local}
    if prefix:
        try:
            listed = list_records(f"{prefix}/savepoints", d)
        except ListingUnreadable as exc:
            if not cands:
                raise ListingUnreadable(f"{ev_unit.name}: no save point in the evidence, and whether the relay holds "
                                        f"one is unknown ({exc}); refusing to train from step 0") from exc
            found = f"evidence (the relay listing was unreadable: {str(exc)[:200]})"
        else:
            for x in listed.get("records", []):
                if x.get("present") and x["record"].get("verified"):
                    cands.setdefault(x["record"]["tag"], x["record"])
            found = f"evidence and relay listing (read in {listed['attempts']} attempt(s))"
    if not cands:
        return None
    rec = max(cands.values(), key=lambda r: r["step"])
    #: every listed save point the evidence did not hold (a pod lost before the dev box pulled its record) enters
    #: savepoints.json, so the rotation retires the superseded ones and keeps counting what the relay holds
    known = {r["tag"] for r in held}
    extra = [{**c, "found": "relay listing"} for t, c in sorted(cands.items()) if t not in known]
    if extra:
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, held + extra)
    ckpt = out_dir / "checkpoints" / rec["tag"]
    nbytes = sum((rec.get("bytes") or {}).values())
    got = R.run_online("get", {"record": rec, "dest": str(ckpt)},
                       timeout_s=max(3600.0, transfer_seconds(nbytes, d, up=False) + 600))
    if not got.get("verified"):
        raise DriverError(f"save point {rec['tag']} did not restore: {str(got)[:500]}")
    (out_dir / "checkpoints" / "latest.txt").write_text(rec["tag"] + "\n")
    #: the trainer appends its log across resumes (`resume_loaded`, then the steps after the save point); on a
    #: replacement pod that log is the one the previous attempt secured, or the mirror would overwrite it with the tail
    if (ev_unit / "train" / "train_log.jsonl").is_file() and not (out_dir / "train_log.jsonl").is_file():
        shutil.copyfile(ev_unit / "train" / "train_log.jsonl", out_dir / "train_log.jsonl")
    return {"tag": rec["tag"], "step": rec["step"], "from": got.get("from"), "found": found, "utc": utc()}


def final_record(ev_unit: Path, prefix: str | None, d: dict | None = None) -> tuple[dict, str] | None:
    """The unit's verified final-weights record: the evidence's ``final.json``, else the one pointer the relay lists
    under ``<prefix>/final`` with its files present (written there before any save point was retired)."""
    from stages.teacher_parent_initialization import relay as R
    p = ev_unit / "final.json"
    rec = read_json(p) if p.is_file() else None
    if rec and rec.get("verified"):
        return rec, "evidence"
    if not prefix:
        return None
    try:
        listed = list_records(f"{prefix}/final", d)
    except ListingUnreadable as exc:
        raise ListingUnreadable(f"{ev_unit.name}: no final.json and {exc}") from exc
    there = [x["record"] for x in listed["records"] if x.get("present") and x["record"].get("verified")]
    if len(there) > 1:
        raise DriverError(f"{ev_unit.name}: the relay lists {len(there)} final weights: {[r['prefix'] for r in there]}")
    return (there[0], "relay listing") if there else None


def restore_final(ev_unit: Path, out_dir: Path, cfg: dict, prefix: str | None = None,
                  d: dict | None = None) -> dict | None:
    """A trained unit's final weights from their verified relay copy; None when the unit has none. The completion
    record is the trainer's own, under either of its names (`a1_units.completion_records`: a resumed run writes
    ``run_completion_resume_step<N>.json``), from the evidence or the final's relay copy, and must be this config's
    finished run (`a1_units.completion`: the config hash, the planned and completed steps); the weights are refused
    unless they hash to the record. A final whose completion does not check out stops the unit: it is never adopted
    and never silently retrained."""
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_parent_initialization import relay as R
    steps = int(cfg["schedule"]["total_steps"])
    d = d or DURABILITY
    hit = final_record(ev_unit, prefix, d)
    if hit is None:
        return None
    rec, found = hit
    if found != "evidence":
        write_json(ev_unit / "final.json", rec)
    train = ev_unit / "train"
    carried = sorted(k for k in rec["files_sha256"] if k.startswith("run_completion"))
    if not AU.completion_records(train) and carried:
        got = R.run_online("get", {"record": rec, "dest": str(train), "only": carried}, timeout_s=600)
        if not got.get("verified"):
            raise DriverError(f"the final's completion record did not restore: {str(got)[:500]}")
    try:
        done, crec = AU.completion(train, config=cfg, steps=steps)
    except AU.CompletionError as exc:
        raise DriverError(f"{ev_unit.name}: final weights are recorded but no completion of this unit's training "
                          f"checks out: {exc}") from exc
    model = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
    only = [n for n in AU.FINAL_FILES if n in rec["files_sha256"]]
    nbytes = sum(v for k, v in (rec.get("bytes") or {}).items() if k in only)
    got = R.run_online("get", {"record": rec, "dest": str(model), "only": only},
                       timeout_s=max(3600.0, transfer_seconds(nbytes, d, up=False) + 600))
    if not got.get("verified"):
        raise DriverError(f"the final weights did not restore: {str(got)[:500]}")
    shutil.copyfile(done, out_dir / done.name)            #: its own name: `a1_units.state` reads either
    if (train / "train_log.jsonl").is_file() and not (out_dir / "train_log.jsonl").is_file():
        shutil.copyfile(train / "train_log.jsonl", out_dir / "train_log.jsonl")
    return {"from": got.get("from"), "found": found, "completion": done.name, "final_step": crec["final_step"],
            "config_sha256": crec["config_sha256"], "model_safetensors_sha256": rec["files_sha256"]["model.safetensors"],
            "utc": utc()}


def fetch_init(u: dict, d: dict | None = None) -> dict:
    """The unit's initialization at the plan's student path, from its verified relay copy, every file re-hashed."""
    from stages.teacher_parent_initialization import relay as R
    rec = u["initialization"]["relay"]
    dest = Path(u["config"]["student_path"])
    sub = {k: v for k, v in rec["files_sha256"].items() if k.startswith("init/")}
    flat = {**rec, "files_sha256": {k.removeprefix("init/"): v for k, v in sub.items()},
            "prefix": f"{rec['prefix']}/init"}
    nbytes = sum(v for k, v in (rec.get("bytes") or {}).items() if k.startswith("init/"))
    got = R.run_online("get", {"record": flat, "dest": str(dest)},
                       timeout_s=max(3600.0, transfer_seconds(nbytes, d or DURABILITY, up=False) + 600))
    if not got.get("verified"):
        raise DriverError(f"{u['unit']}'s initialization did not restore: {str(got)[:500]}")
    want = u["config"]["lineage"]["initialization"]["model_safetensors_sha256"]
    if R.sha256_file(dest / "model.safetensors") != want:
        raise DriverError(f"{u['unit']}'s initialization on the pod is not the plan's")
    return {"path": str(dest), "from": got.get("from"), "utc": utc()}


def training_deadline_s(unit_deadline_hours: float, session_deadline_utc: str | None, savepoint_bytes: float,
                        final_bytes: float, d: dict, now: float | None = None) -> dict:
    """How long the trainer may run: the unit's own bound, shortened so that the last save point and the final weights
    can still be copied and verified at the floor rates before the session's hard deadline. ``seconds`` <= 0 means
    the session cannot secure what training would produce, and training must not start."""
    import calendar
    unit_s = float(unit_deadline_hours) * 3600
    if not session_deadline_utc:
        return {"seconds": unit_s, "rule": "the unit's deadline (no session deadline given)"}
    end = calendar.timegm(time.strptime(session_deadline_utc, "%Y-%m-%dT%H:%M:%SZ"))
    reserve = (transfer_seconds(savepoint_bytes, d) + transfer_seconds(final_bytes, d)
               + float(d["closeout_reserve_seconds"]))
    left = end - (time.time() if now is None else now) - reserve
    return {"seconds": min(unit_s, left), "session_deadline_utc": session_deadline_utc,
            "reserve_seconds": round(reserve, 1), "session_left_after_reserve_s": round(left, 1),
            "rule": "min(the unit's deadline, the session's hard deadline - now - the floor-rate copy of one save "
                    "point and the final - the closeout reserve)"}


def memory_telemetry(out_dir: Path) -> dict:
    """The trainer's ``memory_probe`` and ``memory_oom`` events (`aadistill.training.train.MEMORY_PROBE_ENV`): per
    probed step the peak allocated / reserved GiB and the device's free GiB after each phase, and, after an
    out-of-memory error, the phase it was raised in."""
    log = Path(out_dir) / "train_log.jsonl"
    probes, ooms = [], []
    if log.is_file():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "memory_probe":
                ph = row.get("phases") or []
                probes.append({"step": row.get("step"),
                               "peak_allocated_gib": max((x.get("peak_allocated_gib") or 0 for x in ph), default=None),
                               "peak_reserved_gib": max((x.get("peak_reserved_gib") or 0 for x in ph), default=None),
                               "min_device_free_gib": min((x.get("device_free_gib") for x in ph
                                                           if x.get("device_free_gib") is not None), default=None),
                               "device_total_gib": next((x.get("device_total_gib") for x in ph
                                                         if x.get("device_total_gib")), None),
                               "phases": ph})
            elif row.get("event") == "memory_oom":
                ooms.append({k: v for k, v in row.items() if k not in ("event",)})
    return {"probes": probes, "oom": ooms,
            "oom_phase": ooms[-1].get("phase_in_progress") if ooms else None}


def retire_save_points(ev_unit: Path, name: str) -> None:
    """Every verified save point of the unit, permanently removed from the relay: called only once its final weights
    have handed off, so a replacement attempt finds the final instead."""
    from stages.teacher_parent_initialization import relay as R
    sp_path = ev_unit / "savepoints.json"
    recs = read_json(sp_path) if sp_path.is_file() else []
    for sp in recs:
        if sp.get("verified") and not retired(sp):
            remove_now(sp, f"{name}: {sp['tag']} superseded by the verified final weights")
            write_json(sp_path, recs)       #: a failure stays PENDING: `retry_pending_removals` tries it again


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


def evaluate(model_dir: Path, ev_dir: Path, label: str, mode: str, vllm_py: str, attempt: str, work: Path,
             ev: dict | None = None, package_fn=None) -> dict:
    """Every eval-set-1 unit in ``mode``, resumable by sample, then the formal report (`eval_set_scoring.report`).
    ``ev``: the protocol (`protocol`); TPI1's design's when not given. ``package_fn(model_dir, dest)``: the study's
    evaluation package; A-MIX's (`package`) when not given."""
    from stages.reasoning_inheritance import driver as RID
    import shared.evaluation.eval_set_scoring as S
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}       #: the driver's own (absolute) PYTHONPATH, as RI-LM1's
    ev = ev or protocol()
    pkg = work / "packages" / f"{label}_{mode}"
    rec = {"package": (package_fn or package)(model_dir, pkg)}
    write_json(ev_dir / "package.json", rec["package"])
    rec["fluid"] = RID.fluid(pkg, ev_dir, label, mode, vllm_py, env, attempt, None, ev)
    rec["capability"] = RID.capability(pkg, ev_dir, label, mode, vllm_py, env, attempt, None, ev)
    rep = S.report(ev_dir / "generations", label, {}, ev["eval_set"], rows_out=ev_dir / "rows")
    write_json(ev_dir / "formal_report.json", rep)
    return rec


# ---- the state machine ----------------------------------------------------------------------------------------------------

def is_live(rel: Path) -> bool:
    """A stream still being written: anything under a ``live/`` directory at ANY depth (the evaluation runners write
    theirs beside the units they serve: ``<unit dir>/../../live``, which is ``evaluation/live`` in SOB1's layout)."""
    return "live" in Path(rel).parts[:-1]


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
        if not p.is_file() or p.name.startswith(".") or is_live(rel):
            continue
        dest = out / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        taken.append(str(rel))
    return {"from": str(resume), "files": len(taken), "utc": utc()}


def main(argv=None, *, study_launch=None) -> int:
    """``study_launch``: the module whose ``STUDY``, ``TRAIN_KIND`` and authorization class (``Authorization``, or
    TPI1's ``TPAuthorization``) bind this session; TPI1's launch when not given. The plan (``--plan``), the relay
    prefix root (``--prefix-root``) and each unit's arm (the plan row's ``arm``) come from arguments and the plan."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--vllm-python", default="/opt/vllm/bin/python")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--attempt", default=None)
    ap.add_argument("--plan", default=None, help="the training plan (repo-relative or absolute); TPI1's by default")
    ap.add_argument("--prefix-root", default=None, help="the relay prefix every unit's durable copies live under")
    ap.add_argument("--units", default=None, help="a comma-separated subset of the plan's units, in plan order")
    ap.add_argument("--session-deadline-utc", default=None,
                    help="the session's hard deadline: training stops early enough to secure its last save point")
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
        from stages.teacher_parent_initialization import plan as PL
        from stages.teacher_parent_initialization import relay as R
        from shared import study_engineering as SE
        from shared.pod.teacher_cache import local_snapshot
        if study_launch is None:
            from stages.teacher_parent_initialization import launch as study_launch
        L = study_launch
        auth_cls = getattr(L, "Authorization", None) or L.TPAuthorization
        auth = auth_cls.load(a.authorization)
        record["authorization"] = SE.require_on_pod(auth, L.STUDY, SE.identity_hash(L.STUDY, REPO), str(REPO),
                                                    kind=L.TRAIN_KIND).get("authorization_id")
        if a.resume:
            record["resumed_from"] = take_over(Path(a.resume), out)
        plan_path = PLAN if a.plan is None else (Path(a.plan) if Path(a.plan).is_absolute() else REPO / a.plan)
        plan = read_json(plan_path)
        dur = durability(plan)
        prefix_root = a.prefix_root or plan.get("relay_prefix_root") or R.ROOT_PREFIX
        ev_protocol = protocol(plan)
        record.update(plan=str(plan_path), plan_sha256=R.sha256_file(plan_path), prefix_root=prefix_root,
                      durability=dur)
        if "unit_order" in plan:
            us = [next(u for u in plan["units"] if u["unit"] == n) for n in plan["unit_order"]]
        else:
            us = sorted(plan["units"], key=lambda u: plan["seed_order"].index(u["seed"]))
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
            name, cfg = u["unit"], u["config"]
            row = record["units"].setdefault(name, {})
            ev_unit = out / name
            out_dir = Path(cfg["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            steps = int(cfg["schedule"]["total_steps"])
            evals = [{**e, "path": ev_unit / e["dir"]} for e in u["evaluations"]]
            st = AU.state(out_dir, [e["path"] for e in evals])
            unit_prefix = f"{prefix_root}/{u.get('arm') or PL.ARM}/{name}"
            if st == "fresh":                   #: a replacement pod: the final weights, else the latest save point
                got = restore_final(ev_unit, out_dir, cfg, unit_prefix, dur)
                if got:
                    row["restored_final"] = got
                else:
                    got = restore_savepoint(ev_unit, out_dir, unit_prefix, dur)
                    if got:
                        row["restored_savepoint"] = got
                st = AU.state(out_dir, [e["path"] for e in evals])
            row["state_at_start"] = st
            save()
            if st == "done":
                continue
            if st in ("fresh", "resume"):
                if not Path(cfg["student_path"], "model.safetensors").is_file():
                    row["init"] = fetch_init(u, dur)
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
                sizes = u.get("bytes") or {}
                row["training_deadline"] = training_deadline_s(
                    a.unit_deadline_hours, a.session_deadline_utc, float(sizes.get("save_point", 0)),
                    float(sizes.get("final", 0)), dur)
                save()
                if row["training_deadline"]["seconds"] <= 0:
                    raise DriverError(f"{name}: the session cannot secure a save point before its hard deadline "
                                      f"({row['training_deadline']}); not starting the trainer")
                mirror = SavepointMirror(name, out_dir, ev_unit, steps, unit_prefix, d=dur)
                mirror.start()
                mark(f"TRAIN_START:{name}")
                t0 = time.time()
                #: execution only: the trainer logs each phase's memory for its first steps (`memory_probe`) and the
                #: phase of an out-of-memory error (`memory_oom`), which the driver reads back below
                os.environ["AADISTILL_MEMORY_PROBE_STEPS"] = str(int(dur.get("memory_probe_steps") or 0))
                try:
                    rc = AU.train(cfg_path, resume=st == "resume", deadline_s=row["training_deadline"]["seconds"],
                                  log=out_dir / "train_stdout.log")
                finally:
                    mirror.close()
                row.update(train_rc=rc, train_seconds=round(time.time() - t0, 1), telemetry=AU.train_telemetry(out_dir),
                           memory=memory_telemetry(out_dir))
                save()
                if rc != 0:
                    mirror.mirror_once()        #: the last completed save point is durable before the session stops
                    oom = row["memory"].get("oom_phase")
                    raise DriverError(f"{name}: training exited {rc} ("
                                      f"{'deadline' if rc == 124 else f'out of memory in {oom}' if oom else 'failure'})")
                if row["telemetry"].get("context_strategies") != [PTP.CONTEXT_STRATEGY]:
                    raise DriverError(f"{name}: the trainer logged {row['telemetry'].get('context_strategies')}")
                mark(f"TRAINED:{name}")
            model_dir = AU.final_model(out_dir, steps)
            try:
                done, _ = AU.completion(out_dir, config=cfg, steps=steps)
            except AU.CompletionError as exc:
                raise DriverError(f"{name}: {exc}") from exc
            row["completion_record"] = done.name
            row["final_model_safetensors_sha256"] = R.sha256_file(model_dir / "model.safetensors")
            fin = ev_unit / "final.json"
            prev = read_json(fin) if fin.is_file() else {}
            if not (prev.get("verified") and prev["files_sha256"].get("model.safetensors")
                    == row["final_model_safetensors_sha256"]):
                files = {n: str(model_dir / n) for n in AU.FINAL_FILES if (model_dir / n).is_file()}
                files[done.name] = str(done)    #: the training's completion travels with its weights
                nbytes = sum(Path(f).stat().st_size for f in files.values())
                prev = R.run_online("put", {"files": files, "scratch": str(out_dir),
                                            "prefix": f"{unit_prefix}/final/{row['final_model_safetensors_sha256'][:16]}",
                                            "message": f"{name}: final weights"},
                                    timeout_s=transfer_seconds(nbytes, dur) + 600)
                write_json(fin, prev)
                if not prev.get("verified"):
                    raise DriverError(f"{name}: the final weights' relay copy did not verify: {str(prev)[:500]}")
            if not (prev.get("handoff") or {}).get("verified"):
                #: the weights are on the relay and verified: only their record's publication is retried
                prev["handoff"] = hand_off_with_retries(prev, out_dir, f"{unit_prefix}/final", f"{name}: final weights",
                                                        newest=False, d=dur)
                write_json(fin, prev)
            if prev["handoff"]["verified"]:     #: a replacement attempt finds the final: the save points can go
                retire_save_points(ev_unit, name)
            else:
                row["save_points_kept"] = prev["handoff"].get("why")
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
                    **evaluate(model_dir, e["path"], name, mode, a.vllm_python, attempt, out_dir.parent / "work",
                               ev_protocol, getattr(L, "package", None)),
                    "eval_seconds": round(time.time() - t0, 1), "template": e["template"]}
                save()
                mark(f"EVAL_DONE:{name}:{e['mode']}")
            if not prev["handoff"]["verified"]:     #: one more try after the evaluations, before the unit closes
                prev["handoff"] = hand_off_with_retries(prev, out_dir, f"{unit_prefix}/final",
                                                        f"{name}: final weights", newest=False, d=dur)
                write_json(fin, prev)
                if prev["handoff"]["verified"]:
                    retire_save_points(ev_unit, name)
                    row.pop("save_points_kept", None)
                else:
                    row["save_points_kept"] = prev["handoff"].get("why")
                save()
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

#!/usr/bin/env python3
"""The L40S tool pilot: frozen cells and task file, the pilot-only source admission, GPU telemetry, the report. $0.

    # the frozen pilot plan (cells with exact task ids, every task's sha256) and its task file, from a verified pool
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot build --pool <pool> \\
        --plan-out logs/stages/stage-2/teacher_modes/plans/tool_pilot_plan.json \\
        --tasks-out artifacts/stages/stage-2/teacher_modes/tool_pilot/pilot_tasks.jsonl
    # the pilot-only source admission record (PREPARED: inactive until the maintainer signs it)
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot admission \\
        --write logs/stages/stage-2/teacher_modes/audit/tool_pilot_source_admission.json
    # the application for the maintainer (identity, the runner's hard-cap check per option, commands, gates)
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot application --live-position
    # after a session: the measurements, from its sealed shards and telemetry
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot report --session <secured dir> --write <out>

The pilot is the design the maintainer accepted in principle on 2026-11-04 (review of 2468c1aa): the hybrid in both
modes, one round, the tasks `tool_generation_plan.pilot_sample` derives from the verified plan (794 at the dry pool
v4). Its session is `tool_session.py --pilot` under its own kind ``tool_pilot`` (`token_matched_phase.py`), which
refuses until the maintainer's grant names its hard cap and the source admission below is signed. Nothing here
generates or spends.

* `build` -- the frozen cells: the plan built over a pool that `tool_pool.load_pool` verified, the pilot sample of it,
  every task's canonical sha256, and the task file the pod receives instead of the whole pool (P8.4: the session
  reads these tasks and no other); bound to the pool's content, its sources, the protocol, the design (the
  verification version) and the plan's cells;
* `load` -- the session's check of both files before anything runs (`PilotError` otherwise);
* `source_admission` -- Toucan SFT and Pivot Conversational admitted for THIS pilot's measurement only, by exact task
  ids and sha256; never a training admission, never the canonical pool, never a spend;
* `GpuSampler` / `engine_telemetry` -- peak VRAM from nvidia-smi, and the engine's own lines and summary (model load,
  KV cache size and concurrency, preemptions, KV-cache usage and throughput where the engine reports them);
* `report` -- pass rates per mode, split, category and shape (first and second candidates apart), supervised tokens
  per admitted episode, schema_arguments' supply per split, and the telemetry, beside the plan's priors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import tool_episodes as TE  # noqa: E402
from stages.teacher_modes import tool_generation_plan as TGP  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.tool_pilot_plan/v1"
ADMISSION_SCHEMA = "aadistill.stage2.teacher_modes.tool_pilot_source_admission/v1"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_pilot_plan.json"
TASKS = REPO / "artifacts/stages/stage-2/teacher_modes/tool_pilot/pilot_tasks.jsonl"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_pilot_source_admission.json"
REVIEW = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_calling_source_review_v2.json"
LINEAGE = "hybrid"


class PilotError(RuntimeError):
    """Pilot inputs that are not the frozen ones."""


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def task_sha256(task: dict) -> str:
    return hashlib.sha256(canonical(task).encode()).hexdigest()


# --- the frozen cells ----------------------------------------------------------------------------------------------

def build(pool_dir: Path, *, protocol: dict | None = None, spec: dict = TGP.PILOT) -> tuple[dict, list[dict]]:
    """``(pilot plan, tasks)``: the pilot sample of the plan over a verified pool, frozen (module docstring)."""
    from stages.teacher_modes import tool_pool as TP
    proto = protocol or json.loads(TGP.PROTOCOL.read_text())
    plan = TGP.build(Path(pool_dir), protocol=proto)
    man, splits = TP.load_pool(Path(pool_dir))
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    sample = TGP.pilot_sample(plan, by_id, man, spec)
    pplan = {"lineages": {spec["lineage"]: {"modes": {m: {"cells": cs} for m, cs in sample.items()}}}}
    pr = TGP.price(pplan, by_id, man, proto, rounds=spec["rounds"])
    ids = sorted({t for cs in sample.values() for c in cs for t in c["task_ids"]})
    tasks = [by_id[t] for t in ids]
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "lineage": spec["lineage"], "rounds": spec["rounds"],
           "verification_version": TE.VERIFICATION_VERSION,
           "source": {"pool_content_sha256": man["content_sha256"], "pool_schema": man["schema"],
                      "pool_sources": man.get("sources"), "protocol_sha256": TGP.sha_file(TGP.PROTOCOL),
                      "six_category_design_sha256": TGP.sha_file(TGP.SIX),
                      "plan_cells_sha256": hashlib.sha256(canonical(plan["lineages"]).encode()).hexdigest()},
           "sample_rule": {k: (list(map(list, v)) if k == "whole_cells" else v) for k, v in spec.items()},
           "modes": {m: [{k: c[k] for k in ("split", "category", "unit", "target", "supply")} |
                         {"task_ids": c["task_ids"], "shapes": c["shapes"]} for c in cs] for m, cs in sample.items()},
           "tasks": {t["task_id"]: task_sha256(t) for t in tasks},
           "price": pr["sessions"][spec["lineage"]], "authorizes": "nothing"}
    return doc, tasks


def write(doc: dict, tasks: list[dict], plan_out: Path, tasks_out: Path) -> dict:
    """The task file, then the plan naming its sha256 (the plan is the identity the kind binds)."""
    Path(tasks_out).parent.mkdir(parents=True, exist_ok=True)
    Path(tasks_out).write_text("".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks))
    doc = {**doc, "tasks_file": {"path": _rel(tasks_out), "sha256": sha(tasks_out), "tasks": len(tasks)}}
    Path(plan_out).parent.mkdir(parents=True, exist_ok=True)
    Path(plan_out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def load(plan_path: Path, tasks_path: Path, *, protocol_path: Path = TGP.PROTOCOL,
         six_path: Path = TGP.SIX) -> tuple[dict, dict[str, dict]]:
    """``(pilot plan, tasks by id)`` after every check, or `PilotError`: the task file is the one the plan names (its
    sha256), it holds exactly the plan's tasks, each byte-identical to its frozen sha256 and runnable; every cell's
    ids are among them; the protocol and the design (hence the verification version) are the ones it was frozen
    under, and this verifier is that version."""
    doc = json.loads(Path(plan_path).read_text())
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append("not a tool pilot plan")
    if sha(tasks_path) != (doc.get("tasks_file") or {}).get("sha256"):
        bad.append("the task file is not the one the plan froze (sha256)")
    if doc.get("source", {}).get("protocol_sha256") != sha(protocol_path):
        bad.append("frozen under another tool_generation_protocol.json")
    if doc.get("source", {}).get("six_category_design_sha256") != sha(six_path):
        bad.append("frozen under another six_category_design.json (task protocol, verification version)")
    if doc.get("verification_version") != TE.VERIFICATION_VERSION:
        bad.append(f"frozen for {doc.get('verification_version')}, this verifier is {TE.VERIFICATION_VERSION}")
    by_id = {}
    for line in Path(tasks_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            by_id[t["task_id"]] = t
    if set(by_id) != set(doc.get("tasks") or {}):
        bad.append(f"the task file holds {len(set(by_id) ^ set(doc.get('tasks') or {}))} task(s) not as frozen")
    changed = [t for t, x in by_id.items() if task_sha256(x) != (doc.get("tasks") or {}).get(t)]
    if changed:
        bad.append(f"{len(changed)} task(s) differ from their frozen sha256, e.g. {changed[:2]}")
    stray = sorted({t for cs in (doc.get("modes") or {}).values() for c in cs for t in c["task_ids"]} - set(by_id))
    if stray:
        bad.append(f"cells name {len(stray)} task(s) outside the task file")
    unrunnable = {t: TE.task_problems(x) for t, x in by_id.items() if TE.task_problems(x)}
    if unrunnable:
        bad.append(f"{len(unrunnable)} task(s) the protocol cannot run, e.g. {next(iter(unrunnable.items()))}")
    if bad:
        raise PilotError("the pilot refuses: " + "; ".join(bad))
    return doc, by_id


def cells(doc: dict, mode: str, by_id: dict) -> tuple[list[dict], list[list[dict]]]:
    """``(cells, orders)`` for `tool_session.generate_mode`: each frozen cell attempts exactly its task ids, in their
    frozen order, in round 1 (its bound and expected count are their number)."""
    out, orders = [], []
    for c in doc["modes"][mode]:
        n = len(c["task_ids"])
        out.append({"split": c["split"], "category": c["category"], "unit": c["unit"], "target": c["target"],
                    "min_tasks": 0, "bound": n, "tasks": dict.fromkeys(("lower", "expected", "upper"), n)})
        orders.append([by_id[t] for t in c["task_ids"]])
    return out, orders


# --- the resumable identity ------------------------------------------------------------------------------------------

#: the generation implementation a sealed pilot episode depends on: the runner, the sealing driver, the session, the
#: tool environment and verifier, the core call parser
IMPLEMENTATION = ("scripts/stages/stage-0/calibration_volume/run_battery.py",
                  "scripts/stages/stage-2/reasoning_data/generation_driver.py",
                  "scripts/stages/stage-2/teacher_modes/tool_session.py",
                  "scripts/stages/stage-2/teacher_modes/tool_episodes.py",
                  "src/aadistill/data/tool_calls.py")


def pilot_identity(study_identity: str, plan_path: Path = PLAN, *, protocol_path: Path | None = None,
                   root: Path = REPO) -> dict:
    """The launch identity a pilot's episodes are sealed under and resumed by: the study identity AND the frozen pilot
    plan's content, the tool generation protocol, the verification version and the generation implementation
    (`IMPLEMENTATION`, by sha256). `generation_driver.sealed` refuses a shard sealed under any other, so an earlier
    plan's, protocol's or implementation's episodes are never reused -- the same sealing, no second system."""
    parts = {"study_identity_sha256": study_identity, "pilot_plan_sha256": sha(plan_path),
             "protocol_sha256": sha(protocol_path or TGP.PROTOCOL), "verification_version": TE.VERIFICATION_VERSION,
             "implementation_sha256": {f: sha(Path(root) / f) for f in IMPLEMENTATION}}
    return {"sha256": hashlib.sha256(canonical(parts).encode()).hexdigest(), "parts": parts}


# --- the pilot-only source admission -------------------------------------------------------------------------------

#: the exact scope the record grants and the exact token a signature must approve: nothing broader is a pilot admission
SCOPE = ("the tool_pilot kind's measurement only: the hybrid's teacher-native generation on exactly the tasks below "
         "(by id and sha256), its episodes measurement records by default")
APPROVAL = "tool_pilot measurement only"
DOES_NOT = ["admit any source or task to formal training, calibration or state evaluation",
            "build or replace the canonical tool_pool@v1 under data/",
            "decide any category's supply or any SHORTFALL",
            "authorize a GPU, a pod or any spend (the pilot's grant does that, separately)"]
SIGNER = "maintainer"
SIGNATURE_FIELDS = ("signed_by", "approves", "decision", "date", "pilot_plan_sha256", "tasks_sha256", "tasks")


def admission_problems(rec: dict, plan_path: Path, tasks_path: Path, *, today: str | None = None) -> list[str]:
    """Why ``rec`` does not let the pilot run (empty when it does) -- the ONE check issue (`token_matched_phase.
    pilot_problems`) and the pod (`tool_session.run_pilot`) both apply:

    * the record: this schema, exactly `SCOPE` and `DOES_NOT`, bound to the frozen pilot plan (its sha256), its task
      file (sha256, as the plan names it) and task count, and sources whose dataset, revision and task counts are the
      task file's;
    * the signature: signed by the `SIGNER`, approving exactly `APPROVAL`, a non-empty decision quote, an ISO date
      between the record's writing and today, and the plan sha256, task file sha256 and task count it signs equal to
      the record's. A signature with any field missing or different -- ``signed_by`` alone included -- is not one.
    The agent never fills these fields: the maintainer signs."""
    import datetime as dt
    plan_path, tasks_path = Path(plan_path), Path(tasks_path)
    bad = []
    if rec.get("schema") != ADMISSION_SCHEMA:
        return ["not a pilot-only source admission record"]
    if rec.get("scope") != SCOPE or rec.get("does_not") != DOES_NOT:
        bad.append("its scope is not exactly the tool_pilot measurement scope")
    doc = json.loads(plan_path.read_text())
    if (rec.get("pilot_plan") or {}).get("sha256") != sha(plan_path):
        bad.append("it names another pilot plan")
    tsha = sha(tasks_path) if tasks_path.is_file() else None
    if rec.get("tasks_sha256") != tsha or tsha != (doc.get("tasks_file") or {}).get("sha256"):
        bad.append("it names another task file")
    if rec.get("tasks") != len(doc.get("tasks") or {}):
        bad.append("its task count is not the pilot plan's")
    if tsha:
        rows = [json.loads(x) for x in tasks_path.read_text().splitlines() if x.strip()]
        have = Counter((str(t["source"].get("dataset")), str(t["source"].get("revision"))) for t in rows)
        named = {(x.get("dataset"), x.get("revision")): x.get("tasks") for x in rec.get("sources") or []}
        if named != dict(have):
            bad.append("its sources are not the task file's datasets, revisions and counts")
    sig = rec.get("signature") or {}
    if not any(sig.get(k) for k in SIGNATURE_FIELDS):
        return bad + ["PREPARED, not signed by the maintainer"]
    if sig.get("signed_by") != SIGNER:
        bad.append(f"signed_by is {sig.get('signed_by')!r}, not the {SIGNER}")
    if sig.get("approves") != APPROVAL:
        bad.append(f"the signature approves {sig.get('approves')!r}, not exactly {APPROVAL!r}")
    if not isinstance(sig.get("decision"), str) or not sig["decision"].strip():
        bad.append("the signature carries no decision")
    try:
        day = dt.date.fromisoformat(str(sig.get("date")))
        start = dt.date.fromisoformat(str(rec.get("written_utc", ""))[:10])
        now = dt.date.fromisoformat(today) if today else dt.datetime.now(dt.timezone.utc).date()
        if not start <= day <= now:
            bad.append(f"the signature's date {day} is outside {start} .. {now}")
    except ValueError:
        bad.append(f"the signature's date {sig.get('date')!r} is not an ISO date")
    for k, want in (("pilot_plan_sha256", (rec.get("pilot_plan") or {}).get("sha256")),
                    ("tasks_sha256", rec.get("tasks_sha256")), ("tasks", rec.get("tasks"))):
        if sig.get(k) != want:
            bad.append(f"the signature's {k} is not the record's")
    return bad

def source_admission(doc: dict, plan_path: Path = PLAN, tasks_path: Path | None = None) -> dict:
    """The record admitting the two candidate sources for THIS pilot's measurement only (module docstring). It is
    PREPARED, not active: ``signature.signed_by`` stays null until the maintainer signs, and the ``tool_pilot`` kind
    refuses an unsigned record (`token_matched_phase.kind_conditions`)."""
    review = json.loads(REVIEW.read_text())["candidates"]
    lic = {"Agent-Ark/Toucan-1.5M": review["Agent-Ark/Toucan-1.5M"]["licence"],
           "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1":
               review["nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 + Conversational-Tool-Use-Pivot-v1"]["licence"]}
    tasks = [json.loads(x) for x in Path(tasks_path or REPO / doc["tasks_file"]["path"]).read_text().splitlines()
             if x.strip()]
    per = Counter((str(t["source"].get("dataset")), str(t["source"].get("revision"))) for t in tasks)
    shapes = Counter((str(t["source"].get("dataset")), TE.episode_shape(t)) for t in tasks)
    return {"schema": ADMISSION_SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PREPARED: inactive until the maintainer signs it (signature below)",
            "decided_by": "maintainer 2026-11-04 (review of 2468c1aa), item 5: prepare a source admission limited to "
                          "the pilot, from the accepted candidates Toucan SFT and Pivot Conversational; it must not "
                          "widen into a training-data admission and authorizes no paid action",
            "scope": SCOPE,
            "pilot_plan": {"path": _rel(plan_path), "sha256": sha(plan_path)},
            "pool_content_sha256": doc["source"]["pool_content_sha256"],
            "sources": [{"dataset": ds, "revision": rev, "licence": lic.get(ds), "tasks": n,
                         "episode_shapes": {s: k for (d, s), k in shapes.items() if d == ds},
                         "converted_sha256": {k: v.get("converted_sha256")
                                              for k, v in (doc["source"]["pool_sources"] or {}).items()
                                              if any(ds == x.split("@")[0] for x in v.get("datasets", []))}}
                        for (ds, rev), n in sorted(per.items())],
            "tasks_sha256": doc["tasks_file"]["sha256"], "tasks": len(tasks),
            "conditions": ["each source's licence and attribution kept with every derived record (Apache-2.0 notice "
                           "for Toucan; CC-BY-4.0 attribution for Pivot)",
                           "third-party MCP and tool content may carry residual PII: the pilot's generations stay "
                           "outside git (P15); only counts, hashes and measurements enter the repository",
                           "the pool's decontamination (T1-T5), isolation and verification levels unchanged"],
            "does_not": DOES_NOT,
            "signature": dict.fromkeys(SIGNATURE_FIELDS),
            "_signature": f"the maintainer signs by filling every field: signed_by {SIGNER!r}, approves exactly "
                          f"{APPROVAL!r}, the decision quoted, the date (YYYY-MM-DD), and the pilot plan sha256, task "
                          "file sha256 and task count copied from this record (admission_problems checks all of them); "
                          "the agent never fills them",
            "authorizes": "nothing by itself"}


# --- the volume ------------------------------------------------------------------------------------------------------

PILOT_DIR = "tool_pilot_hybrid"
#: the free space the pod must find before generating: about four times what 794 episodes' shards, raw outputs,
#: inputs and logs take (well under 0.5 GB); the teacher snapshot goes to the container disk (setup's HF_HOME)
MIN_FREE_BYTES = 2 * 10 ** 9
#: what the pilot writes inside its own directory, and nothing else may be there
OWN = re.compile(r"^(pilot_[A-Za-z0-9-]+_r1_pass\d+\..+|pilot_[A-Za-z0-9-]+_r1\.jsonl|inputs|telemetry\.jsonl|"
                 r"tool_session\.json)$")


def storage_problems(out: Path, mount: Path, volume_gb: float | None, *, used=None) -> tuple[list[str], dict]:
    """The pod's volume check before the pilot generates: the provider's size less the bytes actually on the mount
    (`h1pa_driver.used_bytes`: a network volume's df reports the host) leaves at least `MIN_FREE_BYTES`, and the
    pilot's directory is new or holds only the pilot's own files -- so nothing of H1-PA, the mixed calibration or
    Research A on the volume is written over. Returns (problems, the measurement)."""
    from stages.reasoning_data.h1pa_driver import used_bytes
    out, mount = Path(out), Path(mount)
    if volume_gb is None:
        return ["the volume's provider size is unknown (--volume-gb)"], {}
    if mount not in out.resolve().parents and mount.resolve() not in out.resolve().parents:
        return [f"{out} is not on the mount {mount}"], {}
    used_b = (used or used_bytes)(mount)
    free = int(float(volume_gb) * 10 ** 9) - used_b
    names = sorted(f.name for f in out.iterdir()) if out.is_dir() else []
    names += sorted(f"inputs/{f.name}" for f in (out / "inputs").iterdir()) if (out / "inputs").is_dir() else []
    foreign = [n for n in names if not OWN.match(n.split("/")[-1])]
    bad = []
    if free < MIN_FREE_BYTES:
        bad.append(f"{free / 1e9:.2f} GB free on the {volume_gb} GB volume, below the {MIN_FREE_BYTES / 1e9:g} GB the "
                   "pilot requires")
    if foreign:
        bad.append(f"{out.name} holds files the pilot did not write, e.g. {foreign[:3]}: nothing is written over")
    return bad, {"mount": str(mount), "volume_gb": volume_gb, "used_bytes": used_b, "free_bytes": free,
                 "required_free_bytes": MIN_FREE_BYTES, "pilot_dir": out.name, "pilot_dir_entries": len(names)}


STORAGE_RECORD = REPO / "logs/stages/stage-2/reasoning_data/storage/h1pa_volume_am5m373kuq.json"


def storage_precheck(resources=None) -> dict:
    """The $0, read-only look at the pilot's volume before any grant: what the provider reports now (the volume, its
    size and datacenter, the pods), what the records say is on it, where the pilot would write, and why the free
    space itself is measured on the pod (`storage_problems`): the provider reports no used bytes, and a network
    volume's df reports the host."""
    if resources is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key
        resources = RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))).account_resources()
    rec = json.loads(STORAGE_RECORD.read_text())
    known = {"h1pa (4 checkpoints, contents_after_pa2)": sum(m["bytes"] for m in rec["contents_after_pa2"]["members"]),
             "mixed_v2 (2 checkpoints, contents_after_mc2)": sum(m["bytes"]
                                                                 for m in rec["contents_after_mc2"]["members_added"])}
    vol = next((v for v in resources.network_volumes if v.get("id") == VOLUME["id"]), None)
    return {"schema": "aadistill.stage2.teacher_modes.pilot_storage_precheck/v1",
            "read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": {"known": resources.known, "error": resources.error, "volume": vol,
                         "other_volumes": [v.get("id") for v in resources.network_volumes if v.get("id") != VOLUME["id"]],
                         "pods": list(getattr(resources, "pods", ()) or [])},
            "matches_the_planned_volume": bool(vol) and int(vol.get("size") or 0) == VOLUME["size_gb"]
            and vol.get("dataCenterId") == VOLUME["data_center"],
            "recorded_contents_bytes": {**known, "teacher_modes (Research A v1: teacher shards, the common "
                                        "initialization, the recovery units)": "not re-listed since 2026-10-30"},
            "recorded_lower_bound_bytes": sum(known.values()),
            "pilot_writes": {"path": f"{VOLUME['mount']}/teacher_modes/{PILOT_DIR}", "estimate_bytes": "< 5e8",
                             "required_free_bytes": MIN_FREE_BYTES,
                             "teacher_snapshot": "container disk (setup's HF_HOME=/root/.cache/huggingface), not the "
                                                 "volume"},
            "never_written": ["<mount>/h1pa", "<mount>/mixed_v2", "<mount>/teacher_modes/<every other entry>"],
            "pod_side_rule": "tool_session.run_pilot refuses before generating unless the provider's size less the "
                             "bytes on the mount (h1pa_driver.used_bytes) leaves MIN_FREE_BYTES and the pilot's directory "
                             "is new or holds only the pilot's own files (tool_pilot.storage_problems)",
            "4rhubprswo": "never attached: the launch refuses any volume but am5m373kuq "
                          "(token_matched_phase.the_pilot_volume_is_the_planned_one)",
            "authorizes": "nothing"}


# --- telemetry -----------------------------------------------------------------------------------------------------

class GpuSampler:
    """nvidia-smi every ``every`` seconds in a thread while a generation runs: peak and last memory.used per GPU,
    and utilization. A missing or failing nvidia-smi records the error, never raises."""

    QUERY = ("nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits")

    def __init__(self, every: float = 5.0, query=None):
        self.every, self.query = every, query or self._smi
        self.peak, self.samples, self.errors = {}, 0, []
        self._stop = threading.Event()
        self._t = None

    def _smi(self) -> str:
        return subprocess.run(self.QUERY, capture_output=True, text=True, timeout=20, check=True).stdout

    def sample(self) -> None:
        try:
            for line in self.query().strip().splitlines():
                idx, name, used, total, util = [x.strip() for x in line.split(",")]
                p = self.peak.setdefault(idx, {"name": name, "memory_total_mib": int(total), "peak_memory_used_mib": 0,
                                               "peak_utilization_pct": 0})
                p["peak_memory_used_mib"] = max(p["peak_memory_used_mib"], int(used))
                p["peak_utilization_pct"] = max(p["peak_utilization_pct"], int(util))
            self.samples += 1
        except Exception as exc:  # noqa: BLE001 -- telemetry never stops a session
            if len(self.errors) < 5:
                self.errors.append(repr(exc)[:200])

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.sample()
                self._stop.wait(self.every)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=30)
        self.sample()
        return False

    def as_dict(self) -> dict:
        return {"every_seconds": self.every, "samples": self.samples, "gpus": self.peak, "errors": self.errors}


#: the engine's own log lines: model load (run_battery's "engine ready"; vLLM's model loading line), KV-cache size and
#: concurrency, and preemption notices -- as the installed vLLM words them; an absent line is recorded as absent
ENGINE_PATTERNS = {
    "engine_ready_seconds": re.compile(r"engine ready in ([0-9.]+)s"),
    "model_loading_gib_seconds": re.compile(r"Model loading took ([0-9.]+) ?GiB and ([0-9.]+) seconds"),
    "kv_cache_tokens": re.compile(r"KV cache size: ([0-9,]+) tokens"),
    "max_concurrency": re.compile(r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x"),
}
PREEMPT = re.compile(r"preempt", re.IGNORECASE)


def engine_telemetry(log: Path, summary: Path | None = None) -> dict:
    """One engine invocation's telemetry from its log and its run_battery summary (`engine_metrics` there when the
    runner was asked for them): every pattern's first match or None, the preemption lines counted."""
    text = Path(log).read_text(errors="ignore") if Path(log).is_file() else ""
    out = {"log": Path(log).name, "log_found": bool(text)}
    for k, pat in ENGINE_PATTERNS.items():
        m = pat.search(text)
        out[k] = [float(g.replace(",", "")) for g in m.groups()] if m else None
    out["preemption_lines"] = sum(1 for line in text.splitlines() if PREEMPT.search(line))
    if summary is not None and Path(summary).is_file():
        s = json.loads(Path(summary).read_text())
        eng = (s.get("identity") or {}).get("engine") or {}
        out.update(init_seconds=eng.get("init_seconds"), output_tokens=s.get("output_tokens"),
                   generation_seconds=s.get("generation_seconds"),
                   output_tokens_per_second=s.get("output_tokens_per_second"),
                   engine_metrics=s.get("engine_metrics"), vllm_version=eng.get("vllm_version"),
                   max_num_seqs=eng.get("max_num_seqs"), max_num_batched_tokens=eng.get("max_num_batched_tokens"))
    return out


def turn_telemetry(records: dict[str, dict]) -> dict:
    """Per-turn timings the runner records (prompt and generated tokens, TTFT, seconds), summarized. The engine
    batches requests, so a request's TTFT includes its queueing; prefill and decode rates per request are reported
    with that caveat, beside the engine-level throughput of `engine_telemetry`."""
    ttft, decode, prompt = [], [], []
    for rec in records.values():
        for t in rec.get("turns") or []:
            n, s, f = int(t.get("generated_tokens") or len(t.get("output_token_ids") or [])), t.get("seconds"), t.get("ttft_seconds")
            if f:
                ttft.append(float(f))
                if s and float(s) > float(f) and n > 1:
                    decode.append((n - 1) / (float(s) - float(f)))
            if t.get("prompt_tokens"):
                prompt.append(int(t["prompt_tokens"]))

    def q(xs, p):
        return round(sorted(xs)[int(p * (len(xs) - 1))], 3) if xs else UNMEASURED
    return {"turns": len(ttft), "ttft_seconds": {"p50": q(ttft, .5), "p90": q(ttft, .9), "max": q(ttft, 1.0)},
            "per_request_decode_tokens_per_second": {"p50": q(decode, .5), "p10": q(decode, .1)},
            "prompt_tokens": {"p50": q(prompt, .5), "p90": q(prompt, .9), "total": sum(prompt) if prompt else UNMEASURED},
            "_": "batched engine: a request's TTFT includes its queueing"}


# --- the grant: envelope and the pilot-only amendment -----------------------------------------------------------------

AMENDMENT = REPO / "logs/budget/applications/research_a_tool_pilot_2026-11-04/grant_amendment_proposal.json"


def envelope_reconciliation() -> dict:
    """The study grant's envelope read-only (`study_engineering.spent` / `envelope`: every session's closeout cost, an
    unknown refusing), per kind, beside the A1 phase's $77.00 and Research A's cumulative cap as the grant itself
    states them -- what the $2.50 pilot would draw on. Nothing here changes a grant."""
    from shared import study_engineering as SE
    from stages.teacher_modes import token_matched_phase as TM
    g = SE.grant(TM.STUDY)
    spent = SE.spent(TM.STUDY)
    env = SE.envelope(TM.STUDY, g)
    by_kind = Counter()
    for r in spent["runs"]:
        a = json.loads((REPO / TM.STUDY.run_dir(r["run_id"]) / "governance/authorization.json").read_text())
        by_kind[(a.get("stage_conditions") or {}).get("kind")] += r["usd"]
    m = re.search(r"cumulative \$([0-9.]+) is not exceeded \(\$([0-9.]+) spent before the phase\)", g.get("_ceiling", ""))
    cap_a, before = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    storage = float(g["phase_a1"]["phase_caps_usd"]["storage_350gb_14d"])
    phase_sessions = round(spent["sessions_usd"] - by_kind.get("supplement", 0.0), 4)
    cum = round(before + phase_sessions + storage, 4) if before is not None else None
    pilot = CAP_OPTIONS["B"][0]
    return {"read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "grant": str(TM.STUDY.grant),
            "grant_sha256": sha(REPO / TM.STUDY.grant),
            "envelope": {"ceiling_usd": env["ceiling_usd"], "spent_usd": env["spent_usd"],
                         "remaining_usd": env["remaining_usd"], "unknown_runs": spent["unknown"],
                         "runs": len(spent["runs"]), "by_kind_usd": {k: round(v, 4) for k, v in sorted(by_kind.items())},
                         "covers_the_pilot": env["remaining_usd"] >= pilot,
                         "remaining_after_the_pilot_cap_usd": round(env["remaining_usd"] - pilot, 4)},
            "a1_phase": {"hard_cap_usd": g["phase_a1"]["hard_cap_usd"], "sessions_usd": phase_sessions,
                         "storage_booked_usd": storage, "new_spend_usd": round(phase_sessions + storage, 4),
                         "headroom_usd": round(g["phase_a1"]["hard_cap_usd"] - phase_sessions - storage, 4),
                         "_": "the pilot is not in the A1 phase's scope: drawing on this grant needs the amendment"},
            "research_a_cumulative": {"cap_usd": cap_a, "before_the_a1_phase_usd": before,
                                      "now_usd": cum, "with_the_pilot_cap_usd": round(cum + pilot, 4) if cum else None,
                                      "headroom_after_usd": round(cap_a - cum - pilot, 4) if cum else None,
                                      "cross_check": {
                                          "cap_less_the_first_ceiling_plus_the_supplement_usd": round(
                                              cap_a - float(g["amendments"][0]["previous_ceiling_usd"])
                                              + by_kind.get("supplement", 0.0), 4) if cap_a else None,
                                          "_": "Research A v1's spend is the cap less the token-matched grant's first "
                                               "ceiling; with the supplement's closeouts it must equal the spend "
                                               "before the phase"},
                                      "_": "the grant's own statement (_ceiling) for the cap and the spend before the "
                                           "phase; the A1 phase's sessions from the closeouts; the 350 GB volume at its "
                                           "booked 14-day reservation (to 2026-11-15) -- storage past that date is not "
                                           "booked here and is a separate decision"},
            "project_cap_usd": g["project_cumulative_cap_usd"], "raises_nothing": True}


def grant_amendment(plan_path: Path = PLAN) -> dict:
    """The exact pilot-only amendment the maintainer would write into the study grant (a PROPOSAL: nothing applies
    it). Its ``tool_pilot`` block is what `token_matched_phase.pilot_grant_problems` requires at issue; ``decision``
    and ``date`` stay null for the maintainer."""
    from shared import study_engineering as SE
    from stages.teacher_modes import token_matched_phase as TM
    g = SE.grant(TM.STUDY)
    doc = json.loads(Path(plan_path).read_text())
    cap, minutes = CAP_OPTIONS["B"]
    block = {"approves": APPROVAL, "granted_by": "maintainer", "decision": None, "date": None,
             "pilot_plan_sha256": sha(plan_path), "tasks_sha256": doc["tasks_file"]["sha256"],
             "source_admission": _rel(ADMISSION), "hard_cap_usd": cap, "driver_minutes": minutes,
             "gpu": TM.GPU, "network_volume": VOLUME["id"], "ceiling_usd_at_amendment": g["ceiling_usd"],
             "draws_on": "this grant's remaining envelope; the ceiling, the per-session cap, the project cap and the "
                         "card list unchanged",
             "kind": "tool_pilot (teacher_generation only; never formal, training or evaluation)",
             "requires": ["the pilot-only source admission signed (tool_pilot.admission_problems)",
                          "issue, launch and pod checks as the application lists them"]}
    return {"schema": "aadistill.budget.grant_amendment_proposal/v1",
            "status": "PROPOSED for the maintainer: NOT applied; the grant is unchanged until the maintainer writes it",
            "target": {"path": str(TM.STUDY.grant), "sha256_now": sha(REPO / TM.STUDY.grant)},
            "edits": {"session_hard_caps_usd.tool_pilot": cap, "tool_pilot": block,
                      "scope (append)": " || AMENDED <date>: the L40S tool pilot -- kind tool_pilot through "
                                        "scripts/stages/stage-2/teacher_modes/token_matched_phase.py (tool_session.py "
                                        f"--pilot) on the frozen pilot plan {block['pilot_plan_sha256'][:12]}... and task "
                                        f"file {block['tasks_sha256'][:12]}..., measurement only, one L40S on "
                                        f"{VOLUME['id']}, hard cap ${cap:.2f}, inside this grant's remaining envelope",
                      "amendments (append)": {"utc": "<date>", "by": "maintainer decision '<quote>'",
                                              "previous_session_hard_caps_usd": dict(g.get("session_hard_caps_usd") or {}),
                                              "previous_scope_sha256": hashlib.sha256(
                                                  str(g.get("scope", "")).encode()).hexdigest(),
                                              "added": "kind tool_pilot and its block"},
                      "authorizes (append)": "; and one tool_pilot artifact, measurement only, capped by its kind"},
            "unchanged": {"ceiling_usd": g["ceiling_usd"], "per_session_hard_cap_usd": g["per_session_hard_cap_usd"],
                          "project_cumulative_cap_usd": g["project_cumulative_cap_usd"], "gpus": g["gpus"],
                          "gpu_count_max": g["gpu_count_max"], "never": g.get("never"),
                          "session_hard_caps_usd (the A1 kinds)": g.get("session_hard_caps_usd"),
                          "phase_a1": "unchanged, with its spend", "amendments": "kept; one entry appended"},
            "envelope": envelope_reconciliation(), "authorizes": "nothing"}


# --- the application -------------------------------------------------------------------------------------------------

APPLICATION = REPO / "logs/budget/applications/research_a_tool_pilot_2026-11-04/application.json"
#: the pilot's network volume: the retained Research A teacher volume (its storage is already committed in the
#: project position); 4rhubprswo, the v2 Top-1's only copy, is never attached
VOLUME = {"id": "am5m373kuq", "size_gb": 55, "data_center": "EUR-IS-2", "mount": "/runpod-volume"}
DISK_GB = 150
#: hard-cap options for the maintainer: (cap, the driver's planned minutes); priced below at the provider rates
CAP_OPTIONS = {"A": (1.80, 27.0), "B": (2.50, 55.0)}
PRICES = (0.86, 1.1067, 1.19)


def runner_cap(minutes: float, price: float, disk_gb: int = DISK_GB, cap: float | None = None) -> dict:
    """The most the pilot session can bill at ``price``: the shared runner's own plan for the tool_pilot launch
    (`study_engineering.session_spec` -> `BudgetSpec.plan`: setup, transfer, the driver's minutes, contingency, the
    closeout-and-teardown reserve, the artifact-recovery reserve) and `study_engineering.all_in` (GPU and container
    disk to the hard deadline plus the watchdog's time to act) -- the numbers the launch's ``all_in_within_cap``
    precheck refuses on."""
    from types import SimpleNamespace

    from shared import study_engineering as SE
    from stages.teacher_modes import token_matched_phase as TM
    sess = SE.session_spec(TM.STUDY, TM.launch_spec(TM.TOOL_PILOT_KIND, minutes),
                           SimpleNamespace(run_id="price_check", dry_run=False))
    plan = sess.budget.plan(price_per_hour=price, authorized_usd=10 ** 6)
    b = SE.all_in(plan, price, disk_gb, cap)
    return {"price_per_hour": price, "driver_minutes": minutes, "expected_minutes": round(plan.expected_minutes, 1),
            "soft_stop_minutes": round(plan.soft_stop_minutes, 1),
            "hard_terminate_minutes": round(plan.hard_terminate_minutes, 1), "all_in_usd": b["all_in_usd"],
            **({"fits_cap": b["all_in_usd"] <= cap, "highest_price_inside_cap": b["highest_price_inside_cap"]}
               if cap is not None else {})}


def application(plan_path: Path = PLAN, admission_path: Path = ADMISSION, *, position: dict | None = None,
                tests: dict | None = None) -> dict:
    """The pilot's application for the maintainer's review: identity, sample, price, the runner's hard-cap check per
    option, the exact commands, gates, stop conditions, recovery, cleanup. AUTHORIZES NOTHING."""
    doc = json.loads(Path(plan_path).read_text())
    adm = json.loads(Path(admission_path).read_text())
    minutes_expected = round(doc["price"]["expected"]["hours"] * 60 - 35, 1)
    minutes_upper = round(doc["price"]["upper"]["hours"] * 60 - 35, 1)
    caps = {k: {"hard_cap_usd": cap, "driver_minutes": m,
                "at_prices": [runner_cap(m, p, cap=cap) for p in PRICES]} for k, (cap, m) in CAP_OPTIONS.items()}
    sample = {m: {f"{c['split']}/{c['category']}": {"tasks": len(c["task_ids"]), "shapes": c["shapes"]}
                  for c in cs} for m, cs in doc["modes"].items()}
    cmd = "PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_matched_phase.py"
    return {
        "schema": "aadistill.budget.application/tool_pilot_v4",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FINAL AUTHORIZATION EVIDENCE (maintainer review of 56381f55: conditional GO for option B); "
                  "AUTHORIZES NOTHING: no grant change, no signature, no issue, no launch, no pod, no spend",
        "recommended_option": {"option": "B", "hard_cap_usd": CAP_OPTIONS["B"][0], "driver_minutes": CAP_OPTIONS["B"][1],
                               "by": "the maintainer's independent review of 29b62c93: one session should complete "
                                     "the 794 runs, the 29 state-evaluation schema_arguments tasks above all",
                               "caveat": "a ceiling, not a promise: the launch recomputes the all-in at the live quote "
                                         "and refuses above the cap, and the workload may still not finish inside it"},
        "decided_by": ["maintainer 2026-11-04 (review of 647eea0e), item 7: prepare the pilot's application",
                       "maintainer 2026-11-04 (review of 2468c1aa), item 5: the 794-task design accepted in principle; "
                       "complete the $0 execution preparation; not a paid authorization",
                       "maintainer 2026-11-04 (review of 29b62c93): option B recommended; partial-report integrity, "
                       "the admission's signature semantics, the storage precheck; still not a paid authorization"],
        "kind": {"study": "A-TM", "kind": "tool_pilot", "claims": ["teacher_generation"],
                 "driver": "scripts/stages/stage-2/teacher_modes/tool_session.py --pilot",
                 "never": ["formal", "training", "evaluation"]},
        "identity": {"pilot_plan": {"path": _rel(plan_path), "sha256": sha(plan_path)},
                     "tasks_file": doc["tasks_file"], "pool_content_sha256": doc["source"]["pool_content_sha256"],
                     "protocol_sha256": doc["source"]["protocol_sha256"],
                     "six_category_design_sha256": doc["source"]["six_category_design_sha256"],
                     "plan_cells_sha256": doc["source"]["plan_cells_sha256"],
                     "verification_version": doc["verification_version"],
                     "source_admission": {"path": _rel(admission_path), "sha256": sha(admission_path),
                                          "signed": bool((adm.get("signature") or {}).get("signed_by"))}},
        "sample": {"rule": doc["sample_rule"], "unique_tasks": len(doc["tasks"]),
                   "mode_specific_runs": sum(len(c["task_ids"]) for cs in doc["modes"].values() for c in cs),
                   "cells": sample},
        "price_by_the_plan_terms": {**doc["price"], "driver_minutes": {"expected": minutes_expected,
                                                                       "upper": minutes_upper},
                                    "_": "the plan's terms (setup 20 + teardown 15 min, throughput ranges, model "
                                         "loads), one round; the driver's minutes are the hours less those 35 min"},
        "hard_cap_check": {
            "rule": "the launch refuses unless the runner's all-in (GPU and container disk from the create to the hard "
                    "deadline plus the watchdog's time to act) fits the artifact's hard cap at the live quote "
                    "(study_engineering.all_in_within_cap); the hard deadline is the runner's plan: (setup 20 + "
                    "transfer 5 + the driver's minutes) x 1.25 contingency + the 12.5-min closeout-and-teardown "
                    "reserve + the 10-min artifact-recovery reserve",
            "storage": f"the network volume {VOLUME['id']} ({VOLUME['size_gb']} GB, {VOLUME['data_center']}) is retained "
                       "and its storage already committed in the project position; no volume is created; container "
                       f"disk ({DISK_GB} GB) is inside the all-in; 4rhubprswo is never attached",
            "options": caps,
            "finding": f"the plan-priced cap of $1.80 (1.1 x the plan's upper $1.64) does not include the runner's "
                       f"contingency and reserves: at the plan's upper driver time ({minutes_upper} min) the runner's "
                       "all-in is above it at every listed price. Option A keeps $1.80 and gives the driver "
                       f"{CAP_OPTIONS['A'][1]:g} min (the plan's expected {minutes_expected} min), stopping at the "
                       "deadline with everything sealed if the workload is slower; option B covers the plan's upper "
                       f"case in one session for ${CAP_OPTIONS['B'][0]:.2f}"},
        "commands": [
            {"step": "the maintainer signs the pilot-only source admission", "who": "maintainer",
             "how": f"every signature field in {_rel(admission_path)}: signed_by 'maintainer', approves exactly "
                    f"{APPROVAL!r}, the decision quoted, the date, and the plan sha256, task file sha256 and task count "
                    "copied from the record; issue and the pod refuse anything less (admission_problems)"},
            {"step": "the maintainer writes the pilot-only grant amendment", "who": "maintainer",
             "how": f"{_rel(AMENDMENT)} into logs/budget/approvals/teacher_modes_token_matched_grant.json: "
                    f"session_hard_caps_usd.tool_pilot = {CAP_OPTIONS['B'][0]:.2f} and the tool_pilot block (decision, "
                    "date filled) with the scope, amendments and authorizes appends; issue refuses a cap without the "
                    "block (pilot_grant_problems)"},
            {"step": "commit; stage the session bundle of that commit on the relay",
             "how": "stages.phase_c1.bundle.stage_bundle (the procedure of every Research A session); the launch's "
                    "bundle_on_relay gate refuses without it"},
            {"step": "re-read the project position (read-only)",
             "how": "PYTHONPATH=src:scripts .venv/bin/python scripts/shared/project_position.py --live "
                    "--request tool_pilot=<cap>"},
            {"step": "issue the one-use artifact",
             "how": f"{cmd} issue --kind tool_pilot --run-id <run> --expected-usd 1.14 --hard-cap-usd "
                    f"{CAP_OPTIONS['B'][0]:.2f}"},
            {"step": "dry-run every pre-provider gate at the live quote (no create)",
             "how": f"{cmd} issue --kind tool_pilot --run-id <run>_dry --expected-usd 1.14 --hard-cap-usd "
                    f"{CAP_OPTIONS['B'][0]:.2f} --dry-run --out <scratch>/auth.json && {cmd} launch --kind tool_pilot "
                    f"--minutes {CAP_OPTIONS['B'][1]:g} "
                    "--run-id <run>_dry --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live> "
                    f"--gpu 'NVIDIA L40S' --disk-gb {DISK_GB} --network-volume-id {VOLUME['id']} --volume-mount-path "
                    f"{VOLUME['mount']} --data-center-ids {VOLUME['data_center']} --volume-gb {VOLUME['size_gb']} "
                    "--dry-run --authorization <scratch>/auth.json"},
            {"step": "launch (only after the grant and the signature)",
             "how": f"{cmd} launch --kind tool_pilot --minutes {CAP_OPTIONS['B'][1]:g} --run-id <run> --scr <scratch> "
                    "--session-commit <sha> --bundle <name> --max-price <live> --gpu 'NVIDIA L40S' "
                    f"--disk-gb {DISK_GB} --network-volume-id {VOLUME['id']} --volume-mount-path {VOLUME['mount']} "
                    f"--data-center-ids {VOLUME['data_center']} --volume-gb {VOLUME['size_gb']}"},
            {"step": "closeout (the launcher runs it on every terminal path) and the dev box's report",
             "how": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot report --session "
                    "<scratch>/store/tool_pilot_hybrid [--session <a resumed run's copy>] [--study-identity <the "
                    "run's authorization plan_hash, when tool_session.json is missing>] --write <report>"}],
        "gates": {"at_issue": ["the grant names a tool_pilot cap and the cap fits it (kind_cap_problems)",
                               "the grant carries the pilot-only tool_pilot block: approves exactly 'tool_pilot "
                               "measurement only', granted by the maintainer with a decision and a date, the plan and "
                               "task file sha256, the source admission, the cap, the L40S, am5m373kuq and the unchanged "
                               "ceiling (pilot_grant_problems)",
                               "the cap fits the study envelope, the project cap as the provider bills it now and the "
                               "account balance (funding_problems on a live, fail-closed position)",
                               "the L40S check record admits teacher generation (kind_conditions)",
                               "the pilot plan and task file load (tool_pilot.load) and the source admission passes "
                               "tool_pilot.admission_problems: signed by the maintainer, approving exactly 'tool_pilot "
                               "measurement only', a decision and a valid date, the exact scope, and the plan sha256, "
                               "task file sha256, task count and sources it signs equal to the record and the files",
                               "the protocol, design, pilot plan and admission bound by sha256 (KIND_RECORDS)"],
                  "at_launch": ["the artifact is this study's tool_pilot artifact, unused, not a dry run",
                                "one NVIDIA L40S only (the_pilot_runs_on_an_l40s; the grant's price ceiling)",
                                "the all-in fits the hard cap at the live quote (all_in_within_cap)",
                                "the study identity and the bound files are the session commit's",
                                "the project cap holds now: committed spend, remaining cap, balance, pods and volumes "
                                "re-read from the provider (the_project_cap_holds_now; an unreadable provider, a pod in "
                                "the account or an unowned volume refuses)",
                                "the network volume is the planned one as the provider reports it "
                                "(the_volume_is_the_planned_one) and it is am5m373kuq at 55 GB in EUR-IS-2, never "
                                "4rhubprswo (the_pilot_volume_is_the_planned_one)",
                                "the kind's bound records are the committed ones",
                                "the session bundle is on the relay"],
                  "on_the_pod": ["setup re-checks the artifact against the pod's files (require_on_pod)",
                                 "the session re-checks the pilot plan, the task file and the admission with the same "
                                 "admission_problems (tool_session.run_pilot)",
                                 "before generating: at least 2 GB free on the volume (its size less the bytes on the "
                                 "mount) and the pilot's directory new or only the pilot's own files "
                                 "(tool_pilot.storage_problems)"]},
        "storage_precheck": "logs/stages/stage-2/teacher_modes/validations/packed_round/pilot_storage_precheck_20261104.json",
        "grant": {"amendment_proposal": _rel(AMENDMENT),
                  "amendment_proposal_sha256": sha(AMENDMENT) if AMENDMENT.is_file() else None,
                  "envelope": envelope_reconciliation(),
                  "issue_requires": "the grant's tool_pilot block (token_matched_phase.pilot_grant_problems) beside "
                                    "session_hard_caps_usd.tool_pilot: a kind cap alone refuses"},
        "resumption_identity": {"rule": "episodes are sealed and resumed under the pilot identity: the study identity, "
                                        "the pilot plan's sha256, the protocol's sha256, the verification version and "
                                        "the generation implementation's sha256 (tool_pilot.pilot_identity); before "
                                        "resuming, every sealed episode's group input, protocol and runner sha256 must be "
                                        "this round's (tool_session.resumable); anything else refuses",
                                "implementation_sha256": {f: sha(REPO / f) for f in IMPLEMENTATION}},
        "prefill": "UNMEASURED by design: the batched engine interleaves prefill and decode; the formal generation's "
                   "re-pricing must not treat any prefill rate as measured (the TTFT distribution and prompt tokens are "
                   "reported instead)",
        "report_states": {"COMPLETE": "every required attempt sealed and the session ended ALL_DONE",
                          "PARTIAL": "some sealed, some required attempt not (deadline, interruption, failure); pass "
                                     "rates over the sealed attempts only",
                          "INCOMPLETE": "every required attempt sealed, the session's terminal evidence missing or "
                                        "not ALL_DONE",
                          "FAILED": "nothing sealed",
                          "_": "sealed shards are read whether or not their group completed, each verified by sha256 "
                               "and the launch identity; an unattempted task is never a failure; candidates 0 and 1 "
                               "are counted apart and a task once; copies of a resumed run merge, and a task sealed "
                               "differently in two copies refuses; an unreadable measurement is UNMEASURED, never 0"},
        "stop_conditions": ["the sample complete: every task's first candidate, then second candidates of the refused "
                            "tasks only",
                            "the runner's soft stop and hard deadline from the accepted price (watchdog)",
                            "a generation failure: the group is incomplete, the session ends RUN_FAILED with every "
                            "finished episode sealed",
                            "a deterministic failure repeating the previous paid attempt's signature: no new resource "
                            "until diagnosed (session_prechecks.same_failure_gate)",
                            "provider or billing state unknown: nothing created; reconcile and tear down",
                            "an L40S out-of-memory: the run fails; an A100 needs the measured OOM and a separate "
                            "authorization"],
        "recovery": "identity-bound sealed shards on the volume (generation_driver.run_group, sealed every 50 records "
                    "or 300 s); a replacement session -- a new run with a new one-use artifact -- regenerates nothing "
                    "sealed and runs only each label's unsealed tasks; candidate k uses seed_base + k",
        "cleanup_and_reconciliation": "the launcher's terminal path deletes the pod and confirms it gone with the "
                                      "provider, secures tool_pilot_hybrid (shards, tool_session.json, "
                                      "telemetry.jsonl) and closes out; a session that created a resource and records "
                                      "no closeout cost is an unknown, and every later issue refuses until it is "
                                      "reconciled (study_engineering.envelope)",
        "telemetry": ["per engine invocation (telemetry.jsonl): wall time, nvidia-smi peak memory.used and utilization, "
                      "the engine's model-load and KV-cache lines, preemption lines, run_battery's summary (engine "
                      "init seconds, output tokens/s) and its engine metrics (--engine-metrics: preemptions, KV-cache "
                      "usage peaks, token counters, where the installed vLLM reports them)",
                      "per turn (sealed records): prompt and generated tokens, TTFT, seconds"],
        "outputs": "measurement records only by default (sealed episodes, census, telemetry, report); reuse in formal "
                   "data is a separate maintainer decision",
        "position": position, "tests": tests,
        "needed_from_the_maintainer": ["sign the pilot-only source admission (every signature field)",
                                       "write the pilot-only grant amendment (grant_amendment_proposal.json: the "
                                       "tool_pilot block with its decision and date, session_hard_caps_usd.tool_pilot "
                                       "$2.50, the scope, amendments and authorizes appends; nothing else changes)",
                                       "the final authorization"],
        "not_verifiable_at_zero_cost": ["the installed vLLM's metric names and values (KV-cache usage, preemptions) "
                                        "and its log lines: read on the first run, UNMEASURED where absent",
                                        "nvidia-smi on the pod (peak VRAM): UNMEASURED if it gives no sample",
                                        "throughput, TTFT, model-load time and the real pass rates",
                                        "the live L40S quote and stock in EUR-IS-2 at launch",
                                        "the bytes actually on am5m373kuq (the provider reports none; the pod measures "
                                        "them before generating)",
                                        "the shared runner's create, watchdog, deadline, teardown and the provider's "
                                        "confirmation on a real pod (exercised by earlier paid sessions, not here)",
                                        "the bundle staged on the relay for the session commit"],
        "not_requested": ["the bounded tool generation", "the packed full search", "any training", "formal evaluation",
                          "a Thinking-2507 session", "a raised project cap", "any SHORTFALL approval",
                          "anything on 4rhubprswo", "a merge"],
        "authorizes": "nothing"}


# --- the report ------------------------------------------------------------------------------------------------------

#: a report's state: every required attempt sealed and the session ended ALL_DONE; some sealed, some required attempt
#: still unsealed (a deadline, an interruption, a failed generation); every required attempt sealed but the session's
#: terminal evidence missing or not ALL_DONE; nothing sealed at all
STATES = ("COMPLETE", "PARTIAL", "INCOMPLETE", "FAILED")
UNMEASURED = "UNMEASURED"
LABEL = re.compile(r"^(pilot_(?P<mode>[A-Za-z0-9-]+)_r1_pass(?P<k>\d+))\.(?:part\d+\.done|complete|seg\d+\.input\.jsonl)$")


def _labels(dirs: list[Path]) -> dict[tuple[str, int], str]:
    """``(mode, candidate) -> label`` of every generation group any session left a trace of: a sealed shard's marker, a
    completion record or a segment's input -- a group need not be complete to be read."""
    out = {}
    for d in dirs:
        for f in Path(d).iterdir():
            m = LABEL.match(f.name)
            if m:
                out[(m.group("mode"), int(m.group("k")))] = m.group(1)
    return out


def _sealed(dirs: list[Path], label: str, identity: str) -> dict[str, dict]:
    """Every sealed record of ``label`` across ``dirs``, each shard verified by its sha256 marker and the launch
    identity (`generation_driver.sealed`); a task sealed differently in two places refuses: a resumed session never
    changes what was sealed."""
    from stages.reasoning_data import generation_driver as GD
    out: dict[str, dict] = {}
    for d in dirs:
        for tid, rec in GD.sealed(Path(d), label, identity).items():
            if tid in out and canonical(out[tid]) != canonical(rec):
                raise PilotError(f"{label}: {tid} is sealed differently in two session copies")
            out[tid] = rec
    return out


def _attempted(dirs: list[Path], label: str) -> set[str]:
    """The tasks a session submitted to the engine in ``label`` (its segments' inputs), sealed or not."""
    got = set()
    for d in dirs:
        for f in Path(d).glob(f"{label}.seg*.input.jsonl"):
            got |= {json.loads(x)["task_id"] for x in f.read_text().splitlines() if x.strip()}
    return got


def _outcome(task: dict, rec: dict | None, attempted: set, *, mode: str, thinking: bool, stop_ids) -> tuple[str, float]:
    """One candidate's outcome: admitted or rejected (sealed and verified), attempted_unsealed, or unattempted -- an
    unattempted or unsealed task is never a verification failure."""
    if rec is None:
        return ("attempted_unsealed" if task["task_id"] in attempted else "unattempted"), 0.0
    turns = rec.get("turns") or []
    if not TE.verify(task, turns, thinking=thinking).admitted:
        return "rejected", 0.0
    if task["split"] in ("calibration", "state_eval"):
        its = TE.items(task, turns, mode=mode, thinking=thinking)
        return ("admitted", float(sum(len(i["tags"]["assistant"]) for i in its))) if its else ("rejected", 0.0)
    return "admitted", float(sum(len(d.output_ids) for d in TE.documents(task, turns, mode=mode, stop_ids=stop_ids)))


def _measured(v, why: str = "not reported by the engine or the GPU tools on this run"):
    return {"value": v} if v not in (None, {}, []) else {"value": UNMEASURED, "why": why}


def report(session_dirs, *, study_identity: str | None = None, plan_path: Path = PLAN,
           tasks_path: Path = TASKS) -> dict:
    """The pilot's measurements from one or more secured copies of its session directory, complete or not:

    * every sealed record is read -- a group need not have a ``.complete`` -- after `generation_driver.sealed` verifies
      each shard's sha256 and the launch identity; copies of a resumed session merge, and a task sealed differently in
      two copies refuses;
    * per task and candidate: admitted or rejected (sealed, under the frozen verifier), attempted but unsealed, or
      unattempted; candidate 1 is required only where candidate 0 was sealed and rejected; a task counts once;
    * pass rates divide by the SEALED attempts (first candidate) and by the DETERMINED tasks (both candidates), never
      by tasks that were not attempted;
    * the state (`STATES`), and the session evidence present or missing: without ``tool_session.json`` (a forced end)
      the caller gives the study identity (the run's authorization ``plan_hash``), the pilot identity is derived from
      it and the frozen plan (`pilot_identity`), and the report says so;
    * telemetry, every quantity a run could not read marked UNMEASURED, never 0."""
    doc, by_id = load(plan_path, tasks_path)
    dirs = [Path(d) for d in ([session_dirs] if isinstance(session_dirs, (str, Path)) else session_dirs)]
    sessions = {str(d): json.loads((d / "tool_session.json").read_text()) if (d / "tool_session.json").is_file() else None
                for d in dirs}
    idents = {r["identity"] for r in sessions.values() if r and r.get("identity")}
    if study_identity:
        idents.add(pilot_identity(study_identity, plan_path)["sha256"])
    if len(idents) != 1:
        raise PilotError(f"the launch identity is {'unknown' if not idents else 'ambiguous'}: pass the run's "
                         "authorization plan_hash (--study-identity) when tool_session.json is missing")
    ident = idents.pop()
    proto = json.loads(TGP.PROTOCOL.read_text())
    stop_ids = [int(x) for x in proto["execution"]["stop_ids"]]
    labels = _labels(dirs)
    stray = sorted(m for m, _ in labels if m not in doc["modes"])
    if stray:
        raise PilotError(f"generation groups of modes the pilot does not run: {stray}")
    modes, all_sealed, all_required_sealed, every = {}, 0, True, {}
    for mode in doc["modes"]:
        thinking = bool(proto["lineages"][doc["lineage"]]["modes"][mode]["template_kwargs"].get("enable_thinking", True))
        recs = {k: _sealed(dirs, labels[(mode, k)], ident) if (mode, k) in labels else {} for k in (0, 1)}
        att = {k: _attempted(dirs, labels[(mode, k)]) if (mode, k) in labels else set() for k in (0, 1)}
        frozen = {t for c in doc["modes"][mode] for t in c["task_ids"]}
        outside = sorted((set(recs[0]) | set(recs[1])) - frozen)
        if outside:
            raise PilotError(f"{mode}: sealed records of tasks outside the frozen cells, e.g. {outside[:2]}")
        cells = {}
        for c in doc["modes"][mode]:
            for t in c["task_ids"]:
                task = by_id[t]
                key = f"{c['split']}|{c['category']}|{TE.episode_shape(task)}"
                row = cells.setdefault(key, {"offered": 0, "c0": Counter(), "c1": Counter(), "final": Counter(),
                                             "units": 0.0})
                row["offered"] += 1
                o0, u0 = _outcome(task, recs[0].get(t), att[0], mode=mode, thinking=thinking, stop_ids=stop_ids)
                row["c0"][o0] += 1
                o1, u1 = "not_required", 0.0
                if o0 == "rejected":
                    o1, u1 = _outcome(task, recs[1].get(t), att[1], mode=mode, thinking=thinking, stop_ids=stop_ids)
                    row["c1"]["required"] += 1
                    row["c1"][o1] += 1
                elif t in recs[1]:
                    raise PilotError(f"{mode}: {t} has a second candidate its first did not call for")
                final = ("admitted" if "admitted" in (o0, o1) else "rejected" if (o0, o1) == ("rejected", "rejected")
                         else "undetermined")
                row["final"][final] += 1
                row["units"] += u0 + u1
                all_required_sealed &= o0 in ("admitted", "rejected") and o1 in ("admitted", "rejected", "not_required")
        for row in cells.values():
            sealed0 = row["c0"]["admitted"] + row["c0"]["rejected"]
            determined = row["final"]["admitted"] + row["final"]["rejected"]
            row.update(c0=dict(row["c0"]), c1=dict(row["c1"]), final=dict(row["final"]),
                       first_candidate_pass_rate=round(row["c0"]["admitted"] / sealed0, 4) if sealed0 else UNMEASURED,
                       pass_rate_two_candidates=(round(row["final"]["admitted"] / determined, 4) if determined
                                                 else UNMEASURED),
                       units_per_admitted_episode=(round(row["units"] / row["final"]["admitted"], 1)
                                                   if row["final"]["admitted"] else UNMEASURED))
        n = len(recs[0]) + len(recs[1])
        all_sealed += n
        every.update({f"{t}#{k}": r for k, rs in recs.items() for t, r in rs.items()})
        modes[mode] = {"cells": dict(sorted(cells.items())), "sealed_records": n,
                       "attempted": {f"candidate_{k}": len(att[k]) for k in (0, 1)},
                       "groups_complete": {f"candidate_{k}": any((Path(d) / f"{labels[(mode, k)]}.complete").is_file()
                                                                 for d in dirs) if (mode, k) in labels else False
                                           for k in (0, 1)}}
    terminals = [r.get("terminal") for r in sessions.values() if r]
    state = ("FAILED" if not all_sealed else
             ("COMPLETE" if "ALL_DONE" in terminals else "INCOMPLETE") if all_required_sealed else "PARTIAL")
    tel = [json.loads(x) for d in dirs if (d / "telemetry.jsonl").is_file()
           for x in (d / "telemetry.jsonl").read_text().splitlines() if x.strip()]
    gpu = [g for t in tel for g in ((t.get("gpu") or {}).get("gpus") or {}).values()]
    eng = [e for t in tel for e in t.get("engine") or []]
    turns = turn_telemetry(every)
    return {"schema": "aadistill.stage2.teacher_modes.tool_pilot_report/v2", "state": state,
            "pilot_plan_sha256": sha(plan_path), "identity": ident,
            "session_evidence": {d: ({k: r.get(k) for k in ("kind", "authorization", "terminal", "started_utc",
                                                           "finished_utc")} if r else
                                     "MISSING: no tool_session.json (a forced end); the report rests on the sealed "
                                     "shards alone") for d, r in sessions.items()},
            "modes": modes,
            "telemetry": {
                "invocations": len(tel),
                "peak_vram_mib": _measured(max((g["peak_memory_used_mib"] for g in gpu), default=None),
                                           "nvidia-smi gave no sample"),
                "model_load_seconds": _measured([e.get("init_seconds") or (e.get("engine_ready_seconds") or [None])[0]
                                                 for e in eng if e.get("init_seconds") or e.get("engine_ready_seconds")]),
                "kv_cache_tokens": _measured([e["kv_cache_tokens"][0] for e in eng if e.get("kv_cache_tokens")]),
                "kv_cache_usage_peak": _measured([v for e in eng for k, v in ((e.get("engine_metrics") or {})
                                                                               .get("max") or {}).items()
                                                  if "cache_usage" in k]),
                "preemptions": _measured([v for e in eng for k, v in ((e.get("engine_metrics") or {}).get("max") or {})
                                          .items() if "preempt" in k] or None,
                                         "the engine reported no preemption metric (preemption log lines are "
                                         "counted separately)"),
                "preemption_log_lines": sum(e.get("preemption_lines") or 0 for e in eng) if eng else UNMEASURED,
                "decode_output_tokens_per_second": _measured([e["output_tokens_per_second"] for e in eng
                                                              if e.get("output_tokens_per_second")]),
                "ttft_seconds": turns["ttft_seconds"] if turns["turns"] else {"value": UNMEASURED},
                "per_request_decode_tokens_per_second": turns["per_request_decode_tokens_per_second"],
                "prompt_tokens": turns["prompt_tokens"],
                "prefill_tokens_per_second": {"value": UNMEASURED, "why": "the batched engine interleaves prefill "
                                              "and decode; the TTFT distribution and the prompt tokens are reported"}},
            "priors": proto["bounded_plan"]["priors"], "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pool", required=True)
    b.add_argument("--plan-out", default=str(PLAN))
    b.add_argument("--tasks-out", default=str(TASKS))
    s = sub.add_parser("admission")
    s.add_argument("--plan", default=str(PLAN))
    s.add_argument("--write", default=str(ADMISSION))
    ap_ = sub.add_parser("application")
    ap_.add_argument("--write", default=str(APPLICATION))
    ap_.add_argument("--live-position", action="store_true", help="re-read the project position (read-only)")
    ap_.add_argument("--tests", default=None, help="a JSON file of the test evidence to record")
    ga = sub.add_parser("grant-amendment")
    ga.add_argument("--write", default=str(AMENDMENT))
    sp = sub.add_parser("storage-precheck")
    sp.add_argument("--write", required=True)
    r = sub.add_parser("report")
    r.add_argument("--session", action="append", required=True, help="a secured session directory (repeatable)")
    r.add_argument("--study-identity", default=None,
                   help="the run's authorization plan_hash, when tool_session.json is missing")
    r.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        doc, tasks = build(Path(a.pool))
        doc = write(doc, tasks, Path(a.plan_out), Path(a.tasks_out))
        print(json.dumps({"tasks": doc["tasks_file"]["tasks"], "price": doc["price"]}, indent=1))
    elif a.cmd == "admission":
        doc = json.loads(Path(a.plan).read_text())
        rec = source_admission(doc, Path(a.plan))
        Path(a.write).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: rec[k] for k in ("status", "tasks", "tasks_sha256")}, indent=1))
    elif a.cmd == "grant-amendment":
        doc = grant_amendment()
        Path(a.write).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(doc["envelope"], indent=1))
    elif a.cmd == "storage-precheck":
        doc = storage_precheck()
        Path(a.write).write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({k: doc[k] for k in ("provider", "matches_the_planned_volume", "recorded_lower_bound_bytes")},
                         indent=1, default=str))
    elif a.cmd == "application":
        pos = None
        if a.live_position:
            from shared import project_position as PPOS
            from shared import study_engineering as SE
            p = PPOS.position(REPO, snapshot=SE.provider_snapshot())
            pos = {k: p.get(k) for k in ("fetched_utc", "committed_usd", "project_cap_usd", "project_remaining_usd",
                                         "balance_usd", "account_pods", "account_volumes")}
        tests = json.loads(Path(a.tests).read_text()) if a.tests else None
        doc = application(position=pos, tests=tests)
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(doc["hard_cap_check"]["options"], indent=1)[:3000])
    else:
        Path(a.write).write_text(json.dumps(report([Path(x) for x in a.session], study_identity=a.study_identity),
                                            indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

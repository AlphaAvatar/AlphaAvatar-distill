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
* `report` -- the teacher data admission measurements (`METRICS`: first-candidate admission rate, admission yield of
  up to two candidates, each verifier level's pass rate, verification coverage; never a benchmark score) per mode,
  split, category, shape and source, supervised tokens and scored positions, schema_arguments' supply per split, the
  formal cells' projection, and the telemetry, beside the plan's priors.
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


def frozen_protocol(plan_path: Path = PLAN, root: Path = REPO) -> Path:
    """The tool generation protocol AS THE PILOT PLAN FROZE IT: today's file when its sha256 is the plan's, else the
    committed version of that file whose sha256 is (``git log`` of the file, newest first), written beside the
    gitignored pilot artifacts. The protocol has moved on since (maintainer 2026-11-05: the common-support composition
    and the last round's rule); the closed pilot's plan is still verified against what it was frozen under."""
    import subprocess
    want = (json.loads(Path(plan_path).read_text()).get("source") or {}).get("protocol_sha256")
    rel = TGP.PROTOCOL.relative_to(REPO).as_posix()
    if sha(root / rel) == want:
        return root / rel
    log = subprocess.run(["git", "log", "--format=%H", "--", rel], cwd=root, capture_output=True, text=True).stdout
    for commit in log.split():
        b = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=root, capture_output=True).stdout
        if hashlib.sha256(b).hexdigest() == want:
            out = root / "artifacts/stages/stage-2/teacher_modes/tool_pilot" / f"tool_generation_protocol.{want[:12]}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b)
            return out
    raise PilotError(f"no committed {rel} has the sha256 {str(want)[:12]}... the pilot plan was frozen under")


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


# --- the L40S generation-scoped gate -----------------------------------------------------------------------------------

#: the pilot's own modules on its generation path, read as they are (their imports include the authorization's
#: governance, which feeds the engine nothing: the engine command is built from the protocol, the plan and the tasks)
SESSION_MODULES = ("scripts/stages/stage-2/teacher_modes/tool_session.py",
                   "scripts/stages/stage-2/teacher_modes/tool_pilot.py",
                   "scripts/stages/stage-2/teacher_modes/tool_generation_plan.py")


def generation_engine(root: Path = REPO, lineage: str = LINEAGE, kind: str | None = None) -> dict:
    """The engine a tool generation of ``lineage`` runs (`l40s_check.generation_engine`): the lineage of the tool
    generation protocol -- its teacher, per-mode template switch and sampling -- through `tool_session.mode_args`'
    memory fraction and `generation_driver.run_group`'s session limit, in this tree's runner and vLLM pins. A formal
    generation ``kind`` reads its own protocol (`token_matched_phase.KIND_RECORDS`)."""
    from stages.teacher_modes import l40s_check as LC
    from stages.teacher_modes import token_matched_phase as TM
    root = Path(root)
    rel = TM.KIND_RECORDS[kind][0] if kind in TM.GENERATION_LINEAGE else TGP.PROTOCOL.relative_to(REPO)
    proto = json.loads((root / rel).read_text())
    li, ex = proto["lineages"][lineage], proto["execution"]
    return LC.generation_engine(model=li["model"], revision=li["revision"], modes=li["modes"],
                                session_limit=ex["session_limit"], gpu_memory_utilization=ex["gpu_memory_utilization"],
                                stop_ids=ex["stop_ids"], runner_source=(root / LC.RUNNER).read_text(),
                                teacher_identity=json.loads((root / LC.TEACHER_IDENTITY).read_text()),
                                vllm_pins=(root / LC.VLLM_PINS).read_bytes())


def generation_inputs(root: Path = REPO, kind: str = "tool_pilot", lineage: str = LINEAGE) -> tuple[str, ...]:
    """The files a tool generation of ``kind`` reads: the kind's bound records, its task file (the pilot plan's, or
    the bounded plan's for the lineage), the teacher identity (its tokenizer and template), the runner and the vLLM
    pins."""
    from stages.teacher_modes import l40s_check as LC
    from stages.teacher_modes import token_matched_phase as TM
    recs = TM.KIND_RECORDS[kind]
    plan = Path(root) / (recs[2] if kind == TM.TOOL_PILOT_KIND else recs[1])
    doc = json.loads(plan.read_text()) if plan.is_file() else {}
    tasks = ((doc.get("tasks_file") or {}).get("path") if kind == TM.TOOL_PILOT_KIND else
             (((doc.get("lineages") or {}).get(lineage) or {}).get("tasks_file") or {}).get("path"))
    return tuple(recs) + ((tasks,) if tasks else ()) + (LC.TEACHER_IDENTITY, LC.RUNNER, LC.VLLM_PINS)


def generation_path(root: Path = REPO) -> tuple[str, ...]:
    """The modules of the pilot's generation path: the runner, the generation driver, the environment factory the
    protocol names, the verifier and the core parser (`IMPLEMENTATION` but the session driver) with their import
    closures (`aadistill.governance.closure.walk`), and the pilot's own `SESSION_MODULES`."""
    import importlib.util
    from aadistill.governance.closure import walk
    root = Path(root)
    proto = json.loads((root / TGP.PROTOCOL.relative_to(REPO)).read_text())
    env = importlib.util.find_spec(proto["execution"]["env_factory"].split(":")[0])
    entries = tuple(f for f in IMPLEMENTATION if f not in SESSION_MODULES) + (
        Path(env.origin).resolve().relative_to(REPO).as_posix(),)
    files, _ = walk(root, tuple(sorted(set(entries))), roots=("src", "scripts"))
    return tuple(sorted(set(files) | set(SESSION_MODULES)))


def l40s_generation_problems(root: Path = REPO, *, kind: str = "tool_pilot", lineage: str = LINEAGE) -> list[str]:
    """Why the committed L40S check does not admit a tool generation (`l40s_check.generation_scope_problems`): a
    generation-only kind is admitted on the check's generation measurement exactly when it runs the generation the
    check measured -- the same engine, teacher, template and runner (Thinking-2507 is another teacher: refused). It
    is a compatibility check: whether the kind may run is the grant's."""
    from stages.teacher_modes import l40s_check as LC
    return LC.generation_scope_problems(generation_engine(root, lineage, kind), generation_inputs(root, kind, lineage),
                                        generation_path(root), root)


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


# --- storage and placement ---------------------------------------------------------------------------------------

PILOT_DIR = "tool_pilot_hybrid"
#: the free space the pod must find before generating: about four times what 794 episodes' shards, raw outputs,
#: inputs and logs take (well under 0.5 GB); the teacher snapshot goes to the container disk (setup's HF_HOME)
MIN_FREE_BYTES = 2 * 10 ** 9
#: what the pilot writes inside its own directory, and nothing else may be there
OWN = re.compile(r"^(pilot_[A-Za-z0-9-]+_r1_pass\d+\..+|pilot_[A-Za-z0-9-]+_r1\.jsonl|inputs|telemetry\.jsonl|"
                 r"tool_session\.json|receipts\.json)$")
#: Where the pilot runs (maintainer 2026-11-04: every Secure Cloud L40S that meets the runtime, price, resource and
#: persistence requirements, no longer EUR-IS-2 alone). One L40S and NO network volume: the pilot reads nothing a
#: volume holds (its tasks are a local asset, the teacher comes from the hub at its pinned revision), so a volume would
#: only lock the draw to its datacenter. The session writes to the pod's container disk under ``work_root``; every
#: sealed shard is pulled to the dev box on each poll (`token_matched_phase.secure`) and a replacement pod receives
#: the dev box's copies (`tool_session --resume-from`). am5m373kuq stays retained and untouched; 4rhubprswo, the v2
#: Top-1's only copy, is never attached.
PLACEMENT = {"gpu": "NVIDIA L40S", "gpu_count": 1, "cloud": "SECURE", "network_volume": None,
             "data_centers": "any Secure Cloud datacenter offering the L40S at or below the grant's price",
             "work_root": "/workspace",
             "storage": "the pod's container disk; sealed shards pulled to the dev box on every poll and staged back "
                        "to a replacement pod (--resume-from); no network volume"}


def storage_problems(out: Path, mount: Path, volume_gb: float | None, *, used=None, free=None) -> tuple[list[str], dict]:
    """The pod's storage check before the pilot generates: at least `MIN_FREE_BYTES` free where it writes, and the
    pilot's directory new or holding only the pilot's own files -- nothing is written over. On a network volume
    (``volume_gb``, the provider's size) the free space is that size less the bytes on the mount
    (`h1pa_driver.used_bytes`: a network volume's df reports the host); on the container disk (no volume, the pilot's
    placement) it is the filesystem's own free space. Returns (problems, the measurement)."""
    import shutil
    out, mount = Path(out), Path(mount)
    if mount not in out.resolve().parents and mount.resolve() not in out.resolve().parents:
        return [f"{out} is not under {mount}"], {}
    if volume_gb is None:
        used_b, free_b = None, int((free or (lambda p: shutil.disk_usage(p).free))(mount))
    else:
        from stages.reasoning_data.h1pa_driver import used_bytes
        used_b = (used or used_bytes)(mount)
        free_b = int(float(volume_gb) * 10 ** 9) - used_b
    names = sorted(f.name for f in out.iterdir()) if out.is_dir() else []
    names += sorted(f"inputs/{f.name}" for f in (out / "inputs").iterdir()) if (out / "inputs").is_dir() else []
    foreign = [n for n in names if not OWN.match(n.split("/")[-1])]
    bad = []
    if free_b < MIN_FREE_BYTES:
        bad.append(f"{free_b / 1e9:.2f} GB free {'on the ' + str(volume_gb) + ' GB volume' if volume_gb else 'on the disk'}"
                   f", below the {MIN_FREE_BYTES / 1e9:g} GB the pilot requires")
    if foreign:
        bad.append(f"{out.name} holds files the pilot did not write, e.g. {foreign[:3]}: nothing is written over")
    return bad, {"mount": str(mount), "storage": "network_volume" if volume_gb else "container_disk",
                 "volume_gb": volume_gb, "used_bytes": used_b, "free_bytes": free_b,
                 "required_free_bytes": MIN_FREE_BYTES, "pilot_dir": out.name, "pilot_dir_entries": len(names)}


def resume_problems(src: Path, prefix: str = "pilot") -> tuple[list[str], dict]:
    """A dev-box copy of an earlier pilot session (`token_matched_phase.secure` pulled it) offered to a replacement
    pod: every ``.done`` marker's shard present, hashing to it AND receipted (`token_matched_phase.receipt`: received
    and verified off the pod), nothing but the pilot's own files. The pod checks the rest -- the pilot identity, the
    round input, the protocol, the runner (`tool_session.resumable`)."""
    src, bad = Path(src), []
    if not src.is_dir():
        return [f"no secured pilot session at {src}"], {}
    receipts = json.loads((src / "receipts.json").read_text()) if (src / "receipts.json").is_file() else {}
    dones = sorted(src.glob(f"{prefix}_*_r*_pass*.part*.done"))
    for d in dones:
        shard = d.with_suffix(".jsonl")
        if not shard.is_file() or sha(shard) != d.read_text().strip():
            bad.append(f"{shard.name} does not hash to its marker")
        elif (receipts.get(shard.name) or {}).get("sha256") != d.read_text().strip():
            bad.append(f"{shard.name} has no receipt of its sha256: not confirmed received off the pod")
    own = OWN if prefix == "pilot" else re.compile(rf"^({prefix}_[A-Za-z0-9-]+_r\d+_pass\d+\..+|{prefix}_[A-Za-z0-9-]+_r\d+"
                                                   r"\.jsonl|inputs|telemetry\.jsonl|tool_session\.json|receipts\.json)$")
    foreign = [f.name for f in src.iterdir() if not own.match(f.name)]
    if foreign:
        bad.append(f"{src} holds files the pilot did not write, e.g. {foreign[:3]}")
    return bad, {"path": str(src), "sealed_shards": len(dones),
                 "records": sum(len(d.with_suffix(".jsonl").read_text().splitlines()) for d in dones
                                if d.with_suffix(".jsonl").is_file())}


#: RunPod's per-datacenter availability, read-only (`placement`)
DATACENTERS_QUERY = ("query { dataCenters { id name location storageSupport listed "
                     "gpuAvailability { gpuTypeId stockStatus available } } }")


def placement(provider=None, gpu: str = PLACEMENT["gpu"]) -> dict:
    """The $0, read-only placement read before an attempt: every datacenter the provider lists with its stock of the
    pilot's GPU (a HINT: a create can still find none), the account's resources (pods and volumes) and the GPU's
    Secure Cloud price. ``candidates`` lists the datacenters that offer the GPU at all, by stock; a launch passes them
    to ``--data-center-ids`` or passes none and lets the provider choose. No volume is involved (`PLACEMENT`)."""
    if provider is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key
        provider = RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser())))
    obs = provider.observe(DATACENTERS_QUERY)
    price = provider.observe('query { gpuTypes(input:{id:"%s"}) { id securePrice '
                             'lowestPrice(input:{gpuCount:1}) { stockStatus } } }' % gpu)
    rank = {"High": 0, "Medium": 1, "Low": 2}
    rows = []
    for dc in ((obs.data or {}).get("dataCenters") or []) if obs.ok else []:
        for g in dc.get("gpuAvailability") or []:
            if g.get("gpuTypeId") == gpu:
                rows.append({"id": dc.get("id"), "name": dc.get("name"), "location": dc.get("location"),
                             "listed": dc.get("listed"), "storage_support": dc.get("storageSupport"),
                             "stock": g.get("stockStatus"), "available": g.get("available")})
    rows.sort(key=lambda r: (rank.get(str(r["stock"]), 3), str(r["id"])))
    res = provider.account_resources()
    return {"schema": "aadistill.stage2.teacher_modes.pilot_placement/v1",
            "read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "gpu": gpu,
            "placement": PLACEMENT, "datacenters_known": obs.ok, "error": None if obs.ok else obs.error,
            "datacenters": rows, "candidates": [r["id"] for r in rows if r["stock"]],
            "price": (price.data or {}).get("gpuTypes") if price.ok else None,
            "account": {"known": res.known, "pods": list(res.pods),
                        "network_volumes": [v.get("id") for v in res.network_volumes], "error": res.error},
            "_": "stock is the provider's hint, not a reservation; the create's own answer decides",
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
             "gpu": TM.GPU, **PLACEMENT_AMENDMENT, "ceiling_usd_at_amendment": g["ceiling_usd"],
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
                                        f"file {block['tasks_sha256'][:12]}..., measurement only, one Secure Cloud "
                                        f"L40S with no network volume, hard cap ${cap:.2f} over every attempt, inside "
                                        "this grant's remaining envelope",
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
#: the maintainer's 2026-11-04 amendment of the grant's tool_pilot block (`pilot_grant_problems` requires it): the
#: placement above in place of am5m373kuq in EUR-IS-2, and the kind cap made cumulative over every attempt
PLACEMENT_AMENDMENT = {
    "network_volume": None, "placement": PLACEMENT, "cap_is_cumulative": True,
    "attempts": "each attempt is a new run with a new one-use artifact; its hard cap is at most the kind cap less the "
                "closeout all-in of every earlier tool_pilot run (token_matched_phase.kind_cap_problems)"}
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
            "storage": f"{PLACEMENT['storage']}; container disk ({DISK_GB} GB) is inside the all-in; no volume is "
                       "created or attached (am5m373kuq retained untouched, 4rhubprswo never attached)",
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
                    f"--gpu 'NVIDIA L40S' --disk-gb {DISK_GB} [--data-center-ids <placement candidates>] "
                    "[--resume-from <a secured earlier attempt>] --dry-run --authorization <scratch>/auth.json"},
            {"step": "launch (only after the grant and the signature)",
             "how": f"{cmd} launch --kind tool_pilot --minutes {CAP_OPTIONS['B'][1]:g} --run-id <run> --scr <scratch> "
                    "--session-commit <sha> --bundle <name> --max-price <live> --gpu 'NVIDIA L40S' "
                    f"--disk-gb {DISK_GB} [--data-center-ids <placement candidates>] [--resume-from <a secured "
                    "earlier attempt>] --create-attempts <n> --create-retry-seconds <s>"},
            {"step": "closeout (the launcher runs it on every terminal path) and the dev box's report",
             "how": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot report --session "
                    "<scratch>/store/tool_pilot_hybrid [--session <a resumed run's copy>] [--study-identity <the "
                    "run's authorization plan_hash, when tool_session.json is missing>] --write <report>"}],
        "gates": {"at_issue": ["the grant names a tool_pilot cap and the cap fits it (kind_cap_problems)",
                               "the grant carries the pilot-only tool_pilot block: approves exactly 'tool_pilot "
                               "measurement only', granted by the maintainer with a decision and a date, the plan and "
                               "task file sha256, the source admission, the cap, the L40S, the placement and the unchanged "
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
                                "the placement is the pilot's: no network volume, any datacenter named offers the "
                                "L40S, a resume source hashes to its markers (the_pilot_placement_is_the_planned_one)",
                                "the kind's bound records are the committed ones",
                                "the session bundle is on the relay"],
                  "on_the_pod": ["setup re-checks the artifact against the pod's files (require_on_pod)",
                                 "the session re-checks the pilot plan, the task file and the admission with the same "
                                 "admission_problems (tool_session.run_pilot)",
                                 "a resume source's shards copied in and re-verified under the pilot identity "
                                 "(tool_session.resumable)",
                                 "before generating: at least 2 GB free on the container disk and the pilot's "
                                 "directory new or only the pilot's own files (tool_pilot.storage_problems)"]},
        "placement": "tool_pilot placement --write <the attempt's run directory>/evidence/placement.json (read-only)",
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
        "recovery": "identity-bound sealed shards (generation_driver.run_group, sealed every 50 records or 300 s), "
                    "pulled to the dev box on every poll; a replacement session -- a new run with a new one-use "
                    "artifact, given the secured copies with --resume-from -- regenerates nothing "
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
                                        "throughput, TTFT, model-load time and the real admission rates",
                                        "the live L40S quote and stock at launch",
                                        "the container disk's free space (the pod measures "
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


def _outcome(task: dict, rec: dict | None, attempted: set, *, mode: str, thinking: bool,
             stop_ids) -> tuple[str, dict, dict | None]:
    """One candidate's outcome -- admitted or rejected (sealed and verified), attempted_unsealed, or unattempted (an
    unattempted or unsealed task is never a verification failure) -- with its admitted units (supervised tokens of its
    documents, scored positions of its items) and its verdict (`EpisodeVerdict.as_dict`; None when unsealed)."""
    none = {"supervised_tokens": 0, "scored_positions": 0}
    if rec is None:
        return ("attempted_unsealed" if task["task_id"] in attempted else "unattempted"), none, None
    turns = rec.get("turns") or []
    v = TE.verify(task, turns, thinking=thinking)
    if not v.admitted:
        return "rejected", none, v.as_dict()
    if task["split"] in ("calibration", "state_eval"):
        its = TE.items(task, turns, mode=mode, thinking=thinking)
        if not its:
            return "rejected", none, {**v.as_dict(), "admitted": False, "problems": ["item protocol"]}
        return "admitted", {**none, "scored_positions": sum(len(i["tags"]["assistant"]) for i in its)}, v.as_dict()
    docs = TE.documents(task, turns, mode=mode, stop_ids=stop_ids)
    return "admitted", {**none, "supervised_tokens": sum(len(d.output_ids) for d in docs)}, v.as_dict()


def _measured(v, why: str = "not reported by the engine or the GPU tools on this run"):
    return {"value": v} if v not in (None, {}, []) else {"value": UNMEASURED, "why": why}


def _rate(num: int, den: int):
    return round(num / den, 4) if den else UNMEASURED


#: the report's metrics, each by its exact numerator and denominator (docs/shared/METRIC_CONTRACT.md :: tool pilot).
#: They measure TEACHER DATA ADMISSION under the frozen verifier: none is a BFCL, tau-bench or other benchmark score,
#: and none is a model-capability measurement
METRICS = {
    "first_candidate_admission_rate": "candidate-0 episodes admitted / candidate-0 episodes sealed (unattempted and "
                                      "attempted-but-unsealed tasks are in neither)",
    "admission_yield_up_to_two_candidates": "tasks with an admitted episode / tasks whose outcome is determined "
                                            "(admitted at candidate 0, or candidate 0 rejected and candidate 1 sealed). "
                                            "Candidate 1 is generated ONLY for a task whose candidate 0 was sealed and "
                                            "rejected: a bounded data-admission retry, not pass@2 (which samples two "
                                            "candidates for every task) and not pass^k",
    "level_pass_rate": "per verifier level and candidate: sealed episodes that passed it / sealed episodes it was "
                       "checked on (passed + failed); unverified (None) episodes are counted apart, never as passed",
    "protocol_validity": "level protocol: natural termination, the mode's think rule, every tool-call block parsed, "
                         "V1's tail rule",
    "schema_validity": "level schema_validity: every call names an offered tool and its arguments satisfy its JSON "
                       "Schema",
    "gold_tool_call_match": "level step_gold: the call set of every gold-carrying turn equals one allowed gold set "
                            "(name + arguments, canonicalized; order-free) -- exact match against the source's "
                            "recorded expert call (G2: agreement with a model-generated gold, not human-verified)",
    "recorded_execution": "level recorded_execution: every executed call matched a recorded response of the replay "
                          "table (recorded-response replay; no executable environment, no final-state check); None "
                          "when the episode executed no call",
    "target_tools_called": "level target_tools_called: every target tool called at least once (recall; extra or wrong "
                           "calls unchecked) -- not tool-selection accuracy",
    "tool_selection": "level tool_selection: here only no_call -- no call on any turn of a no-tool task",
    "verification_coverage": "admitted episodes by the exact set of levels they were verified at (True), per source "
                             "and episode shape",
    "supervised_tokens": "generated token ids of the admitted training/validation episodes' turns (each counted once)",
    "scored_positions": "assistant positions of the admitted calibration/state-evaluation episodes' items",
}


def _blank() -> dict:
    return {"offered": 0, "c0": Counter(), "c1": Counter(), "final": Counter(), "supervised_tokens": 0,
            "scored_positions": 0}


def _finish(row: dict) -> dict:
    """A cell's counts closed into its metrics (`METRICS`)."""
    c0, c1, final = row["c0"], row["c1"], row["final"]
    sealed0 = c0["admitted"] + c0["rejected"]
    determined = final["admitted"] + final["rejected"]
    units = row["supervised_tokens"] + row["scored_positions"]
    return {"offered": row["offered"],
            "candidate_0": {"attempted": c0["admitted"] + c0["rejected"] + c0["attempted_unsealed"], "sealed": sealed0,
                            **{k: c0[k] for k in ("admitted", "rejected", "attempted_unsealed", "unattempted")}},
            "candidate_1": {"required": c1["required"], "sealed": c1["admitted"] + c1["rejected"],
                            **{k: c1[k] for k in ("admitted", "rejected", "attempted_unsealed", "unattempted")}},
            "final": {k: final[k] for k in ("admitted", "rejected", "undetermined")},
            "first_candidate_admission_rate": _rate(c0["admitted"], sealed0),
            "admission_yield_up_to_two_candidates": _rate(final["admitted"], determined),
            "supervised_tokens": row["supervised_tokens"], "scored_positions": row["scored_positions"],
            "units_per_admitted_episode": (round(units / final["admitted"], 1) if final["admitted"] else UNMEASURED),
            #: the counts as earlier report versions keyed them
            "c0": dict(c0), "c1": dict(c1)}


def _levels(verdicts: list[dict]) -> dict:
    """Per verifier level: passed / failed / unverified over sealed episodes, and the level pass rate."""
    from aadistill.data.tool_calls import LEVELS
    out = {}
    for lv in LEVELS:
        got = Counter({True: "passed", False: "failed", None: "unverified"}[v["checks"].get(lv)] for v in verdicts)
        out[lv] = {"passed": got["passed"], "failed": got["failed"], "unverified": got["unverified"],
                   "pass_rate": _rate(got["passed"], got["passed"] + got["failed"])}
    return out


def _source(task: dict) -> str:
    return str(task["source"].get("dataset"))


def report(session_dirs, *, study_identity: str | None = None, plan_path: Path = PLAN,
           tasks_path: Path = TASKS, supply_by_shape: dict | None = None) -> dict:
    """The pilot's measurements from one or more secured copies of its session directory, complete or not:

    * every sealed record is read -- a group need not have a ``.complete`` -- after `generation_driver.sealed` verifies
      each shard's sha256 and the launch identity; copies of a resumed session merge, and a task sealed differently in
      two copies refuses;
    * per task and candidate: admitted or rejected (sealed, under the frozen verifier), attempted but unsealed, or
      unattempted; candidate 1 is required only where candidate 0 was sealed and rejected; a task counts once;
    * per mode, by split / category / shape and by source / shape: offered, attempted, sealed, admitted, rejected and
      incomplete tasks; the first-candidate admission rate and the admission yield of up to two candidates
      (`METRICS`: data admission, never a benchmark score); each verifier level's passed / failed / unverified counts
      per candidate; the admitted episodes' verification coverage; supervised tokens and scored positions;
    * the non-thinking schema_arguments supply per split, and each formal cell's projection from the measured yield
      (a projection, labelled as one);
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
        cells, by_source, verdicts, coverage = {}, {}, {}, {}
        for c in doc["modes"][mode]:
            for t in c["task_ids"]:
                task = by_id[t]
                shape = TE.episode_shape(task)
                rows = (cells.setdefault(f"{c['split']}|{c['category']}|{shape}", _blank()),
                        by_source.setdefault(f"{_source(task)}|{shape}", _blank()))
                o0, u0, v0 = _outcome(task, recs[0].get(t), att[0], mode=mode, thinking=thinking, stop_ids=stop_ids)
                o1, u1, v1 = "not_required", {"supervised_tokens": 0, "scored_positions": 0}, None
                if o0 == "rejected":
                    o1, u1, v1 = _outcome(task, recs[1].get(t), att[1], mode=mode, thinking=thinking,
                                          stop_ids=stop_ids)
                elif t in recs[1]:
                    raise PilotError(f"{mode}: {t} has a second candidate its first did not call for")
                final = ("admitted" if "admitted" in (o0, o1) else "rejected" if (o0, o1) == ("rejected", "rejected")
                         else "undetermined")
                for row in rows:
                    row["offered"] += 1
                    row["c0"][o0] += 1
                    if o0 == "rejected":
                        row["c1"]["required"] += 1
                        row["c1"][o1] += 1
                    row["final"][final] += 1
                    for u in ("supervised_tokens", "scored_positions"):
                        row[u] += u0[u] + u1[u]
                key = f"{_source(task)}|{shape}"
                for k, v in ((0, v0), (1, v1)):
                    if v is not None:
                        verdicts.setdefault(key, {}).setdefault(k, []).append(v)
                won = v0 if o0 == "admitted" else v1 if o1 == "admitted" else None
                if won is not None:
                    coverage.setdefault(key, Counter())["+".join(won["verified_levels"])] += 1
                all_required_sealed &= o0 in ("admitted", "rejected") and o1 in ("admitted", "rejected", "not_required")
        n = len(recs[0]) + len(recs[1])
        all_sealed += n
        every.update({f"{t}#{k}": r for k, rs in recs.items() for t, r in rs.items()})
        done = {k: _finish(v) for k, v in sorted(cells.items())}
        modes[mode] = {"cells": done, "by_source_and_shape": {k: _finish(v) for k, v in sorted(by_source.items())},
                       "levels": {key: {f"candidate_{k}": _levels(vs) for k, vs in sorted(per.items())}
                                  for key, per in sorted(verdicts.items())},
                       "verification_coverage": {k: dict(v) for k, v in sorted(coverage.items())},
                       "totals": _finish(_sum(cells.values())),
                       "sealed_records": n,
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
    return {"schema": "aadistill.stage2.teacher_modes.tool_pilot_report/v4", "state": state,
            "pilot_plan_sha256": sha(plan_path), "identity": ident,
            "metrics": METRICS, "metric_contract": "docs/shared/METRIC_CONTRACT.md",
            "session_evidence": {d: ({k: r.get(k) for k in ("kind", "authorization", "terminal", "started_utc",
                                                           "finished_utc")} if r else
                                     "MISSING: no tool_session.json (a forced end); the report rests on the sealed "
                                     "shards alone") for d, r in sessions.items()},
            "modes": modes,
            "schema_arguments_non_thinking_supply": _schema_arguments_supply(doc, modes),
            "formal_plan_projection": _projection(doc, modes, supply_by_shape),
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


def _sum(rows) -> dict:
    out = _blank()
    for r in rows:
        out["offered"] += r["offered"]
        for k in ("c0", "c1", "final"):
            out[k].update(r[k])
        for u in ("supervised_tokens", "scored_positions"):
            out[u] += r[u]
    return out


def _schema_arguments_supply(doc: dict, modes: dict) -> dict:
    """The non-thinking mode's schema_arguments data the pilot admitted, per split and shape: supervised tokens for
    training, scored positions for calibration and state evaluation (the gap option 1 addressed)."""
    out = {}
    for key, row in (modes.get("H-N") or {}).get("cells", {}).items():
        split, cat, shape = key.split("|")
        if cat == "schema_arguments":
            out[f"{split}|{shape}"] = {k: row[k] for k in ("offered", "final", "supervised_tokens", "scored_positions",
                                                            "first_candidate_admission_rate",
                                                            "admission_yield_up_to_two_candidates")}
    return out


def supply_from_plan(plan_path: Path) -> dict:
    """``{"<mode>|<split>|<category>": {shape: tasks}}`` from a bounded generation plan (`tool_generation_plan`'s
    ``supply_by_shape``): the per-shape supply a projection multiplies, which the frozen pilot plan does not carry."""
    plan = json.loads(Path(plan_path).read_text())
    return {f"{m}|{c['split']}|{c['category']}": c["supply_by_shape"]
            for li in plan["lineages"].values() for m, mm in li["modes"].items() for c in mm["cells"]
            if "supply_by_shape" in c}


def _projection(doc: dict, modes: dict, supply_by_shape: dict | None = None) -> dict:
    """Per formal cell (the pilot plan carries its target and its pool supply in tasks), three things kept apart:
    the SEALED units (admitted episodes of the pilot, measured); when the pilot attempted the cell's WHOLE supply, that
    is the cell's measured total (REACHED or SHORTFALL_EXHAUSTED: an exact gap, no projection); otherwise the WHOLE
    SUPPLY's projection, shape by shape -- each shape's untried tasks at that shape's measured units per attempted
    task (its admission yield x units per admitted episode) -- which needs the per-shape supply (``supply_by_shape``,
    `supply_from_plan`); without it, or without an admitted episode of a shape, UNMEASURED. A projection from a sample,
    never a measurement or a decision; no SHORTFALL is approved here."""
    out = {}
    for mode, cs in doc["modes"].items():
        got = (modes.get(mode) or {}).get("cells", {})
        for c in cs:
            key = f"{mode}|{c['split']}|{c['category']}"
            rows = {k.split("|")[2]: r for k, r in got.items() if k.startswith(f"{c['split']}|{c['category']}|")}
            det = {sh: r["final"]["admitted"] + r["final"]["rejected"] for sh, r in rows.items()}
            sealed = round(sum(r["supervised_tokens"] + r["scored_positions"] for r in rows.values()), 1)
            row = {"unit": c["unit"], "target": c["target"], "supply_tasks": c["supply"],
                   "attempted_tasks": sum(det.values()), "sealed_units": sealed}
            if sum(det.values()) >= c["supply"]:
                out[key] = {**row, "basis": "measured: the whole supply attempted",
                            "status": "REACHED" if sealed >= c["target"] else "SHORTFALL_EXHAUSTED"}
                continue
            sup = (supply_by_shape or {}).get(key)
            if not sup or any(rows.get(sh, {}).get("final", {}).get("admitted", 0) == 0 for sh in sup if sup[sh]):
                out[key] = {**row, "status": UNMEASURED,
                            "why": "no per-shape supply" if not sup else "a shape with no admitted episode"}
                continue
            by_shape = {}
            proj = sealed
            for sh, n in sup.items():
                r = rows[sh]
                units = r["supervised_tokens"] + r["scored_positions"]
                per = units / det[sh]
                by_shape[sh] = {"supply_tasks": n, "attempted": det[sh], "units_per_attempted_task": round(per, 3),
                                "projected_units": round(units + (n - det[sh]) * per, 1)}
                proj += (n - det[sh]) * per
            out[key] = {**row, "basis": "projection by shape", "by_shape": by_shape,
                        "projected_units_from_whole_supply": round(proj, 1),
                        "status": "REACHABLE" if proj >= c["target"] else "SHORTFALL_PROJECTED"}
    return out


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
    sp = sub.add_parser("placement")
    sp.add_argument("--write", required=True)
    r = sub.add_parser("report")
    r.add_argument("--session", action="append", required=True, help="a secured session directory (repeatable)")
    r.add_argument("--study-identity", default=None,
                   help="the run's authorization plan_hash, when tool_session.json is missing")
    r.add_argument("--write", required=True)
    r.add_argument("--supply-plan", default=None, help="a bounded generation plan: its per-shape supply for the projection")
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
    elif a.cmd == "placement":
        doc = placement()
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({k: doc[k] for k in ("datacenters", "candidates", "price", "account")}, indent=1,
                         default=str))
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
        sup = supply_from_plan(Path(a.supply_plan)) if a.supply_plan else None
        Path(a.write).write_text(json.dumps(report([Path(x) for x in a.session], study_identity=a.study_identity,
                                                   supply_by_shape=sup), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from collections import Counter, defaultdict
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


# --- the pilot-only source admission -------------------------------------------------------------------------------

def source_admission(doc: dict, plan_path: Path = PLAN) -> dict:
    """The record admitting the two candidate sources for THIS pilot's measurement only (module docstring). It is
    PREPARED, not active: ``signature.signed_by`` stays null until the maintainer signs, and the ``tool_pilot`` kind
    refuses an unsigned record (`token_matched_phase.kind_conditions`)."""
    review = json.loads(REVIEW.read_text())["candidates"]
    lic = {"Agent-Ark/Toucan-1.5M": review["Agent-Ark/Toucan-1.5M"]["licence"],
           "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1":
               review["nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 + Conversational-Tool-Use-Pivot-v1"]["licence"]}
    tasks = [json.loads(x) for x in (REPO / doc["tasks_file"]["path"]).read_text().splitlines() if x.strip()]
    per = Counter((str(t["source"].get("dataset")), str(t["source"].get("revision"))) for t in tasks)
    shapes = Counter((str(t["source"].get("dataset")), TE.episode_shape(t)) for t in tasks)
    return {"schema": ADMISSION_SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PREPARED: inactive until the maintainer signs it (signature below)",
            "decided_by": "maintainer 2026-11-04 (review of 2468c1aa), item 5: prepare a source admission limited to "
                          "the pilot, from the accepted candidates Toucan SFT and Pivot Conversational; it must not "
                          "widen into a training-data admission and authorizes no paid action",
            "scope": "the tool_pilot kind's measurement only: the hybrid's teacher-native generation on exactly the "
                     "tasks below (by id and sha256), its episodes measurement records by default",
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
            "does_not": ["admit any source or task to formal training, calibration or state evaluation",
                         "build or replace the canonical tool_pool@v1 under data/",
                         "decide any category's supply or any SHORTFALL",
                         "authorize a GPU, a pod or any spend (the pilot's grant does that, separately)"],
            "signature": {"signed_by": None, "decision": None, "date": None}, "authorizes": "nothing by itself"}


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
        return round(sorted(xs)[int(p * (len(xs) - 1))], 3) if xs else None
    return {"turns": len(ttft), "ttft_seconds": {"p50": q(ttft, .5), "p90": q(ttft, .9), "max": q(ttft, 1.0)},
            "per_request_decode_tokens_per_second": {"p50": q(decode, .5), "p10": q(decode, .1)},
            "prompt_tokens": {"p50": q(prompt, .5), "p90": q(prompt, .9), "total": sum(prompt)},
            "_": "batched engine: a request's TTFT includes its queueing"}


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
        "schema": "aadistill.budget.application/tool_pilot_v2",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "RE-VERIFIED after the execution preparation (maintainer review of 2468c1aa); AUTHORIZES NOTHING: "
                  "no GPU, no pod, no spend",
        "decided_by": ["maintainer 2026-11-04 (review of 647eea0e), item 7: prepare the pilot's application",
                       "maintainer 2026-11-04 (review of 2468c1aa), item 5: the 794-task design accepted in principle; "
                       "complete the $0 execution preparation; not a paid authorization"],
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
             "how": f"its signature fields in {_rel(admission_path)} (signed_by, decision, date); the kind refuses "
                    "until then"},
            {"step": "the maintainer amends the study grant with the pilot's cap", "who": "maintainer",
             "how": "logs/budget/approvals/teacher_modes_token_matched_grant.json :: session_hard_caps_usd.tool_pilot "
                    "= the chosen option's cap, and its scope naming the pilot; issue refuses until then"},
            {"step": "commit; stage the session bundle of that commit on the relay",
             "how": "stages.phase_c1.bundle.stage_bundle (the procedure of every Research A session); the launch's "
                    "bundle_on_relay gate refuses without it"},
            {"step": "re-read the project position (read-only)",
             "how": "PYTHONPATH=src:scripts .venv/bin/python scripts/shared/project_position.py --live "
                    "--request tool_pilot=<cap>"},
            {"step": "issue the one-use artifact",
             "how": f"{cmd} issue --kind tool_pilot --run-id <run> --expected-usd 1.14 --hard-cap-usd <cap>"},
            {"step": "dry-run every pre-provider gate at the live quote (no create)",
             "how": f"{cmd} issue --kind tool_pilot --run-id <run>_dry --expected-usd 1.14 --hard-cap-usd <cap> "
                    f"--dry-run --out <scratch>/auth.json && {cmd} launch --kind tool_pilot --minutes <minutes> "
                    "--run-id <run>_dry --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live> "
                    f"--gpu 'NVIDIA L40S' --disk-gb {DISK_GB} --network-volume-id {VOLUME['id']} --volume-mount-path "
                    f"{VOLUME['mount']} --data-center-ids {VOLUME['data_center']} --volume-gb {VOLUME['size_gb']} "
                    "--dry-run --authorization <scratch>/auth.json"},
            {"step": "launch (only after the grant and the signature)",
             "how": f"{cmd} launch --kind tool_pilot --minutes <minutes> --run-id <run> --scr <scratch> "
                    "--session-commit <sha> --bundle <name> --max-price <live> --gpu 'NVIDIA L40S' "
                    f"--disk-gb {DISK_GB} --network-volume-id {VOLUME['id']} --volume-mount-path {VOLUME['mount']} "
                    f"--data-center-ids {VOLUME['data_center']} --volume-gb {VOLUME['size_gb']}"},
            {"step": "closeout (the launcher runs it on every terminal path) and the dev box's report",
             "how": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot report --session "
                    "<scratch>/store/tool_pilot_hybrid --write <report>"}],
        "gates": {"at_issue": ["the grant names a tool_pilot cap and the cap fits it (kind_cap_problems)",
                               "the cap fits the study envelope, the project cap as the provider bills it now and the "
                               "account balance (funding_problems on a live, fail-closed position)",
                               "the L40S check record admits teacher generation (kind_conditions)",
                               "the pilot plan and task file load (tool_pilot.load) and the source admission is signed, "
                               "pilot-scoped and names this plan (pilot_problems)",
                               "the protocol, design, pilot plan and admission bound by sha256 (KIND_RECORDS)"],
                  "at_launch": ["the artifact is this study's tool_pilot artifact, unused, not a dry run",
                                "one NVIDIA L40S only (the_pilot_runs_on_an_l40s; the grant's price ceiling)",
                                "the all-in fits the hard cap at the live quote (all_in_within_cap)",
                                "the study identity and the bound files are the session commit's",
                                "the project cap holds now: committed spend, remaining cap, balance, pods and volumes "
                                "re-read from the provider (the_project_cap_holds_now; an unreadable provider, a pod in "
                                "the account or an unowned volume refuses)",
                                "the network volume is the planned one (the_volume_is_the_planned_one)",
                                "the kind's bound records are the committed ones",
                                "the session bundle is on the relay"],
                  "on_the_pod": ["setup re-checks the artifact against the pod's files (require_on_pod)",
                                 "the session re-checks the pilot plan, the task file and the signed admission "
                                 "(tool_session.run_pilot)"]},
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
        "needed_from_the_maintainer": ["sign the pilot-only source admission", "amend the grant with the pilot's cap",
                                       "choose cap option A ($1.80, the expected workload) or B ($2.50, the upper "
                                       "workload in one session)"],
        "not_requested": ["the bounded tool generation", "the packed full search", "any training", "formal evaluation",
                          "a Thinking-2507 session", "a raised project cap", "any SHORTFALL approval",
                          "anything on 4rhubprswo", "a merge"],
        "authorizes": "nothing"}


# --- the report ------------------------------------------------------------------------------------------------------

def report(session_dir: Path, plan_path: Path = PLAN, tasks_path: Path = TASKS) -> dict:
    """The pilot's measurements from a secured session directory (its sealed shards, ``tool_session.json`` and
    ``telemetry.jsonl``), every pass rate under the frozen verifier (`tool_admission.admit`), beside the priors."""
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import tool_admission as TA
    doc, by_id = load(plan_path, tasks_path)
    session_dir = Path(session_dir)
    rec = json.loads((session_dir / "tool_session.json").read_text())
    ident = rec["identity"]
    proto = json.loads(TGP.PROTOCOL.read_text())
    stop_ids = [int(x) for x in proto["execution"]["stop_ids"]]
    out = {"schema": "aadistill.stage2.teacher_modes.tool_pilot_report/v1", "pilot_plan_sha256": sha(plan_path),
           "session": {k: rec.get(k) for k in ("kind", "authorization", "terminal", "started_utc", "finished_utc")},
           "modes": {}}
    for mode in doc["modes"]:
        cand = defaultdict(dict)
        for done in sorted(session_dir.glob(f"pilot_{mode}_r1_pass*.complete")):
            label = done.name[:-len(".complete")]
            k = int(label.rsplit("pass", 1)[1])
            for tid, r in GD.sealed(session_dir, label, ident).items():
                cand[k][tid] = r
        tasks = [by_id[t] for c in doc["modes"][mode] for t in c["task_ids"]]
        eps = [{"task_id": t, "turns": r.get("turns") or [], "candidate": k} for k, rs in cand.items() for t, r in rs.items()]
        res = TA.admit(tasks, eps, mode=mode, stop_ids=stop_ids)
        first = TA.admit(tasks, [e for e in eps if e["candidate"] == 0], mode=mode, stop_ids=stop_ids)
        by = defaultdict(lambda: {"offered": 0, "first_candidate_admitted": 0, "admitted": 0, "units": 0.0})
        shape_of = {t["task_id"]: TE.episode_shape(t) for t in tasks}
        for c in doc["modes"][mode]:
            for t in c["task_ids"]:
                by[f"{c['split']}|{c['category']}|{shape_of[t]}"]["offered"] += 1
        admitted = {d["task_id"] for d in res["documents"]} | {i["task_id"] for i in res["items"]}
        first_ok = {d["task_id"] for d in first["documents"]} | {i["task_id"] for i in first["items"]}
        units = Counter()
        for d in res["documents"]:
            units[d["task_id"]] += len(d["output_ids"])
        for i in res["items"]:
            units[i["task_id"]] += len(i["tags"]["assistant"])
        for c in doc["modes"][mode]:
            for t in c["task_ids"]:
                key = f"{c['split']}|{c['category']}|{shape_of[t]}"
                by[key]["first_candidate_admitted"] += t in first_ok
                by[key]["admitted"] += t in admitted
                by[key]["units"] += units[t]
        for v in by.values():
            v["first_candidate_pass_rate"] = round(v["first_candidate_admitted"] / v["offered"], 4) if v["offered"] else None
            v["pass_rate_two_candidates"] = round(v["admitted"] / v["offered"], 4) if v["offered"] else None
            v["units_per_admitted_episode"] = round(v["units"] / v["admitted"], 1) if v["admitted"] else None
        every = {f"{t}#{k}": r for k, rs in cand.items() for t, r in rs.items()}
        out["modes"][mode] = {"cells": dict(sorted(by.items())), "census": res["census"],
                              "turns": turn_telemetry(every)}
    tel = session_dir / "telemetry.jsonl"
    out["telemetry"] = [json.loads(x) for x in tel.read_text().splitlines() if x.strip()] if tel.is_file() else []
    out["priors"] = proto["bounded_plan"]["priors"]
    out["authorizes"] = "nothing"
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
    r = sub.add_parser("report")
    r.add_argument("--session", required=True)
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
        Path(a.write).write_text(json.dumps(report(Path(a.session)), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

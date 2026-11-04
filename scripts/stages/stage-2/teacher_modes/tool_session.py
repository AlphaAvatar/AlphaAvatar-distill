#!/usr/bin/env python3
"""On-pod driver of the BOUNDED teacher-native tool-calling generation (Research A packed round), one lineage per session.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/tool_session.py --authorization <a.json> \\
        --lineage hybrid --durable-root <volume>/teacher_modes --status <file>

`tool_generation_protocol.json` names the lineage's teacher, modes, pool and execution terms; the frozen generation
plan (`tool_generation_plan.py`, bound by the artifact with the pool manifest) names every cell -- split, category,
unit, target and bound. Before anything runs, the plan must name this pool's content and this protocol, and
`tool_generation_plan.verified` re-verifies every split file the lineage reads and the plan's cells: a missing file, a
changed byte, a foreign source, an edited order or target refuses the session.

Per mode, the cells are generated in ROUNDS (`tool_generation_plan.round_tasks`): each round takes every unfinished
cell's next tasks in its frozen order, runs them through `run_battery` in generate mode with the tool environment
(``--env-factory``; each task renders its own system text), then the second candidates of the round's refused
tasks. Every finished episode is verified here (`tool_episodes.verify`) and counted in its cell's own unit --
supervised tokens of its documents, scored positions of its items, or tasks -- the units the dev box's admission
and the packs count again. A cell stops at its target; it never attempts past its bound or after the last round,
and a cell short at either is recorded as SHORTFALL. `generation_driver.run_group` seals every finished episode into
identity-bound shards, and a replacement session recomputes every round from them, so nothing finished is
regenerated and the stopping state comes from the volume. Nothing here admits data: the dev box does
(`tool_admission.py`) from the sealed shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import tool_episodes as TE  # noqa: E402
from stages.teacher_modes import tool_generation_plan as TGP  # noqa: E402

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
SCHEMA = "aadistill.stage2.teacher_modes.tool_session/v2"


class PlanError(RuntimeError):
    """A plan that cannot be run (an unknown unit, a task picked by two cells)."""


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def checked_plan(proto: dict, lineage: str):
    """``(plan, pool manifest or None, tasks by id)``. A plan that writes the lineage's bound task file (P8.4) is read
    with `tool_generation_plan.verified_bound` -- the pod holds that file, never the pool; the dev box checked the
    plan against the pool at issue. Otherwise `tool_generation_plan.verified`: its split files re-verified, the plan
    bound to this pool's content, pool.json, protocol and design, its cells unchanged. Either way the formal source
    admission (`tool_source_admission.problems`) must admit the plan's pool content and every task's source."""
    plan = json.loads((REPO / proto["pool"]["plan"]).read_text())
    if (plan.get("lineages", {}).get(lineage) or {}).get("tasks_file"):
        plan, by_id = TGP.verified_bound(proto, lineage)
        man = None
    else:
        plan, man, by_id = TGP.verified(proto, lineage=lineage)
    from stages.teacher_modes import tool_source_admission as TSA
    bad = TSA.problems(TSA.load(), pool_content_sha256=plan["pool"]["content_sha256"], tasks=by_id.values())
    if bad:
        raise PlanError("the formal source admission refuses: " + "; ".join(bad))
    return plan, man, by_id


def round_body(tasks: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in tasks)


def write_round(tasks: list[dict], out: Path, label: str, default_system: str) -> dict:
    """The round's run_battery input (each task keeps its own ``system``); an existing file must be identical."""
    d = out / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{label}.jsonl"
    body = round_body(tasks)
    if path.is_file() and path.read_text() != body:
        raise RuntimeError(f"{path.name} on the volume is not this session's input")
    path.write_text(body)
    return {"file": path.name, "system": default_system, "tasks": len(tasks),
            "sha256": hashlib.sha256(body.encode()).hexdigest()}


def context(proto: dict, lineage: str, identity: str) -> dict:
    """The `generation_driver.run_group` context for one lineage."""
    li = proto["lineages"][lineage]
    return {"protocol": {"execution": proto["execution"], "teacher": {"model": li["model"], "revision": li["revision"]}},
            "launch_identity_sha256": identity, "protocol_path": str(PROTOCOL.relative_to(REPO))}


def mode_args(proto: dict, lineage: str, mode: str) -> tuple[str, ...]:
    m = proto["lineages"][lineage]["modes"][mode]
    return ("--chat-template-kwargs", json.dumps(m["template_kwargs"], sort_keys=True),
            "--generate-sampling", json.dumps(m["sampling"], sort_keys=True),
            "--gpu-mem-util", str(proto["execution"]["gpu_memory_utilization"]),
            "--env-factory", proto["execution"]["env_factory"])


def units(task: dict, turns: list, unit: str, *, mode: str, thinking: bool, stop_ids) -> float | None:
    """The cell units one episode contributes, or None when it is refused (verification or the item protocol)."""
    if not TE.verify(task, turns or [], thinking=thinking).admitted:
        return None
    if unit == "tasks":
        return 1.0
    if unit == "supervised_tokens":
        return float(sum(len(d.output_ids) for d in TE.documents(task, turns, mode=mode, stop_ids=stop_ids)))
    if unit == "scored_positions":
        its = TE.items(task, turns, mode=mode, thinking=thinking)
        return float(sum(len(it["tags"]["assistant"]) for it in its)) if its else None
    raise PlanError(f"unknown unit {unit!r}")


def generate_mode(proto: dict, plan: dict, lineage: str, mode: str, by_id: dict, man: dict, out: Path,
                  identity: str, run, mark, *, cells: list[dict] | None = None, orders: list | None = None,
                  max_rounds: int | None = None, label_prefix: str = "tool", telemetry: Path | None = None,
                  extra_args: tuple[str, ...] = (), imported: dict | None = None) -> dict:
    """Every cell of one mode, round by round, to its target or its bound; returns the cells' final states. A pilot
    passes its frozen ``cells`` and ``orders`` (`tool_pilot.cells`), one round and its own label prefix; with
    ``telemetry`` every engine invocation appends its GPU and engine telemetry there (`tool_pilot`). ``imported``
    (`tool_generation_import.outcomes`): tasks already DONE in this mode -- each cell starts from their units and
    admissions, and its rounds walk the frozen order without them, so none is attempted or re-sampled again."""
    from stages.reasoning_data import generation_driver as GD
    bp, ex = proto["bounded_plan"], proto["execution"]
    cand, stop_ids = int(ex["candidates"]), [int(x) for x in ex["stop_ids"]]
    thinking = bool(proto["lineages"][lineage]["modes"][mode]["template_kwargs"].get("enable_thinking", True))
    done = imported or {}
    if cells is None:
        cells = plan["lineages"][lineage]["modes"][mode]["cells"]
        orders = [([by_id[t] for t in c["task_ids"]] if "task_ids" in c else
                   TGP.cell_order(man["orders"].get(c["split"], []), by_id, c["category"])) for c in cells]
    state = []
    for i, c in enumerate(cells):
        if "sealed" in c and "task_ids" in c:       #: a plan that bound the import record: its cells carry the outcomes
            state.append({"attempted": 0, "units": float(c["sealed"]["units"]), "admitted": int(c["sealed"]["admitted"]),
                          "imported": int(c["sealed"]["tasks"])})
            continue
        mine = [done[t["task_id"]] for t in orders[i] if t["task_id"] in done]
        units_of = (lambda o: float(o["admitted"])) if c["unit"] == "tasks" else (lambda o: o["units"] if o["admitted"] else 0.0)
        state.append({"attempted": 0, "units": sum(units_of(o) for o in mine), "admitted": sum(o["admitted"] for o in mine),
                      "imported": len(mine)})
        orders[i] = [t for t in orders[i] if t["task_id"] not in done]
    c, groups = context(proto, lineage, identity), []
    for r in range(1, int(max_rounds or bp["rounds"]["max_rounds"]) + 1):
        picks = [TGP.round_tasks(cell, orders[i], state[i], r, bp=bp, mode=mode, candidates=cand)
                 for i, cell in enumerate(cells)]
        ids = [t for p in picks for t in p]
        if not ids:
            break
        if len(set(ids)) != len(ids):
            raise PlanError(f"round {r}: a task picked by two cells")
        label = f"{label_prefix}_{mode}_r{r}"
        f = write_round([by_id[t] for t in ids], out, label, ex["system_message_default"])
        admitted: dict[str, float] = {}
        retry = None
        for k in range(cand):
            if k and not retry:
                break
            t0 = time.time()
            with _gpu(telemetry) as gpu:
                g = GD.run_group(c, out, out / "inputs", f, k, retry, run, mark, label=f"{label}_pass{k}",
                                 extra_args=mode_args(proto, lineage, mode) + tuple(extra_args))
            recs = g.pop("records")
            groups.append(g)
            if telemetry is not None:
                _telemetry(telemetry, out, f"{label}_pass{k}", g, gpu, time.time() - t0, recs)
            if not g["complete"]:
                mark(f"{GD.UNIT_FAILED}:{label}_pass{k}")
                return {"complete": False, "groups": groups, "cells": _report(cells, state)}
            mark(f"{GD.UNIT_OK}:{label}_pass{k}")
            unit_of = {t: cells[i]["unit"] for i, p in enumerate(picks) for t in p}
            for tid, rec in recs.items():
                if tid in admitted:
                    continue
                u = units(by_id[tid], rec.get("turns") or [], unit_of[tid], mode=mode, thinking=thinking,
                          stop_ids=stop_ids)
                if u is not None:
                    admitted[tid] = u
            retry = sorted(t for t in ids if t not in admitted)
        for i, p in enumerate(picks):
            state[i]["attempted"] += len(p)              #: new tasks only; ``imported`` holds the rest
            state[i]["units"] += sum(admitted.get(t, 0.0) for t in p)
            state[i]["admitted"] += sum(t in admitted for t in p)
        mark(f"ROUND_DONE:{label}")
    return {"complete": True, "groups": groups, "cells": _report(cells, state)}


def _gpu(telemetry: Path | None):
    import contextlib
    from stages.teacher_modes import tool_pilot as TPL
    return TPL.GpuSampler() if telemetry is not None else contextlib.nullcontext()


def _telemetry(path: Path, out: Path, label: str, g: dict, gpu, seconds: float, recs: dict) -> dict:
    """One engine invocation's line of ``telemetry.jsonl``: wall time, GPU peaks, each new segment's engine lines and
    summary, the per-turn timings of the records it sealed. Never raises: telemetry does not stop a session."""
    from stages.teacher_modes import tool_pilot as TPL
    try:
        segs = [x["segment"] for x in g.get("segments") or [] if "segment" in x]
        row = {"label": label, "wall_seconds": round(seconds, 1), "complete": g.get("complete"),
               "gpu": gpu.as_dict() if hasattr(gpu, "as_dict") else None,
               "engine": [TPL.engine_telemetry(out / f"{label}.seg{n}.log", out / f"{label}.seg{n}.raw.summary.json")
                          for n in segs], "turns": TPL.turn_telemetry(recs)}
    except Exception as exc:  # noqa: BLE001
        row = {"label": label, "error": repr(exc)[:300]}
    with Path(path).open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def _report(cells: list[dict], state: list[dict]) -> list[dict]:
    out = []
    for cell, s in zip(cells, state):
        done = s["units"] >= cell["target"] and s["admitted"] >= cell.get("min_tasks", 0)
        out.append({k: cell[k] for k in ("split", "category", "unit", "target", "bound")} |
                   {**s, "status": "REACHED" if done else "SHORTFALL"})
    return out


# --- the in-session device admission (a lineage no committed L40S record measured) ---------------------------------

#: per lineage, the device admission its formal session runs first (`t2507_device_admission.json` names why)
DEVICE_ADMISSION = {"thinking_2507": REPO / "configs/stages/stage-2/teacher_modes/t2507_device_admission.json"}
L40S_CHECK_CONFIG = REPO / "configs/stages/stage-2/teacher_modes/l40s_check.json"
VLLM_PINS = REPO / "requirements-vllm.txt"


class DeviceAdmissionRefused(RuntimeError):
    """The device admission measured a teacher, engine or memory margin that is not the admitted one."""


def snapshot_dir(model: str, revision: str) -> Path | None:
    """The pod's local snapshot of ``model`` at ``revision`` (setup downloaded it; offline lookup only)."""
    import os
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model, revision=revision, local_files_only=True))
    except Exception:  # noqa: BLE001 -- fall back to the cache layout
        home = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                    / "hub")
        d = home / f"models--{model.replace('/', '--')}" / "snapshots" / revision
        return d if d.is_dir() else None


def teacher_files(teacher: dict, snap: Path | None) -> dict:
    """Every registered file's and weight shard's sha256 (and each shard's size) in the snapshot, against the record."""
    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()
    if snap is None:
        return {"snapshot": None, "differ": ["no local snapshot of the teacher at its revision"]}
    want = {**teacher["files_sha256"], **teacher["weight_shards_sha256"]}
    got, differ = {}, []
    for f, w in sorted(want.items()):
        p = snap / f
        got[f] = sha(p) if p.is_file() else None
        if got[f] != w:
            differ.append(f)
    for f, n in sorted((teacher.get("weight_shards_bytes") or {}).items()):
        if (snap / f).is_file() and (snap / f).stat().st_size != n:
            differ.append(f"{f} (size)")
    return {"snapshot": str(snap), "sha256": got, "differ": differ}


def pinned_vllm(path: Path = VLLM_PINS) -> str | None:
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith("vllm=="):
            return line.strip().split("==", 1)[1]
    return None


def device_verdict(cfg: dict, proto: dict, lineage: str, mode: str, *, identity: dict, engine: list[dict],
                   gpu: dict | None, complete: bool, turns: int, log_text: str, files: dict, free_wanted_gib: float,
                   vllm_pin: str | None) -> dict:
    """Every admission criterion (`t2507_device_admission.json`) from the measured evidence alone; ``admitted`` is
    their conjunction. ``identity`` is the runner's recorded run identity (its summary), ``engine`` the engine
    telemetry rows (`tool_pilot.engine_telemetry`), ``gpu`` the GPU sampler's record."""
    t, e, li = cfg["teacher"], cfg["engine"], proto["lineages"][lineage]
    eng = identity.get("engine") or {}
    checks = {}

    def check(name, ok, measured, wanted):
        checks[name] = {"ok": bool(ok), "measured": measured, "wanted": wanted}
    check("teacher_model", identity.get("model") == t["model"] == li["model"], identity.get("model"), t["model"])
    check("teacher_revision", identity.get("revision") == t["revision"] == li["revision"], identity.get("revision"),
          t["revision"])
    check("teacher_files", not files.get("differ"), files.get("differ"), "every registered file and shard by sha256")
    check("chat_template", identity.get("chat_template_sha256") == t["chat_template_sha256"],
          identity.get("chat_template_sha256"), t["chat_template_sha256"])
    check("template_kwargs", (identity.get("chat_template_kwargs") or {}) == (li["modes"][mode]["template_kwargs"] or {}),
          identity.get("chat_template_kwargs"), li["modes"][mode]["template_kwargs"])
    check("sampling", identity.get("sampling") == li["modes"][mode]["sampling"], identity.get("sampling"),
          li["modes"][mode]["sampling"])
    check("stop_ids", sorted(identity.get("stop_ids") or []) == sorted(int(x) for x in proto["execution"]["stop_ids"]),
          identity.get("stop_ids"), proto["execution"]["stop_ids"])
    check("vllm_lock", eng.get("vllm_version") == vllm_pin == e["vllm_version"], eng.get("vllm_version"),
          {"requirements-vllm.txt": vllm_pin, "record": e["vllm_version"]})
    limit = int(proto["execution"]["session_limit"])
    lens = [identity.get("session_limit"), identity.get("effective_limit"), eng.get("max_model_len")]
    check("context", all(x == limit == e["session_limit"] for x in lens), lens, limit)
    check("gpu_memory_utilization", eng.get("gpu_memory_utilization") == e["gpu_memory_utilization"]
          == proto["execution"]["gpu_memory_utilization"], eng.get("gpu_memory_utilization"), e["gpu_memory_utilization"])
    check("dtype", eng.get("dtype") == e["dtype"], eng.get("dtype"), e["dtype"])
    started = [r for r in engine if r.get("engine_ready_seconds") and r.get("kv_cache_tokens")]
    kv = min((r["kv_cache_tokens"][0] for r in started), default=None)
    check("engine_started_with_a_kv_cache", bool(started) and kv is not None and kv >= limit, kv,
          f"an engine that starts and reports a KV cache of at least {limit} tokens")
    oom = "out of memory" in log_text.lower()
    check("no_out_of_memory", not oom, "CUDA out-of-memory in the engine log" if oom else None, None)
    check("generation_complete", bool(complete) and turns > 0, {"complete": complete, "turns": turns},
          "the invocation completes with generated turns")
    peaks = [(g.get("name"), g.get("memory_total_mib"), g.get("peak_memory_used_mib"))
             for g in ((gpu or {}).get("gpus") or {}).values()]
    free = [round((tot - peak) / 1024, 2) for _, tot, peak in peaks if tot and peak]
    check("memory_margin", bool(free) and min(free) >= free_wanted_gib and int((gpu or {}).get("samples") or 0) > 0,
          {"gpus": peaks, "free_gib": free, "samples": (gpu or {}).get("samples")}, f">= {free_wanted_gib} GiB free")
    return {"checks": checks, "admitted": all(c["ok"] for c in checks.values()),
            "failed": sorted(k for k, c in checks.items() if not c["ok"])}


def run_device_admission(proto: dict, plan: dict, lineage: str, by_id: dict, out: Path, identity: str, run,
                         mark) -> dict:
    """The lineage's device admission, before any formal invocation: the teacher snapshot's files by sha256, then ONE
    engine invocation (pass 0, the protocol's seed base) on the first task of every cell of the lineage's first mode,
    with GPU and engine telemetry, under ``<out>/device_admission`` -- device evidence only, never a formal shard --
    and `device_verdict` over what it measured. Writes ``<out>/device_admission.json``; raises
    `DeviceAdmissionRefused` when any criterion fails."""
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import tool_pilot as TPL
    cfg = json.loads(DEVICE_ADMISSION[lineage].read_text())
    li = proto["lineages"][lineage]
    if cfg["lineage"] != lineage or (cfg["teacher"]["model"], cfg["teacher"]["revision"]) != (li["model"], li["revision"]):
        raise DeviceAdmissionRefused(f"{DEVICE_ADMISSION[lineage].name} admits another teacher than the protocol's")
    mode = sorted(li["modes"])[0]
    adm = out / "device_admission"
    adm.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = teacher_files(cfg["teacher"], snapshot_dir(li["model"], li["revision"]))
    files_seconds = round(time.time() - t0, 1)
    ids = [c["task_ids"][0] for c in plan["lineages"][lineage]["modes"][mode]["cells"] if c.get("task_ids")]
    label = f"admission_{mode}"
    f = write_round([by_id[t] for t in ids], adm, label, proto["execution"]["system_message_default"])
    t1 = time.time()
    with TPL.GpuSampler() as gpu:
        try:
            g = GD.run_group(context(proto, lineage, identity), adm, adm / "inputs", f, 0, None, run, mark,
                             label=f"{label}_pass0", extra_args=mode_args(proto, lineage, mode) + ("--engine-metrics",))
        except Exception as exc:  # noqa: BLE001 -- measured as a failed invocation, with what the sampler saw
            g = {"complete": False, "records": {}, "segments": [{"segment": 0}], "error": repr(exc)[:2000]}
    recs = g.pop("records")
    row = _telemetry(out / "telemetry.jsonl", adm, f"{label}_pass0", g, gpu, time.time() - t1, recs)
    segs = [x["segment"] for x in g.get("segments") or [] if "segment" in x]
    summ = [adm / f"{label}_pass0.seg{n}.raw.summary.json" for n in segs]
    ident = next((json.loads(p.read_text()).get("identity") or {} for p in summ if p.is_file()), {})
    logs = "".join((adm / f"{label}_pass0.seg{n}.log").read_text(errors="ignore")
                   for n in segs if (adm / f"{label}_pass0.seg{n}.log").is_file())
    free_wanted = float(json.loads(L40S_CHECK_CONFIG.read_text())["free_memory_wanted_gib"])
    v = device_verdict(cfg, proto, lineage, mode, identity=ident, engine=row.get("engine") or [],
                       gpu=row.get("gpu"), complete=bool(g.get("complete")),
                       turns=sum(len(r.get("turns") or []) for r in recs.values()), log_text=logs, files=files,
                       free_wanted_gib=free_wanted, vllm_pin=pinned_vllm())
    rec = {"schema": "aadistill.stage2.teacher_modes.device_admission_result/v1", "lineage": lineage, "mode": mode,
           "config": {"path": str(DEVICE_ADMISSION[lineage].relative_to(REPO)), "sha256": GD.sha(DEVICE_ADMISSION[lineage])},
           "tasks": ids, "teacher_files": files, "teacher_files_seconds": files_seconds,
           "invocation": {"label": f"{label}_pass0", "complete": g.get("complete"), "error": g.get("error"),
                          "wall_seconds": row.get("wall_seconds"),
                          "engine": row.get("engine"), "gpu": row.get("gpu"), "turns": row.get("turns")},
           "runner_identity": ident, **v,
           "outputs": "device evidence only: never sealed under a formal label, imported, admitted, counted or packed"}
    (out / "device_admission.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    mark("DEVICE_ADMISSION_OK" if v["admitted"] else "DEVICE_ADMISSION_FAILED")
    if not v["admitted"]:
        raise DeviceAdmissionRefused(f"the device admission failed: {v['failed']}")
    return rec


class _Done(Exception):
    def __init__(self, ok: bool):
        super().__init__(ok)
        self.ok = ok


def run_pilot(proto: dict, plan_path: Path, tasks_path: Path, out: Path, identity: str, run, mark,
              record: dict, *, mount: Path, volume_gb: float | None, resume_from: Path | None = None) -> bool:
    """The tool pilot (`tool_pilot.py`): its frozen plan and task file verified (`tool_pilot.load`) and its signed,
    pilot-only source admission naming this plan; then each mode's frozen cells in ONE round -- first candidates,
    second candidates for the refused tasks only -- with GPU and engine telemetry beside the sealed shards. A
    replacement session regenerates nothing sealed (`generation_driver.run_group`): on a new pod, ``resume_from`` is
    the dev box's copy of the earlier sessions' sealed shards, copied in first and re-verified like the pod's own."""
    from stages.teacher_modes import tool_pilot as TPL
    doc, by_id = TPL.load(plan_path, tasks_path)
    adm = json.loads(TPL.ADMISSION.read_text())
    bad = TPL.admission_problems(adm, plan_path, tasks_path)
    if bad:
        raise TPL.PilotError("the pilot-only source admission refuses: " + "; ".join(bad))
    if resume_from is not None:
        record["resumed_from"] = bring_sealed(Path(resume_from), out)
    bad, record["storage"] = TPL.storage_problems(out, mount, volume_gb)
    if bad:
        raise TPL.PilotError("the volume refuses: " + "; ".join(bad))
    pid = TPL.pilot_identity(identity, plan_path)
    record.update(pilot_plan_sha256=adm["pilot_plan"]["sha256"], tasks=len(by_id),
                  verification_version=TE.VERIFICATION_VERSION, study_identity=identity, identity=pid["sha256"],
                  identity_parts=pid["parts"])
    for mode in sorted(doc["modes"]):
        resumable(out, mode, doc, by_id, pid["sha256"])
    mark("CONTRACT_OK")
    record["modes"] = {}
    for mode in sorted(doc["modes"]):
        cells, orders = TPL.cells(doc, mode, by_id)
        res = generate_mode(proto, None, doc["lineage"], mode, by_id, None, out, pid["sha256"], run, mark, cells=cells,
                            orders=orders, max_rounds=doc["rounds"], label_prefix="pilot",
                            telemetry=out / "telemetry.jsonl", extra_args=("--engine-metrics",))
        record["modes"][mode] = res
        if not res["complete"]:
            return False
    return True


def bring_sealed(src: Path, out: Path, prefix: str = "pilot") -> dict:
    """Copy an earlier session's sealed pilot shards and their markers into ``out``; a file already there must be
    byte-identical (a shard is never overwritten). What they hold is checked by `resumable` and
    `generation_driver.sealed` exactly as for shards this pod sealed itself."""
    import shutil
    from stages.teacher_modes import tool_pilot as TPL
    bad, _ = TPL.resume_problems(src, prefix=prefix)
    if bad:
        raise TPL.PilotError("the resume source refuses: " + "; ".join(bad))
    out.mkdir(parents=True, exist_ok=True)
    copied = kept = 0
    for f in sorted(src.glob(f"{prefix}_*_r*_pass*.part*.*")):
        if f.suffix not in (".jsonl", ".done"):
            continue
        dest = out / f.name
        if dest.exists():
            if dest.read_bytes() != f.read_bytes():
                raise TPL.PilotError(f"{f.name} differs from the shard already in {out}: nothing sealed is overwritten")
            kept += 1
            continue
        shutil.copy2(f, dest)
        copied += 1
    return {"path": str(src), "copied": copied, "already_present": kept}


def resumable(out: Path, mode: str, doc: dict, by_id: dict, identity: str) -> None:
    """Before a pilot resumes: every episode already sealed for ``mode`` must be this pilot's -- sealed under its pilot
    identity (plan, protocol, verification version, implementation; `generation_driver.sealed` refuses another), from
    exactly this round's input (its sha256), under this protocol file and this runner. Anything else refuses
    (`tool_pilot.PilotError`): an earlier plan's or generation's episodes are never reused."""
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import tool_pilot as TPL
    body = round_body([by_id[t] for c in doc["modes"][mode] for t in c["task_ids"]])
    want = {"group_input_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "protocol_sha256": GD.sha(PROTOCOL), "runner_sha256": GD.sha(REPO / GD.RUNNER)}
    for k in range(int(protocol()["execution"]["candidates"])):
        for tid, rec in GD.sealed(out, f"pilot_{mode}_r1_pass{k}", identity).items():
            got = {f: (rec.get("generation_pass") or {}).get(f) for f in want}
            if got != want:
                raise TPL.PilotError(f"{tid} (pilot_{mode}_r1_pass{k}) was sealed from another input, protocol or "
                                     f"runner: {sorted(f for f in want if got[f] != want[f])}")


def main(argv=None, *, run=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--lineage", required=True, choices=("hybrid", "thinking_2507"))
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--pilot", default=None, help="the frozen pilot plan (tool_pilot.py): runs the tool_pilot kind")
    ap.add_argument("--pilot-tasks", default=None, help="the pilot's task file, by the plan's sha256")
    ap.add_argument("--volume-gb", type=float, default=None, help="a network volume's size, as the provider reports "
                    "it (its free space is checked before generating); none: the container disk's own free space")
    ap.add_argument("--resume-from", default=None, help="the dev box's receipted copy of earlier sessions' sealed "
                    "shards, staged on this pod")
    a = ap.parse_args(argv)
    if bool(a.pilot) != bool(a.pilot_tasks):
        ap.error("--pilot and --pilot-tasks go together")
    if a.pilot and a.lineage != "hybrid":
        ap.error("the pilot is the hybrid's")
    durable, status = Path(a.durable_root), Path(a.status)
    out = durable / (f"tool_pilot_{a.lineage}" if a.pilot else f"tool_generation_{a.lineage}")
    out.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "lineage": a.lineage, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    ok = True
    try:
        from stages.reasoning_data import generation_driver as GD
        from stages.teacher_modes import token_matched_phase as TM
        #: the lineage's own kind: a hybrid artifact never runs Thinking-2507, nor the reverse; a pilot's never runs
        #: the bounded generation, nor the reverse
        kind = TM.TOOL_PILOT_KIND if a.pilot else TM.TOOL_GENERATION_KINDS[a.lineage]
        bound = TM.require_on_pod(Path(a.authorization), kind, REPO)
        record.update(kind=kind, authorization=bound.get("authorization_id"), identity=bound["identity"])
        proto = protocol()
        if a.pilot:
            ok = run_pilot(proto, Path(a.pilot), Path(a.pilot_tasks), out, bound["identity"], run or GD.run_battery,
                           mark, record, mount=durable.parent, volume_gb=a.volume_gb,
                           resume_from=Path(a.resume_from) if a.resume_from else None)
            raise _Done(ok)
        plan, man, by_id = checked_plan(proto, a.lineage)
        record.update(plan_pool_content_sha256=plan["pool"]["content_sha256"], tasks_readable=len(by_id))
        if a.resume_from:
            record["resumed_from"] = bring_sealed(Path(a.resume_from), out, prefix="tool")
        if plan.get("import"):                  #: the cells carry the imported outcomes (the plan binds the record)
            record["imported_from"] = plan["import"]
        mark("CONTRACT_OK")
        if a.lineage in DEVICE_ADMISSION:      #: no committed L40S record measured this teacher: measure it first
            adm = run_device_admission(proto, plan, a.lineage, by_id, out, bound["identity"], run or GD.run_battery,
                                       mark)
            record["device_admission"] = {k: adm[k] for k in ("admitted", "failed", "tasks")}
        record["modes"] = {}
        for mode in sorted(proto["lineages"][a.lineage]["modes"]):
            res = generate_mode(proto, plan, a.lineage, mode, by_id, man, out, bound["identity"],
                                run or GD.run_battery, mark, telemetry=out / "telemetry.jsonl",
                                extra_args=("--engine-metrics",))
            record["modes"][mode] = res
            if not res["complete"]:
                ok = False
                break
    except _Done as done:
        ok = done.ok
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if ok else "RUN_FAILED")
    (out / "tool_session.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

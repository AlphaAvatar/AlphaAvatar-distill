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
    """``(plan, pool manifest, tasks by id)``: `tool_generation_plan.verified` for the lineage -- its split files
    re-verified, the plan bound to this pool's content, pool.json, protocol and design, its cells unchanged."""
    return TGP.verified(proto, lineage=lineage)


def write_round(tasks: list[dict], out: Path, label: str, default_system: str) -> dict:
    """The round's run_battery input (each task keeps its own ``system``); an existing file must be identical."""
    d = out / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{label}.jsonl"
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in tasks)
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
                  extra_args: tuple[str, ...] = ()) -> dict:
    """Every cell of one mode, round by round, to its target or its bound; returns the cells' final states. A pilot
    passes its frozen ``cells`` and ``orders`` (`tool_pilot.cells`), one round and its own label prefix; with
    ``telemetry`` every engine invocation appends its GPU and engine telemetry there (`tool_pilot`)."""
    from stages.reasoning_data import generation_driver as GD
    bp, ex = proto["bounded_plan"], proto["execution"]
    cand, stop_ids = int(ex["candidates"]), [int(x) for x in ex["stop_ids"]]
    thinking = bool(proto["lineages"][lineage]["modes"][mode]["template_kwargs"].get("enable_thinking", True))
    if cells is None:
        cells = plan["lineages"][lineage]["modes"][mode]["cells"]
        orders = [TGP.cell_order(man["orders"].get(c["split"], []), by_id, c["category"]) for c in cells]
    state = [{"attempted": 0, "units": 0.0, "admitted": 0} for _ in cells]
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
            state[i]["attempted"] += len(p)
            state[i]["units"] += sum(admitted.get(t, 0.0) for t in p)
            state[i]["admitted"] += sum(t in admitted for t in p)
        mark(f"ROUND_DONE:{label}")
    return {"complete": True, "groups": groups, "cells": _report(cells, state)}


def _gpu(telemetry: Path | None):
    import contextlib
    from stages.teacher_modes import tool_pilot as TPL
    return TPL.GpuSampler() if telemetry is not None else contextlib.nullcontext()


def _telemetry(path: Path, out: Path, label: str, g: dict, gpu, seconds: float, recs: dict) -> None:
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


def _report(cells: list[dict], state: list[dict]) -> list[dict]:
    out = []
    for cell, s in zip(cells, state):
        done = s["units"] >= cell["target"] and s["admitted"] >= cell.get("min_tasks", 0)
        out.append({k: cell[k] for k in ("split", "category", "unit", "target", "bound")} |
                   {**s, "status": "REACHED" if done else "SHORTFALL"})
    return out


class _Done(Exception):
    def __init__(self, ok: bool):
        super().__init__(ok)
        self.ok = ok


def run_pilot(proto: dict, plan_path: Path, tasks_path: Path, out: Path, identity: str, run, mark,
              record: dict) -> bool:
    """The tool pilot (`tool_pilot.py`): its frozen plan and task file verified (`tool_pilot.load`) and its signed,
    pilot-only source admission naming this plan; then each mode's frozen cells in ONE round -- first candidates,
    second candidates for the refused tasks only -- with GPU and engine telemetry beside the sealed shards. A
    replacement session regenerates nothing sealed (`generation_driver.run_group`)."""
    from stages.teacher_modes import tool_pilot as TPL
    doc, by_id = TPL.load(plan_path, tasks_path)
    adm = json.loads(TPL.ADMISSION.read_text())
    if adm.get("schema") != TPL.ADMISSION_SCHEMA or not (adm.get("signature") or {}).get("signed_by"):
        raise TPL.PilotError("the pilot-only source admission is not signed")
    if adm["pilot_plan"]["sha256"] != hashlib.sha256(Path(plan_path).read_bytes()).hexdigest():
        raise TPL.PilotError("the source admission names another pilot plan")
    record.update(pilot_plan_sha256=adm["pilot_plan"]["sha256"], tasks=len(by_id),
                  verification_version=TE.VERIFICATION_VERSION)
    mark("CONTRACT_OK")
    record["modes"] = {}
    for mode in sorted(doc["modes"]):
        cells, orders = TPL.cells(doc, mode, by_id)
        res = generate_mode(proto, None, doc["lineage"], mode, by_id, None, out, identity, run, mark, cells=cells,
                            orders=orders, max_rounds=doc["rounds"], label_prefix="pilot",
                            telemetry=out / "telemetry.jsonl", extra_args=("--engine-metrics",))
        record["modes"][mode] = res
        if not res["complete"]:
            return False
    return True


def main(argv=None, *, run=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--lineage", required=True, choices=("hybrid", "thinking_2507"))
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--pilot", default=None, help="the frozen pilot plan (tool_pilot.py): runs the tool_pilot kind")
    ap.add_argument("--pilot-tasks", default=None, help="the pilot's task file, by the plan's sha256")
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
                           mark, record)
            raise _Done(ok)
        plan, man, by_id = checked_plan(proto, a.lineage)
        record.update(plan_pool_content_sha256=plan["pool"]["content_sha256"], tasks_readable=len(by_id))
        mark("CONTRACT_OK")
        record["modes"] = {}
        for mode in sorted(proto["lineages"][a.lineage]["modes"]):
            res = generate_mode(proto, plan, a.lineage, mode, by_id, man, out, bound["identity"],
                                run or GD.run_battery, mark)
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

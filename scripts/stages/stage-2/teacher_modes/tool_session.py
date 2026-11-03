#!/usr/bin/env python3
"""On-pod driver of the teacher-native tool-calling generation (Research A packed round), one lineage per session.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/tool_session.py --authorization <a.json> \\
        --lineage hybrid --durable-root <volume>/teacher_modes --status <file>

`tool_generation_protocol.json` names the lineage's teacher, modes, pool splits and execution terms. Per mode, the
lineage's pool tasks (its splits; the training split in the pool's frozen order) go to `run_battery` in generate
mode, grouped by their system text, with the tool environment (``--env-factory``): each turn the official template
renders the conversation and the tools, the replay backend answers the parsed calls, an off-script call ends the
episode. `generation_driver.run_group` seals every finished episode into identity-bound shards as it ends, so a
replacement session resumes without regenerating. Pass 1 runs only for tasks whose pass-0 episode fails
`tool_episodes.verify`. Nothing here admits data: the dev box does (`tool_admission.py`) from the sealed shards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import tool_episodes as TE  # noqa: E402

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
SCHEMA = "aadistill.stage2.teacher_modes.tool_session/v1"


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def lineage_tasks(proto: dict, lineage: str, pool_dir: Path) -> list[dict]:
    """The lineage's tasks: the training split in the pool's frozen order, then the other splits by task id."""
    pool = json.loads((pool_dir / "pool.json").read_text())
    order = {t: i for i, t in enumerate(pool["training_order"])}
    out = []
    for split in proto["lineages"][lineage]["splits"]:
        path = pool_dir / f"{split}.tasks.jsonl"
        if not path.is_file():
            continue
        rows = TE.read_tasks(path)
        rows.sort(key=lambda t: (order.get(t["task_id"], len(order)), t["task_id"]))
        out += rows
    bad = {t["task_id"]: TE.task_problems(t) for t in out if TE.task_problems(t)}
    if bad:
        raise TE.ToolTaskError(f"{len(bad)} pool tasks the protocol cannot run, e.g. {next(iter(bad.items()))}")
    return out


def write_inputs(tasks: list[dict], out: Path, label: str, default_system: str) -> list[dict]:
    """One run_battery input file per system text, deterministic; an existing file must be identical."""
    import hashlib
    by_system = defaultdict(list)
    for t in tasks:
        by_system[t.get("system") or default_system].append(t)
    d = out / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (system, rows) in enumerate(sorted(by_system.items())):
        path = d / f"{label}_g{k}.jsonl"
        body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        if path.is_file() and path.read_text() != body:
            raise RuntimeError(f"{path.name} on the volume is not this session's input")
        path.write_text(body)
        files.append({"file": path.name, "system": system, "tasks": len(rows),
                      "sha256": hashlib.sha256(body.encode()).hexdigest()})
    return files


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


def retry_ids(tasks: dict[str, dict], records: dict[str, dict], *, thinking: bool) -> list[str]:
    """Task ids whose first episode fails the five checks (the second candidate's only reason to run)."""
    return sorted(tid for tid, rec in records.items()
                  if not TE.verify(tasks[tid], rec.get("turns") or [], thinking=thinking).admitted)


def generate(proto: dict, lineage: str, tasks: list[dict], out: Path, identity: str, run, mark) -> dict:
    from stages.reasoning_data import generation_driver as GD
    c = context(proto, lineage, identity)
    by_id = {t["task_id"]: t for t in tasks}
    got = {"groups": [], "complete": True}
    for mode, spec in proto["lineages"][lineage]["modes"].items():
        thinking = bool(spec["template_kwargs"].get("enable_thinking", True))
        files = write_inputs(tasks, out, f"tool_{mode}", proto["execution"]["system_message_default"])
        for f in files:
            ids = None
            for k in range(int(proto["execution"]["candidates"])):
                if k and not ids:
                    break
                label = f"tool_{mode}_pass{k}_{Path(f['file']).stem}"
                g = GD.run_group(c, out, out / "inputs", f, k, ids, run, mark, label=label,
                                 extra_args=mode_args(proto, lineage, mode))
                recs = g.pop("records")
                got["groups"].append(g)
                if not g["complete"]:
                    got["complete"] = False
                    mark(f"{GD.UNIT_FAILED}:{label}")
                    break
                mark(f"{GD.UNIT_OK}:{label}")
                if k == 0:
                    ids = retry_ids(by_id, recs, thinking=thinking)
    return got


def main(argv=None, *, run=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--lineage", required=True, choices=("hybrid", "thinking_2507"))
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    a = ap.parse_args(argv)
    durable, status = Path(a.durable_root), Path(a.status)
    out = durable / f"tool_generation_{a.lineage}"
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
        bound = TM.require_on_pod(Path(a.authorization), "tool_generation", REPO)
        record["authorization"] = bound.get("authorization_id")
        proto = protocol()
        tasks = lineage_tasks(proto, a.lineage, REPO / proto["pool"]["dir"])
        record["tasks"] = len(tasks)
        mark("CONTRACT_OK")
        res = generate(proto, a.lineage, tasks, out, bound["identity"], run or GD.run_battery, mark)
        record.update(groups=res["groups"], complete=res["complete"])
        ok = res["complete"]
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

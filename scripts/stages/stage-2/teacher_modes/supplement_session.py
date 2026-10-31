#!/usr/bin/env python3
"""On-pod driver of Research A v2's supplementary generation: order_v2's cells, by rounds, until each (mode, cell) need is met.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/supplement_session.py --authorization <authorization.json> \\
        --durable-root <volume>/teacher_modes --status <file>

Every value the runner receives is the frozen v1 generation protocol's (`generation_protocol.json`: modes, sampling,
stop ids, the 8,184-token allowance, two candidates except MBPP), except the seed base, which
``supplement_protocol.json`` replaces (a new task's seeds are its own). The need of each (mode, cell) is
``supplement_plan.json``'s, computed on the dev box from the sealed v1 records with every verifier.

ROUNDS. Round r takes, per mode and per cell whose admitted supervised tokens -- the pod's verdict of THIS
session's records, `pod_verdict` -- are still below need x (1 + pod_margin), the cell's next tasks in
``cell_position`` order (families the plan skips for that mode left out), as many as the remaining tokens over
the cell's tokens per generated task (measured in this session once ``MEASURED_MIN`` tasks of the (mode, cell)
are generated, the plan's figure before) times ``sizing``. Round 1 also generates V2 (validation) in both modes.
Each round is one stage per mode: pass 0, then pass 1 for each first candidate the pod rejects. A round's
selection is a function of the plan and of the earlier rounds' sealed records only, so a resumed session
recomputes the same selections and the shared group driver regenerates nothing already sealed. The session ends
when no (mode, cell) is short and has tasks left, or after ``max_rounds``; a cell whose queue ends short is
recorded EXHAUSTED for the dev box's admission to report, never filled from another cell.

The pod's verdict bounds the dev box's: the dev box's admission adds the isolation recheck (R1, R2, R9) and is
the only selection. Generation writes TEXT and runs no teacher program.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import generation_driver as GD  # noqa: E402
from stages.teacher_modes import teacher_session as TS  # noqa: E402

SUPPLEMENT_PROTOCOL = Path("configs/stages/stage-2/teacher_modes/supplement_protocol.json")
MODES = TS.MODES
SCHEMA = "aadistill.stage2.teacher_modes.supplement_session/v1"
MEASURED_MIN = 30


def authorize(path: Path, repo: Path) -> dict:
    from stages.teacher_modes import token_matched_phase as TM
    return TM.require_on_pod(path, "supplement", repo)


def contract(repo: Path, authorization: Path, authorize=authorize) -> dict:
    bound = authorize(authorization, repo)
    sp = json.loads((repo / SUPPLEMENT_PROTOCOL).read_text())
    for ref in [sp["base_protocol"], *sp["inputs"].values()]:
        if GD.sha(repo / ref["path"]) != ref["sha256"]:
            raise GD.ShardError(f"{ref['path']}: not the input the supplement protocol binds")
    proto = copy.deepcopy(json.loads((repo / sp["base_protocol"]["path"]).read_text()))
    proto["execution"]["seed_base"] = int(sp["seed_base"])
    from stages.teacher_modes.build_supplement_tasks import read_jsonl
    order = list(read_jsonl(repo / sp["inputs"]["order"]["path"]))
    plan = json.loads((repo / sp["inputs"]["plan"]["path"]).read_text())
    cal = list(A.jsonl(repo / sp["inputs"]["calibration"]["path"]))
    tasks = {t["task_id"]: t for t in order}
    #: a calibration row is run_battery's own input (teacher_session.write_inputs), the protocol's system message
    tasks.update({t["task_id"]: {k: v for k, v in t.items() if k not in ("modes", "system", "source")} for t in cal})
    return {**bound, "protocol": proto, "protocol_path": SUPPLEMENT_PROTOCOL, "supplement": sp, "plan": plan,
            "order": order, "calibration": cal, "tasks": tasks, "launch_identity_sha256": bound["identity"]}


def family_of(t: dict) -> str:
    return f"{t['generator_id']}/{t.get('family_id')}" if t.get("generator_id") else (t.get("family_id") or t["source"])


def queues(c: dict) -> tuple[dict[str, list[dict]], list[str]]:
    q: dict[str, list[dict]] = defaultdict(list)
    v2 = []
    for t in sorted(c["order"], key=lambda t: (t["cell"], t["cell_position"])):
        (v2.append(t["task_id"]) if t["slice"] == "V2" else q[t["cell"]].append(t))
    return dict(q), v2


def pod_verdict(c: dict, tid: str, mode: str, recs: list[dict]) -> dict:
    """The lowest candidate the pod accepts, with its supervised tokens (generated ids of every turn's document)."""
    from aadistill.data.token_documents import DocumentError, document_from_turn, prompt_mode_problems
    from stages.reasoning_data.generation_adapter import candidate
    task = c["tasks"][tid]
    stop = c["protocol"]["teacher"]["stop_ids"]
    suffix = c["protocol"]["modes"][mode]["generation_suffix_ids"]
    for rec in recs:
        try:
            docs = [document_from_turn(t, doc_id=f"{tid}|{mode}|t{k}", stop_ids=stop) for k, t in enumerate(rec["turns"])]
        except DocumentError:
            continue
        if any(prompt_mode_problems(d, suffix) for d in docs) or \
                not candidate(task, rec, rec["candidate_index"]).accepted_for_training:
            continue
        return {"admitted": True, "supervised_tokens": sum(len(d.output_ids) for d in docs)}
    return {"admitted": False, "supervised_tokens": 0}


class State:
    """Per (mode, cell): queue position, generated tasks, the pod's admitted supervised tokens."""

    def __init__(self, c: dict):
        self.c, (self.q, self.v2) = c, queues(c)
        sp, plan = c["supplement"], c["plan"]
        self.margin, self.sizing, self.min_batch = float(sp["pod_margin"]), float(sp["sizing"]), int(sp["min_batch"])
        self.skip = {m: set(plan["skip"][m]) for m in MODES}
        self.pos, self.gen, self.got = defaultdict(int), defaultdict(int), defaultdict(float)
        self.exhausted: set[tuple[str, str]] = set()

    def eligible(self, mode: str, cell: str) -> list[dict]:
        return [t for t in self.q.get(cell, []) if f"{t['stratum']}/{t['source']}/{family_of(t)}" not in self.skip[mode]]

    def short(self, mode: str, cell: str) -> float:
        need = float(self.c["plan"]["cells"][cell]["modes"][mode]["need"])
        return need * (1 + self.margin) - self.got[(mode, cell)] if need > 0 else 0.0

    def select(self, mode: str, r: int) -> tuple[list[str], dict]:
        ids, why = (list(self.v2) if r == 1 else []), {}
        for cell in sorted(self.c["plan"]["cells"]):
            rest = self.short(mode, cell)
            if rest <= 0:
                continue
            q, p = self.eligible(mode, cell), self.pos[(mode, cell)]
            if p >= len(q):
                self.exhausted.add((mode, cell))
                continue
            plan = self.c["plan"]["cells"][cell]["modes"][mode]
            n = self.gen[(mode, cell)]
            tpt = self.got[(mode, cell)] / n if n >= MEASURED_MIN else float(plan["tokens_per_generated_task"])
            k = max(self.min_batch, math.ceil(rest / max(tpt, 1.0) * self.sizing))
            take = q[p:p + k]
            self.pos[(mode, cell)] += len(take)
            ids += [t["task_id"] for t in take]
            why[cell] = {"short_tokens": round(rest, 1), "tokens_per_task": round(tpt, 2), "tasks": len(take)}
        return ids, why

    def update(self, mode: str, ids: list[str], recs: dict[str, list[dict]]) -> None:
        for tid in ids:
            t = self.c["tasks"][tid]
            if t["slice"] == "V2":
                continue
            v = pod_verdict(self.c, tid, mode, recs.get(tid, []))
            self.gen[(mode, t["cell"])] += 1
            self.got[(mode, t["cell"])] += v["supervised_tokens"]

    def table(self) -> dict:
        return {f"{m}|{cell}": {"need": self.c["plan"]["cells"][cell]["modes"][m]["need"],
                                "pod_admitted_tokens": round(self.got[(m, cell)], 1), "generated_tasks": self.gen[(m, cell)],
                                "queue_position": self.pos[(m, cell)], "exhausted": (m, cell) in self.exhausted}
                for cell in sorted(self.c["plan"]["cells"]) for m in MODES}


def main(argv=None, *, run=GD.run_battery, authorize=authorize, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    args = ap.parse_args(argv)
    repo = Path(repo or REPO)
    root, status = Path(args.durable_root), Path(args.status)
    out = root / "generation"
    out.mkdir(parents=True, exist_ok=True)

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rounds": []}
    evidence = root / "supplement_session.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1, default=str) + "\n")

    try:
        c = contract(repo, Path(args.authorization), authorize)
        record["contract"] = {k: c[k] for k in ("authorization_id", "launch_identity_sha256", "harness_digest") if k in c}
    except BaseException as exc:  # noqa: BLE001
        record["contract"] = {"status": "failed", "error": repr(exc)}
        save()
        mark("RUN_FAILED")
        return 2
    import shutil
    keep = root / "inputs" / Path(c["supplement"]["inputs"]["order"]["path"]).name
    if not keep.is_file():                  #: the heavy input lives outside git: keep a copy beside its records
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / c["supplement"]["inputs"]["order"]["path"], keep)
    save()
    mark("CONTRACT_OK")
    st, complete = State(c), True
    n_cand = int(c["protocol"]["execution"]["candidates"])
    try:
        #: round 0: the calibration supplement, each unit in the modes it is needed for, one candidate (v1's rule)
        row = {"round": 0, "modes": {}}
        for mode in MODES:
            ids = [t["task_id"] for t in c["calibration"] if mode in t["modes"]]
            if not ids:
                continue
            t0 = time.time()
            stage = {"stage": f"s00{'t' if mode == 'H-T' else 'n'}", "slice": "calibration", "modes": (mode,), "ids": ids}
            got = TS.run_stage(c, out, stage, run, mark, candidates=1)
            row["modes"][mode] = {"stage": stage["stage"], "tasks": len(ids), "complete": got["complete"],
                                  "seconds": round(time.time() - t0, 1)}
            if not got["complete"]:
                complete = False
                break
            mark(f"ROUND_OK:{stage['stage']}")
        record["rounds"].append(row)
        save()
        for r in range(1, int(c["supplement"]["max_rounds"]) + 1):
            if not complete:
                break
            row = {"round": r, "modes": {}}
            any_ids = False
            for mode in MODES:
                ids, why = st.select(mode, r)
                if not ids:
                    continue
                any_ids = True
                t0 = time.time()
                stage = {"stage": f"s{r:02d}{'t' if mode == 'H-T' else 'n'}", "slice": "S", "modes": (mode,), "ids": ids}
                got = TS.run_stage(c, out, stage, run, mark, candidates=n_cand)
                row["modes"][mode] = {"stage": stage["stage"], "tasks": len(ids), "cells": why,
                                      "complete": got["complete"], "seconds": round(time.time() - t0, 1),
                                      "generated_tokens": sum(int(t.get("generated_tokens") or 0)
                                                              for rs in got["records"].get(mode, {}).values()
                                                              for rr in rs for t in rr.get("turns", []))}
                if not got["complete"]:
                    complete = False
                    break
                st.update(mode, ids, got["records"].get(mode, {}))
                mark(f"ROUND_OK:{stage['stage']}")
            row["state"] = st.table()
            record["rounds"].append(row)
            save()
            if not complete or not any_ids:
                break
        record["final"] = st.table()
        record["exhausted"] = sorted(f"{m}|{cell}" for m, cell in st.exhausted)
        record["short_after"] = sorted(f"{m}|{cell}" for cell in c["plan"]["cells"] for m in MODES
                                       if st.short(m, cell) > 0)
    except BaseException as exc:  # noqa: BLE001
        complete = False
        record["error"] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if complete else "RUNS_INCOMPLETE")
    save()
    mark("ALL_DONE" if complete else "RUN_FAILED")
    return 0 if complete else 3


if __name__ == "__main__":
    sys.exit(main())

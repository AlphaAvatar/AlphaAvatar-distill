#!/usr/bin/env python3
"""On-pod driver of the hybrid full search's inputs (Research A v2, A1 phase A): the calibration top-up and the
state evaluation's native generation, both modes.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/inputs_session.py --authorization <authorization.json> \\
        --durable-root <volume>/teacher_modes --status <file>

Every value the runner receives is the frozen v1 generation protocol's (`generation_protocol.json`: modes, each
mode's official template switch and sampling, stop ids, the 8,184-token allowance), except the seed base, which
``inputs_protocol.json`` replaces. One candidate per (task, mode): calibration and state evaluation keep a mode's
own naturally terminated trajectories, and correctness is not a criterion for either.

ROUND 0 (``c00t``, ``c00n``): the calibration top-up (calibration_topup_v1.jsonl), each unit in the modes it is listed
for. The dev box's profile build takes the cells' units in D4's order; the list already holds each short cell's
reserve, so nothing here depends on how the round turned out.

STATE-EVALUATION ROUNDS (``e01t``, ``e01n``, ...): per mode and type, while the (type, mode) holds fewer than
``positions`` scored positions or fewer than ``min_tasks`` naturally terminated tasks -- counted exactly as the
dev box's freeze counts them (`terminated_positions`) -- the type's next prompts in queue order, as many as the
remaining positions over the measured positions per generated task (the planning yield until ``MEASURED_MIN`` tasks
of the (type, mode) are generated) times ``sizing``, at least what the remaining task count needs, at least
``min_batch``. Both modes read the same queue from its start. A round's selection is a function of the inputs and
of the earlier rounds' sealed records only, so a resumed session recomputes it and the group driver regenerates
nothing already sealed. The session ends when no (type, mode) is short and has prompts left, or after
``max_rounds``; a queue that ends short is recorded EXHAUSTED for the dev box to report, never filled from another
type or mode. Generation writes TEXT and runs no teacher program.
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

INPUTS_PROTOCOL = Path("configs/stages/stage-2/teacher_modes/inputs_protocol.json")
MODES = TS.MODES
SCHEMA = "aadistill.stage2.teacher_modes.inputs_session/v1"
MEASURED_MIN = 8


def authorize(path: Path, repo: Path) -> dict:
    from stages.teacher_modes import token_matched_phase as TM
    return TM.require_on_pod(path, "inputs", repo)


def contract(repo: Path, authorization: Path, authorize=authorize) -> dict:
    bound = authorize(authorization, repo)
    ip = json.loads((repo / INPUTS_PROTOCOL).read_text())
    for ref in [ip["base_protocol"], *ip["inputs"].values()]:
        if GD.sha(repo / ref["path"]) != ref["sha256"]:
            raise GD.ShardError(f"{ref['path']}: not the input the inputs protocol binds")
    proto = copy.deepcopy(json.loads((repo / ip["base_protocol"]["path"]).read_text()))
    proto["execution"]["seed_base"] = int(ip["seed_base"])
    cal = list(A.jsonl(repo / ip["inputs"]["calibration_topup"]["path"]))
    se = list(A.jsonl(repo / ip["inputs"]["state_evaluation"]["path"]))
    tasks = {}
    for t in cal:                            #: a calibration unit is run_battery's own input row (teacher_session)
        tasks[t["task_id"]] = ({k: v for k, v in t.items() if k not in ("modes", "system", "source")}
                               if "calibration" in t else {k: v for k, v in t.items() if k != "modes"})
    for t in se:
        if t["task_id"] in tasks:
            raise GD.ShardError(f"{t['task_id']}: in both the calibration top-up and the state evaluation")
        tasks[t["task_id"]] = t
    return {**bound, "protocol": proto, "protocol_path": INPUTS_PROTOCOL, "inputs_protocol": ip,
            "calibration": cal, "state_evaluation": se, "tasks": tasks, "launch_identity_sha256": bound["identity"]}


def terminated_positions(rec: dict, stop_ids) -> int | None:
    """Scored positions of a naturally terminated one-turn trajectory (its generated tokens, the stop token
    included), None otherwise -- the freeze's rule (`freeze_state_eval.item_of`)."""
    turns = rec.get("turns") or []
    if len(turns) != 1:
        return None
    t = turns[0]
    o = t.get("output_token_ids") or []
    if t.get("stop_reason") != "eos" or not o or int(o[-1]) not in {int(x) for x in stop_ids}:
        return None
    return len(o)


class Queues:
    """Per (type, mode): queue position, generated tasks, naturally terminated tasks and their scored positions."""

    def __init__(self, c: dict):
        self.c, rules = c, c["inputs_protocol"]["state_evaluation"]
        self.q: dict[str, list[dict]] = defaultdict(list)
        for t in sorted(c["state_evaluation"], key=lambda t: (t["subtype"], int(t["queue_position"]))):
            self.q[t["subtype"]].append(t)
        self.positions, self.min_tasks = int(rules["positions"]), int(rules["min_tasks"])
        self.sizing, self.min_batch = float(rules["sizing"]), int(rules["min_batch"])
        self.yields = {tuple(k.split("|")): v for k, v in rules["planning_yield"].items()}
        self.pos, self.gen = defaultdict(int), defaultdict(int)
        self.got, self.done = defaultdict(int), defaultdict(int)
        self.exhausted: set[tuple[str, str]] = set()

    def short(self, sub: str, mode: str) -> tuple[int, int]:
        return max(0, self.positions - self.got[(sub, mode)]), max(0, self.min_tasks - self.done[(sub, mode)])

    def select(self, mode: str) -> tuple[list[str], dict]:
        ids, why = [], {}
        for sub in sorted(self.q):
            rest_pos, rest_tasks = self.short(sub, mode)
            if not (rest_pos or rest_tasks):
                continue
            q, p = self.q[sub], self.pos[(sub, mode)]
            if p >= len(q):
                self.exhausted.add((sub, mode))
                continue
            n = self.gen[(sub, mode)]
            nat, per = self.yields[(sub, mode)]
            ppt = self.got[(sub, mode)] / n if n >= MEASURED_MIN else nat * per
            tpt = self.done[(sub, mode)] / n if n >= MEASURED_MIN else nat
            k = max(self.min_batch, math.ceil(max(rest_pos / max(ppt, 1.0), rest_tasks / max(tpt, 0.05)) * self.sizing))
            take = q[p:p + k]
            self.pos[(sub, mode)] += len(take)
            ids += [t["task_id"] for t in take]
            why[sub] = {"short_positions": rest_pos, "short_tasks": rest_tasks, "positions_per_task": round(ppt, 1),
                        "tasks": len(take)}
        return ids, why

    def update(self, mode: str, ids: list[str], recs: dict[str, list[dict]]) -> None:
        stop = self.c["protocol"]["teacher"]["stop_ids"]
        for tid in ids:
            sub = self.c["tasks"][tid]["subtype"]
            r = (recs.get(tid) or [None])[0]
            n = terminated_positions(r, stop) if r else None
            self.gen[(sub, mode)] += 1
            if n is not None:
                self.got[(sub, mode)] += n
                self.done[(sub, mode)] += 1

    def table(self) -> dict:
        return {f"{sub}|{m}": {"scored_positions": self.got[(sub, m)], "terminated_tasks": self.done[(sub, m)],
                               "generated_tasks": self.gen[(sub, m)], "queue_position": self.pos[(sub, m)],
                               "queue": len(self.q[sub]), "exhausted": (sub, m) in self.exhausted,
                               "short": any(self.short(sub, m))}
                for sub in sorted(self.q) for m in MODES}


def main(argv=None, *, run=GD.run_battery, authorize=authorize, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    args = ap.parse_args(argv)
    repo = Path(repo or REPO)
    root, status = Path(args.durable_root), Path(args.status)
    out = root / "inputs_generation"
    out.mkdir(parents=True, exist_ok=True)

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rounds": []}
    evidence = root / "inputs_session.json"

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
    save()
    mark("CONTRACT_OK")
    qs, complete = Queues(c), True
    try:
        row = {"round": 0, "modes": {}}
        for mode in MODES:
            ids = [t["task_id"] for t in c["calibration"] if mode in t["modes"]]
            if not ids:
                continue
            t0 = time.time()
            stage = {"stage": f"c00{'t' if mode == 'H-T' else 'n'}", "slice": "calibration", "modes": (mode,), "ids": ids}
            got = TS.run_stage(c, out, stage, run, mark, candidates=1)
            row["modes"][mode] = {"stage": stage["stage"], "tasks": len(ids), "complete": got["complete"],
                                  "seconds": round(time.time() - t0, 1),
                                  "terminated": sum(terminated_positions(rs[0], c["protocol"]["teacher"]["stop_ids"])
                                                    is not None for rs in got["records"].get(mode, {}).values() if rs)}
            if not got["complete"]:
                complete = False
                break
            mark(f"ROUND_OK:{stage['stage']}")
        record["rounds"].append(row)
        save()
        for r in range(1, int(c["inputs_protocol"]["state_evaluation"]["max_rounds"]) + 1):
            if not complete:
                break
            row, any_ids = {"round": r, "modes": {}}, False
            for mode in MODES:
                ids, why = qs.select(mode)
                if not ids:
                    continue
                any_ids = True
                t0 = time.time()
                stage = {"stage": f"e{r:02d}{'t' if mode == 'H-T' else 'n'}", "slice": "state_evaluation",
                         "modes": (mode,), "ids": ids}
                got = TS.run_stage(c, out, stage, run, mark, candidates=1)
                row["modes"][mode] = {"stage": stage["stage"], "tasks": len(ids), "types": why,
                                      "complete": got["complete"], "seconds": round(time.time() - t0, 1)}
                if not got["complete"]:
                    complete = False
                    break
                qs.update(mode, ids, got["records"].get(mode, {}))
                mark(f"ROUND_OK:{stage['stage']}")
            row["state"] = qs.table()
            record["rounds"].append(row)
            save()
            if not complete or not any_ids:
                break
        record["final"] = qs.table()
        record["exhausted"] = sorted(f"{s}|{m}" for s, m in qs.exhausted)
        record["short_after"] = sorted(k for k, v in record["final"].items() if v["short"])
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

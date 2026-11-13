#!/usr/bin/env python3
"""What the time-boxed measurement says a full evaluation would cost ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/cost_profile.py --store <run scr>/store [--store ...] [--out <json>]

For each group's fluid unit (thinking mode): the sealed episodes by termination type and length, the tasks the engine
had STARTED when the time box ended (the keyed submission order makes the started set a random sample of the unit:
sealed + still running, read from the engine-metrics lines of the runner's log), the share of started tasks still
generating at the box's end (long runs: past every degeneration check so far), and the engine's generation throughput
over time. From those: a projection of the unit's GPU hours if the long runs each go to the 40,960-token context,
with the wave time measured (concurrency at a full KV cache x the time one wave takes), and the same in dollars at the
grant's price. A projection, labelled as one; never a measurement of the unit and never a score.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
METRIC = re.compile(r"(\d\d:\d\d:\d\d) \[loggers.py.*Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, "
                    r"Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%")
PRICE = 1.09
CONTEXT = 40960


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def unit_profile(unit: Path) -> dict | None:
    runs_p = unit / "fluid_runs.json"
    if not runs_p.is_file():
        return None
    runs = json.loads(runs_p.read_text())
    recs = [r for p in sorted((unit / "parts").glob("part_*.jsonl")) for r in jl(p)]
    out = {"unit": unit.name, "sealed": len(recs), "runs": []}
    for run in runs:
        log = unit / run["log"]
        pts = []
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                m = METRIC.search(line)
                if m:
                    pts.append({"t": m.group(1), "tok_s": float(m.group(2)), "running": int(m.group(3)),
                                "waiting": int(m.group(4)), "kv_pct": float(m.group(5))})
        last = pts[-1] if pts else {}
        n_req = int(run["tasks_requested"])
        full = [q for q in pts if q["kv_pct"] > 85]
        ids = set(run.get("submitted_order") or [])
        mine = [r for r in recs if r["task_id"] in ids] if ids else recs
        turns = [t for r in mine for t in r["turns"]]
        lens = sorted(int(t["generated_tokens"]) for t in turns)
        out["runs"].append({
            "attempt": run["attempt"], "time_boxed": run.get("time_boxed"), "seconds": run.get("seconds"),
            "execution": run.get("execution"), "tasks_requested": n_req, "sealed_in_run": len(mine),
            "started_at_end": (len([r for r in recs if r["task_id"] in ids]) if ids else len(recs)) +
                              int(last.get("running") or 0) if pts else None,
            "running_at_end": last.get("running"),
            "ends": dict(collections.Counter(r["end"] for r in mine)),
            "stops": dict(collections.Counter(t["stop_reason"] for t in turns)),
            "degeneration_kinds": dict(collections.Counter(t.get("degeneration_kind") for t in turns if t.get("degeneration_kind"))),
            "context_limit_hits": sum(1 for t in turns if t.get("context_limit_reached")),
            "generated_tokens": {"p50": lens[len(lens) // 2] if lens else None, "max": lens[-1] if lens else None,
                                 "over_4096": sum(1 for x in lens if x > 4096)},
            "throughput_tok_s": {"first": pts[0]["tok_s"] if pts else None, "last": last.get("tok_s"),
                                 "min": min((p["tok_s"] for p in pts), default=None)},
            "median_running_at_full_cache": (sorted(q["running"] for q in full)[len(full) // 2] if full else None),
            "median_tok_s_at_full_cache": (sorted(q["tok_s"] for q in full)[len(full) // 2] if full else None),
            "samples": len(pts)})
    return out


def project(p: dict, n_tasks: int = 720) -> dict | None:
    """Long runs = started tasks still running at the end, plus sealed ones past 4,096 tokens; each long run is
    assumed to reach the context (a lower bound on its length, an upper bound on nothing). Waves: at a full cache the
    engine holds `c` full-context runs (the median running count while the cache is >85% full) at the median generation throughput
    of those samples; one wave of `c` runs then takes c x 40,960 / throughput seconds."""
    r = p["runs"][-1]
    if not r.get("started_at_end"):
        return None
    long_runs = int(r.get("running_at_end") or 0) + int(r["generated_tokens"]["over_4096"])
    started = int(r["started_at_end"])
    share = long_runs / started
    c = r.get("median_running_at_full_cache") or r["execution"].get("max_num_seqs") or 1
    tok_s = r.get("median_tok_s_at_full_cache") or r["throughput_tok_s"]["min"] or 1.0
    wave_s = c * CONTEXT / tok_s                              #: c runs of 40,960 tokens at the full-cache rate
    hours = share * n_tasks / c * wave_s / 3600
    return {"started": started, "long_runs": long_runs, "long_share": round(share, 3),
            "concurrency_at_full_cache": c, "full_cache_tok_s": tok_s, "wave_hours": round(wave_s / 3600, 2),
            "projected_unit_gpu_hours": round(hours, 1), "projected_unit_usd": round(hours * PRICE, 1),
            "basis": ("started tasks = sealed in the run + running at the box's end; long runs = still running then "
                      "or sealed past 4,096 tokens; each long run taken to the 40,960-token context at the measured "
                      "full-cache rate; short runs (seconds each) neglected")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", action="append", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {"schema": "aadistill.stage1.reasoning_inheritance.cost_profile/v1", "units": {}}
    for st in a.store:
        for unit in sorted((Path(st) / "out" / "eval").glob("*_H-*")):
            prof = unit_profile(unit)
            if prof:
                prof["projection"] = project(prof)
                res["units"].setdefault(unit.name, []).append({"store": st, **prof})
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

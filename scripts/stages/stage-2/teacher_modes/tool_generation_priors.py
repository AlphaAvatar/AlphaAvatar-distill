#!/usr/bin/env python3
"""The bounded tool generation's priors, MEASURED where the tool pilot measured them. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_priors \\
        --session <secured pilot directory> --write logs/stages/stage-2/teacher_modes/audit/tool_generation_priors.json

The pilot (`a_tool_pilot_20261104_164531`, accepted by the maintainer as a completed `tool_verification@v2`
admission, throughput and cost measurement) attempted each of its tasks once in both hybrid modes under the plan's
own two-candidate rule. Per stratum -- lineage, mode, split, category and episode shape -- this records what it
measured over its SEALED episodes (their v2 verdicts re-derived by `tool_pilot._outcome`, the report's own reading):

* ``attempted`` -- tasks whose first candidate was sealed; ``admitted`` -- tasks with an admitted episode after up to
  two candidates; ``second_candidate_rate`` -- the share whose first candidate was refused (a second was generated);
* ``units`` -- the cell unit the admitted episodes carry (supervised tokens of training / validation turns, scored
  positions of calibration / state-evaluation items), counted before packing (P19);
* ``units_per_attempted_task`` -- ``[low, expected, high]``: the admission proportion's Wilson 90% interval times
  the measured units per admitted episode (the middle is the measured mean); a stratum with no admitted episode has
  no unit estimate (``units_per_admitted`` UNMEASURED);
* ``generated_tokens_per_attempted_task`` and ``prompt_tokens_per_turn`` -- every candidate's turns, the reasoning
  included, which is what the price pays for.

``throughput`` is per mode and engine invocation: output tokens over generation seconds and the engine's start time.
Nothing here is a benchmark score, and nothing measured here stands for another lineage: Thinking-2507 has no
measured stratum.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from stages.teacher_modes import tool_episodes as TE  # noqa: E402
from stages.teacher_modes import tool_pilot as TPL  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.tool_generation_priors/v1"
PRIORS = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_generation_priors.json"
Z90 = 1.6449


def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stratum_key(lineage: str, mode: str, split: str, category: str | None, shape: str) -> str:
    return f"{lineage}|{mode}|{split}|{category}|{shape}"


def measure(session_dirs, *, plan_path: Path = TPL.PLAN, tasks_path: Path = TPL.TASKS,
            study_identity: str | None = None) -> dict:
    doc, by_id = TPL.load(plan_path, tasks_path)
    dirs = [Path(d) for d in ([session_dirs] if isinstance(session_dirs, (str, Path)) else session_dirs)]
    sess = next((json.loads((d / "tool_session.json").read_text()) for d in dirs if (d / "tool_session.json").is_file()),
                {})
    identity = sess.get("identity") or TPL.pilot_identity(study_identity, plan_path)["sha256"]
    stop_ids = [int(x) for x in json.loads(TPL.TGP.PROTOCOL.read_text())["execution"]["stop_ids"]]
    lineage = doc["lineage"]
    acc = defaultdict(lambda: {"attempted": 0, "admitted": 0, "second": 0, "units": 0.0, "generated_tokens": 0,
                               "prompt_tokens": 0, "turns": 0, "candidates": 0})
    for mode in sorted(doc["modes"]):
        thinking = bool(json.loads(TPL.TGP.PROTOCOL.read_text())["lineages"][lineage]["modes"][mode]["template_kwargs"]
                        .get("enable_thinking", True))
        sealed = [TPL._sealed(dirs, f"pilot_{mode}_r1_pass{k}", identity) for k in (0, 1)]
        for cell in doc["modes"][mode]:
            for tid in cell["task_ids"]:
                task = by_id[tid]
                r0 = sealed[0].get(tid)
                if r0 is None:
                    continue
                a = acc[stratum_key(lineage, mode, task["split"], task["category"], TE.episode_shape(task))]
                a["attempted"] += 1
                out, units, _ = TPL._outcome(task, r0, set(), mode=mode, thinking=thinking, stop_ids=stop_ids)
                recs = [r0]
                if out != "admitted":
                    a["second"] += 1
                    r1 = sealed[1].get(tid)
                    if r1 is not None:
                        recs.append(r1)
                        out, units, _ = TPL._outcome(task, r1, set(), mode=mode, thinking=thinking, stop_ids=stop_ids)
                if out == "admitted":
                    a["admitted"] += 1
                    a["units"] += float(units["supervised_tokens"] + units["scored_positions"])
                for r in recs:
                    a["candidates"] += 1
                    for t in r.get("turns") or []:
                        a["turns"] += 1
                        a["generated_tokens"] += int(t.get("generated_tokens") or 0)
                        a["prompt_tokens"] += int(t.get("prompt_tokens") or 0)
    strata = {}
    for key, a in sorted(acc.items()):
        n, k = a["attempted"], a["admitted"]
        lo, hi = wilson(k, n)
        upa = a["units"] / k if k else None
        strata[key] = {
            "attempted": n, "admitted": k, "second_candidate_rate": round(a["second"] / n, 4) if n else None,
            "units": round(a["units"], 1), "units_per_admitted": round(upa, 2) if upa is not None else TPL.UNMEASURED,
            "admission_yield": round(k / n, 4) if n else None, "admission_yield_wilson90": [round(lo, 4), round(hi, 4)],
            "units_per_attempted_task": ([round(lo * upa, 3), round(k / n * upa, 3), round(hi * upa, 3)]
                                         if upa is not None else TPL.UNMEASURED),
            "generated_tokens_per_attempted_task": round(a["generated_tokens"] / n, 1) if n else None,
            "prompt_tokens_per_turn": round(a["prompt_tokens"] / a["turns"], 1) if a["turns"] else None,
            "turns_per_candidate": round(a["turns"] / a["candidates"], 3) if a["candidates"] else None}
    return {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "measured_by": {"run_id": "a_tool_pilot_20261104_164531", "pilot_plan_sha256": TPL.sha(plan_path),
                            "pool_content_sha256": doc["source"]["pool_content_sha256"],
                            "tasks_sha256": doc["tasks_file"]["sha256"], "pilot_identity": identity,
                            "verification_version": TE.VERIFICATION_VERSION, "lineage": lineage},
            "strata": strata, "throughput": throughput(dirs),
            "unmeasured": {"lineages": ["thinking_2507"],
                           "hybrid": "every stratum not listed: the validation split, and calibration / state "
                                     "evaluation in every category but schema_arguments"},
            "interval": "Wilson 90% on the admission proportion x the measured units per admitted episode",
            "authorizes": "nothing"}


def throughput(dirs) -> dict:
    rows = []
    for d in dirs:
        p = Path(d) / "telemetry.jsonl"
        if p.is_file():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows:
        mode = r["label"].split("_")[1]
        for e in r.get("engine") or []:
            by[mode].append({"label": r["label"], "wall_seconds": r.get("wall_seconds"),
                             "engine_start_seconds": (e.get("engine_ready_seconds") or [None])[0],
                             "output_tokens": e.get("output_tokens"), "generation_seconds": e.get("generation_seconds"),
                             "output_tokens_per_second": e.get("output_tokens_per_second")})
    out = {}
    for mode, xs in sorted(by.items()):
        tok = sum(float(x["output_tokens"] or 0) for x in xs)
        sec = sum(float(x["generation_seconds"] or 0) for x in xs)
        out[mode] = {"invocations": xs, "output_tokens_per_second": round(tok / sec, 1) if sec else TPL.UNMEASURED,
                     "slowest_invocation_tokens_per_second": min(float(x["output_tokens_per_second"] or 0) for x in xs),
                     "engine_start_seconds": {"first": xs[0]["engine_start_seconds"],
                                              "later": max(float(x["engine_start_seconds"] or 0) for x in xs[1:])
                                              if len(xs) > 1 else None}}
    return out


def load(path: Path = PRIORS) -> dict:
    return json.loads(Path(path).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", action="append", required=True)
    ap.add_argument("--write", default=str(PRIORS))
    a = ap.parse_args(argv)
    doc = measure([Path(x) for x in a.session])
    Path(a.write).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("attempted", "admitted", "units_per_attempted_task")}
                      for k, v in doc["strata"].items()}, indent=1)[:6000])
    print(json.dumps({m: {k: v for k, v in t.items() if k != "invocations"} for m, t in doc["throughput"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

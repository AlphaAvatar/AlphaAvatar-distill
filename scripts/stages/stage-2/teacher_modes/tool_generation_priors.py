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
Nothing here is a benchmark score, and nothing measured here stands for another lineage.

Thinking-2507's strata (``--add-import``; maintainer 2026-11-05, review of f06ccd4, section 3) are measured the same
way from its completed formal session through that session's import record (`tool_generation_import.build_session`):
every task it attempted, its candidates' outcomes as the record re-derived them, and the turns of the sealed shards
the record names. They are added beside the pilot's strata, which stay as they are, so the plan's bounds for that
lineage rest on what that teacher measured instead of the protocol priors that overestimated its per-task yield.
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


#: tool_pool@v1's two sources keep the unsuffixed stratum key every recorded prior and plan uses; a task of any other
#: source (a pool extension, `tool_pool.extend`) has strata of its own (``|<dataset>``), so it is never priced at
#: another source's measured yield and stays UNMEASURED until a session of its own measures it
POOL_V1_SOURCES = ("Agent-Ark/Toucan-1.5M", "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1")


def stratum_key(lineage: str, mode: str, split: str, category: str | None, shape: str,
                dataset: str | None = None) -> str:
    key = f"{lineage}|{mode}|{split}|{category}|{shape}"
    return key if dataset is None or dataset in POOL_V1_SOURCES else f"{key}|{dataset}"


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
                a = acc[stratum_key(lineage, mode, task["split"], task["category"], TE.episode_shape(task),
                                    (task.get("source") or {}).get("dataset"))]
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
    strata = _strata(acc)
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


def _strata(acc: dict) -> dict:
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
    return strata


def measure_import(record: dict, store: Path, acc=None) -> dict:
    """``{"strata", "throughput", "measured_by"}`` of one formal session from its import record: per stratum of the
    record's lineage, the tasks it lists (attempted), their final outcomes and units, and every listed candidate's
    turns read from the sealed shard the record names (verified by sha256 and the session's identity). ``acc``
    accumulates several sessions of one lineage (`measure_imports`)."""
    from stages.reasoning_data import generation_driver as GD
    lineage, store = record["lineage"], Path(store)
    identity = record.get("identity") or record["pilot_identity"]           #: a formal session's, or the pilot's
    acc = acc if acc is not None else defaultdict(lambda: {"attempted": 0, "admitted": 0, "second": 0, "units": 0.0,
                                                           "generated_tokens": 0, "prompt_tokens": 0, "turns": 0,
                                                           "candidates": 0})
    for mode, m in sorted(record["modes"].items()):
        labels = sorted({c["label"] for t in m["tasks"].values() for c in t["candidates"]})
        sealed = {lb: GD.sealed(store, lb, identity) for lb in labels}
        for tid, t in m["tasks"].items():
            a = acc[stratum_key(lineage, mode, t["split"], t["category"], t["shape"], t.get("dataset"))]
            a["attempted"] += 1
            a["second"] += int(len(t["candidates"]) > 1)
            if t["final"]["outcome"] == "admitted":
                a["admitted"] += 1
                a["units"] += float(t["final"]["units"])
            for c in t["candidates"]:
                if GD.sha(store / c["shard"]) != c["shard_sha256"]:
                    raise SystemExit(f"{c['shard']}: not the shard the import record names")
                a["candidates"] += 1
                for turn in sealed[c["label"]][tid].get("turns") or []:
                    a["turns"] += 1
                    a["generated_tokens"] += int(turn.get("generated_tokens") or 0)
                    a["prompt_tokens"] += int(turn.get("prompt_tokens") or 0)
    return {"strata": _strata(acc), "throughput": throughput([store]),
            "measured_by": {"run_id": record["run_id"],
                            "import_record_session_plan_sha256": (record.get("session_plan_sha256")
                                                                  or record.get("pilot_plan_sha256")),
                            "pool_content_sha256": record["pool_content_sha256"], "identity": identity,
                            "verification_version": record["verification_version"], "lineage": lineage}}


def measure_imports(pairs) -> dict:
    """`measure_import` over several sessions of ONE lineage, ``[(record, store)]``: their tasks are disjoint (each
    later plan imported the earlier sessions), so the strata pool them; the throughput rows are every session's."""
    if len({r["lineage"] for r, _ in pairs}) != 1:
        raise SystemExit("measure_imports pools the sessions of one lineage")
    seen: set = set()
    for r, _ in pairs:
        ids = {(m, t) for m, x in r["modes"].items() for t in x["tasks"]}
        if ids & seen:
            raise SystemExit(f"{r['run_id']}: a task another pooled session already lists")
        seen |= ids
    acc = defaultdict(lambda: {"attempted": 0, "admitted": 0, "second": 0, "units": 0.0, "generated_tokens": 0,
                               "prompt_tokens": 0, "turns": 0, "candidates": 0})
    for r, store in pairs:
        last = measure_import(r, store, acc)
    return {"strata": last["strata"], "throughput": throughput([s for _, s in pairs]),
            "measured_by": [{"run_id": r["run_id"], "identity": r.get("identity") or r["pilot_identity"],
                             "import_record_session_plan_sha256": r.get("session_plan_sha256") or r.get("pilot_plan_sha256"),
                             "pool_content_sha256": r["pool_content_sha256"],
                             "verification_version": r["verification_version"]} for r, _ in pairs]}


def replace_lineage(priors: dict, lineage: str, pairs, record_paths: list[str]) -> dict:
    """``priors`` with ``lineage``'s strata and throughput re-measured over every session in ``pairs``
    (`measure_imports`); every other lineage's strata are kept as they are."""
    got = measure_imports(pairs)
    modes = {m for r, _ in pairs for m in r["modes"]}
    out = dict(priors)
    out["strata"] = {**{k: v for k, v in priors["strata"].items() if not k.startswith(f"{lineage}|")}, **got["strata"]}
    out["throughput"] = {**{k: v for k, v in priors["throughput"].items() if k not in modes}, **got["throughput"]}
    out["measured_by_lineage"] = {**(priors.get("measured_by_lineage") or {}),
                                  lineage: {"sessions": got["measured_by"], "import_records": record_paths}}
    un = dict(priors.get("unmeasured") or {})
    un["lineages"] = [x for x in un.get("lineages") or [] if x != lineage]
    un[lineage] = "every stratum not listed (no task of it attempted in these sessions)"
    out["unmeasured"] = un
    out["written_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def add_import(priors: dict, record: dict, store: Path, record_path: str) -> dict:
    """``priors`` with one formal session's strata and throughput added (a lineage the priors do not measure yet);
    every existing stratum and mode is kept as it is."""
    got = measure_import(record, store)
    clash = sorted(set(got["strata"]) & set(priors["strata"])) + sorted(set(got["throughput"]) & set(priors["throughput"]))
    if clash:
        raise SystemExit(f"already measured: {clash[:3]}")
    out = dict(priors)
    out["strata"] = {**priors["strata"], **got["strata"]}
    out["throughput"] = {**priors["throughput"], **got["throughput"]}
    out["measured_by_lineage"] = {**(priors.get("measured_by_lineage") or {}),
                                  record["lineage"]: {**got["measured_by"], "import_record": record_path}}
    un = dict(priors.get("unmeasured") or {})
    un["lineages"] = [x for x in un.get("lineages") or [] if x != record["lineage"]]
    un[record["lineage"]] = "every stratum not listed (its session generated training and validation only)"
    out["unmeasured"] = un
    out["written_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def throughput(dirs) -> dict:
    by = defaultdict(list)
    for n, d in enumerate(dirs):              #: one directory per session (pod): its first engine start compiles
        p = Path(d) / "telemetry.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
        for r in rows:
            mode = r["label"].split("_")[1]
            for e in r.get("engine") or []:
                by[mode].append({"label": r["label"], "session": n, "wall_seconds": r.get("wall_seconds"),
                                 "engine_start_seconds": (e.get("engine_ready_seconds") or [None])[0],
                                 "output_tokens": e.get("output_tokens"),
                                 "generation_seconds": e.get("generation_seconds"),
                                 "output_tokens_per_second": e.get("output_tokens_per_second")})
    out = {}
    for mode, xs in sorted(by.items()):
        tok = sum(float(x["output_tokens"] or 0) for x in xs)
        sec = sum(float(x["generation_seconds"] or 0) for x in xs)
        firsts, laters = {}, []
        for x in xs:
            if x["session"] in firsts:
                laters.append(float(x["engine_start_seconds"] or 0))
            else:
                firsts[x["session"]] = x["engine_start_seconds"]
        out[mode] = {"invocations": xs, "output_tokens_per_second": round(tok / sec, 1) if sec else TPL.UNMEASURED,
                     "slowest_invocation_tokens_per_second": min(float(x["output_tokens_per_second"] or 0) for x in xs),
                     "engine_start_seconds": {"first": max(float(v or 0) for v in firsts.values()),
                                              "later": max(laters) if laters else None}}
    return out


def load(path: Path = PRIORS) -> dict:
    return json.loads(Path(path).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--add-import", default=None, help="a formal session's import record: its strata are added to "
                    "--priors (with --import-store)")
    ap.add_argument("--import-store", default=None)
    ap.add_argument("--lineage-imports", nargs="+", default=None, metavar="RECORD=STORE",
                    help="re-measure one lineage's strata over every listed session (record=store pairs)")
    ap.add_argument("--priors", default=str(PRIORS))
    ap.add_argument("--write", default=str(PRIORS))
    a = ap.parse_args(argv)
    if a.lineage_imports:
        pairs, paths = [], []
        for x in a.lineage_imports:
            rp, st = x.split("=", 1)
            pairs.append((json.loads(Path(rp).read_text()), Path(st)))
            paths.append(Path(rp).resolve().relative_to(REPO).as_posix())
        doc = replace_lineage(load(Path(a.priors)), pairs[0][0]["lineage"], pairs, paths)
    elif a.add_import:
        rec = json.loads(Path(a.add_import).read_text())
        rel = Path(a.add_import).resolve().relative_to(REPO).as_posix()
        doc = add_import(load(Path(a.priors)), rec, Path(a.import_store), rel)
    elif a.session:
        doc = measure([Path(x) for x in a.session])
    else:
        ap.error("--session or --add-import is required")
    Path(a.write).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("attempted", "admitted", "units_per_attempted_task")}
                      for k, v in doc["strata"].items()}, indent=1)[:6000])
    print(json.dumps({m: {k: v for k, v in t.items() if k != "invocations"} for m, t in doc["throughput"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

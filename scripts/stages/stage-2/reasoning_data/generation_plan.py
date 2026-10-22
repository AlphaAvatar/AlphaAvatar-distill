#!/usr/bin/env python3
"""The paid generation list of one corpus variant: which tasks, how many, at what scale. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.generation_plan \
        --variant stage2.reasoning_master@v1 --pools <pool.jsonl> ... --inputs-dir <dir>

For every (stratum, source) cell the variant gives a positive share
(`source_plan`, which also says whether the variant is ADMITTED):

* a ``reused_targets`` cell generates nothing; its verified sessions are
  counted from the reuse selection (`build_supporting_tasks`) given with
  ``--reuse``;
* a generated cell first DROPS every task without a usable verifier -- an
  external module whose answer type is ``needs_checker`` or names a checker
  `checkers.CHECKERS` lacks, a module that is no longer a candidate, a dataset
  task whose ``verifier`` `supporting_verify.VERIFIERS` lacks (maintainer
  2026-10-22: only modules with a real verifier are generated) -- then takes
  ``ceil(units / p x oversupply)`` tasks, round-robin over the cell's groups
  (generator x difficulty, module x level, ...) in pool order. Nothing is chosen
  by looking at a result. A cell whose verifiable supply cannot fill its units
  even in expectation is a SHORTFALL; one that can, but with less than the
  margin, is BELOW MARGIN; both are reported, never filled from another source.

Rates, each with its basis named per cell: ``units`` is the cell's independent
units (`source_plan`); ``p`` the n = 1 unit accept rate within the 8,192 block
and ``g`` the generated tokens per candidate (whole episode for an episode) at
H1's allowance. A cell needs ``units / q x oversupply`` tasks, with ``q`` the
units a task yields over both candidates (``1 - (1 - p)^2``; ``p`` where there
is no second candidate).
Measured on the same generators where the teacher has run (Stage-0
teacher_cal), otherwise a named PROXY (the same axis on teacher_cal; corpus_v2's
nearest type) or the spec's labelled assumption; a proxied cell takes a larger
oversupply. Expected tokens assume the second candidate for every task the
first fails, except where the verdict cannot be known on the generation pod
(code: the sandbox runs on the dev box only), so the figure is an upper bound
on the policy.

Writes the run_battery inputs (``--inputs-dir``: one file per system message,
`run_battery`'s task contract, with each file's sha256 in the plan) and
`logs/stages/stage-2/reasoning_data/plans/generation_plan.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import source_plan as SP

OUT = A.LOG_DIR / "plans/generation_plan.json"
OVERSUPPLY = {"measured": 1.5, "proxy": 2.0}
#: verifiers that may not run where the teacher runs: no second candidate
DEV_BOX_ONLY = frozenset({"code_asserts"})
SESSION_LIMIT = 8192 - 8          # aadistill.data.sessions.completion_budget's margin
SYSTEM_DEFAULT = "You are a helpful Assistant."


def verifier_of(task: dict) -> tuple[str | None, str | None]:
    """(the verifier that will judge `task`, None) or (None, why there is none)."""
    if task.get("generator_id"):
        return ("episode_replay" if task.get("interactive") else "battery_v2"), None
    if task.get("verifier"):
        from stages.reasoning_data.supporting_verify import VERIFIERS
        return (task["verifier"], None) if task["verifier"] in VERIFIERS else \
            (None, f"no verifier {task['verifier']!r}")
    if task.get("module"):
        from stages.reasoning_data.external_tasks import candidates
        from stages.reasoning_data.generation_adapter import COMPARE
        if (task["source"], task["module"]) not in candidates():
            return None, f"{task['module']}: not a candidate module"
        kind = task.get("answer_type")
        if kind == "checker":
            from stages.reasoning_data.checkers import CHECKERS
            key = f"{task['source']}/{task['module']}"
            return (f"checker:{key}", None) if key in CHECKERS else (None, f"{key}: no project checker")
        return (f"compare:{kind}", None) if kind in COMPARE else (None, f"{task['module']}: answer type {kind}")
    return None, f"no verifier for source {task.get('source')!r}"


def group_of(task: dict) -> str:
    if task.get("generator_id"):
        return f"{task['generator_id']}/{task.get('difficulty', '')}"
    if task.get("module"):
        return f"{task['module']}/{task.get('level', '')}"
    return f"{task.get('family_id', '')}/{task.get('difficulty', '')}"


def round_robin(tasks: list[dict], n: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        groups[group_of(t)].append(t)
    order, k = [], 0
    while len(order) < min(n, len(tasks)):
        for g in sorted(groups):
            if k < len(groups[g]) and len(order) < n:
                order.append(groups[g][k])
        k += 1
    return order


def at_allowance(rec: dict, limit: int = SESSION_LIMIT) -> int:
    """A teacher_cal record's generated tokens had it run at `limit` (it ran at
    16,384): each turn stops at ``limit - prompt``, and a stopped turn ends
    the episode, as run_battery does."""
    total = 0
    for t in rec["turns"]:
        allow = max(limit - int(t["prompt_tokens"]), 0)
        total += min(int(t["generated_tokens"]), allow)
        if int(t["generated_tokens"]) >= allow:
            break
    return total


def rates(spec: dict) -> dict:
    """Per-basis (p, g) from the closed records, each named."""
    audit = A.read_json(A.LOG_DIR / "audit/asset_audit.json")
    tc, pool = audit["teacher_cal"], audit["base_pool_v1_fluid"]
    gen: dict[str, list[int]] = defaultdict(list)
    for rec in A.teacher_cal_records():
        gen[rec["axis"]].append(at_allowance(rec))
    out = {}
    for axis, c in tc["under_target_quality_contract"].items():
        out[f"teacher_cal:{axis}"] = {
            "p": pool["units_by_axis"][axis] / c["units"],
            "g": sum(gen[axis]) / len(gen[axis]),
            "basis": (f"Stage-0 teacher_cal {axis}: units accepted and <= 8,192 (base_pool_v1) / units "
                      f"generated; generated tokens per unit re-counted at the {SESSION_LIMIT} allowance "
                      f"(it ran at 16,384)")}
    cv = audit["corpus_v2"]["under_target_quality_contract"]
    for kind, c in cv["per_type"].items():
        if c["candidates"]:
            out[f"corpus_v2:{kind}"] = {
                "p": c["candidate_accept_rate"],
                "g": min(cv["generated_tokens"][kind]["all_candidates"] / c["candidates"], SESSION_LIMIT),
                "basis": (f"corpus_v2 {kind} under the v2 contract: candidate accept rate (n = 1); generated "
                          f"tokens per candidate, capped at the {SESSION_LIMIT} allowance")}
    code = spec["generation"]["supporting.code_algorithmic"]["yield_from"]
    out["assumption:code"] = {"p": float(code.split(":")[1]), "g": out["corpus_v2:code"]["g"],
                              "basis": f"corpus_spec {code} (a labelled ASSUMPTION: corpus_v2 code had no verifier) "
                                       "x corpus_v2 code tokens per candidate"}
    return out


#: cell -> (basis key, measured | proxy). Project cells read their own axis.
def basis_of(stratum: str, source: str, spec: dict) -> tuple[str, str]:
    axis = spec["strata"][stratum].get("axis")
    if source == "s0_generators_v2_calsplit":
        return f"teacher_cal:{axis}", "measured"
    if stratum.startswith("primary."):
        return f"teacher_cal:{axis}", "proxy"
    return {"math_train": ("corpus_v2:openmath", "proxy"), "mathematics_dataset": ("corpus_v2:openmath", "proxy"),
            "reasoning_gym": ("corpus_v2:openmath" if stratum == "supporting.math_new" else "assumption:code",
                              "proxy"),
            "reasoning_core": ("assumption:code", "proxy"), "mbpp_train": ("assumption:code", "proxy"),
            "squad_v2_train_unanswerable": ("corpus_v2:rag_evidence", "proxy")}[source]


def run_input(task: dict, spec: dict) -> dict:
    """`task` in run_battery's task contract; its own system message, if any, is
    the run's (`--system`), so it is lifted out of the messages."""
    stratum = task.get("stratum") or SP.stratum_of(task, spec)
    msgs = task["messages"]
    row = {"task_id": task["task_id"],
           "messages": msgs[1:] if msgs and msgs[0]["role"] == "system" else msgs,
           "generator_id": task.get("generator_id") or (f"{task['source']}:{task['module']}" if task.get("module")
                                                        else task["source"]),
           "family_id": task.get("family_id") or task["source"],
           "axis": task.get("axis") or spec["strata"][stratum].get("axis") or stratum,
           "difficulty": str(task.get("difficulty") or task.get("level") or ""),
           "interactive": bool(task.get("interactive"))}
    if task.get("interactive"):
        row["env"] = task["env"]
    return row


def build(variant: str, pools: list[Path], reuse: list[Path] = ()) -> dict:
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    sp = SP.plan(variant)
    r = rates(spec)
    tasks_by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for p in pools:
        for t in A.jsonl(Path(p)):
            tasks_by_cell[(t.get("stratum") or SP.stratum_of(t, spec), t.get("source"))].append(t)
    reused = Counter()
    for p in reuse:
        for row in A.jsonl(Path(p)):
            reused[(row["stratum"], row["source"])] += 1
    cells, chosen, shortfalls = [], [], []
    for c in sp["cells"]:
        if c["share"] <= 0:
            continue
        key = (c["stratum"], c["source"])
        row = {k: c[k] for k in ("stratum", "source", "kind", "share", "U_target_with_reserve",
                                 "independent_units_estimate", "supervised_per_unit_estimate")}
        if c["kind"] == "reused_targets":
            row.update(generate=0, reused_units_available=reused[key],
                       shortfall=reused[key] < c["independent_units_estimate"])
        else:
            dropped, ok = Counter(), []
            for t in tasks_by_cell[key]:
                v, why = verifier_of(t)
                (ok.append(t) if v else dropped.update([why]))
            bkey, kind = basis_of(*key, spec)
            rate = r[bkey]
            retry = all(verifier_of(t)[0] not in DEV_BOX_ONLY for t in ok)
            q = 1 - (1 - rate["p"]) ** 2 if retry else rate["p"]      # units per task, both candidates
            n = math.ceil(c["independent_units_estimate"] / q * OVERSUPPLY[kind])
            take = round_robin(ok, n)
            expected_units = len(take) * q
            tokens = len(take) * rate["g"] * ((2 - rate["p"]) if retry else 1)
            row.update(tasks_in_pools=len(tasks_by_cell[key]), verifiable=len(ok),
                       dropped_no_verifier=dict(sorted(dropped.items())), tasks_needed=n, generate=len(take),
                       expected_units=round(expected_units, 1),
                       shortfall=expected_units < c["independent_units_estimate"],
                       below_margin=len(take) < n, rate_basis={"key": bkey, "kind": kind, **rate,
                                                            "oversupply": OVERSUPPLY[kind]},
                       second_candidate="for every task the first fails" if retry else
                       "none: the verdict needs the dev-box sandbox",
                       expected_generated_tokens_upper=round(tokens),
                       groups=dict(sorted(Counter(group_of(t) for t in take).items())))
            chosen += take
        if row["shortfall"]:
            shortfalls.append(f"{key[0]}/{key[1]}")
        cells.append(row)
    return {"spec": spec, "plan": sp, "cells": cells, "chosen": chosen, "shortfalls": shortfalls}


def write_inputs(chosen: list[dict], spec: dict, out_dir: Path) -> list[dict]:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for t in chosen:
        m = t["messages"]
        by_system[m[0]["content"] if m and m[0]["role"] == "system" else SYSTEM_DEFAULT].append(run_input(t, spec))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (system, rows) in enumerate(sorted(by_system.items())):
        path = out_dir / f"generation_input_g{k}.jsonl"
        path.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in rows))
        files.append({"file": path.name, "system": system, "tasks": len(rows), "sha256": A.sha256_file(path),
                      "interactive": sum(x["interactive"] for x in rows)})
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="stage2.reasoning_master@v1")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--reuse", nargs="*", default=[])
    ap.add_argument("--inputs-dir", required=True)
    args = ap.parse_args(argv)
    sp = SP.plan(args.variant)
    if not (sp["admitted"] and sp["chosen"]):
        raise SystemExit(f"{args.variant}: not the chosen, admitted variant ({sp['status']}); no paid list is written")
    got = build(args.variant, [Path(p) for p in args.pools], [Path(p) for p in args.reuse])
    files = write_inputs(got["chosen"], got["spec"], Path(args.inputs_dir))
    gen = [c for c in got["cells"] if c["generate"]]
    doc = {"schema": "aadistill.stage2.reasoning_data.generation_plan/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "corpus_id": args.variant, "admitted": got["plan"]["admitted"], "status": got["plan"]["status"],
           "pools": [{"path": SP.repo_path(Path(p)), "sha256": A.sha256_file(Path(p))} for p in args.pools],
           "reuse": [{"path": SP.repo_path(Path(p)), "sha256": A.sha256_file(Path(p))} for p in args.reuse],
           "cells": got["cells"], "shortfalls": got["shortfalls"],
           "inputs": files, "session_limit": SESSION_LIMIT,
           "below_margin": [f"{c['stratum']}/{c['source']}" for c in got["cells"] if c.get("below_margin")],
           "totals": {"tasks": sum(c["generate"] for c in gen),
                      "episodes": sum(f["interactive"] for f in files),
                      "expected_generated_tokens_upper": sum(c["expected_generated_tokens_upper"] for c in gen),
                      "by_rate_kind": dict(Counter(c["rate_basis"]["kind"] for c in gen))},
           "seed_rule": "pass k runs run_battery with --seed-base <plan base> + k; a turn's seed is "
                        "request_seed(base, task_id, turn), so every candidate and turn has its own recorded seed",
           "_reading": ("an upper bound on the candidate policy at the named rates; the run's own figures replace "
                        "them. A proxy rate is not a measurement of that source"),
           "authorizes": "nothing"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), doc["totals"], "shortfalls:", doc["shortfalls"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

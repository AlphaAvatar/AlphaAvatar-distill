#!/usr/bin/env python3
"""The bounded teacher-native tool generation plan: targets, frozen orders, rounds, bounds and price. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_plan \\
        --pool data/stages/stage-2/teacher_modes/tool_pool_v1 --write logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json

Generation is bounded by what the arms consume, never by the pool's size. The terms are
tool_generation_protocol.json :: bounded_plan:

* cells -- per lineage, mode and split, one target in the unit its consumer counts BEFORE packing: training in unique
  supervised tokens, split over the seven categories by the registered common-support composition
  (six_category_common_support.json, maintainer 2026-11-05); calibration and state evaluation in unique scored
  positions, by six_category_design.json's registered shares (state evaluation also >= 5 tasks per category the
  split holds); validation as registered (the hybrid's 32 tasks, Thinking-2507's 12,624 supervised tokens). A
  category that cannot reach its target is a SHORTFALL, never filled from another;
* order -- each cell walks its split's frozen order (pool.json :: orders, verified by `tool_pool.load_pool`) filtered
  to its category, whole tasks only; every teacher walks the same order and stops at its own target;
* rounds -- round 1 takes, per cell, the prefix whose EXPECTED yield reaches the target; each later round takes
  ``ceil(remaining / observed yield per attempted task x 1.1)`` more; the LAST round (``max_rounds``) takes every
  task left within the bound, which the price's upper level already pays for. Each round is one engine invocation
  per mode for its first candidates and one for the second candidates of its refused tasks;
* bound -- a cell never attempts more than the prefix whose PESSIMISTIC yield (the lower end of its measured
  stratum, else the lower protocol priors) reaches its target, nor more than its supply: that prefix is what the
  price's upper bound buys. A whole-supply cell (six_category_common_support.json :: whole_supply_cells) is bound by
  its whole remaining supply. A cell short at its bound with supply left is short of its BOUND, not of its source;
* imports -- every sealed episode of an earlier session is reused through an import record
  (`tool_generation_import`: the tool pilot's, and completed formal sessions'); a cell starts from their units and
  its rounds walk the frozen order without the imported tasks;
* yield -- a task's expected turns are its recorded steps plus one answer per user turn (a gold task: its gold
  steps); expected units per task = turns x tokens per turn (the mode's prior) x the probability that one of two
  candidates is admitted, ``1 - (1 - p)^2``;
* price -- generated tokens / the supplement's measured throughput (scaled by a tool-workload efficiency range),
  plus the first prompts' prefill, a model load per invocation, setup and teardown, at the L40S rate. A pilot's
  measured pass rates, turn lengths and throughput replace the priors before the cap is final.

`round_tasks` is the rule the session (`tool_session.py`) runs; nothing here generates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
SCHEMA = "aadistill.stage2.teacher_modes.tool_generation_plan/v1"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json"
#: the pilot's measured priors and its import record (maintainer 2026-11-04: the pilot is a completed tool_verification@v2
#: measurement; its episodes are reused, never regenerated); a plan binds both by sha256
PRIORS = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_generation_priors.json"
IMPORT = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import.json"
#: every import record a plan binds, in order (maintainer 2026-11-05, review of f06ccd4, section 1): the pilot's, and
#: Thinking-2507's completed formal sessions a_tool_generation_t2507_20261104_195411, _20261105_064829 and
#: _20261105_072112, and the hybrid's a_tool_generation_20261105_074146 and _20261105_105512
IMPORTS = (IMPORT, REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import_t2507.json",
           REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import_t2507_a3.json",
           REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import_t2507_a4.json",
           REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import_hybrid_a1.json",
           REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import_hybrid_a2.json")
#: the probability a first-round-bound lineage's single round must reach its target with (`reach_bound`)
REACH_PROBABILITY = 0.95
#: the registered training composition (six_category_common_support@v1, maintainer 2026-11-05)
COMMON_SUPPORT = REPO / "configs/stages/stage-2/teacher_modes/six_category_common_support.json"
#: the measured rate and the grant's per-GPU ceiling the price's upper level pays (USD per hour, one L40S)
RATE_EXPECTED, RATE_UPPER = 1.09, 1.19


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_turns(task: Mapping) -> int:
    """Turns an episode of ``task`` is expected to take: its gold steps, else its recorded steps plus one answer per
    user turn."""
    if task.get("gold_steps"):
        return len(task["gold_steps"])
    rec = (task.get("source") or {}).get("recorded") or {}
    return int(rec.get("steps", 0)) + 1 + len(task.get("later_user_turns") or [])


def admit_probability(p: float, candidates: int) -> float:
    return 1.0 - (1.0 - p) ** candidates


def per_task_units(task: Mapping, unit: str, tpt: float, p: float, candidates: int) -> float:
    """Expected units one attempted task yields."""
    a = admit_probability(p, candidates)
    if unit == "tasks":
        return a
    return a * expected_turns(task) * tpt


def prefix_reaching(tasks: Sequence[Mapping], target: float, unit: str, tpt: float, p: float, candidates: int) -> int:
    """The shortest prefix of ``tasks`` whose expected units reach ``target`` (all of them when none does)."""
    got = 0.0
    for i, t in enumerate(tasks):
        if got >= target:
            return i
        got += per_task_units(t, unit, tpt, p, candidates)
    return len(tasks)


def cell_targets(bp: Mapping, shares: Mapping[str, float], lineage: str, mode: str, splits: Sequence[str],
                 training_shares: Mapping[str, float] | None = None,
                 source_groups: Mapping | None = None) -> list[dict]:
    """The lineage-mode's cells: ``{split, category, unit, target, min_tasks}``; the training split is split by
    ``training_shares`` when given (the registered common-support composition), every other split by ``shares``.

    ``source_groups`` (a design's ``{split: {category: [group, ...]}}``) splits a category's cell into one cell per
    source group, each ``{group, datasets}`` with its own target: a fixed ``per_mode`` value, or ``rest`` -- the
    category's target minus the fixed groups'. Every arm then holds the same units of each source in that category
    (a source-composition match); each group keeps the category's min_tasks. Without it, nothing changes."""
    out = []
    tg = bp["targets"]
    for split in splits:
        spec = tg[split]
        if split == "validation":
            v = spec[mode]
            out.append({"split": split, "category": None, "unit": v["unit"], "target": float(v["value"]),
                        "min_tasks": 0})
            continue
        total = float(spec["per_mode"][mode])
        for cat, share in sorted((training_shares if split == "training" and training_shares else shares).items()):
            cell = {"split": split, "category": cat, "unit": spec["unit"], "target": total * share,
                    "min_tasks": int(spec.get("min_tasks_per_category", 0))}
            groups = ((source_groups or {}).get(split) or {}).get(cat)
            if not groups:
                out.append(cell)
                continue
            fixed = sum(float(g["per_mode"][mode]) for g in groups if not g.get("rest"))
            if fixed > cell["target"] or sum(bool(g.get("rest")) for g in groups) != 1:
                raise ValueError(f"{split}/{cat}: source groups need exactly one 'rest' group and fixed groups within "
                                 f"the category's {cell['target']:.1f}")
            for g in groups:
                out.append({**cell, "group": g["group"], "datasets": list(g["datasets"]),
                            "target": cell["target"] - fixed if g.get("rest") else float(g["per_mode"][mode])})
    return out


def cell_order(order: Sequence[str], by_id: Mapping[str, Mapping], category: str | None,
               datasets: Sequence[str] | None = None) -> list[Mapping]:
    return [by_id[t] for t in order if (category is None or by_id[t]["category"] == category)
            and (datasets is None or (by_id[t].get("source") or {}).get("dataset") in datasets)]


class PlanIdentityError(RuntimeError):
    """A plan that does not belong to the pool, protocol or design it is read with, or whose cells drifted."""


def stratum_candidates(lineage: str, mode: str, task: Mapping) -> list[tuple[str, str]]:
    """``[(stratum key, basis)]`` a task is priced from, in order: its own source's stratum on its split
    (``measured``) and on training (``proxy:training``); for a source no session has measured yet (a pool extension:
    `tool_generation_priors.stratum_key`), the same lineage, mode, category and shape measured on tool_pool@v1's
    sources (``proxy:other_source`` / ``proxy:other_source_training``) -- labelled, never reported as measured."""
    from stages.teacher_modes import tool_episodes as TE
    from stages.teacher_modes.tool_generation_priors import stratum_key
    shape, ds = TE.episode_shape(task), (task.get("source") or {}).get("dataset")
    out = [(stratum_key(lineage, mode, task["split"], task["category"], shape, ds), "measured"),
           (stratum_key(lineage, mode, "training", task["category"], shape, ds), "proxy:training")]
    if stratum_key(lineage, mode, "training", task["category"], shape, ds) != \
            stratum_key(lineage, mode, "training", task["category"], shape):
        out += [(stratum_key(lineage, mode, task["split"], task["category"], shape), "proxy:other_source"),
                (stratum_key(lineage, mode, "training", task["category"], shape), "proxy:other_source_training")]
    return out


def task_units(task: Mapping, unit: str, *, lineage: str, mode: str, proto: Mapping,
               priors: Mapping | None) -> tuple[tuple[float, float, float], str]:
    """``((low, expected, high), basis)``: the units ONE attempted task is expected to yield under the two-candidate
    rule. ``measured`` -- the pilot's stratum (lineage, mode, split, category, shape; `tool_generation_priors`);
    ``proxy:training`` -- not measured on this split: the same lineage, mode, category and shape on the training split
    (a calibration or state-evaluation item's scored positions are its assistant turn's tokens, the unit a training
    turn's supervised tokens count); ``proxy:mode`` -- a task-count cell: the mode's pooled training admission yield;
    ``prior`` -- nothing measured in this lineage and mode: the protocol's priors (`per_task_units`)."""
    from stages.teacher_modes import tool_episodes as TE
    strata = (priors or {}).get("strata") or {}
    shape = TE.episode_shape(task)
    if unit == "tasks":
        rows = [v for k, v in strata.items() if k.startswith(f"{lineage}|{mode}|training|") and k.endswith("|" + TE.FULL_EPISODE)]
        n, a = sum(v["attempted"] for v in rows), sum(v["admitted"] for v in rows)
        if n:
            from stages.teacher_modes.tool_generation_priors import wilson
            lo, hi = wilson(a, n)
            return (lo, a / n, hi), "proxy:mode"
    else:
        for key, basis in stratum_candidates(lineage, mode, task):
            v = strata.get(key)
            if v and isinstance(v.get("units_per_attempted_task"), list):
                return tuple(float(x) for x in v["units_per_attempted_task"]), basis
    bp, cand = proto["bounded_plan"], int(proto["execution"]["candidates"])
    tpt, pr = bp["priors"]["supervised_tokens_per_turn"][mode], bp["priors"]["episode_pass_rate"]
    return tuple(per_task_units(task, unit, tpt[i], pr[i], cand) for i in (0, 1, 2)), "prior"


#: attempted tasks from which a split's own stratum decides a reach bound; below it the training stratum stands in
OWN_STRATUM_MIN_ATTEMPTS = 5


def reach_stratum(strata: Mapping, lineage: str, mode: str, task: Mapping) -> tuple[float, float | None] | None:
    """``(lower Wilson admission yield, units per admitted episode)`` a reach bound uses for ``task``: its own split's
    stratum once it holds >= `OWN_STRATUM_MIN_ATTEMPTS` attempted tasks -- a stratum with none admitted then counts at
    yield 0 (its units from the training stratum) -- else the training stratum of the same lineage, mode, category and
    shape. None when neither is measured."""
    cands = dict((b, strata.get(k)) for k, b in stratum_candidates(lineage, mode, task))
    own, train = cands.get("measured"), cands.get("proxy:training")
    if not own and not train:                       #: an unmeasured source: tool_pool@v1's strata stand in
        own, train = cands.get("proxy:other_source"), cands.get("proxy:other_source_training")
    units = lambda v: v["units_per_admitted"] if v and isinstance(v.get("units_per_admitted"), (int, float)) else None  # noqa: E731
    if own and int(own.get("attempted") or 0) >= OWN_STRATUM_MIN_ATTEMPTS:
        u = units(own) or units(train)
        return (float(own["admission_yield_wilson90"][0]) if units(own) else 0.0, u) if u else None
    v = own if units(own) else train
    return (float(v["admission_yield_wilson90"][0]), float(units(v))) if units(v) else None


def reach_bound(tasks: Sequence[Mapping], rest: float, *, lineage: str, mode: str, priors: Mapping | None,
                q: float = REACH_PROBABILITY, admissions: int = 0) -> int | None:
    """The shortest prefix of ``tasks`` whose admitted episodes reach ``rest`` units -- and at least ``admissions``
    admitted episodes (a cell's remaining minimum tasks) -- with probability >= ``q``, each task admitted
    independently at its `reach_stratum` lower Wilson yield and every admitted episode counted at the smallest units
    per admitted episode among the prefix's strata. None when a task has no measured stratum; the whole list when
    even all of it does not reach (a stratum measured at yield 0 among them). For a lineage whose budget holds about
    one round, where the mean-yield prefix of a small gap is a coin flip (`lineage_cells`)."""
    strata = (priors or {}).get("strata") or {}
    if rest <= 0 and admissions <= 0:
        return 0
    dist, upa = [1.0], None                         #: P(j admissions so far)
    for i, t in enumerate(tasks):
        got = reach_stratum(strata, lineage, mode, t)
        if got is None:
            return None
        p, u = got
        upa = min(upa, u) if upa else u
        dist = [(dist[j] if j < len(dist) else 0.0) * (1 - p) + (dist[j - 1] * p if j else 0.0)
                for j in range(len(dist) + 1)]
        k = max(math.ceil(rest / upa) if rest > 0 else 0, admissions)
        if sum(dist[k:]) >= q:
            return i + 1
    return len(tasks)


def prefix_reaching_units(units: Sequence[float], target: float) -> int:
    """The shortest prefix of per-task ``units`` whose sum reaches ``target`` (all of them when none does)."""
    got = 0.0
    for i, u in enumerate(units):
        if got >= target:
            return i
        got += u
    return len(units)


#: a cell's state: its sealed units already reach the target; its bound reaches it at the expected yield; every task
#: is attempted and it is short (an exact gap); its whole supply is short at the expected yield (a projection); it
#: reaches only beyond the expected yield's reach of the bound (AT_RISK); or nothing measured decides it
STATUSES = ("REACHED", "REACHABLE", "SHORTFALL_EXHAUSTED", "SHORTFALL_PROJECTED", "AT_RISK")


def lineage_cells(proto: Mapping, shares: Mapping[str, float], lineage: str, man: Mapping, by_id: Mapping,
                  priors: Mapping | None = None, imported: Sequence[Mapping] | Mapping | None = None, *,
                  training_shares: Mapping[str, float] | None = None, whole_supply: Sequence[Mapping] = (),
                  source_groups: Mapping | None = None) -> dict:
    """One lineage's ``{mode: {"cells": [...]}}``. Per cell, apart: what is SEALED (the import records' episodes of
    this cell, `tool_generation_import.merged`), the WHOLE SUPPLY's projection (sealed + every remaining task at the
    measured, proxy or prior yield), what the BOUND buys (the remaining tasks the plan may attempt), and the share
    of that resting on UNMEASURED strata (a proxy or a prior); the round-1 prefix and the bound count NEW tasks only,
    walked in the frozen order with the imported ones skipped. A ``whole_supply`` cell's bound is its whole remaining
    supply."""
    from stages.teacher_modes import tool_episodes as TE
    from stages.teacher_modes import tool_generation_import as TGI
    li = proto["lineages"][lineage]
    records = [imported] if isinstance(imported, Mapping) else list(imported or [])
    wide = {(w["lineage"], w["mode"], w["split"], w["category"]) for w in whole_supply}
    first_bound = lineage in (proto["bounded_plan"]["rounds"].get("first_round_bound_lineages") or [])
    modes = {}
    for mode in sorted(li["modes"]):
        done = TGI.merged(records, lineage, mode)
        cells = []
        for c in cell_targets(proto["bounded_plan"], shares, lineage, mode, li["splits"], training_shares,
                              source_groups):
            tasks = cell_order(man["orders"].get(c["split"], []), by_id, c["category"], c.get("datasets"))
            mine = {t["task_id"]: done[t["task_id"]] for t in tasks if t["task_id"] in done}
            new = [t for t in tasks if t["task_id"] not in mine]
            sealed = sum(o["units"] for o in mine.values() if o["admitted"]) if c["unit"] != "tasks" else \
                float(sum(o["admitted"] for o in mine.values()))
            per = [task_units(t, c["unit"], lineage=lineage, mode=mode, proto=proto, priors=priors) for t in new]
            rest = max(c["target"] - sealed, 0.0)
            tasks_at = {lvl: prefix_reaching_units([u[0][i] for u in per], rest)
                        for lvl, i in (("lower", 2), ("expected", 1), ("upper", 0))}
            admitted = sum(o["admitted"] for o in mine.values())
            whole_cell = (lineage, mode, c["split"], c["category"]) in wide
            reach = (reach_bound(new, rest, lineage=lineage, mode=mode, priors=priors,
                                 admissions=max(0, c["min_tasks"] - admitted))
                     if first_bound and c["unit"] != "tasks" and not (sealed >= c["target"]
                                                                      and admitted >= c["min_tasks"]) else None)
            bound = len(new) if whole_cell else min(len(new), max(tasks_at["upper"], reach or 0,
                                                                  max(0, c["min_tasks"] - admitted)))
            whole = [round(sealed + sum(u[0][i] for u in per), 1) for i in (0, 1, 2)]
            in_bound = [round(sealed + sum(u[0][i] for u in per[:bound]), 1) for i in (0, 1, 2)]
            unmeasured = round(sum(u[0][1] for u in per[:bound] if u[1] != "measured"), 1)
            if sealed >= c["target"] and admitted >= c["min_tasks"]:
                status = "REACHED"
            elif not new:
                status = "SHORTFALL_EXHAUSTED"
            elif whole[1] < c["target"]:
                status = "SHORTFALL_PROJECTED"
            elif in_bound[1] >= c["target"] and whole[0] >= c["target"]:
                status = "REACHABLE"
            else:
                status = "AT_RISK"
            basis = {}
            for t, u in zip(new, per):
                basis.setdefault(f"{TE.episode_shape(t)}:{u[1]}", 0)
                basis[f"{TE.episode_shape(t)}:{u[1]}"] += 1
            cells.append({**c, **({"whole_supply": True} if whole_cell else {}),
                          **({"first_round": "bound", "reach_bound": reach} if first_bound else {}),
                          "supply": len(tasks),
                          "supply_by_shape": {sh: sum(TE.episode_shape(t) == sh for t in tasks)
                                              for sh in TE.EPISODE_SHAPES if any(TE.episode_shape(t) == sh for t in tasks)},
                          "sealed": {"tasks": len(mine), "admitted": admitted, "units": round(sealed, 1)},
                          "new_supply": len(new), "tasks": tasks_at, "bound": bound,
                          "projection": {"whole_supply_units": whole, "within_bound_units": in_bound,
                                         "unmeasured_units_within_bound": unmeasured,
                                         "_": "[low, expected, high]; sealed units included"},
                          "basis_of_new_tasks": dict(sorted(basis.items())), "status": status,
                          #: what the session may attempt, in order: the new tasks up to the bound (P8.4: the pod
                          #: receives these tasks, `bound_tasks_file`, never the whole pool)
                          "task_ids": [t["task_id"] for t in new[:bound]],
                          "shortfall_at_pessimistic_yield": whole[0] < c["target"],
                          "expected_turns_in_bound": sum(expected_turns(t) for t in new[:bound])})
        modes[mode] = {"cells": cells}
    return modes


def design_shares() -> dict[str, float]:
    return {k: float(v["share"]) for k, v in json.loads(SIX.read_text())["tool_calling"]["categories"].items()}


def training_shares(path: Path = COMMON_SUPPORT) -> dict[str, float]:
    """The registered training composition: six_category_common_support.json :: tool_calling.categories."""
    return {k: float(v["share"]) for k, v in json.loads(Path(path).read_text())["tool_calling"]["categories"].items()}


def source_groups(path: Path = COMMON_SUPPORT) -> dict | None:
    """A composition file's ``source_groups`` (`cell_targets`), or None when it registers none."""
    return json.loads(Path(path).read_text()).get("source_groups") or None


def whole_supply_cells(path: Path = COMMON_SUPPORT) -> list[dict]:
    return [{k: w[k] for k in ("lineage", "mode", "split", "category")}
            for w in json.loads(Path(path).read_text()).get("whole_supply_cells") or []]


def pool_contents(man: Mapping) -> tuple[str, ...]:
    """The content identities a pool's tasks were sealed under: its own and, for an extension (`tool_pool.extend`),
    its base's -- every base task is in the extension byte for byte, in its place."""
    return (man["content_sha256"],) + ((man["base"]["content_sha256"],) if isinstance(man.get("base"), Mapping) else ())


def import_records(pool_content_sha256: str | Sequence[str], paths: Sequence[Path] = IMPORTS, *,
                   by_id: Mapping[str, Mapping] | None = None) -> list[tuple[Path, dict]]:
    """``[(path, record)]`` of every import record that exists and applies to this pool. A record sealed under this
    pool's content applies whole. One sealed under another content (a base pool, an earlier extension) applies by
    TASK identity: when every task it lists is in ``by_id`` with the task sha256 it was sealed with -- so a pool that
    only adds tasks keeps every sealed outcome, and no task is attempted twice. A record sealed under a content this
    pool names as its base (`pool_contents`) whose tasks do not all match refuses the plan (`PlanIdentityError`); any
    other non-matching record belongs to another pool and is skipped."""
    contents = (pool_content_sha256,) if isinstance(pool_content_sha256, str) else tuple(pool_content_sha256)
    from stages.teacher_modes import tool_pilot as TPL
    out = []
    for p in paths:
        if not Path(p).is_file():
            continue
        r = json.loads(Path(p).read_text())
        if r.get("pool_content_sha256") == contents[0]:
            out.append((Path(p), r))
            continue
        if by_id is None:
            continue
        bad = sorted(tid for m in r["modes"].values() for tid, t in m["tasks"].items()
                     if tid not in by_id or TPL.task_sha256(by_id[tid]) != t["task_sha256"])
        if not bad:
            out.append((Path(p), r))
        elif r.get("pool_content_sha256") in contents:
            raise PlanIdentityError(f"{Path(p).name}: sealed under this pool's base, but {len(bad)} of its task(s) are "
                                    f"absent or changed, e.g. {bad[:3]}")
    return out


def build(pool_dir: Path, *, protocol: Mapping | None = None, composition: Path | None = COMMON_SUPPORT,
          protocol_path: Path | None = None, priors_path: Path | None = None,
          import_paths: Sequence[Path] | None = None) -> dict:
    """The plan over a verified pool. ``composition`` names the training composition file the plan binds -- the
    registered common-support design by default, or a design that registers its own shares and ``source_groups``
    (cells per source within a category: `cell_targets`); None plans the registered six-category shares.
    ``protocol_path`` is the generation protocol the plan is built under (a design's own file), ``priors_path`` and
    ``import_paths`` the measured priors and the import records it binds -- each design's own, so one design's later
    records never move another design's plan."""
    from stages.teacher_modes import tool_pool as TP
    protocol_path, priors_path = Path(protocol_path or PROTOCOL), Path(priors_path or PRIORS)
    import_paths = IMPORTS if import_paths is None else import_paths
    proto = protocol or json.loads(protocol_path.read_text())
    bp = proto["bounded_plan"]
    shares = design_shares()
    man, splits = TP.load_pool(Path(pool_dir))
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    #: the pilot's priors and import apply to the pool the pilot ran on, and to no other (a test pool, a rebuilt one)
    priors = json.loads(priors_path.read_text()) if priors_path.is_file() else None
    if priors and (priors.get("measured_by") or {}).get("pool_content_sha256") not in pool_contents(man):
        priors = None
    imports = import_records(pool_contents(man), tuple(Path(x) for x in import_paths), by_id=by_id)
    comp = Path(composition) if composition is not None and Path(composition).is_file() else None
    tshares = training_shares(comp) if comp else None
    wide = whole_supply_cells(comp) if comp else []
    groups = source_groups(comp) if comp else None
    lineages = {li: {"modes": lineage_cells(proto, shares, li, man, by_id, priors, [r for _, r in imports],
                                            training_shares=tshares, whole_supply=wide, source_groups=groups),
                     "splits": proto["lineages"][li]["splits"]} for li in sorted(proto["lineages"])}
    plan = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pool": {"dir": str(Path(pool_dir)), "content_sha256": man["content_sha256"],
                     "pool_json_sha256": sha_file(Path(pool_dir) / "pool.json")},
            "protocol_sha256": sha_file(protocol_path),
            **({"protocol_path": Path(protocol_path).resolve().relative_to(REPO).as_posix()}
               if Path(protocol_path).resolve() != PROTOCOL.resolve() else {}),
            "six_category_design_sha256": sha_file(SIX),
            "shares": shares, "rules": bp["rounds"],
            "common_support": ({"path": comp.resolve().relative_to(REPO).as_posix(), "sha256": sha_file(comp),
                                "training_shares": tshares, "whole_supply_cells": wide,
                                **({"source_groups": groups} if groups else {})} if tshares else None),
            "approved_shortfalls": list(bp.get("approved_shortfalls") or []),
            "priors": ({"path": priors_path.resolve().relative_to(REPO).as_posix(), "sha256": sha_file(priors_path)}
                       if priors else None),
            "imports": [{"path": p.resolve().relative_to(REPO).as_posix(), "sha256": sha_file(p), "lineage": r["lineage"],
                         "run_id": r["run_id"]} for p, r in imports],
            "lineages": lineages, "authorizes": "nothing"}
    plan["price"] = price_measured(plan, by_id, man, proto, priors) if priors else price(plan, by_id, man, proto)
    return plan


def verified(protocol: Mapping | None = None, pool_dir: Path | None = None, lineage: str | None = None,
             protocol_path: Path | None = None) -> tuple[dict, dict, dict]:
    """``(plan, pool manifest, tasks by id)`` after checking every identity the plan binds, for the session and for
    every dev-box consumer alike: `tool_pool.load_pool` re-verifies the split files, their task counts, sources,
    frozen orders and the pool's content sha256; the plan must name that content, the exact bytes of pool.json, the
    protocol file and the design (the shares) it is read with; and its cells must be the cells this pool, protocol
    and design produce now -- an edited target, bound, order or binding refuses (`PlanIdentityError`). ``lineage``
    restricts the reading and the cell check to that lineage's splits; ``protocol_path`` is the protocol file the
    plan must name (a design's own: its ``pool.dir`` and ``pool.plan`` are read)."""
    from stages.teacher_modes import tool_pool as TP
    protocol_path = Path(protocol_path or PROTOCOL)
    proto = protocol or json.loads(protocol_path.read_text())
    pool_dir = Path(pool_dir) if pool_dir else REPO / proto["pool"]["dir"]
    splits_wanted = proto["lineages"][lineage]["splits"] if lineage else None
    man, splits = TP.load_pool(pool_dir, splits_wanted)
    plan = json.loads((REPO / proto["pool"]["plan"]).read_text())
    bad = []
    if plan.get("schema") != SCHEMA:
        bad.append("not a tool generation plan")
    pool = plan.get("pool") or {}
    if pool.get("content_sha256") != man["content_sha256"]:
        bad.append("built from another pool content")
    if pool.get("pool_json_sha256") != sha_file(pool_dir / "pool.json"):
        bad.append("built from another pool.json (its orders, files or sources differ)")
    if plan.get("protocol_sha256") != sha_file(protocol_path):
        bad.append("built under another protocol")
    if plan.get("six_category_design_sha256") != sha_file(SIX) or plan.get("shares") != design_shares():
        bad.append("built under other category shares")
    if plan.get("approved_shortfalls") != list(proto["bounded_plan"].get("approved_shortfalls") or []):
        bad.append("its approved shortfalls are not the protocol's")
    cs = plan.get("common_support")
    if cs and (not (REPO / cs["path"]).is_file() or sha_file(REPO / cs["path"]) != cs["sha256"]
               or cs["training_shares"] != training_shares(REPO / cs["path"])
               or cs.get("source_groups") != source_groups(REPO / cs["path"])
               or cs["whole_supply_cells"] != whole_supply_cells(REPO / cs["path"])):
        bad.append("built under another common-support composition")
    bound_records = {}
    refs = [("priors", plan.get("priors"))] + [("import", r) for r in plan.get("imports") or []]
    for key, ref in refs:
        if ref:
            path = REPO / ref["path"]
            if not path.is_file() or sha_file(path) != ref["sha256"]:
                bad.append(f"its {key} record {ref['path']} is not the one it was built with")
            else:
                bound_records.setdefault(key, []).append(json.loads(path.read_text()))
    if not bad:
        by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
        for li in ([lineage] if lineage else sorted(proto["lineages"])):
            now = lineage_cells(proto, plan["shares"], li, man, by_id, (bound_records.get("priors") or [None])[0],
                                bound_records.get("import") or [],
                                training_shares=(cs or {}).get("training_shares"),
                                whole_supply=(cs or {}).get("whole_supply_cells") or (),
                                source_groups=(cs or {}).get("source_groups"))
            if json.dumps(now, sort_keys=True) != json.dumps((plan.get("lineages") or {}).get(li, {}).get("modes"),
                                                             sort_keys=True):
                bad.append(f"its {li} cells are not the cells this pool, protocol and design produce")
    if not bad:
        for li in ([lineage] if lineage else sorted(proto["lineages"])):
            tf = (plan["lineages"].get(li) or {}).get("tasks_file")
            if tf:
                want = sorted({t for m in plan["lineages"][li]["modes"].values() for c in m["cells"] for t in c["task_ids"]})
                path = REPO / tf["path"]
                if not path.is_file() or sha_file(path) != tf["sha256"]:
                    bad.append(f"its {li} task file {tf['path']} is not the one it was written with")
                elif [json.loads(x)["task_id"] for x in path.read_text().splitlines()] != want:
                    bad.append(f"its {li} task file does not hold exactly the cells' tasks")
    if bad:
        raise PlanIdentityError("the generation plan refuses: " + "; ".join(bad))
    return plan, man, by_id


#: where a lineage's bound task file goes (gitignored artifacts; the plan binds it by sha256)
TASKS_DIR = REPO / "artifacts/stages/stage-2/teacher_modes/tool_generation"


def bound_tasks_file(plan: dict, by_id: Mapping, lineage: str, out_dir: Path = TASKS_DIR) -> dict:
    """Write ``<out_dir>/<lineage>/tasks.jsonl``: every task some cell of the lineage may attempt (its ``task_ids``),
    once, sorted by id, as the pool holds it -- the session's whole input (P8.4) -- and record it in the plan."""
    ids = sorted({t for m in plan["lineages"][lineage]["modes"].values() for c in m["cells"] for t in c["task_ids"]})
    path = Path(out_dir) / lineage / "tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(by_id[t], sort_keys=True, ensure_ascii=False) + "\n" for t in ids))
    ref = {"path": path.relative_to(REPO).as_posix() if REPO in path.resolve().parents else str(path),
           "sha256": sha_file(path), "tasks": len(ids), "bytes": path.stat().st_size}
    plan["lineages"][lineage]["tasks_file"] = ref
    return ref


def verified_bound(protocol: Mapping | None = None, lineage: str = "hybrid", plan_path: Path | None = None,
                   protocol_path: Path | None = None) -> tuple[dict, dict]:
    """``(plan, tasks by id)`` on the pod, which holds the lineage's bound task file and not the pool: the plan (bound
    by the session's artifact) must be this schema, name this protocol and design, and the task file must hash to the
    plan's sha256 and hold exactly its cells' ``task_ids``. The dev box ran `verified` against the pool at issue."""
    protocol_path = Path(protocol_path or PROTOCOL)
    proto = protocol or json.loads(protocol_path.read_text())
    plan = json.loads(Path(plan_path or REPO / proto["pool"]["plan"]).read_text())
    bad = []
    if plan.get("schema") != SCHEMA:
        bad.append("not a tool generation plan")
    if plan.get("protocol_sha256") != sha_file(protocol_path):
        bad.append("built under another protocol")
    if plan.get("six_category_design_sha256") != sha_file(SIX):
        bad.append("built under other category shares")
    cs = plan.get("common_support")
    if cs and (not (REPO / cs["path"]).is_file() or sha_file(REPO / cs["path"]) != cs["sha256"]):
        bad.append("built under another common-support composition")
    tf = (plan.get("lineages", {}).get(lineage) or {}).get("tasks_file")
    if not tf:
        raise PlanIdentityError(f"the plan writes no task file for {lineage}")
    path = REPO / tf["path"]
    if not path.is_file() or sha_file(path) != tf["sha256"]:
        bad.append(f"the {lineage} task file is not the plan's")
    if bad:
        raise PlanIdentityError("the generation plan refuses: " + "; ".join(bad))
    by_id = {}
    for line in path.read_text().splitlines():
        t = json.loads(line)
        by_id[t["task_id"]] = t
    want = {t for m in plan["lineages"][lineage]["modes"].values() for c in m["cells"] for t in c["task_ids"]}
    if set(by_id) != want:
        raise PlanIdentityError("the generation plan refuses: its task file does not hold exactly the cells' tasks")
    return plan, by_id


#: per price level: (the task count used, the prior index for tokens per turn, for the pass rate, for the throughput
#: ranges). Lower and expected are consistent worlds; upper bounds: the cells' bound prefixes, long turns, the low
#: pass rate (more second candidates), the slow end of every throughput range and every round used.
LEVELS = {"lower": ("lower", 2, 2, 2), "expected": ("expected", 1, 1, 1), "upper": ("bound", 2, 0, 0)}


def price(plan: Mapping, by_id: Mapping, man: Mapping, proto: Mapping, *, rounds: int | None = None) -> dict:
    """lower / expected / upper per lineage session (USD), from the plan's task counts and the protocol's terms. A
    cell may name its tasks (``task_ids``, a pilot's sample) instead of a prefix count; ``rounds`` fixes the engine
    invocations' rounds for every level (a pilot runs one)."""
    bp = proto["bounded_plan"]
    tp = bp["throughput"]
    cand = int(proto["execution"]["candidates"])
    rate = float(bp["rate_usd_per_hour"])
    out = {}
    for lineage, li in plan["lineages"].items():
        row = {}
        for lvl, (count, i_tpt, i_p, i_tp) in LEVELS.items():
            hours = (tp["setup_minutes"] + tp["teardown_minutes"]) / 60
            gen_tokens = prefill = 0.0
            loads = 0
            for mode, m in li["modes"].items():
                tpt = bp["priors"]["supervised_tokens_per_turn"][mode][i_tpt]
                p = bp["priors"]["episode_pass_rate"][i_p]
                retry = 1.0 + (1.0 - p) * (cand - 1)
                mode_tokens = mode_prefill = 0.0
                for c in m["cells"]:
                    n = c["bound"] if count == "bound" else c["tasks"][count]
                    tasks = ([by_id[t] for t in c["task_ids"]] if "task_ids" in c else
                             cell_order(man["orders"].get(c["split"], []), by_id, c["category"], c.get("datasets"))[:n])
                    mode_tokens += sum(expected_turns(t) for t in tasks) * tpt * retry
                    mode_prefill += sum((t.get("first_prompt_tokens") or tp["prompt_tokens_default"])
                                        * expected_turns(t) for t in tasks) * retry
                decode_tps = tp["decode_tokens_per_s"]["thinking" if mode in tp["thinking_modes"] else "non_thinking"]
                hours += mode_tokens / (decode_tps * tp["tool_workload_efficiency"][i_tp]) / 3600
                hours += mode_prefill / tp["prefill_tokens_per_s"][i_tp] / 3600
                r = rounds or (bp["rounds"]["max_rounds"] if count == "bound" else bp["rounds"]["expected_rounds"][count])
                loads += r * cand
                gen_tokens += mode_tokens
                prefill += mode_prefill
            hours += loads * tp["model_load_minutes"][i_tp] / 60
            row[lvl] = {"usd": round(hours * rate, 2), "hours": round(hours, 2), "generated_tokens": int(gen_tokens),
                        "prefill_tokens": int(prefill), "engine_invocations": loads}
        out[lineage] = row
    tot = {lvl: round(sum(out[li][lvl]["usd"] for li in out), 2) for lvl in LEVELS}
    return {"sessions": out, "total_usd": tot, "proposed_hard_cap_usd": round(tot["upper"] * 1.1, 2),
            "rate_usd_per_hour": rate, "levels": {k: list(v) for k, v in LEVELS.items()},
            "basis": "plan task counts (expected / bound) x expected turns x the mode's tokens per turn x (1 + the "
                     "second candidates' share); decode at the supplement's measured throughput x a tool-workload "
                     "efficiency; prefill of each turn's prompt (first prompt x turns; prefix caching would lower "
                     "it); a model load per engine invocation (rounds x candidates per mode); setup and teardown"}


#: measured session overheads (a_tool_pilot_20261104_164531: 2.1 min to SSH, 2.6 min setup, 2.1 min closeout and
#: teardown) at the expected level; the runner's own setup and teardown allowances at the upper level
OVERHEAD_MINUTES = {"expected": 10.0, "upper": 35.0}


def task_generated_tokens(task: Mapping, *, lineage: str, mode: str, proto: Mapping, priors: Mapping | None
                          ) -> tuple[float, str]:
    """``(generated tokens, basis)`` ONE attempted task costs, every candidate's turns and reasoning included: the
    pilot's stratum (measured, at its measured second-candidate rate), the training split's (proxy), else the
    protocol priors (expected turns x tokens per turn x the expected second-candidate share)."""
    from stages.teacher_modes import tool_episodes as TE
    strata = (priors or {}).get("strata") or {}
    shape = TE.episode_shape(task)
    for key, basis in stratum_candidates(lineage, mode, task):
        v = strata.get(key)
        if v and v.get("generated_tokens_per_attempted_task"):
            return float(v["generated_tokens_per_attempted_task"]), basis
    if task.get("category") is None or task["split"] == "validation":
        rows = [v for k, v in strata.items() if k.startswith(f"{lineage}|{mode}|training|") and k.endswith("|" + shape)]
        if rows:
            return (sum(v["generated_tokens_per_attempted_task"] * v["attempted"] for v in rows)
                    / sum(v["attempted"] for v in rows)), "proxy:mode"
    bp, cand = proto["bounded_plan"], int(proto["execution"]["candidates"])
    tpt, pr = bp["priors"]["supervised_tokens_per_turn"][mode][1], bp["priors"]["episode_pass_rate"][1]
    return expected_turns(task) * tpt * (1.0 + (1.0 - pr) * (cand - 1)), "prior"


def runner_all_in(driver_minutes: float, rate: float, kind: str = "tool_generation") -> dict:
    """The shared runner's own plan for a session of ``kind`` whose driver needs ``driver_minutes``
    (`study_engineering.session_spec` -> `BudgetSpec.plan`: setup, transfer, the driver, contingency, the closeout
    reserve, the recovery reserve) and its all-in at ``rate`` (`study_engineering.all_in`: GPU and container disk to
    the hard deadline plus the watchdog's time to act): what a launch's ``all_in_within_cap`` refuses on."""
    from types import SimpleNamespace

    from shared import study_engineering as SE
    from stages.teacher_modes import token_matched_phase as TM
    spec = TM.launch_spec(kind, driver_minutes)
    sess = SE.session_spec(TM.STUDY, spec, SimpleNamespace(run_id="price_check", dry_run=False))
    plan = sess.budget.plan(price_per_hour=rate, authorized_usd=10 ** 6)
    b = SE.all_in(plan, rate, spec.disk_gb)
    return {"driver_minutes": round(driver_minutes, 1), "hard_terminate_minutes": round(plan.hard_terminate_minutes, 1),
            "rate_usd_per_hour": rate, "all_in_usd": b["all_in_usd"]}


def price_measured(plan: Mapping, by_id: Mapping, man: Mapping, proto: Mapping, priors: Mapping) -> dict:
    """lower / expected / upper per lineage session from what the pilot measured (module docstring's priors
    replaced): per NEW task its stratum's generated tokens over every candidate, at its measured second-candidate
    rate; the mode's measured output tokens per second (the slowest invocation at the upper level); a measured engine
    start per invocation (two per round and mode: first and second candidates); the measured session overhead
    (expected) or the runner's allowance (upper); $1.09/h measured, $1.19/h (the grant's ceiling) at the upper level.
    A stratum the pilot did not measure is priced by its proxy or the protocol prior and counted apart; Thinking-2507
    has no measured stratum or throughput (protocol terms, ``prior``)."""
    bp, tp = proto["bounded_plan"], proto["bounded_plan"]["throughput"]
    cand = int(proto["execution"]["candidates"])
    thr = (priors or {}).get("throughput") or {}
    levels = {"lower": ("lower", 1), "expected": ("expected", bp["rounds"]["expected_rounds"]["expected"]),
              "upper": ("bound", bp["rounds"]["max_rounds"])}
    out = {}
    for lineage, li in plan["lineages"].items():
        row = {}
        for lvl, (count, rounds) in levels.items():
            minutes = OVERHEAD_MINUTES["upper" if lvl == "upper" else "expected"]
            gen = 0.0
            by_basis: dict[str, float] = {}
            starts = 0.0
            for mode, m in li["modes"].items():
                mode_tokens = 0.0
                from stages.teacher_modes import tool_generation_import as TGI
                imported = set(TGI.merged([json.loads((REPO / r["path"]).read_text()) for r in plan.get("imports") or []],
                                          lineage, mode))
                for c in m["cells"]:
                    n = c["bound"] if count == "bound" or (c.get("first_round") == "bound" and count != "lower") \
                        else c["tasks"][count]
                    new = [t for t in cell_order(man["orders"].get(c["split"], []), by_id, c["category"],
                                                 c.get("datasets"))
                           if t["task_id"] not in imported][:n]
                    for t in new:
                        g, basis = task_generated_tokens(t, lineage=lineage, mode=mode, proto=proto, priors=priors)
                        mode_tokens += g
                        by_basis[basis] = by_basis.get(basis, 0.0) + g
                t_m = thr.get(mode)
                if t_m:
                    tps = float(t_m["slowest_invocation_tokens_per_second"] if lvl == "upper" else t_m["output_tokens_per_second"])
                    first = float(t_m["engine_start_seconds"]["first"] or 0)
                    later = float(t_m["engine_start_seconds"]["later"] or first)
                    #: a pod's first engine start compiles; its later starts reuse that cache (measured on one pod)
                    starts += ((first + later * (rounds * cand - 1)) if lvl == "upper" else later * rounds * cand) / 60
                else:
                    kind = "thinking" if mode in tp["thinking_modes"] else "non_thinking"
                    eff = tp["tool_workload_efficiency"][{"lower": 2, "expected": 1, "upper": 0}[lvl]]
                    tps = tp["decode_tokens_per_s"][kind] * eff
                    starts += tp["model_load_minutes"][{"lower": 2, "expected": 1, "upper": 0}[lvl]] * rounds * cand
                minutes += mode_tokens / tps / 60
                gen += mode_tokens
            minutes += starts
            rate = RATE_UPPER if lvl == "upper" else RATE_EXPECTED
            row[lvl] = {"usd": round(minutes / 60 * rate, 2), "minutes": round(minutes, 1),
                        "generated_tokens": int(gen), "generated_tokens_by_basis": {k: int(v) for k, v in sorted(by_basis.items())},
                        "rounds": rounds, "rate_usd_per_hour": rate}
        drive = row["upper"]["minutes"] - OVERHEAD_MINUTES["upper"]
        kind = "tool_generation" if lineage == "hybrid" else "tool_generation_t2507"
        row["runner_cap"] = runner_all_in(max(drive, 1.0), RATE_UPPER, kind)
        out[lineage] = row
    tot = {lvl: round(sum(out[li][lvl]["usd"] for li in out), 2) for lvl in levels}
    return {"sessions": out, "total_usd": tot,
            "proposed_hard_caps_usd": {li: out[li]["runner_cap"]["all_in_usd"] for li in out},
            "levels": {k: {"tasks": v[0], "rounds": v[1]} for k, v in levels.items()},
            "basis": "the pilot's measured generated tokens per attempted task (every candidate, its measured second-"
                     "candidate rate), measured output tokens per second and engine starts, measured session overhead; "
                     "the cap is the runner's own all-in at $1.19/h for the upper level's driver minutes; strata the "
                     "pilot did not measure are priced by proxy or prior and listed in generated_tokens_by_basis"}


#: The L40S pilot (tool_pilot.py; logs/budget/applications/research_a_tool_pilot_2026-11-04):
#: the hybrid lineage, both modes, one round (first candidates, then second candidates of refused tasks). Per mode:
#: the first ``training_per_category`` tasks of every training cell's frozen order (full episodes, the order's first
#: tier); the first ``single_step_training`` single-step tasks of the schema_arguments training cell; and EVERY
#: calibration and state-evaluation schema_arguments task of both shapes -- their supply is below every target, so
#: generating all of them under the unchanged verifier is what settles whether that category's remaining SHORTFALL is
#: real. Proposal parameters, not a plan term: nothing reads them to generate.
PILOT = {"lineage": "hybrid", "training_per_category": 24, "single_step_training": 100,
         "whole_cells": (("calibration", "schema_arguments"), ("state_eval", "schema_arguments")), "rounds": 1}


def pilot_sample(plan: Mapping, by_id: Mapping, man: Mapping, spec: Mapping = PILOT) -> dict:
    """``{mode: [cell with task_ids]}``: the pilot's tasks, each cell's in its frozen order."""
    from stages.teacher_modes import tool_episodes as TE
    out = {}
    for mode, m in plan["lineages"][spec["lineage"]]["modes"].items():
        cells = []
        for c in m["cells"]:
            order = cell_order(man["orders"].get(c["split"], []), by_id, c["category"], c.get("datasets"))
            if (c["split"], c["category"]) in {tuple(x) for x in spec["whole_cells"]}:
                ids = [t["task_id"] for t in order]
            elif c["split"] == "training":
                full = [t["task_id"] for t in order if TE.episode_shape(t) == TE.FULL_EPISODE]
                one = [t["task_id"] for t in order if TE.episode_shape(t) == TE.SINGLE_STEP]
                ids = full[:spec["training_per_category"]] + one[:spec["single_step_training"]]
            else:
                continue
            if ids:
                cells.append({**c, "task_ids": ids, "bound": len(ids), "tasks": dict.fromkeys(
                    ("lower", "expected", "upper"), len(ids)),
                              "shapes": {s: sum(TE.episode_shape(by_id[t]) == s for t in ids) for s in TE.EPISODE_SHAPES}})
        out[mode] = cells
    return out


def round_tasks(cell: Mapping, order_tasks: Sequence[Mapping], state: Mapping, round_no: int, *,
                bp: Mapping, mode: str, candidates: int) -> list[str]:
    """The task ids a cell attempts in round ``round_no`` (1-based), given its ``state`` after earlier rounds:
    ``{"attempted": int, "units": float, "admitted": int, "imported": int}`` -- ``attempted`` counts the NEW tasks of
    ``order_tasks`` (the frozen order without the imported ones), ``units`` and ``admitted`` include the imported
    tasks' outcomes, ``imported`` is how many there are. Empty when the cell is done, bounded or out of rounds."""
    if round_no > int(bp["rounds"]["max_rounds"]):
        return []
    attempted, units = int(state.get("attempted", 0)), float(state.get("units", 0.0))
    seen = attempted + int(state.get("imported", 0))
    done = units >= cell["target"] and int(state.get("admitted", 0)) >= cell.get("min_tasks", 0)
    room = min(int(cell["bound"]), len(order_tasks)) - attempted
    if done or room <= 0:
        return []
    if round_no == int(bp["rounds"]["max_rounds"]) and round_no > 1:
        return [t["task_id"] for t in order_tasks[attempted:attempted + room]]   #: the last round: the whole bound
    if round_no == 1:
        n = int(cell["bound"]) if cell.get("first_round") == "bound" else int(cell["tasks"]["expected"])
    else:
        rest = order_tasks[attempted:]
        if units > 0 and seen:
            y = units / seen
            n = math.ceil(max(cell["target"] - units, 0.0) / y * float(bp["rounds"]["top_up_margin"]))
        else:                                   #: nothing admitted yet: the pessimistic prefix of what is left
            pr, tpt = bp["priors"]["episode_pass_rate"][0], bp["priors"]["supervised_tokens_per_turn"][mode][0]
            n = prefix_reaching(rest, cell["target"] - units, cell["unit"], tpt, pr, candidates)
        if int(state.get("admitted", 0)) < cell.get("min_tasks", 0):
            n = max(n, cell["min_tasks"] - int(state.get("admitted", 0)))
    n = max(1, min(n, room))
    return [t["task_id"] for t in order_tasks[attempted:attempted + n]]


def select(order: Sequence[str], units: Mapping[str, float], target: float, min_tasks: int = 0
           ) -> tuple[list[str], float]:
    """A consumer's selection: admitted tasks (``units``) in the frozen ``order``, whole, until ``target`` units and
    ``min_tasks`` tasks -- the same rule the session stopped generation by."""
    out, got = [], 0.0
    for tid in order:
        if got >= target and len(out) >= min_tasks:
            break
        if tid in units:
            out.append(tid)
            got += units[tid]
    return out, got


def cells_for(plan: Mapping, lineage: str, mode: str, split: str) -> list[dict]:
    return [c for c in plan["lineages"][lineage]["modes"][mode]["cells"] if c["split"] == split]


def for_consumers(protocol: Mapping | None = None, pool_dir: Path | None = None, protocol_path: Path | None = None
                  ) -> tuple[dict, dict]:
    """``(plan, pool orders)`` for a dev-box consumer (admission's packs, packed inputs), through `verified`: the
    same identity checks the session runs before it generates."""
    plan, man, _ = verified(protocol, pool_dir, protocol_path=protocol_path)
    return plan, man["orders"]


def approved(plan: Mapping, split: str, mode: str, category: str | None) -> dict | None:
    """The maintainer's recorded approval of a cell's SHORTFALL (``bounded_plan.approved_shortfalls``), or None."""
    return next((a for a in plan.get("approved_shortfalls") or []
                 if (a.get("split"), a.get("mode"), a.get("category")) == (split, mode, category)), None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--composition", default=str(COMMON_SUPPORT),
                    help="the training composition file the plan binds ('none': the registered shares)")
    ap.add_argument("--no-task-files", action="store_true",
                    help="a dry plan: write no bound task file (the frozen plan's task files stay untouched)")
    ap.add_argument("--protocol", default=str(PROTOCOL), help="the generation protocol (a design's own file)")
    ap.add_argument("--priors", default=str(PRIORS), help="the measured priors the plan binds")
    ap.add_argument("--import-record", action="append", default=None,
                    help="an import record the plan binds (repeatable; default: IMPORTS)")
    a = ap.parse_args(argv)
    proto_path = Path(a.protocol).resolve()
    plan = build(Path(a.pool), composition=None if a.composition == "none" else Path(a.composition),
                 protocol_path=proto_path, priors_path=Path(a.priors).resolve(),
                 import_paths=[Path(x).resolve() for x in a.import_record] if a.import_record else IMPORTS)
    #: a design's protocol names its own task-file directory, so its plan never overwrites another plan's files
    tasks_dir = REPO / (json.loads(proto_path.read_text())["pool"].get("tasks_dir") or TASKS_DIR.relative_to(REPO))
    summary = {li: {m: {f"{c['split']}/{c['category']}": (c["supply"], c["tasks"]["expected"], c["bound"], c.get("status"))
                        for c in mm["cells"]} for m, mm in v["modes"].items()} for li, v in plan["lineages"].items()}
    if a.write:
        from stages.teacher_modes import tool_pool as TP
        if not a.no_task_files:
            _, splits = TP.load_pool(Path(a.pool))
            by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
            for li in plan["lineages"]:
                bound_tasks_file(plan, by_id, li, tasks_dir)
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"price": plan["price"]["total_usd"], "caps": plan["price"].get("proposed_hard_caps_usd"),
                      "cells": summary}, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

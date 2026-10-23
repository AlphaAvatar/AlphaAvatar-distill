#!/usr/bin/env python3
"""A corpus candidate on the formal ladder at $0: build it, or say exactly why not.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.corpus_feasibility \
        --candidate configs/stages/stage-2/reasoning_data/corpus_v2_candidate.json \
        --run-id h1gen_20261022_190653 --store <scr>/store \
        --pack-out artifacts/stages/stage-2/reasoning_data/packs/B_v2 \
        [--plan-2 configs/stages/stage-2/reasoning_data/corpus_v3_proposal.json]

1. DATA. P1's accepted units come from `p1v.verify_run` (the shards re-hashed,
   every record re-verified and selected); the reuse comes from its sha-checked
   selection. Nothing else enters, and a session in a cell the candidate does
   not allocate refuses (`build_arm_b.place`).
2. THE FORMAL PATH. `build_arm_b.build` -> `build_master_ladder.build`: whole
   units rendered at their real token lengths, the reserve drawn, the nested
   order, the rungs cut WITHOUT composition drift, packed best-fit. A refusal
   is the verdict. Drift is never allowed and the salt is never varied.
3. THE CONTRACT, reported whether or not the candidate builds:
   - each cell's supply against its share of 2.96M plus the reserve;
   - where each cell runs dry in the training order, and which rungs that
     blocks;
   - the validation tail's composition and families;
   - unit independence: one unit per task, no repeated prompt, no repeated
     target;
   - training/validation disjointness, by unit and by prompt;
   - each cell's families;
   - R0-R9 for the generated units, re-run with their teacher text, and the
     reuse selection's own isolation record.
4. A BUILT CANDIDATE gets its own pack, beside @v1's and never over it,
   hash-bound to the candidate file, P1's launch identity, the sessions and
   the reuse. The trainer's reader must take it back. Adoption stays the
   maintainer's.
5. PLAN 2 (`--plan-2`, only when the candidate does not build).
   - The same ladder runs on the real units, plus stand-ins for the units a
     top-up would add to the cells the proposal can fill only with new data.
     Stand-ins are sized as that cell's measured accepted units.
   - The fewest stand-ins with which the ladder builds is the minimum. It is
     turned into tasks at the measured acceptance, and into an application
     priced from P1's measured session (`plans/abstract_topup_application.json`).
   - A stand-in is a forecast. It is never a pack and never data.

Writes ``logs/stages/stage-2/reasoning_data/audit/corpus_<version>_feasibility.json``.
Exits 0 when the candidate built, 3 when it did not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_arm_b as AB
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import generation_plan as GP
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import p1v
from stages.reasoning_data import source_plan as SP

from aadistill.data.nested_ladder import LadderError, LadderUnit, build_ladder
from maintenance.consolidation import provider_billing as PB

TARGET = 2_960_000
APPLICATION = A.LOG_DIR / "plans/abstract_topup_application.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_key(session: dict) -> str:
    return _sha(next(m["content"] for m in session["messages"] if m["role"] == "user"))


def rung_reach(order: list[LadderUnit], targets: list[int]) -> dict[int, int | None]:
    """The size of the shortest prefix reaching each target (cut_rungs' rule)."""
    out, cum, n = {}, 0, 0
    for t in targets:
        while n < len(order) and cum < t:
            cum += order[n].size
            n += 1
        out[t] = cum if cum >= t else None
    return out


def contract(units: list[LadderUnit], shares: dict[str, float], *, salt: str, reserve: int,
             targets: list[int]) -> dict:
    """What the formal ladder does with these units, read before any rung is
    cut: the same reserve and order `build_ladder` cuts from, with no rung and
    no drift. A cell blocks a rung when it runs dry below that rung's prefix --
    exactly the refusal `cut_rungs` raises."""
    lad = build_ladder(units, shares, [], salt=salt, reserve=reserve)
    byid = {u.unit_id: u for u in units}
    order = [byid[i] for i in lad["order"]]
    reserved = [byid[i] for i in lad["reserved"]]
    reach = rung_reach(order, targets)
    left, cum, dry = Counter(u.stratum for u in order), 0, {}
    for u in order:
        left[u.stratum] -= 1
        cum += u.size
        if left[u.stratum] == 0:
            dry[u.stratum] = cum
    top = max(targets) + reserve
    cells = {}
    for c, s in sorted(shares.items()):
        mine = [u for u in units if u.stratum == c]
        res = [u for u in reserved if u.stratum == c]
        supply = sum(u.size for u in mine)
        blocks = [t for t in targets if reach[t] is not None and c in dry and dry[c] < reach[t]]
        cells[c] = {"share": s, "units": len(mine), "supervised_tokens": supply,
                    "supervised_tokens_needed_with_reserve": round(top * s),
                    "covered": supply >= top * s,
                    "families": len({u.family for u in mine}),
                    "family_ids": sorted({u.family for u in mine if u.family})[:12],
                    "reserve_units": len(res), "reserve_supervised_tokens": sum(u.size for u in res),
                    "reserve_families": len({u.family for u in res}),
                    "runs_dry_at_prefix": dry.get(c), "blocks_rungs": blocks}
    absent = sorted(c for c, v in cells.items() if not v["reserve_units"])
    return {"composition_capacity": lad["composition_capacity"],
            "rung_prefix_sizes": {str(t): reach[t] for t in targets},
            "cells": cells,
            "validation_tail": {"units": len(reserved), "supervised_tokens": sum(u.size for u in reserved),
                                "cells_present": len(cells) - len(absent), "cells_absent": absent,
                                "families": len({(u.stratum, u.family) for u in reserved}),
                                "max_share_deviation": (lad["reserved_accounting"] or {}).get("max_share_deviation")},
            "_reading": ("runs_dry_at_prefix: the training-order prefix size at which the cell's last unit is "
                         "placed; past it the interleave cannot serve the cell, so a rung whose prefix is larger "
                         "is refused (nested_ladder.composition_capacity). A one-unit cell runs dry where its "
                         "unit is placed, so it either sits in a rung and caps the capacity there, or sits in "
                         "none; no order or salt satisfies both.")}, lad


def independence(sessions: list[dict], reserved: set[str]) -> dict:
    """One unit per task, no prompt or target shared by two units, and no
    prompt shared by the validation tail and the training units."""
    prompts: dict[str, set] = defaultdict(set)
    targets: dict[str, set] = defaultdict(set)
    for s in sessions:
        prompts[prompt_key(s)].add(s["unit_id"])
        targets[json.dumps(s.get("target"), sort_keys=True)].add(s["unit_id"])
    shared_prompt = [sorted(v) for v in prompts.values() if len(v) > 1]
    shared_target = [sorted(v) for k, v in targets.items() if len(v) > 1 and k != "null"]
    val_keys = {k for k, v in prompts.items() if v & reserved}
    train_keys = {k for k, v in prompts.items() if v - reserved}
    return {"units": len({s["unit_id"] for s in sessions}),
            "units_sharing_a_prompt": sum(len(v) for v in shared_prompt), "examples": shared_prompt[:5],
            "units_sharing_a_target": sum(len(v) for v in shared_target),
            "units_without_target": len(targets.get("null", ())),
            "validation_prompts_also_in_training": len(val_keys & train_keys)}


def attempt(sessions: list[dict], spec: dict, shares: dict[str, float], tok) -> dict:
    try:
        built, ladder_spec = AB.build(sessions, spec, shares, spec["rung_targets"], tok)
    except LadderError as e:
        return {"built": False, "refusal": str(e)}
    return {"built": True, "result": built, "ladder_spec": ladder_spec}


def unsatisfied(con: dict) -> list[str]:
    out = []
    for c, v in con["cells"].items():
        if not v["share"]:
            continue
        if v["blocks_rungs"]:
            out.append(f"{c}: {v['units']} unit(s), {v['supervised_tokens']} supervised; runs dry at prefix "
                       f"{v['runs_dry_at_prefix']}, below rungs {v['blocks_rungs']} (composition capacity "
                       f"{con['composition_capacity']}); the formal path refuses without drift")
        if not v["covered"]:
            out.append(f"{c}: {v['supervised_tokens']} supervised of the {v['supervised_tokens_needed_with_reserve']} "
                       f"its share of 2.96M + the reserve needs")
        if v["units"] == 1:
            out.append(f"{c}: ONE unit from one task ({', '.join(v['family_ids'])}): it cannot be both trained "
                       f"on and in the validation tail (reserve units: {v['reserve_units']}), and one task is "
                       "not source coverage")
    return out


# --- plan 2 -------------------------------------------------------------------

def stand_ins(cell: str, measured: list[LadderUnit], k: int, *, order: str) -> list[LadderUnit]:
    """k forecast units for `cell`, sized as its measured accepted units:
    ``ascending`` cycles the measured sizes smallest first (the conservative
    bound on how many units are needed), ``median`` repeats the median size.
    Families cycle over the measured ones, so the reserve's family rule acts on
    them as it would on real units."""
    sizes = sorted(u.size for u in measured)
    fams = sorted({u.family for u in measured if u.family})
    med = int(statistics.median(sizes))
    return [LadderUnit(unit_id=f"standin/{cell}/{i}", stratum=cell,
                       size=sizes[i % len(sizes)] if order == "ascending" else med,
                       family=fams[i % len(fams)] if fams else None)
            for i in range(k)]


def with_stand_ins(units: list[LadderUnit], k: dict[str, int], *, order: str) -> list[LadderUnit]:
    return units + [x for c, n in sorted(k.items())
                    for x in stand_ins(c, [u for u in units if u.stratum == c], n, order=order)]


def builds(units: list[LadderUnit], shares: dict, *, salt: str, reserve: int, targets: list[int]) -> bool:
    try:
        lad = build_ladder(units, shares, targets, salt=salt, reserve=reserve)
    except LadderError:
        return False
    return all(r["reachable"] for r in lad["rungs"])


def minimum_new_units(units: list[LadderUnit], shares: dict, cells: list[str], *, salt: str, reserve: int,
                      targets: list[int], order: str, limit: int = 400) -> dict[str, int] | None:
    """The fewest stand-ins per fill cell with which the formal ladder cuts
    every rung without drift: each cell's count is raised one at a time until
    that cell neither falls short nor runs dry below a rung, with the other
    cells at their current counts, and the sweep repeats until no count moves.
    None when a count passes `limit` or the whole ladder still refuses."""
    k = {c: 0 for c in cells}
    for _ in range(10):
        moved = False
        for c in cells:
            while True:
                con, _ = contract(with_stand_ins(units, k, order=order), shares, salt=salt, reserve=reserve,
                                  targets=targets)
                v = con["cells"][c]
                if v["covered"] and not v["blocks_rungs"]:
                    break
                k[c] += 1
                moved = True
                if k[c] > limit:
                    return None
        if not moved:
            break
    ok = builds(with_stand_ins(units, k, order=order), shares, salt=salt, reserve=reserve, targets=targets)
    return k if ok else None


def planning_units(units: list[LadderUnit], shares: dict, k_min: dict[str, int], *, prefix_top: int,
                   reserve: int) -> dict[str, int]:
    """The robust count a top-up should aim at, per cell: the fewest (at least
    the mechanism minimum, sized smallest first) for which the cell holds its
    share of the top rung's prefix plus the reserve AND two of its largest
    measured units -- one the reserve may draw beyond its share, one the order
    may place past the prefix. Real units have other sizes and other ids
    (other positions) than the stand-ins, so the bare minimum can miss by one."""
    out = {}
    for c, k in k_min.items():
        measured = [u for u in units if u.stratum == c]
        need = shares[c] * (prefix_top + reserve) + 2 * max(u.size for u in measured)
        have = sum(u.size for u in measured)
        sizes = sorted(u.size for u in measured)
        n = 0
        while have + sum(sizes[i % len(sizes)] for i in range(n)) < need or n < k:
            n += 1
        out[c] = n
    return out


def tasks_for(units: int, q: float, *, oversupply: float = 1.5, confidence: float = 0.9) -> dict:
    """Tasks to generate for `units` accepted units at unit rate q: the larger
    of generation_plan's measured-rate oversupply and the fewest tasks whose
    binomial chance of yielding at least `units` reaches `confidence`."""
    over = math.ceil(units / q * oversupply)
    tail = lambda n: sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(units, n + 1))
    n = units
    while tail(n) < confidence:
        n += 1
    return {"tasks": max(over, n), "by_oversupply": over, f"by_binomial_p{int(confidence * 100)}": n,
            "chance_of_enough_units": round(tail(max(over, n)), 3)}


def generation_measured(recs: list[dict], tasks: dict, got: dict, cell: str, spec: dict) -> dict:
    """The cell's measured acceptance and generated tokens in P1, from the
    records and the adapter's verdicts, overall and per generation group
    (`generation_plan.group_of`: module or family, and level)."""
    in_cell = {tid for tid, t in tasks.items() if p1v.cell_of(t, spec) == cell}
    gen = Counter()
    for r in recs:
        if r["task_id"] in in_cell:
            gen[r["candidate_index"]] += sum(t["generated_tokens"] for t in r["turns"])
    cand, acc = Counter(), Counter()
    group_tasks, group_units = Counter(), Counter()
    for c in got["candidates"]:
        if c.unit_id in in_cell:
            cand[c.candidate_index] += 1
            acc[c.candidate_index] += bool(c.accepted_for_training)
            g = GP.group_of(tasks[c.unit_id])
            group_tasks[g] += c.candidate_index == 0
            group_units[g] += bool(c.accepted_for_training)
    tasks_n = cand[0]
    return {"tasks_generated": tasks_n,
            "candidates": dict(cand), "accepted": dict(acc),
            "candidate_accept_rate": {k: round(acc[k] / cand[k], 4) for k in cand},
            "unit_accept_rate": round(sum(acc.values()) / tasks_n, 4),
            "units_by_group": {g: f"{group_units[g]}/{group_tasks[g]}" for g in sorted(group_tasks)},
            "generated_tokens": dict(gen),
            "generated_tokens_per_task": round(sum(gen.values()) / tasks_n)}


def by_group_expectation(selected: list[dict], gen: dict, k: int) -> dict:
    """Units the actual round-robin selection would yield at P1's per-group
    rates (small samples), and the chance of at least k (Poisson binomial)."""
    rate = {g: int(v.split("/")[0]) / int(v.split("/")[1]) for g, v in gen["units_by_group"].items()}
    ps = [rate.get(GP.group_of(t), gen["unit_accept_rate"]) for t in selected]
    dist = [1.0]
    for p in ps:
        dist = [a * (1 - p) + (dist[i - 1] * p if i else 0) for i, a in enumerate(dist + [0.0])]
    return {"expected_units": round(sum(ps), 1), "chance_of_at_least": {str(k): round(sum(dist[k:]), 3)},
            "groups_selected": dict(sorted(Counter(GP.group_of(t) for t in selected).items())),
            "_": "P1's per-group rates rest on 4-8 tasks per group; a group P1 never drew takes the cell's rate"}


def decode_speed(recs: list[dict]) -> float:
    """P1's median per-sequence decode speed (tokens/s) over every turn above
    1,000 tokens, measured under P1's full load; a top-up runs far fewer
    sequences at once, so this bounds its speed from below."""
    return round(statistics.median(t["generated_tokens"] / (t["seconds"] - t["ttft_seconds"])
                                   for r in recs for t in r["turns"]
                                   if t["generated_tokens"] > 1000 and t["seconds"] > t["ttft_seconds"]), 1)


def session_measured(run_id: str, store: Path) -> dict:
    """P1's measured session phases: pod to driver, engine start, teardown."""
    tl = A.read_json(A.LOG_DIR / f"runs/{run_id}/runtime/session.json")["timeline"]
    at = lambda pat: next(e["elapsed_min"] for e in tl if re.search(pat, e["msg"]))
    inits = []
    for log in sorted(store.glob("extracted/audit/*/*.seg0.log")):
        m = re.search(r"init engine .*? took ([\d.]+) s", log.read_text(errors="replace"))
        if m:
            inits.append(float(m.group(1)))
    out = A.read_json(A.LOG_DIR / f"runs/{run_id}/closeout/outcome.json")["money"]
    return {"pod_to_driver_minutes": at(r"^driver detached"),
            "all_done_to_pod_deleted_minutes": round(at(r"pod deleted") - at(r"polling ended: ALL_DONE"), 2),
            "engine_start_seconds_by_pass": inits,
            "price_per_hour": out["price_per_hour"],
            "all_in_over_gpu": round(out["all_in_usd"] / out["gpu_usd"], 5),
            "source": f"runs/{run_id}/runtime/session.json timeline; the generators' logs in the run's store; "
                      f"runs/{run_id}/closeout/outcome.json :: money"}


def provider_billed(snap: Path) -> float:
    """All-time provider billing in a committed read-only snapshot (pods + volumes)."""
    s = json.loads(snap.read_text())
    return round(sum(p["usd"] for p in PB.billed_by_pod(s).values()) + sum(PB.volumes_by_day(s).values()), 4)


def topup_application(*, rows: dict, sess: dict, speed: float, limit: int, proposal: dict, run_id: str) -> dict:
    """One A100 session for every fill cell's tasks, priced from P1's measured
    session: setup and teardown as measured, two passes each with the slowest
    measured engine start and the longest possible candidate at the decode
    speed bound; P1's hard-cap rule (1.5 x expected + closeout reserve)."""
    grant = A.read_json(A.REPO / "logs/budget/approvals/h1_p1_generation_grant.json")
    snap = sorted((A.REPO / "logs/budget/provider").glob("runpod_billing_*.json"))[-1]
    billed = provider_billed(snap)
    per_pass_s = limit / speed
    engine_s = max(sess["engine_start_seconds_by_pass"] or [180.0])
    minutes = (sess["pod_to_driver_minutes"] + 2 * (engine_s + per_pass_s) / 60
               + sess["all_done_to_pod_deleted_minutes"])
    rate = sess["price_per_hour"] * sess["all_in_over_gpu"] / 60
    expected = minutes * rate
    reserve = (sess["pod_to_driver_minutes"] + sess["all_done_to_pod_deleted_minutes"] + 15) * rate
    hard = 1.5 * (expected + reserve)
    cap = grant["project_cumulative_cap_usd"]
    total_tasks = sum(r["tasks"]["tasks"] for r in rows.values())
    doc = {
        "schema": "aadistill.stage2.reasoning_data.topup_application/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "APPLICATION for the maintainer's decision; NOT authorized, nothing launched",
        "route": (f"plan 2: analogy kept at @v1's 0.09 with the minimum new teacher data (corpus "
                  f"{proposal['corpus_id']}, {proposal['candidate']['path']})"),
        "what": (f"P1t: ONE A100-80GB session generating teacher candidates for {total_tasks} unused tasks in "
                 f"{len(rows)} cells under P1's frozen generation protocol (two candidates, the second only when "
                 "the first is rejected; the same teacher, sampling, allowance and verifiers), then the same P1v "
                 "verification and the formal ladder at $0"),
        "cells": rows,
        "tasks_total": total_tasks,
        "predicted_generated_tokens": sum(r["predicted_generated_tokens"] for r in rows.values()),
        "_prediction": ("measured generated tokens per task in each cell (both candidates) x its tasks; a "
                        "PREDICTION, not a limit: the hard bound is the cap, enforced by the watchdog"),
        "cost": {"measured_basis": sess,
                 "decode_tokens_per_second_per_sequence": speed,
                 "_decode": ("P1's median per-sequence decode speed under its full load (decode_speed); a "
                             "top-up runs far fewer sequences at once, so the bound is conservative"),
                 "per_pass_bound_seconds": round(per_pass_s),
                 "_per_pass": f"the longest possible candidate ({limit} tokens) at that speed",
                 "session_minutes_expected": round(minutes, 1),
                 "expected_usd": round(expected, 2),
                 "closeout_reserve_usd": round(reserve, 2),
                 "hard_cap_usd": round(hard, 2),
                 "_hard_cap": "1.5 x (expected + closeout reserve of setup + teardown + 15 min), P1's rule"},
        "budget": {"provider_snapshot": {"path": SP.repo_path(snap), "sha256": A.sha256_file(snap)},
                   "provider_billed_total_usd": billed,
                   "project_cumulative_cap_usd": cap,
                   "cap_source": "logs/budget/approvals/h1_p1_generation_grant.json (absolute; never raised by a snapshot)",
                   "headroom_usd": round(cap - billed, 2),
                   "after_hard_cap_usd": round(billed + hard, 2),
                   "fits_under_the_existing_cap": billed + hard <= cap,
                   "_": ("occupancy as P1's grant counts it: the provider's all-time billing once; the 34 "
                         "historical pods and the anchor stay OPEN")},
        "launch": ("NOT built. If this route is chosen, the top-up list, its generation plan, launch identity and "
                   "harness digest are built at $0 first; a grant must name both (governance.issue), and the "
                   "existing launcher, shard securing, teardown and closeout apply unchanged"),
        "not_requested": ["P1g", "P2", "student recovery", "formal evaluation", "a network volume",
                          "a top-up of any cell not listed", "a change of D4 Top-1, the recovery objective, M1 or "
                          "eval-set-1"],
        "limitations": [
            "post hoc: the route and its size were set after P1's yield was seen (no student or evaluation result read)",
            "analogy would rest on the project generator alone; Reasoning Core's analogy module leaves the stratum",
            "two generation sessions of the same teacher and protocol pooled into one corpus; every unit keeps its "
            "own record identity",
            "the counts are forecasts from stand-ins sized as the measured units; the built ladder decides",
        ],
        "run_id_basis": run_id,
        "authorizes": "nothing",
    }
    APPLICATION.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def plan_2(path: Path, units: list[LadderUnit], tasks: dict, got: dict, store: Path, run_id: str) -> dict:
    spec = SP.candidate_spec(path)
    shares = AB.cell_shares(spec, spec["corpus_id"])
    reserve, targets = int(spec["reserve_tokens"]), list(spec["rung_targets"])
    mine = [u for u in units if u.stratum in shares]
    con, _ = contract(mine, shares, salt=spec["salt"], reserve=reserve, targets=targets)
    fill = sorted(c for c, v in con["cells"].items() if v["share"] and (v["blocks_rungs"] or not v["covered"]))
    doc = {"proposal": spec["candidate"] | {"corpus_id": spec["corpus_id"], "salt": spec["salt"]},
           "fill_cells": fill,
           "on_the_real_units_alone": {c: con["cells"][c] for c in fill},
           "composition_capacity_without_new_units": con["composition_capacity"]}
    k = {o: minimum_new_units(mine, shares, fill, salt=spec["salt"], reserve=reserve, targets=targets, order=o)
         for o in ("ascending", "median")}
    if k["ascending"] is None:
        doc["stop"] = "no number of stand-ins up to the limit lets the ladder build"
        return doc
    top = max(targets)
    con_min, _ = contract(with_stand_ins(mine, k["ascending"], order="ascending"), shares, salt=spec["salt"],
                          reserve=reserve, targets=targets)
    plan_k = planning_units(mine, shares, k["ascending"], prefix_top=con_min["rung_prefix_sizes"][str(top)],
                            reserve=reserve)
    forecast = with_stand_ins(mine, plan_k, order="ascending")
    con_k, _ = contract(forecast, shares, salt=spec["salt"], reserve=reserve, targets=targets)
    lad_k = build_ladder(forecast, shares, targets, salt=spec["salt"], reserve=reserve)
    recs = [r for s in sorted(store.glob("pass*.part*.jsonl")) for r in A.jsonl(s)]
    used = {c.unit_id for c in got["candidates"]}
    rows = {}
    for c in fill:
        gen = generation_measured(recs, tasks, got, c, spec)
        pool = [t for t in tasks.values() if p1v.cell_of(t, spec) == c]
        unused = [t for t in pool if t["task_id"] not in used]
        need = tasks_for(plan_k[c], gen["unit_accept_rate"])
        selected = GP.round_robin(unused, need["tasks"])
        measured = [u for u in mine if u.stratum == c]
        rows[c] = {"new_units": {"mechanism_minimum": k["ascending"][c], "at_the_median_size": k["median"][c],
                                 "planning": plan_k[c]},
                   "new_supervised_tokens_planning": sum(u.size for u in stand_ins(c, measured, plan_k[c],
                                                                                   order="ascending")),
                   "measured": gen, "tasks": need,
                   "unused_independent_tasks": {"total": len(unused), "pool": len(pool),
                                                "by_family_level": dict(sorted(Counter(
                                                    f"{t.get('family_id')}/{t.get('difficulty') or t.get('level')}"
                                                    for t in unused).items()))},
                   "enough_unused_tasks": len(unused) >= need["tasks"],
                   "families": sorted({u.family for u in measured if u.family}),
                   "selection": "generation_plan.round_robin over the unused tasks' groups: P1's own rule; no "
                                "group is chosen by its acceptance",
                   "selected_task_ids_sha256": _sha(json.dumps(sorted(t["task_id"] for t in selected))),
                   "expected_new_units": round(need["tasks"] * gen["unit_accept_rate"], 1),
                   "expected_new_units_by_group_rates": by_group_expectation(selected, gen, plan_k[c]),
                   "predicted_generated_tokens": need["tasks"] * gen["generated_tokens_per_task"]}
    sess = session_measured(run_id, store)
    limit = int(A.read_json(A.CONFIG_DIR / "generation_protocol.json")["execution"]["session_limit"])
    app = topup_application(rows=rows, sess=sess, speed=decode_speed(recs), limit=limit, proposal=spec,
                            run_id=run_id)
    doc.update({
        "cells": rows,
        "_counts": ("mechanism_minimum: the fewest stand-ins (smallest measured sizes first) with which the formal "
                    "ladder builds under the proposal's salt; at_the_median_size: the same with median-sized "
                    "stand-ins; planning: the robust count the tasks are sized for (planning_units)"),
        "contract_with_the_planning_forecast": {
            "label": "FORECAST with stand-ins: not a pack, not data",
            "rungs": [{k2: r.get(k2) for k2 in ("target", "n_units", "size", "reachable")} |
                      {"max_share_deviation": r.get("accounting", {}).get("max_share_deviation"),
                       "distinct_families": r.get("accounting", {}).get("distinct_families")} for r in lad_k["rungs"]],
            "cells": {c: {k2: v[k2] for k2 in ("share", "units", "supervised_tokens", "covered", "runs_dry_at_prefix",
                                                 "blocks_rungs", "reserve_units")} for c, v in con_k["cells"].items()},
            "validation_tail": con_k["validation_tail"]},
        "application": {"path": SP.repo_path(APPLICATION), "tasks_total": app["tasks_total"],
                        "expected_usd": app["cost"]["expected_usd"], "hard_cap_usd": app["cost"]["hard_cap_usd"],
                        "fits_under_the_existing_cap": app["budget"]["fits_under_the_existing_cap"]}})
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--pack-out", required=True)
    ap.add_argument("--plan-2", default=None)
    args = ap.parse_args(argv)
    spec = SP.candidate_spec(Path(args.candidate))
    cand = A.read_json(Path(args.candidate))
    v = p1v.verify_run(args.run_id, Path(args.store))
    got, tasks, tok = v["got"], v["tasks"], v["tok"]
    reuse = [A.REPO / r["path"] for r in cand["data"]["reuse"]]
    for r, p in zip(cand["data"]["reuse"], reuse):
        if A.sha256_file(p) != r["sha256"]:
            raise SystemExit(f"{r['path']} is not the reuse selection the candidate names")
    if v["auth"].plan_hash != cand["data"]["launch_identity_sha256"]:
        raise SystemExit("the run's launch identity is not the one the candidate names")
    sessions = got["sessions"] + AB._reused(reuse)
    shares = AB.cell_shares(spec, spec["corpus_id"])
    present = Counter(f"{s['stratum']}/{(s.get('attributes') or {}).get('source', '')}" for s in sessions)
    unallocated = sorted(c for c in present if c not in shares)
    if unallocated:
        raise SystemExit(f"units in cells {spec['corpus_id']} does not allocate: {unallocated}")
    sessions_sha = _sha("".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions))
    placed = AB.place(sessions, shares)
    rendered, excluded, _ = B.render_all(tok, placed, block_len=int(spec["block_len"]))
    units = B.ladder_units(rendered, tok.convert_tokens_to_ids("</think>"))
    reserve, targets = int(spec["reserve_tokens"]), list(spec["rung_targets"])
    con, lad = contract(units, shares, salt=spec["salt"], reserve=reserve, targets=targets)
    ind = independence(placed, set(lad["reserved"]))
    gen_units = {s["unit_id"]: s for s in got["sessions"]}
    items = [{**tasks[uid], "unit_id": uid, "messages": s["messages"]} for uid, s in sorted(gen_units.items())]
    iso = IC.check_pool(items)
    reuse_iso = [{"path": SP.repo_path(p.with_name(p.name.replace(".jsonl", ".isolation.json"))),
                  "pass": A.read_json(p.with_name(p.name.replace(".jsonl", ".isolation.json")))["pass"]}
                 for p in reuse]
    tried = attempt(sessions, spec, shares, tok)
    doc = {"schema": "aadistill.stage2.reasoning_data.corpus_feasibility/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "candidate": spec["candidate"] | {"corpus_id": spec["corpus_id"], "salt": spec["salt"]},
           "data": {"run_id": args.run_id, "launch_identity_sha256": v["auth"].plan_hash,
                    "shards": v["ins"]["shards"], "records": v["ins"]["records"],
                    "generated_units_selected": len(got["selected"]),
                    "reuse_units": len({s["unit_id"] for s in sessions}) - len(got["selected"]),
                    "sessions": len(sessions), "sessions_sha256": sessions_sha,
                    "admission": B.exclusion_summary(excluded)},
           "formal_path": {"built": tried["built"], "refusal": tried.get("refusal"),
                           "path": "build_arm_b.build -> build_master_ladder.build (render_all, ladder_units, "
                                   "build_ladder without drift, pack_best_fit)"},
           "contract": con, "independence": ind,
           "isolation": {"generated_units_R0_R9": {"pass": iso["pass"], "n_items": iso["n_items"],
                                                   "violations": iso["violations"][:20],
                                                   "R2_flagged": iso["results"]["R2_near_duplicate"].get("flagged")},
                         "reuse_selection": reuse_iso},
           "authorizes": "nothing"}
    doc["unsatisfied"] = unsatisfied(con) + ([] if iso["pass"] else ["R0-R9 on the generated units"]) \
        + ([] if all(r["pass"] for r in reuse_iso) else ["the reuse selection's isolation record"]) \
        + ([f"{ind['units_sharing_a_prompt']} units share a prompt"] if ind["units_sharing_a_prompt"] else []) \
        + ([f"{ind['units_sharing_a_target']} units share a target"] if ind["units_sharing_a_target"] else [])
    out = Path(args.pack_out)
    if tried["built"] and not doc["unsatisfied"]:
        built, ladder_spec = tried["result"], tried["ladder_spec"]
        meta = B.write_pack(out, built, ladder_spec, {
            "corpus_id": spec["corpus_id"], "candidate_sha256": spec["candidate"]["sha256"],
            "run_id": args.run_id, "launch_identity_sha256": v["auth"].plan_hash, "sessions_sha256": sessions_sha,
            "reuse_sha256": {r["path"]: r["sha256"] for r in cand["data"]["reuse"]}})
        from aadistill.data.ladder import ladder_blocks
        _, val, stats = ladder_blocks(out, TARGET, n_val=16)
        doc["pack"] = {"built": True, "dir": SP.repo_path(out), "outputs": meta["outputs"],
                       "ladder_sha256": A.sha256_file(out / "ladder.json"),
                       "rungs": [{k: r.get(k) for k in ("target_supervised_tokens", "n_units", "n_blocks",
                                                        "actual_supervised_tokens", "real_tokens", "padding_tokens")}
                                 for r in meta["rungs"]],
                       "validation_tail": meta["validation_tail"],
                       "trainer_reader": {"val_blocks": int(val[0].shape[0]),
                                          "val_disjoint_from_all_rungs": stats["val_disjoint_from_all_rungs"]}}
    else:
        doc["pack"] = {"built": False,
                       "why": tried.get("refusal") or "; ".join(doc["unsatisfied"]),
                       "rule": ("no composition drift, no other salt, no lower gate, no repetition: the "
                                "unsatisfied items are reported and nothing is packed")}
        if args.plan_2:
            doc["plan_2"] = plan_2(Path(args.plan_2), units, tasks, got, Path(args.store), args.run_id)
    dest = A.LOG_DIR / f"audit/corpus_{spec['corpus_id'].split('@')[1]}_feasibility.json"
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    print(dest.relative_to(A.REPO), "built" if doc["pack"]["built"] else "NOT built", doc["unsatisfied"][:3])
    return 0 if doc["pack"]["built"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

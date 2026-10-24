#!/usr/bin/env python3
"""H1-PA's fluid top-up: the approved 115 unused tasks as a frozen generation plan, its run inputs, and the grant request. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_topup_plan [--request --run-id <run>]

The sibling of `topup_plan` (P1t), for phase ``PA1`` (`governance.PHASE_PLANS`):

1. THE TASKS. For every primary cell the application allocates
   (``plans/h1pa_topup_application.json :: requested``): the tasks of P1's pools
   in that cell that neither P1 nor P1t generated -- each session's inputs,
   whose launch identity on disk must still be the one its authorization binds
   -- drawn by `generation_plan.round_robin` (P1's rule) to the application's
   count. The ids must hash to the approved task-set sha256: this is the list
   the maintainer approved, not a new draw. Every task needs a verifier; no
   prompt may repeat one of P1's or P1t's or another of this list's; the
   cells must be primary fluid-reasoning cells; R0-R9 run on the prompts.
   Unlike P1t, episodes are allowed: P1 generated them under the same protocol.
2. THE PLAN. ``plans/h1pa_topup/generation_plan.json``, its inputs beside it,
   P1's pools by sha256 and P1's generation protocol byte for byte. No
   timestamp, so a rebuild keeps the launch identity.
3. THE REQUEST (``--request``). What a grant must name, computed from disk: the
   launch identity, the harness digest, the launcher's budget plan and its
   all-in bound at the application's price, and the provider snapshot.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone

from stages.reasoning_data import assets as A
from stages.reasoning_data import corpus_feasibility as CF
from stages.reasoning_data import generation_plan as GP
from stages.reasoning_data import governance as GOV
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import p1v
from stages.reasoning_data import source_plan as SP
from stages.reasoning_data import topup_plan as TP

PHASE = "PA1"
CORPUS = A.CONFIG_DIR / "corpus_v3.json"          #: the cell definitions the pool and the tasks share
APPLICATION = A.LOG_DIR / "plans/h1pa_topup_application.json"
PLAN = A.REPO / GOV.PHASE_PLANS[PHASE]
INPUTS = PLAN.parent / "generation_inputs"
REQUEST = PLAN.parent / "authorization_request.json"
P1T_GRANT = A.REPO / "logs/budget/approvals/h1_p1t_generation_grant.json"


def p1t_bound() -> tuple[dict, set[str]]:
    """P1t's plan and the tasks it generated, while its launch identity on disk
    is still the one its authorization binds."""
    grant = A.read_json(P1T_GRANT)
    auth = GOV.load(A.REPO / GOV.authorization_path(grant["run_id"]))
    if GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN) != auth.plan_hash:
        raise SystemExit(f"{GOV.TOPUP_PLAN} on disk is not the plan P1t's authorization binds")
    plan = A.read_json(A.REPO / GOV.TOPUP_PLAN)
    used = {r["task_id"] for f in plan["inputs"] for r in A.jsonl(A.REPO / GOV.inputs_dir(plan) / f["file"])}
    return {"run_id": grant["run_id"], "launch_identity_sha256": auth.plan_hash, "plan": plan}, used


def requested(app: dict) -> dict:
    return app["scenarios"][app["requested"]]


def select(spec: dict, row: dict, pools: list[dict], used: set[str]) -> dict[str, list[dict]]:
    tasks = {t["task_id"]: t for p in pools for t in A.jsonl(A.REPO / p["path"])}
    out = {}
    for cell, n in sorted(row["tasks_by_cell"].items()):
        if not cell.startswith("primary."):
            raise SystemExit(f"{cell} is not a primary fluid-reasoning cell")
        unused = [t for t in tasks.values() if p1v.cell_of(t, spec) == cell and t["task_id"] not in used]
        out[cell] = GP.round_robin(unused, n)
        if len(out[cell]) != n:
            raise SystemExit(f"{cell}: {len(out[cell])} unused tasks, the application needs {n}")
    flat = [t for ts in out.values() for t in ts]
    if TP._ids_sha(flat) != row["selected_task_ids_sha256"]:
        raise SystemExit(f"the draw hashes to {TP._ids_sha(flat)[:12]}, not the approved "
                         f"{row['selected_task_ids_sha256'][:12]}")
    return out


def checks(chosen: dict[str, list[dict]], pools: list[dict], used: set[str]) -> dict:
    flat = [t for ts in chosen.values() for t in ts]
    no_verifier = {t["task_id"]: GP.verifier_of(t)[1] for t in flat if GP.verifier_of(t)[0] is None}
    pool = {t["task_id"]: t for p in pools for t in A.jsonl(A.REPO / p["path"])}
    seen = {TP._prompt(pool[t]) for t in used}
    mine = Counter(TP._prompt(t) for t in flat)
    sources = {s["source_id"]: s for s in A.read_json(A.CONFIG_DIR / "sources.json")["sources"]}
    unapproved = sorted({c.split("/", 1)[1] for c in chosen if not sources[c.split("/", 1)[1]].get("approved_for_h1")})
    iso = IC.check_pool(flat)
    found = {"tasks_without_a_verifier": no_verifier,
             "episodes": sum(bool(t.get("interactive")) for t in flat),
             "tasks_also_in_P1_or_P1t": len({t["task_id"] for t in flat} & used),
             "prompts_also_in_P1_or_P1t": sum(1 for k in mine if k in seen),
             "prompts_repeated_in_the_top_up": sum(n - 1 for n in mine.values() if n > 1),
             "sources_not_approved_for_h1": unapproved,
             "isolation_R0_R9": {"pass": iso["pass"], "n_items": iso["n_items"], "violations": iso["violations"][:20],
                                 "R2_flagged": iso["results"]["R2_near_duplicate"].get("flagged"),
                                 "_": "prompts only; P1v reruns R0-R9 with the teacher text"}}
    bad = [k for k in ("tasks_without_a_verifier", "tasks_also_in_P1_or_P1t", "prompts_also_in_P1_or_P1t",
                       "prompts_repeated_in_the_top_up", "sources_not_approved_for_h1") if found[k]]
    if bad or not iso["pass"]:
        raise SystemExit(f"the top-up list fails {bad or ['R0-R9']}: {json.dumps(found)[:600]}")
    return found


def build(inputs_dir=INPUTS) -> dict:
    spec, app = SP.candidate_spec(CORPUS), A.read_json(APPLICATION)
    row = requested(app)
    p1, used1 = TP.p1_bound()
    p1t, used2 = p1t_bound()
    if p1t["plan"]["pools"] != p1["plan"]["pools"]:
        raise SystemExit("P1t's pools are not P1's")
    used = used1 | used2
    chosen = select(spec, row, p1["plan"]["pools"], used)
    found = checks(chosen, p1["plan"]["pools"], used)
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)                   # only this plan's own inputs, rebuilt whole
    files = GP.write_inputs([t for ts in chosen.values() for t in ts], spec, inputs_dir)
    protocol = {"path": str(GOV.PROTOCOL), "sha256": A.sha256_file(A.REPO / GOV.PROTOCOL)}
    if protocol != GOV.launch_identity(A.REPO, GOV.PLAN)["generation_protocol"]:
        raise SystemExit("the generation protocol is not the bytes P1 ran under")
    cells = {c: {"tasks": len(ts), "selected_task_ids_sha256": TP._ids_sha(ts),
                 "groups": dict(sorted(Counter(GP.group_of(t) for t in ts).items())),
                 "verifiers": dict(sorted(Counter(GP.verifier_of(t)[0] for t in ts).items())),
                 "episodes": sum(bool(t.get("interactive")) for t in ts)}
             for c, ts in chosen.items()}
    return {"schema": "aadistill.stage2.reasoning_data.topup_generation_plan/v1",
            "phase": PHASE, "study": "stage2.reasoning_data.h1pa", "corpus_id": spec["corpus_id"],
            "corpus": {"path": SP.repo_path(CORPUS), "sha256": A.sha256_file(CORPUS),
                       "_": "the cell definitions the existing fluid pool and these tasks share; no @v3 quota applies"},
            "admitted": True,
            "status": ("FROZEN LIST for the maintainer's H1-PA top-up grant: the application's 115 tasks, drawn by "
                       "P1's rule and matched to the approved task-set sha256; nothing is generated without a grant "
                       "naming this plan's launch identity"),
            "application": {"path": SP.repo_path(APPLICATION), "sha256": A.sha256_file(APPLICATION),
                            "scenario": app["requested"]},
            "task_ids_sha256": row["selected_task_ids_sha256"],
            "excludes": [{"phase": "P1", "run_id": p1["run_id"], "launch_identity_sha256": p1["launch_identity_sha256"],
                          "tasks": len(used1)},
                         {"phase": "P1t", "run_id": p1t["run_id"], "launch_identity_sha256": p1t["launch_identity_sha256"],
                          "tasks": len(used2)}],
            "pools": p1["plan"]["pools"], "reuse": [],
            "protocol": protocol | {"same_as": "P1's launch identity :: generation_protocol (byte for byte)"},
            "cells": cells, "shortfalls": [], "checks": found,
            "inputs": files, "inputs_dir": str(INPUTS.relative_to(A.REPO)),
            "session_limit": GP.SESSION_LIMIT, "seed_rule": p1["plan"]["seed_rule"],
            "totals": {"tasks": sum(c["tasks"] for c in cells.values()), "episodes": found["episodes"],
                       "expected_generated_tokens": row["generated_tokens"]["expected"],
                       "_": "the application's forecast: a PREDICTION, not a limit; the cap is the bound"},
            "on_short_corpus": ("if the accepted units leave the fluid increment short of 2.96M, the gap is reported and "
                                "the phase stops: no further task, no lower gate, no repetition, no supporting data "
                                "(maintainer 2026-10-24)"),
            "authorizes": "nothing"}


def request(run_id: str) -> dict:
    from aadistill.governance.authorization import harness_source_digest
    from stages.reasoning_data import h1_generation_launch as L
    plan, app = A.read_json(PLAN), A.read_json(APPLICATION)
    row = requested(app)
    price, cap = row["price_per_hour"], row["hard_cap_usd"]
    b = L.budget(PHASE)
    bp = b.plan(price_per_hour=price, authorized_usd=cap)
    disk_gb = L.build_parser().get_default("disk_gb")
    bound = L.all_in(bp, price, disk_gb, cap)
    snap = sorted((A.REPO / "logs/budget/provider").glob("runpod_billing_*.json"))[-1]
    live = A.read_json(snap)["account"]
    billed = CF.provider_billed(snap)
    project_cap = A.read_json(A.REPO / "configs/stages/stage-1/phase_c1/authorization.json")[
        "accepted_pricing"]["cumulative_cap_usd"]
    return {
        "schema": "aadistill.stage2.reasoning_data.h1pa_topup_authorization_request/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "REQUEST for the maintainer's grant; NOT authorized, nothing launched",
        "phase": PHASE, "run_id": run_id,
        "generation_plan": {"path": str(GOV.PHASE_PLANS[PHASE]), "sha256": A.sha256_file(PLAN),
                            "tasks": plan["totals"]["tasks"], "task_ids_sha256": plan["task_ids_sha256"],
                            "cells": {c: v["tasks"] for c, v in plan["cells"].items()}},
        "launch_identity_sha256": GOV.launch_hash(A.REPO, GOV.PHASE_PLANS[PHASE]),
        "harness_digest": harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"],
        "not_p1_or_p1t": "P1's and P1t's grants and authorizations are consumed and fund nothing here; this is its own run",
        "cost": {"application": plan["application"], "expected_usd": row["expected_usd"],
                 "conservative_usd": row["conservative_usd"], "hard_cap_usd": cap,
                 "launcher_budget": bp.as_dict(), "all_in_bound": bound,
                 "termination_minutes": L.TERMINATION_MINUTES,
                 "fits_a_live_secure_price_up_to": bound["highest_price_inside_cap"]},
        "budget": {"provider_snapshot": {"path": SP.repo_path(snap), "sha256": A.sha256_file(snap),
                                         "fetched_utc": A.read_json(snap)["fetched_utc"]},
                   "provider_billed_total_usd": billed,
                   "live": {"pods": len(live["pods"]), "network_volumes": len(live["network_volumes"]),
                            "spend_per_hour": live["current_spend_per_hr"],
                            "balance_usd": round(live["client_balance_usd"], 4)},
                   "project_cumulative_cap_usd": project_cap, "cap_raised": False,
                   "after_hard_cap_usd": round(billed + cap, 4), "fits_under_the_project_cap": billed + cap <= project_cap},
        "resources": {"gpu": f"1 x {L.build_parser().get_default('gpu')} (the measured generation hardware), one session",
                      "image": L.build_parser().get_default("image"), "container_disk_gb": disk_gb,
                      "network_volume": None},
        "persistence": {"scr": f"/home/ecs-user/aad-scratch/{run_id}",
                        "route": "P1's: sealed shards pulled and re-hashed on every poll into <scr>/store, re-hashed at "
                                 "closeout; then artifacts/stages/stage-2/reasoning_data/generation/<run>/ and the relay "
                                 "(stage2_h1/generation/<run>/), read back"},
        "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    if args.request:
        if not args.run_id:
            raise SystemExit("--request needs --run-id")
        REQUEST.write_text(json.dumps(request(args.run_id), indent=1) + "\n")
        print(REQUEST.relative_to(A.REPO))
        return 0
    doc = build()
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(doc, indent=1) + "\n")
    print(PLAN.relative_to(A.REPO), doc["totals"], "launch identity",
          GOV.launch_hash(A.REPO, GOV.PHASE_PLANS[PHASE])[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

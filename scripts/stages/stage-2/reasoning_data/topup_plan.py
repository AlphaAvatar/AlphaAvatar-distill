#!/usr/bin/env python3
"""P1t's frozen generation list -- the @v3 top-up's 59 unused tasks -- its run inputs, and the request a grant answers. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.topup_plan \
        [--request --run-id h1p1t_20261023_094007]

1. THE CORPUS. `corpus_v3.json`, the maintainer's choice for arm B
   (2026-10-23), bound by its sha256; its proposal is the one the application
   was priced on (`audit/corpus_v2_feasibility.json :: plan_2.proposal`).
2. THE TASKS. For every cell the application fills: the tasks of P1's own
   pools in that cell that P1 did not generate (P1's inputs, whose launch
   identity on disk must still be the one P1's authorization binds), drawn by
   `generation_plan.round_robin` -- P1's rule, never a group chosen by its
   acceptance -- to the application's count. The ids must hash to the
   application's ``selected_task_ids_sha256``: this is the list the maintainer
   approved, not a new draw. Every task needs a verifier
   (`generation_plan.verifier_of`); none may be an episode; no prompt may
   repeat one of P1's or another top-up task's; R0-R9 run on the prompts
   (`isolation_check.check_pool`; P1v reruns them with the teacher text).
3. THE PLAN. ``plans/p1t/generation_plan.json``, its inputs in
   ``plans/p1t/generation_inputs/`` (`generation_plan.write_inputs`, the
   format the driver reads), P1's pools by sha256 and P1's generation protocol
   byte for byte. No timestamp, so a rebuild is byte-identical and keeps the
   launch identity (`governance.launch_hash(plan_path=TOPUP_PLAN)`).
4. THE REQUEST (``--request``). ``plans/p1t/authorization_request.json``: the
   run id, the launch identity, the harness digest, the launcher's budget plan
   at P1's measured price, the provider snapshot against the cumulative cap,
   and where the records go. It authorizes nothing; a grant naming its values
   is what `governance.issue` takes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import corpus_feasibility as CF
from stages.reasoning_data import generation_plan as GP
from stages.reasoning_data import governance as GOV
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import p1v
from stages.reasoning_data import source_plan as SP

CORPUS = A.CONFIG_DIR / "corpus_v3.json"
APPLICATION = A.LOG_DIR / "plans/abstract_topup_application.json"
AUDIT = A.LOG_DIR / "audit/corpus_v2_feasibility.json"
PLAN = A.REPO / GOV.TOPUP_PLAN
INPUTS = PLAN.parent / "generation_inputs"
REQUEST = PLAN.parent / "authorization_request.json"
P1_GRANT = A.REPO / "logs/budget/approvals/h1_p1_generation_grant.json"


def _ids_sha(tasks: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sorted(t["task_id"] for t in tasks)).encode()).hexdigest()


def _prompt(task: dict) -> str:
    return hashlib.sha256(next(m["content"] for m in task["messages"] if m["role"] == "user").encode()).hexdigest()


def p1_bound() -> tuple[dict, set[str]]:
    """P1's plan, and the tasks P1 generated -- only while P1's launch identity
    on disk is still the one its authorization binds."""
    grant = A.read_json(P1_GRANT)
    auth = GOV.load(A.REPO / GOV.authorization_path(grant["run_id"]))
    if GOV.launch_hash(A.REPO, GOV.PLAN) != auth.plan_hash:
        raise SystemExit(f"{GOV.PLAN} on disk is not the plan P1's authorization binds")
    plan = A.read_json(A.REPO / GOV.PLAN)
    used = {r["task_id"] for f in plan["inputs"] for r in A.jsonl(A.REPO / GOV.inputs_dir(plan) / f["file"])}
    return {"run_id": grant["run_id"], "launch_identity_sha256": auth.plan_hash, "plan": plan}, used


def select(spec: dict, app: dict, p1_plan: dict, used: set[str]) -> dict[str, list[dict]]:
    tasks = {t["task_id"]: t for p in p1_plan["pools"] for t in A.jsonl(A.REPO / p["path"])}
    out = {}
    for cell, row in sorted(app["cells"].items()):
        unused = [t for t in tasks.values() if p1v.cell_of(t, spec) == cell and t["task_id"] not in used]
        sel = GP.round_robin(unused, row["tasks"]["tasks"])
        if len(sel) != row["tasks"]["tasks"] or _ids_sha(sel) != row["selected_task_ids_sha256"]:
            raise SystemExit(f"{cell}: the draw is not the application's list ({len(sel)} tasks, "
                             f"{_ids_sha(sel)[:12]} vs {row['selected_task_ids_sha256'][:12]})")
        out[cell] = sel
    return out


def checks(chosen: dict[str, list[dict]], p1_plan: dict, used: set[str], spec: dict) -> dict:
    flat = [t for ts in chosen.values() for t in ts]
    no_verifier = {t["task_id"]: GP.verifier_of(t)[1] for t in flat if GP.verifier_of(t)[0] is None}
    pool = {t["task_id"]: t for p in p1_plan["pools"] for t in A.jsonl(A.REPO / p["path"])}
    p1_prompts = {_prompt(pool[t]) for t in used}
    mine = Counter(_prompt(t) for t in flat)
    sources = {s["source_id"]: s for s in A.read_json(A.CONFIG_DIR / "sources.json")["sources"]}
    unapproved = sorted({c.split("/", 1)[1] for c, share in CF.AB.cell_shares(spec, spec["corpus_id"]).items()
                         if not sources[c.split("/", 1)[1]].get("approved_for_h1")})
    iso = IC.check_pool(flat)
    found = {"tasks_without_a_verifier": no_verifier,
             "episodes": sum(bool(t.get("interactive")) for t in flat),
             "tasks_also_in_P1": len({t["task_id"] for t in flat} & used),
             "prompts_also_in_P1": sum(1 for k in mine if k in p1_prompts),
             "prompts_repeated_in_the_top_up": sum(n - 1 for n in mine.values() if n > 1),
             "corpus_sources_not_approved_for_h1": unapproved,
             "isolation_R0_R9": {"pass": iso["pass"], "n_items": iso["n_items"], "violations": iso["violations"][:20],
                                 "R2_flagged": iso["results"]["R2_near_duplicate"].get("flagged"),
                                 "establishes": iso.get("establishes"),
                                 "_": "prompts only; P1v reruns R0-R9 with the teacher text"}}
    bad = [k for k in ("tasks_without_a_verifier", "episodes", "tasks_also_in_P1", "prompts_also_in_P1",
                       "prompts_repeated_in_the_top_up", "corpus_sources_not_approved_for_h1") if found[k]]
    if bad or not iso["pass"]:
        raise SystemExit(f"the top-up list fails {bad or ['R0-R9']}: {json.dumps(found)[:600]}")
    return found


def build(inputs_dir: Path = INPUTS) -> dict:
    """The plan document; its run inputs are written to `inputs_dir` (the
    plan's own directory, rebuilt whole; a test passes another)."""
    spec = SP.candidate_spec(CORPUS)
    cand, app, audit = A.read_json(CORPUS), A.read_json(APPLICATION), A.read_json(AUDIT)
    if cand.get("chosen", {}).get("corpus_id") != spec["corpus_id"]:
        raise SystemExit(f"{CORPUS.name} is not the corpus the maintainer chose")
    if cand["proposal"]["sha256"] != audit["plan_2"]["proposal"]["sha256"]:
        raise SystemExit("the corpus is not the proposal the application was priced on")
    p1, used = p1_bound()
    chosen = select(spec, app, p1["plan"], used)
    found = checks(chosen, p1["plan"], used, spec)
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)                   # only this plan's own inputs, rebuilt whole
    files = GP.write_inputs([t for ts in chosen.values() for t in ts], spec, inputs_dir)
    p1_ident = GOV.launch_identity(A.REPO, GOV.PLAN)
    protocol = {"path": str(GOV.PROTOCOL), "sha256": A.sha256_file(A.REPO / GOV.PROTOCOL)}
    if protocol != p1_ident["generation_protocol"]:
        raise SystemExit("the generation protocol is not the bytes P1 ran under")
    cells = {c: {"tasks": len(ts), "selected_task_ids_sha256": _ids_sha(ts),
                 "unused_before": app["cells"][c]["unused_independent_tasks"]["total"],
                 "groups": dict(sorted(Counter(GP.group_of(t) for t in ts).items())),
                 "verifiers": dict(sorted(Counter(GP.verifier_of(t)[0] for t in ts).items())),
                 "new_units_planning": app["cells"][c]["new_units"]["planning"],
                 "expected_new_units": app["cells"][c]["expected_new_units"],
                 "expected_new_units_by_group_rates":
                     app["cells"][c]["expected_new_units_by_group_rates"]["expected_units"],
                 "predicted_generated_tokens": app["cells"][c]["predicted_generated_tokens"]}
             for c, ts in chosen.items()}
    return {"schema": "aadistill.stage2.reasoning_data.topup_generation_plan/v1",
            "phase": "P1t", "corpus_id": spec["corpus_id"],
            "corpus": {"path": SP.repo_path(CORPUS), "sha256": A.sha256_file(CORPUS)},
            "admitted": True,
            "status": ("FROZEN LIST for the maintainer's P1t grant: the application's 59 tasks, re-drawn by P1's rule "
                       "and matched to its id hashes; nothing is generated without a grant naming this plan's launch "
                       "identity"),
            "application": {"path": SP.repo_path(APPLICATION), "sha256": A.sha256_file(APPLICATION)},
            "excludes": {"phase": "P1", "run_id": p1["run_id"], "launch_identity_sha256": p1["launch_identity_sha256"],
                         "tasks": len(used), "tasks_shared": found["tasks_also_in_P1"]},
            "pools": p1["plan"]["pools"], "reuse": [],
            "protocol": protocol | {"same_as": "P1's launch identity :: generation_protocol (byte for byte)"},
            "cells": cells, "shortfalls": [], "checks": found,
            "inputs": files, "inputs_dir": str(INPUTS.relative_to(A.REPO)),
            "session_limit": GP.SESSION_LIMIT, "seed_rule": p1["plan"]["seed_rule"],
            "totals": {"tasks": sum(c["tasks"] for c in cells.values()), "episodes": found["episodes"],
                       "predicted_generated_tokens": sum(c["predicted_generated_tokens"] for c in cells.values()),
                       "_predicted": "the application's measured tokens per task x its tasks: a PREDICTION, not a limit"},
            "on_short_corpus": ("if the accepted units still do not build B on the formal ladder, the gap is reported "
                                "and P1t stops there: no automatic further paid top-up, no lower gate, no repetition, "
                                "no other source or mix (maintainer 2026-10-23)"),
            "_reading": ("the comp/Reasoning Gym forecast rests on a few easy groups (largest_island L0 7/7, "
                         "tower_of_hanoi L0 6/6, string_manipulation L0 2/7; 11 of 14 module-levels yielded 0): "
                         "a prediction, not a promise"),
            "authorizes": "nothing"}


def request(run_id: str) -> dict:
    """What a P1t grant has to name, computed from disk now."""
    from aadistill.governance.authorization import harness_source_digest
    from stages.reasoning_data import h1_generation_launch as L
    plan, app = A.read_json(PLAN), A.read_json(APPLICATION)
    m = app["cost"]["measured_basis"]
    hard_cap = app["cost"]["hard_cap_usd"]
    b = L.topup_budget(A.REPO)
    bp = b.plan(price_per_hour=m["price_per_hour"], authorized_usd=hard_cap)
    hard_min = bp.as_dict()["hard_terminate_minutes"]
    snap = sorted((A.REPO / "logs/budget/provider").glob("runpod_billing_*.json"))[-1]
    live = A.read_json(snap)["account"]
    billed = CF.provider_billed(snap)
    cap = A.read_json(P1_GRANT)["project_cumulative_cap_usd"]
    gen = b.other_phases[0].minutes
    return {
        "schema": "aadistill.stage2.reasoning_data.p1t_authorization_request/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "REQUEST for the maintainer's P1t grant; NOT authorized, nothing launched",
        "phase": "P1t", "run_id": run_id,
        "corpus": plan["corpus"],
        "generation_plan": {"path": str(GOV.TOPUP_PLAN), "sha256": A.sha256_file(PLAN),
                            "tasks": plan["totals"]["tasks"],
                            "cells": {c: v["tasks"] for c, v in plan["cells"].items()}},
        "launch_identity_sha256": GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN),
        "launch_identity": GOV.launch_identity(A.REPO, GOV.TOPUP_PLAN),
        "harness_digest": harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"],
        "harness_files": len(GOV.HARNESS),
        "not_p1s": {"P1_launch_identity_sha256": plan["excludes"]["launch_identity_sha256"],
                    "P1_grant_harness_digest": A.read_json(P1_GRANT)["harness_digest"],
                    "_": "P1's grant and authorization are consumed and fund nothing here; P1t is its own run"},
        "cost": {"application": plan["application"],
                 "expected_usd": app["cost"]["expected_usd"], "closeout_reserve_usd": app["cost"]["closeout_reserve_usd"],
                 "hard_cap_usd": hard_cap,
                 "launcher_budget_at_p1_price": bp.as_dict(),
                 "launcher_hard_threshold_all_in_usd": round(bp.hard_terminate_usd * m["all_in_over_gpu"], 4),
                 "fits_a_live_secure_price_up_to": math.floor(hard_cap / (hard_min / 60) * 100) / 100,
                 "setup_may_take_up_to_minutes": round(hard_min - gen - m["all_done_to_pod_deleted_minutes"], 1),
                 "_": ("h1_generation_launch.topup_budget: the application's rule, 1.5 x (expected + setup + teardown "
                       "+ 15 min); the watchdog's hard deadline follows the cap and the live price. P1 measured "
                       f"{m['pod_to_driver_minutes']} min pod to driver; any setup up to the figure above still "
                       "leaves both passes and the teardown inside the hard deadline")},
        "budget": {"provider_snapshot": {"path": SP.repo_path(snap), "sha256": A.sha256_file(snap),
                                         "fetched_utc": A.read_json(snap)["fetched_utc"]},
                   "provider_billed_total_usd": billed,
                   "live": {"pods": len(live["pods"]), "network_volumes": len(live["network_volumes"]),
                            "spend_per_hour": live["current_spend_per_hr"],
                            "balance_usd": round(live["client_balance_usd"], 4)},
                   "account_balance_required_usd": b.account_balance_required_usd,
                   "project_cumulative_cap_usd": cap, "cap_raised": False,
                   "headroom_usd": round(cap - billed, 4),
                   "after_hard_cap_usd": round(billed + hard_cap, 4),
                   "fits_under_the_existing_cap": billed + hard_cap <= cap,
                   "_": "conservative occupancy: the provider's all-time billing counted once; the cap is absolute"},
        "resources": {"gpu": "1 x NVIDIA A100-SXM4-80GB (P1's), one session, one billing resource",
                      "image": "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404 (P1's)",
                      "container_disk_gb": 150, "network_volume": None},
        "persistence": {"scr": f"/home/ecs-user/aad-scratch/{run_id}",
                        "route": ("P1's: the driver seals finished candidates into hash-marked shards, the launcher "
                                  "pulls every marked shard on each poll into <scr>/store and re-hashes it, the closeout "
                                  "re-hashes all of them; then a copy to artifacts/stages/stage-2/reasoning_data/"
                                  "generation/<run>/ and to the relay (stage2_h1/generation/<run>/), read back"),
                        "expected_bytes": "about 6 MB of shards (P1: 141 MB for 2,164 records; P1t at most 118)",
                        "dev_box_free_bytes": shutil.disk_usage("/home/ecs-user").free},
        "merge": ("corpus_feasibility --candidate corpus_v3.json --store <P1 run>=<store> --store <P1t run>=<store>: "
                  "each session verified under its own authorization; refused on a shared task, unit, protocol or "
                  "runner; units keep their session's identity; cells at 0.0 set aside by name; the pack only when "
                  "the formal ladder cuts every rung with the reserve and isolation passes"),
        "grant_document": {"fields": ["granted_by", "decision", "phase (= P1t)", "run_id", "launch_identity_sha256",
                                      "harness_digest", "expected_usd", "hard_cap_usd", "scope (optional)"],
                           "issued_by": "governance.issue: recomputes both identities from disk and refuses a mismatch"},
        "at_launch_read_only": ["stage the session-commit bundle on the relay (bundle_on_relay)",
                                "a fresh provider snapshot: 0 pods, 0 volumes, $0/h, billing inside the cap",
                                "the live securePrice at or below the figure above (check_gpu_offered)",
                                "dev-box free space for <scr>/store"],
        "on_short_corpus": plan["on_short_corpus"],
        "not_requested": ["P1g", "P2", "student recovery", "formal evaluation", "a network volume",
                          "any further top-up", "a change of D4 Top-1, the recovery objective, M1, eval-set-1 or the "
                          "150k validation reserve"],
        "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    if args.request:
        if not args.run_id:
            raise SystemExit("--request needs --run-id (the corpus's declared P1t run)")
        declared = [s for s in A.read_json(CORPUS)["data"]["generation_sessions"] if s["phase"] == "P1t"]
        if [s["run_id"] for s in declared] != [args.run_id]:
            raise SystemExit(f"{args.run_id} is not the P1t run {CORPUS.name} declares")
        REQUEST.write_text(json.dumps(request(args.run_id), indent=1) + "\n")
        print(REQUEST.relative_to(A.REPO))
        return 0
    doc = build()
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(doc, indent=1) + "\n")
    print(PLAN.relative_to(A.REPO), doc["totals"], "launch identity", GOV.launch_hash(A.REPO, GOV.TOPUP_PLAN)[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Propose ONE fixed initialization from the 12 Stage-0 candidates, by a rule that
reads no recovered outcome.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.select_initialization

Writes `logs/stages/stage-2/reasoning_data/plans/initialization_selection.json`.

**The rule.** Stage 0/1's own objective is the student's step-0 state against
the teacher (AGENTS.md 4.2), and every candidate was scored on it by the same
suite (`state_eval@v1`, one hash across all three searches). Each search ranked
its leaves by three objectives, all minimized: equal-domain teacher KL,
worst-domain teacher KL and critical-token KL. The proposal is the candidate
that Pareto-dominates every other candidate on those three. If no single
candidate does, the script says so and proposes nothing.

**What it does not read.** Recovered usability and correctness are reported
beside the selection, never used by it: over 24 recovered students they sit at
the reasoning floor and follow the recovery seed (+3.9 to +24.9 points of
usability between seeds), so selecting on them would select a lucky seed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from stages.reasoning_data import assets as A

OUT = A.LOG_DIR / "plans/initialization_selection.json"
OBJECTIVES = ("state.teacher_kl.equal_domain_mean", "state.teacher_kl.worst_domain",
              "state.critical_token_kl")
DOSES = {
    "d1": "runs/dose_d1_20261019_055515/evidence/d1",
    "d2": "runs/dose_d2_20261020_002929/evidence/d2",
    "d4": "runs/dose_d4_20261020_212310/evidence/d4",
}


def candidates() -> list[dict]:
    plan = A.read_json(A.S0 / "plans/top4_extension.json")["candidates"]["by_dose"]
    d4 = A.read_json(A.S0 / DOSES["d4"] / "finalists/finalists_manifest.json")["candidates"]["d4"]
    registry = A.read_json(A.REPO / "logs/maintenance/inventories/checkpoint_registry.json")
    out, suites = [], set()
    for dose, root in DOSES.items():
        sel_path = A.S0 / root / "search/stage1_selection.json"
        sel = A.read_json(sel_path)
        suites.add((sel["suite"]["qualified_id"], sel["suite"]["hash"]))
        decisions = {d["state_id"]: d for d in sel["decisions"]}
        record = A.read_json(A.S0 / root / "search/search_record.json")
        finalists = d4 if dose == "d4" else plan[dose]["finalists"]
        for f in finalists:
            d = decisions[f["state_id"]]
            reg = next((v for k, v in registry["scientific_artifacts"].items()
                        if k.startswith(f"stage0_calvol_{dose}_top{f['rank']}_")), None)
            out.append({
                "dose": dose, "rank": f["rank"], "state_id": f["state_id"],
                "path": f["path"], "checkpoint_sha256": f["checkpoint_sha256"],
                "artifact_digest": f["artifact_digest"], "weights_digest": f["weights_digest"],
                "num_parameters": f["num_parameters"],
                "objectives": {k: d["objectives"][k] for k in OBJECTIVES},
                "diagnostics": d["diagnostics"],
                "selection_record": str(sel_path.relative_to(A.REPO)),
                "execution_fingerprint": record["session"]["baseline"]["numerical_execution_fingerprint"],
                "prediction_rows": record["execution"]["operators"]["prediction_rows"],
                "registry_entry": reg and reg["artifact_id"],
                "relay_copy": reg and reg["relay_copy"],
                "local_copies": reg and reg["local_copies"],
            })
    if len(suites) != 1:
        raise SystemExit(f"candidates were scored by different suites: {sorted(suites)}")
    for c in out:
        c["suite"] = dict(zip(("qualified_id", "hash"), next(iter(suites))))
    return out


def dominates(a: dict, b: dict) -> bool:
    x, y = a["objectives"], b["objectives"]
    return all(x[k] <= y[k] for k in OBJECTIVES) and any(x[k] < y[k] for k in OBJECTIVES)


def recovered(dose: str, rank: int) -> list[dict]:
    per = A.read_json(A.S0 / "design/extension_analysis.json")["per_unit"]
    rows = []
    for key, u in sorted(per.items()):
        if u["dose"] != dose or u["rank"] != rank:
            continue
        f = u["eval_set_1"]["fluid_reasoning_720/dev_v1"]
        rows.append({"unit": key, "seed": u["seed"],
                     "fluid_static_correct": f"{f['static']['correct']['count']}/570",
                     "fluid_static_usable": f["static"]["usable"]["rate"],
                     "episode_success": f"{f['episodes']['success']['count']}/144",
                     "final_val_ce": u["recovery"]["final_val"]["val_ce"]})
    return rows


def main() -> int:
    cands = candidates()
    for c in cands:
        c["dominated_by"] = sorted(f"{o['dose']}/r{o['rank']}" for o in cands if dominates(o, c))
    front = [c for c in cands if not c["dominated_by"]]
    dominant = [c for c in cands if all(dominates(c, o) for o in cands if o is not c)]
    chosen = dominant[0] if len(dominant) == 1 else None
    doc = {
        "schema": "aadistill.stage2.reasoning_data.initialization_selection/v1",
        "status": "PROPOSED -- a maintainer decision; nothing is restored, copied or trained by this record",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": ("the candidate that Pareto-dominates all others on the searches' three minimized "
                 "step-0 objectives " + ", ".join(OBJECTIVES) + ", under the one shared state_eval suite; "
                 "no recovered outcome is read"),
        "candidates": cands,
        "pareto_front": [f"{c['dose']}/r{c['rank']}" for c in front],
        "dominant": [f"{c['dose']}/r{c['rank']}" for c in dominant],
        "proposal": None if chosen is None else {
            "dose": chosen["dose"], "rank": chosen["rank"], "state_id": chosen["state_id"],
            "checkpoint_sha256": chosen["checkpoint_sha256"],
            "artifact_digest": chosen["artifact_digest"], "weights_digest": chosen["weights_digest"],
            "relay_copy": chosen["relay_copy"], "registry_entry": chosen["registry_entry"],
            "step0": {**chosen["objectives"],
                      "state.nll.pooled_all_domains": chosen["diagnostics"]["state.nll.pooled_all_domains"],
                      "state.nll.reasoning_math": chosen["diagnostics"]["state.nll.reasoning_math"],
                      "state.nll.teacher_reference_pooled": chosen["diagnostics"]["state.nll.teacher_reference_pooled"]},
            "recovered_beside_it_not_used": recovered(chosen["dose"], chosen["rank"]),
        },
        "materialization_identity": {
            "fact": "every candidate was materialized by a D1 full search whose session baseline records "
                    "the forward profile's numerical execution fingerprint and prediction_rows scored_v1",
            "fingerprints_seen": sorted({c["execution_fingerprint"] for c in cands}),
            "forward_baseline_record": "configs/shared/baselines/initialization_forward_baseline_v1.json still lists "
                                       "forward_materializations: [] -- the open record inconsistency #1 of "
                                       "docs/shared/STAGE2_3_HANDOFF.md section 11; this proposal does not edit or "
                                       "reselect it",
        },
        "limits": [
            "step-0 objectives are a structural-initialization measure (AGENTS.md 4.2); that a lower step-0 "
            "KL recovers better reasoning is NOT established -- it is what Stage 3 now tests",
            "the three objectives are computed on state_eval@v1 (mixed-domain), not on eval-set-1",
            "the proposal fixes ONE initialization for every arm; it makes no claim about the other 11",
        ],
        "authorizes": "nothing",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), "->", doc["dominant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

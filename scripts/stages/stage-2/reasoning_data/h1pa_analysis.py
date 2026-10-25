#!/usr/bin/env python3
"""H1-PA's result at $0, from the six formal units' evidence: the preregistered rule, then every capability apart.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_analysis --evidence <dir>

``<dir>/<unit>/`` holds each unit's extracted evidence (``report/report.json``,
``evaluation/rows/*.rows.jsonl``, ``evaluation/generations/*.summary.json``,
``evaluation/formal_report.json``). Nothing is rescored or re-generated.

1. IDENTITY. All six units must be present. Each report must be formal and
   name its arm, seed and execution, and its inputs must equal what
   `h1pa_driver.unit_inputs` derives from the frozen study. That covers the
   preregistration, the eval set, the decoding protocol, the PCA checkpoint
   or the initialization, the pack and the recipe identity. Every generation
   summary must carry the preregistered decoding protocol
   (`h1pa_driver.generation_refusals`). A unit that is missing, or that does
   not match its frozen self, refuses.
2. THE DECISION (`h1pa_statistics.decide`, bound by the preregistration):
   - one row per (arm, seed, item) on the FORMAL static fluid items;
   - cluster = the item's rule family;
   - every row carries the decoding identity its unit's fluid summary records;
   - A and B are decided, C is descriptive;
   - interactive episodes are exploratory and reported beside the decision,
     never inside it.
3. EVERY CAPABILITY APART: each fluid axis by arm, seed and generator tier;
   every supporting and guard-rail unit's headline from its formal report. No
   pooled score is made.
4. COMPUTE beside the result: optimizer steps and training FLOPs per arm
   (`h1pa.json :: arms.*.schedule`). B adds data and compute together.

Writes ``logs/stages/stage-2/reasoning_data/analyses/h1pa_result.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import h1pa_driver as D
from stages.reasoning_data import h1pa_statistics as HS
from stages.reasoning_data.h1_analysis import FLUID, STATIC_AXES, capabilities, fluid_meta, rate

OUT = A.LOG_DIR / "analyses/h1pa_result.json"
FLUID_KEY = "__".join(FLUID)


def expected_inputs(plan: dict, u: dict) -> dict:
    """What the formal driver bound for `u`, derived from the frozen study alone
    (D4-Augmented's schedule from the study's record of the merged set, which
    the driver refused to differ from)."""
    if u["execution"] == "evaluate":
        return D.unit_inputs(plan, u, None, None)
    sched = None
    if u["arm"] == "D4-Augmented":
        e1 = plan["recipes"][int(plan["study"]["seeds"][0])]
        rung = plan["study"]["datasets"][plan["study"]["arms"][u["arm"]]["dataset"]]["rung"]
        sched = {"rung": int(rung["target_supervised_tokens"]),
                 **D.e1_schedule(int(rung["n_blocks"]), epochs=3, batch_size=int(e1["batch"]["blocks_per_step"]))}
    recipe = D.derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-")
    return D.unit_inputs(plan, u, sched, D.recipe_identity(recipe))


def unit_evidence(plan: dict, u: dict, d: Path) -> dict:
    rep = json.loads((d / "report/report.json").read_text())
    if (rep.get("formal") is not True or rep.get("unit") != u["unit"] or rep.get("execution") != u["execution"]
            or rep.get("inputs") != json.loads(json.dumps(expected_inputs(plan, u)))):
        raise SystemExit(f"{u['unit']}: its report is not this frozen unit's ({d / 'report/report.json'})")
    want = plan["prereg"]["evaluation"]["decoding"]["sha256"]
    bad = D.generation_refusals(d / "evaluation/generations", want)
    if bad:
        raise SystemExit(f"{u['unit']}: generations not under the preregistered decoding: {bad[:3]}")
    summary = json.loads((d / f"evaluation/generations/{FLUID_KEY}.summary.json").read_text())
    rows = {f.name.removesuffix(".rows.jsonl"): [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for f in sorted((d / "evaluation/rows").glob("*.rows.jsonl"))}
    return {"report": rep, "rows": rows, "decoding_sha256": D._decoding_of(summary),
            "formal_report": json.loads((d / "evaluation/formal_report.json").read_text())}


def analyse(evidence: Path) -> dict:
    plan = D.load_plan()
    items, eval_only = fluid_meta()
    units = {u["unit"]: {**u, **unit_evidence(plan, u, evidence / u["unit"])} for u in plan["units"]}
    decision_rows, by = [], defaultdict(list)
    for u in units.values():
        for r in u["rows"][FLUID_KEY]:
            if not r.get("formal"):
                continue
            it = items[r["id"]]
            tier = "unseen_format" if it["generator_id"] in eval_only else "unseen_rule"
            by[(u["arm"], u["seed"], it["axis"], tier)].append(r)
            if it.get("interactive"):
                continue
            decision_rows.append({"arm": u["arm"], "seed": u["seed"], "item_id": r["id"], "cluster": it["family_id"],
                                  "axis": it["axis"], "correct": bool(r.get("correct")),
                                  "decoding_sha256": u["decoding_sha256"]})
    decision = HS.decide(decision_rows, prereg=plan["prereg"])
    axes = defaultdict(dict)
    for (arm, seed, axis, tier), rows in sorted(by.items()):
        axes[axis].setdefault(f"{arm}-s{seed}", {})[tier] = rate(rows)
    for axis, cells in axes.items():
        for cell, tiers in cells.items():
            tiers["all"] = rate([r for (a, s, ax, _t), rows in by.items() if ax == axis and f"{a}-s{s}" == cell
                                 for r in rows])
    st = plan["study"]
    return {"schema": "aadistill.stage2.reasoning_data.h1pa_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preregistration": {"path": str(D.PREREG.relative_to(A.REPO)), "sha256": plan["prereg_sha256"]},
            "decision_rule_bound_sha256": HS.bound_identity(),
            "units": {n: {"arm": u["arm"], "seed": u["seed"], "execution": u["execution"],
                          "run_id": u["report"]["run_id"], "attempt": u["report"]["attempt"],
                          "checkpoint": u["report"]["checkpoint"], "seconds": u["report"].get("seconds"),
                          "evaluation_descriptor_sha256": u["report"]["evaluation_descriptor_sha256"]}
                      for n, u in units.items()},
            "decision": decision,
            "primary_comparison": decision["primary_comparison"],
            "primary_label": decision["primary_label"],
            "by_axis_and_tier": dict(axes),
            "interactive_exploratory": {a: c for a, c in axes.items() if a not in STATIC_AXES},
            "capabilities_by_unit": {n: capabilities(u["formal_report"]) for n, u in units.items()},
            "compute": {arm: {k: st["arms"][arm]["schedule"].get(k) for k in ("optimizer_steps", "training_flops")}
                        for arm in ("D4-Original", "D4-Augmented")},
            "reading": ("A and B are decided under the preregistered rule on static fluid correctness under nucleus_v1; "
                        "B adds the fluid increment together with its optimizer steps and FLOPs and is never read as "
                        "the data's isolated effect; C is descriptive; a floor is NOT_RESOLVED (FLOOR), never "
                        "equivalence; loss, KL, format or termination are never read as reasoning"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args(argv)
    doc = analyse(Path(args.evidence))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(OUT.relative_to(A.REPO), {k: (v.get("primary") or v).get("label")
                                    for k, v in doc["decision"]["comparisons"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

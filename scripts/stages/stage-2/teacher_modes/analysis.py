#!/usr/bin/env python3
"""Research A's result: S and C under the registered rule, every axis and unit beside them. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.analysis --evidence <dir with units/ and teacher_eval/>

The evidence is what the recovery and teacher sessions secured, extracted:
``units/<arm>-s<seed>/`` (report, evaluation rows and summaries) and
``teacher_eval/teacher_<mode>/``. Every unit's report must be the formal one of
the inputs the frozen plan derives (`recovery_units.load_plan`), every
generation under nucleus_v1. Reported:

* the decision (`mode_statistics.decide`): S, and C or its registered fallback;
* the four P10.2 primary axes -- task-contained deduction, inductive rule
  discovery, compositional generalization (static, decided) and interactive
  adaptation and planning (the 144 episodes, exploratory) -- with
  abstract/analogical as its registered sub-category, each axis apart, by
  unseen-rule and unseen-format tier; every other unit of eval-set-1 beside,
  never pooled;
* the teacher in both modes on the same items, and each arm's gap to it;
* the training side: sequences, supervised tokens, optimizer steps per arm;
  the length-limited (right-censored) shares and the ``LENGTH_CENSORED``
  warnings of the admission record, in full.

Nothing here is a selection of Research B's teacher: that is the maintainer's.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import h1pa_driver as HD  # noqa: E402
from stages.reasoning_data.h1_analysis import FLUID, STATIC_AXES, capabilities, fluid_meta, rate  # noqa: E402
from stages.teacher_modes import recovery_units as RD  # noqa: E402
from stages.teacher_modes import mode_statistics as ST  # noqa: E402

OUT = REPO / "logs/stages/stage-2/teacher_modes/analyses/d4_scale_result.json"
FLUID_KEY = "__".join(FLUID)
PRIMARY_AXES = ("task_contained_deduction", "inductive_rule_discovery", "compositional_generalization",
                "interactive_adaptation")
SUB_CATEGORY = {"abstract_analogical": "a registered sub-category of the primary axes, reported as its own axis"}


def expected_inputs(plan: dict, u: dict, pack_root: Path) -> dict:
    sched = HD.schedule_for(plan, u["arm"], pack_root)
    recipe = HD.derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-")
    return HD.unit_inputs(plan, u, sched, HD.recipe_identity(recipe))


def evidence_of(d: Path, want_decoding: str) -> dict:
    bad = HD.generation_refusals(d / "evaluation/generations", want_decoding)
    if bad:
        raise SystemExit(f"{d}: generations not under the registered decoding: {bad[:3]}")
    summary = json.loads((d / f"evaluation/generations/{FLUID_KEY}.summary.json").read_text())
    rows = {f.name.removesuffix(".rows.jsonl"): [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for f in sorted((d / "evaluation/rows").glob("*.rows.jsonl"))}
    return {"rows": rows, "decoding_sha256": HD._decoding_of(summary),
            "formal_report": json.loads((d / "evaluation/formal_report.json").read_text())}


def by_axis(rows: list[dict], items: dict, eval_only: set) -> dict:
    cells = defaultdict(list)
    for r in rows:
        if not r.get("formal"):
            continue
        it = items[r["id"]]
        cells[(it["axis"], "unseen_format" if it["generator_id"] in eval_only else "unseen_rule")].append(r)
    out = defaultdict(dict)
    for (axis, tier), rs in sorted(cells.items()):
        out[axis][tier] = rate(rs)
    for axis in list(out):
        out[axis]["all"] = rate([r for (a, _t), rs in cells.items() if a == axis for r in rs])
    return dict(out)


def analyse(evidence: Path, pack_root: Path) -> dict:
    plan = RD.load_plan(frozen=True)
    want = plan["prereg"]["evaluation"]["decoding"]["sha256"]
    items, eval_only = fluid_meta()
    units, decision_rows = {}, []
    for u in plan["units"]:
        d = evidence / "units" / u["unit"]
        rep = json.loads((d / "report/report.json").read_text())
        if (rep.get("formal") is not True or rep.get("unit") != u["unit"]
                or rep.get("inputs") != json.loads(json.dumps(expected_inputs(plan, u, pack_root)))):
            raise SystemExit(f"{u['unit']}: its report is not this frozen unit's")
        ev = evidence_of(d, want)
        units[u["unit"]] = {**u, "report": rep, **ev}
        for r in ev["rows"][FLUID_KEY]:
            it = items[r["id"]]
            if r.get("formal") and not it.get("interactive"):
                decision_rows.append({"arm": u["arm"], "seed": u["seed"], "item_id": r["id"], "cluster": it["family_id"],
                                      "axis": it["axis"], "correct": bool(r.get("correct")),
                                      "decoding_sha256": ev["decoding_sha256"]})
    adm = plan["admission"]
    fallback = not adm["C"]["state"].startswith("FEASIBLE")
    decision = ST.decide(decision_rows, prereg=plan["prereg"], c_fallback=fallback)
    teacher = {}
    for mode in ("H-T", "H-N"):
        d = evidence / "teacher_eval" / f"teacher_{mode}"
        if (d / "evaluation").is_dir():
            ev = evidence_of(d, want)
            teacher[mode] = {"by_axis": by_axis(ev["rows"][FLUID_KEY], items, eval_only),
                             "capabilities": capabilities(ev["formal_report"])}
    per_unit = {n: {"arm": u["arm"], "seed": u["seed"], "by_axis": by_axis(u["rows"][FLUID_KEY], items, eval_only),
                    "capabilities": capabilities(u["formal_report"]), "run_id": u["report"]["run_id"],
                    "seconds": u["report"].get("seconds")} for n, u in units.items()}
    gap = {}
    for n, u in per_unit.items():
        t = teacher.get(RD.ARMS_MODE[u["arm"]])
        if t:
            gap[n] = {ax: round((t["by_axis"].get(ax, {}).get("all", {}).get("rate") or 0)
                                - (v.get("all", {}).get("rate") or 0), 6) for ax, v in u["by_axis"].items()}
    return {"schema": "aadistill.stage2.teacher_modes.d4_scale_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preregistration": {"path": str(RD.PREREG.relative_to(REPO)), "sha256": plan["prereg_sha256"]},
            "decision_rule_bound_sha256": ST.bound_identity(), "decision": decision,
            "axes": {"primary": list(PRIMARY_AXES), "sub_category": SUB_CATEGORY, "decided_static": sorted(STATIC_AXES),
                     "exploratory": "interactive_adaptation (the 144 interactive episodes)"},
            "units": per_unit, "teacher": teacher, "teacher_minus_student": gap,
            "training": {arm: {k: adm["arms"][arm][k] for k in ("sequences", "documents", "supervised_tokens",
                                                                 "optimizer_steps")} for arm in adm["arms"]},
            "data": {"stopping": adm["stopping"], "I_tasks": adm["I"]["tasks"], "C": adm["C"]["state"],
                     "length_limit": {"LENGTH_CENSORED": adm["length_limit"]["LENGTH_CENSORED"],
                                      "_": adm["length_limit"]["_"]}},
            "reading": ("S and C are decided on static fluid correctness under nucleus_v1 and read separately; the "
                        "interactive axis is exploratory; NLL, KL, loss or usable rollout are never read as reasoning; "
                        "which teacher Research B uses is the maintainer's decision"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--pack-root", default=str(REPO / "artifacts/stages/stage-2/teacher_modes/packs"))
    a = ap.parse_args(argv)
    doc = analyse(Path(a.evidence), Path(a.pack_root))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(OUT.relative_to(REPO), {k: v["primary"]["label"] for k, v in doc["decision"]["comparisons"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 3 of six_category_schema_restored@v1: the six units' result on eval-set-1, every axis apart. $0, dev box.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.stage3_sr_analysis \\
        --units <dir holding <unit>/{train,evaluation}> [--units-record <arm>_units.json ...] [--out <json>]

The units are `packed_training_plan.json`'s (A1-6cat-sr / A2-6cat-sr / C1-6cat-sr x seeds 20260726, 20260801), each
evaluated on the frozen eval-set-1 v2 under nucleus_v1 in its own mode (`a1_units.py`). Reported, never pooled
across capabilities (P10.2, P10.3):

* per unit and per arm (the two seeds' item-paired mean, bootstrap 95% over items): the 570 formal static fluid items
  over the four static axes ("pooled") and by axis, with each unit's worst axis; correctness among usable rollouts
  beside the usable rate; the 144 interactive episodes, exploratory; rollout integrity (natural termination,
  degeneration stops, context-limit hits right-censored, reasoning and answer tokens); every other capability unit as
  the formal report measured it, with each capability's worst unit;
* the registered comparison (`token_matched_preregistration.json :: decision_rule`, `mode_statistics.compare`, bound
  by sha256): A1 - A2, the decided contrast -- the same hybrid teacher and initialization, matched unique supervised
  tokens and composition, NOT matched compute -- and A1 - C1, descriptive end to end (another teacher lineage AND
  another initialization search); each item-paired over both seeds, rule-family cluster bootstrap stratified by axis,
  97.5% interval, SESOI 5 pp, per-axis Holm at 8 pp; never merged;
* each arm's data and compute from its frozen records: unique supervised tokens before packing (reasoning / answer /
  tool-call), packed sequences, optimizer steps, supervised-token exposure, non-padding tokens trained, tool-calling
  turns and calls, and each unit's measured training (steps, seconds per step, peak memory, validation losses).

What this does not say: eval-set-1's tool coverage is not closed-loop tool-task success; a lower loss, KL, format
rate or natural termination is never read as reasoning; no release or Stage 4/5 decision follows from it.
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

from stages.reasoning_data.h1_analysis import STATIC_AXES, capabilities, fluid_meta  # noqa: E402
from stages.teacher_modes import a1_analysis as AA  # noqa: E402
from stages.teacher_modes import mode_statistics as MS  # noqa: E402
from stages.teacher_modes import packed_training_plan as PTP  # noqa: E402

OUT = REPO / "logs/stages/stage-2/teacher_modes/analyses/stage3_sr_result.json"
PREREG = REPO / "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json"
#: (X, Y): the contrast is Y - X; the registered rule's X is the non-thinking arm
COMPARISONS = {"A1_minus_A2": ("A2-6cat-sr", "A1-6cat-sr", "decided (the study's primary comparison)"),
               "A1_minus_C1": ("C1-6cat-sr", "A1-6cat-sr", "descriptive, end to end: another teacher lineage and "
                                                            "another initialization search, not a single variable")}
#: the rule's labels name the thinking arm Y; for the descriptive contrast they are read as Y / X better
READING = {"GO_THINKING": "Y better by more than the SESOI", "GO_NON_THINKING": "X better by more than the SESOI",
           "NO_GO": "a difference beyond the SESOI excluded", "INCONCLUSIVE": "neither excluded nor shown",
           "NOT_RESOLVED": "too few discordant pairs or informative clusters, or both arms at the floor"}


def usable_correct(ev: dict, items: dict) -> dict:
    """Static formal correctness among the usable rollouts (an answer extracted), beside the usable rate."""
    rows = [r for r in ev["rows"][AA.FLUID_KEY] if r.get("formal") and not items[r["id"]].get("interactive")]
    use = [r for r in rows if r.get("reason") != "unusable"]
    return {"n": len(rows), "usable": len(use), "usable_rate": round(len(use) / len(rows), 4) if rows else None,
            "correct_among_usable": AA.rate(use)}


def decision_rows(name: str, arm: str, seed: int, f: dict, items: dict) -> list[dict]:
    return [{"arm": arm, "seed": seed, "item_id": i, "cluster": items[i]["family_id"], "axis": items[i]["axis"],
             "correct": bool(c), "decoding_sha256": AA.DECODING_SHA} for i, c in sorted(f["item_correct"].items())]


def worst(cells: dict) -> dict | None:
    rated = {k: v["rate"] for k, v in cells.items() if isinstance(v, dict) and v.get("rate") is not None}
    if not rated:
        return None
    k = min(rated, key=lambda x: (rated[x], x))
    return {"group": k, "rate": rated[k]}


def data_and_compute(plan: dict) -> dict:
    fr = PTP.frozen()["arms"]
    out = {}
    for arm in PTP.ARMS:
        u = next(x for x in plan["units"] if x["arm"] == arm)
        es, f = u["effective_schedule"], fr[arm]
        cells = (f.get("tool_composition") or {}).get("cells") or {}
        out[arm] = {"before_packing": f["before_packing"]["training"],
                    "packed_sequences": es["packed_sequences"], "optimizer_steps": es["optimizer_steps"],
                    "supervised_token_exposure": es["supervised_token_exposure"],
                    "non_padding_tokens_trained": es["non_padding_tokens_trained"],
                    "tool_calling": {"turns": sum(c.get("turns", 0) for c in cells.values()),
                                     "tool_calls": sum(c.get("tool_calls", 0) for c in cells.values()),
                                     "episodes": sum(c.get("episodes", 0) for c in cells.values()),
                                     "_": "turns are the tool-calling decisions (call, answer or refuse to call)"}}
    a1, a2 = out["A1-6cat-sr"], out["A2-6cat-sr"]
    ratio = {k: round(a2[k] / a1[k], 4) for k in ("packed_sequences", "optimizer_steps", "supervised_token_exposure",
                                                   "non_padding_tokens_trained")}
    ratio["tool_calling_turns"] = round(a2["tool_calling"]["turns"] / max(1, a1["tool_calling"]["turns"]), 4)
    return {"arms": out, "A2_over_A1": ratio,
            "_": "training FLOPs for one shared student geometry follow the packed-sequence ratio (every sequence is "
                 "8,192 positions under causal attention); the teacher's forward cost follows it too"}


def analyse(units_dir: Path, records: list[Path]) -> dict:
    plan = json.loads(PTP.BOUND.read_text())
    prereg = json.loads(PREREG.read_text())
    MS.check_bound(prereg)
    items, _ = fluid_meta()
    per, rows, by_arm = {}, [], defaultdict(list)
    for u in plan["units"]:
        d = Path(units_dir) / u["unit"]
        if not (d / "evaluation" / "formal_report.json").is_file():
            per[u["unit"]] = {"status": "not_run"}
            continue
        ev = AA.evidence(d)
        f = AA.fluid(ev, items)
        f["_axis"] = {i: items[i]["axis"] for i in f["item_correct"]}
        rows += decision_rows(u["unit"], u["arm"], u["seed"], f, items)
        by_arm[u["arm"]].append(f)
        caps = capabilities(ev["formal_report"])
        per[u["unit"]] = {"arm": u["arm"], "seed": u["seed"],
                          "fluid": {k: v for k, v in f.items() if k not in ("item_correct", "_axis")},
                          "worst_static_axis": worst(f["by_axis"]), "usable": usable_correct(ev, items),
                          "capabilities": caps,
                          "worst_unit_per_capability": {c: worst(v["units"]) for c, v in caps.items()},
                          "training": AA.training(d)}
    arms = {}
    for arm, fs in by_arm.items():
        arms[arm] = {"seeds": len(fs), "static_all": AA.paired_mean(fs, None) if len(fs) == 2 else None,
                     "by_axis": {ax: AA.paired_mean(fs, ax) for ax in STATIC_AXES} if len(fs) == 2 else None}
    comps = {}
    for key, (x, y, role) in COMPARISONS.items():
        if len(by_arm.get(x, [])) == 2 and len(by_arm.get(y, [])) == 2:
            c = MS.compare(rows, x, y)
            comps[key] = {"role": role, **c, "reading": READING[c["primary"]["label"]]}
        else:
            comps[key] = {"role": role, "status": "not_run"}
    measured = {}
    for p in records:
        rec = json.loads(Path(p).read_text())
        for name, row in (rec.get("units") or {}).items():
            measured[name] = {k: row.get(k) for k in ("train_seconds", "eval_seconds", "telemetry", "resume_from",
                                                       "final_model_safetensors_sha256", "final_model_path",
                                                       "off_volume_copy") if row.get(k) is not None}
    return {"schema": "aadistill.stage2.teacher_modes.stage3_sr_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "study_id": PTP.STUDY_ID, "plan_sha256": __import__("hashlib").sha256(PTP.BOUND.read_bytes()).hexdigest(),
            "set": "eval-set-1 v2 (frozen), scoring protocol v2, validity mask v1: 570 static + 144 interactive formal",
            "decoding": {"protocol_id": "nucleus_v1", "sha256": AA.DECODING_SHA},
            "rule": {"id": MS.RULE_ID, "bound_sha256": MS.bound_identity(),
                     "constants": prereg["decision_rule"]["constants"]},
            "units": per, "arms": arms, "comparisons": comps, "data_and_compute": data_and_compute(plan),
            "measured_runs": measured,
            "limits": ["A1 - A2 is matched in unique supervised tokens before packing and in category composition, "
                       "not in compute, optimizer steps, supervised-token exposure or tool-calling decisions",
                       "A1 - C1 compares two lineages end to end: C1's initialization came from its own root's D4 "
                       "search under forward baseline v1, A1's from the packed search; not a pure teacher difference",
                       "eval-set-1 covers tool calling only indirectly: no number here is a closed-loop tool-task "
                       "success rate; that needs a separately authorized real tool-calling evaluation",
                       "interactive episodes are exploratory and outside the decision"],
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", required=True)
    ap.add_argument("--units-record", action="append", default=[])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    doc = analyse(Path(a.units), [Path(p) for p in a.units_record])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: (v.get("primary", {}).get("label"), v.get("primary", {}).get("delta"),
                          v.get("primary", {}).get("ci")) if "primary" in v else v.get("status")
                      for k, v in doc["comparisons"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

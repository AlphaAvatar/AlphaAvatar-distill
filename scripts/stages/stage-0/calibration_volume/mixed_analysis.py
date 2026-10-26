#!/usr/bin/env python3
"""The mixed-calibration study's result at $0, from the units' evidence: the preregistered rule, then every capability apart.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/mixed_analysis.py [--mc1-store <dir>]

1. IDENTITY. The two Mix-Augmented units (``EVIDENCE_MIX/<unit>/``, MC2's
   evidence archives extracted) must be formal, name their arm, seed and
   execution, and carry exactly the inputs `h1pa_driver.unit_inputs` derives
   from the mixed plan -- the preregistration, the eval set, the decoding
   protocol, the MC1 Top-1's identity, the merged set, D4-Augmented's recipe
   identity and schedule. The REUSED units (D4-Augmented x2, PCA-Original x2,
   ``EVIDENCE_CONTROL``) must be the ones the preregistration names: their
   evaluation descriptors hash to the recorded sha256s. Nothing is
   regenerated or rescored under another protocol; every generation summary
   must carry the preregistered decoding protocol.
2. THE DECISION (`mixed_statistics.decide`, bound by the preregistration): M =
   Mix-Augmented - D4-Augmented decided, E = Mix-Augmented - PCA-Original
   descriptive, on the formal static fluid items, clustered by rule family.
   Interactive episodes are exploratory, beside the decision.
3. EVERY CAPABILITY APART, each unit on its own: the fluid axes by generator
   tier (unseen rule vs unseen format), every eval-set-1 unit correct AND usable
   side by side (`h1pa_fullset.unit_table`), protocol integrity (natural
   termination, degeneration stops, context-limit hits), and the step-0
   teacher KL as a diagnostic only. No pooled score is made.

Writes ``logs/stages/stage-0/calibration_volume/analyses/mixed_result.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EVIDENCE_CONTROL = REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/units/evidence"
EVIDENCE_MIX = REPO / "artifacts/stages/stage-0/calibration_volume/mixed/units/evidence"
OUT = REPO / "logs/stages/stage-0/calibration_volume/analyses/mixed_result.json"
FLUID_KEY = "fluid_reasoning_720__dev_v1"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _driver():
    spec = importlib.util.spec_from_file_location(
        "mixed_recovery_driver", REPO / "scripts/stages/stage-0/calibration_volume/mixed_recovery_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expected_mix_inputs(plan: dict, u: dict) -> dict:
    from stages.reasoning_data import h1pa_driver as HD
    sched = dict(plan["mixed"]["control_schedule"])
    probe = HD.derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-")
    return HD.unit_inputs(plan, u, sched, HD.recipe_identity(probe))


def load_unit(d: Path, decoding_sha256: str) -> dict:
    from stages.reasoning_data import h1pa_driver as HD
    rep = json.loads((d / "report/report.json").read_text())
    bad = HD.generation_refusals(d / "evaluation/generations", decoding_sha256)
    if bad:
        raise SystemExit(f"{d.name}: generations not under the preregistered decoding: {bad[:3]}")
    summary = json.loads((d / f"evaluation/generations/{FLUID_KEY}.summary.json").read_text())
    rows = {f.name.removesuffix(".rows.jsonl"): [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            for f in sorted((d / "evaluation/rows").glob("*.rows.jsonl"))}
    return {"dir": d, "report": rep, "rows": rows, "decoding_sha256": HD._decoding_of(summary),
            "formal_report": json.loads((d / "evaluation/formal_report.json").read_text()),
            "descriptor_sha256": _sha(d / "evaluation.descriptor.json")}


def integrity(d: Path) -> dict:
    """Natural termination, degeneration stops and context-limit hits, per generated unit."""
    out = {}
    for s in sorted((d / "evaluation/generations").glob("*.summary.json")):
        doc = json.loads(s.read_text())
        key = s.name.removesuffix(".summary.json")
        out[key] = ({"ends": doc["ends"]} if "ends" in doc else
                    {k: doc.get(k) for k in ("stop_reasons", "natural_termination_rate", "degeneration_rate",
                                             "context_limit_rate", "right_censored_rate")})
    return out


def step0_kl(d: Path) -> dict | None:
    log = d / "training/trainer/train_log.jsonl"
    if not log.is_file():
        return None
    for line in log.read_text().splitlines():
        j = json.loads(line)
        if j.get("event") == "eval_result" and j.get("step") == 0:
            return {"val_kd": j.get("val_kd"), "val_ce": j.get("val_ce"),
                    "_": "diagnostic only: E1's 16 validation sequences at step 0, never a reasoning result"}
    return None


def analyse(mc1_store: Path | None = None) -> dict:
    from stages.calibration_volume import mixed_statistics as MS
    from stages.reasoning_data import h1pa_fullset as HF
    from stages.reasoning_data.h1_analysis import STATIC_AXES, capabilities, fluid_meta, rate
    from shared import eval_sets as ES
    M = _driver()
    plan = M.load_plan()
    prereg = plan["prereg"]
    MS.check_bound(prereg)
    dec = prereg["evaluation"]["decoding"]["sha256"]
    units = {}
    for u in plan["units"]:
        ev = load_unit(EVIDENCE_MIX / u["unit"], dec)
        rep = ev["report"]
        if (rep.get("formal") is not True or rep.get("unit") != u["unit"] or rep.get("execution") != "train"
                or rep.get("inputs") != json.loads(json.dumps(expected_mix_inputs(plan, u)))):
            raise SystemExit(f"{u['unit']}: its report is not this frozen unit's")
        units[u["unit"]] = {"arm": u["arm"], "seed": u["seed"], **ev}
    for name, rec in prereg["control_evidence"]["units"].items():
        ev = load_unit(EVIDENCE_CONTROL / name, dec)
        if ev["descriptor_sha256"] != rec["evaluation_descriptor_sha256"]:
            raise SystemExit(f"{name}: its evaluation descriptor is not the one the preregistration reuses")
        arm, seed = name.rsplit("-s", 1)
        units[name] = {"arm": arm, "seed": int(seed), **ev}
    items, eval_only = fluid_meta()
    rows, by = [], defaultdict(list)
    for u in units.values():
        for r in u["rows"][FLUID_KEY]:
            if not r.get("formal"):
                continue
            it = items[r["id"]]
            tier = "unseen_format" if it["generator_id"] in eval_only else "unseen_rule"
            by[(u["arm"], u["seed"], it["axis"], tier)].append(r)
            if it.get("interactive"):
                continue
            rows.append({"arm": u["arm"], "seed": u["seed"], "item_id": r["id"], "cluster": it["family_id"],
                         "axis": it["axis"], "correct": bool(r.get("correct")),
                         "decoding_sha256": u["decoding_sha256"]})
    decision = MS.decide(rows, prereg=prereg)
    axes = defaultdict(dict)
    for (arm, seed, axis, tier), rs in sorted(by.items()):
        axes[axis].setdefault(f"{arm}-s{seed}", {})[tier] = rate(rs)
    for axis, cells in axes.items():
        for cell, tiers in cells.items():
            tiers["all"] = rate([r for (a, s, ax, _t), rs in by.items() if ax == axis and f"{a}-s{s}" == cell
                                 for r in rs])
    cap_items = {u: {r["id"]: r for r in ES.items("eval-set-1", HF.CAP, u)} for u in
                 ("math_verified", "gsm8k", "multihop", "rag", "answerability_paired", "safety_paired", "knowledge",
                  "eval_behavior_v0")}
    cap_items["legacy"] = {r["id"]: r for r in ES.items("eval-set-1", "stage3_legacy_150", "items")}
    with tempfile.TemporaryDirectory() as tmp:
        fullset = {n: HF.unit_table(u["dir"] / "evaluation", Path(tmp) / n, cap_items) for n, u in units.items()}
    top1 = json.loads((REPO / plan["mixed"]["initialization"]["record"]).read_text())
    return {"schema": "aadistill.stage0.calibration_volume.mixed_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preregistration": {"path": str(M.PREREG.relative_to(REPO)), "sha256": plan["prereg_sha256"]},
            "decision_rule_bound_sha256": MS.bound_identity(),
            "initialization": {"mix_top1": top1["materialization_identity"],
                               "record": plan["mixed"]["initialization"]["record"]},
            "units": {n: {"arm": u["arm"], "seed": u["seed"], "run_id": u["report"]["run_id"],
                          "attempt": u["report"]["attempt"], "checkpoint": u["report"]["checkpoint"],
                          "seconds": u["report"].get("seconds"),
                          "evaluation_descriptor_sha256": u["descriptor_sha256"],
                          "reused": u["arm"] != "Mix-Augmented"} for n, u in units.items()},
            "decision": decision,
            "primary_comparison": decision["primary_comparison"],
            "primary_label": decision["primary_label"],
            "by_axis_and_tier": dict(axes),
            "interactive_exploratory": {a: c for a, c in axes.items() if a not in STATIC_AXES},
            "eval_set_1_correct_and_usable": fullset,
            "capabilities_by_unit": {n: capabilities(u["formal_report"]) for n, u in units.items()},
            "protocol_integrity": {n: integrity(u["dir"]) for n, u in units.items()},
            "step0_teacher_kl_diagnostic": {n: step0_kl(u["dir"]) for n, u in units.items()},
            "reading": ("M is decided under the preregistered rule on static fluid correctness under nucleus_v1; the "
                        "two arms differ only in the initialization the recovery starts from, and inside it "
                        "calibration composition, volume, interactive data, profile choice and DEPTH weighting change "
                        "together, so M is never read as any one of them; E is descriptive; a floor is NOT_RESOLVED "
                        "(FLOOR), never equivalence; usable rollout, step-0 KL and loss are never read as reasoning"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc1-store", default=None)
    args = ap.parse_args(argv)
    doc = analyse(Path(args.mc1_store) if args.mc1_store else None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(OUT.relative_to(REPO), {k: (v.get("primary") or {}).get("label", v.get("label"))
                                  for k, v in doc["decision"]["comparisons"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

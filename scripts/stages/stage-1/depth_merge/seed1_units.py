#!/usr/bin/env python3
"""Seed-1 descriptive reading beside the stop check (`seed1_check.py`): for V1 (depth_merge) and SOB1's depth_only,
seed 20260726, both modes, the registered scorer's formal rows of every eval-set-1 unit other than the static fluid
core -- the interactive episodes and each supporting unit -- as n and accuracy. Descriptive only: one seed, no test.
`stage3_legacy_150__items` is in-sample (training prompts), never a generalization number.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python seed1_units.py <V1 secured out> <record.json>
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for p in ("src", "scripts", "scripts/stages/stage-1"):
    sys.path.insert(0, str(REPO / p))

from stages.depth_merge import analysis as AN  # noqa: E402
from stages.depth_merge import common as DC  # noqa: E402
from stages.single_operator_budget import analysis as SA  # noqa: E402

SEED = 20260726
a_out, dest = Path(sys.argv[1]), Path(sys.argv[2])
modes = DC.design()["evaluation"]["protocol"]["modes"]
res = {}
with tempfile.TemporaryDirectory() as tmp:
    for arm, out in (("depth_merge", a_out), (DC.SOB1_DEPTH, DC.sob1_depth_evidence())):
        for m in AN.MODES:
            gd = out / f"{arm}-s{SEED}" / modes[m]["dir"] / "generations"
            r = SA._set_reading(gd, f"{arm}_s{SEED}", Path(tmp) / f"{arm}_{m}")
            n_i = len(r["interactive"])
            res[f"{arm}/{m}"] = {
                "interactive_episodes": {"n": n_i, "accuracy_pct": round(100 * sum(r["interactive"].values()) / max(1, n_i), 2)},
                "units": {u: {"n": len(v), "accuracy_pct": round(100 * sum(v.values()) / max(1, len(v)), 2)}
                          for u, v in r["units"].items() if v},
                "digest": r["digest"]}
doc = {"schema": "aadistill.dmr1.seed1_supporting_units/v1", "seed": SEED, "pair": ["depth_merge", "depth_only"],
       "scope": "one training seed, descriptive: no test, no verdict; eval-set-1 is development-exposed; "
                "stage3_legacy_150__items is in-sample (training prompts)",
       "evidence": {"depth_merge": str(a_out), "depth_only": str(DC.sob1_depth_evidence())}, "readings": res}
dest.write_text(json.dumps(doc, indent=1) + "\n")
for k, v in res.items():
    print(k, v["interactive_episodes"], {u: x["accuracy_pct"] for u, x in v["units"].items()})

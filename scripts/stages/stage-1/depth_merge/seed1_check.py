#!/usr/bin/env python3
"""The maintainer's seed-1 stop rule (2026-11-27): V1 (depth_merge) seed 20260726 against SOB1's depth_only seed
20260726, paired by item, on eval-set-1's static fluid formal core (570 items, 558 groups), the registered scorer
(`depth_merge.analysis.read_arm`), the registered per-seed statistic (`single_operator_budget.analysis._paired`: mean
paired difference and its 95% group-bootstrap interval).

STOP (do not train seed 2) iff, in thinking mode (the registered primary): delta <= -0.050 AND the interval's upper
bound < 0. Non-thinking is reported, not part of the rule. Fixed before any seed-1 result was read.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/seed1_check.py \\
        <V1 secured out> <record.json>
Exit 10 = STOP, 0 = CONTINUE, 2 = not readable yet (never a stop).
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
from stages.teacher_parent_initialization import analysis as TA  # noqa: E402

SEED, DELTA_MAX, = 20260726, -0.050
a_out, dest = Path(sys.argv[1]), Path(sys.argv[2])
d = DC.design()
modes = d["evaluation"]["protocol"]["modes"]
primary = TA.registered_items(d)
groups = TA.independence_groups(primary, d)
for m in AN.MODES:
    c = a_out / f"depth_merge-s{SEED}" / modes[m]["dir"] / "completion.json"
    if not c.is_file() or json.loads(c.read_text()).get("complete") is not True:
        print(f"NOT_READY: {c} absent or not complete")
        raise SystemExit(2)
with tempfile.TemporaryDirectory() as tmp:
    ya, da, ta = AN.read_arm(a_out, "depth_merge", [SEED], primary, Path(tmp) / "a")
    yb, db, tb = AN.read_arm(DC.sob1_depth_evidence(), DC.SOB1_DEPTH, [SEED], primary, Path(tmp) / "b")
rows = {}
for m in AN.MODES:
    if sorted(ya[m][SEED]) != sorted(primary) or sorted(yb[m][SEED]) != sorted(primary):
        print(f"NOT_READY: {m} does not cover the {len(primary)} registered items")
        raise SystemExit(2)
    r = SA._paired({SEED: ya[m][SEED]}, {SEED: yb[m][SEED]}, groups, [SEED])
    rows[m] = {"delta_pp": round(100 * r["delta"], 2), "accuracy_pct": {"depth_merge": round(100 * r["accuracy"][0], 2),
                                                                         "depth_only": round(100 * r["accuracy"][1], 2)},
               "ci95_pp": [round(100 * x, 2) for x in r["effect_interval"]["ci95_percentile"]],
               "n_paired_records": r["n_paired_records"], "raw": r}
ht = rows["H-T"]["raw"]
stop = ht["delta"] <= DELTA_MAX and ht["effect_interval"]["ci95_percentile"][1] < 0
doc = {"schema": "aadistill.dmr1.seed1_stop_check/v1", "seed": SEED, "pair": ["depth_merge", "depth_only"],
       "rule": "STOP iff thinking-mode delta <= -5.0 pp AND its 95% group-bootstrap upper bound < 0 (maintainer, "
               "2026-11-27: stop before seed 2 if seed 1 is much worse; threshold fixed before any seed-1 result)",
       "decision": "STOP" if stop else "CONTINUE", "modes": rows, "scored_digests": {**da, **db},
       "terminations": {**ta, **tb}, "n_items": len(primary), "n_groups": len(groups),
       "evidence": {"depth_merge": str(a_out), "depth_only": str(DC.sob1_depth_evidence())},
       "scope": "one training seed, descriptive: no test, no verdict; eval-set-1 is development-exposed"}
dest.write_text(json.dumps(doc, indent=1, default=str) + "\n")
print(json.dumps({"decision": doc["decision"], **{m: {k: v for k, v in r.items() if k != "raw"} for m, r in rows.items()}}))
raise SystemExit(10 if stop else 0)

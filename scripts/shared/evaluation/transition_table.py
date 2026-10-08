#!/usr/bin/env python
"""What did scaling actually repair? Per-prompt transitions between two arms.

    PYTHONPATH=src python scripts/evaluation/transition_table.py \
        --from P2-ceheavy --to E4-P2-1600k \
        --out artifacts/audit/e5_transition_table.json

Aggregate rates say usable rollout rose +0.20 and correctness did not move. That
is compatible with several different stories — a uniform lift, a swap where as
many prompts broke as were fixed, or a repair concentrated in one failure mode.
Only the per-prompt transition tells them apart, and the answer determines what
a recovery corpus should be built from.

Cells are the (before, after) product of two binary axes: whether the rollout was
usable, and whether the answer was correct. `unusable → usable_wrong` is the
interesting one for E5: prompts where scaling bought a well-formed trajectory
that still gets the wrong answer.

Failure attribution uses the first failing component in the fixed
`usable_rollout` order, which is a presentation aid — the components co-occur,
so the per-component before/after rates are reported beside it.

Nothing is generated and no model is loaded: retained generations are re-read and
correctness is re-scored with the corrected scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.evaluation import usable_rollout as ur  # noqa: E402
from aadistill.evaluation.behavior import split_generation  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from run_three_mode_diagnostic import score  # noqa: E402

AUDIT = REPO_ROOT / "artifacts/audit"
SESSIONS = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
CELLS = ["usable_correct", "usable_wrong", "unusable"]


def load_arm(alias: str) -> dict:
    """Per-prompt state for one evaluated arm."""
    recs = [json.loads(line)
            for line in (AUDIT / "three_mode" / alias / "free.generations.jsonl").open()
            if line.strip()]
    sessions = {}
    for line in SESSIONS.open():
        s = json.loads(line)
        sessions[s["id"]] = s
    out = {}
    for r in recs:
        body = split_generation(r["raw"], think_preopened=True)["answer"]
        correct = bool(score(sessions[r["id"]], body.strip()))
        comp = ur.components(r)
        usable = all(comp.values())
        first_fail = next((k for k in ur.COMPONENTS if not comp[k]), None)
        out[r["id"]] = {
            "usable": usable,
            "correct": correct,
            "cell": ("usable_correct" if usable and correct
                     else "usable_wrong" if usable else "unusable"),
            "first_failure": first_fail,
            "components": comp,
            "data_type": r["data_type"],
            "generated_tokens": r["generated_tokens"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="P2-ceheavy")
    ap.add_argument("--to", dest="dst", default="E4-P2-1600k")
    ap.add_argument("--seeds", nargs="+", default=["sa", "sb"])
    ap.add_argument("--out", type=Path, default=AUDIT / "e5_transition_table.json")
    args = ap.parse_args()

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "from": args.src, "to": args.dst,
        "note": ("cells are (usable, correct) products; 'unusable' ignores "
                 "correctness because an unusable rollout's answer is not a "
                 "meaningful measurement"),
        "per_seed": {}, "pooled": {},
    }
    pooled_tx: Counter = Counter()
    pooled_repaired: Counter = Counter()
    pooled_broken: Counter = Counter()
    pooled_type: dict[str, Counter] = {}

    for seed in args.seeds:
        a = load_arm(f"{args.src}-{seed}")
        b = load_arm(f"{args.dst}-{seed}")
        ids = sorted(set(a) & set(b))
        tx: Counter = Counter()
        repaired: Counter = Counter()   # first-failure type of prompts that became usable
        broken: Counter = Counter()     # first-failure type prompts fell into
        by_type: dict[str, Counter] = {}
        for i in ids:
            tx[(a[i]["cell"], b[i]["cell"])] += 1
            by_type.setdefault(a[i]["data_type"], Counter())[
                (a[i]["cell"], b[i]["cell"])] += 1
            if not a[i]["usable"] and b[i]["usable"]:
                repaired[a[i]["first_failure"]] += 1
            if a[i]["usable"] and not b[i]["usable"]:
                broken[b[i]["first_failure"]] += 1
        pooled_tx += tx
        pooled_repaired += repaired
        pooled_broken += broken
        for t, c in by_type.items():
            pooled_type.setdefault(t, Counter()).update(c)

        comp_rates = {
            k: {"before": round(sum(a[i]["components"][k] for i in ids) / len(ids), 4),
                "after": round(sum(b[i]["components"][k] for i in ids) / len(ids), 4)}
            for k in ur.COMPONENTS}
        report["per_seed"][seed] = {
            "n": len(ids),
            "transitions": {f"{x}->{y}": n for (x, y), n in sorted(tx.items())},
            "repaired_by_original_first_failure": dict(repaired.most_common()),
            "broken_into_first_failure": dict(broken.most_common()),
            "component_rates": comp_rates,
        }

    n = sum(pooled_tx.values())
    report["pooled"] = {
        "n": n,
        "transitions": {f"{x}->{y}": v for (x, y), v in sorted(pooled_tx.items())},
        "transition_share": {f"{x}->{y}": round(v / n, 4)
                             for (x, y), v in sorted(pooled_tx.items())},
        "repaired_by_original_first_failure": dict(pooled_repaired.most_common()),
        "broken_into_first_failure": dict(pooled_broken.most_common()),
        "by_task": {t: {f"{x}->{y}": v for (x, y), v in sorted(c.items())}
                    for t, c in sorted(pooled_type.items())},
    }
    # The quantity E5 exists to attack.
    gained_usable = sum(v for (x, y), v in pooled_tx.items()
                        if x == "unusable" and y.startswith("usable"))
    gained_wrong = pooled_tx[("unusable", "usable_wrong")]
    report["pooled"]["headline"] = {
        "became_usable": gained_usable,
        "of_which_still_wrong": gained_wrong,
        "share_of_repairs_still_wrong": (round(gained_wrong / gained_usable, 4)
                                         if gained_usable else None),
        "still_unusable_after": sum(v for (x, y), v in pooled_tx.items()
                                    if y == "unusable"),
        "usable_wrong_after": sum(v for (x, y), v in pooled_tx.items()
                                  if y == "usable_wrong"),
    }
    report["code_state"] = code_state(REPO_ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(f"pooled over {args.seeds}, n={n}\n")
    print(f"{'from \\\\ to':18s}" + "".join(f"{c:>16s}" for c in CELLS))
    for x in CELLS:
        row = "".join(f"{pooled_tx[(x, y)]:16d}" for y in CELLS)
        print(f"{x:18s}{row}")
    h = report["pooled"]["headline"]
    print(f"\nbecame usable: {h['became_usable']}, of which still wrong: "
          f"{h['of_which_still_wrong']} ({h['share_of_repairs_still_wrong']:.1%})")
    print(f"still unusable after: {h['still_unusable_after']}  |  "
          f"usable-but-wrong after: {h['usable_wrong_after']}")
    print(f"\nrepaired, by the failure they originally showed:")
    for k, v in report["pooled"]["repaired_by_original_first_failure"].items():
        print(f"  {k:24s} {v}")
    if report["pooled"]["broken_into_first_failure"]:
        print("regressed into:")
        for k, v in report["pooled"]["broken_into_first_failure"].items():
            print(f"  {k:24s} {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score generations on a registered eval set, per unit and per capability.

    PYTHONPATH=src:scripts python scripts/shared/evaluation/eval_set_scoring.py conformance
    PYTHONPATH=src:scripts python scripts/shared/evaluation/eval_set_scoring.py report \
        --generations <dir of <component>__<unit>.jsonl> --label <model> --out <report.json>

The cross-stage entry for formal capability measurement (AGENTS.md P10.3). The
set is read through `shared.eval_sets`, which refuses it unless its bytes match
the registered identity. Each component is scored by the adapter its manifest
names, and each adapter delegates to the scorer that component was frozen
under -- nothing here re-implements a rule:

* `score_fluid`      -> stages.calibration_volume.score_battery (static + episodes)
* `score_capability` -> aadistill.evaluation.capability.SCORERS / strict_answer.score_numeric
* `score_behavior`   -> aadistill.evaluation.behavior.score_sample
* `score_legacy`     -> stages.d0.run_three_mode_diagnostic.score (free mode)

`conformance` re-scores the registered conformance vectors and fails on any
difference: that is how a change of scoring semantics under a protected set is
caught, without freezing the scorer's bytes.

`report` writes one row per unit and one per declared capability and NEVER a
pooled number across units or capabilities. A unit with no generations is
`not_run`; a withheld unit without its verified local copy is `withheld`; a
unit the model's interface cannot take (`--unsupported unit=reason`) is
`unsupported`. None of these is ever counted as zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation import eval_registry as ER  # noqa: E402
from shared import eval_sets as ES  # noqa: E402

CAPABILITY_RECORD_FIELDS = ("raw", "think_preopened", "natural_termination",
                            "degeneration_triggered", "degeneration_kind")
BEHAVIOR_RECORD_FIELDS = ("raw", "stop_reason")


def _plain(x):
    return json.loads(json.dumps(x))


# -- fluid_reasoning_720 ------------------------------------------------------
#: The end-of-turn marker a terminated generation's decoded text ends with; the
#: protocol check requires it, so a synthetic reply without it is unusable.
EOS_TEXT = "<|im_end|>"


def synthetic_turn(answer_text: str, *, terminated: bool = True) -> dict:
    """A minimal well-formed turn whose final answer is `answer_text`."""
    end = EOS_TEXT if terminated else ""
    return {"raw": f"reasoning</think>\n\n{answer_text}{end}", "answer": answer_text,
            "natural_termination": terminated, "degeneration_triggered": False,
            "context_limit_reached": False, "think_preopened": True,
            "generated_tokens": 8, "stop_reason": "eos" if terminated else "length"}


def _full_turn(t: dict) -> dict:
    eos = t.get("stop_reason") == "eos"
    return {"raw": t.get("raw", t.get("answer") or ""), "answer": t.get("answer") or "",
            "natural_termination": t.get("natural_termination", eos),
            "degeneration_triggered": t.get("degeneration_triggered", False),
            "context_limit_reached": t.get("context_limit_reached", False),
            "think_preopened": t.get("think_preopened", True),
            "generated_tokens": t.get("generated_tokens", 0),
            "stop_reason": t.get("stop_reason")}


def score_fluid(task: dict, record: dict) -> dict:
    from stages.calibration_volume import score_battery as SB
    rec = {"task_id": task["task_id"], "label": "conformance",
           "turns": [_full_turn(t) for t in record["turns"]], "end": record.get("end")}
    if task["interactive"]:
        out = SB.score_episode(task, rec)
        return _plain({"correct": out["correct"], "success": out["env"].get("success")})
    out = SB.score_static(task, rec)
    return _plain({"correct": out["correct"], "extracted": out.get("extracted"),
                   "reason": out["reason"]})


# -- stage3_capability_846 ----------------------------------------------------
def score_capability(set_name: str, sample: dict, record: dict) -> dict:
    from aadistill.evaluation.capability import SCORERS
    from aadistill.evaluation.strict_answer import score_numeric
    v = (score_numeric(record, sample["gsm8k_answer"]) if set_name == "gsm8k"
         else SCORERS[set_name](record, sample))
    return _plain({"correct": v["correct"], "reason": v.get("reason")})


def score_behavior(sample: dict, record: dict) -> dict:
    from aadistill.evaluation.behavior import score_sample
    v = score_sample(sample, record.get("raw", ""),
                     hit_cap=record.get("stop_reason") not in (None, "eos"))
    return _plain({k: val for k, val in v.items() if not isinstance(val, (dict, list))})


# -- stage3_legacy_150 --------------------------------------------------------
def score_legacy(item: dict, record: dict) -> dict:
    from aadistill.evaluation.behavior import split_generation
    from stages.d0.run_three_mode_diagnostic import score
    answer = split_generation(record["raw"], think_preopened=True)["answer"]
    return _plain({"correct": score(item, answer)})


ADAPTERS = {
    "fluid_reasoning_720": lambda unit, item, rec: score_fluid(item, rec),
    "stage3_capability_846": lambda unit, item, rec: (
        score_behavior(item, rec) if unit == "eval_behavior_v0"
        else score_capability(unit, item, rec)),
    "stage3_legacy_150": lambda unit, item, rec: score_legacy(item, rec),
}


def conformance(eval_set_id: str | None = None) -> dict:
    """Re-score every registered conformance vector; report every difference."""
    eval_set_id = eval_set_id or ES.default_formal()
    m = ES.manifest(eval_set_id)
    root = ES.set_root(eval_set_id)
    out = {}
    for cname, comp in m["components"].items():
        path = root / comp["scorer"]["conformance"]
        index = {}
        for uname, spec in comp["units"].items():
            if spec.get("withheld"):
                continue
            index[uname] = {r[spec["id_field"]]: r for r in ES.items(eval_set_id, cname, uname)}
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        diffs = []
        for r in rows:
            got = ADAPTERS[cname](r["unit"], index[r["unit"]][r["id"]], r["input"])
            if got != r["expected"]:
                diffs.append({"unit": r["unit"], "id": r["id"], "variant": r["variant"],
                              "expected": r["expected"], "got": got})
        out[cname] = {"vectors": len(rows), "differences": len(diffs), "examples": diffs[:5]}
    return out


def _slice(unit_key: str) -> tuple[str, str, tuple[str, str] | None]:
    base, _, sel = unit_key.partition("[")
    comp, unit = base.split("/", 1)
    if not sel:
        return comp, unit, None
    field, value = sel.rstrip("]").split("=", 1)
    return comp, unit, (field, value)


def unit_metrics(cname: str, uname: str, items: dict, recs: list[dict], id_field: str,
                 repeat: set[str]) -> tuple[dict, list[dict]]:
    """A unit's registered metrics, from the aggregator its scorer was frozen with.

    Correctness units (fluid, legacy, the capability sets) headline their rate;
    the paired sets headline PAIR accuracy (`score_battery.aggregate`); the
    behaviour set has no single correctness and reports `behavior.aggregate`'s
    per-group metrics with no headline, so nothing reads it as a zero."""
    rows = []
    for rec in recs:
        rid = rec.get(id_field, rec.get("id"))
        rows.append({"id": rid, **ADAPTERS[cname](uname, items[rid], rec)})
    if cname == "stage3_capability_846" and uname == "eval_behavior_v0":
        from aadistill.evaluation.behavior import aggregate, score_sample
        scored = [score_sample(items[r.get(id_field, r.get("id"))], r.get("raw", ""),
                               hit_cap=r.get("stop_reason") not in (None, "eos")) for r in recs]
        return {"headline": None, "metrics": aggregate(scored)}, rows
    if cname == "stage3_capability_846":
        from aadistill.evaluation.capability import PAIRED_SETS, SCORERS
        from aadistill.evaluation.strict_answer import score_numeric
        from shared.evaluation.score_battery import aggregate
        verdicts, ordered = [], []
        for rec in recs:
            sample = items[rec["id"]]
            verdicts.append(score_numeric(rec, sample["gsm8k_answer"]) if uname == "gsm8k"
                            else SCORERS[uname](rec, sample))
            ordered.append(sample)
        agg = aggregate(uname, verdicts, ordered)
        head = "pair_correct" if uname in PAIRED_SETS else "correct"
        return {"headline": head, "headline_value": agg.get(head), "metrics": agg}, rows
    correct = sum(bool(r.get("correct")) for r in rows)
    indep = [r for r in rows if r["id"] not in repeat]
    n_ind = len(set(items) - repeat)
    return {"headline": "correct", "correct": correct, "rate": round(correct / len(items), 4),
            "n_independent": n_ind,
            "rate_independent": round(sum(bool(r.get("correct")) for r in indep) / n_ind, 4)}, rows


def report(generations: Path, label: str, unsupported: dict[str, str],
           eval_set_id: str | None = None) -> dict:
    """Per unit and per capability; generations are `<component>__<unit>.jsonl`
    rows `{id, ...record}` (static) or `{task_id, turns}` (episodes)."""
    eval_set_id = eval_set_id or ES.default_formal()
    m = ES.manifest(eval_set_id)
    per_unit_rows: dict[str, list[dict]] = {}
    outcomes: dict[str, dict] = {}
    for cname, comp in m["components"].items():
        for uname, spec in comp["units"].items():
            key = f"{cname}/{uname}"
            if key in unsupported:
                outcomes[key] = {"status": "unsupported", "reason": unsupported[key]}
                continue
            gen = generations / f"{cname}__{uname}.jsonl"
            if not gen.is_file():
                outcomes[key] = {"status": "not_run"}
                continue
            try:
                items = {r[spec["id_field"]]: r for r in ES.items(eval_set_id, cname, uname)}
            except FileNotFoundError:
                outcomes[key] = {"status": "withheld", "reason": "no verified local copy"}
                continue
            recs = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
            # a duplicate group inside this unit is ONE independent item
            repeat = {x.split(":", 1)[1] for g in comp.get("duplicate_groups", [])
                      if all(y.split(":", 1)[0] == uname for y in g) for x in g[1:]}
            metrics, rows = unit_metrics(cname, uname, items, recs, spec["id_field"], repeat)
            per_unit_rows[key] = rows
            outcomes[key] = {"status": "measured", "n": len(rows),
                             "missing": len(set(items) - {r["id"] for r in rows}), **metrics}
    # capability slices of correctness units
    cap_outcomes = dict(outcomes)
    for cap, units in m["capabilities"].items():
        for u in units:
            comp, unit, sel = _slice(u)
            base = f"{comp}/{unit}"
            if sel is None or base not in per_unit_rows:
                if sel is not None:
                    cap_outcomes[u] = outcomes.get(base, {"status": "not_run"})
                continue
            items = {r[m["components"][comp]["units"][unit]["id_field"]]: r
                     for r in ES.items(eval_set_id, comp, unit)}
            ids = {i for i, r in items.items() if str(r.get(sel[0])) == sel[1]}
            if outcomes[base].get("headline") is None:      # behaviour groups
                cap_outcomes[u] = {"status": "measured", "n": len(ids),
                                   "metrics": outcomes[base]["metrics"]["by_group"].get(sel[1])
                                   if "by_group" in outcomes[base]["metrics"] else None}
                continue
            rows = [r for r in per_unit_rows[base] if r["id"] in ids]
            cap_outcomes[u] = {"status": "measured", "n": len(ids),
                               "correct": sum(bool(r.get("correct")) for r in rows),
                               "rate": round(sum(bool(r.get("correct")) for r in rows) / len(ids), 4)
                               if ids else None}
    return {"schema": "aadistill.eval_set_report/v1", "label": label,
            "eval_set_id": eval_set_id,
            "eval_set_identity": ER.current_identity(ER.entry(ES.registry(), eval_set_id)),
            "eval_set_status": ER.entry(ES.registry(), eval_set_id)["status"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "units": outcomes,
            "capabilities": ER.capability_table(m["capabilities"], cap_outcomes),
            "pooled": "none by contract: units and capabilities are reported separately"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conformance")
    c.add_argument("--eval-set", default=None)
    r = sub.add_parser("report")
    r.add_argument("--eval-set", default=None)
    r.add_argument("--generations", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--unsupported", nargs="*", default=[],
                   help="component/unit=reason for units the model's interface cannot take")
    args = ap.parse_args(argv)
    if args.cmd == "conformance":
        res = conformance(args.eval_set)
        print(json.dumps(res, indent=1))
        return 1 if any(v["differences"] for v in res.values()) else 0
    unsupported = dict(u.split("=", 1) for u in args.unsupported)
    out = report(args.generations, args.label, unsupported, args.eval_set)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["status"] for k, v in out["capabilities"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import collections
import json
import sys
from datetime import datetime, timezone
from collections.abc import Mapping
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


def score_fluid(task: dict, record: dict, protocol: str = "v1") -> dict:
    """`protocol` is the STATIC scoring protocol; an episode is always replayed
    under the action protocol its record was generated with (absent: v1)."""
    from stages.calibration_volume import score_battery as SB
    rec = {"task_id": task["task_id"], "label": "conformance",
           "turns": [_full_turn(t) for t in record["turns"]], "end": record.get("end"),
           **({"action_protocol": record["action_protocol"]}
              if record.get("action_protocol") else {})}
    if task["interactive"]:
        out = SB.score_episode(task, rec)
        return _plain({"correct": out["correct"], "success": out["env"].get("success")})
    out = SB.score_static(task, rec, protocol)
    return _plain({"correct": out["correct"], "extracted": out.get("extracted"),
                   "reason": out["reason"]})


# -- stage3_capability_846 ----------------------------------------------------
#: Capability protocol v2 (maintainer decision 2026-10-19): the v1 scorer,
#: applied to the reply with the surface forms that do not change an answer
#: normalized first (`tasks.common.normalize_v2`: Unicode minus and spaces,
#: `\\text{}` / `\\mathrm{}` / `\\mathbf{}`-style wrappers, LaTeX spacing,
#: markdown emphasis). An audit probe found `\\boxed{−12}`, `\\boxed{\\text{12}}`
#: and `\\boxed{\\mathbf{12}}` scored wrong by the strict numeric rule. Applied to
#: answer-correctness units only: the refusal classifier (safety_paired) and the
#: behaviour scorer (tool-call JSON, format checks) keep v1, where rewriting the
#: reply could change what they judge.
CAPABILITY_V2_UNITS = frozenset({"math_verified", "gsm8k", "multihop", "rag",
                                 "answerability_paired", "knowledge"})


def normalized_record(record: dict) -> dict:
    from stages.calibration_volume.tasks.common import normalize_v2
    return {**record, "raw": normalize_v2(record.get("raw") or "")}


def score_capability(set_name: str, sample: dict, record: dict, protocol: str = "v1") -> dict:
    from aadistill.evaluation.capability import SCORERS
    from aadistill.evaluation.strict_answer import score_numeric
    if protocol == "v2" and set_name in CAPABILITY_V2_UNITS:
        record = normalized_record(record)
    v = (score_numeric(record, sample["gsm8k_answer"]) if set_name == "gsm8k"
         else SCORERS[set_name](record, sample))
    return _plain({"correct": v["correct"], "reason": v.get("reason")})


def score_behavior(sample: dict, record: dict) -> dict:
    from aadistill.evaluation.behavior import score_sample
    v = score_sample(sample, record.get("raw", ""),
                     hit_cap=record.get("stop_reason") not in (None, "eos"))
    return _plain({k: val for k, val in v.items() if not isinstance(val, (dict, list))})


# -- stage3_legacy_150 --------------------------------------------------------
def score_legacy(item: dict, record: dict, protocol: str = "v1") -> dict:
    from aadistill.evaluation.behavior import split_generation
    from stages.d0.run_three_mode_diagnostic import score
    if protocol == "v2":
        record = normalized_record(record)
    answer = split_generation(record["raw"], think_preopened=True)["answer"]
    return _plain({"correct": score(item, answer)})


ADAPTERS = {
    "fluid_reasoning_720": lambda unit, item, rec, p="v1": score_fluid(item, rec, p),
    "stage3_capability_846": lambda unit, item, rec, p="v1": (
        score_behavior(item, rec) if unit == "eval_behavior_v0"
        else score_capability(unit, item, rec, p)),
    "stage3_legacy_150": lambda unit, item, rec, p="v1": score_legacy(item, rec, p),
}


def conformance_files(scorer: Mapping) -> dict[str, str]:
    """{protocol: file}. A manifest written before scorer protocols were
    versioned names one file, which pins v1."""
    conf = scorer["conformance"]
    return dict(conf) if isinstance(conf, Mapping) else {"v1": conf}


def formal_protocol(scorer: Mapping) -> str:
    return scorer.get("formal_protocol", "v1")


def conformance(eval_set_id: str | None = None) -> dict:
    """Re-score every registered conformance vector; report every difference."""
    eval_set_id = eval_set_id or ES.default_formal()
    m = ES.manifest(eval_set_id)
    root = ES.set_root(eval_set_id)
    out = {}
    for cname, comp in m["components"].items():
        index = {}
        for uname, spec in comp["units"].items():
            if spec.get("withheld"):
                continue
            index[uname] = {r[spec["id_field"]]: r for r in ES.items(eval_set_id, cname, uname)}
        for protocol, rel in conformance_files(comp["scorer"]).items():
            rows = [json.loads(l) for l in (root / rel).read_text().splitlines() if l.strip()]
            diffs = []
            for r in rows:
                got = ADAPTERS[cname](r["unit"], index[r["unit"]][r["id"]], r["input"], protocol)
                if got != r["expected"]:
                    diffs.append({"unit": r["unit"], "id": r["id"], "variant": r["variant"],
                                  "expected": r["expected"], "got": got})
            key = cname if protocol == "v1" and len(conformance_files(comp["scorer"])) == 1 \
                else f"{cname}@{protocol}"
            out[key] = {"vectors": len(rows), "differences": len(diffs), "examples": diffs[:5]}
    return out


def _slice(unit_key: str) -> tuple[str, str, tuple[str, str] | None]:
    base, _, sel = unit_key.partition("[")
    comp, unit = base.split("/", 1)
    if not sel:
        return comp, unit, None
    field, value = sel.rstrip("]").split("=", 1)
    return comp, unit, (field, value)


def unit_masks(eval_set_id: str, cname: str, comp: Mapping) -> dict[str, dict]:
    """Per unit: which rows the FORMAL numbers exclude, whether the unit is in the
    formal core, and its verified duplicate / equivalence groups (each counted
    once in independent statistics). A manifest without these files masks nothing."""
    root = ES.set_root(eval_set_id)
    out = {u: {"exclude": {}, "in_core": True, "groups": []} for u in comp["units"]}
    for g in comp.get("duplicate_groups", []):
        units = {x.split(":", 1)[0] for x in g}
        if len(units) == 1:
            out[units.pop()]["groups"].append([x.split(":", 1)[1] for x in g])
    if comp.get("validity"):
        v = json.loads((root / comp["validity"]["file"]).read_text())
        out[v["unit"]]["exclude"].update({e["id"]: f"validity: {e['class']}" for e in v["excluded"]})
    if comp.get("independence"):
        d = json.loads((root / comp["independence"]["file"]).read_text())
        out[d["unit"]]["groups"] += [g["ids"] for g in d["groups"]]
    if comp.get("core"):
        c = json.loads((root / comp["core"]["file"]).read_text())
        for u, spec in c["units"].items():
            out[u]["in_core"] = bool(spec.get("in_formal_core", True))
            out[u]["exclude"].update({i: "outside the formal core"
                                      for i in spec.get("excluded_from_core", {})})
            out[u]["exclude"].update({i: "outside held-out correctness (training overlap)"
                                      for i in spec.get("excluded_from_held_out", {})})
    return out


def unit_metrics(cname: str, uname: str, items: dict, recs: list[dict], id_field: str,
                 groups: list[list[str]], protocol: str = "v1",
                 keep: set[str] | None = None) -> tuple[dict, list[dict]]:
    """A unit's registered metrics over `keep` (default: every item), from the
    aggregator its scorer was frozen with.

    Correctness units (fluid, legacy, the capability sets) headline their rate;
    the paired sets headline PAIR accuracy (`score_battery.aggregate`); the
    behaviour set has no single correctness and reports `behavior.aggregate`'s
    per-group metrics with no headline, so nothing reads it as a zero. Each
    group in `groups` is ONE independent item (`rate_independent`)."""
    keep = set(items) if keep is None else set(keep)
    recs = [r for r in recs if r.get(id_field, r.get("id")) in keep]
    rows = []
    for rec in recs:
        rid = rec.get(id_field, rec.get("id"))
        rows.append({"id": rid, **ADAPTERS[cname](uname, items[rid], rec, protocol)})
    if cname == "stage3_capability_846" and uname == "eval_behavior_v0":
        from aadistill.evaluation.behavior import aggregate, score_sample
        scored = [score_sample(items[r.get(id_field, r.get("id"))], r.get("raw", ""),
                               hit_cap=r.get("stop_reason") not in (None, "eos")) for r in recs]
        return {"headline": None, "n_items": len(keep), "metrics": aggregate(scored)}, rows
    if cname == "stage3_capability_846":
        from aadistill.evaluation.capability import PAIRED_SETS, SCORERS
        from aadistill.evaluation.strict_answer import score_numeric
        from shared.evaluation.score_battery import aggregate
        verdicts, ordered = [], []
        for rec in recs:
            sample = items[rec["id"]]
            if protocol == "v2" and uname in CAPABILITY_V2_UNITS:
                rec = normalized_record(rec)
            verdicts.append(score_numeric(rec, sample["gsm8k_answer"]) if uname == "gsm8k"
                            else SCORERS[uname](rec, sample))
            ordered.append(sample)
        agg = aggregate(uname, verdicts, ordered)
        head = "pair_correct" if uname in PAIRED_SETS else "correct"
        return {"headline": head, "headline_value": agg.get(head), "n_items": len(keep),
                "metrics": agg}, rows
    correct = sum(bool(r.get("correct")) for r in rows)
    repeat = {x for g in groups for x in [i for i in g if i in keep][1:]}
    indep = [r for r in rows if r["id"] not in repeat]
    n_ind = len(keep - repeat)
    return {"headline": "correct", "n_items": len(keep), "correct": correct,
            "rate": round(correct / len(keep), 4) if keep else None,
            "n_independent": n_ind,
            "rate_independent": round(sum(bool(r.get("correct")) for r in indep) / n_ind, 4)
            if n_ind else None}, rows


def report(generations: Path, label: str, unsupported: dict[str, str],
           eval_set_id: str | None = None, rows_out: Path | None = None) -> dict:
    """Per unit and per capability; generations are `<component>__<unit>.jsonl`
    rows `{id, ...record}` (static) or `{task_id, turns}` (episodes).

    Every unit is scored under its component's FORMAL protocol. A unit's row
    carries the formal metrics (its validity / core / held-out exclusions
    applied) and, when anything was excluded, the metrics over every registered
    item beside them. A unit outside the formal core is still measured and
    reported, flagged `in_formal_core: false`."""
    eval_set_id = eval_set_id or default_formal_set()
    m = ES.manifest(eval_set_id)
    per_unit_rows: dict[str, list[dict]] = {}
    all_rows: dict[str, list[dict]] = {}
    formal_ids: dict[str, set] = {}
    outcomes: dict[str, dict] = {}
    for cname, comp in m["components"].items():
        protocol = formal_protocol(comp["scorer"])
        masks = unit_masks(eval_set_id, cname, comp)
        for uname, spec in comp["units"].items():
            key = f"{cname}/{uname}"
            mask = masks[uname]
            flags = {"in_formal_core": mask["in_core"], "protocol": protocol}
            if key in unsupported:
                outcomes[key] = {"status": "unsupported", "reason": unsupported[key], **flags}
                continue
            gen = generations / f"{cname}__{uname}.jsonl"
            if not gen.is_file():
                outcomes[key] = {"status": "not_run", **flags}
                continue
            try:
                items = {r[spec["id_field"]]: r for r in ES.items(eval_set_id, cname, uname)}
            except FileNotFoundError:
                outcomes[key] = {"status": "withheld", "reason": "no verified local copy", **flags}
                continue
            recs = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
            keep = set(items) - set(mask["exclude"])
            formal_ids[key] = keep
            formal, rows = unit_metrics(cname, uname, items, recs, spec["id_field"],
                                        mask["groups"], protocol, keep)
            per_unit_rows[key] = rows
            all_rows[key] = rows
            got = {r.get(spec["id_field"], r.get("id")) for r in recs}
            outcomes[key] = {"status": "measured", **flags, "n": len(rows),
                             "missing": len(keep - got), **formal}
            if mask["exclude"]:
                everything, all_rows[key] = unit_metrics(cname, uname, items, recs,
                                                         spec["id_field"], mask["groups"],
                                                         protocol)
                outcomes[key]["excluded_from_formal"] = dict(sorted(
                    collections.Counter(mask["exclude"].values()).items()))
                outcomes[key]["all_registered_items"] = everything
    if rows_out is not None:
        #: Every scored row, formal flag included: the per-item evidence a paired
        #: or clustered analysis reads, written once beside the report.
        rows_out.mkdir(parents=True, exist_ok=True)
        for key, rows in all_rows.items():
            body = "".join(json.dumps({**r, "formal": r["id"] in formal_ids[key]},
                                      ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
            (rows_out / f"{key.replace('/', '__')}.rows.jsonl").write_text(body)
    cap_outcomes = dict(outcomes)
    for cap, units in m["capabilities"].items():
        for u in units:
            comp, unit, sel = _slice(u)
            base = f"{comp}/{unit}"
            if sel is None or base not in per_unit_rows:
                if sel is not None:
                    cap_outcomes[u] = outcomes.get(base, {"status": "not_run"})
                continue
            spec = m["components"][comp]["units"][unit]
            items = {r[spec["id_field"]]: r for r in ES.items(eval_set_id, comp, unit)}
            ids = {i for i, r in items.items()
                   if str(r.get(sel[0])) == sel[1] and i in formal_ids[base]}
            flags = {k: outcomes[base][k] for k in ("in_formal_core", "protocol")}
            if outcomes[base].get("headline") is None:      # behaviour groups
                from aadistill.evaluation.behavior import aggregate, score_sample
                recs = [json.loads(l) for l in (generations / f"{comp}__{unit}.jsonl")
                        .read_text().splitlines() if l.strip()]
                scored = [score_sample(items[r["id"]], r.get("raw", ""),
                                       hit_cap=r.get("stop_reason") not in (None, "eos"))
                          for r in recs if r["id"] in ids]
                cap_outcomes[u] = {"status": "measured", **flags, "n": len(ids),
                                   "metrics": aggregate(scored) if scored else None}
                continue
            rows = [r for r in per_unit_rows[base] if r["id"] in ids]
            cap_outcomes[u] = {"status": "measured", **flags, "n": len(ids),
                               "correct": sum(bool(r.get("correct")) for r in rows),
                               "rate": round(sum(bool(r.get("correct")) for r in rows) / len(ids), 4)
                               if ids else None}
    entry = ER.entry(ES.registry(), eval_set_id)
    return {"schema": "aadistill.eval_set_report/v2", "label": label,
            "eval_set_id": eval_set_id,
            "eval_set_identity": ER.current_identity(entry),
            "eval_set_status": entry["status"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "formal_core": m.get("formal_core"),
            "units": outcomes,
            "capabilities": ER.capability_table(m["capabilities"], cap_outcomes),
            "pooled": "none by contract: units and capabilities are reported separately"}


def default_formal_set() -> str:
    return ES.default_formal()


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

#!/usr/bin/env python3
"""The Stage-0 Top-4 extension's analysis, as registered. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_extension.py \
        --top1 d1=<dir> --top1 d2=<dir> --top1 d4=<dir> \
        --probes <store> [--probes <store> ...] --out <json>

`--top1` is a dose's secured records as analyze_formal reads them (`search/`,
`recovery/`, `eval/`); `--probes` is one probe session's secured store, one
dir per unit (`report.json`, `rows/`, `generations/`, the recovery's
`train_log.jsonl` and `recovery_products.json`). Implements
`plans/top4_extension.json :: analysis` and nothing else:

* per unit -- every eval-set-1 unit's formal numbers, CORRECT and USABLE side
  by side (maintainer 2026-10-21), usable_rollout with every component, and the
  recovery's training record;
* within dose -- rank k vs rank 1 at seed 20260727, seed 20260726 vs 20260727
  for ranks 2..4 (the reference spread), and the A100 rank-1 unit beside the
  L40S one (reported, never attributed), each a task-paired difference with the
  preregistration's bootstrap and, for fluid correctness, its SESOI label; the
  search's quality order beside the recovered outcome, described;
* across doses -- the 1x/2x/4x shape of the above, described.

Usable is the five-component `aadistill.evaluation.usable_rollout` on the stored
replies. eval-set-1 registers it for the fluid battery (via score_battery) and
legacy_150; for the stage3_capability units it is the same check applied to the
stored generations, with tools_offered read from the item. No SESOI is
registered for usable, so its contrasts carry intervals and no label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
import sys  # noqa: E402
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation import usable_rollout as UR  # noqa: E402
from stages.calibration_volume import score_battery as SB  # noqa: E402
from stages.calibration_volume.analyze_formal import PREREG, STATIC_AXES, cluster_bootstrap, label  # noqa: E402

PLAN = REPO / "logs/stages/stage-0/calibration_volume/plans/top4_extension.json"
SESSION_PLAN = REPO / "configs/stages/stage-0/calibration_volume/probes_{dose}_session.json"
EVAL_SET = "eval-set-1"
FLUID = "fluid_reasoning_720__dev_v1"
CAP = "stage3_capability_846"
BEHAVIOR = f"{CAP}__eval_behavior_v0"
#: units contrasted item by item (the paired sets go by pair); eval_behavior is
#: reported per unit only, as in the formal analysis (5-12 items a group)
ITEM_UNITS = (f"{CAP}__math_verified", f"{CAP}__gsm8k", f"{CAP}__multihop", f"{CAP}__rag",
              f"{CAP}__knowledge", "stage3_legacy_150__items")
PAIRED_UNITS = (f"{CAP}__answerability_paired", f"{CAP}__safety_paired")
ROLE = {"stage3_legacy_150__items": "in-sample regression, not generalization",
        f"{CAP}__knowledge": "supplementary (outside the formal core)"}
A100_SEED, SECOND_SEED = 20260726, 20260727


def jl(path: Path) -> list[dict]:
    """JSON lines split on "\\n" only: a stored reply may hold U+2028 or U+0085,
    which str.splitlines() treats as line ends and so cuts a record in two."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rate(x: int, n: int) -> dict:
    return {"n": n, "count": x, "rate": round(x / n, 4) if n else None}


class Unit:
    """One recovered student's eval-set-1 rows, scored for correct AND usable."""

    def __init__(self, key: str, dose: str, rank: int, seed: int, hardware: str,
                 eval_dir: Path, rec_dir: Path, items: dict, tasks: dict):
        self.key, self.dose, self.rank, self.seed, self.hardware = key, dose, rank, seed, hardware
        self.eval_dir, self.rec_dir = eval_dir, rec_dir
        self.report = json.loads((eval_dir / "report.json").read_text())
        self.read = [eval_dir / "report.json"]
        self.rows = {}
        for unit in (FLUID, BEHAVIOR, *ITEM_UNITS, *PAIRED_UNITS):
            self.rows[unit] = self._score(unit, items, tasks)

    def _formal(self, unit: str) -> dict:
        path = self.eval_dir / "rows" / f"{unit}.rows.jsonl"
        self.read.append(path)
        return {r["id"]: r for r in jl(path) if r["formal"]}

    def _generations(self, unit: str) -> list[dict]:
        path = self.eval_dir / "generations" / f"{unit}.jsonl"
        self.read.append(path)
        return jl(path)

    def _score(self, unit: str, items: dict, tasks: dict) -> dict:
        formal = self._formal(unit)
        recs = self._generations(unit)
        if unit == FLUID:
            scored = SB.score(tasks, [r for r in recs if r["task_id"] in formal], "v2")
            out = {}
            for r in scored:
                # the battery scorer must reproduce the report's own verdict
                assert bool(r["correct"]) == bool(formal[r["task_id"]]["correct"]), (self.key, r["task_id"])
                out[r["task_id"]] = r
            assert set(out) == set(formal), (self.key, unit)
            return out
        comp_unit = unit.split("__")
        out = {}
        for rec in recs:
            if rec["id"] not in formal:
                continue
            item = items[unit][rec["id"]]
            comp = UR.components(rec, think_preopened=bool(rec.get("think_preopened", True)),
                                 tools_offered=bool(item.get("tools")))
            out[rec["id"]] = {**formal[rec["id"]], **comp, "usable": all(comp.values()),
                              "pair_id": item.get("pair_id"), "group": item.get("group")}
        assert set(out) == set(formal), (self.key, comp_unit)
        return out

    # -- per unit ---------------------------------------------------------
    def table(self) -> dict:
        U = self.report["units"]
        f = self.rows[FLUID]
        static = [r for r in f.values() if not r["interactive"]]
        episodes = [r for r in f.values() if r["interactive"]]
        fl = U["fluid_reasoning_720/dev_v1"]
        assert fl["correct"] == sum(r["correct"] for r in f.values()) and fl["n_items"] == len(f)
        out = {"fluid_reasoning_720/dev_v1": {
            "formal_n": fl["n_items"], "formal_correct": rate(fl["correct"], fl["n_items"]),
            "static": {"correct": rate(sum(r["correct"] for r in static), len(static)),
                       "usable": rate(sum(r["usable"] for r in static), len(static)),
                       "usable_components": {c: rate(sum(r[c] for r in static), len(static))
                                             for c in UR.COMPONENTS},
                       "answer_correct_ignoring_usable": rate(sum(r["answer_correct"] for r in static),
                                                              len(static))},
            "episodes": {"success": rate(sum(r["correct"] for r in episodes), len(episodes)),
                         "all_turns_usable": rate(sum(r["all_turns_usable"] for r in episodes), len(episodes)),
                         "mean_usable_turn_rate": round(sum(r["usable_turn_rate"] for r in episodes)
                                                        / len(episodes), 4) if episodes else None},
            "axes": {ax: {"correct": rate(sum(r["correct"] for r in static if r["axis"] == ax),
                                          sum(r["axis"] == ax for r in static)),
                          "usable": rate(sum(r["usable"] for r in static if r["axis"] == ax),
                                         sum(r["axis"] == ax for r in static))}
                     for ax in STATIC_AXES}}}
        for unit in ITEM_UNITS:
            rows = list(self.rows[unit].values())
            u = U[unit.replace("__", "/")]
            n_correct = sum(bool(r["correct"]) for r in rows)
            head = u.get("correct", round(u["headline_value"] * u["n_items"]) if "headline_value" in u else None)
            assert head == n_correct and u["n_items"] == len(rows), (self.key, unit, head, n_correct)
            out[unit.replace("__", "/")] = {
                "role": ROLE.get(unit, "held out"), "in_formal_core": u["in_formal_core"],
                "correct": rate(n_correct, len(rows)), "usable": rate(sum(r["usable"] for r in rows), len(rows)),
                "usable_components": {c: rate(sum(r[c] for r in rows), len(rows)) for c in UR.COMPONENTS}}
        for unit in PAIRED_UNITS:
            rows = self.rows[unit]
            pairs = defaultdict(list)
            for r in rows.values():
                pairs[r["pair_id"]].append(r)
            m = U[unit.replace("__", "/")]["metrics"]
            pc = sum(all(bool(x["correct"]) for x in v) for v in pairs.values())
            assert m["pairs"] == len(pairs) and round(m["pair_correct"] * m["pairs"]) == pc, (self.key, unit)
            out[unit.replace("__", "/")] = {
                "role": "held out", "unit_of_analysis": "pair",
                "pairs_correct": rate(pc, len(pairs)),
                "pairs_both_usable": rate(sum(all(x["usable"] for x in v) for v in pairs.values()), len(pairs)),
                "rows_usable": rate(sum(r["usable"] for r in rows.values()), len(rows)),
                "rows_usable_components": {c: rate(sum(r[c] for r in rows.values()), len(rows))
                                           for c in UR.COMPONENTS}}
        beh = self.rows[BEHAVIOR]
        groups = defaultdict(list)
        for r in beh.values():
            groups[r["group"]].append(r)
        tc = groups["tool_calling"]
        out[f"{CAP}/eval_behavior_v0"] = {
            "tool_calling": {"correct": rate(sum(bool(r["tool_call_exact_match"]) for r in tc), len(tc)),
                             "usable": rate(sum(r["usable"] for r in tc), len(tc))},
            **{g: {"correct": "n/a: eval-set-1 registers no correctness scorer for this group",
                   "format_ok": rate(sum(bool(r["format_ok"]) for r in groups[g]), len(groups[g])),
                   "usable": rate(sum(r["usable"] for r in groups[g]), len(groups[g]))}
               for g in ("instruction", "short_realtime")},
            "protocol_integrity_all_groups": {
                "terminated": rate(sum(bool(r["terminated"]) for r in beh.values()), len(beh)),
                "usable": rate(sum(r["usable"] for r in beh.values()), len(beh)),
                "usable_components": {c: rate(sum(r[c] for r in beh.values()), len(beh))
                                      for c in UR.COMPONENTS}}}
        return out

    def recovery(self) -> dict:
        log = jl(self.rec_dir / "train_log.jsonl")
        prod = json.loads((self.rec_dir / "recovery_products.json").read_text())
        self.read += [self.rec_dir / "train_log.jsonl", self.rec_dir / "recovery_products.json"]
        steps = [e for e in log if e.get("event") == "train_step"]
        evals = [e for e in log if e.get("event") == "eval_result"]
        end = next((e for e in log if e.get("event") == "run_end"), {})
        assert prod.get("seed") in (None, self.seed), (self.key, prod.get("seed"))
        return {"state_id": prod.get("state_id"), "seed": self.seed, "model_sha256": prod.get("model_sha256"),
                "config_sha256": prod.get("config_sha256"), "steps": end.get("steps"),
                "seconds": end.get("seconds"),
                "last_train_step": {k: steps[-1].get(k) for k in ("step", "loss", "ce", "kd", "tokens_seen")}
                if steps else None,
                "final_val": {k: evals[-1].get(k) for k in ("step", "val_ce", "val_ppl", "val_kd")}
                if evals else None}


# -- contrasts ------------------------------------------------------------
def _boot(job):
    name, diffs, cl, st, seed = job
    return name, cluster_bootstrap(diffs, cl, st, seed)


def contrast_jobs(tag: str, a: Unit, b: Unit, tasks: dict) -> list:
    """(b - a) on every contrasted unit, correct and usable; returns bootstrap jobs."""
    jobs = []
    fa, fb = a.rows[FLUID], b.rows[FLUID]
    ids = sorted(set(fa) & set(fb))
    static = [i for i in ids if not tasks[i]["interactive"]]
    fam = {i: f"{tasks[i]['generator_id']}/{tasks[i]['family_id']}" for i in ids}
    axis = {i: tasks[i]["axis"] for i in ids}

    def add(name, sel, field_a, field_b, cl, st, seed):
        if sel:
            jobs.append((f"{tag}|{name}", {i: float(bool(field_b(i))) - float(bool(field_a(i))) for i in sel},
                         {i: cl[i] for i in sel}, {i: st[i] for i in sel}, seed))

    for metric, off in (("correct", 0), ("usable", 100)):
        add(f"fluid.static_pooled|{metric}", static, lambda i: fa[i][metric], lambda i: fb[i][metric],
            fam, axis, 1 + off)
        for k, ax in enumerate(STATIC_AXES):
            add(f"fluid.{ax}|{metric}", [i for i in static if axis[i] == ax],
                lambda i: fa[i][metric], lambda i: fb[i][metric], fam, axis, 10 + k + off)
    inter = [i for i in ids if tasks[i]["interactive"]]
    add("fluid.interactive_adaptation|correct", inter, lambda i: fa[i]["correct"], lambda i: fb[i]["correct"],
        fam, axis, 20)
    add("fluid.interactive_adaptation|all_turns_usable", inter, lambda i: fa[i]["all_turns_usable"],
        lambda i: fb[i]["all_turns_usable"], fam, axis, 120)
    for unit in ITEM_UNITS:
        ra, rb = a.rows[unit], b.rows[unit]
        sel = sorted(set(ra) & set(rb))
        one = {i: i for i in sel}
        flat = {i: "all" for i in sel}
        for metric, seed in (("correct", 40), ("usable", 140)):
            add(f"{unit}|{metric}", sel, lambda i, r=ra, m=metric: r[i][m],
                lambda i, r=rb, m=metric: r[i][m], one, flat, seed)
    for unit in PAIRED_UNITS:
        pa, pb = defaultdict(list), defaultdict(list)
        for r in a.rows[unit].values():
            pa[r["pair_id"]].append(r)
        for r in b.rows[unit].values():
            pb[r["pair_id"]].append(r)
        sel = sorted(set(pa) & set(pb))
        one = {p: p for p in sel}
        flat = {p: "all" for p in sel}
        for metric, seed in (("correct", 30), ("usable", 130)):
            add(f"{unit}|pair_{metric}", sel, lambda p, d=pa, m=metric: all(bool(x[m]) for x in d[p]),
                lambda p, d=pb, m=metric: all(bool(x[m]) for x in d[p]), one, flat, seed)
    return jobs


def main(argv=None) -> int:
    from shared import eval_sets as ES
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1", action="append", required=True, help="d1=<dose dir>")
    ap.add_argument("--probes", action="append", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    sesoi = json.loads(PREREG.read_text())["analysis"]["statistics"]["sesoi"]
    tasks = {t["task_id"]: t for t in ES.items(EVAL_SET, "fluid_reasoning_720", "dev_v1")}
    items = {}
    for unit in (BEHAVIOR, *ITEM_UNITS, *PAIRED_UNITS):
        comp, u = unit.split("__")
        items[unit] = {r["id"]: r for r in ES.items(EVAL_SET, comp, u)}

    top1 = {k: Path(v) for k, v in (x.split("=", 1) for x in args.top1)}
    candidates, sessions, units = {}, {}, {}
    for dose, d in top1.items():
        sp = SESSION_PLAN.with_name(SESSION_PLAN.name.format(dose=dose))
        sessions[dose] = {"plan": str(sp.relative_to(REPO)), "sha256": sha256(sp)}
        selection = json.loads((d / "search" / "stage1_selection.json").read_text())
        objectives = {x["state_id"]: x["objectives"] for x in selection["decisions"]}
        cands = {}
        for u in json.loads(sp.read_text())["units"]:
            cands.setdefault(u["rank"], {"rank": u["rank"], "quality_order": u["quality_order"],
                                         "state_id": u["state_id"], "path": u["path"],
                                         "checkpoint_sha256": u["checkpoint_sha256"],
                                         "search_objectives": objectives[u["state_id"]]})
        candidates[dose] = [cands[k] for k in sorted(cands)]
        rec = json.loads((d / "recovery" / "recovery_products.json").read_text())
        assert rec.get("state_id") in (None, cands[1]["state_id"]), (dose, rec.get("state_id"))
        key = f"{dose}/r1/s{A100_SEED}/A100"
        units[key] = Unit(key, dose, 1, A100_SEED, "NVIDIA A100-SXM4-80GB (preregistered unit)",
                          d / "eval", d / "recovery", items, tasks)
    for store in args.probes:
        session = json.loads((store / "probe_session.json").read_text())
        for name, st in session["units"].items():
            assert st["status"] == "ok", (name, st["status"])
            _, _, dose, r, state8, s = name.split("_")
            rank, seed = int(r[1:]), int(s[1:])
            cand = candidates[dose][rank - 1]
            assert cand["state_id"].startswith(state8), name
            key = f"{dose}/r{rank}/s{seed}/L40S"
            units[key] = Unit(key, dose, rank, seed, f"{session['gpu']['name']} ({session['run_id']})",
                              store / name, store / name, items, tasks)
    expected = {f"{d}/r{k}/s{s}/L40S" for d in top1 for k in range(1, 5)
                for s in (plan["units"]["seeds"]["top1"] if k == 1 else plan["units"]["seeds"]["others"])}
    missing = sorted(expected - set(units))

    per_unit = {k: {"dose": u.dose, "rank": u.rank, "seed": u.seed, "hardware": u.hardware,
                    "recovery": u.recovery(), "eval_set_1": u.table(),
                    "report_sha256": sha256(u.eval_dir / "report.json")} for k, u in units.items()}

    jobs, plans = [], {}
    for dose in top1:
        def get(k, s, hw="L40S"):
            return units.get(f"{dose}/r{k}/s{s}/{hw}")
        pairs = [(f"{dose}|rank{k}_vs_rank1_s{SECOND_SEED}", get(1, SECOND_SEED), get(k, SECOND_SEED))
                 for k in (2, 3, 4)]
        pairs += [(f"{dose}|rank{k}_s{SECOND_SEED}_vs_s{A100_SEED}", get(k, A100_SEED), get(k, SECOND_SEED))
                  for k in (2, 3, 4)]
        pairs.append((f"{dose}|rank1_L40S_s{SECOND_SEED}_vs_A100_s{A100_SEED}",
                      get(1, A100_SEED, "A100"), get(1, SECOND_SEED)))
        for tag, a, b in pairs:
            if a is None or b is None:
                continue
            plans[tag] = {"a": a.key, "b": b.key}
            jobs += contrast_jobs(tag.replace("|", "#"), a, b, tasks)
    with ProcessPoolExecutor(args.workers) as ex:
        results = dict(ex.map(_boot, jobs, chunksize=4))
    diffs_of = {j[0]: j[1] for j in jobs}
    contrasts = {}
    for name, (p, lo, hi) in results.items():
        tagh, unit, metric = name.split("|")
        tag = tagh.replace("#", "|")
        diffs = diffs_of[name]
        a, b = units[plans[tag]["a"]], units[plans[tag]["b"]]
        ra, rb = side_rates(unit, metric, a, b, list(diffs), tasks)
        cell = {"n": len(diffs), "rate_a": ra, "rate_b": rb, "diff": round(p, 4),
                "ci95": [round(lo, 4), round(hi, 4)]}
        if ra == 0 and rb == 0:
            cell["floor"] = True
        if unit.startswith("fluid.") and metric == "correct":
            s = sesoi["pooled_primary"] if unit == "fluid.static_pooled" else sesoi["per_axis"]
            cell.update({"sesoi": s, "label": label(lo, hi, s)})
        elif unit.startswith("fluid."):
            cell["label"] = "none: no SESOI is registered for usable"
        else:
            cell["role"] = ROLE.get(unit, "held out")
        dose, kind = tag.split("|")
        contrasts.setdefault(dose, {}).setdefault(kind, {"a": plans[tag]["a"], "b": plans[tag]["b"],
                                                         "diff_is": "b - a"})
        contrasts[dose][kind].setdefault(unit, {})[metric] = cell

    def head(k):
        t = per_unit[k]["eval_set_1"]["fluid_reasoning_720/dev_v1"]
        return {"fluid_static_correct": t["static"]["correct"]["rate"],
                "fluid_static_usable": t["static"]["usable"]["rate"],
                "episodes_success": t["episodes"]["success"]["rate"],
                "episodes_all_turns_usable": t["episodes"]["all_turns_usable"]["rate"]}

    order = {dose: [{**c, "recovered": {k: head(k) for k in per_unit
                                         if k.startswith(f"{dose}/r{c['rank']}/")}}
                    for c in candidates[dose]] for dose in top1}
    metrics = ("fluid_static_correct", "fluid_static_usable", "episodes_success", "episodes_all_turns_usable")
    shape = {dose: {m: {k: head(k)[m] for k in per_unit if k.startswith(f"{dose}/")} for m in metrics}
             for dose in top1}
    for dose in shape:
        for m, vals in shape[dose].items():
            v = sorted(vals.values())
            vals["_range"] = [v[0], v[-1]]

    def spread(unit, metric, marker):
        return {dose: {kind: c.get(unit, {}).get(metric, {}).get("diff")
                       for kind, c in kinds.items() if marker in kind}
                for dose, kinds in contrasts.items()}

    reference = {f"{unit}|{metric}": {"seed_to_seed": spread(unit, metric, f"_vs_s{A100_SEED}"),
                                      "rank_k_vs_rank_1": spread(unit, metric, "_vs_rank1"),
                                      "hardware_and_seed": spread(unit, metric, "_vs_A100")}
                 for unit, metric in (("fluid.static_pooled", "correct"), ("fluid.static_pooled", "usable"),
                                      ("fluid.interactive_adaptation", "all_turns_usable"))}

    read = sorted({p for u in units.values() for p in u.read})
    out = {"schema": "aadistill.stage0.calibration_volume.extension_analysis/v1",
           "plan": str(PLAN.relative_to(REPO)), "plan_sha256": sha256(PLAN),
           "preregistration": str(PREREG.relative_to(REPO)),
           "eval_set": EVAL_SET,
           "usable_definition": __doc__.split("Usable is", 1)[1].strip(),
           "session_plans": sessions,
           "units_expected": len(expected) + len(top1), "units_present": len(units),
           "units_missing": missing,
           "inputs": {str(p): sha256(p) for p in read},
           "per_unit": per_unit,
           "within_dose": contrasts,
           "quality_order_beside_outcome": {"read": "described (n = 4 candidates), never tested",
                                            **order},
           "across_doses": {"read": "the 1x/2x/4x shape, described: each dose's 8 units (7 L40S + the "
                                    "preregistered A100 rank 1)", **shape},
           "reference_spread": {"read": "each contrast's paired difference, side by side: seed-to-seed "
                                        "(same candidate, same L40S) is the spread every candidate "
                                        "difference is read against", **reference},
           "hardware_note": plan["units"]["hardware_note"],
           "claims": plan["analysis"]["not_claimed"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"units {len(units)} (missing {missing}); contrasts {len(results)} -> {args.out}")
    return 0


def side_rates(unit: str, metric: str, a: Unit, b: Unit, ids: list, tasks: dict) -> tuple[float, float]:
    """Each side's rate over the contrasted ids (items, or pairs for paired sets)."""
    def val(u: Unit, i):
        if unit.startswith("fluid."):
            return bool(u.rows[FLUID][i][metric])
        key = next(k for k in (*ITEM_UNITS, *PAIRED_UNITS) if k == unit)
        if key in PAIRED_UNITS:
            m = metric.removeprefix("pair_")
            return all(bool(r[m]) for r in u.rows[key].values() if r["pair_id"] == i)
        return bool(u.rows[key][i][metric])
    n = len(ids)
    return (round(sum(val(a, i) for i in ids) / n, 4), round(sum(val(b, i) for i in ids) / n, 4))


if __name__ == "__main__":
    raise SystemExit(main())

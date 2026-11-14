#!/usr/bin/env python3
"""The four groups on eval-set-1 and the state suite: per-unit results, structural contrasts, retention, failure types
($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/analysis.py --store <run scr>/store [--store <later run>/store ...] \\
        [--out logs/stages/stage-1/reasoning_inheritance/analyses/structural_ablation.json]

Inputs: T's reused evaluation (`teacher_reuse.py` unpacks it under ``artifacts/.../teacher_eval``) and D / W / DW
from each attempt's secured evidence (``<store>/out/eval/<group>_<mode>/generations``; a later ``--store`` wins for a
unit both hold). Every group is rescored here with the current scorer: the fluid unit by
``score_battery.py --protocol v2`` (which carries the usable-rollout components and the interactive replay), every
other unit by ``eval_set_scoring.report``; a unit without generations is ``not_run``.

Per group and mode (H-T and H-N never pooled): formal correctness and usable rollout of every unit, the fluid axes,
static (570) and interactive (144) apart, termination types, generation lengths. Contrasts on each axis and unit with
all four groups present (`design.json :: analysis.contrasts`): depth alone T-D, non-depth alone T-W, depth given
non-depth W-DW, non-depth given depth D-DW, interaction D+W-T-DW, each with a paired 95% interval from a cluster
bootstrap that resamples whole rule families (fluid ``family_id``; a paired unit's pair; otherwise the item), so items
sharing a rule or template are never counted as independent. Retention: of the items T answered correctly in the same
mode, the share each group keeps. Outcomes and failure types: what every item that did not score ended as -- a
generation stopped before an answer, a protocol error, a well-formed reply with no answer the registered scorer can
read, or an answer read and wrong -- for the static fluid items and for every capability unit (`OUTCOMES`); none of it
changes a score. Stop records: every degeneration-stopped record checked as sealed (`stop_record_integrity`). The
fluid intervals are also computed under the registered independence grouping (`contrasts_registered_grouping`).

eval-set-1 is development-exposed (P10.3): nothing here is an untouched test, and no per-item result feeds back into a
mapping, a kept block or a calibration share.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/design.json"
TEACHER_EVAL = REPO / "artifacts/stages/stage-1/reasoning_inheritance/teacher_eval"
OUT = REPO / "logs/stages/stage-1/reasoning_inheritance/analyses/structural_ablation.json"
GROUPS, MODES = ("T", "D", "W", "DW"), ("H-T", "H-N")
FLUID = "fluid_reasoning_720__dev_v1"
CAP = "stage3_capability_846"
STATIC_AXES = ("task_contained_deduction", "inductive_rule_discovery", "compositional_generalization",
               "abstract_analogical")
CAP_UNITS = ("math_verified", "gsm8k", "multihop", "rag", "answerability_paired", "safety_paired", "knowledge")


def dir_digest(d: Path) -> str:
    """sha256 over the directory's files in name order (name, then each file's own sha256): what was scored."""
    h = hashlib.sha256()
    for f in sorted(p for p in d.iterdir() if p.is_file()):
        h.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def gens_dir(g: str, mode: str, stores: list[Path]) -> Path | None:
    if g == "T":
        d = TEACHER_EVAL / mode / "unpacked" / "evaluation" / "generations"
        return d if d.is_dir() else None
    for st in reversed(stores):
        d = st / "out" / "eval" / f"{g}_{mode}" / "generations"
        if d.is_dir():
            return d
    return None


def fluid_rows(gens: Path, label: str, tmp: Path) -> dict | None:
    """score_battery v2 rows of the fluid unit, restricted to the formal core (the scorer's own mask)."""
    from shared import eval_sets as ES
    f = gens / f"{FLUID}.jsonl"
    if not f.is_file():
        return None
    out = tmp / f"battery_{label}"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "scripts/stages/stage-0/calibration_volume/score_battery.py",
                    "--tasks", str(ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1")), "--records", str(f),
                    "--out-dir", str(out), "--protocol", "v2"], cwd=REPO, env={"PYTHONPATH": "src:scripts"},
                   capture_output=True, check=True)
    return {r["task_id"]: r for r in jl(next(out.glob("*.rows.jsonl")))}


def unit_rows(gens: Path, label: str, tmp: Path) -> tuple[dict, dict]:
    """The formal report and every unit's rows (eval_set_scoring.report), keyed by item id."""
    import shared.evaluation.eval_set_scoring as S
    rows_dir = tmp / f"rows_{label}"
    rep = S.report(gens, label, {}, "eval-set-1", rows_out=rows_dir)
    rows = {p.name.removesuffix(".rows.jsonl"): {r["id"]: r for r in jl(p)} for p in rows_dir.glob("*.rows.jsonl")}
    return rep, rows


def usable_of(gens: Path, unit: str, items: dict) -> dict:
    from aadistill.evaluation import usable_rollout as UR
    f = gens / f"{unit}.jsonl"
    if not f.is_file():
        return {}
    out = {}
    for rec in jl(f):
        c = UR.components(rec, think_preopened=bool(rec.get("think_preopened", True)),
                          tools_offered=bool((items.get(rec["id"]) or {}).get("tools")), think_state=rec.get("think_state"))
        out[rec["id"]] = all(c.values())
    return out


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def static_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    lens = [int(r.get("generated_tokens") or 0) for r in rows]
    stops = collections.Counter(r.get("stop_reason") for r in rows)
    return {"n": n, "correct": sum(bool(r["correct"]) for r in rows),
            "accuracy": round(sum(bool(r["correct"]) for r in rows) / n, 4),
            "usable": sum(bool(r.get("usable")) for r in rows),
            "usable_rate": round(sum(bool(r.get("usable")) for r in rows) / n, 4),
            #: descriptive: each group finishes a different subset, so this is not a matched comparison
            "correct_given_usable": sum(bool(r["correct"]) and bool(r.get("usable")) for r in rows),
            "components": {k: round(sum(bool(r.get(k)) for r in rows) / n, 4)
                           for k in ("non_empty", "natural_termination", "no_severe_repetition", "no_context_limit",
                                     "protocol_valid")},
            "stop_reasons": dict(stops),
            "degeneration_rate": round(sum(1 for r in rows if not r.get("no_severe_repetition")) / n, 4),
            "context_limit_rate": round(sum(1 for r in rows if not r.get("no_context_limit")) / n, 4),
            "generated_tokens": {"mean": round(sum(lens) / n, 1), "p10": pct(lens, 0.1), "p50": pct(lens, 0.5),
                                 "p90": pct(lens, 0.9), "max": max(lens)},
            "reasons": dict(collections.Counter(r.get("reason") for r in rows)),
            "outcomes": {k: v for k, v in ((o, sum(outcome(r) == o for r in rows)) for o in OUTCOMES) if v}}


def episode_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {"n": n, "success": sum(bool(r["correct"]) for r in rows),
            "success_rate": round(sum(bool(r["correct"]) for r in rows) / n, 4),
            "all_turns_usable": sum(bool(r.get("all_turns_usable")) for r in rows),
            "ends": dict(collections.Counter(r.get("end") for r in rows)),
            "turns_mean": round(sum(int(r.get("turns") or 0) for r in rows) / n, 2),
            "generated_tokens_mean": round(sum(int(r.get("generated_tokens") or 0) for r in rows) / n, 1)}


#: what an item that did not score ended as, in the scorer's terms: the generation stopped before an answer
#: (`degeneration`, `context_limit`, `not_terminated`), it ended but broke the protocol (`protocol_invalid`: empty,
#: or think delimiters invalid), it ended well-formed but the registered scorer found no final answer to read
#: (`no_extractable_answer`: no \boxed{} / no final-answer line), or an answer was read and is wrong
#: (`answered_wrong`). Only the last is a wrong answer; none of them changes a score.
OUTCOMES = ("correct", "answered_wrong", "no_extractable_answer", "protocol_invalid", "degeneration", "context_limit",
            "not_terminated")


def failure_type(r: dict) -> str:
    """What a wrong static fluid item ended as (first matching, in this order)."""
    if not r.get("no_context_limit"):
        return "context_limit"
    if not r.get("no_severe_repetition"):
        return "degeneration"
    if not r.get("natural_termination"):
        return "not_terminated"
    if not r.get("non_empty") or not r.get("protocol_valid"):
        return "protocol_invalid"
    if r.get("reason") == "no_boxed":
        return "no_extractable_answer"
    return "answered_wrong"


def outcome(r: dict) -> str:
    return "correct" if r["correct"] else failure_type(r)


def cap_outcome(reason: str, stop_reason: str | None) -> str:
    """A capability row's `eval_set_scoring` reason in the same terms; `protocol:not_terminated` is split by the
    generation record's own stop reason. Wrong behaviour on the paired and RAG units (echoed, answered an
    unanswerable question, refused an answerable or benign one, complied with an unsafe one) is an answer, read and
    scored wrong."""
    if reason == "ok":
        return "correct"
    if reason == "protocol:not_terminated":
        return stop_reason if stop_reason in ("degeneration", "context_limit") else "not_terminated"
    if str(reason).startswith("protocol:"):
        return "protocol_invalid"
    if reason in ("no_final_answer", "no_boxed"):
        return "no_extractable_answer"
    return "answered_wrong"


def stop_record_integrity(gens: Path, check_every: int) -> dict:
    """Every degeneration-stopped generation (each turn of an episode) as sealed: its saved output token ids number
    exactly its recorded length, that length is a multiple of the check interval (the stop happened at a check) and
    it is inside the request's allowance. A continuation appended after the stop would break the first two. This shows
    the sealed records were not overwritten; it does not show a rerun after the abort repair would reproduce them."""
    c = collections.Counter()
    for f in sorted(gens.glob("*.jsonl")):
        for rec in jl(f):
            for t in (rec["turns"] if "turns" in rec else [rec]):
                if t.get("stop_reason") != "degeneration":
                    continue
                ids = t.get("output_token_ids")
                ids = json.loads(ids) if isinstance(ids, str) else ids
                n = int(t.get("generated_tokens") or 0)
                c["degeneration_stops"] += 1
                c["ids_equal_recorded_length"] += int(ids is not None and len(ids) == n)
                c["at_a_check"] += int(n % check_every == 0)
                c["inside_allowance"] += int(n < int(t.get("generation_allowance") or n + 1))
    return dict(c)


TEACHER_LENGTH_BINS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, None))


def kept_by_teacher_length(t_rows: list[dict], g_rows: list[dict]) -> list[dict]:
    """Descriptive, added after the results were seen (never a registered analysis): of the static items T answered
    correctly, the share the group keeps, by the length of T's own generation on the item. T's length goes with the
    item's difficulty, so this does not separate "needs a long derivation" from "is hard"."""
    g_ok = {r["task_id"]: bool(r["correct"]) for r in g_rows}
    t_ok = [r for r in t_rows if r["correct"] and r["task_id"] in g_ok]
    out = []
    for lo, hi in TEACHER_LENGTH_BINS:
        sel = [r for r in t_ok if int(r.get("generated_tokens") or 0) >= lo and (hi is None or int(r["generated_tokens"]) < hi)]
        out.append({"teacher_tokens": [lo, hi], "teacher_correct": len(sel), "kept": sum(g_ok[r["task_id"]] for r in sel)})
    return out


def bootstrap(clusters: dict, values: dict, stat, *, n: int, seed: int) -> dict:
    """Percentile interval of ``stat(sample)`` over cluster resamples; ``values[item] -> per-group 0/1``."""
    keys = sorted(clusters)
    rng = random.Random(seed)
    point = stat([i for k in keys for i in clusters[k]])
    draws = []
    for _ in range(n):
        items = [i for k in (rng.choice(keys) for _ in keys) for i in clusters[k]]
        draws.append(stat(items))
    draws.sort()
    return {"estimate": round(point, 4), "ci95": [round(draws[int(0.025 * n)], 4), round(draws[int(0.975 * n) - 1], 4)],
            "clusters": len(keys), "resamples": n}


def contrasts(acc: dict, clusters: dict, *, n: int, seed: int) -> dict:
    """``acc[g][item]`` in {0,1} for the four groups over the same items."""
    items = sorted(set.intersection(*(set(acc[g]) for g in GROUPS)))
    cl = {k: [i for i in v if i in set(items)] for k, v in clusters.items()}
    cl = {k: v for k, v in cl.items() if v}

    def mean(g, its):
        return sum(acc[g][i] for i in its) / len(its)
    defs = {"depth_alone (T-D)": lambda s: mean("T", s) - mean("D", s),
            "non_depth_alone (T-W)": lambda s: mean("T", s) - mean("W", s),
            "depth_given_non_depth (W-DW)": lambda s: mean("W", s) - mean("DW", s),
            "non_depth_given_depth (D-DW)": lambda s: mean("D", s) - mean("DW", s),
            "interaction (D+W-T-DW)": lambda s: mean("D", s) + mean("W", s) - mean("T", s) - mean("DW", s)}
    out = {"n_items": len(items), "accuracy": {g: round(mean(g, items), 4) for g in GROUPS}}
    for j, (k, f) in enumerate(defs.items()):
        out[k] = bootstrap(cl, acc, f, n=n, seed=seed + j)
    return out


def retention(acc: dict, clusters: dict, *, n: int, seed: int) -> dict:
    """Of the items T answered correctly, the share each group keeps (and the overall rate beside it)."""
    t_ok = [i for i, v in acc["T"].items() if v]
    out = {"n_teacher_correct": len(t_ok)}
    for j, g in enumerate(g for g in GROUPS if g != "T"):
        if not acc.get(g):
            continue
        its = [i for i in t_ok if i in acc[g]]
        cl = {k: [i for i in v if i in set(its)] for k, v in clusters.items()}
        cl = {k: v for k, v in cl.items() if v}
        out[g] = bootstrap(cl, acc, lambda s, g=g: sum(acc[g][i] for i in s) / len(s), n=n, seed=seed + 100 + j)
        out[g]["kept"] = sum(acc[g][i] for i in its)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", action="append", default=[])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    from shared import eval_sets as ES
    d = json.loads(DESIGN.read_text())
    bs = d["analysis"]["bootstrap"]
    stores = [Path(s) for s in a.store]
    fluid_items = {t["task_id"]: t for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1")}
    cap_items = {u: {r["id"]: r for r in ES.items("eval-set-1", CAP, u)} for u in (*CAP_UNITS, "eval_behavior_v0")}
    res: dict = {"schema": "aadistill.stage1.reasoning_inheritance.structural_ablation/v1",
                 "eval_set": "eval-set-1 (development-exposed; not an untouched test)", "groups": {}, "contrasts": {},
                 "retention": {}, "failure_types": {}, "inputs": {}}
    acc: dict = {}
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        for mode in MODES:
            for g in GROUPS:
                gd = gens_dir(g, mode, stores)
                key = f"{g}_{mode}"
                if gd is None:
                    res["groups"][key] = {"status": "not_run"}
                    continue
                res["inputs"][key] = {"generations": str(gd.resolve().relative_to(REPO)), "sha256": dir_digest(gd)}
                fr = fluid_rows(gd, key, tmp)
                rep, rows = unit_rows(gd, key, tmp)
                entry: dict = {"status": "measured", "units": {},
                               "stop_records": stop_record_integrity(gd, int(d["evaluation"]["degeneration_check_every"]))}
                if fr is not None:
                    formal = rows.get(FLUID, {})
                    core = [fr[i] for i in fr if (formal.get(i) or {}).get("formal")]
                    static = [r for r in core if not r["interactive"]]
                    episodes = [r for r in core if r["interactive"]]
                    entry["fluid"] = {
                        "formal_core": {"n": len(core), "correct": sum(bool(r["correct"]) for r in core)},
                        "static": static_summary(static), "interactive": episode_summary(episodes),
                        "axes": {ax: static_summary([r for r in static if r["axis"] == ax]) for ax in STATIC_AXES},
                        "all_registered_items": {"n": len(fr), "correct": sum(bool(r["correct"]) for r in fr.values())}}
                    for ax in (*STATIC_AXES, "static_all", "interactive_adaptation"):
                        sel = (core if ax == "fluid_formal_core" else
                               static if ax == "static_all" else
                               episodes if ax == "interactive_adaptation" else [r for r in static if r["axis"] == ax])
                        acc.setdefault((mode, f"fluid/{ax}"), {})[g] = {r["task_id"]: int(bool(r["correct"])) for r in sel}
                    res["failure_types"].setdefault(mode, {})[g] = static
                for u in CAP_UNITS:
                    unit = f"{CAP}__{u}"
                    rr = {i: r for i, r in rows.get(unit, {}).items() if r.get("formal")}
                    if not rr:
                        entry["units"][u] = {"status": "not_run"}
                        continue
                    us = usable_of(gd, unit, cap_items[u])
                    info = rep["units"][f"{CAP}/{u}"]
                    stops = {r["id"]: r.get("stop_reason") for r in jl(gd / f"{unit}.jsonl")}
                    oc = collections.Counter(cap_outcome(r.get("reason"), stops.get(i)) for i, r in rr.items())
                    e = {"in_formal_core": info.get("in_formal_core"), "n": len(rr),
                         "correct_rows": sum(bool(r["correct"]) for r in rr.values()),
                         "headline": info.get("headline"), "headline_value": info.get("headline_value"),
                         "usable": sum(bool(us.get(i)) for i in rr), "metrics": info.get("metrics"),
                         "outcomes_by_row": {o: oc[o] for o in OUTCOMES if oc[o]}}
                    if u.endswith("_paired"):
                        pairs = collections.defaultdict(list)
                        for i in rr:
                            pairs[cap_items[u][i]["pair_id"]].append((bool(rr[i]["correct"]), bool(us.get(i))))
                        e["pairs"] = len(pairs)
                        e["pairs_correct"] = sum(all(c for c, _ in v) for v in pairs.values())
                        e["pairs_both_usable"] = sum(all(x for _, x in v) for v in pairs.values())
                        acc.setdefault((mode, f"{CAP}/{u}"), {})[g] = {p: int(all(c for c, _ in v)) for p, v in pairs.items()}
                    else:
                        acc.setdefault((mode, f"{CAP}/{u}"), {})[g] = {i: int(bool(r["correct"])) for i, r in rr.items()}
                    entry["units"][u] = e
                beh = rows.get(f"{CAP}__eval_behavior_v0", {})
                if beh:
                    us = usable_of(gd, f"{CAP}__eval_behavior_v0", cap_items["eval_behavior_v0"])
                    fb = {i: r for i, r in beh.items() if r.get("formal")}
                    by = collections.defaultdict(list)
                    for i, r in fb.items():
                        by[r.get("group")].append((r, bool(us.get(i))))
                    entry["units"]["eval_behavior_v0"] = {
                        g2: {"n": len(v), "usable": sum(x for _, x in v),
                             "format_ok": sum(bool(r.get("format_ok")) for r, _ in v),
                             "terminated": sum(bool(r.get("terminated")) for r, _ in v)} for g2, v in by.items()}
                    bm = rep["units"][f"{CAP}/eval_behavior_v0"].get("metrics") or {}
                    entry["units"]["eval_behavior_v0"]["metrics"] = bm
                    tc = ((bm.get("by_group") or {}).get("tool_calling") or {}).get("tool_call_exact_match")
                    if tc is not None and "tool_calling" in entry["units"]["eval_behavior_v0"]:
                        n_tc = entry["units"]["eval_behavior_v0"]["tool_calling"]["n"]
                        entry["units"]["eval_behavior_v0"]["tool_calling"]["tool_call_exact"] = round(tc * n_tc)
                lg = rows.get("stage3_legacy_150__items", {})
                if lg:
                    entry["units"]["legacy_150"] = {"n": len(lg), "correct": sum(bool(r["correct"]) for r in lg.values()),
                                                    "role": "in_sample_regression (never generalization)"}
                res["groups"][key] = entry

    #: cluster structure: fluid rule families; paired units are already pair-level; other units item-level.
    #: The design registered the eval set's independence groups for the fluid unit (`independence_v1.json`: six
    #: groups of renamed knights-and-knaves items, every other item its own cluster); the reported intervals resample
    #: whole rule families instead, a coarser clustering chosen when the analysis was written.
    #: Both are computed; the registered one is kept beside the reported one (`contrasts_registered_grouping`).
    fam = collections.defaultdict(list)
    for i, t in fluid_items.items():
        fam[t["family_id"]].append(i)
    fluid_path = ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1")
    indep_path = fluid_path.with_name("independence_v1.json")
    want = ES.manifest("eval-set-1")["files"]["fluid_reasoning_720/independence_v1.json"]["sha256"]
    if hashlib.sha256(indep_path.read_bytes()).hexdigest() != want:
        raise SystemExit(f"{indep_path} is not the registered independence file")
    indep_doc = json.loads(indep_path.read_text())
    if indep_doc["items_sha256"] != hashlib.sha256(fluid_path.read_bytes()).hexdigest():
        raise SystemExit("independence_v1.json does not describe this fluid unit")
    in_group = {i: n for n, grp in enumerate(indep_doc["groups"]) for i in grp["ids"]}
    indep = collections.defaultdict(list)
    for i in fluid_items:
        indep[f"g{in_group[i]}" if i in in_group else i].append(i)
    res["bootstrap_grouping"] = {
        "reported": "fluid: whole rule families (family_id, %d clusters); paired units: the pair; other units: the item"
                    % len(fam),
        "registered": "design.json :: analysis.bootstrap.unit -- the eval set's independence groups "
                      "(fluid_reasoning_720/independence_v1.json, %d clusters); paired units: the pair" % len(indep),
        "why_they_differ": "the analysis resampled rule families, which also treats items of one family as dependent; "
                           "the registered grouping treats only verified equivalent items as one. Both intervals "
                           "are recorded; the registration is unchanged"}
    for (mode, unit), by_g in sorted(acc.items()):
        if not all(by_g.get(g) for g in GROUPS):
            res["contrasts"].setdefault(mode, {})[unit] = {"status": "incomplete",
                                                           "groups_present": sorted(by_g)}
            continue
        cl = fam if unit.startswith("fluid/") else {i: [i] for i in by_g["T"]}
        res["contrasts"].setdefault(mode, {})[unit] = contrasts(by_g, cl, n=int(bs["resamples"]), seed=int(bs["seed"]))
        res["retention"].setdefault(mode, {})[unit] = retention(by_g, cl, n=int(bs["resamples"]), seed=int(bs["seed"]))
        if unit.startswith("fluid/"):
            reg = res.setdefault("contrasts_registered_grouping", {}).setdefault(mode, {})
            reg[unit] = {"contrasts": contrasts(by_g, indep, n=int(bs["resamples"]), seed=int(bs["seed"])),
                         "retention": retention(by_g, indep, n=int(bs["resamples"]), seed=int(bs["seed"]))}
    #: failure types of the static items T answered correctly and the group did not
    ft = {}
    for mode, by_g in res["failure_types"].items():
        t_ok = {r["task_id"] for r in by_g.get("T", []) if r["correct"]}
        ft[mode] = {}
        for g, rows in by_g.items():
            if g == "T":
                continue
            lost = [r for r in rows if r["task_id"] in t_ok and not r["correct"]]
            allw = [r for r in rows if not r["correct"]]
            ft[mode][g] = {"teacher_correct_lost": len(lost),
                           "lost_by_type": dict(collections.Counter(failure_type(r) for r in lost)),
                           "all_wrong_by_type": dict(collections.Counter(failure_type(r) for r in allw))}
            ft[mode][g]["kept_by_teacher_length"] = kept_by_teacher_length(by_g["T"], rows)
            deg = [int(r.get("generated_tokens") or 0) for r in rows if r.get("stop_reason") == "degeneration"]
            ft[mode][g]["degeneration_stop_tokens"] = {"n": len(deg), "p25": pct(deg, 0.25), "p50": pct(deg, 0.5),
                                                       "p75": pct(deg, 0.75)}
    res["failure_types"] = ft
    #: state evaluation
    se = {}
    for st in stores:
        p = st / "out" / "records" / "state_eval.json"
        if p.is_file():
            se = json.loads(p.read_text())
    res["state_eval"] = {g: {"values": v.get("values"), "nll": (v.get("detail") or {}).get("per_domain_nll"),
                             "per_domain_kl": (v.get("detail") or {}).get("per_domain_kl")}
                         for g, v in (se.get("groups") or {}).items()}
    res["state_eval_protocol"] = {k: se.get(k) for k in ("measurement_protocol_id", "suite", "suite_content_sha256")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: (v.get("fluid", {}).get("static", {}).get("accuracy") if isinstance(v, dict) else v)
                      for k, v in res["groups"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

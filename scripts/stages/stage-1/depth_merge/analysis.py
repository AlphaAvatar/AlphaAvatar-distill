#!/usr/bin/env python3
"""DMR1's registered analyses, from the sealed post-recovery sets ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/analysis.py \\
        comparison --id C1 --a-evidence <depth_merge's secured out> [--write]
    ... recurrence --kept-evidence <depth_loop_kept's secured out> --merged-evidence <depth_loop_merged's> [--write]

* **C1** (variant 1's design, ``configs/stages/stage-1/depth_merge/design.json``; registered 2026-11-26): depth_merge
  against SOB1's depth_only, its own family -- the thinking-mode test at alpha 0.05 on the static fluid formal core
  (570 items in 558 independence groups), both seeds pooled; the non-thinking test only if the thinking one rejects;
  formal only when both arms' records of both seeds cover all 570 items in the mode (TPI1's / SOB1's statistic, test
  and interval, `single_operator_budget.analysis.pair_comparison`). Variant 1 was stopped with one seed (2026-11-28),
  so C1 stays descriptive.
* **recurrence** (variant 2 as revised 2026-11-28, ``configs/stages/stage-1/depth_recurrence/design.json :: analysis``):
  one seed (20260726), four conditions -- V2-D (depth_loop_kept), V2-M (depth_loop_merged), D (SOB1's depth_only),
  M (variant 1's depth_merge) -- and four estimates per mode, each with its percentile 95% group-bootstrap interval
  and no test: R1 = V2-D - D, R2 = V2-M - M, R3 = V2-M - V2-D, R4 = I = (V2-M - V2-D) - (M - D). R4's item contrast
  is formed per item from the four records of that item, so a group resample keeps all four together. Read in a mode
  only when every condition it names covers all 570 items; the four conditions' accuracies and terminations beside.

A set is read where it is: SOB1's depth arm from its COMPLETE session's secured evidence, variant 1's seed-20260726
sets from its training session's (`common.merged_control_evidence`), the loop arms' from theirs. A unit's sets count
only when its final and both sets are complete on their files (`completion.session`, the arm's design bound).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

SCHEMA = "aadistill.dmr1.comparison/v1"
RECURRENCE_SCHEMA = "aadistill.dmr1.recurrence_comparisons/v1"
ALPHA = 0.05
RESULTS = DC.LOGS / "results"
RECURRENCE_RESULTS = DC.RECURRENCE_LOGS / "results"
MODES = ("H-T", "H-N")
#: variant 1's registered comparison (its design's 2026-11-26 registry, kept in the recurrence design's history)
C1 = {"pair": ["depth_merge", "depth_only"],
      "question": "alpha-merged initialization vs direct deletion, same structure, same recovery"}
#: the revised variant 2's conditions: arm -> its role in R1-R4
CONDITIONS = {"V2-D": "depth_loop_kept", "V2-M": "depth_loop_merged", "D": DC.SOB1_DEPTH, "M": "depth_merge"}


def comparisons() -> dict:
    """The registered comparisons by id: C1 (variant 1's) and R1-R4 (the revised variant 2's design)."""
    return {"C1": C1, **DC.recurrence_design()["analysis"]["comparisons"]}


def protocol_identity(d: dict | None = None) -> dict:
    """What a comparison record was scored under, by identity: the set's manifest, the decoding, the action protocol,
    the modes' template arguments, the registered primary items and the independence file. A consumer (the
    initialization selection) checks a record's against this tree's before reading it."""
    d = d or DC.design()
    ev, an = d["evaluation"]["protocol"], d["analysis"]
    return {"eval_set": ev["eval_set"], "manifest_sha256": ev["manifest_sha256"],
            "decoding_sha256": ev["decoding"]["sha256"], "action_protocol": ev["action_protocol"],
            "modes": {m: ev["modes"][m]["template_kwargs"] for m in MODES},
            "primary_items": {"n": int(an["primary_items"]["n"]), "sha256": an["primary_items"]["sha256"]},
            "independence": {"groups": int(an["units_of_independence"]["groups"]),
                             "sha256": an["units_of_independence"]["sha256"]},
            "seeds": [int(x) for x in d["training"]["seeds"]]}


def result_path(cid: str) -> Path:
    return RESULTS / "comparison_1.json" if cid == "C1" else RECURRENCE_RESULTS / "recurrence_seed20260726.json"


def unit_sets_complete(out: Path, arm: str, seed: int) -> dict:
    """Whether ``arm``'s unit of ``seed`` is complete on its files in ``out``: its final without problems and each
    mode's set complete (`completion.session` with the arm's design and plan bound); ``{"final": bool, mode: bool}``."""
    from stages.single_operator_budget import common as SC
    from stages.single_operator_budget import completion as CP
    before = SC.bind(**{k.lower(): v for k, v in SC.SOB1_FILES.items() if k in ("DESIGN", "LOGS")}) \
        if arm == DC.SOB1_DEPTH else DC.bind_for(arm)
    try:
        plan = None
        if arm != DC.SOB1_DEPTH:
            from stages.depth_merge import plan as PL
            plan = json.loads(PL.out_path(arm).read_text())
        ses = CP.session(Path(out), arm, plan=plan)
        unit = f"{arm}-s{int(seed)}"
        fin = (ses["finals"].get(unit) or {})
        got = {"final": not fin.get("problems", ["absent"]) and bool(fin.get("model_safetensors_sha256")),
               "final_sha256": fin.get("model_safetensors_sha256"), "set_weights_sha256": {}}
        for m in MODES:
            row = ses["sets"].get(f"{arm}_s{int(seed)}/{m}") or {}
            got[m] = bool(row.get("complete"))
            got["set_weights_sha256"][m] = ((row.get("binding") or {}).get("weights_sha256") or {}).get("model.safetensors")
        return got
    finally:
        SC.restore(before)


def session_complete(out: Path, arm: str) -> dict:
    """The arm's session on its files (`completion.session` with SOB1's chain bound to the arm's design; SOB1's own
    files for the depth control)."""
    from stages.single_operator_budget import common as SC
    from stages.single_operator_budget import completion as CP
    before = SC.bind(**{k.lower(): v for k, v in SC.SOB1_FILES.items() if k in ("DESIGN", "LOGS")}) \
        if arm == DC.SOB1_DEPTH else DC.bind_for(arm)
    try:
        plan = None
        if arm != DC.SOB1_DEPTH:
            from stages.depth_merge import plan as PL
            plan = json.loads(PL.out_path(arm).read_text())
        ses = CP.session(Path(out), arm, plan=plan)
        return {"complete": ses["complete"], "n_complete": ses["n_complete"], "n_sets": ses["n_sets"],
                "finals": {u: f["model_safetensors_sha256"] for u, f in ses["finals"].items()}}
    finally:
        SC.restore(before)


def read_arm(out: Path, arm: str, seeds: list[int], primary: list[str], tmp: Path) -> tuple[dict, dict, dict]:
    """``y[mode][seed][item]`` on the static fluid formal core (the registered scorer, rescored here), the scored
    digests and the terminations, for one arm's two seeds and both modes."""
    from stages.single_operator_budget import analysis as SA
    dirs = DC.read_json(DC.DESIGN)["evaluation"]["protocol"]["modes"]
    y, digests, terms = {}, {}, {}
    for mode in MODES:
        for sd in seeds:
            gd = Path(out) / f"{arm}-s{sd}" / dirs[mode]["dir"] / "generations"
            label = f"{arm}_s{sd}"
            if not gd.is_dir():
                continue
            r = SA._set_reading(gd, label, tmp / f"{label}_{mode}")
            y.setdefault(mode, {})[sd] = r["static"]
            digests[f"{label}/{mode}"] = r["digest"]
            terms[f"{label}/{mode}"] = SA.terminations(gd, primary)
    return y, digests, terms


def compare(y_a: dict, y_b: dict, groups: dict, seeds: list[int], pair: tuple[str, str]) -> dict:
    """One comparison in both modes (module docstring)."""
    from stages.single_operator_budget import analysis as SA
    from stages.teacher_parent_initialization import analysis as TA
    items = [i for v in groups.values() for i in v]
    y = {pair[0]: y_a, pair[1]: y_b}
    out: dict = {"_rule": {"alpha": ALPHA, "family": "this comparison alone: thinking at 0.05, non-thinking at 0.05 "
                                                     "only if thinking rejects (fixed sequence)",
                           "formal_requires": "both arms x both seeds cover all 570 items in the mode",
                           "pairing": "same training seed, same item", "seeds": seeds}, "modes": {}}
    gate = True
    for mode in MODES:
        complete = all(TA.covers((y.get(a) or {}).get(mode), seeds, items) for a in pair)
        row: dict = {"complete": complete}
        if complete:
            res = SA.pair_comparison(y, pair, mode, groups, seeds, test=gate)
            row.update(res)
            row["status"] = "formal" if gate else "descriptive_only"
            if gate:
                rejected = res["p_two_sided"] <= ALPHA
                row.update(rejected=rejected, verdict=SA.verdict(pair, res["delta"], rejected))
            else:
                row["why"] = "not tested: the thinking-mode test did not reject (fixed sequence)"
            per = {str(sd): SA._paired({sd: y[pair[0]][mode][sd]}, {sd: y[pair[1]][mode][sd]}, groups, [sd])
                   for sd in seeds}
            signs = {(r["delta"] > 0) - (r["delta"] < 0) for r in per.values()}
            row["per_seed"] = per
            row["seeds_agree_in_sign"] = len(signs) == 1
        else:
            row.update(status="descriptive_only", why="an arm's records do not cover every item in every seed")
        out["modes"][mode] = row
        gate = gate and complete and bool(row.get("rejected"))
    return out


def comparison(cid: str, a_out: Path, b_out: Path | None = None) -> dict:
    """Variant 1's registered comparison C1 from the sealed records (module docstring). ``b_out``: the control's
    secured evidence (SOB1's depth session's by default)."""
    if cid != "C1":
        raise SystemExit(f"{cid}: only C1 is a two-seed comparison; the revised variant 2's are `recurrence`")
    from stages.teacher_parent_initialization import analysis as TA
    reg = comparisons()[cid]
    a_arm, b_arm = reg["pair"]
    b_out = Path(b_out) if b_out is not None else DC.sob1_depth_evidence()
    d = DC.design()
    primary = TA.registered_items(d)
    groups = TA.independence_groups(primary, d)
    seeds = [int(x) for x in d["training"]["seeds"]]
    with tempfile.TemporaryDirectory() as tmp:
        ya, da, ta = read_arm(a_out, a_arm, seeds, primary, Path(tmp) / "a")
        yb, db, tb = read_arm(b_out, b_arm, seeds, primary, Path(tmp) / "b")
    sessions = {a_arm: session_complete(a_out, a_arm), b_arm: session_complete(b_out, b_arm)}
    res = compare(ya, yb, groups, seeds, (a_arm, b_arm))
    if not all(s["complete"] for s in sessions.values()):
        for row in res["modes"].values():         #: a set not complete on its files is never formal
            if row.get("status") == "formal":
                row.update(status="descriptive_only", why="a session is not complete on its files")
                row.pop("rejected", None)
                row.pop("verdict", None)
    for row in res["modes"].values():
        row["complete"] = bool(row.get("complete")) and all(s["complete"] for s in sessions.values())
    return {"schema": SCHEMA, "id": cid, "pair": [a_arm, b_arm], "question": reg["question"],
            "protocol": protocol_identity(d), "n_items": len(primary), "n_groups": len(groups), "seeds": seeds,
            "sessions": sessions, "evidence": {a_arm: str(a_out), b_arm: str(b_out)}, **res,
            "scored_digests": {**da, **db}, "terminations": {**ta, **tb},
            "exposure": "eval-set-1 is development-exposed: nothing here is a never-seen blind result",
            "measures": "item and group sampling with each arm's two checkpoints fixed, not training randomness",
            "error_rate_scope": "this comparison's own family (thinking at 0.05, non-thinking only after it)"}


def registered_controls(d: dict | None = None) -> dict:
    """The single-execution controls the recurrence design FIXES (``base.kept_control`` / ``base.merged_control``):
    each one's arm, unit, final weights' sha256 and completion record (path and sha256)."""
    b = (d or DC.recurrence_design())["base"]
    seed = int((d or DC.recurrence_design())["arms"]["depth_loop_kept"]["seeds"][0])
    kc, mc = b["kept_control"]["sessions"], b["merged_control"]["session"]
    d_unit, m_unit = f"{DC.SOB1_DEPTH}-s{seed}", f"depth_merge-s{seed}"
    return {"D": {"arm": DC.SOB1_DEPTH, "unit": d_unit, "final_sha256": kc["finals_sha256"][d_unit],
                  "completion": kc["completion"]},
            "M": {"arm": "depth_merge", "unit": m_unit, "final_sha256": mc["final_sha256"][m_unit],
                  "completion": mc["completion"]}}


def control_problems(cond: str, unit: dict, root: Path = REPO) -> list[str]:
    """Why the D or M records read (``unit``: `unit_sets_complete`'s answer for that evidence) are not the registered
    control (empty when they are): the completion record by sha256, the final weights' sha256, and each mode's set
    bound to those weights. A copy anywhere passes; a different, internally consistent record does not."""
    reg = registered_controls()[cond]
    bad = []
    cp = Path(root) / reg["completion"]["path"]
    from stages.teacher_parent_initialization.relay import sha256_file
    if not cp.is_file() or sha256_file(cp) != reg["completion"]["sha256"]:
        bad.append(f"{cond}: the completion record {reg['completion']['path']} is not the registered one")
    if unit.get("final_sha256") != reg["final_sha256"]:
        bad.append(f"{cond}: final {str(unit.get('final_sha256'))[:12]} is not the registered {reg['final_sha256'][:12]}")
    for m in MODES:
        if (unit.get("set_weights_sha256") or {}).get(m) != reg["final_sha256"]:
            bad.append(f"{cond}: the {m} set is bound to {str((unit.get('set_weights_sha256') or {}).get(m))[:12]}, "
                       f"not the registered final {reg['final_sha256'][:12]}")
    return bad


def interaction_groups(y: dict[str, dict], groups: dict[str, list[str]]) -> tuple[list[str], "np.ndarray",
                                                                                 "np.ndarray"]:
    """(group keys, d_g, n_g) for R4: d_g sums, over the group's items, the item's interaction
    (y_V2M - y_V2D) - (y_M - y_D) from that item's four records; n_g = |g|. Every item must be present in all four."""
    import numpy as np
    keys, d, n = sorted(groups), [], []
    for k in keys:
        tot = 0
        for i in groups[k]:
            missing = [c for c in CONDITIONS if i not in y[c]]
            if missing:
                raise ValueError(f"group {k}: item {i} has no record in {missing}")
            tot += (int(y["V2-M"][i]) - int(y["V2-D"][i])) - (int(y["M"][i]) - int(y["D"][i]))
        d.append(tot)
        n.append(len(groups[k]))
    return keys, np.asarray(d, dtype=np.int64), np.asarray(n, dtype=np.int64)


def recurrence_modes(y: dict[str, dict[str, dict]], groups: dict[str, list[str]], seed: int) -> dict:
    """R1-R4 in each mode from ``y[mode][condition][item]`` (module docstring): a comparison is computed in a mode only
    when every condition it names covers all the registered items there."""
    from stages.single_operator_budget import analysis as SA
    from stages.teacher_parent_initialization import analysis as TA
    items = [i for v in groups.values() for i in v]
    pairs = {"R1": ("V2-D", "D"), "R2": ("V2-M", "M"), "R3": ("V2-M", "V2-D")}
    out = {}
    for m in MODES:
        ym = y.get(m) or {}
        covered = {c: ym.get(c) is not None and all(i in ym[c] for i in items) for c in CONDITIONS}
        row = {"accuracy_pct": {c: (round(100 * sum(int(ym[c][i]) for i in items) / len(items), 2) if covered[c]
                                    else None) for c in CONDITIONS},
               "covered": covered}
        for rid, (a, b) in pairs.items():
            if covered[a] and covered[b]:
                r = SA._paired({seed: ym[a]}, {seed: ym[b]}, groups, [seed])
                row[rid] = {"estimate_pp": round(100 * r["delta"], 2),
                            "ci95_pp": [round(100 * x, 2) for x in r["effect_interval"]["ci95_percentile"]],
                            "n_items": r["n_paired_records"], "raw": r}
            else:
                row[rid] = {"status": "not_computed", "why": f"{a} or {b} does not cover the {len(items)} items"}
        if all(covered.values()):
            _k, d, n = interaction_groups(ym, groups)
            ei = TA.LA.group_bootstrap(d, n, draws=TA.BOOT_DRAWS, seed=SA.BOOT_SEED)
            row["R4"] = {"estimate_pp": round(100 * float(d.sum() / n.sum()), 2),
                         "ci95_pp": [round(100 * x, 2) for x in ei["ci95_percentile"]], "n_items": int(n.sum()),
                         "raw": {"estimate": float(d.sum() / n.sum()), "effect_interval": ei},
                         "check": "equals R3 - (M - D) item for item"}
        else:
            row["R4"] = {"status": "not_computed", "why": "a condition does not cover the items"}
        out[m] = row
    return out


def recurrence(kept_out: Path, merged_out: Path, *, d_out: Path | None = None, m_out: Path | None = None) -> dict:
    """The revised variant 2's comparisons R1-R4 at seed 20260726 (module docstring)."""
    from stages.teacher_parent_initialization import analysis as TA
    s6 = DC.recurrence_design()
    seed = int(s6["arms"]["depth_loop_kept"]["seeds"][0])
    if [int(x) for x in s6["arms"]["depth_loop_merged"]["seeds"]] != [seed]:
        raise SystemExit("the two loop arms do not register the same single seed")
    d = DC.design()
    primary = TA.registered_items(d)
    groups = TA.independence_groups(primary, d)
    outs = {"V2-D": Path(kept_out), "V2-M": Path(merged_out),
            "D": Path(d_out) if d_out else DC.sob1_depth_evidence(),
            "M": Path(m_out) if m_out else DC.merged_control_evidence()}
    y, digests, terms, units = {}, {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for c, arm in CONDITIONS.items():
            units[c] = {"arm": arm, "evidence": str(outs[c]), **unit_sets_complete(outs[c], arm, seed)}
            if c in ("D", "M"):
                bad = control_problems(c, units[c])
                if bad:
                    raise SystemExit(f"the {c} control read from {outs[c]} is not the registered one: {bad}")
            yc, dc, tc = read_arm(outs[c], arm, [seed], primary, Path(tmp) / c)
            digests.update(dc)
            terms.update(tc)
            for m in MODES:
                if units[c].get(m) and units[c]["final"] and seed in (yc.get(m) or {}):
                    y.setdefault(m, {})[c] = yc[m][seed]
    return {"schema": RECURRENCE_SCHEMA, "seed": seed, "conditions": CONDITIONS,
            "comparisons": s6["analysis"]["comparisons"],
            "protocol": protocol_identity(d) | {"seeds": [seed]}, "n_items": len(primary), "n_groups": len(groups),
            "units": units, "modes": recurrence_modes(y, groups, seed), "scored_digests": digests,
            "terminations": terms, "primary": s6["analysis"]["primary"], "statistic": s6["analysis"]["statistic"],
            "effect_interval": s6["analysis"]["effect_interval"], "test": s6["analysis"]["test"],
            "interpretation": s6["analysis"]["interpretation"],
            "exposure": "eval-set-1 is development-exposed: nothing here is a never-seen blind result",
            "measures": "item and group sampling with each checkpoint fixed (one training seed), not training randomness"}


SPLIT_SCHEMA = "aadistill.dmr1.split_comparisons/v1"
#: v1-new's conditions: its own arm and the two reused controls (the registered D and M finals of its seed)
SPLIT_CONDITIONS = {"v1-new": "depth_merge_split", "D": DC.SOB1_DEPTH, "M": "depth_merge"}


def split_comparisons(new_out: Path, *, d_out: Path | None = None, m_out: Path | None = None) -> dict:
    """v1-new's comparisons X1 (v1-new - D) and X2 (v1-new - M), descriptive, at its one seed (split_design.json
    ``analysis``): the paired difference over the registered items, the percentile 95% group-bootstrap interval, no
    test. D and M are the registered controls, read and checked by `control_problems`, never retrained."""
    from stages.single_operator_budget import analysis as SA
    from stages.teacher_parent_initialization import analysis as TA
    sd = DC.design_of("depth_merge_split")
    seeds = [int(x) for x in sd["arms"]["depth_merge_split"]["seeds"]]
    if len(seeds) != 1:
        raise SystemExit(f"v1-new registers one seed, not {seeds}")
    seed = seeds[0]
    if int(DC.recurrence_design()["arms"]["depth_loop_kept"]["seeds"][0]) != seed:
        raise SystemExit("the registered controls are of another seed")
    d = DC.design()
    primary = TA.registered_items(d)
    groups = TA.independence_groups(primary, d)
    items = [i for v in groups.values() for i in v]
    outs = {"v1-new": Path(new_out), "D": Path(d_out) if d_out else DC.sob1_depth_evidence(),
            "M": Path(m_out) if m_out else DC.merged_control_evidence()}
    y, digests, terms, units = {}, {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for c, arm in SPLIT_CONDITIONS.items():
            units[c] = {"arm": arm, "evidence": str(outs[c]), **unit_sets_complete(outs[c], arm, seed)}
            if c in ("D", "M"):
                bad = control_problems(c, units[c])
                if bad:
                    raise SystemExit(f"the {c} control read from {outs[c]} is not the registered one: {bad}")
            yc, dc, tc = read_arm(outs[c], arm, [seed], primary, Path(tmp) / c)
            digests.update(dc)
            terms.update(tc)
            for m in MODES:
                if units[c].get(m) and units[c]["final"] and seed in (yc.get(m) or {}):
                    y.setdefault(m, {})[c] = yc[m][seed]
    modes = {}
    for m in MODES:
        ym = y.get(m) or {}
        covered = {c: ym.get(c) is not None and all(i in ym[c] for i in items) for c in SPLIT_CONDITIONS}
        row = {"accuracy_pct": {c: (round(100 * sum(int(ym[c][i]) for i in items) / len(items), 2) if covered[c]
                                    else None) for c in SPLIT_CONDITIONS}, "covered": covered}
        for xid, (a, b) in (("X1", ("v1-new", "D")), ("X2", ("v1-new", "M"))):
            if covered[a] and covered[b]:
                r = SA._paired({seed: ym[a]}, {seed: ym[b]}, groups, [seed])
                row[xid] = {"estimate_pp": round(100 * r["delta"], 2),
                            "ci95_pp": [round(100 * x, 2) for x in r["effect_interval"]["ci95_percentile"]],
                            "n_items": r["n_paired_records"], "raw": r}
            else:
                row[xid] = {"status": "not_computed", "why": f"{a} or {b} does not cover the {len(items)} items"}
        modes[m] = row
    an = sd["analysis"]
    return {"schema": SPLIT_SCHEMA, "seed": seed, "conditions": SPLIT_CONDITIONS, "comparisons": an["comparisons"],
            "protocol": protocol_identity(d) | {"seeds": [seed]}, "n_items": len(primary), "n_groups": len(groups),
            "units": units, "modes": modes, "scored_digests": digests, "terminations": terms,
            "structures": an["structures"], "reading": an["reading"], "test": an["test"],
            "exposure": "eval-set-1 is development-exposed: nothing here is a never-seen blind result",
            "measures": "item and group sampling with each checkpoint fixed (one training seed), not training randomness"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("comparison")
    c.add_argument("--id", required=True, choices=("C1",))
    c.add_argument("--a-evidence", required=True)
    c.add_argument("--b-evidence", default=None)
    c.add_argument("--write", action="store_true")
    r = sub.add_parser("recurrence")
    r.add_argument("--kept-evidence", required=True)
    r.add_argument("--merged-evidence", required=True)
    r.add_argument("--write", action="store_true")
    s = sub.add_parser("split")
    s.add_argument("--evidence", required=True, help="v1-new's secured training evidence")
    s.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "comparison":
        doc = comparison(a.id, Path(a.a_evidence), Path(a.b_evidence) if a.b_evidence else None)
        p = result_path(a.id)
        brief = {"id": doc["id"], "pair": doc["pair"],
                 "modes": {m: {k: r_.get(k) for k in ("status", "delta", "p_two_sided", "verdict")}
                           for m, r_ in doc["modes"].items()}}
    elif a.cmd == "split":
        doc = split_comparisons(Path(a.evidence))
        p = RESULTS / f"split_seed{doc['seed']}.json"
        brief = {m: {k: (v.get("estimate_pp") if isinstance(v, dict) and "estimate_pp" in v else v)
                     for k, v in row.items() if k in ("accuracy_pct", "X1", "X2")} for m, row in doc["modes"].items()}
    else:
        doc = recurrence(Path(a.kept_evidence), Path(a.merged_evidence))
        p = result_path("R")
        brief = {m: {k: (v.get("estimate_pp") if isinstance(v, dict) and "estimate_pp" in v else v)
                     for k, v in row.items() if k in ("accuracy_pct", "R1", "R2", "R3", "R4")}
                 for m, row in doc["modes"].items()}
    if a.write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(p.relative_to(REPO))
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

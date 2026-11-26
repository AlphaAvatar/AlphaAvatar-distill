#!/usr/bin/env python3
"""DMR1's registered analysis (``configs/stages/stage-6/depth_recurrence/design.json :: analysis``), from the sealed
post-recovery sets ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/analysis.py \\
        comparison --id C1 --a-evidence <depth_merge's secured out> [--write]
    ... comparison --id C2 --a-evidence <depth_loop's out> [--b-evidence <depth_merge's out, on the MERGED path>] [--write]
    ... comparison --id C3 --a-evidence <depth_loop_hidden's out> --b-evidence <depth_loop's out> [--write]
    ... sensitivity [--write]          # Holm over the three thinking-mode p-values, once all three exist

Each comparison (A, B) is its own family (the design's ``family``): the thinking-mode test at alpha 0.05 on the static
fluid formal core (570 items in 558 independence groups), both seeds pooled, each record paired with the other arm's
record of the same training seed and item; the non-thinking test at 0.05 only if the thinking one rejects (fixed
sequence), otherwise descriptive. Formal only when both arms' records of both seeds cover all 570 items in the mode;
otherwise deltas and intervals only. The statistic, the test (B = 100,000) and the interval are TPI1's / SOB1's
(`single_operator_budget.analysis.pair_comparison`); per-seed deltas, sign agreement and the terminations are reported
beside. Nothing here pools across comparisons; `sensitivity` is a separate line, never a verdict.

The control arm's sealed sets are read where they are: SOB1's depth arm from its COMPLETE session's secured evidence,
variant 1's and the loop arms' from theirs. A set is read only when its arm's session is complete on its files
(`completion.session`, SOB1's chain bound to that arm's design).
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
ALPHA = 0.05
RESULTS = DC.LOGS / "results"
STAGE6_RESULTS = DC.STAGE6_LOGS / "results"
MODES = ("H-T", "H-N")


def comparisons() -> dict:
    return DC.stage6_design()["analysis"]["comparisons"]


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


def bound_init(arm: str, root: Path = REPO) -> dict:
    """The recurrent initialization a loop arm's committed training plan binds, read through the plan (never through
    the current selection file): the init record (re-hashed), its path (KEPT / MERGED), the selection it names and
    whether that selection still re-derives from comparison 1 (`schedule.selection_problems`)."""
    from stages.depth_merge import plan as PL
    from stages.depth_recurrence import schedule as SCH
    plan_p = PL.out_path(arm)
    if plan_p.is_relative_to(REPO) and Path(root) != REPO:
        plan_p = Path(root) / plan_p.relative_to(REPO)
    plan = SCH.read_json(plan_p)
    ref = plan["init_record"]
    p = Path(root) / ref["path"]
    if not p.is_file() or SCH.sha256_file(p) != ref["sha256"]:
        raise SystemExit(f"{arm}: the init record its plan binds ({ref['path']}) is not here as bound")
    rec = SCH.read_json(p)
    if rec.get("path") not in SCH.PATHS or (rec.get("selection") or {}).get("choice") != rec["path"]:
        raise SystemExit(f"{arm}: the bound init record names no consistent selection")
    return {"init_record": ref, "path": rec["path"], "selection": rec["selection"],
            "selection_problems": SCH.selection_problems(rec["selection"], root=root)}


def result_path(cid: str) -> Path:
    return (RESULTS if cid == "C1" else STAGE6_RESULTS) / f"comparison_{cid[1:]}.json"


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
    """One registered comparison from the sealed records (module docstring). ``b_out``: the control's secured
    evidence (SOB1's depth session's by default for C1 and the KEPT path of C2)."""
    from stages.teacher_parent_initialization import analysis as TA
    reg = comparisons()[cid]
    a_arm = reg["pair"][0]
    b_arm = reg["pair"][1]
    binding = None
    if cid in ("C2", "C3"):
        #: the control is fixed by the initialization the TRAINED units bound, never by a later selection file
        binding = {a: bound_init(a) for a in (("depth_loop",) if cid == "C2" else ("depth_loop_hidden", "depth_loop"))}
        bad = {a: b["selection_problems"] for a, b in binding.items() if b["selection_problems"]}
        if bad:
            raise SystemExit(f"{cid}: a bound selection does not re-derive from comparison 1: {bad}")
        if cid == "C3" and binding["depth_loop_hidden"]["init_record"] != binding["depth_loop"]["init_record"]:
            raise SystemExit("C3: the two loop arms bound different initializations")
    if cid == "C2":
        b_arm = {"KEPT": DC.SOB1_DEPTH, "MERGED": "depth_merge"}[binding["depth_loop"]["path"]]
    b_out = Path(b_out) if b_out is not None else (DC.sob1_depth_evidence() if b_arm == DC.SOB1_DEPTH else None)
    if b_out is None:
        raise SystemExit(f"{cid}: the control {b_arm}'s evidence is needed (--b-evidence)")
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
    out = {"schema": SCHEMA, "id": cid, "pair": [a_arm, b_arm], "question": reg["question"],
           "protocol": protocol_identity(d), "n_items": len(primary), "n_groups": len(groups), "seeds": seeds,
           "sessions": sessions, "evidence": {a_arm: str(a_out), b_arm: str(b_out)}, **res,
           "scored_digests": {**da, **db}, "terminations": {**ta, **tb},
           "exposure": "eval-set-1 is development-exposed: nothing here is a never-seen blind result",
           "measures": "item and group sampling with each arm's two checkpoints fixed, not training randomness",
           "error_rate_scope": "this comparison's own family (thinking at 0.05, non-thinking only after it); no "
                               "error rate is claimed for the round's three comparisons together"}
    if binding is not None:
        out["initialization"] = binding
    if cid == "C2":
        out["selection_conditioning"] = (
            "both arms start from the initialization comparison 1 chose on these same eval-set-1 items (the "
            "selection read C1's thinking-mode deltas; C1's control or variant 1 is this comparison's control): C2 "
            "estimates the recurrence's effect AT the chosen initialization, conditional on that selection -- not "
            "its effect at an initialization chosen independently of this set")
    return out


def sensitivity() -> dict:
    """Holm over the three thinking-mode p-values (m = 3), once all three exist: reported, never the decision."""
    from stages.single_operator_budget import analysis as SA
    pv = {}
    for cid in ("C1", "C2", "C3"):
        p = result_path(cid)
        if not p.is_file():
            return {"status": "INCOMPLETE", "why": f"{cid} has no record yet"}
        row = DC.read_json(p)["modes"]["H-T"]
        if row.get("status") != "formal":
            return {"status": "INCOMPLETE", "why": f"{cid}'s thinking-mode test is not formal"}
        pv[cid] = float(row["p_two_sided"])
    return {"status": "COMPLETE", "role": "a sensitivity line beside the per-comparison verdicts, never a verdict",
            "holm_m3": SA.holm(pv, m=3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("comparison")
    c.add_argument("--id", required=True, choices=("C1", "C2", "C3"))
    c.add_argument("--a-evidence", required=True)
    c.add_argument("--b-evidence", default=None)
    c.add_argument("--write", action="store_true")
    s = sub.add_parser("sensitivity")
    s.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    doc = comparison(a.id, Path(a.a_evidence), Path(a.b_evidence) if a.b_evidence else None) if a.cmd == "comparison" \
        else sensitivity()
    if a.write:
        p = result_path(a.id) if a.cmd == "comparison" else STAGE6_RESULTS / "sensitivity.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(p.relative_to(REPO))
    print(json.dumps({k: v for k, v in doc.items() if k in ("id", "pair", "status")} |
                     {"modes": {m: {k: r.get(k) for k in ("status", "delta", "p_two_sided", "verdict")}
                                for m, r in (doc.get("modes") or {}).items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

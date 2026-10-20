#!/usr/bin/env python3
"""The formal Stage-0 volume study's analysis, as preregistered. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_formal.py \
        --dose d1=<dir> --dose d2=<dir> --dose d4=<dir> --out <json>

Each `<dir>` is one dose's secured records (`<scr>/store/<treatment>/`):
`search/` (search record, selection, structural record, journal, root WIDTH
projections), `recovery/` (product identity, training log), `eval/` (the
eval-set-1 report and its per-row file). Implements
`plans/volume_preregistration.json :: analysis` and nothing else:

* H1 -- per dose the committed Top-1's path; root-level selection agreement
  between doses for every operator and profile (the same parent, the teacher,
  so the comparison isolates the calibration data), each beside chance; Top-1
  per-operator agreement only where both doses applied the operator to the
  same parent;
* H2 -- every unit and capability's formal numbers per dose, and the paired
  contrasts 2x-1x, 4x-2x, 4x-1x with bootstrap intervals over EVALUATION
  sampling (fluid: rule-family clusters within axis; capability units: items;
  paired sets: pairs), labelled by the preregistered SESOI;
* H3 -- the three doses' shape, described;
* separation -- structure, the Top-1's step-0 state evaluation, the recovery's
  own log and the recovered student's capabilities, side by side, never
  combined.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
import sys  # noqa: E402
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume.analyze_structure import agreement, load_roots  # noqa: E402

PREREG = REPO / "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"
CONTRASTS = (("d2", "d1"), ("d4", "d2"), ("d4", "d1"))
B = 10_000
STATIC_AXES = ("inductive_rule_discovery", "compositional_generalization",
               "task_contained_deduction", "abstract_analogical")


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def label(lo: float, hi: float, sesoi: float) -> str:
    if lo > sesoi:
        return "GO"
    if lo > -sesoi and hi < sesoi:
        return "NO-GO"
    return "INCONCLUSIVE"


def cluster_bootstrap(diffs: dict[str, float], cluster: dict[str, str],
                      strata: dict[str, str], seed: int) -> tuple[float, float, float]:
    """Mean paired difference with a percentile interval: clusters resampled
    with replacement inside each stratum."""
    by = defaultdict(lambda: defaultdict(list))
    for i, d in diffs.items():
        by[strata[i]][cluster[i]].append(d)
    rng = random.Random(seed)
    point = sum(diffs.values()) / len(diffs)
    stats = []
    for _ in range(B):
        tot, n = 0.0, 0
        for cl in by.values():
            keys = list(cl)
            for _ in keys:
                v = cl[keys[rng.randrange(len(keys))]]
                tot += sum(v)
                n += len(v)
        stats.append(tot / n)
    stats.sort()
    return point, stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def h2(doses: dict[str, Path], sesoi: dict) -> dict:
    from shared import eval_sets as ES
    tasks = {t["task_id"]: t for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1")}
    rows = {d: {u.name.split(".rows")[0]: {r["id"]: r for r in jl(u)}
                for u in (p / "eval" / "rows").glob("*.rows.jsonl")} for d, p in doses.items()}
    reports = {d: json.loads((p / "eval" / "report.json").read_text())
               for d, p in doses.items() if (p / "eval" / "report.json").is_file()}
    out = {"per_dose": {d: {"units": r["units"], "capabilities": r["capabilities"],
                            "eval_set_identity": r["eval_set_identity"]}
                        for d, r in reports.items()}, "contrasts": {}}
    fluid = "fluid_reasoning_720__dev_v1"
    for b_, a_ in CONTRASTS:
        if b_ not in rows or a_ not in rows:
            continue
        key = f"{b_}-{a_}"
        res = {}
        ra, rb = rows[a_].get(fluid, {}), rows[b_].get(fluid, {})
        ids = sorted(i for i in ra if i in rb and ra[i].get("formal") and rb[i].get("formal"))

        def contrast(sel, name, sesoi_v, seed):
            diffs = {i: float(bool(rb[i]["correct"])) - float(bool(ra[i]["correct"])) for i in sel}
            if not diffs:
                return
            cl = {i: f"{tasks[i]['generator_id']}/{tasks[i]['family_id']}" for i in sel}
            st = {i: tasks[i]["axis"] for i in sel}
            p, lo, hi = cluster_bootstrap(diffs, cl, st, seed)
            res[name] = {"n": len(sel), "rate_a": round(sum(bool(ra[i]["correct"]) for i in sel) / len(sel), 4),
                         "rate_b": round(sum(bool(rb[i]["correct"]) for i in sel) / len(sel), 4),
                         "diff": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "sesoi": sesoi_v, "label": label(lo, hi, sesoi_v)}

        static = [i for i in ids if not tasks[i]["interactive"]]
        contrast(static, "fluid.static_pooled", sesoi["pooled_primary"], 1)
        for k, ax in enumerate(STATIC_AXES):
            contrast([i for i in static if tasks[i]["axis"] == ax], f"fluid.{ax}",
                     sesoi["per_axis"], 10 + k)
        contrast([i for i in ids if tasks[i]["interactive"]], "fluid.interactive_adaptation",
                 sesoi["per_axis"], 20)
        for unit in sorted(set(rows[a_]) & set(rows[b_]) - {fluid, "stage3_capability_846__eval_behavior_v0"}):
            ua, ub = rows[a_][unit], rows[b_][unit]
            comp, u = unit.split("__")
            items = {r["id"]: r for r in ES.items("eval-set-1", comp, u)} \
                if ES.unit_path("eval-set-1", comp, u) else {}
            sel = sorted(i for i in ua if i in ub and ua[i].get("formal") and ub[i].get("formal"))
            if not sel or "correct" not in ua[sel[0]]:
                continue
            if items and "pair_id" in items.get(sel[0], {}):
                pairs = defaultdict(list)
                for i in sel:
                    pairs[items[i]["pair_id"]].append(i)
                diffs = {p: float(all(ub[i]["correct"] for i in m)) - float(all(ua[i]["correct"] for i in m))
                         for p, m in pairs.items()}
                p, lo, hi = cluster_bootstrap(diffs, {k: k for k in diffs}, {k: "all" for k in diffs}, 30)
                res[unit] = {"n_pairs": len(diffs), "unit_of_analysis": "pair", "diff": round(p, 4),
                             "ci95": [round(lo, 4), round(hi, 4)]}
            else:
                diffs = {i: float(bool(ub[i]["correct"])) - float(bool(ua[i]["correct"])) for i in sel}
                p, lo, hi = cluster_bootstrap(diffs, {k: k for k in diffs}, {k: "all" for k in diffs}, 40)
                res[unit] = {"n": len(sel), "diff": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            res[unit]["role"] = ("in-sample regression, not generalization"
                                 if comp == "stage3_legacy_150" else
                                 "supplementary (outside the formal core)" if u == "knowledge"
                                 else "held out")
        out["contrasts"][key] = res
    return out


def top1(path: Path) -> dict:
    sel = json.loads((path / "search" / "stage1_selection.json").read_text())["selected"][0]
    journal, traces = {}, {}
    for r in jl(path / "search" / "search" / "states.jsonl"):
        journal[r["state_id"]] = r
        for i, st in enumerate(r.get("steps") or []):
            if st.get("trace"):
                traces.setdefault((r["state_id"], i), st["trace"])
    s = journal.get(sel["state_id"], {})
    #: a resumed search journals restored states again with empty step traces,
    #: and a leaf built on one inherits them; the ancestor's first record has it
    chain, cur = [], sel["state_id"]
    while cur in journal and cur not in chain:
        chain.append(cur)
        cur = journal[cur].get("parent_id")
    steps = [dict(st, trace=st.get("trace") or next(
                 (traces[(c, i)] for c in chain if (c, i) in traces), {}))
             for i, st in enumerate(s.get("steps") or [])]
    return {"state_id": sel["state_id"], "path": sel.get("path"),
            "order": [st.get("kind") for st in steps],
            "profiles": [st.get("profile_id") for st in steps],
            "num_parameters": s.get("num_parameters"),
            "artifact_digest": s.get("artifact_digest"),
            "state_evaluation_step0": (s.get("evaluation") or {}).get("values"),
            "per_step": [{"kind": st.get("kind"), "profile": st.get("profile_id"),
                          "parent_spec": (journal.get(s.get("parent_id")) or {}).get("arch_spec_hash"),
                          "trace": {k: v for k, v in (st.get("trace") or {}).items()
                                    if k in ("removed_layers", "kept_layers", "removal_order")}}
                         for st in steps]}


def h1(doses: dict[str, Path]) -> dict:
    import torch
    from safetensors.torch import load_file
    roots, proj = {}, {}
    for d, p in doses.items():
        sp = p / "search" / "structure.json"
        if not sp.is_file():
            continue
        roots[d] = load_roots(sp)
        for pid, r in roots[d].get("RESIDUAL_WIDTH", {}).items():
            f = p / "search" / f"width_projection_{r['state_id']}.safetensors"
            if f.is_file():
                q, _ = torch.linalg.qr(load_file(str(f))["P"].double())
                proj[(d, pid)] = q
    out = {"top1": {d: top1(p) for d, p in doses.items()
                    if (p / "search" / "stage1_selection.json").is_file()},
           "root_agreement": {}}

    def ptype(pid: str) -> str:
        return "base" if "_base_" in pid else "rh"

    for (da, db) in itertools.combinations(sorted(roots), 2):
        res = {}
        for kind in sorted(set(roots[da]) & set(roots[db])):
            for pa, ra in roots[da][kind].items():
                for pb, rb in roots[db][kind].items():
                    if ptype(pa) != ptype(pb):
                        continue
                    pj = {pa: proj.get((da, pa)), pb: proj.get((db, pb))}
                    pj = {k: v for k, v in pj.items() if v is not None}
                    res[f"{kind}/{ptype(pa)}"] = agreement(kind, ra.get("decision") or {},
                                                           rb.get("decision") or {}, pj, pa, pb)
        out["root_agreement"][f"{da}~{db}"] = res
    return out


def separation(doses: dict[str, Path], h2_out: dict, h1_out: dict) -> dict:
    out = {}
    for d, p in doses.items():
        log = jl(p / "recovery" / "train_log.jsonl")
        evals = [r for r in log if r.get("event") in ("eval", "eval_result") or "val_loss" in r]
        steps = [r for r in log if "loss" in r]
        rep = (h2_out["per_dose"].get(d) or {}).get("units", {})
        fl = rep.get("fluid_reasoning_720/dev_v1", {})
        out[d] = {"structure": {k: (h1_out["top1"].get(d) or {}).get(k)
                                for k in ("path", "order", "profiles", "num_parameters")},
                  "step0_state_evaluation": (h1_out["top1"].get(d) or {}).get("state_evaluation_step0"),
                  "recovery": {"last_train_record": steps[-1] if steps else None,
                               "last_eval_record": evals[-1] if evals else None,
                               "products": json.loads((p / "recovery" / "recovery_products.json").read_text())
                               if (p / "recovery" / "recovery_products.json").is_file() else None},
                  "capability": {"fluid_formal_correct": fl.get("correct"),
                                 "fluid_formal_n": fl.get("n_items"),
                                 "fluid_formal_rate": fl.get("rate")}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", action="append", required=True, help="d1=<dir>")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    prereg = json.loads(PREREG.read_text())
    doses = {k: Path(v) for k, v in (x.split("=", 1) for x in args.dose)}
    sesoi = prereg["analysis"]["statistics"]["sesoi"]
    h1_out = h1(doses)
    h2_out = h2(doses, sesoi)
    out = {"schema": "aadistill.stage0.calibration_volume.formal_analysis/v1",
           "preregistration": str(PREREG.relative_to(REPO)),
           "doses": {d: str(p) for d, p in doses.items()},
           "H1": h1_out, "H2": h2_out,
           "H3": {"read": "the 1x/2x/4x shape of H1 and H2 above; described, never tested"},
           "separation": separation(doses, h2_out, h1_out),
           "claims": prereg["analysis"]["statistics"]["not_claimed"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v.get("label", v.get("diff")) for c in h2_out["contrasts"].values()
                      for k, v in c.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive a D2 search's evidence summaries from its raw records, at $0.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/analyze_d2_search.py <dir> \
        [--selection stage1_selection.json] [--d1-states <d1 states.jsonl>] \
        [--out-dir <run>/evidence]

<dir> holds the driver's `states.jsonl` and `telemetry.jsonl` (and optionally
`d2_search.json`): a relay directory mid-run, or the extracted store at
closeout. READS ONLY; every number below is derived from those records, so the
committed summaries can be regenerated from the archived journal.

Writes, under --out-dir:

* `complete_leaf_ranking.json` -- every complete leaf: the committed selection's
  own order and reason, the objective beside the raw union KL, the root
  teacher-top-k KL and D1's metric, per domain, with digests;
* `depth_choices.json` -- every DEPTH expansion: the layers it removed, which
  layer each KL form ranks first in round 0 (objective / raw union KL / root
  top-k KL / parent KL) and how often a greedy round's objective choice is also
  the raw-KL argmin;
* `search_analysis.json` -- operator timing, cache modes and counters, the beam
  trajectory level by level, leaf rank correlations, and (with --d1-states) the
  metric-alignment check on states D1 and D2 share by artifact digest.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

OBJ = "state.objective.equal_domain_mean"
RAW = "state.objective.raw_kl.equal_domain_mean"
TOPK = "state.objective.kl_reference_topk.equal_domain_mean"
CONF = "state.objective.reference_confidence_weight.equal_domain_mean"
D1KL = "state.teacher_kl.equal_domain_mean"
DOMAINS = ("code", "general", "rag_multihop", "reasoning_math", "tool")


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def last_records(path: Path) -> dict[str, dict]:
    """The journal appends a state more than once; its last line is current."""
    out: dict[str, dict] = {}
    for r in jsonl(path):
        out[r["state_id"]] = r
    return out


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else None


def vals(r: dict) -> dict:
    return (r.get("evaluation") or {}).get("values") or {}


def operator_timing(tel: list[dict]) -> dict:
    rows = [r for r in tel if r.get("event", "expansion") == "expansion"]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["impl_id"]].append(r)
    per_impl = {}
    for impl, rs in sorted(by.items()):
        secs = [float(r.get("operator_seconds") or 0) for r in rs]
        over = [float(r.get("parent_load_seconds") or 0) + sum(
            float((r.get(k) or {}).get("seconds") or 0) for k in (
                "identify_seconds", "materialize_seconds",
                "state_evaluation_seconds", "validation_seconds",
                "canonical_reload_seconds")) for r in rs]
        per_impl[impl] = {
            "expansions": len(rs),
            "operator_minutes_total": round(sum(secs) / 60, 2),
            "operator_minutes_mean": round(statistics.fmean(secs) / 60, 2),
            "operator_minutes_min": round(min(secs) / 60, 2),
            "operator_minutes_max": round(max(secs) / 60, 2),
            "per_expansion_overhead_minutes_total": round(sum(over) / 60, 2)}
    depth = [r for r in rows if str(r.get("impl_id", "")).startswith("depth.")]
    counters: collections.Counter = collections.Counter()
    for r in depth:
        counters.update(r.get("reference_counters") or {})

    def timing(key: str) -> float:
        return round(sum(float((r.get("operator_timing") or {}).get(key) or 0)
                         for r in depth), 2)

    return {
        "per_impl": per_impl,
        "expansions": len(rows),
        "operator_minutes_total": round(sum(float(r.get("operator_seconds") or 0)
                                            for r in rows) / 60, 2),
        "depth_cache": {
            "reference_cache_modes": dict(collections.Counter(
                str((r.get("reference_cache") or {}).get("mode")) for r in depth)),
            "parent_reference_cache_modes": dict(collections.Counter(
                str((r.get("parent_reference_cache") or {}).get("mode"))
                for r in depth)),
            "reference_counters_summed": dict(counters),
            "objective_seconds_total": timing("objective_seconds"),
            "parent_kl_seconds_total": timing("parent_kl_seconds"),
            "ablated_seconds_total": timing("ablated_seconds"),
            "distortion_seconds_total": timing("distortion_seconds")},
        "root_reference_stores": [
            {"profile_hash": r.get("profile"),
             "build_seconds": r.get("build_seconds"),
             "identity_hash": r.get("identity_hash"),
             "content_sha256": r.get("content_sha256"),
             "mode": (r.get("decision") or {}).get("mode"),
             "teacher_confidence": {
                 k: (r.get("confidence") or {}).get(k) for k in (
                     "positions", "mean", "sd", "min", "effective_fraction")}
             | {"quantiles": (r.get("confidence") or {}).get("quantiles")}}
            for r in tel if r.get("event") == "root_reference_built"]}


def beam(states: dict[str, dict]) -> dict:
    by_depth: dict[int, list[dict]] = collections.defaultdict(list)
    for r in states.values():
        by_depth[int(r.get("depth") or 0)].append(r)
    parents = {r.get("parent_id") for r in states.values()}
    levels = []
    for d in sorted(by_depth):
        rs = by_depth[d]

        def row(r: dict) -> dict:
            return {"state_id": r["state_id"], "path": r.get("path_label"),
                    "objective": vals(r).get(OBJ), "raw_union_kl": vals(r).get(RAW)}

        kept = sorted((row(r) for r in rs if r["state_id"] in parents),
                      key=lambda x: x["objective"])
        levels.append({
            "depth": d, "states": len(rs), "expanded": len(kept),
            "complete_leaves": sum(1 for r in rs if r.get("is_complete_leaf")),
            "evaluated": sum(1 for r in rs if vals(r)),
            "invalid": sum(1 for r in rs if r.get("invalid_reason")),
            "pruned": sum(1 for r in rs if r.get("prune_reason")),
            "expanded_states": kept,
            "best_not_expanded": sorted(
                (row(r) | {"prune_reason": r.get("prune_reason")} for r in rs
                 if r["state_id"] not in parents and not r.get("is_complete_leaf")),
                key=lambda x: x["objective"])[:6]})
    return {"distinct_states_in_journal": len(states), "levels": levels}


def leaf_ranking(states: dict[str, dict], selection: dict | None) -> dict:
    decisions = {d["state_id"]: d for d in (selection or {}).get("decisions", [])}
    rows = []
    for r in states.values():
        if not r.get("is_complete_leaf"):
            continue
        v = vals(r)
        dec = decisions.get(r["state_id"], {})
        rows.append({
            "state_id": r["state_id"], "path": r.get("path_label"),
            "impl_ids": r.get("impl_ids"),
            "quality_order": dec.get("quality_order"), "front": dec.get("front"),
            "selected": dec.get("selected"), "selection_reason": dec.get("reason"),
            "objective_equal_domain_mean": v.get(OBJ),
            "objective_worst_domain": v.get("state.objective.worst_domain"),
            "objective_critical_token": v.get("state.objective.critical_token"),
            "raw_union_kl_equal_domain_mean": v.get(RAW),
            "root_teacher_topk_kl_equal_domain_mean": v.get(TOPK),
            "d1_metric_teacher_kl_equal_domain_mean": v.get(D1KL),
            "reference_confidence_weight_mean": v.get(CONF),
            "objective_over_raw": (v[OBJ] / v[RAW]) if v.get(RAW) else None,
            "per_domain": {dom: {"objective": v.get(f"state.objective.domain.{dom}"),
                                 "raw_union_kl": v.get(f"state.teacher_kl.{dom}")}
                           for dom in DOMAINS},
            "removed_layers": next((s.get("trace", {}).get("removed_layers")
                                    for s in r.get("steps") or []
                                    if str(s.get("impl_id", "")).startswith("depth.")),
                                   None),
            "artifact_digest": r.get("artifact_digest"),
            "weights_digest": (r.get("artifact") or {}).get("weights_digest"),
            "single_shard_sha256": (r.get("artifact") or {}).get("single_shard_sha256"),
            "num_parameters": r.get("num_parameters")})
    for key, name in ((lambda x: x["objective_equal_domain_mean"], "rank_objective"),
                      (lambda x: x["raw_union_kl_equal_domain_mean"],
                       "rank_raw_union_kl"),
                      (lambda x: x["root_teacher_topk_kl_equal_domain_mean"],
                       "rank_root_teacher_topk_kl")):
        for i, x in enumerate(sorted(rows, key=key)):
            x[name] = i + 1
    rows.sort(key=lambda x: (x["quality_order"] or 99, x["rank_objective"]))
    col = lambda k: [x[k] for x in rows]  # noqa: E731
    return {
        "n_complete_leaves": len(rows),
        "spearman": {
            "objective_vs_raw_union_kl": spearman(
                col("objective_equal_domain_mean"),
                col("raw_union_kl_equal_domain_mean")),
            "objective_vs_root_teacher_topk_kl": spearman(
                col("objective_equal_domain_mean"),
                col("root_teacher_topk_kl_equal_domain_mean"))},
        "top4": {
            "committed_quality_order": [x["state_id"] for x in rows if x["selected"]],
            "by_objective_equal_domain_mean": [
                x["state_id"] for x in sorted(
                    rows, key=lambda x: x["objective_equal_domain_mean"])[:4]],
            "by_raw_union_kl_equal_domain_mean": [
                x["state_id"] for x in sorted(
                    rows, key=lambda x: x["raw_union_kl_equal_domain_mean"])[:4]]},
        "leaves": rows}


def depth_choices(states: dict[str, dict]) -> list[dict]:
    out = []
    for r in states.values():
        step = next((s for s in r.get("steps") or []
                     if str(s.get("impl_id", "")).startswith("depth.")), None)
        if step is None or r.get("impl_ids", [None])[-1] != step.get("impl_id"):
            continue
        art = step.get("artifacts") or {}
        cands = art.get("objective_candidates") or {}
        rounds = art.get("search_rounds") or []
        agree = []
        for rd in rounds:
            scored = []
            for c in rd["table"]:
                key = ",".join(str(x) for x in sorted(rd["removed_before"]
                                                     + [c["candidate"]]))
                if key in cands:
                    scored.append((c["candidate"], cands[key]))
            if scored:
                raw_arg = min(scored, key=lambda t: t[1]["raw_kl"])[0]
                agree.append(raw_arg == rd["chosen"])
        round0 = {}
        if rounds:
            c0 = [(c["candidate"], cands.get(str(c["candidate"])))
                  for c in rounds[0]["table"]]
            c0 = [(c, x) for c, x in c0 if x]
            for field in ("objective", "raw_kl", "kl_reference_topk", "parent_kl"):
                have = [(c, x[field]) for c, x in c0 if x.get(field) is not None]
                if have:
                    round0[f"argmin_{field}"] = min(have, key=lambda t: t[1])[0]
            for field in ("raw_kl", "kl_reference_topk", "parent_kl"):
                if c0 and all(x.get(field) is not None for _, x in c0):
                    round0[f"spearman_objective_vs_{field}"] = spearman(
                        [x["objective"] for _, x in c0], [x[field] for _, x in c0])
        trace = step.get("trace") or {}
        pref = art.get("parent_reference_cache")
        out.append({
            "state_id": r["state_id"], "parent_id": r.get("parent_id"),
            "depth": r.get("depth"), "path": r.get("path_label"),
            "is_root_child": int(r.get("depth") or 0) == 1,
            "removed_layers": trace.get("removed_layers"),
            "removal_order": trace.get("removal_order"),
            "round0": round0,
            "rounds_objective_choice_is_raw_kl_argmin": sum(agree),
            "rounds": len(agree),
            "parent_kl_diagnostic": art.get("parent_kl_diagnostic"),
            "reference_cache_mode": (art.get("reference_cache") or {}).get("mode"),
            "parent_reference_cache_mode": pref.get("mode")
            if isinstance(pref, dict) else None,
            "reference_counters": art.get("reference_counters"),
            "operator_wall_seconds": step.get("wall_seconds")})
    return sorted(out, key=lambda x: (x["depth"] or 0, x["path"] or ""))


def d1_alignment(states: dict[str, dict], d1_states: Path) -> dict:
    """States D1 and D2 share by artifact digest, measured by both evaluators."""
    d1 = {r.get("artifact_digest"): r for r in last_records(d1_states).values()}
    rows = []
    for r in states.values():
        twin = d1.get(r.get("artifact_digest"))
        if twin is None:
            continue
        rows.append({"path": r.get("path_label"),
                     "artifact_digest": r.get("artifact_digest"),
                     "d1_teacher_kl": vals(twin).get(D1KL),
                     "d2_root_teacher_topk_kl": vals(r).get(TOPK),
                     "d2_raw_union_kl": vals(r).get(RAW),
                     "d2_objective": vals(r).get(OBJ)})
    return {"shared_states": len(rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--d1-states", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    states = last_records(a.dir / "states.jsonl")
    tel = jsonl(a.dir / "telemetry.jsonl")
    selection = json.loads(a.selection.read_text()) if a.selection else None
    leaves = leaf_ranking(states, selection)
    depth = depth_choices(states)
    summary = {"schema": "aadistill.phase_d2.search_analysis/v1",
               "inputs_sha256": {name: _sha256(a.dir / name) for name in (
                   "states.jsonl", "telemetry.jsonl")},
               "operator_timing": operator_timing(tel), "beam": beam(states),
               "leaf_spearman": leaves["spearman"], "top4": leaves["top4"]}
    if a.d1_states:
        summary["d1_metric_alignment"] = d1_alignment(states, a.d1_states)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for name, doc in (("complete_leaf_ranking.json", leaves),
                          ("depth_choices.json", depth),
                          ("search_analysis.json", summary)):
            (a.out_dir / name).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"leaves": leaves["n_complete_leaves"],
                      "depth_expansions": len(depth),
                      "expansions": summary["operator_timing"]["expansions"],
                      "top4": leaves["top4"], "spearman": leaves["spearman"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

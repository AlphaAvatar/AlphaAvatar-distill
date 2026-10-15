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

VERSION 2 (2026-10-15, erratum to the version-1 files committed at 7dd31165,
which are kept unchanged): the leaf and summary files are written as
`complete_leaf_ranking_v2.json` and `search_analysis_v2.json`.

* Version 1 labelled `state.teacher_kl.equal_domain_mean` of a D2 state as
  "d1_metric_teacher_kl". Under D2 v3 that field is the UNWEIGHTED UNION-support
  KL (it equals `state.objective.raw_kl`), not D1's historical teacher Top-K +
  tail KL. Version 2 names every KL by what it is and states each one's
  support, reference and teacher-logit numerical protocol (`METRICS`).
* Version 1's alignment table listed two rows that were two D2 STATE RECORDS of
  ONE artifact (the two root DEPTH expansions produced identical bytes). Version
  2 groups by artifact: records, unique artifacts and independent comparisons
  are counted separately.
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
ANALYSIS_VERSION = 2

#: What each KL in these files IS. Four quantities that look alike and are not:
#: a reader comparing two of them must compare these rows first.
METRICS = {
    "d2_objective": {
        "field": OBJ,
        "what": "teacher-confidence-weighted union KL (D2 v3's objective)",
        "support": "reference_student_union_topk_tail_v1: teacher Top-200 "
                   "UNION candidate Top-200, + one tail; deterministic "
                   "tie-break (value desc, index asc)",
        "reference": "the root (original) teacher",
        "weighting": "c_T(t) = 1 - H^{K+1}_T(t)/log(K+1), the frozen "
                     "teacher's Top-K + tail confidence; denominator 1/T_x",
        "teacher_logits": "root reference store: float32 product of the "
                          "model-dtype final hidden state and the output head, "
                          "accumulated in float32 (never rounded to bf16); "
                          "candidate-only entries retrieved from the stored "
                          "hidden state and log-normalizer",
        "prediction_rows": "scored_v1 (this search)"},
    "d2_unweighted_union_kl": {
        "field": RAW,
        "same_as": D1KL + " OF A D2 STATE (under v3 the evaluator's KL is the "
                   "union-support KL, so the two fields are equal)",
        "what": "the same union KL, UNWEIGHTED",
        "support": "as d2_objective", "reference": "the root teacher",
        "weighting": "none", "teacher_logits": "as d2_objective",
        "prediction_rows": "scored_v1 (this search)"},
    "d2_root_teacher_topk_kl": {
        "field": TOPK,
        "what": "teacher Top-K + tail KL of the same pair (D2's built-in "
                "KL-only control on D1's support family)",
        "support": "reference_topk_tail_v1 shape: teacher Top-200 + one tail, "
                   "with D2's deterministic tie-break",
        "reference": "the root teacher", "weighting": "none",
        "teacher_logits": "as d2_objective (float32 head product from the "
                          "root store)",
        "prediction_rows": "scored_v1 (this search)"},
    "d1_historical_teacher_topk_kl": {
        "field": D1KL + " OF A D1 STATE (D1's journal)",
        "what": "D1's state-evaluation KL",
        "support": "reference_topk_tail_v1: teacher Top-200 + one tail, "
                   "torch.topk (no deterministic tie-break)",
        "reference": "the root teacher at state evaluation",
        "weighting": "none",
        "teacher_logits": "the teacher's own forward: bf16 logits from the "
                          "model, upcast to float32 (teacher(ids).logits"
                          ".float())",
        "prediction_rows": "all_v1 (D1's search)"},
}


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
            #: v1 called this "d1_metric_teacher_kl". It is D2's own
            #: evaluator field, which under v3 is the UNION-support KL.
            "d2_state_teacher_kl_equal_domain_mean_is_union_kl": v.get(D1KL),
            "state_teacher_kl_equals_raw_union_kl": (
                v.get(D1KL) is not None and v.get(D1KL) == v.get(RAW)),
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
    """Artifacts D1 and D2 both measured, grouped BY ARTIFACT.

    One artifact measured by both evaluators is ONE comparison, however many
    state records name it; version 1 counted records.
    """
    d1_records = last_records(d1_states)
    d1_by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in d1_records.values():
        d1_by[r.get("artifact_digest")].append(r)
    d2_by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in states.values():
        if r.get("artifact_digest") in d1_by:
            d2_by[r.get("artifact_digest")].append(r)
    artifacts = []
    for digest, d2_rows in sorted(d2_by.items()):
        d1_rows = d1_by[digest]
        d1_vals = sorted({vals(t).get(D1KL) for t in d1_rows} - {None})
        d2_topk = sorted({vals(r).get(TOPK) for r in d2_rows} - {None})
        one = (len(d1_vals) == 1 and len(d2_topk) == 1)
        artifacts.append({
            "artifact_digest": digest,
            "d2_state_records": [{"state_id": r["state_id"],
                                  "path": r.get("path_label")}
                                 for r in d2_rows],
            "d1_state_records": [{"state_id": t["state_id"],
                                  "path": t.get("path_label")}
                                 for t in d1_rows],
            "d1_historical_teacher_topk_kl": d1_vals,
            "d2_root_teacher_topk_kl": d2_topk,
            "d2_unweighted_union_kl": sorted(
                {vals(r).get(RAW) for r in d2_rows} - {None}),
            "d2_objective": sorted({vals(r).get(OBJ) for r in d2_rows}
                                   - {None}),
            "abs_difference_root_topk_vs_d1": (
                abs(d2_topk[0] - d1_vals[0]) if one else None),
            "rel_difference_root_topk_vs_d1": (
                abs(d2_topk[0] - d1_vals[0]) / abs(d1_vals[0])
                if one and d1_vals[0] else None)})
    return {
        "n_d2_state_records": sum(len(a["d2_state_records"]) for a in artifacts),
        "n_d1_state_records": sum(len(a["d1_state_records"]) for a in artifacts),
        "n_unique_artifacts": len(artifacts),
        "independent_comparisons": len(artifacts),
        "artifacts": artifacts,
        "compares": ("d2_root_teacher_topk_kl against "
                     "d1_historical_teacher_topk_kl: the same support FAMILY "
                     "under different teacher-logit numerical protocols, "
                     "tie-breaks and prediction rows (see METRICS); an "
                     "agreement on one artifact is one comparison, not a "
                     "validation of the two evaluators in general")}


def cross_search_proxy(leaves: dict, d1_states: Path) -> dict:
    """Best complete leaf of each search, each on ITS OWN evaluator's KL."""
    d1_leaves = [r for r in last_records(d1_states).values()
                 if r.get("is_complete_leaf") and vals(r).get(D1KL) is not None]
    d1_best = min(d1_leaves, key=lambda r: vals(r)[D1KL])
    d2 = sorted(leaves["leaves"],
                key=lambda x: x["root_teacher_topk_kl_equal_domain_mean"])
    return {
        "d2_best_root_teacher_topk_kl": d2[0][
            "root_teacher_topk_kl_equal_domain_mean"],
        "d2_best_unweighted_union_kl": min(
            x["raw_union_kl_equal_domain_mean"] for x in leaves["leaves"]),
        "d1_best_historical_teacher_topk_kl": vals(d1_best)[D1KL],
        "d1_best_state_id": d1_best["state_id"],
        "comparable_as": ("an approximately comparable step-0 PROXY: same "
                          "support family, different teacher-logit numerical "
                          "protocol (float32 head product vs bf16 forward "
                          "logits upcast), tie-break and prediction rows; the "
                          "one shared artifact agrees to the difference shown "
                          "in d1_metric_alignment. Not behaviour, not a "
                          "D1-vs-D2 verdict"),
        "not_comparable": ("d2_unweighted_union_kl and the D2 objective: a "
                           "different support (and weighting) from D1's KL")}


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
    leaves["schema"] = "aadistill.phase_d2.complete_leaf_ranking/v2"
    leaves["metrics"] = METRICS
    summary = {"schema": "aadistill.phase_d2.search_analysis/v2",
               "metrics": METRICS,
               "inputs_sha256": {name: _sha256(a.dir / name) for name in (
                   "states.jsonl", "telemetry.jsonl")},
               "operator_timing": operator_timing(tel), "beam": beam(states),
               "leaf_spearman": leaves["spearman"], "top4": leaves["top4"]}
    if a.d1_states:
        summary["d1_metric_alignment"] = d1_alignment(states, a.d1_states)
        summary["cross_search_proxy"] = cross_search_proxy(leaves, a.d1_states)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        #: Version 2 writes NEW files; the version-1 files are frozen derived
        #: evidence and are corrected by the erratum, never overwritten.
        for name, doc in (("complete_leaf_ranking_v2.json", leaves),
                          ("depth_choices.json", depth),
                          ("search_analysis_v2.json", summary)):
            (a.out_dir / name).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"leaves": leaves["n_complete_leaves"],
                      "depth_expansions": len(depth),
                      "expansions": summary["operator_timing"]["expansions"],
                      "top4": leaves["top4"], "spearman": leaves["spearman"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The D3 probes' EXPLORATORY comparison with q2 and B, off-pod at `$0`.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/analyze_d3_probes.py \
        --run d3_probes_<utc> --store <scr>/store [--write]

What the registration froze before any D3 outcome existed, applied here:

1. every D3 unit is VERIFIED by the gate D2's units passed
   (`rank_d2_screening.verify_unit` under the shared CONFIRMATION rung): its
   identity, admission, rows and generations, a re-score by the pinned scorer
   reproducing every figure, its training records; its rows aligned id-for-id
   with q2's and B's imported rows at the same seed;
2. per comparator (q2 = the primary METHOD comparison, B = the anchor):
   per-seed and pooled correct_overall, paired per-sample counts, the
   two-sided 95% stratified prompt-cluster bootstrap interval of the
   prompt-mean seed-mean paired difference at the registered seed, the seed
   signs, and the registered DESCRIPTIVE label (BEHAVIOURAL_IMPROVEMENT /
   BEHAVIOURAL_REGRESSION / INCONCLUSIVE); usable rollout the same way,
   separately, never moving the label;
3. per stratum and per seed: usable and every component, correct_given_usable,
   natural termination, degeneration, context-limit hits, lengths.

EXPLORATORY: the battery and q2/B were observed before this design; no alpha,
no verdict contract, no SHARED_GO, no promotion; the mixed-domain battery does
not measure fluid reasoning (AGENTS.md P10.2).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d3.probes_exploratory_comparison/v1"
IMPROVEMENT = "BEHAVIOURAL_IMPROVEMENT"
REGRESSION = "BEHAVIOURAL_REGRESSION"
INCONCLUSIVE = "INCONCLUSIVE"
#: Two-sided 95%: the bootstrap's alpha is per tail.
ALPHA_PER_TAIL = 0.025
COMPONENTS = ("non_empty", "natural_termination", "no_severe_repetition",
              "no_context_limit", "protocol_valid")


class AnalysisRefused(RuntimeError):
    pass


def label(interval: tuple[float, float], per_seed: Sequence[float]) -> str:
    """The registered descriptive label. Pure; table-tested."""
    lo, hi = interval
    pos = sum(1 for d in per_seed if d > 0)
    neg = sum(1 for d in per_seed if d < 0)
    if lo > 0 and pos >= 2:
        return IMPROVEMENT
    if hi < 0 and neg >= 2:
        return REGRESSION
    return INCONCLUSIVE


def _by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out = {}
    for r in rows:
        if r["id"] in out:
            raise AnalysisRefused(f"duplicate prompt id {r['id']!r}")
        out[r["id"]] = r
    return out


def compare(candidate: Mapping[int, Sequence[Mapping[str, Any]]],
            reference: Mapping[int, Sequence[Mapping[str, Any]]], *,
            seeds: Sequence[int], bootstrap_seed: int,
            iterations: int) -> dict[str, Any]:
    """D3 - comparator: correctness (the label) and usable (descriptive).

    Correctness pairs the SCORABLE prompts (the shared protocol's estimand);
    usable pairs ALL prompts (its denominator is every row), stratified by
    set. Both are prompt-means of seed-mean paired differences with seeds as
    fixed blocks.
    """
    from stages.phase_c1.isolation import (
        paired_differences, stratified_cluster_bootstrap,
    )

    if sorted(candidate) != sorted(seeds) or sorted(reference) != sorted(seeds):
        raise AnalysisRefused(f"both arms need exactly the seeds {list(seeds)}")
    cand = {s: _by_id(candidate[s]) for s in seeds}
    ref = {s: _by_id(reference[s]) for s in seeds}
    ids = set(cand[seeds[0]])
    for s in seeds:
        if set(cand[s]) != ids or set(ref[s]) != ids:
            raise AnalysisRefused(f"seed {s}: the prompt sets differ")
    scorable = {j for j in ids if all(bool(cand[s][j]["scorable"])
                                      and bool(ref[s][j]["scorable"])
                                      for s in seeds)}
    strata = {j: str(cand[seeds[0]][j].get("set")) for j in ids}

    def block(metric: str, prompts: set[str]) -> dict[str, Any]:
        c = {s: {j: bool(cand[s][j][metric]) for j in prompts} for s in seeds}
        r = {s: {j: bool(ref[s][j][metric]) for j in prompts} for s in seeds}
        d = paired_differences(r, c)
        boot = stratified_cluster_bootstrap(
            d, {j: strata[j] for j in d}, iterations=iterations,
            seed=bootstrap_seed, alpha=ALPHA_PER_TAIL)
        per_seed = [sum(c[s][j] - r[s][j] for j in prompts) / len(prompts)
                    for s in seeds]
        return {"delta": boot["delta"],
                "ci_two_sided_95": [boot["lcb_one_sided"],
                                    boot["ucb_one_sided"]],
                "per_seed_delta": per_seed,
                "seeds_positive": sum(1 for x in per_seed if x > 0),
                "seeds_negative": sum(1 for x in per_seed if x < 0),
                "n_prompts": len(prompts),
                "candidate_mean": sum(sum(c[s].values()) for s in seeds)
                / (len(prompts) * len(seeds)),
                "comparator_mean": sum(sum(r[s].values()) for s in seeds)
                / (len(prompts) * len(seeds))}

    correct = block("correct", scorable)
    usable = block("usable", ids)
    paired = {}
    for s in seeds:
        counts = {"both": 0, "candidate_only": 0, "comparator_only": 0,
                  "neither": 0}
        for j in scorable:
            a, b = bool(cand[s][j]["correct"]), bool(ref[s][j]["correct"])
            counts["both" if a and b else "candidate_only" if a else
                   "comparator_only" if b else "neither"] += 1
        paired[str(s)] = counts
    return {"correct_overall": {
                **correct,
                "label": label(tuple(correct["ci_two_sided_95"]),
                               correct["per_seed_delta"])},
            "usable_rollout_descriptive": usable,
            "paired_per_seed": paired,
            "bootstrap": {"iterations": iterations, "seed": bootstrap_seed,
                          "stratified_by": "set",
                          "seeds": "fixed blocks, not resampled"},
            "_claim_boundary": (
                "prompt-distribution uncertainty CONDITIONAL ON the three "
                "seeds; descriptive, exploratory, no alpha")}


def profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One unit's behaviour, overall and per set."""
    def summarize(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        usable = [r for r in rs if r["usable"]]
        scorable = [r for r in rs if r["scorable"]]
        us = [r for r in usable if r["scorable"]]
        return {"n": n,
                "usable_rollout_rate": round(len(usable) / n, 4) if n else None,
                **{c: round(sum(bool(r[c]) for r in rs) / n, 4)
                   for c in COMPONENTS if n},
                "correct_overall": (round(sum(bool(r["correct"])
                                              for r in scorable)
                                          / len(scorable), 4)
                                    if scorable else None),
                "correct_given_usable": (round(sum(bool(r["correct"])
                                                   for r in us) / len(us), 4)
                                         if us else None)}
    sets = sorted({r.get("set") for r in rows})
    return {"overall": summarize(rows),
            "per_set": {s: summarize([r for r in rows if r.get("set") == s])
                        for s in sets}}


def analyze(candidate: Mapping[int, Sequence[Mapping[str, Any]]], *,
            repo_root: Path = REPO_ROOT,
            evidence_root: Path | None = None,
            registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Both comparisons and the per-seed profiles, from the candidate's rows
    and the imported, hash-checked q2/B rows."""
    from stages.d_series import shared_evaluation as SE
    from stages.phase_d3 import d3_design as D

    reg = registration or D.registration(repo_root)
    rung = reg["behavioural_rung"]
    rule = rung["exploratory_classification"]
    seeds = [int(s) for s in rung["seeds"]]
    out: dict[str, Any] = {"comparisons": {}, "profiles": {}}
    for arm in ("q2", "B"):
        rows = SE.load_reference_rows("shared_confirmation", arm,
                                      repo_root=repo_root,
                                      evidence_root=evidence_root)
        out["comparisons"][arm] = {
            "role": rung["comparators"][arm]["role"],
            **compare(candidate, rows, seeds=seeds,
                      bootstrap_seed=int(rule["bootstrap_seeds"][arm]),
                      iterations=int(rule["bootstrap_iterations"]))}
        out["profiles"][arm] = {str(s): profile(rows[s]) for s in seeds}
    out["profiles"]["d3"] = {str(s): profile(candidate[s]) for s in seeds}
    out["primary_label_vs_q2"] = out["comparisons"]["q2"]["correct_overall"][
        "label"]
    return out


def analyze_run(session: Mapping[str, Any], store: Path,
                repo_root: Path = REPO_ROOT,
                evidence_root: Path | None = None,
                rescore: Any = None) -> dict[str, Any]:
    """The real session: every unit verified before anything is compared."""
    from stages.d_series import shared_evaluation as SE
    from stages.phase_d2 import rank_d2_screening as R
    from stages.phase_d3 import d3_probes as P

    if session.get("status") != "COMPLETE":
        raise AnalysisRefused(f"the session status is {session.get('status')!r}")
    contract = P.session_contract(repo_root, verify_bytes=False)
    if session.get("contract_hash") != P.contract_hash(contract):
        raise AnalysisRefused("the session ran under a different contract")
    digest = contract["initialization"]["identities"]["artifact_digest"]
    reference = {arm: SE.load_reference_rows(
        "shared_confirmation", arm, repo_root=repo_root,
        evidence_root=evidence_root) for arm in ("q2", "B")}
    by_probe = {p["probe_id"]: p for p in session.get("probes", [])}
    problems, units, rows_by_seed = [], {}, {}
    for probe in contract["probes"]:
        pid, seed = probe["probe_id"], int(probe["seed"])
        if not by_probe.get(pid, {}).get("scored"):
            problems.append(f"{pid}: not scored")
            continue
        unit = R.verify_unit(
            store / "probes" / pid, by_probe[pid],
            {"probe_id": pid, "arm": probe["arm"], "seed": seed,
             "state_id": digest, "artifact_digest": digest},
            repo_root=repo_root, rescore=rescore,
            shared_rung="shared_confirmation",
            config_path=(store / "extracted/audit/autoinit_d3_probes/configs"
                         / f"{pid}.json"))
        for arm in ("q2", "B"):
            unit[f"aligned_with_{arm}"] = R.aligned(unit["rows"],
                                                    reference[arm][seed])
        if not (unit["verified"] and unit["aligned_with_q2"]
                and unit["aligned_with_B"]):
            problems.append(
                f"{pid}: {[k for k, v in unit['checks'].items() if v is not True]}")
        rows_by_seed[seed] = unit["rows"]
        units[pid] = {k: v for k, v in unit.items() if k != "rows"}
    if problems:
        raise AnalysisRefused("not comparable: " + "; ".join(problems))
    result = analyze(rows_by_seed, repo_root=repo_root,
                     evidence_root=evidence_root)
    return {"schema": SCHEMA, "run_id": session["run_id"],
            "contract_hash": session["contract_hash"],
            "initialization": contract["initialization"],
            "units_verified": units, **result,
            "_claim_boundary": (
                "EXPLORATORY on an observed battery: not a fresh independent "
                "confirmation, no alpha, no SHARED_GO, no promotion; the "
                "mixed-domain battery does not establish fluid-reasoning "
                "superiority. If it cannot separate the two DEPTH references, "
                "both are carried to a future capability-aligned evaluation.")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    run_dir = REPO_ROOT / "logs/stages/stage-1/phase_d3/runs" / args.run
    session = json.loads((run_dir / "evidence" / "d3_probes.json").read_text())
    out = analyze_run(session, Path(args.store))
    print(json.dumps({"primary_label_vs_q2": out["primary_label_vs_q2"],
                      "vs_q2": out["comparisons"]["q2"]["correct_overall"],
                      "vs_B": out["comparisons"]["B"]["correct_overall"]},
                     indent=1))
    if args.write:
        (run_dir / "evidence" / "d3_probes_comparison.json").write_text(
            json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

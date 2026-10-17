#!/usr/bin/env python3
"""Fix the D3 parent-reference comparability gate from GPU evidence, at $0.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/write_d3_admission.py [--write]

The treated DEPTH step records, for every round-0 candidate, the KL against
the intact PARENT beside the teacher-reference KL that decides. Round 0's
candidates are single layers of q2's own FFN parent under q2's own reduction,
so that diagnostic must reproduce q2's recorded round-0 scores. Maintainer
review 2026-10-17 (§2A) makes that agreement an execution-validity gate: a
run that fails it is INADMISSIBLE, never a fourth outcome.

The tolerance is fixed here, before any D3 execution, from GPU measurements
of the same quantity that already exist:

* SAME EXECUTION: q2's DEPTH step, run independently by two D1 search
  sessions (`d1_search_20261006_101725`, `..._210210`), agreed to the bit at
  all 260 candidates of all eight rounds. Expected difference: zero.
* ACROSS SESSIONS: the D1 and D2 searches both expanded DEPTH on the root,
  under each profile, at the same micro-batch and packing and host driver.
  D2's numbers are its round-0 parent-reference DIAGNOSTIC with a root store
  resident (the path D3 runs), against D1's decision scores (the path q2
  ran). Between them the code commit, the reference cache mode, the
  prediction-row policy and the store's residency all differed. Every
  candidate's difference is the benign variation the gate must admit.
* THE DRIVER: D3's host runs the shared protocol's 595.91.07 and q2's scores
  were computed on 580.126.09. No score-level DEPTH measurement exists on
  595.91.07; `d1_replay_002` there reproduced all four of q2's step
  identities, FFN's included -- the operator whose kept set moves under any
  perturbation of its reduction. That is digest-level evidence only, and the
  record says so: a driver effect beyond the tolerance would make the run
  inadmissible, never admit it.

`atol` and `rtol` are those pairs' maximum absolute and relative (to the D1
side) differences, each rounded UP at one significant figure. Every other
DEPTH round-0 table in both journals over the same candidates is then
checked against q2's: each must be REFUSED, or the gate cannot tell q2's
parent and reduction from another's.

Without `--write` it prints the record and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d3.parent_reference_admission/v1"
DEPTH_IMPL = "depth.causal_kl_greedy_v1"
#: The completed searches whose journals hold the measurements. Out of tree,
#: in the durable artifact store; the record keeps every value it uses.
D1_REPEAT_JOURNAL = ("logs/stages/stage-1/phase_d1/runs/d1_search_20261006_101725/"
                     "evidence/states_partial_oob_38states.jsonl")
D2_JOURNAL = ("/home/ecs-user/aad-artifacts/phase_d2/d2_search_20261015_051722/"
              "relay/states.jsonl")
#: The sessions whose host drivers the record states, read from their own
#: session records rather than typed.
SESSIONS = {
    "d1_search": "logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210",
    "d2_search": "logs/stages/stage-1/phase_d2/runs/d2_search_20261015_051722",
    "d1_replay": "logs/stages/stage-1/phase_d1_replay/runs/d1_replay_002"}


class AdmissionEvidenceError(RuntimeError):
    """The journals do not hold the measurements the gate is fixed from."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_states(path: Path) -> dict[str, dict[str, Any]]:
    """state id -> row, first occurrence; a truncated tail line is skipped."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.setdefault(row["state_id"], row)
    return out


def ceil_one_significant(x: float) -> float:
    """`x` rounded UP at one significant figure: 4.949e-05 -> 5e-05."""
    if not x > 0:
        raise AdmissionEvidenceError(f"cannot round {x!r}")
    d = Decimal(repr(x))
    exponent = d.adjusted()
    mantissa = (d.scaleb(-exponent)).to_integral_value(rounding=ROUND_CEILING)
    return float(mantissa.scaleb(exponent))


def _depth(row: dict[str, Any]) -> dict[str, Any] | None:
    step = (row.get("steps") or [{}])[-1]
    return step if step.get("impl_id") == DEPTH_IMPL else None


def round0_scores(row: dict[str, Any]) -> dict[int, float]:
    """A DEPTH state's round-0 decision scores, by candidate."""
    rounds = (_depth(row) or {}).get("artifacts", {}).get("search_rounds") or []
    if not rounds or rounds[0].get("removed_before"):
        raise AdmissionEvidenceError(f"{row['state_id']}: no round-0 table")
    return {int(e["candidate"]): float(e["score"]) for e in rounds[0]["table"]}


def round0_parent_diagnostic(row: dict[str, Any]) -> dict[int, float]:
    """A DEPTH state's round-0 PARENT-reference diagnostic, by candidate."""
    cands = (_depth(row) or {}).get("artifacts", {}).get(
        "objective_candidates") or {}
    out = {int(k): float(v["parent_kl"]) for k, v in cands.items()
           if "," not in k and "parent_kl" in v}
    if not out:
        raise AdmissionEvidenceError(f"{row['state_id']}: no round-0 "
                                     "parent-reference diagnostic")
    return out


def compare(base: dict[int, float], other: dict[int, float]) -> dict[str, Any]:
    if set(base) != set(other):
        raise AdmissionEvidenceError("the two tables test different candidates")
    rows = [{"candidate": c, "base": base[c], "other": other[c],
             "abs_difference": abs(other[c] - base[c]),
             "rel_difference": abs(other[c] - base[c]) / abs(base[c])}
            for c in sorted(base)]
    worst_abs = max(rows, key=lambda r: r["abs_difference"])
    worst_rel = max(rows, key=lambda r: r["rel_difference"])
    return {"n": len(rows),
            "n_bit_identical": sum(r["abs_difference"] == 0.0 for r in rows),
            "max_abs_difference": worst_abs["abs_difference"],
            "max_abs_at_candidate": worst_abs["candidate"],
            "max_rel_difference": worst_rel["rel_difference"],
            "max_rel_at_candidate": worst_rel["candidate"],
            "argmin_base": min(base, key=lambda c: (base[c], c)),
            "argmin_other": min(other, key=lambda c: (other[c], c)),
            "rows": rows}


def session_driver(repo_root: Path, run_dir: str) -> dict[str, Any]:
    session = json.loads((repo_root / run_dir / "runtime" / "session.json"
                          ).read_text())
    image = session.get("image_identity") or {}
    commit = (session.get("session_commit_check") or {}).get("session_commit")
    return {"run": run_dir.rsplit("/", 1)[1], "drivers": image.get("observed"),
            "image_identity": image.get("digest"), "session_commit": commit}


def _within(q2: dict[int, float], table: dict[int, float],
            atol: float, rtol: float) -> int:
    return sum(not abs(table[c] - q2[c]) <= atol + rtol * abs(q2[c])
               for c in q2)


def build(d1_journal: Path, d1_repeat: Path, d2_journal: Path,
          repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from aadistill.infrastructure.manifest import sha256_json
    from stages.phase_d3 import d3_design as D

    control_path = repo_root / D.CONTROL_RECORD
    control = json.loads(control_path.read_text())
    reg = D.registration(repo_root)
    leaf = D.control_leaf(repo_root)
    at = D.treated_index(leaf)
    treated_profile = leaf.steps[at].profile_id
    q2_depth_state = control["steps"][at]["state_id"]
    q2 = {int(e["candidate"]): float(e["score"])
          for e in control["depth"]["rounds"][0]["table"]}
    first = int(control["depth"]["removal_order"][0])

    d1, d1b, d2 = (load_states(d1_journal), load_states(d1_repeat),
                   load_states(d2_journal))
    #: The root, derived: the parent of q2's first step.
    root_state = d1[control["steps"][0]["state_id"]]["parent_id"]

    # --- same execution: q2's own DEPTH, twice ---------------------------------
    a, b = d1[q2_depth_state], d1b[q2_depth_state]
    ra = _depth(a)["artifacts"]["search_rounds"]
    rb = _depth(b)["artifacts"]["search_rounds"]
    per_round = []
    for x, y in zip(ra, rb):
        tx = {e["candidate"]: e["score"] for e in x["table"]}
        ty = {e["candidate"]: e["score"] for e in y["table"]}
        if set(tx) != set(ty):
            raise AdmissionEvidenceError(f"round {x['round']}: the repeat "
                                         "tested other candidates")
        per_round.append({"round": x["round"], "n": len(tx),
                          "max_abs_difference": max(abs(tx[k] - ty[k])
                                                    for k in tx),
                          "chosen": [x["chosen"], y["chosen"]]})
    if round0_scores(a) != q2:
        raise AdmissionEvidenceError("the journal's q2 round 0 is not the "
                                     "control record's")
    same_execution = {
        "state_id": q2_depth_state,
        "sessions": ["d1_search_20261006_101725", "d1_search_20261006_210210"],
        "independent_executions": {
            "wall_seconds": [round(_depth(b)["wall_seconds"], 1),
                             round(_depth(a)["wall_seconds"], 1)],
            "_reads": "different wall times: two executions, not one row "
                      "copied"},
        "rounds": len(per_round),
        "candidates": sum(r["n"] for r in per_round),
        "max_abs_difference": max(r["max_abs_difference"] for r in per_round),
        "per_round": per_round}

    # --- across sessions: the root's DEPTH, D1 decision vs D2 diagnostic --------
    pairs = []
    for sid1, r1 in sorted(d1.items()):
        if r1.get("parent_id") != root_state or _depth(r1) is None:
            continue
        profile = _depth(r1)["profile_id"]
        twins = [r2 for r2 in d2.values()
                 if r2.get("parent_id") == root_state and _depth(r2)
                 and _depth(r2)["profile_id"] == profile]
        if len(twins) != 1:
            raise AdmissionEvidenceError(f"{len(twins)} D2 root DEPTH states "
                                         f"under {profile}")
        r2 = twins[0]
        t1, t2 = _depth(r1).get("trace") or {}, _depth(r2).get("trace") or {}
        cmp = compare(round0_scores(r1), round0_parent_diagnostic(r2))
        pairs.append({
            "profile_id": profile, "parent_state_id": root_state,
            "d1": {"state_id": sid1, "value": "round-0 decision score "
                                              "(parent_state reference)",
                   "reference_cache_mode": _depth(r1)["artifacts"][
                       "reference_cache"]["mode"],
                   "micro_batch_size": t1.get("micro_batch_size"),
                   "calibration_batch_packing": t1.get(
                       "calibration_batch_packing"),
                   "prediction_rows": t1.get("prediction_rows", "all_v1")},
            "d2": {"state_id": r2["state_id"],
                   "value": "round-0 parent-reference diagnostic, root "
                            "reference store resident",
                   "parent_reference_cache_mode": _depth(r2)["artifacts"][
                       "parent_reference_cache"]["mode"],
                   "micro_batch_size": t2.get("micro_batch_size"),
                   "calibration_batch_packing": t2.get(
                       "calibration_batch_packing"),
                   "prediction_rows": t2.get("prediction_rows", "all_v1")},
            **cmp})
    if not pairs or any(p["d1"]["micro_batch_size"] != p["d2"]["micro_batch_size"]
                        or p["d1"]["calibration_batch_packing"]
                        != p["d2"]["calibration_batch_packing"] for p in pairs):
        raise AdmissionEvidenceError("no cross-session pair at one execution "
                                     "shape")
    drivers = {k: session_driver(repo_root, v) for k, v in SESSIONS.items()}
    if drivers["d1_search"]["drivers"] != drivers["d2_search"]["drivers"]:
        raise AdmissionEvidenceError("the cross-session pair changed driver; "
                                     "restate what the evidence covers")
    replay = json.loads((repo_root / SESSIONS["d1_replay"] / "closeout"
                         / "outcome.json").read_text())["result"]
    if replay["differ"] or not replay["q2"]["adopted"]:
        raise AdmissionEvidenceError("the replay did not reproduce q2")
    max_abs = max(p["max_abs_difference"] for p in pairs)
    max_rel = max(p["max_rel_difference"] for p in pairs)
    atol, rtol = ceil_one_significant(max_abs), ceil_one_significant(max_rel)
    headroom = max(r["abs_difference"] / (atol + rtol * abs(r["base"]))
                   for p in pairs for r in p["rows"])

    # --- discrimination: every OTHER round-0 table must be refused --------------
    others = []
    for label, states, read in (("d1", d1, round0_scores),
                                ("d2", d2, round0_parent_diagnostic)):
        for sid, row in sorted(states.items()):
            if _depth(row) is None or sid == q2_depth_state:
                continue
            try:
                table = read(row)
            except AdmissionEvidenceError:
                continue
            if set(table) != set(q2):
                continue
            others.append({"journal": label, "state_id": sid,
                           "parent_state_id": row.get("parent_id"),
                           "profile_id": _depth(row)["profile_id"],
                           "violations": _within(q2, table, atol, rtol),
                           "argmin": min(table, key=lambda c: (table[c], c))})
    accepted = [o for o in others if o["violations"] == 0]
    if accepted:
        raise AdmissionEvidenceError(
            f"the tolerance admits other parents' tables: {accepted}")
    q2_parent = d1[q2_depth_state]["parent_id"]
    closest = min(others, key=lambda o: o["violations"])
    margins = []
    for r in control["depth"]["rounds"]:
        s = sorted(e["score"] for e in r["table"])
        margins.append(s[1] - s[0])

    body = {
        "schema": SCHEMA,
        "experiment_id": D.EXPERIMENT_ID,
        "registration_hash": reg["registration_hash"],
        "control_record": D.CONTROL_RECORD,
        "control_record_sha256": _sha256(control_path),
        "declared_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "fixed_before_execution": True,
        "what_this_is": (
            "An EXECUTION-VALIDITY gate on the D3 structural session "
            "(maintainer review 2026-10-17, section 2A). It is not a "
            "scientific outcome and changes nothing in the treatment "
            "objective: a run that fails it is INADMISSIBLE -- its evidence "
            "is preserved, and it yields no structural conclusion and no "
            "probe proposal."),
        "checks": {
            "diagnostic_complete": (
                "the treated DEPTH step carries round 0, and a finite "
                "parent-reference KL recorded at round 0 for every one of "
                "its candidates"),
            "candidate_set": "round 0 tests exactly the control's round-0 "
                             "candidates, each once",
            "numerical": "for every candidate: |d3 - q2| <= atol + rtol * |q2|",
            "argmin": ("the parent-reference argmin (ties by candidate id) is "
                       "q2's first removed layer")},
        "expected": {"n_round0_candidates": len(q2),
                     "first_removed_layer": first,
                     "q2_round0_margin": margins[0],
                     "q2_smallest_round_margin": min(margins)},
        "tolerance": {
            "atol": atol, "rtol": rtol,
            "form": "|d3 - q2| <= atol + rtol * |q2|, per candidate",
            "rule": ("atol and rtol are the cross-session pairs' maximum "
                     "absolute and relative (to the D1 side) differences, "
                     "each rounded UP at one significant figure"),
            "from_max_abs_difference": max_abs,
            "from_max_rel_difference": max_rel,
            "worst_evidence_fraction_of_tolerance": headroom,
            "tolerance_at_q2_first_removed": atol + rtol * abs(q2[first])},
        "evidence": {
            "same_execution_repeat": same_execution,
            "cross_session": pairs,
            "host_drivers": {
                **drivers,
                "d1_replay_reproduced": {
                    "leaf": replay["q2"]["state_id"],
                    "identities": replay["identities_required"],
                    "steps": len(replay["q2"]["steps"]),
                    "differ": replay["differ"]},
                "_reads": (
                    "the cross-session pair shares q2's driver, so it does "
                    "not measure a driver change. D3 runs the protocol's "
                    "driver, where q2 was reproduced at every step identity, "
                    "FFN's included, with no score-level measurement. A "
                    "driver effect beyond the tolerance makes the run "
                    "INADMISSIBLE; it cannot make one admissible")},
            "_reads": (
                "Same execution reproduces to the bit, so the expected "
                "difference is zero. Across sessions -- another code commit, "
                "cache mode and prediction-row policy, with a root store "
                "resident -- the same quantity moved by at most the maxima "
                "above, with the argmin unchanged. D3 holds q2's micro-batch, "
                "packing and all-row prediction")},
        "discrimination": {
            "tables_checked": len(others),
            "tables_refused": len(others),
            "min_violations": closest["violations"],
            "closest": closest,
            "same_parent_other_profile": [
                o for o in others if o["parent_state_id"] == q2_parent],
            "_reads": (
                "every other DEPTH round-0 table over the same candidates in "
                "both journals is refused, including q2's own parent under "
                "the other profile, whose argmin is ALSO layer 26: the "
                "argmin check alone would not have caught it, the "
                "numerical check does")},
        "journals": {
            "d1": {"path": str(d1_journal), "sha256": _sha256(d1_journal)},
            "d1_repeat": {"path": str(d1_repeat), "sha256": _sha256(d1_repeat)},
            "d2": {"path": str(d2_journal), "sha256": _sha256(d2_journal)}},
        "on_failure": (
            "INADMISSIBLE: the driver stops after the DEPTH step, before WIDTH "
            "and ATTENTION; the closeout re-derives the verdict off-pod from "
            "the persisted step record; nothing is classified, nothing is "
            "proposed, nothing is retried to seek agreement"),
    }
    return {**body, "admission_hash": sha256_json(body)}


def main(argv: list[str] | None = None) -> int:
    from stages.phase_d1 import replay_specs as R
    from stages.phase_d3 import d3_design as D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d1-journal", default=R.DEFAULT_JOURNAL)
    ap.add_argument("--d1-repeat", default=str(REPO_ROOT / D1_REPEAT_JOURNAL))
    ap.add_argument("--d2-journal", default=D2_JOURNAL)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    record = build(Path(args.d1_journal), Path(args.d1_repeat),
                   Path(args.d2_journal))
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    if not math.isfinite(record["tolerance"]["atol"]):
        raise SystemExit("non-finite tolerance")
    if args.write:
        out = REPO_ROOT / D.ADMISSION
        out.write_text(text)
        print(f"wrote {out.relative_to(REPO_ROOT)}  "
              f"admission_hash={record['admission_hash'][:16]}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

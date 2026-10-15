#!/usr/bin/env python3
"""D2 screening's ranking and selection, computed OFF-POD at `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/rank_d2_screening.py \
        --run d2_screening_<utc> [--store <scr>] [--write]

Reads the secured evidence of a COMPLETE screening session and applies the
frozen rule (`d2_screening.rank`, i.e. the shared protocol's
`rank_screening` over D1's pooling, veto and selection functions):

1. every candidate unit is VERIFIED before it is ranked: its result's self-hash
   and per-sample sha256, its admission under the shared rung's evaluation
   protocol, its trained run's training identity equal to the protocol's
   recovery block on every key but the initialization, and its per-sample rows
   aligned id-for-id with B's imported rows at the same seed;
2. the RANKING: pooled seed-mean `correct_overall` delta vs B's imported rows,
   usable-rollout as a VETO ONLY, tie-break by the frozen quality position then
   state id, exactly one candidate advances or none;
3. DESCRIPTIVE ONLY, never read by the rule: per candidate and seed, the paired
   per-sample correctness against B (both / candidate-only / B-only / neither)
   and the text-level rollout profile; q2's rows likewise, labelled with their
   winner's curse.

A unit that fails (1) is not ranked and the script refuses: ranking a partial
or non-comparable field selects on who happened to pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages.phase_d2 import d2_screening as SC  # noqa: E402


class RankingRefused(RuntimeError):
    """The secured evidence does not support a ranking."""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def verify_unit(probe_dir: Path, session_probe: dict[str, Any], *,
                repo_root: Path = REPO_ROOT, config_path: Path | None = None,
                manifest_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    """One candidate unit, from its secured files. Never trusts the record."""
    from aadistill.infrastructure.manifest import sha256_json
    from stages.d_series import shared_evaluation as SE

    record = SC.protocol(repo_root)
    rung = record["rungs"][SC.SHARED_RUNG]
    result = json.loads((probe_dir / "result.json").read_text())
    body = {k: v for k, v in result.items() if k != "result_sha256"}
    rows_bytes = (probe_dir / "per_sample.jsonl").read_bytes()
    checks = {
        "result_self_hash": sha256_json(body) == result["result_sha256"],
        "per_sample_sha256": hashlib.sha256(rows_bytes).hexdigest()
        == result.get("per_sample_sha256"),
        "admitted_under_the_shared_rung": (
            (session_probe.get("generation_admission") or {}).get(
                "evaluation_protocol_hash") == rung["evaluation_protocol_hash"]),
        "result_is_the_session_record": (
            result["result_sha256"] == session_probe.get("result_sha256")),
    }
    manifest = next((m for m in (probe_dir / "training" / "run_manifest.json",
                                 *manifest_paths) if m.is_file()),
                    probe_dir / "training" / "run_manifest.json")
    training: dict[str, Any] = {}
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        ex = m["execution"]
        training = {
            "trainer_source_digest": ex["trainer_source"]["digest"],
            "training_runtime_digest": ex["runtime_digest"],
            "device_class": ",".join(m["hardware"]["cuda_devices"]),
            "teacher": f"{m['teacher']['model_id']}@{m['teacher']['revision']}",
            "data_digest": m["ladder"]["blocks_sha256"]}
        block = record["recovery"]
        checks["training_identity_is_the_protocols"] = all(
            training[k] == block[k] for k in training)
    else:
        checks["training_identity_is_the_protocols"] = False
    config = config_path or probe_dir / "config.json"
    if config.is_file():
        checks["recipe_digest_is_the_protocols"] = (
            SE.recipe_digest(json.loads(config.read_text()),
                             SE.allowed_overrides(repo_root))
            == record["recovery"]["recipe_digest"])
    return {"checks": checks, "training": training,
            "verified": all(checks.values()),
            "correct_overall": result["correct_overall"],
            "usable_rollout_rate": result["usable_rollout_rate"],
            "rows": _rows(probe_dir / "per_sample.jsonl")}


def aligned(candidate: list[dict], reference: list[dict]) -> bool:
    """Same prompt ids, same scorable flags: one row per prompt on each side."""
    ref = {r["id"]: r for r in reference}
    return (len(ref) == len(reference) == len(candidate)
            and all(c["id"] in ref and bool(c["scorable"])
                    == bool(ref[c["id"]]["scorable"]) for c in candidate))


def paired_counts(candidate: list[dict], reference: list[dict]) -> dict[str, int]:
    ref = {r["id"]: r for r in reference}
    out = {"both": 0, "candidate_only": 0, "reference_only": 0, "neither": 0}
    for c in candidate:
        r = ref[c["id"]]
        if not (c["scorable"] and r["scorable"]):
            continue
        key = ("both" if c["correct"] and r["correct"] else
               "candidate_only" if c["correct"] else
               "reference_only" if r["correct"] else "neither")
        out[key] += 1
    return out


def rank_run(session: dict[str, Any], store: Path,
             repo_root: Path = REPO_ROOT,
             evidence_root: Path | None = None) -> dict[str, Any]:
    from stages.d_series import shared_evaluation as SE

    if session.get("status") != "COMPLETE":
        raise RankingRefused(f"the session status is {session.get('status')!r};"
                             " only a complete rung is ranked")
    contract = SC.session_contract(repo_root, verify_bytes=False)
    if session.get("contract_hash") != SC.contract_hash(contract):
        raise RankingRefused("the session ran under a different contract")
    reference = {r: SE.load_reference_rows(SC.SHARED_RUNG, r,
                                           repo_root=repo_root,
                                           evidence_root=evidence_root)
                 for r in ("B", "q2")}
    by_probe = {p["probe_id"]: p for p in session["probes"]}
    units, scored, problems = {}, [], []
    for probe in contract["probes"]:
        pid = probe["probe_id"]
        if not by_probe.get(pid, {}).get("scored"):
            problems.append(f"{pid}: not scored")
            continue
        unit = verify_unit(
            store / "probes" / pid, by_probe[pid], repo_root=repo_root,
            config_path=(store / "extracted/audit/autoinit_d2_screening/"
                                 "configs" / f"{pid}.json"),
            manifest_paths=(store.parent / "products" / "trained_unscored"
                            / pid / "run_manifest.json",))
        unit["aligned_with_B"] = aligned(unit["rows"],
                                         reference["B"][probe["seed"]])
        if not (unit["verified"] and unit["aligned_with_B"]):
            problems.append(f"{pid}: {unit['checks']}, aligned "
                            f"{unit['aligned_with_B']}")
        unit["paired_vs_B"] = paired_counts(unit["rows"],
                                            reference["B"][probe["seed"]])
        unit["paired_vs_q2_descriptive"] = paired_counts(
            unit["rows"], reference["q2"][probe["seed"]])
        units[pid] = {k: v for k, v in unit.items() if k != "rows"}
        scored.append({"arm": probe["arm"], "seed": probe["seed"],
                       "correct_overall": unit["correct_overall"],
                       "usable_rollout_rate": unit["usable_rollout_rate"]})
    if problems:
        raise RankingRefused("the field is not rankable: " + "; ".join(problems))
    ranked = SC.rank(scored, repo_root)
    return {
        "schema": "aadistill.phase_d2.screening_ranking/v1",
        "run_id": session["run_id"],
        "contract_hash": session["contract_hash"],
        "units_verified": units,
        **ranked,
        "statistical_integrity": SC.rule(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--store", required=True,
                    help="the session's <scr>/store (holds probes/<id>/)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    run_dir = REPO_ROOT / "logs/stages/stage-1/phase_d2/runs" / args.run
    session = json.loads((run_dir / "evidence" / "d2_screening.json").read_text())
    out = rank_run(session, Path(args.store))
    if args.write:
        (run_dir / "evidence" / "d2_screening_ranking.json").write_text(
            json.dumps(out, indent=1) + "\n")
    print(json.dumps({"selection": out["selection"].get("arm")
                      or out["selection"].get("outcome"),
                      "ranked": [(r["arm"], r["delta_vs_b"], r["vetoed"])
                                 for r in out["ranking"]["ranked"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

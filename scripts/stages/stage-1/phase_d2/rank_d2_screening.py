#!/usr/bin/env python3
"""D2 screening's ranking and selection, computed OFF-POD at `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/rank_d2_screening.py \
        --run d2_screening_<utc> [--store <scr>] [--write]

Reads the secured evidence of a COMPLETE screening session and applies the
frozen rule (`d2_screening.rank`, i.e. the shared protocol's
`rank_screening` over D1's pooling, veto and selection functions):

1. every candidate unit is VERIFIED before it is ranked (`verify_unit`): the
   result's self-hash and that it is the session's record; its label, seed,
   arm and initialization digest equal to the FROZEN unit's; its admission
   record under the shared rung's evaluation protocol with the result's
   generation fingerprint; its rows and per-stratum generations against the
   result by the shared protocol's own checks; the pinned scorer re-run on the
   stored generations reproducing the rows byte-exactly and every figure of
   the result; its secured training records equal to the session's digests,
   bound to the result, completed, and showing the protocol's recovery block,
   recipe and the committed candidate at the frozen seed; and its rows aligned
   id-for-id with B's imported rows at the same seed;
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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() \
        else None


#: The fields a re-score legitimately writes differently from the original:
#: when and how it ran, where its files were, and the hash over all of that.
#: The generations are compared by content (sha256 and count), not path.
RESCORE_VOLATILE = ("created_utc", "command", "generations", "per_sample_path",
                    "trained_run", "result_sha256")


def rescore_matches(result: dict[str, Any], again: dict[str, Any] | None
                    ) -> bool:
    """Does the pinned scorer, re-run on the stored generations, produce THIS
    result: every count, rate, per-set and per-domain figure, the battery and
    scoring identities -- everything but the volatile fields above?"""
    if not again:
        return False
    keys = (set(result) | set(again)) - set(RESCORE_VOLATILE)
    if any(result.get(k) != again.get(k) for k in keys):
        return False
    content = lambda r: {n: (g.get("sha256"), g.get("n"))      # noqa: E731
                         for n, g in (r.get("generations") or {}).items()}
    return content(result) == content(again)


def verify_unit(probe_dir: Path, session_probe: dict[str, Any],
                expected: dict[str, Any], *, repo_root: Path = REPO_ROOT,
                config_path: Path | None = None,
                rescore: Any = None) -> dict[str, Any]:
    """One candidate unit, from its secured files. Never trusts the record.

    `expected` is the FROZEN unit, read from the contract: `probe_id`, `arm`,
    `seed`, and the arm's committed `state_id` and `artifact_digest`. A record
    whose self-hash holds but whose label, seed, arm, initialization, training
    or scores contradict the frozen unit -- or each other -- does not verify.
    The evidence checks and the re-score are the shared protocol's own
    (`scored_evidence_checks`, `rescore_unit` with the pinned scorer), so a
    candidate is verified by the same definitions B's imported units were.
    """
    from aadistill.infrastructure.manifest import sha256_json
    from stages.d_series import shared_evaluation as SE

    record = SC.protocol(repo_root)
    rung = record["rungs"][SC.SHARED_RUNG]
    block = record["recovery"]
    pid, seed, arm = expected["probe_id"], int(expected["seed"]), expected["arm"]
    result = _read_json(probe_dir / "result.json") or {}
    checks: dict[str, Any] = {}

    # -- the record: its own hash, and that it is the session's ---------------
    body = {k: v for k, v in result.items() if k != "result_sha256"}
    checks["result_self_hash"] = bool(result) and \
        sha256_json(body) == result.get("result_sha256")
    checks["result_is_the_session_record"] = bool(result) and \
        result.get("result_sha256") == session_probe.get("result_sha256")

    # -- WHICH unit: the result and the session entry against the frozen one --
    checks["label_is_the_frozen_probe"] = result.get("label") == pid
    checks["seed_is_the_frozen_seed"] = (
        type(result.get("seed")) is int and result["seed"] == seed)
    checks["arm_is_the_frozen_arm"] = result.get("screening_arm") == arm
    checks["initialization_is_the_committed_candidate"] = (
        result.get("initialization_artifact_digest")
        == expected["artifact_digest"])
    checks["session_entry_is_the_frozen_probe"] = (
        session_probe.get("probe_id") == pid
        and session_probe.get("arm") == arm
        and session_probe.get("seed") == seed
        and session_probe.get("state_id") == expected["state_id"])

    # -- admission: the session's, the secured record's, the result's ---------
    fingerprint = result.get("generation_protocol_fingerprint")
    admitted = session_probe.get("generation_admission") or {}
    checks["admitted_under_the_shared_rung"] = (
        admitted.get("evaluation_protocol_hash")
        == rung["evaluation_protocol_hash"])
    admission = _read_json(probe_dir / "admission.json") or {}
    checks["admission_record_is_this_probes"] = (
        admission.get("probe_id") == pid
        and admission.get("comparable") is True
        and admission.get("evaluation_protocol_hash")
        == rung["evaluation_protocol_hash"]
        and admission.get("attested_evaluation_protocol_hash")
        == rung["evaluation_protocol_hash"]
        and bool(fingerprint)
        and admission.get("generation_fingerprint") == fingerprint
        and admitted.get("generation_fingerprint") == fingerprint)

    # -- the evidence: rows and generations, by the shared definitions --------
    checks.update(SE.scored_evidence_checks(probe_dir, result))
    rows = (_rows(probe_dir / "per_sample.jsonl")
            if (probe_dir / "per_sample.jsonl").is_file() else [])
    checks["rows_carry_the_frozen_unit"] = bool(rows) and all(
        r.get("label") == pid and r.get("seed") == seed for r in rows)
    checks["rows_counts_are_the_results"] = bool(rows) and (
        len(rows) == result.get("n")
        and sum(1 for r in rows if r.get("scorable"))
        == result.get("n_scorable"))

    # -- the scores: the pinned scorer, re-run on the stored generations -----
    run = rescore or SE.rescore_unit
    again = run(SC.SHARED_RUNG, probe_dir / "generations", label=pid,
                seed=seed, screening_arm=arm, repo_root=repo_root,
                extra_args=("--init-digest", str(expected["artifact_digest"]),
                            "--generation-fingerprint", str(fingerprint)))
    checks["rescore_reproduces_per_sample"] = (
        again["rows_sha256"] is not None
        and again["rows_sha256"] == result.get("per_sample_sha256"))
    checks["rescore_reproduces_the_result"] = rescore_matches(result,
                                                             again["result"])

    # -- training: the secured records, bound to the session and the result --
    training_dir = probe_dir / "training"
    manifest_path = training_dir / "run_manifest.json"
    completion_path = training_dir / "run_completion.json"
    recorded = session_probe.get("training_records_sha256") or {}
    checks["training_records_are_the_sessions"] = all(
        recorded.get(name) and _sha256(training_dir / name) == recorded[name]
        for name in SC.TRAINING_RECORDS)
    manifest = _read_json(manifest_path) or {}
    completion = _read_json(completion_path) or {}
    trained_run = result.get("trained_run") or {}
    checks["completion_is_the_results"] = bool(completion) and (
        _sha256(completion_path) == trained_run.get("run_completion_sha256"))
    checks["training_completed_every_step"] = (
        completion.get("completed_all_steps") is True
        and completion.get("final_step") is not None
        and completion.get("final_step")
        == completion.get("planned_total_steps"))
    config = manifest.get("config") or {}
    checks["config_is_bound_end_to_end"] = bool(config) and (
        sha256_json(config) == manifest.get("config_sha256")
        == completion.get("config_sha256") == trained_run.get("config_sha256"))
    training: dict[str, Any] = {}
    try:
        ex = manifest["execution"]
        training = {
            "trainer_source_digest": ex["trainer_source"]["digest"],
            "training_runtime_digest": ex["runtime_digest"],
            "training_image_digest": ex["runtime"]["image_digest"],
            "device_class": ",".join(manifest["hardware"]["cuda_devices"]),
            "teacher": (f"{manifest['teacher']['model_id']}@"
                        f"{manifest['teacher']['revision']}"),
            "data_digest": manifest["ladder"]["blocks_sha256"]}
        checks["training_identity_is_the_protocols"] = all(
            training[k] == block[k] for k in training)
    except (KeyError, TypeError):
        checks["training_identity_is_the_protocols"] = False
    checks["recipe_digest_is_the_protocols"] = bool(config) and (
        SE.recipe_digest(config, SE.allowed_overrides(repo_root))
        == block["recipe_digest"])
    student = str(config.get("student_path") or "")
    checks["trained_from_the_candidate_at_the_seed"] = bool(config) and (
        config.get("seed") == seed and config.get("run_name") == pid
        and Path(student).name == expected["state_id"]
        and manifest.get("student_source") == student)
    if config_path is not None and config_path.is_file():
        checks["config_file_is_the_manifests"] = (
            _read_json(config_path) == config)
    return {"checks": checks, "training": training,
            "verified": all(v is True for v in checks.values()),
            "correct_overall": result.get("correct_overall"),
            "usable_rollout_rate": result.get("usable_rollout_rate"),
            "rows": rows}


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
    arms = {a["arm"]: a for a in contract["field"]["arms"]}
    units, scored, problems = {}, [], []
    #: The session verified every arm's bytes at all four identities before
    #: anything trained; a session record without that is not this rung's.
    verified_arms = {a.get("state_id") for a in
                     (session.get("arm_verification") or {}).get("arms", [])}
    if verified_arms != {a["state_id"] for a in arms.values()}:
        problems.append(f"the session's arm verification covers "
                        f"{sorted(verified_arms)}, not the committed field")
    for probe in contract["probes"]:
        pid = probe["probe_id"]
        if not by_probe.get(pid, {}).get("scored"):
            problems.append(f"{pid}: not scored")
            continue
        arm = arms[probe["arm"]]
        unit = verify_unit(
            store / "probes" / pid, by_probe[pid],
            {"probe_id": pid, "arm": probe["arm"], "seed": probe["seed"],
             "state_id": arm["state_id"],
             "artifact_digest": arm["identities"]["artifact_digest"]},
            repo_root=repo_root,
            config_path=(store / "extracted/audit/autoinit_d2_screening/"
                                 "configs" / f"{pid}.json"))
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

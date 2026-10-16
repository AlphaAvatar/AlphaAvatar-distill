#!/usr/bin/env python3
"""Close out ONE D2 screening session, off-pod, at `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/closeout_d2_screening.py \
        --run d2_screening_<utc> --scr <scr> [--rung confirmation] [--write]

ONE CLOSEOUT, TWO RUNGS: the screening records its frozen RANKING, the shared
confirmation its frozen VERDICT (`verdict_d2_confirmation.verdict_run`); the
rest is the same.

What a finished screening session owes the repository, in one command:

1. the SMALL evidence copied into the run directory -- the driver's session
   record, the attested protocol and engine probe, and per probe the result,
   admission and training records (run manifest, completion record, train
   log). The per-sample rows, raw generations, per-sample rescoring rows and
   preserved checkpoints stay in the durable store `<scr>/store`, named here
   with their sha256;
2. the RANKING, by `rank_d2_screening.rank_run` -- the frozen rule against B's
   imported units, refusing a field any unit of which fails verification;
3. the DIAGNOSTICS, descriptive only and never read by the rule: per probe the
   token-id sidecar's alignment and the rescoring's completeness and its mean
   position-level quantities;
4. the MONEY: the runner's GPU figure plus the container disk model, ceiled to
   the 4-decimal quantum, with the provider balance as a cross-check -- and
   `provider_teardown_verified` only when the provider confirms the pod gone.

Without `--write` it prints what it would record and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_REL = "logs/stages/stage-1/phase_d2/runs"
AUDIT = "autoinit_d2_screening"


def _names(rung: str) -> dict[str, str]:
    return {"audit": f"autoinit_d2_{rung}",
            "record": f"d2_{rung}.json",
            "attested": f"d2_{rung}_attested_protocol.json",
            "rule_file": ("d2_screening_ranking.json" if rung == "screening"
                          else "d2_confirmation_verdict.json"),
            "diagnostics": f"d2_{rung}_diagnostics.json",
            "schema": f"aadistill.phase_d2.{rung}_closeout/v1",
            "shared_rung": ("shared_screening" if rung == "screening"
                            else "shared_confirmation")}
DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730.0


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() \
        else None


def _session_record(scr: Path, rung: str = "screening") -> dict[str, Any]:
    """The driver's own record: the closeout fetch first, the live mirror
    second."""
    name = _names(rung)["record"]
    for path in (scr / "store" / name, scr / "relay" / name):
        if path.is_file():
            return json.loads(path.read_text())
    raise SystemExit(f"no driver record under {scr}")


def evidence_plan(run_dir: Path, scr: Path, record: dict[str, Any],
                  rung: str = "screening") -> list[tuple[Path, Path]]:
    """(source, destination) for every small file the run directory keeps."""
    store = scr / "store"
    ev = run_dir / "evidence"
    plan = [(store / name, ev / name) for name in (
        _names(rung)["attested"], "engine_probe.json")]
    for probe in record.get("probes", []):
        pid = probe["probe_id"]
        src = store / "probes" / pid
        plan += [(src / "result.json", ev / "results" / f"{pid}.json"),
                 (src / "admission.json", ev / "admissions" / f"{pid}.json"),
                 (store / "extracted" / "audit" / _names(rung)["audit"]
                  / "configs"
                  / f"{pid}.json", ev / "configs" / f"{pid}.json")]
        plan += [(src / "training" / name, ev / "training" / pid / name)
                 for name in ("run_manifest.json", "run_completion.json",
                              "train_log.jsonl")]
    plan += [(scr / "launch.log", run_dir / "runtime" / "launcher.log")]
    plan += [(p, run_dir / "runtime" / p.name)
             for p in sorted(scr.glob("watchdog_*.jsonl"))]
    plan += [(p, ev / "provider" / p.name)
             for p in sorted(scr.glob("create_raw_*.txt"))
             if '"id"' in p.read_text(errors="replace")]
    return plan


def durable_store(scr: Path, record: dict[str, Any]) -> dict[str, Any]:
    """What stays outside git, and how a reader checks it."""
    out = {}
    for probe in record.get("probes", []):
        pid = probe["probe_id"]
        src = scr / "store" / "probes" / pid
        result = json.loads((src / "result.json").read_text()) \
            if (src / "result.json").is_file() else {}
        out[pid] = {
            "per_sample_sha256": _sha(src / "per_sample.jsonl"),
            "per_sample_sha256_in_result": result.get("per_sample_sha256"),
            "generations": {name: g.get("sha256") for name, g in
                            sorted((result.get("generations") or {}).items())},
            "generations_present": sorted(
                p.name for p in (src / "generations").glob("*.jsonl")),
            "rescoring_sha256": _sha(src / "rescoring.json"),
            "trained_sha256": probe.get("trained_sha256"),
            "trained_bytes": probe.get("trained_bytes")}
    return {"path": str(scr), "probes": out}


def rollout_profile(gen_dir: Path) -> dict[str, Any]:
    """Text-level rollout behaviour of one unit, from its stored generations.

    DESCRIPTIVE ONLY: length, natural termination, context-limit hits and
    degeneration, as the generator recorded them per sample (P18)."""
    rows = []
    for path in sorted(gen_dir.glob("*.generations.jsonl")):
        rows += [json.loads(x) for x in path.read_text().splitlines()
                 if x.strip()]
    if not rows:
        return {"n": 0}
    toks = sorted(int(r.get("generated_tokens") or 0) for r in rows)
    rate = lambda key: round(sum(1 for r in rows if r.get(key))  # noqa: E731
                             / len(rows), 4)
    return {"n": len(rows),
            "generated_tokens_mean": round(statistics.fmean(toks), 1),
            "generated_tokens_median": statistics.median(toks),
            "generated_tokens_p90": toks[int(0.9 * (len(toks) - 1))],
            "natural_termination_rate": rate("natural_termination"),
            "context_limit_rate": rate("context_limit_reached"),
            "degeneration_rate": rate("degeneration_triggered")}


def reference_profiles(repo_root: Path,
                       shared_rung: str = "shared_screening") -> dict[str, Any]:
    """B's and q2's imported screening units, from D1's durable store."""
    from stages.d_series import shared_evaluation as SE

    store = SE.durable_store(shared_rung, repo_root) / "store" / "probes"
    out = {}
    for unit in SE.historical_units(repo_root):
        if unit.rung == shared_rung:
            out[unit.probe_id] = {"arm": unit.arm, "seed": unit.seed,
                                  **rollout_profile(store / unit.probe_id
                                                    / "generations")}
    return out


def diagnostics(scr: Path, record: dict[str, Any],
                rung: str = "screening") -> dict[str, Any]:
    """Descriptive only: alignment and rescoring, per probe."""
    probes = {}
    for probe in record.get("probes", []):
        pid = probe["probe_id"]
        path = scr / "store" / "probes" / pid / "rescoring.json"
        row: dict[str, Any] = {
            "generation_via": (probe.get("generation") or {}).get("via"),
            "token_id_sidecar": probe.get("token_id_sidecar"),
            "rescoring_run": probe.get("rescoring")}
        if path.is_file():
            doc = json.loads(path.read_text())
            means = {}
            for key in ("student_confidence", "student_entropy",
                        "teacher_surprisal", "kl_teacher_student",
                        "student_topk_mass"):
                vals = [r["summary"]["all"][key] for r in doc.get("rows", [])
                        if (r.get("summary") or {}).get("all", {}).get(key)
                        is not None]
                means[key] = round(statistics.fmean(vals), 6) if vals else None
            row.update({
                "n_samples": doc.get("n_samples"),
                "n_rescored": doc.get("n_rescored"),
                "n_exact_engine_ids": doc.get("n_exact_engine_ids"),
                "n_retokenized_approximation":
                    doc.get("n_retokenized_approximation"),
                "complete": doc.get("complete"),
                "failures": len(doc.get("failures") or []),
                "mean_over_samples_of_position_means": means,
                "thresholds_from": doc.get("thresholds_from")})
        probes[pid] = row
    for probe in record.get("probes", []):
        probes[probe["probe_id"]]["rollout_profile"] = rollout_profile(
            scr / "store" / "probes" / probe["probe_id"] / "generations")
    return {"schema": "aadistill.phase_d2.screening_diagnostics/v1",
            "reference_rollout_profiles": reference_profiles(
                REPO_ROOT, _names(rung)["shared_rung"]),
            "_role": ("DESCRIPTIVE ONLY. Never read by the screening rule, "
                      "never a guardrail: the token-id sidecar's alignment "
                      "and the rescoring of each candidate's own rollouts "
                      "against the pinned teacher."),
            "probes": probes}


def money(session: dict[str, Any], *, disk_gb: int,
          balance_after: float | None) -> dict[str, Any]:
    cost = session.get("cost") or {}
    gpu = float(cost.get("actual_usd") or 0.0)
    minutes = float(cost.get("elapsed_minutes") or 0.0)
    disk = disk_gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH * minutes / 60.0
    before = (session.get("account_balance") or {}).get("client_balance_usd")
    return {"all_in_usd": math.ceil((gpu + disk) * 1e4) / 1e4,
            "gpu_usd": round(gpu, 4), "disk_usd": round(disk, 4),
            "elapsed_minutes": minutes,
            "_basis": (f"the runner's GPU figure plus {disk_gb} GB container "
                       "disk at $0.10/GB-month over the session's elapsed "
                       "minutes, ceiled to the 4-decimal quantum"),
            "provider_balance_cross_check": {
                "before_usd": before, "after_usd": balance_after,
                "measured_delta_usd": (round(before - balance_after, 4)
                                       if before is not None
                                       and balance_after is not None
                                       else None)}}


def build(run: str, scr: Path, *, balance_after: float | None,
          rung: str = "screening",
          disk_gb: int) -> tuple[dict[str, Any], list[tuple[Path, Path]],
                                 dict[str, Any], dict[str, Any] | None,
                                 str | None]:
    from stages.phase_d2 import rank_d2_screening as R
    from stages.phase_d2 import verdict_d2_confirmation as V

    names = _names(rung)

    run_dir = REPO_ROOT / RUNS_REL / run
    session = json.loads((run_dir / "runtime" / "session.json").read_text())
    auth = json.loads((run_dir / "governance" / "authorization.json"
                       ).read_text())
    record = _session_record(scr, rung)
    plan = evidence_plan(run_dir, scr, record, rung)
    diag = diagnostics(scr, record, rung)
    ranking, refused = None, None
    try:
        ranking = (R.rank_run(record, scr / "store", REPO_ROOT)
                   if rung == "screening" else
                   V.verdict_run(record, scr / "store", REPO_ROOT))
    except (R.RankingRefused, V.VerdictRefused) as exc:
        refused = str(exc)
    gone = bool(session.get("provider_confirms_gone"))
    state = session.get("final_pod_state") or {}
    complete = (record.get("status") == "COMPLETE" and ranking is not None)
    doc = {
        "schema": names["schema"],
        "run_id": run, "stage_id": "1", "experiment_id": "phase_d2",
        "classification": (
            f"COMPLETE {rung.upper()} RUNG: every frozen probe trained, "
            "generated, scored and verified off-pod; the frozen rule applied"
            if complete else
            f"INCOMPLETE: driver status {record.get('status')!r}; "
            f"ranking {'refused: ' + str(refused) if refused else 'absent'}"),
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "session_commit": (session.get("session_commit_check") or {}).get(
            "session_commit"),
        "authorized_base_commit": auth["authorized_session_commit"],
        "authorization_sha256": auth["authorization_sha256"],
        "authorization_file_sha256": _sha(run_dir / "governance"
                                          / "authorization.json"),
        "contract_hash": auth["contract_hash"],
        "driver_status": record.get("status"),
        "runner_terminal": session.get("terminal"),
        "provider_resource_created": bool(
            session.get("provider_resource_created")),
        "provider_teardown_verified": bool(
            gone and state.get("exists") is False
            and state.get("billing") is False),
        "teardown_evidence": {"provider_confirms_gone": gone,
                              "final_pod_state": state,
                              "teardown_gate": session.get("teardown_gate")},
        "pod": {"pod_id": session.get("pod_id"),
                "image_identity": session.get("image_identity"),
                "draws": session.get("draws")},
        "runtime_comparability": {
            "training_runtime": record.get("training_runtime"),
            "attested_evaluation_protocol_hash":
                record.get("attested_evaluation_protocol_hash"),
            "attested_matches_the_shared_rung":
                record.get("attested_matches_the_shared_rung"),
            "scoring_contract": record.get("scoring_contract"),
            "arm_verification": record.get("arm_verification")},
        "money": money(session, disk_gb=disk_gb,
                       balance_after=balance_after),
        "durable_store": durable_store(scr, record),
        "required_products_secured": session.get("required_products_secured"),
        "artifact_loss_risk": session.get("artifact_loss_risk"),
        "ranking": (f"evidence/{names['rule_file']}" if ranking else None),
        "ranking_refused": refused,
        **({"selection": (ranking or {}).get("selection")}
           if rung == "screening" else
           {"outcome": (ranking or {}).get("outcome"),
            "decision": ((ranking or {}).get("verdict") or {}).get(
                "primary_vs_B", {}).get("decision")}),
        "diagnostics": f"evidence/{names['diagnostics']}",
        "what_this_authorizes": (
            "nothing. Screening nominates at most one candidate for the "
            "shared confirmation, which needs its own maintainer grant; it "
            "never promotes."),
    }
    return doc, plan, diag, ranking, refused


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--balance-after", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--rung", default="screening",
                    choices=("screening", "confirmation"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    scr = Path(args.scr)
    doc, plan, diag, ranking, refused = build(
        args.run, scr, balance_after=args.balance_after, disk_gb=args.disk_gb,
        rung=args.rung)
    names = _names(args.rung)
    run_dir = REPO_ROOT / RUNS_REL / args.run
    present = [(s, d) for s, d in plan if s.is_file()]
    print(json.dumps({"copy": len(present), "missing": [
        str(s) for s, _ in plan if not s.is_file()][:12],
        "selection": doc.get("selection") or doc.get("outcome"),
        "ranking_refused": refused,
        "money": doc["money"],
        "provider_teardown_verified": doc["provider_teardown_verified"]},
        indent=1, default=str))
    if not args.write:
        return 0
    for src, dest in present:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    record = _session_record(scr, args.rung)
    (run_dir / "evidence" / names["record"]).write_text(
        json.dumps(record, indent=2, default=str) + "\n")
    (run_dir / "evidence" / names["diagnostics"]).write_text(
        json.dumps(diag, indent=1) + "\n")
    if ranking is not None:
        (run_dir / "evidence" / names["rule_file"]).write_text(
            json.dumps(ranking, indent=1) + "\n")
    (run_dir / "closeout").mkdir(parents=True, exist_ok=True)
    (run_dir / "closeout" / "outcome.json").write_text(
        json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {run_dir.relative_to(REPO_ROOT)}/closeout/outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

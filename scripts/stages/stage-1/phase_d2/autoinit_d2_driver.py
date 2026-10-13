#!/usr/bin/env python3
"""The D2 formal SEARCH driver. D1's four stages, D2's objective, evidence always.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d2/autoinit_d2_driver.py \
        --out DIR --run-id ID --authorization PATH [--deadline-s N] \
        [--prediction-rows all_v1|scored_v1]

    A  setup: registries, the session, and its CONTRACT against the
       preregistration -- before the teacher is resident;
    B  the root teacher once; the evaluator primed with that object;
    C  the beam search, minimizing the D2 objective, under the wall clock;
    D  commit the preregistered Top-K by the objective ranking, and record the
       KL-only control: D1's ranking of the SAME complete leaves on the raw KL
       every state carries. Reported, never used.

The generic machinery -- journal, markers, environment probe, teacher staging
-- is D1's driver's, imported unchanged; D1's files are not edited.
AUTHORIZES NOTHING: a paid run needs `--authorization`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages.phase_d1.autoinit_d1_driver import (  # noqa: E402
    CHECK_ONLY_MARKER, FAILURE_MARKER, SUCCESS_MARKER, Journal,
    _load_root_teacher, mark, probe_environment, say, terminal_marker,
)

__all__ = ["CHECK_ONLY_MARKER", "FAILURE_MARKER", "SUCCESS_MARKER", "main"]


class D2DriverError(RuntimeError):
    pass


def load_authorization(path: str, *, contract: dict[str, Any],
                       run_id: str, prediction_rows: str) -> dict[str, Any]:
    """The typed grant, bound to THIS session's identities, or refused."""
    from stages.phase_d2.d2_authorization import D2Authorization

    auth = D2Authorization.load(path)
    auth.require_plan(contract["preregistration_hash"])
    auth.require_run_id(run_id)
    for key in ("measurement_protocol_id", "config_hash", "objective_hash",
                "ranking_policy_hash"):
        if getattr(auth, key) != contract.get(key):
            raise D2DriverError(
                f"the authorization binds {key} {str(getattr(auth, key))[:12]} "
                f"and this session is {str(contract.get(key))[:12]}; the grant "
                "priced another run")
    if auth.prediction_rows != prediction_rows:
        raise D2DriverError(
            f"the authorization binds prediction_rows {auth.prediction_rows!r} "
            f"and this session runs {prediction_rows!r}")
    if auth.allows_recovery or auth.allows_behavioural:
        raise D2DriverError("a D2 search grant may not claim recovery or "
                            "behavioural permission")
    return {"authorization_id": auth.authorization_id,
            "hard_cap_usd": auth.hard_cap_usd,
            "authorized_session_commit": auth.authorized_session_commit,
            "preregistration_hash": auth.preregistration_hash,
            "config_hash": auth.config_hash,
            "pricing": dict(auth.pricing)}


def cache_regression(telemetry_path: Path) -> dict[str, Any] | None:
    """The first DEPTH expansion whose reference cache was NOT cached, or None."""
    if not telemetry_path.is_file():
        return None
    for line in telemetry_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not str(row.get("impl_id", "")).startswith("depth."):
            continue
        mode = (row.get("reference_cache") or {}).get("mode")
        if mode != "cached":
            return {"state_id": row.get("state_id"), "mode": mode,
                    "operator_seconds": row.get("operator_seconds")}
    return None


def guarded_deadline(seconds: float, *, started: float, telemetry_path: Path,
                     guard_cache: bool):
    """The soft stop, plus the reference-cache regression guard.

    A search priced on the CACHED DEPTH cell (the allocator-hoard fix validated
    on one root profile) must not keep running on that cheap expectation once
    a DEPTH expansion recomputes: the price no longer holds. The guard reads
    the search's own telemetry (at most every 30 s) and, on the first
    regression, expires the deadline -- the search fails closed with every
    record written, and the regression is in the evidence. A search priced on
    D1's recompute-mode maxima needs no guard: its price already assumes it.
    """
    from dataclasses import dataclass, field

    from aadistill.initialization.planning.search import Deadline

    @dataclass
    class CacheGuardDeadline(Deadline):
        telemetry_path: Path = field(default=Path("/nonexistent"))
        guard_cache: bool = False
        regression: dict | None = None
        _checked_at: float = 0.0

        def remaining(self) -> float:
            base = super().remaining()
            if not self.guard_cache or self.regression is not None:
                return 0.0 if self.regression is not None else base
            now = time.monotonic()
            if now - self._checked_at >= 30.0:
                self._checked_at = now
                self.regression = cache_regression(self.telemetry_path)
                if self.regression is not None:
                    self.fired_at = (
                        "reference-cache regression at "
                        f"{self.regression['state_id']} "
                        f"(mode {self.regression['mode']}): the cached-path "
                        "price no longer holds")
                    return 0.0
            return base

    return CacheGuardDeadline(seconds=float(seconds), started=started,
                              telemetry_path=telemetry_path,
                              guard_cache=guard_cache)


def _run_search(session, teacher, deadline) -> tuple[dict[str, Any], Any]:
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.planning.search import BeamSearch
    from aadistill.initialization.specs.arch import get_adapter
    from stages.phase_d1 import d1_session as D1S

    root = D1S.root_teacher_identity()
    search = BeamSearch(
        adapter=get_adapter(session.config.target_spec.family),
        config=session.config,
        root_teacher_id=root["root_teacher_id"],
        root_teacher_sha256=root["root_teacher_sha256"],
        root_loader=lambda: teacher,
        calibration_loader=lambda p: prepare_calibration_items(
            p.resolve(REPO), profile_id=p.qualified_id),
        measurer=session.measurer(),
        execution=session.evaluator.execution,
        numerics=D1S.numerics(),
        deadline=deadline)
    result = search.run()
    return {"n_states": len(result.states), "n_leaves": len(result.leaves),
            "n_complete_leaves": len(result.complete_leaves),
            "resumed": list(result.resumed),
            "finished_utc": result.finished_utc,
            "config_hash": session.config.config_hash,
            "root_teacher_id": root["root_teacher_id"]}, result


def objective_telemetry(telemetry_path: Path) -> dict[str, Any]:
    """What the objective cost, from the search's own per-expansion rows.

    The cost model's measured input for D3: DEPTH's `objective_seconds` and
    every expansion's state-evaluation seconds, beside the cache mode that
    decides DEPTH's wall clock.
    """
    if not telemetry_path.is_file():
        return {"available": False}
    rows = [json.loads(l) for l in telemetry_path.read_text().splitlines()
            if l.strip()]
    depth = [r for r in rows if str(r.get("impl_id", "")).startswith("depth.")]
    return {
        "available": True, "expansions": len(rows),
        "depth_expansions": len(depth),
        "depth_operator_seconds": round(sum(float(r.get("operator_seconds") or 0)
                                            for r in depth), 2),
        "depth_objective_seconds": round(sum(float(
            (r.get("operator_timing") or {}).get("objective_seconds") or 0)
            for r in depth), 2),
        "depth_cache_modes": sorted({str((r.get("reference_cache") or {})
                                         .get("mode")) for r in depth}),
        "state_evaluation_seconds": round(sum(float(
            (r.get("state_evaluation_seconds") or {}).get("seconds") or 0)
            for r in rows), 2)}


def commit(session, result, out: Path, k: int) -> dict[str, Any]:
    """The preregistered Top-K by the objective ranking, committed for real."""
    from aadistill.initialization.planning import stage1_selection

    ranking = result.finalists(session.config.policy, k)
    if len(ranking.selected) != k:
        raise D2DriverError(
            f"the objective ranking selected {len(ranking.selected)} recovery-"
            f"admissible leaves and the preregistration commits {k}; the search "
            "is NOT COMPLETE")
    path = stage1_selection.commit(
        search_config=session.config, ranking=ranking, suite=session.config.suite,
        policy=session.config.policy, profiles=session.config.profiles,
        journal_path=out / "search" / "states.jsonl", directory=out)
    selected = json.loads(Path(path).read_text())["selected"]
    for row in selected:
        for required in ("state_id", "checkpoint_path", "artifact_digest",
                         "arch_signature", "weights_digest"):
            if not row.get(required):
                raise D2DriverError(f"the committed selection omits {required!r} "
                                    f"for {row.get('state_id')}")
    return {"selection_path": str(Path(path).relative_to(out)), "k": k,
            "n_selected": len(selected),
            "selected_rows": [{f: r.get(f) for f in (
                "state_id", "path", "checkpoint_path", "artifact_digest",
                "weights_digest", "arch_signature", "single_shard_sha256",
                "num_parameters")} for r in selected],
            "state_ids": [r["state_id"] for r in selected],
            "ranking_policy": session.config.policy.qualified_id,
            "retention": ranking.retention}


def kl_only_control(result, committed_ids: list[str], k: int) -> dict[str, Any]:
    """D1's ranking of D2's complete leaves on their raw KL. Reported only.

    Every D2 state carries D1's `state.teacher_kl.*` beside the objective, so
    PARETO_V1 ranks the same leaves exactly as D1's search would have. The
    overlap separates "the objective chose different leaves" from "the search
    under the objective reached different leaves".
    """
    from aadistill.initialization.planning.ranking import PARETO_V1

    ranking = result.finalists(PARETO_V1, k)
    ids = list(ranking.selected_ids)
    return {"ranking_policy": PARETO_V1.qualified_id, "selected": ids,
            "overlap_with_committed": sorted(set(ids) & set(committed_ids)),
            "n_overlap": len(set(ids) & set(committed_ids)),
            "used_for_selection": False}


def main(argv: list[str] | None = None) -> int:
    from stages.phase_d2 import d2_session as S

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True,
                    help="enters SearchConfig.config_hash, which the grant binds")
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--deadline-s", type=int, default=0,
                    help="seconds from THIS process's start; the launcher "
                         "derives it from the plan's soft_stop_minutes minus "
                         "the session minutes already elapsed")
    ap.add_argument("--prediction-rows", default="all_v1",
                    choices=list(S.PREDICTION_ROWS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--status", default=S.STATUS_PATH)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    process_started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    journal = Journal(out / "journal.jsonl")
    record: dict[str, Any] = {
        "schema": "aadistill.phase_d2.search_session/v1",
        "experiment_id": S.EXPERIMENT_ID, "stage_id": S.STAGE_ID,
        "run_id": args.run_id, "prediction_rows": args.prediction_rows,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "STARTED"}
    evidence_written = False
    result = None
    mark(status, "DRIVER_START")
    try:
        with journal.stage("A", "environment") as st:
            record["environment"] = probe_environment()
            st.result = {"gpu": record["environment"]["gpu_name"]}
        with journal.stage("A", "session_and_contract") as st:
            session = S.build_session(workdir=out / "search", run_id=args.run_id,
                                      device=args.device,
                                      prediction_rows=args.prediction_rows)
            record["contract"] = S.assert_session_contract(session)
            record["wiring"] = session.as_record()
            st.result = {"config_hash": record["contract"]["config_hash"][:16]}
        if args.authorization:
            with journal.stage("A", "authorization") as st:
                record["authorization"] = load_authorization(
                    args.authorization, contract=record["contract"],
                    run_id=args.run_id, prediction_rows=args.prediction_rows)
                st.result = {"hard_cap_usd":
                             record["authorization"]["hard_cap_usd"]}
        elif not args.check_only:
            raise D2DriverError("--authorization is required for a paid session")
        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0

        with journal.stage("B", "teacher_and_reference") as st:
            from stages.phase_d1 import d1_session as D1S

            teacher, teacher_id = _load_root_teacher()
            record["root_teacher"] = D1S.verify_staged_teacher(teacher_id["path"])
            session.evaluator.prime_reference(teacher)
            st.result = {"revision": str(teacher_id.get("revision", ""))[:12]}
        with journal.stage("C", "beam_search") as st:
            pricing = (record.get("authorization") or {}).get("pricing") or {}
            deadline = (guarded_deadline(
                args.deadline_s, started=process_started,
                telemetry_path=out / "search" / "telemetry.jsonl",
                guard_cache=pricing.get("pricing_in_force")
                == "cache_fix_validated") if args.deadline_s > 0 else None)
            record["deadline"] = {"seconds_from_driver_start": args.deadline_s,
                                  "cache_guard": bool(deadline
                                                      and deadline.guard_cache)}
            try:
                record["search"], result = _run_search(session, teacher,
                                                       deadline)
            finally:
                if deadline is not None:
                    record["deadline"].update(deadline.as_dict())
                    record["deadline"]["cache_regression"] = deadline.regression
            st.result = {"states": record["search"]["n_states"]}
        with journal.stage("D", "commit_top_k") as st:
            k = int(S.preregistration()["retention"]["k"])
            record["commit"] = commit(session, result, out, k)
            record["kl_only_control"] = kl_only_control(
                result, record["commit"]["state_ids"], k)
            st.result = {"committed": record["commit"]["state_ids"]}
        record["status"] = "COMPLETE"
    except BaseException as exc:                      # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        journal.event(stage="driver", status="failed", error=str(exc)[:400])
    finally:
        record["objective_telemetry"] = objective_telemetry(
            out / "search" / "telemetry.jsonl")
        record["stages"] = journal.stages
        record["elapsed_seconds"] = round(time.time() - journal.started, 3)
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            (out / "d2_search.json").write_text(
                json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
            evidence_written = True
            say(f"\nwrote {out}/d2_search.json  status={record['status']}")
        except OSError as exc:
            say(f"\nCOULD NOT WRITE {out}/d2_search.json: {exc}")
        mark(status, terminal_marker(check_only=bool(args.check_only),
                                     status=str(record["status"]),
                                     evidence_written=evidence_written))
    if args.check_only:
        return 0 if record["status"] == "CHECK_ONLY_OK" else 1
    return 0 if record["status"] == "COMPLETE" and evidence_written else 1


if __name__ == "__main__":
    raise SystemExit(main())

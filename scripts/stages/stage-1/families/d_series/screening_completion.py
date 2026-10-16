"""When a D-series behavioural rung (SCREENING, or the SHARED CONFIRMATION)
counts as complete -- for the budget. `$0`.

The confirmation is the same contract over its own records: its frozen rule's
output is the verdict file (`outcome`), where the screening's is the ranking
file (`selection.outcome`); `ScreeningCompletionSpec.outcome_key` names which.

The search's rule, for a screening rung (`search_completion.py` says why a
release of reservation is a machine-readable contract and never a sentence).
A screening rung may take several one-use attempts under one cumulative grant;
every attempt's cost is inside the live limits through its own closeout, so
the rung leaves the programme plan once exactly ONE attempt satisfies every
clause and no attempt's cost is unknown:

    authorized_identity     the run's one-use grant loads (self-hash) and the
                            closeout and grant name the same run, grant hash
                            and the contract the caller derives now
    terminal_complete       the runtime record ALL_DONE and passed; the
                            closeout's runner terminal ALL_DONE and the
                            driver's status COMPLETE
    every_unit_verified     the off-pod ranking exists, binds the same
                            contract, and verified EVERY frozen probe -- each
                            unit's checks all true and aligned with B's rows --
                            and carries a selection
    products_secured        the runtime's required products secured and its
                            teardown gate allowed without emergency
    provider_teardown       the closeout says verified AND the runtime record
                            shows the provider confirms the pod gone, not
                            billing
    cost_closed             a positive all-in that is GPU + disk, GPU equal to
                            the runtime's, within the grant, not below the
                            provider debit, BOOKED by the budget authority
                            under money.all_in_usd
    no_unsettled_resources  every refused draw released and confirmed not
                            billing; no teardown-ownership block; and every
                            OTHER attempt of the rung has a known cost

It never raises on a bad record; a failed clause keeps the reservation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CLAUSES = ("authorized_identity", "terminal_complete", "every_unit_verified",
           "products_secured", "provider_teardown", "cost_closed",
           "no_unsettled_resources")
DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730.0


@dataclass(frozen=True)
class ScreeningCompletionSpec:
    """The instance facts, from the caller."""

    experiment_id: str
    closeout_schema: str
    contract_hash: str
    probe_ids: tuple[str, ...]
    load_authorization: Callable[[Path], Any]
    ranking_filename: str
    disk_gb: int
    booked_cost: Callable[[str, str], tuple[float, str] | None]
    #: (run_id) -> (cost, basis) or (None, why) for every attempt of the rung.
    attempt_cost: Callable[[Path], tuple[float | None, str]]
    #: Where the frozen rule's outcome sits in the rule's output file.
    outcome_key: tuple[str, ...] = ("selection", "outcome")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def assess(run_dir: Path, spec: ScreeningCompletionSpec) -> dict[str, Any]:
    """Every clause for one attempt. Never raises on a bad record."""
    problems: dict[str, list[str]] = {c: [] for c in CLAUSES}

    def need(clause: str, ok: Any, why: str) -> None:
        if not ok:
            problems[clause].append(why)

    run_id = run_dir.name
    closeout = _json(run_dir / "closeout/outcome.json") or {}
    runtime = _json(run_dir / "runtime/session.json") or {}
    ranking = _json(run_dir / "evidence" / spec.ranking_filename) or {}
    money = closeout.get("money") or {}

    c = "authorized_identity"
    need(c, closeout.get("schema") == spec.closeout_schema,
         f"closeout schema {closeout.get('schema')!r}")
    need(c, closeout.get("run_id") == run_id
         and closeout.get("experiment_id") == spec.experiment_id,
         "the closeout names another run or experiment")
    auth = None
    try:
        auth = spec.load_authorization(run_dir / "governance/authorization.json")
    except Exception as exc:                                  # noqa: BLE001
        need(c, False, f"the grant does not load: {type(exc).__name__}")
    hard = None
    if auth is not None:
        raw = _json(run_dir / "governance/authorization.json") or {}
        hard = _num(getattr(auth, "hard_cap_usd", None))
        need(c, getattr(auth, "run_id", None) == run_id,
             "the grant was issued for another run")
        need(c, raw.get("authorization_sha256")
             == closeout.get("authorization_sha256"),
             "the closeout's authorization_sha256 is not the grant's")
        need(c, getattr(auth, "contract_hash", None)
             == closeout.get("contract_hash") == spec.contract_hash,
             "grant, closeout and the derived contract disagree")

    c = "terminal_complete"
    need(c, runtime.get("terminal") == "ALL_DONE"
         and runtime.get("passed") is True,
         f"runtime terminal {runtime.get('terminal')!r}, passed "
         f"{runtime.get('passed')!r}")
    need(c, closeout.get("runner_terminal") == "ALL_DONE"
         and closeout.get("driver_status") == "COMPLETE",
         "the closeout does not record ALL_DONE and a COMPLETE driver")

    c = "every_unit_verified"
    units = ranking.get("units_verified") or {}
    need(c, ranking.get("contract_hash") == spec.contract_hash,
         "the ranking binds another contract")
    need(c, sorted(units) == sorted(spec.probe_ids),
         f"the ranking verified {sorted(units)}, not the frozen probes")
    bad = [pid for pid, u in units.items()
           if not (u.get("verified") is True
                   and u.get("aligned_with_B") is True
                   and u.get("checks")
                   and all(v is True for v in u["checks"].values()))]
    need(c, not bad, f"units not verified: {bad}")
    node: Any = ranking
    for key in spec.outcome_key:
        node = (node or {}).get(key) if isinstance(node, dict) else None
    need(c, bool(node), "the rule's output carries no outcome")

    c = "products_secured"
    need(c, (runtime.get("required_products_secured") or {}).get("ok") is True,
         "the runtime's required products are not secured")
    gate = runtime.get("teardown_gate") or {}
    need(c, gate.get("allowed") is True and gate.get("emergency") is False,
         "the teardown gate did not allow a normal teardown")
    need(c, not runtime.get("artifact_loss_risk"),
         "the runtime records an artifact-loss risk")

    c = "provider_teardown"
    state = runtime.get("final_pod_state") or {}
    need(c, closeout.get("provider_teardown_verified") is True
         and runtime.get("provider_confirms_gone") is True
         and state.get("exists") is False and state.get("billing") is False,
         "the provider does not confirm the pod gone and not billing")

    c = "cost_closed"
    all_in, gpu = _num(money.get("all_in_usd")), _num(money.get("gpu_usd"))
    minutes = _num(money.get("elapsed_minutes"))
    need(c, all_in is not None and all_in > 0, f"all-in {all_in!r}")
    if None not in (all_in, gpu, minutes):
        disk = spec.disk_gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH \
            * minutes / 60.0
        need(c, abs(all_in - math.ceil((gpu + disk) * 1e4) / 1e4) <= 1e-4,
             "all-in is not GPU + disk")
    runtime_gpu = _num((runtime.get("cost") or {}).get("actual_usd"))
    need(c, runtime_gpu is not None and gpu is not None
         and abs(runtime_gpu - gpu) <= 1e-4,
         "the runtime's GPU cost is not the closeout's")
    need(c, hard is not None and all_in is not None and all_in <= hard,
         f"all-in {all_in} is not within the grant {hard}")
    debit = _num((money.get("provider_balance_cross_check") or {})
                 .get("measured_delta_usd"))
    need(c, debit is None or (all_in is not None and all_in + 1e-4 >= debit),
         f"the booked {all_in} is below the provider debit {debit}")
    booked = spec.booked_cost(spec.experiment_id, run_id)
    need(c, booked is not None and all_in is not None
         and abs(booked[0] - all_in) <= 1e-4
         and booked[1] == "money.all_in_usd",
         f"the budget authority books {booked!r}")

    c = "no_unsettled_resources"
    draws = runtime.get("draws") or []
    need(c, all((d.get("release") or {}).get("confirmed_not_billing") is True
                for d in draws),
         "a refused draw is not confirmed released")
    need(c, not runtime.get("blocked_teardown_ownership"),
         "an unresolved teardown-ownership block")
    for other in sorted(run_dir.parent.glob(f"{run_id.rsplit('_', 2)[0]}_*")):
        if other == run_dir or not other.is_dir():
            continue
        cost, why = spec.attempt_cost(other)
        need(c, cost is not None, f"attempt {other.name}: {why}")

    failed = [k for k, v in problems.items() if v]
    return {"run_id": run_id, "complete": not failed,
            "failed_clauses": failed,
            "problems": {k: v for k, v in problems.items() if v},
            "clauses": list(CLAUSES), "booked_all_in_usd": all_in,
            "authorized_hard_usd": hard,
            "claims_complete": str(closeout.get("classification", ""))
            .startswith("COMPLETE")}


def completion(runs_root: Path, spec: ScreeningCompletionSpec, *,
               pattern: str) -> dict[str, Any]:
    """Every closed attempt, and the one that counts (exactly one)."""
    runs = [assess(p.parent.parent, spec)
            for p in sorted(runs_root.glob(f"{pattern}/closeout/outcome.json"))]
    passing = [r for r in runs if r["complete"]]
    contradictions = [f"{r['run_id']} claims completion and fails "
                      f"{r['failed_clauses']}"
                      for r in runs if r["claims_complete"] and not r["complete"]]
    if len(passing) > 1:
        contradictions.append(
            f"{len(passing)} attempts satisfy the contract; one rung has one "
            "complete attempt")
    return {"runs": runs,
            "complete": passing[0] if len(passing) == 1 and not contradictions
            else None,
            "contradictions": contradictions}


__all__ = ["CLAUSES", "ScreeningCompletionSpec", "assess", "completion"]

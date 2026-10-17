"""When a D-series formal SEARCH counts as complete -- for the budget. `$0`.

A programme view stops planning a search at its hard ceiling once that search
is spent: its actual all-in cost is then inside the live limits through its
closeout. Releasing that reservation is a money decision, so the condition is a
machine-readable CONTRACT over the run's own records, not a sentence. The first
version released ~$24 of reservation on `classification.startswith("COMPLETE
FORMAL SEARCH")` -- a string anyone can type.

Nine clauses, each read from structured fields and cross-checked between the
records that state them independently (closeout, governance, runtime, driver
evidence, manifest, selection, the budget authority's own booking):

    authorized_identity     the run's one-use grant loads (self-hash), and the
                            closeout, the grant and the runtime record name the
                            same run, preregistration, commit lineage and hash
    terminal_complete       ALL_DONE and passed, in the closeout AND the runtime
                            record; the driver's own status COMPLETE
    full_search             every stage ok, the deadline not fired, the journal
                            and the level table agree on the expansion count,
                            the last level all complete leaves
    top_k_committed         K = the preregistration's; the committed members,
                            the selection record (hash-verified) and the
                            driver's commit name the same leaves in one order
    checkpoints_secured     K required, K secured, every identity verified, and
                            the teardown gate's product checks passed
    manifest_and_archive    the session manifest is complete; the evidence in
                            git hashes to its entries; the archive and the
                            state journal carry recorded sha256
    provider_teardown       the provider confirms the resource gone, not billing
    cost_closed             a positive all-in figure that equals GPU + disk, is
                            within the grant, is BOOKED by the budget authority
                            under that shape, and is not below the provider debit
    no_unsettled_resources  every resource the session created is accounted for
                            and none is in an unknown ownership state

A contract that fails keeps the reservation. It never raises on a bad record:
missing or contradictory evidence is a reason to keep money aside, and the
programme view reports WHICH clause failed.

The clauses are generic over a search experiment; the instance facts (which
experiment, which schema, which grant type, which K) come from the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

CLAUSES = ("authorized_identity", "terminal_complete", "full_search",
           "top_k_committed", "checkpoints_secured", "manifest_and_archive",
           "provider_teardown", "cost_closed", "no_unsettled_resources")


@dataclass(frozen=True)
class SearchCompletionSpec:
    """The instance facts a search-completion check needs, from the caller."""

    experiment_id: str
    closeout_schema: str
    k: int
    preregistration_hash: str
    load_authorization: Callable[[Path], Any]
    evidence_filename: str
    #: (experiment_id, run_id) -> (booked all-in, shape) as the budget
    #: authority books it; `None` when it does not book the run at all.
    booked_cost: Callable[[str, str], tuple[float, str] | None]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def assess(run_dir: Path, spec: SearchCompletionSpec) -> dict[str, Any]:
    """Every clause, for one run directory. Never raises on a bad record."""
    problems: dict[str, list[str]] = {c: [] for c in CLAUSES}

    def need(clause: str, ok: Any, why: str) -> None:
        if not ok:
            problems[clause].append(why)

    closeout = _json(run_dir / "closeout/outcome.json") or {}
    runtime = _json(run_dir / "runtime/session.json") or {}
    evidence = _json(run_dir / "evidence" / spec.evidence_filename) or {}
    manifest = _json(run_dir / "evidence/manifest.json") or {}
    execution = closeout.get("execution") or {}
    money = closeout.get("money") or {}
    run_id = run_dir.name

    # -- authorized_identity ------------------------------------------------
    c = "authorized_identity"
    need(c, closeout.get("schema") == spec.closeout_schema,
         f"closeout schema {closeout.get('schema')!r} is not "
         f"{spec.closeout_schema!r}")
    need(c, closeout.get("run_id") == run_id,
         f"closeout names run {closeout.get('run_id')!r}, directory is {run_id}")
    need(c, closeout.get("experiment_id") == spec.experiment_id,
         f"closeout experiment {closeout.get('experiment_id')!r}")
    auth = None
    try:
        auth = spec.load_authorization(run_dir / "governance/authorization.json")
    except Exception as exc:                                  # noqa: BLE001
        need(c, False, f"the run's grant does not load: {type(exc).__name__}: "
                       f"{exc}")
    if auth is not None:
        auth_raw = _json(run_dir / "governance/authorization.json") or {}
        need(c, getattr(auth, "run_id", None) == run_id,
             f"the grant was issued for run {getattr(auth, 'run_id', None)!r}")
        need(c, auth_raw.get("authorization_sha256")
             == closeout.get("authorization_sha256"),
             "the closeout's authorization_sha256 is not the grant's")
        need(c, getattr(auth, "plan_hash", None)
             == closeout.get("preregistration_hash") == spec.preregistration_hash,
             "grant, closeout and the approved preregistration disagree on the "
             "preregistration hash")
        need(c, getattr(auth, "authorized_session_commit", None)
             == closeout.get("authorized_base_commit"),
             "the closeout's authorized base commit is not the grant's")
    check = runtime.get("session_commit_check") or {}
    need(c, check.get("session_commit") == closeout.get("session_commit")
         and bool(closeout.get("session_commit")),
         "the runtime record and the closeout name different session commits")
    need(c, check.get("harness_matches") is True
         and check.get("commit_carries_this_authorization") is True,
         "the session commit was not verified to carry the grant and its "
         "harness")
    need(c, runtime.get("one_use_grant_consumed") is True
         and money.get("one_use_grant_consumed") is True,
         "the one-use grant is not recorded consumed in both records")

    # -- terminal_complete --------------------------------------------------
    c = "terminal_complete"
    need(c, execution.get("terminal") == "ALL_DONE"
         and execution.get("passed") is True,
         f"closeout terminal {execution.get('terminal')!r}, passed "
         f"{execution.get('passed')!r}")
    need(c, runtime.get("terminal") == "ALL_DONE"
         and runtime.get("passed") is True,
         f"runtime terminal {runtime.get('terminal')!r}, passed "
         f"{runtime.get('passed')!r}")
    need(c, evidence.get("status") == "COMPLETE",
         f"the driver's own status is {evidence.get('status')!r}")

    # -- full_search --------------------------------------------------------
    c = "full_search"
    stages = evidence.get("stages") or []
    need(c, bool(stages) and all(s.get("status") == "ok" for s in stages),
         "a driver stage did not end ok")
    need(c, (evidence.get("deadline") or {}).get("expired") is False,
         "the soft stop fired or is unrecorded")
    trajectory = closeout.get("search_trajectory") or {}
    levels = trajectory.get("levels") or []
    expansions = trajectory.get("expansions")
    need(c, isinstance(expansions, int) and expansions > 0
         and expansions == trajectory.get("states_journal")
         == sum(int(lv.get("states") or 0) for lv in levels),
         "the expansion count, the state journal and the level table disagree")
    need(c, bool(levels) and int(levels[-1].get("expanded", -1)) == 0
         and levels[-1].get("complete_leaves") == levels[-1].get("states")
         == trajectory.get("complete_leaves"),
         "the last level is not entirely complete leaves")
    need(c, all(int(lv.get("invalid") or 0) == 0 for lv in levels),
         "a level carries invalid states")
    search = evidence.get("search") or {}
    need(c, search.get("n_complete_leaves") == trajectory.get("complete_leaves")
         and search.get("n_states") == trajectory.get("n_states_with_root"),
         "the driver's leaf/state counts disagree with the closeout")
    need(c, isinstance(trajectory.get("complete_leaves"), int)
         and trajectory["complete_leaves"] >= spec.k,
         f"fewer complete leaves than K={spec.k}")

    # -- top_k_committed ----------------------------------------------------
    c = "top_k_committed"
    committed = closeout.get("committed_top_4") or closeout.get("committed_top_k") \
        or {}
    members = [m.get("state_id") for m in committed.get("members") or []]
    need(c, len(members) == spec.k,
         f"{len(members)} committed members, the preregistration commits "
         f"{spec.k}")
    selection = None
    try:
        from aadistill.initialization.planning import stage1_selection

        selection = stage1_selection.load(run_dir / "evidence/stage1_selection.json")
    except Exception as exc:                                  # noqa: BLE001
        need(c, False, f"the selection record does not verify: {exc}")
    if selection is not None:
        need(c, [r.get("state_id") for r in selection.get("selected") or []]
             == members, "the selection record and the closeout name "
                         "different leaves or a different order")
        need(c, selection.get("selection_sha256")
             == committed.get("selection_sha256"),
             "the closeout binds a different selection hash")
    commit = evidence.get("commit") or {}
    need(c, [r.get("state_id") for r in commit.get("selected_rows") or []]
         == members and commit.get("k") == spec.k,
         "the driver's commit disagrees with the closeout")

    # -- checkpoints_secured ------------------------------------------------
    c = "checkpoints_secured"
    secured = closeout.get("products_secured") or {}
    need(c, secured.get("required") == secured.get("secured") == spec.k
         and secured.get("all_identity_verified") is True,
         f"products secured {secured.get('secured')!r} of "
         f"{secured.get('required')!r}, identity verified "
         f"{secured.get('all_identity_verified')!r}")
    gate = execution.get("teardown_gate") or {}
    checks = gate.get("checks") or {}
    need(c, checks.get("checkpoint_hashes_matched") is True
         and checks.get("required_products_secured") is True,
         "the teardown gate did not pass its product checks")

    # -- manifest_and_archive ------------------------------------------------
    c = "manifest_and_archive"
    need(c, manifest.get("ok") is True and manifest.get("missing") == []
         and manifest.get("final_streams_quiescent") is True,
         "the session manifest is not complete")
    entries = {Path(e.get("path", "")).name: e for e in
               manifest.get("entries") or []}
    for name in (spec.evidence_filename, "stage1_selection.json"):
        e = entries.get(name) or {}
        need(c, bool(e) and _sha256(run_dir / "evidence" / name)
             == e.get("sha256"),
             f"evidence/{name} does not hash to its manifest entry")
    out_of_tree = (closeout.get("evidence_in_tree") or {}).get("out_of_tree") \
        or {}
    need(c, _is_hex64(out_of_tree.get("archive_sha256")),
         "no archive sha256 is recorded")
    states = entries.get("states.jsonl") or {}
    need(c, _is_hex64(out_of_tree.get("states_jsonl_sha256"))
         and out_of_tree.get("states_jsonl_sha256") == states.get("sha256"),
         "the state journal's recorded sha256 is not the manifest's")

    # -- provider_teardown ----------------------------------------------------
    c = "provider_teardown"
    final = execution.get("final_pod_state") or {}
    need(c, execution.get("provider_confirms_gone") is True
         and runtime.get("provider_confirms_gone") is True,
         "the provider has not confirmed the resource gone in both records")
    need(c, final.get("exists") is False and final.get("billing") is False,
         f"final pod state {final!r}")
    need(c, gate.get("allowed") is True, "the teardown gate did not allow")

    # -- cost_closed ----------------------------------------------------------
    c = "cost_closed"
    all_in, gpu, disk = (_num(money.get(k)) for k in
                         ("all_in_usd", "gpu_usd", "disk_usd"))
    hard = _num(money.get("authorized_hard_usd"))
    need(c, all_in is not None and all_in > 0,
         f"all-in cost {money.get('all_in_usd')!r} is not a positive number")
    if all_in is not None and gpu is not None and disk is not None:
        need(c, abs(gpu + disk - all_in) <= 1e-4,
             f"all-in {all_in} != GPU {gpu} + disk {disk}")
    else:
        need(c, False, "GPU or disk cost is missing")
    if all_in is not None and hard is not None:
        need(c, all_in <= hard, f"all-in {all_in} exceeds the grant {hard}")
    else:
        need(c, False, "no authorized hard ceiling is recorded")
    runtime_gpu = _num((runtime.get("cost") or {}).get("actual_usd"))
    need(c, runtime_gpu is not None and gpu is not None
         and abs(runtime_gpu - gpu) <= 1e-4,
         "the runtime record's GPU cost is not the closeout's")
    debit = _num((money.get("provider_balance_cross_check") or {})
                 .get("delta_usd"))
    if debit is not None and all_in is not None:
        need(c, all_in + 1e-4 >= debit,
             f"the booked {all_in} is below the provider debit {debit}")
    booked = spec.booked_cost(spec.experiment_id, run_id)
    need(c, booked is not None and all_in is not None
         and abs(booked[0] - all_in) <= 1e-4
         and booked[1] == "money.all_in_usd",
         f"the budget authority books {booked!r}, not money.all_in_usd = "
         f"{all_in}")

    # -- no_unsettled_resources ---------------------------------------------
    c = "no_unsettled_resources"
    created = money.get("provider_resources_created")
    need(c, created == 1 and bool(money.get("pod_id"))
         and money.get("pod_id") == runtime.get("pod_id"),
         f"{created!r} resources created; this contract accounts for exactly "
         "one, named identically in both records")
    need(c, not runtime.get("blocked_teardown_ownership"),
         "the runtime record carries an unresolved teardown ownership block")

    failed = [k for k, v in problems.items() if v]
    return {"run_id": run_id, "complete": not failed,
            "failed_clauses": failed,
            "problems": {k: v for k, v in problems.items() if v},
            "clauses": list(CLAUSES),
            "booked_all_in_usd": all_in,
            "authorized_hard_usd": hard,
            "claims_complete": str(closeout.get("classification", ""))
            .startswith("COMPLETE")}


def completion(runs_root: Path, spec: SearchCompletionSpec, *,
               pattern: str = "*") -> dict[str, Any]:
    """Every closed search run under `runs_root`, and the one that counts.

    `complete` is a run only when exactly ONE run satisfies every clause: two
    complete formal searches of one preregistration is itself a contradiction,
    and money is kept aside until a reader resolves it.
    """
    runs = [assess(path.parent.parent, spec)
            for path in sorted(runs_root.glob(f"{pattern}/closeout/outcome.json"))]
    passing = [r for r in runs if r["complete"]]
    contradictions = []
    for r in runs:
        if r["claims_complete"] and not r["complete"]:
            contradictions.append(
                f"{r['run_id']} claims completion and fails "
                f"{r['failed_clauses']}")
    if len(passing) > 1:
        contradictions.append(
            f"{len(passing)} runs satisfy the contract "
            f"({[r['run_id'] for r in passing]}); one preregistration has one "
            "formal search")
    return {"runs": runs,
            "complete": passing[0] if len(passing) == 1 else None,
            "contradictions": contradictions}


def budget_authority_booking(repo_root: Path) -> Callable[[str, str],
                                                           tuple[float, str] | None]:
    """The budget authority's OWN booking, so 'closed' means 'booked there'."""
    import importlib.util

    path = Path(repo_root) / "scripts/maintenance/consolidation/derive_budget.py"
    module_spec = importlib.util.spec_from_file_location("derive_budget", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    sessions = module.project_sessions(Path(repo_root))

    def booked(experiment_id: str, run_id: str) -> tuple[float, str] | None:
        for row in sessions:
            if (row.get("experiment_id"), row.get("run_id")) == (
                    experiment_id, run_id) and row.get("cost_usd") is not None:
                return float(row["cost_usd"]), str(row.get("shape"))
        return None

    return booked


__all__ = ["CLAUSES", "SearchCompletionSpec", "assess",
           "budget_authority_booking", "completion"]

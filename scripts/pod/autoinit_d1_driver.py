#!/usr/bin/env python3
"""The D1 formal SEARCH driver. One session, four stages, evidence on every path.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_d1_driver.py \
        --out DIR --authorization PATH [--arm supervised_target] [--deadline-s N]

Stages, and what each is allowed to do:

    A  setup: registries, frozen assets, the session, and the CONTRACT ASSERTED
       before any expansion. A miswired session must cost nothing.
    B  the root teacher materializes once; the evaluator is primed with THAT
       object and the search is handed the same one.
    C  the beam search over the frozen space, at the authorized ceiling.
    D  commit_top_k and package. No recovery, no behavioural work, no decision.

**The contract is asserted before the teacher is loaded.** Every wiring
requirement is a thing the core permits a caller to omit, and a run that omits one
produces records that look valid and are scientifically unusable. Checking after
the search would mean discovering it after paying for it.

**Evidence is written on EVERY path out**, including a failure in stage A, because
a failed session's evidence is the point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    p = str(REPO / extra)
    if p not in sys.path:
        sys.path.insert(0, p)


class D1DriverError(RuntimeError):
    """A premise of the session does not hold."""


def say(message: str) -> None:
    print(message, flush=True)


class Journal:
    """Append-only stage events. The record of a failed run, not a log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.time()
        self.stages: list[dict[str, Any]] = []

    def event(self, **fields: Any) -> None:
        row = {"t": round(time.time() - self.started, 3), **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        say(f"[{row['t']:9.3f}] " + " ".join(
            f"{k}={v}" for k, v in fields.items() if k != "t"))

    def stage(self, letter: str, name: str):
        return _Stage(self, letter, name)


class _Stage:
    def __init__(self, journal: Journal, letter: str, name: str) -> None:
        self.j, self.letter, self.name = journal, letter, name
        self.result: dict[str, Any] | None = None

    def __enter__(self):
        self.t0 = time.time()
        self.j.event(stage=self.letter, name=self.name, status="start")
        return self

    def __exit__(self, exc_type, exc, tb):
        seconds = round(time.time() - self.t0, 3)
        row = {"stage": self.letter, "name": self.name, "seconds": seconds}
        if exc is None:
            row["status"] = "ok"
            if self.result:
                row["result"] = self.result
        else:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        self.j.stages.append(row)
        self.j.event(**row)
        return False


def probe_environment() -> dict[str, Any]:
    """CUDA, or refuse. A D1 search on the host would answer a different question."""
    import torch

    if not torch.cuda.is_available():
        raise D1DriverError(
            "no CUDA device. A formal D1 search materializes a 4B-class teacher "
            "and reduces KL over its logits; a CPU fallback would produce "
            "evidence about the wrong machine.")
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gib": round(props.total_memory / 2**30, 3),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "device_count": torch.cuda.device_count(),
    }


def load_authorization(path: str, *, arm: str, design_hash: str):
    """The one-use artifact, read through its own loader and bound to this run."""
    from experiments.phase_d1.d1_authorization import (
        D1Authorization, D1AuthorizationRefused,
    )

    auth = D1Authorization.load(path)
    auth.require_plan(design_hash)
    if auth.arm != arm:
        raise D1AuthorizationRefused(
            f"the authorization covers the {auth.arm!r} arm and this session "
            f"runs {arm!r}")
    if not auth.authorizes_d1_search or not auth.allows_beam_search:
        raise D1AuthorizationRefused(
            "the artifact does not authorize a D1 beam search")
    if auth.allows_recovery or auth.allows_behavioural:
        raise D1AuthorizationRefused(
            "the artifact claims recovery or behavioural permission; a D1 search "
            "session has neither and would be running unauthorized work")
    return auth


def main(argv: list[str] | None = None) -> int:
    from experiments.phase_d1 import d1_session as S

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--authorization", default=None,
                    help="the one-use artifact; omitted only by --check-only")
    ap.add_argument("--arm", default=S.TREATMENT_ARM, choices=list(S.ARMS))
    ap.add_argument("--deadline-s", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-only", action="store_true",
                    help="stages A alone: the registries, the frozen assets, the "
                         "session and its contract, on THIS interpreter. Seconds, "
                         "before the teacher is resident.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal = Journal(out / "journal.jsonl")
    record: dict[str, Any] = {
        "schema": "aadistill.phase_d1.search_session/v1",
        "experiment_id": S.EXPERIMENT_ID,
        "stage_id": S.STAGE_ID,
        "arm": args.arm,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "STARTED",
    }
    session = None
    try:
        with journal.stage("A", "environment") as st:
            record["environment"] = probe_environment()
            st.result = {"gpu": record["environment"]["gpu_name"]}

        with journal.stage("A", "registries_and_design") as st:
            S._register_frozen_operators()
            record["design_hash"] = S.design_hash()
            record["open_blockers"] = list(S.open_blockers())
            if record["open_blockers"]:
                raise D1DriverError(
                    f"the design reports open blockers {record['open_blockers']}; "
                    "a formal session may not run over one")
            st.result = {"design_hash": record["design_hash"][:16]}

        with journal.stage("A", "session_and_contract") as st:
            session = S.build_session(
                arm=args.arm, workdir=out / "search", run_id=out.name,
                device=args.device)
            #: BEFORE ANYTHING EXPENSIVE. Each requirement the core permits a
            #: caller to omit is checked against the DESIGN, not against this
            #: driver's intentions.
            record["contract"] = S.assert_session_contract(session)
            record["wiring"] = session.as_record()
            st.result = {
                "protocol": record["contract"]["measurement_protocol_id"][:16],
                "config_hash": record["contract"]["config_hash"][:16]}

        if args.authorization:
            with journal.stage("A", "authorization") as st:
                auth = load_authorization(
                    args.authorization, arm=args.arm,
                    design_hash=record["design_hash"])
                record["authorization"] = {
                    "authorization_id": auth.authorization_id,
                    "hard_cap_usd": auth.hard_cap_usd,
                    "session_commit": auth.session_commit,
                    "authorized_stages": list(auth.authorized_stages),
                    "measurement_protocol_id": auth.measurement_protocol_id,
                    "config_hash": auth.config_hash,
                }
                #: THE IDENTITIES THE MONEY WAS AUTHORIZED AGAINST must be the
                #: ones this session carries, or the grant priced another run.
                if auth.measurement_protocol_id != \
                        record["contract"]["measurement_protocol_id"]:
                    raise D1DriverError(
                        "the authorization binds protocol "
                        f"{auth.measurement_protocol_id[:12]} and this session is "
                        f"{record['contract']['measurement_protocol_id'][:12]}")
                if auth.config_hash != record["contract"]["config_hash"]:
                    raise D1DriverError(
                        f"the authorization binds config {auth.config_hash[:12]} "
                        f"and this session is {record['contract']['config_hash'][:12]}")
                st.result = {"hard_cap_usd": auth.hard_cap_usd}
        elif not args.check_only:
            raise D1DriverError(
                "--authorization is required for a paid session; a search that "
                "cannot name its authorization is unauthorized work")

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the CUDA probe, the four registries, the frozen state-eval "
                "asset, the session and its contract resolved on this "
                "interpreter. No teacher was loaded and no expansion ran.")
            return 0

        with journal.stage("B", "teacher_and_reference") as st:
            teacher, teacher_id = _load_root_teacher()
            #: THE STAGED BYTES, against the frozen binding, BEFORE expansion. A
            #: declared root identity binds nothing on its own: the lineage would
            #: record the frozen revision while the search read whatever was
            #: staged.
            record["root_teacher"] = S.verify_staged_teacher(teacher_id["path"])
            session.evaluator.prime_reference(teacher)
            record["teacher"] = teacher_id
            st.result = {"revision": teacher_id.get("revision", "")[:12],
                         "root_sha256":
                             record["root_teacher"]["root_teacher_sha256"][:12]}

        with journal.stage("C", "beam_search") as st:
            summary, result_obj = _run_search(session, teacher, args, journal)
            record["search"] = summary
            #: The live object, for the ranking. Popped before the record is
            #: written: a `SearchResult` is not JSON and the record must be.
            record["_search_result"] = result_obj
            st.result = {"states": summary["n_states"],
                         "leaves": summary["n_leaves"]}

        with journal.stage("D", "commit_top_k") as st:
            record["commit"] = _commit(session, record, out)
            st.result = {"committed": record["commit"]["n_selected"],
                         "state_ids": record["commit"]["state_ids"]}

        record["status"] = "COMPLETE"
    except BaseException as exc:                      # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        journal.event(stage="driver", status="failed", error=str(exc)[:400])
    finally:
        #: NOT JSON, and never was meant to be in the record.
        record.pop("_search_result", None)
        record["stages"] = journal.stages
        record["elapsed_seconds"] = round(time.time() - journal.started, 3)
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        #: WRITTEN ON EVERY PATH. A failed session's evidence is the point.
        (out / "d1_search.json").write_text(
            json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
        say(f"\nwrote {out}/d1_search.json  status={record['status']}")
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 1


def _load_root_teacher():
    """The frozen root teacher, at the revision the path spec pins."""
    from experiments.phase_a3 import a3_session as A3S
    from aadistill.initialization.specs.arch import get_adapter

    spec = A3S.path_spec(workdir_device="cuda")
    local = os.environ.get("D1_TEACHER_PATH") or os.environ.get("TEACHER_PATH")
    if not local:
        raise D1DriverError(
            "no teacher path: set D1_TEACHER_PATH to the staged root teacher. A "
            "driver that downloads it mid-session is a driver whose first paid "
            "minute is a network transfer.")
    adapter = get_adapter(spec.family)
    model = adapter.load(local, dtype="bfloat16", device="cuda")
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False
    return model, {"repo_id": spec.root_repo_id, "revision": spec.root_revision,
                   "path": str(local), "family": spec.family}


def _run_search(session, teacher, args, journal) -> dict[str, Any]:
    """The real beam, with the authorized wall clock."""
    from aadistill.initialization.planning.search import BeamSearch, Deadline
    from experiments.phase_d1 import d1_session as S
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )

    def calibration_loader(profile):
        return prepare_calibration_items(profile.resolve(REPO),
                                        profile_id=profile.qualified_id)

    deadline = Deadline(seconds=float(args.deadline_s)) if args.deadline_s > 0 \
        else None
    #: THE TEACHER's OWN IDENTITY. `root_materialization_id()` refuses an empty id
    #: or hash -- an unpinned root is a lineage that starts from nothing in
    #: particular, which every child inherits -- and the first version passed
    #: `f"d1/{arm}"` with an empty hash. The ARM belongs to the scoring protocol
    #: identity, which carries it through `position_policy` and `config_hash`; the
    #: root identity describes the teacher.
    root = S.root_teacher_identity()
    search = BeamSearch(
        adapter=get_adapter(session.config.target_spec.family),
        config=session.config,
        root_teacher_id=root["root_teacher_id"],
        root_teacher_sha256=root["root_teacher_sha256"],
        root_loader=lambda: teacher,
        calibration_loader=calibration_loader,
        measurer=lambda model, digest: session.evaluator.evaluate(model, digest),
        execution=session.evaluator.execution,
        numerics=S.numerics(),
        deadline=deadline)
    result = search.run()
    return {
        "n_states": len(result.states),
        "n_leaves": len(result.leaves),
        "n_complete_leaves": len(result.complete_leaves),
        "resumed": list(result.resumed),
        "finished_utc": result.finished_utc,
        "levels": [getattr(level, "as_dict", lambda: {})() for level in
                   result.levels],
        "config_hash": session.config.config_hash,
        "root_teacher_id": root["root_teacher_id"],
        "root_teacher_sha256": root["root_teacher_sha256"],
    }, result


def _commit(session, record, out: Path) -> dict[str, Any]:
    """`commit_top_k`: the REAL Stage-1 ranking, written by the REAL committer.

    `result.top_n(PARETO_V1, k)` then `stage1_selection.commit(...)`. Not a second
    ranking or a second record format: the first version of this function returned
    `top_k: []`, which cannot be a successful D1 search and would have let a
    session mark ALL_DONE with no candidate set at all.

    EXACTLY `k` recovery-admissible leaves, or the search is NOT COMPLETE. The
    ranking's own guard is what stops a depth-only intermediate -- which often
    scores BETTER on teacher KL than any fully compressed leaf -- from being
    promoted into a recovery probe it could never be a candidate for.
    """
    from aadistill.initialization.planning import stage1_selection
    from aadistill.initialization.planning.ranking import PARETO_V1
    from experiments.phase_d1 import d1_session as S

    doc = S.design()
    k = int(doc["behavioural_design"]["top_k"])
    result = record["_search_result"]
    ranking = result.top_n(PARETO_V1, k)
    if len(ranking.selected) != k:
        raise D1DriverError(
            f"the ranking selected {len(ranking.selected)} recovery-admissible "
            f"leaves and the design commits {k}. The formal search is NOT "
            "COMPLETE: a partial or empty candidate set must not be marked done, "
            "because the behavioural rungs would then screen something the search "
            "did not choose.")
    path = stage1_selection.commit(
        search_config=session.config, ranking=ranking, suite=session.config.suite,
        policy=PARETO_V1, profiles=session.config.profiles,
        journal_path=out / "search" / "states.jsonl",
        directory=out)
    committed = json.loads(Path(path).read_text())
    selected = committed["selected"]
    #: EVERY IDENTITY A TRANSFER NEEDS. `verify_transferred_leaf` rebuilds the
    #: identity from the bytes that arrive and takes `arch_signature` and
    #: `weights_digest` from this record because no file carries them; omitting
    #: one once made every transfer report NOT MATCHED on a KeyError while the
    #: bytes were correct.
    for row in selected:
        for required in ("state_id", "checkpoint_path", "artifact_digest",
                         "arch_signature", "weights_digest"):
            if not row.get(required):
                raise D1DriverError(
                    f"the committed selection omits {required!r} for "
                    f"{row.get('state_id')}, so its checkpoint could not be "
                    "identity-verified after transfer")
    return {
        "selection_path": str(Path(path).relative_to(out)),
        "k": k,
        "n_selected": len(selected),
        #: THE ROWS THE LAUNCHER'S PRODUCT FETCHER READS, by the name it reads
        #: them under. A launcher consuming a field the driver never writes is the
        #: writer/consumer gap this project has paid for more than once -- and the
        #: consequence here would be silent: no rows means no products, and the pod
        #: is deleted with the selected checkpoints on it.
        "selected_rows": [
            {k2: r.get(k2) for k2 in
             ("state_id", "path", "checkpoint_path", "artifact_digest",
              "weights_digest", "arch_signature", "single_shard_sha256",
              "num_parameters")}
            for r in selected],
        "state_ids": [r["state_id"] for r in selected],
        "checkpoint_paths": [r["checkpoint_path"] for r in selected],
        "artifact_digests": [r["artifact_digest"] for r in selected],
        "ranking_policy": PARETO_V1.qualified_id,
        "_what_this_is": (
            "the committed candidate set the behavioural rungs will screen. This "
            "session ranks and commits; it trains nothing and evaluates no "
            "behaviour."),
        "_products": (
            "these checkpoint directories are this session's PRODUCTS and must be "
            "secured off-pod before teardown: the next stage trains recovery "
            "probes FROM these initializations, and the screening price does not "
            "fund re-running two complete structural paths."),
        "_next": ("screening, as a SEPARATELY authorized session. Neither "
                  "behavioural rung could be bound before this set existed."),
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""The D1 REPLAY's pod-environment readiness. Its own session, its own view.

WHY THIS EXISTS, and it is not symmetry with the other eight entries. The
replay has been swept four times through `--experiment phase_d1`, and that
registry entry binds `autoinit_d1_launch` — the SEARCH's launcher. So every one
of those sweeps derived the staged view of a *different session*: the search
stages three frozen calibration/eval assets, the replay stages those three plus
its resolved plan, and the sweep therefore modelled a pod without the one file
the replay's driver is invoked with (`--plan`).

It cost nothing for four subruns because no test in the selection read the
plan. The moment one did, the sweep reported it as an unexpected environment
skip — and the obvious reading of that was "the test's skip guard is wrong",
which sent the repair one level too shallow: a file-granularity bug in
`staged_files` was real and was fixed, and the sweep still hid the plan,
because the session it was modelling does not stage it.

The recorder's own registry says the same thing five times over, about five
other sessions: a session that binds its own launcher, session id, executable
closure and staged view needs its own entry, or its chain is unusable at
exactly the step a launch rests on. The replay binds all four.

**Nothing here is transcribed.** The session id, the pod selection, the setup
environment and the staged view come from the real
`autoinit_d1_replay_launch.spec(...)`; the harness file list from the issuer
that binds it into the authorization. A second copy of any of those is a second
thing that can disagree with the session that actually launches.

**RUN-OWNED, no pointer.** Same reasoning as D1's search contract: this record
is evidence about one attempt, there is no phase-level fallback, and it must
not touch the Phase-C1 pointer that `logs/state/current.json ::
latest_verification` derives from.

**The record lives under `phase_d1_replay/`**, where this session's run
directory, session record and stage attribution already are — not under
`phase_d1/`, where the first four landed because the registry sent them to the
search's experiment id.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from aadistill.runtime import pod_environment as _pe
from aadistill.runtime import pod_environment as _runtime
from aadistill.runtime.pod_environment import (
    LAUNCH_BOUND,
    ReadinessGroups,
    RecordContract,
    SweepContract,
)

from stages.phase_d1 import d1_session as D1S

REPO = Path(__file__).resolve().parents[4]

#: THIS session's experiment id, which is not the search's. `rel_run_dir` keys
#: on it, so the record lands beside the session record rather than among the
#: search's attempts.
EXPERIMENT_ID = "phase_d1_replay"
STAGE_ID = D1S.STAGE_ID

#: THE WHOLE POINT OF THIS MODULE.
LAUNCHER_MODULE = "autoinit_d1_replay_launch"

#: Its own schema, so no other record can satisfy this verifier or it theirs.
SCHEMA = "aadistill.autoinit.d1_replay_pod_environment/v1"

RUN_READINESS_ROLE = "governance/readiness.json"


class D1ReplayReadinessError(RuntimeError):
    """A replay readiness premise does not hold. The recorder reports it as a
    refusal naming this experiment, never as a traceback."""


@functools.lru_cache(maxsize=1)
def _spec() -> Any:
    """The REAL replay `SessionSpec`, loaded the way the recorder loads it."""
    import sys

    for extra in ("scripts/pod", "tests"):
        p = str(REPO / extra)
        if p not in sys.path:
            sys.path.insert(0, p)
    from support.session_specs import load_session_launcher, session_args

    launcher = load_session_launcher(LAUNCHER_MODULE)
    return launcher.spec(session_args(launcher))


def session_id() -> str:
    """Read off the spec, not typed: a launcher that renamed its session would
    otherwise have its staged view derived under the old name."""
    return _spec().session_id


def pod_test_selection() -> tuple[str, ...]:
    """What the replay's pod gate collects, from its own `test_paths`.

    It happens to be D1's suite, which is correct -- the replay's blocking gate
    runs the experiment's own tests -- but it is DERIVED, so a replay that
    narrowed its selection would be swept at the narrower one.
    """
    paths = tuple(_spec().setup.test_paths)
    if not paths:
        raise D1ReplayReadinessError(
            "the replay session declares no test_paths, so its pod gate would "
            "collect the core suite and this contract would watch nothing. A "
            "readiness group over zero modules passes vacuously.")
    return paths


def readiness_groups() -> ReadinessGroups:
    """The claim: the declared selection passes with no unexpected skips.

    No `expected_skips`. That is deliberate and it is what caught this
    module's absence: a skip inside the selection is a finding, because a test
    that quietly stops running under a pod-like environment is
    indistinguishable from one that never existed.
    """
    return ReadinessGroups(
        expected_skips={},
        must_pass={},
        staged_role_nodeid=None,
        known_non_environment_skips=(),
        watched=pod_test_selection(),
        refusal_notes={},
    )


def pod_test_environment_files() -> tuple[str, ...]:
    """Files that decide the gate outcome and lie OUTSIDE the harness.

    Same derivation as the search's: the simulator, the two conftests that
    apply to a selection outside `tests/`, and every `.py` under the declared
    selection. A new test is exactly as capable of failing on a pod as a new
    line of production code.
    """
    named = [
        "scripts/shared/pod/simulate_pod_env.sh",
        "conftest.py",
        "scripts/conftest.py",
    ]
    for selection in pod_test_selection():
        root = REPO / selection
        if not root.is_dir():
            raise D1ReplayReadinessError(
                f"the declared pod test selection {selection!r} is not a "
                "directory; digesting a smaller environment than the gate runs "
                "would certify a tree the pod does not have")
        named += [str(p.relative_to(REPO)) for p in sorted(root.rglob("*.py"))]
    missing = [rel for rel in named if not (REPO / rel).is_file()]
    if missing:
        raise D1ReplayReadinessError(
            f"declared pod test-environment sources are missing: {missing}")
    return tuple(sorted(set(named)))


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives. A run id is REQUIRED."""
    if not run_id:
        raise D1ReplayReadinessError(
            "a replay readiness record has no location without a run id: it is "
            "evidence about one attempt, and a shared path is how one attempt's "
            "evidence comes to describe another's tree")
    from shared.run_layout import rel_run_dir

    return (f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id or STAGE_ID)}"
            f"/{RUN_READINESS_ROLE}")


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The replay's executable identity, from the ONE place that binds it.

    `issue_d1_replay_authorization.HARNESS_FILES` is what goes into the
    authorization's `harness_source_digest`, which the launcher re-derives and
    refuses on disagreement. Reading it here means the readiness record and the
    authorization describe the same executable, rather than two lists that can
    drift.

    The search's closure is an AST import walk over its driver and session; the
    replay's is nine declared files, because its closure IS small and a walk
    would add a dependency on the walker for no coverage this cannot state.
    """
    import hashlib
    import json
    import sys

    p = str(REPO / "scripts/autoinit")
    if p not in sys.path:
        sys.path.insert(0, p)
    from stages.phase_d1.issue_d1_replay_authorization import HARNESS_FILES

    root = Path(repo_root)
    missing = [rel for rel in HARNESS_FILES if not (root / rel).is_file()]
    if missing:
        raise D1ReplayReadinessError(
            f"declared replay harness files are missing: {missing}")
    payload = {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in HARNESS_FILES
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"digest": digest, "n_files": len(HARNESS_FILES),
            "files": [{"path": rel, "sha256": sha}
                      for rel, sha in sorted(payload.items())]}


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def bundle_name(commit: str) -> str:
    """The replay fetches the same D1 bundle, from D1's own `TransportSpec`."""
    return D1S.canonical_bundle_name(commit)


def replay_record_contract(run_id: str | None = None,
                           stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="d1_replay_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=pod_test_environment_files(),
        harness_label="D1 replay harness",
    )


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or readiness_groups())


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs) -> tuple[bool, str]:
    kwargs.setdefault("contract", replay_record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    return _pe.load_record(repo_root,
                           record_path=record_path_for(run_id, stage_id))


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str | None = None) -> SweepContract:
    """How to DRIVE a replay readiness sweep, for the generic recorder."""
    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=replay_record_contract(run_id, stage_id),
        groups=readiness_groups(),
        launcher_module=LAUNCHER_MODULE,
        session_id=session_id(),
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="d1_replay_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the D1 REPLAY's own declared pod "
            f"selection — {', '.join(pod_test_selection())} — under the "
            "environment the replay's SessionRunner really hands setup: empty "
            "HOME, isolated HF_HOME, synthetic HF_TOKEN, and every gitignored "
            "artifact THIS session does not stage hidden. The distinction from "
            "the search's contract is the staged view: the search stages three "
            "frozen assets, the replay stages those three plus the resolved "
            "plan its driver is invoked with, and four earlier subruns were "
            "swept through the search's entry and therefore modelled a pod "
            "without that file. RUN-OWNED: no pointer, no global readiness "
            "file, and no claim about any other experiment."),
        pointer_path=None,
    )


__all__ = ["EXPERIMENT_ID", "LAUNCHER_MODULE", "LAUNCH_BOUND",
           "RUN_READINESS_ROLE", "SCHEMA", "STAGE_ID",
           "D1ReplayReadinessError", "bundle_name", "evaluate_sweep",
           "harness", "harness_digest", "load_record",
           "pod_test_environment_files", "pod_test_selection",
           "readiness_groups", "record_path_for", "replay_record_contract",
           "session_id", "sweep_contract", "verify_record"]

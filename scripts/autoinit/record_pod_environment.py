"""Run the pod-like sweep once, and record what it proved.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/record_pod_environment.py \
        --experiment phase_c2 --run-id attempt2 --stage-id 1

This drives the real `scripts/pod/simulate_pod_env.sh` — empty HOME, isolated
`HF_HOME`, synthetic `HF_TOKEN`, gitignored artifacts hidden, the session's own
pytest selection — and writes that run's readiness record.

**One mechanism, one experiment per invocation.** Everything specific to an
experiment — its launcher, session id, harness digest, record schema, record
key names, readiness groups and pointer, if it has one — arrives through the
`SweepContract` its own `pod_environment` module declares. This file held C1's
by name until 2026-09-15: a module-level `c1_harness_digest` import, the literal
session id `autoinit-c1`, the record key `c1_harness_n_files`, a
`c1_readiness_pointer` schema. A second session's only options were to copy
seven hundred lines of scar tissue or to have no readiness record at all.
`--experiment` defaults to `phase_c1`, so every existing invocation still means
what it meant.

It runs the simulator itself rather than accepting somebody's transcript of one,
so the command in the record is literally the command that produced the counts.
A readiness record whose command field was typed by hand is a claim, not
evidence.

The sweep takes seconds — it runs the session's own pod selection, which is one
preflight directory per experiment. It took thirteen minutes until 2026-09-13,
when C1's selection was the whole repository minus four modules, and a $0 probe
measured the same shape for C2: 3976 tests, 907 s. The record still exists
because `verify_record` re-checks in milliseconds that the recorded proof
describes the LIVE executable, which is a different question from re-running the
sweep: a pre-provider gate must not have to run anything while a pod waits.

`--kind` defaults to `diagnostic`, the weaker claim: it says the pod-like suite
passes on this tree and that the machinery works. A `launch_bound` record is the
one a maintainer-approved launch rests on, and is owed when that grant is issued.

Exit 0 when the sweep passes and the record is written; non-zero otherwise, and
the record is still written so the failure is inspectable.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aadistill.runtime.pod_environment import (  # noqa: E402
    evaluate_sweep as _evaluate_sweep,
    head_commit,
    pod_test_environment_digest as _env_digest,
    read_junit,
    self_hash,
    tree_is_clean,
)

from aadistill.runtime.staging_contract import (  # noqa: E402
    derive_contract,
    describe,
    hidden_files,
)

SIMULATOR = "scripts/pod/simulate_pod_env.sh"

#: `--experiment` -> the factory returning that experiment's `SweepContract`.
#:
#: Every value that used to be a top-level import, a literal session id or a
#: hardcoded record key in this file now arrives through one of these. The
#: alternative was a second seven-hundred-line copy of a script whose every
#: comment records a paid failure — and a copy diverges, so the repair for the
#: next failure would land in one of them.
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "phase_c1": ("experiments.phase_c1.pod_environment", "c1_sweep_contract"),
    "phase_c2": ("experiments.phase_c2.pod_environment", "c2_sweep_contract"),
    #: Baseline completion is a THIRD entry rather than a mode of the second.
    #: It binds a different launcher, a different session id, a different
    #: staging contract and a different executable closure, and its record
    #: declares its own schema -- so a Search-1 record cannot satisfy its
    #: verifier and its own cannot satisfy Search-1's. One registry entry is
    #: the whole cost of that separation.
    "phase_c2_baseline_completion": (
        "experiments.phase_c2.baseline_completion_pod_environment",
        "sweep_contract"),
    #: A FOURTH entry, for the same reason the third is not a mode of the
    #: second: the full joint re-search binds its own launcher, session id and
    #: executable closure, and its record declares its own schema, so no other
    #: C2 record can satisfy its verifier or it theirs.
    #:
    #: Without this line the chain is unusable at exactly the step a launch
    #: review authorizes -- the launch-bound sweep could not be DRIVEN, and the
    #: launcher's readiness gate would refuse forever with a message about a
    #: missing record rather than a missing registration. A new experiment's
    #: dispatch entry is the thing this repository has forgotten before:
    #: `SESSION_KIND=phase_b` had no branch in the setup script and cost $0.23.
    "phase_c2_full_search": (
        "experiments.phase_c2.full_search_pod_environment",
        "sweep_contract"),
    #: A FIFTH, and the comment above is why: the replay binds its own launcher,
    #: session id and executable closure, and its record declares its own
    #: schema, so no other C2 record can satisfy its verifier or it theirs. It
    #: also runs its OWN pod selection -- the full search's asserts that
    #: session's staged assets and Search-1's asserts a canonical control this
    #: session does not stage, so either would fail a correct replay tree.
    "phase_c2_replay": (
        "experiments.phase_c2.replay_pod_environment",
        "sweep_contract"),
    #: A SIXTH, and the warning three entries above is why this line exists at
    #: all: without it the behavioural chain is unusable at exactly the step a
    #: launch review authorizes. The launch-bound sweep could not be DRIVEN,
    #: and `readiness_gate` would refuse forever with a message about a missing
    #: record rather than a missing registration.
    #:
    #: It binds its own launcher, its own session id, its own executable
    #: closure and its own pod selection — `tests/c2_behavioural_preflight`,
    #: which the other five would each fail on a correct behavioural tree — and
    #: its record declares its own schema, so no other C2 record can satisfy
    #: its verifier or it theirs.
    "phase_c2_behavioural": (
        "experiments.phase_c2.behavioural_pod_environment",
        "sweep_contract"),
}


def sweep_contract(experiment: str, run_id: str | None, stage_id: str | None,
                   kind: str | None = None):
    """The named experiment's sweep contract, or a refusal naming the choices."""
    import importlib

    if experiment not in EXPERIMENTS:
        raise SystemExit(
            f"--experiment {experiment!r} is not declared. Known: "
            f"{sorted(EXPERIMENTS)}. An experiment declares its readiness "
            "contract in its own `pod_environment` module; this tool holds the "
            "mechanism and none of the instances.")
    module, factory = EXPERIMENTS[experiment]
    build = getattr(importlib.import_module(module), factory)
    try:
        return build(run_id, stage_id, kind)
    except RuntimeError as exc:
        #: An experiment may REQUIRE a run — C2 does, because it has no
        #: repository-level readiness file to fall back to. Reported as a
        #: refusal with the experiment's own reason, not as a traceback.
        raise SystemExit(f"{experiment}: {exc}") from exc


def derive_session(sweep):
    """The session's own SessionSpec: the staged view AND the setup environment.

    Refuses rather than falling back. A readiness sweep that cannot say what this
    session stages, or under what environment it runs, must not run at all —
    falling back to the generic simulator list is the exact failure being
    repaired, and a fallback would reintroduce it the first time this broke.

    The environment comes from `SessionSpec.setup_environment`, the SAME
    production method `SessionRunner._launch` calls, never a reconstruction of a
    subset. Attempt 4's record hashed the manifest and then launched pytest under
    nothing but PODSIM/HIDDEN_PATHS, so the contract was hashed but never
    realized — the child process never saw `SESSION_KIND=c1`, and any test that
    keys on it behaved as though it were some other session.

    `session_commit` is the clean HEAD being swept and `bundle` is the canonical
    name for it, so a pre-authorization diagnostic describes the tree it actually
    ran on. The authorization path is the DECLARED path; no live authorization is
    needed and none is created.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    _sys.path.insert(0, str(REPO_ROOT / "tests/pod"))
    from session_specs import session_args

    launcher = launcher_module(sweep.launcher_module)
    spec = launcher.spec(session_args(launcher))
    head = head_commit(REPO_ROOT)
    setup_env = spec.setup_environment(session_commit=head,
                                       bundle=sweep.bundle_name(head))
    contract = derive_contract(spec.setup, session_id=sweep.session_id)
    return spec, contract, describe(contract, REPO_ROOT), setup_env


@functools.lru_cache(maxsize=4)
def launcher_module(name: str):
    """The launcher module, loaded once per name.

    Shared by the session derivation and the host-local scan so both describe
    the same launcher, and memoized because `load_session_launcher` builds a
    fresh module object each call.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    _sys.path.insert(0, str(REPO_ROOT / "tests/pod"))
    from session_specs import load_session_launcher
    return load_session_launcher(name)


def gate_documents(launcher, repo_root: Path) -> list[str]:
    """Every committed JSON document the launcher's own gates read.

    Taken from the launcher module's path constants rather than from a list kept
    here, so a gate added tomorrow with a new document is covered the day it
    lands instead of the day a pod fails on it.
    """
    out = set()
    for name, value in vars(launcher).items():
        if (name.isupper() and isinstance(value, str) and value.endswith(".json")
                and not value.startswith("/") and (repo_root / value).is_file()):
            out.add(value)
    return sorted(out)


def host_local_stores(launcher, repo_root: Path) -> list[str]:
    """Absolute paths OUTSIDE the repository that those documents name.

    A path like this is a host-local store: something the machine that froze an
    asset holds, and that no pod receives, because a session stages an asset's
    BYTES and never the store they were frozen in. `hidden_files` cannot see
    one — it is the complement of the manifest within the repository's own
    gitignored set, and a store outside the tree is in neither term.

    That blind spot cost C1 attempt 14 $0.40 and 22 minutes: `battery_staged_gate`
    compares the staged battery against `battery.json`'s `canonical_path`, the
    dev box had it, the launch-bound sweep passed, and the pod — which by design
    has the bytes and not the store — failed the gate test at its CPU gate.

    Derived from the documents, not listed here, and restricted to paths that
    actually exist: hiding is a rename, and a rename of something absent is a
    no-op the simulator already skips.
    """
    def strings(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from strings(v)
        elif isinstance(node, str):
            yield node

    root = repo_root.resolve()
    found = set()
    for rel in gate_documents(launcher, repo_root):
        try:
            doc = json.loads((repo_root / rel).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for s in strings(doc):
            #: A command line is not a path. Anything with whitespace is prose or
            #: an invocation, and several plan documents record both.
            if not s.startswith("/") or any(c.isspace() for c in s):
                continue
            p = Path(s)
            try:
                p.resolve().relative_to(root)
            except ValueError:
                if p.exists():
                    found.add(s)
    return sorted(found)
#: One directory per sweep, named by the tree the sweep is ABOUT.
#:
#: Both of these were fixed paths, so every sweep overwrote the previous one's
#: raw output -- including a sweep whose record is cited as evidence, whose
#: `evidence.junit` and `evidence.pytest_log` then pointed at a different
#: sweep's results. The record survived; what it referenced did not.
#:
#: Keyed on the head commit rather than a timestamp: two sweeps of the SAME tree
#: are the same claim and may share a directory, while a sweep of a different
#: tree is a different claim and gets its own. A timestamp would also make the
#: path unreproducible, and a record that names an unreproducible path cannot be
#: checked later.
SWEEP_ROOT = "/home/ecs-user/aad-scratch/podsim"


def sweep_dir(head: str, *, root: str = SWEEP_ROOT) -> Path:
    """Where executions of THIS tree live. Source identity, not execution."""
    return Path(root) / f"sweep-{head[:12]}"


def allocate_execution(head: str, *, root: str = SWEEP_ROOT
                       ) -> tuple[str, str, str]:
    """Claim a fresh execution directory. `(execution_id, junit, log)`.

    **Source identity and execution identity are different things.** Keying the
    path on the commit alone said "the same tree is the same claim", and that is
    wrong: a sweep that failed and a sweep that then passed are two observations
    of one tree, and the second overwrote the first. The simulator redirects with
    `>`, so the earlier `junit.xml` and `pytest.log` were truncated, including
    ones a committed record cites as its evidence.

    `mkdir(exist_ok=False)` IS the allocation, so two executions cannot claim the
    same directory even concurrently, and an existing directory is never entered.
    The id is recorded in the record together with the paths; nothing has to
    re-derive a filename from a commit.
    """
    base = sweep_dir(head, root=root)
    n = 1
    while True:
        d = base / f"exec-{n:03d}"
        try:
            d.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            n += 1
            if n > 9999:
                raise SystemExit(f"{base} holds 9999 executions; refusing")
            continue
        return (f"sweep-{head[:12]}/exec-{n:03d}",
                str(d / "junit.xml"), str(d / "pytest.log"))


def existing_executions(head: str, *, root: str = SWEEP_ROOT) -> list[str]:
    """Execution directories already recorded for this tree. READ ONLY."""
    base = sweep_dir(head, root=root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def min_free_gib() -> int:
    """How much room THIS sweep needs before it moves anything.

    Derived here because this is the caller that knows the footprint, and
    derived from a measurement rather than chosen: on 2026-09-11 one full suite
    left about 3.5 GiB under `/tmp/pytest-of-*`. Doubling that covers a run that
    keeps more; the headroom covers the estimate simply being wrong, which is
    the case that matters, because being wrong here does not fail the sweep --
    it displaces the repository's gitignored artifacts and reports nothing.

    **This figure describes the current suite on the current tree.** It is not a
    standing disk policy and must not be inherited by an operation with a
    different footprint: a larger student, a longer battery or a sweep that
    retains more per test needs its own number, measured the same way. Override
    with `AAD_PODSIM_MIN_FREE_GIB` when the workload is known to differ, which
    is the supported way to say so rather than editing this function.
    """
    override = os.environ.get("AAD_PODSIM_MIN_FREE_GIB")
    if override:
        return int(override)
    measured_tmp_gib = 4      # observed peak of one full-suite tmp_path tree
    headroom_gib = 16         # what a wrong estimate must not be able to eat
    return measured_tmp_gib * 2 + headroom_gib


def check_invocation_matches(contract, setup_env, pytest_cmd, child_env):
    """Refuse a PASS when the declaration and the invocation disagree.

    This is the attempt-4 failure mode expressed as code. That record hashed a
    SetupManifest and then launched pytest under an environment built from
    nothing but PODSIM variables, so the contract was *hashed but never
    realized*: the child never saw `SESSION_KIND=c1`, the staged view was the
    generic default, and the recorded pytest command was a hand-written string
    naming two ignores while the sweep ran a different selection. Every one of
    those could disagree with the manifest and nothing noticed.

    So the three facts a manifest declares -- test selection, environment,
    staging -- are each compared against what was actually handed to the
    subprocess, and any mismatch becomes a `problem`, which forces the verdict to
    FAIL before a record can be written.
    """
    problems: list[str] = []

    declared_ignores = list(contract["test_ignores"])
    invoked_ignores = re.findall(r"--ignore=(\S+)", pytest_cmd)
    if invoked_ignores != declared_ignores:
        problems.append(
            f"test selection mismatch: the manifest declares {declared_ignores} "
            f"and the invocation passes {invoked_ignores}")

    env_mismatch = {k: (v, child_env.get(k)) for k, v in setup_env.items()
                    if child_env.get(k) != v}
    if env_mismatch:
        problems.append(
            f"setup environment mismatch between the manifest and the child "
            f"process: {sorted(env_mismatch)}")

    if child_env.get("HIDDEN_PATHS") is None:
        problems.append("no derived HIDDEN_PATHS was passed; the simulation would "
                        "fall back to the generic default")
    if child_env.get("PODSIM_CMD") != pytest_cmd:
        problems.append("PODSIM_CMD is not the command this record describes")

    return {
        "declared_test_ignores": declared_ignores,
        "invoked_test_ignores": invoked_ignores,
        "setup_environment_keys": sorted(setup_env),
        "setup_environment_realized": not env_mismatch,
        "staging_contract_digest": contract["digest"],
        "hidden_paths_passed": child_env.get("HIDDEN_PATHS") is not None,
        "problems": problems,
        "rule": ("a manifest fact that is hashed but not realized in the "
                 "invocation is refused here, before any PASS record exists"),
    }


def _sha256_of(path: str | None) -> str | None:
    """Hash a raw-output file, or `None` when it is not there.

    Recorded so "this record's evidence" is checkable rather than asserted: a
    path can be right and its contents replaced, which is exactly what the
    shared-path scheme did.
    """
    import hashlib

    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pointer_for(sweep, run_id: str, stage_id: str, record: dict,
                record_rel: str) -> dict:
    """The navigation document, shared by the sweep and by `--repoint`.

    One shape written from two places was how the pointer came to disagree with
    the run it names: it said `FAIL` at `c5dfd9ea` while `attempt13`'s own record
    said `PASS` at `5aafc549`.
    """
    return {
        "schema": sweep.pointer_schema,
        "_what_this_is": (
            "a pointer to the run that owns the live readiness record. NOT "
            "the record: this file used to be the record, which meant each "
            "run overwrote the evidence the previous one launched under."),
        "run_id": run_id,
        "stage_id": stage_id,
        "record": record_rel,
        "record_self_sha256": record.get("self_sha256"),
        "record_kind": record.get("record_kind"),
        "verdict": record.get("verdict"),
        "swept_base_commit": record.get("swept_base_commit"),
        "history": sweep.pointer_history,
        "_it_lags_by_design": (
            "a launch_bound sweep writes ONLY the run-owned record: rewriting "
            "this tracked file would put an unpermitted change into the tree it "
            "just certified, and the session-lineage rule would refuse the "
            "launch. So between a launch-bound sweep and the launch it feeds, "
            "the summary here describes the PREVIOUS state of that run. The "
            "record named above is the authority; `--repoint` refreshes this at "
            "the next convergence."),
        "authorizes": "nothing",
    }


def repoint(sweep, root: Path) -> int:
    """Point the experiment's pointer at its newest run-owned record.

    Runs nothing. A `launch_bound` sweep leaves the pointer alone on purpose: it
    is tracked, and rewriting it would put an unpermitted change into the very
    tree the sweep is certifying. The cost is that navigation lags by one run,
    and nothing was closing that gap — the pointer still named attempt 13 after
    attempts 13 and 14 had both been swept, closed and consumed.

    **Scoped to `sweep.experiment_id`.** The glob was `logs/stages/*/*/runs/*`,
    across every experiment, which was harmless while exactly one experiment had
    run-owned readiness records and became wrong the moment a second did: C2's
    newest record would have been written into C1's pointer, under C1's schema,
    as C1's live readiness evidence.
    """
    if not sweep.pointer_path:
        print(f"{sweep.experiment_id} declares no readiness pointer: its record "
              "is run-owned and there is nothing to repoint")
        return 0
    records = sorted(
        root.glob(f"logs/stages/*/{sweep.experiment_id}/runs/*/governance/"
                  "readiness.json"),
        key=lambda p: (int("".join(c for c in p.parents[1].name if c.isdigit())
                           or -1), p.parents[1].name))
    if not records:
        print("no run-owned readiness record exists; pointer left alone")
        return 0
    newest = records[-1]
    doc = json.loads(newest.read_text())
    rel = newest.relative_to(root).as_posix()
    #: logs/stages/stage-<id>/<experiment>/runs/<run>/governance/readiness.json
    #: The run is `parents[1]`, the same handle the sort above uses. Reading it
    #: positionally out of `parts` put "runs" in the `run_id` field.
    stage_id = newest.relative_to(root).parts[2].split("-", 1)[1]
    pointer = pointer_for(sweep, newest.parents[1].name, stage_id, doc, rel)
    p = root / sweep.pointer_path
    body = json.dumps(pointer, indent=1) + "\n"
    if p.is_file() and p.read_text() == body:
        print(f"pointer already names {rel}")
        return 0
    p.write_text(body)
    print(f"pointer -> {rel} ({doc.get('record_kind')} {doc.get('verdict')})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="phase_c1",
                    choices=sorted(EXPERIMENTS),
                    help="whose readiness this sweep records. Defaults to "
                         "phase_c1, so every existing invocation of this tool "
                         "means exactly what it meant before.")
    #: Defaults are DERIVED per sweep, below, once the head commit is known.
    ap.add_argument("--junit", default=None,
                    help="raw JUnit path. Defaults to a freshly ALLOCATED "
                         "execution directory under this tree's sweep "
                         "directory; an explicit path that already holds "
                         "output is refused rather than overwritten")
    ap.add_argument("--log", default=None)
    ap.add_argument("--from-existing", action="store_true",
                    help="READ ONLY: parse a sweep that already ran. Requires "
                         "--junit, allocates nothing and executes nothing")
    ap.add_argument("--run-id", default=None,
                    help="the run this readiness evidence belongs to. With it "
                         "the record is written into that run's governance "
                         "area and the repository-root file becomes a pointer; "
                         "without it the root file is the record, which is "
                         "where every pre-2026-09-12 sweep is.")
    ap.add_argument("--stage-id", default=None,
                    help="the run's declared stage; required with --run-id")
    ap.add_argument("--repoint", action="store_true",
                    help="MAINTENANCE, runs nothing: rewrite the repository-root "
                         "pointer from the newest run-owned record that already "
                         "exists. A launch_bound sweep deliberately leaves the "
                         "pointer alone -- rewriting a tracked file mid-sweep "
                         "puts an unpermitted change into the tree it swept -- "
                         "so navigation lags by one run until this is run. Do it "
                         "BEFORE a sweep, never after.")
    ap.add_argument("--kind", default="diagnostic",
                    choices=("diagnostic", "launch_bound"),
                    help=("`diagnostic` proves the machinery on the current tree; "
                          "`launch_bound` is the sweep a maintainer-approved "
                          "launch rests on. Default is deliberately the weaker "
                          "claim."))
    args = ap.parse_args()

    # Captured BEFORE the record is written: writing it into logs/ is itself a
    # tree modification, and the verdict must describe the tree that was swept.
    if args.run_id and not args.stage_id:
        raise SystemExit("--run-id needs --stage-id: the run's location is "
                         "derived from the stage its experiment declares, and "
                         "guessing it would put the evidence in a second place")
    sweep = sweep_contract(args.experiment, args.run_id, args.stage_id,
                           args.kind)

    if args.repoint:
        return repoint(sweep, REPO_ROOT)

    record_rel = sweep.record.record_path

    clean_before = tree_is_clean(REPO_ROOT)
    head = head_commit(REPO_ROOT)
    #: Resolved now that the tree's identity is known. A caller may still name
    #: its own paths; what it may not get is a shared default that silently
    #: replaces another sweep's raw output.
    #: Reading an existing result and producing a new one are separate paths.
    #: They shared one branch, so `--from-existing` still resolved a DEFAULT
    #: output path -- which for a fresh default would have been an empty
    #: directory it had just created.
    if args.from_existing:
        if not args.junit:
            raise SystemExit(
                "--from-existing parses a sweep that already ran and cannot "
                "guess which one. Name it with --junit. Executions of this "
                f"tree: {existing_executions(head) or 'none'}")
        if not Path(args.junit).is_file():
            raise SystemExit(f"--from-existing: {args.junit} does not exist")
        args.log = args.log or str(Path(args.junit).with_name("pytest.log"))
        execution_id = f"imported:{Path(args.junit).parent.name}"
    elif args.junit is None and args.log is None:
        execution_id, args.junit, args.log = allocate_execution(head)
        print(f"execution {execution_id}")
    else:
        #: A caller may name its own paths; what it may not do is write over
        #: output that already exists, which is how a cited record lost the
        #: evidence it pointed at.
        for named in (args.junit, args.log):
            if named and Path(named).exists():
                raise SystemExit(
                    f"{named} already exists. A new execution never overwrites "
                    "an existing one; use --from-existing to read it, or name "
                    "an unused path.")
        args.log = args.log or str(Path(args.junit).with_name("pytest.log"))
        execution_id = f"explicit:{Path(args.junit).parent.name}"
    Path(args.junit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    harness = sweep.harness(REPO_ROOT)
    env_digest = _env_digest(REPO_ROOT,
                             named_files=sweep.record.named_files)

    # --- the staged view, DERIVED from the session that will be launched -----
    #
    # Attempt 4's sweep used simulate_pod_env.sh's generic default HIDDEN_PATHS,
    # a hand-maintained complement whose own comment claimed every pod session
    # stages artifacts/stage3/corpus_v2. C1 stages no such thing, so the sweep
    # modelled a machine 55 tests more generous than the pod and certified a tree
    # that then failed six ways for $0.6986. The visible set now comes from the
    # same SetupManifest the SessionRunner launches, and the hidden set is
    # computed as its complement rather than declared.
    spec, contract, staged_view, setup_env = derive_session(sweep)
    #: The complement inside the repository, plus the host-local stores outside
    #: it that the session's own documents name. Both are things this machine
    #: holds and the pod does not; only the first was modelled until attempt 14
    #: paid to find the second.
    in_tree = hidden_files(contract, REPO_ROOT)
    host_local = host_local_stores(launcher_module(sweep.launcher_module),
                                   REPO_ROOT)
    hidden = [*in_tree, *host_local]
    pytest_cmd = (".venv/bin/python -m pytest tests/ -q "
                  + " ".join(f"--ignore={i}" for i in contract["test_ignores"]))
    # The production setup environment is MERGED IN, so the child pytest runs
    # under SESSION_KIND=c1 and the rest of what the pod is given.
    env = {**os.environ, **setup_env,
           "PODSIM_JUNIT": args.junit, "PODSIM_LOG": args.log,
           "HIDDEN_PATHS": "\n".join(hidden), "PODSIM_CMD": pytest_cmd,
           # The interpreter the simulator uses to emit the CPU-test contract.
           # Explicit, because on a pod there is no repo venv and the ambient
           # fallback that used to cover that gap cost attempt 6 its CPU gate.
           "PODSIM_PYTHON": sys.executable,
           # Derived from THIS sweep's own footprint rather than guessed, and
           # passed from here because this is where the footprint is known.
           "PODSIM_MIN_FREE_GIB": str(min_free_gib())}
    command = (f"<SessionSpec.setup_environment: {len(setup_env)} keys> "
               f"HIDDEN_PATHS=<{len(hidden)} derived paths, contract "
               f"{contract['digest'][:12]}> PODSIM_CMD=<derived> "
               f"PODSIM_JUNIT={args.junit} PODSIM_LOG={args.log} bash {SIMULATOR}")
    print(f"staging contract {contract['digest'][:12]}… — "
          f"{staged_view['n_staged_files']} staged visible, {len(hidden)} hidden; "
          f"{len(setup_env)} setup env keys incl SESSION_KIND="
          f"{setup_env.get('SESSION_KIND')}; ignores {contract['test_ignores']}")

    # THE ATTEMPT-4 FAILURE MODE, refused before a PASS can be written: the
    # declaration and the invocation must agree. Hashing a contract the sweep did
    # not actually run under is what produced a green record for a machine 55
    # tests more generous than the pod.
    realization = check_invocation_matches(contract, setup_env, pytest_cmd, env)
    started = time.time()
    if args.from_existing:
        rc, seconds = 0, 0.0
        print(f"parsing the existing sweep at {args.junit}")
    else:
        print(f"running: {command}")
        # A sweep supersedes the record it is about to replace, and the suite it
        # runs CONTAINS the two tests that verify that record. Leaving the old one
        # in place makes them assert a stale artifact against the tree being
        # swept, and they fail — which is a statement about the previous sweep,
        # not this one. Move it aside for the duration; the tests skip when it is
        # absent, which is the honest reading of "not yet recorded".
        stash = None
        live = REPO_ROOT / record_rel
        try:
            if live.is_file():
                stash = Path(tempfile.mkdtemp(prefix="podsim-record-")) / live.name
                shutil.move(str(live), str(stash))
                print(f"moved the previous record aside -> {stash}")
            proc = subprocess.run(["bash", str(REPO_ROOT / SIMULATOR)],
                                  cwd=str(REPO_ROOT), env=env)
            rc = proc.returncode
        finally:
            # Restored only if this run does not go on to write a new one; the
            # write below overwrites it either way, so a failed sweep leaves the
            # tree exactly as it found it.
            if stash and stash.is_file() and not live.exists():
                shutil.move(str(stash), str(live))
                print("restored the previous record")
        seconds = round(time.time() - started, 1)

    if not Path(args.junit).is_file():
        # An interrupted sweep leaves no report. Say so plainly and write
        # nothing: a readiness record is evidence, and half a sweep is not.
        print(f"\nNO RECORD WRITTEN: {args.junit} does not exist — the sweep did "
              f"not finish (simulator exit {rc}). Re-run it.")
        return 2

    junit = read_junit(args.junit, REPO_ROOT)
    findings = _evaluate_sweep(junit["outcomes"], junit.get("skip_reasons"),
                               groups=sweep.groups)

    record = {
        # The wire format is the SESSION's, read off the contract the verifier
        # will check against. Writing one string here and comparing another
        # somewhere else is how a record and its gate come to disagree.
        "schema": sweep.record.schema,
        "_what_this_is": sweep.what_this_is,
        #: THE commit the sweep ran on, clean. `pod_environment_gate` requires
        #: the session commit to descend from it with no tracked change beyond
        #: the readiness record (and, once issued, the authorization artifact).
        #: Captured before the sweep starts, when the tree is verified clean.
        "swept_base_commit": head,
        "record_kind": args.kind,
        "staging_contract_digest": contract["digest"],
        "staging_contract": staged_view,
        "staging_contract_rule": (
            "derived from spec(args).setup -- the same SetupManifest the "
            "SessionRunner launches. Relay inputs are modelled at FILE "
            "granularity (a RelayInput stages one named file into its dest, not "
            "the directory), local assets as whole trees. The hidden set is the "
            "computed complement, never a declared list."),
        "record_kind_note": (
            "diagnostic: proves the pod-like suite passes on this exact tree and "
            "that the readiness machinery works. A launch-bound record is the one "
            "a maintainer-approved launch rests on. It is produced on the final "
            "CLEAN PRE-AUTHORIZATION tree -- after the grant and all metadata are "
            "committed, BEFORE the authorization is issued -- because a sweep run "
            "after issuance adds a second path to the lineage diff and "
            "session_commit_gate refuses."),
        "tree_clean": clean_before,
        sweep.record.harness_field: harness["digest"],
        sweep.harness_n_files_field: harness["n_files"],
        "pod_test_environment_digest": env_digest["digest"],
        "pod_test_environment_n_files": env_digest["n_files"],
        "pod_test_environment_named_files": env_digest["named_files"],
        "binding_rule": (
            "this record binds the EXECUTABLE via two digests, and the REPOSITORY "
            "via swept_base_commit lineage. The digests ignore logs/ and docs/ so "
            "that paperwork does not invalidate a sweep; the lineage rule then "
            "permits exactly two post-sweep tracked paths -- this record, and (in "
            "an issued session) the canonical authorization artifact -- because "
            "the pod suite READS repository state and a documentation commit can "
            "change what it asserts."),
        "simulator": SIMULATOR,
        "simulator_command": command,
        "environment": {
            "home_mode": "empty — a fresh directory created by the simulator",
            "hf_home_mode": "isolated — $PODSIM_ENV_ROOT/hf, outside the dev cache",
            "hf_hub_cache_resolution": "$HF_HOME/hub, set explicitly as HF_HUB_CACHE",
            "token_mode": "synthetic, non-empty; never printed, never a real credential",
            "dev_hf_cache_visible": False,
            "artifact_hiding": (
                f"manifest-derived positive staged view: {staged_view['n_staged_files']} "
                f"files visible from the C1 SetupManifest, {len(in_tree)} hidden as "
                f"the computed complement and {len(host_local)} host-local store(s) "
                f"hidden as well. NOT the simulator's generic default."),
            "host_local_stores_hidden": host_local,
            "host_local_rule": (
                "an absolute path outside the repository named by a document the "
                "launcher's gates read. The session stages an asset's bytes, never "
                "the store it was frozen in, so a pod has none of these. Derived "
                "from those documents rather than listed, and hidden by rename for "
                "the duration of the sweep."),
        },
        # THE command handed to PODSIM_CMD, not a transcription of one. A record
        # that restates the command cannot be checked against the JUnit it claims
        # to describe; attempt 4's said two ignores while the sweep ran a
        # different selection entirely.
        "pytest_command": pytest_cmd,
        "pytest_command_note": (
            "verbatim from PODSIM_CMD. Test SELECTION is the session's own "
            "SESSION_TEST_IGNORES; --junitxml is appended by the simulator as a "
            "reporting flag so every skip and pass can be named exactly."),
        "setup_environment": {k: v for k, v in sorted(setup_env.items())},
        "setup_environment_source": (
            "SessionSpec.setup_environment(session_commit=<swept HEAD>, "
            "bundle=canonical_bundle_name(<swept HEAD>)) -- the same production "
            "method SessionRunner._launch calls. Not a reconstructed subset. No "
            "secret is stored: none of these keys carries one."),
        "invocation_realized": realization,
        "simulator_exit_code": rc,
        "seconds": seconds,
        "counts": findings["counts"],
        "n_tests": junit["total"],
        "failed_nodeids": findings["failed_nodeids"],
        # The WHOLE findings block, not a hand-picked subset. Cherry-picking is
        # how the record came to omit the battery, host-local and dev-box skip
        # groups while `evaluate_sweep` was computing them: a new group had to be
        # remembered in two places, and the second was forgotten. Same failure
        # shape as the transcribed pytest command.
        "findings": findings,
        "unexpected_environment_skips": findings["unexpected_environment_skips"],
        "expected_environment_skips": findings["expected_environment_skips"],
        "problems": findings["problems"],
        "verdict": findings["verdict"] if rc == 0 else "FAIL",
        #: The execution's OWN identity and its own raw output, with hashes so
        #: a later reader can verify that what it finds is what was recorded.
        #: Two executions of one commit are two observations, and each keeps
        #: its own; the paths are recorded rather than re-derived from the
        #: commit, because a commit does not identify an execution.
        "execution_id": execution_id,
        "evidence": {"junit": args.junit, "pytest_log": args.log,
                     "junit_sha256": _sha256_of(args.junit),
                     "pytest_log_sha256": _sha256_of(args.log),
                     "executions_of_this_tree": existing_executions(head)},
        #: Whatever else this experiment states about its own readiness. C1
        #: names the renderer-parity validation here; C2 has no renderer.
        **dict(sweep.extra_record_fields),
    }
    if realization["problems"]:
        record["problems"] = list(record["problems"]) + realization["problems"]
        record["verdict"] = "FAIL"
    if rc != 0 and not record["problems"]:
        record["problems"] = [f"the simulator exited {rc} with no failing nodeid"]
    record["self_sha256"] = self_hash(record)

    #: Into the run that owns it. A run's readiness evidence is part of that
    #: run, not a repository-level file the next run replaces.
    out = REPO_ROOT / record_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    #: A launch-bound sweep writes the RUN-OWNED RECORD AND NOTHING ELSE.
    #:
    #: The pointer is tracked; the record is the run's. Rewriting the pointer
    #: during a launch-bound sweep put a tracked change into the tree that the
    #: session-lineage rule does not permit after a sweep -- and the rule is
    #: deliberately narrow, because a grant committed after a sweep must still
    #: invalidate it. That made the chain unsatisfiable in both directions:
    #: leaving the pointer dirty fails the issuer's clean-tree requirement, and
    #: committing it fails lineage, with no sweep able to converge because each
    #: one writes its own swept_base_commit into the pointer.
    #:
    #: So the pointer is navigation, not the authority for one formal session.
    #: The launch gate does not read it -- `pod_environment_gate` resolves the
    #: run-owned record from run_id and stage_id -- and ordinary state
    #: maintenance may refresh it once launch lineage no longer depends on the
    #: pre-authorization tree.
    #: An experiment with no pointer has nothing to keep in step: its record is
    #: run-owned and that is the only place it lives.
    if sweep.pointer_path and args.run_id and args.kind != "launch_bound":
        #: A POINTER, not a second record: it says where the live evidence is
        #: and what it hashes to, so one stable path still answers "which sweep
        #: is current" without becoming a copy that can drift.
        pointer = pointer_for(sweep, args.run_id, args.stage_id, record,
                              record_rel)
        (REPO_ROOT / sweep.pointer_path).write_text(
            json.dumps(pointer, indent=1) + "\n")
        print(f"pointer: {sweep.pointer_path} -> {record_rel}")
    elif sweep.pointer_path and args.run_id:
        print(f"pointer: {sweep.pointer_path} left UNCHANGED — a launch_bound "
              "sweep writes only the run-owned record, so the tree it swept "
              "stays the tree it describes")

    c = record["counts"]
    print(f"\n{record['verdict']}: {c['passed']} passed, {c['skipped']} skipped, "
          f"{c['failed']} failed, {c['error']} error  (rc={rc}, {seconds}s)")
    for p in record["problems"]:
        print(f"  problem: {p}")
    print(f"record: {record_rel} ({record['self_sha256'][:12]}…)")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the pod-like sweep once, and record what it proved.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/record_pod_environment.py

This drives the real `scripts/pod/simulate_pod_env.sh` — empty HOME, isolated
`HF_HOME`, synthetic `HF_TOKEN`, gitignored artifacts hidden, the pod's own pytest
selection — and writes `logs/c1_pod_environment_verification.json`.

It runs the simulator itself rather than accepting somebody's transcript of one,
so the command in the record is literally the command that produced the counts.
A readiness record whose command field was typed by hand is a claim, not
evidence.

The sweep takes about thirteen minutes. That is why the record exists:
`aadistill.autoinit.pod_environment.verify_record` re-checks in milliseconds that
the recorded proof still describes the live executable, so a pre-provider gate
never has to re-run this while a pod waits.

`--kind` defaults to `diagnostic`, the weaker claim: it says the pod-like suite
passes on this tree and that the machinery works. A `launch_bound` record is the
one a maintainer-approved launch rests on, and is owed when that grant is issued.

Exit 0 when the sweep passes and the record is written; non-zero otherwise, and
the record is still written so the failure is inspectable.
"""

from __future__ import annotations

import argparse
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

from aadistill.autoinit.c1_authorization import c1_harness_digest  # noqa: E402
from aadistill.autoinit.pod_environment import (  # noqa: E402
    LEAF_TRANSPORT_NODEIDS, RECORD_PATH, RENDERER_PARITY_NODEIDS, SCHEMA,
    evaluate_sweep, head_commit, pod_test_environment_digest, read_junit,
    self_hash, tree_is_clean,
)

from aadistill.autoinit.staging_contract import (  # noqa: E402
    derive_contract, describe, hidden_files,
)

SIMULATOR = "scripts/pod/simulate_pod_env.sh"


def derive_c1_session():
    """C1's own SessionSpec: the staged view AND the setup environment.

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
    from session_specs import load_session_launcher, session_args
    from aadistill.autoinit.c1_bundle import canonical_bundle_name

    launcher = load_session_launcher("autoinit_c1_launch")
    spec = launcher.spec(session_args(launcher))
    head = head_commit(REPO_ROOT)
    setup_env = spec.setup_environment(session_commit=head,
                                       bundle=canonical_bundle_name(head))
    contract = derive_contract(spec.setup, session_id="autoinit-c1")
    return spec, contract, describe(contract, REPO_ROOT), setup_env
DEFAULT_JUNIT = "/home/ecs-user/aad-scratch/podsim_junit.xml"
DEFAULT_LOG = "/home/ecs-user/aad-scratch/podsim_pytest.log"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--junit", default=DEFAULT_JUNIT)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--from-existing", action="store_true",
                    help="parse a sweep that already ran instead of running one")
    ap.add_argument("--kind", default="diagnostic",
                    choices=("diagnostic", "launch_bound"),
                    help=("`diagnostic` proves the machinery on the current tree; "
                          "`launch_bound` is the sweep a maintainer-approved "
                          "launch rests on. Default is deliberately the weaker "
                          "claim."))
    args = ap.parse_args()

    # Captured BEFORE the record is written: writing it into logs/ is itself a
    # tree modification, and the verdict must describe the tree that was swept.
    clean_before = tree_is_clean(REPO_ROOT)
    head = head_commit(REPO_ROOT)
    harness = c1_harness_digest(REPO_ROOT)
    env_digest = pod_test_environment_digest(REPO_ROOT)

    # --- the staged view, DERIVED from the session that will be launched -----
    #
    # Attempt 4's sweep used simulate_pod_env.sh's generic default HIDDEN_PATHS,
    # a hand-maintained complement whose own comment claimed every pod session
    # stages artifacts/stage3/corpus_v2. C1 stages no such thing, so the sweep
    # modelled a machine 55 tests more generous than the pod and certified a tree
    # that then failed six ways for $0.6986. The visible set now comes from the
    # same SetupManifest the SessionRunner launches, and the hidden set is
    # computed as its complement rather than declared.
    spec, contract, staged_view, setup_env = derive_c1_session()
    hidden = hidden_files(contract, REPO_ROOT)
    pytest_cmd = (".venv/bin/python -m pytest tests/ -q "
                  + " ".join(f"--ignore={i}" for i in contract["test_ignores"]))
    # The production setup environment is MERGED IN, so the child pytest runs
    # under SESSION_KIND=c1 and the rest of what the pod is given.
    env = {**os.environ, **setup_env,
           "PODSIM_JUNIT": args.junit, "PODSIM_LOG": args.log,
           "HIDDEN_PATHS": "\n".join(hidden), "PODSIM_CMD": pytest_cmd}
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
        print(f"running: {command}\n(this takes roughly 13 minutes)")
        # A sweep supersedes the record it is about to replace, and the suite it
        # runs CONTAINS the two tests that verify that record. Leaving the old one
        # in place makes them assert a stale artifact against the tree being
        # swept, and they fail — which is a statement about the previous sweep,
        # not this one. Move it aside for the duration; the tests skip when it is
        # absent, which is the honest reading of "not yet recorded".
        stash = None
        live = REPO_ROOT / RECORD_PATH
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
    findings = evaluate_sweep(junit["outcomes"])

    record = {
        "schema": SCHEMA,
        "_what_this_is": (
            "one complete pod-like sweep of the CPU test suite: the condition a "
            "fresh C1 pod is actually in, which is what C1 attempt 3R's setup "
            "test gate refused for $0.3482 with zero scientific stages run."),
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
            "a maintainer-approved C1 launch rests on. It is produced on the final "
            "CLEAN PRE-AUTHORIZATION tree -- after the grant and all metadata are "
            "committed, BEFORE the authorization is issued -- because a sweep run "
            "after issuance adds a second path to the lineage diff and "
            "session_commit_gate refuses."),
        "tree_clean": clean_before,
        "c1_harness_digest": harness["digest"],
        "c1_harness_n_files": harness["n_files"],
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
                f"files visible from the C1 SetupManifest, {len(hidden)} hidden as "
                f"the computed complement. NOT the simulator's generic default."),
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
        "expected_renderer_parity_skips": list(RENDERER_PARITY_NODEIDS),
        "leaf_transport_nodeids": list(LEAF_TRANSPORT_NODEIDS),
        "renderer_parity_skipped_as_expected":
            findings["renderer_parity_skipped_as_expected"],
        "leaf_transport_all_passed": findings["leaf_transport_all_passed"],
        "unexpected_environment_skips": findings["unexpected_environment_skips"],
        "expected_environment_skips": findings["expected_environment_skips"],
        "problems": findings["problems"],
        "verdict": findings["verdict"] if rc == 0 else "FAIL",
        "evidence": {"junit": args.junit, "pytest_log": args.log},
        "renderer_parity_is_proved_by": "logs/c1_renderer_parity.json",
    }
    if realization["problems"]:
        record["problems"] = list(record["problems"]) + realization["problems"]
        record["verdict"] = "FAIL"
    if rc != 0 and not record["problems"]:
        record["problems"] = [f"the simulator exited {rc} with no failing nodeid"]
    record["self_sha256"] = self_hash(record)

    out = REPO_ROOT / RECORD_PATH
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    c = record["counts"]
    print(f"\n{record['verdict']}: {c['passed']} passed, {c['skipped']} skipped, "
          f"{c['failed']} failed, {c['error']} error  (rc={rc}, {seconds}s)")
    for p in record["problems"]:
        print(f"  problem: {p}")
    print(f"record: {RECORD_PATH} ({record['self_sha256'][:12]}…)")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

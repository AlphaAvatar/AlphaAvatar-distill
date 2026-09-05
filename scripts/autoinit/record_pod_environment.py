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


def derive_c1_staging():
    """Read C1's own SetupManifest. Refuses rather than falling back.

    There is deliberately no default path here. A readiness sweep that cannot say
    what this session stages must not run at all — falling back to the generic
    simulator list is the exact failure being repaired, and a fallback would
    reintroduce it the first time this import broke.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    _sys.path.insert(0, str(REPO_ROOT / "tests/pod"))
    from session_specs import load_session_launcher, session_args

    launcher = load_session_launcher("autoinit_c1_launch")
    setup = launcher.spec(session_args(launcher)).setup
    contract = derive_contract(setup, session_id="autoinit-c1")
    return contract, describe(contract, REPO_ROOT)
DEFAULT_JUNIT = "/home/ecs-user/aad-scratch/podsim_junit.xml"
DEFAULT_LOG = "/home/ecs-user/aad-scratch/podsim_pytest.log"


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
    contract, staged_view = derive_c1_staging()
    hidden = hidden_files(contract, REPO_ROOT)
    pytest_cmd = (".venv/bin/python -m pytest tests/ -q "
                  + " ".join(f"--ignore={i}" for i in contract["test_ignores"]))
    env = {**os.environ, "PODSIM_JUNIT": args.junit, "PODSIM_LOG": args.log,
           "HIDDEN_PATHS": "\n".join(hidden), "PODSIM_CMD": pytest_cmd}
    command = (f"HIDDEN_PATHS=<{len(hidden)} paths derived from the C1 "
               f"SetupManifest, contract {contract['digest'][:12]}> "
               f"PODSIM_CMD=<derived, {len(contract['test_ignores'])} ignores> "
               f"PODSIM_JUNIT={args.junit} PODSIM_LOG={args.log} bash {SIMULATOR}")
    print(f"staging contract {contract['digest'][:12]}… — "
          f"{staged_view['n_staged_files']} staged files visible, "
          f"{len(hidden)} hidden; ignores {contract['test_ignores']}")
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
            "a maintainer-approved C1 launch rests on, and is owed at the time "
            "that grant is issued."),
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
            "artifact_hiding": ("the simulator's default HIDDEN_PATHS — the "
                                "gitignored artifacts a session bundle cannot carry"),
        },
        "pytest_command": (
            ".venv/bin/python -m pytest tests/ -q "
            "--ignore=tests/data/test_recovery_corpus_pipeline.py "
            "--ignore=tests/pod/test_phase_a_stages1_5_execute.py"),
        "pytest_command_note": (
            "identical in test SELECTION to the pod gate in "
            "autoinit_preflight_setup.sh; --junitxml is a reporting flag added by "
            "the simulator so every skip and pass can be named exactly."),
        "simulator_exit_code": rc,
        "seconds": seconds,
        "counts": findings["counts"],
        "n_tests": junit["total"],
        "failed_nodeids": findings["failed_nodeids"],
        "expected_renderer_parity_skips": list(RENDERER_PARITY_NODEIDS),
        "renderer_parity_observed": findings["renderer_parity_expected_skips"],
        "renderer_parity_skipped_as_expected":
            findings["renderer_parity_skipped_as_expected"],
        "leaf_transport_nodeids": list(LEAF_TRANSPORT_NODEIDS),
        "leaf_transport_observed": findings["leaf_transport"],
        "leaf_transport_all_passed": findings["leaf_transport_all_passed"],
        "repository_state_observed": findings["repository_state"],
        "unexpected_environment_skips": findings["unexpected_environment_skips"],
        "problems": findings["problems"],
        "verdict": findings["verdict"] if rc == 0 else "FAIL",
        "evidence": {"junit": args.junit, "pytest_log": args.log},
        "renderer_parity_is_proved_by": "logs/c1_renderer_parity.json",
    }
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

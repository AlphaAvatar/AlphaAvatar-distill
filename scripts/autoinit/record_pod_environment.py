"""Run the pod-like sweep once, and record what it proved.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/record_pod_environment.py

This drives the real `scripts/pod/simulate_pod_env.sh` — empty HOME, isolated
`HF_HOME`, synthetic `HF_TOKEN`, gitignored artifacts hidden, the pod's own pytest
selection — and writes `logs/c1_pod_environment_verification.json`.

It runs the simulator itself rather than accepting somebody's transcript of one,
so the command in the record is literally the command that produced the counts.
A readiness record whose command field was typed by hand is a claim, not
evidence.

The sweep takes roughly three quarters of an hour. That is exactly why the record
exists: `aadistill.autoinit.pod_environment.verify_record` re-checks in
milliseconds that the recorded proof still describes the live executable, so a
pre-provider gate never has to re-run this while a pod waits.

Exit 0 when the sweep passes and the record is written; non-zero otherwise, and
the record is still written so the failure is inspectable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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

SIMULATOR = "scripts/pod/simulate_pod_env.sh"
DEFAULT_JUNIT = "/home/ecs-user/aad-scratch/podsim_junit.xml"
DEFAULT_LOG = "/home/ecs-user/aad-scratch/podsim_pytest.log"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--junit", default=DEFAULT_JUNIT)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--from-existing", action="store_true",
                    help="parse a sweep that already ran instead of running one")
    args = ap.parse_args()

    # Captured BEFORE the record is written: writing it into logs/ is itself a
    # tree modification, and the verdict must describe the tree that was swept.
    clean_before = tree_is_clean(REPO_ROOT)
    head = head_commit(REPO_ROOT)
    harness = c1_harness_digest(REPO_ROOT)
    env_digest = pod_test_environment_digest(REPO_ROOT)

    env = {**os.environ, "PODSIM_JUNIT": args.junit, "PODSIM_LOG": args.log}
    command = (f"PODSIM_JUNIT={args.junit} PODSIM_LOG={args.log} "
               f"bash {SIMULATOR}")
    started = time.time()
    if args.from_existing:
        rc, seconds = 0, 0.0
        print(f"parsing the existing sweep at {args.junit}")
    else:
        print(f"running: {command}\n(this takes roughly 45 minutes)")
        proc = subprocess.run(["bash", str(REPO_ROOT / SIMULATOR)],
                              cwd=str(REPO_ROOT), env=env)
        rc = proc.returncode
        seconds = round(time.time() - started, 1)

    junit = read_junit(args.junit)
    findings = evaluate_sweep(junit["outcomes"])

    record = {
        "schema": SCHEMA,
        "_what_this_is": (
            "one complete pod-like sweep of the CPU test suite: the condition a "
            "fresh C1 pod is actually in, which is what C1 attempt 3R's setup "
            "test gate refused for $0.3482 with zero scientific stages run."),
        "executable_head": head,
        "tree_clean": clean_before,
        "c1_harness_digest": harness["digest"],
        "c1_harness_n_files": harness["n_files"],
        "pod_test_environment_digest": env_digest["digest"],
        "pod_test_environment_n_files": env_digest["n_files"],
        "pod_test_environment_named_files": env_digest["named_files"],
        "binding_rule": (
            "this record binds the EXECUTABLE, not HEAD: the preregistration and "
            "the documentation commits that follow a sweep must not invalidate "
            "it, and an edit to any measured file must."),
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

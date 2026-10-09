"""Phase C1's readiness record contract: its strings, and its wrapper.

Split from `tests/runtime/test_readiness_wire_contract.py` by the 2026-10-03
convergence round. That file imported this package at MODULE SCOPE, so
collecting the core suite needed a closed experiment to import; it now proves the
mechanism — a `RecordContract`'s schema, harness key and record path drive
verification — with two synthetic contracts, which is the actual property.

What belongs here is C1's INSTANCE:

* its schema string and harness key must not move, because every committed C1
  record already carries them and their self-hashes were computed over them;
* its wrapper must supply the contract, which is the defect this closed on C1's
  own side — `pod_environment_gate` calls `C1.verify_record(record, REPO, ...)`
  with no harness digest, and under the previous signature that returned "no
  harness_digest provider was supplied" on the real launch path while every test
  passed one explicitly and never saw it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.runtime import pod_environment as PE  # noqa: E402
from stages.phase_c1 import pod_environment as C1  # noqa: E402


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


def record_for(contract, *, kind: str = "diagnostic", **over) -> dict:
    rec = {
        "schema": contract.schema,
        "swept_base_commit": head(),
        "tree_clean": True,
        contract.harness_field: contract.harness_digest(REPO),
        "pod_test_environment_digest": PE.pod_test_environment_digest(
            REPO, named_files=contract.named_files)["digest"],
        "counts": {"passed": 3000, "skipped": 50, "failed": 0, "error": 0},
        "verdict": "PASS",
        "record_kind": kind,
        "staging_contract_digest": "a" * 64,
        "problems": [],
    }
    rec.update(over)
    rec["self_sha256"] = PE.self_hash(rec)
    return rec


def test_the_schema_string_is_exactly_what_records_carry():
    assert C1.C1_RECORD_CONTRACT.schema == (
        "aadistill.autoinit.c1_pod_environment_verification/v1")


def test_the_harness_field_is_exactly_the_key_records_use():
    assert C1.C1_RECORD_CONTRACT.harness_field == "c1_harness_digest"


def test_a_c1_shaped_record_verifies_through_the_real_function():
    ok, why = PE.verify_record(record_for(C1.C1_RECORD_CONTRACT), REPO,
                               contract=C1.C1_RECORD_CONTRACT)
    assert ok, why


def test_the_wrapper_supplies_the_contract_so_the_launcher_needs_no_kwargs():
    ok, why = C1.verify_record(record_for(C1.C1_RECORD_CONTRACT), REPO)
    assert ok, why

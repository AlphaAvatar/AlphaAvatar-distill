"""A second caller can have a readiness record. Until now it could not.

`aadistill.runtime.pod_environment` is the reusable mechanism: how a sweep
record is verified, how the test-environment digest is computed, what lineage a
session commit must have. Extracting `ReadinessGroups` moved the *expectations*
out — which node ids a session watches — and left three things behind that are
just as much one experiment's:

* `SCHEMA`, a module constant, compared for equality. Any other caller's record
  was "unexpected schema".
* `record["c1_harness_digest"]`, read by that literal key. A second caller's
  harness would have been looked for under a name its record does not carry.
* the success message, which printed "7 renderer skips, leaf transport 5/5" —
  two of C1's group names and two of C1's counts — whatever the record said.

So this drives BOTH callers through the real `verify_record`: C1's own contract
against C1's own record shape, and a deliberately unlike one — different schema,
different harness field, different harness value, a different NUMBER of declared
groups. If any of the three had stayed in the runtime, the second caller fails
and the first one's summary lies.

`test_the_summary_reports_what_the_record_says` is the one that catches a
regression to a hardcoded message: it asserts the numbers move with the data.

Nothing here writes a record, runs a sweep, or touches the committed C1
artifact. Every record is built in-memory and self-hashed the way the real
recorder does it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.runtime import pod_environment as PE  # noqa: E402
from experiments.phase_c1 import pod_environment as C1  # noqa: E402


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


# --- two callers, deliberately unlike --------------------------------------

#: A second session's wire format. Nothing about it resembles C1's: its own
#: schema, its own harness key, its own harness value, its own record path.
OTHER = PE.RecordContract(
    schema="example.other_session_readiness/v3",
    harness_field="other_harness_sha256",
    harness_digest=lambda repo_root: "b" * 64,
    record_path="logs/other_session_readiness.json",
    named_files=(),
    harness_label="other-session harness",
)


def record_for(contract: PE.RecordContract, *, kind: str = "diagnostic",
               groups: dict | None = None, **over) -> dict:
    """A record binding the LIVE tree under `contract`, self-hashed last."""
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
        **(groups or {}),
    }
    rec.update(over)
    rec["self_sha256"] = PE.self_hash(rec)
    return rec


# --- 1. the original C1 contract still works --------------------------------

class TestTheC1ContractIsUnchanged:
    """Its schema string and its harness key must NOT move: every committed C1
    record already carries them, and their self-hashes were computed over them.
    Renaming either would invalidate real evidence."""

    def test_the_schema_string_is_exactly_what_records_carry(self):
        assert C1.C1_RECORD_CONTRACT.schema == (
            "aadistill.autoinit.c1_pod_environment_verification/v1")

    def test_the_harness_field_is_exactly_the_key_records_use(self):
        assert C1.C1_RECORD_CONTRACT.harness_field == "c1_harness_digest"

    def test_the_writer_sources_both_values_from_the_contract(self):
        """The producer/consumer property, and the reason neither value moved.

        This first read `logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json` off disk —
        which the pod-simulated sweep caught at `$0`, because that record is one
        of the 1056 paths a C1 pod does not receive. A test that reads a
        repository artifact the session does not stage passes on a dev box and
        fails on a pod, and this suite runs inside the pod's setup gate.

        Reading the WRITER is better than reading one artifact anyway: it checks
        that the record and the gate take their schema and their harness key
        from the same place, which is the property that keeps every record
        verifiable — not just the one currently on disk.
        """
        src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        assert '"schema": C1_RECORD_CONTRACT.schema' in code, (
            "the recorder writes a schema string it did not read from the "
            "contract the verifier checks against")
        assert "C1_RECORD_CONTRACT.harness_field: harness" in code, (
            "the recorder writes the harness under a literal key rather than "
            "the contract's field name")
        assert '"c1_harness_digest":' not in code, (
            "a literal harness key survives in the recorder")

    def test_a_c1_shaped_record_verifies_through_the_real_function(self):
        ok, why = PE.verify_record(record_for(C1.C1_RECORD_CONTRACT), REPO,
                                   contract=C1.C1_RECORD_CONTRACT)
        assert ok, why

    def test_the_wrapper_supplies_the_contract_so_the_launcher_needs_no_kwargs(self):
        """THE DEFECT this closes on C1's own side.

        `pod_environment_gate` calls `C1.verify_record(record, REPO, ...)` and
        passes no harness digest. Under the previous signature that returned
        "no harness_digest provider was supplied" — on the real launch path,
        while every test passed one explicitly and never saw it.
        """
        ok, why = C1.verify_record(record_for(C1.C1_RECORD_CONTRACT), REPO)
        assert ok, why


# --- 2. a non-C1 caller works, end to end -----------------------------------

class TestASecondCallerCanHaveARecord:
    def test_its_own_schema_is_accepted(self):
        ok, why = PE.verify_record(record_for(OTHER), REPO, contract=OTHER)
        assert ok, why

    def test_its_harness_is_read_under_ITS_key(self):
        """Not `c1_harness_digest`. A record carrying only the other key must
        verify, and one carrying only C1's must not."""
        rec = record_for(OTHER)
        assert "c1_harness_digest" not in rec
        assert rec["other_harness_sha256"] == "b" * 64
        assert PE.verify_record(rec, REPO, contract=OTHER)[0]

    def test_c1s_schema_is_refused_for_this_caller(self):
        rec = record_for(OTHER, schema=C1.C1_RECORD_CONTRACT.schema)
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=OTHER)
        assert not ok and "unexpected schema" in why

    def test_and_its_schema_is_refused_for_C1(self):
        """Symmetric. Neither caller may verify the other's record."""
        ok, why = PE.verify_record(record_for(OTHER), REPO,
                                   contract=C1.C1_RECORD_CONTRACT)
        assert not ok and "unexpected schema" in why

    def test_its_record_path_drives_the_post_sweep_allowance(self):
        assert PE.permitted_post_sweep_paths(OTHER.record_path)[0] == \
            OTHER.record_path
        assert C1.RECORD_PATH not in PE.permitted_post_sweep_paths(
            OTHER.record_path)


# --- 3. a DIFFERENT number of checks ----------------------------------------

class TestADifferentNumberOfChecks:
    """C1 declares nine groups. A caller with two must not be described in C1's
    counts, and a caller with none must not have a message invented for it."""

    TWO = {
        "alpha": {"tests/a.py::x": "passed", "tests/a.py::y": "passed"},
        "alpha_all_passed": True,
        "beta_expected_skips": {"tests/b.py::z": "skipped"},
        "beta_skipped_as_expected": True,
    }

    def test_a_two_group_caller_verifies(self):
        ok, why = PE.verify_record(record_for(OTHER, groups=self.TWO), REPO,
                                   contract=OTHER)
        assert ok, why

    def test_the_summary_uses_the_callers_own_group_names(self):
        _, why = PE.verify_record(record_for(OTHER, groups=self.TWO), REPO,
                                  contract=OTHER)
        assert "alpha 2/2 passed" in why
        assert "beta 1/1 skipped" in why
        assert "renderer" not in why and "leaf transport" not in why

    def test_the_summary_reports_what_the_record_says(self):
        """The regression a hardcoded message cannot survive: change the data,
        the numbers must change with it."""
        bigger = {
            "alpha": {f"tests/a.py::{i}": "passed" for i in range(5)},
            "alpha_all_passed": True,
        }
        _, why = PE.verify_record(record_for(OTHER, groups=bigger), REPO,
                                  contract=OTHER)
        assert "alpha 5/5 passed" in why

    def test_a_caller_with_no_declared_groups_says_so(self):
        _, why = PE.verify_record(record_for(OTHER), REPO, contract=OTHER)
        assert "no declared groups" in why

    def test_the_c1_label_is_gone_from_a_second_callers_refusal(self):
        rec = record_for(OTHER)
        rec["other_harness_sha256"] = "c" * 64
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=OTHER)
        assert not ok
        assert "other-session harness" in why and "C1" not in why


# --- 4. every fail-closed behaviour survives --------------------------------

class TestNothingWasWeakened:
    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_wrong_harness_digest_is_refused(self, contract):
        rec = record_for(contract)
        rec[contract.harness_field] = "f" * 64
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "the pod sweep is owed again" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_tampered_record_fails_the_self_hash(self, contract):
        """Edited in place WITHOUT recomputing the hash, which is what tampering
        looks like. The check must fire before anything else reads the field."""
        rec = record_for(contract)
        rec["counts"] = {"passed": 1, "skipped": 0, "failed": 0, "error": 0}
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "self-hash" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_promoting_the_kind_in_place_is_still_tampering(self, contract):
        rec = record_for(contract, kind="diagnostic")
        rec["record_kind"] = PE.LAUNCH_BOUND
        ok, why = PE.verify_record(rec, REPO, contract=contract,
                                   required_kind=PE.LAUNCH_BOUND)
        assert not ok and "self-hash" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_failing_verdict_is_refused(self, contract):
        ok, why = PE.verify_record(
            record_for(contract, verdict="FAIL", problems=["x"]), REPO,
            contract=contract)
        assert not ok and "verdict" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_dirty_sweep_is_refused(self, contract):
        ok, why = PE.verify_record(record_for(contract, tree_clean=False), REPO,
                                   contract=contract)
        assert not ok and "dirty working tree" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_diagnostic_is_still_refused_by_a_launch_bound_caller(self, contract):
        ok, why = PE.verify_record(record_for(contract), REPO, contract=contract,
                                   required_kind=PE.LAUNCH_BOUND)
        assert not ok and "launch_bound" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_missing_swept_base_is_refused(self, contract):
        rec = record_for(contract)
        del rec["swept_base_commit"]
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "swept_base_commit" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_staging_contract_mismatch_is_refused(self, contract):
        ok, why = PE.verify_record(record_for(contract), REPO, contract=contract,
                                   staging_contract_digest="9" * 64)
        assert not ok and "staging contract" in why

    @pytest.mark.parametrize("contract", [C1.C1_RECORD_CONTRACT, OTHER],
                             ids=["c1", "other"])
    def test_a_launch_bound_record_without_a_staging_contract_is_refused(self, contract):
        rec = record_for(contract, kind=PE.LAUNCH_BOUND)
        del rec["staging_contract_digest"]
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=contract,
                                   required_kind=PE.LAUNCH_BOUND)
        assert not ok and "staging_contract_digest" in why


# --- 5. the contract itself fails closed ------------------------------------

class TestTheContractRefusesToBeIncomplete:
    @pytest.mark.parametrize("field", ["schema", "harness_field", "record_path"])
    def test_a_blank_required_field_is_refused(self, field):
        kwargs = dict(schema="s", harness_field="h", record_path="p",
                      harness_digest=lambda r: "x")
        kwargs[field] = ""
        with pytest.raises(ValueError, match=field):
            PE.RecordContract(**kwargs)

    def test_a_non_callable_harness_digest_is_refused(self):
        """It is re-derived against the live tree; a stored string would make
        the check compare the record against itself."""
        with pytest.raises(ValueError, match="callable"):
            PE.RecordContract(schema="s", harness_field="h", record_path="p",
                              harness_digest="a" * 64)

    def test_the_runtime_defines_no_schema_of_its_own(self):
        """The literal removal. A module constant here is a schema the runtime
        would accept from anybody."""
        assert not hasattr(PE, "SCHEMA")

    def test_no_c1_name_survives_in_executable_code(self):
        """Code only. The comments explaining what was removed necessarily
        quote it, and a scan that cannot tell a citation from a claim would
        force the explanation out along with the defect.
        """
        import ast

        src = (REPO / "src/aadistill/runtime/pod_environment.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:] or [ast.Pass()]
        code = ast.unparse(tree).lower()
        for literal in ("c1_harness_digest", "renderer", "leaf transport",
                        "c1_pod_environment_verification"):
            assert literal not in code, (
                f"{literal!r} is still executable in the reusable runtime")

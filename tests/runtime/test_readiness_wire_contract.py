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


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


# --- two callers, deliberately unlike --------------------------------------

#: A FIRST caller, synthetic. This used to be
#: `experiments.phase_c1.pod_environment.C1_RECORD_CONTRACT`, imported at module
#: scope -- so collecting the CORE suite required a closed experiment's package
#: to import. The mechanism under test is "a RecordContract's schema, harness key
#: and record path drive verification", and that is a statement about arbitrary
#: contracts; proving it with two synthetic ones proves it for every caller,
#: including the ones that do not exist yet.
#:
#: Phase C1's concrete instance -- that ITS schema string and ITS harness key have
#: not moved, and that its wrapper supplies the contract -- is asserted in
#: `scripts/experiments/stage-1/phase_c1/tests/test_c1_record_contract.py`, where
#: a reader looking for C1's wiring will look.
FIRST = PE.RecordContract(
    schema="example.first_session_readiness/v1",
    harness_field="first_harness_digest",
    harness_digest=lambda repo_root: "a" * 64,
    record_path="logs/first_session_readiness.json",
    named_files=(),
    harness_label="first-session harness",
)

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


# --- 1. a contract's own strings are what verification reads --------------------------------

class TestAContractsOwnStringsAreUsedVerbatim:
    """A contract's schema and harness key are read from the contract, not from
    a literal. Asserted on a synthetic contract, because that is the property;
    which strings one experiment chose is that experiment's record.
    """

    def test_the_schema_string_is_the_contracts_own(self):
        assert FIRST.schema == "example.first_session_readiness/v1"

    def test_the_harness_field_is_the_contracts_own(self):
        assert FIRST.harness_field == "first_harness_digest"

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
        #: These read `C1_RECORD_CONTRACT.schema` and
        #: `C1_RECORD_CONTRACT.harness_field` until 2026-09-15, when the
        #: recorder stopped importing any experiment's contract by name and
        #: started taking it from the `SweepContract` the invocation names. The
        #: PROPERTY is the same and is stronger now: the schema and the harness
        #: key come from the contract the verifier checks against, for whichever
        #: experiment is being swept, rather than from C1's contract regardless.
        assert '"schema": sweep.record.schema' in code, (
            "the recorder writes a schema string it did not read from the "
            "contract the verifier checks against")
        assert "sweep.record.harness_field: harness" in code, (
            "the recorder writes the harness under a literal key rather than "
            "the contract's field name")
        assert "sweep.harness_n_files_field: harness" in code, (
            "the recorder writes the harness file COUNT under a literal key")
        for literal in ('"c1_harness_digest":', '"c1_harness_n_files":',
                        '"c2_harness_digest":', '"c2_harness_n_files":'):
            assert literal not in code, (
                f"the literal harness key {literal} survives in the recorder")

# --- 2. a SECOND caller works, end to end -----------------------------------

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

    def test_the_other_callers_schema_is_refused_for_this_one(self):
        rec = record_for(OTHER, schema=FIRST.schema)
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=OTHER)
        assert not ok and "unexpected schema" in why

    def test_and_its_schema_is_refused_for_the_first(self):
        """Symmetric. Neither caller may verify the other's record."""
        ok, why = PE.verify_record(record_for(OTHER), REPO,
                                   contract=FIRST)
        assert not ok and "unexpected schema" in why

    def test_its_record_path_drives_the_post_sweep_allowance(self):
        assert PE.permitted_post_sweep_paths(OTHER.record_path)[0] == \
            OTHER.record_path
        assert FIRST.record_path not in PE.permitted_post_sweep_paths(
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
    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_wrong_harness_digest_is_refused(self, contract):
        rec = record_for(contract)
        rec[contract.harness_field] = "f" * 64
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "the pod sweep is owed again" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_tampered_record_fails_the_self_hash(self, contract):
        """Edited in place WITHOUT recomputing the hash, which is what tampering
        looks like. The check must fire before anything else reads the field."""
        rec = record_for(contract)
        rec["counts"] = {"passed": 1, "skipped": 0, "failed": 0, "error": 0}
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "self-hash" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_promoting_the_kind_in_place_is_still_tampering(self, contract):
        rec = record_for(contract, kind="diagnostic")
        rec["record_kind"] = PE.LAUNCH_BOUND
        ok, why = PE.verify_record(rec, REPO, contract=contract,
                                   required_kind=PE.LAUNCH_BOUND)
        assert not ok and "self-hash" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_failing_verdict_is_refused(self, contract):
        ok, why = PE.verify_record(
            record_for(contract, verdict="FAIL", problems=["x"]), REPO,
            contract=contract)
        assert not ok and "verdict" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_dirty_sweep_is_refused(self, contract):
        ok, why = PE.verify_record(record_for(contract, tree_clean=False), REPO,
                                   contract=contract)
        assert not ok and "dirty working tree" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_diagnostic_is_still_refused_by_a_launch_bound_caller(self, contract):
        ok, why = PE.verify_record(record_for(contract), REPO, contract=contract,
                                   required_kind=PE.LAUNCH_BOUND)
        assert not ok and "launch_bound" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_missing_swept_base_is_refused(self, contract):
        rec = record_for(contract)
        del rec["swept_base_commit"]
        rec["self_sha256"] = PE.self_hash(
            {k: v for k, v in rec.items() if k != "self_sha256"})
        ok, why = PE.verify_record(rec, REPO, contract=contract)
        assert not ok and "swept_base_commit" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
                             ids=["c1", "other"])
    def test_a_staging_contract_mismatch_is_refused(self, contract):
        ok, why = PE.verify_record(record_for(contract), REPO, contract=contract,
                                   staging_contract_digest="9" * 64)
        assert not ok and "staging contract" in why

    @pytest.mark.parametrize("contract", [FIRST, OTHER],
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

"""Rehearsal of the characterization continuation: every way it must refuse.

The continuation session spends money on one thing — characterizing two controls
it did not train — so the load-bearing property is that an imported artifact
which is not the permanent control is rejected *before* any measurement is bound
to it.

    wrong imported weights hash          -> import refused
    wrong control_binding                -> import refused
    missing material evidence            -> import refused, field named
    attestation mismatch                 -> characterization refused
    v2 tool smoke failure                -> stop before the controls
    normal success path                  -> through collection and teardown

The happy path runs against the **real** permanent controls where they are
staged, so the gate is exercised on the artifacts it will actually admit rather
than on a fixture built to satisfy it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1,
    CONTINUATION_SCOPE,
    IMPORT_REQUIRED_FIELDS,
    ControlImportError,
    continuation_manifest,
    import_permanent_control,
)
from aadistill.autoinit.recovery import RecoveryAdmissionError  # noqa: E402

RECORDS = REPO / "logs/autoinit_permanent_controls"
STAGED = Path("/home/ecs-user/aad-artifacts/autoinit")
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")

have_controls = pytest.mark.skipif(
    not (STAGED / CONTROLS[0] / "step_001023/model/model.safetensors").is_file(),
    reason="the permanent controls are local artifacts, not tracked in git")


def staged(name: str) -> Path:
    return STAGED / name / "step_001023/model"


def evidence_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / f"evidence_{name}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(RECORDS / f"{name}_run_manifest.json", d / "run_manifest.json")
    shutil.copy(RECORDS / f"{name}_run_completion.json", d / "run_completion.json")
    return d


def record_copy(tmp_path: Path, name: str, **edits) -> Path:
    record = json.loads((RECORDS / f"{name}_probe_identity.json").read_text())
    for dotted, value in edits.items():
        target, _, leaf = dotted.rpartition(".")
        node = record
        for part in filter(None, target.split(".")):
            node = node[part]
        if value is _DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    path = tmp_path / f"{name}_record.json"
    path.write_text(json.dumps(record, indent=2))
    return path


class _Delete:
    pass


_DELETE = _Delete()


# --- the success path -------------------------------------------------------


@have_controls
def test_the_real_permanent_controls_import(tmp_path):
    """The gate admits exactly what it should, on the real artifacts."""
    for name, seed in zip(CONTROLS, (20260726, 20260801)):
        imported = import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name),
            run_evidence_dir=evidence_dir(tmp_path, name),
            repo_root=REPO, strict=True)
        assert imported.seed == seed
        assert imported.reconstructed_from_run_evidence, (
            "the protocol must be re-derived from the run's own evidence, not "
            "taken from the record")
        reconstruction = imported.evidence["reconstruction"]
        assert reconstruction["fingerprint"] == imported.observed_protocol_fingerprint
        assert reconstruction["missing_fields"] == []
        assert reconstruction["step_accounting"]["completed_all_steps"] is True
        assert reconstruction["pack_recomputed"].startswith("6f324cb0")
        assert imported.weights_sha256 == \
            imported.control_binding["checkpoint_weights_sha256"]
    # Both controls share one protocol and differ only by seed — the
    # single-variable structure Phase A needs, re-established on import.
    a, b = (import_permanent_control(
        n, record_path=RECORDS / f"{n}_probe_identity.json",
        checkpoint_dir=staged(n), run_evidence_dir=evidence_dir(tmp_path, n),
        repo_root=REPO, strict=True) for n in CONTROLS)
    assert a.observed_protocol_fingerprint == b.observed_protocol_fingerprint
    assert a.seed != b.seed and a.probe_id != b.probe_id
    assert a.initialization_artifact_digest == b.initialization_artifact_digest


# --- 1. wrong imported weights ----------------------------------------------


@have_controls
def test_wrong_imported_weights_hash_is_refused(tmp_path):
    name = CONTROLS[0]
    record = record_copy(tmp_path, name, weights_sha256="0" * 64)
    with pytest.raises(ControlImportError, match="not the same checkpoint"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


@have_controls
def test_a_checkpoint_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ControlImportError, match="nothing was staged"):
        import_permanent_control(
            CONTROLS[0], record_path=RECORDS / f"{CONTROLS[0]}_probe_identity.json",
            checkpoint_dir=tmp_path / "absent", repo_root=REPO, strict=True)


# --- 2. wrong control_binding -----------------------------------------------


@have_controls
def test_a_control_binding_that_disagrees_is_refused(tmp_path):
    name = CONTROLS[0]
    for edit, expect in (
        ({"control_binding.checkpoint_weights_sha256": "1" * 64},
         "not the ones the binding was issued for"),
        ({"control_binding.probe_id": "2" * 64},
         "disagrees with the recorded probe id"),
    ):
        record = record_copy(tmp_path, name, **edit)
        with pytest.raises(ControlImportError, match=expect):
            import_permanent_control(name, record_path=record,
                                     checkpoint_dir=staged(name),
                                     run_evidence_dir=evidence_dir(tmp_path, name),
                                     repo_root=REPO, strict=True)


@have_controls
def test_two_copies_of_the_fingerprint_must_agree(tmp_path):
    """The record states it twice; a record that contradicts itself is not one."""
    name = CONTROLS[0]
    record = record_copy(tmp_path, name,
                         **{"control_binding.observed_protocol_fingerprint": "3" * 64})
    with pytest.raises(ControlImportError, match="disagree"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


# --- 3. missing material evidence -------------------------------------------


@have_controls
def test_missing_material_evidence_is_refused_and_named(tmp_path):
    name = CONTROLS[0]
    for dotted, field in (("weights_sha256", "weights_sha256"),
                          ("probe_id", "probe_id"),
                          ("seed", "seed"),
                          ("initialization_artifact_digest",
                           "initialization_artifact_digest"),
                          ("control_binding", "control_binding")):
        record = record_copy(tmp_path, name, **{dotted: _DELETE})
        with pytest.raises(ControlImportError, match=field):
            import_permanent_control(name, record_path=record,
                                     checkpoint_dir=staged(name),
                                     run_evidence_dir=evidence_dir(tmp_path, name),
                                     repo_root=REPO, strict=True)


@have_controls
def test_no_run_evidence_is_refused_in_strict_mode(tmp_path):
    """A sidecar summary is not evidence that a protocol was executed."""
    name = CONTROLS[0]
    with pytest.raises(ControlImportError, match="could not be re-derived"):
        import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name), run_evidence_dir=None,
            repo_root=REPO, strict=True)


@have_controls
def test_run_evidence_that_reconstructs_a_different_protocol_is_refused(tmp_path):
    name = CONTROLS[0]
    evidence = evidence_dir(tmp_path, name)
    manifest = json.loads((evidence / "run_manifest.json").read_text())
    manifest["execution"]["optimizer_defaults"]["lr"] = 9e-9
    manifest["config"]["optim"]["lr"] = 9e-9
    (evidence / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ControlImportError, match="reconstructs protocol"):
        import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name), run_evidence_dir=evidence,
            repo_root=REPO, strict=True)


@have_controls
def test_a_control_that_never_verified_cannot_be_imported(tmp_path):
    name = CONTROLS[0]
    record = record_copy(tmp_path, name, protocol_verified=False)
    with pytest.raises(ControlImportError, match="protocol_verified is false"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


# --- 4/5. the plan's own gating ---------------------------------------------


def test_the_continuation_plan_orders_the_gates_and_blocks_on_them():
    stages = {s.stage: s for s in CONTINUATION_PLAN_V1.stages}
    assert [stages[i].name for i in range(4)] == [
        "import the permanent controls",
        "current evaluation attestation",
        "v2 tool and RAG generation smoke",
        "characterize the imported controls"]
    # Import, attestation and smoke all block; characterization does not, so a
    # scoring failure cannot retroactively invalidate what came before it.
    assert all(stages[i].blocking for i in (0, 1, 2))
    assert stages[3].blocking is False

    # An attestation failure stops before the smoke and before characterization.
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        CONTINUATION_PLAN_V1.advance_to(
            2, {0: {"passed": True}, 1: {"passed": False, "reason": "drift"}})
    # A smoke failure stops before characterization.
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        CONTINUATION_PLAN_V1.advance_to(
            3, {0: {"passed": True}, 1: {"passed": True},
                2: {"passed": False, "reason": "tool prompt did not render"}})
    # And a missing predecessor is refused outright, not assumed.
    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        CONTINUATION_PLAN_V1.advance_to(3, {0: {"passed": True}, 1: {"passed": True}})


def test_the_smoke_stage_names_the_tool_set_as_its_reason_to_exist():
    smoke = next(s for s in CONTINUATION_PLAN_V1.stages if s.stage == 2)
    joined = " ".join(smoke.stop_conditions) + smoke.purpose + " ".join(smoke.produces)
    assert "tool" in joined and "rag" in joined
    assert "renders no tool prompt" in joined, (
        "the stage must say that a smoke which never renders a tool prompt is "
        "the failure it exists to prevent")


def test_the_continuation_does_not_train_and_cannot_reach_phase_a():
    scope = CONTINUATION_SCOPE.as_dict()
    assert scope["trains_anything"] is False
    assert scope["retrains_controls"] is False
    assert scope["reaches_phase_a"] is False
    assert scope["battery"] == "recovery_search_v2"
    assert sorted(scope["controls"]) == sorted(CONTROLS)
    manifest = continuation_manifest()
    assert manifest["manifest_sha256"]
    assert manifest["import_required_fields"] == list(IMPORT_REQUIRED_FIELDS)
    # It is a different plan from the preflight, not a mutation of it.
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1
    assert CONTINUATION_PLAN_V1.plan_hash != PREFLIGHT_PLAN_V1.plan_hash
    assert CONTINUATION_PLAN_V1.plan_id != PREFLIGHT_PLAN_V1.plan_id


def test_advance_to_was_not_weakened():
    """The preflight gate must still refuse what it refused before."""
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1

    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        PREFLIGHT_PLAN_V1.advance_to(3, {0: {"passed": True}, 1: {"passed": True}})
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        PREFLIGHT_PLAN_V1.advance_to(
            2, {0: {"passed": True}, 1: {"passed": False, "reason": "peak memory"}})

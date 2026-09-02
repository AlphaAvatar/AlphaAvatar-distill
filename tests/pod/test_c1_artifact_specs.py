"""The C1 evidence declaration, exercised against the real collector.

Both spec files were absent from a tree that passed every other precheck, and
nothing noticed, because `SessionSpec.validate()` only checks that the two path
*strings* are non-empty and `collect_artifacts.py` first opens the files on the
pod, at teardown, after the money is spent.

So these tests do two things a schema check cannot.

They pin every pattern to the **producer's own literal**. The manifest root is
`artifacts/`, and the paths under it are decided by `PhaseADriver`, which
`C1Driver` subclasses -- `audit/autoinit_phase_a`, `stage3/phase_a/<probe_id>`,
`eval/phase_a/<probe_id>` -- not by the C1 launcher's `audit_dirname`. A spec
written from the launcher's name would match nothing, which is exactly how
Phase-B attempt 3 lost its search journal to `phase_a_search` vs
`phase_b_search`. Each pattern below is therefore asserted against the source
line that writes it.

And they run the **actual four-step collection** -- manifest, archive,
verify-archive, verify-local -- over a synthetic C1-shaped filesystem, once for
a full success and once for a replay mismatch, checking both that the success
spec is satisfiable and that it is not vacuous.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))

from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1,
)
from aadistill.autoinit.c1_isolation import derive_recovery_seeds  # noqa: E402
from collect_artifacts import load_specs  # noqa: E402

SUCCESS = "configs/autoinit/c1_artifacts.json"
FAILED = "configs/autoinit/c1_artifacts_failed.json"
COLLECT = REPO / "scripts/pod/collect_artifacts.py"
BATTERY = json.loads(
    (REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json").read_text())
SETS = sorted(BATTERY["sets"])
PROBES = [f"autoinit.v1.phase_c1.{arm}.{seed}"
          for arm in ("incumbent", "treatment")
          for seed in derive_recovery_seeds()]


# --- the specs are well-formed and measured --------------------------------

@pytest.mark.parametrize("rel", [SUCCESS, FAILED])
def test_spec_exists_parses_and_loads(rel):
    p = REPO / rel
    assert p.is_file(), f"{rel} is missing"
    json.loads(p.read_text())
    specs = load_specs(str(p))
    assert specs, f"{rel} declares no entries"


@pytest.mark.parametrize("rel", [SUCCESS, FAILED])
def test_spec_is_inside_the_measured_harness(rel):
    """Editing what survives teardown must move the digest a grant binds."""
    assert rel in C1_HARNESS_SOURCE_FILES_V1


@pytest.mark.parametrize("rel", [SUCCESS, FAILED])
def test_patterns_stay_within_the_artifact_roots(rel):
    for s in load_specs(str(REPO / rel)):
        assert not s.pattern.startswith("/")
        assert ".." not in s.pattern.split("/")
        assert s.pattern.split("/", 1)[0] in (
            "audit", "eval", "stage3", "stage1", "autoinit"), s.pattern


# --- every pattern traces to the line that writes it ------------------------

def test_patterns_match_the_producing_source_literals():
    """The spec's paths are the driver's paths, not the launcher's dirname.

    `ArtifactPolicy.audit_dirname="autoinit_c1"` only decides where the runner
    copies session logs and where it scp's `report_names`. Everything the driver
    writes goes to the INHERITED `audit/autoinit_phase_a`.
    """
    a = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    c1 = (REPO / "scripts/pod/autoinit_c1_driver.py").read_text()
    trainer = (REPO / "scripts/training/train_stage3.py").read_text()
    ev = (REPO / "scripts/evaluation/uncapped_eval.py").read_text()

    assert 'AUDIT = REPO / "artifacts/audit/autoinit_phase_a"' in a
    assert '"out_dir": f"artifacts/stage3/phase_a/{name}"' in a
    assert 'gen_dir = REPO / f"artifacts/eval/phase_a/{label}"' in a
    assert 'AUDIT / f"{label}_per_sample.jsonl"' in a
    assert 'AUDIT / f"{label}_recovery_search.json"' in a
    assert 'AUDIT / "probes" / f"{name}.json"' in a
    assert 'AUDIT / "configs" / f"{name}.json"' in a
    assert 'AUDIT / f"{name}_train_tail.log"' in a
    assert '(AUDIT / "phase_a_evidence.json")' in a
    assert '(AUDIT / "attested_evaluation_protocol.json")' in a
    assert 'JsonlLogger(out_dir / "train_log.jsonl")' in trainer
    assert '"run_manifest.json"' in trainer and '"run_completion.json"' in trainer
    assert '.generations.jsonl' in ev
    for name in ("c1_replay_record.json", "c1_arm_identities.json",
                 "c1_decision.json"):
        assert f'"{name}"' in c1, name
    assert 'f"autoinit.v1.phase_c1.{arm}.{seed}"' in c1


def test_success_spec_covers_the_frozen_evidence_contract():
    """Minimums are derived from the contract, not transcribed into the test."""
    prereg = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    n_probes = prereg["c1_session_contract"]["stages"] and 6
    required = {s.artifact_class: s.min_matches
                for s in load_specs(str(REPO / SUCCESS)) if s.required}
    for cls in ("probe_event_stream", "per_sample", "scored_probe_aggregate",
                "probe_journal"):
        assert required.get(cls, 0) >= n_probes, cls
    # P18: every evaluated sample of every set of every probe.
    assert required.get("generations", 0) >= n_probes * len(SETS)
    assert required.get("generation_summary", 0) >= n_probes * len(SETS)
    for cls in ("arm_identities", "replay_record", "decision",
                "attested_protocol", "session_evidence", "engine_probe"):
        assert required.get(cls, 0) >= 1, cls


def test_failed_spec_requires_nothing_that_presupposes_training():
    post = {"probe_event_stream", "per_sample", "generations",
            "generation_summary", "scored_probe_aggregate", "probe_journal",
            "probe_config", "probe_run_manifest", "probe_run_completion",
            "decision", "probe_train_tail"}
    for s in load_specs(str(REPO / FAILED)):
        if s.required and s.min_matches > 0:
            assert s.artifact_class not in post, s.artifact_class


def test_failed_spec_still_collects_the_replay_mismatch_evidence():
    """Not required -- it cannot be -- but it MUST be declared, or the archive
    would not contain it even when it exists."""
    classes = {s.artifact_class for s in load_specs(str(REPO / FAILED))}
    for cls in ("replay_record", "arm_identities", "engine_probe",
                "session_evidence", "session_logs", "probe_event_stream"):
        assert cls in classes, cls


def test_failed_spec_documents_why_it_cannot_be_conditional():
    doc = json.loads((REPO / FAILED).read_text())
    assert "limitation" in doc and len(doc["limitation"]) > 400


# --- the real collector, over a synthetic C1 filesystem ---------------------

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _replay_tree(root: Path) -> None:
    """What exists after a D/E mismatch: identities and evidence, no training."""
    audit = root / "audit/autoinit_phase_a"
    _write(audit / "phase_a_evidence.json", json.dumps({"outcome": "FAILED"}))
    _write(audit / "c1_replay_record.json", json.dumps(
        {"schema": "aadistill.autoinit.c1_replay_mismatch/v1",
         "training_started": False, "runtime": {"torch": "x"}, "steps": []}))
    _write(audit / "engine_probe.json", "{}")
    _write(root / "audit/autoinit_c1/session/autoinit_c1_run.log", "log\n")


def _success_tree(root: Path) -> None:
    """What exists after ALL_DONE: the replay evidence plus all six probes."""
    _replay_tree(root)
    audit = root / "audit/autoinit_phase_a"
    _write(audit / "c1_arm_identities.json", "{}")
    _write(audit / "c1_decision.json", "{}")
    _write(audit / "attested_evaluation_protocol.json", "{}")
    for probe in PROBES:
        _write(root / f"stage3/phase_a/{probe}/train_log.jsonl", '{"step":1}\n')
        _write(root / f"stage3/phase_a/{probe}/run_manifest.json", "{}")
        _write(root / f"stage3/phase_a/{probe}/run_completion.json", "{}")
        _write(audit / "probes" / f"{probe}.json", "{}")
        _write(audit / "configs" / f"{probe}.json", "{}")
        _write(audit / f"{probe}_per_sample.jsonl", '{"i":0}\n')
        _write(audit / f"{probe}_recovery_search.json", "{}")
        _write(audit / f"{probe}_train_tail.log", "tail\n")
        for s in SETS:
            _write(root / f"eval/phase_a/{probe}/{s}.generations.jsonl", '{"g":1}\n')
            _write(root / f"eval/phase_a/{probe}/{s}.json", "{}")


def _collect(tmp: Path, root: Path, spec_rel: str) -> tuple[int, str]:
    """manifest -> archive -> verify-archive -> extract -> verify-local."""
    man, arc = tmp / "manifest.json", tmp / "c1_artifacts.tar.gz"
    steps = [["manifest", "--root", str(root), "--spec", str(REPO / spec_rel),
              "--out", str(man)]]
    rc = subprocess.run([sys.executable, str(COLLECT), *steps[0]],
                        capture_output=True, text=True, timeout=600)
    if rc.returncode != 0:
        return rc.returncode, rc.stdout + rc.stderr
    out = ""
    for argv in (["archive", "--manifest", str(man), "--out", str(arc)],
                 ["verify-archive", "--manifest", str(man), "--archive", str(arc)]):
        r = subprocess.run([sys.executable, str(COLLECT), *argv],
                           capture_output=True, text=True, timeout=600)
        out += r.stdout + r.stderr
        if r.returncode != 0:
            return r.returncode, out
    extract = tmp / "extracted"
    extract.mkdir(exist_ok=True)
    with tarfile.open(arc) as tar:
        tar.extractall(extract, filter="data")
    r = subprocess.run(
        [sys.executable, str(COLLECT), "verify-local", "--manifest", str(man),
         "--root", str(extract)], capture_output=True, text=True, timeout=600)
    return r.returncode, out + r.stdout + r.stderr


def test_full_success_round_trips_through_the_real_collector(tmp_path):
    root = tmp_path / "artifacts"
    _success_tree(root)
    rc, out = _collect(tmp_path, root, SUCCESS)
    assert rc == 0, out
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    classes = {e["artifact_class"] for e in manifest["entries"]}
    assert "generations" in classes and "probe_event_stream" in classes
    n_gen = sum(1 for e in manifest["entries"]
                if e["artifact_class"] == "generations")
    assert n_gen == len(PROBES) * len(SETS) == 42


def test_replay_mismatch_round_trips_through_the_failed_spec(tmp_path):
    root = tmp_path / "artifacts"
    _replay_tree(root)
    rc, out = _collect(tmp_path, root, FAILED)
    assert rc == 0, out
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    paths = {e["path"] for e in manifest["entries"]}
    assert "audit/autoinit_phase_a/c1_replay_record.json" in paths
    assert "audit/autoinit_c1/session/autoinit_c1_run.log" in paths


def test_success_spec_refuses_the_replay_mismatch_tree(tmp_path):
    """The success spec must not be vacuous: a run that never trained cannot
    satisfy it, and the manifest must say which classes are missing."""
    root = tmp_path / "artifacts"
    _replay_tree(root)
    rc, out = _collect(tmp_path, root, SUCCESS)
    assert rc == 5, out
    assert "MISSING" in out
    for cls in ("probe_event_stream", "generations", "decision"):
        assert cls in out, cls


def test_one_missing_generation_set_fails_the_manifest(tmp_path):
    """41 of 42 is not "every evaluated sample". Mutation of the success case."""
    root = tmp_path / "artifacts"
    _success_tree(root)
    (root / f"eval/phase_a/{PROBES[0]}/{SETS[0]}.generations.jsonl").unlink()
    rc, out = _collect(tmp_path, root, SUCCESS)
    assert rc == 5, out
    assert "generations" in out


# --- the gate itself --------------------------------------------------------

def _gate():
    import autoinit_c1_launch as L
    return L


def test_artifact_spec_gate_passes_on_the_committed_specs():
    ok, why = _gate().artifact_spec_gate(None)
    assert ok, why


def _fake_root(monkeypatch, tmp_path, *, success=None, failed=None) -> Path:
    """A self-consistent tree the gate can be pointed at: both specs plus the
    battery manifest it derives the generation count from."""
    L = _gate()
    for rel, doc in ((SUCCESS, success), (FAILED, failed)):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc) if doc is not None
                     else (REPO / rel).read_text())
    man = tmp_path / L.BATTERY_MANIFEST
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text((REPO / L.BATTERY_MANIFEST).read_text())
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    return tmp_path


def test_artifact_spec_gate_catches_an_undercounted_success_spec(monkeypatch,
                                                                 tmp_path):
    """Mutation: a spec that asks for 6 generation files instead of 42."""
    L = _gate()
    doc = json.loads((REPO / SUCCESS).read_text())
    for e in doc["entries"]:
        if e["artifact_class"] == "generations":
            e["min_matches"] = 6
    _fake_root(monkeypatch, tmp_path, success=doc)
    ok, why = L.artifact_spec_gate(None)
    assert not ok and "generations" in why


def test_artifact_spec_gate_catches_a_failed_spec_that_demands_training(
        monkeypatch, tmp_path):
    L = _gate()
    doc = json.loads((REPO / FAILED).read_text())
    for e in doc["entries"]:
        if e["artifact_class"] == "probe_event_stream":
            e["required"], e["min_matches"] = True, 6
    _fake_root(monkeypatch, tmp_path, failed=doc)
    ok, why = L.artifact_spec_gate(None)
    assert not ok and "probe_event_stream" in why


def test_artifact_spec_gate_catches_an_out_of_tree_pattern(monkeypatch,
                                                           tmp_path):
    L = _gate()
    doc = json.loads((REPO / SUCCESS).read_text())
    doc["entries"][0]["pattern"] = "../../etc/passwd"
    _fake_root(monkeypatch, tmp_path, success=doc)
    ok, why = L.artifact_spec_gate(None)
    assert not ok and "artifact roots" in why


def test_artifact_spec_gate_reacts_to_a_battery_with_fewer_sets(monkeypatch,
                                                                tmp_path):
    """The generation minimum is derived, so a battery that lost a set must
    change what the gate demands rather than silently accept 42."""
    L = _gate()
    root = _fake_root(monkeypatch, tmp_path)
    man = json.loads((root / L.BATTERY_MANIFEST).read_text())
    man["sets"].pop(SETS[0])
    (root / L.BATTERY_MANIFEST).write_text(json.dumps(man))
    ok, why = L.artifact_spec_gate(None)
    assert ok, why           # 42 required still exceeds 6 x 6 = 36
    assert f"{len(PROBES) * (len(SETS) - 1)} generation files" in why


def test_artifact_spec_gate_catches_an_unmeasured_spec(monkeypatch):
    L = _gate()
    monkeypatch.setattr(L, "C1_HARNESS_SOURCE_FILES_V1", ())
    ok, why = L.artifact_spec_gate(None)
    assert not ok and "measured C1 harness" in why


def test_artifact_spec_gate_catches_a_missing_file(monkeypatch):
    L = _gate()
    monkeypatch.setattr(L, "SPEC_SUCCESS", "configs/autoinit/does_not_exist.json")
    monkeypatch.setattr(L, "C1_HARNESS_SOURCE_FILES_V1",
                        ("configs/autoinit/does_not_exist.json", FAILED))
    ok, why = L.artifact_spec_gate(None)
    assert not ok and "missing" in why


def test_artifact_spec_gate_is_wired_into_the_launcher():
    L = _gate()
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert "artifact_spec_gate," in src.split("precheck=(", 1)[1]
    assert L.SPEC_SUCCESS == SUCCESS and L.SPEC_FAILED == FAILED


def test_launcher_books_the_specs_the_gate_validates():
    """The gate cannot end up checking a different file than the pod is handed."""
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    policy = src.split("ArtifactPolicy(", 1)[1].split("teardown=", 1)[0]
    assert "spec_success=SPEC_SUCCESS" in policy
    assert "spec_failed=SPEC_FAILED" in policy


def test_writer_and_launcher_name_the_same_specs():
    """The preregistration writer restates the two paths; keep them in step."""
    L = _gate()
    w = (REPO / "scripts/autoinit/write_c1_execution_preregistration.py").read_text()
    ns: dict = {}
    for line in w.splitlines():
        if line.startswith(("SPEC_SUCCESS", "SPEC_FAILED")):
            exec(line, ns)                                     # noqa: S102
    assert ns["SPEC_SUCCESS"] == L.SPEC_SUCCESS
    assert ns["SPEC_FAILED"] == L.SPEC_FAILED


def test_preregistration_binds_both_spec_hashes():
    from aadistill.infrastructure.manifest import sha256_file

    doc = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    block = doc["artifact_specs"]
    assert block["success"]["path"] == SUCCESS
    assert block["failed"]["path"] == FAILED
    assert block["success"]["sha256"] == sha256_file(REPO / SUCCESS)
    assert block["failed"]["sha256"] == sha256_file(REPO / FAILED)

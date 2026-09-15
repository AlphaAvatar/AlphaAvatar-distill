"""What C2 attempt 3 cost $0.1674 to learn, in checks that cost nothing.

Attempt 3 reached a $1.09/h pod, completed setup in 2m33s, started its driver
and died one second later. Three defects, in a chain:

1. **Stage A resolved calibration profiles from an EMPTY registry.** The four
   mixtures are data in `configs/calibration/profiles.json`; nothing in
   `src/aadistill` names them, and `scripts/experiments/calibration.py` is the
   application bootstrap that loads them. The C2 driver imported that bootstrap
   inside stage B while stage A already called `get_profile`, so stage A raised
   `no calibration profile 'calib.domain_balanced@v1'; registered: []`.

2. **Both C2 artifact specs declared `"lifecycle": "best_effort"`**, which is
   not a lifecycle -- `ArtifactSpec.__post_init__` accepts only
   `final_required` and `mutable_snapshot`. The document was unloadable, so the
   collector exited 1 before looking at a file and no manifest existed. The
   same word sat in the SUCCESS spec, where a completed ten-hour Search-1 would
   have produced no manifest, failed the teardown gate and held the pod until
   the 828-minute watchdog.

3. **With no manifest, a session that declares no event streams could not
   satisfy the emergency gate's naming rule**, so the launcher raised
   `ArtifactError` mid-teardown and reported that instead of the real failure.

Why no `$0` test saw the first one: **`tests/conftest.py` imports the
bootstrap.** The profile registry is process-global, so it is full in every
pytest process, and the empty registry exists only in a fresh interpreter --
which is exactly what a pod runs. Every check here that depends on that
distinction therefore leaves the process.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/pod"))

from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    ArtifactError, LIFECYCLES, build_manifest, evaluate_teardown)
from aadistill.infrastructure.session_runner import streams_at_risk  # noqa: E402
from collect_artifacts import load_specs  # noqa: E402

SPEC_SCHEMA = "aadistill.artifact_spec/v1"
C2_SPECS = ("configs/autoinit/c2_artifacts.json",
            "configs/autoinit/c2_artifacts_failed.json")

#: The device C1 froze. `FixedPathSpec.as_dict` includes it, so the B spec hash
#: is device-dependent and `cuda:0` or `cpu` produce a different, correct-for-
#: their-device hash that is not B's. The launcher passes `--device cuda`.
FROZEN_DEVICE = "cuda"


def _fresh(code: str, *argv: str) -> subprocess.CompletedProcess:
    """Run `code` in an interpreter that has never imported this test session.

    No `-S`, no environment scrubbing: the point is only that `conftest.py` has
    not run, because that import is what hides the defect.
    """
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True, text=True, timeout=900, cwd=str(REPO))


def _an_issued_authorization() -> str:
    """Any C2 authorization actually issued, repository-relative.

    The driver's constructor reads one for its `authorization_id`; stage A does
    not care which. Sorted and first so the check does not change meaning as
    attempts accrue.
    """
    found = sorted(
        p.relative_to(REPO).as_posix() for p in REPO.glob(
            "logs/stages/stage-1/phase_c2/runs/*/governance/authorization.json"))
    assert found, ("no C2 authorization has ever been issued, so this check "
                   "cannot construct the driver; that is a repository state "
                   "problem, not a passing test")
    return found[0]


# --- 1. the failure itself ---------------------------------------------------


_STAGE_A = """
import argparse, json, pathlib, sys
REPO = pathlib.Path(sys.argv[1])
for rel in ("src", "scripts", "scripts/pod"):
    sys.path.insert(0, str(REPO / rel))
import autoinit_phase_c2_driver as D
tmp = pathlib.Path(sys.argv[2])
# The pod's /workspace and audit root do not exist here. Stage A's arithmetic
# does not depend on either; redirecting them is what lets the real stage run.
D.WS, D.STATUS, D.AUDIT = tmp, tmp / "status", tmp / "audit"
driver = D.PhaseC2Driver(argparse.Namespace(
    stage="all", image_digest="zero-cost-probe", rate=1.09, spent_usd=0.0,
    soft_stop_usd=14.49, authorized_usd=15.0446, search_minutes=1.0,
    search_deadline_minutes=1.0, baseline_rebuild_minutes=27.665,
    authorization_path=sys.argv[3], device=sys.argv[4], top_n=5))
ok = driver.bind_identities()
print("RESULT " + json.dumps(
    {"ok": ok, "stage": driver.ev["stages"].get("bind_identities")}, default=str))
"""


def test_stage_a_executes_in_a_fresh_interpreter(tmp_path):
    """The real stage A, in the condition the pod runs it in.

    Not a proxy for the failing line -- the failing line, in a process where
    `conftest.py` has not pre-filled the registry. It resolves both mixtures
    against their recorded content hashes, registers the C2 operators and
    verifies the frozen baseline construction, which is everything stage A is
    for.
    """
    r = _fresh(_STAGE_A, str(REPO), str(tmp_path),
               _an_issued_authorization(), FROZEN_DEVICE)
    assert r.returncode == 0, f"stage A crashed:\n{r.stdout}\n{r.stderr}"
    line = next((ln for ln in r.stdout.splitlines()
                 if ln.startswith("RESULT ")), None)
    assert line, f"stage A produced no result:\n{r.stdout}\n{r.stderr}"
    result = json.loads(line[len("RESULT "):])
    assert result["ok"] is True, (
        f"stage A failed in a fresh interpreter: {result['stage']}")
    detail = result["stage"]["detail"]
    assert set(detail["calibration"]) == set(detail["space"]["profiles"]), (
        "stage A must resolve every profile the space declares")
    for qualified_id, m in detail["calibration"].items():
        assert m["n_items"] > 0, f"{qualified_id} resolved to nothing"
    # The pinned B spec. A stage A that passed while constructing a DIFFERENT
    # baseline would be worse than one that failed.
    construction = detail["baseline"]["construction"]
    assert construction["verified"] is True
    assert construction["spec_hash"] == construction["expected_spec_hash"]
    assert construction["spec_hash"] == detail["baseline"]["frozen_spec_hash"]


_EMPTY_REGISTRY = """
import pathlib, sys
REPO = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
from aadistill.initialization.calibration.profiles import (
    get_profile, registered_profiles)
print("REGISTERED " + repr(registered_profiles()))
try:
    get_profile("calib.domain_balanced@v1")
except KeyError as exc:
    print("RAISED " + str(exc))
else:
    print("RESOLVED")
"""


def test_the_core_registry_is_empty_without_the_application_bootstrap():
    """Why the check above has to leave the process, and why it is not vacuous.

    If the core registered the mixtures at import, `test_stage_a_executes...`
    would pass no matter where the driver imported the bootstrap, and the
    regression would be theatre. This pins the real mechanism: importing only
    `src` gives an EMPTY registry and the exact error attempt 3 died on.
    """
    r = _fresh(_EMPTY_REGISTRY, str(REPO))
    assert r.returncode == 0, r.stderr
    assert "REGISTERED []" in r.stdout, (
        "importing the core alone registered calibration mixtures; experiment "
        f"data has moved back into src/aadistill:\n{r.stdout}")
    assert "RAISED" in r.stdout and "registered: []" in r.stdout, r.stdout


def test_conftest_pre_fills_the_registry_for_every_in_process_test():
    """The blind spot, asserted, so the checks above are not 'simplified'.

    An in-process version of `test_stage_a_executes_in_a_fresh_interpreter`
    passes whether or not the driver imports the bootstrap. If this ever stops
    being true the subprocesses are no longer load-bearing -- but until then,
    deleting them restores the hole that cost a paid pod.
    """
    conftest = (REPO / "tests/conftest.py").read_text()
    assert "experiments.calibration" in conftest, (
        "tests/conftest.py no longer imports the calibration bootstrap; "
        "re-check whether in-process tests can now see an empty registry")
    from aadistill.initialization.calibration.profiles import registered_profiles
    assert registered_profiles(), (
        "this pytest process has no registered profiles, which contradicts "
        "the conftest import above")


# --- 2. the specs ------------------------------------------------------------


def _spec_documents() -> list[Path]:
    out = []
    for p in sorted((REPO / "configs/autoinit").glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and doc.get("spec") == SPEC_SCHEMA:
            out.append(p)
    return out


def test_every_artifact_spec_document_loads():
    """Through the collector's own loader, which is what runs on the pod.

    Generic on purpose. The invented lifecycle was in C2's two specs, but the
    failure mode -- an unloadable document, discovered by a pod during
    teardown -- belongs to every session, and each one has exactly one moment
    where it matters and it is not a cheap one.
    """
    docs = _spec_documents()
    assert len(docs) >= len(C2_SPECS)
    for path in docs:
        specs = load_specs(str(path))
        assert specs, f"{path.name} declares no artifacts"


def test_lifecycle_values_come_from_the_vocabulary():
    """Asserted on parsed VALUES, not on the file text.

    Both specs now explain in prose why `best_effort` was wrong, so the word
    appears in the documents and a substring check would fail on its own
    explanation.
    """
    assert LIFECYCLES == ("final_required", "mutable_snapshot")
    for path in _spec_documents():
        for entry in json.loads(path.read_text())["entries"]:
            assert entry.get("lifecycle", "final_required") in LIFECYCLES, (
                f"{path.name}: {entry['artifact_class']} declares "
                f"{entry.get('lifecycle')!r}. `required: false` is what makes "
                "an artifact best-effort; `lifecycle` says what a matching "
                "file means.")


def test_the_failed_spec_over_an_early_failure_is_complete_and_quiescent(tmp_path):
    """Attempt 3's exact artifact shape, through the real manifest builder.

    An early failure has the session record and the copied session logs and
    nothing else. Every other entry is `required: false`, so its absence must
    not appear in `missing`, and quiescence must hold -- that is what routes
    teardown through the clean emergency path instead of the strict one.
    """
    root = tmp_path / "artifacts"
    audit = root / "audit/autoinit_phase_c2"
    (audit / "session").mkdir(parents=True)
    (audit / "c2_evidence.json").write_text(
        (REPO / "logs/stages/stage-1/phase_c2/runs/attempt3/evidence/"
         "c2_evidence.json").read_text())
    (audit / "session/autoinit_phase_c2_run.log").write_text(
        "2026-09-15T18:24:46Z MARKER:STAGE_FAILED:bind_identities\n")

    manifest = build_manifest(
        str(root), load_specs(str(REPO / "configs/autoinit/c2_artifacts_failed.json")),
        created_utc="2026-09-16T00:00:00+00:00", settle_seconds=0)
    assert manifest.missing == [], manifest.missing
    assert manifest.ok
    assert manifest.final_streams_quiescent
    assert {e.artifact_class for e in manifest.entries} == {
        "session_evidence", "session_logs"}


# --- 3. the teardown route ---------------------------------------------------


#: Attempt 3's own teardown state, read from what the session recorded rather
#: than retyped, so this cannot drift into a shape no run ever had.
def _attempt3_state() -> dict:
    rec = json.loads(
        (REPO / "logs/stages/stage-1/phase_c2/runs/attempt3/runtime/"
         "session.json").read_text())
    gate = rec.get("teardown_gate") or {}
    if gate.get("checks"):
        return dict(gate["checks"])
    # The launcher raised before the decision was recorded, which is the defect
    # this file exists for. The state it was built from is what mattered.
    return {"training_complete": False, "evaluation_complete": False,
            "artifact_manifest_created": False, "required_files_present": False,
            "final_streams_quiescent": False, "archive_created": False,
            "transfer_complete": False, "local_hashes_verified": False,
            "checkpoint_hashes_matched": True, "report_inputs_verified": False,
            "required_products_secured": True}


def test_a_stream_less_session_records_the_loss_instead_of_raising():
    """No manifest, no declared streams: the recorded-loss route, not an abort.

    This is the composition the runner performs, with the helper in place of
    the inline branch it used to be.
    """
    state = _attempt3_state()
    decision = evaluate_teardown(
        state, emergency_budget=True,
        emergency_reason="a blocking stage failed (PHASE_C2_FAILED)",
        incomplete_event_streams=(),
        streams_at_risk=streams_at_risk(None, ()))
    assert decision.allowed and decision.emergency
    assert "No event stream was truncated" in decision.reason


def test_declared_streams_with_no_manifest_still_get_the_strict_rule():
    """The mutation that must keep failing.

    If `streams_at_risk` answered `()` whenever the manifest was missing, a
    session whose journal really was mid-write would be torn down with its
    truncation unrecorded. The weaker rule is earned by declaring no streams,
    not by losing a manifest.
    """
    assert streams_at_risk(None, ("artifacts/x/train_log.jsonl",)) is None
    with pytest.raises(ArtifactError, match="name the streams"):
        evaluate_teardown(
            _attempt3_state(), emergency_budget=True,
            emergency_reason="a blocking stage failed",
            incomplete_event_streams=(),
            streams_at_risk=streams_at_risk(
                None, ("artifacts/x/train_log.jsonl",)))


def test_a_manifest_is_believed_over_the_declaration():
    """When the manifest exists it is the evidence, streams declared or not."""
    class _M:
        completion_marker_failures = ["run.log does not contain DONE"]
        still_being_written = ["artifacts/x/train_log.jsonl"]

    assert streams_at_risk(_M(), ()) == (
        "run.log does not contain DONE", "artifacts/x/train_log.jsonl")
    assert streams_at_risk(_M(), ("a",)) == (
        "run.log does not contain DONE", "artifacts/x/train_log.jsonl")

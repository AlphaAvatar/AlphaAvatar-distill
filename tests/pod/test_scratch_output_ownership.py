"""A run collects its OWN outputs, or it refuses.

`open_c1_run` refused a colliding `run_id` under `logs/runs/`, and stopped
there. It said nothing about `--scr`, which is where the session's outputs
actually accumulate: `SessionRunner` does `mkdir(parents=True, exist_ok=True)`
on it, and `close_c1_run` copied every path in `_RUN_COLLECT` that happened to
exist.

So a fresh `run_id` pointed at a previous attempt's scratch, failing in
precheck or setup — before a driver ever starts — came home with the PREVIOUS
attempt's `c1_evidence.json`, driver log, status stream and artifact manifest
registered in its own manifest, as though this execution had produced them.
Nothing was overwritten and nothing looked wrong: the roles were present, the
files existed, `verify_run_manifest` passed.

That is the failure this module reproduces and then forbids. The negative case
is the point; the positive case is kept beside it so the fix cannot be "refuse
everything".
"""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from experiments.run_layout import RunConventionError, read_run
from session_specs import load_session_launcher

REPO = Path(__file__).resolve().parents[2]

#: What a previous attempt leaves behind, with content that names it.
STALE = {
    "launch.log": "attempt9 launcher\n",
    "watchdog.jsonl": '{"attempt": 9}\n',
    "relay/autoinit_c1_run.log": "attempt9 driver ran\n",
    "relay/autoinit_c1.status": "MARKER:ALL_DONE attempt9\n",
    "relay/c1_evidence.json": '{"attempt": 9, "stages": ["B", "C", "D"]}',
    "store/manifest.json": '{"attempt": 9, "artifacts": 42}',
}


@pytest.fixture(scope="module")
def L():
    return load_session_launcher("autoinit_c1_launch")


def _repo(tmp_path, L):
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True, exist_ok=True)
    (repo / L.PRICING).write_bytes((REPO / L.PRICING).read_bytes())
    (repo / L.AUTH_PATH).write_text('{"authorization_id": "test"}')
    return repo


def _fill(scr: Path) -> dict[str, str]:
    digests = {}
    for rel, body in STALE.items():
        p = scr / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        digests[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return digests


def _stale_scratch(tmp_path) -> tuple[Path, dict[str, str]]:
    """A scratch holding a previous attempt's outputs, with NO claim.

    The pre-convention shape: whatever a launcher left behind before output
    ownership was recorded at all. Ownership is unknown, not merely old.
    """
    scr = tmp_path / "shared_scr"
    scr.mkdir(parents=True, exist_ok=True)
    return scr, _fill(scr)


def _owned_stale_scratch(tmp_path, L, repo) -> tuple[Path, dict[str, str]]:
    """The same, but produced in the real order: attempt 9 claims, then writes.

    Claiming an EMPTY directory and producing into it is what a session
    actually does, so the collision this exercises is the one that can happen.
    """
    scr = tmp_path / "shared_scr"
    L.open_c1_run(_args(scr, "attempt9"), repo)
    return scr, _fill(scr)


def _args(scr, run_id):
    return types.SimpleNamespace(
        run_id=run_id, scr=str(scr), session_commit="a" * 40,
        bundle="aad_autoinit_aaaaaaaa.bundle")


def _write_session(repo, args, **over):
    body = {"session_id": "autoinit-c1", "session_plan_hash": "ph",
            "harness_source_digest": "hd", "passed": False, "terminal": None,
            "pod_id": "", "cost": None, "provider_confirms_gone": None}
    body.update(over)
    p = repo / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body))


def _unchanged(scr: Path, digests: dict[str, str]) -> None:
    for rel, want in digests.items():
        got = hashlib.sha256((scr / rel).read_bytes()).hexdigest()
        assert got == want, f"{rel} was modified; old evidence must be inert"


# --- the negative case ------------------------------------------------------

def test_a_new_run_may_not_take_over_another_runs_scratch(tmp_path, L):
    """The reproduction. A different `run_id`, the same output location.

    The refusal must happen at OPEN — before the `SessionSpec` is built and so
    before anything can be created or billed.
    """
    repo = _repo(tmp_path, L)
    scr, digests = _owned_stale_scratch(tmp_path, L, repo)

    #: Attempt 10 points at it. This is the thing that used to be silent.
    with pytest.raises(RunConventionError) as exc:
        L.open_c1_run(_args(scr, "attempt10"), repo)
    assert "attempt9" in str(exc.value)
    _unchanged(scr, digests)


def test_the_owning_run_may_reopen_its_own_scratch(tmp_path, L):
    """The claim is idempotent for its owner.

    A guard that tripped on the run that created the marker would make the
    launcher unable to run at all.
    """
    repo = _repo(tmp_path, L)
    scr, _ = _owned_stale_scratch(tmp_path, L, repo)
    from experiments.run_layout import claim_output_root

    again = claim_output_root(scr, "phase_c1", "attempt9",
                              outputs=L.RUN_OUTPUTS)
    assert again["run_id"] == "attempt9"


def test_an_unclaimed_scratch_holding_outputs_is_refused_as_ambiguous(tmp_path,
                                                                      L):
    """No claim at all, but the outputs are already there.

    Ownership is UNKNOWN, not absent. Guessing from mtime would be a guess, and
    deleting it would destroy evidence, so the run refuses and says so.
    """
    repo = _repo(tmp_path, L)
    scr, digests = _stale_scratch(tmp_path)

    with pytest.raises(RunConventionError) as exc:
        L.open_c1_run(_args(scr, "attempt10"), repo)
    msg = str(exc.value)
    assert "relay/c1_evidence.json" in msg or "6" in msg
    _unchanged(scr, digests)


def test_the_stale_evidence_never_reaches_a_manifest(tmp_path, L):
    """The consequence, stated directly.

    Even reaching closeout with a foreign scratch must not register another
    run's driver evidence as this execution's.
    """
    repo = _repo(tmp_path, L)
    scr, digests = _owned_stale_scratch(tmp_path, L, repo)

    hijack = _args(scr, "attempt10")
    hijack.out = "logs/runs/phase_c1/attempt10/runtime/session.json"
    layout = L.layout_for_run(repo, "attempt10")
    layout.create(dict(L.C1_RUN_ROLES))
    _write_session(repo, hijack)

    with pytest.raises(RunConventionError):
        L.close_c1_run(layout, hijack, repo)

    assert not layout.path("evidence/c1_evidence.json").exists()
    assert not layout.path("artifacts/manifest.json").exists()
    _unchanged(scr, digests)


# --- the positive case, kept beside it --------------------------------------

def test_a_run_with_its_own_scratch_collects_normally(tmp_path, L):
    """The fix must not be "refuse everything"."""
    repo = _repo(tmp_path, L)
    args = _args(tmp_path / "own_scr", "attempt10")
    layout = L.open_c1_run(args, repo)

    scr = Path(args.scr)
    for rel, body in STALE.items():
        p = scr / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body.replace("attempt9", "attempt10").replace('"attempt": 9', '"attempt": 10'))
    _write_session(repo, args, passed=True, terminal="ALL_DONE")

    doc = L.close_c1_run(layout, args, repo)
    assert {"driver_evidence", "driver_log", "driver_status",
            "artifact_manifest"} <= set(doc["roles"])
    assert "attempt10" in layout.path("evidence/driver_status.txt").read_text()
    assert read_run(repo, "phase_c1", "attempt10")["self_sha256"] == doc["self_sha256"]


def test_reopening_the_same_run_id_on_its_own_scratch_is_the_run_id_rule(
        tmp_path, L):
    """Scratch ownership does not weaken the run-directory rule.

    A recorded run is still not reopened, and the message must be about the
    run, not about the scratch — the two guards answer different questions.
    """
    repo = _repo(tmp_path, L)
    args = _args(tmp_path / "own_scr", "attempt10")
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args)
    L.close_c1_run(layout, args, repo)

    with pytest.raises(RunConventionError) as exc:
        L.open_c1_run(_args(tmp_path / "own_scr", "attempt10"), repo)
    assert "manifest.json" in str(exc.value)


def test_a_shared_read_only_input_does_not_make_a_scratch_ambiguous(tmp_path, L):
    """Inputs and caches are not this run's outputs.

    A scratch may legitimately hold a shared model cache or a staged input. Only
    the paths this run declares as its OWN outputs decide ownership; refusing on
    any pre-existing file would make the guard unusable.
    """
    repo = _repo(tmp_path, L)
    scr = tmp_path / "with_cache"
    (scr / "hf_cache" / "models").mkdir(parents=True)
    (scr / "hf_cache" / "models" / "blob.bin").write_bytes(b"\x00" * 32)
    (scr / "assets").mkdir()
    (scr / "assets" / "battery.tar").write_bytes(b"\x00" * 16)

    layout = L.open_c1_run(_args(scr, "attempt10"), repo)
    assert layout.rel_root == "phase_c1/attempt10"
    assert (scr / "hf_cache" / "models" / "blob.bin").exists()


# --- the other real consumer uses the same mechanism ------------------------

def _cuda():
    import importlib.util

    path = REPO / "scripts/validation/cuda_engineering_launch.py"
    spec = importlib.util.spec_from_file_location("cuda_engineering_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cuda_engineering(scr: Path, run_id: str):
    """`write_evidence`'s receiver, without the constructor's provider setup."""
    mod = _cuda()
    eng = object.__new__(mod.Engineering)
    eng.a = types.SimpleNamespace(run_id=run_id, execution_sha="a" * 40,
                                  image="img:tag")
    eng.scr = scr
    eng.ev = {"verdict": "CUDA ENGINEERING VALIDATION PASS", "pod_id": "p1",
              "subrun_cost_usd": 0.0182, "campaign_cost_after_usd": 0.04}
    return mod, eng


def test_the_cuda_validation_claims_and_requires_the_same_way(tmp_path):
    """One mechanism, two experiments, disjoint output vocabularies."""
    from experiments.run_layout import claim_output_root

    mod = _cuda()
    assert set(mod.RUN_OUTPUTS) == {"validation_stdout.txt", "watchdog.jsonl",
                                    "artifacts"}
    L = load_session_launcher("autoinit_c1_launch")
    #: The RUN-ROLE vocabularies are disjoint — that is the mechanism-vs-instance
    #: property. The scratch OUTPUTS are not, and must not be assumed to be:
    #: both sessions detach the same `watchdog.py` and both call its journal
    #: `watchdog.jsonl`. Which is the point — a filename cannot say which run
    #: produced it, so ownership is decided by the claim on the directory.
    assert not set(mod.RUN_ROLES) & set(L.C1_RUN_ROLES)
    assert set(mod.RUN_OUTPUTS) & set(L.RUN_OUTPUTS) == {"watchdog.jsonl"}

    repo, scr = tmp_path / "repo", tmp_path / "scr"
    (repo / "logs").mkdir(parents=True)
    scr.mkdir()
    claim_output_root(scr, "cuda_stage_f", "subrun_a", outputs=mod.RUN_OUTPUTS)
    (scr / "validation_stdout.txt").write_text("subrun_a stdout\n")

    _, eng = _cuda_engineering(scr, "subrun_a")
    eng.write_evidence(repo)
    doc = read_run(repo, "cuda_stage_f", "subrun_a")
    assert "validation_stdout" in doc["roles"]


def test_a_cuda_subrun_may_not_collect_another_subruns_scratch(tmp_path):
    """The negative case, for the second consumer.

    Same defect shape: `--scr` is a writable output root independent of
    `logs/runs/`, and three subruns of one campaign share a machine.
    """
    from experiments.run_layout import claim_output_root

    mod = _cuda()
    repo, scr = tmp_path / "repo", tmp_path / "scr"
    (repo / "logs").mkdir(parents=True)
    scr.mkdir()
    claim_output_root(scr, "cuda_stage_f", "subrun_a", outputs=mod.RUN_OUTPUTS)
    (scr / "validation_stdout.txt").write_text("subrun_a stdout\n")
    before = hashlib.sha256((scr / "validation_stdout.txt").read_bytes()).hexdigest()

    _, eng = _cuda_engineering(scr, "subrun_b")
    with pytest.raises(RunConventionError) as exc:
        eng.write_evidence(repo)
    assert "subrun_a" in str(exc.value)

    assert not (repo / "logs/runs/cuda_stage_f/subrun_b").exists()
    after = hashlib.sha256((scr / "validation_stdout.txt").read_bytes()).hexdigest()
    assert after == before, "the other subrun's stdout must be inert"

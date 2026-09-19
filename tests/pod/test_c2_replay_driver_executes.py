"""The replay driver, executed for real at toy scale.

Four paid pods in this programme have died in lines no rehearsal ever ran, so
nothing here stubs the work. `materialize_fixed_path` really runs, the digest
gate really fires, the identity assertion really compares, the leaf is really
copied and the sidecar is really written. The only thing replaced is the
function that reads attempt 3's committed evidence off disk — which has its own
tests against the real files — because a toy run cannot reproduce a 4B teacher's
digests.

The mismatch case matters as much as the success case: it is the one branch that
must never be reached on the pod, which is exactly why it has never been
executed there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "tests/autoinit"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import (  # noqa: E402
    register_profile, unregister_profile,
)
from aadistill.initialization.operators import attention_activation  # noqa: E402
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathSpec, FixedPathStep, materialize_fixed_path,
)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

import importlib  # noqa: E402
import importlib.util  # noqa: E402


def _load(name, relative):
    """Load by path under a unique name.

    `tests/conftest.py` already owns the module name `conftest`, so a plain
    import of the autoinit fixtures silently returns the wrong module.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_autoinit_fixtures = _load("autoinit_conftest", "tests/autoinit/conftest.py")
TEACHER_GEOMETRY = _autoinit_fixtures.TEACHER_GEOMETRY
build_tiny_model = _autoinit_fixtures.build_tiny_model
make_items = _autoinit_fixtures.make_items
make_profile = _autoinit_fixtures.make_profile

driver_mod = importlib.import_module("pod.autoinit_c2_replay_driver")

TARGET = dict(TEACHER_GEOMETRY, intermediate_size=24, num_attention_heads=2)
PROFILE_ID = "test.balanced@v1"


@pytest.fixture(autouse=True)
def operators_registered():
    attention_activation.register(replace=True)
    profile = make_profile("balanced")
    register_profile(profile, replace=True)
    yield
    attention_activation.unregister()
    unregister_profile(profile.profile_id)


def toy_spec(path_id, pinned=(None, None)):
    return FixedPathSpec(
        path_id=path_id, family="qwen3", target_spec=ArchSpec.of("qwen3", TARGET),
        steps=(
            FixedPathStep("ffn.activation_importance_v0", PROFILE_ID,
                          expected_artifact_digest=pinned[0], label="ffn"),
            FixedPathStep("attention.weight_proxy_v0", "calib.none@v1",
                          expected_artifact_digest=pinned[1], label="attn"),
        ),
        root_repo_id="test/teacher", root_revision="deadbeef", device="cpu")


def true_digests(tmp_path, calib):
    """Run the toy path once UNPINNED to learn what it really produces.

    The pins must come from a real execution, not from a constant: a test that
    pinned a hand-written digest would only prove the gate compares strings.
    """
    results = materialize_fixed_path(
        toy_spec("probe"), adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=tmp_path / "probe", calibration_items=calib)
    return [r.identity for r in results]


class Args:
    def __init__(self, **kw):
        self.device = "cpu"
        self.rate = 1.09
        self.authorized_usd = 5.0
        self.soft_stop_usd = 4.0
        self.image_digest = "sha256:toy"
        self.already_spent_usd = 0.0
        #: The collector patterns the evidence here, not in the
        #: workdir; the real launcher passes it explicitly.
        self.audit_dir = None
        self.__dict__.update(kw)


def install_toy_leaves(monkeypatch, leaves, *, calib):
    """Replace only the evidence reader; everything downstream stays real."""
    from experiments.phase_c2 import replay_specs

    monkeypatch.setattr(replay_specs, "build_replay_leaves",
                        lambda *a, **k: leaves)
    monkeypatch.setattr(replay_specs, "source_binding", lambda *a, **k: {
        "source_session_commit": "0" * 40,
        "selection_sha256": "1" * 64,
        "operators": {"verdict": "toy"},
    })
    monkeypatch.setattr(driver_mod, "N_LEAVES", len(leaves))
    #: The toy path's artifacts are kilobytes; the real headroom rule is
    #: exercised by its own test below rather than by inflating this one.
    monkeypatch.setattr(driver_mod, "WORST_PATH_BYTES", 1024)
    monkeypatch.setattr(driver_mod, "LEAF_BYTES", 1024)


def build_leaf(identity, spec, state_id):
    from experiments.phase_c2.replay_specs import ReplayLeaf

    return ReplayLeaf(
        state_id=state_id, path_label="FFN->ATTENTION", lineage="toy",
        spec=spec,
        artifact_digest=identity.artifact_digest,
        weights_digest=identity.weights_digest,
        single_shard_sha256=identity.single_shard_sha256,
        arch_signature=identity.arch_signature,
        num_parameters=identity.num_parameters,
        step_digests=(identity.artifact_digest,),
        bounded_minutes=0.5)


@pytest.fixture
def calib():
    return {PROFILE_ID: make_items()}


@pytest.fixture
def toy_root(monkeypatch, calib):
    """Replace the ONE function a toy run cannot execute, and record its calls.

    `load_root` downloads a 4B teacher. Everything else in the driver —
    `materialize_fixed_path`, the digest gate, the identity assertion, the copy,
    the sidecar, the markers, the cleanup — runs for real.

    Before this seam existed the driver read the teacher from a module constant,
    so the first version of this rehearsal spent 9 minutes 39 seconds pulling
    Qwen3-4B into a unit test and then failed on a digest that could not have
    matched. The calls are recorded so the wiring the download would have
    exercised is still asserted.
    """
    import aadistill.initialization.planning.fixed_path as fp

    calls = []

    def fake_load_root(spec, device):
        calls.append({"root_repo_id": spec.root_repo_id,
                      "root_revision": spec.root_revision,
                      "device": device, "path_id": spec.path_id})
        return build_tiny_model(TEACHER_GEOMETRY)

    monkeypatch.setattr(driver_mod, "load_root", fake_load_root)

    #: The driver deliberately passes no `calibration_items`, so production
    #: resolves profiles from the repository. A toy profile has no repository
    #: data, so the items are supplied by WRAPPING the real materializer rather
    #: than replacing it.
    original = fp.materialize_fixed_path

    def wrapper(spec, **kw):
        kw.setdefault("calibration_items", calib)
        return original(spec, **kw)

    monkeypatch.setattr(fp, "materialize_fixed_path", wrapper)
    return calls


def test_the_driver_reconstructs_pinned_leaves_and_secures_each_one(
        tmp_path, monkeypatch, calib, toy_root):
    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    leaves = [build_leaf(ids[-1], toy_spec(f"leaf{i}", pins), f"state{i}")
              for i in range(2)]
    install_toy_leaves(monkeypatch, leaves, calib=calib)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"))
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 0

    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["outcome"] == "ALL_DONE"
    assert ev["n_reconstructed"] == 2
    #: The claims a reviewer checks instead of reading the prose.
    for claim in ("runs_beam_search", "ranks_or_reranks", "produces_a_selection",
                  "injects_canonical_control", "trains_anything",
                  "measures_behaviour", "substitutes_candidates"):
        assert ev[claim] is False, claim

    for i in range(2):
        dest = work / "leaves" / f"state{i}"
        sidecar = json.loads((dest / "replay_leaf.json").read_text())
        assert sidecar["identity"]["artifact_digest"] == ids[-1].artifact_digest
        assert sidecar["identity"]["tokenizer_sha256"] is None
        assert (dest / "model.safetensors").exists()

    #: Secured before the next path starts, and the intermediates released after.
    assert not (work / "paths" / "state0").exists()
    assert not (work / "paths" / "state1").exists()

    #: The root came from each SPEC, not from a module constant, and each path
    #: got its own load. This is the wiring the 4B download would have covered.
    assert [c["path_id"] for c in toy_root] == ["leaf0", "leaf1"]
    assert {c["root_repo_id"] for c in toy_root} == {"test/teacher"}
    assert {c["root_revision"] for c in toy_root} == {"deadbeef"}
    assert {c["device"] for c in toy_root} == {"cpu"}


def test_a_digest_mismatch_stops_the_session_without_retrying(
        tmp_path, monkeypatch, calib, toy_root, capsys):
    ids = true_digests(tmp_path, calib)
    #: Pin the SECOND step to something the path cannot produce. The first step
    #: still matches, so this proves the gate catches a divergence part-way in
    #: rather than only at the end.
    leaves = [build_leaf(ids[-1],
                         toy_spec("bad", (ids[0].artifact_digest, "0" * 64)),
                         "state0")]
    install_toy_leaves(monkeypatch, leaves, calib=calib)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"))
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1

    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["outcome"] == "FAILED"
    assert ev["digest_mismatch"]["step_index"] == 1
    assert ev["digest_mismatch"]["expected"] == "0" * 64
    assert ev["digest_mismatch"]["actual"] == ids[1].artifact_digest
    mismatch = [s for s in ev["stages"] if s["stage"] == "digest_mismatch"]
    assert len(mismatch) == 1, "recorded once — a mismatch is not retried"
    assert "not retried and not substituted" in mismatch[0]["verdict"].lower()

    out = capsys.readouterr().out
    assert f"MARKER:{driver_mod.MISMATCH_MARKER}:state0" in out
    assert f"MARKER:{driver_mod.FAILURE_MARKER}" in out
    #: No leaf may be published when the path diverged.
    assert not (work / "leaves" / "state0").exists()


def test_an_identity_that_matches_the_digest_but_not_attempt_3_is_refused(
        tmp_path, monkeypatch, calib, toy_root):
    """The digest gate cannot see the fields it does not hash.

    This programme has twice shipped checkpoints whose identity gates all passed
    and which could not be used, so the parameter count and the shard hash are
    compared against the SELECTION even though the digest already matched.
    """
    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    leaf = build_leaf(ids[-1], toy_spec("leaf", pins), "state0")
    from dataclasses import replace

    wrong = replace(leaf, num_parameters=leaf.num_parameters + 1)
    install_toy_leaves(monkeypatch, [wrong], calib=calib)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"))
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1

    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    failed = [s for s in ev["stages"] if not s["ok"]][-1]
    assert "num_parameters" in failed["error"]
    assert not (work / "leaves" / "state0").exists()


def test_an_unpinned_intermediate_is_refused_before_anything_is_computed(
        tmp_path, monkeypatch, calib):
    """Pinning only the leaf would leave its operators unverified."""
    ids = true_digests(tmp_path, calib)
    leaf = build_leaf(ids[-1], toy_spec("leaf", (None, ids[1].artifact_digest)),
                      "state0")
    install_toy_leaves(monkeypatch, [leaf], calib=calib)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"))
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1
    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["failed_stage"] == "bind_identities"
    assert "1 of 2 steps pinned" in ev["stages"][-1]["error"]


def test_insufficient_disk_is_refused_at_bind_not_at_path_four(
        tmp_path, monkeypatch, calib):
    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    leaves = [build_leaf(ids[-1], toy_spec("leaf", pins), "state0")]
    install_toy_leaves(monkeypatch, leaves, calib=calib)
    #: More than any test filesystem has.
    monkeypatch.setattr(driver_mod, "WORST_PATH_BYTES", 1 << 60)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"))
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1
    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["failed_stage"] == "bind_identities"
    assert "Provision the disk" in ev["stages"][-1]["error"]


def test_the_soft_stop_keeps_the_leaves_it_already_secured(
        tmp_path, monkeypatch, calib, toy_root):
    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    leaves = [build_leaf(ids[-1], toy_spec(f"leaf{i}", pins), f"state{i}")
              for i in range(2)]
    install_toy_leaves(monkeypatch, leaves, calib=calib)

    work = tmp_path / "work"
    #: A soft stop already exhausted when the driver starts, so the FIRST path
    #: is refused and the record must still be coherent.
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"),
                soft_stop_usd=0.0, authorized_usd=5.0)
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1
    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["stopped_at_soft_stop"] is True
    assert ev["n_reconstructed"] == 0
    assert "does not invalidate them" in ev["_partial_is_not_waste"]


def test_the_soft_stop_must_leave_room_for_teardown():
    parser = driver_mod.build_parser()
    args = parser.parse_args([
        "--workdir", "/tmp/x", "--rate", "1.09",
        "--authorized-usd", "5.00", "--soft-stop-usd", "5.00"])
    assert args.soft_stop_usd == 5.00
    #: main() is where the relation is enforced; the parser does not know it.
    sys.argv = ["driver", "--workdir", "/tmp/x", "--rate", "1.09",
                "--authorized-usd", "5.00", "--soft-stop-usd", "5.00"]
    with pytest.raises(SystemExit) as exc:
        driver_mod.main()
    assert "teardown" in str(exc.value)


def test_setup_spend_counts_against_the_soft_stop(
        tmp_path, monkeypatch, calib, toy_root):
    """The driver's own clock is not the session's clock.

    Setup in this programme has taken 5 minutes and 150 minutes for the same
    script and image. A soft stop measured only from the driver's start would
    believe the whole budget remained while most of it was already billed.
    """
    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    leaves = [build_leaf(ids[-1], toy_spec("leaf0", pins), "state0")]
    install_toy_leaves(monkeypatch, leaves, calib=calib)

    work = tmp_path / "work"
    #: Setup already spent past the soft stop before the driver ever ran.
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"),
                already_spent_usd=4.10, soft_stop_usd=4.00)
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1

    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["stopped_at_soft_stop"] is True
    assert ev["already_spent_usd_at_driver_start"] == 4.10
    assert ev["n_reconstructed"] == 0
    assert not (work / "leaves" / "state0").exists()


def test_a_path_is_not_started_unless_the_budget_can_see_it_finish(
        tmp_path, monkeypatch, calib, toy_root):
    """Admission control, not a trip-wire.

    Checking only that SOMETHING remains would let a 30-minute path begin on two
    minutes of budget and overshoot by the whole difference, which is how a soft
    stop quietly becomes advisory. The first path is affordable here and the
    second is not, so exactly one leaf must be reconstructed and secured.
    """
    from dataclasses import replace

    ids = true_digests(tmp_path, calib)
    pins = (ids[0].artifact_digest, ids[1].artifact_digest)
    cheap = build_leaf(ids[-1], toy_spec("leaf0", pins), "state0")
    #: Bounded well beyond what the remaining budget can fund.
    dear = replace(build_leaf(ids[-1], toy_spec("leaf1", pins), "state1"),
                   bounded_minutes=600.0)
    install_toy_leaves(monkeypatch, [cheap, dear], calib=calib)

    work = tmp_path / "work"
    args = Args(workdir=str(work), leaf_dir=str(work / "leaves"),
                audit_dir=str(work / "audit"),
                rate=1.09, soft_stop_usd=1.00, authorized_usd=5.00)
    rc = driver_mod.ReplayDriver(args).run()
    assert rc == 1

    ev = json.loads((work / "audit" / "c2_replay_evidence.json").read_text())
    assert ev["stopped_at_soft_stop"] is True
    assert ev["n_reconstructed"] == 1
    stop = [s for s in ev["stages"] if s["stage"] == "soft_stop"][0]
    #: 600 min at $1.09/h is $10.90, against $1.00 of budget.
    assert stop["path_bounded_usd"] == pytest.approx(10.90, abs=0.01)
    assert stop["reconstructed"] == 1

    #: The affordable leaf was still secured before the stop.
    assert (work / "leaves" / "state0" / "replay_leaf.json").exists()
    assert not (work / "leaves" / "state1").exists()

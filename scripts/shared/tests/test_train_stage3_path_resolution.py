"""A frozen config's paths must reach the migrated physical layout.

E6b's two arm configs are hashed by `e6b_registration.json`, so they keep the
spellings they were registered with — `artifacts/stage1/...`,
`artifacts/stage3/...`. The 2026-10-08 migration moved the objects those
strings name. Its driver and setup script compute the MIGRATED paths. The
trainer sat between the two reading `REPO_ROOT / cfg[...]` directly, so it
would have loaded from and written to directories neither side uses — in
particular the driver would have waited on an output directory the trainer
never created.

`train_stage3.at()` is the single boundary where a config's declared path
becomes a physical one. These tests prove it lands where the driver expects,
that a post-migration config is unaffected, and that the frozen config
dictionary and its hash are never touched.

CPU only. No training runs here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

E6B = REPO / "configs/stages/stage-3/e6b"


@pytest.fixture(scope="module")
def trainer():
    """The trainer module, loaded without running it."""
    spec = importlib.util.spec_from_file_location(
        "train_stage3_under_test",
        REPO / "scripts/shared/training/train_stage3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver_out_dir():
    """How the E6b driver computes an arm's output directory."""
    src = (REPO / "scripts/stages/stage-3/e6b/e6b_driver.py").read_text()
    line = next(l for l in src.splitlines()
                if 'return REPO / f"artifacts' in l)
    prefix = line.split('f"')[1].split('{')[0]      # e.g. artifacts/stages/stage-3/
    return lambda name: prefix + name


# --- the defect, stated as a test ------------------------------------------

@pytest.mark.parametrize("seed", ["sa", "sb"])
def test_the_arm_output_dir_is_where_the_driver_waits_for_it(
        trainer, driver_out_dir, seed):
    cfg = json.loads((E6B / f"e6b_p2_r2960k_{seed}.json").read_text())
    name = f"e6b_p2_r2960k_{seed}"

    # the frozen config keeps its registered spelling ...
    assert cfg["out_dir"] == f"artifacts/stage3/{name}"
    # ... the driver computes the migrated one ...
    assert driver_out_dir(name) == f"artifacts/stages/stage-3/{name}"
    # ... and what the TRAINER will actually use reconciles them. Asked of
    # `effective_paths`, which is the mapping `main()` consumes, so this
    # fails if the resolution is removed from the execution path rather than
    # merely from one expression.
    paths = trainer.effective_paths(cfg)
    assert paths["out_dir"] == REPO / driver_out_dir(name)
    assert paths["checkpoints"] == REPO / driver_out_dir(name) / "checkpoints"
    assert paths["train_log"] == REPO / driver_out_dir(name) / "train_log.jsonl"


@pytest.mark.parametrize("seed", ["sa", "sb"])
def test_every_read_path_in_the_frozen_arm_lands_in_the_owner_first_layout(
        trainer, seed):
    """`student_path` and `data_dir` keep their registered spellings and
    resolve into the migrated layout.

    Existence is asserted only for objects a dev box actually holds: the
    token pack is a gitignored artifact the setup script fetches onto the
    pod, so requiring it here would make the test describe this machine
    rather than the resolution.
    """
    cfg = json.loads((E6B / f"e6b_p2_r2960k_{seed}.json").read_text())
    paths = trainer.effective_paths(cfg)
    for key in ("student_path", "data_dir"):
        assert cfg[key].startswith(("artifacts/stage1/", "artifacts/stage3/")), (
            f"{key} no longer carries its registered spelling: {cfg[key]!r}")
        resolved = paths[key]
        assert resolved != REPO / cfg[key], (
            f"{key} resolved to its own literal, so nothing was resolved")
        rel = resolved.relative_to(REPO).as_posix()
        assert rel.startswith("artifacts/stages/stage-"), rel
        if (REPO / cfg[key]).exists():
            pytest.fail(f"{key}: the pre-migration path still exists, so this "
                        "test cannot tell resolution from a leftover copy")

    # the student init IS present on a dev box, so check it for real
    assert paths["student_path"].is_dir()


def test_the_setup_script_and_the_trainer_agree_on_the_corpus(trainer):
    """E6b's setup stages the pack at the migrated path and asserts it is
    there; the trainer must read that same directory."""
    setup = (REPO / "scripts/stages/stage-3/e6b/e6b_setup.sh").read_text()
    cfg = json.loads((E6B / "e6b_p2_r2960k_sa.json").read_text())
    resolved = trainer.effective_paths(cfg)["data_dir"].relative_to(
        REPO).as_posix()
    assert resolved in setup, (
        f"the trainer would read {resolved}, which the setup script never "
        "stages or checks")


# --- what must NOT change --------------------------------------------------

@pytest.mark.parametrize("seed", ["sa", "sb"])
def test_resolution_does_not_touch_the_frozen_config_or_its_hash(
        trainer, seed):
    """The registered identity is the config dictionary. Resolution happens
    on the way to the filesystem and nowhere else."""
    raw = (E6B / f"e6b_p2_r2960k_{seed}.json").read_text()
    cfg = json.loads(raw)
    before = sha256_json(cfg)

    trainer.effective_paths(cfg)

    assert cfg == json.loads(raw), "the config dictionary was mutated"
    assert sha256_json(cfg) == before
    reg = json.loads((REPO / "logs/stages/stage-3/e6b/analyses/"
                      "e6b_registration.json").read_text())
    alias = f"P2-2.96M-{seed}"
    assert before == reg["arms"][alias]["config_sha256"], (
        "the arm no longer hashes to its prospectively registered identity")


def test_a_current_probe_config_resolves_to_itself(trainer):
    """D1's recovery probes are built at runtime from the Phase-A driver's
    template, which already spells canonical paths. For those the boundary
    must be the identity function, or this repair would have changed the
    behaviour of current work.

    The template's own values are read from the driver rather than retyped,
    so if it ever emits a different spelling this test sees it.
    """
    src = (REPO / "scripts/stages/stage-1/phase_a/autoinit_phase_a_driver.py"
           ).read_text()
    pack = next(l.split('"')[1] for l in src.splitlines()
                if l.startswith("PACK_DIR = "))
    out = next(l.split('f"')[1].split('"')[0] for l in src.splitlines()
               if '"out_dir": f"artifacts' in l)
    out = out.replace("{name}", "probe_x")
    emitted = {"data_dir": pack, "out_dir": out,
               "student_path": "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"}
    paths = trainer.effective_paths(emitted)
    for key, value in emitted.items():
        assert paths[key] == REPO / value, (
            f"{key}: a canonical path was rewritten by resolution ({value})")


def test_a_canonical_path_is_never_rewritten(trainer):
    """Stated generically: anything already under the owner-first layout is
    returned unchanged."""
    for rel in ("artifacts/stages/stage-3/ladder_uniform",
                "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint",
                "artifacts/stages/stage-3/e6b_p2_r2960k_sa"):
        assert trainer.at(rel) == REPO / rel


def test_the_boundary_is_used_everywhere_a_config_path_is_opened(trainer):
    """A path read that bypasses `at()` is the defect returning. Checked in
    the source, because the alternative is running a training job."""
    src = (REPO / "scripts/shared/training/train_stage3.py").read_text()
    for expr in ('REPO_ROOT / cfg["out_dir"]',
                 'REPO_ROOT / cfg["student_path"]',
                 'REPO_ROOT / cfg["data_dir"]',
                 'REPO_ROOT / cfg["extra_stream"]["data_dir"]'):
        assert expr not in src, f"{expr} bypasses the resolution boundary"
    assert "paths = effective_paths(cfg)" in src, (
        "main() no longer derives its paths from the one mapping")
    for key in ("out_dir", "student_path", "data_dir"):
        assert f'at(cfg["{key}"])' in src, f"{key} is not resolved"

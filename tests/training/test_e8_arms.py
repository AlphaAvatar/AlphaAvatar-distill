"""E8 arm identity: the depth map is the only variable, asserted from the files.

The shipped configs are checked, not a reconstruction of them, because the thing
that trains is the file on disk. Every field that could change a gradient is
compared against the E1/P1 2.96M control it forked from, and the token budget is
read out of the pack rather than copied from the preregistration.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.init.sandwich import depth_span_map  # noqa: E402

ARMS = json.loads((REPO / "configs/stage3/e8/arms.json").read_text())
ALLOWED_DIFF = {"student_path", "run_name", "out_dir", "_purpose"}
GRADIENT_RELEVANT = (
    "loss", "optim", "schedule", "batch", "trainable_patterns", "dtype",
    "autocast_bf16", "gradient_checkpointing", "rung", "block_len", "data_dir",
    "packing", "groups", "intervals", "val_blocks", "checkpoint", "teacher",
)


def cfg_of(arm: dict) -> dict:
    return json.loads((REPO / arm["path"]).read_text())


def control_of(arm: dict) -> dict:
    return json.loads((REPO / arm["control"]).read_text())


def test_there_are_exactly_two_arms_on_two_seeds_from_one_init():
    assert len(ARMS["arms"]) == 2
    assert {a["seed"] for a in ARMS["arms"]} == {20260726, 20260801}
    assert {cfg_of(a)["student_path"] for a in ARMS["arms"]} == \
        {"artifacts/stage1/e8_contribution_init_v1/checkpoint"}
    assert ARMS["allowed_diff"] == sorted(ALLOWED_DIFF)


@pytest.mark.parametrize("arm", ARMS["arms"], ids=lambda a: a["name"])
def test_the_realized_diff_against_the_control_is_exactly_the_allowed_set(arm):
    cfg, control = cfg_of(arm), control_of(arm)
    realized = {k for k in set(cfg) | set(control)
                if json.dumps(cfg.get(k), sort_keys=True)
                != json.dumps(control.get(k), sort_keys=True)}
    assert realized == ALLOWED_DIFF


@pytest.mark.parametrize("arm", ARMS["arms"], ids=lambda a: a["name"])
def test_every_gradient_relevant_field_is_byte_identical_to_the_control(arm):
    cfg, control = cfg_of(arm), control_of(arm)
    for key in GRADIENT_RELEVANT:
        assert json.dumps(cfg.get(key), sort_keys=True) == \
            json.dumps(control.get(key), sort_keys=True), key


@pytest.mark.parametrize("arm", ARMS["arms"], ids=lambda a: a["name"])
def test_the_arm_is_the_kd_heavy_2960k_recipe_and_not_something_adjacent(arm):
    cfg = cfg_of(arm)
    assert cfg["rung"] == 2_960_000
    assert cfg["loss"] == {"ce_weight": 0.25, "kd_weight": 1.0,
                           "kd_temperature": 1.0, "kd_scope": "all"}
    assert cfg["schedule"]["total_steps"] == 2916
    assert cfg["schedule"]["warmup_steps"] == 146
    assert cfg["batch"] == {"blocks_per_step": 2, "micro_blocks": 1}
    assert cfg["block_len"] == 8192
    assert cfg["optim"]["lr"] == 5e-05
    assert cfg["packing"] == "ladder"


def test_the_stage1_treatment_config_changes_only_the_depth_map():
    base = json.loads(
        (REPO / "configs/stage1/qwen3_0p6b_from_4b_thinking.json").read_text())
    treat = json.loads(
        (REPO / "configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json"
         ).read_text())
    realized = {k for k in set(base) | set(treat)
                if json.dumps(base.get(k), sort_keys=True)
                != json.dumps(treat.get(k), sort_keys=True)}
    assert realized == {"depth_map_path", "output_dir",
                        "save_random_baseline", "_purpose"}
    # The parts that must not move: same teacher, revision, seed, statistics,
    # geometry. The projection, head rule, FFN rule and norm treatment are code
    # reached identically by both, which test_contribution.py proves bitwise.
    for key in ("teacher_model_id", "teacher_revision", "dtype", "seed",
                "stats_dir", "student_geometry", "stage"):
        assert base[key] == treat[key], key
    assert treat["save_random_baseline"] is False


def test_the_token_budget_the_arms_will_consume_is_what_the_pack_holds():
    from aadistill.data.ladder import ladder_blocks

    pack = REPO / "artifacts/stage3/ladder_uniform_probe"
    if not (pack / "blocks.npz").is_file():
        pytest.skip("canonical pack not present on this machine")
    arm = ARMS["arms"][0]
    cfg = cfg_of(arm)
    train, _, _ = ladder_blocks(pack, cfg["rung"], n_val=cfg["val_blocks"])
    ids, mask, _ = train
    assert int(ids.shape[0]) == 1944
    assert int(mask.sum()) == 2_960_507
    steps, bps = cfg["schedule"]["total_steps"], cfg["batch"]["blocks_per_step"]
    assert steps * bps == 3 * int(ids.shape[0])
    assert int(mask.sum()) * 3 == 8_881_521


@pytest.mark.skipif(
    not (REPO / "artifacts/stage3/ladder_uniform_probe/blocks.npz").is_file(),
    reason="canonical pack absent; the validator reads the budget from it")
def test_the_validator_passes_its_config_checks_and_fails_closed_without_the_init(
        tmp_path):
    """Both halves matter: the config gate passes now, the init gate does not.

    Guarded on the pack: `validate_e8_arms.py` re-derives the token budget from
    it, so without the pack the validator cannot run at all. E8's search pod does
    not stage the pack — it does not train — and this test failing there cost a
    draw before the guard existed.
    """
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
           "HOME": str(Path.home())}
    script = str(REPO / "scripts/training/validate_e8_arms.py")
    ok = subprocess.run([sys.executable, script, "--out",
                         str(tmp_path / "configs.json")],
                        capture_output=True, text=True, env=env, cwd=REPO)
    assert ok.returncode == 0, ok.stdout + ok.stderr[-2000:]
    report = json.loads((tmp_path / "configs.json").read_text())
    assert report["all_passed"] and report["init_checks_run"] is False
    assert report["budget"]["cumulative_ce_exposure"] == 8_881_521

    gated = subprocess.run([sys.executable, script, "--require-init", "--out",
                            str(tmp_path / "gated.json")],
                           capture_output=True, text=True, env=env, cwd=REPO)
    treatment_exists = (
        REPO / "artifacts/stage1/e8_contribution_init_v1/checkpoint"
        / "model.safetensors").is_file()
    if treatment_exists:
        pytest.skip("the treatment initialization exists; the negative case is "
                    "no longer reachable from this machine's state")
    assert gated.returncode == 6
    failed = json.loads((tmp_path / "gated.json").read_text())["failed"]
    assert "init_present:treatment" in failed
    assert "init_nll_gate:baseline" in failed


def test_the_positional_map_this_experiment_replaces_is_what_we_think_it_is():
    kept = [s["representative"] for s in depth_span_map(36, 28)]
    assert len(kept) == 28
    assert sorted(set(range(36)) - set(kept)) == [5, 7, 9, 11, 13, 15, 17, 19]
    # Recorded in the pinned init's own manifest, so the control's map is not
    # merely re-derived here but confirmed against the artifact.
    manifest = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/manifest.json"
    if manifest.is_file():
        diag = json.loads(manifest.read_text())["init_diagnostics"]
        assert [d["representative"] for d in diag["depth_map"]] == kept

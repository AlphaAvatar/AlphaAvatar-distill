"""The KD chunk fallback is regime-wide, or it is a confound.

`expandable_segments:True` alone passed a 20-step gate at 77.15 GiB and the DP-sa arm
then OOM'd at step ~110 of 1,761, with allocated memory having climbed to 77.37 GiB by
step 70. So the preregistered fallback — KD `chunk` 512 → 128 — was adopted.

The danger it introduces is not memory, it is comparability. Chunking changes the
float32 accumulation order (~7e-8 relative, `test_kd_chunk_invariance.py`), so an arm
at 128 and its partner at 512 would differ by something other than the depth map. These
tests pin that the change lands on **all four** depth-only arms and on **neither**
compressed arm, and that the objective itself is untouched.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ARMS = json.loads((REPO / "configs/stage3/e8b/arms.json").read_text())
DEPTH_ARMS = ("e8b_dp_r1600k_sa", "e8b_dc_r1600k_sa",
              "e8b_dp_r1600k_sb", "e8b_dc_r1600k_sb")
COMPRESSED_ARMS = ("e8b_fc_r1600k_sa", "e8b_fc_r1600k_sb")
CANONICAL_OBJECTIVE = {"ce_weight": 0.25, "kd_weight": 1.0,
                       "kd_temperature": 1.0, "kd_scope": "all"}


def cfg(name: str) -> dict:
    return json.loads((REPO / f"configs/stage3/e8b/{name}.json").read_text())


@pytest.mark.parametrize("name", DEPTH_ARMS)
def test_every_depth_only_arm_carries_chunk_128(name):
    assert cfg(name)["loss"].get("kd_chunk") == 128


@pytest.mark.parametrize("name", COMPRESSED_ARMS)
def test_no_compressed_arm_carries_it(name):
    # FC's regime needs 23-27 GB on an L40S and its retained FP control trained at the
    # default. Applying the fallback there would perturb FP-vs-FC for no reason.
    assert "kd_chunk" not in cfg(name)["loss"]


def test_all_four_depth_arms_agree_so_the_pairs_stay_single_variable():
    chunks = {n: cfg(n)["loss"]["kd_chunk"] for n in DEPTH_ARMS}
    assert len(set(chunks.values())) == 1, (
        f"depth arms disagree on kd_chunk: {chunks} — DC-DP would then confound the "
        "depth map with the reduction order")


@pytest.mark.parametrize("name", DEPTH_ARMS + COMPRESSED_ARMS)
def test_the_objective_itself_is_unchanged(name):
    loss = {k: v for k, v in cfg(name)["loss"].items() if k != "kd_chunk"}
    assert loss == CANONICAL_OBJECTIVE


def test_the_e1_control_was_not_touched():
    for control in ("e1_r1600k_sa_pca.json", "e1_r1600k_sb_pca.json"):
        loss = json.loads((REPO / "configs/stage3/e1" / control).read_text())["loss"]
        assert loss == CANONICAL_OBJECTIVE
        assert "kd_chunk" not in loss


def test_within_pair_identity_is_still_exactly_four_keys():
    # The chunk cancels inside a pair because both members carry it, so the
    # single-variable claim is unaffected.
    for pair in ARMS["within_cell_identity"]:
        assert set(pair["diff"]) == {"student_path", "run_name", "out_dir", "_purpose"}


def test_the_trainer_defaults_to_512_when_the_key_is_absent():
    """An absent key must mean the historical behaviour, not an error or a new value."""
    import inspect

    from aadistill.training.train import KD_CHUNK_DEFAULT, kd_forward_kl
    assert inspect.signature(kd_forward_kl).parameters["chunk"].default == 512
    # The two call sites read the config through one named default, which the
    # run manifest's execution record also reports. Named rather than repeated
    # so a manifest cannot claim a chunk the loss did not use; the value it
    # names is still 512, which is the historical behaviour this pins.
    assert KD_CHUNK_DEFAULT == 512
    src = (REPO / "src/aadistill/training/train.py").read_text()
    assert src.count('chunk=loss_cfg.get("kd_chunk", KD_CHUNK_DEFAULT)') == 1
    assert src.count('chunk=self.cfg["loss"].get("kd_chunk", KD_CHUNK_DEFAULT)') == 1
    # The execution record resolves it through the same expression, so a run
    # manifest cannot report a chunk the loss did not use.
    assert src.count('"kd_chunk": int(loss_cfg.get("kd_chunk", KD_CHUNK_DEFAULT))') == 1


def test_a_bad_chunk_is_rejected_by_config_validation():
    from aadistill.training.train import validate_train_config
    base = cfg("e8b_dp_r1600k_sa")
    for bad in (0, -1, 1.5, "128"):
        broken = json.loads(json.dumps(base))
        broken["loss"]["kd_chunk"] = bad
        with pytest.raises(ValueError, match="kd_chunk"):
            validate_train_config(broken)


def test_the_recorded_hashes_match_the_regenerated_configs():
    from aadistill.infrastructure.manifest import sha256_json
    for arm in ARMS["arms"]:
        got = sha256_json(json.loads((REPO / arm["path"]).read_text()))
        assert got == arm["config_sha256"], f"{arm['name']} hash drifted"

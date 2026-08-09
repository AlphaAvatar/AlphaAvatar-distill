"""A derived artifact must announce that it is derived.

E6b's machine-readable training event streams were destroyed. The console log
survived, and parsing it back gives a usable curve — but only the fields the
console printed, at the precision it printed them. The output must therefore be
impossible to mistake for the original, and must say per field what it lost.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/pod/reconstruct_training_events.py"
E6B = Path("/home/ecs-user/aad-artifacts/e6b")
COMMITTED = REPO / "logs/e6b_reconstructed_training_events.json"

RUN_LOG = """[17:25:51] $ /opt/train/bin/python scripts/training/train_stage3.py --config /workspace/aad/configs/stage3/e6b/{name}.json
device cuda; loading packed token ladder ...
eval step 0: {{'val_blocks': 16, 'val_ce': 10.919939, 'val_ppl': 55267.4502, 'val_kd': 10.603207}}
step 10/2916  loss 11.8970  ce 9.6091  kd 9.1515  lr 3.42e-06  4.25s
step 20/2916  loss 9.3378  ce 7.4468  kd 7.5641  lr 6.85e-06  3.87s
eval step 2916 (final): {{'val_blocks': 16, 'val_ce': 1.169355, 'val_ppl': 3.2199, 'val_kd': 1.044711}}
"""

STATUS = ("2026-08-08T17:25:51.541177+00:00 MARKER:ARMS_VALIDATED\n"
          "2026-08-08T20:50:30.255480+00:00 MARKER:TRAIN_DONE:P2-2.96M-sa\n")


def run_script(tmp_path, config_name="e6b_p2_r2960k_sa"):
    log = tmp_path / "run.log"
    log.write_text(RUN_LOG.format(name=config_name))
    status = tmp_path / "run.status"
    status.write_text(STATUS)
    out = tmp_path / "events.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-log", str(log), "--status",
         str(status), "--config",
         str(REPO / f"configs/stage3/e6b/{config_name}.json"),
         "--out", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text()), proc


def test_the_output_declares_its_provenance(tmp_path):
    doc, _ = run_script(tmp_path)
    assert doc["provenance"] == "reconstructed_from_driver_console"
    assert doc["original_event_stream_available"] is False
    assert doc["original_event_stream_paths"] == [
        "artifacts/stage3/e6b_p2_r2960k_sa/train_log.jsonl"]
    assert "not a substitute" in doc["original_loss_note"]


def test_it_never_calls_itself_the_original(tmp_path):
    doc, proc = run_script(tmp_path)
    blob = json.dumps(doc) + proc.stdout
    assert "derived artifact, not the original event stream" in proc.stdout
    lowered = blob.lower()
    assert "recovered event stream" not in lowered
    assert "restored train_log" not in lowered


def test_every_original_field_is_classified(tmp_path):
    """A reader must be able to tell exact from truncated from gone."""
    doc, _ = run_script(tmp_path)
    fp = doc["field_provenance"]
    assert set(fp) == {"exact", "truncated", "derived_from_config",
                       "bounded_only", "unrecoverable"}
    # The fields the trainer emits per step, from
    # aadistill.training.train.Trainer.step_once.
    emitted = {"step", "loss", "ce", "kd", "lr", "grad_norm", "ce_targets",
               "kd_positions", "logical_block_tokens", "executed_positions",
               "executed_nonpad_tokens", "supervised_tokens",
               "truncate_padding", "seconds", "tokens_seen", "gpu_mem_gb"}
    classified = set().union(*(set(v) for v in fp.values()))
    assert emitted <= classified, (
        f"unclassified fields: {sorted(emitted - classified)} — a field that "
        "is neither promised nor disclaimed is the worst of both")
    assert "grad_norm" in fp["unrecoverable"]
    assert "loss" in fp["truncated"] and "6 decimals" in fp["truncated"]["loss"]


def test_events_are_extracted_with_derived_token_counts(tmp_path):
    doc, _ = run_script(tmp_path)
    arm = doc["arms"][0]
    steps = [e for e in arm["events"] if e["event"] == "train_step"]
    assert [e["step"] for e in steps] == [10, 20]
    assert steps[0] == {"event": "train_step", "step": 10, "loss": 11.8970,
                        "lr": 3.42e-06, "seconds": 4.25, "ce": 9.6091,
                        "kd": 9.1515, "tokens_seen": 10 * 2 * 8192}
    evals = [e for e in arm["events"] if e["event"] == "eval_result"]
    assert [e["val_set"] for e in evals] == ["val", "final"]
    assert arm["final_eval"]["val_ce"] == 1.169355


def test_timestamps_are_bounded_not_invented(tmp_path):
    doc, _ = run_script(tmp_path)
    arm = doc["arms"][0]
    assert all("time" not in e for e in arm["events"]), (
        "no per-event timestamp was printed; inventing one would make the "
        "derived artifact look more complete than it is")
    bounds = arm["time_bounds"]
    assert bounds["driver_command_clock_utc"] == "17:25:51"
    assert bounds["train_done_marker"] == "TRAIN_DONE:P2-2.96M-sa"
    assert bounds["marker_association"] == "positional", (
        "the driver's arm labels and the config run names are different "
        "vocabularies; the pairing must say how it was made")


def test_the_two_step_time_measures_are_kept_separate(tmp_path):
    doc, _ = run_script(tmp_path)
    arm = doc["arms"][0]
    assert arm["step_seconds"]["mean"] == pytest.approx(4.06, abs=0.01)
    assert "including_eval_and_checkpoint" in json.dumps(arm["wall_clock"])
    assert arm["wall_clock"]["seconds_per_step_including_eval_and_checkpoint"] \
        != arm["step_seconds"]["mean"]


def test_the_source_log_is_hashed(tmp_path):
    doc, _ = run_script(tmp_path)
    assert len(doc["source"]["run_log_sha256"]) == 64
    assert doc["source"]["status_sha256"]


@pytest.mark.skipif(not COMMITTED.is_file(), reason="artifact not generated")
def test_the_committed_e6b_reconstruction_matches_the_run(tmp_path):
    """Guards the file in `logs/` against silent edits."""
    doc = json.loads(COMMITTED.read_text())
    assert doc["provenance"] == "reconstructed_from_driver_console"
    assert doc["original_event_stream_available"] is False
    assert [a["run_name"] for a in doc["arms"]] == [
        "e6b_p2_r2960k_sa", "e6b_p2_r2960k_sb"]
    for arm in doc["arms"]:
        # 2916 steps at log_every 10.
        assert arm["counts"]["train_step"] == 291
        assert arm["counts"]["eval_result"] == 10
        assert 4.0 < arm["step_seconds"]["mean"] < 4.3
        assert 4.1 < arm["wall_clock"][
            "seconds_per_step_including_eval_and_checkpoint"] < 4.3
    finals = [a["final_eval"]["val_ce"] for a in doc["arms"]]
    assert all(1.16 < ce < 1.18 for ce in finals), (
        f"the reported P2 final val CE is ~1.17; got {finals}")


@pytest.mark.skipif(not (E6B / "e6b_run.log").is_file(),
                    reason="E6b console log is not on this box")
def test_it_reproduces_from_the_surviving_console_log(tmp_path):
    out = tmp_path / "events.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-log", str(E6B / "e6b_run.log"),
         "--status", str(E6B / "e6b.status"),
         "--config", str(REPO / "configs/stage3/e6b/e6b_p2_r2960k_sa.json"),
         "--config", str(REPO / "configs/stage3/e6b/e6b_p2_r2960k_sb.json"),
         "--out", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert proc.returncode == 0, proc.stderr
    fresh = json.loads(out.read_text())
    committed = json.loads(COMMITTED.read_text())
    assert fresh["source"]["run_log_sha256"] == \
        committed["source"]["run_log_sha256"]
    assert [a["events"] for a in fresh["arms"]] == \
        [a["events"] for a in committed["arms"]], (
            "the committed artifact must be reproducible from the log it "
            "claims to derive from")

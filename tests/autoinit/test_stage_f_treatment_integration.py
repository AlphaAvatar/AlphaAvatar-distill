"""Stage F end to end, through the code the pod actually runs.

Attempt 9 did not fail in a unit. It failed in stage F, inside
`materialize_fixed_path_suffix` -> `_run_steps` -> `impl.execute` ->
`attention_activation.apply` -> `head_write_energy`. Every `$0` regression that
covered the treatment operator called `impl.execute` directly, and every one that
covered the suffix entry point ran on a single device — so the composition of the
two, which is what stage F is, had no failing case to find.

This module is that composition:

* the real four-step frozen path and the real `materialize_fixed_path_suffix`;
* the real registered `attention.activation_importance_v1`, never a stand-in —
  `test_D4` substitutes one and requires the integration to go red, so a future
  agent cannot quietly satisfy this file with a fake;
* the claim that steps 0/1/2 do not execute proven from the checkpoints on disk
  rather than from patched `execute` methods, because a monkeypatched recorder
  proves what the patch saw, not what the path wrote.

The device topology is the one thing that cannot be real on this box, so it is
modelled with `device_split` — the instrument the repository already has. The
operator, the materializer, the record writer and the checkpoint layout are all
production code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.autoinit.fixed_path import (  # noqa: E402
    FixedPathError, write_suffix_execution_record,
)
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.operators.base import get_implementation  # noqa: E402

from device_split import CrossDeviceUse, on_cache_device  # noqa: E402

#: The real two-arm world, built by the module that owns it. Importing the
#: fixtures rather than rebuilding them means stage F here rests on the same
#: genuinely-gated parent that the suffix contract tests use.
from test_fixed_path_verified_suffix import (  # noqa: E402,F401
    ATTENTION, START, _run_suffix, _treatment_registered, arms,
)


def _split_the_snapshot(monkeypatch):
    """Put the statistics snapshot on a different logical device than the model.

    This labels what `AttentionHeadStatsCollector.state()` returns — the real
    host-resident snapshot, at the real boundary. It does not replace the
    collector, the operator or any arithmetic.
    """
    real_state = attention_activation.AttentionHeadStatsCollector.state

    def labelled_state(self):
        return on_cache_device(real_state(self))

    monkeypatch.setattr(attention_activation.AttentionHeadStatsCollector,
                        "state", labelled_state)


# --- D. the production path, unpatched --------------------------------------

def test_D_stage_f_runs_the_real_treatment_operator_through_the_real_suffix(
        arms, tmp_path):
    """No monkeypatching anywhere. What stage F does, and what it leaves."""
    work = tmp_path / "stage_f"
    results, evidence = _run_suffix(arms, work)

    assert len(results) == 1
    r = results[0]
    assert r.index == START == 3
    assert r.impl_id == ATTENTION == "attention.activation_importance_v1"
    assert r.kind == "ATTENTION"
    assert Path(r.checkpoint_path).name == "03_attention"

    # The treatment really ran: its own trace and its own selection evidence.
    assert r.trace["score"] == "mean_t ||W_o,h a_h(t)||^2"
    assert r.trace["calibration_tokens"] > 0
    assert "op.attention.retained_write_energy_mean" in r.local_metrics

    assert evidence["executed_step_indices"] == [3]
    assert evidence["prefix_step_indices_not_executed"] == [0, 1, 2]
    assert evidence["actual_path_hash"] == arms.treatment.spec_hash


def test_D2_the_prefix_left_no_checkpoints_because_it_never_ran(arms, tmp_path):
    """Proven from the filesystem, not from a patched `execute`.

    A recorder that patches every operator proves the patch was not called. This
    asks the stronger question: did anything write a step-0/1/2 checkpoint into
    stage F's workdir? DEPTH is not even applicable to this parent, so if the
    prefix had run at all there would be an exception rather than a directory —
    but the artifact check is the one that stays true if that ever changes.
    """
    work = tmp_path / "stage_f"
    _run_suffix(arms, work)

    steps = sorted(p.name for p in (work / "steps").iterdir() if p.is_dir())
    assert steps == ["03_attention"], f"stage F wrote {steps}"
    assert (work / "steps" / "03_attention" / "config.json").is_file()


def test_D3_the_checkpoint_stage_g_would_load_is_the_treatment_child(arms,
                                                                     tmp_path):
    """The output has to be loadable and have the treatment's geometry."""
    from test_calibration_item_preparation import TARGET

    results, _ = _run_suffix(arms, tmp_path / "stage_f")
    child = QWEN3_ADAPTER.load(results[0].checkpoint_path, device="cpu")
    spec = QWEN3_ADAPTER.spec_of(child)
    assert spec["num_attention_heads"] == TARGET["num_attention_heads"]
    assert spec["num_key_value_heads"] == TARGET["num_key_value_heads"]
    assert spec["num_hidden_layers"] == TARGET["num_hidden_layers"]


# --- D. the device boundary, inside stage F ---------------------------------

def test_D_the_repaired_operator_survives_stage_f_across_a_device_split(
        arms, tmp_path, monkeypatch):
    """The case attempt 9 would have failed, run through stage F's own code."""
    _split_the_snapshot(monkeypatch)
    results, evidence = _run_suffix(arms, tmp_path / "split")
    assert results[0].index == 3
    assert Path(results[0].checkpoint_path).name == "03_attention"
    assert evidence["executed_step_indices"] == [3]


def test_D_mutation_the_old_operator_reproduces_attempt_9_inside_stage_f(
        arms, tmp_path, monkeypatch):
    """Remove the transfer and stage F dies where it died on the L40S.

    The failure surfaces from `materialize_fixed_path_suffix`, in the same call
    chain the driver logged — not from a directly-invoked operator.
    """
    _split_the_snapshot(monkeypatch)
    monkeypatch.setattr(attention_activation, "stats_to",
                        lambda state, device: state)          # the old behaviour

    with pytest.raises(CrossDeviceUse, match="persistent cache device"):
        _run_suffix(arms, tmp_path / "old")

    assert not (tmp_path / "old" / "steps" / "03_attention").exists(), (
        "a failed stage F must not leave a treatment checkpoint behind")


def test_D4_mutation_a_stand_in_stage_f_operator_makes_this_module_red(
        arms, tmp_path, monkeypatch):
    """The guard on this file: substitute the operator and the path refuses.

    If the integration above could be satisfied by a fake, it would be proving
    nothing about `attention.activation_importance_v1`.
    """
    impl = get_implementation(ATTENTION)

    class _PassThrough:
        """Returns the parent unchanged — a plausible-looking stub."""

        def execute(self, ctx):
            from aadistill.autoinit.operators.base import OperatorOutcome
            return OperatorOutcome(model=ctx.model, local_metrics=None,
                                   trace={}, artifacts={})

    monkeypatch.setattr(impl, "execute", _PassThrough().execute)
    with pytest.raises(FixedPathError, match="is not the planned"):
        _run_suffix(arms, tmp_path / "fake")


# --- D. the treatment record, written and read back -------------------------

def test_D5_the_treatment_record_round_trips_with_the_repair_recorded(
        arms, tmp_path, monkeypatch):
    """What stage F would hand the closeout, produced across the device split."""
    _split_the_snapshot(monkeypatch)
    results, evidence = _run_suffix(arms, tmp_path / "rec_run")

    path = write_suffix_execution_record(
        arms.treatment, results, tmp_path / "treatment_record.json",
        runtime={"torch": torch.__version__}, suffix_evidence=evidence,
        calibration={"profile_id": arms.pid})
    rec = json.loads(path.read_text())

    assert rec["schema"] == "aadistill.autoinit.fixed_path_suffix_execution/v1"
    assert rec["is_replay"] is False
    assert rec["n_pinned"] == 0
    assert rec["output_digest_was_pre_pinned"] is False
    assert rec["path_hash"] == arms.treatment.spec_hash
    assert rec["executed_step_indices"] == [3]
    assert rec["output"]["index"] == 3
    assert rec["output"]["impl_id"] == ATTENTION
    assert rec["output"]["artifact_digest"] == results[0].identity.artifact_digest
    assert rec["verified_prefix"]["expected_parent_artifact_digest"] == \
        arms.parent_digest

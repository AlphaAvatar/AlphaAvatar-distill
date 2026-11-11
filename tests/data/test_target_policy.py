"""`aadistill.data.target_policy.drop_template_prefill`, and the trainer applying it."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from aadistill.data.target_policy import drop_template_prefill

GP, PF = [7, 8, 9], [3, 4]


def _row(*parts):
    ids, mask = [], []
    for toks, sup in parts:
        ids += toks
        mask += [sup] * len(toks)
    return torch.tensor([ids]), torch.tensor([mask])


def test_only_the_prefill_after_the_generation_prompt_stops_being_a_target():
    ids, mask = _row(([1, 2] + GP, False), (PF + [11, 12, 13], True), ([5, 6] + GP, False),
                     (PF + [21], True), ([30], False), ([31, 32], True))
    new, st = drop_template_prefill(ids, mask, generation_prompt=GP, prefill=PF)
    expect = mask.clone()
    expect[0, 5:7] = False
    expect[0, 15:17] = False
    assert torch.equal(new, expect)
    assert st["runs"] == 3 and st["matched_runs"] == 2 and st["removed_positions"] == 4
    assert st["unmatched_runs_after_generation_prompt"] == 0
    assert torch.equal(ids, _row(([1, 2] + GP, False), (PF + [11, 12, 13], True), ([5, 6] + GP, False),
                                 (PF + [21], True), ([30], False), ([31, 32], True))[0])


def test_a_run_after_the_prompt_without_the_prefill_is_counted_not_changed():
    ids, mask = _row((GP, False), ([40, 41], True))
    new, st = drop_template_prefill(ids, mask, generation_prompt=GP, prefill=PF)
    assert torch.equal(new, mask) and st["unmatched_runs_after_generation_prompt"] == 1


def test_the_trainer_refuses_a_template_the_policy_does_not_describe(tmp_path):
    from test_train import tiny_model, toy_cfg
    from aadistill.training.train import Trainer

    ids, mask = _row(([1] + GP, False), (PF + [11, 12, 13, 14], True), (GP, False), ([40] * 5, True),
                     ([2] * 13, False))
    ids, mask = ids.repeat(2, 1), mask.repeat(2, 1)
    cfg = toy_cfg(tmp_path, loss={"exclude_template_prefill": {"generation_prompt": GP, "prefill": PF}})
    with pytest.raises(ValueError, match="without the declared prefill"):
        Trainer(cfg, tiny_model(1), (ids, mask), device="cpu")
    ok_ids, ok_mask = _row(([1] + GP, False), (PF + [11, 12, 13, 14], True), ([2] * 23, False))
    tr = Trainer(cfg, tiny_model(1), (ok_ids.repeat(2, 1), ok_mask.repeat(2, 1)), device="cpu")
    assert tr.target_policy["train"]["removed_positions"] == 4
    assert int(tr.train_mask.sum()) == 2 * 4 and tr.execution_record()["target_policy"] == tr.target_policy

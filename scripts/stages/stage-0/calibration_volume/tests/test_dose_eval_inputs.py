"""The prompt rows the dose driver builds must carry what uncapped_eval's in-run
scorer reads. d1 attempt 2 (dose_d1_20261019_055515) generated every capability
unit and then died on `KeyError: 'prompt_text'` scoring the legacy 150, which
the driver builds itself rather than reading from the set.
"""
from __future__ import annotations

import json

from aadistill.evaluation.behavior import aggregate, score_sample
from shared import eval_sets as ES
from stages.calibration_volume.dose_driver import SET_ID, legacy_prompts


def test_every_legacy_row_scores_in_run(tmp_path):
    rows = [json.loads(l) for l in legacy_prompts(tmp_path / "legacy.jsonl").read_text().splitlines()]
    items = {it["id"]: it for it in ES.items(SET_ID, "stage3_legacy_150", "items")}
    assert len(rows) == len(items) == 150
    for r in rows:
        assert r["prompt_text"] == ES.prompt_text(items[r["id"]]) != ""
    aggregate([score_sample(r, "<think>\nx\n</think>\n\n42<|im_end|>") for r in rows])

"""A decoding protocol is pure nucleus sampling, and its seeds depend on the request alone.

`scripts/shared/evaluation/decoding.py` is what both evaluation runners read a
protocol through (maintainer, 2026-10-24: no greedy decoding in new
evaluation). These tests pin the two properties a comparison across
checkpoints rests on: a protocol that is not pure top-p is refused, and a
request's seed is a function of the protocol and the request -- never of the
model -- so every checkpoint draws with the same seed for the same item.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from shared.evaluation import decoding as DEC

REPO = Path(__file__).resolve().parents[2]
NUCLEUS = REPO / "configs/shared/evaluation/decoding/nucleus_v1.json"


def test_the_registered_protocol_loads_as_pure_nucleus_sampling():
    p = DEC.load(NUCLEUS)
    s = DEC.sampling(p)
    assert s["temperature"] > 0 and 0 < s["top_p"] < 1 and s["top_k"] == -1 and s["min_p"] == 0
    assert DEC.identity(p)["sha256"] == p["sha256"] and p["samples_per_item"] == 1


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.0), ("top_p", 1.0), ("top_k", 20), ("min_p", 0.05),
    ("repetition_penalty", 1.1), ("presence_penalty", 0.5)])
def test_a_protocol_that_is_not_pure_top_p_is_refused(tmp_path, field, value):
    doc = json.loads(NUCLEUS.read_text())
    doc["sampling"][field] = value
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        DEC.load(bad)


def test_seeds_depend_on_the_request_and_never_on_the_model():
    p = DEC.load(NUCLEUS)
    assert "model" not in inspect.signature(DEC.request_seed).parameters
    seeds = {DEC.request_seed(p, ns, item, turn, sample)
             for ns in ("fluid_reasoning_720__dev_v1", "stage3_capability_846__gsm8k")
             for item in ("a", "b") for turn in (0, 1) for sample in (0, 1)}
    assert len(seeds) == 16 and all(0 <= s < 2 ** 63 for s in seeds)
    assert DEC.request_seed(p, "u", "a", 2) == DEC.request_seed(DEC.load(NUCLEUS), "u", "a", 2)

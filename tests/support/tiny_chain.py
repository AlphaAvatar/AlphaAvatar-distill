"""The incumbent operator chain and a REAL materialized tiny mixture.

Shared because the same four operator ids and the same on-disk mixture are what
an initialization test, a Phase-C3 batching test and a verified-suffix test all
build against. They lived in `test_calibration_item_preparation.py`, so every
other suite imported a test module to get them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from aadistill.initialization.calibration.profiles import (
    CalibrationProfile,
    CalibrationSource,
    mixture_content_sha256,
)
from aadistill.initialization.calibration.datasets import DatasetRole

from support.toy import TEACHER_GEOMETRY

#: The frozen current-best implementation per structural kind.
DEPTH = "depth.causal_kl_greedy_v1"
FFN = "ffn.activation_importance_v0"
WIDTH = "width.global_pca_v0"
ATTENTION = "attention.activation_importance_v1"

TINY_PROFILE_ID = "test.raw_materialized@v1"

#: A target the tiny teacher can actually reach: shallower, narrower, fewer heads.
TARGET = dict(TEACHER_GEOMETRY, num_hidden_layers=4, intermediate_size=24,
              hidden_size=16, num_attention_heads=2)


def raw_item(item_id: str, domain: str, subtype: str, ids: list[int]) -> dict:
    """Exactly the shape a frozen materialized mixture stores on disk.

    `ids`, not `input_ids` — that is what `mixture_content_sha256` hashes, and
    therefore what the pinned content identity is defined over.
    """
    return {"item_id": item_id, "ids": list(ids), "domain": domain,
            "subtype": subtype, "n_prediction_positions": len(ids) - 1}


def write_tiny_mixture(root: Path, *, seq_len: int = 24, vocab: int = 128,
                       n_per_domain: int = 2, seed: int = 4242):
    """A REAL materialized profile: JSONL on disk, both hashes derived from it.

    Built through the production rules rather than around them — the file hash
    and `mixture_content_sha256` are computed from the bytes actually written, so
    `CalibrationProfile.resolve()` runs its full fail-closed check here exactly as
    it does against the frozen mixtures.
    """
    g = torch.Generator().manual_seed(seed)
    items = []
    for domain, subtype in (("general", "text"), ("math", "arith")):
        for k in range(n_per_domain):
            ids = torch.randint(0, vocab, (seq_len,), generator=g).tolist()
            items.append(raw_item(f"{subtype}-{k}", domain, subtype, ids))

    rel = "artifacts/tiny_calibration/items.jsonl"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i) + "\n" for i in items))

    profile = CalibrationProfile(
        profile_id="test.raw_materialized", version=1,
        description="a real materialized mixture that stores tokens under 'ids'",
        sources=(CalibrationSource("test/raw", "local", "general", n_per_domain),
                 CalibrationSource("test/raw", "local", "math", n_per_domain)),
        domain_weights={"general": 0.5, "math": 0.5},
        token_budget=seq_len * len(items), sample_rule="fixed", seed=seed,
        role=DatasetRole.OPERATOR_CALIBRATION,
        materialized=True, items_path=rel,
        content_sha256=mixture_content_sha256(items),
        items_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return profile, items

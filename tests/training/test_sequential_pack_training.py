"""Stage-3 training on a sequential causal pack (`token_documents.pack_sequential`).

The trainer needs no new attention path: a sequential pack carries no position
ids, so both models read each packed sequence with standard causal attention and
continuous positions (``batch.context_strategy: cross_context``, the default).
What these tests pin is the contract around it: CE and KD land exactly on the
pack's supervised positions (fragments included), teacher and student read the
same packed context, and a pack without position ids cannot be declared
document-isolated.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.data.token_documents import TokenDocument, pack_sequential  # noqa: E402
from aadistill.training.train import Trainer  # noqa: E402
from test_train import VOCAB, tiny_model, toy_cfg  # noqa: E402


def _pack(seed: int = 0):
    rng = random.Random(seed)
    docs = []
    for i in range(14):
        prompt = tuple(rng.randint(10, VOCAB - 1) for _ in range(rng.randint(3, 8)))
        output = tuple(rng.randint(10, VOCAB - 1) for _ in range(rng.randint(2, 12))) + (5,)
        docs.append(TokenDocument(f"doc-{i:02d}", prompt, output))
    return pack_sequential(docs, 32, pad_id=0, order_seed=7)


def _cfg(tmp_path, **batch):
    return toy_cfg(tmp_path, loss={"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0,
                                   "kd_scope": "assistant",
                                   "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": 8}},
                   batch={"blocks_per_step": 2, "micro_blocks": 1, **batch})


def test_ce_and_kd_land_on_the_packs_supervised_positions(tmp_path):
    packed = _pack()
    assert packed.stats()["fragments"] > 0, "the toy pack must contain a fragment"
    tr = Trainer(_cfg(tmp_path), tiny_model(1), packed.blocks(), teacher=tiny_model(2), device="cpu")
    ids, mask, content = packed.blocks()
    with torch.no_grad():
        ce, ce_n, kd, kd_n = tr._micro_losses(ids[:2], mask[:2], content[:2])
    want = int(mask[:2, 1:].sum())
    assert ce_n == kd_n == want > 0
    assert torch.isfinite(ce) and torch.isfinite(kd)
    # no document's first (prompt) token is ever a target, so the position
    # after a document never supervises the next one
    for b, members in enumerate(packed.members[:2]):
        for m in members:
            assert not bool(mask[b, m["start"]])


def test_a_step_trains_on_the_packed_context(tmp_path):
    packed = _pack(1)
    tr = Trainer(_cfg(tmp_path), tiny_model(1), packed.blocks(), teacher=tiny_model(2), device="cpu")
    out = tr.step_once()
    assert out["context_strategy"] == "cross_context"
    assert out["ce_targets"] == out["kd_positions"] > 0


def test_a_pack_without_position_ids_cannot_be_declared_isolated(tmp_path):
    packed = _pack()
    with pytest.raises(ValueError, match="document_isolated"):
        Trainer(_cfg(tmp_path, context_strategy="document_isolated"), tiny_model(1),
                packed.blocks(), teacher=tiny_model(2), device="cpu")

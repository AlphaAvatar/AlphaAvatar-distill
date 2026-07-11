"""Teacher model loading with logged identity.

Fails loudly if the requested model or revision is unavailable (AGENTS.md 2.3).
"""

from __future__ import annotations

import hashlib

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_revision(model_id: str, revision: str | None) -> str:
    """Pin a branch/tag revision to its exact commit hash."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=revision)
    return info.sha


def tokenizer_hash(tokenizer) -> str:
    """Stable hash of the tokenizer's full vocabulary."""
    vocab = tokenizer.get_vocab()
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    h = hashlib.sha256()
    for token, idx in items:
        h.update(f"{idx}:{token}\n".encode())
    return h.hexdigest()


def load_teacher(
    model_id: str,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
):
    """Load teacher model + tokenizer and return them with an identity record."""
    pinned = resolve_revision(model_id, revision)
    config = AutoConfig.from_pretrained(model_id, revision=pinned)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=pinned)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=pinned, dtype=DTYPES[dtype]
    ).to(device)
    model.eval()
    identity = {
        "model_id": model_id,
        "revision": pinned,
        "dtype": dtype,
        "device": device,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "architecture": config.architectures,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "tokenizer_sha256": tokenizer_hash(tokenizer),
    }
    return model, tokenizer, identity

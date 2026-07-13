"""Student model construction for Stage 1.

The student is a standard transformers Qwen3 dense model whose geometry comes
from the recipe config and whose positional/normalization/tokenizer settings
are inherited from the teacher config, so the initialized checkpoint stays
loadable by any Qwen3-compatible runtime with no custom modeling code.
"""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

# Inherited from the teacher so student and teacher share tokenizer identity,
# RoPE basis, norm epsilon, and special-token behavior.
INHERITED_KEYS = [
    "vocab_size", "rope_theta", "rope_scaling", "max_position_embeddings",
    "rms_norm_eps", "hidden_act", "attention_bias", "attention_dropout",
    "bos_token_id", "eos_token_id",
]


def build_student_config(teacher_config, geometry: dict) -> Qwen3Config:
    required = ["hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "tie_word_embeddings"]
    missing = [k for k in required if k not in geometry]
    if missing:
        raise ValueError(f"Student geometry missing keys: {missing}")
    inherited = {k: getattr(teacher_config, k) for k in INHERITED_KEYS
                 if hasattr(teacher_config, k)}
    return Qwen3Config(**inherited, **geometry)


def build_student(config: Qwen3Config, dtype: torch.dtype, seed: int) -> Qwen3ForCausalLM:
    """Fresh student with the standard (random) init; deterministic via seed.

    The random state is both the Stage 1 baseline comparator and the tensor
    container that ``init_student`` overwrites.
    """
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config).to(dtype)
    model.eval()
    return model

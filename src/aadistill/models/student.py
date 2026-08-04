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


def assert_rope_matches_config(model, config, path: str = "") -> float:
    """Fail if the loaded model's RoPE base is not the one the config stores.

    transformers 5.x writes `rope_theta` inside a nested `rope_parameters` dict.
    A 4.x reader loading such a config silently falls back to the class default
    (10000) — `config.rope_theta` reads 10000 while `config.rope_parameters`
    still says 5000000, and the model is built with the wrong basis. Nothing
    warns, and the run looks normal.

    So the check is made against the *runtime* frequencies rather than any config
    attribute: `inv_freq[1] == base ** (-2/head_dim)` inverts to the base the
    model will actually use. Returns it, and raises when it disagrees with what
    the checkpoint recorded.
    """
    stored = None
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        stored = float(params["rope_theta"])
    elif getattr(config, "rope_theta", None) is not None:
        stored = float(config.rope_theta)
    if stored is None:
        return float("nan")

    inv = model.model.rotary_emb.inv_freq
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    runtime = float(inv[1]) ** (-head_dim / 2.0)
    if abs(runtime - stored) / stored > 1e-3:
        raise ValueError(
            f"RoPE base mismatch for {path or 'model'}: the checkpoint records "
            f"{stored:,.0f} but the model was built with {runtime:,.0f}. This is "
            "the transformers 4.x/5.x `rope_parameters` skew — the installed "
            "transformers cannot read this checkpoint's config correctly. "
            "Install the locked version rather than training on a 500x-wrong "
            "positional basis.")
    return runtime

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


def stored_rope_base(config) -> float | None:
    """The RoPE base a config *records*, from either library's field layout.

    Read the nested transformers-5 field first: when both are present the flat
    `rope_theta` is the 4.x reader's class default, i.e. the wrong one.
    """
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        return float(params["rope_theta"])
    if getattr(config, "rope_theta", None) is not None:
        return float(config.rope_theta)
    return None


def assert_rope_matches_config(model, config, path: str = "") -> float:
    """Fail if the loaded model's RoPE base is not the one the config stores.

    transformers 5.x writes `rope_theta` inside a nested `rope_parameters` dict.
    A 4.x reader loading such a config silently falls back to the class default
    (10000) — `config.rope_theta` reads 10000 while `config.rope_parameters`
    still says 5000000, and the model is built with the wrong basis. Nothing
    warns, and the run looks normal.

    So the check is made against the *runtime* frequencies rather than any config
    attribute: `inv_freq[i] == base ** (-2i/head_dim)` inverts to the base the
    model will actually use. Returns it, and raises when it disagrees with what
    the checkpoint recorded.

    **Which entry to invert matters.** Inverting `inv_freq[1]` needs the exponent
    `-head_dim/2` — 64 here — which amplifies the buffer's relative error 64x. On
    an fp32 buffer that is harmless (recovers 4,999,983 of 5,000,000), but
    `build_student` casts the whole module to bf16, and a bf16 `inv_freq[1]`
    recovers 5,282,142 — a 5.6% error that trips a 0.1% tolerance while nothing
    is actually wrong. The **last** entry needs the exponent
    `-head_dim/(head_dim-2)` ≈ 1.016, so error is barely amplified at all: the
    same bf16 buffer recovers 4,986,576, and the 500x skew this function exists to
    catch is still four orders of magnitude outside the tolerance.
    """
    stored = stored_rope_base(config)
    if stored is None:
        return float("nan")

    inv = model.model.rotary_emb.inv_freq.detach().double()
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    n = inv.shape[0]
    if n < 2:
        return float("nan")
    # Highest index available, i.e. the smallest exponent to invert.
    index = n - 1
    runtime = float(inv[index]) ** (-head_dim / (2.0 * index))
    if abs(runtime - stored) / stored > 1e-2:
        raise ValueError(
            f"RoPE base mismatch for {path or 'model'}: the checkpoint records "
            f"{stored:,.0f} but the model was built with {runtime:,.0f}. This is "
            "the transformers 4.x/5.x `rope_parameters` skew — the installed "
            "transformers cannot read this checkpoint's config correctly. "
            "Install the locked version rather than training on a 500x-wrong "
            "positional basis.")
    return runtime

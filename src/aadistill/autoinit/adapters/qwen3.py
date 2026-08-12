"""Qwen3 dense architecture adapter.

The only module in the AutoInitializer that knows a Qwen3 block keeps attention
under ``.self_attn``, its FFN under ``.mlp``, and its pre-attention norm under
``.input_layernorm``. Everything above this file addresses those through role
names (``"q"``, ``"gate"``, ``"attn_out"``, ...) so an MLA or MoE family can
answer the same questions with a different module tree.

``param_count`` is exact arithmetic rather than a materialized ``numel`` sum: the
cost model has to price a ~4.xB intermediate state without building it, and the
search has to validate a leaf's parameter count before deciding to keep it. It is
pinned by test against the two counts this project has frozen — the teacher's
4,022,468,096 and the target's 596,049,920.
"""

from __future__ import annotations

from typing import Any

import torch

from ...init.collect import ActivationStatsCollector
from ..arch import ArchitectureAdapter, ArchSpec, Capability, register_adapter

#: Config keys that describe *structure*. Anything outside this list (rope base,
#: norm epsilon, tokenizer ids, dtype) is inherited and must survive every
#: operator untouched; `build_config` copies it rather than restating it.
QWEN3_STRUCTURAL_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
)


class Qwen3Adapter(ArchitectureAdapter):
    family = "qwen3"
    adapter_version = "qwen3.dense_v1"
    capabilities = frozenset({
        Capability.RESIDUAL_STREAM,
        Capability.PRENORM_BLOCKS,
        Capability.BLOCK_LIST,
        Capability.DENSE_FFN,
        Capability.ATTENTION_GQA,
        Capability.TIED_EMBEDDINGS,
        Capability.RMS_NORM,
        Capability.ACTIVATION_STATS,
        Capability.LOGIT_COMPARABLE,
    })
    structural_fields = QWEN3_STRUCTURAL_FIELDS

    # --- spec algebra ------------------------------------------------------

    def spec_from_config(self, config: Any) -> ArchSpec:
        missing = [k for k in self.structural_fields if getattr(config, k, None) is None]
        if missing:
            raise ValueError(f"qwen3 config is missing structural fields {missing}")
        return ArchSpec.of(self.family, {k: getattr(config, k) for k in self.structural_fields})

    def build_config(self, base_config: Any, spec: ArchSpec) -> Any:
        if spec.family != self.family:
            raise ValueError(f"cannot build a qwen3 config from a {spec.family} spec")
        raw = base_config.to_dict()

        # `layer_types` is *derived* from the layer count, and transformers
        # validates the two against each other. Carrying the teacher's 36-entry
        # list onto a 28-layer child raises; carrying it silently would be worse.
        # Dropping it lets the config regenerate a list of the right length —
        # correct only while every entry is identical, so anything else is
        # refused rather than guessed. A sliding-window family that removes
        # specific layers has to decide which pattern survives, and that decision
        # belongs to a depth operator for that family, not to a default here.
        layer_types = raw.pop("layer_types", None)
        if layer_types and len(set(layer_types)) > 1:
            raise ValueError(
                f"teacher config mixes attention layer types {sorted(set(layer_types))}; "
                "regenerating the pattern for a different layer count would change "
                "which layers are sliding-window. This adapter version only handles "
                "a uniform pattern.")

        # Apply the spec into the dict rather than by setattr afterwards: the
        # config regenerates derived fields in __init__, so a post-hoc setattr
        # leaves them describing the old geometry.
        raw.update(spec.as_dict())
        config = type(base_config).from_dict(raw)
        # transformers 5 keeps the rope base in a nested dict and 4.x in a flat
        # field; `to_dict`/`from_dict` roundtrips whichever the installed library
        # wrote, so the positional basis is inherited rather than restated here.
        # See STATE.md 0.5 for what happens when it is not.
        return config

    def param_count(self, spec: ArchSpec) -> int:
        d = spec["hidden_size"]
        layers = spec["num_hidden_layers"]
        inter = spec["intermediate_size"]
        n_q = spec["num_attention_heads"]
        n_kv = spec["num_key_value_heads"]
        head_dim = spec["head_dim"]
        vocab = spec["vocab_size"]
        tied = bool(spec["tie_word_embeddings"])

        attn = (
            n_q * head_dim * d          # q_proj
            + n_kv * head_dim * d       # k_proj
            + n_kv * head_dim * d       # v_proj
            + d * n_q * head_dim        # o_proj
            + 2 * head_dim              # q_norm, k_norm
        )
        mlp = 3 * inter * d
        norms = 2 * d                   # input_layernorm, post_attention_layernorm
        per_layer = attn + mlp + norms
        embed = vocab * d
        return embed * (1 if tied else 2) + layers * per_layer + d

    def validate_target(self, spec: ArchSpec) -> None:
        if spec.family != self.family:
            raise ValueError(f"{spec.family} spec is not a qwen3 target")
        for key in ("hidden_size", "num_hidden_layers", "intermediate_size",
                    "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"):
            value = spec[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"target {key}={value!r} must be a positive int")
        n_q, n_kv = spec["num_attention_heads"], spec["num_key_value_heads"]
        if n_q % n_kv:
            raise ValueError(
                f"target has {n_q} query heads and {n_kv} KV heads; GQA needs "
                "query heads divisible by KV heads")
        if not isinstance(spec["tie_word_embeddings"], bool):
            raise ValueError("target tie_word_embeddings must be a bool")

    # --- model lifecycle ---------------------------------------------------

    def build_model(self, config: Any, dtype: Any, seed: int) -> Any:
        from ...models.student import build_student

        return build_student(config, dtype, seed)

    def save(self, model: Any, path: str) -> None:
        model.save_pretrained(path)

    def load(self, path: str, dtype: Any = None, device: str = "cpu") -> Any:
        from transformers import Qwen3ForCausalLM

        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = Qwen3ForCausalLM.from_pretrained(path, **kwargs)
        return model.to(device).eval()

    def weight_file(self, path: str) -> str:
        return "model.safetensors"

    # --- structure accessors ----------------------------------------------

    def blocks(self, model: Any) -> list[Any]:
        return list(model.model.layers)

    def set_blocks(self, model: Any, blocks) -> None:
        kept = list(blocks)
        model.model.layers = torch.nn.ModuleList(kept)
        model.config.num_hidden_layers = len(kept)

    def attention(self, block: Any) -> Any:
        return block.self_attn

    def ffn(self, block: Any) -> Any:
        return block.mlp

    def stream_in_projections(self, block: Any) -> dict[str, tuple[Any, Any]]:
        attn_norm, ffn_norm = block.input_layernorm, block.post_attention_layernorm
        return {
            "q": (block.self_attn.q_proj, attn_norm),
            "k": (block.self_attn.k_proj, attn_norm),
            "v": (block.self_attn.v_proj, attn_norm),
            "gate": (block.mlp.gate_proj, ffn_norm),
            "up": (block.mlp.up_proj, ffn_norm),
        }

    def stream_out_projections(self, block: Any) -> dict[str, Any]:
        return {"attn_out": block.self_attn.o_proj, "ffn_out": block.mlp.down_proj}

    def attn_norm(self, block: Any) -> Any:
        return block.input_layernorm

    def ffn_norm(self, block: Any) -> Any:
        return block.post_attention_layernorm

    def embedding(self, model: Any) -> Any:
        return model.model.embed_tokens

    def final_norm(self, model: Any) -> Any:
        return model.model.norm

    def head_groups(self, spec: ArchSpec) -> tuple[int, int, int]:
        return spec["num_attention_heads"], spec["num_key_value_heads"], spec["head_dim"]

    def stats_collector(self, model: Any) -> ActivationStatsCollector:
        return ActivationStatsCollector(model)


QWEN3_ADAPTER = register_adapter(Qwen3Adapter())

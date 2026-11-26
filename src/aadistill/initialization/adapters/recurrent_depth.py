"""The recurrent-depth family: a decoder whose logical layers execute fewer parameter blocks, repeatedly.

This module is the family's structural definition, so it lives with the architecture adapters (AGENTS.md P3), beside
the projected-residual family. Nothing in it names a teacher, a geometry, a schedule or an experiment: every
structural fact is a config field, and a schedule is built from data the caller supplies.

**Structure.** A ``RecurrentDepthConfig`` is a Qwen3 config plus two fields:

``num_hidden_layers`` (L)
    the number of LOGICAL layers the forward executes, one after another. It is also the number of hidden states the
    forward produces and the number of KV-cache layers a generation keeps: each logical layer has its own cache
    entry, because each sees a different input even when it shares its parameters with another.
``layer_to_block`` (length L)
    logical layer ``p`` executes parameter block ``layer_to_block[p]`` on the output of logical layer ``p - 1``.
    Blocks are numbered ``0 .. num_parameter_blocks - 1``; every block is executed at least once.
``num_parameter_blocks`` (B)
    how many decoder blocks hold parameters. The checkpoint stores exactly these, under ``model.layers.0 ..
    model.layers.B-1`` -- the names of a B-layer Qwen3 -- so a B-layer Qwen3 checkpoint and a recurrent one over the
    same blocks have the same weight file and differ only in their config (which is part of the checkpoint's
    identity: the same weights executed on another schedule are another model).

What one logical layer computes is a whole Qwen3 decoder block -- input RMSNorm, attention with its q/k norms and
RoPE, residual add, post-attention RMSNorm, gated FFN, residual add -- with the block's shared parameters. Positions
do not advance between logical layers (the same token keeps its position id however many times a block runs), and
attention at logical layer ``p`` reads the keys and values logical layer ``p`` wrote for earlier tokens, never
another logical layer's, even one that shares its parameters.

**Explicit execution context, no mutated module state.** A block's attention module keeps the ``layer_idx`` it was
built with (the first logical layer that executes it) and is never re-pointed. Which cache entry a call reads and
writes is passed with the call: logical layer ``p`` receives ``LogicalLayerCache(cache, p)``, whose ``update`` writes
entry ``p`` whatever ``layer_idx`` the module passes. Gradient checkpointing re-runs a block's forward with the same
arguments, so a recomputation sees the same context.

**Shared parameters, one registration.** The B blocks are B modules; repetition is in the call schedule, never in
module aliasing, so ``named_parameters`` lists each parameter once, an optimizer holds one state per parameter, and
autograd sums the gradient of every call into it. ``save_pretrained`` writes each tensor once.

**Hidden-state supervision.** ``RecurrentDepthModel.forward`` accepts ``hidden_supervision`` (an object with
``targets`` and ``add(layer, hidden)``; `aadistill.training.hidden_state.HiddenSupervision`). After logical layer
``p`` runs, its OUTPUT (the residual stream after the block, before any final norm) is handed to ``add`` when ``p``
is one of the supervised layers. This happens in the logical-layer loop, outside the checkpointed block call, so a
recomputation during the backward never registers a loss a second time.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

MODEL_TYPE = "aad_recurrent_depth"
ARCHITECTURE = "RecurrentDepthForCausalLM"
#: fields this family adds to the Qwen3 config
EXTRA_FIELDS = ("layer_to_block", "num_parameter_blocks")

__all__ = [
    "ARCHITECTURE", "MODEL_TYPE", "RecurrentDepthConfig", "RecurrentDepthModel", "RecurrentDepthForCausalLM",
    "LogicalLayerCache", "register_auto_classes", "schedule_from_spans", "recurrent_config_from",
    "block_calls", "first_layer_of_block",
]


class RecurrentDepthError(ValueError):
    pass


def schedule_from_spans(spans: Sequence[Sequence[int]]) -> dict:
    """The execution schedule of a depth-reduced child whose block ``j`` stands for the span ``spans[j]`` of
    consecutive root blocks and is executed once per root block of its span, in order.

    Returns ``{"layer_to_block": [...], "root_layer_of_layer": [...]}``: logical layer ``p`` runs block
    ``layer_to_block[p]`` and stands where root block ``root_layer_of_layer[p]`` stood. The spans must tile ``0 ..
    n-1`` in order (`layer_merge.spans_from_kept_blocks` gives such spans); a span of one is a block run once."""
    flat = [int(b) for s in spans for b in s]
    if not spans or any(len(s) == 0 for s in spans) or flat != list(range(len(flat))):
        raise RecurrentDepthError(f"spans {list(map(list, spans))} do not tile 0 .. n-1 in order")
    return {"layer_to_block": [j for j, s in enumerate(spans) for _ in s], "root_layer_of_layer": flat}


def block_calls(layer_to_block: Sequence[int]) -> dict[int, list[int]]:
    """block -> the logical layers that execute it, in order."""
    out: dict[int, list[int]] = {}
    for p, b in enumerate(layer_to_block):
        out.setdefault(int(b), []).append(p)
    return out


def first_layer_of_block(layer_to_block: Sequence[int]) -> list[int]:
    """For each block, the first logical layer that executes it (the ``layer_idx`` its modules are built with)."""
    calls = block_calls(layer_to_block)
    return [calls[b][0] for b in range(len(calls))]


class RecurrentDepthConfig(Qwen3Config):
    """A Qwen3 config whose ``num_hidden_layers`` logical layers execute ``num_parameter_blocks`` blocks."""

    model_type = MODEL_TYPE

    def __init__(self, layer_to_block: Sequence[int] | None = None, num_parameter_blocks: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        n = int(self.num_hidden_layers)
        sched = [int(b) for b in (layer_to_block if layer_to_block is not None else range(n))]
        blocks = int(num_parameter_blocks) if num_parameter_blocks is not None else (max(sched) + 1 if sched else 0)
        if len(sched) != n:
            raise RecurrentDepthError(f"layer_to_block has {len(sched)} entries for {n} logical layers")
        if blocks < 1 or sorted(set(sched)) != list(range(blocks)):
            raise RecurrentDepthError(f"layer_to_block {sched} does not execute every one of {blocks} blocks")
        types = list(getattr(self, "layer_types", None) or ["full_attention"] * n)
        for b, layers in block_calls(sched).items():
            if len({types[p] for p in layers}) != 1:
                raise RecurrentDepthError(f"block {b} is executed by logical layers {layers} of different attention "
                                          f"types {[types[p] for p in layers]}")
        self.layer_to_block = sched
        self.num_parameter_blocks = blocks


class LogicalLayerCache:
    """The cache one logical layer reads and writes: ``update`` goes to entry ``layer`` whatever ``layer_idx`` the
    attention module passes; everything else is the shared cache's."""

    __slots__ = ("cache", "layer")

    def __init__(self, cache: Cache, layer: int):
        self.cache = cache
        self.layer = int(layer)

    def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
        return self.cache.update(key_states, value_states, self.layer, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.cache, name)


class RecurrentDepthPreTrainedModel(Qwen3PreTrainedModel):
    config_class = RecurrentDepthConfig
    config: RecurrentDepthConfig
    _no_split_modules = ["Qwen3DecoderLayer"]
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer, "attentions": Qwen3Attention}


class RecurrentDepthModel(RecurrentDepthPreTrainedModel):
    """`Qwen3Model` over ``num_parameter_blocks`` blocks executed on the ``layer_to_block`` schedule."""

    #: the trainer's hidden-state loss reads logical-layer outputs through `forward(hidden_supervision=...)`
    supports_hidden_supervision = True

    def __init__(self, config: RecurrentDepthConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        first = first_layer_of_block(config.layer_to_block)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, first[b]) for b in range(config.num_parameter_blocks)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values: Cache | None = None,
                inputs_embeds=None, use_cache: bool | None = None, hidden_supervision=None,
                **kwargs) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)       #: one entry per LOGICAL layer
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen).unsqueeze(0)
        if not isinstance(mask_mapping := attention_mask, dict):
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                           "past_key_values": past_key_values, "position_ids": position_ids}
            mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        wanted = set(hidden_supervision.targets) if hidden_supervision is not None else set()
        for p, b in enumerate(self.config.layer_to_block):
            hidden_states = self.layers[b](
                hidden_states,
                attention_mask=mask_mapping[self.config.layer_types[p]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None if past_key_values is None else LogicalLayerCache(past_key_values, p),
                use_cache=use_cache,
                **kwargs,
            )
            if p in wanted:
                hidden_supervision.add(p, hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None)


class RecurrentDepthForCausalLM(RecurrentDepthPreTrainedModel, Qwen3ForCausalLM):
    """Causal LM head over `RecurrentDepthModel`; its forward and generation are Qwen3's."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: RecurrentDepthConfig):
        RecurrentDepthPreTrainedModel.__init__(self, config)
        self.model = RecurrentDepthModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def register_auto_classes() -> None:
    """Make `AutoConfig` / `AutoModelForCausalLM` resolve ``aad_recurrent_depth``: something a caller DOES, never an
    import side effect (`families.register_custom_model_families`). Idempotent."""
    AutoConfig.register(MODEL_TYPE, RecurrentDepthConfig, exist_ok=True)
    AutoModelForCausalLM.register(RecurrentDepthConfig, RecurrentDepthForCausalLM, exist_ok=True)


def recurrent_config_from(base_config: dict, layer_to_block: Sequence[int]) -> dict:
    """The ``config.json`` of the recurrent model over a B-layer Qwen3's blocks: the base config (a dict, as saved)
    with ``num_hidden_layers`` the schedule's length, the layer types expanded to the logical layers (each takes its
    block's type), ``layer_to_block``, ``num_parameter_blocks`` = the base's layer count, and the family's model type
    and architecture. The weights are the base's, unchanged and under the same names."""
    cfg = dict(base_config)
    n_blocks = int(cfg["num_hidden_layers"])
    sched = [int(b) for b in layer_to_block]
    if sorted(set(sched)) != list(range(n_blocks)):
        raise RecurrentDepthError(f"the schedule executes blocks {sorted(set(sched))}, the base has {n_blocks}")
    base_types = list(cfg.get("layer_types") or ["full_attention"] * n_blocks)
    if len(base_types) != n_blocks:
        raise RecurrentDepthError(f"the base declares {len(base_types)} layer types for {n_blocks} layers")
    for k in ("auto_map", "transformers_version", "_name_or_path"):
        cfg.pop(k, None)
    cfg.update(model_type=MODEL_TYPE, architectures=[ARCHITECTURE], num_hidden_layers=len(sched),
               layer_types=[base_types[b] for b in sched], layer_to_block=sched, num_parameter_blocks=n_blocks)
    if "max_window_layers" in cfg and cfg.get("use_sliding_window") is not True:
        cfg["max_window_layers"] = len(sched)
    RecurrentDepthConfig(**{k: v for k, v in cfg.items() if k != "model_type"})        #: validates
    return cfg

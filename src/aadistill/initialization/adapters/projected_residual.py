"""The projected-residual family: a dense decoder whose sub-layers run narrower than its stream.

This module is the family's structural definition, so it lives with the
architecture adapters (AGENTS.md P3): it is the one place that knows this
family's block keeps ``.self_attn``, ``.mlp`` and two full-width norms around a
per-layer ``projection``.


The residual stream, the token embedding and the output head keep the parent's
full width ``hidden_size``. Each decoder layer ``l`` owns one projection
``P_l`` of shape ``[hidden_size, internal_size]``; its attention and its FFN
read the normalized residual through ``P_l`` and write back through
``P_l^T``::

    h = x + P_l^T-up( attn( norm_attn(x) @ P_l ) )        # down: @ P_l, up: @ P_l^T
    y = h + mlp( norm_ffn(h) @ P_l ) @ P_l^T

so the attention heads, the KV cache and the FFN are those of a model of width
``internal_size``, while the stream they add into is never compressed.

**Every structural fact is a config field.** ``model_type`` is
``aad_projected_residual``; ``internal_size`` and ``projection_sharing`` say
what the projection is. Nothing about a particular teacher, width or ratio
lives here (AGENTS.md P3): the geometry arrives in the config.

**The model is a reparametrization of a standard Qwen3 model.** Multiplying
``P_l`` into the sub-layer weights gives a stock ``Qwen3ForCausalLM`` of width
``hidden_size`` whose sub-layer weights have rank at most ``internal_size``.
``fold_to_qwen3`` does that, exactly in real arithmetic, so a runtime with no
custom modeling code can evaluate the *function* this model computes. It does
not evaluate its *cost*: the folded form does more arithmetic than the
factored one (see ``aadistill.models.cost_model``).

Only ``projection_sharing = "transpose_shared"`` exists: one ``P_l`` per layer
shared by both sub-layers, the down map ``P_l`` and the up map ``P_l^T``.
Nothing keeps ``P_l`` orthonormal once it trains; ``projection_orthogonality``
measures how far it has drifted, and whether to constrain it is a recipe
decision, not this module's.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

MODEL_TYPE = "aad_projected_residual"
PROJECTION_SHARING = ("transpose_shared",)
#: Fields this family adds to the Qwen3 config; they never reach the inner one.
EXTRA_FIELDS = ("internal_size", "projection_sharing")

__all__ = [
    "MODEL_TYPE",
    "register_auto_classes",
    "ProjectedResidualConfig",
    "ProjectedResidualDecoderLayer",
    "ProjectedResidualModel",
    "ProjectedResidualForCausalLM",
    "fold_to_qwen3",
    "projection_orthogonality",
    "projections",
]


class ProjectedResidualConfig(Qwen3Config):
    """A Qwen3 config plus the width the sub-layers compute in."""

    model_type = MODEL_TYPE

    def __init__(self, internal_size: int | None = None,
                 projection_sharing: str = "transpose_shared", **kwargs):
        super().__init__(**kwargs)
        self.internal_size = int(internal_size) if internal_size is not None else int(self.hidden_size)
        if projection_sharing not in PROJECTION_SHARING:
            raise ValueError(f"projection_sharing must be one of {PROJECTION_SHARING}, "
                             f"got {projection_sharing!r}")
        self.projection_sharing = projection_sharing
        if not 0 < self.internal_size <= self.hidden_size:
            raise ValueError(f"internal_size {self.internal_size} must lie in (0, hidden_size "
                             f"{self.hidden_size}]")

    def inner_config(self) -> Qwen3Config:
        """The config the sub-layers are built from: this one at width ``internal_size``."""
        raw = self.to_dict()
        for key in (*EXTRA_FIELDS, "model_type", "architectures", "auto_map", "transformers_version"):
            raw.pop(key, None)
        raw["hidden_size"] = self.internal_size
        inner = Qwen3Config(**raw)
        inner._attn_implementation = self._attn_implementation
        return inner


class ProjectedResidualDecoderLayer(GradientCheckpointingLayer):
    """A pre-norm block whose attention and FFN read and write through ``P_l``."""

    def __init__(self, config: ProjectedResidualConfig, layer_idx: int):
        super().__init__()
        inner = config.inner_config()
        self.hidden_size = config.hidden_size
        self.internal_size = config.internal_size
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #: [hidden_size, internal_size]: the down map; its transpose is the up map.
        self.projection = nn.Parameter(torch.empty(config.hidden_size, config.internal_size))
        self.self_attn = Qwen3Attention(config=inner, layer_idx=layer_idx)
        # Built at the inner width, then bound to the OUTER config: attention
        # reads `config._attn_implementation` at call time, and only the outer
        # config is the one `set_attn_implementation` / `from_pretrained` update.
        self.self_attn.config = config
        self.mlp = Qwen3MLP(inner)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        down = self.projection
        residual = hidden_states
        h = self.input_layernorm(hidden_states) @ down
        h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                              position_ids=position_ids, past_key_values=past_key_values,
                              use_cache=use_cache, position_embeddings=position_embeddings,
                              **kwargs)
        hidden_states = residual + h @ down.T
        residual = hidden_states
        h = self.mlp(self.post_attention_layernorm(hidden_states) @ down)
        return residual + h @ down.T


class ProjectedResidualPreTrainedModel(Qwen3PreTrainedModel):
    config_class = ProjectedResidualConfig
    config: ProjectedResidualConfig
    _no_split_modules = ["ProjectedResidualDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": ProjectedResidualDecoderLayer,
        "attentions": Qwen3Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ProjectedResidualDecoderLayer):
            # A random orthonormal basis: the identity on its own span, so a
            # fresh model is a well-conditioned reparametrization. A real
            # student overwrites it with the per-layer PCA basis.
            init.orthogonal_(module.projection)


class ProjectedResidualModel(ProjectedResidualPreTrainedModel):
    """`Qwen3Model` with projected blocks; its forward is Qwen3Model's, unchanged."""

    forward = Qwen3Model.forward

    def __init__(self, config: ProjectedResidualConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ProjectedResidualDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()


class ProjectedResidualForCausalLM(ProjectedResidualPreTrainedModel, Qwen3ForCausalLM):
    """Causal LM head over `ProjectedResidualModel`; forward and generation are Qwen3's."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ProjectedResidualConfig):
        ProjectedResidualPreTrainedModel.__init__(self, config)
        self.model = ProjectedResidualModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def register_auto_classes() -> None:
    """Make `AutoConfig` / `AutoModelForCausalLM` resolve ``aad_projected_residual``.

    Something a caller DOES, never an import side effect (the adapter
    bootstrap rule): a loader that reads a saved checkpoint through the Auto
    classes -- a trainer resuming one, an evaluation reading one -- calls this
    first. Idempotent.
    """
    AutoConfig.register(MODEL_TYPE, ProjectedResidualConfig, exist_ok=True)
    AutoModelForCausalLM.register(ProjectedResidualConfig, ProjectedResidualForCausalLM, exist_ok=True)


def projections(model: ProjectedResidualForCausalLM) -> list[torch.Tensor]:
    """Every layer's ``P_l``, in layer order."""
    return [layer.projection for layer in model.model.layers]


@torch.no_grad()
def projection_orthogonality(model: ProjectedResidualForCausalLM) -> list[dict]:
    """How far each ``P_l`` is from orthonormal columns.

    ``rel_fro`` is ``||P^T P - I||_F / sqrt(r)``; the singular-value range says
    whether the drift is a rotation-preserving rescale or a loss of rank.
    """
    out = []
    for i, p in enumerate(projections(model)):
        p = p.detach().double()
        r = p.shape[1]
        gram = p.T @ p
        s = torch.linalg.svdvals(p)
        out.append({"layer": i,
                    "rel_fro": float(torch.linalg.matrix_norm(gram - torch.eye(r, dtype=p.dtype,
                                                                               device=p.device)) / r ** 0.5),
                    "sigma_min": float(s.min()), "sigma_max": float(s.max())})
    return out


@torch.no_grad()
def fold_to_qwen3(model: ProjectedResidualForCausalLM, dtype: torch.dtype | None = None):
    """The stock `Qwen3ForCausalLM` computing the same function.

    Each sub-layer weight absorbs its layer's projection: readers become
    ``W P_l^T`` (``[out, hidden]``) and writers ``P_l W`` (``[hidden, in]``).
    Norms, embedding and head are copied. Exact in real arithmetic; in floating
    point the two differ by the rounding of the products, which the caller
    measures rather than assumes.
    """
    cfg = model.config
    if cfg.attention_bias:
        raise ValueError("fold_to_qwen3 handles bias-free attention only")
    raw = cfg.to_dict()
    for key in (*EXTRA_FIELDS, "model_type", "architectures", "auto_map", "transformers_version"):
        raw.pop(key, None)
    raw["architectures"] = ["Qwen3ForCausalLM"]
    folded_cfg = Qwen3Config(**raw)
    dtype = dtype or model.dtype
    with torch.device("meta"):
        folded = Qwen3ForCausalLM(folded_cfg)
    state = {}
    src = model.state_dict()
    for name, value in src.items():
        if ".projection" in name or name == "lm_head.weight":
            continue
        state[name] = value
    for i, layer in enumerate(model.model.layers):
        p = layer.projection.detach().double()
        pre = f"model.layers.{i}."
        for rd in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
            state[pre + rd + ".weight"] = src[pre + rd + ".weight"].double() @ p.T
        for wr in ("self_attn.o_proj", "mlp.down_proj"):
            state[pre + wr + ".weight"] = p @ src[pre + wr + ".weight"].double()
    state = {k: v.to(dtype) for k, v in state.items()}
    folded.load_state_dict(state, strict=False, assign=True)
    folded.tie_weights()
    # `inv_freq` is a non-persistent buffer: built on meta it has no values, and
    # no state dict carries it. Rebuild it from the config, on the host.
    folded.model.rotary_emb = Qwen3RotaryEmbedding(config=folded_cfg)
    missing = [n for n, t in [*folded.named_parameters(), *folded.named_buffers()] if t.is_meta]
    if missing:
        raise RuntimeError(f"folding left tensors unset: {missing[:5]}")
    return folded.eval()

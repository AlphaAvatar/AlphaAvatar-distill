"""The projected-residual family: a dense decoder whose sub-layers run narrower than its stream.

This module is the family's structural definition, so it lives with the
architecture adapters (AGENTS.md P3): it is the one place that knows this
family's block keeps ``.self_attn``, ``.mlp`` and two full-width norms around
per-layer projections.

The residual stream, the token embedding and the output head keep the parent's
full width ``hidden_size``. Each decoder layer's attention and FFN compute at
``internal_size``: each reads the normalized stream through a DOWN projection
``[internal_size, hidden_size]`` and writes back through an UP projection
``[hidden_size, internal_size]``. ``projection_sharing`` says how many such
matrices a layer owns:

``independent``
    four, none shared and none tied by a transpose: ``attn_down``,
    ``attn_up``, ``ffn_down`` and ``ffn_up``::

        a = phi_ai(attn_down(norm_attn(x)))
        h = x + attn_up(phi_ao(attn(a)))
        f = phi_fi(ffn_down(norm_ffn(h)))
        y = h + ffn_up(phi_fo(ffn(f)))

``transpose_shared``
    one ``P_l`` ``[hidden_size, internal_size]`` per layer, the down map of both
    sub-layers, its transpose their up map (the historical 902,911,488-parameter
    design of B-E; kept so that record's code still runs).

``projection_activation`` says what ``phi`` is. ``None`` is the identity: the
model is then linear across every projection and ``fold_to_qwen3`` turns it
into a stock ``Qwen3ForCausalLM`` of width ``hidden_size``, exactly in real
arithmetic. ``learnable_silu_residual`` is the identity-initialized learnable
SiLU residual gate::

    phi(z) = z + alpha * silu(z)

with one static, trainable ``alpha`` ``[internal_size]`` per position (four per
layer), initialized to zero: at initialization ``phi`` is the identity and its
input Jacobian is the identity matrix, while ``d phi / d alpha = silu(z)`` is
non-zero, so the gates train from the first step. It is not ReZero or
LayerScale (those scale a residual BRANCH; this adds a scaled nonlinearity to
a projected activation). A model with it is no longer a reparametrization of a
standard Qwen3 and does not fold; it is served by its own inference code.

**Every structural fact is a config field.** ``model_type`` is
``aad_projected_residual``; ``internal_size``, ``projection_sharing`` and
``projection_activation`` say what the projections are. Nothing about a
particular teacher, width or ratio lives here (AGENTS.md P3).

Nothing keeps a projection orthonormal once it trains;
``projection_orthogonality`` measures how far each has drifted, and
``gate_statistics`` what the gates learned.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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
PROJECTION_SHARING = ("transpose_shared", "independent")
PROJECTION_ACTIVATIONS = (None, "learnable_silu_residual")
#: The four projection sites of an `independent` layer, as (sub-layer, direction).
SITES = (("attn", "down"), ("attn", "up"), ("ffn", "down"), ("ffn", "up"))
#: The four gate positions: after each down map, before each up map.
GATES = ("attn_in_gate", "attn_out_gate", "ffn_in_gate", "ffn_out_gate")
#: Fields this family adds to the Qwen3 config; they never reach the inner one.
EXTRA_FIELDS = ("internal_size", "projection_sharing", "projection_activation")

__all__ = [
    "MODEL_TYPE",
    "register_auto_classes",
    "ProjectedResidualConfig",
    "ProjectedResidualDecoderLayer",
    "ProjectedResidualModel",
    "ProjectedResidualForCausalLM",
    "LearnableSiLUGate",
    "fold_to_qwen3",
    "gate_statistics",
    "projection_orthogonality",
    "projections",
]


class ProjectedResidualConfig(Qwen3Config):
    """A Qwen3 config plus the width the sub-layers compute in and how they reach it."""

    model_type = MODEL_TYPE

    def __init__(self, internal_size: int | None = None,
                 projection_sharing: str = "transpose_shared",
                 projection_activation: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.internal_size = int(internal_size) if internal_size is not None else int(self.hidden_size)
        if projection_sharing not in PROJECTION_SHARING:
            raise ValueError(f"projection_sharing must be one of {PROJECTION_SHARING}, "
                             f"got {projection_sharing!r}")
        if projection_activation not in PROJECTION_ACTIVATIONS:
            raise ValueError(f"projection_activation must be one of {PROJECTION_ACTIVATIONS}, "
                             f"got {projection_activation!r}")
        self.projection_sharing = projection_sharing
        self.projection_activation = projection_activation
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


class LearnableSiLUGate(nn.Module):
    """``phi(z) = z + alpha * silu(z)``, ``alpha`` per channel, zero at initialization."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.alpha * F.silu(z)


class ProjectedResidualDecoderLayer(GradientCheckpointingLayer):
    """A pre-norm block whose attention and FFN read and write through projections."""

    def __init__(self, config: ProjectedResidualConfig, layer_idx: int):
        super().__init__()
        inner = config.inner_config()
        d, r = config.hidden_size, config.internal_size
        self.hidden_size = d
        self.internal_size = r
        self.sharing = config.projection_sharing
        self.input_layernorm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        if self.sharing == "transpose_shared":
            #: [hidden_size, internal_size]: the down map; its transpose is the up map.
            self.projection = nn.Parameter(torch.empty(d, r))
        else:
            for sub, direction in SITES:
                shape = (d, r) if direction == "down" else (r, d)
                setattr(self, f"{sub}_{direction}", nn.Linear(*shape, bias=False))
        self.gated = config.projection_activation == "learnable_silu_residual"
        if self.gated:
            for name in GATES:
                setattr(self, name, LearnableSiLUGate(r))
        self.self_attn = Qwen3Attention(config=inner, layer_idx=layer_idx)
        # Built at the inner width, then bound to the OUTER config: attention
        # reads `config._attn_implementation` at call time, and only the outer
        # config is the one `set_attn_implementation` / `from_pretrained` update.
        self.self_attn.config = config
        self.mlp = Qwen3MLP(inner)

    def _down(self, sub: str, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.projection if self.sharing == "transpose_shared" else getattr(self, f"{sub}_down")(x)
        return getattr(self, f"{sub}_in_gate")(z) if self.gated else z

    def _up(self, sub: str, z: torch.Tensor) -> torch.Tensor:
        if self.gated:
            z = getattr(self, f"{sub}_out_gate")(z)
        return z @ self.projection.T if self.sharing == "transpose_shared" else getattr(self, f"{sub}_up")(z)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        residual = hidden_states
        h = self._down("attn", self.input_layernorm(hidden_states))
        h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                              position_ids=position_ids, past_key_values=past_key_values,
                              use_cache=use_cache, position_embeddings=position_embeddings,
                              **kwargs)
        hidden_states = residual + self._up("attn", h)
        residual = hidden_states
        h = self.mlp(self._down("ffn", self.post_attention_layernorm(hidden_states)))
        return residual + self._up("ffn", h)


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
        if isinstance(module, LearnableSiLUGate):
            init.zeros_(module.alpha)
        if isinstance(module, ProjectedResidualDecoderLayer):
            # Random orthonormal bases, each up map the transpose of its down
            # map: a fresh model is then a well-conditioned reparametrization.
            # A real student overwrites them with its PCA bases.
            if module.sharing == "transpose_shared":
                init.orthogonal_(module.projection)
            else:
                for sub in ("attn", "ffn"):
                    down = getattr(module, f"{sub}_down").weight
                    init.orthogonal_(down)
                    init.copy_(getattr(module, f"{sub}_up").weight, down.T)


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


def projections(model: ProjectedResidualForCausalLM) -> list[dict[str, torch.Tensor]]:
    """Every layer's projection matrices, in layer order, as ``{site: [hidden, internal]}``.

    Each is returned in the ``[hidden_size, internal_size]`` orientation whatever
    its storage: the down map's transpose, the up map as stored. Under
    ``transpose_shared`` the one ``P_l`` is the single entry ``"shared"``.
    """
    out = []
    for layer in model.model.layers:
        if layer.sharing == "transpose_shared":
            out.append({"shared": layer.projection})
        else:
            out.append({f"{sub}_{direction}": (getattr(layer, f"{sub}_{direction}").weight.T
                                               if direction == "down"
                                               else getattr(layer, f"{sub}_{direction}").weight)
                        for sub, direction in SITES})
    return out


@torch.no_grad()
def projection_orthogonality(model: ProjectedResidualForCausalLM) -> list[dict]:
    """How far each projection is from orthonormal columns (``[hidden, internal]`` orientation).

    ``rel_fro`` is ``||P^T P - I||_F / sqrt(r)``; the singular-value range says
    whether the drift is a rotation-preserving rescale or a loss of rank. For
    independent projections, ``up_down_rel_fro`` also says how far each
    sub-layer's up map is from the transpose of its down map.
    """
    out = []
    for i, sites in enumerate(projections(model)):
        for site, p in sites.items():
            p = p.detach().double()
            r = p.shape[1]
            gram = p.T @ p
            s = torch.linalg.svdvals(p)
            out.append({"layer": i, "site": site,
                        "rel_fro": float(torch.linalg.matrix_norm(
                            gram - torch.eye(r, dtype=p.dtype, device=p.device)) / r ** 0.5),
                        "sigma_min": float(s.min()), "sigma_max": float(s.max())})
        if "attn_down" in sites:
            for sub in ("attn", "ffn"):
                down, up = sites[f"{sub}_down"].detach().double(), sites[f"{sub}_up"].detach().double()
                out.append({"layer": i, "site": f"{sub}_up_vs_down",
                            "up_down_rel_fro": float(torch.linalg.matrix_norm(up - down)
                                                     / torch.linalg.matrix_norm(down))})
    return out


@torch.no_grad()
def gate_statistics(model: ProjectedResidualForCausalLM) -> list[dict]:
    """Per layer and gate: mean, mean |alpha|, quantiles of |alpha|, and the gradient norm if held."""
    out = []
    for i, layer in enumerate(model.model.layers):
        if not layer.gated:
            continue
        for name in GATES:
            a = getattr(layer, name).alpha
            x = a.detach().double()
            q = torch.quantile(x.abs(), torch.tensor([0.5, 0.9, 1.0], dtype=x.dtype, device=x.device))
            out.append({"layer": i, "gate": name, "mean": float(x.mean()), "mean_abs": float(x.abs().mean()),
                        "abs_p50": float(q[0]), "abs_p90": float(q[1]), "abs_max": float(q[2]),
                        "grad_norm": (None if a.grad is None else float(a.grad.detach().double().norm()))})
    return out


@torch.no_grad()
def fold_to_qwen3(model: ProjectedResidualForCausalLM, dtype: torch.dtype | None = None):
    """The stock `Qwen3ForCausalLM` computing the same function, for a linear model.

    Each sub-layer weight absorbs its projections: readers ``W D`` (``[out,
    hidden]``, ``D`` the sub-layer's down map ``[internal, hidden]``) and writers
    ``U W`` (``[hidden, in]``, ``U`` its up map). Norms, embedding and head are
    copied. Exact in real arithmetic; in floating point the two differ by the
    rounding of the products, which the caller measures rather than assumes.
    Refused for a model with a projection activation: it is not linear across
    the projections, so no stock Qwen3 computes its function.
    """
    cfg = model.config
    if cfg.attention_bias:
        raise ValueError("fold_to_qwen3 handles bias-free attention only")
    if cfg.projection_activation is not None:
        raise ValueError(f"projection_activation {cfg.projection_activation!r} is not linear: "
                         "this model does not fold to a stock Qwen3")
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
        if name == "lm_head.weight" or ".projection" in name or any(
                f".{sub}_{direction}." in name for sub, direction in SITES):
            continue
        state[name] = value
    for i, sites in enumerate(projections(model)):
        pre = f"model.layers.{i}."
        maps = {}
        for sub in ("attn", "ffn"):
            if "shared" in sites:
                p = sites["shared"].detach().double()
                maps[sub] = (p.T, p)                       # down [r, d], up [d, r]
            else:
                maps[sub] = (sites[f"{sub}_down"].detach().double().T,
                             sites[f"{sub}_up"].detach().double())
        for rd, sub in (("self_attn.q_proj", "attn"), ("self_attn.k_proj", "attn"),
                        ("self_attn.v_proj", "attn"), ("mlp.gate_proj", "ffn"), ("mlp.up_proj", "ffn")):
            state[pre + rd + ".weight"] = src[pre + rd + ".weight"].double() @ maps[sub][0]
        for wr, sub in (("self_attn.o_proj", "attn"), ("mlp.down_proj", "ffn")):
            state[pre + wr + ".weight"] = maps[sub][1] @ src[pre + wr + ".weight"].double()
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

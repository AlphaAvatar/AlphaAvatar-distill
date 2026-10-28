"""The projected-residual family in vLLM, as an out-of-tree model.

Imported only inside the vLLM environment (it imports ``vllm``). vLLM finds it
through its general-plugin entry point (group ``vllm.general_plugins``, loaded
in every vLLM process: the front end, the engine core and each worker), which
:func:`register` serves: it registers this package's Hugging Face config class
with ``AutoConfig`` -- vLLM reads ``config.json`` through it -- and the model
class with vLLM's ``ModelRegistry`` under the checkpoint's architecture name.
The entry point's metadata lives in ``scripts/shared/rollout/vllm_plugin/``,
which a caller puts on the vLLM interpreter's ``PYTHONPATH`` together with
``src`` (``scripts/shared/rollout/vllm_plugin_path.py``); nothing is
installed into the engine's environment.

The model is built from vLLM's own Qwen3 components at the INTERNAL width
(``Qwen3Attention``, the Qwen2/3 MLP, the fused residual RMSNorm at the full
width) plus the family's projections (``ReplicatedLinear``) and its gates, and
computes, per layer::

    a = phi(attn_down(norm(x)));   h = x + attn_up(phi(attn(a)))
    f = phi(ffn_down(norm(h)));    y = h + ffn_up(phi(ffn(f)))

exactly as :mod:`aadistill.initialization.adapters.projected_residual` does in
Hugging Face; the HF model is the reference implementation, and agreement
between the two is measured on a GPU, never assumed. Weight names are the HF
checkpoint's; :func:`vllm_parameter_name` is the whole mapping, a pure function
the CPU tests pin.
"""
from __future__ import annotations

from collections.abc import Iterable

ARCHITECTURE = "ProjectedResidualForCausalLM"
#: HF weight-name fragments vLLM stacks into one fused parameter (Qwen2/3's own).
STACKED = ((".q_proj", ".qkv_proj", "q"), (".k_proj", ".qkv_proj", "k"), (".v_proj", ".qkv_proj", "v"),
           (".gate_proj", ".gate_up_proj", 0), (".up_proj", ".gate_up_proj", 1))


def vllm_parameter_name(hf_name: str):
    """``(vllm_name, shard_id)`` for an HF weight name; ``shard_id`` is ``None`` when unstacked.

    Only Qwen's own fused projections stack. The family's own modules --
    ``attn_down``, ``attn_up``, ``ffn_down``, ``ffn_up``, ``projection`` and the
    ``*_gate.alpha`` vectors -- keep their names: none contains a stacked
    fragment, which a test asserts.
    """
    for frag, fused, shard in STACKED:
        if frag + "." in hf_name or hf_name.endswith(frag):
            return hf_name.replace(frag, fused), shard
    return hf_name, None


def register() -> None:
    """The plugin entry point: idempotent, safe to call in every vLLM process."""
    from vllm import ModelRegistry

    from aadistill.initialization.adapters.projected_residual import register_auto_classes

    register_auto_classes()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, f"{__name__}:ProjectedResidualForCausalLM")


try:  # the model classes exist only where vLLM does
    import torch
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import VllmConfig
    from vllm.distributed import get_pp_group
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.interfaces import SupportsPP
    from vllm.model_executor.models.qwen2 import Qwen2Model
    from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
    from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
    from vllm.transformers_utils.config import set_default_rope_theta
except ImportError:  # pragma: no cover - the training environment has no vLLM
    _HAVE_VLLM = False
else:
    _HAVE_VLLM = True

if _HAVE_VLLM:
    import torch.nn.functional as F

    class LearnableSiLUGate(nn.Module):
        """``phi(z) = z + alpha * silu(z)``; ``alpha`` loaded from the checkpoint."""

        def __init__(self, dim: int):
            super().__init__()
            self.alpha = nn.Parameter(torch.zeros(dim), requires_grad=False)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return z + self.alpha * F.silu(z)

    class ProjectedResidualDecoderLayer(nn.Module):
        def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
            super().__init__()
            d, r = config.hidden_size, config.internal_size
            set_default_rope_theta(config, default_theta=1000000)
            self.sharing = config.projection_sharing
            self.gated = config.projection_activation == "learnable_silu_residual"
            if config.projection_activation not in (None, "learnable_silu_residual"):
                raise ValueError(f"unknown projection_activation {config.projection_activation!r}")
            self.self_attn = Qwen3Attention(
                hidden_size=r, num_heads=config.num_attention_heads,
                max_position=config.max_position_embeddings, num_kv_heads=config.num_key_value_heads,
                rms_norm_eps=config.rms_norm_eps, qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None), cache_config=cache_config,
                quant_config=quant_config, rope_parameters=config.rope_parameters,
                prefix=f"{prefix}.self_attn")
            self.mlp = Qwen3MLP(hidden_size=r, intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act, quant_config=quant_config,
                                prefix=f"{prefix}.mlp")
            self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
            if self.sharing == "transpose_shared":
                self.projection = nn.Parameter(torch.empty(d, r), requires_grad=False)
                self.projection.weight_loader = default_weight_loader
            else:
                for sub in ("attn", "ffn"):
                    setattr(self, f"{sub}_down", ReplicatedLinear(d, r, bias=False, quant_config=quant_config,
                                                                  prefix=f"{prefix}.{sub}_down"))
                    setattr(self, f"{sub}_up", ReplicatedLinear(r, d, bias=False, quant_config=quant_config,
                                                                prefix=f"{prefix}.{sub}_up"))
            if self.gated:
                for name in ("attn_in_gate", "attn_out_gate", "ffn_in_gate", "ffn_out_gate"):
                    setattr(self, name, LearnableSiLUGate(r))

        def _down(self, sub: str, x: torch.Tensor) -> torch.Tensor:
            z = x @ self.projection if self.sharing == "transpose_shared" else getattr(self, f"{sub}_down")(x)[0]
            return getattr(self, f"{sub}_in_gate")(z) if self.gated else z

        def _up(self, sub: str, z: torch.Tensor) -> torch.Tensor:
            if self.gated:
                z = getattr(self, f"{sub}_out_gate")(z)
            return z @ self.projection.T if self.sharing == "transpose_shared" else getattr(self, f"{sub}_up")(z)[0]

        def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            h = self.self_attn(positions=positions, hidden_states=self._down("attn", hidden_states))
            hidden_states, residual = self.post_attention_layernorm(self._up("attn", h), residual)
            return self._up("ffn", self.mlp(self._down("ffn", hidden_states))), residual

    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1,
                                             "intermediate_tensors": 0, "inputs_embeds": 0})
    class ProjectedResidualModel(Qwen2Model):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix,
                             decoder_layer_type=ProjectedResidualDecoderLayer)

    class ProjectedResidualForCausalLM(nn.Module, SupportsPP):
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                                  "gate_up_proj": ["gate_proj", "up_proj"]}

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.config = config
            self.model = ProjectedResidualModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
            if get_pp_group().is_last_rank:
                self.lm_head = (self.model.embed_tokens if config.tie_word_embeddings else
                                ParallelLMHead(config.vocab_size, config.hidden_size,
                                               quant_config=vllm_config.quant_config,
                                               prefix=maybe_prefix(prefix, "lm_head")))
            else:
                self.lm_head = PPMissingLayer()
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model.embed_input_ids(input_ids)

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

        def compute_logits(self, hidden_states: torch.Tensor):
            return self.logits_processor(self.lm_head, hidden_states)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            """HF names through :func:`vllm_parameter_name`; every parameter must be loaded."""
            params = dict(self.named_parameters())
            loaded: set[str] = set()
            for hf_name, tensor in weights:
                if "rotary_emb.inv_freq" in hf_name:
                    continue
                if hf_name.startswith("lm_head.") and self.config.tie_word_embeddings:
                    continue
                name, shard = vllm_parameter_name(hf_name)
                if name not in params:
                    raise KeyError(f"checkpoint weight {hf_name} has no vLLM parameter ({name})")
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)
                loaded.add(name)
            missing = sorted(set(params) - loaded - ({"lm_head.weight"} if self.config.tie_word_embeddings
                                                     else set()))
            if missing:
                raise KeyError(f"parameters the checkpoint did not provide: {missing[:8]}")
            return loaded

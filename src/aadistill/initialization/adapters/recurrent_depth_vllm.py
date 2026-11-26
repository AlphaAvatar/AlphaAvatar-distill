"""The recurrent-depth family in vLLM, as an out-of-tree model.

Imported only inside the vLLM environment (it imports ``vllm``). vLLM finds it through its general-plugin entry point
(group ``vllm.general_plugins``, declared in ``scripts/shared/rollout/vllm_plugin/``), which :func:`register` serves:
it registers the family's Hugging Face config class with ``AutoConfig`` -- vLLM reads ``config.json`` through it --
and the model class with vLLM's ``ModelRegistry`` under the checkpoint's architecture name. Nothing is installed into
the engine's environment; a caller puts ``src`` and the plugin directory on the vLLM interpreter's ``PYTHONPATH``
(`shared.rollout.vllm_plugin_path`).

**What it executes** is :mod:`aadistill.initialization.adapters.recurrent_depth`'s schedule: ``num_hidden_layers``
logical layers, logical layer ``p`` running parameter block ``layer_to_block[p]`` (a whole Qwen3 decoder block built
from vLLM's own components: the fused QKV and gate/up projections, q/k RMSNorm, RoPE, the fused residual RMSNorm).
The HF model is the reference implementation; agreement between the two is measured on a GPU, never assumed.

**One KV cache per logical layer.** vLLM allocates a KV cache for every ``Attention`` module it finds by its prefix
(``static_forward_context``; `bind_kv_cache` orders them by the one integer in the name). The B blocks therefore own
NO ``Attention``: their attention keeps only its weights (``qkv_proj``, ``o_proj``, ``q_norm``, ``k_norm``) and the
RoPE, and the model owns one ``Attention`` per logical layer (``<prefix>.attn_layers.<p>``), which logical layer
``p`` calls. Two logical layers that share a block's parameters never share a cache entry; positions are the same
for a token at every logical layer.

**Weights** are the HF checkpoint's names (``model.layers.<b>.*``, B blocks); `vllm_parameter_name` (the projected
family's pure mapping of Qwen's fused projections) is the whole renaming, and every parameter must be loaded.

Pipeline parallelism is refused (the schedule is one chain over all logical layers).
"""
from __future__ import annotations

from collections.abc import Iterable

from aadistill.initialization.adapters.projected_residual_vllm import vllm_parameter_name
from aadistill.initialization.adapters.recurrent_depth import ARCHITECTURE

__all__ = ["ARCHITECTURE", "register", "attention_prefix", "vllm_parameter_name"]


def attention_prefix(model_prefix: str, layer: int) -> str:
    """The prefix of logical layer ``layer``'s ``Attention``: one integer in the name, the logical layer's."""
    return f"{model_prefix}.attn_layers.{int(layer)}"


def register() -> None:
    """The plugin entry point: idempotent, safe to call in every vLLM process."""
    from vllm import ModelRegistry

    from aadistill.initialization.adapters.recurrent_depth import register_auto_classes

    register_auto_classes()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, f"{__name__}:RecurrentDepthForCausalLM")


try:  # the model classes exist only where vLLM does
    import torch
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import VllmConfig
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, maybe_prefix
    from vllm.transformers_utils.config import set_default_rope_theta
except ImportError:  # pragma: no cover - the training environment has no vLLM
    _HAVE_VLLM = False
else:
    _HAVE_VLLM = True

if _HAVE_VLLM:

    class RecurrentAttentionWeights(nn.Module):
        """Qwen3 attention's weights and RoPE without an ``Attention`` (the caller passes the logical layer's)."""

        def __init__(self, config, quant_config=None, prefix: str = ""):
            super().__init__()
            tp = get_tensor_model_parallel_world_size()
            self.total_num_heads = config.num_attention_heads
            self.total_num_kv_heads = config.num_key_value_heads
            assert self.total_num_heads % tp == 0
            if self.total_num_kv_heads >= tp:
                assert self.total_num_kv_heads % tp == 0
            else:
                assert tp % self.total_num_kv_heads == 0
            self.num_heads = self.total_num_heads // tp
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
            self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.scaling = self.head_dim ** -0.5
            self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, self.total_num_heads,
                                              self.total_num_kv_heads, bias=getattr(config, "attention_bias", False),
                                              quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
            self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
                                            quant_config=quant_config, prefix=f"{prefix}.o_proj")
            self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                       rope_parameters=config.rope_parameters)
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        def qkv(self, positions: torch.Tensor, hidden_states: torch.Tensor):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = self.q_norm(q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)).view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
            return q, k, v

    class RecurrentDecoderBlock(nn.Module):
        """One parameter block: a Qwen3 decoder block's weights (its attention without an ``Attention``)."""

        def __init__(self, config, quant_config=None, prefix: str = ""):
            super().__init__()
            self.self_attn = RecurrentAttentionWeights(config, quant_config, prefix=f"{prefix}.self_attn")
            self.mlp = Qwen2MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act, quant_config=quant_config, prefix=f"{prefix}.mlp")
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1,
                                             "intermediate_tensors": 0, "inputs_embeds": 0})
    class RecurrentDepthModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            config = vllm_config.model_config.hf_config.get_text_config()
            if get_pp_group().world_size != 1:
                raise ValueError("the recurrent-depth model does not support pipeline parallelism")
            if getattr(config, "use_sliding_window", False):
                raise ValueError("the recurrent-depth vLLM model implements full attention only")
            set_default_rope_theta(config, default_theta=1000000)
            cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
            self.config = config
            self.layer_to_block = tuple(int(b) for b in config.layer_to_block)
            n_blocks = int(config.num_parameter_blocks)
            if len(self.layer_to_block) != int(config.num_hidden_layers) or sorted(set(self.layer_to_block)) != list(
                    range(n_blocks)):
                raise ValueError(f"layer_to_block {self.layer_to_block} is not a schedule over {n_blocks} blocks and "
                                 f"{config.num_hidden_layers} logical layers")
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                       quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
            self.layers = nn.ModuleList([RecurrentDecoderBlock(config, quant_config, prefix=f"{prefix}.layers.{b}")
                                         for b in range(n_blocks)])
            w = self.layers[0].self_attn
            self.attn_layers = nn.ModuleList([
                Attention(w.num_heads, w.head_dim, w.scaling, num_kv_heads=w.num_kv_heads, cache_config=cache_config,
                          quant_config=quant_config, prefix=attention_prefix(prefix, p))
                for p in range(len(self.layer_to_block))])
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed_tokens(input_ids)

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
            for p, b in enumerate(self.layer_to_block):
                blk = self.layers[b]
                if residual is None:
                    residual = hidden
                    hidden = blk.input_layernorm(hidden)
                else:
                    hidden, residual = blk.input_layernorm(hidden, residual)
                q, k, v = blk.self_attn.qkv(positions, hidden)
                hidden, _ = blk.self_attn.o_proj(self.attn_layers[p](q, k, v))
                hidden, residual = blk.post_attention_layernorm(hidden, residual)
                hidden = blk.mlp(hidden)
            hidden, _ = self.norm(hidden, residual)
            return hidden

    class RecurrentDepthForCausalLM(nn.Module):
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                                  "gate_up_proj": ["gate_proj", "up_proj"]}

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.config = config
            self.model = RecurrentDepthModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
            self.lm_head = (self.model.embed_tokens if config.tie_word_embeddings else
                            ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=vllm_config.quant_config,
                                           prefix=maybe_prefix(prefix, "lm_head")))
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model.embed_input_ids(input_ids)

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

        def compute_logits(self, hidden_states: torch.Tensor):
            return self.logits_processor(self.lm_head, hidden_states)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            """HF names through `vllm_parameter_name`; every parameter must be loaded."""
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

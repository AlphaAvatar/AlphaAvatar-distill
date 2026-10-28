"""The projected-residual family's vLLM integration, at $0 on the host.

The engine itself needs CUDA and the vLLM environment, so its agreement with the
Hugging Face reference is a GPU measurement. What the host CAN pin: the plugin
entry point is discoverable from the PYTHONPATH the drivers hand vLLM, the
module imports without vLLM, and the weight-name mapping covers every
parameter of a real checkpoint layout with nothing but Qwen's own fusions.
"""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import torch

from aadistill.initialization.adapters.projected_residual import (
    ProjectedResidualConfig, ProjectedResidualForCausalLM)
from aadistill.initialization.adapters import projected_residual_vllm as V
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from shared.rollout.vllm_plugin_path import PLUGIN_DIR as VLLM_PLUGIN_DIR, vllm_plugin_pythonpath  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _hf_names(**over):
    cfg = dict(vocab_size=67, hidden_size=48, internal_size=16, intermediate_size=40,
               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               tie_word_embeddings=True, **over)
    with torch.device("meta"):
        m = ProjectedResidualForCausalLM(ProjectedResidualConfig(**cfg))
    return list(m.state_dict())


def test_weight_names_map_through_qwens_own_fusions_only():
    names = _hf_names(projection_sharing="independent", projection_activation="learnable_silu_residual")
    mapped = {n: V.vllm_parameter_name(n) for n in names}
    assert mapped["model.layers.0.self_attn.q_proj.weight"] == ("model.layers.0.self_attn.qkv_proj.weight", "q")
    assert mapped["model.layers.1.self_attn.v_proj.weight"] == ("model.layers.1.self_attn.qkv_proj.weight", "v")
    assert mapped["model.layers.0.mlp.gate_proj.weight"] == ("model.layers.0.mlp.gate_up_proj.weight", 0)
    assert mapped["model.layers.0.mlp.up_proj.weight"] == ("model.layers.0.mlp.gate_up_proj.weight", 1)
    family = [n for n in names if any(f".{s}" in n for s in (
        "attn_down", "attn_up", "ffn_down", "ffn_up", "attn_in_gate", "attn_out_gate",
        "ffn_in_gate", "ffn_out_gate"))]
    assert len(family) == 2 * 8
    for n in family:
        assert mapped[n] == (n, None), n
    for n in ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight",
              "model.layers.0.self_attn.q_norm.weight", "model.embed_tokens.weight", "model.norm.weight"):
        assert mapped[n] == (n, None)
    shared = _hf_names()
    assert V.vllm_parameter_name("model.layers.0.projection") == ("model.layers.0.projection", None)
    assert "model.layers.0.projection" in shared


def test_the_plugin_entry_point_is_found_on_the_drivers_pythonpath():
    path = vllm_plugin_pythonpath(REPO)
    assert path.split(":")[:2] == [str(REPO / "src"), str(REPO / VLLM_PLUGIN_DIR)]
    code = ("import importlib.metadata as m; "
            "print([(e.name, e.value) for e in m.entry_points(group='vllm.general_plugins')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": path, "PATH": "/usr/bin:/bin"}).stdout
    assert ("aadistill_projected_residual", "aadistill.initialization.adapters.projected_residual_vllm:register") in eval(out)
    # importable where vLLM is absent; the register function is the entry point's target
    assert callable(V.register) and V.ARCHITECTURE == "ProjectedResidualForCausalLM"
    assert importlib.metadata is not None

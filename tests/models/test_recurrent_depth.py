"""The recurrent-depth family (`aadistill.initialization.adapters.recurrent_depth`) on tiny real Qwen3 geometries.

What must hold for a recurrent model to be what it claims, each checked against an independent reference:

* its forward is the forward of an explicitly unrolled Qwen3 whose L layers ARE the B blocks (the same module objects,
  so the same tensors) in schedule order -- the same logits, bit for bit;
* parameters are registered once (B blocks), and each shared parameter's gradient is the SUM of the gradients its
  calls receive in an unrolled model of independent copies;
* generation with the KV cache equals the no-cache forward, every logical layer keeps its own cache entry (two layers
  of one block hold different keys), and positions do not advance between logical layers;
* gradient checkpointing on or off gives the same loss and gradients;
* save / load keeps the schedule, the sharing and the logits; a B-layer Qwen3 checkpoint with
  `recurrent_config_from` loads as the recurrent model over the same weights, and the identity schedule IS the Qwen3.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from aadistill.initialization.adapters import recurrent_depth as RD
from aadistill.initialization.adapters.families import register_custom_model_families

GEOM = dict(vocab_size=67, hidden_size=32, intermediate_size=40, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, tie_word_embeddings=True, max_position_embeddings=256, rope_theta=10000.0)
SCHED = [0, 1, 1, 1, 2, 3, 3]                      #: 4 blocks, 7 logical layers: a span of 3 and a span of 2


def recurrent(seed=0, sched=SCHED, dtype=torch.float64):
    torch.manual_seed(seed)
    cfg = RD.RecurrentDepthConfig(num_hidden_layers=len(sched), layer_to_block=sched,
                                  num_parameter_blocks=max(sched) + 1, **GEOM)
    m = RD.RecurrentDepthForCausalLM(cfg).to(dtype).eval()
    with torch.no_grad():                           #: non-unit norms so a norm mix-up cannot hide
        for mod in m.modules():
            if hasattr(mod, "variance_epsilon"):
                mod.weight.uniform_(0.5, 1.5)
    return m


def unrolled_tied(m):
    """A Qwen3ForCausalLM of L layers whose layer p IS block layer_to_block[p] (the same module)."""
    cfg = Qwen3Config(num_hidden_layers=m.config.num_hidden_layers, **GEOM)
    ref = Qwen3ForCausalLM(cfg).to(m.dtype).eval()
    ref.model.embed_tokens = m.model.embed_tokens
    ref.model.norm = m.model.norm
    ref.model.layers = torch.nn.ModuleList([m.model.layers[b] for b in m.config.layer_to_block])
    ref.lm_head = m.lm_head
    return ref


def unrolled_copies(m):
    """A Qwen3ForCausalLM of L INDEPENDENT copies (the cache-correct reference: each copy has its own layer_idx)."""
    cfg = Qwen3Config(num_hidden_layers=m.config.num_hidden_layers, **GEOM)
    ref = Qwen3ForCausalLM(cfg).to(m.dtype).eval()
    sd = {}
    for n, t in m.state_dict().items():
        if n.startswith("model.layers."):
            b = int(n.split(".")[2])
            for p, bb in enumerate(m.config.layer_to_block):
                if bb == b:
                    sd[n.replace(f"model.layers.{b}.", f"model.layers.{p}.", 1)] = t.clone()
        else:
            sd[n] = t.clone()
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}
    ref.tie_weights()
    return ref


def ids(seed=1, n=2, t=11):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, GEOM["vocab_size"], (n, t), generator=g)


# ---- schedules and configs ----------------------------------------------------------------------------------------

def test_schedule_from_spans_is_each_block_once_per_root_block_of_its_span():
    s = RD.schedule_from_spans([(0,), (1, 2, 3), (4,), (5, 6)])
    assert s == {"layer_to_block": [0, 1, 1, 1, 2, 3, 3], "root_layer_of_layer": [0, 1, 2, 3, 4, 5, 6]}
    with pytest.raises(RD.RecurrentDepthError):
        RD.schedule_from_spans([(0,), (2, 3)])          #: does not tile 0 .. n-1
    assert RD.block_calls([0, 1, 1, 2]) == {0: [0], 1: [1, 2], 2: [3]}
    assert RD.first_layer_of_block([0, 1, 1, 2]) == [0, 1, 3]


def test_a_config_refuses_an_unused_block_and_a_wrong_length():
    with pytest.raises(RD.RecurrentDepthError):
        RD.RecurrentDepthConfig(num_hidden_layers=3, layer_to_block=[0, 0, 2], num_parameter_blocks=3, **GEOM)
    with pytest.raises(RD.RecurrentDepthError):
        RD.RecurrentDepthConfig(num_hidden_layers=4, layer_to_block=[0, 1, 1], **GEOM)


def test_the_real_schedule_has_16_blocks_and_36_logical_layers_from_the_mapping():
    from aadistill.initialization.planning.layer_merge import spans_from_kept_blocks
    kept = [0, 1, 4, 5, 6, 7, 10, 12, 18, 19, 22, 23, 24, 29, 31, 34]    #: SOB1's committed depth selection
    s = RD.schedule_from_spans(spans_from_kept_blocks(36, kept))
    assert len(s["layer_to_block"]) == 36 and max(s["layer_to_block"]) + 1 == 16
    assert s["root_layer_of_layer"] == list(range(36))
    assert [len(c) for c in RD.block_calls(s["layer_to_block"]).values()] == [1, 3, 1, 1, 1, 3, 2, 6, 1, 3, 1, 1, 5, 2,
                                                                              3, 2]


# ---- the forward against the unrolled reference -------------------------------------------------------------------

def test_the_forward_is_the_unrolled_tied_qwen3_bit_for_bit():
    m = recurrent()
    x = ids()
    with torch.no_grad():
        got = m(x, use_cache=False).logits
        want = unrolled_tied(m)(x, use_cache=False).logits
    assert torch.equal(got, want)


def test_parameters_are_registered_once_and_each_gradient_sums_its_calls():
    m = recurrent()
    names = [n for n, _ in m.named_parameters()]
    assert len(names) == len(set(names))
    assert sorted({int(n.split(".")[2]) for n in names if n.startswith("model.layers.")}) == [0, 1, 2, 3]
    copies = unrolled_copies(m)
    blocks = sum(p.numel() for n, p in m.named_parameters() if n.startswith("model.layers."))
    per_block = blocks // 4
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in copies.parameters()) - 3 * per_block
    x = ids()
    m.train()
    copies.train()
    m(x, labels=x, use_cache=False).loss.backward()
    copies(x, labels=x, use_cache=False).loss.backward()
    for n, p in m.named_parameters():
        if not n.startswith("model.layers."):
            continue
        b = int(n.split(".")[2])
        want = sum(dict(copies.named_parameters())[n.replace(f"layers.{b}.", f"layers.{q}.", 1)].grad
                   for q, bb in enumerate(SCHED) if bb == b)
        assert torch.allclose(p.grad, want, atol=1e-12, rtol=1e-10), n


def test_gradient_checkpointing_changes_neither_the_loss_nor_the_gradients():
    out = {}
    for gc in (False, True):
        m = recurrent()
        m.train()
        if gc:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loss = m(ids(), labels=ids(), use_cache=False).loss
        loss.backward()
        out[gc] = (float(loss), {n: p.grad.clone() for n, p in m.named_parameters()})
    assert out[True][0] == pytest.approx(out[False][0], rel=1e-12)
    for n in out[False][1]:
        assert torch.allclose(out[True][1][n], out[False][1][n], atol=1e-12, rtol=1e-10), n


# ---- the cache ----------------------------------------------------------------------------------------------------

def test_cached_decoding_equals_the_no_cache_forward_and_each_logical_layer_has_its_own_entry():
    m = recurrent()
    x = ids(n=1, t=6)
    with torch.no_grad():
        out = m(x, use_cache=True)
        cache = out.past_key_values
        assert len(cache.layers) == len(SCHED)
        #: logical layers 1, 2, 3 share block 1's parameters and see different inputs: their keys differ
        k1, k2 = cache.layers[1].keys, cache.layers[2].keys
        assert not torch.allclose(k1, k2)
        seq, logits_step = x, out.logits[:, -1]
        for _ in range(4):
            nxt = logits_step.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
            step = m(nxt, past_key_values=cache, use_cache=True)
            logits_step = step.logits[:, -1]
            full = m(seq, use_cache=False).logits[:, -1]
            assert torch.allclose(logits_step, full, atol=1e-10, rtol=1e-9)
        #: and the cache-correct unrolled reference agrees token for token
        ref = unrolled_copies(m)
        g1 = m.generate(x, max_new_tokens=5, do_sample=False)
        g2 = ref.generate(x, max_new_tokens=5, do_sample=False)
    assert torch.equal(g1, g2)


def test_positions_do_not_advance_between_logical_layers():
    """With position ids given explicitly, logical layers read the same positions: shifting them all by a constant
    changes the logits only through RoPE's relative phase, which is invariant -- the recurrent model must agree with
    the unrolled reference at the same shifted positions."""
    m = recurrent()
    x = ids(n=1, t=9)
    pos = (torch.arange(9) + 17).unsqueeze(0)
    with torch.no_grad():
        a = m(x, position_ids=pos, use_cache=False).logits
        b = unrolled_tied(m)(x, position_ids=pos, use_cache=False).logits
    assert torch.equal(a, b)


# ---- persistence and conversion -----------------------------------------------------------------------------------

def test_save_and_load_keep_the_schedule_the_sharing_and_the_logits(tmp_path):
    register_custom_model_families()
    m = recurrent(dtype=torch.float32)
    m.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["layer_to_block"] == SCHED and cfg["num_parameter_blocks"] == 4 and cfg["num_hidden_layers"] == 7
    from safetensors import safe_open
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as h:
        blocks = sorted({int(k.split(".")[2]) for k in h.keys() if k.startswith("model.layers.")})
    assert blocks == [0, 1, 2, 3]
    back = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    assert type(back).__name__ == "RecurrentDepthForCausalLM" and back.config.layer_to_block == SCHED
    with torch.no_grad():
        assert torch.equal(back(ids(), use_cache=False).logits, m(ids(), use_cache=False).logits)


def test_a_qwen3_checkpoint_becomes_the_recurrent_model_over_the_same_weights(tmp_path):
    register_custom_model_families()
    torch.manual_seed(3)
    base = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=4, **GEOM)).float().eval()
    base.save_pretrained(tmp_path / "base")
    base_cfg = json.loads((tmp_path / "base/config.json").read_text())
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec/config.json").write_text(json.dumps(RD.recurrent_config_from(base_cfg, SCHED)))
    (tmp_path / "rec/model.safetensors").write_bytes((tmp_path / "base/model.safetensors").read_bytes())
    rec = AutoModelForCausalLM.from_pretrained(tmp_path / "rec", dtype=torch.float32).eval()
    for n, t in base.state_dict().items():
        assert torch.equal(rec.state_dict()[n], t), n
    with torch.no_grad():
        assert torch.equal(rec(ids(), use_cache=False).logits, unrolled_tied(rec)(ids(), use_cache=False).logits)
    #: the identity schedule over the same blocks IS the base
    (tmp_path / "ident").mkdir()
    (tmp_path / "ident/config.json").write_text(json.dumps(RD.recurrent_config_from(base_cfg, [0, 1, 2, 3])))
    (tmp_path / "ident/model.safetensors").write_bytes((tmp_path / "base/model.safetensors").read_bytes())
    ident = AutoModelForCausalLM.from_pretrained(tmp_path / "ident", dtype=torch.float32).eval()
    with torch.no_grad():
        assert torch.equal(ident(ids(), use_cache=False).logits, base(ids(), use_cache=False).logits)
    with pytest.raises(RD.RecurrentDepthError):
        RD.recurrent_config_from(base_cfg, [0, 1, 1, 2])           #: block 3 never executed


def test_the_logical_layer_cache_never_mutates_the_shared_module():
    m = recurrent()
    before = [layer.self_attn.layer_idx for layer in m.model.layers]
    assert before == RD.first_layer_of_block(SCHED)
    with torch.no_grad():
        m.generate(ids(n=1, t=4), max_new_tokens=3, do_sample=False)
    assert [layer.self_attn.layer_idx for layer in m.model.layers] == before


def test_the_vllm_weight_names_map_the_hf_blocks_and_attention_prefixes_have_one_integer():
    from aadistill.initialization.adapters import recurrent_depth_vllm as RV
    assert RV.vllm_parameter_name("model.layers.15.self_attn.k_proj.weight") == (
        "model.layers.15.self_attn.qkv_proj.weight", "k")
    assert RV.vllm_parameter_name("model.layers.3.mlp.up_proj.weight") == ("model.layers.3.mlp.gate_up_proj.weight", 1)
    assert RV.vllm_parameter_name("model.layers.3.self_attn.q_norm.weight") == (
        "model.layers.3.self_attn.q_norm.weight", None)
    for p in range(36):
        name = RV.attention_prefix("model", p)
        assert [int(x) for x in name.split(".") if x.isdigit()] == [p]

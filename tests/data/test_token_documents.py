"""Token-in/token-out documents, isolated packing, and the trainer path that uses them.

The load-bearing check is conditional-context equality: under isolation every
logit a target is trained against equals the logit of its document run alone,
for the student and the teacher alike, so KD compares the two models on the
context the teacher generated in. Without position ids the same packed
sequence leaks the earlier document into the later one; that is the existing
trainer's behaviour and is asserted here so the difference stays visible.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from aadistill.data.token_documents import (
    DocumentError,
    TokenDocument,
    document_from_turn,
    generation_suffix,
    isolation_report,
    pack_isolated,
    prompt_mode_problems,
)
from aadistill.evaluation.behavior import split_generation, template_think_state
from aadistill.training.train import Trainer, nonpad_extent

STOP = (151645,)
HYBRID = ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=50, hidden_size=32, num_hidden_layers=2, intermediate_size=40,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      tie_word_embeddings=True, max_position_embeddings=128)
    return Qwen3ForCausalLM(cfg).double().eval()


def docs():
    g = torch.Generator().manual_seed(1)
    out = []
    for i, (p, o) in enumerate([(5, 4), (3, 6), (7, 2), (4, 4), (6, 9)]):
        ids = torch.randint(1, 50, (p + o,), generator=g).tolist()
        out.append(TokenDocument(f"d{i}", tuple(ids[:p]), tuple(ids[p:])))
    return out


def test_document_from_turn_refuses_what_it_cannot_supervise():
    ok = {"prompt_token_ids": [1, 2], "output_token_ids": [3, 151645], "stop_reason": "eos"}
    d = document_from_turn(ok, doc_id="t", stop_ids=STOP)
    assert d.target_mask == (False, False, True, True)
    with pytest.raises(DocumentError, match="stop id"):
        document_from_turn({**ok, "output_token_ids": [3, 4]}, doc_id="t", stop_ids=STOP)
    with pytest.raises(DocumentError, match="carries no"):
        document_from_turn({**ok, "prompt_token_ids": None}, doc_id="t", stop_ids=STOP)
    with pytest.raises(DocumentError, match="empty prompt"):
        TokenDocument("x", (), (1,))


def test_packing_keeps_documents_whole_restarts_positions_and_pads_a_suffix():
    ds = docs() + [TokenDocument("long", tuple(range(1, 20)), tuple(range(1, 10)))]
    packed = pack_isolated(ds, 16, pad_id=0, order_seed=3)
    assert [r["doc_id"] for r in packed.refused] == ["long"]
    placed = sorted(m["doc_id"] for ms in packed.members for m in ms)
    assert placed == sorted(d.doc_id for d in ds[:-1])
    by_id = {d.doc_id: d for d in ds}
    for b, ms in enumerate(packed.members):
        for m in ms:
            s, k = m["start"], m["length"]
            assert packed.input_ids[b, s:s + k].tolist() == list(by_id[m["doc_id"]].ids)
            assert packed.position_ids[b, s:s + k].tolist() == list(range(k))
            assert packed.target_mask[b, s:s + k].tolist() == list(by_id[m["doc_id"]].target_mask)
            assert not packed.target_mask[b, s]          # a document starts with context
        nonpad_extent(packed.content_mask[b:b + 1])      # padding is a contiguous suffix
    st = packed.stats()
    assert st["supervised_tokens"] == sum(len(d.output_ids) for d in ds[:-1])
    assert pack_isolated(ds, 16, 0, order_seed=3).input_ids.equal(packed.input_ids)


def test_isolated_logits_equal_each_document_alone_and_the_unisolated_path_leaks():
    m = tiny()
    ds = docs()
    packed = pack_isolated(ds, 24, pad_id=0)
    multi = [b for b, ms in enumerate(packed.members) if len(ms) > 1]
    assert multi
    rep = isolation_report(m, packed, {d.doc_id: d for d in ds}, sequences=range(len(packed.members)))
    assert rep["max_abs_diff"] < 1e-10
    b = multi[0]
    second = packed.members[b][1]
    leaked = m(packed.input_ids[b:b + 1], use_cache=False).logits[0]
    alone = m(torch.tensor([ds[int(second["doc_id"][1:])].ids]), use_cache=False).logits[0]
    s, k = second["start"], second["length"]
    assert (leaked[s:s + k - 1] - alone[:k - 1]).abs().max() > 1e-6


def _cfg(tmp_path, iso):
    return {
        "stage": "stage3_recovery", "run_name": "iso", "student_path": "unused", "teacher": None,
        "data_dir": "unused", "groups": None, "block_len": 24, "dtype": "float32", "device": "cpu",
        "seed": 3, "trainable_patterns": "all",
        "loss": {"ce_weight": 1.0, "kd_weight": 1.0, "kd_temperature": 1.0, "kd_scope": "assistant"},
        "optim": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 2, "warmup_steps": 1, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 1, "micro_blocks": 1, "document_isolation": iso},
        "checkpoint": {"save_every": 0, "keep_last": 1},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(tmp_path / "run"),
    }


def test_trainer_losses_under_isolation_are_the_per_document_sums(tmp_path):
    ds = docs()
    packed = pack_isolated(ds, 24, pad_id=0)
    tr = Trainer(_cfg(tmp_path, True), tiny(1), packed.blocks(), teacher=tiny(2))
    b = max(range(len(packed.members)), key=lambda i: len(packed.members[i]))
    ce, ce_n, kd, kd_n = tr._micro_losses(packed.input_ids[b:b + 1], packed.target_mask[b:b + 1],
                                          packed.content_mask[b:b + 1], packed.position_ids[b:b + 1])
    exp_ce = exp_kd = 0.0
    n = 0
    for mem in packed.members[b]:
        d = next(x for x in ds if x.doc_id == mem["doc_id"])
        ids = torch.tensor([d.ids])
        mask = torch.tensor([d.target_mask])
        c, cn, k, kn = tr._micro_losses(ids, mask, torch.ones_like(mask), torch.arange(len(d)).unsqueeze(0))
        exp_ce += float(c)
        exp_kd += float(k)
        n += cn
    assert ce_n == kd_n == n == int(packed.target_mask[b].sum())
    # the losses are reduced in float32 by design (training.train), hence the tolerance
    assert abs(float(ce) - exp_ce) < 1e-5 * max(1.0, abs(exp_ce))
    assert abs(float(kd) - exp_kd) < 1e-5 * max(1.0, abs(exp_kd))


def test_isolation_flag_and_position_ids_must_agree(tmp_path):
    packed = pack_isolated(docs(), 24, pad_id=0)
    teacher = tiny(2)
    with pytest.raises(ValueError, match="document_isolation"):
        Trainer(_cfg(tmp_path, False), tiny(), packed.blocks(), teacher=teacher)
    with pytest.raises(ValueError, match="document_isolation"):
        Trainer(_cfg(tmp_path, True), tiny(), packed.blocks()[:3], teacher=teacher)
    with pytest.raises(ValueError, match="content mask"):
        Trainer(_cfg(tmp_path, True), tiny(), (packed.input_ids, packed.target_mask, None,
                                               packed.position_ids), teacher=teacher)


@pytest.fixture(scope="module")
def hybrid():
    try:
        return AutoTokenizer.from_pretrained(HYBRID[0], revision=HYBRID[1], local_files_only=True)
    except Exception:
        pytest.skip("hybrid tokenizer not in the local HF cache")


MSGS = [{"role": "system", "content": "You are a helpful Assistant."},
        {"role": "user", "content": "What is 2+3?"}]


def test_hybrid_generation_suffix_and_think_state_per_mode(hybrid):
    think = generation_suffix(hybrid, {"enable_thinking": True})
    plain = generation_suffix(hybrid, {"enable_thinking": False})
    assert hybrid.decode(list(think)) == "<|im_start|>assistant\n"
    assert hybrid.decode(list(plain)) == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    for kw, state in (({"enable_thinking": True}, "self_opening"), ({"enable_thinking": False}, "preclosed")):
        prompt = hybrid.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True, **kw)
        assert template_think_state(prompt) == state
    # a document generated under one mode is refused under the other
    p_ids = hybrid(hybrid.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False), add_special_tokens=False)["input_ids"]
    d = TokenDocument("n", tuple(p_ids), (20, 151645))
    assert prompt_mode_problems(d, plain) == []
    assert prompt_mode_problems(d, think)


def test_hybrid_rerendering_is_not_prompt_plus_generation(hybrid):
    """Why documents are taken token-in/token-out: the template rewrites a finished turn."""
    enc = lambda s: hybrid(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    for gen, kw in (("<think>\n\nwork\n\n</think>\n\n5", {"enable_thinking": True}),
                    ("\n5", {"enable_thinking": False})):
        prompt = enc(hybrid.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True, **kw))
        native = prompt + enc(gen + "<|im_end|>")
        rerendered = enc(hybrid.apply_chat_template(MSGS + [{"role": "assistant", "content": gen}],
                                                    tokenize=False, **kw))
        assert rerendered[:len(native)] != native


def test_hybrid_history_drops_earlier_reasoning(hybrid):
    hist = MSGS + [{"role": "assistant", "content": "5", "reasoning_content": "add"},
                   {"role": "user", "content": "And 5+5?"}]
    for kw in ({"enable_thinking": True}, {"enable_thinking": False}):
        prompt = hybrid.apply_chat_template(hist, tokenize=False, add_generation_prompt=True, **kw)
        assert "add" not in prompt and "<|im_start|>assistant\n5<|im_end|>" in prompt


def test_preclosed_split_reads_the_whole_body_as_the_answer():
    ok = split_generation("The answer is 5.<|im_end|>", think_state="preclosed")
    assert ok["think_closed"] and ok["answer"] == "The answer is 5." and ok["no_stray_markers"]
    for bad in ("<think>x</think>5<|im_end|>", "x</think>5<|im_end|>"):
        parts = split_generation(bad, think_state="preclosed")
        assert not (parts["think_closed"] and parts["no_stray_markers"])
    # the two historical states are unchanged by the new argument
    for state, raw in (("preopened", "r</think>\n\n5<|im_end|>"), ("self_opening", "<think>r</think>5<|im_end|>")):
        assert split_generation(raw, think_state=state) == split_generation(
            raw, think_preopened=state == "preopened")

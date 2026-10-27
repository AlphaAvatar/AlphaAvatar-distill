"""Token-in/token-out documents, both context strategies, and the trainer path that uses them.

``document_isolated``: every logit a target is trained against equals the logit
of its document run alone, for the student and the teacher alike, so KD
compares the two models on the context the teacher generated in.
``cross_context``: the documents share one causal context, so a later
document's logits are conditioned on the earlier ones -- a different training
distribution, not an error -- and teacher and student see the same composed
sequence. Both are asserted, and the trainer refuses a config that does not say
which one its blocks were packed for.
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
    context_exposure,
    isolation_report,
    message_terminator,
    pack_cross_context,
    pack_documents,
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


def test_isolated_logits_equal_each_document_alone_and_cross_context_conditions_on_neighbours():
    m = tiny()
    ds = docs()
    packed = pack_isolated(ds, 24, pad_id=0)
    multi = [b for b, ms in enumerate(packed.members) if len(ms) > 1]
    assert multi
    rep = isolation_report(m, packed, {d.doc_id: d for d in ds}, sequences=range(len(packed.members)))
    assert rep["max_abs_diff"] < 1e-10
    b = multi[0]
    second = packed.members[b][1]
    # the same sequence run with continuous positions is the cross-context distribution
    composed = m(packed.input_ids[b:b + 1], use_cache=False).logits[0]
    alone = m(torch.tensor([ds[int(second["doc_id"][1:])].ids]), use_cache=False).logits[0]
    s, k = second["start"], second["length"]
    assert (composed[s:s + k - 1] - alone[:k - 1]).abs().max() > 1e-6


def _cfg(tmp_path, strategy):
    return {
        "stage": "stage3_recovery", "run_name": "iso", "student_path": "unused", "teacher": None,
        "data_dir": "unused", "groups": None, "block_len": 24, "dtype": "float32", "device": "cpu",
        "seed": 3, "trainable_patterns": "all",
        "loss": {"ce_weight": 1.0, "kd_weight": 1.0, "kd_temperature": 1.0, "kd_scope": "assistant"},
        "optim": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 2, "warmup_steps": 1, "min_lr_frac": 0.1},
        "batch": {"blocks_per_step": 1, "micro_blocks": 1, "context_strategy": strategy},
        "checkpoint": {"save_every": 0, "keep_last": 1},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(tmp_path / "run"),
    }


def test_trainer_losses_under_isolation_are_the_per_document_sums(tmp_path):
    ds = docs()
    packed = pack_isolated(ds, 24, pad_id=0)
    tr = Trainer(_cfg(tmp_path, "document_isolated"), tiny(1), packed.blocks(), teacher=tiny(2))
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


def test_context_strategy_and_position_ids_must_agree(tmp_path):
    packed = pack_isolated(docs(), 24, pad_id=0)
    teacher = tiny(2)
    with pytest.raises(ValueError, match="context_strategy"):
        Trainer(_cfg(tmp_path, "cross_context"), tiny(), packed.blocks(), teacher=teacher)
    with pytest.raises(ValueError, match="context_strategy"):
        Trainer(_cfg(tmp_path, "document_isolated"), tiny(), packed.blocks()[:3], teacher=teacher)
    with pytest.raises(ValueError, match="content mask"):
        Trainer(_cfg(tmp_path, "document_isolated"), tiny(), (packed.input_ids, packed.target_mask, None,
                                                            packed.position_ids), teacher=teacher)
    with pytest.raises(ValueError, match="context_strategy must be one of"):
        Trainer(_cfg(tmp_path, "isolated"), tiny(), packed.blocks(), teacher=teacher)


def _prefixed_docs():
    """Documents sharing one 3-token 'system block', as a template renders them."""
    g = torch.Generator().manual_seed(4)
    prefix = (7, 8, 9)
    out = []
    for i, (p, o) in enumerate([(3, 4), (2, 6), (4, 2), (3, 3), (5, 7)]):
        ids = torch.randint(10, 50, (p + o,), generator=g).tolist()
        out.append(TokenDocument(f"c{i}", prefix + tuple(ids[:p]), tuple(ids[p:])))
    return prefix, out


def test_cross_context_composes_once_prefixed_bodies_and_records_the_composition():
    prefix, ds = _prefixed_docs()
    sep = (6,)
    packed = pack_cross_context(ds, 24, pad_id=0, shared_prefix=prefix, separator_ids=sep, order_seed=5)
    by_id = {d.doc_id: d for d in ds}
    assert sorted(m["doc_id"] for ms in packed.members for m in ms) == sorted(by_id)
    for b, ms in enumerate(packed.members):
        row = packed.input_ids[b].tolist()
        assert tuple(row[:3]) == prefix                          # the shared block once, first
        at = 3
        for j, m in enumerate(ms):
            d = by_id[m["doc_id"]]
            if j:
                assert tuple(row[at:at + 1]) == sep             # the separator is context
                assert not packed.target_mask[b, at]
                at += 1
            assert m["start"] == at and m["foreign_context"] == at - 3
            body = d.ids[3:]
            assert tuple(row[at:at + len(body)]) == body
            assert packed.target_mask[b, at:at + len(body)].tolist() == list(d.target_mask[3:])
            at += len(body)
        nonpad_extent(packed.content_mask[b:b + 1])
        assert packed.position_ids[b].tolist() == list(range(24))
    assert len(packed.blocks()) == 3                             # the trainer's default positions
    assert packed.composition() == [[m["doc_id"] for m in ms] for ms in packed.members]
    assert packed.stats()["supervised_tokens"] == sum(len(d.output_ids) for d in ds)
    again = pack_documents(ds, 24, 0, strategy="cross_context", shared_prefix=prefix,
                           separator_ids=sep, order_seed=5)
    assert again.input_ids.equal(packed.input_ids) and again.composition() == packed.composition()
    with pytest.raises(DocumentError, match="shared prefix"):
        pack_cross_context(ds + [TokenDocument("x", (1, 2, 3, 4), (5,))], 24, 0, shared_prefix=prefix)
    with pytest.raises(ValueError, match="unknown context strategy"):
        pack_documents(ds, 24, 0, strategy="sessions")


def test_cross_context_first_document_is_its_own_context_and_later_ones_are_conditioned():
    m = tiny()
    prefix, ds = _prefixed_docs()
    packed = pack_cross_context(ds, 24, pad_id=0, shared_prefix=prefix, separator_ids=(6,))
    b = max(range(len(packed.members)), key=lambda i: len(packed.members[i]))
    first, second = packed.members[b][0], packed.members[b][1]
    out = m(packed.input_ids[b:b + 1], use_cache=False).logits[0]
    for mem, conditioned in ((first, False), (second, True)):
        d = next(x for x in ds if x.doc_id == mem["doc_id"])
        alone = m(torch.tensor([d.ids]), use_cache=False).logits[0]
        s, k = mem["start"], mem["length"]
        diff = (out[s:s + k - 1] - alone[3:3 + k - 1]).abs().max()
        assert (diff > 1e-6) == conditioned


def test_trainer_runs_cross_context_blocks_with_teacher_and_student_on_one_context(tmp_path):
    prefix, ds = _prefixed_docs()
    packed = pack_cross_context(ds, 24, pad_id=0, shared_prefix=prefix, separator_ids=(6,))
    tr = Trainer(_cfg(tmp_path, "cross_context"), tiny(1), packed.blocks(), teacher=tiny(2))
    b = 0
    ce, ce_n, kd, kd_n = tr._micro_losses(packed.input_ids[b:b + 1], packed.target_mask[b:b + 1],
                                          packed.content_mask[b:b + 1], None)
    assert ce_n == kd_n == int(packed.target_mask[b, 1:].sum())
    # KD at T=1 is zero exactly when student and teacher are one model on one context
    same = Trainer(_cfg(tmp_path / "same", "cross_context"), tiny(1), packed.blocks(), teacher=tiny(1))
    _, _, kd_same, _ = same._micro_losses(packed.input_ids[b:b + 1], packed.target_mask[b:b + 1],
                                          packed.content_mask[b:b + 1], None)
    assert abs(float(kd_same)) < 1e-9
    assert tr.context_strategy == "cross_context"


def test_context_exposure_is_zero_isolated_and_grows_with_shorter_documents():
    prefix, ds = _prefixed_docs()
    iso = context_exposure(pack_isolated(ds, 40, pad_id=0))
    assert iso["share_of_supervised_tokens_with_foreign_context"] == 0.0
    long_ = context_exposure(pack_cross_context(ds, 40, 0, shared_prefix=prefix, separator_ids=(6,)))
    short = [TokenDocument(d.doc_id, d.prompt_ids, d.output_ids[:1]) for d in ds]
    short_ = context_exposure(pack_cross_context(short, 40, 0, shared_prefix=prefix, separator_ids=(6,)))
    assert short_["documents_per_sequence_mean"] > long_["documents_per_sequence_mean"]
    assert short_["share_of_supervised_tokens_with_foreign_context"] > 0


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


def test_hybrid_cross_context_composition_reads_as_the_template_writes_turns(hybrid):
    """Composed with the template's own message terminator, two documents join as
    rendered consecutive turns do: ``<|im_end|>\\n<|im_start|>user``."""
    sep = message_terminator(hybrid, hybrid.convert_tokens_to_ids("<|im_end|>"), {"enable_thinking": False})
    assert hybrid.decode(list(sep)) == "\n"
    enc = lambda s: hybrid(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    system = enc(hybrid.apply_chat_template(MSGS[:1], tokenize=False))
    ds = []
    for i, q in enumerate(("What is 2+3?", "And 5+5?")):
        prompt = enc(hybrid.apply_chat_template([MSGS[0], {"role": "user", "content": q}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False))
        ds.append(TokenDocument(f"q{i}", tuple(prompt), tuple(enc("5<|im_end|>"))))
    packed = pack_cross_context(ds, 128, pad_id=0, shared_prefix=system, separator_ids=sep)
    text = hybrid.decode(packed.input_ids[0][packed.content_mask[0]].tolist())
    assert text.count("<|im_start|>system") == 1
    assert "5<|im_end|>\n<|im_start|>user\n" in text


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

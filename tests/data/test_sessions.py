"""Tests for session rendering and system-prompt-aware packing.

Split deliberately in two. The packing invariants (§4-§8 of the packing spec) are
exercised with synthetic sessions and **no tokenizer**, so they run everywhere
and a missing HF cache can never quietly skip the checks that protect the
training data. The rendering tests need the pinned teacher's chat template and
skip when it is absent, following `test_dataset.py`.

The regression test that matters most here is
`test_packed_render_preserves_every_trace`: applying the official template to a
multi-session message list silently deletes the `<think>` block of every
assistant turn except the last, which would destroy most of the supervision in
every packed block without raising anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.sessions import (
    SYSTEM_DEFAULT,
    PackedBlock,
    RenderedSession,
    completion_budget,
    pack_group,
    pack_sessions,
    render_session,
    render_system_block,
    split_system,
    system_group_key,
)

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

PAD = 151643
N_SYS = 6  # stand-in system-block length for the synthetic tests
BLOCK = 64


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            TEACHER, revision=REVISION, local_files_only=True)
    except Exception:
        pytest.skip("teacher tokenizer not in local HF cache")


# --------------------------------------------------------------------------
# synthetic sessions: packing algebra, no tokenizer
# --------------------------------------------------------------------------

def fake_session(sid, n_prompt, n_answer, *, key="K", dtype="t", vocab=None):
    """A session whose body is `n_prompt` unsupervised then `n_answer` supervised."""
    total = n_prompt + n_answer
    base = (abs(hash(sid)) % 97) + 3
    ids = [base + i for i in range(total)]
    if vocab is not None:
        ids = [3 + (v % (vocab - 3)) for v in ids]
    return RenderedSession(
        session_id=sid, data_type=dtype, system_text="S", system_key=key,
        body_ids=ids, body_mask=[False] * n_prompt + [True] * n_answer,
        n_system_tokens=N_SYS,
    )


def sys_ids(n=N_SYS, vocab=None):
    ids = [900 + i for i in range(n)]
    return [3 + (v % (vocab - 3)) for v in ids] if vocab else ids


def test_every_block_is_exactly_block_len_and_starts_with_the_system_block():
    sessions = [fake_session(f"s{i}", 3, 7) for i in range(20)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    assert blocks
    for b in blocks:
        assert len(b.input_ids) == BLOCK
        assert len(b.ce_mask) == BLOCK and len(b.content_mask) == BLOCK
        assert b.input_ids[:N_SYS] == sys_ids()
        assert b.audit["unpadded_length"] <= BLOCK
        assert b.audit["final_length"] == BLOCK
        # The system block is never supervised.
        assert not any(b.ce_mask[:N_SYS])


def test_padding_is_excluded_from_every_mask_and_from_accounting():
    sessions = [fake_session(f"s{i}", 3, 7) for i in range(7)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    padded = [b for b in blocks if b.audit["padding_length"] > 0]
    assert padded, "expected at least one padded block"
    for b in padded:
        n = b.audit["unpadded_length"]
        assert all(v == PAD for v in b.input_ids[n:])
        assert not any(b.ce_mask[n:])
        assert not any(b.content_mask[n:])
        assert all(b.content_mask[:n])
        assert b.n_supervised == sum(b.ce_mask[:n])


def test_supervised_count_is_the_authoritative_post_packing_number():
    sessions = [fake_session(f"s{i}", 3, 7) for i in range(20)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        assert b.audit["supervised_tokens"] == b.n_supervised == int(sum(b.ce_mask))
        # And it equals the sum of what each member says it retained.
        assert b.n_supervised == sum(m["supervised_retained"] for m in b.audit["sessions"])


def test_only_the_final_session_of_a_block_may_be_truncated():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    saw_truncation = False
    for b in blocks:
        members = b.audit["sessions"]
        for m in members[:-1]:
            assert not m["truncated"], "a non-terminal session was truncated"
            assert m["supervised_discarded"] == 0
        if members[-1]["truncated"]:
            saw_truncation = True
            assert b.audit["terminal_truncated"] is True
            assert b.audit["terminal_truncation"]["session_id"] == members[-1]["session_id"]
    assert saw_truncation, "test data did not exercise terminal truncation"


def test_a_truncated_suffix_is_never_repacked_into_a_later_block():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    seen: list[str] = []
    for b in blocks:
        seen.extend(m["session_id"] for m in b.audit["sessions"])
    assert len(seen) == len(set(seen)), "a session appears in more than one block"


def test_terminal_session_is_not_appended_when_it_would_retain_no_supervision():
    # Room for the whole of s0 then 3 spare tokens; s1's body opens with 8
    # unsupervised prompt tokens, so appending it would add prompt and no target.
    sessions = [fake_session("s0", 2, BLOCK - N_SYS - 2 - 3), fake_session("s1", 8, 8)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    assert [m["session_id"] for m in blocks[0].audit["sessions"]] == ["s0"]
    assert blocks[0].audit["padding_length"] == 3
    assert blocks[0].audit["terminal_truncated"] is False
    # s1 opens the next block intact.
    assert [m["session_id"] for m in blocks[1].audit["sessions"]] == ["s1"]
    assert blocks[1].audit["sessions"][0]["truncated"] is False


def test_no_synthetic_terminal_token_is_added_at_the_cut():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        cut = b.audit["terminal_truncation"]
        if not cut:
            continue
        member = b.audit["sessions"][-1]
        original = fake_session(member["session_id"], 4, 9)
        kept = cut["kept_body_tokens"]
        # The retained region is a byte-exact prefix of the original body: no
        # EOS, no closing tag, nothing manufactured at the boundary.
        assert b.input_ids[member["start"]:member["end"]] == original.body_ids[:kept]
        assert b.ce_mask[member["start"]:member["end"]] == original.body_mask[:kept]


def test_truncation_accounting_splits_retained_and_discarded_exactly():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        for m in b.audit["sessions"]:
            original = fake_session(m["session_id"], 4, 9)
            assert m["supervised_retained"] + m["supervised_discarded"] == original.n_supervised


def test_groups_are_never_mixed_and_each_block_carries_its_own_system():
    a = [fake_session(f"a{i}", 3, 7, key="KA") for i in range(6)]
    b = [fake_session(f"b{i}", 3, 7, key="KB") for i in range(6)]
    system_by_key = {"KA": sys_ids(), "KB": [800 + i for i in range(N_SYS)]}
    blocks = pack_sessions(a + b, system_by_key, block_len=BLOCK, pad_id=PAD)
    for block in blocks:
        prefixes = {sid[0] for sid in block.audit["session_ids"]}
        assert len(prefixes) == 1, "a block mixed two system-prompt groups"
        expected = system_by_key["KA" if prefixes == {"a"} else "KB"]
        assert block.input_ids[:N_SYS] == expected


def test_packing_is_deterministic():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(25)]
    first = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    second = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    assert [b.input_ids for b in first] == [b.input_ids for b in second]
    assert [b.ce_mask for b in first] == [b.ce_mask for b in second]
    assert [b.audit for b in first] == [b.audit for b in second]


def test_blocks_are_prefix_nested_as_sessions_are_added():
    """The property the token ladder depends on (§10)."""
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(40)]
    short = pack_group(sessions[:15], sys_ids(), block_len=BLOCK, pad_id=PAD)
    long = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    # Every block of the short pack that is full (i.e. not the trailing partial
    # one) must appear unchanged at the same index in the long pack.
    complete = [b for b in short if b.audit["padding_length"] == 0]
    assert len(complete) >= 2
    for i, block in enumerate(complete):
        assert long[i].input_ids == block.input_ids
        assert long[i].ce_mask == block.ce_mask
        assert long[i].audit == block.audit


def test_audit_metadata_carries_every_required_field():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    required_block = {
        "system_sha256", "n_system_tokens", "sessions", "session_ids",
        "supervision_spans", "unpadded_length", "padding_length", "final_length",
        "terminal_truncated", "terminal_truncation", "supervised_tokens",
    }
    required_member = {
        "session_id", "data_type", "candidate_index", "candidate_sha256", "start",
        "end", "original_rendered_tokens", "original_body_tokens", "truncated",
        "supervised_retained", "supervised_discarded",
    }
    required_cut = {
        "session_id", "original_body_tokens", "packed_start_offset",
        "truncation_offset", "kept_body_tokens", "supervised_retained",
        "supervised_discarded", "cut_boundary_kind", "cut_token_id",
    }
    saw_cut = False
    for b in blocks:
        assert required_block <= set(b.audit)
        for m in b.audit["sessions"]:
            assert required_member <= set(m)
        if b.audit["terminal_truncation"]:
            saw_cut = True
            assert required_cut <= set(b.audit["terminal_truncation"])
    assert saw_cut


def test_supervision_spans_match_the_ce_mask():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        rebuilt = [False] * len(b.ce_mask)
        for start, end in b.audit["supervision_spans"]:
            for i in range(start, end):
                rebuilt[i] = True
        assert rebuilt == list(b.ce_mask)


def test_session_boundaries_reconstruct_the_block():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        cursor = N_SYS
        for m in b.audit["sessions"]:
            assert m["start"] == cursor
            cursor = m["end"]
        assert cursor == b.audit["unpadded_length"]


def test_serialization_roundtrip_reproduces_ids_masks_and_audit():
    sessions = [fake_session(f"s{i}", 4, 9) for i in range(20)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    wire = json.dumps([{
        "input_ids": b.input_ids,
        "ce_mask": [int(v) for v in b.ce_mask],
        "content_mask": [int(v) for v in b.content_mask],
        "audit": b.audit,
    } for b in blocks])
    back = [PackedBlock(r["input_ids"], [bool(v) for v in r["ce_mask"]],
                        [bool(v) for v in r["content_mask"]], r["audit"])
            for r in json.loads(wire)]
    assert len(back) == len(blocks)
    for a, b in zip(blocks, back):
        assert a.input_ids == b.input_ids
        assert list(a.ce_mask) == list(b.ce_mask)
        assert list(a.content_mask) == list(b.content_mask)
        assert a.audit == b.audit
        assert a.n_supervised == b.n_supervised


# --------------------------------------------------------------------------
# §9: an untruncated prefix and the terminally truncated block must agree
# --------------------------------------------------------------------------

def _block_with_truncation(vocab=None):
    sessions = [fake_session(f"s{i}", 4, 9, vocab=vocab) for i in range(30)]
    blocks = pack_group(sessions, sys_ids(vocab=vocab), block_len=BLOCK, pad_id=PAD)
    for b in blocks:
        if b.audit["terminal_truncated"]:
            return sessions, b
    raise AssertionError("no truncated block produced")


def test_truncation_leaves_every_earlier_token_and_mask_identical():
    sessions, block = _block_with_truncation()
    by_id = {s.session_id: s for s in sessions}
    # Rebuild the same content with the terminal session left whole.
    full_ids, full_mask = list(sys_ids()), [False] * N_SYS
    for m in block.audit["sessions"]:
        s = by_id[m["session_id"]]
        full_ids.extend(s.body_ids)
        full_mask.extend(s.body_mask)

    n = block.audit["unpadded_length"]
    assert block.input_ids[:n] == full_ids[:n]
    assert list(block.ce_mask[:n]) == full_mask[:n]
    # And the discarded suffix is strictly beyond the retained region.
    cut = block.audit["terminal_truncation"]
    assert cut["truncation_offset"] == n
    assert cut["supervised_discarded"] == sum(full_mask[n:])


def test_truncation_does_not_change_causal_logits_at_retained_positions():
    """The causal argument made empirical, per §9."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    vocab = 64
    sessions, block = _block_with_truncation(vocab=vocab)
    by_id = {s.session_id: s for s in sessions}
    full_ids = list(sys_ids(vocab=vocab))
    for m in block.audit["sessions"]:
        full_ids.extend(by_id[m["session_id"]].body_ids)

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=2,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=256,
        eos_token_id=2, pad_token_id=PAD % vocab,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()

    n = block.audit["unpadded_length"]
    with torch.no_grad():
        packed = model(torch.tensor([block.input_ids])).logits[0, :n]
        untruncated = model(torch.tensor([full_ids])).logits[0, :n]
    assert torch.allclose(packed, untruncated, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# rendering against the real template
# --------------------------------------------------------------------------

def session(sid, q, r, c, system=None, dtype="rag_evidence"):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages += [
        {"role": "user", "content": q},
        {"role": "assistant", "reasoning_content": r, "content": c},
    ]
    return {"id": sid, "data_type": dtype, "messages": messages}


def test_split_system_preserves_a_source_prompt_and_defaults_otherwise():
    text, body = split_system([{"role": "system", "content": "CUSTOM"},
                               {"role": "user", "content": "q"}])
    assert text == "CUSTOM" and [m["role"] for m in body] == ["user"]
    text, body = split_system([{"role": "user", "content": "q"}])
    assert text == SYSTEM_DEFAULT and [m["role"] for m in body] == ["user"]


def test_system_group_key_separates_different_system_prompts():
    assert system_group_key("A", None) == system_group_key("A", None)
    assert system_group_key("A", None) != system_group_key("B", None)
    # Tools land inside the rendered system block, so they are part of the key.
    assert system_group_key("A", None) != system_group_key("A", [{"x": 1}])


def test_render_session_round_trips_through_the_official_template(tokenizer):
    r = render_session(tokenizer, session("s1", "Q", "REASONING", "ANSWER"))
    assert r.system_text == SYSTEM_DEFAULT
    assert r.n_rendered_tokens == r.n_system_tokens + r.n_body_tokens
    assert r.n_supervised > 0
    text = tokenizer.decode(r.body_ids)
    assert "REASONING" in text and "ANSWER" in text
    assert "<|im_start|>system" not in text  # the system block is not in the body


def test_packed_render_preserves_every_trace(tokenizer):
    """Regression test for the template's dropped-reasoning behaviour.

    Applying the template to `[system, u1, a1, u2, a2, u3, a3]` keeps only a3's
    `<think>` block. Per-session rendering plus token concatenation keeps all
    three, which is what the packer does.
    """
    traces = ["TRACE-ALPHA", "TRACE-BETA", "TRACE-GAMMA"]
    rendered = [render_session(tokenizer, session(f"s{i}", f"Q{i}", t, f"A{i}"))
                for i, t in enumerate(traces)]
    key = rendered[0].system_key
    system = tokenizer(render_system_block(tokenizer, SYSTEM_DEFAULT),
                       add_special_tokens=False).input_ids
    blocks = pack_sessions(rendered, {key: system}, block_len=8192, pad_id=PAD)
    assert len(blocks) == 1
    n = blocks[0].audit["unpadded_length"]
    text = tokenizer.decode(blocks[0].input_ids[:n])

    for t in traces:
        assert t in text, f"{t} was dropped from the packed render"
    assert text.count("<|im_start|>system") == 1
    assert text.startswith("<|im_start|>system")
    assert text.count("<think>") == len(traces)

    # The naive approach the spec's wording suggests loses all but the last.
    naive = [{"role": "system", "content": SYSTEM_DEFAULT}]
    for i, t in enumerate(traces):
        naive += [{"role": "user", "content": f"Q{i}"},
                  {"role": "assistant", "reasoning_content": t, "content": f"A{i}"}]
    naive_text = tokenizer.apply_chat_template(naive, tokenize=False,
                                               add_generation_prompt=False)
    assert sum(t in naive_text for t in traces) == 1


def test_packed_supervision_covers_only_assistant_content(tokenizer):
    rendered = [render_session(tokenizer, session(f"s{i}", f"Question {i}",
                                                 f"Reasoning {i}", f"Answer {i}"))
                for i in range(3)]
    key = rendered[0].system_key
    system = tokenizer(render_system_block(tokenizer, SYSTEM_DEFAULT),
                       add_special_tokens=False).input_ids
    block = pack_sessions(rendered, {key: system}, block_len=8192, pad_id=PAD)[0]
    supervised = [i for i, v in enumerate(block.ce_mask) if v]
    text = tokenizer.decode([block.input_ids[i] for i in supervised])
    for i in range(3):
        assert f"Answer {i}" in text and f"Reasoning {i}" in text
        assert f"Question {i}" not in text
    assert "You are a helpful Assistant." not in text


def test_completion_budget_shrinks_with_prompt_length(tokenizer):
    short = completion_budget(tokenizer, [{"role": "user", "content": "hi"}])
    long = completion_budget(
        tokenizer, [{"role": "user", "content": "word " * 500}])
    assert short > long
    prompt_len = len(tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_DEFAULT},
             {"role": "user", "content": "hi"}],
            tokenize=False, add_generation_prompt=True),
        add_special_tokens=False).input_ids)
    assert short == 8192 - prompt_len - 8


def test_render_session_rejects_a_session_over_the_limit(tokenizer):
    big = session("s-big", "q", "R " * 200, "a")
    with pytest.raises(ValueError, match="render_overflow"):
        render_session(tokenizer, big, block_len=64)


def test_source_system_prompt_is_preserved_in_the_render(tokenizer):
    r = render_session(tokenizer, session("s1", "Q", "R", "A", system="CUSTOM SYS"))
    assert r.system_text == "CUSTOM SYS"
    block = render_system_block(tokenizer, "CUSTOM SYS")
    assert "CUSTOM SYS" in block
    assert r.system_key == system_group_key("CUSTOM SYS", None)


# --------------------------------------------------------------------------
# turn expansion: same-source examples must never co-occur in a block
# --------------------------------------------------------------------------

def fake_expanded(source, turn, n_prompt, n_answer, *, key="K"):
    s = fake_session(f"{source}#t{turn}", n_prompt, n_answer, key=key)
    s.source_id = source
    return s


def test_same_source_examples_never_share_a_block():
    """Turn-expanded siblings are prefixes of one another — co-packing leaks."""
    sessions = []
    for conv in range(6):
        for turn in (1, 3, 5):
            sessions.append(fake_expanded(f"c{conv}", turn, 3, 7))
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    assert blocks
    for b in blocks:
        sources = [m["session_id"].split("#")[0] for m in b.audit["sessions"]]
        assert len(sources) == len(set(sources)), (
            f"block packed two examples from {sources}")
    # Nothing is lost: every example still lands in exactly one block.
    placed = [m["session_id"] for b in blocks for m in b.audit["sessions"]]
    assert sorted(placed) == sorted(s.session_id for s in sessions)


def test_same_source_collisions_are_deferred_not_dropped():
    sessions = [fake_expanded("c0", t, 3, 7) for t in (1, 2, 3)]
    blocks = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    # Three siblings cannot share a block, so they occupy three blocks even
    # though all three would fit in one by length.
    assert len(blocks) == 3
    assert sum(b.audit["deferred_same_source"] for b in blocks) > 0


def test_deferral_preserves_prefix_nesting():
    sessions = []
    for conv in range(12):
        for turn in (1, 2):
            sessions.append(fake_expanded(f"c{conv}", turn, 4, 9))
    short = pack_group(sessions[:12], sys_ids(), block_len=BLOCK, pad_id=PAD)
    long = pack_group(sessions, sys_ids(), block_len=BLOCK, pad_id=PAD)
    complete = [b for b in short if b.audit["padding_length"] == 0]
    assert complete
    for i, block in enumerate(complete):
        assert long[i].input_ids == block.input_ids
        assert long[i].audit["sessions"] == block.audit["sessions"]

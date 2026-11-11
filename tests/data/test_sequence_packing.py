"""Random shuffle -> sequential packing -> truncation, against the reference.

The maintainer's `packing_conversation` is the definition. These tests hold
`sequential_pack` to its index lists exactly, pin the token materialization the
module states (every placement a prefix; a fragment restarts in the next
sequence), and check the accounting of repeated, truncated and padding tokens
closes. Then the Stage-3 packer: the tokens in each sequence are the documents'
prefixes at the recorded offsets, positions are continuous, padding is a suffix
of the last sequence only, and the position after a document never supervises
the next document's prompt.
"""
from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.data.sequence_packing import (  # noqa: E402
    PackingError, SequencePackingSpec, packing_conversation, plan_indices,
    plan_report, sequential_pack, shuffled_order,
)
from aadistill.data.token_documents import (  # noqa: E402
    SEQUENTIAL_CAUSAL, TokenDocument, pack_documents, pack_sequential, write_pack,
)


def _random_lengths(seed: int, n: int, capacity: int) -> list[int]:
    rng = random.Random(seed)
    return [max(1, rng.choice([1, 2, capacity // 3, capacity // 2, capacity - 1, capacity,
                               capacity + 1, 2 * capacity + 3, rng.randint(1, 2 * capacity)]))
            for _ in range(n)]


class TestTheReferenceAlgorithm:

    @pytest.mark.parametrize("seed", range(40))
    def test_index_lists_equal_the_reference(self, seed):
        capacity = random.Random(seed).choice([2, 3, 8, 17, 64])
        lengths = _random_lengths(seed, 30, capacity)
        assert plan_indices(sequential_pack(lengths, capacity)) == \
            packing_conversation(lengths, capacity, True)

    def test_a_worked_example(self):
        # [5, 5, 5] into 8: 0 fits, 1's first 3 tokens fill the rest, 1 restarts, ...
        plan = sequential_pack([5, 5, 5], 8)
        assert [[(p.index, p.tokens, p.final) for p in s] for s in plan] == [
            [(0, 5, True), (1, 3, False)],
            [(1, 5, True), (2, 3, False)],
            [(2, 5, True)]]

    def test_an_example_longer_than_the_capacity(self):
        # mid-sequence: a fragment of the room, then a capacity-truncated restart
        plan = sequential_pack([3, 20, 2], 8)
        assert [[(p.index, p.tokens, p.final) for p in s] for s in plan] == [
            [(0, 3, True), (1, 5, False)],
            [(1, 8, True)],
            [(2, 2, True)]]

    @pytest.mark.parametrize("seed", range(20))
    def test_the_materialization_rule(self, seed):
        capacity = 16
        lengths = _random_lengths(seed, 25, capacity)
        plan = sequential_pack(lengths, capacity)
        finals = {}
        for seq in plan:
            room = capacity
            for k, p in enumerate(seq):
                assert p.tokens == min(lengths[p.index], room)
                if not p.final:
                    assert k == len(seq) - 1, "a fragment is the last placement"
                    assert p.tokens < lengths[p.index]
                room -= lengths[p.index]
            if seq is not plan[-1]:
                assert sum(p.tokens for p in seq) == capacity, "padding only at the end"
            for p in seq:
                if p.final:
                    assert p.index not in finals
                    finals[p.index] = p.tokens
        assert finals == {i: min(n, capacity) for i, n in enumerate(lengths)}

    @pytest.mark.parametrize("seed", range(10))
    def test_the_accounting_closes(self, seed):
        capacity = 12
        lengths = _random_lengths(seed, 20, capacity)
        rng = random.Random(seed)
        masks = [[rng.random() < 0.5 for _ in range(n)] for n in lengths]
        plan = sequential_pack(lengths, capacity)
        r = plan_report(lengths, plan, capacity, supervised=masks)
        assert r["unique_tokens"] - r["truncated_tokens"] + r["repeated_tokens"] == r["placed_tokens"]
        assert r["repeated_tokens"] == sum(p.tokens for s in plan for p in s if not p.final)
        assert r["truncated_tokens"] == sum(max(0, n - capacity) for n in lengths)
        assert r["padding_tokens"] == capacity * len(plan) - r["placed_tokens"]
        assert r["unique_supervised_tokens"] == sum(map(sum, masks))

    def test_refusals(self):
        with pytest.raises(PackingError):
            sequential_pack([3, 0], 8)
        with pytest.raises(PackingError):
            sequential_pack([3], 1)


class TestTheShuffle:

    def test_a_pure_function_of_the_ids_and_the_seed(self):
        ids = [f"doc-{i}" for i in range(50)]
        a = [ids[i] for i in shuffled_order(ids, 7)]
        b = [list(reversed(ids))[i] for i in shuffled_order(list(reversed(ids)), 7)]
        assert a == b and a != ids
        assert a != [ids[i] for i in shuffled_order(ids, 8)]

    def test_removing_an_example_keeps_the_others_order(self):
        ids = [f"doc-{i}" for i in range(30)]
        full = [ids[i] for i in shuffled_order(ids, 3)]
        less = [x for x in ids if x != "doc-11"]
        assert [less[i] for i in shuffled_order(less, 3)] == [x for x in full if x != "doc-11"]

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(PackingError, match="duplicate"):
            shuffled_order(["a", "b", "a"], 1)

    def test_the_spec_names_its_rules(self):
        spec = SequencePackingSpec(sequence_length=8192, shuffle_seed=1)
        d = spec.as_dict()
        assert d["packing"] == "sequential_truncate_v1" and d["truncate"] is True
        assert d["attention"] == "causal_continuous_positions_v1"
        assert spec.spec_hash != SequencePackingSpec(sequence_length=8192, shuffle_seed=2).spec_hash


def _docs(n: int = 9, seed: int = 0) -> list[TokenDocument]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        prompt = tuple(rng.randint(10, 90) for _ in range(rng.randint(2, 6)))
        output = tuple(rng.randint(10, 90) for _ in range(rng.randint(1, 9))) + (5,)
        out.append(TokenDocument(f"d{i:02d}", prompt, output))
    return out


class TestTheStage3Packer:

    def test_tokens_offsets_masks_and_positions(self):
        docs = _docs()
        by_id = {d.doc_id: d for d in docs}
        packed = pack_sequential(docs, 16, pad_id=0, order_seed=11)
        assert packed.strategy == SEQUENTIAL_CAUSAL
        for b, members in enumerate(packed.members):
            at = 0
            for m in members:
                d = by_id[m["doc_id"]]
                assert m["start"] == at and m["foreign_context"] == at
                k = m["length"]
                assert packed.input_ids[b, at:at + k].tolist() == list(d.ids[:k])
                assert packed.target_mask[b, at:at + k].tolist() == list(d.target_mask[:k])
                assert m["final"] == (k == len(d))
                at += k
            assert packed.content_mask[b].tolist() == [True] * at + [False] * (16 - at)
            if b < len(packed.members) - 1:
                assert at == 16, "only the last sequence carries padding"
        assert torch.equal(packed.position_ids, torch.arange(16).expand_as(packed.position_ids))
        # every document's first token is a prompt token, so the position before it
        # never supervises it
        for b, members in enumerate(packed.members):
            for m in members:
                assert not bool(packed.target_mask[b, m["start"]])

    def test_independent_of_the_input_order(self):
        docs = _docs()
        a = pack_sequential(docs, 16, 0, order_seed=3)
        b = pack_sequential(list(reversed(docs)), 16, 0, order_seed=3)
        assert torch.equal(a.input_ids, b.input_ids) and a.members == b.members

    def test_the_statistics_separate_unique_and_placed(self):
        docs = _docs(12, seed=4)
        packed = pack_documents(docs, 12, 0, strategy=SEQUENTIAL_CAUSAL, order_seed=5)
        st = packed.stats()
        assert st["unique_supervised_tokens"] == sum(len(d.output_ids) for d in docs)
        assert st["supervised_tokens"] == st["unique_supervised_tokens"] + st["repeated_supervised_tokens"] \
            - st["truncated_supervised_tokens"]
        assert st["fragments"] == sum(not m["final"] for ms in packed.members for m in ms)

    def test_the_ladder_round_trip_has_no_position_ids(self, tmp_path):
        docs = _docs(10, seed=2)
        train = pack_sequential(docs[:8], 16, 0, order_seed=1)
        val = pack_sequential(docs[8:], 16, 0, order_seed=1)
        rec = write_pack(train, val, tmp_path, data_type={d.doc_id: "t" for d in docs})
        assert rec["packing"] == "token_documents.sequential_causal"
        assert rec["packing_spec"]["packing"] == "sequential_truncate_v1"
        assert rec["rungs"][0]["target_supervised_tokens"] == train.stats()["unique_supervised_tokens"]
        rung = rec["rungs"][0]["target_supervised_tokens"]
        tr, va, stats = ladder_blocks(tmp_path, rung, n_val=len(val.members))
        assert len(tr) == 3 and len(va) == 3
        assert stats["train_unique_supervised_tokens"] == rung
        rows = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
        assert all("final" in s for r in rows for s in r["sessions"])

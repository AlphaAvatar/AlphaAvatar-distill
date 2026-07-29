"""Tests for the v1 mixture builder (scripts/build_stage2_v1.py).

Pure-logic tests: normalization, dedup-seed compatibility with the v0 sink
digest, v0-offset skipping, chat cleaning, and the xlam conversion. No
network, no downloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import build_stage2_v0 as v0
import build_stage2_v1 as v1
from aadistill.data.dataset import validate_sample


def chat_sample(sid: str, content: str = "hi") -> dict:
    return {
        "id": sid, "group": "instruction", "source": "test", "format": "chat",
        "messages": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": content}],
    }


class TestNormalizeGsm8k:
    def test_strips_calc_and_rewrites_final(self):
        raw = ("Ava has 3 apples and buys 2 more.\n"
               "3 + 2 = <<3+2=5>>5 apples.\n#### 5")
        out = v1.normalize_gsm8k_answer(raw)
        assert "<<" not in out and ">>" not in out
        assert "####" not in out
        assert out.endswith("The answer is 5.")
        assert "3 + 2 = 5 apples." in out

    def test_noop_on_clean_text(self):
        clean = "Just a plain answer.\nThe answer is 7."
        assert v1.normalize_gsm8k_answer(clean) == clean

    def test_idempotent(self):
        raw = "x = <<1*2=2>>2\n#### 2"
        once = v1.normalize_gsm8k_answer(raw)
        assert v1.normalize_gsm8k_answer(once) == once

    def test_final_line_answer_with_commas(self):
        out = v1.normalize_gsm8k_answer("Total is big.\n#### 1,200")
        assert out.endswith("The answer is 1,200.")


class TestDigestSeedCompat:
    def test_digest_matches_v0_sink_dedup(self):
        # A sample added to a v0 sink must be rejected by a V1Sink seeded
        # with content_digest(sample), even under a different id.
        s = chat_sample("test-000001")
        seeded = v1.V1Sink({v1.content_digest(s)})
        seeded.start_source("test", {"instruction": 10_000})
        twin = chat_sample("test-000999")
        assert seeded.add("instruction", twin) is False
        assert seeded.counters["test"]["skipped_duplicates"] == 1

    def test_fresh_content_accepted(self):
        s = chat_sample("test-000001")
        seeded = v1.V1Sink({v1.content_digest(s)})
        seeded.start_source("test", {"instruction": 10_000})
        assert seeded.add("instruction", chat_sample("test-000002", "new")) is True


class TestSkipUntil:
    def test_skips_v0_consumed_indices(self):
        sink = v1.V1Sink(set())
        sink.start_source("test", {"instruction": 10_000}, skip_until=100)
        assert sink.add("instruction", chat_sample("test-000100")) is False
        assert sink.counters["test"]["skipped_v0_consumed"] == 1
        assert sink.add("instruction", chat_sample("test-000101")) is True

    def test_budget_untouched_by_skipped(self):
        sink = v1.V1Sink(set())
        sink.start_source("test", {"instruction": 10_000}, skip_until=5)
        sink.add("instruction", chat_sample("test-000003"))
        assert sink.budgets["instruction"] == 10_000


class TestCleanChat:
    def test_keeps_system_turn(self):
        msgs = v1._clean_chat([
            {"role": "system", "content": "Rewrite the text formally."},
            {"role": "user", "content": "yo what up"},
            {"role": "assistant", "content": "Greetings."},
        ])
        assert msgs is not None and msgs[0]["role"] == "system"

    def test_rejects_bad_alternation_and_nonassistant_end(self):
        assert v1._clean_chat([
            {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"}]) is None
        assert v1._clean_chat([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]) is None

    def test_rejects_empty_or_overlong(self):
        assert v1._clean_chat([
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "x"}]) is None
        assert v1._clean_chat([
            {"role": "user", "content": "x" * (v0.MSG_CHAR_CAP + 1)},
            {"role": "assistant", "content": "y"}]) is None


class TestSmoltalkRouting:
    def test_short_vs_instruction(self):
        sink = v1.V1Sink(set())
        sink.start_source("smol_smoltalk",
                          {"instruction": 100_000, "short_realtime": 100_000})
        rows = [
            {"messages": [{"role": "user", "content": "Hi!"},
                          {"role": "assistant", "content": "Hello, how can I help?"}]},
            {"messages": [{"role": "user", "content": "Explain X in detail."},
                          {"role": "assistant", "content": "long " * 300}]},
        ]
        v1.build_smoltalk(rows, sink)
        assert len(sink.samples["short_realtime"]) == 1
        assert len(sink.samples["instruction"]) == 1
        for group in ("short_realtime", "instruction"):
            validate_sample(sink.samples[group][0])


class TestRefusalPool:
    def test_pool_size_and_hygiene(self):
        pool = v1.UNANSWERABLE_RESPONSES_V1
        assert len(pool) == 12 and len(set(pool)) == 12
        from aadistill.data.dataset import FORBIDDEN_MARKERS
        for resp in pool:
            assert all(m not in resp for m in FORBIDDEN_MARKERS)

    def test_build_squad_cycles_widened_pool(self):
        sink = v1.V1Sink(set())
        sink.start_source("squad_v2", {"rag_evidence": 10 ** 6,
                                       "refusal_uncertainty": 10 ** 6})
        rows = [{"context": f"Context {i}.", "question": f"Q{i}?",
                 "answers": {"text": []}} for i in range(14)]
        v1.build_squad_v1(rows, sink)
        answers = [s["messages"][1]["content"]
                   for s in sink.samples["refusal_uncertainty"]]
        assert len(set(answers[:12])) == 12  # all 12 templates used
        assert answers[12] == answers[0]     # cycles


class TestXlamConversion:
    ROW = {
        "query": "What's the weather in Paris and London?",
        "tools": json.dumps([{"name": "get_weather",
                              "description": "Get weather",
                              "parameters": {"city": {"type": "str"}}}]),
        "answers": json.dumps([
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "London"}},
        ]),
    }

    def test_valid_row_converts(self):
        sink = v1.V1Sink(set())
        sink.start_source("xlam_fc_60k", {"tool_calling": 10 ** 6})
        v1.build_xlam([self.ROW], sink)
        [s] = sink.samples["tool_calling"]
        validate_sample(s)
        assert len(s["messages"][1]["tool_calls"]) == 2
        assert s["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris"}

    def test_malformed_rows_counted_not_added(self):
        sink = v1.V1Sink(set())
        sink.start_source("xlam_fc_60k", {"tool_calling": 10 ** 6})
        bad = [
            {**self.ROW, "answers": "not json"},
            {**self.ROW, "answers": json.dumps([{"name": "f", "arguments": "s"}])},
            {**self.ROW, "query": "  "},
        ]
        v1.build_xlam(bad, sink)
        assert not sink.samples["tool_calling"]
        assert sink.counters["xlam_fc_60k"]["parse_failures"] == 3

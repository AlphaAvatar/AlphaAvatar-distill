"""The behavioural diagnostics D2/D3 add, at $0.

Two instruments: a text-level rollout profile that any secured unit supports,
and a per-position rescoring of a student's OWN rollout that only a unit with
a live checkpoint supports. The tests pin what each measures -- in particular
that the rescoring reads the student's distributions on the student's own
prefix, with the causal shift, against an independent computation.
"""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from shared.evaluation import rollout_rescoring as RR  # noqa: E402
from stages.d_series import rollout_profile as RP  # noqa: E402


class _Tok:
    """Whitespace tokenizer: enough to count tokens deterministically."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _gen(i, raw, n_tokens, *, stop="eos", natural=True):
    return {"id": f"p{i}", "raw": raw, "generated_tokens": n_tokens,
            "think_preopened": True, "stop_reason": stop,
            "natural_termination": natural, "context_limit_reached": False}


def _row(i, *, correct, usable=True, scorable=True):
    return {"id": f"p{i}", "set": "gsm8k", "correct": correct,
            "usable": usable, "scorable": scorable}


@pytest.fixture
def unit(tmp_path):
    gens = [
        _gen(0, "a b c </think> x y<|im_end|>", 6),
        _gen(1, "a b c d e f </think> x<|im_end|>", 8),
        _gen(2, "a b a b a b a b", 8, stop="degeneration", natural=False),
    ]
    (tmp_path / "gsm8k.generations.jsonl").write_text(
        "\n".join(json.dumps(g) for g in gens) + "\n")
    rows = [_row(0, correct=True), _row(1, correct=False),
            _row(2, correct=False, usable=False)]
    return tmp_path, rows


class TestTheTextProfile:

    def test_rows_join_scores_to_generations(self, unit):
        path, rows = unit
        out = RP.unit_rows(path, rows, _Tok())
        by = {r["id"]: r for r in out}
        assert by["p0"]["think_closed"] and by["p0"]["correct"]
        assert by["p0"]["reasoning_tokens"] == 3
        assert by["p2"]["think_closed"] is False
        assert by["p2"]["stop_reason"] == "degeneration"

    def test_the_engine_count_is_kept_and_retokenization_flagged(self, unit):
        path, rows = unit
        out = {r["id"]: r for r in RP.unit_rows(path, rows, _Tok())}
        assert out["p1"]["generated_tokens"] == 8
        assert out["p2"]["retokenization_exact"] is True

    def test_a_generation_without_a_score_is_refused(self, unit):
        path, rows = unit
        with pytest.raises(ValueError, match="no per-sample row"):
            RP.unit_rows(path, rows[:2], _Tok())

    def test_summary_rates(self, unit):
        path, rows = unit
        s = RP.summarize(RP.unit_rows(path, rows, _Tok()))
        assert s["correct_rate_scorable"] == pytest.approx(1 / 3, abs=1e-4)
        assert s["degeneration_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert s["generated_tokens"]["correct"]["n"] == 1

    def test_paired_counts_and_length_deltas(self):
        a = {1: [{"id": "x", "scorable": True, "correct": True,
                  "generated_tokens": 10},
                 {"id": "y", "scorable": True, "correct": True,
                  "generated_tokens": 30}]}
        b = {1: [{"id": "x", "scorable": True, "correct": True,
                  "generated_tokens": 25},
                 {"id": "y", "scorable": True, "correct": False,
                  "generated_tokens": 30}]}
        p = RP.paired(a, b)
        assert (p["both_correct"], p["candidate_only_correct"],
                p["reference_only_correct"]) == (1, 1, 0)
        assert p["generated_token_delta"]["both_correct"]["p50"] == -15


@pytest.fixture(scope="module")
def pair():
    from support.toy import TEACHER_GEOMETRY, build_tiny_model
    from aadistill.initialization.adapters import register_builtin_adapters

    register_builtin_adapters()
    teacher = build_tiny_model(TEACHER_GEOMETRY)
    student = build_tiny_model(TEACHER_GEOMETRY, seed=19)
    return teacher, student


class TestRescoringReadsTheStudentOnItsOwnPrefix:

    def test_against_an_independent_computation(self, pair):
        teacher, student = pair
        g = torch.Generator().manual_seed(1)
        prompt = torch.randint(1, 128, (1, 7), generator=g)
        completion = torch.randint(1, 128, (1, 9), generator=g)
        terms = RR.rescore_trajectory(student, teacher, prompt, completion,
                                      support_top_k=16, entropy_top_k=8)
        ids = torch.cat([prompt, completion], dim=1)
        with torch.no_grad():
            s = student(ids).logits[0, 6:15].double()
            t = torch.log_softmax(teacher(ids).logits[0, 6:15].double(), -1)
        s_logp = torch.log_softmax(s, -1)
        tok = completion[0]
        assert torch.allclose(terms["student_token_logprob"].double(),
                              s_logp.gather(1, tok.view(-1, 1)).squeeze(1),
                              atol=1e-4)
        assert torch.allclose(terms["teacher_surprisal"].double(),
                              -t.gather(1, tok.view(-1, 1)).squeeze(1),
                              atol=1e-4)
        top = torch.topk(s, 8, dim=-1).values
        q = torch.softmax(top, -1)
        h = -(q * q.log()).sum(-1)
        assert torch.allclose(terms["student_confidence"].double(),
                              (1 - h / math.log(8)).clamp(0, 1), atol=1e-4)

    def test_the_same_model_is_at_zero_kl(self, pair):
        teacher, _ = pair
        prompt = torch.tensor([[3, 5, 7]])
        completion = torch.tensor([[11, 13, 17, 19]])
        terms = RR.rescore_trajectory(teacher, teacher, prompt, completion,
                                      support_top_k=16, entropy_top_k=8)
        assert float(terms["kl_teacher_student"].abs().max()) < 1e-5

    def test_the_summary_splits_reasoning_and_uses_given_thresholds(self):
        terms = {"student_confidence": torch.tensor([0.9, 0.9, 0.1, 0.95]),
                 "kl_teacher_student": torch.tensor([2.0, 0.1, 2.0, 3.0]),
                 "teacher_surprisal": torch.tensor([1.0, 1.0, 1.0, 1.0])}
        s = RR.summarize_trajectory(terms, reasoning_positions=2,
                                    high_confidence=0.8, high_kl=1.0)
        assert s["high_confidence_high_kl_fraction"] == pytest.approx(0.5)
        assert s["reasoning"]["student_confidence"] == pytest.approx(0.9)
        assert s["answer"]["kl_teacher_student"] == pytest.approx(2.5)

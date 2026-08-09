"""General-text diagnostics: correct arithmetic, and correctly labelled.

Validated against known-answer cases before anything is spent on them — a
project habit that has caught three real scorer defects
(`logs/EXPERIMENTS.md`). A diagnostic that is wrong in the same direction as the
hypothesis is worse than no diagnostic.
"""

import math

import pytest
import torch

from aadistill.evaluation.general_text import general_text_metrics


class Table(torch.nn.Module):
    """Logits are a pure function of the input token: `logits[b,t] = table[ids[b,t]]`.

    A genuine function of the input, not a positional slice. That matters here:
    a double that ignores which blocks it was handed would make the batching
    invariance tests pass vacuously.
    """

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = table

    def forward(self, ids):
        class Out:
            pass
        o = Out()
        o.logits = self.table[ids]
        return o


def successor_table(vocab: int, sharp: float = 50.0) -> torch.Tensor:
    """`table[k]` puts all its mass on `k + 1`, so predicting a run is perfect."""
    t = torch.full((vocab, vocab), -sharp)
    for k in range(vocab):
        t[k, (k + 1) % vocab] = sharp
    return t


def random_table(vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, vocab, generator=g)


def test_a_perfect_model_scores_zero_nll_and_rank_one():
    ids = torch.tensor([[1, 2, 3, 4]])           # a successor run
    m = general_text_metrics(Table(successor_table(8)), ids)
    assert m["positions"] == 3
    assert m["nll"] == pytest.approx(0.0, abs=1e-6)
    assert m["top1"] == 1.0
    assert m["mean_rank"] == 1.0
    assert m["mean_target_prob"] == pytest.approx(1.0, abs=1e-6)
    assert m["ppl"] == pytest.approx(1.0, abs=1e-4)


def test_a_uniform_model_scores_log_vocab():
    ids = torch.tensor([[1, 2, 3, 4]])
    m = general_text_metrics(Table(torch.zeros(8, 8)), ids)
    assert m["nll"] == pytest.approx(math.log(8), abs=1e-5)
    assert m["mean_entropy"] == pytest.approx(math.log(8), abs=1e-5)
    # Every logit ties, so nothing strictly beats the target: rank 1 by the
    # strict-greater convention, and argmax picks index 0.
    assert m["mean_rank"] == 1.0


def test_kl_is_zero_against_the_same_teacher():
    torch.manual_seed(0)
    ids = torch.randint(0, 11, (2, 9))
    tab = random_table(11, 10)
    m = general_text_metrics(Table(tab), ids, teacher=Table(tab))
    assert m["kl"] == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_and_nll_independent_against_a_different_teacher():
    """The two numbers can come apart, which is why both are reported."""
    torch.manual_seed(1)
    ids = torch.randint(0, 11, (2, 9))
    s, t = random_table(11, 11), random_table(11, 12)
    m = general_text_metrics(Table(s), ids, teacher=Table(t))
    same = general_text_metrics(Table(s), ids)
    assert m["kl"] > 0
    assert m["nll"] == pytest.approx(same["nll"], abs=1e-9), (
        "adding a teacher must not change the student's own NLL")


def test_chunking_does_not_change_the_result():
    torch.manual_seed(2)
    ids = torch.randint(0, 13, (3, 17))
    model = Table(random_table(13, 13))
    a = general_text_metrics(model, ids, chunk=4)
    b = general_text_metrics(model, ids, chunk=1000)
    for k in ("nll", "top1", "mean_rank", "mean_target_prob", "mean_entropy"):
        assert a[k] == pytest.approx(b[k], abs=1e-5), k


def test_micro_blocks_do_not_change_the_result():
    torch.manual_seed(3)
    ids = torch.randint(0, 13, (4, 11))
    model = Table(random_table(13, 14))
    a = general_text_metrics(model, ids, micro_blocks=1)
    b = general_text_metrics(model, ids, micro_blocks=4)
    assert a["positions"] == b["positions"] == 4 * 10
    assert a["nll"] == pytest.approx(b["nll"], abs=1e-5)


def test_position_count_is_exactly_n_times_len_minus_one():
    torch.manual_seed(4)
    ids = torch.randint(0, 7, (5, 23))
    m = general_text_metrics(Table(random_table(7, 15)), ids)
    assert m["positions"] == 5 * 22


def test_max_blocks_truncates_and_says_so():
    torch.manual_seed(5)
    ids = torch.randint(0, 7, (6, 9))
    m = general_text_metrics(Table(random_table(7, 16)), ids, max_blocks=2)
    assert m["blocks"] == 2 and m["positions"] == 2 * 8


def test_an_empty_stream_raises_rather_than_reporting_a_number():
    with pytest.raises(ValueError, match="no blocks"):
        general_text_metrics(Table(random_table(5, 17)),
                             torch.zeros(0, 4, dtype=torch.long))


def test_eval_mode_is_restored():
    torch.manual_seed(6)
    model = Table(random_table(7, 18))
    model.train()
    general_text_metrics(model, torch.randint(0, 7, (2, 5)))
    assert model.training is True

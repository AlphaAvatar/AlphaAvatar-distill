"""Contribution-guided depth selection tests (CPU, tiny random Qwen3).

Three things carry the experiment and are checked independently of each other:

1. **bypass really is a residual pass-through** — verified against a model whose
   block forward has been replaced by the identity, which shares no code with
   the ModuleList swap under test;
2. **the objective is the objective** — `distortion` checked against
   hand-computed KL, and the domain aggregation checked against a case where
   token weighting and domain balancing give different answers;
3. **greedy is not Top-N** — a synthetic objective with conditional redundancy,
   where the one-shot ranking picks a fatal pair and the iterative search does
   not.

And the load-bearing single-variable proof: feeding `init_student` the positional
map's own representatives as an explicit map must reproduce the positional
initialization **bitwise**, so a depth-map experiment changes the depth map and
nothing else.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.init.contribution import (  # noqa: E402
    bypassed_blocks,
    distortion,
    domain_balanced_score,
    expected_evaluations,
    greedy_removal,
)
from aadistill.init.sandwich import (  # noqa: E402
    depth_span_map,
    explicit_depth_map,
    init_student,
)
from aadistill.models.student import build_student, build_student_config  # noqa: E402

from test_stage1 import collect_stats, geometry, tiny_teacher  # noqa: E402


def _ids(vocab: int = 128, n: int = 40, seed: int = 5):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (1, n))


# --- bypass -------------------------------------------------------------------


def test_bypassing_nothing_reproduces_the_intact_model_exactly():
    model = tiny_teacher()
    model.config.use_cache = False
    ids = _ids()
    with torch.no_grad():
        intact = model(ids).logits
        with bypassed_blocks(model, set()):
            same = model(ids).logits
    assert torch.equal(intact, same)


def test_bypass_equals_replacing_the_block_with_the_identity():
    """An independent path to the same model: the block returns its input."""
    model = tiny_teacher()
    model.config.use_cache = False
    ids = _ids()
    skip = 2
    with torch.no_grad():
        with bypassed_blocks(model, {skip}):
            via_bypass = model(ids).logits
        layer = model.model.layers[skip]
        original_forward = layer.forward
        layer.forward = lambda hidden_states, **kwargs: hidden_states
        try:
            via_identity = model(ids).logits
        finally:
            layer.forward = original_forward
    assert torch.equal(via_bypass, via_identity)


def test_bypass_changes_the_output_and_restores_the_layer_list():
    model = tiny_teacher()
    model.config.use_cache = False
    ids = _ids()
    before = list(model.model.layers)
    with torch.no_grad():
        intact = model(ids).logits
        with bypassed_blocks(model, {1, 3}):
            assert len(model.model.layers) == 2
            ablated = model(ids).logits
    assert list(model.model.layers) == before
    assert not torch.allclose(intact, ablated)


def test_the_layer_list_is_restored_even_when_the_body_raises():
    model = tiny_teacher()
    model.config.use_cache = False
    before = list(model.model.layers)
    with pytest.raises(RuntimeError):
        with bypassed_blocks(model, {0}):
            raise RuntimeError("forward blew up")
    assert list(model.model.layers) == before


def test_bypass_refuses_impossible_or_unsafe_requests():
    model = tiny_teacher()
    model.config.use_cache = False
    with pytest.raises(ValueError, match="out of range"):
        with bypassed_blocks(model, {99}):
            pass
    with pytest.raises(ValueError, match="cannot bypass all"):
        with bypassed_blocks(model, {0, 1, 2, 3}):
            pass
    model.config.use_cache = True
    with pytest.raises(ValueError, match="use_cache"):
        with bypassed_blocks(model, {0}):
            pass


# --- the objective ------------------------------------------------------------


def test_identical_distributions_have_zero_distortion():
    logits = torch.randn(6, 11)
    d = distortion(logits, logits.clone(), torch.randint(0, 11, (6,))).as_dict()
    assert d["positions"] == 6
    assert abs(d["kl"]) < 1e-6
    assert abs(d["reverse_kl"]) < 1e-6
    assert abs(d["ce_delta"]) < 1e-6
    assert d["top1_agreement"] == 1.0


def test_distortion_matches_hand_computed_kl():
    # Two positions, three-token vocabulary. The expectation is computed in
    # float64 from an independent formula; `distortion` reduces in float32, so
    # the tolerance is the float32 reduction's, not a fudge factor.
    ref = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    abl = torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0])
    p = torch.softmax(ref[0].double(), 0)
    q = torch.softmax(abl[0].double(), 0)
    expect_pos0 = float((p * (p.log() - q.log())).sum())
    d = distortion(ref, abl, targets).as_dict()
    assert d["kl"] == pytest.approx(expect_pos0 / 2, rel=1e-5)
    # Position 1 is untouched, so all of the CE change comes from position 0.
    assert d["ce_delta"] == pytest.approx(
        float(-torch.log(q[0]) + torch.log(p[0])) / 2, rel=1e-5)
    assert d["top1_agreement"] == 0.5


def test_tagged_positions_are_reported_and_never_pooled_into_the_primary():
    ref = torch.zeros(4, 5)
    abl = torch.zeros(4, 5)
    abl[2, 0] = 50.0  # only the tagged position is distorted
    tags = {"think_close": torch.tensor([False, False, True, False])}
    d = distortion(ref, abl, torch.zeros(4, dtype=torch.long), tags=tags).as_dict()
    assert d["tagged"]["think_close"]["positions"] == 1
    assert d["tagged"]["think_close"]["kl"] > d["kl"]      # the mean dilutes it
    assert d["kl"] == pytest.approx(d["tagged"]["think_close"]["kl"] / 4, rel=1e-5)


def test_distortion_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="shape mismatch"):
        distortion(torch.zeros(3, 5), torch.zeros(3, 6), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="position count"):
        distortion(torch.zeros(3, 5), torch.zeros(3, 5), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="wrong length"):
        distortion(torch.zeros(3, 5), torch.zeros(3, 5), torch.zeros(3, dtype=torch.long),
                   tags={"x": torch.zeros(2, dtype=torch.bool)})


def test_domain_balancing_beats_the_token_weighted_mean_it_replaces():
    # `code` has a huge score; under token weighting (not used here) its longer
    # sequences would dominate. Two levels of unweighted means give each of the
    # two domains exactly half the influence regardless of sub-type count.
    scores = {"general": 1.0, "gsm8k": 3.0, "openmath": 5.0}
    primary, per_domain = domain_balanced_score(
        scores, {"general": ["general"], "math": ["gsm8k", "openmath"]})
    assert per_domain == {"general": 1.0, "math": 4.0}
    assert primary == pytest.approx(2.5)


def test_a_missing_subtype_raises_instead_of_reweighting_its_domain():
    with pytest.raises(ValueError, match="missing sub-types"):
        domain_balanced_score({"a": 1.0}, {"d": ["a", "b"]})
    with pytest.raises(ValueError, match="no sub-types"):
        domain_balanced_score({"a": 1.0}, {"d": []})
    with pytest.raises(ValueError, match="no domains"):
        domain_balanced_score({"a": 1.0}, {})


# --- greedy vs one-shot -------------------------------------------------------


def _conditional_redundancy_objective(skip):
    """Layers 1 and 2 are individually cheap but fatal together.

    Each layer has an independent cost; the pair {1, 2} adds a large interaction
    penalty. A one-shot ranking removes the two cheapest layers and pays it.
    """
    base = {0: 9.0, 1: 0.1, 2: 0.2, 3: 1.0, 4: 1.1}
    total = sum(base[i] for i in skip)
    if {1, 2} <= set(skip):
        total += 100.0
    return total


def test_greedy_avoids_the_pair_that_one_shot_top_n_would_take():
    one_shot = sorted(range(5), key=lambda i: _conditional_redundancy_objective({i}))[:2]
    assert set(one_shot) == {1, 2}      # the fatal pair, by independent scores
    result = greedy_removal(_conditional_redundancy_objective, 5, 2)
    assert set(result["removed"]) != {1, 2}
    assert result["removed"] == [1, 3]
    assert result["kept"] == [0, 2, 4]
    assert result["evaluations"] == 5 + 4


def test_every_round_table_is_kept_whole_and_in_index_order():
    result = greedy_removal(_conditional_redundancy_objective, 5, 2)
    assert [r["round"] for r in result["rounds"]] == [0, 1]
    first, second = result["rounds"]
    assert [e["candidate"] for e in first["table"]] == [0, 1, 2, 3, 4]
    assert first["removed_before"] == [] and second["removed_before"] == [1]
    assert [e["candidate"] for e in second["table"]] == [0, 2, 3, 4]
    assert second["table"][1]["score"] == pytest.approx(0.1 + 0.2 + 100.0)


def test_the_search_size_is_the_preregistered_260_for_36_to_28():
    assert expected_evaluations(36, 8) == 36 + 35 + 34 + 33 + 32 + 31 + 30 + 29
    assert expected_evaluations(36, 8) == 260
    calls = []

    def counting(skip):
        calls.append(frozenset(skip))
        return float(sum(skip))            # trivial: removes the low indices

    result = greedy_removal(counting, 36, 8)
    assert len(calls) == 260 == result["evaluations"]
    assert result["removed"] == list(range(8))
    assert result["kept"] == list(range(8, 36))


def test_ties_break_on_the_lower_layer_index():
    result = greedy_removal(lambda skip: 1.0, 4, 2)
    assert result["removal_order"] == [0, 1]


def test_protect_and_bounds_are_enforced():
    result = greedy_removal(lambda skip: float(sum(skip)), 5, 2, protect={0, 1})
    assert result["removed"] == [2, 3]
    assert result["protect"] == [0, 1]
    with pytest.raises(ValueError, match="cannot remove"):
        greedy_removal(lambda skip: 0.0, 4, 4)
    with pytest.raises(ValueError, match="too few removable"):
        greedy_removal(lambda skip: 0.0, 5, 3, protect={0, 1, 2})
    with pytest.raises(ValueError, match="returned"):
        greedy_removal(lambda skip: float("nan"), 4, 1)


def test_a_resumed_search_replays_finished_rounds_instead_of_rescoring():
    full = greedy_removal(_conditional_redundancy_objective, 5, 2)
    calls = []

    def counting(skip):
        calls.append(frozenset(skip))
        return _conditional_redundancy_objective(skip)

    resumed = greedy_removal(counting, 5, 2, completed_rounds=[full["rounds"][0]])
    assert resumed["removed"] == full["removed"]
    assert len(calls) == 4                      # round 0 was not re-scored
    assert resumed["rounds"][0]["resumed"] is True
    with pytest.raises(ValueError, match="re-removes"):
        greedy_removal(counting, 5, 2,
                       completed_rounds=[{"chosen": 1}, {"chosen": 1}])
    with pytest.raises(ValueError, match="protected"):
        greedy_removal(counting, 5, 2, protect={1}, completed_rounds=[{"chosen": 1}])


# --- explicit depth map -------------------------------------------------------


def test_an_explicit_map_of_the_positional_representatives_is_that_map():
    positional = depth_span_map(36, 28)
    kept = [s["representative"] for s in positional]
    assert kept == [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, *range(20, 36)]
    assert explicit_depth_map(kept, 36) == positional


def test_an_explicit_map_covers_only_what_it_keeps():
    spans = explicit_depth_map([1, 3], 5)
    assert spans == [
        {"student": 0, "teacher_span": [1, 3], "representative": 1},
        {"student": 1, "teacher_span": [3, 5], "representative": 3},
    ]


def test_an_explicit_map_refuses_reordered_or_invalid_layer_lists():
    with pytest.raises(ValueError, match="empty"):
        explicit_depth_map([], 4)
    with pytest.raises(ValueError, match="duplicates"):
        explicit_depth_map([1, 1], 4)
    with pytest.raises(ValueError, match="increasing teacher order"):
        explicit_depth_map([2, 1], 4)
    with pytest.raises(ValueError, match="outside range"):
        explicit_depth_map([0, 4], 4)


def test_the_explicit_positional_map_reproduces_the_init_bitwise():
    """The single-variable guarantee: same map in, same weights out."""
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(teacher.config,
                                                       num_hidden_layers=3))
    positional_kept = [s["representative"] for s in depth_span_map(4, 3)]

    a = build_student(cfg, torch.float32, seed=1)
    diag_a = init_student(teacher, a, state)
    b = build_student(cfg, torch.float32, seed=2)
    diag_b = init_student(teacher, b, state, kept_layers=positional_kept)

    assert diag_a["depth_map_source"] == "positional_pairwise_merge"
    assert diag_b["depth_map_source"] == "explicit_kept_layers"
    assert diag_a["depth_map"] == diag_b["depth_map"]
    assert diag_a["kept_teacher_layers"] == diag_b["kept_teacher_layers"]
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert set(sd_a) == set(sd_b)
    differing = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])]
    assert differing == []


def test_a_different_kept_set_changes_the_student_and_is_recorded():
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(teacher.config,
                                                       num_hidden_layers=3))
    a = build_student(cfg, torch.float32, seed=1)
    init_student(teacher, a, state, kept_layers=[0, 1, 2])
    b = build_student(cfg, torch.float32, seed=1)
    diag = init_student(teacher, b, state, kept_layers=[0, 1, 3])

    assert diag["kept_teacher_layers"] == [0, 1, 3]
    assert diag["removed_teacher_layers"] == [2]
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    # The projection is a function of the statistics alone, so it must not move
    # when the depth map does — that is what keeps E8 a single-variable change.
    assert torch.equal(sd_a["model.embed_tokens.weight"],
                       sd_b["model.embed_tokens.weight"])


def test_a_kept_list_of_the_wrong_length_is_refused():
    teacher = tiny_teacher()
    state = collect_stats(teacher)
    cfg = build_student_config(teacher.config, geometry(teacher.config,
                                                       num_hidden_layers=3))
    student = build_student(cfg, torch.float32, seed=1)
    with pytest.raises(ValueError, match="kept_layers has 2 entries"):
        init_student(teacher, student, state, kept_layers=[0, 1])

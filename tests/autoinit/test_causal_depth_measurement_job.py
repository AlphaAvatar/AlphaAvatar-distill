"""The GPU measurement job, executed for real at toy scale on the dev box.

`scripts/autoinit/measure_causal_depth_runtime.py` cannot run its `main()` here —
it refuses without CUDA, deliberately, because measuring the accelerator path on
the host would re-measure the defect. But four paid pods have now died inside
lines no $0 path had ever executed, so its *body* runs here: `run_measurement` is
a seam, and this drives it end to end against a tiny real Qwen3, the production
reference-cache class, and the real E8a `Searcher`.

The three defects the reviewer found in the first version are each pinned:

* it timed every sample at cardinality 8, which would have overstated
  evaluations/min and understated the 260-evaluation runtime;
* it built its own cache, bypassing the operator's 0.66 gate and recompute
  fallback;
* it accepted an unpinned Hub revision.

and the unbacked "reports GPU utilization" claim is now a real sampler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "training"))


def load_job():
    path = REPO / "scripts/autoinit/measure_causal_depth_runtime.py"
    spec = importlib.util.spec_from_file_location("measure_causal_depth", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_causal_depth"] = mod
    spec.loader.exec_module(mod)
    return mod


JOB = load_job()


def tiny_teacher(n_layers: int = 6):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(5)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=32, num_hidden_layers=n_layers,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=128,
    )
    m = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    m.config.use_cache = False
    return m


def toy_items(vocab: int):
    """Deliberately unequal lengths: that is what makes E8a's position-weighted
    subtype mean differ from the operator's unweighted one."""
    g = torch.Generator().manual_seed(9)
    return [{"item_id": f"i{k}", "subtype": ["a", "b"][k % 2], "domain": "d0",
             "input_ids": torch.randint(0, vocab, (1, 24 + 8 * k), generator=g)}
            for k in range(4)]


# --- 1. the sampling scheme -------------------------------------------------

def test_skip_sets_are_deterministic_and_the_right_size():
    for c in range(1, 9):
        for j in range(4):
            s = JOB.skip_set(c, j)
            assert len(s) == c, f"cardinality {c} sample {j} collided: {sorted(s)}"
            assert s == JOB.skip_set(c, j), "not deterministic"
            assert all(0 <= x < 36 for x in s)


def test_skip_sets_spread_rather_than_clump():
    """A block of eight adjacent layers is not a representative ablation, and it
    would also be the cheapest possible one to run."""
    s = sorted(JOB.skip_set(8, 0))
    assert max(b - a for a, b in zip(s, s[1:])) > 1, f"clumped: {s}"


def test_the_schedule_is_the_real_one():
    """36,35,...,29 summing to 260 — the greedy rounds, not a guess."""
    assert JOB.SCHEDULE == {1: 36, 2: 35, 3: 34, 4: 33, 5: 32, 6: 31, 7: 30, 8: 29}
    assert sum(JOB.SCHEDULE.values()) == 260
    from aadistill.init.contribution import expected_evaluations
    assert sum(JOB.SCHEDULE.values()) == expected_evaluations(36, 8)


def test_every_cardinality_is_sampled_not_only_the_cheapest(tmp_path):
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=1)
    # 6 layers, remove 3 -> cardinalities 1..3, weights 6,5,4.
    assert core["schedule"] == {1: 6, 2: 5, 3: 4}
    assert set(core["timings"]) == set(core["schedule"])
    assert all(len(v) == 1 for v in core["timings"].values())
    assert core["n_timed"] == len(core["schedule"])


def test_the_weighted_extrapolation_differs_from_the_flat_one():
    """The exact defect: timing everything at the largest skip set understates
    the real runtime, because those forwards run the fewest layers."""
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=1)
    # Recompute the weighting from the parts, so the assertion is on arithmetic
    # rather than on a number the code happened to print.
    expect = sum(core["schedule"][c] * core["means_by_c"][c]
                 for c in core["schedule"])
    assert core["weighted_s"] == pytest.approx(expect)
    assert core["flat_s"] == pytest.approx(
        sum(core["schedule"].values()) * core["means_by_c"][max(core["schedule"])])


# --- 2. the production cache path -------------------------------------------

def test_the_job_uses_the_operators_own_reference_cache():
    """Not a private dict: the operator's `_ReferenceLogits`, so the 0.66 gate
    and the recompute fallback are the ones that really run."""
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "_ReferenceLogits" in src and "_forward_logits" in src
    body = src[src.index("def run_measurement"):src.index("def main")]
    assert "reference = _ReferenceLogits(" in body
    assert "reference.get(item)" in body
    assert "cache_decision" in body
    # And it must not quietly build its own.
    assert "reference: dict" not in body, "the job reintroduced a private cache"


def test_the_cache_decision_is_reported_and_is_the_operators_own():
    from aadistill.autoinit.operators.depth import _ReferenceLogits

    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=1)
    d = core["cache_decision"]
    assert set(d) == set(_ReferenceLogits(model, items, "cpu").decision())
    assert "cached" in d and "budget_fraction" in d
    assert d["budget_fraction"] == _ReferenceLogits.BUDGET_FRACTION


# --- 3. the pinned identities -----------------------------------------------

def test_the_teacher_revision_is_pinned_to_the_frozen_one():
    assert JOB.TEACHER_REVISION == "768f209d9ea81521153ed38c47d515654e938aea"
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "refusing to measure against an unpinned Hub HEAD" in src
    # The default must BE the pin, not merely mention it.
    assert 'ap.add_argument("--teacher-revision", default=TEACHER_REVISION)' in src


def test_the_frozen_revision_matches_the_rest_of_the_repository():
    launcher = (REPO / "scripts/pod/autoinit_phase_a_launch.py").read_text()
    assert JOB.TEACHER_REVISION in launcher, (
        "the measurement job pins a different teacher revision from the session")


# --- 4. GPU utilization is sampled, not claimed ------------------------------

def test_the_sampler_records_real_samples():
    class Fake(JOB.GpuSampler):
        def _read(self):
            return 73

    s = Fake(torch.device("cpu"), period_s=0.01)
    with s:
        import time as _t
        _t.sleep(0.08)
    r = s.report()
    assert r["samples"] >= 2, r
    assert r["mean_pct"] == 73 and r["max_pct"] == 73


def test_the_sampler_withdraws_the_claim_when_it_cannot_measure():
    """Better to say nothing than to report a number nobody sampled."""
    class Blind(JOB.GpuSampler):
        def _read(self):
            self.method = "unavailable"
            return None

    s = Blind(torch.device("cpu"), period_s=0.01)
    with s:
        import time as _t
        _t.sleep(0.05)
    r = s.report()
    assert r["samples"] == 0 and "withdrawn" in r["note"]


# --- 5. the E8a comparison, and what it must not conflate --------------------

def test_the_e8a_pairing_compares_per_item_and_reports_the_aggregation_gap():
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=2)
    assert len(core["paired"]) == 2
    for p in core["paired"]:
        # Same forward, same reduction, same device -> the per-item sums must be
        # identical. This is the backend claim.
        assert p["max_per_item_kl_delta"] == pytest.approx(0.0, abs=1e-12), (
            f"per-item KL differs between E8a and the port: {p}")
        # And the aggregated scores differ by the DECLARED aggregation, which is
        # why the per-item delta is the comparison and this is only reported.
        assert "aggregated_difference" in p


def test_the_declared_aggregation_difference_is_real_and_would_mislead():
    """If this ever becomes zero the framing above is wrong and the report's
    explanation must change — so it is pinned rather than assumed."""
    from aadistill.init.contribution import DistortionSums, distortion

    torch.manual_seed(3)
    per_item = []
    for n in (30, 90):                      # unequal lengths, as the mixture is
        per_item.append(distortion(torch.randn(n, 40), torch.randn(n, 40),
                                   torch.randint(0, 40, (n,)), chunk=512))
    merged = DistortionSums()
    for s in per_item:
        merged.merge(s)
    e8a = merged.as_dict()["kl"]                                  # position-weighted
    port = sum(s.as_dict()["kl"] for s in per_item) / len(per_item)   # unweighted
    assert e8a != port
    assert abs(e8a - port) > 8.195e-05, (
        "the aggregation gap is now below the smallest real decision margin; the "
        "measurement report's explanation of why it compares per-item needs "
        "revisiting")


def test_the_job_runs_no_search_and_selects_no_depth_map():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "greedy_removal" not in src, "the measurement job runs a greedy search"
    assert "depth_span_map" not in src
    assert "not_a_phase_a_attempt" in src


# --- 6. one reference cache at a time ---------------------------------------
#
# GPU-only in its consequence: each frozen cache is ~16.9 GiB, so holding the
# production one and E8a's together would either OOM the measurement or make the
# reported peak describe a dual-cache arrangement no Phase-A run ever has. The
# consequence cannot be reproduced on this box; the arrangement that causes it
# can, and is what these assert.

def test_the_production_peak_is_captured_before_the_comparison_path_exists():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    body = src[src.index("def run_measurement"):src.index("def main")]
    prod = body.index("production_peak =")
    release = body.index("reference._cache.clear()")
    e8a = body.index("from search_depth_map import")
    assert prod < release < e8a, (
        "the production peak must be taken BEFORE the cache is released and "
        "before E8a's path is built; otherwise it is not the Phase-A number")


def test_the_production_cache_is_released_before_e8a_is_constructed():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    body = src[src.index("def run_measurement"):src.index("def main")]
    assert "reference._cache.clear()" in body and "del reference" in body
    assert "gc.collect()" in body
    assert "torch.cuda.empty_cache()" in body
    assert body.index("del reference") < body.index("Searcher(")


def test_e8a_runs_without_a_second_reference_cache():
    """The pairs validate backend numerics, not E8a throughput, and E8a defines
    recomputation as numerically identical."""
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "cache_reference=False" in src
    assert "cache_reference=True" not in src


def test_the_two_peaks_are_reported_separately_and_not_substituted():
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=2)
    assert "production_peak_bytes" in core and "comparison_peak_bytes" in core
    assert core["e8a_cache_reference"] is False
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert '"which_is_the_phase_a_number": "production_peak_gib"' in src


def test_the_released_cache_does_not_cost_the_port_results():
    """Releasing is only safe because the port's per-item results are already
    scalars. If they ever become tensors this test is the one that should fail."""
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=2)
    for p in core["paired"]:
        assert isinstance(p["port_aggregated_score"], float)
        assert isinstance(p["max_per_item_kl_delta"], float)


def test_the_e8a_pairs_are_the_smallest_and_largest_skip_sets():
    """Explicit, not 'whichever two came first in a dict'. They bracket the
    schedule, so a backend difference appearing only at depth cannot hide."""
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=2)
    cards = [p["cardinality"] for p in core["paired"]]
    assert cards == [1, 3], f"expected the smallest and largest, got {cards}"
    assert core["paired"][0]["skip"] == sorted(JOB.skip_set(1, 0, 6))
    assert core["paired"][1]["skip"] == sorted(JOB.skip_set(3, 0, 6))


def test_the_pair_count_was_not_increased():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert 'ap.add_argument("--e8a-pairs", type=int, default=2,' in src


def test_the_production_peak_is_read_before_the_comparison_peak():
    """Behavioural, not textual. Both peaks are None on a CPU box, so a test
    comparing only their values cannot tell "captured before the comparison"
    from "captured after and copied" — a mutation kept the assignment in place
    and aliased the value, and the source-ordering test passed. An injected
    counter makes the order observable here."""
    model = tiny_teacher()
    items = toy_items(model.config.vocab_size)
    reads = []

    def counter():
        reads.append(len(reads) + 1)
        return reads[-1]

    core = JOB.run_measurement(model, items, torch.device("cpu"), n_layers=6,
                               n_remove=3, samples_per_cardinality=1,
                               e8a_pairs=2, peak_reader=counter)
    assert len(reads) == 2, f"expected exactly two peak reads, got {reads}"
    assert core["production_peak_bytes"] == 1, "the production peak was not read first"
    assert core["comparison_peak_bytes"] == 2
    assert core["production_peak_bytes"] != core["comparison_peak_bytes"], (
        "the two peaks are the same reading; the production number would then "
        "include E8a's path")

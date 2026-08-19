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
import inspect
import json
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


# --- 7. the ENTRYPOINT, executed ------------------------------------------
#
# Measurement attempt 2 died at $0.18 on `ImportError: cannot import name
# 'as_operator_items' from 'aadistill.autoinit.datasets'` — a line inside
# `main()`, which refuses to start without CUDA, so no $0 path had ever reached
# it. `run_measurement` had a seam and was hammered; `main()` did not, and that
# is exactly where the defect was.
#
# These drive `run_entrypoint` — the real one, the same function the paid CLI
# calls — with a toy model and a stand-in for the CUDA-only bookkeeping. There is
# no second implementation of the entrypoint anywhere.

class FakeHardware:
    """Only the CUDA-specific bookkeeping. Everything else is the real path."""

    def __init__(self, device):
        self.device = device
        self.reset_calls = 0

    def reset_peak(self):
        self.reset_calls += 1

    def name(self):
        return "FakeDevice"

    def total_gib(self):
        return 44.99


def test_the_entrypoint_runs_end_to_end_on_the_dev_box(tmp_path, monkeypatch):
    """Argument defaults, the pinned identity, loading, calibration resolution,
    `as_operator_items`, identity assembly, the measurement, report assembly,
    the stop conditions and the artifact write — all of it, for free."""
    model = tiny_teacher(n_layers=6)
    items = toy_items(model.config.vocab_size)

    def loader(args, device):
        assert args.teacher_revision == JOB.TEACHER_REVISION, (
            "the entrypoint did not carry the pinned revision to the loader")
        return model

    class Profile:
        qualified_id = "calib.toy@v0"
        profile_hash = "0" * 64

    def calibration(repo_root):
        return Profile(), Path(repo_root) / "toy_items.jsonl", items

    args = JOB.build_parser().parse_args(
        ["--out", "result.json", "--samples-per-cardinality", "1",
         "--e8a-pairs", "2"])
    monkeypatch.setattr(JOB, "STATUS", tmp_path / "status")

    report = JOB.run_entrypoint(
        args, hardware=FakeHardware(torch.device("cpu")),
        teacher_loader=loader, calibration=calibration,
        repo_root=tmp_path, n_layers=6, n_remove=3)

    # The artifact was written where the CLI would write it.
    out = tmp_path / "result.json"
    assert out.is_file()
    written = json.loads(out.read_text())
    # Through JSON, because that is what was written: `mean_seconds_by_cardinality`
    # is keyed by int and JSON has only string keys. Comparing the round-trip
    # asserts "what landed on disk is what was produced" without pretending the
    # coercion did not happen.
    assert written == json.loads(json.dumps(report))

    # And it carries what the grant asks it to report.
    assert report["identities"]["revision_pinned"] is True
    assert report["identities"]["revision"] == JOB.TEACHER_REVISION
    assert report["device"]["name"] == "FakeDevice"
    # Not `> 0`: a toy evaluation is ~8 ms, so the weighted total rounds to 0.0
    # at two decimals. The rate does not round away, and the labelled-wrong flat
    # figure must be present beside the weighted one whatever the scale.
    timing = report["timing"]
    assert timing["weighted_evaluations_per_minute"] > 0
    assert timing["weighted_260_eval_minutes"] >= 0
    assert "flat_cardinality_8_minutes_WRONG" in timing
    assert set(timing["mean_seconds_by_cardinality"]) == {1, 2, 3}
    assert report["vram"]["which_is_the_phase_a_number"] == "production_peak_gib"
    assert len(report["e8a_backend_comparison"]["paired"]) == 2
    assert report["reference_cache_decision"]["budget_fraction"] == 0.66

    # And the success marker fired, which means both stop conditions passed.
    assert "MARKER:ALL_DONE" in (tmp_path / "status").read_text()


def test_the_entrypoint_imports_as_operator_items_from_its_real_owner():
    """The $0.18 line. `as_operator_items` is defined in
    `scripts/autoinit/phase_a_search.py` and nowhere else."""
    import importlib.util

    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "from phase_a_search import as_operator_items" in src
    assert "from aadistill.autoinit.datasets import as_operator_items" not in src

    # And it really is there, rather than merely spelled differently.
    spec = importlib.util.spec_from_file_location(
        "phase_a_search_probe", REPO / "scripts/autoinit/phase_a_search.py")
    assert spec is not None
    owner = (REPO / "scripts/autoinit/phase_a_search.py").read_text()
    assert "def as_operator_items(" in owner
    datasets = (REPO / "src/aadistill/autoinit/datasets.py").read_text()
    assert "def as_operator_items(" not in datasets, (
        "as_operator_items moved; the import in the measurement job needs "
        "re-deriving rather than this test relaxing")


def test_the_entrypoint_refuses_an_unpinned_revision_before_loading_anything():
    """Fail-closed, and BEFORE the teacher is pulled — an unpinned measurement
    should cost nothing, not a model download."""
    args = JOB.build_parser().parse_args(["--teacher-revision", ""])
    loaded = []

    with pytest.raises(SystemExit, match="unpinned Hub HEAD"):
        JOB.run_entrypoint(args, hardware=FakeHardware(torch.device("cpu")),
                           teacher_loader=lambda a, d: loaded.append(1))
    assert not loaded, "the teacher was loaded before the pin was checked"


def test_the_entrypoint_refuses_a_host_run_when_no_hardware_is_injected():
    """The production guard is still there: without CUDA and without an injected
    stand-in, the entrypoint stops rather than measuring the host.

    The loader is stubbed to EXPLODE rather than left at its default. This test
    is the only one that calls `run_entrypoint` with nothing injected, so if the
    guard ever stops firing it would fall through to the real `load_teacher` —
    a 7.6 GB checkpoint and a full CPU measurement inside the $0 suite. A
    mutation experiment did exactly that and hung for 900 s. A $0 test must stay
    $0 even when the thing it is guarding is broken.
    """
    if torch.cuda.is_available():                       # pragma: no cover
        pytest.skip("this box has CUDA; the guard cannot be exercised here")
    args = JOB.build_parser().parse_args([])

    def never(args, device):                            # pragma: no cover
        raise AssertionError(
            "the no-CUDA guard did not fire and the entrypoint reached the "
            "teacher loader; on the real default that is a 7.6 GB load")

    with pytest.raises(SystemExit, match="no CUDA device"):
        JOB.run_entrypoint(args, teacher_loader=never)


def test_main_is_only_parsing_and_a_call_to_the_seam():
    """No orchestration may live in `main()`, or it is untested again."""
    import ast

    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "run_entrypoint" in called
    for forbidden in ("run_measurement", "apply_cpu_budget", "load_teacher",
                      "resolve_calibration", "mark"):
        assert forbidden not in called, (
            f"main() calls {forbidden} directly; that orchestration belongs "
            "behind run_entrypoint where a $0 test can reach it")


def test_the_real_loader_passes_the_pinned_revision_through(monkeypatch):
    """`load_teacher` is the one function the entrypoint test injects past, so
    its body is the one place the seam cannot cover. A mutation dropping the
    revision here passed every other test in this file — this is that hole.

    `from_pretrained` is stubbed, so nothing is downloaded.
    """
    import transformers

    seen = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.update({"name": name, **kw})

            class M:
                config = type("C", (), {"use_cache": True})()

                def to(self, device):
                    seen["device"] = device
                    return self

                def eval(self):
                    return self
            return M()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAuto)
    args = JOB.build_parser().parse_args([])
    model = JOB.load_teacher(args, torch.device("cpu"))

    assert seen["name"] == JOB.TEACHER_ID
    assert seen["revision"] == JOB.TEACHER_REVISION, (
        "the loader dropped the pinned revision; a paid measurement would run "
        "against whatever the Hub published that morning")
    assert seen["dtype"] is torch.bfloat16
    assert seen["device"] == torch.device("cpu")
    assert model.config.use_cache is False, (
        "bypassed_blocks requires use_cache=False and the loader must set it")


def test_the_real_calibration_resolver_runs_and_returns_a_path_not_a_mixture():
    """`resolve_calibration` is the second function the entrypoint test injects
    past, and the second place a defect hid behind an injection point.

    `DOMAIN_BALANCED_V1.resolve()` returns the loaded ROWS. Passing that on as
    the report's `calibration_path` wrote **734,042 characters** of serialized
    mixture — every item, every token id — into a field labelled with a path.
    The measurement itself was unaffected; the artifact was not.

    This executes the real resolver against the real frozen asset, which the dev
    box has, so the import that cost $0.18 is *run* here rather than matched as
    a string.
    """
    profile, path, items = JOB.resolve_calibration(JOB.REPO_ROOT)

    assert isinstance(path, Path), f"got {type(path).__name__}"
    assert path.is_file() and path.suffix == ".jsonl"
    assert len(str(path)) < 200, (
        f"calibration_path is {len(str(path))} chars; resolve() returns rows, "
        "not a filename, and the report field must not carry the mixture")

    # The frozen mixture, unchanged: this test must fail if the profile moves.
    assert profile.qualified_id == "calib.domain_balanced@v1"
    assert len(items) == 67
    assert sum(int(i["input_ids"].shape[1]) - 1 for i in items) == 59_763


def test_the_report_field_carries_the_path_and_not_the_items(tmp_path, monkeypatch):
    """End to end, through the real resolver: what lands in the artifact."""
    model = tiny_teacher(n_layers=4)
    monkeypatch.setattr(JOB, "STATUS", tmp_path / "status")
    args = JOB.build_parser().parse_args(
        ["--out", "r.json", "--samples-per-cardinality", "1", "--e8a-pairs", "2"])

    real_profile, real_path, _ = JOB.resolve_calibration(JOB.REPO_ROOT)

    def calibration(repo_root):
        # The real profile and the real path, with toy items so this stays free.
        return real_profile, real_path, toy_items(model.config.vocab_size)

    report = JOB.run_entrypoint(
        args, hardware=FakeHardware(torch.device("cpu")),
        teacher_loader=lambda a, d: model, calibration=calibration,
        repo_root=tmp_path, n_layers=4, n_remove=2)

    ident = report["identities"]
    assert ident["calibration_path"] == str(real_path)
    assert len(ident["calibration_path"]) < 200
    assert ident["calibration_profile"] == "calib.domain_balanced@v1"
    assert ident["calibration_profile_hash"] == real_profile.profile_hash
    assert len((tmp_path / "r.json").read_bytes()) < 200_000, (
        "the report is enormous; something serialized the mixture into it")


def test_the_real_hardware_object_is_the_default_and_its_dispatch_is_exercised():
    """The third and last injection point.

    Unlike the other two, this one CANNOT be fully covered at $0: the CUDA calls
    inside it need a CUDA device. What is coverable is that the real class is the
    default the paid CLI gets, and that its branch dispatch is correct — so a
    stand-in cannot quietly become the production object, and the non-CUDA
    branches (which a `meta`/`cpu` device takes) do what they claim.

    **The remaining gap is stated rather than papered over:** the three
    `torch.cuda.*` calls are verified by a paid run and by nothing here.
    """
    sig = inspect.signature(JOB.run_entrypoint)
    assert sig.parameters["hardware"].default is None, (
        "hardware defaults to something other than None; the paid CLI must "
        "construct the real Hardware, never inherit a test's stand-in")
    assert sig.parameters["teacher_loader"].default is JOB.load_teacher
    assert sig.parameters["calibration"].default is JOB.resolve_calibration

    hw = JOB.Hardware(torch.device("meta"))
    hw.reset_peak()                      # a no-op off CUDA, and must not raise
    assert hw.name() == "meta"
    assert hw.total_gib() == 0.0
    assert JOB.Hardware.available() is torch.cuda.is_available()

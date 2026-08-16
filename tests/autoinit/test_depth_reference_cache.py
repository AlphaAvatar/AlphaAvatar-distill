"""`depth.causal_kl_greedy_v1`'s reference cache: bounded, and free.

The operator caches the unbypassed parent's logits so that each of the ~260
candidate evaluations costs one ablated forward instead of two. It used to do so
**unconditionally**, in float32, on the host. For the frozen
`calib.domain_balanced@v1` mixture that is 59,763 prediction positions x 151,936
vocabulary x 4 B = **33.8 GiB** per invocation, and the OOM killer took the first
run that ever fed it the real mixture
(`logs/autoinit_phase_a_full_mixture_depth.json`).

`scripts/training/search_depth_map.py` — the E8a script whose algorithm this
operator declares it re-runs — already sized the cache and fell back to
recomputing. The port dropped that. These tests pin the two properties that
make the restored fallback safe to take automatically:

1. it produces **identical** numbers, so a run that falls back is not a
   different experiment;
2. it is **chosen from the memory that actually binds**, not from a host-wide
   number that a container over-reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.operators import depth as depth_module  # noqa: E402
from aadistill.autoinit.operators.base import (  # noqa: E402
    OperatorContext,
    get_implementation,
)

ADAPTER = get_adapter("qwen3")
IMPL = "depth.causal_kl_greedy_v1"


def apply_depth(model, parent_spec, target_spec, items, profile, *, cached: bool):
    """Run the operator with the cache forced on or off."""
    original = depth_module._ReferenceLogits.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.enabled = cached

    depth_module._ReferenceLogits.__init__ = patched
    try:
        return get_implementation(IMPL).execute(OperatorContext(
            adapter=ADAPTER, model=model, parent_spec=parent_spec,
            target_spec=target_spec, profile=profile, calibration_items=items,
            seed=1234))
    finally:
        depth_module._ReferenceLogits.__init__ = original


def test_recomputing_the_reference_gives_bitwise_identical_scores(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """The fallback is automatic, so it must not be a different measurement.

    Not `approx`: the reference is the same deterministic no-grad forward and
    `distortion` upcasts to float32 in chunks either way, so the reduced values
    are the same floats. Anything looser would let a real divergence through.
    """
    hot = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                      profile, cached=True)
    cold = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                       profile, cached=False)

    assert hot.artifacts["reference_cache"]["cached"] is True
    assert cold.artifacts["reference_cache"]["cached"] is False
    assert hot.trace["removal_order"] == cold.trace["removal_order"]
    assert hot.trace["kept_layers"] == cold.trace["kept_layers"]

    hot_rounds = hot.artifacts["search_rounds"]
    cold_rounds = cold.artifacts["search_rounds"]
    assert len(hot_rounds) == len(cold_rounds)
    for a, b in zip(hot_rounds, cold_rounds):
        assert a["table"] == b["table"], (
            "the cached and recomputed reference disagree on a candidate score; "
            "the fallback would silently change the depth map")


def test_the_operator_records_which_memory_path_it_took(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """A run that quietly took the 2x-forward path must say so."""
    outcome = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                          profile, cached=False)
    decision = outcome.artifacts["reference_cache"]
    assert decision["fallback"], "the slow path was taken and left no record"
    assert decision["estimate_bytes"] > 0
    assert decision["budget_fraction"] == pytest.approx(0.66)
    assert decision["headroom_source"]


def test_the_estimate_is_positions_times_vocabulary_times_the_model_dtype(
        teacher, calibration_items):
    """The number the budget is checked against must be the real footprint."""
    cache = depth_module._ReferenceLogits(teacher, calibration_items, "cpu")
    positions = sum(int(i["input_ids"].shape[1]) - 1 for i in calibration_items)
    itemsize = next(teacher.parameters()).dtype.itemsize
    assert cache.estimate_bytes == positions * teacher.config.vocab_size * itemsize

    # And the frozen mixture's real footprint, which is the case that OOMed.
    assert 59_763 * 151_936 * 4 == pytest.approx(36_320_604_672)


def test_a_cache_larger_than_the_budget_is_refused(
        teacher, calibration_items, monkeypatch):
    """The decision must actually depend on the measurement."""
    monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                        lambda: (10, "test"))
    assert depth_module._ReferenceLogits(
        teacher, calibration_items, "cpu").enabled is False

    monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                        lambda: (10 * 2**40, "test"))
    assert depth_module._ReferenceLogits(
        teacher, calibration_items, "cpu").enabled is True


def test_the_cgroup_grant_wins_over_the_hosts_memory(tmp_path, monkeypatch):
    """Inside a container `/proc/meminfo` reports the HOST.

    This project has already shipped one bug from trusting a host-wide number
    over the cgroup limit (`nproc` versus the CPU quota). The smaller of the two
    is the one that can kill the process.
    """
    host = tmp_path / "meminfo"
    host.write_text("MemTotal:       800000000 kB\nMemAvailable:   700000000 kB\n")
    cgroup_max = tmp_path / "memory.max"
    cgroup_max.write_text(str(8 * 2**30))
    cgroup_cur = tmp_path / "memory.current"
    cgroup_cur.write_text(str(2 * 2**30))

    real_read = Path.read_text

    def routed(self, *args, **kwargs):
        mapping = {"/proc/meminfo": host,
                   "/sys/fs/cgroup/memory.max": cgroup_max,
                   "/sys/fs/cgroup/memory.current": cgroup_cur}
        target = mapping.get(str(self))
        if target is not None:
            return real_read(target, *args, **kwargs)
        if str(self).startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", routed)
    available, source = depth_module._host_available_memory_bytes()
    assert source == "cgroup.v2"
    assert available == 6 * 2**30, (
        "the host's 700 GB was used instead of the cgroup's 6 GiB of headroom")


def test_an_unlimited_cgroup_falls_through_to_meminfo(tmp_path, monkeypatch):
    host = tmp_path / "meminfo"
    host.write_text("MemAvailable:   1048576 kB\n")
    unlimited = tmp_path / "memory.max"
    unlimited.write_text("max")

    real_read = Path.read_text

    def routed(self, *args, **kwargs):
        if str(self) == "/proc/meminfo":
            return real_read(host, *args, **kwargs)
        if str(self) == "/sys/fs/cgroup/memory.max":
            return real_read(unlimited, *args, **kwargs)
        if str(self).startswith("/sys/fs/cgroup"):
            raise FileNotFoundError(str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", routed)
    available, source = depth_module._host_available_memory_bytes()
    assert (available, source) == (2**30, "proc.meminfo")


def test_the_reference_is_the_unbypassed_parent_even_when_recomputed(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """The one ordering bug the restructure could have introduced.

    Recomputing the reference inside the scoring loop puts it next to the
    ablated forward. Taking it *inside* the bypass would make the objective
    KL(bypassed || bypassed) — identically zero, and every candidate would tie.
    """
    outcome = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                          profile, cached=False)
    scores = [row["score"] for r in outcome.artifacts["search_rounds"]
              for row in r["table"]]
    assert scores, "no candidate was scored"
    assert any(s > 0 for s in scores), (
        "every distortion is zero; the reference was taken with the blocks "
        "already bypassed")
    assert len(set(scores)) > 1, "every candidate scored identically"


def test_a_cached_reference_is_computed_once_per_item(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """The whole point of the cache: (evals + 1) * n_items, not 2 * evals."""
    calls: list[str] = []
    real_forward = depth_module._forward_logits

    def counting(model, item, device, skip=frozenset()):
        if not skip:
            calls.append(item["item_id"])
        return real_forward(model, item, device, skip)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(depth_module, "_forward_logits", counting)
    try:
        apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                    profile, cached=True)
    finally:
        monkeypatch.undo()
    assert sorted(calls) == sorted(i["item_id"] for i in calibration_items), (
        "the cache did not hold: an item's reference was recomputed")


def test_the_memory_decision_stays_out_of_the_deterministic_trace(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """`trace` is compared field-for-field between two invocations.

    Recording free memory there made `test_operator_is_deterministic` fail on
    an environment reading, not on a result. The decision is a byproduct, so it
    lives in `artifacts`.
    """
    hot = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                      profile, cached=True)
    cold = apply_depth(teacher, teacher_spec, target_spec, calibration_items,
                       profile, cached=False)
    assert "reference_cache" not in hot.trace
    assert hot.trace == cold.trace, (
        "the two memory paths produced different traces; the trace would no "
        "longer describe the result alone")
    assert hot.artifacts["reference_cache"] != cold.artifacts["reference_cache"]


def test_the_headroom_probe_measures_the_host_not_the_accelerator(monkeypatch):
    """The cache is `.cpu()`, so free VRAM is the wrong number.

    E8a kept its cache on the accelerator, so `torch.cuda.mem_get_info` was the
    right probe there. Carried over unchanged it would have compared a 16.9 GiB
    host allocation against 38 GiB of free L40S VRAM and cached regardless of
    how much host memory the cgroup actually granted.
    """
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: called.append(1) or (10 * 2**40, 10 * 2**40))
    _, source = depth_module._host_available_memory_bytes()
    assert not called, "the accelerator was measured for a host-resident cache"
    assert source in {"cgroup.v2", "cgroup.v1", "proc.meminfo", "unknown"}

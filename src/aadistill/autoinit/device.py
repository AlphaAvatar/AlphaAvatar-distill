"""The device contract for the Phase-A Stage-1 search path.

Three paid pods have now been lost to one class of defect: code that assumes a
model and a tensor share a device, which is true on the CPU-only dev box and
false on a GPU. Attempt 6 died in the search's reload validation; attempt 7 died
in the activation collector, in different code, having got strictly further.
Patching whichever site the last pod reached only reveals the next, so the rule
is written down here, in one place, and the sites are made to follow it.

Four categories, deliberately distinguished because they have different answers:

**1. Model execution device.** Where a model's parameters actually are. It is
read from the weights (:func:`model_device`), never assumed from a config field:
an operator's child comes from ``ChildBuilder`` -> ``build_student``, which sets
the dtype and does NOT place the model, so a parent on CUDA routinely coexists
with a freshly built child on the host. ``SearchConfig.device`` is the device the
search *intends* to run on; it is not evidence about any particular object.

**2. Persistent activation/statistics cache device: HOST, always.**
``StatsCache`` holds one entry and a 4B parent's statistics are **1.81 GiB**,
almost all of it ``res_sqsum`` at ``(L+1, H, H)`` float64. That is a cache, not a
working set: it survives across operator invocations, and pinning it in VRAM
would hold 1.81 GiB for the whole expansion of a parent so that two operators
can avoid one transfer. Accumulation still happens on the model's device — doing
it on the host would push every ``X^T X`` across PCIe, roughly 1.85 GiB per
calibration item — and the result is transferred to the host **once**, at
:meth:`ActivationStatsCollector.state`. A consumer that needs the statistics for
compute moves them back explicitly with :func:`stats_to`, for the duration of one
invocation.

**3. Ephemeral tensors, indices and projections that interact with model
parameters.** Created on, or explicitly moved to, the device of the parameters
they meet. An index built with ``torch.tensor([...])`` is on the host whatever
the weight it indexes; some torch ops tolerate that and some raise, which is a
worse property than either.

**4. The serialization/artifact boundary.** ``adapter.save`` writes from wherever
the model is. The canonical reload is placed explicitly, validated against the
produced model **on the produced model's device** so the save/reload comparison
runs on one numerical backend, and only then moved to the search device to be
measured.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

#: Bumped when the contract's *meaning* changes, so a run record can cite which
#: rules it executed under.
DEVICE_CONTRACT_VERSION = 1
DEVICE_CONTRACT_ID = f"autoinit.stage1_device_contract@v{DEVICE_CONTRACT_VERSION}"


def model_device(model: Any) -> torch.device:
    """Where a model's weights actually are, not where they were asked to be.

    Falls back to CPU only for a parameterless module, which the search does not
    produce.
    """
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def stats_to(state: Mapping[str, torch.Tensor], device: Any) -> dict[str, torch.Tensor]:
    """Move a statistics dict to a compute device. The explicit transfer.

    The persistent cache stays on the host; this is the per-invocation working
    copy, and it is freed when the operator returns. Non-tensor values pass
    through untouched so a caller can hand this a mixed record without losing
    fields.
    """
    target = torch.device(device)
    return {k: (v.to(target) if isinstance(v, torch.Tensor) else v)
            for k, v in state.items()}


def stats_bytes(state: Mapping[str, torch.Tensor]) -> int:
    """What a statistics dict costs, for the memory accounting the contract
    requires before anything is pinned anywhere."""
    return sum(v.numel() * v.element_size()
               for v in state.values() if isinstance(v, torch.Tensor))


def as_dict() -> dict[str, Any]:
    """The contract, serializable, so a session record can carry it."""
    return {
        "contract": DEVICE_CONTRACT_ID,
        "version": DEVICE_CONTRACT_VERSION,
        "model_execution_device": (
            "read from the weights via model_device(); SearchConfig.device is an "
            "intent, not evidence about any object. A parent on CUDA routinely "
            "coexists with a freshly built child on the host."),
        "persistent_stats_cache_device": (
            "HOST, always. One entry, 1.81 GiB at the 4B parent, dominated by "
            "res_sqsum (L+1, H, H) float64. Accumulated on the model's device, "
            "transferred to the host once in state(); consumers move a working "
            "copy back with stats_to() for one invocation."),
        "ephemeral_tensors_indices_projections": (
            "created on, or explicitly moved to, the device of the parameters "
            "they interact with"),
        "serialization_artifact_boundary": (
            "save from wherever the model is; reload placed explicitly and "
            "validated against the produced model ON THE PRODUCED MODEL'S "
            "DEVICE, so the save/reload comparison is one numerical backend; "
            "only then moved to the search device to be measured"),
        "scope": (
            "the Phase-A Stage-1 GPU search path: search.py, the five frozen "
            "operators and their shared helpers, the directly-used "
            "aadistill.init helpers, and the Qwen3 adapter/build/load boundary. "
            "NOT a whole-project audit and it does not reopen the frozen "
            "science."),
    }

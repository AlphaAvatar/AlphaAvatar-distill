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

**5. Fresh tensor factories on the device-coupled path (added 2026-08-19).**
``torch.zeros``/``ones``/``empty``/``eye``/``full``/``tensor``/``arange`` default
to the *host*, whatever they are about to meet. Every such call in this scope is
therefore one of two things, and which one must be legible at the call site:

* **device-coupled** — it participates in arithmetic or indexing with model
  parameters or with a statistics working copy. It must name a device derived
  from what it meets (``device=state["residual_sqsum"].device``,
  ``device=proj.device``, ``device=w.device``), or be a ``*_like`` form that
  inherits one.
* **intentional host-only** — a control or diagnostic assembled from Python
  scalars and reduced back to Python scalars, never meeting a parameter. These
  are correct as they are and must **not** be mechanically moved;
  ``attention.py``'s per-head ``scores`` and ``sandwich.select_q_heads``'s are
  the worked examples.

  ``depth.py``'s distortion reduction was listed here as a third example until
  2026-08-19. It was not one: it ran on the host because the port of
  ``scripts/training/search_depth_map.py`` inserted ``.cpu()``, and E8a runs that
  reduction on the accelerator. Attempt 10 spent $11.43 discovering it. The
  reduction is device-resident again, and the lesson is that "intentional
  host-only" is a claim to check against the implementation being ported, not a
  label to apply to whatever is already on the host.

This category exists because category 3 already stated the rule and nothing
checked it. Attempt 9 died at $0.34 on ``project.py``'s ``avg``, allocated with a
dtype and no device two call levels below the operator; a second, latent instance
(``torch.eye`` in the same function's orthonormality diagnostic) would have cost
the next session. The :class:`~tests.autoinit.device_split.HostCacheTensor` split
cannot observe this class by construction: it labels the persistent cache, and
after ``stats_to`` returns plain tensors a freshly allocated host tensor is
plain too. Placement is asserted instead by intercepting the factory calls —
``tests/autoinit/factory_placement.py`` — which tests the *intent*, not whether
the arithmetic happens to succeed on a one-device box.
"""

from __future__ import annotations

import os
from pathlib import Path
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
        "fresh_tensor_factories": (
            "added 2026-08-19 after Phase-A attempt 9. A factory on this path is "
            "either DEVICE-COUPLED, and must name a device derived from what it "
            "meets (or be a *_like form that inherits one), or INTENTIONALLY "
            "HOST-ONLY, assembled from Python scalars and reduced back to them "
            "without meeting a parameter -- and those must not be mechanically "
            "moved. Category 3 stated the rule and nothing checked it; the "
            "HostCacheTensor split cannot observe this class, because after "
            "stats_to a freshly allocated host tensor is as plain as a moved "
            "one. Asserted by intercepting the factory calls: "
            "tests/autoinit/factory_placement.py"),
        "scope": (
            "the Phase-A Stage-1 GPU search path: search.py, the five frozen "
            "operators and their shared helpers, the directly-used "
            "aadistill.init helpers, and the Qwen3 adapter/build/load boundary. "
            "NOT a whole-project audit and it does not reopen the frozen "
            "science."),
    }


# ---------------------------------------------------------------------------
# CPU budget. Added 2026-08-19 after Phase-A attempt 10.
# ---------------------------------------------------------------------------

def cpu_budget(cap: int = 16) -> tuple[int, str]:
    """Usable CPUs: the **cgroup grant**, never what the kernel advertises.

    `autoinit_preflight_setup.sh` has computed this correctly since E8b and
    applies it — to the test suite only. The driver inherited nothing, so on
    attempt 10 torch sized its pools from the 128 vCPUs the container could see
    while the cgroup granted **13**: 192 threads on 13 CPUs, measured, through a
    bandwidth-bound reduction whose BLAS barriers make every thread wait for the
    slowest.

    NOT `os.cpu_count()`, and not `nproc`: coreutils documents that `nproc`
    honours `OMP_NUM_THREADS`, so once anything sets that variable `nproc` stops
    reporting the machine and starts reporting our own cap.

    Returns `(n, source)` so a run record can say which limit bound.
    """
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", "v2"),                       # cgroup v2
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "v1"),          # cgroup v1
    ):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        try:
            if parse == "v2":
                quota_s, period_s = raw.split()
                if quota_s == "max":
                    continue
                quota, period = int(quota_s), int(period_s)
            else:
                quota = int(raw)
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
                             .read_text().strip())
            if quota > 0 and period > 0:
                return max(1, min(quota // period, cap)), f"cgroup.{parse}"
        except (OSError, ValueError):
            continue

    # No quota: a bare host, where the affinity mask is the truth.
    try:
        return max(1, min(len(os.sched_getaffinity(0)), cap)), "sched_getaffinity"
    except AttributeError:                                   # pragma: no cover
        return max(1, min(os.cpu_count() or 1, cap)), "cpu_count"


def apply_cpu_budget(cap: int = 16) -> dict[str, Any]:
    """Hold torch to the CPUs we were actually granted. Returns what it did.

    Called by the driver before any heavy work. `torch.set_num_threads` is the
    part that binds: the environment variables are set too, because a subprocess
    or a library that reads them at import would otherwise re-derive the wrong
    number from the same visible-CPU count.
    """
    n, source = cpu_budget(cap)
    before = torch.get_num_threads()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(n, 4)))
    except RuntimeError:
        pass                    # already initialized; the intra-op cap is what matters
    return {"threads": n, "source": source, "torch_threads_before": before,
            "torch_threads_after": torch.get_num_threads(),
            "visible_cpus": os.cpu_count()}

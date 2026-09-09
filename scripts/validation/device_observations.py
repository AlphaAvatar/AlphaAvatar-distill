"""Watch where tensors are while the real treatment operator runs.

Attempt 9 died at `attention_activation.apply` -> `head_write_energy`, on
statistics that were on the host and an `o_proj.weight` that was on `cuda:0`.
The repair is a per-invocation working copy. Nothing on a single-device box can
tell whether that repair works, because there both operands are trivially
co-located — so what this module does is record the placements *while the real
operator executes on a real device*, and let the caller assert them.

**It observes; it does not substitute.** Each wrapper calls the production
function and returns its real value unchanged. The one exception is
`torch.empty`, which is intercepted for the duration of a single
`head_write_energy` call to learn what device the score buffer was ALLOCATED on
— the arithmetic that follows would succeed either way on one device, so the
allocation intent is the only thing that carries the information.

A wrapper that never fires must not be read as agreement. Every observation is
counted, and `report()` marks a proof `observed: false` when its site was never
reached — a check nobody ran is not a check that passed. That distinction is the
whole reason this file exists: the CPU regressions all "passed" while the defect
was live.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any


def _devices(obj: Any) -> list[str]:
    """Every tensor device inside a state mapping, as strings."""
    import torch

    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                out.append(str(v.device))
    elif isinstance(obj, torch.Tensor):
        out.append(str(obj.device))
    return out


def _kind(device: str) -> str:
    """`cuda:0` -> `cuda`. Placement is about the backend, not the ordinal."""
    return device.split(":")[0]


@dataclass
class DeviceObservations:
    """What was seen, per site, during one operator execution."""

    snapshot_devices: list[list[str]] = field(default_factory=list)
    working_copy_devices: list[list[str]] = field(default_factory=list)
    energy_operands: list[dict] = field(default_factory=list)
    score_alloc_devices: list[str] = field(default_factory=list)
    score_return_devices: list[str] = field(default_factory=list)

    # --- the six claims, each derived rather than assumed --------------------

    def report(self, compute: str) -> dict:
        """The proofs, with `observed` separate from `holds`.

        `holds` is only meaningful when `observed` is true; a site that never
        ran yields `holds: false` so that an unreached proof can never be
        summed into a pass.
        """
        host, dev = "cpu", _kind(compute)

        def proof(observed: bool, holds: bool, detail: Any, claim: str) -> dict:
            return {"claim": claim, "observed": observed,
                    "holds": bool(observed and holds), "detail": detail}

        return {
            "collector_state_is_the_host_snapshot": proof(
                bool(self.snapshot_devices),
                all(_kind(d) == host for ds in self.snapshot_devices for d in ds),
                self.snapshot_devices,
                "AttentionHeadStatsCollector.state() returns a host-resident "
                "snapshot -- the persistent cache form, by design"),
            "one_working_copy_reaches_the_model_device": proof(
                len(self.working_copy_devices) == 1,
                bool(self.working_copy_devices)
                and all(_kind(d) == dev for d in self.working_copy_devices[0]),
                {"n_calls": len(self.working_copy_devices),
                 "devices": self.working_copy_devices},
                "stats_to builds EXACTLY ONE working copy on the compute "
                "device; more than one would mean the transfer moved into a "
                "loop and the (layers, heads, d, d) tensor is duplicated"),
            "statistics_and_o_proj_are_co_located": proof(
                bool(self.energy_operands),
                all(o["stats"] == o["o_proj"] for o in self.energy_operands),
                self.energy_operands,
                "at every head_write_energy call the attention statistics and "
                "o_proj.weight are on the same device -- the exact pair that "
                "raised on the L40S"),
            "score_vector_is_allocated_on_the_operand_device": proof(
                bool(self.score_alloc_devices),
                all(_kind(d) == dev for d in self.score_alloc_devices),
                self.score_alloc_devices,
                "the per-head score buffer is created on w.device, not on the "
                "host default -- otherwise the first scores[h] assignment is a "
                "second, latent cross-device use"),
            "returned_score_vector_is_host_resident": proof(
                bool(self.score_return_devices),
                all(_kind(d) == host for d in self.score_return_devices),
                self.score_return_devices,
                "the small returned vector is moved to the host ONCE, after "
                "the whole vector is built; the ranking and tie-break that "
                "follow are host-side and unchanged"),
        }

    def all_hold(self, compute: str) -> bool:
        return all(p["holds"] for p in self.report(compute).values())


@contextlib.contextmanager
def observing(module):
    """Install the observers around one execution, then remove them.

    `module` is `aadistill.initialization.operators.attention_activation`,
    passed in rather than imported here so this file names no experiment and a
    test can drive it against the same production module. Both `stats_to` and
    `head_write_energy` are bound INTO that module's namespace by its imports,
    which is why patching the module attributes is what the operator's own
    lookups resolve to.
    """
    import torch

    obs = DeviceObservations()
    real_state = module.AttentionHeadStatsCollector.state
    real_stats_to = module.stats_to
    real_energy = module.head_write_energy
    real_empty = torch.empty

    def state(self):
        out = real_state(self)
        obs.snapshot_devices.append(_devices(out))
        return out

    def stats_to(state_, device):
        out = real_stats_to(state_, device)
        obs.working_copy_devices.append(_devices(out))
        return out

    def head_write_energy(state_, layer, o_proj_weight, num_heads, head_dim):
        obs.energy_operands.append({
            "layer": int(layer),
            "stats": str(state_["attn_head_sqsum"].device),
            "o_proj": str(o_proj_weight.device),
        })
        # Only for the duration of this call: `torch.empty` is used everywhere,
        # and a global patch would record allocations that have nothing to do
        # with the score buffer.
        seen: list[str] = []

        def empty(*a, **kw):
            t = real_empty(*a, **kw)
            seen.append(str(t.device))
            return t

        torch.empty = empty
        try:
            out = real_energy(state_, layer, o_proj_weight, num_heads, head_dim)
        finally:
            torch.empty = real_empty
        # The score buffer is the one sized (num_heads,); anything else this
        # call allocated is not the tensor the claim is about.
        obs.score_alloc_devices.extend(seen)
        obs.score_return_devices.append(str(out.device))
        return out

    module.AttentionHeadStatsCollector.state = state
    module.stats_to = stats_to
    module.head_write_energy = head_write_energy
    try:
        yield obs
    finally:
        module.AttentionHeadStatsCollector.state = real_state
        module.stats_to = real_stats_to
        module.head_write_energy = real_energy
        torch.empty = real_empty

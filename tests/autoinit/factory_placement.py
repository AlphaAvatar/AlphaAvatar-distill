"""Observe where a fresh tensor was *told* to go, on a box with one device.

The companion to `device_split.py`, and deliberately its dual.
`HostCacheTensor` labels the persistent statistics cache and catches an unmoved
cache tensor meeting a model-side one. It cannot catch what killed Phase-A
attempt 9: after `stats_to` the working copy is a plain tensor, so a *freshly
allocated* host tensor mixed into the same arithmetic is plain as well, and
there is nothing for a label to bite on.

On one device the consequence is invisible — every tensor is on it and the
arithmetic is correct — so this asserts the **intent** instead. A
`TorchFunctionMode` records each factory call and whether it named a device.
`torch.zeros(n, n, dtype=torch.float64)` and
`torch.zeros(n, n, dtype=torch.float64, device=state[...].device)` are
indistinguishable by their results here and trivially distinguishable by this.

Not a FakeTensor or meta-device framework: ~40 lines, no dispatch of its own, and
used only by `test_stage1_factory_placement.py`.
"""

from __future__ import annotations

import torch
from torch.overrides import TorchFunctionMode

#: The `torch.*` entry points that allocate storage from scratch rather than
#: deriving it from an operand. `*_like` forms are deliberately absent: they
#: inherit the device of their argument and are correct by construction.
FACTORY_NAMES = frozenset({
    "zeros", "ones", "empty", "eye", "full", "tensor", "arange", "linspace",
    "rand", "randn", "randint", "as_tensor", "from_numpy",
})


class FactoryCall(tuple):
    """`(name, device)` — what was allocated and where it was told to go."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return self[0]

    @property
    def device(self):
        return self[1]

    @property
    def placed(self) -> bool:
        return self[1] is not None

    def __repr__(self) -> str:                       # pragma: no cover - debug
        return f"torch.{self[0]}(device={self[1]!r})"


class RecordFactories(TorchFunctionMode):
    """Record every fresh-allocation call made inside the `with` block."""

    def __init__(self) -> None:
        self.calls: list[FactoryCall] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        # Module check keeps `Tensor.new_zeros` and friends out: those derive
        # their device from the tensor they are called on.
        if name in FACTORY_NAMES and getattr(func, "__module__", None) == "torch":
            self.calls.append(FactoryCall((name, kwargs.get("device"))))
        return func(*args, **kwargs)

    def unplaced(self, *, ignore: frozenset[str] = frozenset()) -> list[FactoryCall]:
        """Calls that named no device, excluding names the caller has
        classified as intentionally host-only."""
        return [c for c in self.calls if not c.placed and c.name not in ignore]

"""Make "the cache is on another device" real on a box that has one device.

The Phase-A device defects are unobservable on the dev box for a structural
reason: with one device, every tensor is on it. Attempts 6 and 7 both certified
green at $0 and then died on a GPU, in different code, for the same reason.

So the split is modelled instead of performed. A statistics tensor is wrapped in
`HostCacheTensor`, which carries a logical "I am on the persistent cache device"
label. Any torch op that mixes an unmoved cache tensor with a model-side tensor
raises `CrossDeviceUse` — which is what CUDA would have done. Calling `.to(...)`
on one returns a plain tensor: that is the explicit transfer at the compute
boundary, and after it the value is model-side and may be used freely.

Labels propagate through arithmetic by the ordinary subclass rule, so a
projection derived from unmoved statistics is still unmoved, and multiplying it
by a parameter is caught at exactly the line CUDA would have caught it.

This is a ~60-line test helper, not a framework, and it is used only by
`test_search_operator_device_split.py`.
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten


class CrossDeviceUse(RuntimeError):
    """An unmoved cache tensor met a model-side tensor in one operation."""


#: Ops that inspect rather than compute, and that a caller may legitimately
#: apply across the split. `to`/`cpu`/`cuda` are the transfer itself — but ONLY
#: when they name a device. `x.to(torch.float64)` is a dtype cast, and treating
#: it as a transfer silently released the label on the first `.to(torch.float64)`
#: inside a projection helper, which made this instrument pass a defect it was
#: built to catch. Verified by
#: `test_the_split_actually_bites_when_the_transfer_is_removed`.
_ALWAYS_TRANSFER = {torch.Tensor.cpu, torch.Tensor.cuda}
_METADATA = {torch.Tensor.__repr__, torch.Tensor.size, torch.Tensor.dim,
             torch.Tensor.numel, torch.Tensor.element_size,
             torch.Tensor.__format__, torch.Tensor.item}


class HostCacheTensor(torch.Tensor):
    """A tensor that behaves as if it lived on the persistent cache device."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _ALWAYS_TRANSFER or (func is torch.Tensor.to
                                        and _names_a_device(args[1:], kwargs)):
            # The explicit transfer: the result is model-side from here on.
            plain = super().__torch_function__(
                func, (torch.Tensor,), tuple(_unwrap(a) for a in args),
                {k: _unwrap(v) for k, v in kwargs.items()})
            return plain.as_subclass(torch.Tensor)
        if func not in _METADATA:
            flat, _ = tree_flatten((args, kwargs))
            tensors = [a for a in flat if isinstance(a, torch.Tensor)]
            cached = [a for a in tensors if isinstance(a, HostCacheTensor)]
            model_side = [a for a in tensors if not isinstance(a, HostCacheTensor)]
            if cached and model_side:
                raise CrossDeviceUse(
                    f"{getattr(func, '__name__', func)} mixes a statistics tensor "
                    f"still on the persistent cache device with "
                    f"{len(model_side)} model-side tensor(s). On a GPU this is "
                    "'Expected all tensors to be on the same device'. Move the "
                    "working copy explicitly at the compute boundary — see "
                    "aadistill.autoinit.device.stats_to.")
        return super().__torch_function__(func, types, args, kwargs)


def _names_a_device(args, kwargs) -> bool:
    """Did this `.to(...)` ask for a device, or only for a dtype?"""
    if "device" in kwargs:
        return kwargs["device"] is not None
    for a in args:
        if isinstance(a, torch.device):
            return True
        if isinstance(a, str):
            try:
                torch.device(a)
            except (RuntimeError, ValueError):
                return False
            return True
        if isinstance(a, torch.Tensor):      # `x.to(other_tensor)` copies both
            return True
    return False


def _unwrap(x):
    return x.as_subclass(torch.Tensor) if isinstance(x, HostCacheTensor) else x


def on_cache_device(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Label a statistics dict as living on the persistent cache device."""
    return {k: HostCacheTensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in state.items()}

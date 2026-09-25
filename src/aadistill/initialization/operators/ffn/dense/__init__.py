"""Dense feed-forward blocks: one gate/up/down circuit per block.

An operator here may assume every intermediate neuron is evaluated for every
token, so per-neuron activation statistics describe the whole block. A
mixture-of-experts FFN breaks that assumption — its neurons are routed, so an
average over tokens is an average over a different denominator per expert — and
it gets its own topology package when one is implemented.

Importing this package registers nothing.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()

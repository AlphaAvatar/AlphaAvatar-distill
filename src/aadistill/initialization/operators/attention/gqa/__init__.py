"""Grouped-query attention: many query heads share one key/value head.

What every operator here may assume, and what an adapter must therefore
provide: query heads outnumber KV heads by a whole factor, each head owns a
contiguous ``head_dim`` slice of the concatenated attention output, and
reducing query heads leaves the KV heads, the grouping and the RoPE basis
intact. Those assumptions are what make the shared selection topology in
:mod:`._common` correct, and they are exactly what MLA or linear attention
would not satisfy.

Importing this package registers nothing.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()

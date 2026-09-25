"""ATTENTION operators, organised by the head topology they transform.

The kind is ``ATTENTION``; the sub-packages are **topologies** — structural
mechanisms an adapter declares through a capability, not model families. A
Qwen3 adapter and a future Llama adapter both declare
``Capability.ATTENTION_GQA`` and therefore both reach the operators under
:mod:`.gqa` unchanged. ``attention/qwen3/`` would put family knowledge on the
wrong side of that boundary; ``.q_proj`` and ``.o_proj`` stay in the adapter.

Further topologies (MHA, MLA, linear attention) get their own sub-package when
the first real implementation needs one, and not before.

Importing this package registers nothing — see
:mod:`aadistill.initialization.operators.register`.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()

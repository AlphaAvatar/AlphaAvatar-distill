"""Architecture adapters.

Importing this package registers the adapters the project ships. A family
defined elsewhere registers itself by calling
``aadistill.autoinit.arch.register_adapter`` — no edit here and no edit to the
search engine.
"""

from .qwen3 import QWEN3_ADAPTER, Qwen3Adapter

__all__ = ["QWEN3_ADAPTER", "Qwen3Adapter"]

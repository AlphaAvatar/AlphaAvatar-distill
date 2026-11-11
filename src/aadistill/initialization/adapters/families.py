"""Hugging Face model types this package defines itself, and their explicit registration.

A checkpoint of one of these loads through ``AutoModelForCausalLM`` only after
its family's ``register_auto_classes`` ran -- something a caller DOES, never an
import side effect (the adapter bootstrap rule). A loader that reads arbitrary
checkpoints (a trainer resuming one, an evaluation, an engine check) calls
:func:`register_custom_model_families` first.
"""
from __future__ import annotations

import importlib

#: ``model_type`` -> the module whose ``register_auto_classes`` resolves it
CUSTOM_MODEL_FAMILIES = {
    "aad_projected_residual": "aadistill.initialization.adapters.projected_residual",
}

__all__ = ["CUSTOM_MODEL_FAMILIES", "register_custom_model_families"]


def register_custom_model_families() -> tuple[str, ...]:
    """Register every custom model type with the Auto classes. Idempotent."""
    for module in CUSTOM_MODEL_FAMILIES.values():
        importlib.import_module(module).register_auto_classes()
    return tuple(CUSTOM_MODEL_FAMILIES)

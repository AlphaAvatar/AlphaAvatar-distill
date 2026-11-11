"""The PYTHONPATH a vLLM interpreter needs to load this repository's custom model families.

vLLM loads its general plugins in every one of its processes (the front end, the
engine core, each worker) from the entry points ``importlib.metadata`` finds on
``sys.path``. This repository declares one (``vllm_plugin/``, a dist-info with the
entry point ``aadistill.initialization.adapters.projected_residual_vllm:register``) without
installing anything: putting ``src`` and that directory on the interpreter's
``PYTHONPATH`` through the environment reaches every vLLM process whichever start
method it uses. A stock model is unaffected.
"""
from __future__ import annotations

from pathlib import Path

#: the plugin metadata directory, relative to the repository root
PLUGIN_DIR = "scripts/shared/rollout/vllm_plugin"


def vllm_plugin_pythonpath(repo_root, existing: str | None = None) -> str:
    """``src`` and the plugin metadata directory, ahead of ``existing``."""
    root = Path(repo_root)
    parts = [str(root / "src"), str(root / PLUGIN_DIR)]
    if existing:
        parts.append(existing)
    return ":".join(parts)

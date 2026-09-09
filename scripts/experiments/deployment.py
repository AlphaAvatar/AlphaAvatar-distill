"""Deployment facts this repository's sessions run under.

The application layer. `aadistill.infrastructure` holds the mechanism -- how a
session is priced, launched, watched and torn down -- and must not know where
this deployment installed its provider CLI or which artifact store it uses.

Both used to be read from `configs/` by `src/aadistill` at import time, which
made the reusable core reach into the repository it was supposed to be
independent of: a second deployment would have had to place files at paths the
framework chose.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PROVIDER_CLI_CONFIG = REPO / "configs/infrastructure/provider_cli.json"
ARTIFACT_STORE_CONFIG = REPO / "configs/infrastructure/artifact_store.json"
POD_IMAGE_CONFIG = REPO / "configs/infrastructure/pod_image.json"

#: The provider CLI's name on PATH, then the fallback locations this deployment
#: declares. An empty fallback list is legitimate: it means the CLI is expected
#: on PATH and this deployment would rather fail than search a home directory.
_CLI = json.loads(PROVIDER_CLI_CONFIG.read_text()) if PROVIDER_CLI_CONFIG.is_file() else {}
PROVIDER_CLI_NAME: str = _CLI.get("name", "runpodctl")


def provider_cli_candidates() -> tuple[str, ...]:
    """Ordered places to look for the provider CLI, PATH first.

    Resolved by the APPLICATION and handed to the runner, so the runner picks
    the first that exists without knowing any of these names.
    """
    on_path = shutil.which(PROVIDER_CLI_NAME)
    fallbacks = tuple(os.path.expanduser(p)
                      for p in _CLI.get("fallback_paths", ()))
    return ((on_path,) if on_path else ()) + fallbacks


_IMAGE = json.loads(POD_IMAGE_CONFIG.read_text())

#: The image layout every session in THIS deployment runs inside. The runner
#: held `WS = "/workspace"`, `REPO = f"{WS}/aad"` and `--min-cuda-version 13.0`
#: as module constants, so a second image could only be supported by patching
#: the framework's globals.
#:
#: Spread into `ExecutionCommands` by each launcher, so a session that runs a
#: different image says so in its own declaration.
POD_IMAGE = {
    "workspace_root": _IMAGE["workspace_root"],
    "checkout_root": _IMAGE["checkout_root"],
    "remote_python": _IMAGE["remote_python"],
    "min_cuda_version": _IMAGE["min_cuda_version"],
}


def deployment_commands(**overrides) -> dict:
    """This deployment's image facts, with a caller's explicit values winning."""
    return {**POD_IMAGE, "provider_cli_candidates": provider_cli_candidates(),
            **overrides}


#: WHICH artifact repository a `RelayInput` means when it does not say.
#: `RelayInput.repo` had this as a module-level default computed at import from
#: `configs/`, so importing the core read the repository.
MAIN_RELAY: str = json.loads(ARTIFACT_STORE_CONFIG.read_text())["main_relay"]

__all__ = ["ARTIFACT_STORE_CONFIG", "MAIN_RELAY", "POD_IMAGE",
           "POD_IMAGE_CONFIG", "PROVIDER_CLI_CONFIG", "PROVIDER_CLI_NAME",
           "deployment_commands", "provider_cli_candidates"]

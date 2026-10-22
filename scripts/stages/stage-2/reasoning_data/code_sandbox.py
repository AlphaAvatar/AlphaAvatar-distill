"""Where teacher-written code may run: a probed sandbox, never the bare host.

Teacher programs are executed only by the $0 verification step, never on the
generation pod: generation writes text and nothing else, so the pod's provider
credentials, its network and its mounted volume never meet model-written code.
On the verifying machine the program runs inside `aadistill.data.code_exec`'s
bubblewrap sandbox, and only after a canary run inside that sandbox shows it
reaches no network interface, no environment variable and none of the paths
below -- every entry of the home directory (the interpreter's own prefix is the
one thing bound in), the repository, the artifact tree and the usual volume
mount points. A machine where the probe fails verifies no teacher code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from stages.reasoning_data import assets as A

from aadistill.data.code_exec import Limits, Sandbox, bwrap_sandbox, probe_sandbox

MOUNTS = ("/workspace", "/runpod-volume", "/mnt", "/data", "/root")
CREDENTIALS = (".runpod/config.toml", ".cache/huggingface/token", ".git-credentials",
               ".ssh", ".netrc", ".config/gh")


def protected_paths() -> list[str]:
    home = Path.home()
    prefix = Path(os.path.realpath(sys.executable)).parents[1]
    out = {str(home / c) for c in CREDENTIALS}
    if home.is_dir():
        for entry in home.iterdir():
            if entry != prefix and entry not in prefix.parents:
                out.add(str(entry))
    out |= {str(A.REPO), str(A.REPO / "artifacts"), *MOUNTS}
    return sorted(out)


def verified_sandbox(limits: Limits = Limits(wall_seconds=10, cpu_seconds=5, memory_mb=256)
                     ) -> tuple[Sandbox, dict]:
    """The sandbox and its clean probe, or SystemExit naming what leaked."""
    sb = bwrap_sandbox()
    paths = protected_paths()
    probe = probe_sandbox(sb, paths, limits=limits)
    if not probe["isolated"]:
        raise SystemExit(f"refusing to run teacher code: the sandbox is not isolated: "
                         f"{probe.get('error') or probe['reached']}")
    return sb, {"isolated": True, "protected_paths_checked": len(paths),
                "network_interfaces": probe["seen"]["network_interfaces"],
                "environment": probe["seen"]["environment"]}

"""Environment fingerprinting and determinism helpers.

Every experiment record embeds the output of these functions so a run can be
reproduced from its manifest alone (AGENTS.md P4, P5, P8.1).
"""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
import sys

import torch


def code_state(repo_root: str) -> dict:
    """Git commit plus a hash of the uncommitted diff (including untracked files)."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    commit = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    h = hashlib.sha256(diff.encode())
    for path in untracked.splitlines():
        h.update(path.encode())
        try:
            with open(f"{repo_root}/{path}", "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<unreadable>")
    return {
        "git_commit": commit,
        "dirty": bool(diff or untracked),
        "uncommitted_state_sha256": h.hexdigest(),
        "untracked_files": untracked.splitlines(),
    }


def hardware_report() -> dict:
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_count": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if report["cuda_available"]:
        report["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return report


def set_determinism(seed: int) -> dict:
    """Seed all RNGs and enable deterministic algorithms where available."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"seed": seed, "deterministic_algorithms": True}

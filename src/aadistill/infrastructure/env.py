"""Environment fingerprinting and determinism helpers.

Every experiment record embeds the output of these functions so a run can be
reproduced from its manifest alone (AGENTS.md P4, P5, P8.1).
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys

import torch


def code_state(repo_root: str) -> dict:
    """Git commit plus a hash of the uncommitted diff (including untracked files).

    **Never raises.** This is called when a manifest is written, which is the
    last step of work that may have cost hours of GPU time, so a failure here
    would throw away the record of work already done rather than degrade it.
    That is not hypothetical: the 2026-07-30 corpus build finished all 752
    prompts and then lost its manifest because `git` is not installed in vLLM's
    official image (AGENTS.md P8.1 — do not assume a tool exists).

    When git is unavailable the commit is taken from `AADISTILL_CODE_COMMIT` if
    the caller set it, and the source is recorded either way. It is never
    guessed: an unknown commit is reported as unknown, because a manifest that
    silently invents its code state is worse than one that admits it (P4/P14).
    """

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        diff = _git("diff", "HEAD")
        untracked = _git("ls-files", "--others", "--exclude-standard")
        source = "git"
    except (OSError, subprocess.SubprocessError) as exc:
        env_commit = os.environ.get("AADISTILL_CODE_COMMIT")
        return {
            "git_commit": env_commit,
            "git_commit_source": "env:AADISTILL_CODE_COMMIT" if env_commit
            else "unavailable",
            "dirty": None,
            "uncommitted_state_sha256": None,
            "untracked_files": None,
            "code_state_error": f"{type(exc).__name__}: {exc}",
        }
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
        "git_commit_source": source,
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

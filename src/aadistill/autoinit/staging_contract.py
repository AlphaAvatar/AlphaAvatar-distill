"""What a pod can actually see, derived from the session manifest that stages it.

C1 attempt 4 died at the pod CPU test gate for `$0.6986` with six failures, and
the readiness sweep that had certified the same tree passed. The sweep ran with
`simulate_pod_env.sh`'s generic default `HIDDEN_PATHS` — a hand-maintained list of
paths *believed* absent, whose own comment asserted that every pod session stages
`artifacts/stage3/corpus_v2`. C1 stages no such thing. The simulation was 55 tests
more generous than the machine it claimed to model: 49 extra skips and 6 failures,
exactly the pass delta.

The defect was never the list's contents. It was the direction. A complement —
"everything except these paths is present" — cannot be checked against anything,
so it drifts silently every time a session's staging changes. This module inverts
it: the session's own `SetupManifest` already declares, exactly, what the pod
receives, so the visible set is **derived** from that declaration and the hidden
set is whatever is left. There is no second copy of the staging list to maintain,
and no way for the two to disagree.

Three properties matter.

**File granularity.** A `RelayInput` stages one named file into a destination
directory; it does not stage the directory. C1 puts four files into
`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, and the dev box holds six there
— `model.safetensors` and `generation_config.json` are not staged and must not be
visible. Modelling that destination as "present" would hide precisely the class of
error that has now cost four paid aborts.

**Local assets are whole trees.** A `LocalAsset` is scp'd and installed as a
directory, so every file beneath it is staged, and that difference from
`RelayInput` is part of the contract rather than an implementation detail.

**Tooling is not an artifact.** `.venv`, `__pycache__` and the pytest caches are
gitignored but are not session inputs — the pod has its own interpreter at
`/opt/train`. They are excluded by prefix, and that exclusion is the only
judgement in here; everything else follows from the manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

#: Gitignored, but not artifacts: the interpreter and regenerable caches. A pod
#: has its own venv, so hiding ours would model nothing and break the run.
TOOLING_PREFIXES: tuple[str, ...] = (
    ".venv/", ".pytest_cache/", ".ruff_cache/", ".mypy_cache/", ".git/",
)
TOOLING_SUBSTRINGS: tuple[str, ...] = ("__pycache__/",)

SCHEMA = "aadistill.autoinit.staging_contract/v1"


def _is_tooling(rel: str) -> bool:
    return (rel.startswith(TOOLING_PREFIXES)
            or any(s in rel for s in TOOLING_SUBSTRINGS)
            or rel.endswith(".pyc"))


def derive_contract(setup: Any, *, session_id: str = "") -> dict[str, Any]:
    """The staged view, read straight off the `SetupManifest` the runner uses.

    Nothing here is transcribed. `SessionRunner` builds setup's environment
    entirely from this same object, so a contract derived from it describes the
    pod that would actually be created.
    """
    relay = []
    for r in setup.relay_inputs:
        dest = getattr(r, "dest", None)
        if not dest:
            # `dest=None` means setup does not stage it — the declaration buys a
            # $0 precheck only. It is recorded, and it stages nothing.
            relay.append({"repo": r.repo, "path": r.path, "dest": None,
                          "staged_path": None, "staged": False})
            continue
        relay.append({
            "repo": r.repo, "path": r.path, "dest": dest,
            "filename": Path(r.path).name,
            "staged_path": f"{dest.rstrip('/')}/{Path(r.path).name}",
            "staged": True,
        })

    #: `install_to` is the PARENT directory and `dest_name` is the tree's name
    #: under it, so the staged tree is their join. Reading `install_to` alone
    #: marks all of `artifacts/stage1` and `artifacts/stage3` as staged — which
    #: is the very over-generous modelling this contract exists to remove, and it
    #: is what the first draft of this function did.
    local = [{
        "repo_path": a.repo_path, "dest_name": a.dest_name,
        "install_to": a.install_to,
        "staged_tree": f"{a.install_to.rstrip('/')}/{a.dest_name}",
    } for a in setup.local_assets]

    contract = {
        "schema": SCHEMA,
        "session_id": session_id,
        "session_kind": dict(setup.env).get("SESSION_KIND"),
        "setup_env": {k: v for k, v in sorted(dict(setup.env).items())},
        "required_env": list(setup.required_env),
        "relay_inputs": relay,
        "local_assets": local,
        "test_ignores": list(setup.test_ignores),
        "pytest_selection": (
            "python -m pytest tests/ -q "
            + " ".join(f"--ignore={p}" for p in setup.test_ignores)),
        "cpu_affinity_contract": (
            "the pod derives NCPU from its cgroup quota (never bare nproc, which "
            "reports the host's CPUs inside a container and also honours "
            "OMP_NUM_THREADS), pins the suite with `taskset -c 0-(NCPU-1)`, and "
            "caps OMP/MKL/OPENBLAS at min(NCPU, 8). Attempt 4 observed 128 vCPUs "
            "visible, cgroup budget 15, cpu set 0-14."),
        "teacher_revision": setup.teacher_revision,
        "tests_max_seconds": setup.tests_max_seconds,
        "granularity": (
            "RelayInput stages ONE NAMED FILE into its dest directory, not the "
            "directory. LocalAsset installs a whole tree. Modelling a relay dest "
            "as wholly present is the error this contract exists to prevent."),
    }
    contract["digest"] = contract_digest(contract)
    return contract


def contract_digest(contract: dict[str, Any]) -> str:
    """Deterministic over everything that changes what the pod can see.

    Relay inputs, local assets, test ignores, session kind and the staging
    mapping all feed it, so any of them moving invalidates a readiness record
    bound to the old value.
    """
    body = {k: v for k, v in contract.items() if k != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def staged_files(contract: dict[str, Any], repo_root: str | Path = ".") -> set[str]:
    """Every repo-relative path the pod receives, at file granularity."""
    root = Path(repo_root)
    out: set[str] = set()
    for r in contract["relay_inputs"]:
        if r.get("staged") and r.get("staged_path"):
            out.add(r["staged_path"])
    for a in contract["local_assets"]:
        tree = root / a["staged_tree"]
        if tree.is_dir():
            out |= {str(p.relative_to(root)) for p in tree.rglob("*") if p.is_file()}
    return out


def gitignored_files(repo_root: str | Path = ".") -> list[str]:
    """Non-tracked, non-tooling files: what a bundle checkout does NOT carry."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--ignored",
         "--exclude-standard"], capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\n") if p.strip() and not _is_tooling(p)]


def hidden_files(contract: dict[str, Any], repo_root: str | Path = ".") -> list[str]:
    """The complement, COMPUTED — never declared.

    This is the list the simulator moves aside. It exists only as the arithmetic
    difference between what the manifest stages and what the dev box happens to
    hold, so it cannot drift away from the session it models.
    """
    staged = staged_files(contract, repo_root)
    return sorted(p for p in gitignored_files(repo_root) if p not in staged)


def describe(contract: dict[str, Any], repo_root: str | Path = ".") -> dict[str, Any]:
    """Counts and destinations, for the readiness record and for a human."""
    staged = staged_files(contract, repo_root)
    hidden = hidden_files(contract, repo_root)
    return {
        "digest": contract["digest"],
        "n_staged_files": len(staged),
        "n_hidden_files": len(hidden),
        "relay_file_destinations": sorted(
            r["staged_path"] for r in contract["relay_inputs"] if r.get("staged")),
        "local_asset_destinations": sorted(
            a["staged_tree"] for a in contract["local_assets"]),
        "test_ignores": list(contract["test_ignores"]),
        "session_kind": contract["session_kind"],
    }

"""The C1 harness must measure everything the paid path executes.

An authorization binds a digest over `C1_HARNESS_SOURCE_FILES_V1`. If a module
the launcher or the driver actually imports is outside that set, the grant
certifies less code than runs — and the gap is invisible, because the digest
verifies perfectly against the smaller list.

This derives the set from the **real imports**, by walking the launcher and the
driver with `ast` — module level and function level alike, since stage B's
`huggingface_hub` and stage D's adapter are both imported inside methods.

It is deliberately the DIRECT set, not arbitrary repository closure. Following
imports transitively from these two files reaches most of `aadistill`:
`autoinit/__init__` pulls in the search, the Qwen3 adapter pulls in the student
model, the session runner pulls in the log relay and the remote. A harness that
large stops describing what C1 executes and starts describing the repository.
The direct set is what the C1 code names, so a new dependency is always a line
somebody wrote in one of these two files.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1, c1_harness_digest,
)
from aadistill.autoinit.c1_scoring import C1_SCORING_FILES_V1  # noqa: E402

ENTRY_POINTS = ("scripts/pod/autoinit_c1_launch.py",
                "scripts/pod/autoinit_c1_driver.py")

#: Search roots for an in-repo module name, in the order Python would resolve
#: them given each entry point's own `sys.path` inserts.
SEARCH = ("src", "scripts/pod", "scripts/autoinit", ".")

#: Covered elsewhere, each for a stated reason. Not a pattern — a list, so that
#: adding one is a decision somebody made rather than a glob that widened.
COVERED_ELSEWHERE = {
    # Hashing helpers. They can move `result_sha256`; they cannot move a count,
    # a rate or a digest gate's verdict.
    "src/aadistill/infrastructure/manifest.py": "in the harness set already",
}


def _module_file(name: str) -> Path | None:
    rel = name.replace(".", "/")
    for root in SEARCH:
        for candidate in (REPO / root / f"{rel}.py",
                          REPO / root / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: resolve against pkg
                pkg = path.relative_to(REPO / "src").parent
                parts = list(pkg.parts)
                if node.level > 1:
                    parts = parts[:-(node.level - 1)]
                out.add(".".join([*parts, node.module]) if node.module
                        else ".".join(parts))
            elif node.module:
                out.add(node.module)
                for a in node.names:             # `from pkg import module`
                    out.add(f"{node.module}.{a.name}")
    return out


def in_repo_closure() -> dict[str, list[str]]:
    """Every in-repo module the launcher or driver imports DIRECTLY.

    Depth one, on purpose. Transitive closure from these two files reaches most
    of `aadistill` — `autoinit/__init__` pulls in the search, the adapters pull in
    the student model, the session runner pulls in the relay and the remote — and
    a harness that large stops describing what C1 executes and starts describing
    the repository. The direct set is what the C1 code names, so a new dependency
    is a line somebody wrote in one of these two files.

    Function-level imports count: `ast.walk` sees them, and stage B's
    `huggingface_hub` and stage D's adapter are both imported inside methods.
    """
    reached: dict[str, list[str]] = {}
    for entry in ENTRY_POINTS:
        for name in sorted(_imports(REPO / entry)):
            f = _module_file(name)
            if f is None:
                continue                          # stdlib or third-party
            rel = str(f.relative_to(REPO))
            if rel not in ENTRY_POINTS:
                reached.setdefault(rel, []).append(entry)
    for entry in ENTRY_POINTS:
        reached.setdefault(entry, ["entry point"])
    return reached


def test_every_module_the_paid_path_imports_is_measured():
    closure = in_repo_closure()
    declared = set(C1_HARNESS_SOURCE_FILES_V1)
    missing = sorted(set(closure) - declared - set(COVERED_ELSEWHERE))
    assert not missing, (
        "these in-repo modules are reachable from the C1 launcher or driver and "
        "are outside the measured harness:\n"
        + "\n".join(f"  {m}  <- {closure[m][0]}" for m in missing))


def test_the_declared_harness_has_no_file_that_does_not_exist():
    for rel in C1_HARNESS_SOURCE_FILES_V1:
        assert (REPO / rel).is_file(), rel
    c1_harness_digest(REPO)


def test_the_paid_path_does_not_reach_the_phase_a_launcher_or_driver():
    """Eliminated rather than declared. C1 has its own parser and its own driver."""
    closure = in_repo_closure()
    for forbidden in ("scripts/pod/autoinit_phase_a_launch.py",
                      "scripts/pod/autoinit_phase_a_driver.py",
                      "scripts/autoinit/phase_a_search.py"):
        assert forbidden not in closure, f"{forbidden} <- {closure.get(forbidden)}"


def test_the_scoring_closure_is_inside_the_harness():
    """A scorer file measured by the scoring contract but not by the grant would
    let the code that produces C1's numbers change without moving the digest an
    authorization binds."""
    assert set(C1_SCORING_FILES_V1) <= set(C1_HARNESS_SOURCE_FILES_V1)


def test_the_artifact_specs_are_measured():
    for rel in ("configs/autoinit/c1_artifacts.json",
                "configs/autoinit/c1_artifacts_failed.json"):
        assert rel in C1_HARNESS_SOURCE_FILES_V1


def test_the_preregistration_records_the_live_harness():
    doc = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    live = c1_harness_digest(REPO)
    assert doc["c1_harness"]["digest"] == live["digest"]
    assert doc["c1_harness"]["n_files"] == len(C1_HARNESS_SOURCE_FILES_V1)

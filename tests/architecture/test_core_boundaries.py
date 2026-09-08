"""The maintainer's architecture rules, enforced over all of `src/aadistill`.

Every rule here is checked against the **syntax tree**, never the source text.
That distinction has bitten this repository three times in two sessions: a guard
that greps for `rope_input_gate` matches the comment explaining why
`rope_input_gate` is conditional, and a guard that greps for `Path.home()`
matches the docstring explaining why `Path.home()` was removed. Comments are not
in the AST at all, and docstrings are excluded explicitly, so no finding below
can be produced by prose.

**Why a baseline instead of a flat refusal.** The tree violates these rules
today — that is the maintainer's finding, and fixing it is a multi-session
migration touching 164 importing files and every frozen source-set declaration.
A test that simply fails would be red for the whole migration and would stop
telling anyone anything. So each rule is a **ratchet**: the known violations are
committed in `configs/architecture/core_boundary_baseline.json` with an exact
count, and the assertion is

    observed <= baseline,  and every observed violation is already listed

A NEW violation fails immediately. A FIXED violation fails too — loudly, asking
for the baseline to be lowered — so the debt cannot silently stay flat while
files move around. `test_the_baseline_only_shrinks` is what makes it a ratchet
rather than a permanent excuse.

The baseline is debt, not permission. Its counts are the remaining work.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "src" / "aadistill"
INVENTORY = REPO / "logs/architecture_inventory.json"
BASELINE = REPO / "configs/architecture/core_boundary_baseline.json"


@pytest.fixture(scope="module")
def inventory() -> dict:
    """The committed inventory, re-derived if the tree has moved since.

    Re-derived rather than trusted: an inventory that described an older tree
    would let a new violation pass, which is the one thing a ratchet must never
    do.
    """
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/architecture/inventory.py")],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-2000:]
    # the script prints counts; the full document is what we need, so import it
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    modules = INV.scan()
    graph = INV.edges(modules)
    return {"modules": modules, "graph": graph}


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text())


def _violations(inventory: dict) -> dict[str, list[str]]:
    """Every rule, evaluated. Keys match the baseline's."""
    mods = inventory["modules"]
    out: dict[str, list[str]] = {k: [] for k in (
        "experiment_named_modules", "sha256_literals", "repo_id_literals",
        "path_literals", "family_access_outside_adapters",
        "import_time_registration", "package_cycles", "core_imports_scripts")}

    for rel, m in mods.items():
        if m["classification"] == "experiment_instance":
            out["experiment_named_modules"].append(rel)
        is_adapter = m["classification"] == "model_family_adapter"
        for lit in m["literals"]:
            site = f"{rel}::{lit['owner']}"
            if "sha256" in lit["kinds"] or "git_sha" in lit["kinds"]:
                out["sha256_literals"].append(site)
            if "repo_id_shape" in lit["kinds"]:
                out["repo_id_literals"].append(site)
            if "path" in lit["kinds"]:
                out["path_literals"].append(site)
        if not is_adapter:
            for a in m["family_attribute_access"]:
                out["family_access_outside_adapters"].append(
                    f"{rel}::{a['owner']}")
        for c in m["import_time_calls"]:
            if "register" in c["call"]:
                out["import_time_registration"].append(f"{rel}::{c['call']}")
        for imp in m["imports"]:
            t = imp["target"]
            if t.startswith(("scripts.", "tests.")) or t in ("scripts", "tests"):
                out["core_imports_scripts"].append(f"{rel}::{t}")

    out["package_cycles"] = ["<->".join(c) for c in inventory["graph"]["package_cycles"]]
    return {k: sorted(set(v)) for k, v in out.items()}


# --- the ratchet ------------------------------------------------------------

RULES = (
    ("experiment_named_modules",
     "a module under src/aadistill named for one experiment"),
    ("sha256_literals",
     "a frozen digest or git sha owned by core rather than by config"),
    ("repo_id_literals",
     "a concrete model repo id in core rather than in a recipe"),
    ("path_literals",
     "a logs/ configs/ artifacts/ or home path read from core"),
    ("family_access_outside_adapters",
     "model-family member access outside an adapter"),
    ("import_time_registration",
     "a registration side effect at import time"),
    ("package_cycles",
     "a reverse/cyclic dependency between core packages"),
    ("core_imports_scripts",
     "core importing from scripts or tests"),
)


@pytest.mark.parametrize("rule,description", RULES, ids=[r[0] for r in RULES])
def test_no_new_violation_of_a_core_boundary(rule, description, inventory,
                                             baseline):
    """The ratchet. New violations fail; existing debt is named, not hidden."""
    observed = _violations(inventory)[rule]
    allowed = set(baseline["allow"][rule])
    novel = sorted(set(observed) - allowed)
    assert not novel, (
        f"NEW violation — {description}:\n  " + "\n  ".join(novel) +
        f"\n\nThis rule allows {len(allowed)} known site(s), listed in "
        f"{BASELINE.relative_to(REPO)}. That list is debt, not permission: it "
        "may shrink, never grow. If this site is genuinely generic, the fix is "
        "to make the inventory recognise it, not to append it here.")


@pytest.mark.parametrize("rule,description", RULES, ids=[r[0] for r in RULES])
def test_the_baseline_only_shrinks(rule, description, inventory, baseline):
    """A fixed violation must lower the baseline in the same commit.

    Without this the ratchet has no teeth in the good direction: debt could be
    paid off and the allowance would silently stay, ready to absorb a future
    regression at the same site.
    """
    observed = set(_violations(inventory)[rule])
    allowed = set(baseline["allow"][rule])
    stale = sorted(allowed - observed)
    assert not stale, (
        f"{len(stale)} baselined site(s) for {rule!r} no longer violate it — "
        "lower the baseline in this commit:\n  " + "\n  ".join(stale))


def test_the_baseline_records_its_own_counts_honestly(inventory, baseline):
    """The headline numbers must match the list, so the debt cannot be understated."""
    v = _violations(inventory)
    for rule, _ in RULES:
        assert baseline["counts"][rule] == len(baseline["allow"][rule]), (
            f"{rule}: counts says {baseline['counts'][rule]}, the list has "
            f"{len(baseline['allow'][rule])}")
        assert len(v[rule]) <= baseline["counts"][rule], (
            f"{rule}: {len(v[rule])} observed > {baseline['counts'][rule]} allowed")


def test_the_baseline_is_debt_and_says_so(baseline):
    assert baseline["authorizes"] == "nothing"
    assert "debt" in baseline["_contract"].lower()
    assert baseline["direction"] == "may only shrink"


# --- rules with no debt: these must hold everywhere, now --------------------

def test_core_never_imports_an_experiment_config_or_recipe(inventory):
    """No allowance. Core reading a recipe would invert the dependency."""
    offenders = []
    for rel, m in inventory["modules"].items():
        for imp in m["imports"]:
            t = imp["target"]
            if t.startswith(("configs.", "recipes.", "logs.")):
                offenders.append(f"{rel}::{t}")
    assert not offenders, offenders


def test_the_inventory_excludes_docstrings_from_literal_collection():
    """The property the whole file rests on, proven against a fixture.

    If docstrings were collected, every module explaining a digest would be
    reported as owning one, and the baseline would be noise.
    """
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    src = (
        '"""A module docstring naming logs/runs and Qwen/Qwen3-4B."""\n'
        'def f():\n'
        '    """A function docstring naming artifacts/stage1."""\n'
        '    return 1\n'
        'REAL = "logs/actually_read_this"\n'
    )
    tree = ast.parse(src)
    w = INV.Walker("fake", src)
    doc = ast.get_docstring(tree, clean=False)
    if doc is not None:
        w._docstrings.add(id(tree.body[0].value))
    w.visit(tree)
    values = [l["value"] for l in w.literals]
    assert values == ["logs/actually_read_this"], values


def test_comments_cannot_produce_a_finding():
    """Comments are not in the AST. Stated as a test because three guards in
    this repository have been defeated by exactly this."""
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    src = ("# logs/secret Qwen/Qwen3-4B "
           "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
           "X = 1\n")
    w = INV.Walker("fake", src)
    w.visit(ast.parse(src))
    assert w.literals == []

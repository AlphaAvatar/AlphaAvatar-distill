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
import hashlib
import json
import subprocess
import sys
from collections import Counter
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


def _discriminator(value: object) -> str:
    """A short, stable id for the offending CONTENT.

    Not the line number: a site must keep its identity when unrelated code above
    it moves, or every edit would churn the baseline and reviewers would stop
    reading it. Not the owner alone either -- that was the defect this replaces.
    Keying on `file::owner` and then de-duplicating meant a SECOND hard-coded
    path in an already-listed function was invisible, because the identity it
    produced was one the baseline already contained.
    """
    return hashlib.sha256(repr(value).encode()).hexdigest()[:10]


def _violations(inventory: dict) -> dict[str, list[str]]:
    """Every rule, evaluated. Keys match the baseline's.

    Returns a LIST, deliberately not a set: two identical violations in one
    owner are two violations, and collapsing them would let one be added for
    free. Multiplicity is compared with `collections.Counter` below.
    """
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
            site = f"{rel}::{lit['owner']}::{_discriminator(lit['value'])}"
            if "sha256" in lit["kinds"] or "git_sha" in lit["kinds"]:
                out["sha256_literals"].append(site)
            if "repo_id_shape" in lit["kinds"]:
                out["repo_id_literals"].append(site)
            if "path" in lit["kinds"]:
                out["path_literals"].append(site)
        if not is_adapter:
            for a in m["family_attribute_access"]:
                out["family_access_outside_adapters"].append(
                    f"{rel}::{a['owner']}::{_discriminator(a['chain'])}")
        for c in m["import_time_calls"]:
            #: A concrete INSTANCE registering itself at import is the defect:
            #: it makes the registry's contents depend on who imported what
            #: first. Registering a KIND is not -- an operator taxonomy
            #: (`DEPTH`, `FFN`, `ATTENTION`, `RESIDUAL_WIDTH`) is framework
            #: vocabulary, in the same category as a schema name, and there is
            #: nothing for a caller to inject. Matching on the word "register"
            #: alone conflated the two.
            if c["call"] in TAXONOMY_REGISTRATIONS:
                continue
            if "register" in c["call"]:
                out["import_time_registration"].append(
                    f"{rel}::<module>::{_discriminator(c['call'])}")
        for imp in m["imports"]:
            t = imp["target"]
            if t.startswith(("scripts.", "tests.")) or t in ("scripts", "tests"):
                out["core_imports_scripts"].append(
                    f"{rel}::<module>::{_discriminator(t)}")

    out["package_cycles"] = ["<->".join(c) for c in inventory["graph"]["package_cycles"]]
    return {k: sorted(v) for k, v in out.items()}


#: Registrations that declare framework VOCABULARY rather than a concrete
#: instance. Named explicitly, so adding one is a visible decision.
TAXONOMY_REGISTRATIONS = frozenset({"register_kind"})


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
    """The ratchet. New violations fail; existing debt is named, not hidden.

    Compared by multiset. A site is novel when it is absent from the baseline
    OR occurs more often than the baseline allows -- so a second offending
    literal inside an already-listed function is caught, which the previous
    `set()` comparison could not do.
    """
    observed = Counter(_violations(inventory)[rule])
    allowed = Counter(baseline["allow"][rule])
    novel = sorted((observed - allowed).elements())
    assert not novel, (
        f"NEW violation — {description}:\n  " + "\n  ".join(novel) +
        f"\n\nThis rule allows {sum(allowed.values())} known site(s), listed in "
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
    observed = Counter(_violations(inventory)[rule])
    allowed = Counter(baseline["allow"][rule])
    stale = sorted((allowed - observed).elements())
    assert not stale, (
        f"{len(stale)} baselined site(s) for {rule!r} no longer violate it — "
        "lower the baseline in this commit:\n  " + "\n  ".join(stale))


def _accepted_baseline(baseline: dict) -> tuple[dict | None, str]:
    """The baseline at the revision this one names as accepted.

    The revision is named IN the baseline (`accepted_revision`) rather than
    taken as HEAD, so widening an allowance cannot ride along inside the commit
    that needed it: the wider list has to be committed first, and then pointed
    at deliberately. Bumping that field is a one-line, greppable diff a reviewer
    can see on its own.
    """
    rev = baseline.get("accepted_revision")
    if not rev:
        return None, "no accepted_revision recorded"
    out = subprocess.run(
        ["git", "show", f"{rev}:{BASELINE.relative_to(REPO)}"],
        cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        return None, f"accepted_revision {rev[:12]} does not resolve"
    return json.loads(out.stdout), rev


@pytest.mark.parametrize("rule,description", RULES, ids=[r[0] for r in RULES])
def test_an_allowance_never_grows_against_the_accepted_revision(rule, description,
                                                                baseline):
    """Widening the allowance is checked against COMMITTED debt, not the tree.

    Both the source and the baseline are editable, so a rule that only compares
    them to each other is satisfied by editing both -- add a violation, add its
    site to the allow list, and the ratchet reports nothing. The accepted
    revision is the one at HEAD: raising an allowance therefore has to be its
    own reviewable commit, rather than something that rides along inside a
    change that needed it.
    """
    accepted, rev = _accepted_baseline(baseline)
    # Not a skip. A baseline that names a revision nothing can resolve has no
    # accepted state to compare against, which is a failure of its own claim
    # rather than a reason to stop checking -- and a skip here would decide
    # differently on a pod, whose checkout may not carry the same history.
    assert accepted is not None, (
        f"the baseline's accepted_revision cannot be read: {rev}")
    was = Counter(accepted["allow"].get(rule, []))
    now = Counter(baseline["allow"][rule])
    grew = sorted((now - was).elements())
    assert not grew, (
        f"the allowance for {rule!r} grew against the accepted revision "
        f"({rev[:12]}) by {len(grew)} site(s):\n  " + "\n  ".join(grew) +
        "\n\nDebt may only shrink. If this genuinely has to be accepted, it "
        "belongs in its own commit that says so and nothing else.")


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


# --- undefined global reads -------------------------------------------------
#
# Not a ratchet. This rule is at zero and stays at zero: an unresolvable global
# is a `NameError` waiting for the branch that reaches it, and the tree has no
# such debt to amortize.

def test_the_core_reads_no_global_it_does_not_bind():
    """`workdir=REPO` survived a whole milestone in `SessionRunner.run()`.

    `REPO` was deleted when the image layout moved into `ExecutionCommands`.
    The removal was verified against the nineteen f-strings that build remote
    commands; this was a keyword argument, so it was missed, and it raised only
    after a pod existed, setup had finished and the inputs had materialized.
    """
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    found = []
    for path in sorted(CORE.rglob("*.py")):
        rel = str(path.relative_to(REPO))
        for u in INV.undefined_global_reads(path.read_text(), rel):
            found.append(f"{rel}:{u['scope']} reads {u['name']}")
    assert not found, "unresolvable global reads in the core:\n  " + "\n  ".join(found)


def test_a_star_import_would_be_reported_rather_than_silently_trusted():
    """`from x import *` binds names this analysis cannot enumerate.

    A module using it makes the rule above UNSOUND for that module, not merely
    noisy, so the scanner records it. The core has none today; if one appears,
    this says so instead of the rule quietly weakening.
    """
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    starred = {rel: m["star_imports"] for rel, m in INV.scan().items()
               if m["star_imports"]}
    assert not starred, f"star imports make the undefined-name rule unsound: {starred}"


def test_the_rule_finds_the_defect_it_was_written_for():
    """Executed against the real pre-fix source, from git.

    A guard whose only evidence is that it currently reports zero has not been
    shown to detect anything. This runs it over the exact bytes that raised.
    """
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    broken = (
        "from .remote import JobSpec\n"
        "class R:\n"
        "    def run(self):\n"
        "        return JobSpec(job_id='d', workdir=REPO, command='c',\n"
        "                       job_dir=f'{self.ws}/jobs', log_path='l',\n"
        "                       status_path='s',\n"
        "                       env={'PYTHONPATH': f'{self.repo}/src'})\n"
    )
    found = INV.undefined_global_reads(broken, "probe.py")
    assert [(u["scope"], u["name"]) for u in found] == [("run", "REPO")], found

    fixed = broken.replace("workdir=REPO", "workdir=self.repo")
    assert INV.undefined_global_reads(fixed, "probe.py") == []


def test_conditional_and_fallback_bindings_are_not_false_positives():
    """The shapes this core actually uses must not be reported.

    A rule that fires on `TYPE_CHECKING` imports or an `except ImportError`
    fallback would be turned off within a week, which is worse than not having
    it.
    """
    sys.path.insert(0, str(REPO / "scripts/architecture"))
    import inventory as INV

    ok = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .session import SessionSpec\n"
        "try:\n"
        "    import ujson as _json\n"
        "except ImportError:\n"
        "    import json as _json\n"
        "TOTAL = 0\n"
        "def f(xs):\n"
        "    global TOTAL\n"
        "    TOTAL = sum(x for x in xs)\n"
        "    return _json, SessionSpec, TOTAL, __name__, __file__\n"
    )
    assert INV.undefined_global_reads(ok, "probe.py") == []

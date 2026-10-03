"""The core suite may not reach into a specific experiment.

AGENTS.md §2.8a draws the line; this keeps it drawn. The suite under `tests/`
grew to 266 files and 43 minutes by accretion, and most of what it validated was
closed experiments' records — no single commit did that, so no single review
caught it. A static check is enough to stop it happening again, and a static
check is all this is: it reads imports and string literals, it builds nothing.

**Two rules, and the distinction between them is the whole point.**

1. A core test may not import a SPECIFIC experiment package —
   `experiments.phase_c1`, `experiments.phase_a3`, `experiments.phase_d1`, an
   `eN` module. Those carry one campaign's arms, seeds, digests, budgets and
   wiring. A test that needs them is that experiment's test.
2. A core test MAY import the SHARED APPLICATION LAYER —
   `experiments.run_layout`, `experiments.calibration`, `experiments.datasets`,
   `experiments.deployment` and their siblings at the top of
   `scripts/experiments/`. These are cross-stage conventions, and a deliberate
   core↔application contract test is legitimate: `run_layout`'s convention is
   checked against `aadistill.runtime.run_layout`'s mechanism precisely because
   the two must agree.

The allowed set is derived from the tree, not listed here: a module directly
under `scripts/experiments/` is shared, a module inside a stage directory
belongs to an experiment. So adding an experiment cannot quietly widen the
allowance, and adding a shared module needs no edit.

**A core test also may not load a named historical launcher.** Driving
`autoinit_c1_launch` to prove that `open_run` refuses a duplicate run id is
proving a generic mechanism through a closed experiment's wiring; the generic
version builds a synthetic caller under `tmp_path`, which is what
`tests/runtime/test_run_convention.py` does. Enumerating ALL launchers is a
different thing and stays allowed — `support.session_specs` exists for the
properties every session must satisfy.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CORE = REPO / "tests"
EXPERIMENTS = REPO / "scripts" / "experiments"


def shared_application_modules() -> frozenset[str]:
    """Modules at the top of `scripts/experiments/` — owned by no stage.

    Derived from the tree so the allowance cannot drift from what is actually
    shared. A package inside `stage-*/` is an experiment's and is not here.
    """
    out = set()
    for entry in EXPERIMENTS.iterdir():
        if entry.name.startswith((".", "_")) or entry.name == "tests":
            continue
        if entry.is_file() and entry.suffix == ".py":
            out.add(entry.stem)
        elif entry.is_dir() and not entry.name.startswith("stage-"):
            out.add(entry.name)
    return frozenset(out)


def experiment_packages() -> frozenset[str]:
    """Package names living inside a stage directory — each one an experiment."""
    out = set()
    for stage in sorted(EXPERIMENTS.glob("stage-*")):
        for entry in stage.iterdir():
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                out.add(entry.name)
    return frozenset(out)


#: This file is exempt from its own string rules, and only from those: it has to
#: QUOTE a forbidden launcher name to prove the pattern matches one, and the
#: mutation test at the bottom writes synthetic violations. Its own imports are
#: checked like everyone else's.
SELF = pathlib.Path(__file__).resolve()


def core_test_files() -> list[pathlib.Path]:
    #: `support/` is included deliberately: a helper that reaches into one
    #: experiment makes every core test that imports it do the same, which is
    #: how `historical_declarations` carried Phase-C2 replay knowledge into
    #: `tests/support/` in the first place.
    return sorted(p for p in CORE.rglob("*.py") if p.name != "conftest.py") + \
        [CORE / "conftest.py"]


def experiment_imports(path: pathlib.Path) -> set[str]:
    """`experiments.*` module names this file imports, at any depth."""
    out = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:                                   # pragma: no cover
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("experiments"):
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("experiments"):
                    out.add(alias.name)
    return out


def test_the_allowed_and_forbidden_sets_are_disjoint_and_non_empty():
    """The derivation must actually separate the two, or the guard is vacuous."""
    shared, experiments = shared_application_modules(), experiment_packages()
    assert shared and experiments
    assert not (shared & experiments), (
        f"a name is both shared and experiment-owned: {sorted(shared & experiments)}")
    #: Spot-check the two the rule is written about, so a refactor that moved
    #: `run_layout` into a stage, or lifted `phase_c1` out of one, fails here.
    assert "run_layout" in shared and "calibration" in shared
    assert "phase_c1" in experiments and "phase_a3" in experiments


def test_no_core_file_imports_a_specific_experiment():
    """Rule 1. ONE test, every offender named.

    Not parametrised per file: 150 files x 3 rules is 450 ids for a static check,
    and a single message listing every violation is what a reader needs anyway.
    """
    experiments = experiment_packages()
    offenders = []
    for path in core_test_files():
        for module in sorted(experiment_imports(path)):
            owner = module.split(".")
            if len(owner) >= 2 and owner[1] in experiments:
                offenders.append(
                    f"{path.relative_to(REPO)} -> {module}  "
                    f"(move to scripts/experiments/stage-*/{owner[1]}/tests/)")
    assert not offenders, (
        "a core test reaches into a specific experiment:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the assertion to that experiment's suite and leave the "
        "generic mechanism here, proved against a synthetic caller. "
        "AGENTS.md 2.8a.")


def test_every_experiments_import_from_core_is_a_shared_module():
    """Rule 2, stated positively: whatever a core file imports from
    `experiments` must be a module owned by no stage."""
    shared = shared_application_modules()
    offenders = []
    for path in core_test_files():
        for module in sorted(experiment_imports(path)):
            parts = module.split(".")
            if len(parts) > 1 and parts[1] not in shared:
                offenders.append(f"{path.relative_to(REPO)} -> {module}")
    assert not offenders, (
        "not part of the shared application layer "
        f"({sorted(shared)}):\n  " + "\n  ".join(offenders))


#: Launchers a core test may not load BY NAME. Derived from the pod directory:
#: anything named for a phase, a continuation or an E-series session is one
#: experiment's entry point.
_NAMED_EXPERIMENT_LAUNCHER = re.compile(
    r"load_session_launcher\(\s*[\"'](?P<name>[a-z0-9_]*"
    r"(?:phase_[abcd]|_c1_|_c2_|_c3_|_a3_|continuation|e6b?|e8b?)[a-z0-9_]*)[\"']")


def test_no_core_file_loads_a_named_experiment_launcher():
    """A generic mechanism proved through a closed experiment's launcher is a
    test of that experiment.

    Enumerating every launcher is allowed and is a different statement — the
    properties every session must satisfy are genuinely generic, which is what
    `support.session_specs` is for.
    """
    offenders = []
    for path in core_test_files():
        if path.resolve() == SELF:        # it must quote one to prove the match
            continue
        hits = sorted({m.group("name") for m
                       in _NAMED_EXPERIMENT_LAUNCHER.finditer(path.read_text())})
        if hits:
            offenders.append(f"{path.relative_to(REPO)} -> {hits}")
    assert not offenders, (
        "a core test loads a named experiment launcher:\n  "
        + "\n  ".join(offenders)
        + "\n\nDrive the generic entry point with a synthetic caller, or move "
        "the test to that experiment's suite. Enumerating EVERY launcher is a "
        "different statement and stays allowed. AGENTS.md 2.8a.")


def test_the_guard_would_catch_a_violation(tmp_path):
    """MUTATION. A guard that has never refused anything is not known to work.

    Three synthetic files, one per rule, written here rather than against the
    real tree: a check that can only be verified by breaking the repository is a
    check nobody verifies.
    """
    experiments, shared = experiment_packages(), shared_application_modules()

    bad_import = tmp_path / "test_bad_import.py"
    bad_import.write_text(
        f"from experiments.{sorted(experiments)[0]} import authorization\n")
    assert experiment_imports(bad_import), "the reader missed a plain import"
    module = next(iter(experiment_imports(bad_import)))
    assert module.split(".")[1] in experiments, "rule 1 would not fire"

    deferred = tmp_path / "test_deferred.py"
    deferred.write_text(
        "def test_x():\n"
        f"    from experiments.{sorted(experiments)[0]} import pod_environment\n")
    assert experiment_imports(deferred), (
        "an import inside a function is still an import; a guard that only reads "
        "module scope would pass the file that moved C1 into core")

    launcher = tmp_path / "test_launcher.py"
    launcher.write_text('mod = load_session_launcher("autoinit_c1_launch")\n')
    assert _NAMED_EXPERIMENT_LAUNCHER.search(launcher.read_text()), \
        "rule 3 would not fire"

    concrete = tmp_path / "test_concrete_run.py"
    concrete.write_text(
        'D = REPO / "logs/stages/stage-1/phase_a/runs/attempt12"\n')
    assert any(_CONCRETE_RUN.match(l)
               for l in _repo_joined_literals(concrete)), "rule 4 would not fire"

    prose = tmp_path / "test_prose.py"
    prose.write_text(
        '"""see logs/stages/stage-1/phase_c1/runs/attempt9 for why."""\n'
        'entry = {"root": "logs/stages/stage-1/phase_c1/runs/attempt10"}\n')
    assert not [l for l in _repo_joined_literals(prose) if _CONCRETE_RUN.match(l)], (
        "rule 4 would flag a docstring or synthetic test data, which is the "
        "false positive that would have made it not worth having")

    allowed = tmp_path / "test_allowed.py"
    allowed.write_text("from experiments.run_layout import RunLayout\n")
    assert all(m.split(".")[1] in shared for m in experiment_imports(allowed)), \
        "the guard would refuse a legitimate shared-application contract"

#: A CONCRETE historical run used as core test INPUT.
#:
#: `test_stage1_import.py` bypassed both rules above: it imported nothing from
#: `experiments` and loaded no launcher, yet it read
#: `logs/stages/stage-1/phase_a/runs/attempt12` and a host-local checkpoint store
#: under a `skipif`. The core suite's result therefore depended on whether a
#: closed experiment's bytes were still on this machine.
#:
#: The signal is deliberately narrow: a string literal naming a concrete run,
#: JOINED TO `REPO`. That is what reading the real tree looks like. It does NOT
#: match the same path in a docstring (`test_run_layout_all_stages` explains
#: attempt 9's two surviving components), nor in synthetic test data
#: (`test_log_organisation` builds index entries naming `runs/attempt10` and a
#: tree under `tmp_path`), nor generic discovery through `logs/index.json`, which
#: is how a governance test should find runs. Three such files exist today and
#: all three are legitimate; a pattern that flagged them would have been dropped
#: rather than grown into an ownership framework.
_CONCRETE_RUN = re.compile(r"^logs/stages/stage-[0-9]+/[a-z0-9_]+/runs/[a-z0-9_]+")


def _repo_joined_literals(path: pathlib.Path) -> set[str]:
    """String literals this file joins to `REPO` with `/`."""
    out = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:                                   # pragma: no cover
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        base = node.left
        #: `REPO / "x"` and `REPO / "x" / "y"` both have REPO at the far left.
        while isinstance(base, ast.BinOp):
            base = base.left
        if isinstance(base, ast.Name) and base.id in ("REPO", "REPO_ROOT"):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                out.add(node.right.value)
    return out


def test_no_core_file_reads_a_concrete_historical_run():
    """A generic mechanism tested against one old run is that run's test."""
    offenders = []
    for path in core_test_files():
        if path.resolve() == SELF:
            continue
        for literal in sorted(_repo_joined_literals(path)):
            if _CONCRETE_RUN.match(literal):
                offenders.append(f"{path.relative_to(REPO)} -> REPO / {literal!r}")
    assert not offenders, (
        "a core test reads a concrete historical run:\n  "
        + "\n  ".join(offenders)
        + "\n\nBuild the input under `tmp_path`, or move the verification to the "
        "experiment that owns the run. Discovering runs through `logs/index.json` "
        "is still fine. AGENTS.md 2.8a.")

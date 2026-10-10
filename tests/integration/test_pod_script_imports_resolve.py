"""Every script a pod EXECUTES can resolve its own imports. At `$0`.

THE FAILURE THIS EXISTS FOR. `scripts/shared/pod/autoinit_engine_probe.py`
inserted `scripts/evaluation` on `sys.path` and imported
`shared.evaluation.uncapped_eval`. The information-architecture migration moved
the evaluator under `scripts/shared/evaluation/` and rewrote the import --
which needs `scripts` on the path, since `shared` is a package under it -- but
left the insert naming a directory that no longer exists. The probe could not
import its own dependency on ANY machine, and no suite noticed: every phase
that runs it (C1, C3, A3) closed before the migration, and the module is only
ever executed as a subprocess on a pod.

D1's screening attempt 3 paid for the discovery: five arms materialized, the
first probe trained for 60.9 minutes, and the attestation refused because no
generation protocol could be observed. 164 minutes, $2.98.

**A stale insert is not automatically a defect.** 127 of them exist across the
tree, and nearly all are harmless: the module is importable through another
entry the script also adds, so the dead path is a no-op. What is NOT harmless
is a script whose imports cannot resolve AT ALL under the paths it sets for
itself. That is what this checks, by compiling each module in a subprocess
under exactly its own `sys.path` -- no ambient `PYTHONPATH`, which is how a
pod runs it and how the dev box hides the problem.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Scripts a DRIVER spawns as a BARE subprocess, declared POSITIVELY.
#:
#: The distinction is the whole contract. A driver sets its own `sys.path` in
#: process and that does NOT put anything in `os.environ`, so every subprocess
#: it spawns inherits an environment with no `PYTHONPATH` and must resolve its
#: own first-party imports. These five are spawned exactly that way --
#: `/opt/train/bin/python <script>` or `/opt/vllm/bin/python <script>` -- and
#: the engine probe is the one that proved what happens when one cannot.
#:
#: `collect_artifacts.py` and `watchdog.py` are deliberately NOT here: the
#: shared `SessionRunner` invokes both WITH an explicit
#: `PYTHONPATH=<repo>/src`, so they are not required to be self-sufficient.
#: That contract is asserted below rather than assumed, because if the runner
#: ever stopped supplying it they would fail exactly like the probe did.
DRIVER_SPAWNED_BARE: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_engine_probe.py",
    "scripts/shared/evaluation/uncapped_eval.py",
    "scripts/shared/training/train_stage3.py",
)

#: Invoked by the runner WITH a PYTHONPATH it supplies.
RUNNER_SUPPLIED: tuple[str, ...] = (
    "scripts/shared/pod/collect_artifacts.py",
    "scripts/shared/pod/watchdog.py",
)

#: D1's behavioural chain, whose pod-side entry points this round added.
D1_BEHAVIOURAL_POD: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py",
    "scripts/stages/stage-1/phase_d1/score_d1_screening.py",
    "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py",
)


def _repo_root_depth(tree: ast.AST) -> dict[str, int]:
    """`NAME = Path(__file__).resolve().parents[N]` -> {NAME: N}."""
    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        text = ast.unparse(node.value)
        if "__file__" in text and "parents[" in text:
            try:
                depth = int(text.split("parents[")[1].split("]")[0])
            except (IndexError, ValueError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = depth
    return out


def _self_inserted_paths(path: Path) -> list[Path]:
    """The repo-relative paths a module puts on `sys.path` for itself.

    Both shapes in this tree are covered: a direct
    `sys.path.insert(0, str(REPO / "src"))`, and the loop form
    `for extra in ("src", "scripts"): sys.path.insert(0, str(ROOT / extra))`.
    Parsed, never executed -- executing it is the thing being made safe.
    """
    tree = ast.parse(path.read_text())
    depths = _repo_root_depth(tree)
    if not depths:
        return []
    #: One root is the norm; when a module names several, the shallowest that
    #: actually contains `src` is the repository.
    bases = {name: path.resolve().parents[d] for name, d in depths.items()
             if d < len(path.resolve().parents)}
    base = next((b for b in bases.values() if (b / "src").is_dir()),
                next(iter(bases.values())))

    literals: list[str] = []
    for node in ast.walk(tree):
        #: A statement that touches sys.path, or a loop whose body does.
        if isinstance(node, (ast.For, ast.Expr, ast.Assign, ast.With)):
            text = ast.unparse(node)
            if "sys.path" not in text:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    literals.append(sub.value)
    out: list[Path] = []
    for literal in literals:
        #: Only repo-relative fragments; a message or a module name is not a
        #: path, and an absolute one is not ours to resolve.
        if literal.startswith("/") or " " in literal:
            continue
        candidate = base / literal
        if candidate.is_dir() and candidate not in out:
            out.append(candidate)
    return out


FIRST_PARTY = ("shared", "stages", "aadistill", "experiments", "support")


def _first_party_imports(path: Path) -> list[str]:
    """Top-level first-party modules the file imports AT ANY DEPTH.

    THIS USED TO READ `tree.body` ONLY, and that made the whole test vacuous
    on the module it was written for. `autoinit_engine_probe.py` does its
    `from shared.evaluation.uncapped_eval import ...` inside `main()`, so a
    module-level-only scan found nothing, the test skipped, and the stale
    `sys.path` insert it exists to catch sailed through -- after costing D1's
    screening attempt 3 sixty-one minutes of training and $2.98. The driver
    skipped for the same reason.

    Depth is the wrong axis. The paths a script inserts for itself are
    inserted AT MODULE LEVEL and persist for the life of the process, so an
    import inside a function is resolved against exactly those paths. What
    matters is whether the set of paths the module installs can resolve every
    first-party module it will ask for, whenever it asks.
    """
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module.split(".")[0])
    return sorted({n for n in names if n in FIRST_PARTY})


@pytest.mark.parametrize("rel", DRIVER_SPAWNED_BARE + D1_BEHAVIOURAL_POD)
def test_first_party_imports_resolve_under_the_scripts_own_paths(rel: str
                                                                 ) -> None:
    """Resolvable with NO ambient PYTHONPATH, exactly as a pod runs it.

    `find_spec` rather than import, because a real import needs the GPU
    libraries a pod has and this box does not -- while the failure being
    guarded is purely one of path resolution.
    """
    path = REPO / rel
    assert path.is_file(), rel
    #: ASSERTED, NOT SKIPPED. This was `pytest.skip(...)` for the hypothetical
    #: module that imports nothing first-party, and C1's skip-predicate audit
    #: was right to flag it UNRESOLVED: the condition is a function result, so
    #: no signal matches it and nothing said the pod and the dev box must
    #: decide it the same way. The deeper problem is that skipping is the wrong
    #: behaviour -- every entry in these lists is declared BECAUSE a pod
    #: executes it and it imports first-party code, so a module with no such
    #: import is a declaration to fix, and silently skipping would hide the
    #: day someone removed the import this guard exists for.
    wanted = _first_party_imports(path)
    assert wanted, (
        f"{rel} is declared as a pod-executed script but imports no "
        "first-party module at module level. Either its imports moved inside "
        "functions -- in which case this check no longer covers it and the "
        "declaration should say so -- or it no longer belongs in the list.")
    paths = _self_inserted_paths(path)
    assert paths, (
        f"{rel} imports {wanted} at module level and puts nothing resolvable "
        "on sys.path for itself; a pod has no ambient PYTHONPATH")
    code = (
        "import importlib.util, sys\n"
        f"sys.path[:0] = {[str(p) for p in paths]!r}\n"
        f"missing = [n for n in {wanted!r} "
        "if importlib.util.find_spec(n) is None]\n"
        "print('MISSING=' + ','.join(missing))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=REPO,
                          env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, f"{rel}: {proc.stderr}"
    missing = [m for m in proc.stdout.strip().split("=", 1)[1].split(",") if m]
    assert not missing, (
        f"{rel} sets its own sys.path to "
        f"{[p.name for p in paths]} and still cannot resolve {missing}. A pod "
        "executes this as a subprocess with no ambient PYTHONPATH, so an "
        "unresolvable first-party import is discovered on a billing machine "
        "-- which is what the engine probe cost D1's screening.")


def test_the_engine_probe_puts_scripts_on_the_path_not_a_subdirectory() -> None:
    """The specific regression, named, so a future edit cannot undo it
    silently: `shared.evaluation` needs `scripts`, and `scripts/evaluation`
    has not existed since the migration."""
    src = (REPO / "scripts/shared/pod/autoinit_engine_probe.py").read_text()
    assert 'REPO / "scripts"' in src
    assert 'REPO / "scripts/evaluation"' not in src
    assert "from shared.evaluation.uncapped_eval import" in src
    assert not (REPO / "scripts/evaluation").exists(), (
        "scripts/evaluation exists again; this regression's premise changed")


@pytest.mark.parametrize("rel", RUNNER_SUPPLIED)
def test_the_runner_supplies_the_path_for_what_it_invokes(rel: str) -> None:
    """The other half of the contract, checked rather than assumed.

    These two import `aadistill` at module level and set no path for
    themselves, which is correct ONLY because `SessionRunner` passes
    `PYTHONPATH=<repo>/src` when it invokes them. If that ever stopped, they
    would fail on a pod exactly as the engine probe did -- so the guarantee
    they depend on is asserted at the invocation site.
    """
    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "PYTHONPATH" in runner
    name = Path(rel).stem
    #: The runner names each of these through `ExecutionCommands`, so the
    #: assertion is that it never invokes one without supplying the path.
    for line in runner.splitlines():
        if name in line and "python" in line.lower():
            assert "PYTHONPATH" in line or "env=" in line, (
                f"{rel} is invoked at {line.strip()!r} without a PYTHONPATH, "
                "and it sets none for itself")
    assert not _self_inserted_paths(REPO / rel), (
        f"{rel} now sets its own path; move it to DRIVER_SPAWNED_BARE so the "
        "stronger check applies")


def test_the_bare_group_is_what_the_d1_driver_actually_spawns() -> None:
    """Non-vacuity: the declared bare group must match the driver's real
    subprocess calls, so a new spawned script cannot escape the check by not
    being listed."""
    src = (REPO / "scripts/stages/stage-1/phase_d1/"
                  "autoinit_d1_behavioural_driver.py").read_text()
    for rel in ("scripts/shared/pod/autoinit_engine_probe.py",):
        assert Path(rel).name in src, rel
    #: The driver reaches the trainer, the evaluator and both scorers through
    #: constants imported from their owners, so their names appear as symbols
    #: rather than paths.
    for symbol in ("TRAINER", "UNCAPPED_EVAL", "SCORER",
                   "CONFIRMATION_SCORER"):
        assert symbol in src, symbol


def test_the_check_is_not_vacuous_on_the_module_it_was_written_for() -> None:
    """The engine probe MUST contribute a first-party import to check.

    It did not, for the whole life of the first version: the scan read module
    level only, the probe imports `shared.evaluation.uncapped_eval` inside
    `main()`, so the parametrized case skipped and the stale `sys.path` insert
    went unchecked by its own regression. A guard that silently covers nothing
    is worse than no guard, so the premise is asserted rather than assumed.
    """
    wanted = _first_party_imports(REPO / "scripts/shared/pod/autoinit_engine_probe.py")
    assert "shared" in wanted, (
        "the engine probe no longer imports `shared`; this regression's "
        "premise changed")
    assert "aadistill" in wanted


@pytest.mark.parametrize("rel", DRIVER_SPAWNED_BARE + D1_BEHAVIOURAL_POD)
def test_every_declared_pod_script_contributes_something_to_check(rel: str
                                                                  ) -> None:
    """Non-vacuity for the whole declared set, not just the one example."""
    wanted = _first_party_imports(REPO / rel)
    assert wanted, (
        f"{rel} is declared as a pod-executed script but imports no "
        "first-party module at any depth, so the resolution check covers "
        "nothing for it. Either it no longer belongs in the list, or the "
        "import it was declared for has gone.")

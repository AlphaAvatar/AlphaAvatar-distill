"""The real-CUDA-validated execution surface is byte-identical, and the rest of
the core changed only in prose.

The review that accepted execution SHA `7027a8f4` named a surface that must not
move — `attention_activation`, the attention statistics, `stats_to`/device
behaviour, the adapters the validation used, fixed-path/suffix execution,
treatment-record generation, and the two geometries — and said in as many words:
do not silently claim that SHA validates different code.

"I did not change those files" is a claim about intent. This is the check.
Two things are asserted, both read from git rather than argued:

1. every file on the declared surface is byte-identical to the reviewed tip;
2. every OTHER core file that did change is prose-only — its AST, with
   docstrings stripped, is identical to what it was.

(2) matters as much as (1). A docstring sweep across 44 modules is exactly the
shape of change that could carry one edited expression through review unnoticed,
and reading 44 diffs by eye is how that gets missed.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: One entry per independent review round on this branch, oldest first: the tip
#: that round was reviewed at, and the core files that round deliberately
#: changed the BEHAVIOUR of. Everything else it touched must be prose-only.
#:
#: A list rather than one constant because each round is bound to its own
#: reviewed tip. Collapsing them would either lose the earlier round's
#: declaration or measure this round's changes from the wrong base, and both
#: weaken the check that catches an undeclared edit riding along with a
#: docstring sweep.
ROUNDS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("573d6cfc92497c176afe4645335df1a2534f93df",
     "post-CUDA evidence and reproducibility closure",
     {
        "src/aadistill/infrastructure/session.py":
            "ExecutionCommands: interpreter default removed, canonical "
            "DeploymentBinding added",
        "src/aadistill/infrastructure/session_runner.py":
            "records the resolved provider CLI in the session record",
     }),
    ("b2ecdff83ee0a7653eea7872b5af6739a4de4381",
     "readiness wire-contract core/application separation",
     {
        "src/aadistill/runtime/pod_environment.py":
            "RecordContract: the schema string, the harness field name, the "
            "harness digest callable and the record path are the CALLER's; the "
            "success summary is derived from the record instead of naming two "
            "of one experiment's groups. THIS IS A DECLARED SEMANTIC CHANGE to "
            "readiness verification and is deliberately not described as prose.",
     }),
)

#: The tip the CURRENT round was reviewed at.
REVIEWED_TIP = ROUNDS[-1][0]

#: The GPU-validated execution commit. Named here because it is what the
#: preserved surface's bytes are claimed to still match.
EXECUTION_SHA = "7027a8f4c0a7684c892b483a2193025cc26c1b58"

#: Verbatim from the review's clause 5, resolved to files.
CUDA_VALIDATED_SURFACE = (
    "src/aadistill/initialization/operators/attention_activation.py",
    "src/aadistill/initialization/statistics/attention.py",
    "src/aadistill/initialization/device.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/adapters/__init__.py",
    "src/aadistill/initialization/adapters/qwen3.py",
)


def declared_from(round_index: int) -> dict[str, str]:
    """Every semantic change declared by this round and every later one.

    Measured from an older base, a later round's declared change is also in the
    diff -- so the expected set for round i is the union from i onward, and
    nothing else is admissible.
    """
    out: dict[str, str] = {}
    for _tip, _label, changes in ROUNDS[round_index:]:
        out.update(changes)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture(scope="module")
def changed_core() -> list[str]:
    out = git("diff", "--name-only", REVIEWED_TIP).split()
    return [f for f in out if f.startswith("src/aadistill/") and f.endswith(".py")]


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            body = getattr(n, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                n.body = body[1:] or [ast.Pass()]
    return tree


def shape(source: str) -> str:
    """Everything except the prose."""
    return ast.dump(strip_docstrings(ast.parse(source)))


# --- 1. the preserved surface did not move ----------------------------------

@pytest.mark.parametrize("path", CUDA_VALIDATED_SURFACE)
def test_the_validated_file_is_byte_identical_to_the_reviewed_tip(path):
    assert (REPO / path).is_file(), path
    before = git("show", f"{REVIEWED_TIP}:{path}")
    assert (REPO / path).read_text() == before, (
        f"{path} is on the CUDA-validated execution surface and has changed. "
        "Report whether a CUDA rerun is required; do not claim execution SHA "
        f"{EXECUTION_SHA[:8]} validates it.")


@pytest.mark.parametrize("path", CUDA_VALIDATED_SURFACE)
def test_it_is_also_identical_to_the_gpu_execution_commit(path):
    """The stronger form: identical to the commit that actually ran on CUDA,
    not merely to the tip this session started from."""
    assert (REPO / path).read_text() == git("show", f"{EXECUTION_SHA}:{path}"), (
        f"{path} differs from the commit the GPU evidence names")


def test_none_of_them_appears_in_this_sessions_diff(changed_core):
    overlap = sorted(set(changed_core) & set(CUDA_VALIDATED_SURFACE))
    assert overlap == [], f"validated surface touched: {overlap}"


# --- 2. every other change is prose-only, or declared ----------------------

@pytest.mark.parametrize("index", range(len(ROUNDS)),
                         ids=[r[1] for r in ROUNDS])
def test_every_other_core_change_is_prose_only_or_declared(index):
    """The check that reading dozens of diffs by eye would not reliably make.

    Run once per review round, from that round's own reviewed tip. From an older
    base a later round's declared change is also in the diff, so the admissible
    set is the union from that round onward -- and nothing else. An edit that
    rode along with a docstring sweep appears here as an unexplained entry.
    """
    base = ROUNDS[index][0]
    changed = [f for f in git("diff", "--name-only", base).split()
               if f.startswith("src/aadistill/") and f.endswith(".py")]
    semantic = {f for f in changed
                if shape(git("show", f"{base}:{f}")) != shape((REPO / f).read_text())}
    expected = set(declared_from(index))
    assert semantic == expected, (
        f"from {base[:8]} ({ROUNDS[index][1]}), core semantic changes disagree "
        "with what is declared:\n"
        + "\n".join(f"  {f}" for f in sorted(semantic ^ expected)))


def test_this_rounds_declaration_is_not_empty():
    """A round that declares nothing while changing behaviour would pass the
    check above only by making the expected set match a wrong observation."""
    assert ROUNDS[-1][2], "the current round declares no semantic change"


def test_the_readiness_change_is_declared_as_semantic_not_prose():
    """Named explicitly, because describing it as a docstring sweep is exactly
    the misreport this file exists to prevent."""
    declared = ROUNDS[-1][2]
    assert "src/aadistill/runtime/pod_environment.py" in declared
    why = declared["src/aadistill/runtime/pod_environment.py"]
    assert "SEMANTIC CHANGE" in why
    # And it really is one, measured rather than asserted.
    base = ROUNDS[-1][0]
    path = "src/aadistill/runtime/pod_environment.py"
    assert shape(git("show", f"{base}:{path}")) != shape((REPO / path).read_text())


def test_the_prose_sweep_actually_covered_the_core():
    """A guard on the guard above: if almost nothing changed in prose,
    the parametrized check would pass close to vacuously."""
    changed = [f for f in git("diff", "--name-only", ROUNDS[0][0]).split()
               if f.startswith("src/aadistill/") and f.endswith(".py")]
    prose_only = set(changed) - set(declared_from(0))
    assert len(prose_only) >= 20, (
        f"only {len(prose_only)} core files changed in prose; the ownership "
        "sweep was expected to reach far more than that")


def test_no_declared_change_touches_the_validated_surface():
    assert not (set(declared_from(0)) & set(CUDA_VALIDATED_SURFACE))


# --- the geometries the validation ran are unchanged too --------------------

def test_the_two_validation_geometries_are_unchanged():
    """`suffix_narrow` and `suffix_mid`, from the config the check reads."""
    cfg = "configs/validation/cuda_engineering.json"
    assert (REPO / cfg).read_text() == git("show", f"{EXECUTION_SHA}:{cfg}"), (
        "the validation workload changed; the accepted evidence describes a "
        "different configuration")

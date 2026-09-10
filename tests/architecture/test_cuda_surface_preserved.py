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

#: The reviewed tip this closure started from. Everything below is measured
#: against it, so the assertions describe THIS session's changes and no others.
REVIEWED_TIP = "573d6cfc92497c176afe4645335df1a2534f93df"

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

#: The two files this closure deliberately changed the behaviour of, and the
#: reason. Anything else appearing in the semantic list is unexplained.
EXPECTED_SEMANTIC_CHANGES = {
    "src/aadistill/infrastructure/session.py":
        "ExecutionCommands: interpreter default removed, canonical DeploymentBinding added",
    "src/aadistill/infrastructure/session_runner.py":
        "records the resolved provider CLI in the session record",
}


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


# --- 2. everything else that changed changed only in prose ------------------

def test_every_other_core_change_is_prose_only(changed_core):
    """The check that reading 44 diffs by eye would not reliably make."""
    semantic = [f for f in changed_core
                if shape(git("show", f"{REVIEWED_TIP}:{f}"))
                != shape((REPO / f).read_text())]
    assert set(semantic) == set(EXPECTED_SEMANTIC_CHANGES), (
        "unexplained semantic change in core:\n"
        + "\n".join(f"  {f}" for f in sorted(set(semantic)
                                             ^ set(EXPECTED_SEMANTIC_CHANGES))))


def test_the_prose_sweep_actually_covered_the_core(changed_core):
    """A guard on the guard above: if the sweep touched almost nothing,
    `test_every_other_core_change_is_prose_only` would pass vacuously."""
    prose_only = set(changed_core) - set(EXPECTED_SEMANTIC_CHANGES)
    assert len(prose_only) >= 20, (
        f"only {len(prose_only)} core files changed in prose; the ownership "
        "sweep was expected to reach far more than that")


def test_the_two_semantic_changes_are_outside_the_validated_surface():
    assert not (set(EXPECTED_SEMANTIC_CHANGES) & set(CUDA_VALIDATED_SURFACE))


# --- the geometries the validation ran are unchanged too --------------------

def test_the_two_validation_geometries_are_unchanged():
    """`suffix_narrow` and `suffix_mid`, from the config the check reads."""
    cfg = "configs/validation/cuda_engineering.json"
    assert (REPO / cfg).read_text() == git("show", f"{EXECUTION_SHA}:{cfg}"), (
        "the validation workload changed; the accepted evidence describes a "
        "different configuration")

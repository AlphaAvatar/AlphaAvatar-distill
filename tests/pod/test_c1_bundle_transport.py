"""The transport gate: can a pod obtain the exact authorized code, right now?

C1 attempt 1 passed all eight pre-provider gates, created a pod and died at
`SETUP_RC=1` fetching `transfer/c1` — an alias for nothing. `$0.0786` for a 404.
Every gate verified the *contents* of the session commit; none asked whether the
pod could reach it.

These tests drive the real `roundtrip` against real git bundles, built from this
repository, through an injected `download` that hands back a local file instead
of calling the relay. Nothing here is a mock of the verification: the bundles are
real, `git bundle verify` really runs, the clone really happens, and the harness
digest is really recomputed from the round-tripped checkout. Only the transport
is local, so the suite costs nothing and needs no network.

Every failure mode that could have produced attempt 1 has a case.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))

from aadistill.autoinit.c1_bundle import (  # noqa: E402
    BUNDLE_PREFIX, C1BundleError, build_bundle, canonical_bundle_name,
    canonical_repo_path, require_canonical_bundle_arg, roundtrip, sha256_bytes,
)

AUTH_PATH = "logs/autoinit_c1_authorization.json"


def _git(*a, cwd=None):
    out = subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


HEAD = _git("rev-parse", "HEAD", cwd=REPO)
#: The session commit these tests bundle. HEAD, not the commit that first
#: introduced the authorization: a real session commit carries BOTH the
#: authorization artifact and the current harness, and the attempt-1 commit
#: predates two harness files, so using it made the round-trip fail for a reason
#: no real session could hit.
AUTH_COMMIT = HEAD


# --- the canonical name -----------------------------------------------------

def test_the_name_is_derived_from_the_commit():
    assert canonical_bundle_name(HEAD) == f"{BUNDLE_PREFIX}{HEAD[:8]}.bundle"
    assert canonical_repo_path(HEAD) == f"transfer/{canonical_bundle_name(HEAD)}"


def test_an_alias_fails_at_zero_dollars():
    """The exact defect: `--bundle c1` named a file that never existed."""
    with pytest.raises(C1BundleError, match="is not the canonical name"):
        require_canonical_bundle_arg("c1", HEAD)
    assert require_canonical_bundle_arg(canonical_bundle_name(HEAD), HEAD)


@pytest.mark.parametrize("bad", ["", "nope", "zzzzzzzz", "c1"])
def test_a_non_commit_yields_no_canonical_name(bad):
    with pytest.raises(C1BundleError):
        canonical_bundle_name(bad)


# --- the round-trip ---------------------------------------------------------

def _auth_bytes(commit: str) -> bytes:
    out = subprocess.run(["git", "show", f"{commit}:{AUTH_PATH}"],
                         capture_output=True, cwd=REPO)
    return out.stdout


def _harness(commit: str, files: tuple[str, ...]) -> str:
    entries = []
    for rel in sorted(files):
        blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                              capture_output=True, cwd=REPO)
        assert blob.returncode == 0, rel
        entries.append({"path": rel,
                        "sha256": hashlib.sha256(blob.stdout).hexdigest()})
    return hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()


@pytest.fixture(scope="module")
def files() -> tuple[str, ...]:
    from aadistill.autoinit.c1_authorization import C1_HARNESS_SOURCE_FILES_V1
    return C1_HARNESS_SOURCE_FILES_V1


def _serve(path: Path):
    """An injected `download` that hands back a local file, byte-identical."""
    def download(repo_id, path_in_repo, dest_dir):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(path_in_repo).name
        shutil.copyfile(path, dest)
        return dest
    return download


def _absent():
    def download(repo_id, path_in_repo, dest_dir):
        raise FileNotFoundError(
            f"404 Entry Not Found for url: .../{path_in_repo}")
    return download


def _rt(tmp_path, *, commit, bundle_path, local_sha, files, auth=None,
        harness=None):
    return roundtrip(
        session_commit=commit, local_bundle_sha256=local_sha,
        authorization_bytes=_auth_bytes(commit) if auth is None else auth,
        authorization_path=AUTH_PATH,
        expected_harness_digest=_harness(commit, files) if harness is None
        else harness,
        harness_files=files, download=_serve(bundle_path),
        workdir=tmp_path / "rt")


@pytest.fixture(scope="module")
def good_bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    built = build_bundle(REPO, AUTH_COMMIT, d / canonical_bundle_name(AUTH_COMMIT))
    return Path(built["path"]), built["sha256"]


def test_the_exact_canonical_round_trip_passes(tmp_path, good_bundle, files):
    path, sha = good_bundle
    ev = _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=path, local_sha=sha,
             files=files)
    assert ev["ok"] is True
    assert ev["roundtrip_head"] == AUTH_COMMIT
    assert ev["remote_sha256"] == sha == ev["local_sha256"]
    assert ev["authorization_matches"] is True
    assert ev["roundtrip_harness_digest"] == ev["expected_harness_digest"]
    assert ev["canonical_bundle_name"] == canonical_bundle_name(AUTH_COMMIT)
    assert ev["bytes"] > 0


def test_a_missing_remote_object_fails(tmp_path, good_bundle, files):
    _, sha = good_bundle
    with pytest.raises(Exception, match="404|Entry Not Found"):
        roundtrip(session_commit=AUTH_COMMIT, local_bundle_sha256=sha,
                  authorization_bytes=_auth_bytes(AUTH_COMMIT),
                  authorization_path=AUTH_PATH,
                  expected_harness_digest=_harness(AUTH_COMMIT, files),
                  harness_files=files, download=_absent(),
                  workdir=tmp_path / "rt")


def test_remote_bytes_differing_from_the_staged_bundle_fail(tmp_path, good_bundle,
                                                            files):
    path, sha = good_bundle
    with pytest.raises(C1BundleError, match="different bytes than were prepared"):
        _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=path, local_sha="0" * 64,
            files=files)


def test_a_stale_bundle_for_a_previous_commit_fails(tmp_path, files):
    """It verifies, it clones, and it is the wrong tree — the case a name-only
    check cannot see."""
    parent = _git("rev-parse", f"{AUTH_COMMIT}^", cwd=REPO)
    built = build_bundle(REPO, parent, tmp_path / "stale.bundle")
    with pytest.raises(C1BundleError, match="does not contain|not the session commit"):
        _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=Path(built["path"]),
            local_sha=built["sha256"], files=files)


def test_a_bundle_that_does_not_contain_the_commit_is_refused_at_build(tmp_path):
    """`git bundle create` cannot be asked for a commit it will not include, so
    the equivalent failure is caught by re-deriving the heads afterwards."""
    built = build_bundle(REPO, AUTH_COMMIT, tmp_path / "b.bundle")
    assert AUTH_COMMIT in built["heads"]
    with pytest.raises(C1BundleError):
        build_bundle(REPO, "0" * 40, tmp_path / "nope.bundle")


def test_a_checkout_carrying_a_different_authorization_fails(tmp_path, good_bundle,
                                                             files):
    path, sha = good_bundle
    with pytest.raises(C1BundleError, match="not the artifact this launcher"):
        _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=path, local_sha=sha,
            files=files, auth=b'{"schema": "someone elses grant"}\n')


def test_a_checkout_whose_harness_digest_differs_fails(tmp_path, good_bundle,
                                                       files):
    path, sha = good_bundle
    with pytest.raises(C1BundleError, match="harness inside the bundle digests"):
        _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=path, local_sha=sha,
            files=files, harness="f" * 64)


def test_a_missing_authorization_in_the_checkout_fails(tmp_path, files):
    """A bundle for the PRE-authorization base: the ordering error the runbook
    warns about, caught rather than trusted."""
    base = _git("rev-parse", f"{AUTH_COMMIT}^", cwd=REPO)
    built = build_bundle(REPO, base, tmp_path / "base.bundle")
    with pytest.raises(C1BundleError):
        roundtrip(session_commit=base, local_bundle_sha256=built["sha256"],
                  authorization_bytes=_auth_bytes(AUTH_COMMIT),
                  authorization_path="logs/does_not_exist_here.json",
                  expected_harness_digest=_harness(base, files),
                  harness_files=files, download=_serve(Path(built["path"])),
                  workdir=tmp_path / "rt")


# --- mutations --------------------------------------------------------------

def test_mutation_dropping_canonical_name_enforcement_is_caught():
    """If `require_canonical_bundle_arg` stopped comparing, `c1` would pass."""
    import inspect
    src = inspect.getsource(require_canonical_bundle_arg)
    assert "if bundle != want" in src and "raise C1BundleError" in src


def test_mutation_dropping_the_checkout_commit_check_is_caught(tmp_path, files,
                                                               monkeypatch):
    """Without the HEAD equality, the stale-bundle case passes silently."""
    import aadistill.autoinit.c1_bundle as B

    parent = _git("rev-parse", f"{AUTH_COMMIT}^", cwd=REPO)
    built = build_bundle(REPO, parent, tmp_path / "stale.bundle")
    with pytest.raises(C1BundleError, match="does not contain|not the session commit"):
        _rt(tmp_path, commit=AUTH_COMMIT, bundle_path=Path(built["path"]),
            local_sha=built["sha256"], files=files)
    #: Two independent refusals, both retained: the checkout must succeed AND
    #: the resulting HEAD must equal the session commit.
    import inspect
    assert "does not contain" in inspect.getsource(B.checkout_from_bundle)
    assert "if head != session_commit" in inspect.getsource(B.roundtrip)


def _executable_text(fn) -> str:
    """Identifiers and live strings only.

    The first version of this check grepped the raw source and tripped on the
    gate's own docstring, which names `stage_c1_bundle.py` and says it uploads
    nothing -- precisely to explain that it does not. A check that cannot tell an
    explanation from a call site forces the explanation to be deleted.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    doc = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
           if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))}
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in doc:
                parts.append(node.value)
    return "\n".join(parts)


def test_mutation_dropping_the_remote_roundtrip_is_caught():
    """The gate must verify the REMOTE object, not a local record."""
    import autoinit_c1_launch as L
    code = _executable_text(L.bundle_staged_gate)
    assert "roundtrip" in code
    assert "hf_download" in code
    #: read-only: the gate must not stage or upload
    assert "stage_bundle" not in code
    assert "upload_file" not in code


def test_the_gate_is_wired_and_the_count_is_ten():
    """Ten now. Eight passed while attempt 1 died on transport; nine passed while
    attempt 2 died on the ROPE_OK staging input. Each abort added the gate that
    would have refused it at $0."""
    import autoinit_c1_launch as L

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", HEAD,
         "--bundle", canonical_bundle_name(HEAD)])
    spec = L.spec(args)
    names = [getattr(g, "__name__", "session_commit_and_lineage")
             for g in spec.precheck]
    assert "bundle_staged_gate" in names, names
    assert "rope_input_gate" in names, names
    assert len(spec.precheck) == 10, names


def test_preparation_is_a_separate_command_that_may_mutate_the_relay():
    p = REPO / "scripts/autoinit/stage_c1_bundle.py"
    assert p.is_file()
    src = p.read_text()
    assert "stage_bundle" in src
    #: it refuses a bundle for a commit that does not carry the authorization
    assert "does not carry" in src


def test_stage_refuses_to_overwrite_a_different_remote_object():
    import inspect

    from aadistill.autoinit.c1_bundle import stage_bundle
    src = inspect.getsource(stage_bundle)
    assert "Refusing to overwrite" in src

"""The relay credential the detached driver never had, and the $3.27 it cost.

E8b S1 measured all four initializations, then lost its publish step to

    MARKER:STAGE_FAILED:publish_step0:RepositoryNotFoundError

There was nothing wrong with the repository. The launcher starts the driver with an
explicit minimal environment — `env={"PYTHONPATH": ...}` — so `HF_TOKEN`, exported by
the setup shell, is not inherited. `HfApi().upload_file(token=None)` on a private
repo cannot see the repo and reports it as missing, which is why the failure reads as
a path or naming problem rather than a missing credential.

The same line appeared in `stage_fetch_step0`, so every downstream session would have
failed the same way *after* paying for setup — S2 and S3 on an A100.

These tests pin the fix: relay calls resolve the token through `hf_token()`, which
reads `/workspace/hf/token` (where setup stages it) and treats the environment
variable as an override rather than a requirement.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/pod"))

DRIVER = REPO / "scripts/pod/e8b_driver.py"
TOKEN_FILE = "/workspace/hf/token"


def _tree() -> ast.Module:
    return ast.parse(DRIVER.read_text())


def _token_kwargs() -> list[ast.keyword]:
    """Every `token=` keyword argument anywhere in the driver."""
    return [kw for node in ast.walk(_tree()) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == "token"]


def test_driver_has_token_kwargs_to_check():
    # A guard on the guard: if the relay calls are ever restructured away, the
    # assertions below would pass vacuously.
    assert _token_kwargs(), "no token= call sites found; this test has gone stale"


def test_no_relay_call_reads_the_token_from_the_environment_alone():
    for kw in _token_kwargs():
        src = ast.unparse(kw.value)
        assert "environ" not in src, (
            f"token={src} depends on environment inheritance, which the detached "
            "driver does not get — route it through hf_token()")
        assert src == "hf_token()", f"unexpected token source: {src}"


def test_token_file_path_is_where_setup_stages_it():
    setup = (REPO / "scripts/pod/e8b_setup.sh").read_text()
    # Setup writes WS=/workspace and reads $WS/hf/token; the driver must agree.
    assert "WS=/workspace" in setup
    assert 'export HF_TOKEN="$(cat $WS/hf/token)"' in setup
    assert f'Path("{TOKEN_FILE}")' in DRIVER.read_text(), (
        f"the driver must read {TOKEN_FILE}, the path setup stages")


def _driver_module():
    import importlib
    return importlib.import_module("e8b_driver")


def test_env_var_overrides_the_file(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert _driver_module().hf_token() == "from-env"


def test_falls_back_to_the_staged_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    staged = tmp_path / "token"
    staged.write_text("from-file\n")
    assert _driver_module().hf_token(staged) == "from-file"


def test_missing_credential_fails_loudly_rather_than_as_a_missing_repo(
        monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        _driver_module().hf_token(tmp_path / "absent")
    msg = str(exc.value)
    # The failure must name the credential, since the symptom it replaces
    # (RepositoryNotFoundError) pointed at the wrong thing entirely.
    assert "token" in msg.lower()
    assert "absent" in msg


def test_an_empty_staged_file_is_not_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    staged = tmp_path / "token"
    staged.write_text("   \n")
    with pytest.raises(SystemExit):
        _driver_module().hf_token(staged)


@pytest.mark.parametrize("stage", ["stage_publish_step0", "stage_fetch_step0"])
def test_the_relay_stages_call_the_resolver(stage):
    fn = next(n for n in ast.walk(_tree())
              if isinstance(n, ast.FunctionDef) and n.name == stage)
    calls = {ast.unparse(kw.value) for node in ast.walk(fn)
             if isinstance(node, ast.Call)
             for kw in node.keywords if kw.arg == "token"}
    assert calls == {"hf_token()"}, f"{stage} resolves the token as {calls}"

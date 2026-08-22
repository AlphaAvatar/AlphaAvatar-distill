"""A relay input names its repository, and every declared repo is prechecked.

Continuation attempt 2 died staging the first Stage-1 leaf: the launcher pushes
each `LOCAL_ASSET` by scp under a hard-coded 600 s timeout, and one 1.110 GiB
leaf needs **1.99 MB/s** to fit that against a dev box observed at 0.44–0.72
MB/s. The main relay could not take them either — 1.60 GiB of headroom against
5.55 GiB.

So the leaves moved to a private transport repo and the pod now **pulls** them.
The risk that creates is a *second* fetch path that nothing declares and no `$0`
gate checks — which is the exact defect `RelayInput` was introduced to remove.
These tests keep the repository inside the manifest contract:

* the declaration carries the repo, and the serialized env carries it too;
* the shared setup fetches each item from **its** declared repo and names none
  of its own;
* the `$0` precheck groups by repository and checks every path in every repo.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    MAIN_RELAY, RelayInput, SetupManifest,
)

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"
RUNNER = REPO / "src/aadistill/infrastructure/session_runner.py"
TRANSPORT = "AlphaAvatar/aadistill-transport"


# --- the declaration carries the repository ---------------------------------

def test_a_relay_input_defaults_to_the_main_relay():
    """Every existing declaration keeps its meaning without being touched."""
    r = RelayInput(path="a/b.bin", dest="artifacts/x")
    assert r.repo == MAIN_RELAY == "AlphaAvatar/aadistill-artifacts"
    assert r.as_record()["repo"] == MAIN_RELAY


def test_the_record_carries_the_repo_so_the_session_record_preserves_it():
    r = RelayInput(path="a/b.bin", dest="artifacts/x", repo=TRANSPORT,
                   sha256="ab" * 32)
    rec = r.as_record()
    assert rec == {"repo": TRANSPORT, "path": "a/b.bin", "dest": "artifacts/x",
                   "sha256": "ab" * 32, "also_stage_to": None}


def test_the_serialized_env_carries_the_repo():
    """`SESSION_RELAY_INPUTS` is what setup consumes; a repo it does not carry
    is a repo the shell would have to know, which is the defect."""
    man = SetupManifest(relay_inputs=(
        RelayInput(path="sci/a.bin", dest="artifacts/sci"),
        RelayInput(path="leaf/m.safetensors", dest="artifacts/leaf",
                   repo=TRANSPORT)))
    items = json.loads(man.relay_env())
    assert [i["repo"] for i in items] == [MAIN_RELAY, TRANSPORT]


# --- the shell names no repository of its own -------------------------------

def test_the_setup_block_fetches_from_the_declared_repo():
    src = SETUP.read_text()
    fetch = src.split("SESSION_RELAY_INPUTS")[2]
    assert "hf_hub_download(repo, path" in src
    assert 'repo = item.get("repo")' in src
    assert "FETCH FAILED {repo}:{path}" in src


def test_the_setup_block_refuses_an_input_with_no_repo():
    """Fail closed rather than silently defaulting: an item without a repo means
    the manifest and this shell disagree about who decides."""
    src = SETUP.read_text()
    assert "with no repo; this script" in src


def test_the_relay_fetch_block_hardcodes_no_repository():
    """The staging block must name no repository. The bundle fetch and the
    teacher/vLLM snapshot steps are separate concerns and keep their own."""
    src = SETUP.read_text()
    block = src.split("staging the session's declared science inputs")[1]
    block = block.split("FETCHEOF")[0]
    assert "AlphaAvatar/" not in block, (
        "the relay staging block names a repository of its own again")


# --- the $0 precheck is multi-repo ------------------------------------------

class _Spec:
    def __init__(self, inputs, local=()):
        self.setup = SimpleNamespace(relay_inputs=tuple(inputs),
                                     local_assets=tuple(local))
        self.precheck = ()


def runner_with(inputs, present_by_repo, monkeypatch, local=()):
    r = object.__new__(SR.SessionRunner)
    r.spec = _Spec(inputs, local)
    r.repo_root = REPO
    r.ev = {"timeline": []}
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: None
    r.a = SimpleNamespace()
    r.context = lambda **kw: SimpleNamespace(evidence=r.ev)
    asked: list[str] = []

    class FakeApi:
        def list_repo_files(self, repo, repo_type="model"):
            asked.append(repo)
            if repo not in present_by_repo:
                raise RuntimeError(f"404 {repo}")
            return list(present_by_repo[repo])

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return r, asked


def test_every_declared_repository_is_listed(monkeypatch):
    inputs = [RelayInput(path="sci/a.bin", dest="artifacts/sci"),
              RelayInput(path="leaf/m.bin", dest="artifacts/leaf", repo=TRANSPORT)]
    r, asked = runner_with(
        inputs, {MAIN_RELAY: ["sci/a.bin"], TRANSPORT: ["leaf/m.bin"]}, monkeypatch)
    assert r.run_prechecks() is True
    assert sorted(asked) == sorted([MAIN_RELAY, TRANSPORT]), (
        "a declared repository was never listed")
    ev = r.ev["precheck"]
    assert ev["relay_repos"] == {MAIN_RELAY: 1, TRANSPORT: 1}


def test_a_leaf_in_the_wrong_repo_fails_before_provider_creation(monkeypatch):
    """The bytes exist — in the other repo. That must not pass."""
    inputs = [RelayInput(path="leaf/m.bin", dest="artifacts/leaf", repo=TRANSPORT)]
    r, _ = runner_with(
        inputs, {MAIN_RELAY: ["leaf/m.bin"], TRANSPORT: []}, monkeypatch)
    assert r.run_prechecks() is False
    assert f"{TRANSPORT}:leaf/m.bin" in " ".join(r.ev["said"])


def test_changing_only_the_repo_id_fails(monkeypatch):
    """Path and hash untouched; repository wrong. Under the single-repo check
    this was invisible, because nothing compared the repo at all."""
    present = {MAIN_RELAY: ["sci/a.bin"], TRANSPORT: ["leaf/m.bin"]}
    good = [RelayInput(path="leaf/m.bin", dest="artifacts/leaf", repo=TRANSPORT)]
    r, _ = runner_with(good, present, monkeypatch)
    assert r.run_prechecks() is True

    typo = [RelayInput(path="leaf/m.bin", dest="artifacts/leaf",
                       repo="AlphaAvatar/aadistill-transport-typo")]
    r2, _ = runner_with(typo, present, monkeypatch)
    assert r2.run_prechecks() is False


def test_one_missing_remote_leaf_fails(monkeypatch):
    inputs = [RelayInput(path=f"leaf/{i}.bin", dest="artifacts/leaf",
                         repo=TRANSPORT) for i in range(5)]
    present = {TRANSPORT: [f"leaf/{i}.bin" for i in range(4)]}   # one short
    r, _ = runner_with(inputs, present, monkeypatch)
    assert r.run_prechecks() is False
    assert "leaf/4.bin" in " ".join(r.ev["said"])


def test_an_unreachable_repository_aborts_rather_than_assuming(monkeypatch):
    inputs = [RelayInput(path="leaf/m.bin", dest="artifacts/leaf", repo=TRANSPORT)]
    r, _ = runner_with(inputs, {MAIN_RELAY: []}, monkeypatch)   # transport 404s
    assert r.run_prechecks() is False
    assert f"cannot list {TRANSPORT}" in " ".join(r.ev["said"])


def test_the_ten_main_relay_science_inputs_are_unchanged(monkeypatch):
    """The whole point of a default: nothing that already worked moved."""
    sys.path.insert(0, str(REPO / "scripts/pod"))
    from session_specs import load_session_launcher, session_args

    mod = load_session_launcher("autoinit_phase_a_launch")
    spec = mod.spec(session_args(mod))
    inputs = spec.setup.relay_inputs
    assert len(inputs) == 10, f"expected the 10 science inputs, got {len(inputs)}"
    assert {r.repo for r in inputs} == {MAIN_RELAY}
    env = json.loads(spec.setup.relay_env())
    assert all(i["repo"] == MAIN_RELAY for i in env)


# --- setup and the precheck cannot diverge ----------------------------------

def test_a_repo_reachable_by_setup_is_also_checked_at_zero_dollars():
    """Structural: the precheck must derive its repositories from the SAME
    declaration setup fetches from. A second repo added to the manifest but not
    to the precheck would be discovered on a billing pod.
    """
    fn = next(n for n in ast.walk(ast.parse(RUNNER.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "run_prechecks")
    body = ast.unparse(fn)
    assert "by_repo.setdefault(r.repo, [])" in body, (
        "the precheck no longer groups declared inputs by repository")
    assert "for repo, paths in sorted(by_repo.items())" in body
    assert "list_repo_files(repo" in body
    # and it must not have gone back to a single hard-coded repository
    assert "aadistill-artifacts" not in body, (
        "the precheck names a repository instead of reading the declaration")

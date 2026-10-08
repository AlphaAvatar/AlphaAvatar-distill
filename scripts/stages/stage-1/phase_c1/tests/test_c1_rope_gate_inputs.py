"""C1's staged checkpoint inputs land where the RoPE gate looks for them.

Split from `tests/integration/test_session_architecture.py` by the 2026-10-03
convergence round. Both tests here load `autoinit_c1_launch` by name: one to read
C1's declared relay inputs, the other to run the real gate body against the tree
those inputs produce. The GATE's own behaviour -- that it refuses a missing
checkpoint and reads the stored rope base -- stays in core, where it is exercised
without any experiment's launcher.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest  # noqa: F401

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests"))

from support.session_specs import load_session_launcher, session_args  # noqa: E402

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"


def extract_rope_block() -> str:
    """The `ROPE_OK` check body, verbatim from the shell script."""
    text = SETUP.read_text()
    start = text.index('say "checking the RoPE base resolves in every venv')
    body = text[text.index('$PY -c "', start) + len('$PY -c "'):]
    body = body[:body.index('\n"\ndone')]
    assert "no staged checkpoint to check" in body and "stored_rope_base" in body
    return body.replace('\\"', '"')


def run_rope_check(repo_root):
    """Execute the real check body against a tree. Returns (rc, message)."""
    body = extract_rope_block().replace("/workspace/aad", str(repo_root))
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                         capture_output=True, text=True, timeout=300)
    return out.returncode, (out.stdout + out.stderr).strip()


def _c1_relay_inputs():
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod, ()))
    return [r for r in spec.setup.staged_relay_inputs()
            if r.path.startswith("stage1/qwen3_0p6b_init_v0/checkpoint/")]


def test_the_c1_checkpoint_inputs_land_where_the_rope_gate_globs(tmp_path,
                                                                 monkeypatch):
    """The REAL staging block, driven by C1's REAL declared manifest."""
    import hashlib
    import json as _json

    local = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    inputs, relay = [], {}
    for r in _c1_relay_inputs():
        src = local / Path(r.path).name
        if not src.is_file():
            pytest.skip(f"{src} is not present on this machine")
        relay[r.path] = src.read_bytes()
        inputs.append({"repo": MAIN_RELAY, "path": r.path, "dest": r.dest,
                       "sha256": r.sha256, "also_stage_to": None})
    assert len(inputs) == 4, [i["path"] for i in inputs]

    repo, fetched = run_staging(tmp_path, inputs, relay, monkeypatch)
    dest = repo / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    landed = {p.name for p in dest.iterdir() if p.is_file()}
    assert landed == {"config.json", "tokenizer.json", "tokenizer_config.json",
                      "chat_template.jinja"}, landed
    #: byte-identical to the pinned relay object
    got = hashlib.sha256((dest / "config.json").read_bytes()).hexdigest()
    assert got == "a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054"
    #: and the weights were never asked for
    assert not [p for p in fetched if p.endswith("model.safetensors")]

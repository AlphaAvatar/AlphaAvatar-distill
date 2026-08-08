"""Any pod setup that builds a vLLM venv must install ninja.

vLLM's flashinfer sampling path JIT-compiles a kernel the first time a sampler
runs, and shells out to `ninja` to build it. Without ninja the failure is
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` raised from
inside engine-core startup — so it does not surface at install time, or at import
time, or anywhere a setup script would notice. It surfaces at the first `LLM()`
construction, which is *after* uv sync, after the vLLM venv, and after every
checkpoint has been downloaded and hash-verified.

E4's setup installed it. E6's did not, and the pod reached its first evaluation
arm before dying. This test is why the next one will not: the requirement lives
in a test rather than in whichever previous script somebody happened to copy.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"

# Every setup script that creates a vLLM environment.
VLLM_SETUPS = sorted(
    p for p in POD.glob("*_setup.sh")
    if re.search(r"pip install [^\n]*\bvllm\b", p.read_text())
)


def test_at_least_one_vllm_setup_is_detected():
    """A detection bug here would make every assertion below vacuous."""
    assert VLLM_SETUPS, "no vLLM setup scripts found; the detector is broken"


@pytest.mark.parametrize("script", VLLM_SETUPS, ids=lambda p: p.name)
def test_vllm_setup_installs_ninja(script):
    text = script.read_text()
    assert re.search(r"apt-get install[^\n]*\bninja-build\b", text), (
        f"{script.name} builds a vLLM venv but never installs ninja-build; "
        "flashinfer will JIT-compile a sampling kernel and fail at the first "
        "LLM() call, after the whole setup has been paid for")


@pytest.mark.parametrize("script", VLLM_SETUPS, ids=lambda p: p.name)
def test_vllm_setup_verifies_ninja_is_on_path(script):
    """Installing is not the same as being callable; flashinfer needs it on PATH."""
    text = script.read_text()
    assert re.search(r"command -v ninja", text), (
        f"{script.name} installs ninja-build but never checks that `ninja` "
        "resolves; the package and the binary name differ and a silent PATH "
        "problem would present as the same late failure")

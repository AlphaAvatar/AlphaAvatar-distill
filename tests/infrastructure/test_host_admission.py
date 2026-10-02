"""A host whose properties make the result incomparable must be redrawn.

`a3_attempt35` trained three probes, generated the first, and was refused by
the protocol admission at `$4.33`: every material generation field matched
attempt75's controls, and the host NVIDIA driver BRANCH had moved 580 -> 595,
which `generation_compat` declares a real runtime event rather than
provenance. The refusal was correct. What was wrong is that the question was
asked after the money was spent, on a property knowable one ssh round trip
after the pod answers.

`SessionSpec.host_admission` asks it before setup, and a refusal is
redrawable exactly like a cold host. Default `None` admits everything, so no
existing session changes behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.infrastructure.session import SessionSpec  # noqa: E402


def test_the_hook_defaults_to_admitting_every_host():
    """Every session that predates this is unaffected, by construction."""
    import dataclasses

    field = next(f for f in dataclasses.fields(SessionSpec)
                 if f.name == "host_admission")
    assert field.default is None


def test_a_refusal_is_a_redrawable_outcome():
    """Read from the runner's own condition, not restated.

    A refusal that ended the session instead of redrawing would turn a host
    lottery into a lost chain, which is the opposite of the repair.
    """
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert 'return "host_not_admitted"' in src, (
        "the runner no longer reports a non-admitted host")
    assert '"host_not_admitted"' in src.split("host_draws")[0].rsplit(
        "if (outcome in", 1)[-1] + "host_draws", (
        "`host_not_admitted` is not in the redrawable set beside `cold`")


def test_the_check_runs_before_setup_and_after_the_image_is_confirmed():
    """Order is the whole point: after the provider names the image, before a
    dollar of setup. Asserted on the source order, because the alternative is
    to run a session."""
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    i_identity = src.index('image identity {self.image_digest}')
    i_admit = src.index("admit = getattr(self.spec, \"host_admission\", None)")
    i_setup = src.index('draw {draw}: running setup')
    assert i_identity < i_admit < i_setup, (
        "the admission must sit between the confirmed image identity and the "
        "setup invocation")


def test_a_raising_check_refuses_rather_than_admitting():
    """Fail closed. A check that cannot answer has not answered yes."""
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    block = src[src.index("admit = getattr(self.spec"):
                src.index('draw {draw}: running setup')]
    assert "except Exception" in block and "ok, why = False" in block, (
        "a raising host-admission check must refuse, not admit")


# --- A3's rule, which is the only caller today ----------------------------

def test_a3_admits_the_controls_driver_branch_and_refuses_another():
    from experiments.phase_c3 import a3_session as A3S

    want = A3S.control_driver_branch()
    assert want == "580", (
        f"attempt75's controls were measured on branch {want!r}; this test "
        "pins the fact that A3's admission is derived from them")

    for patch in ("580.159.03", "580.126.09", "580.65.06"):
        ok, why = A3S.host_admission(f"img@{patch}")
        assert ok, f"{patch} is in the controls' branch and was refused: {why}"
        assert "provenance" in why

    for other in ("595.91.07", "570.124.06"):
        ok, why = A3S.host_admission(f"img@{other}")
        assert not ok, f"{other} is a different branch and was admitted"
        assert "branch" in why and "Redraw" in why


def test_a3_fails_closed_when_the_controls_branch_cannot_be_read(monkeypatch):
    from experiments.phase_c3 import a3_session as A3S

    monkeypatch.setattr(A3S, "control_driver_branch", lambda *a, **k: None)
    ok, why = A3S.host_admission("img@580.159.03")
    assert not ok and "unknown comparability premise" in why


def test_a3_refuses_an_image_identity_with_no_driver():
    """`read_image_digest` appends the driver only when
    `/etc/podinfo/image_digest` is absent, so a bare tag is a host whose
    comparability cannot be established."""
    from experiments.phase_c3 import a3_session as A3S

    ok, why = A3S.host_admission("runpod/pytorch:1.1.0-cu1300")
    assert not ok and "no" in why.lower() and "driver" in why


def test_the_a3_session_wires_it():
    sys.path.insert(0, str(REPO / "scripts/pod"))
    sys.path.insert(0, str(REPO / "tests/pod"))
    from session_specs import load_session_launcher, session_args

    mod = load_session_launcher("autoinit_a3_launch")
    spec = mod.spec(session_args(mod))
    assert spec.host_admission is not None, (
        "A3 declares no host admission, so a wrong-branch host would be paid "
        "for and refused at the protocol admission instead")
    ok, _ = spec.host_admission("img@595.91.07")
    assert ok is False

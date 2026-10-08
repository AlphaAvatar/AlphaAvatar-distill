"""A3's concrete host-admission rule: the controls driver branch.

Split from `tests/infrastructure/test_host_admission.py` by the 2026-10-03
convergence round. That file keeps the MECHANISM — the hook defaults to
admitting, a refusal is redrawable, it is asked after the provider-confirmed
image and before setup, a raising check refuses — proved with synthetic
callables. These four are about A3's instance: which NVIDIA driver branch its
controls were generated on, that it fails closed when the branch cannot be read,
and that the launcher actually wires it.

`a3_attempt35` trained three probes and was refused by the protocol admission at
`$4.46`: every material generation field matched attempt75's controls and the
host driver BRANCH had moved 580 -> 595, which `generation_compat` declares a
real runtime event. The refusal was correct; asking after the money was spent was
not.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))


def test_a3_admits_the_controls_driver_branch_and_refuses_another():
    from stages.phase_a3 import a3_session as A3S

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
    from stages.phase_a3 import a3_session as A3S

    monkeypatch.setattr(A3S, "control_driver_branch", lambda *a, **k: None)
    ok, why = A3S.host_admission("img@580.159.03")
    assert not ok and "unknown comparability premise" in why


def test_a3_refuses_an_image_identity_with_no_driver():
    """`read_image_digest` appends the driver only when
    `/etc/podinfo/image_digest` is absent, so a bare tag is a host whose
    comparability cannot be established."""
    from stages.phase_a3 import a3_session as A3S

    ok, why = A3S.host_admission("runpod/pytorch:1.1.0-cu1300")
    assert not ok and "no" in why.lower() and "driver" in why


def test_the_a3_session_wires_it():
    sys.path.insert(0, str(REPO / "scripts/pod"))
    sys.path.insert(0, str(REPO / "tests/pod"))
    from support.session_specs import load_session_launcher, session_args

    mod = load_session_launcher("autoinit_a3_launch")
    spec = mod.spec(session_args(mod))
    assert spec.host_admission is not None, (
        "A3 declares no host admission, so a wrong-branch host would be paid "
        "for and refused at the protocol admission instead")
    ok, _ = spec.host_admission("img@595.91.07")
    assert ok is False

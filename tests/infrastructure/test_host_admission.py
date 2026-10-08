"""A host whose properties make the result incomparable must be redrawn.

THE MECHANISM, with synthetic hooks. The four A3 tests that drove A3's own
`host_admission` callable and asserted its controls driver branch moved to
`scripts/stages/stage-1/phase_a3/tests/test_a3_host_admission.py` in the
2026-10-03 convergence round: a core test should prove the hook works for an
arbitrary callable, not that one closed experiment's instance is wired a
particular way.

`a3_attempt35` trained three probes, generated the first, and was refused by
the protocol admission: every material generation field matched
attempt75's controls, and the host NVIDIA driver BRANCH had moved 580 -> 595,
which `generation_compat` declares a real runtime event rather than
provenance. The refusal was correct. What was wrong is that the question was
asked after the money was spent -- that session cost `$4.46`, owned by its
closeout -- on a property knowable one ssh round trip after the pod answers.

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


#: WHAT IS DELIBERATELY NOT HERE. A3's concrete admission rule — the controls
#: driver branch, its fail-closed read, and the launcher that wires it — is A3's
#: and lives in A3's suite. The generic properties above (default-admit, refusal
#: is redrawable, asked after the image and before setup, a raising check
#: refuses) are what every future session inherits, and none of them needs an
#: experiment package to be true.

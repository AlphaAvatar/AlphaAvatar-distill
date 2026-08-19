"""What the shared setup REQUIRES, against what each session DECLARES.

Two paid pods have now died on the same reasoning error, sixteen days apart:

    device canary retry   $0.0637   LOCAL_ASSETS = () because it needed neither
    measurement           $0.0700   LOCAL_ASSETS = () because it reads only the
                                    calibration and the teacher, both from the relay

Both statements were true. Both were about what the session *needed*.
`autoinit_preflight_setup.sh` runs `scripts/autoinit/verify_frozen_assets.py`
**unconditionally** at the `ASSETS_READY` gate, and that verifier checks its
frozen roots whatever the session is doing — so what binds is what the shared
setup requires, not what the session uses.

The 2026-08-18 `SESSION_ASSETS` fix stopped the setup *copying* assets a session
had not declared. It did not, and could not, tell a session which assets it must
declare: that is this file.

Derived, not transcribed
------------------------
The required roots come from `verify_frozen_assets.FROZEN` itself. Encoding
today's two filenames here would close today's instance and leave the class open
— a third frozen asset added to the verifier would sail past. The third mutation
below is the one that matters.

Declarations, not the filesystem
--------------------------------
The comparison is `install_to/dest_name` against the verifier's `root`. Checking
whether the files exist on *this* machine would pass on the dev box, where they
all do, and say nothing about a pod — which is the exact blindness that let both
failures through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

from session_specs import SESSION_LAUNCHERS, all_specs  # noqa: E402


def verifier_required_local_roots() -> set[str]:
    """Repository-relative roots `verify_frozen_assets.py` will demand.

    Imported from the verifier so the requirement has ONE definition. A future
    frozen asset joins this set the moment it joins the verifier, without anyone
    remembering to update a test.
    """
    from verify_frozen_assets import FROZEN

    return {spec["root"].strip("/") for spec in FROZEN.values()}


def installed_local_roots(spec) -> set[str]:
    """Where a session's declared assets LAND, in repository terms.

    `LocalAsset(repo_path, dest_name, install_to)` is copied to
    `install_to/dest_name` on the pod, which is what the verifier then looks for.
    """
    return {f"{a.install_to.strip('/')}/{a.dest_name}"
            for a in spec.setup.local_assets}


def test_the_verifier_actually_declares_roots():
    """A discovery bug here would make every assertion below vacuous."""
    roots = verifier_required_local_roots()
    assert roots, "no frozen roots found; the extractor is broken"
    assert all(r.startswith("artifacts/") for r in roots), roots


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_session_installs_every_root_the_shared_setup_verifies(name, extra):
    """`verifier_required_local_roots ⊆ session_installed_local_roots`.

    The invariant the two failures violated, checked against declarations so it
    gives the same answer on the dev box, in the simulator and on a pod.
    """
    spec = dict((n, s) for n, _m, _a, s in all_specs())[name]
    required = verifier_required_local_roots()
    installed = installed_local_roots(spec)
    missing = sorted(required - installed)
    assert not missing, (
        f"{name} does not install {missing}, which "
        "scripts/autoinit/verify_frozen_assets.py checks unconditionally at the "
        "shared setup's ASSETS_READY gate. A session declares what the SETUP "
        "requires, not what the session reads — declaring only what it needs is "
        "what cost the device-canary retry $0.0637 and the measurement $0.0700.")


def test_the_contract_holds_for_every_session_at_once():
    """Stated once more as a set relation, so the invariant is readable as the
    one line it is rather than inferred from a parametrized sweep."""
    required = verifier_required_local_roots()
    for name, _m, _a, spec in all_specs():
        assert required <= installed_local_roots(spec), name


def test_a_session_may_install_more_than_the_verifier_requires():
    """Subset, not equality. The continuation installs a third asset — its
    permanent-control records — and that is not a violation of anything."""
    required = verifier_required_local_roots()
    installed = {n: installed_local_roots(s) for n, _m, _a, s in all_specs()}
    assert any(len(v) > len(required) for v in installed.values()), (
        "no session installs an extra asset; if that is now true the test is "
        "vacuous and should be re-derived rather than deleted")


def test_the_requirement_is_read_from_the_verifier_not_copied_into_this_file():
    """The class, not the instance.

    If this file hard-coded `state_eval_v1` and `recovery_search_v2`, a third
    frozen asset would be required by the setup and unnoticed here — which is the
    same hidden contract in a new place.
    """
    src = Path(__file__).read_text()
    body = src[src.index("def verifier_required_local_roots"):
               src.index("def installed_local_roots")]
    assert "from verify_frozen_assets import FROZEN" in body
    for literal in ("state_eval_v1", "recovery_search_v2"):
        assert literal not in body, (
            f"the required roots are transcribed ({literal!r}) rather than "
            "derived; a new frozen asset would not be caught")

"""The Phase-A search entry point's profile/items seam.

`run_phase_a_search` used to resolve `active_profile` and `calibration`
independently: `profile or DOMAIN_BALANCED_V1` for the label, and an
unconditional `DOMAIN_BALANCED_V1.resolve()` for the items. Passing `profile=X`
without `calibration_items=` therefore produced a run *labelled* X and *fed* the
domain-balanced mixture, and every operator's statistics, every state id and
every recorded profile hash would have said X.

Nothing triggered it, because every call site passed both arguments or neither —
and Phase B is the first thing that would wire a second profile. These tests pin
the repaired contract at the seam rather than at the whole function, so they cost
nothing and run everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))
sys.path.insert(0, str(REPO / "tests/autoinit"))

from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1  # noqa: E402
from conftest import make_profile  # noqa: E402
from phase_a_search import build_calibration_loader, resolve_profiles  # noqa: E402

A = make_profile("balanced", seed=1)
B = make_profile("reasoning", seed=2)


# --- which profiles are searched -------------------------------------------


def test_the_default_is_still_exactly_phase_a():
    assert resolve_profiles(None, None) == (DOMAIN_BALANCED_V1,)


def test_a_single_profile_and_an_explicit_tuple_both_work():
    assert resolve_profiles(A, None) == (A,)
    assert resolve_profiles(None, (A, B)) == (A, B)


def test_ambiguous_or_degenerate_profile_arguments_are_refused():
    with pytest.raises(ValueError, match="not both"):
        resolve_profiles(A, (A, B))
    with pytest.raises(ValueError, match="at least one"):
        resolve_profiles(None, ())
    with pytest.raises(ValueError, match="repeats a profile"):
        resolve_profiles(None, (A, A))


# --- what the loader answers ------------------------------------------------


def test_the_loader_answers_per_profile_from_a_mapping():
    items = {A.qualified_id: ["a"], B.qualified_id: ["b"]}
    load = build_calibration_loader((A, B), items, REPO)
    assert load(A) == ["a"] and load(B) == ["b"], "the loader ignored its argument"


def test_a_mapping_that_does_not_cover_every_searched_profile_is_refused():
    with pytest.raises(ValueError, match="missing"):
        build_calibration_loader((A, B), {A.qualified_id: ["a"]}, REPO)


def test_a_bare_sequence_is_refused_for_a_multi_profile_search():
    """The historical shape. A list cannot say which mixture it is."""
    with pytest.raises(ValueError, match="cannot say which"):
        build_calibration_loader((A, B), ["items"], REPO)


def test_a_bare_sequence_is_accepted_for_one_profile_and_bound_to_it():
    load = build_calibration_loader((A,), ["items"], REPO)
    assert load(A) == ["items"]
    # The repaired behaviour: asked about a profile it was not built for, it
    # raises instead of handing back the one mixture it happens to hold.
    with pytest.raises(ValueError, match="labelled with one profile and fed another"):
        load(B)


def test_the_omitted_case_resolves_each_profile_itself_rather_than_defaulting():
    """The defect, at its narrowest: an unbuilt profile must NOT silently
    resolve to the domain-balanced mixture."""
    from aadistill.autoinit.calibration import REASONING_HEAVY_V1, CalibrationError

    load = build_calibration_loader((REASONING_HEAVY_V1,), None, REPO)
    with pytest.raises(CalibrationError, match="not built"):
        load(REASONING_HEAVY_V1)

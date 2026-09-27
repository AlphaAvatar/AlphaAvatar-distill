"""The C3 preregistration binds itself, and the documents quote what it says.

WHY THIS EXISTS. On 2026-09-28 the preregistration carried
`preregistration_sha256: fdf1d4c2...`, `logs/state/current.md` and
`logs/state/current.json` both described the design as "hash-bound
(fdf1d4c266facd95)", and the value reproduced under **no** canonicalization of
the document. It had been written by hand: unlike C1, phase B and
continuation B, the C3 plan had no producer script -- and, more to the point,
no consumer either. Nothing in `src/`, `scripts/` or `tests/` read the file,
so a stamp that bound nothing could sit in a committed tree and be reported to
the maintainer as a binding.

It was caught before any launch, so the fix was to stamp the true hash over an
unchanged body. That is only safe while no result exists; once a probe has
run, a preregistration whose stamp does not verify is a scientific problem and
not a clerical one, because nothing then distinguishes "the stamp was wrong"
from "the document moved after the science started". This file is what makes
the difference detectable at all.

The convention is the one `autoinit_c1_launch.py` verifies: `sha256_json` over
the document with `preregistration_sha256` removed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

PREREG = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"
STATE_MD = REPO / "logs/state/current.md"
SNAPSHOT = REPO / "logs/state/current.json"
FIELD = "preregistration_sha256"


def _prereg() -> dict:
    if not PREREG.is_file():
        pytest.skip(f"{PREREG.relative_to(REPO)} does not exist")
    return json.loads(PREREG.read_text())


def _body_digest(doc: dict) -> str:
    return sha256_json({k: v for k, v in doc.items() if k != FIELD})


def test_the_preregistration_stamp_binds_its_own_body():
    """The defect itself. A stamp that reproduces nothing binds nothing."""
    doc = _prereg()
    stated = doc.get(FIELD)
    assert stated, f"the C3 preregistration declares no {FIELD}"
    got = _body_digest(doc)
    assert got == stated, (
        f"{FIELD} is {stated} but sha256_json(body) is {got}. The stamp does "
        f"not bind this document. If no probe has run, restamp it and say so "
        f"in the document; if one has, the document moved after the science "
        f"started and that is a maintainer decision, not a repair.")


def test_the_convention_is_the_one_the_launcher_verifies():
    """Not a private hash. C1's launcher must agree about what is hashed.

    A correct-looking stamp computed a different way would pass the test
    above and be refused by the paid gate -- the failure mode the C1 harness
    digest already produced once.
    """
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert f'k != "{FIELD}"' in src, (
        "autoinit_c1_launch.py no longer excludes only the stamp field; this "
        "test's convention and the launcher's have drifted apart")
    doc = _prereg()
    #: Recomputed the launcher's way, from its own exclusion, not from ours.
    launcher_way = sha256_json({k: v for k, v in doc.items() if k != FIELD})
    assert launcher_way == doc[FIELD]


def test_every_document_quoting_the_stamp_quotes_the_live_one():
    """The second half of the defect: two state documents quoted the bad value.

    A stamp can be corrected in the plan and left stale in the handoff, and
    the handoff is what a reader believes -- so the prefix that appears in
    prose is checked against the file, not against itself.

    The window must name C3. A first cut matched any sentence containing
    "prereg" and flagged three digests that belong to C1's preregistration
    and Phase B's amendments ledger -- a guard that reports other
    experiments' correct hashes as this one's drift would be turned off
    within a round.
    """
    doc = _prereg()
    live = doc[FIELD]
    offenders = []
    for path in (STATE_MD, SNAPSHOT):
        if not path.is_file():
            continue
        text = path.read_text()
        for window in re.findall(r"[^.]*?(?:prereg|hash-bound)[^.]*\.", text,
                                 flags=re.IGNORECASE):
            if not re.search(r"\bc3\b|c3_preregistration", window, re.IGNORECASE):
                continue
            for hexrun in re.findall(r"\b[0-9a-f]{12,64}\b", window):
                if not live.startswith(hexrun):
                    offenders.append(
                        f"{path.relative_to(REPO)}: {hexrun} is presented as "
                        f"C3's preregistration hash; the file says {live[:16]}")
    #: Belt and braces, and the one that would have caught the original
    #: defect on its own: the known-bad value may not survive anywhere.
    for path in (STATE_MD, SNAPSHOT, PREREG):
        if path.is_file() and "fdf1d4c266facd95" in path.read_text():
            if path == PREREG and "_sha256_correction" in _prereg():
                continue          # the correction note records it deliberately
            offenders.append(f"{path.relative_to(REPO)} still carries the "
                             f"unbinding stamp fdf1d4c266facd95")
    assert not offenders, offenders


def test_the_correction_is_recorded_rather_than_silent():
    """A restamp must leave a trace, because the body did not change.

    Without this note the two versions are indistinguishable to a reader: the
    body hashes the same before and after, so only prose can say that the
    stamp moved and the science did not.

    Unconditional, deliberately. A first cut skipped when the key was absent,
    which is the wrong default twice over: deleting the note would turn the
    test green rather than red, and the skip-predicate audit had no
    classification for it, so the pod and the dev box were not guaranteed to
    decide it the same way. This document WAS corrected; that is a fact about
    it, not a branch. A future C3 preregistration written correctly from the
    start would drop this test in the same commit that drops the note.
    """
    doc = _prereg()
    assert "_sha256_correction" in doc, (
        "the 2026-09-28 restamp is no longer recorded; a reader cannot then "
        "tell that the stamp moved and the body did not")
    note = doc["_sha256_correction"]
    assert "fdf1d4c2" in note, "the corrected-from value is not recorded"
    for claim in ("NOTHING IN THE DOCUMENT CHANGED", "before any launch"):
        assert claim in note, f"the correction does not state {claim!r}"


def test_the_design_the_stamp_binds_is_the_three_arm_one():
    """Guard the content the stamp is protecting, not only its integrity.

    A stamp that binds correctly to the WRONG document is worse than a stamp
    that binds to nothing, because it looks verified. The supplement fixed
    three arms, three seeds and which contrast is primary; all three are
    frozen-before-results facts, so they belong here.
    """
    doc = _prereg()
    blob = json.dumps(doc)
    assert "217230555" in blob and "1151307191" in blob and "2045359208" in blob, (
        "the three mechanically derived seeds are not in the preregistration")
    assert "654678655" in blob, "the bootstrap seed is not preregistered"
    #: The primary contrast is named, and it is the operator-isolating one.
    assert re.search(r"causal[_-]?B1\s*(?:-|−|minus)\s*B", blob, re.IGNORECASE), (
        "the primary contrast causal-B1 - B is not stated")

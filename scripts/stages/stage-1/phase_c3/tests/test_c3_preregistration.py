"""The C3 preregistration binds itself, and says ONE thing about each contrast.

WHY THIS EXISTS, in two layers.

**Layer 1 — the stamp.** On 2026-09-28 the document carried
`preregistration_sha256: fdf1d4c2...`, both state documents described the
design as "hash-bound (fdf1d4c266facd95)", and the value reproduced under
**no** canonicalization. It had been written by hand: unlike C1, phase B and
continuation B, the C3 plan had no producer script -- and, more to the point,
no consumer either, so a stamp that bound nothing sat in a committed tree and
was reported to the maintainer as a binding.

**Layer 2 — the contradiction the stamp was hiding.** Fixing the stamp proved
nothing about the document being *coherent*. Maintainer review of `b937aebb`
found that the same frozen document said three different things:

    claim_boundary.primary_contrast   "causal-B1 - incumbent B"
    estimand.primary                  "causal-KL-B3 treatment - incumbent B"
    decision_rule._applies_to         "the PRIMARY contrast, causal-B1 - B"

and that `terminal_outcomes.GO` promoted `B3/length_sorted_v1` automatically
while `interpretation_matrix` case B said B3 must NOT be auto-promoted. A
correctly-stamped document that contradicts itself is worse than an unstamped
one, because the stamp makes it look checked. So this file verifies the
*content* as well as the integrity: every place that names the primary
contrast must name the same one, and the decision rule must have exactly one
owner.

Both were caught before any launch and before any behavioural result, which
is the only time a preregistration may be repaired. After formal execution
begins there are no further scientific edits.

The stamp convention is the one `autoinit_c1_launch.py` verifies: `sha256_json`
over the document with `preregistration_sha256` removed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

PREREG = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"
STATE_MD = REPO / "logs/state/current.md"
SNAPSHOT = REPO / "logs/state/current.json"
FIELD = "preregistration_sha256"

#: Every stamp this document has ever carried and no longer does. A live
#: document quoting one of these is quoting a superseded freeze.
SUPERSEDED_STAMPS = (
    "fdf1d4c266facd95",      # hand-written, bound nothing
    "6d1d1121e5fded88",      # correct, but of the self-contradictory body
    "1632066081200aff",      # corrected contrasts, before `authorizes: nothing`
)

PRIMARY = "causal-B1 - incumbent B"
SECONDARY = "causal-B3 - causal-B1"
PRACTICAL = "causal-B3 - incumbent B"


def _prereg() -> dict:
    if not PREREG.is_file():
        pytest.skip(f"{PREREG.relative_to(REPO)} does not exist")
    return json.loads(PREREG.read_text())


def _norm(text: str) -> str:
    """Compare contrasts by content, not by dash or spacing."""
    return re.sub(r"\s+", " ", text.replace("−", "-")).strip().lower()


# --------------------------------------------------------------------------
# Layer 1: integrity
# --------------------------------------------------------------------------

def test_the_preregistration_stamp_binds_its_own_body():
    """The original defect. A stamp that reproduces nothing binds nothing."""
    doc = _prereg()
    stated = doc.get(FIELD)
    assert stated, f"the C3 preregistration declares no {FIELD}"
    got = sha256_json({k: v for k, v in doc.items() if k != FIELD})
    assert got == stated, (
        f"{FIELD} is {stated} but sha256_json(body) is {got}. The stamp does "
        f"not bind this document. If no probe has run, restamp it and record "
        f"the correction; if one has, the document moved after the science "
        f"started and that is a maintainer decision, not a repair.")


def test_the_convention_is_the_one_the_launcher_verifies():
    """Not a private hash. C1's launcher must agree about what is hashed."""
    src = (REPO / "scripts/stages/stage-1/phase_c1/autoinit_c1_launch.py").read_text()
    assert f'k != "{FIELD}"' in src, (
        "autoinit_c1_launch.py no longer excludes only the stamp field; this "
        "test's convention and the launcher's have drifted apart")
    doc = _prereg()
    assert sha256_json({k: v for k, v in doc.items() if k != FIELD}) == doc[FIELD]


def test_no_live_document_quotes_a_superseded_stamp():
    """A stamp can be corrected in the plan and left stale in the handoff."""
    doc = _prereg()
    live = doc[FIELD]
    offenders = []
    for path in (STATE_MD, SNAPSHOT, PREREG):
        if not path.is_file():
            continue
        text = path.read_text()
        for stale in SUPERSEDED_STAMPS:
            if stale not in text:
                continue
            #: The plan's own correction history names them deliberately.
            if path is PREREG:
                continue
            offenders.append(f"{path.relative_to(REPO)} quotes the superseded "
                             f"stamp {stale}; live is {live[:16]}")
    #: And any hex presented as C3's preregistration hash must BE it.
    for path in (STATE_MD, SNAPSHOT):
        if not path.is_file():
            continue
        for window in re.findall(r"[^.]*?(?:prereg|hash-bound)[^.]*\.",
                                 path.read_text(), flags=re.IGNORECASE):
            if not re.search(r"\bc3\b|c3_preregistration", window, re.IGNORECASE):
                continue                    # C1's and phase B's, not ours
            for hexrun in re.findall(r"\b[0-9a-f]{12,64}\b", window):
                if not live.startswith(hexrun):
                    offenders.append(
                        f"{path.relative_to(REPO)}: {hexrun} is presented as "
                        f"C3's preregistration hash; the file says {live[:16]}")
    assert not offenders, offenders


def test_the_correction_history_is_honest_about_what_changed():
    """A restamp and a body edit are different events and must read that way.

    The first correction could truthfully say the body was unchanged. The
    second could not -- it changed `estimand` and `terminal_outcomes` -- and
    carrying the old "NOTHING IN THE DOCUMENT CHANGED" wording forward would
    have been a false claim inside the very document whose integrity is the
    point. So the history is a list of events, each stating whether the body
    moved, and the body-changing one must name what it touched and what it
    did not.
    """
    doc = _prereg()
    hist = doc.get("_correction_history")
    assert isinstance(hist, list) and hist, "no correction history recorded"
    assert "_sha256_correction" not in doc, (
        "the superseded single-correction field is still present; it claimed "
        "NOTHING IN THE DOCUMENT CHANGED, which stopped being true")

    body_edits = [e for e in hist if e.get("body_changed")]
    assert body_edits, "no body-changing correction recorded"
    for e in body_edits:
        assert e.get("_what_changed"), "a body edit that names nothing it changed"
        assert e.get("_what_did_not_change"), (
            "a body edit must also name what it did NOT touch; that is what "
            "makes 'the science is unchanged' checkable rather than asserted")
        assert e.get("superseded_sha256"), "no superseded stamp recorded"
        assert "no C3 behavioural result exists" in e.get("_why_this_is_legitimate", "")
    #: Every superseded stamp this test knows about is accounted for.
    named = json.dumps(hist)
    for stale in SUPERSEDED_STAMPS:
        assert stale in named, f"{stale} is not accounted for in the history"


# --------------------------------------------------------------------------
# Layer 2: the document says ONE thing about each contrast
# --------------------------------------------------------------------------

def test_every_field_that_names_the_primary_contrast_names_the_same_one():
    """The b937aebb defect. Three fields, three owners, two answers.

    `claim_boundary` said causal-B1, `estimand` said causal-B3, and
    `decision_rule._applies_to` said causal-B1 again. Asking only one of them
    would not have caught it -- two of the three were already right.
    """
    doc = _prereg()
    sites = {
        "claim_boundary.primary_contrast": doc["claim_boundary"]["primary_contrast"],
        "estimand.primary.contrast": doc["estimand"]["primary"]["contrast"],
        "decision_rule._applies_to": doc["decision_rule"]["_applies_to"],
    }
    wrong = {k: v for k, v in sites.items()
             if "causal-b1" not in _norm(v) or "causal-b3" in _norm(v)}
    assert not wrong, (
        f"these fields do not name the primary contrast {PRIMARY!r}: {wrong}. "
        f"The primary is the OPERATOR-ISOLATION contrast; B3 is a numerical "
        f"protocol, not the scientific hypothesis.")


def test_the_three_contrast_roles_are_distinct_and_correctly_assigned():
    """Primary, secondary and practical each name a different pair of arms."""
    est = _prereg()["estimand"]
    for role, expected in (("primary", PRIMARY), ("secondary", SECONDARY),
                           ("practical", PRACTICAL)):
        got = est[role]["contrast"]
        assert _norm(got) == _norm(expected), (
            f"estimand.{role}.contrast is {got!r}, expected {expected!r}")
        assert est[role]["metric"] == "correct_overall"
    assert len({_norm(est[r]["contrast"])
                for r in ("primary", "secondary", "practical")}) == 3


def test_only_the_primary_contrast_owns_the_verdict():
    """One decision owner. The secondaries are reported, never promoted."""
    doc = _prereg()
    est, rule = doc["estimand"], doc["decision_rule"]
    assert "causal-b1" in _norm(rule["_applies_to"])
    assert est["primary"]["_owns"], "the primary does not claim the verdict"
    assert est["secondary"]["_does_not_replace_the_primary"] is True
    assert est["practical"][
        "_may_not_replace_the_primary_after_results_are_visible"] is True
    #: The decision rule's thresholds exist and are not duplicated per role.
    for role in ("secondary", "practical"):
        blob = json.dumps(est[role]).lower()
        for verdict in ("go", "no_go", "inconclusive"):
            assert f'"{verdict}"' not in blob, (
                f"estimand.{role} carries a {verdict} rule; only the primary "
                f"may own a verdict")


def test_all_three_contrasts_share_one_aggregation_and_one_seed_set():
    """Different differences, identical machinery -- or they are not paired."""
    doc = _prereg()
    est = doc["estimand"]
    assert est["aggregation"] == ("prompt mean of the mean over the three "
                                 "fixed paired seeds")
    shared = est["_all_three_use_the_same_machinery"].lower()
    for token in ("identical aggregation", "bootstrap", "same three"):
        assert token in shared, f"the shared-machinery clause omits {token!r}"
    assert len(doc["seeds"]["recovery"]) == 3


def test_GO_does_not_automatically_promote_the_B3_protocol():
    """The second contradiction: GO auto-promoted B3, case B forbade it."""
    doc = _prereg()
    go = doc["terminal_outcomes"]["GO"]
    assert isinstance(go, dict), "GO is still free prose, not a checkable object"
    assert go["_no_automatic_promotion_of_b3"] is True
    assert "not automatically decided" in go["b1_vs_b3"].lower()
    assert "maintainer" in go["b1_vs_b3"].lower()
    #: And it must not read as a promotion instruction.
    assert "promote" not in _norm(go["means"]), (
        "the GO verdict still describes promoting a protocol; it settles the "
        "OPERATOR question only")
    #: C4 eligible, never started.
    for verdict in ("GO", "NO_GO", "INCONCLUSIVE"):
        c4 = json.dumps(doc["terminal_outcomes"][verdict]).lower()
        assert "not authorized" in c4 or "do not start c4" in c4, (
            f"terminal_outcomes.{verdict} does not forbid starting C4")


def test_no_b3_vs_b1_non_inferiority_margin_was_invented():
    """The correct response to a missing margin is not to make one up."""
    doc = _prereg()
    blob = json.dumps(doc).lower()
    assert "non-inferiority" not in blob or "none is invented" in blob, (
        "a non-inferiority notion appears without the clause refusing to "
        "invent one")
    #: The only preregistered numeric threshold is the primary SESOI.
    thresholds = re.findall(r"[+-]?0\.0\d{2,}", json.dumps(doc["decision_rule"]))
    assert set(thresholds) <= {"+0.010", "0.010"}, (
        f"decision_rule carries thresholds beyond the +0.010 SESOI: {thresholds}")


def test_the_terminal_semantics_and_the_interpretation_matrix_agree():
    """Case C and the GO clause must say the same thing about a strong B3."""
    doc = _prereg()
    case_c = doc["interpretation_matrix"]["C"].lower()
    clause = doc["terminal_outcomes"][
        "_if_b3_looks_strong_while_primary_is_not_GO"].lower()
    for text, where in ((case_c, "interpretation_matrix.C"),
                        (clause, "terminal_outcomes clause")):
        assert "relabel" in text, f"{where} does not forbid relabelling"
    assert "case c" in clause, "the clause does not cite the matrix it mirrors"


def test_the_design_the_stamp_binds_is_the_three_arm_one():
    """A stamp binding the WRONG document looks verified. Guard the content."""
    doc = _prereg()
    assert set(doc["arms"]) >= {"A_incumbent", "B_causal_b1", "C_causal_b3"}
    assert doc["arms"]["B_causal_b1"]["config"] == {
        "calibration_forward_batch_size": 1,
        "calibration_batch_packing": "original_order_v1"}
    assert doc["arms"]["C_causal_b3"]["config"] == {
        "calibration_forward_batch_size": 3,
        "calibration_batch_packing": "length_sorted_v1"}
    assert doc["seeds"]["recovery"] == [217230555, 1151307191, 2045359208]
    assert doc["seeds"]["bootstrap"] == 654678655

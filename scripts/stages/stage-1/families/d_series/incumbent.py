"""Which checkpoint is the STANDING INCUMBENT, derived from C1's own outcome.

Every D-series round is a challenger experiment against the incumbent, so every
round needs one answer to "which checkpoint is B". Until this module existed the
answer was a hand-typed constant in each round's own files, and D1's constants
named the wrong arm:

    write_d1_design.py  INCUMBENT_STATE_ID = "fe9683e6a9c783bbc6fe276a78c851c6"
                        INCUMBENT_DIGEST   = "c313d1b4081b"

That is C1's INCUMBENT arm -- `attention.weight_proxy_v0` -- which is the arm
C1 measured and **beat**. C1's verdict was GO at a pooled delta of +0.013725
against a SESOI of 0.010, so the checkpoint that stands is C1's TREATMENT arm,
`attention.activation_importance_v1` at `53e30566...`. A D1 candidate compared
against `c313d1b4` would be credited with C1's already-banked effect on top of
its own, and that effect is larger than the SESOI the decision rule tests
against.

**So the identity is DERIVED from the verdict, never restated beside it.**
`standing_incumbent()` reads C1's decision record and C1's measured arm
identities and returns the arm the verdict selected. If a future round reopens
the question, the answer moves because the record moved -- which is the property
a typed constant cannot have, and the one that was missing here.

**One fact, one owner.** C2 and C3 are closed and keep their own frozen
constants (`phase_c2.baseline.B_ARTIFACT_DIGEST`, the C3 stage-E gate), which is
correct: a closed experiment's records describe what it ran. They are not
rewritten to import this; instead `tests/` asserts they AGREE with it, so a
disagreement is a failure rather than a discovery made by the next round.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]

#: THE C1 RUN THAT PRODUCED THE VERDICT. Attempt 18 is the one that completed:
#: `phase_c1.authorization_payload.ATTEMPT_18_PREREGISTRATION` is the document
#: the repository pins C1 to, and the digest gates in `phase_c1.session` are
#: that attempt's. Named here as a path rather than globbed, because "the latest
#: attempt directory" is not the same question as "the attempt whose
#: measurement stands".
C1_RUN_REL = "logs/stages/stage-1/phase_c1/runs/attempt18"
C1_DECISION_REL = f"{C1_RUN_REL}/evidence/c1_decision.json"
C1_ARM_IDENTITIES_REL = f"{C1_RUN_REL}/evidence/c1_arm_identities.json"

#: Which arm a verdict selects. C1's hypothesis was that the treatment beats the
#: incumbent, so GO promotes the treatment and anything else leaves the
#: incumbent standing. Written as a mapping rather than an `if` so an
#: unrecognised verdict raises instead of silently falling through to one arm.
VERDICT_SELECTS: dict[str, str] = {
    "GO": "treatment",
    "NO_GO": "incumbent",
    "INCONCLUSIVE": "incumbent",
}

#: The four identities a downstream consumer may bind an arm by. Enumerated
#: because a consumer that compared only `artifact_digest` would accept a
#: checkpoint whose weights or architecture differed under a colliding
#: construction identity -- the defect A3 found, where two differing artifacts
#: shared one state id.
IDENTITY_FIELDS: tuple[str, ...] = (
    "artifact_digest", "weights_digest", "single_shard_sha256",
    "arch_signature",
)


class IncumbentUndetermined(RuntimeError):
    """The standing incumbent cannot be derived from the records on hand.

    Raised rather than defaulted. An unknown incumbent is not a cosmetic gap: a
    challenger round that guessed one would measure its candidates against a
    control nobody selected.
    """


def _read(rel: str, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(repo_root) / rel
    if not path.is_file():
        raise IncumbentUndetermined(
            f"{rel} is missing, so which arm C1's verdict selected cannot be "
            "derived. This module will not fall back to a typed identity.")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise IncumbentUndetermined(f"{rel} is not parseable JSON: {exc}") from exc


def c1_verdict(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """C1's recorded verdict and the delta it rests on. Read, never recomputed."""
    doc = _read(C1_DECISION_REL, repo_root)
    verdict = str(doc.get("verdict") or "")
    if verdict not in VERDICT_SELECTS:
        raise IncumbentUndetermined(
            f"C1's decision records verdict {verdict!r}, which is not one of "
            f"{sorted(VERDICT_SELECTS)}. An unrecognised verdict selects no arm.")
    return {
        "verdict": verdict,
        "selects_arm": VERDICT_SELECTS[verdict],
        "delta": doc.get("delta"),
        "sesoi": doc.get("sesoi"),
        "lcb_one_sided": doc.get("lcb_one_sided"),
        "no_forced_winner": bool(doc.get("no_forced_winner")),
        "source": C1_DECISION_REL,
    }


def standing_incumbent(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The checkpoint every later round is challenging, with its four identities.

    DERIVED: the verdict names an arm, and the arm's identities come from the
    same run's MEASURED `c1_arm_identities.json` -- the digests C1 observed, not
    the digests anything expected. A pinned expectation and an observation agree
    until they do not, and the observation is what the bytes were.
    """
    outcome = c1_verdict(repo_root)
    arm_name = outcome["selects_arm"]
    identities = _read(C1_ARM_IDENTITIES_REL, repo_root)
    arm = identities.get(arm_name)
    if not isinstance(arm, dict):
        raise IncumbentUndetermined(
            f"C1's verdict selects the {arm_name!r} arm and "
            f"{C1_ARM_IDENTITIES_REL} carries no such block; it has "
            f"{sorted(k for k in identities if isinstance(identities.get(k), dict))}")
    missing = [f for f in IDENTITY_FIELDS if not arm.get(f)]
    if missing:
        raise IncumbentUndetermined(
            f"the {arm_name!r} arm records no {missing}; a control arm bound by "
            "fewer than its four identities can be satisfied by a checkpoint "
            "that is not it")
    return {
        "role": "standing_incumbent",
        "label": "B",
        "c1_arm": arm_name,
        "verdict": outcome["verdict"],
        "selected_because": (
            f"C1's verdict was {outcome['verdict']} at a pooled delta of "
            f"{outcome['delta']} against a SESOI of {outcome['sesoi']}, which "
            f"selects the {arm_name} arm"),
        "impl_id": arm.get("impl_id"),
        "profile_id": arm.get("profile_id"),
        "kind": arm.get("kind"),
        "num_parameters": arm.get("num_parameters"),
        "config_sha256": arm.get("config_sha256"),
        **{f: arm[f] for f in IDENTITY_FIELDS},
        "measured_by": C1_ARM_IDENTITIES_REL,
        "verdict_from": C1_DECISION_REL,
        "_not_a_state_id": (
            "C1's arms were built as fixed paths and its treatment arm has no "
            "search state id -- `c1_arm_identities.json` records `state_id: "
            "null` for it. A round that keys its control on a state id is "
            "keying on something the promoted arm does not have; the four "
            "content identities are what it does have."),
    }


def disagreements(declared: dict[str, Any],
                  repo_root: str | Path = REPO_ROOT) -> list[str]:
    """Which of `declared`'s identity fields disagree with the standing one.

    Compared by PREFIX on the declared value's own length, because records
    legitimately carry a 12-hex prefix -- and never truncated to the shorter of
    the two, which would let a 12-hex declaration pass against any digest
    sharing those twelve characters by making the comparison narrower than the
    evidence.
    """
    standing = standing_incumbent(repo_root)
    out: list[str] = []
    for field in IDENTITY_FIELDS:
        want = declared.get(field)
        if want in (None, ""):
            continue
        if not str(standing[field]).startswith(str(want)):
            out.append(
                f"{field}: declared {want} and the standing incumbent is "
                f"{str(standing[field])[:len(str(want))]}")
    return out


__all__ = ["C1_ARM_IDENTITIES_REL", "C1_DECISION_REL", "C1_RUN_REL",
           "IDENTITY_FIELDS", "IncumbentUndetermined", "VERDICT_SELECTS",
           "c1_verdict", "disagreements", "standing_incumbent"]

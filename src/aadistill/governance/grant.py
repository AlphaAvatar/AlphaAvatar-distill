"""What a maintainer states, and what an issuer derives. No experiment here.

Every phase of this project has re-implemented the same contract: a grant is a
human decision about *money and permission*, an authorization is a machine
derivation about *identity*, and the failure mode is a grant that asserts an
identity nobody computed. Every launchable phase in this repository carries its
own copy of the check.

The mechanism is experiment-agnostic and lives here:

* **stated fields** — what only a person can say (who approved, what it covers,
  the spend at approval, the cap). Absent or blank is a refusal.
* **derived fields** — what the issuer computes (commit, digests, plan hashes).
  A grant that asserts one of these is refused, because a document asserting an
  identity it did not compute is not evidence of one.
* **the cap arithmetic** — `spend + ceiling <= cap`, refused otherwise, with the
  numbers in the message. Raising a cap is a maintainer decision, not an
  issuer's.

What this module does NOT know: which fields a particular experiment uses, what
its ceiling is, where its preregistration lives, or what it is called. Those are
arguments. `GrantContract` is the whole vocabulary, and an experiment supplies
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class GrantRefused(Exception):
    """The maintainer-stated half does not support issuing anything."""


@dataclass(frozen=True)
class GrantContract:
    """The field split for one experiment's grants. Supplied, never assumed."""

    contract_id: str
    stated_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    #: Names of the two money fields, so the arithmetic can be checked without
    #: this module knowing what either is called in a given experiment.
    spend_field: str = "cumulative_spend_at_approval_usd"
    cap_field: str = "cumulative_cap_usd"

    def __post_init__(self) -> None:
        overlap = sorted(set(self.stated_fields) & set(self.derived_fields))
        if overlap:
            raise GrantRefused(
                f"{self.contract_id}: {overlap} are declared both stated and "
                "derived. A field is one or the other; if a maintainer may "
                "state it, the issuer must not compute it.")


def validate_grant(grant: Mapping[str, Any], contract: GrantContract, *,
                   ceiling_usd: float, expected_cap_usd: float | None = None
                   ) -> dict[str, Any]:
    """Refuse anything a grant cannot legitimately be. Returns it unchanged."""
    missing = [f for f in contract.stated_fields
               if f not in grant
               or (isinstance(grant[f], str) and not grant[f].strip())
               or grant[f] is None]
    if missing:
        raise GrantRefused(
            f"the grant is missing {missing}. A grant states who permitted "
            "what, at what cumulative spend, and what it does not cover.")

    asserted = [f for f in contract.derived_fields if f in grant]
    if asserted:
        raise GrantRefused(
            f"the grant asserts {asserted}, which the issuer derives. A grant "
            "that asserts an identity it did not compute is not evidence of "
            "anything.")

    cap = float(grant[contract.cap_field])
    if expected_cap_usd is not None and cap != expected_cap_usd:
        raise GrantRefused(
            f"the grant names cap ${cap:.4f}, not ${expected_cap_usd:.4f}")
    spent = float(grant[contract.spend_field])
    if spent + ceiling_usd > cap:
        raise GrantRefused(
            f"${spent:.4f} already spent plus a ${ceiling_usd:.4f} ceiling "
            f"exceeds the ${cap:.4f} cap. Raising a cap is a maintainer "
            "decision, not an issuer's.")
    return dict(grant)


def refuse_a_future_dated_grant(granted_utc: str, *,
                                now: "date | None" = None) -> str:
    """A grant cannot be issued from the future. Returns the date it states.

    A grant is written BY HAND and its date field says `utc`. C2-behavioural's
    attempt3 said `2026-09-21` while the session ran on `2026-09-20` UTC — a
    local-timezone date in a UTC field, which nothing detected because nothing
    read it. It distorted no money there, because run costs are attributed from
    closeouts through the run index, but a grant dated after the work it
    authorizes is not a record anyone can reason from.

    Experiment-agnostic on purpose: the check is one comparison against the
    clock, and every launchable phase needs it. `issue_c2_behavioural_
    authorization.py` keeps its own inline copy, deliberately unedited — that
    phase is closed, and rewriting a closed phase's issuer to remove a
    duplication is a worse trade than the duplication.

    An empty or absent date is NOT refused here: whether a grant must carry one
    is the phase's contract, not this function's.
    """
    from datetime import datetime, timezone

    stated = str(granted_utc or "")[:10]
    if not stated:
        return stated
    today = (now or datetime.now(timezone.utc).date()).isoformat()
    if stated > today:
        raise GrantRefused(
            f"the grant is dated {stated} and today is {today} UTC. A grant "
            "cannot be issued from the future: either the date was written "
            "from a local timezone -- which is what happened to C2's attempt3 "
            "-- or the clock is wrong. Fix the grant before issuing against it.")
    return stated


def budget_headroom(*, cumulative_usd: float, cap_usd: float,
                    ceiling_usd: float) -> dict[str, Any]:
    """Derive whether one more ceiling-sized run fits. Never restate it.

    This exists because the arithmetic was once written out in prose and the
    conclusion contradicted it: `15.9002 - 15.1475 = +0.7527` was described as
    falling *short* of the ceiling. A caller that asks for the verdict cannot
    make that mistake; a caller that retypes it can.
    """
    remaining = round(cap_usd - cumulative_usd, 4)
    reserve = round(cap_usd - cumulative_usd - ceiling_usd, 4)
    return {
        "cumulative_usd": cumulative_usd, "cap_usd": cap_usd,
        "ceiling_usd": ceiling_usd, "remaining_usd": remaining,
        "reserve_after_one_usd": reserve,
        "one_more_fits": cumulative_usd + ceiling_usd <= cap_usd,
        "worst_case_usd": round(cumulative_usd + ceiling_usd, 4),
        "note": "headroom is not permission",
    }

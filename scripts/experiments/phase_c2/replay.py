"""Governance for the replay-only artifact reconstruction.

One session, one purpose: rebuild the five checkpoints behind attempt 3's frozen
Top-5 and secure them off the pod. It decides nothing. The authorization type
exists so that fact is enforced rather than promised — a full-search
authorization cannot run this session, and this one cannot run a search, a
baseline rebuild, or any behavioural work.

The money is small and the scope is narrow, so the machinery is too. There is no
protocol document, no pricing record and no separate plan: the plan IS the
source binding, whose hash is the plan hash, because what this session will do
is entirely determined by the five pinned paths it reconstructs. Adding a second
document that restated them would create a way for the two to disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aadistill.infrastructure.manifest import sha256_json

from experiments.phase_c2.session import C2Authorization

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_replay_authorization/v1"

PLAN_ID = "autoinit.v1.phase_c2.replay"
SESSION_ID = "autoinit-phase-c2-replay"

#: The ONE stage sequence, and it ends.
AUTHORIZED_STAGES: tuple[str, ...] = ("bind_identities", "reconstruct")

#: What this session executes. The issuer is included for the same reason the
#: full search includes its own: the code that decides what an authorization
#: SAYS belongs to the executable identity that authorization binds.
ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_c2_replay_launch.py",
    "scripts/pod/autoinit_c2_replay_driver.py",
    "scripts/experiments/phase_c2/replay.py",
    "scripts/experiments/phase_c2/replay_specs.py",
    "scripts/autoinit/issue_c2_replay_authorization.py",
    "scripts/pod/collect_artifacts.py",
)

#: The maintainer's stated money for this session. Both figures are stated, not
#: derived: the review set them directly, and a derivation that produced a
#: different number would be overriding the approval rather than implementing
#: it. They live here — in this task's governance artifact — and not in
#: reusable core, per P12.1.
SOFT_STOP_USD = 4.00
HARD_CEILING_USD = 5.00

#: Reconstructing five pinned paths is GPU work of roughly two hours; the
#: session is bounded by money, not by a separately invented deadline. The
#: window is derived from the ceiling and the accepted rate at issue time.
TEARDOWN_RESERVE_USD = 0.25


def plan_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """The source binding IS the plan.

    What this session does is fully determined by the five digest-pinned paths
    and the commit they came from, so the plan hash is the hash of that binding.
    A second plan document would be a second place to edit.
    """
    from experiments.phase_c2.replay_specs import source_binding

    return sha256_json(source_binding(repo_root))


def window_minutes(rate_usd_per_hour: float) -> float:
    """How long the ceiling buys at the accepted rate, less the teardown reserve.

    Derived from price, never fixed independently of it: a deadline set without
    reference to the rate is a deadline that can outlive the budget.
    """
    if rate_usd_per_hour <= 0:
        raise ValueError("a rate must be positive to derive a window from it")
    usable = HARD_CEILING_USD - TEARDOWN_RESERVE_USD
    return (usable / rate_usd_per_hour) * 60.0


class ReplayAuthorization(C2Authorization):
    """Permits exactly one replay-only artifact reconstruction.

    Structurally a `C2Authorization` — same commit binding, same derived-harness
    rule, same hash-of-itself check — and a different type with a different
    schema, so it cannot stand in for the full search's, Search-1's or the
    baseline completion's, nor they for it.
    """

    @property
    def authorizes_c2_search1(self) -> bool:
        """NEVER. Search-1's beam is a consumed, frozen measurement."""
        return False

    @property
    def authorizes_c2_baseline_completion(self) -> bool:
        """NEVER. B is measured and frozen; this session compares nothing."""
        return False

    @property
    def authorizes_c2_full_search(self) -> bool:
        """NEVER, and this is the property that keeps the ruling's line.

        The search is COMPLETE. Its Top-5 is accepted and frozen, and the review
        was explicit that there must be no fourth Full Search attempt, no rerun
        of the beam, no re-ranking of the 14 leaves and no regeneration of a
        Top-5 from the journal. An artifact that could authorize a beam would be
        able to buy exactly the thing that was forbidden, at a tenth of the
        price and under a name that sounds like bookkeeping.
        """
        return False

    @property
    def authorizes_behavioural_selection(self) -> bool:
        """NEVER. Screening and confirmation are separately authorized."""
        return False

    @property
    def authorizes_c2_replay(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        #: From the properties, never literals.
        payload["authorizes_c2_search1"] = self.authorizes_c2_search1
        payload["authorizes_c2_baseline_completion"] = (
            self.authorizes_c2_baseline_completion)
        payload["authorizes_c2_full_search"] = self.authorizes_c2_full_search
        payload["authorizes_behavioural_selection"] = (
            self.authorizes_behavioural_selection)
        payload["authorizes_c2_replay"] = self.authorizes_c2_replay
        payload["scope"] = (
            "ONE replay-only reconstruction of the five checkpoints behind "
            "attempt 3's frozen Top-5, each path pinned at EVERY step to the "
            "artifact digest attempt 3 recorded, and each finished leaf secured "
            "off the pod. NOT a fourth Full Search attempt, NOT a beam, NOT a "
            "re-ranking, NOT a regeneration of a Top-5, NOT a "
            "selection-bearing evaluation, NOT a control comparison, and NOT "
            "behavioural work of any kind.")
        payload["forbids"] = [
            "any beam search or expansion",
            "ranking or re-ranking the 14 complete leaves",
            "producing or replacing a selection",
            "any selection-bearing evaluation",
            "injecting or comparing against the canonical control",
            "substituting a functionally similar checkpoint for a mismatch",
            "retrying a deterministic digest mismatch",
            "recovery training of any kind",
            "any behavioural screening or confirmation probe",
        ]
        payload["on_mismatch"] = (
            "STOP. A digest mismatch is a replay mismatch and a scientific "
            "finding: preserve the evidence, tear down, and refer it to review. "
            "It is not an ordinary engineering failure and must not be retried, "
            "because the path is deterministic and would diverge identically.")
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "ReplayAuthorization":
        record = json.loads(Path(path).read_text())
        if record.get("schema") != SCHEMA:
            raise ValueError(
                f"{path} carries schema {record.get('schema')!r}, not {SCHEMA!r}. "
                "The replay session refuses an authorization issued for another "
                "kind of session, whatever its money says.")
        return super().load(path)

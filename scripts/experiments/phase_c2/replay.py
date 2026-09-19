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

from aadistill.governance.closure import ClosureError, derive, digest_of
from aadistill.infrastructure.manifest import sha256_json

from experiments.phase_c2.session import C2Authorization


class ReplayGovernanceError(RuntimeError):
    """The replay's own governance cannot be derived from this tree."""

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

#: Where the closure follows imports. Same roots as the full search: the replay
#: runs the same operators from the same trees.
SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                 "scripts/autoinit")


def declared_inputs(repo_root: str | Path = REPO_ROOT) -> tuple[str, ...]:
    """Every non-python input whose BYTES decide what this session does.

    Attempt 3's selection and journal are here because they are not context —
    they ARE the plan. The specs, the pins and the identities every leaf is
    checked against are read out of those two files, so a change to either is a
    change to what the session executes, and the executable identity must move
    with them. Two paid pods in this programme died one per producer because a
    non-source input was shipped for one consumer and not the other.
    """
    from experiments.phase_c2.replay_specs import JOURNAL_REL, SELECTION_REL, TELEMETRY_REL

    return (
        SELECTION_REL, JOURNAL_REL, TELEMETRY_REL,
        "configs/experiments/phase_c2/replay_frozen_assets.json",
        "configs/autoinit/c2_replay_artifacts.json",
        "configs/autoinit/c2_replay_artifacts_failed.json",
        "scripts/pod/autoinit_preflight_setup.sh",
    )


def staged_assets(repo_root: str | Path = REPO_ROOT):
    """Dev-box artifacts the launcher stages, DERIVED from what this session reads.

    These cannot travel in the bundle: they live in the out-of-tree artifact
    store, which is gitignored, so the relay is how they reach a pod.

    Derived from the profiles THE FIVE PATHS NAME, not from a constant and not
    from the profile list the search declared. The first version staged only
    `calib.domain_balanced@v1` — under a comment claiming both were needed — and
    the launch-bound sweep caught it: two of the twenty steps name
    `calib.reasoning_heavy@v2`, so the session would have run path 1 and then
    died resolving a mixture that never arrived.

    A profile whose implementation needs no calibration stages nothing.
    `depth.positional_v0` carries the `calib.none@v1` sentinel, which is not a
    registered profile and has no items by construction; asking the registry for
    it would raise on a step that reads no items at all.

    No metric suite: this session evaluates nothing.
    """
    from aadistill.infrastructure.session import LocalAsset
    from aadistill.initialization.calibration.profiles import get_profile

    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c2.replay_specs import build_replay_leaves

    #: Registered HERE rather than assumed. `get_profile` raises on an empty
    #: registry, and a governance function that asked a registry somebody else
    #: was supposed to fill would answer "no assets to stage" in a fresh process
    #: — so the launcher would stage nothing and the pod would die resolving the
    #: mixtures. Idempotent.
    register_builtin_profiles()

    wanted: list[str] = []
    for leaf in build_replay_leaves(repo_root, device="cpu"):
        for step in leaf.spec.steps:
            if step.profile_id not in wanted:
                wanted.append(step.profile_id)

    roots: list[str] = []
    for qualified in wanted:
        try:
            profile = get_profile(qualified)
        except KeyError:
            #: The no-calibration sentinel. Not an asset and not an error.
            continue
        items = getattr(profile, "items_path", None)
        if not items:
            raise ReplayGovernanceError(
                f"calibration profile {qualified!r} is registered but resolves "
                "no items_path, so the launcher cannot stage what a step that "
                "names it will read")
        #: The DIRECTORY, because the manifest beside the items is part of the
        #: asset's identity.
        root = str(Path(items).parent)
        if root not in roots:
            roots.append(root)

    return tuple(LocalAsset(root, Path(root).name, str(Path(root).parent))
                 for root in roots)


def current_executable(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What a replay session would execute NOW, derived live from the tree."""
    try:
        return derive(Path(repo_root), "phase_c2_replay",
                      ENTRY_POINTS, declared_inputs(repo_root),
                      roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise ReplayGovernanceError(
            f"cannot derive the replay executable set: {exc}") from exc


def executable_digest(repo_root: str | Path = REPO_ROOT) -> str:
    live = current_executable(repo_root)
    return live["digest"] if isinstance(live, dict) else digest_of(live)

#: This session's money, and the two halves of it are kept apart on purpose.
#:
#: The figures moved twice and the second move was a CORRECTION, not a
#: negotiation. The review set $4.00/$5.00. A first derivation bounded each step
#: by the worst observation of its operator KIND and came to $5.27, over the
#: ceiling, so the session stopped before creating a resource and returned the
#: requirement. That bound was wrong: two DEPTH implementations appear in the
#: selected paths, `depth.causal_kl_greedy_v1` at up to 25.7 minutes and
#: `depth.positional_v0` at 0.6, and bounding by kind charged the cheap one at
#: the expensive one's rate — about 25 minutes of phantom cost on one leaf.
#: Bounding per IMPLEMENTATION gives 125.7 minutes of reconstruction and the
#: figures below, which fit the ORIGINAL $5.00 ceiling. The raise that was
#: approved in the meantime is not needed and is not taken.
#:
#: `GPU_HARD_USD` is what the budget planner is authorized against: it prices
#: GPU minutes and nothing else. `DISK_USD` is billed separately by the provider
#: at $0.10/GB/month and is NOT GPU money; a ceiling that folded the two together
#: would either refuse a fitting plan or hide the disk, and a GPU-only ceiling
#: has already missed $1.69 in this programme. `ALL_IN_USD` is their sum and is
#: the figure the ledger and any review read.
GPU_HARD_USD = 4.69
DISK_USD = 0.08
ALL_IN_USD = 4.77

#: Kept for callers that ask for one number. It is the ALL-IN figure, because a
#: single number that excluded the disk would understate the session.
HARD_CEILING_USD = ALL_IN_USD

#: The soft stop is DERIVED, not stated: `BudgetSpec.plan` computes it from the
#: same phases the ceiling comes from, so raising the ceiling moves it in step.
#: Pinning it at the old $4.00 under the new ceiling would have left admission
#: control refusing the fifth path for budget that exists — the precise thing
#: raising the ceiling was meant to buy.
TEARDOWN_RESERVE_USD = 0.25


def plan_payload(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The decision-bearing part of the source binding.

    What this session DOES is fully determined by the five digest-pinned paths,
    the selection they come from and the attempt-3 commit that produced them.
    Everything else in the binding is provenance.

    The distinction is not cosmetic. `source_binding` records the LIVE `head`
    alongside the source commit, so hashing the whole document made the plan
    hash move with every commit — including the commit that carries the
    authorization itself. An authorization is issued against the plan hash and
    checked against it at launch, so a plan hash containing `head` can never
    survive from issuance to launch: it was a binding that could not hold.
    """
    from experiments.phase_c2.replay_specs import source_binding

    binding = source_binding(repo_root)
    return {
        "reconstructs": binding["reconstructs"],
        "source_session_commit": binding["source_session_commit"],
        "selection_sha256": binding["selection_sha256"],
        "journal_file_sha256": binding["journal_file_sha256"],
        "full_journal_sha256": binding["full_journal_sha256"],
        "search": binding["search"],
        "policy": binding["policy"],
        "suite": binding["suite"],
        "profiles": binding["profiles"],
        "leaves": binding["leaves"],
    }


def plan_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """The plan IS the five pinned paths. A second document would be a second
    place to edit."""
    return sha256_json(plan_payload(repo_root))


def window_minutes(rate_usd_per_hour: float) -> float:
    """How long the ceiling buys at the accepted rate, less the teardown reserve.

    Derived from price, never fixed independently of it: a deadline set without
    reference to the rate is a deadline that can outlive the budget.
    """
    if rate_usd_per_hour <= 0:
        raise ValueError("a rate must be positive to derive a window from it")
    #: GPU money buys GPU minutes. The separately billed disk does not
    #: shorten the window, so it is not subtracted from it.
    usable = GPU_HARD_USD - TEARDOWN_RESERVE_USD
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
        """Its own schema check, and an EXPLICIT field mapping.

        Not `super().load()`: the parent pins the Search-1 schema, so
        delegating refuses this artifact by construction. And not a by-name
        filter either — `as_dict` serialises `plan_hash` as
        `phase_a_session_plan_hash` and `science_plan_hash` as
        `phase_a_science_plan_hash`, so a by-name round trip drops both and the
        constructor fails on a required argument. Every sibling loader maps by
        hand for the same reason. Reasoning from what this subclass NEEDS rather
        than from what the parent REQUIRES is how a session dies one step after
        a gate it passed.
        """
        from aadistill.governance.authorization import AuthorizationError

        from experiments.phase_c2.session import C2ResourceScope

        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "A search, a baseline rebuild and a behavioural session each "
                "price different work; none of them can authorize this replay, "
                "and this cannot authorize any of them.")
        for forbidden in ("authorizes_c2_full_search", "authorizes_c2_search1",
                          "authorizes_c2_baseline_completion",
                          "authorizes_behavioural_selection",
                          "allows_phase_a", "allows_recovery_training"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. This session reconstructs "
                    "artifacts behind a frozen selection and decides nothing; "
                    "an artifact claiming more is not one of these.")
        stages = tuple(raw.get("authorized_stages") or ())
        if stages != AUTHORIZED_STAGES:
            raise AuthorizationError(
                f"{path} authorizes stages {stages}, not {AUTHORIZED_STAGES}.")
        scope = raw.get("resource_scope")
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"], granted_by=raw["granted_by"],
            plan_id=raw["plan_id"],
            plan_hash=raw["phase_a_session_plan_hash"],
            science_plan_hash=raw["phase_a_science_plan_hash"],
            expected_usd=float(raw["expected_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw["scope_note"],
            authorized_session_commit=raw.get("authorized_session_commit"),
            harness_source_digest=raw.get("harness_source_digest"),
            harness_source_files=tuple(raw.get("harness_source_files") or ()),
            resource_scope=(C2ResourceScope.from_dict(scope)
                            if scope is not None else None),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))

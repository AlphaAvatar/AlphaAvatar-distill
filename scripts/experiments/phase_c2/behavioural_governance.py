"""Governance for the Phase-C2 behavioural selection session.

What this session is: twelve recovery probes in two rungs, one screening
ranking, one confirmation verdict under the frozen Phase-C rule. What it is
not, and cannot become, is written into the authorization type below as
properties rather than prose — a search, a re-ranking, a rebuild of a frozen
measurement, or any later cycle.

`behavioural.py` PROPOSES: it derives the candidates, the schedule, the storage
and the money. This module is the launch-side binding: the executable closure,
the plan hash, the ceiling and the one authorization type that can permit this
work. The split is the same one every C2 session uses, and it exists so the
document that asks for money is not the document that grants it.

**This module authorizes nothing by existing.** It defines the shape of an
authorization; issuing one is a separate, maintainer-gated act.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from aadistill.infrastructure.manifest import sha256_json
from aadistill.governance.closure import ClosureError, derive, digest_of

from experiments.phase_c2 import behavioural as BH
from experiments.phase_c2.session import C2Authorization

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_behavioural_authorization/v1"

PLAN_ID = BH.PLAN_ID
SESSION_ID = BH.SESSION_ID


class BehaviouralGovernanceError(RuntimeError):
    """The session cannot be bound to what it would execute."""


#: The stage sequence, and it ends at the verdict. `decide` is terminal: GO,
#: NO_GO and INCONCLUSIVE are all complete results, and none of them is a
#: reason to run anything else in this session.
AUTHORIZED_STAGES: tuple[str, ...] = (
    "prepare", "screen", "rank", "confirm", "decide")

#: What this session executes. The issuer belongs in it for the reason every
#: other C2 session includes its own: the code deciding what an authorization
#: SAYS is part of the executable identity that authorization binds.
ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_c2_behavioural_launch.py",
    "scripts/pod/autoinit_c2_behavioural_driver.py",
    "scripts/experiments/phase_c2/behavioural.py",
    "scripts/experiments/phase_c2/behavioural_governance.py",
    "scripts/experiments/phase_c2/behavioural_schedule.py",
    "scripts/experiments/phase_c2/behavioural_decision.py",
    "scripts/experiments/phase_c2/scoring.py",
    "scripts/autoinit/score_c2_screening.py",
    "scripts/autoinit/score_c1_confirmation.py",
    "scripts/pod/collect_artifacts.py",
)

SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                 "scripts/autoinit")


def declared_inputs(repo_root: str | Path = REPO_ROOT) -> tuple[str, ...]:
    """Every non-python input whose BYTES decide what this session does.

    The protocol is here because it is not context — it IS the plan: the seeds,
    the batteries, the schedule, the SESOI, the guardrails and the bootstrap
    seed are read out of it, so a change to it is a change to what executes.
    The two battery identity records are here for the same reason, and the
    pricing record because the ceiling is derived from it.

    Two paid pods in this programme died one per producer because a non-source
    input was shipped for one consumer and not the other.
    """
    return (
        BH.PROTOCOL,
        BH.PRICING,
        "logs/stages/stage-1/phase_c2/plans/c2_screening_battery.json",
        "logs/stages/stage-1/phase_c1/plans/execution_preregistration.json",
        "logs/stages/stage-1/phase_c1/plans/teacher_binding.json",
        "configs/stage3/e1/e1_r0860k_sa_pca.json",
        BH.STORAGE_PRICING,
        "scripts/pod/autoinit_preflight_setup.sh",
    )


#: WHY THE SIX ARMS ARE BUILT ON THE POD RATHER THAN SHIPPED TO IT.
#:
#: Each arm is a 1.19 GB checkpoint and there are six, 7.1 GB in total. Neither
#: transport this project has can carry that:
#:
#: * **The scp path cannot.** `local_assets` are copied AFTER the pod exists,
#:   so they bill, and the shared runner gives each asset a hardcoded 600-second
#:   timeout. One 1.19 GB asset needs 1.99 MB/s sustained against a dev-box
#:   uplink measured at 0.44-0.79 MB/s. Recovery-continuation attempt 2 died on
#:   exactly this, staging exactly this size of leaf; the write-up concluded the
#:   failure "was arithmetic rather than luck". Even with a longer timeout it is
#:   ~138 billed minutes of upload.
#: * **The hub relay cannot.** The fix continuation attempt 3 adopted — publish
#:   to a private transport repo at $0, let the pod pull at hub speed — needs
#:   quota this account does not have. Asked directly at $0 via the LFS batch
#:   endpoint on 2026-09-19 and again on 2026-09-20: one 1.19 GB object is
#:   ACCEPTED, 5.95 GB is REFUSED with "Private repository storage limit
#:   reached". Headroom bisected to 1.756 GiB against 93.08 GiB occupied.
#:   Freeing or buying storage is a maintainer decision, never an autonomous
#:   repair.
#:
#: What IS available is the mechanism the replay just proved: every candidate
#: reconstructs from the teacher, byte-for-byte, through a path pinned to the
#: artifact digest attempt 3 recorded at EVERY step. Five leaves, 125.68
#: bounded minutes, verified exact. B is built the same way for the same reason
#: — its bytes no longer exist — so this is one mechanism for all six arms
#: rather than two, and each arm is identity-gated at the moment it is built.
#:
#: This is NOT a search. No beam, no expansion, no ranking, no selection: a
#: fixed path per arm with a pinned expected digest per step, which stops on a
#: mismatch as a scientific finding.
CANDIDATE_TRANSPORT = "materialize_on_pod_from_pinned_paths"


def candidate_leaves(repo_root: str | Path = REPO_ROOT, *, device: str = "cuda"):
    """The five candidates as digest-pinned fixed paths, from the replay's owner.

    `build_replay_leaves` is not re-implemented here. It already resolves each
    path from attempt 3's frozen selection and journal, pins every step to the
    digest that attempt recorded, and carries the evidence-bound root config
    override the replay needed to reproduce step 0. Those five paths are the
    ones that reconstructed byte-for-byte; a second derivation would be a second
    thing to keep correct.
    """
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )

    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c2.replay_specs import build_replay_leaves
    from experiments.phase_c2.search_space import register_c2_operators

    register_builtin_profiles()
    register_builtin_operators()
    register_c2_operators()
    return build_replay_leaves(repo_root, device=device)


def arm_specs(repo_root: str | Path = REPO_ROOT, *, device: str = "cuda"):
    """The construction spec of every arm: five candidates, then B."""
    from experiments.phase_c2 import baseline as BL

    return tuple([leaf.spec for leaf in candidate_leaves(repo_root, device=device)]
                 + [BL.frozen_baseline_spec(device=device)])


def materialization_minutes(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Bounded minutes to build all six arms on the pod, before any probe runs.

    Both terms are BOUNDS from the same telemetry and the same rule — the worst
    observation of each operator IMPLEMENTATION anywhere in attempt 3's record.
    Bounding by operator KIND instead charged a 0.6-minute DEPTH step at a
    25.7-minute DEPTH step's rate and inflated a sibling session's estimate by
    about 25 minutes on one leaf.

    This is initialization, not probes. The protocol is twelve probes and stays
    twelve; the pod is simply alive longer because the arms have to exist.
    """
    leaves = candidate_leaves(repo_root)
    b = BH.b_preparation_minutes(repo_root)
    candidates = sum(float(leaf.bounded_minutes) for leaf in leaves)
    return {
        "candidates_minutes": round(candidates, 2),
        "n_candidates": len(leaves),
        "incumbent_b_minutes": float(b["bounded_minutes"]),
        "total_minutes": round(candidates + float(b["bounded_minutes"]), 2),
        "basis": b["_basis"],
        "transport": CANDIDATE_TRANSPORT,
        "_teacher_loaded_once": (
            "the per-leaf bound includes a teacher load each; six sequential "
            "materializations in one process load it once, so the real cost is "
            "below this bound. A bound is not an estimate and is not lowered "
            "by an optimization that has not been measured here."),
    }


def staged_assets(repo_root: str | Path = REPO_ROOT):
    """Dev-box artifacts the launcher stages, DERIVED from what this session reads.

    Only SMALL things travel this way — the calibration mixtures the six arms'
    construction steps name, totalling under 2 MiB. The arms themselves are
    built on the pod; see `CANDIDATE_TRANSPORT` for why the two transports that
    could have shipped them cannot.

    Derived from what the session actually names, not from a constant. A
    sibling session staged one mixture under a comment claiming both were
    needed, and only the launch-bound sweep caught that two of its steps named
    the other.
    """
    from aadistill.infrastructure.session import LocalAsset
    from aadistill.initialization.calibration.profiles import get_profile

    from experiments.calibration import register_builtin_profiles

    #: Registered HERE rather than assumed: `get_profile` raises on an empty
    #: registry, and a governance function that asked a registry somebody else
    #: was supposed to fill would answer "nothing to stage" in a fresh process.
    register_builtin_profiles()

    assets: list = []
    #: Every mixture named by ANY of the six arms' paths — the five candidates'
    #: and B's. Asking only B's would stage `calib.domain_balanced@v1` and miss
    #: `calib.reasoning_heavy@v2` for a candidate step that reads it.
    wanted: list[str] = []
    for spec in arm_specs(repo_root):
        for step in spec.steps:
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
            raise BehaviouralGovernanceError(
                f"calibration profile {qualified!r} is registered but resolves "
                "no items_path, so the launcher cannot stage what a step that "
                "names it will read")
        #: The DIRECTORY, because the manifest beside the items is part of the
        #: asset's identity.
        root = str(Path(items).parent)
        if root not in roots:
            roots.append(root)
    assets.extend(LocalAsset(r, Path(r).name, str(Path(r).parent))
                  for r in roots)
    return tuple(assets)


def current_executable(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What a behavioural session would execute NOW, derived live from the tree."""
    try:
        return derive(Path(repo_root), "phase_c2_behavioural",
                      ENTRY_POINTS, declared_inputs(repo_root),
                      roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise BehaviouralGovernanceError(
            f"cannot derive the behavioural executable set: {exc}") from exc


def executable_digest(repo_root: str | Path = REPO_ROOT) -> str:
    live = current_executable(repo_root)
    return live["digest"] if isinstance(live, dict) else digest_of(live)


# -- money -------------------------------------------------------------------
#: The provisioned container disk, in provider GB. Derived by
#: `behavioural.storage_requirement` from what this session actually holds — the
#: teacher, six staged initializations, B's materialization working set, one
#: probe's training working set and the probe outputs it retains — and NOT
#: inherited from the full search's 400 GB, which was sized for a beam holding
#: sixty compressed states.
PROVISION_GB = 120

#: The rate the proposal was priced at, recorded as a BASIS and not a promise.
#: RunPod's L40S secure price moves, and a launch re-quotes: a ceiling derived
#: from a stale quote is a ceiling that does not bound the bill. Reported by
#: `gpuTypes.securePrice` — never `communityPrice`, which is a different, lower
#: number the launcher does not pay and which has already been reported to a
#: maintainer as if it were the price.
QUOTED_RATE_USD_PER_HOUR = 1.09

#: The teardown reserve, held out of the window so a session that reaches its
#: deadline can still stop and delete its pod.
TEARDOWN_RESERVE_USD = 0.25


def ceiling(repo_root: str | Path = REPO_ROOT, *,
            gpu_rate_usd_per_hour: float = QUOTED_RATE_USD_PER_HOUR,
            provision_gb: int = PROVISION_GB) -> dict[str, Any]:
    """Expected and hard-ceiling cost at a live rate. Derived, never typed in.

    GPU money and separately billed container disk are kept apart all the way
    through and summed only at the end: a GPU-only ceiling has already missed
    $1.69 in this programme, and folding disk into the GPU figure would instead
    refuse plans that fit.

    The preparation term covers ALL SIX arms, not just B. An earlier derivation
    charged only B's 30.15 minutes because the five candidates were assumed to
    arrive as staged inputs; they cannot (see `CANDIDATE_TRANSPORT`), so the
    session builds them and pays for the minutes.
    """
    prep = materialization_minutes(repo_root)
    out = BH.money(repo_root, gpu_rate_usd_per_hour=gpu_rate_usd_per_hour,
                   provision_gb=provision_gb,
                   b_preparation_minutes=prep["total_minutes"])
    out["materialization"] = prep
    return out


def authorized_gpu_usd(repo_root: str | Path = REPO_ROOT) -> float:
    """The GPU dollars an authorization would carry: the ceiling at the QUOTE.

    A fixed figure. An authorization is a dollar amount granted once, not a
    formula re-evaluated at launch.
    """
    return float(ceiling(repo_root,
                         gpu_rate_usd_per_hour=QUOTED_RATE_USD_PER_HOUR
                         )["hard_ceiling"]["gpu_usd"])


def window_minutes(rate_usd_per_hour: float,
                   repo_root: str | Path = REPO_ROOT, *,
                   gpu_hard_usd: float | None = None) -> float:
    """How long a FIXED authorized amount buys at the LIVE rate.

    The dollars are fixed and the rate is live, which is the only arrangement
    in which a deadline cannot outlive its budget. Re-deriving the ceiling at
    the live rate instead — which an earlier version did — makes the ceiling
    scale with the price and the window nearly rate-independent, so a launch at
    double the quoted rate would run just as long and spend twice as much. The
    preflight caught that by asserting the obvious property: a dearer card must
    buy fewer minutes.

    The separately billed container disk does not shorten the GPU window, so it
    is not subtracted from it.
    """
    if rate_usd_per_hour <= 0:
        raise ValueError("a rate must be positive to derive a window from it")
    gpu = authorized_gpu_usd(repo_root) if gpu_hard_usd is None else gpu_hard_usd
    return ((gpu - TEARDOWN_RESERVE_USD) / rate_usd_per_hour) * 60.0


# -- the plan ----------------------------------------------------------------
def plan_payload(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The decision-bearing part of the binding. NO LIVE HEAD.

    What this session does is fully determined by the frozen protocol, the five
    candidate identities, B's required identity, the two batteries and the
    recovery recipe. Provenance that moves with every commit is deliberately
    excluded: an authorization is issued against this hash and checked against
    it at launch, so a hash containing the live head could never survive from
    issuance to launch. That has already happened once.
    """
    proto = BH.protocol(repo_root)["behavioural_selection"]
    binding = BH.b_binding(repo_root, device="cuda")
    return {
        "plan_id": PLAN_ID,
        "protocol_sha256": BH.protocol(repo_root)["protocol_sha256"],
        "schedule": proto["schedule"],
        "seeds": proto["seeds"],
        "batteries": {
            "screening": proto["batteries"]["screening"]["content_sha256"],
            "confirmation": proto["batteries"]["confirmation"]["content_sha256"],
        },
        "candidates": [
            {"state_id": c["state_id"],
             "artifact_digest": c["artifact_digest"],
             "weights_digest": c["weights_digest"],
             "arch_signature": c["arch_signature"],
             "num_parameters": c["num_parameters"],
             "rank_in_frozen_selection": c["rank_in_frozen_selection"]}
            for c in BH.candidate_manifest(repo_root)],
        "incumbent_b": {
            "spec_hash": binding["construction"]["spec_hash"],
            "required_identity": binding["required_identity"],
        },
        "recovery_recipe": proto["frozen_science"]["recovery_recipe"],
        "scoring_contract": proto["frozen_science"]["scoring_contract"],
        "authorized_stages": list(AUTHORIZED_STAGES),
    }


def plan_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """The plan IS the protocol plus the bound arms. One place to edit."""
    return sha256_json(plan_payload(repo_root))


# -- the authorization -------------------------------------------------------
class BehaviouralAuthorization(C2Authorization):
    """Permits exactly one Phase-C2 behavioural selection session.

    Structurally a `C2Authorization` — same commit binding, same derived-harness
    rule, same hash-of-itself check — and a different type with a different
    schema, so it cannot stand in for the search's, Search-1's, the baseline
    completion's or the replay's, nor they for it.

    **This is the one C2 session that trains**, which is why
    `allows_recovery_training` is overridden to True. The parent returns False
    and its `load` refuses any document claiming otherwise, so delegating to it
    would refuse this artifact by construction — the exact failure a sibling
    loader already hit by inheriting a contract instead of satisfying it.
    """

    @property
    def allows_recovery_training(self) -> bool:
        """TRUE, uniquely among C2 sessions. Twelve probes are the experiment."""
        return True

    @property
    def authorizes_c2_search1(self) -> bool:
        """NEVER. Search-1's beam is a consumed, frozen measurement."""
        return False

    @property
    def authorizes_c2_full_search(self) -> bool:
        """NEVER. The search is COMPLETE and its Top-5 is frozen and accepted.

        An artifact that could authorize a beam would be able to buy the one
        thing the review forbade, under a name that sounds like bookkeeping.
        """
        return False

    @property
    def authorizes_c2_baseline_completion(self) -> bool:
        """NEVER.

        This session MATERIALIZES B — it rebuilds the bytes behind an identity
        that is already frozen, and gates them on that identity. It does not
        re-measure B's state-evaluation, which is a completed measurement whose
        result stands. Building an artifact and re-deriving a number about it
        are different acts, and only the first is permitted here.
        """
        return False

    @property
    def authorizes_c2_replay(self) -> bool:
        """NEVER. The replay is closed and its five products are inputs here."""
        return False

    @property
    def authorizes_later_cycles(self) -> bool:
        """NEVER. C3 and C4 challenge whatever incumbent THIS session leaves.

        They are separate experiments against a comparator that does not exist
        until this one finishes, and an authorization written before its own
        result cannot price them.
        """
        return False

    @property
    def authorizes_behavioural_selection(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        #: From the properties, never literals.
        payload["allows_recovery_training"] = self.allows_recovery_training
        payload["authorizes_c2_search1"] = self.authorizes_c2_search1
        payload["authorizes_c2_full_search"] = self.authorizes_c2_full_search
        payload["authorizes_c2_baseline_completion"] = (
            self.authorizes_c2_baseline_completion)
        payload["authorizes_c2_replay"] = self.authorizes_c2_replay
        payload["authorizes_later_cycles"] = self.authorizes_later_cycles
        payload["authorizes_behavioural_selection"] = (
            self.authorizes_behavioural_selection)
        payload["scope"] = (
            "ONE Phase-C2 behavioural selection: six screening probes on one "
            "preregistered seed over five reconstructed candidates plus the "
            "frozen C1 treatment B, a mechanical ranking that advances exactly "
            "one candidate, six confirmation probes on three disjoint "
            "preregistered seeds, and one verdict under C1's frozen decision "
            "rule. Plus ONE materialization of B, gated on its frozen "
            "identity, because its bytes no longer exist. NOT a search, NOT a "
            "re-ranking of the frozen Top-5, NOT a re-measurement of B's state "
            "evaluation, and NOT any part of a later cycle.")
        payload["forbids"] = [
            "any beam search or expansion",
            "re-ranking the frozen Top-5 or regenerating a selection",
            "re-measuring B's completed state evaluation",
            "retraining a probe that already completed, for a different outcome",
            "ranking before all six screening results exist",
            "any verdict from a partial confirmation field",
            "a fourth seed, a tie-break rung or a forced winner",
            "any C3 or C4 work",
        ]
        payload["terminal_results"] = ["GO", "NO_GO", "INCONCLUSIVE"]
        payload["_no_forced_winner"] = (
            "NO_GO and INCONCLUSIVE are complete results. Neither is a reason "
            "to retry, extend or re-seed this experiment.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "BehaviouralAuthorization":
        """Its own schema check, and an EXPLICIT field mapping.

        Not `super().load()`: the parent pins the Search-1 schema AND refuses
        any document claiming `allows_recovery_training`, which this one must
        claim. And not a by-name filter either — `as_dict` serialises
        `plan_hash` as `phase_a_session_plan_hash` and `science_plan_hash` as
        `phase_a_science_plan_hash`, so a by-name round trip drops both and the
        constructor fails on a required argument.
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
                "A search, a replay, a baseline rebuild and a behavioural "
                "session each price different work; none of them can authorize "
                "this session, and this cannot authorize any of them.")
        for forbidden in ("authorizes_c2_full_search", "authorizes_c2_search1",
                          "authorizes_c2_baseline_completion",
                          "authorizes_c2_replay", "authorizes_later_cycles",
                          "allows_phase_a"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. This session runs twelve "
                    "probes against a frozen protocol and does nothing else; "
                    "an artifact claiming more is not one of these.")
        if not raw.get("allows_recovery_training"):
            raise AuthorizationError(
                f"{path} does not claim allows_recovery_training, but every "
                "one of this session's twelve probes is a recovery training "
                "run. An artifact that does not permit training cannot permit "
                "this.")
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

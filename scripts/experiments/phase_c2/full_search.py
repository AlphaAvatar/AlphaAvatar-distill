"""The governance layer for ONE Phase-C2 full-joint-re-search session.

Search-1 searched a RESTRICTED space: three operators fixed at the incumbent's
assignment and one varied. Its result is frozen, its B->C comparison is computed
and accepted, and the maintainer decision of 2026-09-11 withdrew the local
Search-2 refinement in favour of a FULL JOINT re-search — because an ATTENTION
change may have moved the best DEPTH, FFN and WIDTH assignments, and a
restricted search cannot see that.

This module is the permission and identity surface for that session. It permits
one beam over the derived joint space and one committed Top-5. It does not
permit a probe, a recovery run, a behavioural measurement, or anything that
could name an incumbent.

**It is a different TYPE, and the difference is the point.** A
`FullSearchAuthorization` declares its own schema and reports
`authorizes_c2_search1 = False`, so:

* the Search-1 launcher cannot load one, and this launcher cannot load a
  Search-1 or a baseline-completion authorization -- `C2Authorization.load`
  refuses a foreign schema in both directions;
* a full-search grant therefore cannot be spent on Search-1's consumed beam or
  on a baseline rebuild, by construction rather than by instruction.

**THE SESSION TERMINATES AT A COMMITTED TOP-5.** There is no authorized stage
after `commit_top_k`, `authorizes_behavioural_selection` is a property that
returns False, and the driver this binds has no import path into recovery or
scoring. The behavioural session is separately authorized and its candidate
identities do not exist until this search succeeds.

**The inputs are DERIVED, never listed.** Two paid subruns of the CUDA
validation died one per producer of non-source input, because a hand-maintained
shipping list can only name the producer its author thought of. `staged_assets`
and `declared_inputs` below ask the code instead -- `TELEMETRY_SOURCES` for what
the cost model reads and each profile's `items_path` for what the search
resolves -- and the launcher stages exactly what they return.

Every number comes from the full-search pricing record and every identity from
the full-search protocol; nothing here is retyped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import AuthorizationError
from aadistill.governance.closure import ClosureError, derive, digest_of
from aadistill.infrastructure.manifest import sha256_json
from aadistill.infrastructure.session import LocalAsset

from experiments.phase_c2 import full_search_space as FS
from experiments.phase_c2.session import C2Authorization, C2ResourceScope

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_full_search_authorization/v1"

PLAN_ID = "autoinit.v1.phase_c2.full_search"
SESSION_ID = "autoinit-phase-c2-full-search"

PROTOCOL = "logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json"
PRICING = "logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_pricing.json"

#: The ONE stage sequence this session may execute, and it ends.
AUTHORIZED_STAGES: tuple[str, ...] = ("bind_identities", "full_joint_search",
                                      "commit_top_k")

#: What this session executes.
ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_phase_c2_full_search_launch.py",
    "scripts/pod/autoinit_phase_c2_full_search_driver.py",
    #: The assembler and the issuer, because the code that decides what the
    #: AUTHORIZATION says belongs to the executable identity that authorization
    #: binds. An issuer outside the digest could change what a grant means
    #: without changing the digest the grant commits to.
    "scripts/experiments/phase_c2/full_search_authorization.py",
    "scripts/autoinit/issue_c2_full_search_authorization.py",
    "scripts/pod/collect_artifacts.py",
)

#: Non-python inputs whose BYTES decide what runs, and which travel in the
#: bundle because they are tracked. The telemetry pair is not documentation: the
#: cost model pools its per-expansion table over it, and a session reaching the
#: pod without it refuses at stage A rather than pricing from a partial history.
#: That refusal cost $0.0299 to observe, so the files are named in the identity
#: the grant binds.
SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                 "scripts/autoinit")


def _protocol_doc(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    doc = json.loads((Path(repo_root) / PROTOCOL).read_text())
    stated = doc.get("protocol_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "protocol_sha256"})
    if stated != recomputed:
        raise AuthorizationError(
            f"{PROTOCOL} does not match its own protocol_sha256; it has been "
            "edited since it was written")
    return doc


def protocol(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The protocol, verified against its own hash."""
    return _protocol_doc(repo_root)


def plan_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """The protocol IS the plan for this session."""
    return _protocol_doc(repo_root)["protocol_sha256"]


def pricing(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    doc = json.loads((Path(repo_root) / PRICING).read_text())
    stated = doc.get("pricing_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "pricing_sha256"})
    if stated != recomputed:
        raise AuthorizationError(
            f"{PRICING} does not match its own pricing_sha256; it has been "
            "edited since it was priced")
    return doc


def standing_beam_width() -> int:
    """6, from the frozen schedule rather than from a literal here.

    The maintainer has twice stated that beam width 6 is the standing scientific
    design and that shrinking it to fit a cap is not permitted. Reading it from
    `SCHEDULE_V1` is how a narrower width cannot become this session's width by
    someone editing a constant in a governance file: it would have to change the
    schedule, which is a change of experiment.
    """
    return int(FS.SCHEDULE_V1.width)


def _standing_row(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The priced row for the standing width. Selected, never assumed first."""
    width = standing_beam_width()
    for row in pricing(repo_root)["search"]["widths"]:
        if int(row["beam_width"]) == width:
            return row
    raise AuthorizationError(
        f"{PRICING} prices no beam width {width}, which is the standing design. "
        "The narrower widths it prices are scientific ALTERNATIVES and none of "
        "them is this session.")


def hard_ceiling_usd(repo_root: str | Path = REPO_ROOT) -> float:
    """The enforceable ceiling for the standing design, at the priced basis.

    Planning evidence with a date on it. `derive_ceiling_usd` below is what a
    launch uses, because the rate is re-quoted and the ceiling moves with it.
    """
    return float(_standing_row(repo_root)["hard_ceiling_usd"])


def expected_usd(repo_root: str | Path = REPO_ROOT) -> float:
    return float(_standing_row(repo_root)["expected_usd"])


def price_per_hour_basis(repo_root: str | Path = REPO_ROOT) -> float:
    """The rate the record was priced at. NOT the rate a launch pays."""
    return float(pricing(repo_root)["search"]["price_per_hour_basis"])


def derive_ceiling_usd(price_per_hour: float,
                       repo_root: str | Path = REPO_ROOT) -> float:
    """The ceiling at a LIVE rate, from the priced minutes.

    The minutes are the plan; the dollars are the minutes times whatever the
    provider charges today. So a rate change re-derives the ceiling
    mechanically and never shrinks the beam: the session buys the same work at
    a different price, and if that price no longer fits the envelope the answer
    is a maintainer decision, not a narrower search.

    Rounded UP at four decimals, because a ceiling that rounds DOWN
    under-authorizes the plan it is supposed to cover -- the C1 grant found
    this at $15.147403 against a recorded $15.1474.
    """
    import math

    minutes = float(_standing_row(repo_root)["hard_ceiling_minutes"])
    exact = minutes / 60.0 * float(price_per_hour)
    return math.ceil(exact * 10_000) / 10_000


def plan(price_per_hour: float, authorized_usd: float,
         repo_root: str | Path = REPO_ROOT):
    """The `BudgetPlan` for this session, through the code that priced it.

    Not a second decomposition of the same cost. `full_search_space.price` owns
    the phase list, the costly-operator-early expected path and the
    beam-composition reserve, and the pricing record was written by calling it —
    so a plan built here reproduces that record by construction rather than by
    two hand-maintained copies agreeing.

    RAISES `BudgetError` when the plan does not fit `authorized_usd`, which is
    how a search that has outgrown its authorization says so instead of being
    quietly shrunk.
    """
    space = FS.full_joint_space(repo_root)
    return FS.price(space, price_per_hour=price_per_hour,
                    authorized_usd=authorized_usd,
                    beam_width=standing_beam_width(), repo_root=repo_root)


def budget_spec(repo_root: str | Path = REPO_ROOT):
    """The `BudgetSpec` the session runner plans from.

    Built from the SAME ingredients `full_search_space.price` uses -- the shared
    session phases, the costly-operator-early expected trajectory and the
    beam-composition reserve -- because `BudgetSpec.plan` and that function both
    end in `plan_session` with the same arguments. So the runner's plan is the
    pricing record's plan, rather than two decompositions of one cost that have
    to be kept agreeing by hand.

    `arms=0`: a search trains nothing, so the step term multiplies out. The
    measured floor is passed only so the below-floor guard cannot fire on a
    figure that means nothing here.
    """
    from aadistill.infrastructure.budget import (
        MEASURED_STEP_SECONDS, Phase)
    from aadistill.infrastructure.session import BudgetSpec

    FS.register_c2_operators()
    space = FS.full_joint_space(repo_root)
    cost = FS.cost_model(repo_root)
    width = standing_beam_width()
    warmup = int(FS.SCHEDULE_V1.warmup_levels)

    expected = FS.trajectory(space, prefer_costly=True, beam_width=width,
                             repo_root=repo_root)
    limit = FS.bound(space, beam_width=width, repo_root=repo_root)
    phases = dict(FS.SESSION_PHASE_MINUTES)
    setup = phases["setup_and_asset_staging"]
    transfer = phases["bundle_transfer"]

    return BudgetSpec(
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: a search session trains nothing, so arms=0 and "
                     "the step term is zero. The measured floor is passed so "
                     "the below-floor guard cannot be satisfied by accident"),
        setup_minutes=setup,
        transfer_minutes=transfer,
        other_phases=(
            *(Phase(name, minutes)
              for name, minutes in FS.SESSION_PHASE_MINUTES
              if name not in ("setup_and_asset_staging", "bundle_transfer")),
            Phase("beam_search_costly_early", expected["minutes"]),
        ),
        contingency_fraction=0.10,
        #: The difference between the expected path and the structural worst
        #: case, named rather than folded in: an identified, bounded risk that is
        #: not on the expected path belongs after the contingency multiplier, so
        #: it protects the work instead of merely moving the watchdog's kill
        #: time. `warmup={warmup}` is carried by the schedule, not restated.
        soft_stop_reserves=(
            Phase("beam_composition_risk",
                  round(limit.max_minutes - expected["minutes"], 2)),),
        artifact_recovery_reserve_minutes=30.0,
    )


# --------------------------------------------------------------------------
# how much disk the beam holds at once
# --------------------------------------------------------------------------

def _state_gib(spec) -> float:
    """A materialized state's bf16 footprint, from its geometry.

    Parameter count for the tied-head Qwen3 shape, times two bytes. Derived
    rather than measured, because the point is to know the number BEFORE a
    volume is provisioned -- and it is checkable against the 596M target, whose
    real checkpoint the CUDA validation weighed at 1.11 GiB.
    """
    g = dict(spec.fields)
    h, layers = g["hidden_size"], g["num_hidden_layers"]
    inter, vocab = g["intermediate_size"], g["vocab_size"]
    heads, kv, head_dim = (g["num_attention_heads"], g["num_key_value_heads"],
                           g["head_dim"])
    attn = h * heads * head_dim + 2 * h * kv * head_dim + heads * head_dim * h
    params = vocab * h + layers * (attn + 3 * h * inter + 2 * h) + h
    return params * 2 / 1024 ** 3


def peak_resident_gib(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The most search state resident at once, DERIVED from the real space.

    Not inherited from Search-1. Search-1's launcher provisions for a 87.4 GiB
    peak, reached at level 1 with five retained level-0 states expanded into
    eighteen children. The joint space is a different shape: every operator kind
    may go first, so level 0 generates eleven children of which nine continue,
    and level 1 expands those nine into SIXTY. Carrying the 87.4 across would
    under-provision by a factor of nearly three, and a volume that fills at
    level 1 loses every state measured up to that point.

    The residency model is the search's own behaviour rather than an assumption:
    `BeamSearch` generates a whole level, ranks it, and only THEN calls
    `_release_weights` on what it pruned -- so every child of a level is on disk
    simultaneously. Completed leaves are never released at all, so they
    accumulate, and the walk adds them.

    The beam kept at each level is the most EXPENSIVE admissible one, not the
    average: a bound over compositions, for the same reason the cost ceiling is.
    """
    from experiments import search_cost_model as _M

    FS.register_c2_operators()
    space = FS.full_joint_space(repo_root)
    cost = FS.cost_model(repo_root)
    width = int(FS.SCHEDULE_V1.width)
    warmup = int(FS.SCHEDULE_V1.warmup_levels)
    #: A finished leaf is the target geometry, whatever path reached it.
    target_gib = _state_gib(space.target)

    beam: tuple[tuple[frozenset[str], int], ...] = ((frozenset(), 1),)
    level, accumulated, levels = 0, 0.0, []
    while beam:
        _minutes, partial, generated = _M.level_children(
            space, beam, cost, statistic="max")
        parents = sum(_state_gib(space.spec_of(cls)) * n for cls, n in beam)
        children = sum(_state_gib(space.spec_of(cls)) * n
                       for cls, n in partial.items())
        n_leaves = generated - sum(partial.values())
        resident = parents + children + n_leaves * target_gib + accumulated
        levels.append({
            "level": level,
            "parents": sum(n for _c, n in beam),
            "generated": generated,
            "completed_leaves": n_leaves,
            "parents_gib": round(parents, 2),
            "children_gib": round(children, 2),
            "accumulated_leaves_gib": round(accumulated, 2),
            "resident_gib": round(resident, 2),
        })
        accumulated += n_leaves * target_gib
        if not partial:
            break
        #: Largest classes first: a storage bound takes the worst admissible
        #: beam, never the mean.
        ordered = sorted(partial.items(),
                         key=lambda kv: -_state_gib(space.spec_of(kv[0])))
        if level < warmup:
            beam = tuple(ordered)
        else:
            kept, left = [], width
            for cls, n in ordered:
                take = min(n, left)
                if take:
                    kept.append((cls, take))
                    left -= take
            beam = tuple(kept)
        level += 1
        if level > 8:                        # structural backstop, never reached
            raise AuthorizationError(
                "the space walk exceeded eight levels; the required-kind set "
                "bounds a decomposition at four, so this is a defect")

    peak = max(row["resident_gib"] for row in levels)
    return {
        "peak_resident_gib": round(peak, 1),
        "peak_at_level": max(levels, key=lambda r: r["resident_gib"])["level"],
        "target_state_gib": round(target_gib, 2),
        "beam_width": width,
        "warmup_levels": warmup,
        "levels": levels,
        "_model": ("BeamSearch generates a whole level, ranks it, and releases "
                   "pruned weights only afterwards, so every child of a level "
                   "is resident at once. Completed leaves are never released."),
        "_excludes": ("the teacher (7.5 GiB in bf16), the checkout, the venv "
                      "and the staged assets. The launcher's provision covers "
                      "those on top of this figure."),
    }


# --------------------------------------------------------------------------
# what the session reads, derived from the code that reads it
# --------------------------------------------------------------------------

def tracked_non_source_inputs() -> tuple[str, ...]:
    """In-repository files the search reads that are not source or config.

    The telemetry the cost model pools over, and the level record one run
    carries. Tracked, so they travel in the bundle; named here, so their bytes
    are inside the identity a grant binds.

    Derived from `TELEMETRY_SOURCES` rather than listed: the CUDA validation's
    first subrun died at stage A for $0.0299 because a hand-written ship list
    named none of them, and a list is short again the next time a source is
    added.
    """
    out: list[str] = []
    for _name, telemetry, result in FS.TELEMETRY_SOURCES:
        out.append(telemetry)
        if result:
            out.append(result)
    return tuple(dict.fromkeys(out))


def staged_assets(repo_root: str | Path = REPO_ROOT) -> tuple[LocalAsset, ...]:
    """Dev-box artifacts the launcher stages, DERIVED from what resolves them.

    These cannot travel in the bundle: they live in the out-of-tree artifact
    store, which is gitignored, so the relay is how they reach a pod. Two of
    them are the real calibration mixtures the search resolves and one is the
    frozen metric suite every candidate is ranked on.

    Derived, because the validation's SECOND subrun died on exactly this half:
    a1's repair declared the cost model's inputs and a2 then died resolving the
    profiles' `items_path` files, which no one had thought to name. Asking each
    profile where its items are closes the class -- a reweighted mixture or a
    third profile is covered without anyone remembering.
    """
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.calibration import register_builtin_profiles

    #: Registered HERE rather than assumed. `get_profile` raises on an empty
    #: registry and this project has already lost a paid pod to exactly that:
    #: registration is explicit by design, and a governance function that asked
    #: a registry somebody else was supposed to fill would answer "no assets to
    #: stage" on a fresh process -- so the launcher would stage nothing and the
    #: pod would die resolving the mixtures, which is the a2 failure again with
    #: a longer fuse. Idempotent: `register_profile` returns the existing entry
    #: for an identical specification and raises on a conflicting one.
    register_builtin_profiles()

    roots: list[str] = []
    for qualified in FS.PROFILE_IDS:
        items = getattr(get_profile(qualified), "items_path", None)
        if not items:
            raise AuthorizationError(
                f"calibration profile {qualified!r} resolves no items_path, so "
                "the launcher cannot stage what the search will read")
        #: The DIRECTORY, because the manifest beside the items is part of the
        #: asset's identity and the frozen-asset check hashes the tree.
        roots.append(str(Path(items).parent))
    #: The metric suite is named by the frozen-asset expectation rather than by
    #: a profile, because nothing in the search resolves it -- the evaluator
    #: does. One owner either way: the expectation document.
    suite = json.loads(
        (Path(repo_root) / "configs/experiments/phase_c2/frozen_assets.json"
         ).read_text())["assets"]
    for name, entry in sorted(suite.items()):
        roots.append(entry["root"])

    seen, assets = set(), []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        assets.append(LocalAsset(root, Path(root).name, str(Path(root).parent)))
    return tuple(assets)


def declared_inputs(repo_root: str | Path = REPO_ROOT) -> tuple[str, ...]:
    """Every non-python input whose bytes belong in the executable identity."""
    return (
        PROTOCOL, PRICING,
        "configs/experiments/phase_c2/frozen_assets.json",
        "configs/experiments/phase_c2/full_search_authorization.json",
        "scripts/pod/autoinit_preflight_setup.sh",
        *tracked_non_source_inputs(),
    )


def current_executable(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What a full-search session would execute NOW, derived live from the tree."""
    try:
        return derive(Path(repo_root), "phase_c2_full_search",
                      ENTRY_POINTS, declared_inputs(repo_root),
                      roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise AuthorizationError(
            f"cannot derive the full-search executable set: {exc}") from exc


def executable_digest(repo_root: str | Path = REPO_ROOT) -> str:
    live = current_executable(repo_root)
    return live["digest"] if isinstance(live, dict) else digest_of(live)


# --------------------------------------------------------------------------
# the authorization type
# --------------------------------------------------------------------------

class FullSearchAuthorization(C2Authorization):
    """Permits exactly one full joint beam ending at a committed Top-5.

    Structurally a `C2Authorization` -- same commit binding, same derived-harness
    rule, same hash-of-itself check -- and a different type with a different
    schema, so it cannot stand in for Search-1's or baseline completion's, nor
    they for it.
    """

    @property
    def authorizes_c2_search1(self) -> bool:
        """NEVER. Search-1's beam is a consumed, frozen measurement.

        An artifact that could authorize another one would let this grant be
        spent re-running a completed experiment, which is both a budget
        expansion and a scientific change.
        """
        return False

    @property
    def authorizes_c2_baseline_completion(self) -> bool:
        """NEVER. B is measured and frozen; this session compares nothing."""
        return False

    @property
    def authorizes_c2_full_search(self) -> bool:
        return True

    @property
    def authorizes_behavioural_selection(self) -> bool:
        """NEVER, and this is the property the whole session rests on.

        The search ranks states on the frozen cheap `state_eval` metrics. It
        trains no probe, measures no `correct_overall` and cannot name an
        incumbent. Screening and confirmation are a separate session under a
        separate authorization, and their candidate identities DO NOT EXIST
        until this search commits a Top-5 -- so an artifact that claimed to
        authorize them would be authorizing work against candidates nobody has
        selected yet.
        """
        return False

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
        payload["scope"] = (
            "ONE beam over the DERIVED full joint space at the standing beam "
            "width, and ONE committed Top-5 candidate set. The session ends at "
            "commit_top_k. NOT a recovery probe, NOT a behavioural measurement, "
            "NOT a re-run of Search-1, NOT a remeasurement of B or any frozen C "
            "candidate, and nothing that names an incumbent.")
        payload["forbids"] = [
            "recovery training of any kind",
            "any behavioural screening or confirmation probe",
            "naming a C2 incumbent",
            "re-running the Search-1 beam",
            "remeasuring B or any frozen C candidate",
            "Search-2, which is withdrawn",
            "C3 or C4 work",
        ]
        payload["terminates_at"] = "commit_top_k"
        payload["_why_it_terminates_there"] = (
            "the Top-5 is a PREREGISTERED candidate set: committed before any "
            "probe result exists, so it cannot grow once one does. A session "
            "that continued into screening would be selecting and confirming on "
            "the same evidence.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "FullSearchAuthorization":
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
                "Search-1's authorization prices a restricted beam and baseline "
                "completion's prices one rebuild; neither can authorize a full "
                "joint re-search, and this cannot authorize either of them.")
        for forbidden in ("authorizes_c2_search1",
                          "authorizes_c2_baseline_completion",
                          "authorizes_behavioural_selection",
                          "allows_phase_a", "allows_recovery_training"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. This session runs one beam and "
                    "commits a Top-5; an artifact claiming more is not one of "
                    "these.")
        stages = tuple(raw.get("authorized_stages") or ())
        if stages != AUTHORIZED_STAGES:
            raise AuthorizationError(
                f"{path} authorizes stages {stages}, not {AUTHORIZED_STAGES}. "
                "The sequence is the scope: a stage after commit_top_k is the "
                "behavioural session, which is separately authorized.")
        #: Mapped EXPLICITLY, not by field name. `as_dict` serialises
        #: `plan_hash` as `phase_a_session_plan_hash` and `science_plan_hash` as
        #: `phase_a_science_plan_hash`, so a by-name filter drops both and the
        #: constructor then fails on a required argument. Every sibling loader
        #: maps them by hand for the same reason, and reasoning from what this
        #: subclass NEEDS rather than from what the parent REQUIRES is how a
        #: session comes to die one step after a gate it passed.
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


__all__ = [
    "AUTHORIZED_STAGES", "ENTRY_POINTS", "FullSearchAuthorization", "PLAN_ID",
    "PRICING", "PROTOCOL", "SCHEMA", "SESSION_ID", "SOURCE_ROOTS",
    "current_executable", "declared_inputs", "derive_ceiling_usd",
    "budget_spec", "executable_digest", "expected_usd",
    "hard_ceiling_usd",
    "peak_resident_gib", "plan",
    "plan_hash", "price_per_hour_basis", "pricing", "protocol",
    "staged_assets", "standing_beam_width", "tracked_non_source_inputs",
]

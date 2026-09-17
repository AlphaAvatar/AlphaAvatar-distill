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
import math
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
    """The most search state resident at once, following the REAL lifecycle.

    Not inherited from Search-1, and not the first model I wrote. Search-1's
    launcher provisions for a 87.4 GiB peak; the joint space is a different
    shape, because every operator kind may go first, so level 0 generates
    eleven children of which nine continue and level 1 expands those nine into
    sixty.

    **What is actually released, from `BeamSearch.run`.** `_release_weights` is
    called in exactly one place: on the PARTIAL children that the current
    level's ranking pruned. Nothing else is ever released. So the filesystem
    also holds, permanently:

    * the ROOT state;
    * every state that was SELECTED into a beam and then expanded -- it leaves
      the beam when its children are ranked, but no call drops its weights;
    * every DEAD END, a parent with no admissible expansion that is not a leaf;
    * every COMPLETED LEAF, which is the point of the search.

    An earlier version of this function modelled `current parents + current
    children + accumulated leaves` and therefore missed the expanded ancestors
    entirely. That was not a sound upper bound, and review caught it before a
    volume was provisioned from it. The walk below adds a state when it is
    generated and removes it only where the search removes it, so the model is
    the lifecycle rather than a summary of it.

    Two choices keep it an upper BOUND rather than an estimate:

    * the peak is taken after a level's children are all materialized and
      BEFORE its pruning, because that is the order `run` executes in;
    * the beam kept at each level is the most EXPENSIVE admissible one, which
      also minimises what gets released -- both push residency up.

    Classes are level-stratified (a state at level L has exactly L operators
    applied), so a newly generated child can never be confused with an ancestor
    of the same class when the pruned ones are subtracted.
    """
    from experiments import search_cost_model as _M

    FS.register_c2_operators()
    space = FS.full_joint_space(repo_root)
    cost = FS.cost_model(repo_root)
    width = int(FS.SCHEDULE_V1.width)
    warmup = int(FS.SCHEDULE_V1.warmup_levels)
    #: A finished leaf is the target geometry, whatever path reached it.
    target_gib = _state_gib(space.target)

    def size_of(multiset: dict) -> float:
        return sum(_state_gib(space.spec_of(cls)) * n
                   for cls, n in multiset.items() if n)

    #: The root is generated before the loop and never released.
    resident: dict[frozenset, int] = {frozenset(): 1}
    resident_leaf_gib = 0.0
    beam: tuple[tuple[frozenset[str], int], ...] = ((frozenset(), 1),)
    level, levels = 0, []
    while beam:
        _minutes, partial, generated = _M.level_children(
            space, beam, cost, statistic="max")
        n_partial = sum(partial.values())
        n_leaves = generated - n_partial

        #: Every child of this level is materialized before any is ranked.
        for cls, n in partial.items():
            resident[cls] = resident.get(cls, 0) + n
        resident_leaf_gib += n_leaves * target_gib
        ancestors_gib = size_of({c: n for c, n in resident.items()
                                 if c not in partial})
        children_gib = size_of(partial)
        peak_here = ancestors_gib + children_gib + resident_leaf_gib

        #: Now the level prunes -- and only the partial children it rejected.
        if level < warmup:
            kept = dict(partial)
        else:
            kept, left = {}, width
            for cls, n in sorted(partial.items(),
                                 key=lambda kv: -_state_gib(space.spec_of(kv[0]))):
                take = min(n, left)
                if take:
                    kept[cls] = take
                    left -= take
        released = {cls: n - kept.get(cls, 0) for cls, n in partial.items()}
        for cls, n in released.items():
            if n:
                resident[cls] -= n

        levels.append({
            "level": level,
            "parents": sum(n for _c, n in beam),
            "generated": generated,
            "completed_leaves": n_leaves,
            "partial_children": n_partial,
            "kept_in_beam": sum(kept.values()),
            "released_here": sum(released.values()),
            "retained_ancestors_gib": round(ancestors_gib, 2),
            "children_gib": round(children_gib, 2),
            "accumulated_leaves_gib": round(resident_leaf_gib, 2),
            "resident_gib": round(peak_here, 2),
            "resident_after_pruning_gib": round(
                size_of(resident) + resident_leaf_gib, 2),
        })

        beam = tuple(kept.items())
        level += 1
        if level > 8:                        # structural backstop, never reached
            raise AuthorizationError(
                "the space walk exceeded eight levels; the required-kind set "
                "bounds a decomposition at four, so this is a defect")

    peak_row = max(levels, key=lambda r: r["resident_gib"])
    return {
        "peak_resident_gib": round(peak_row["resident_gib"], 1),
        "peak_at_level": peak_row["level"],
        "final_resident_gib": round(levels[-1]["resident_after_pruning_gib"], 1),
        "target_state_gib": round(target_gib, 2),
        "root_state_gib": round(_state_gib(space.spec_of(frozenset())), 2),
        "beam_width": width,
        "warmup_levels": warmup,
        "levels": levels,
        "_model": (
            "`BeamSearch.run` releases weights ONLY for the partial children a "
            "level's ranking pruned. The root, every expanded ancestor, every "
            "dead end and every completed leaf stay on disk for the whole "
            "search -- so residency is cumulative, not per-level. The peak is "
            "taken after a level's children are materialized and before its "
            "pruning, which is the order `run` executes in."),
        "_bound_not_estimate": (
            "the beam kept at each level is the most expensive admissible one, "
            "which both maximises future parents and minimises what is "
            "released. A cheaper beam cannot exceed this."),
        "_excludes": (
            "the teacher, the checkout, the venv and the staged assets. The "
            "launcher's provision covers those on top of this figure."),
    }


def teacher_and_environment_gib(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What sits on the volume besides search states, derived where possible.

    The teacher dominates and is computable from its own geometry rather than
    remembered: the session loads it in bf16 and holds it for the whole search.
    The rest -- the checkout, the venv, the staged assets, the journal -- is a
    stated allowance, because they are not derivable from the space and are
    small beside the teacher.
    """
    from aadistill.initialization.specs.arch import ArchSpec

    FS.register_c2_operators()
    space = FS.full_joint_space(repo_root)
    teacher = _state_gib(space.spec_of(frozenset()))
    assets = 0.0
    for asset in staged_assets(repo_root):
        root = Path(repo_root) / asset.repo_path
        if root.is_dir():
            assets += sum(f.stat().st_size for f in root.rglob("*")
                          if f.is_file()) / 1024 ** 3
    #: The HF snapshot on disk as well as the resident copy: the pod downloads
    #: the teacher before it loads it, and both occupy the volume.
    checkout_and_venv = 12.0
    return {
        "teacher_resident_gib": round(teacher, 2),
        "teacher_snapshot_gib": round(teacher, 2),
        "_why_twice": ("the pod downloads the teacher to the HF cache and then "
                       "materializes it; both occupy the volume at once"),
        "staged_assets_gib": round(assets, 3),
        "checkout_and_venv_gib": checkout_and_venv,
        "_checkout_and_venv_is_an_allowance": (
            "stated, not derived: a git checkout plus a built venv with torch "
            "and CUDA wheels is not computable from the search space, and it "
            "is small beside the teacher"),
        "total_gib": round(teacher * 2 + assets + checkout_and_venv, 2),
    }


STORAGE_PRICING = "configs/infrastructure/provider_storage_pricing.json"

#: Headroom over the derived requirement. Disk is cents and a volume that fills
#: at the residency peak loses every state the search has measured, so the
#: provision is deliberately generous -- but it is a MULTIPLIER on a derived
#: figure, not a remembered number.
PROVISION_HEADROOM = 1.25


def storage_pricing(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The provider's separately billed storage basis. NOT a live quote.

    `gpuTypes.securePrice` is re-quoted before every authorization; there is no
    equivalent query for Container Disk -- `oneMonthPrice` is null and the pod
    type exposes no disk field -- so this is a dated stated basis and the
    document says so in a field a machine can read. GPU securePrice being live
    is not evidence that separately priced storage is free.
    """
    doc = json.loads((Path(repo_root) / STORAGE_PRICING).read_text())
    if doc["container_disk"].get("provider_api_exposes_this") is not False:
        raise AuthorizationError(
            f"{STORAGE_PRICING} claims the provider API exposes container-disk "
            "pricing. It does not, and a stated basis that presents itself as "
            "a quote is worse than no basis.")
    return doc


def provision_gb(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The `--container-disk-in-gb` value, DERIVED end to end.

    Residency peak plus the teacher and environment, converted from GiB into
    the provider's decimal GB -- the SMALLER unit, so the conversion rounds the
    request UP rather than silently delivering 7% less capacity than the
    requirement -- then multiplied by the headroom and rounded up to a round
    number.

    An earlier version carried a 25.0 GiB environment allowance and a 350 GiB
    provision as constants, and got the unit direction backwards in its own
    note. Both are computed here now.
    """
    states = peak_resident_gib(repo_root)
    env = teacher_and_environment_gib(repo_root)
    pricing = storage_pricing(repo_root)
    gb_per_gib = float(pricing["gb_versus_gib"]["gb_per_gib"])

    required_gib = states["peak_resident_gib"] + env["total_gib"]
    required_gb = required_gib * gb_per_gib
    with_headroom = required_gb * PROVISION_HEADROOM
    #: Rounded up to the next 50, because a provider flag takes an integer and
    #: a round number is what a reader can check against the gate's message.
    provision = int(math.ceil(with_headroom / 50.0) * 50)
    return {
        "peak_resident_states_gib": states["peak_resident_gib"],
        "peak_at_level": states["peak_at_level"],
        "teacher_and_environment_gib": env["total_gib"],
        "required_gib": round(required_gib, 1),
        "required_gb": round(required_gb, 1),
        "headroom_multiple": PROVISION_HEADROOM,
        "provision_gb": provision,
        "_unit": ("the provider's flag is GB and the requirement is GiB; the "
                  "conversion assumes DECIMAL GB, the smaller unit, so the "
                  "request cannot deliver less capacity than the requirement"),
    }


def storage_cost_usd(hours: float, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What the container disk costs over `hours`, at the derived provision.

    Billed on the integer handed to `--container-disk-in-gb`, for the pod's
    whole lifetime rather than only while the GPU is busy -- this session tears
    down on every path, so the billed window is its wall clock.
    """
    pricing = storage_pricing(repo_root)
    gb = provision_gb(repo_root)["provision_gb"]
    per_gb_month = float(pricing["container_disk"]["usd_per_gb_month"])
    hours_per_month = float(pricing["proration"]["hours_per_month"])
    per_hour = per_gb_month / hours_per_month * gb
    return {
        "provisioned_gb": gb,
        "usd_per_gb_month": per_gb_month,
        "hours_per_month": hours_per_month,
        "usd_per_hour": round(per_hour, 6),
        "hours": round(hours, 4),
        "usd": math.ceil(per_hour * hours * 10_000) / 10_000,
        "_rounded_up": "a storage bound rounds up for the same reason a ceiling does",
    }


def effective_rate_usd_per_hour(price_per_hour: float,
                                repo_root: str | Path = REPO_ROOT) -> float:
    """GPU rate plus the container disk's hourly share.

    Both accrue against the same wall clock, so the session's real hourly cost
    is their sum -- and a watchdog given only the GPU rate would stop the pod
    at a money figure the provider had already exceeded.
    """
    per_hour = storage_cost_usd(1.0, repo_root)["usd_per_hour"]
    return round(float(price_per_hour) + per_hour, 6)


def total_ceiling_usd(price_per_hour: float,
                      repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The ceiling that covers EVERY separately billed provider resource.

    The GPU ceiling alone was $33.1827 while the launcher provisioned hundreds
    of GB of container disk that the GPU securePrice says nothing about. The
    minutes are unchanged -- they are the work bound -- and the money is those
    minutes at the effective rate.
    """
    minutes = float(_standing_row(repo_root)["hard_ceiling_minutes"])
    hours = minutes / 60.0
    gpu = derive_ceiling_usd(price_per_hour, repo_root)
    disk = storage_cost_usd(hours, repo_root)
    effective = effective_rate_usd_per_hour(price_per_hour, repo_root)
    total = math.ceil(hours * effective * 10_000) / 10_000
    return {
        "hard_ceiling_minutes": minutes,
        "hard_ceiling_hours": round(hours, 4),
        "gpu_rate_usd_per_hour": round(float(price_per_hour), 4),
        "gpu_usd": gpu,
        "_gpu_rate_is_live": ("re-quoted from gpuTypes.securePrice before "
                              "authorization and bounded by the grant"),
        "container_disk": disk,
        "_disk_basis_is_stated_not_quoted": (
            "the provider exposes no storage price; see "
            f"{STORAGE_PRICING}. This is the distinction that made the earlier "
            "ceiling wrong: a live GPU quote is not evidence about separately "
            "priced storage."),
        "network_volume_usd": 0.0,
        "_no_volume": "the launcher passes --volume-in-gb 0",
        "effective_rate_usd_per_hour": effective,
        "total_hard_ceiling_usd": total,
        "_total_covers": ("GPU runtime + container disk + any other separately "
                          "billed provider resource this session uses, which "
                          "is none"),
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
    #: And the measured-optimization record the cost table is REFRESHED by.
    #: Named unconditionally, not "if it exists": `cost_model()` falls back to
    #: the pooled pre-optimization figures when it is missing, so a pod without
    #: this file would build a DIFFERENT, more expensive cost model than the one
    #: the grant priced -- and nothing would say so, because the fallback is by
    #: design and silent. Declaring it turns that into a bundle failure.
    out.append(FS.MEASURED_OPTIMIZATION)
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
    "PROVISION_HEADROOM", "STORAGE_PRICING",
    "effective_rate_usd_per_hour", "provision_gb", "storage_cost_usd",
    "storage_pricing", "teacher_and_environment_gib", "total_ceiling_usd",
    "plan_hash", "price_per_hour_basis", "pricing", "protocol",
    "staged_assets", "standing_beam_width", "tracked_non_source_inputs",
]

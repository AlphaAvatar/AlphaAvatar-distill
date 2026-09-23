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
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from aadistill.infrastructure.manifest import sha256_json
from aadistill.governance.closure import ClosureError, derive, digest_of

from experiments.phase_c2 import behavioural as BH
from experiments.phase_c2.session import C2Authorization

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_behavioural_authorization/v1"

PLAN_ID = BH.PLAN_ID
SESSION_ID = BH.SESSION_ID

#: THE SCIENTIFIC CAMPAIGN, which is not the invocation.
#:
#: One campaign is one 12-probe behavioural experiment: six screening probes,
#: one mechanical ranking, six confirmation probes, one verdict. A run id names
#: a launcher invocation and the provider resource it draws; it changes whenever
#: a resource is replaced, and that is exactly what evidence and logs need it
#: for.
#:
#: They used to be the same string — the launcher passed `--campaign
#: <run_id>` — which made the preregistered continuation policy unreachable by
#: construction. R1 permits reuse only inside one campaign, so a replacement
#: resource became a DIFFERENT campaign and had to refuse every probe the
#: previous resource had trained and verified off-pod. The rule meant to prevent
#: cross-experiment pooling was instead preventing continuation of one
#: experiment.
#:
#: Stable, and bound by the authorization: an artifact that does not name this
#: campaign cannot permit work under it, so the campaign cannot be renamed into
#: a fresh permission.
CAMPAIGN_ID = "c2-behavioural-12probe-v1"


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

    **And attempt 3's evidence, because the prepare stage READS it.** The
    behavioural prepare stage calls `replay_specs.build_replay_leaves`, whose
    output is decided by the frozen selection, the compact state journal, the
    telemetry and the architecture-spec lineage: they determine each path's
    ancestry, its calibration profiles, every pinned intermediate digest, the
    historical root-config override that made step 0 reproduce, and the
    materialization admission bounds. Four files that decide which six
    checkpoints get built, and none of them was named here. "Recorded closure ==
    live closure" means nothing until the closure actually names everything the
    executable reads. They are named through `replay_specs`' own constants
    rather than retyped, so a relocation moves one string.

    Two paid pods in this programme died one per producer because a non-source
    input was shipped for one consumer and not the other.
    """
    from experiments.phase_c2 import replay_specs as RS

    return (
        BH.PROTOCOL,
        BH.PRICING,
        "logs/stages/stage-1/phase_c2/plans/c2_screening_battery.json",
        "logs/stages/stage-1/phase_c1/plans/execution_preregistration.json",
        "logs/stages/stage-1/phase_c1/plans/teacher_binding.json",
        "configs/stage3/e1/e1_r0860k_sa_pca.json",
        BH.STORAGE_PRICING,
        "scripts/pod/autoinit_preflight_setup.sh",
        #: The runtime scientific inputs of the prepare stage.
        RS.SELECTION_REL,
        RS.JOURNAL_REL,
        RS.TELEMETRY_REL,
        RS.ARCH_SPEC_LINEAGE_REL,
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
#:   endpoint on 2026-09-19, 2026-09-20, and again on 2026-09-22 after a
#:   maintainer-authorized cleanup recovered 1.110 GiB: it ACCEPTS 1 GiB and
#:   REFUSES 2 GiB with "Private repository storage limit reached", against
#:   91.029 GiB occupied of a recorded ~93.13 limit. A trained probe is 2.221
#:   GiB, so the relay cannot carry even one. Everything larger on the account
#:   is preserved fail-closed — e1_scaling_20260801 (42.4 GiB) and stage3
#:   (26.8 GiB) have NO local copy, so Hugging Face is their only one, and
#:   e1_r0860k_* is the provenance of this campaign's own recovery recipe.
#:   Buying storage remains a maintainer decision.
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

#: THE CUMULATIVE CAMPAIGN CEILING, as approved by the maintainer on
#: 2026-09-23 UTC. Everything else in this module is DERIVED — from the frozen
#: record, from the live rate, from the schedule. This one number is not: it is
#: a maintainer decision about how much of the project's remaining budget this
#: campaign may consume in total, across every attempt and every provider
#: resource it takes.
#:
#: History, in three decisions:
#:
#: 1. $33.2099, which at the time coincided exactly with a single fresh
#:    session's all-in ceiling — so the two were the same number and the code
#:    could not tell them apart;
#: 2. $35.7600, after attempt3 spent $2.5425 and stopped at ENOSPC without a
#:    verdict, so one more full-length attempt still fitted;
#: 3. $42.0000, to absorb TRANSPORT VARIABILITY. A durable backend's
#:    throughput depends on time of day, routing and provider load, so no
#:    single measurement of it is a stable constant and repeatedly
#:    benchmarking one wastes engineering time to manufacture false
#:    precision. The campaign carries an explicit conservative reserve
#:    instead, and the execution proceeds.
#:
#: The session ceiling did NOT move at any of the three and must not. This
#: figure funds OPERATIONAL VARIANCE: it buys no runtime, no disk, no probes,
#: no seeds, no arms and no scientific scope, all of which come from the frozen
#: record through the session ceiling alone. A grant states this number and
#: issuance refuses a grant that disagrees with it, because a hand-written
#: grant is exactly where a ceiling typo would enter unreviewed.
CAMPAIGN_ALL_IN_CEILING_USD = 42.00


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
                   materialization_minutes=prep["total_minutes"])
    out["materialization"] = prep
    return out


#: The five amounts an authorization must carry DISTINCTLY. Named here because
#: the launcher reads them by name and a missing one is a refusal, never a
#: default: a window derived from a constant rate while the authorization was
#: issued at a different one is a window the money does not fund.
AUTHORIZATION_AMOUNT_FIELDS: tuple[str, ...] = (
    "rate_usd_per_hour", "hard_runtime_minutes", "gpu_hard_usd",
    "disk_hard_usd", "all_in_hard_usd")

#: THE CAMPAIGN CEILING, and it is a DIFFERENT QUANTITY from the session's.
#:
#: `all_in_hard_usd` is what ONE session may cost and it is reconciled against
#: `gpu_hard_usd + disk_hard_usd`; that identity is what stops a GPU-only cap
#: from silently permitting the disk amount of extra spend. The campaign
#: ceiling is cumulative across every attempt and resource, and after a failed
#: attempt it necessarily exceeds one session's — so it CANNOT be the same
#: field. Raising `all_in_hard_usd` to carry a campaign figure would break the
#: session identity and, worse, would be read by everything that derives the
#: session window as permission for a longer, dearer session.
#:
#: It buys exactly one thing: room for a fresh attempt beside spend that has
#: already settled. It does not extend runtime, GPU dollars, disk, probes,
#: seeds or scope — those all come from the session amounts and the frozen
#: decomposition, and nothing here reaches them.
CAMPAIGN_AMOUNT_FIELD = "campaign_all_in_hard_usd"


#: Where each attempt of this campaign records what it actually cost. The
#: closeout's `money.all_in_usd` is the AUTHORITATIVE figure -- it is what the
#: project ledger reads -- and it is all-in: `cost.actual_usd` carries GPU
#: alone, because the runner reads GPU alone from the provider and container
#: disk is billed separately.
#:
#: DERIVED from `rel_run_dir`, the one function that knows the log convention,
#: rather than written out. Relocating `logs/` has silently repointed declared
#: executables in this repository before, and a hardcoded path here would
#: enumerate nothing and report a campaign's paid predecessors as free.
EXPERIMENT_ID = "phase_c2_behavioural"
STAGE_ID = "1"

#: THE SUFFIX A DRY RUN'S OUTPUT LOCATIONS TAKE, and the reason it lives here
#: rather than in the launcher: two separate enumerations of this campaign's
#: run directories have to agree about what is NOT a run.
#:
#: A `--dry-run` writes its evidence to its own run directory so it cannot
#: consume the chain it exists to de-risk. That directory then sits under
#: `runs/` beside the real attempts -- and `campaign_attempts` enumerates
#: exactly that, so `attempt7_dryrun` was read as a prior RESOURCE of the
#: campaign whose spend could not be established, and the gate refused the
#: launch it had just rehearsed. A dry run creates no provider resource by
#: construction; that is what the flag means.
DRY_RUN_SUFFIX = "_dryrun"


def is_dry_run_id(run_id: str) -> bool:
    """Is this a dry run's output id rather than a real run attempt?

    Asked by every enumeration of this campaign's run directories. A dry run
    contacts no provider, so it is not a resource, not a predecessor, and not
    something whose billing state could be unknown.
    """
    return str(run_id).endswith(DRY_RUN_SUFFIX)


def campaign_runs_rel() -> str:
    """This campaign's run directories, repo-relative."""
    from experiments.run_layout import rel_run_dir

    return str(Path(rel_run_dir(EXPERIMENT_ID, "_", STAGE_ID)).parent)


def settled_campaign_all_in(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What this campaign has ALREADY spent, all-in, from its own closeouts.

    THE CANONICAL OWNER of this quantity. It had two derivations -- one in the
    continuation pricer and one inside the launcher's gate -- and they answer
    slightly different questions: the gate reconciles each predecessor RESOURCE
    under R9 and refuses an unknown one, while this reads the settled figure
    each attempt published. Both are needed, but only one of them may be the
    number a ceiling is derived from, or a session could be capped against one
    figure and charged against the other.

    This is the one a ceiling is derived from, because it is independent of any
    authorization: the gate's disk term is a function of the authorization's
    own disk rate, and capping an authorization with a figure derived FROM that
    authorization is circular.

    An attempt whose cost is genuinely unknown RAISES. It is never defaulted to
    `$0`: a predecessor that billed an unknown amount is a stop condition, and
    treating it as free is how a cumulative ceiling stops being cumulative.
    """
    root = Path(repo_root) / campaign_runs_rel()
    per_attempt: dict[str, float] = {}
    total = 0.0
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        #: A DRY RUN IS NOT A RUN. It contacts no provider and spends nothing,
        #: so its directory must not enter a settled-spend total -- see
        #: `is_dry_run_id`.
        if is_dry_run_id(d.name):
            continue
        outcome = d / "closeout/outcome.json"
        if not outcome.is_file():
            continue
        doc = json.loads(outcome.read_text())
        value = (doc.get("money") or {}).get("all_in_usd")
        if value is None:
            #: An affirmative "no resource was created" is a STATED $0, which
            #: is different from a missing figure.
            value = 0.0 if doc.get("provider_resource_created") is False else None
        if value is None:
            raise BehaviouralGovernanceError(
                f"{d.name}: its closeout states no all-in cost and does not "
                "affirm that no provider resource was created, so settled "
                "campaign spend is UNKNOWN and cannot be defaulted to $0")
        per_attempt[d.name] = float(value)
        total += float(value)
    return {"per_attempt": per_attempt, "total_usd": round(total, 4)}


def session_ceiling_under_campaign(hard: Mapping[str, Any], *,
                                   rate_usd_per_hour: float,
                                   campaign_all_in_hard_usd: float,
                                   settled_campaign_all_in_usd: float,
                                   ) -> dict[str, Any]:
    """One session's amounts, shortened to what the campaign can still fund.

    **The gap this closes.** The continuation gate charges the campaign ceiling
    for the work a continuation PLANS. Nothing charged it for what that session
    could cost if the work went wrong: `all_in_hard_usd` came from the frozen
    full-session decomposition and was the same figure for the first attempt
    and the fifth. While settled spend was small that was safe by accident —
    attempt5 ran with `$2.5425` settled against a `$35.76` campaign ceiling and
    a `$33.2099` session ceiling, which fits by one cent. It stops being safe
    the moment settled spend is large: `$22.2466` settled plus a `$33.2099`
    session is `$55.4565` against a `$42.0000` campaign ceiling, so a session
    that hung would bill straight through the campaign's bound while every
    gate it passed said yes.

    So a session may be authorized for no more than the campaign has left. The
    runtime is shortened with the money, because a window the money does not
    fund is the defect this whole separation exists to prevent — and it is
    shortened by FLOORING: a limit rounds down, or the amount it derives sits
    a fraction above what was actually available.

    This does not shrink the SCIENCE. It shrinks the runaway bound. Whether the
    remaining work still fits inside the shortened window is a different
    question, asked by `campaign_continuation_gate`, which knows what the work
    is; this function does not and must not guess.
    """
    full_all_in = float(hard["all_in_usd"])
    full_minutes = float(hard["minutes"])
    settled = max(0.0, float(settled_campaign_all_in_usd))
    available = float(campaign_all_in_hard_usd) - settled
    if available <= 0:
        raise BehaviouralGovernanceError(
            f"the campaign has spent ${settled:.4f} of its "
            f"${float(campaign_all_in_hard_usd):.4f} ceiling and has nothing "
            "left to fund a session with. Remaining money is not permission, "
            "and absent money is a refusal.")
    if available >= full_all_in:
        return {"minutes": full_minutes, "gpu_usd": float(hard["gpu_usd"]),
                "disk_usd": float(hard["disk_usd"]), "all_in_usd": full_all_in,
                "shortened": False, "campaign_available_usd": round(available, 4)}

    #: Per-minute cost of the WHOLE session: GPU at the live rate plus the
    #: separately billed container disk, derived from the full window's own
    #: figures so the disk rate is the one the authorization carries.
    disk_per_minute = float(hard["disk_usd"]) / full_minutes
    per_minute = rate_usd_per_hour / 60.0 + disk_per_minute
    minutes = math.floor(available / per_minute * 100) / 100.0
    gpu = minutes / 60.0 * rate_usd_per_hour
    disk = disk_per_minute * minutes
    return {"minutes": minutes, "gpu_usd": round(gpu, 4),
            "disk_usd": round(disk, 4), "all_in_usd": round(gpu + disk, 4),
            "shortened": True, "campaign_available_usd": round(available, 4),
            "_full_session_would_have_been": {
                "minutes": full_minutes, "all_in_usd": full_all_in},
            "_why": (
                "the campaign's remaining money, not the frozen full-session "
                "derivation, because a session may never be authorized to "
                "spend past its campaign's cumulative ceiling")}


def authorization_terms(repo_root: str | Path = REPO_ROOT, *,
                        rate_usd_per_hour: float,
                        campaign_all_in_hard_usd: float,
                        settled_campaign_all_in_usd: float = 0.0,
                        provision_gb: int = PROVISION_GB) -> dict[str, Any]:
    """The amounts an authorization carries, derived MECHANICALLY at a live rate.

    Issuance re-quotes `gpuTypes.securePrice` and calls this with that number.
    Nothing here reads `QUOTED_RATE_USD_PER_HOUR`: that constant is the basis
    the PROPOSAL was priced at, recorded so a reader can see what moved, and an
    authorization derived from it after the price changed would carry a dollar
    window that does not match its own runtime.

    The amounts are kept apart all the way through. GPU money and separately
    billed container disk are different bills — folding them together has
    already cost `$1.69` unledgered once — and the authorized runtime is a
    quantity that follows from neither: a card offered below the authorized rate
    buys more minutes than the experiment is authorized to use, and the session
    must not take them.
    """
    if rate_usd_per_hour <= 0:
        raise BehaviouralGovernanceError(
            "an authorization cannot be derived at a non-positive rate")
    c = ceiling(repo_root, gpu_rate_usd_per_hour=rate_usd_per_hour,
                provision_gb=provision_gb)
    full, expected = c["hard_ceiling"], c["expected"]
    #: SHORTENED TO WHAT THE CAMPAIGN HAS LEFT. A session ceiling larger than
    #: the campaign's remaining money is a runaway bound that every gate would
    #: pass; see `session_ceiling_under_campaign`.
    hard = session_ceiling_under_campaign(
        full, rate_usd_per_hour=rate_usd_per_hour,
        campaign_all_in_hard_usd=campaign_all_in_hard_usd,
        settled_campaign_all_in_usd=settled_campaign_all_in_usd)
    #: The campaign ceiling is a MAINTAINER NUMBER, not a derivation: it is
    #: whatever decision funds this campaign, and it is passed in rather than
    #: computed so nothing here can quietly grant more of it.
    campaign = float(campaign_all_in_hard_usd)
    if campaign < float(full["all_in_usd"]) - DOLLAR_QUANTUM_USD:
        raise BehaviouralGovernanceError(
            f"the campaign ceiling ${campaign} is below one session's all-in "
            f"${hard['all_in_usd']}. A campaign that cannot fund a single "
            "attempt of its own experiment is not a ceiling, it is a refusal "
            "written as a number.")
    return {
        CAMPAIGN_AMOUNT_FIELD: campaign,
        "_campaign_ceiling_is_cumulative": (
            "bounds this campaign across every run attempt and provider "
            "resource, INCLUDING spend that has already settled. It is not a "
            "fresh allocation per attempt, and it is deliberately a different "
            "field from all_in_hard_usd, which bounds ONE session and is "
            "reconciled against gpu_hard_usd + disk_hard_usd."),
        "_campaign_ceiling_buys_no_science": (
            "it does not extend hard_runtime_minutes, gpu_hard_usd, "
            "disk_hard_usd, the provisioned disk, the probe count, the seeds "
            "or the scope. The session window is derived from gpu_hard_usd and "
            "hard_runtime_minutes, and the work from the frozen decomposition; "
            "neither reads this field."),
        "rate_usd_per_hour": float(rate_usd_per_hour),
        "hard_runtime_minutes": float(hard["minutes"]),
        "gpu_hard_usd": float(hard["gpu_usd"]),
        "disk_hard_usd": float(hard["disk_usd"]),
        "all_in_hard_usd": float(hard["all_in_usd"]),
        #: CLAMPED TO THE CEILING. `expected` is the FULL protocol's expected
        #: cost; when the ceiling has been shortened to what the campaign has
        #: left, a full session's expectation can exceed this session's hard
        #: bound, and "expected $23.88, hard $19.75" is not a forecast, it is
        #: two quantities that cannot both be about the same session. The
        #: authoritative expectation for a continuation is the launcher's own
        #: plan, which prices the REMAINING work; this field is recorded, not
        #: spent against.
        "expected_all_in_usd": min(float(expected["all_in_usd"]),
                                   float(hard["all_in_usd"])),
        "_expected_is_clamped_to_the_ceiling": (
            "the frozen decomposition's expected cost is for the WHOLE "
            "protocol. A session shortened to the campaign's remaining money "
            "may have a smaller ceiling than that, and an expected figure "
            "above a hard bound describes no session. What this session "
            "expects to spend comes from the launcher's remaining-work plan."),
        "provisioned_disk_gb": int(provision_gb),
        "campaign_id": CAMPAIGN_ID,
        "_rate_is_live_at_issuance": (
            "re-quoted from gpuTypes.securePrice — never communityPrice, which "
            "is a lower number the launcher does not pay. If the quote moves "
            "before issuance the proposal is regenerated and a materially "
            "changed dollar authorization goes back to the maintainer."),
        "_the_runtime_binds_too": (
            "hard_runtime_minutes is the authorized length of the experiment. A "
            "card offered below the authorized rate buys more minutes than "
            "this; the session may not use them."),
        #: `_all_in_is_the_campaign_ceiling` was emitted here and said
        #: "all_in_hard_usd bounds the campaign cumulatively across every
        #: resource and run attempt". It was true while the two ceilings were
        #: one number and became false the moment they separated, and the same
        #: document already states the correct contract in
        #: `_campaign_ceiling_is_cumulative` above. It is DELETED rather than
        #: reworded into a compatibility note: a governance artifact that
        #: contradicts itself is worse evidence than one that says less, and
        #: the reader cannot tell which of two statements the code obeys.
    }


def window_minutes(rate_usd_per_hour: float, *,
                   gpu_hard_usd: float,
                   hard_runtime_minutes: float) -> float:
    """The session's deadline: the shorter of what the money buys and the runtime.

    Both bounds are AUTHORIZATION-BOUND and both are required. There is no
    default, because the only available default was `QUOTED_RATE_USD_PER_HOUR`'s
    derived ceiling — so a valid authorization re-derived at a different live
    quote would still have inherited the old `$1.09/h` dollar window and could
    have truncated the experiment.

    Two distinct ways a deadline can be wrong, and both are closed here:

    * **too long for the money.** The dollars are fixed and the rate is live,
      which is the only arrangement in which a deadline cannot outlive its
      budget. Re-deriving the ceiling at the live rate instead — which an
      earlier version did — makes the ceiling scale with the price and the
      window nearly rate-independent, so a launch at double the quoted rate
      would run just as long and spend twice as much.
    * **too long for the experiment.** A card at half the authorized rate funds
      twice the minutes. Those minutes are not authorized: the grant permits a
      12-probe experiment of a stated length, not as much work as the money
      happens to reach.

    The separately billed container disk does not shorten the GPU window, so it
    is not subtracted from it.
    """
    if rate_usd_per_hour <= 0:
        raise ValueError("a rate must be positive to derive a window from it")
    if gpu_hard_usd <= TEARDOWN_RESERVE_USD:
        raise ValueError(
            f"a GPU authorization of ${gpu_hard_usd} cannot fund the "
            f"${TEARDOWN_RESERVE_USD} teardown reserve, let alone an experiment")
    if hard_runtime_minutes <= 0:
        raise ValueError("an authorized runtime must be positive")
    funded = ((gpu_hard_usd - TEARDOWN_RESERVE_USD) / rate_usd_per_hour) * 60.0
    return min(funded, float(hard_runtime_minutes))


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
        #: The SCIENTIFIC campaign, in the plan an authorization binds. A
        #: continuation of this experiment is the same campaign under a new run
        #: attempt; a different campaign is a different experiment and needs its
        #: own plan, its own money and its own decision.
        "campaign_id": CAMPAIGN_ID,
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
#: The dollar quantum the proposal's ceiling rounds to. `money()` derives every
#: amount with `math.ceil(... * 10_000) / 10_000` — a ceiling rounds UP, so a
#: stated amount can exceed its own arithmetic by up to this much and no more.
#: It is the tolerance every reconciliation of a stated amount against its
#: derivation uses, and it is named once so a reader can check which quantity it
#: bounds rather than meeting an anonymous epsilon.
DOLLAR_QUANTUM_USD = 1e-4


def _load_amounts(path: str | Path, raw: dict[str, Any]) -> dict[str, float]:
    """Read the five amounts and RECONCILE them. A stated amount is a claim.

    Three things are checked, and each has a way of being wrong that a reader
    of the document could not see:

    * every field is PRESENT and positive — a missing money field must refuse,
      never default;
    * `gpu_hard_usd` is what `hard_runtime_minutes` costs at
      `rate_usd_per_hour` — otherwise the dollars and the deadline describe two
      different experiments, and the launcher would take whichever is longer;
    * `all_in_hard_usd` is `gpu_hard_usd + disk_hard_usd` — a GPU-only ceiling
      has already missed `$1.69` of separately billed container disk in this
      programme, and an all-in figure that is not the sum hides it again.

    `hard_cap_usd` must equal `gpu_hard_usd`: the runner compares the plan's
    hard threshold, which is GPU dollars, against `hard_cap_usd`. An all-in cap
    there would silently permit the disk amount of extra GPU overrun.
    """
    from aadistill.governance.authorization import AuthorizationError

    out: dict[str, float] = {}
    for field in AUTHORIZATION_AMOUNT_FIELDS:
        value = raw.get(field)
        if value is None:
            raise AuthorizationError(
                f"{path} states no {field}. The launcher derives its deadline "
                "and its spend checks from the authorization's own amounts; "
                "with one missing there is nothing to derive them from, and the "
                "only available default was a constant quoted rate that may no "
                "longer be the price.")
        try:
            out[field] = float(value)
        except (TypeError, ValueError) as exc:
            raise AuthorizationError(
                f"{path} states {field}={value!r}, which is not a number") from exc
        if out[field] <= 0:
            raise AuthorizationError(
                f"{path} states {field}={value!r}; every authorized amount must "
                "be positive")

    derived_gpu = out["hard_runtime_minutes"] / 60.0 * out["rate_usd_per_hour"]
    if abs(out["gpu_hard_usd"] - derived_gpu) > DOLLAR_QUANTUM_USD:
        raise AuthorizationError(
            f"{path} authorizes {out['hard_runtime_minutes']} min at "
            f"${out['rate_usd_per_hour']}/h, which is ${derived_gpu:.4f}, but "
            f"states gpu_hard_usd=${out['gpu_hard_usd']}. The dollars and the "
            "deadline describe different experiments; a launcher given both "
            "would run to whichever is longer.")
    all_in = out["gpu_hard_usd"] + out["disk_hard_usd"]
    if abs(out["all_in_hard_usd"] - all_in) > DOLLAR_QUANTUM_USD:
        raise AuthorizationError(
            f"{path} states all_in_hard_usd=${out['all_in_hard_usd']} against "
            f"${out['gpu_hard_usd']} of GPU plus ${out['disk_hard_usd']} of "
            "separately billed container disk. An all-in figure that is not the "
            "sum is how $1.69 of disk went unledgered once already.")
    #: THE CAMPAIGN CEILING, loaded and checked as its OWN quantity. The
    #: session identity below is unaffected by it and must stay that way: an
    #: authorization whose all_in_hard_usd had been raised to carry a campaign
    #: figure would fail the gpu+disk identity, and if it somehow passed, every
    #: consumer that derives a session window would read the larger number as
    #: permission for a longer, dearer session.
    campaign = raw.get(CAMPAIGN_AMOUNT_FIELD)
    if campaign is None:
        raise AuthorizationError(
            f"{path} states no {CAMPAIGN_AMOUNT_FIELD}. The campaign ceiling is "
            "cumulative across every attempt and resource; without it "
            "campaign_continuation_gate has nothing to bound prior spend "
            "against, and a session ceiling is not a substitute — after one "
            "failed attempt they are different numbers.")
    try:
        out[CAMPAIGN_AMOUNT_FIELD] = float(campaign)
    except (TypeError, ValueError) as exc:
        raise AuthorizationError(
            f"{path} states {CAMPAIGN_AMOUNT_FIELD}={campaign!r}, which is not "
            "a number") from exc
    if out[CAMPAIGN_AMOUNT_FIELD] < out["all_in_hard_usd"] - DOLLAR_QUANTUM_USD:
        raise AuthorizationError(
            f"{path} states a campaign ceiling of "
            f"${out[CAMPAIGN_AMOUNT_FIELD]} below its own session all-in of "
            f"${out['all_in_hard_usd']}. A campaign that cannot fund a single "
            "attempt of its own experiment is not a ceiling.")

    cap = float(raw.get("hard_cap_usd", 0.0))
    if abs(cap - out["gpu_hard_usd"]) > DOLLAR_QUANTUM_USD:
        raise AuthorizationError(
            f"{path} states hard_cap_usd=${cap} and gpu_hard_usd="
            f"${out['gpu_hard_usd']}. The runner checks the planned hard "
            "threshold — which is GPU dollars — against hard_cap_usd, so these "
            "must be the same number. An all-in cap there permits the disk "
            "amount of extra GPU spend.")
    return out


@dataclass(frozen=True)
class BehaviouralAuthorization(C2Authorization):
    """Permits exactly one Phase-C2 behavioural selection CAMPAIGN.

    Structurally a `C2Authorization` — same commit binding, same derived-harness
    rule, same hash-of-itself check — and a different type with a different
    schema, so it cannot stand in for the search's, Search-1's, the baseline
    completion's or the replay's, nor they for it.

    **This is the one C2 session that trains**, which is why
    `allows_recovery_training` is overridden to True. The parent returns False
    and its `load` refuses any document claiming otherwise, so delegating to it
    would refuse this artifact by construction — the exact failure a sibling
    loader already hit by inheriting a contract instead of satisfying it.

    It carries six fields the parent does not, and each is load-bearing rather
    than descriptive:

    * `campaign_id` — WHICH 12-probe experiment this permits. A run attempt is
      an invocation; the campaign is the science, and reuse of a completed probe
      is scoped to it.
    * `rate_usd_per_hour`, `hard_runtime_minutes`, `gpu_hard_usd`,
      `disk_hard_usd`, `all_in_hard_usd` — the amounts, kept DISTINCT. The
      launcher derives its deadline from these and not from
      `QUOTED_RATE_USD_PER_HOUR`, so an authorization re-derived at a different
      live quote cannot inherit the old dollar window.
    """

    #: `None` fails closed: `load` refuses a document that omits any of them,
    #: and nothing constructs one of these without going through `load` or an
    #: issuer. A permissive default on a money field is the one shape this type
    #: must not have.
    campaign_id: str | None = None
    #: CUMULATIVE across the campaign, and deliberately not `all_in_hard_usd`.
    #: See CAMPAIGN_AMOUNT_FIELD for why these cannot be one field.
    campaign_all_in_hard_usd: float | None = None
    rate_usd_per_hour: float | None = None
    hard_runtime_minutes: float | None = None
    gpu_hard_usd: float | None = None
    disk_hard_usd: float | None = None
    all_in_hard_usd: float | None = None

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
        #: The campaign and the five amounts, from the fields. A document that
        #: carried a scope the object does not have would be worse than none.
        payload["campaign_id"] = self.campaign_id
        for field in AUTHORIZATION_AMOUNT_FIELDS:
            payload[field] = getattr(self, field)
        payload[CAMPAIGN_AMOUNT_FIELD] = self.campaign_all_in_hard_usd
        payload["_session_versus_campaign"] = (
            "all_in_hard_usd bounds ONE session and equals gpu_hard_usd + "
            "disk_hard_usd. campaign_all_in_hard_usd bounds the CAMPAIGN "
            "cumulatively, including spend that has already settled, and is "
            "the only figure campaign_continuation_gate compares against. "
            "Neither the session window nor the scientific work reads it.")
        payload["_amounts_are_distinct"] = (
            "GPU money, separately billed container disk and the authorized "
            "runtime are three different quantities. The launcher derives its "
            "deadline from gpu_hard_usd and hard_runtime_minutes at the LIVE "
            "rate, never from a constant.")
        #: `_the_ceiling_is_cumulative_over_the_campaign` was emitted here and
        #: attributed the cumulative bound to `all_in_hard_usd`. That was the
        #: contract while the two ceilings were one number; it is now false,
        #: and it contradicted `_session_versus_campaign` in the same payload.
        #: DELETED rather than reworded: the fact it was reaching for — that a
        #: replacement resource receives no fresh allocation — is stated once,
        #: above, against the field that actually bounds it.
        payload["scope"] = (
            "ONE Phase-C2 behavioural selection campaign: SIX deterministic "
            "fixed-path arm materializations — the five frozen Top-5 C "
            "candidates and the frozen C1 treatment B — each built from the "
            "verified teacher along a path pinned at every step and gated on "
            "its exact recorded identity before any probe starts, followed by "
            "exactly twelve behavioural probes: six screening probes on one "
            "preregistered seed over those six arms, a mechanical ranking that "
            "advances exactly one candidate, six confirmation probes on three "
            "disjoint preregistered seeds, and one verdict under C1's frozen "
            "decision rule. All six arms are materialized because neither "
            "available transport can carry six 1.19 GB checkpoints to a pod; "
            "B's bytes no longer exist at all. NOT a search, NOT a beam or "
            "expansion, NOT a re-ranking of the frozen Top-5, NOT a "
            "re-measurement of B's completed state evaluation, NOT a fourth "
            "seed, and NOT any part of C3 or C4.")
        payload["authorized_materializations"] = 6
        payload["authorized_probes"] = 12
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
        if raw.get("campaign_id") != CAMPAIGN_ID:
            raise AuthorizationError(
                f"{path} names campaign {raw.get('campaign_id')!r}, not "
                f"{CAMPAIGN_ID!r}. One authorization funds one 12-probe "
                "behavioural campaign; an artifact naming another campaign "
                "permits a different experiment, and renaming the campaign is "
                "not a way to obtain a second one.")
        amounts = _load_amounts(path, raw)
        scope = raw.get("resource_scope")
        return cls(
            campaign_id=raw["campaign_id"],
            **amounts,
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

"""Governance for the Phase-C2 behavioural selection. PROPOSES; authorizes nothing.

Twelve probes, exactly: six screening and six confirmation. It is the only
evidence that may name a C2 incumbent, and only its confirmation rung may do so.

Everything scientific here is READ from the frozen full-search protocol rather
than restated — the schedule, the seeds, the batteries, the anchor, the ranking
rule and the tie-break all have one owner, and a second copy is how two
documents come to disagree about an experiment that has not run yet.

What this module adds is the part the protocol deliberately left open: which
five candidates, at which exact artifact identities, and what the session costs
on real hardware. The candidates come from attempt 3's frozen selection joined
to the destination-verified durable products the replay reconstructed — never
hand-transcribed, and refusing if a product's identity does not match what the
selection commits.

The storage provision is DERIVED from what this session actually holds: the
teacher, six staged initializations, one probe's training working set, and the
probe outputs it retains. It does not inherit the full search's 400 GB, which
was sized for a beam holding sixty compressed states at a level boundary and has
nothing to do with twelve sequential probes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_behavioural_proposal/v1"

PLAN_ID = "autoinit.v1.phase_c2.behavioural"
SESSION_ID = "autoinit-phase-c2-behavioural"

#: The frozen protocol. Its `behavioural_selection` block is the science.
PROTOCOL = ("logs/stages/stage-1/phase_c2/plans/"
            "phase_c2_full_search_protocol.json")

#: The measured probe cost, from six real probes on a real L40S.
PRICING = "logs/stages/stage-1/phase_c2/plans/phase_c2_full_search_pricing.json"

#: Where the replay left the five reconstructed products.
DURABLE_STORE = ("/home/ecs-user/aad-artifacts/phase_c2_full_search"
                 "/attempt3_replay")

STORAGE_PRICING = "configs/infrastructure/provider_storage_pricing.json"


class BehaviouralProposalError(RuntimeError):
    """The proposal cannot be derived from the committed evidence."""


# --------------------------------------------------------------------------
# the science, read from its owner
# --------------------------------------------------------------------------

def protocol(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen full-search protocol, verified against its own hash."""
    from aadistill.infrastructure.manifest import sha256_json

    doc = json.loads((Path(repo_root) / PROTOCOL).read_text())
    stated = doc.get("protocol_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "protocol_sha256"})
    if stated != recomputed:
        raise BehaviouralProposalError(
            f"{PROTOCOL} does not match its own protocol_sha256; it has been "
            "edited since it was frozen, and this proposal binds to it.")
    return doc


def schedule(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The 12-probe schedule, exactly as frozen."""
    block = protocol(repo_root)["behavioural_selection"]
    sched = block["schedule"]
    total = int(sched["total_probes"])
    screening, confirmation = sched["screening"], sched["confirmation"]
    if int(screening["probes"]) + int(confirmation["probes"]) != total:
        raise BehaviouralProposalError(
            f"the frozen schedule's rungs sum to "
            f"{screening['probes']} + {confirmation['probes']} and it declares "
            f"{total} probes")
    if total != 12:
        raise BehaviouralProposalError(
            f"the frozen schedule declares {total} probes, not 12. This "
            "proposal is for the 12-probe protocol and does not reshape it.")
    return sched


# --------------------------------------------------------------------------
# the inputs, joined rather than transcribed
# --------------------------------------------------------------------------

def candidate_manifest(repo_root: str | Path = REPO_ROOT,
                       store: str | Path = DURABLE_STORE) -> list[dict[str, Any]]:
    """The five candidates: frozen selection JOINED to durable products.

    Each entry carries the identity the selection commits AND the location of
    the product that was re-identified at the destination. The join is the
    check: a product whose identity differs from the selection's is refused
    here, so a behavioural session cannot be pointed at a checkpoint nobody
    reconstructed.
    """
    from experiments.phase_c2 import replay_specs as RS

    selection = RS.load_selection(repo_root)
    store = Path(store)
    out: list[dict[str, Any]] = []
    for rank, entry in enumerate(selection["selected"]):
        sid = entry["state_id"]
        d = store / sid
        ack = d / "durable_ack.json"
        sidecar = d / "replay_leaf.json"
        if not ack.is_file() or not sidecar.is_file():
            raise BehaviouralProposalError(
                f"candidate {sid} has no destination-verified product at {d}. "
                "The behavioural session consumes reconstructed checkpoints; "
                "it does not reconstruct them.")
        identity = json.loads(sidecar.read_text())["identity"]
        for field in ("artifact_digest", "weights_digest", "single_shard_sha256",
                      "arch_signature", "num_parameters"):
            if identity[field] != entry[field]:
                raise BehaviouralProposalError(
                    f"candidate {sid}: the durable product's {field} is "
                    f"{identity[field]!r} and the frozen selection commits "
                    f"{entry[field]!r}. These are different artifacts.")
        shard = d / "model.safetensors"
        out.append({
            "rank_in_frozen_selection": rank,
            "state_id": sid,
            "path": entry["path"],
            "lineage": next((x.get("lineage") for x in selection["decisions"]
                             if x["state_id"] == sid), ""),
            "artifact_digest": entry["artifact_digest"],
            "weights_digest": entry["weights_digest"],
            "single_shard_sha256": entry["single_shard_sha256"],
            "arch_signature": entry["arch_signature"],
            "num_parameters": int(entry["num_parameters"]),
            "durable_path": str(d),
            "bytes": shard.stat().st_size,
            "destination_verified_utc": json.loads(ack.read_text())["verified_utc"],
        })
    if len(out) != 5:
        raise BehaviouralProposalError(f"{len(out)} candidates, expected 5")
    return out


# --------------------------------------------------------------------------
# the sixth arm: incumbent B
# --------------------------------------------------------------------------

#: Where a prepared B is kept, beside the reconstructed candidates. One
#: directory, so "is B available" has a single answer and a later infrastructure
#: failure does not force another rebuild of something already built once.
B_DURABLE_PATH = ("/home/ecs-user/aad-artifacts/phase_c2_full_search"
                  "/incumbent_b")


def b_binding(repo_root: str | Path = REPO_ROOT, *,
              durable_path: str | Path = B_DURABLE_PATH,
              device: str = "cuda") -> dict[str, Any]:
    """The incumbent arm: its construction, its identity, and whether it exists.

    B is the sixth screening arm and the only anchor, and it is NOT a staged
    durable input the way the five candidates are. Baseline-completion attempt 8
    rebuilt it exactly, but that session's artifact manifest preserved evidence
    and logs rather than checkpoint bytes — so the bytes are gone and the
    identity is not.

    This binds the CONSTRUCTION from its canonical owner rather than
    re-declaring it: `baseline.frozen_baseline_spec` builds the path from C1's
    own `build_arm_specs`, and `assert_frozen_construction` refuses anything
    whose spec hash is not what C1's preregistration froze. Re-deriving the
    steps here would be a second construction of the thing whose sameness is the
    point.

    Preparing B is INITIALIZATION, not a probe. The protocol is twelve probes
    and stays twelve: B's materialization produces the arm that six of them
    measure against.
    """
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )

    from experiments.phase_c2 import baseline as BL
    from experiments.phase_c2.search_space import register_c2_operators

    #: EXPLICIT, and before the spec is built. `build_arm_specs` resolves every
    #: impl_id against the registry and B's last step is
    #: `attention.activation_importance_v1`, which is not a shipped default. A
    #: builder that only worked when some other entry point had registered
    #: would be a builder that works by luck.
    register_builtin_operators()
    register_c2_operators()

    spec = BL.frozen_baseline_spec(device=device)
    construction = BL.assert_frozen_construction(spec)

    d = Path(durable_path)
    available, observed = False, None
    if (d / "model.safetensors").is_file() and (d / "config.json").is_file():
        ack = d / "durable_ack.json"
        if ack.is_file():
            observed = json.loads(ack.read_text())
            available = bool(observed.get("re_identified_from_delivered_bytes"))

    return {
        "role": "frozen_c1_treatment_b -- the behavioural incumbent and the only anchor",
        "construction": {
            "owner": "scripts/experiments/phase_c2/baseline.py",
            "built_by": "experiments.phase_c1.session.build_arm_specs",
            "spec_hash": construction["spec_hash"],
            "expected_spec_hash": BL.B_SPEC_HASH,
            "path_label": construction["path_label"],
            "steps": construction["steps"],
            "parent_digest": BL.B_PARENT_DIGEST,
            "_bound_not_redeclared": (
                "the steps come from C1's own constructor and the spec hash is "
                "checked against the value C1's preregistration froze. A second "
                "declaration of this path is how two constructions of one thing "
                "come to differ."),
        },
        "required_identity": {
            "artifact_digest": BL.B_ARTIFACT_DIGEST,
            "weights_digest": BL.B_WEIGHTS_DIGEST,
            "config_sha256": BL.B_CONFIG_SHA256,
            "arch_signature": BL.B_ARCH_SIGNATURE,
            "single_shard_sha256": BL.B_SINGLE_SHARD_SHA256,
            "num_parameters": BL.B_NUM_PARAMETERS,
            "_observed_not_preregistered": (
                "these come from c1_arm_identities.json, which recorded what "
                "attempt 18 actually built. `treatment_output_digest_was_pre_"
                "pinned` is False there, because that attempt was the "
                "operator's first execution. It is the strongest available "
                "content identity of B, and this says which it is."),
        },
        "availability": {
            "durable_path": str(d),
            "available": available,
            "must_materialize": not available,
            "observed_ack": observed,
            "_why_it_is_not_staged": (
                "baseline-completion attempt 8 rebuilt B exactly and its "
                "artifact manifest preserved evidence and logs, not checkpoint "
                "bytes. The identity survived; the weights did not."),
        },
        "gate": (
            "the materialized B must match every field of required_identity. "
            "If it does not, NO SCREENING PROBE MAY START: an anchor that is "
            "not the frozen incumbent makes every delta meaningless."),
        "_not_a_thirteenth_probe": (
            "preparing B is initialization. The protocol is twelve probes and "
            "remains twelve; B's construction produces the arm six of them are "
            "measured against."),
    }


def b_preparation_minutes(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What materializing B costs, bounded from the replay's own measurements.

    B's path is the same four structural kinds the replay reconstructed, so its
    cost is bounded the same way: each step at the worst observation of THAT
    IMPLEMENTATION anywhere in attempt 3's telemetry, plus one teacher load.
    Bounding by operator kind rather than by implementation is what overpriced
    a replay path by 25 minutes.
    """
    from experiments.phase_c2 import baseline as BL
    from experiments.phase_c2 import replay_specs as RS

    worst = RS.worst_seconds_by_impl(repo_root)
    per_step = []
    total = 1.5          # one teacher load, as the replay charges per path
    for kind, impl_id, profile in BL.B_PATH:
        if impl_id not in worst:
            raise BehaviouralProposalError(
                f"no measured timing for {impl_id}; B's preparation cannot be "
                "bounded from evidence and will not be guessed")
        minutes = worst[impl_id] / 60.0
        per_step.append({"kind": kind, "impl_id": impl_id,
                         "profile_id": profile, "bounded_minutes": round(minutes, 2)})
        total += minutes
    return {
        "steps": per_step,
        "teacher_load_minutes": 1.5,
        "bounded_minutes": round(total, 2),
        "_basis": ("the worst observation of each implementation anywhere in "
                   "attempt 3's telemetry, the same bound the replay spent "
                   "against and which reproduced all five paths"),
    }


# --------------------------------------------------------------------------
# storage, derived from what the session holds
# --------------------------------------------------------------------------

#: Bytes per parameter for the pieces a training step keeps resident. The model
#: is bf16; AdamW's two moments and the gradient accumulate in fp32.
_BF16, _FP32 = 2, 4

#: Measured, not assumed: the pinned teacher's snapshot on this machine.
TEACHER_GIB = 7.51

#: Headroom for the image, the built environment and the HF cache's own
#: bookkeeping. C1's sessions ran the same stack.
IMAGE_AND_ENV_GIB = 30.0


#: The worst single fixed path's intermediates, measured during the replay: a
#: four-step construction holds its predecessors while it builds. B's path is
#: four steps of the same kinds, so this bounds its transient residency.
B_MATERIALIZATION_TRANSIENT_GIB = 16.12


def storage_requirement(candidates: list[dict[str, Any]],
                        sched: dict[str, Any],
                        repo_root: str | Path = REPO_ROOT, *,
                        b_must_materialize: bool = True) -> dict[str, Any]:
    """What this session actually needs on disk, component by component.

    Derived from the artifacts rather than inherited. The full search provisions
    400 GB because a beam generates a whole level before pruning and holds sixty
    compressed states at once; twelve sequential probes hold one training working
    set at a time.
    """
    params = candidates[0]["num_parameters"]
    if any(c["num_parameters"] != params for c in candidates):
        raise BehaviouralProposalError(
            "the candidates do not share a parameter count; the training "
            "working set cannot be derived from one of them")

    leaf_gib = candidates[0]["bytes"] / 2**30
    n_screening_arms = int(sched["screening"]["arms"])

    #: One probe at a time: weights, gradient and the two AdamW moments.
    working_set = (params * _BF16 + params * _FP32 * 3) / 2**30

    #: Every probe writes a trained checkpoint, and P18 requires the complete
    #: raw generation for every evaluated sample. The generations are text and
    #: small beside the weights, but they are not nothing.
    trained = int(sched["total_probes"]) * leaf_gib
    generations = 2.0

    components = {
        "teacher": round(TEACHER_GIB, 3),
        "staged_initializations": round(n_screening_arms * leaf_gib, 3),
        "b_materialization_transient": (
            B_MATERIALIZATION_TRANSIENT_GIB if b_must_materialize else 0.0),
        "_b_materialization_transient_is": (
            "ONE arm's construction intermediates, resident while that arm "
            "builds and released when it is verified. A four-step path holds "
            "its predecessors as it goes; this is the worst single path's "
            "intermediates as measured during the replay. Counted once, not "
            "six times, because the arms are built SEQUENTIALLY -- at the last "
            "build, five finished arms are resident plus this transient, which "
            "is what the sum below already charges"
            if b_must_materialize else
            "zero: every arm already exists and is staged, not rebuilt"),
        "_staged_initializations_is": (
            f"{n_screening_arms} screening arms -- the five reconstructed "
            "candidates and the incumbent B -- resident before any probe runs. "
            "They are MATERIALIZED on the pod rather than shipped to it: each "
            "is a 1.19 GB checkpoint, the scp path allows one asset 600 s "
            "against a dev-box uplink needing ~1650, and the hub relay refuses "
            "5.95 GB for private-storage quota. Residency is the same either "
            "way; only the transient above is added by building them here"),
        "one_probe_training_working_set": round(working_set, 3),
        "_working_set_is": (
            f"{params:,} parameters as bf16 weights plus an fp32 gradient and "
            "AdamW's two fp32 moments. Probes run sequentially, so exactly one "
            "of these is resident at a time"),
        "trained_probe_checkpoints": round(trained, 3),
        "_trained_is": f"{sched['total_probes']} probes x {leaf_gib:.3f} GiB retained",
        "saved_generations": generations,
        "batteries_and_ladder": 1.0,
        "image_and_environment": IMAGE_AND_ENV_GIB,
    }
    subtotal = sum(v for k, v in components.items() if not k.startswith("_"))
    #: A provision is an integer handed to the provider and it is billed whole,
    #: so it rounds UP, with a margin that is named rather than folded in.
    margin = 0.25
    with_margin_gib = subtotal * (1 + margin)

    #: GiB -> GB, through the repository's OWN recorded conversion rather than a
    #: second local convention. The residency above is derived in GiB (2^30) and
    #: `--container-disk-in-gb` says GB; if the provider means decimal GB, a
    #: request of N delivers only 0.931*N GiB, so treating the flag as GiB
    #: UNDER-PROVISIONS by 7%. This file made exactly that error -- it rounded a
    #: GiB subtotal straight into a GB flag -- which is the unit bug the full
    #: search had already been repaired for.
    conv = storage_pricing(repo_root)["gb_versus_gib"]
    gb_per_gib = float(conv["gb_per_gib"])
    with_margin_gb = with_margin_gib * gb_per_gib
    provision = int(math.ceil(with_margin_gb / 10.0) * 10)
    return {
        "components_gib": components,
        "subtotal_gib": round(subtotal, 3),
        "margin_fraction": margin,
        "with_margin_gib": round(with_margin_gib, 4),
        "gb_per_gib": gb_per_gib,
        "with_margin_gb": round(with_margin_gb, 4),
        "provision_gb": provision,
        "_units": (
            "the residency is derived in GiB and the provider's flag is GB. "
            "The conversion is the one recorded in "
            "configs/infrastructure/provider_storage_pricing.json, not a local "
            "convention: rounding a GiB subtotal straight into a GB flag "
            "under-provisions by 7%."),
        "_why_not_400": (
            "the full search's 400 GB was derived for a beam that generates a "
            "whole level before pruning and holds sixty compressed states at "
            "the level boundary. This session holds one training working set at "
            "a time and retains twelve small checkpoints; carrying 400 GB here "
            "would bill for storage nothing uses."),
        "_why_it_rounds_up": (
            "a volume that fills mid-probe loses the probe. The margin is 25% "
            "over a component sum that is itself derived from measured artifact "
            "sizes."),
    }


# --------------------------------------------------------------------------
# money
# --------------------------------------------------------------------------

def storage_pricing(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    doc = json.loads((Path(repo_root) / STORAGE_PRICING).read_text())
    if doc["container_disk"].get("provider_api_exposes_this") is not False:
        raise BehaviouralProposalError(
            f"{STORAGE_PRICING} claims the provider exposes container-disk "
            "pricing. It does not, and a stated basis presenting itself as a "
            "quote is worse than no basis.")
    return doc


#: THE canonical phase decomposition of the complete behavioural session, and
#: the reason it exists as one function.
#:
#: The launcher used to build a SECOND decomposition on top of this module's
#: FINAL figure. `ceiling()` reported a hard window of 1800.53 min that already
#: contained the frozen probe model's named reserves, its 10% contingency and
#: its artifact-recovery reserve; the launcher then subtracted the
#: materialization term back out, fed the remainder into a fresh `BudgetSpec`
#: beside `setup`, `transfer` and `materialize` phases, and applied a SECOND
#: contingency and a SECOND artifact-recovery reserve. `plan_session` answered
#: 2036.62 hard minutes — about `$36.9987` of GPU at `$1.09/h`, larger than the
#: whole proposed all-in ceiling. That is not extra conservatism: a correct
#: authorization derived from the proposal would have REFUSED the launch, at the
#: gate, for reserves nobody granted twice.
#:
#: So both consumers read this one decomposition. The proposal prices it; the
#: launcher's `BudgetSpec` carries its phases and its reserves verbatim and
#: applies no contingency of its own, because the contingency is already here as
#: a named number of minutes.
def session_decomposition(repo_root: str | Path = REPO_ROOT, *,
                          materialization_minutes: float,
                          probes_remaining: int = 12,
                          restore_minutes: float = 0.0) -> dict[str, Any]:
    """Expected phases, named reserves and the recovery reserve. ONE owner.

    **It prices REMAINING work, and a fresh campaign's remaining work is all of
    it.** The defaults — twelve probes, no restore — reproduce the full session
    exactly: 1294.87 expected and 1800.53 hard minutes, the figures the proposal
    is authorized against. A continuation passes what its campaign still owes.

    That generalisation is the whole of it, and it is here rather than in a
    second function because a continuation budget derived beside this one would
    be the same defect the launcher's duplicate budget was. The probe-derived
    terms are linear in the probe count by construction — the frozen record
    derives each from a per-probe mean or maximum times twelve — so scaling them
    is arithmetic on the model, not a new model.

    A continuation may not re-reserve work it cannot execute. A completed probe
    is never retrained, so it is not owed; the arms of the probes that DO remain
    must be rebuilt on a fresh pod, so they are; and restoring verified probes
    from the durable destination is billed pod time, so it is a phase.

    Every figure is read from the frozen pricing record or from the pricing
    model that wrote it, and the two are RECONCILED here rather than trusted:

    * the overhead phases must sum to the record's own expected-minus-probe
      remainder, or the record and the model disagree about what a session does
      besides train;
    * the three named reserves plus the recovery reserve must sum to the
      record's own `hard_ceiling_minutes` minus its `expected_minutes`, or the
      reserve block has moved since the record was frozen.

    A mismatch raises. A window derived from a decomposition that no longer
    reconstructs its own source is a window that bounds nothing.

    The 10% contingency is carried as a named RESERVE in minutes rather than as
    a fraction, for two reasons. It is the frozen model's contingency on the
    frozen model's expected path, so re-deriving it from a phase sum that also
    contains the materialization would silently change a frozen figure. And the
    materialization term is already the worst observation of each operator
    IMPLEMENTATION anywhere in attempt 3's telemetry — a bound, not a mean — so
    multiplying it by a contingency built for means would charge a risk margin
    on a number that is already the risk margin.

    **There is no separate probe-transfer phase, and that is deliberate.** The
    launcher's old second budget added `12 x 1.7 = 20.4` minutes of probe
    transfer as a serial phase and then reused the same 20.4 as its recovery
    reserve. Probes leave the pod from the runner's POLL LOOP, concurrently with
    the next probe's training, so those minutes are not serial wall clock; the
    frozen model already carries `artifact_synchronization` for the part that
    is, and a 30-minute recovery reserve — larger than the 20.4 it replaces —
    for the collection that happens after the soft stop.
    """
    from experiments.phase_c2 import selection_pricing as SP

    beh = json.loads(
        (Path(repo_root) / PRICING).read_text())["behavioural_selection"]
    bounding = beh["bounding_basis"]
    probe_expected = float(beh["expected_minutes"])
    probe_hard = float(beh["hard_ceiling_minutes"])
    probe_minutes = float(bounding["probe_minutes_expected"])

    overheads = tuple((name, float(minutes))
                      for name, minutes in SP.SESSION_PHASE_MINUTES)
    overhead_total = sum(m for _, m in overheads)
    if abs(overhead_total - (probe_expected - probe_minutes)) > 0.01:
        raise BehaviouralProposalError(
            f"the pricing model's session overheads sum to {overhead_total} min "
            f"and the frozen record's expected path leaves "
            f"{probe_expected - probe_minutes} min beside its probes. The record "
            "and the model that wrote it disagree about what a session does "
            "besides train, so neither can be decomposed into phases.")

    recovery = float(SP.ARTIFACT_RECOVERY_RESERVE_MINUTES)
    #: The FULL-session reserve block, reconciled against the frozen record
    #: before anything is scaled. Reconciling a scaled block against an
    #: unscaled record would be checking arithmetic against itself.
    full_reserves = (
        ("probe_model_contingency",
         round(probe_expected * SP.CONTINGENCY_FRACTION, 2)),
        ("probe_duration_risk",
         round(float(bounding["probe_minutes_observed_max"]) - probe_minutes, 2)),
        ("generation_length_risk",
         float(bounding["generation_length_reserve_minutes"])),
    )
    full_reserve_total = sum(m for _, m in full_reserves)
    if abs(probe_hard - (probe_expected + full_reserve_total + recovery)) > 0.01:
        raise BehaviouralProposalError(
            f"the named reserves ({full_reserve_total} min) plus the recovery "
            f"reserve ({recovery} min) do not reconstruct the frozen record's "
            f"hard ceiling: {probe_expected} + {full_reserve_total} + "
            f"{recovery} != {probe_hard}. The reserve block has moved since the "
            "record was frozen and this decomposition would bound the wrong "
            "window.")

    total_probes = int(beh["n_probes"])
    if not 0 <= int(probes_remaining) <= total_probes:
        raise BehaviouralProposalError(
            f"{probes_remaining} probes remaining is outside 0..{total_probes}. "
            "A continuation owes some subset of the frozen protocol's probes, "
            "never more of them.")
    #: Linear in the probe count BY CONSTRUCTION: the frozen record derives
    #: each probe term from a per-probe mean or observed maximum times twelve.
    #: Scaling is arithmetic on that model, not a second model.
    scale = int(probes_remaining) / total_probes
    probe_minutes_owed = round(probe_minutes * scale, 2)

    #: The contingency follows the work it covers — the session overheads plus
    #: the probes that remain. At the default it is exactly the frozen model's
    #: own figure, `1139.04 x 0.10`.
    reserves = (
        ("probe_model_contingency",
         round((overhead_total + probe_minutes_owed) * SP.CONTINGENCY_FRACTION, 2)),
        ("probe_duration_risk", round(dict(full_reserves)["probe_duration_risk"]
                                      * scale, 2)),
        ("generation_length_risk",
         round(dict(full_reserves)["generation_length_risk"] * scale, 2)),
    )
    reserve_total = sum(m for _, m in reserves)

    #: Materializing an arm is INITIALIZATION, not a probe. The protocol is
    #: twelve probes and stays twelve; the pod is alive longer because the arms
    #: a remaining probe measures against have to exist first.
    phases: list[tuple[str, float]] = [
        *overheads,
        ("materialize_arms", round(float(materialization_minutes), 2)),
    ]
    if restore_minutes:
        #: Billed pod minutes: `local_assets` and any scp to a pod happen AFTER
        #: it exists. Restoring verified probes from the durable destination is
        #: real wall clock on a running meter, so it is a phase and not a
        #: rounding note.
        phases.append(("restore_verified_probes",
                       round(float(restore_minutes), 2)))
    phases.append((f"{int(probes_remaining)}_probes_remaining",
                   probe_minutes_owed))
    expected_phases = tuple(phases)
    expected = round(sum(m for _, m in expected_phases), 2)
    soft_stop = round(expected + reserve_total, 2)
    hard = round(soft_stop + recovery, 2)
    return {
        "expected_phases": expected_phases,
        "soft_stop_reserves": reserves,
        "probes_remaining": int(probes_remaining),
        "probes_in_protocol": total_probes,
        "restore_minutes": round(float(restore_minutes), 2),
        "_remaining_work_only": (
            "a completed probe is never retrained, so it is not priced again. "
            "The arms its remaining probes measure against ARE priced, because "
            "a replacement resource has a fresh filesystem and must rebuild "
            "them; that is a replacement runtime necessity, not completed "
            "science charged twice."),
        "full_session_reserves": full_reserves,
        "artifact_recovery_reserve_minutes": recovery,
        #: For a `BudgetSpec`. ZERO, deliberately: see the module note above.
        "contingency_fraction": 0.0,
        "_contingency_is_a_named_reserve": (
            "the frozen probe model's 10% contingency is carried in "
            "soft_stop_reserves as a fixed number of minutes. A consumer that "
            "also set contingency_fraction=0.10 would apply it twice, which is "
            "the defect this decomposition exists to remove."),
        "_overhead_phase_names_are_the_pricing_models": (
            "the first six phases are selection_pricing.SESSION_PHASE_MINUTES "
            "verbatim, because that is what the frozen record's 95 non-probe "
            "minutes ARE and renaming them would break the reconciliation. "
            "They are a cost model shared with the full search, not a claim "
            "about this session's stages: `selection_commit_and_artifact_"
            "manifest` funds the closeout minutes, and this session commits no "
            "selection — it consumes one that is already frozen. They are NOT "
            "scaled by the probe count: a replacement resource pays setup, the "
            "bundle, the teacher fetch and the machine gates in full."),
        "expected_minutes": expected,
        "soft_stop_minutes": soft_stop,
        "hard_minutes": hard,
        "probe_model": {
            "record": PRICING,
            "expected_minutes": probe_expected,
            "hard_ceiling_minutes": probe_hard,
            "probe_minutes_expected": probe_minutes,
            "contingency_fraction": SP.CONTINGENCY_FRACTION,
        },
        "_reconciled": (
            "the overhead phases reconstruct the record's non-probe expected "
            "minutes and the reserves reconstruct its hard ceiling; both are "
            "checked here rather than assumed"),
    }


def money(repo_root: str | Path = REPO_ROOT, *,
          gpu_rate_usd_per_hour: float,
          provision_gb: int,
          materialization_minutes: float = 0.0) -> dict[str, Any]:
    """Expected and hard-ceiling cost, GPU and separately billed storage apart.

    The minutes come from `session_decomposition`, which is the ONE canonical
    phase decomposition of this session: the frozen probe model's overheads and
    probe minutes, the six arms' materialization bound, and the frozen model's
    own named reserves. This module does not re-derive any of them — it prices
    that decomposition at a live rate and adds the storage the old record never
    costed for this session.
    """
    decomposition = session_decomposition(
        repo_root, materialization_minutes=materialization_minutes)
    expected_min = decomposition["expected_minutes"]
    hard_min = decomposition["hard_minutes"]
    beh = json.loads(
        (Path(repo_root) / PRICING).read_text())["behavioural_selection"]
    probe_expected = float(beh["expected_minutes"])
    probe_hard = float(beh["hard_ceiling_minutes"])

    disk = storage_pricing(repo_root)
    per_gb_month = float(disk["container_disk"]["usd_per_gb_month"])
    hours_per_month = float(disk["proration"]["hours_per_month"])
    disk_per_hour = per_gb_month / hours_per_month * provision_gb

    def usd(minutes: float) -> float:
        return math.ceil(minutes / 60.0 * gpu_rate_usd_per_hour * 10_000) / 10_000

    def disk_usd(minutes: float) -> float:
        return math.ceil(minutes / 60.0 * disk_per_hour * 10_000) / 10_000

    gpu_expected, gpu_hard = usd(expected_min), usd(hard_min)
    disk_expected, disk_hard = disk_usd(expected_min), disk_usd(hard_min)
    return {
        "gpu_rate_usd_per_hour": gpu_rate_usd_per_hour,
        "_gpu_rate_is_live": ("re-quoted from gpuTypes.securePrice; the "
                              "authorization must re-quote again at issue"),
        "probe_minutes_basis": beh["probe_cost"]["source"],
        "probe_minutes": {"expected": probe_expected, "hard_ceiling": probe_hard},
        "materialization_minutes": materialization_minutes,
        "decomposition": decomposition,
        "_one_decomposition": (
            "session_decomposition is the single canonical phase decomposition "
            "of this session. The proposal prices it here and the launcher's "
            "BudgetSpec carries the same phases and reserves, so the two cannot "
            "derive different hard windows."),
        "_materialization_is_initialization": (
            "added to the session's runtime, not to the probe count. The "
            "protocol is twelve probes and stays twelve; the pod is alive "
            "longer because the six arms have to be built before any probe can "
            "measure against them."),
        "expected": {"minutes": expected_min, "gpu_usd": gpu_expected,
                     "disk_usd": disk_expected,
                     "all_in_usd": round(gpu_expected + disk_expected, 4)},
        "hard_ceiling": {"minutes": hard_min, "gpu_usd": gpu_hard,
                         "disk_usd": disk_hard,
                         "all_in_usd": round(gpu_hard + disk_hard, 4)},
        "container_disk": {
            "provisioned_gb": provision_gb,
            "usd_per_gb_month": per_gb_month,
            "hours_per_month": hours_per_month,
            "usd_per_hour": round(disk_per_hour, 6),
            "_billed_for_the_pods_whole_lifetime": True,
        },
        "_ceiling_includes": beh["_ceiling_includes"],
        "_two_numbers": (
            "the GPU rate is re-quotable from the provider and the storage "
            "price is not, so they are derived apart and summed. A live GPU "
            "quote is not evidence that separately priced storage is free."),
    }

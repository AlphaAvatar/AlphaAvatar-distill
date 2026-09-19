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


def storage_requirement(candidates: list[dict[str, Any]],
                        sched: dict[str, Any]) -> dict[str, Any]:
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
        "_staged_initializations_is": (
            f"{n_screening_arms} screening arms -- the five reconstructed "
            "candidates and the incumbent B, each staged before any probe runs"),
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
    provision = int(math.ceil(subtotal * (1 + margin) / 10.0) * 10)
    return {
        "components_gib": components,
        "subtotal_gib": round(subtotal, 3),
        "margin_fraction": margin,
        "provision_gb": provision,
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


def money(repo_root: str | Path = REPO_ROOT, *,
          gpu_rate_usd_per_hour: float,
          provision_gb: int) -> dict[str, Any]:
    """Expected and hard-ceiling cost, GPU and separately billed storage apart.

    The GPU minutes come from the frozen pricing record, which derived them from
    six real probes: the expected path on per-probe means, the ceiling on the
    observed maxima plus a named generation-length reserve. This module does not
    re-derive them — it re-prices them at a live rate and adds the storage the
    old record never costed for this session.
    """
    pricing = json.loads((Path(repo_root) / PRICING).read_text())
    beh = pricing["behavioural_selection"]
    expected_min = float(beh["expected_minutes"])
    hard_min = float(beh["hard_ceiling_minutes"])

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

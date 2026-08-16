"""Generation-runtime comparability, v2. Additive; nothing historical is rewritten.

Phase-A attempt 4 was refused at $0.2052 by a binding that was working exactly as
specified, on a difference that carries no generation semantics. Every observed
field that governs decoding matched the Stage-3 controls — vLLM, transformers,
torch, dtype, engine settings, tokenizer, chat template, context, stop ids — and
the protocol hashes still differed, because inside the protocol the runtime is a
single opaque ``runtime_digest`` and inside *that* the ``image_digest`` was:

    stage 3     runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.159.03
    attempt 4   runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09

Same image tag. The suffix is the **host NVIDIA driver version** that
``Preflight.read_image_digest`` appends when ``/etc/podinfo/image_digest`` is
absent, so the field named ``image_digest`` is really ``imageName@driver`` — two
concepts fused into one. Since the provider assigns whatever host is free, an
exact-equality rule over that fused field is a host lottery.

**This does not say drivers are irrelevant.** It says an exact *patch* is not by
itself a generation-semantic difference, and it keeps an explicit compatibility
constraint — for this frozen study, the same **580 driver branch** — rather than
claiming universal irrelevance. A branch change is a real event and still fails.

Scope, deliberately narrow:

* This is for **generation/evaluation comparability only.** The historical
  recovery/training runtime identity in `recovery.py` is untouched, and so is
  every file inside `generation_source_digest` and the scoring contract —
  changing any of those would move the very hashes this reconciles.
* Nothing historical is rewritten. `250f72ef…` remains the protocol the Stage-3
  controls attested, and their artifacts stay byte-for-byte.
* Threshold values, `pooled_counts@v2`, the search, the recovery design, the
  seeds and the science plan are not touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..infrastructure.manifest import sha256_json


class ComparabilityError(RuntimeError):
    """Two evaluation protocols are not comparable under the declared rule."""


#: Fields carried in the protocol that are provenance rather than generation
#: semantics, and are therefore recorded but excluded from the comparable
#: identity.
#:
#: `runtime_digest` is opaque and fuses the image with the host driver; it is
#: replaced by the explicit split below.
#:
#: `generation_protocol_fingerprint` is **derived, not independent**: it is
#: `sha256_json(identity())`, and `as_dict()` already spreads every field of
#: `identity()` individually into the same block. Dropping it therefore removes
#: no information — every input remains separately material, so a mutation to
#: any of them is still caught by that field itself. It is dropped only because
#: it transitively contains `runtime_digest`, which would otherwise smuggle the
#: demoted driver patch back into the identity. Verified against the source
#: rather than assumed.
NON_MATERIAL_PROTOCOL_FIELDS = ("runtime_digest", "evaluation_protocol_hash",
                                "generation_protocol_fingerprint")

#: Runtime fields that ARE material: the user-space stack the engine runs on.
MATERIAL_RUNTIME_FIELDS = ("python_version", "torch_version",
                           "transformers_version", "cuda_runtime",
                           "attention_backend")


@dataclass(frozen=True)
class GenerationRuntimeComparability:
    """How two generation runtimes are compared, versioned so it can be cited."""

    rule_id: str = "generation_runtime_comparability"
    version: int = 2
    #: The compatibility constraint retained for this frozen study. Not a claim
    #: that drivers never matter — a claim that within one branch the patch does
    #: not move generation semantics, and that a branch change must still fail.
    driver_branch_must_match: bool = True
    supersedes: str = ("exact equality over the fused imageName@driver field, "
                       "which made comparability a host lottery")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id, "version": self.version,
            "qualified_id": f"{self.rule_id}@v{self.version}",
            "driver_branch_must_match": self.driver_branch_must_match,
            "material": {
                "image": ("the real container image digest when the host exposes "
                          "one; otherwise the image ref/tag alone"),
                "user_space_stack": list(MATERIAL_RUNTIME_FIELDS),
                "generation": ("every field the protocol already declares except "
                               f"{list(NON_MATERIAL_PROTOCOL_FIELDS)}"),
                "also": "scoring contract + digest, and battery identity",
            },
            "recorded_not_material": ["nvidia_driver_version", "host_provenance"],
            "supersedes": self.supersedes,
        }


GENERATION_RUNTIME_COMPARABILITY_V2 = GenerationRuntimeComparability()


def split_image_identity(image_digest: str | None) -> dict[str, Any]:
    """Separate a real image digest from the ``imageName@driver`` fallback.

    A real digest is ``<ref>@sha256:<hex>`` and is material in full. The fallback
    is ``<ref>@<driver>``, where only the ref is material and the driver is
    provenance. A bare ref with no ``@`` is simply the ref.
    """
    if not image_digest:
        return {"image_ref": None, "container_image_digest": None,
                "nvidia_driver_version": None, "form": "absent"}
    ref, sep, suffix = image_digest.rpartition("@")
    if not sep:
        return {"image_ref": image_digest, "container_image_digest": None,
                "nvidia_driver_version": None, "form": "ref_only"}
    if suffix.startswith("sha256:"):
        return {"image_ref": ref, "container_image_digest": suffix,
                "nvidia_driver_version": None, "form": "content_digest"}
    return {"image_ref": ref, "container_image_digest": None,
            "nvidia_driver_version": suffix, "form": "ref_plus_driver"}


def driver_branch(version: str | None) -> str | None:
    """`580.159.03` -> `580`. The branch is the compatibility unit."""
    if not version:
        return None
    return version.split(".")[0]


def comparable_generation_identity(*, protocol: Mapping[str, Any],
                                   runtime: Mapping[str, Any],
                                   host_provenance: Mapping[str, Any] | None = None,
                                   ) -> dict[str, Any]:
    """The v2 comparable identity of one evaluation protocol.

    ``protocol`` is a `RecoveryEvaluationProtocol.as_dict()`; ``runtime`` is the
    expanded runtime block the engine probe records, which is where the image and
    the user-space stack actually live (the protocol only carries their opaque
    digest).

    Everything the protocol declares stays material **except** the two provenance
    fields, so a change to any generation-semantic field still moves the identity
    by construction. Only the driver patch is demoted, and it is demoted to
    *recorded*, not discarded.
    """
    generation = dict(protocol.get("generation") or {})
    dropped = {k: generation.pop(k, None) for k in NON_MATERIAL_PROTOCOL_FIELDS}

    image = split_image_identity(runtime.get("image_digest"))
    material_image = (image["container_image_digest"]
                      if image["container_image_digest"] else image["image_ref"])

    material = {
        "rule": GENERATION_RUNTIME_COMPARABILITY_V2.as_dict()["qualified_id"],
        "image": {"material_value": material_image, "form": image["form"]},
        "runtime_stack": {k: runtime.get(k) for k in MATERIAL_RUNTIME_FIELDS},
        "generation": generation,
        "scoring_contract": protocol.get("scoring_contract"),
        "scoring_digest": protocol.get("scoring_digest"),
        "battery": protocol.get("battery"),
    }
    return {
        "schema": "aadistill.autoinit.comparable_generation_identity/v2",
        "rule": GENERATION_RUNTIME_COMPARABILITY_V2.as_dict(),
        "material": material,
        "comparable_identity": sha256_json(material),
        "recorded_not_material": {
            "nvidia_driver_version": image["nvidia_driver_version"],
            "driver_branch": driver_branch(image["nvidia_driver_version"]),
            "fused_image_digest_field": runtime.get("image_digest"),
            "dropped_protocol_fields": dropped,
            "host_provenance": dict(host_provenance or {}),
        },
    }


def require_comparable(live: Mapping[str, Any], historical: Mapping[str, Any], *,
                       context: str = "") -> dict[str, Any]:
    """Refuse unless the two identities agree AND the driver branch matches.

    Returns the comparison so a caller can record precisely what was equal and
    what was merely recorded — the point of the split is that the difference is
    visible, not that it is hidden.
    """
    where = f" ({context})" if context else ""
    a, b = live["comparable_identity"], historical["comparable_identity"]
    lb = live["recorded_not_material"]["driver_branch"]
    hb = historical["recorded_not_material"]["driver_branch"]
    comparison = {
        "rule": GENERATION_RUNTIME_COMPARABILITY_V2.as_dict()["qualified_id"],
        "live_identity": a, "historical_identity": b, "identities_equal": a == b,
        "live_driver": live["recorded_not_material"]["nvidia_driver_version"],
        "historical_driver": historical["recorded_not_material"]["nvidia_driver_version"],
        "driver_branch_equal": lb == hb,
        "driver_patch_differs": (
            live["recorded_not_material"]["nvidia_driver_version"]
            != historical["recorded_not_material"]["nvidia_driver_version"]),
    }
    if a != b:
        diff = sorted(
            k for k in set(live["material"]) | set(historical["material"])
            if live["material"].get(k) != historical["material"].get(k))
        comparison["differing_material_keys"] = diff
        raise ComparabilityError(
            f"evaluation protocols are not comparable{where} under "
            f"{comparison['rule']}: material identity {a} != {b}; differing "
            f"{diff}")
    if GENERATION_RUNTIME_COMPARABILITY_V2.driver_branch_must_match and lb != hb:
        raise ComparabilityError(
            f"evaluation protocols are not comparable{where}: the material "
            f"identity matches, but the NVIDIA driver branch moved {hb} -> {lb}. "
            "This study treats a patch within a branch as provenance and a branch "
            "change as a real runtime event; re-characterize or re-authorize "
            "rather than assuming it is inert.")
    return comparison

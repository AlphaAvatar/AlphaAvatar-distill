"""The C2 baseline, B: found if the search re-derived it, rebuilt exactly once if not.

Search-1 asks whether re-optimizing composition and order beats the frozen C1
treatment. That question needs B measured on the same cheap metric as every
candidate, and **no C1 checkpoint bytes survived**: attempt 18's artifact spec
collects twenty-two classes and not one is a weight file. What survived is B's
*identity*, and identity is enough, because B is deterministically derivable
from the teacher.

So B reaches the comparison one of two ways, and which one is not a judgement
call:

1. **The search re-derives it.** B's order and its ATTENTION mixture are both
   inside the Search-1 space, and the search carries the same frozen seed, so a
   beam that reaches `DEPTH→FFN→RESIDUAL_WIDTH→ATTENTION(domain_balanced)`
   produces B byte for byte. Then B is already a measured leaf and nothing is
   rebuilt.
2. **The beam did not reach it** — a beam of 6 out of 18 is not a guarantee —
   and B is rebuilt **once**, through the frozen C1 treatment fixed path, and
   passed through the same `identify_checkpoint` → canonical reload →
   `state_eval` semantics every searched candidate goes through, via the
   `retained_candidates` seam.

**Never both.** Their identities collide by construction — that is the whole
reason the search can re-derive B at all — so injecting a rebuilt B beside a
searched one would put two states with one `artifact_digest` into the candidate
set. `BaselineFallback` is single-shot and says so.

**Every gate fails closed, and the cheap one runs first.** The frozen spec hash
is checkable before a single tensor moves; the parent digest and the output
digest are checked by machinery that already exists (`materialize_fixed_path`
gates a step against `expected_artifact_digest`, and `make_retained_state`
refuses a checkpoint whose bytes are not what the evidence describes). On a
mismatch the artifact digest is DECOMPOSED — weights, config, arch signature,
tokenizer, index — because "the digest differs" is not a diagnosis and the
components mean different things: different weights is a scientific finding,
a different config is a runtime one, and a reader must be able to tell.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.initialization.planning.fixed_path import (
    FixedPathSpec,
    FixedPathStep,
    materialize_fixed_path,
)
from aadistill.initialization.specs.artifact import CheckpointIdentity


class BaselineError(RuntimeError):
    """The baseline cannot be established as the frozen one. Fail closed."""


# --- the frozen identities --------------------------------------------------
#
# All of these are read out of committed C1 evidence by
# `tests/autoinit/test_phase_c2_baseline.py`, which re-derives each one from the
# file it came from rather than trusting this block. Two different documents own
# them, and the distinction matters:
#
#   * the PREREGISTRATION owns the construction — what C1 declared it would
#     build, frozen before it ran;
#   * the ARM IDENTITIES record owns the result — what it actually built,
#     observed once.

PREREGISTRATION = "logs/stages/stage-1/phase_c1/plans/execution_preregistration.json"
ARM_IDENTITIES = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
                  "c1_arm_identities.json")

#: `execution_preregistration.json:fixed_path.treatment_spec_hash`. The
#: construction, frozen BEFORE C1 ran, and the only identity checkable at $0
#: before any compute — which is why it is the first gate.
B_SPEC_HASH = "3a233a9017b3b8a717ff18fc1aaa171dad84f36adc97920765d181ca98c53612"

#: `execution_preregistration.json:replay_gates.expected_parent_digest`, and the
#: pin the frozen prefix already carries: the shared `DEPTH→FFN→WIDTH` parent.
#: C1 replayed it successfully three times (attempts 9, 17, 18).
B_PARENT_DIGEST = "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722"

#: `c1_arm_identities.json:treatment.*`. OBSERVED, not preregistered —
#: `treatment_output_digest_was_pre_pinned` is False in that record, because
#: attempt 18 was this operator's first execution and there was no earlier
#: digest to pin it to. That makes it the strongest available content identity
#: of B and not a prior expectation, and this module says which it is.
B_ARTIFACT_DIGEST = "53e30566c5f795f1870d76c1fa6a970ddc507fa5459047f3010ffab8aa890342"
B_WEIGHTS_DIGEST = "85c5c50de3cd7e5242097e05e6f09057d9b27ae98ec49050219e75a2f49e2335"
B_CONFIG_SHA256 = "f69c2fb3a120db405cd48c4b643b28d10a38b7b2e3ba9a3618e110888f2a6841"
B_ARCH_SIGNATURE = "09147a5c08d8ad7ad5bb18131bade73548a5fff08f0e0f99d6dd6ede227606c4"
B_SINGLE_SHARD_SHA256 = (
    "b24f0b63afd3b99f14d34d4fc508abbb26f65421dc5784f3738bd9565608d010")
B_NUM_PARAMETERS = 596049920

#: The runtime attempt 18 built B under. NOT a gate — a different image is
#: legitimate. It is recorded beside a mismatch so that a `config_sha256` that
#: moved while the weights did not can be read as what it is.
B_RUNTIME = {"image_digest": "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@595.91.07",
             "torch": "2.11.0+cu128", "transformers": "5.13.1",
             "cuda_runtime": "12.8", "gpu": "NVIDIA L40S"}

#: B's construction as the SEARCH records it: ordered `(kind, impl_id,
#: profile_id)`. Detection is by construction rather than by digest, so a search
#: state that built B's path but produced different bytes is a finding rather
#: than an absence — and the digest check below is what turns it into one.
B_PATH: tuple[tuple[str, str, str], ...] = (
    ("DEPTH", "depth.causal_kl_greedy_v1", "calib.domain_balanced@v1"),
    ("FFN", "ffn.activation_importance_v0", "calib.domain_balanced@v1"),
    ("RESIDUAL_WIDTH", "width.global_pca_v0", "calib.reasoning_heavy@v2"),
    ("ATTENTION", "attention.activation_importance_v1", "calib.domain_balanced@v1"),
)

#: The state id a rebuilt B is injected under. Deliberately NOT B's searched
#: content id: `make_retained_state` keeps the id it is given, and giving the
#: rebuild the searched id would make two provenances indistinguishable in the
#: journal. The digests are what prove sameness; the id records how it got here.
REBUILT_STATE_ID = "c2-baseline-rebuilt-b"
REBUILT_PATH_ID = "autoinit.v1.phase_c2.baseline_rebuild"


def path_identity(state: Any) -> tuple[tuple[str, str, str], ...]:
    """A searched state's construction, in the shape `B_PATH` is written in."""
    return tuple((s.kind, s.impl_id, s.profile_id) for s in state.steps)


# --- 1. did the search already produce it? ----------------------------------


def find_searched_baseline(result: Any) -> Any | None:
    """The complete leaf whose construction IS B, or None.

    Looks at every complete leaf rather than at the top-N ranking: a leaf that
    ranked poorly is still measured, still hash-bound, and still the baseline.
    The ranking decides what is *interesting*, not what B is.
    """
    matches = [leaf for leaf in result.leaves
               if leaf.is_complete_leaf() and path_identity(leaf) == B_PATH]
    if not matches:
        return None
    if len(matches) > 1:
        raise BaselineError(
            f"{len(matches)} searched leaves carry B's construction. A kind is "
            "applied at most once per path and a profile choice is part of the "
            "identity, so two states with this path cannot both exist unless "
            "duplicate-identity collapse failed. Refusing to pick one.")
    found = matches[0]
    if found.artifact_digest != B_ARTIFACT_DIGEST:
        raise BaselineError(
            f"a searched leaf has B's exact construction {leaf_label()} but "
            f"digests to {found.artifact_digest[:12]}…, where C1 recorded "
            f"{B_ARTIFACT_DIGEST[:12]}…. The same operators, profiles, seed and "
            "teacher produced different bytes. That is a finding about "
            "determinism or about an operator, not a baseline: comparing "
            "candidates against it would silently compare them against a "
            "different B. Stop and report.")
    return found


def leaf_label() -> str:
    return "->".join(f"{kind}({profile})" for kind, _impl, profile in B_PATH)


# --- 2. the rebuild, and what proves it is the same B ------------------------


def frozen_baseline_spec(*, device: str = "cuda") -> FixedPathSpec:
    """The frozen C1 treatment path, from C1's own constructor.

    Imported rather than re-declared. `build_arm_specs` is what produced the
    hash the preregistration froze, so re-deriving the same steps here would be
    a second construction of the thing whose sameness is the point — and two
    constructions of one path is exactly how they come to differ.

    `FixedPathSpec.as_dict` includes `device`, so the hash is device-dependent
    and `"cuda"` is the value C1 froze. The check is pure arithmetic over
    strings and needs no GPU.
    """
    from experiments.phase_c1.session import build_arm_specs

    return build_arm_specs(workdir_device=device)["treatment"]


def assert_frozen_construction(spec: FixedPathSpec) -> dict[str, Any]:
    """The $0 gate, before a single tensor moves.

    A wrong recipe caught here costs nothing. Caught after the rebuild it costs
    the rebuild, and caught not at all it costs the experiment's meaning.
    """
    evidence = {"path_id": spec.path_id, "path_label": spec.path_label,
                "spec_hash": spec.spec_hash, "expected_spec_hash": B_SPEC_HASH,
                "steps": [s.as_dict() for s in spec.steps]}
    if spec.spec_hash != B_SPEC_HASH:
        raise BaselineError(
            f"the baseline construction hashes to {spec.spec_hash[:12]}…, but "
            f"C1's preregistration froze {B_SPEC_HASH[:12]}…. This is not the "
            "frozen treatment path. Refusing to rebuild a different baseline "
            f"and call it B.\n{json.dumps(evidence, indent=1)}")
    if path_identity_of_spec(spec) != B_PATH:
        raise BaselineError(
            f"the spec hash matches but the ordered construction does not: "
            f"{path_identity_of_spec(spec)} != {B_PATH}. One of the two is "
            "wrong and neither may be trusted.")
    evidence["verified"] = True
    return evidence


def path_identity_of_spec(spec: FixedPathSpec) -> tuple[tuple[str, str, str], ...]:
    return tuple(zip(spec.kinds,
                     (s.impl_id for s in spec.steps),
                     (s.profile_id for s in spec.steps)))


def digest_components(artifact: CheckpointIdentity) -> dict[str, Any]:
    """The five things `artifact_digest` is computed over, each named.

    `artifact_digest` is `sha256_json` over weights / index / config / arch /
    tokenizer, so a mismatch always decomposes. Reporting the aggregate alone
    would leave a reader unable to tell a different initialization from a
    differently-serialized config, and only one of those is a scientific fact.
    """
    return {
        "artifact_digest": artifact.artifact_digest,
        "weights_digest": artifact.weights_digest,
        "config_sha256": artifact.config_sha256,
        "arch_signature": artifact.arch_signature,
        "tokenizer_sha256": artifact.tokenizer_sha256,
        "index_sha256": artifact.index_sha256,
        "single_shard_sha256": artifact.single_shard_sha256,
        "num_parameters": artifact.num_parameters,
    }


def explain_mismatch(artifact: CheckpointIdentity) -> dict[str, Any]:
    """Which component moved, and what that would mean."""
    observed = digest_components(artifact)
    expected = {"artifact_digest": B_ARTIFACT_DIGEST,
                "weights_digest": B_WEIGHTS_DIGEST,
                "config_sha256": B_CONFIG_SHA256,
                "arch_signature": B_ARCH_SIGNATURE,
                "single_shard_sha256": B_SINGLE_SHARD_SHA256,
                "num_parameters": B_NUM_PARAMETERS}
    moved = {k: {"expected": v, "observed": observed.get(k)}
             for k, v in expected.items() if observed.get(k) != v}
    meanings = {
        "weights_digest": ("the initialization itself differs. A scientific "
                           "finding about determinism or about an operator, "
                           "not a serialization detail"),
        "config_sha256": ("the weights may be identical while the model is "
                          "CONFIGURED differently. This is measurable: a "
                          "rope_theta that moves between transformers major "
                          "versions changed a holdout NLL by 0.35 nats in this "
                          "project with nothing raising"),
        "arch_signature": "the geometry differs; this is not the target",
        "num_parameters": "the parameter count differs; this is not the target",
        "single_shard_sha256": ("the shard bytes differ while the aggregate may "
                                "not; check the weights digest first"),
    }
    return {"moved": moved,
            "what_each_would_mean": {k: meanings[k] for k in moved
                                     if k in meanings},
            "c1_runtime": B_RUNTIME,
            "runtime_is_not_a_gate": (
                "a different image is legitimate; it is recorded so that a "
                "config-only difference is readable as a runtime fact"),
            "observed": observed}


# --- 3. the single-shot fallback -------------------------------------------


@dataclass
class BaselineFallback:
    """`conditional_candidates` for `run_phase_a_search`. Single-shot.

    Constructed with everything the rebuild needs and nothing it does not, so
    that a caller cannot accidentally point it at a different teacher or a
    different mixture than the search used.
    """

    adapter: Any
    workdir: Path
    #: The rebuild's OWN allowance, in minutes, and the only clock it runs on.
    #:
    #: It is a NUMBER and not a `Deadline`, deliberately. A `Deadline` starts
    #: counting the moment it is constructed, so accepting one here would make
    #: the rebuild inherit a clock that started before the beam did — and by the
    #: time the rebuild is reached the beam may have spent all of it. The
    #: pricing record separates `beam_composition_risk` from
    #: `baseline_rebuild_reserve` for exactly that reason; a shared clock would
    #: let the beam consume a reserve the accounting says is not its to spend.
    #:
    #: The `Deadline` is built inside `rebuild()`, which is the first moment the
    #: deterministic rule has established that a rebuild is needed at all.
    rebuild_minutes: float
    #: `(minutes, what) -> None`, raising if starting that work would cross the
    #: soft stop. Checked INDEPENDENTLY of the beam's own check, immediately
    #: before materialization: the beam's affordability was decided against a
    #: different envelope and hours earlier.
    afford: Any
    repo_root: Path = REPO_ROOT
    calibration_items: Any = None
    device: str = "cuda"
    say: Any = print
    #: Set once the hook has run. A second invocation is a caller defect, not a
    #: reason to rebuild again.
    outcome: dict[str, Any] | None = field(default=None, init=False)
    #: The clock the rebuild actually ran on, kept for the evidence. `None`
    #: until `rebuild()` creates it, which is what a test asserts.
    rebuild_deadline: Any = field(default=None, init=False)

    def __call__(self, result: Any, root_loader: Any) -> list[dict[str, Any]]:
        """`conditional_candidates`: `(SearchResult, root_loader) -> entries`.

        `root_loader` is the SEARCH's teacher, supplied by the caller that owns
        its lifetime. Taking a copy here would put a second 4B model beside the
        first at the one moment the expensive work is done, and could rebuild
        the baseline from a different revision than the search it is compared
        against.
        """
        if self.outcome is not None:
            raise BaselineError(
                "the baseline fallback was invoked twice. It decides once, "
                "because rebuilding a B the search already produced injects two "
                "states with one artifact digest.")

        found = find_searched_baseline(result)
        if found is not None:
            self.outcome = {
                "resolution": "searched",
                "rebuilt": False,
                "state_id": found.state_id,
                "path_label": found.path_label,
                "artifact_digest": found.artifact_digest,
                "matches_c1_recorded_digest": True,
                "note": ("the beam re-derived the frozen baseline; nothing was "
                         "rebuilt and nothing was injected. The identities "
                         "collide by construction, which is why this is the "
                         "expected branch"),
            }
            self.say(f"  baseline B: SEARCHED — {found.state_id[:12]}… "
                     f"digest {found.artifact_digest[:12]}…, no rebuild")
            return []

        self.say(f"  baseline B: ABSENT from {len(result.leaves)} complete "
                 "leaves — rebuilding once through the frozen C1 treatment path")
        return [self.rebuild(root_loader)]

    # -- the rebuild ------------------------------------------------------
    def rebuild(self, root_loader: Any) -> dict[str, Any]:
        from aadistill.initialization.planning.search import Deadline

        spec = frozen_baseline_spec(device=self.device)
        construction = assert_frozen_construction(spec)
        self.say(f"  frozen construction verified: {spec.spec_hash[:12]}… "
                 f"== {B_SPEC_HASH[:12]}…")

        #: Its own affordability check, against the CURRENT spend rather than
        #: the projection the beam was approved on hours ago. The beam may have
        #: run to its full envelope; whether the rebuild still fits is a
        #: different question with a different answer.
        self.afford(self.rebuild_minutes, "the baseline rebuild")

        #: And its own clock, STARTED HERE. This is the first moment the
        #: deterministic rule has established that B is absent, which is the
        #: earliest point at which the reserve is the rebuild's to spend.
        self.rebuild_deadline = Deadline.from_minutes(self.rebuild_minutes)
        self.say(f"  baseline rebuild allowance: {self.rebuild_minutes:.2f} min, "
                 "clock starts now")

        #: The output digest is PINNED for the rebuild, which C1 could not do:
        #: attempt 18 was this operator's first run and had no prior digest. Now
        #: there is one, so `materialize_fixed_path` gates the final step itself
        #: and a divergence stops inside the executor rather than downstream.
        last = len(spec.steps) - 1
        pinned = spec.replace_tail(
            last,
            FixedPathStep(spec.steps[last].impl_id, spec.steps[last].profile_id,
                          expected_artifact_digest=B_ARTIFACT_DIGEST,
                          label="C2 baseline B, pinned to C1's recorded digest"),
            path_id=REBUILT_PATH_ID)

        out = Path(self.workdir) / "baseline_rebuild"
        out.mkdir(parents=True, exist_ok=True)
        steps = materialize_fixed_path(
            pinned, adapter=self.adapter, root_loader=root_loader,
            workdir=out, repo_root=self.repo_root,
            calibration_items=self.calibration_items,
            deadline=self.rebuild_deadline,
            on_step=lambda r: self.say(
                f"    step {r.index} {r.impl_id}: {r.seconds:.1f}s "
                f"{r.identity.artifact_digest[:12]}…"))

        final = steps[-1]
        directory = Path(final.checkpoint_path)
        #: The executor's own `CheckpointIdentity`, not a second one computed
        #: here. Re-identifying would be a different read of the same directory
        #: and could disagree with the digest the step gate just enforced —
        #: which is the whole failure mode two implementations of one identity
        #: produce.
        artifact = final.identity
        components = digest_components(artifact)
        if artifact.artifact_digest != B_ARTIFACT_DIGEST:
            #: Unreachable while the pin above holds — kept because the pin is a
            #: different code path in a different module, and a baseline the
            #: experiment's meaning rests on should not be protected by exactly
            #: one check in somebody else's file.
            raise BaselineError(
                "the rebuilt baseline is not B.\n"
                + json.dumps(explain_mismatch(artifact), indent=1))

        self.outcome = {
            "resolution": "rebuilt",
            "rebuilt": True,
            "rebuilds": 1,
            "state_id": REBUILT_STATE_ID,
            "construction": construction,
            "pinned_path_id": pinned.path_id,
            "pinned_spec_hash": pinned.spec_hash,
            "_pinned_hash_is_not_the_frozen_hash": (
                "the pinned variant carries a different path_id and an output "
                "pin, so its own spec_hash differs from the frozen "
                f"{B_SPEC_HASH[:12]}… by construction. The frozen hash was "
                "verified before pinning; this one records what executed"),
            "identity": components,
            "identity_matches": {
                "artifact_digest": artifact.artifact_digest == B_ARTIFACT_DIGEST,
                "weights_digest": artifact.weights_digest == B_WEIGHTS_DIGEST,
                "config_sha256": artifact.config_sha256 == B_CONFIG_SHA256,
                "arch_signature": artifact.arch_signature == B_ARCH_SIGNATURE,
                "num_parameters": artifact.num_parameters == B_NUM_PARAMETERS,
            },
            "steps": [s.as_dict() for s in steps],
            "checkpoint_dir": str(directory),
            "allowance": {
                "reserve": "baseline_rebuild_reserve",
                "minutes": self.rebuild_minutes,
                "clock": self.rebuild_deadline.as_dict(),
                "_independent_of_the_beam": (
                    "started when the rebuild began, not when the session did. "
                    "The beam ran on base + beam_composition_risk and could "
                    "not spend any of this."),
            },
        }
        self.say(f"  baseline B: REBUILT and proven — artifact "
                 f"{artifact.artifact_digest[:12]}…, weights "
                 f"{artifact.weights_digest[:12]}…")

        #: The `retained_candidates` entry shape. `expected_artifact_digest`
        #: makes `make_retained_state` refuse it a third time, independently, and
        #: from there it takes the identical canonical-reload / state_eval path
        #: every searched candidate takes.
        return {
            "candidate_id": REBUILT_STATE_ID,
            "checkpoint_dir": str(directory),
            "expected_artifact_digest": B_ARTIFACT_DIGEST,
            "provenance": "c2_baseline_rebuilt_from_frozen_fixed_path",
            "description": (
                "the frozen C1 treatment B, rebuilt from the teacher because "
                "the Search-1 beam did not re-derive it. Proven identical to "
                "C1's recorded arm identity by artifact digest, which covers "
                "weights, config, arch signature, tokenizer and index"),
        }

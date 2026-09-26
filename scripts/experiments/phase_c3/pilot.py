"""The C3 batching-adoption pilot: what it is, and what it must not vary.

The question is NOT whether padded batching reproduces every intermediate
top-k choice. It is whether a B4 causal scorer is materially faster while
producing a downstream model practically equivalent to the B1 reference after
the frozen recovery process.

Two arms, one difference:

    causal-B1    calibration_forward_batch_size = 1
    causal-B4    calibration_forward_batch_size = 4

Everything before `attention.causal_kl_v1` is identical by construction, and
identical means B=1.

**Why the prefix is pinned and not defaulted.** The frozen pre-ATTENTION parent
`eea90c91…` was produced by the historical one-item path.
`materialize_fixed_path` defaults to `DEFAULT_EXECUTION`, which on this branch
is `micro_batch_size = 4`, and B4 activation statistics are now MEASURED to
move FFN and WIDTH structural decisions -- 31 of 36 layers at the parent. So
replaying the prefix under the default would build a different parent. The
artifact-digest gate would catch it, but only after a pod had been paid for,
and a gate is the wrong place to learn which execution mode you used.

`PREFIX_EXECUTION` is therefore a literal, not an alias: an alias would track
the default, which is exactly the coupling the frozen prefix must not have.
"""

from __future__ import annotations

import hashlib

from aadistill.initialization.execution import ExecutionConfig
from aadistill.initialization.planning.fixed_path import FixedPathStep

__all__ = [
    "PREFIX_EXECUTION", "PREFIX_STEPS", "CAUSAL_IMPL_ID",
    "FROZEN_PARENT_DIGEST", "CAUSAL_PROFILE_ID", "BATCH_SIZE_CONFIG_KEY",
    "PILOT_SEED_NAMESPACE", "C0_PREREGISTRATION_SHA256", "CAUSAL_STEP_LABEL",
    "C1_PATH_RECORD", "C1_ARM_IDENTITIES",
    "prefix_steps", "causal_step", "derive_pilot_seed", "PILOT_SEED",
]

#: EXPLICIT, and deliberately not `DEFAULT_EXECUTION`. See the module docstring.
PREFIX_EXECUTION = ExecutionConfig(micro_batch_size=1)

#: The historical C1 pre-ATTENTION prefix: (impl_id, profile_id), in order.
#:
#: THE PROFILES ARE NOT UNIFORM and must not be inferred from ATTENTION's.
#: WIDTH ran against `calib.reasoning_heavy@v2`, not `domain_balanced`. The
#: first version of this module assigned one profile to all three and would
#: have replayed WIDTH against the wrong mixture, reconstructing a different
#: parent -- the digest gate would have refused it, but only after an
#: expensive replay had already been paid for.
#:
#: Source: the committed C1 replay record and arm identities, cross-checked
#: against the C2 baseline-completion protocol. `C1_PATH_RECORD` below is what
#: the test reads, so this constant cannot drift from the frozen path without
#: something turning red.
PREFIX_STEPS = (
    ("depth.causal_kl_greedy_v1", "calib.domain_balanced@v1"),
    ("ffn.activation_importance_v0", "calib.domain_balanced@v1"),
    ("width.global_pca_v0", "calib.reasoning_heavy@v2"),
)

#: The committed record the prefix is checked against. Evidence, not a copy.
C1_PATH_RECORD = "logs/stages/stage-1/phase_c1/runs/attempt9/c1_replay_record.json"
C1_ARM_IDENTITIES = ("logs/stages/stage-1/phase_c1/runs/attempt18/evidence/"
                     "c1_arm_identities.json")

CAUSAL_IMPL_ID = "attention.causal_kl_v1"
#: ATTENTION's own profile. Deliberately a separate constant from the prefix's:
#: sharing one name is what let a single profile be applied to every step.
CAUSAL_PROFILE_ID = "calib.domain_balanced@v1"

#: THE SAME LABEL IN BOTH ARMS. `label` participates in `as_dict()` and so in
#: the step hash, so "causal ATTENTION B1" / "B4" would introduce a SECOND
#: identity difference -- and then the two steps would differ even if the
#: protocol config were dropped. The arm names belong in the pilot record;
#: the only intentional step-identity difference is the batch size.
CAUSAL_STEP_LABEL = "causal ATTENTION"

#: `replay_digests.parent` from the C1 authorization. The prefix replay is
#: gated on reproducing exactly this.
FROZEN_PARENT_DIGEST = (
    "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722")

#: The one config key that separates the two arms. It lives in the causal
#: STEP's config rather than in a repository-wide identity, because exactly one
#: operator has measured output sensitivity to it.
BATCH_SIZE_CONFIG_KEY = "calibration_forward_batch_size"

#: The C0 preregistration digest -- an identity frozen before any candidate.
C0_PREREGISTRATION_SHA256 = (
    "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280")
PILOT_SEED_NAMESPACE = "phase-c3:batch-adoption-pilot:0"


def prefix_steps(*, pin_parent: bool = False) -> list[FixedPathStep]:
    """The three pre-ATTENTION steps, each with ITS OWN profile.

    NONE of them carries a config: the B1/B4 variation begins at ATTENTION,
    and a config on a prefix step would move that boundary and change the
    parent both arms are supposed to share.

    `pin_parent` pins the LAST step -- WIDTH -- to the frozen digest, which is
    where the gate belongs: that step's output IS the pre-ATTENTION parent.
    """
    steps = [FixedPathStep(impl_id=impl, profile_id=profile)
             for impl, profile in PREFIX_STEPS]
    if pin_parent:
        last = steps[-1]
        steps[-1] = FixedPathStep(impl_id=last.impl_id,
                                  profile_id=last.profile_id,
                                  expected_artifact_digest=FROZEN_PARENT_DIGEST,
                                  label="pre-ATTENTION parent")
    return steps


def causal_step(batch_size: int, *, label: str = "") -> FixedPathStep:
    """The one step that binds the numerical protocol into its identity.

    Same `impl_id` in both arms -- the implementation is not what differs. The
    config is what makes `causal-B1` and `causal-B4` distinct states, which
    they must be, because they can materialize different head maps.
    """
    if batch_size < 1:
        raise ValueError(f"calibration forward batch size must be >= 1, "
                         f"got {batch_size}")
    return FixedPathStep(
        impl_id=CAUSAL_IMPL_ID,
        profile_id=CAUSAL_PROFILE_ID,
        label=label or CAUSAL_STEP_LABEL,
        config={BATCH_SIZE_CONFIG_KEY: int(batch_size)})


def derive_pilot_seed(base: str = C0_PREREGISTRATION_SHA256,
                      namespace: str = PILOT_SEED_NAMESPACE) -> int:
    """`uint32_be(SHA256(base + ':' + namespace)[0:4]) mod 2**31`.

    The same rule `derive_recovery_seeds` uses, against an identity frozen
    before any candidate existed, so no discretion is left to exercise.

    This seed is **pilot-only**. It is not a historical A/B/C1/C2 recovery
    seed, and it must not be reused as a formal C3 recovery seed -- no formal
    C3 seed set has been frozen, and this namespace is deliberately distinct so
    that one cannot later collide with it by accident.
    """
    digest = hashlib.sha256(f"{base}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


PILOT_SEED = derive_pilot_seed()

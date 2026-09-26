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
    "PREFIX_EXECUTION", "PREFIX_IMPL_IDS", "CAUSAL_IMPL_ID",
    "FROZEN_PARENT_DIGEST", "CALIBRATION_PROFILE_ID", "BATCH_SIZE_CONFIG_KEY",
    "PILOT_SEED_NAMESPACE", "C0_PREREGISTRATION_SHA256",
    "prefix_steps", "causal_step", "derive_pilot_seed", "PILOT_SEED",
]

#: EXPLICIT, and deliberately not `DEFAULT_EXECUTION`. See the module docstring.
PREFIX_EXECUTION = ExecutionConfig(micro_batch_size=1)

#: The historical C1 pre-ATTENTION prefix, in order.
PREFIX_IMPL_IDS = ("depth.causal_kl_greedy_v1",
                   "ffn.activation_importance_v0",
                   "width.global_pca_v0")

CAUSAL_IMPL_ID = "attention.causal_kl_v1"
CALIBRATION_PROFILE_ID = "calib.domain_balanced@v1"

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


def prefix_steps(*, parent_digest: str | None = None) -> list[FixedPathStep]:
    """The three pre-ATTENTION steps. NONE of them carries a config.

    The B1/B4 variation begins at ATTENTION; a config on a prefix step would
    move that boundary and change the parent both arms are supposed to share.
    """
    steps = [FixedPathStep(impl_id=i, profile_id=CALIBRATION_PROFILE_ID)
             for i in PREFIX_IMPL_IDS]
    if parent_digest is not None:
        last = steps[-1]
        steps[-1] = FixedPathStep(impl_id=last.impl_id,
                                  profile_id=last.profile_id,
                                  expected_artifact_digest=parent_digest,
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
        profile_id=CALIBRATION_PROFILE_ID,
        label=label or f"causal ATTENTION B{batch_size}",
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

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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aadistill.initialization.execution import ExecutionConfig
from aadistill.initialization.planning.fixed_path import (
    FixedPathSpec, FixedPathStep, StepResult, materialize_fixed_path)
from aadistill.initialization.specs.arch import ArchSpec

__all__ = [
    "PREFIX_EXECUTION", "PREFIX_STEPS", "CAUSAL_IMPL_ID",
    "FROZEN_PARENT_DIGEST", "CAUSAL_PROFILE_ID", "BATCH_SIZE_CONFIG_KEY",
    "PILOT_SEED_NAMESPACE", "C0_PREREGISTRATION_SHA256", "CAUSAL_STEP_LABEL",
    "C1_PATH_RECORD", "C1_ARM_IDENTITIES", "root_binding",
    "prefix_steps", "causal_step", "derive_pilot_seed", "PILOT_SEED",
    "target_spec", "arm_spec", "replay_prefix", "ARM_IDS",
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


#: The two arms, named once. The batch size is the arm.
ARM_IDS = {1: "causal-B1", 4: "causal-B4"}


def _path_record(repo_root: str | Path) -> dict:
    import json

    return json.loads((Path(repo_root) / C1_PATH_RECORD).read_text())["path"]


def root_binding(repo_root: str | Path = ".") -> tuple[str, str]:
    """`(root_repo_id, root_revision)`, READ from the committed path record.

    The first version of this module wrote `"main"` here. The frozen path
    pins `768f209d…`, and `main` is a moving reference: the pilot would have
    replayed the prefix from whatever that tag pointed at on the day it ran,
    which is not the checkpoint the frozen parent came from. Nothing about
    the root is retyped now, and a test compares this against the record.
    """
    path = _path_record(repo_root)
    repo_id, revision = path["root_repo_id"], path["root_revision"]
    if len(revision) != 40 or not all(c in "0123456789abcdef" for c in revision):
        raise ValueError(
            f"the committed root revision {revision!r} is not a pinned commit "
            "sha; a moving reference cannot reproduce the frozen parent")
    return repo_id, revision


def target_spec(repo_root: str | Path = ".") -> ArchSpec:
    """The ATTENTION target, read from the committed C1 path record.

    Not restated here. The pilot reuses the exact frozen path, and a target
    typed into this module could drift from the one that path was built for.
    """
    path = _path_record(repo_root)
    return ArchSpec.of(path["family"], path["target_spec"])


def arm_spec(batch_size: int, *, repo_root: str | Path = ".",
             device: str = "cpu", seed: int = 0,
             pin_parent: bool = True,
             max_shard_size: str | int | None = None) -> FixedPathSpec:
    """One arm's complete path: the frozen prefix plus the causal step.

    THE ONLY DIFFERENCE BETWEEN THE TWO ARMS IS `batch_size`. Both are built
    by this function from the same prefix, so "identical by construction" is a
    property of the code rather than a claim about it.
    """
    target = target_spec(repo_root)
    repo_id, revision = root_binding(repo_root)
    steps = tuple(prefix_steps(pin_parent=pin_parent)) + (causal_step(batch_size),)
    return FixedPathSpec(
        path_id=f"autoinit.v1.phase_c3.pilot.{ARM_IDS[int(batch_size)]}",
        family=target.family, target_spec=target, steps=steps,
        root_repo_id=repo_id, root_revision=revision,
        device=device, seed=seed, max_shard_size=max_shard_size)


def replay_prefix(
    spec: FixedPathSpec,
    *,
    adapter: Any,
    root_loader: Callable[[], Any],
    workdir: str | Path,
    repo_root: str | Path = ".",
    calibration_items: Mapping[str, Sequence[Any]] | None = None,
    on_step: Callable[[StepResult], None] | None = None,
    deadline: Any = None,
) -> list[StepResult]:
    """THE ONE PLACE the pilot materializes anything. B1, explicitly.

    Every caller -- the launcher, the driver, the $0 readiness evidence --
    goes through this function, so there is exactly one line in the pilot that
    decides which execution mode the frozen prefix replays under. A test that
    built its own `materialize_fixed_path` call could not detect this function
    forgetting to pass `PREFIX_EXECUTION`; running this one can.

    `execution` is NOT a parameter. Making it one would put the decision back
    in the caller, which is the coupling `PREFIX_EXECUTION` exists to remove.
    """
    return materialize_fixed_path(
        spec, adapter=adapter, root_loader=root_loader, workdir=workdir,
        repo_root=repo_root, calibration_items=calibration_items,
        on_step=on_step, deadline=deadline, execution=PREFIX_EXECUTION)

"""The D1 formal search session: one contract, and every wiring requirement in it.

`d1_design.json :: execution_wiring_required` lists three things the core PERMITS
a caller to omit and that a formal run must not: the frozen suite content hash, the
declared position policy and numerical environment, and a `measurement_protocol_id`
declared BEFORE the first expansion. Every one of them produces a run whose records
are valid-looking and scientifically unusable.

So they are discharged HERE, once, in the function a driver calls — not left to a
driver to remember. `build_session()` returns the evaluator and the `SearchConfig`
already wired, and `assert_session_contract()` checks the result against the design
rather than against this module's intentions.

NOTHING IN THIS FILE IS REUSABLE CORE. The frozen K, the batch size, the packing,
the profiles, the operator set, the policies and the paths are all this experiment's
instance facts. `src/aadistill` knows none of them and must not.

AUTHORIZES NOTHING. A session still needs a one-use authorization issued against
the four money conditions; see `d1_authorization.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]

EXPERIMENT_ID = "phase_d1"
STAGE_ID = "1"
DESIGN_PATH = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
#: The FROZEN state-eval asset. Its `manifest.json` carries the `content_sha256`
#: that must reach `StateEvaluator`; the loader refuses an asset without one.
STATE_EVAL_ROOT = "artifacts/stages/stage-1/state_eval_v1"

POD_WORKSPACE = "/workspace"
STATUS_PATH = f"{POD_WORKSPACE}/autoinit_d1.status"
RUN_LOG_PATH = f"{POD_WORKSPACE}/autoinit_d1_run.log"
DRIVER_JOB_ID = "autoinit_d1"

#: The two arms, by the design's own names. A session runs ONE of them.
TREATMENT_ARM = "supervised_target"
CONTROL_ARM = "all_positions"
ARMS = (TREATMENT_ARM, CONTROL_ARM)


class D1SessionError(RuntimeError):
    """A D1 session premise does not hold. Never a warning."""


def design(repo_root: str | Path = REPO) -> dict[str, Any]:
    path = Path(repo_root) / DESIGN_PATH
    if not path.is_file():
        raise D1SessionError(
            f"no D1 design at {DESIGN_PATH}; a session cannot be built against "
            "a design that does not exist")
    return json.loads(path.read_text())


def design_hash(repo_root: str | Path = REPO) -> str:
    return design(repo_root)["design_hash"]


def open_blockers(repo_root: str | Path = REPO) -> tuple[str, ...]:
    """What the DESIGN says still blocks D1. Read, never restated."""
    return tuple(design(repo_root)["open_blockers"])


def position_policy(arm: str):
    """The arm's scoring-position policy, by object and not by name."""
    from aadistill.initialization.scoring.positions import (
        ALL_POSITIONS_V1, SUPERVISED_TARGET_V1,
    )

    if arm == TREATMENT_ARM:
        return SUPERVISED_TARGET_V1
    if arm == CONTROL_ARM:
        return ALL_POSITIONS_V1
    raise D1SessionError(
        f"unknown D1 arm {arm!r}; this experiment has exactly {ARMS}")


def numerics():
    """The DECLARED numerical environment. A term of the protocol identity.

    Declared rather than probed: the identity must say what the run was MEANT to
    execute under, and a probe would make the identity a property of the machine
    that happened to answer.
    """
    from aadistill.initialization.specs.materialization import NumericalEnvironment

    return NumericalEnvironment(
        device_type="cuda", compute_dtype="bfloat16",
        accumulation_dtype="float32")


def execution():
    """`bsz=3` with length-sorted packing, from the D-series protocol module.

    Read from `scoring_protocol`, not retyped: the batch size and the packing are
    the same decision as the support and must not be able to drift apart from it.
    """
    from aadistill.initialization.execution import ExecutionConfig
    from stages.d_series import scoring_protocol as SP

    return ExecutionConfig(
        micro_batch_size=SP.D_SERIES_MICRO_BATCH_SIZE,
        calibration_batch_packing=SP.D_SERIES_BATCH_PACKING)


def verify_state_eval_bytes(repo_root: str | Path = REPO) -> dict[str, Any]:
    """Do the staged state-eval BYTES match the pins the asset itself ships?

    Not a second definition of any identity: the manifest already records
    `outputs.items.sha256` for its own items file, and this reads that pin and
    checks it. The shared loader deliberately does not -- its docstring says
    what it owes is that `content_sha256` EXISTS and comes from the asset's
    record rather than being recomputed -- so nothing anywhere confirmed that
    the items file beside that record is the one it describes.

    That gap matters because this asset does not travel in the bundle. It is
    untracked in git and is scp'd onto the pod as a declared local asset, and
    `LocalAsset` carries no digest field: a truncated or half-copied
    `items.jsonl` would yield a SMALLER suite while the session still bound the
    full `content_sha256` and every record claimed the frozen suite. A
    measurement bound to a content hash its own prompts do not produce is
    exactly the failure `suite_content_sha256` exists to prevent.

    Checked on the pod at stage A, before the teacher is resident, and again at
    `$0` on the dev box by the launcher's precheck over the bytes it is about
    to send.
    """
    from aadistill.infrastructure.manifest import sha256_file, sha256_json

    root = Path(repo_root) / STATE_EVAL_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise D1SessionError(
            f"the frozen state-eval asset is not staged at {STATE_EVAL_ROOT}. A "
            "formal session may not run against a suite it cannot bind.")
    doc = json.loads(manifest_path.read_text())

    #: THE MANIFEST'S OWN SELF-HASH first, so the pins below are read from a
    #: record that has not been edited since the builder wrote it.
    stated_manifest = doc.get("manifest_sha256")
    if stated_manifest:
        recomputed = sha256_json(
            {k: v for k, v in doc.items() if k != "manifest_sha256"})
        if recomputed != stated_manifest:
            raise D1SessionError(
                f"{STATE_EVAL_ROOT}/manifest.json does not match its own "
                f"manifest_sha256 ({recomputed[:12]} vs {stated_manifest[:12]}); "
                "it has been edited since the builder wrote it")

    pinned = ((doc.get("outputs") or {}).get("items") or {}).get("sha256")
    if not pinned:
        raise D1SessionError(
            f"{STATE_EVAL_ROOT}/manifest.json pins no sha256 for its items "
            "file, so the staged prompts cannot be checked against the record "
            "that describes them")
    items_file = root / "items.jsonl"
    if not items_file.is_file():
        raise D1SessionError(f"{STATE_EVAL_ROOT}/items.jsonl is absent")
    got = sha256_file(items_file)
    if got != pinned:
        raise D1SessionError(
            f"{STATE_EVAL_ROOT}/items.jsonl hashes to {got[:12]} and the "
            f"asset's manifest pins {pinned[:12]}. The staged prompts are not "
            "the frozen suite; a measurement taken on them would be recorded "
            "under a content hash they do not produce.")
    return {"manifest_sha256": stated_manifest,
            "items_sha256": got,
            "content_sha256": doc.get("content_sha256"),
            "n_items": (doc.get("counts") or {}).get("n_items")}


def load_state_eval_suite(repo_root: str | Path = REPO):
    """`(suite, items, content_sha256)` from the frozen asset.

    The content hash comes from the asset's own manifest through the shared
    loader, which refuses an asset that carries none. Recomputing it here would
    be a second definition of the same identity -- but the BYTES beside the
    record are checked against the pin the record carries for them; see
    `verify_state_eval_bytes`.
    """
    import sys

    verify_state_eval_bytes(repo_root)

    sys.path.insert(0, str(Path(repo_root) / "scripts/autoinit"))
    from shared.evaluation.load_state_eval import load as load_suite

    root = Path(repo_root) / STATE_EVAL_ROOT
    if not (root / "manifest.json").is_file():
        raise D1SessionError(
            f"the frozen state-eval asset is not staged at {STATE_EVAL_ROOT}. A "
            "formal session may not run against a suite it cannot bind.")
    suite, items, manifest = load_suite(root)
    content = manifest.get("content_sha256")
    if not content:
        raise D1SessionError(
            f"{STATE_EVAL_ROOT}/manifest.json carries no content_sha256")
    return suite, items, content


#: THE BUNDLE. The name is DERIVED from the session commit, never chosen: an
#: alias must fail at `$0` rather than at `SETUP_RC=1` on a billing pod.
D1_TRANSPORT = None  # built lazily; see `transport()`


def transport():
    """D1's bundle transport. One declaration, so the name has one derivation."""
    global D1_TRANSPORT
    if D1_TRANSPORT is None:
        from aadistill.infrastructure.bundle_transport import TransportSpec
        from shared.deployment import MAIN_RELAY

        D1_TRANSPORT = TransportSpec(
            label="Phase-D1 formal search",
            relay_repo=MAIN_RELAY,
            transfer_prefix="transfer")
    return D1_TRANSPORT


def canonical_bundle_name(session_commit: str) -> str:
    return transport().bundle_name(session_commit)


def canonical_bundle_path(session_commit: str) -> str:
    return transport().repo_path(session_commit)


def require_canonical_bundle(bundle: str, session_commit: str) -> str:
    """An alias fails here, at `$0`, not after setup has been paid for."""
    return transport().require_canonical(bundle, session_commit)


#: THE FROZEN ROOT TEACHER's own record. Owned by C1's plan, used by the whole
#: search lineage, and NOT re-typed here: a second copy of a teacher identity is a
#: second thing that can disagree about which bytes the lineage started from.
TEACHER_BINDING = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"


def root_teacher_identity(repo_root: str | Path = REPO) -> dict[str, Any]:
    """`(id, sha256)` for the root, from the frozen binding. Never the arm.

    `root_materialization_id()` REFUSES an empty id or hash -- "an unpinned root is
    a lineage that starts from nothing in particular, which every child would then
    inherit" -- and the first version of this session passed `f"d1/{arm}"` with an
    empty hash. Both halves were wrong: the hash was absent, and the ID described
    the SCORING ARM. The arm belongs to the scoring/search protocol identity, which
    already carries it through `position_policy` and `config_hash`; the root
    identity describes the teacher.

    The single hash is derived from the binding's per-shard digests in shard order,
    so it moves if any shard's content moves and is stable otherwise.
    """
    import hashlib

    path = Path(repo_root) / TEACHER_BINDING
    if not path.is_file():
        raise D1SessionError(
            f"no teacher binding at {TEACHER_BINDING}; a formal session may not "
            "start from a root it cannot pin")
    doc = json.loads(path.read_text())
    shards = list(doc["weight_shards"])
    digests = doc["expected_shard_sha256"]
    missing = [s for s in shards if not digests.get(s)]
    if missing:
        raise D1SessionError(
            f"the teacher binding names shards with no sha256: {missing}")
    combined = hashlib.sha256(
        "".join(f"{s}:{digests[s]}\n" for s in shards).encode()).hexdigest()
    return {
        "root_teacher_id": f"{doc['repo_id']}@{doc['revision']}",
        "root_teacher_sha256": combined,
        "repo_id": doc["repo_id"],
        "revision": doc["revision"],
        "n_shards": int(doc["n_shards"]),
        "shard_sha256": {s: digests[s] for s in shards},
        "_owner": TEACHER_BINDING,
        "_combined_over": ("the per-shard sha256 digests in shard order, so the "
                           "root hash moves if any shard's content moves"),
    }


def verify_staged_teacher(path: str | Path,
                          repo_root: str | Path = REPO) -> dict[str, Any]:
    """Check the teacher ON DISK against the frozen binding, before expansion.

    A root identity that is merely DECLARED binds nothing: the lineage would
    record the frozen revision while the search read whatever bytes were staged.
    Every named shard is hashed and compared.
    """
    from aadistill.infrastructure.manifest import sha256_file

    identity = root_teacher_identity(repo_root)
    staged = Path(path)
    checked, problems = {}, []
    for shard, expected in identity["shard_sha256"].items():
        f = staged / shard
        if not f.is_file():
            problems.append(f"{shard} is absent from {staged}")
            continue
        got = sha256_file(f)
        checked[shard] = got
        if got != expected:
            problems.append(
                f"{shard} hashes to {got[:12]} and the binding records "
                f"{expected[:12]}")
    if problems:
        raise D1SessionError(
            "the staged teacher is not the frozen root:\n  - "
            + "\n  - ".join(problems))
    return {"verified": sorted(checked), "root_teacher_id":
            identity["root_teacher_id"],
            "root_teacher_sha256": identity["root_teacher_sha256"]}


@dataclass(frozen=True)
class D1Session:
    """A wired D1 search session. Built by `build_session`, never by hand."""

    arm: str
    evaluator: Any
    config: Any
    suite_content_sha256: str
    design_hash: str

    def as_record(self) -> dict[str, Any]:
        """What a run's evidence states about how it was wired."""
        from aadistill.initialization.scoring.positions import policy_config

        return {
            "experiment_id": EXPERIMENT_ID,
            "stage_id": STAGE_ID,
            "arm": self.arm,
            "design_hash": self.design_hash,
            "suite_content_sha256": self.suite_content_sha256,
            "measurement_protocol_id": self.config.measurement_protocol_id,
            "config_hash": self.config.config_hash,
            "position_policy": policy_config(self.config.position_policy)
                               or {"position_policy": "incumbent"},
            "distribution_support": self.config.distribution_support.as_dict(),
            "execution": {
                "micro_batch_size": self.config.device and
                                    self.evaluator.execution.micro_batch_size,
                "calibration_batch_packing":
                    self.evaluator.execution.calibration_batch_packing,
            },
            "root_teacher": root_teacher_identity(),
            "beam": {"width": self.config.schedule.width,
                     "warmup_levels": self.config.schedule.warmup_levels},
            "profiles": [p.qualified_id for p in self.config.profiles],
            "allowed_impls": list(self.config.allowed_impls or ()),
            "seed": self.config.seed,
        }


def teacher_vocab_size(repo_root: str | Path = REPO) -> int:
    """The vocabulary, from the frozen A3 path spec's TARGET geometry.

    `CACHE_IN_MEMORY` checks its budget BEFORE allocating and refuses without
    this -- a budget checked afterwards is an OOM. Read from the spec rather than
    typed: a vocabulary literal in an experiment module is one more place for the
    model's shape to disagree with itself.

    The TARGET's, because no frozen operator modifies `vocab_size` -- the teacher
    and every child share it -- and the spec carries the target, not the parent.
    """
    from stages.phase_a3 import a3_session as A3S

    spec = A3S.path_spec(workdir_device="cpu")
    vocab = int(spec.target_spec["vocab_size"])
    if vocab < 2:
        raise D1SessionError(f"implausible teacher vocabulary {vocab}")
    return vocab


def build_session(*, arm: str, workdir: Path, run_id: str,
                  device: str = "cuda",
                  repo_root: str | Path = REPO,
                  reference_strategy: Any = None) -> D1Session:
    """The ONE place a D1 search session is wired. Every requirement, discharged.

    The three the design names, and where each is satisfied:

    * the frozen suite's `content_sha256` reaches
      `StateEvaluator(suite_content_sha256=...)`, so the protocol identity
      distinguishes two suites with the same structure and different items;
    * the evaluator is constructed with the ARM's position policy and the declared
      `NumericalEnvironment`, both terms of `measurement_protocol_id`;
    * `SearchConfig.measurement_protocol_id` is SET from the evaluator before the
      search is constructed, so the config hash carries the protocol from the
      first expansion and a resume can tell what it is adopting.

    And one the design did not have to name because it did not exist yet: the
    D-series distribution support reaches `SearchConfig.distribution_support`, from
    which `_expand_one` declares it and hands the object to every operator.
    """
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1
    from aadistill.initialization.planning.search import SearchConfig
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from stages.phase_a3 import a3_session as A3S
    from stages.d_series import scoring_protocol as SP

    if arm not in ARMS:
        raise D1SessionError(f"unknown D1 arm {arm!r}; expected one of {ARMS}")
    doc = design(repo_root)
    suite, items, content = load_state_eval_suite(repo_root)
    policy = position_policy(arm)
    env = numerics()
    exec_cfg = execution()

    evaluator = StateEvaluator(
        suite, items, device=device,
        vocab_size=teacher_vocab_size(repo_root),
        suite_content_sha256=content,
        position_policy=policy,
        distribution_support=SP.D_SERIES_SUPPORT,
        execution=exec_cfg,
        numerics=env,
        #: RECOMPUTE, EXPLICITLY, because it is the strategy every real search in
        #: this repository recorded. `CACHE_IN_MEMORY` was never usable at the real
        #: vocabulary -- the frozen suite's reference logits are 41.9 GiB against a
        #: 2 GiB budget -- and the evaluator refuses it at construction, as its own
        #: comment says. The Top-K win is NOT this cache: it is the O(T*K) SKETCH
        #: cache inside the recompute path, which is what took the reference state
        #: from 16.91 GiB to 137.5 MiB.
        reference_strategy=(reference_strategy or ReferenceStrategy.RECOMPUTE))

    #: THE TARGET AND THE SPACE come from the frozen A3 path spec, which is what
    #: `search_space` prices and what the design enumerates. Re-deriving the
    #: target here would be a second definition of the experiment's own shape.
    spec = A3S.path_spec(workdir_device=device)
    config = SearchConfig(
        run_id=run_id,
        target_spec=spec.target_spec,
        schedule=SCHEDULE_V1,
        seed=spec.seed,
        workdir=Path(workdir),
        profiles=tuple(_profiles(doc)),
        policy=PARETO_V1,
        suite=suite,
        position_policy=policy,
        #: The D-series partition, so `_expand_one` declares it and hands the
        #: object to every operator. Without this the search would fall back to
        #: the full vocabulary and measure a different estimand.
        distribution_support=SP.D_SERIES_SUPPORT,
        #: DECLARED, before any expansion. The search refuses a declared id that
        #: disagrees with its measurer at construction, which is the point.
        measurement_protocol_id=evaluator.measurement_protocol_id,
        allowed_impls=tuple(sorted(_frozen_impl_ids())),
        device=device,
    )
    return D1Session(arm=arm, evaluator=evaluator, config=config,
                     suite_content_sha256=content,
                     design_hash=doc["design_hash"])


def _profiles(doc: dict[str, Any]):
    """Both calibration profiles the design declares, as objects."""
    from aadistill.initialization.calibration.profiles import get_profile
    from shared.calibration import register_builtin_profiles

    register_builtin_profiles()
    declared = (doc["search_stage"].get("profiles")
                or doc["search_stage"].get("coverage", {}).get("profiles"))
    ids = declared if isinstance(declared, list) else sorted(declared or ())
    ids = [str(p) for p in ids]
    if not ids:
        raise D1SessionError(
            "the design declares no calibration profiles; a session must not "
            "invent its own mixture")
    return [get_profile(p) for p in sorted(ids)]


def _frozen_impl_ids() -> tuple[str, ...]:
    """D1's frozen operator set, from the space the design prices."""
    from stages.phase_d1 import search_space as d1

    _register_frozen_operators()
    #: KIND -> impl_id, so the VALUES are the implementations. Iterating the
    #: mapping yields the kinds, which is a set of four strings that look
    #: plausible and resolve to nothing.
    return tuple(d1.size_report()["frozen_implementations"].values())


def _register_frozen_operators() -> None:
    """Every registry a D1 session needs, in the order the search expects."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    from shared.calibration import register_builtin_profiles
    from stages.phase_c2.search_space import register_c2_operators

    register_builtin_adapters()
    register_builtin_profiles()
    register_builtin_operators()
    register_c2_operators()
    #: Registered by its own module -- the frozen-set escape route its docstring
    #: describes -- so a session must ask for it explicitly.
    activation_importance.register()


def assert_session_contract(session: D1Session,
                            repo_root: str | Path = REPO) -> dict[str, Any]:
    """Check a built session against the DESIGN. Raises, never warns.

    Called by the driver before the first expansion, so a miswired session costs
    nothing rather than producing unusable records at $1.09/h. It checks against
    the design document rather than against this module's constants, because
    agreeing with itself is not a check.
    """
    from aadistill.initialization.scoring.positions import policy_config
    from stages.d_series import scoring_protocol as SP

    doc = design(repo_root)
    cfg, ev = session.config, session.evaluator
    problems: list[str] = []

    if not cfg.measurement_protocol_id:
        problems.append(
            "SearchConfig.measurement_protocol_id is unset, so the config hash "
            "will not carry the protocol until after work has been done under it")
    if cfg.measurement_protocol_id != ev.measurement_protocol_id:
        problems.append(
            f"the search declares protocol {str(cfg.measurement_protocol_id)[:12]} "
            f"and the evaluator reports {str(ev.measurement_protocol_id)[:12]}")
    if getattr(ev, "suite_content_sha256", None) != session.suite_content_sha256:
        problems.append(
            "the evaluator did not bind the frozen suite's content hash, so its "
            "protocol id cannot tell two suites with the same structure apart")
    if session.suite_content_sha256 in (None, "", "unbound", "UNBOUND_SUITE_CONTENT"):
        problems.append("the suite content identity is unbound")

    want_support = SP.D_SERIES_SUPPORT
    if cfg.distribution_support.as_dict() != want_support.as_dict():
        problems.append(
            f"the search reduces over {cfg.distribution_support} and D1 runs "
            f"{want_support}")
    if ev.distribution_support.as_dict() != want_support.as_dict():
        problems.append(
            f"the evaluator reduces over {ev.distribution_support} and D1 runs "
            f"{want_support}")

    want_policy = position_policy(session.arm)
    if cfg.position_policy.policy_hash != want_policy.policy_hash:
        problems.append(
            f"the search scores {cfg.position_policy.qualified_id} and the "
            f"{session.arm} arm is {want_policy.qualified_id}")
    if ev.position_policy.policy_hash != want_policy.policy_hash:
        problems.append(
            f"the evaluator scores {ev.position_policy.qualified_id} and the "
            f"{session.arm} arm is {want_policy.qualified_id}")
    declared_hashes = {
        a: doc["scoring_policy"][a]["policy_hash"]
        for a in ("treatment", "control") if a in doc["scoring_policy"]}
    if want_policy.policy_hash not in declared_hashes.values():
        problems.append(
            f"{want_policy.qualified_id} is not a policy the design declares "
            f"({sorted(declared_hashes)})")

    proto = doc["execution_protocol"]
    if ev.execution.micro_batch_size != proto["micro_batch_size"]:
        problems.append(
            f"the evaluator batches {ev.execution.micro_batch_size} and the "
            f"design declares {proto['micro_batch_size']}")
    if ev.execution.calibration_batch_packing != proto[
            "calibration_batch_packing"]:
        problems.append(
            f"the evaluator packs {ev.execution.calibration_batch_packing!r} and "
            f"the design declares {proto['calibration_batch_packing']!r}")

    beam = doc["search_stage"]["schedule"]
    if cfg.schedule.width != beam["width"]:
        problems.append(f"beam width {cfg.schedule.width} != {beam['width']}")
    if cfg.schedule.warmup_levels != beam["warmup_levels"]:
        problems.append(
            f"warmup levels {cfg.schedule.warmup_levels} != "
            f"{beam['warmup_levels']}")

    frozen = set(doc["search_stage"]["frozen_implementations"].values())
    if set(cfg.allowed_impls or ()) != frozen:
        problems.append(
            f"the space is {sorted(cfg.allowed_impls or ())} and the design "
            f"freezes {sorted(frozen)}")
    declared_profiles = {p.qualified_id for p in cfg.profiles}
    if declared_profiles != set(doc["search_stage"]["profiles"]):
        problems.append(
            f"the profiles are {sorted(declared_profiles)} and the design "
            f"declares {sorted(doc['search_stage']['profiles'])}")

    if session.design_hash != doc["design_hash"]:
        problems.append(
            "the session was built against a different design revision than the "
            "one on disk")

    if problems:
        raise D1SessionError(
            "the D1 session does not implement the frozen design:\n  - "
            + "\n  - ".join(problems))
    return {
        "checked": [
            "measurement_protocol_id declared and equal to the evaluator's",
            "frozen suite content hash bound into the evaluator",
            "distribution support on the search AND the evaluator",
            "arm position policy on both, and declared by the design",
            "micro_batch_size and packing",
            "beam width and warmup levels",
            "frozen operator set and both calibration profiles",
            "design revision",
        ],
        "arm": session.arm,
        "design_hash": session.design_hash,
        "measurement_protocol_id": cfg.measurement_protocol_id,
        "config_hash": cfg.config_hash,
        "suite_content_sha256": session.suite_content_sha256,
        "position_policy_hash": want_policy.policy_hash,
        "distribution_support": cfg.distribution_support.as_dict(),
        "_authorizes": ("nothing. A wired session is not a funded one; the "
                        "one-use authorization is issued separately."),
    }

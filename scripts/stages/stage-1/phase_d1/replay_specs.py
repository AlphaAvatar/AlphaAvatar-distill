"""Rebuild two of D1's quality-order finalists as digest-pinned fixed paths.

The completed search `d1_search_20261006_210210` retained checkpoint bytes for
the four leaves its own Stage-D policy committed -- quality positions 1, 3, 5 and
11 under lineage rotation. The standing retention policy of 2026-10-07 then made
the behavioural finalists the quality-order Top-4, which are positions 1 to 4. Two
of those, at positions 2 and 4, were never retained: the search measured them,
ranked them and then discarded their weights with the pod.

So this module turns that frozen record back into something executable. It is a
REPLAY, not a search. Nothing here ranks, evaluates or selects, and the committed
ranking is not recomputed. Every step of every path is pinned to the artifact
digest the search recorded for it, so the only two outcomes are "byte-identical
to the search" and "STOP".

Three properties the reconstruction depends on, and all three are the same ones
the C2 replay established:

* **Every intermediate is pinned, not just the leaf.** A path that agreed only
  at its final digest would leave three unverified operators behind it, and a
  compensating pair of errors would be indistinguishable from a correct replay.
  `FixedPathStep.expected_artifact_digest` is set on all four steps.
* **The ancestry comes from the journal, not from the path label.** The label
  says which operator and profile ran at each position; it does not say what that
  position produced. Each step's expected digest is read from the recorded state
  that the search actually created there, walked by `parent_id`.
* **The source is the search's own evidence**, recorded here as data rather than
  as a claim in a message.

The journal carries 172 rows over 92 distinct states: the search appends a row
when a state is created and again when it is evaluated. Deduplication is by
`state_id`, and the duplicates must agree on identity -- if two rows for one
state disagree, the journal is not a record and this module refuses.

WHAT THIS MODULE DOES NOT DECIDE: which candidates are finalists. That is the
maintainer's retention policy, recorded in
`logs/stages/stage-1/phase_d1/decisions/post_search_finalist_retention.json`, and
read from there rather than recomputed, so a replay cannot quietly re-select.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]

#: THE SOURCE RUN. Data, not a claim in a review message.
SOURCE_RUN_ID = "d1_search_20261006_210210"
SOURCE_SESSION_COMMIT = "77aedc90f93f7f8b69e62bdf03de4dcc22ca273f"
RUN_REL = f"logs/stages/stage-1/phase_d1/runs/{SOURCE_RUN_ID}"
SELECTION_REL = f"{RUN_REL}/evidence/stage1_selection.json"
RANKING_REL = f"{RUN_REL}/evidence/complete_leaf_ranking.json"
DECISION_REL = ("logs/stages/stage-1/phase_d1/decisions/"
                "post_search_finalist_retention.json")

#: The full state journal is 67 MB and lives OUT OF TREE with the run's other
#: heavy bytes; only the 12 complete leaves are committed. Ancestry needs every
#: intermediate, so the journal is read from the durable store and its location
#: is a parameter rather than a constant -- a replay prepared on another host
#: must be able to say where it read the journal from.
DEFAULT_JOURNAL = (f"/home/ecs-user/aad-scratch/{SOURCE_RUN_ID}"
                   "/oob_products/states.jsonl")

#: Fields two rows for the same state must agree on.
_IDENTITY_KEYS = ("artifact_digest", "checkpoint_sha256", "impl_ids",
                  "path_label", "parent_id", "target_spec_hash")


class ReplaySourceError(RuntimeError):
    """The search evidence cannot support a digest-pinned replay."""


@dataclass(frozen=True)
class ReplayStep:
    """One operator application, with the digest the search recorded for it."""

    index: int
    kind: str
    impl_id: str
    profile_id: str
    seed: int
    state_id: str
    expected_artifact_digest: str
    expected_single_shard_sha256: str
    #: The recorded identity of the checkpoint's `config.json`, and the geometry
    #: it describes. Carried because together they make the ROOT STATE checkable
    #: without a GPU: a child's config is built from its parent's `to_dict()`
    #: plus the spec, so a step-0 config sha is a fingerprint of the root the
    #: search expanded from. See `derive_root_state`.
    expected_config_sha256: str = ""
    arch_spec: tuple[tuple[str, Any], ...] = ()
    #: The `config_hash` the search recorded for this step -- the identity of
    #: the OPERATOR INPUTS, as distinct from `expected_config_sha256`, which is
    #: the identity of the child model's config.json. Two paid subruns were
    #: spent on a replay whose inputs hashed to sha256 of `{}` while the
    #: search's hashed to 464cb782ea8095, and the only symptom was a digest
    #: mismatch reported after the operator had run.
    expected_config_hash: str = ""
    #: The EXECUTION knobs the search recorded for this step. Deliberately not
    #: part of any hash -- they are runtime choices -- which is exactly why
    #: they have to be carried and checked separately: `config_hash` cannot
    #: see them, and the FFN operator's top-k flips when the reduction order
    #: does. Read from the step's own trace.
    expected_micro_batch_size: int = 0
    expected_batch_packing: str = ""


@dataclass(frozen=True)
class ReplayLeaf:
    """One finalist to reconstruct, and every identity adoption requires."""

    state_id: str
    quality_position: int
    path_label: str
    steps: tuple[ReplayStep, ...]
    root_teacher_id: str
    root_teacher_sha256: str
    target_spec_hash: str
    seed: int
    num_parameters: int
    expected_artifact_digest: str
    expected_weights_digest: str
    expected_single_shard_sha256: str
    expected_arch_signature: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "quality_position": self.quality_position,
            "path_label": self.path_label,
            "root_teacher_id": self.root_teacher_id,
            "root_teacher_sha256": self.root_teacher_sha256,
            "target_spec_hash": self.target_spec_hash,
            "seed": self.seed,
            "num_parameters": self.num_parameters,
            "expected": {
                "artifact_digest": self.expected_artifact_digest,
                "weights_digest": self.expected_weights_digest,
                "single_shard_sha256": self.expected_single_shard_sha256,
                "arch_signature": self.expected_arch_signature,
            },
            "steps": [
                {"index": s.index, "kind": s.kind, "impl_id": s.impl_id,
                 "profile_id": s.profile_id, "seed": s.seed,
                 "state_id": s.state_id,
                 "expected_artifact_digest": s.expected_artifact_digest,
                 "expected_config_sha256": s.expected_config_sha256,
                 "expected_config_hash": s.expected_config_hash,
                 "expected_micro_batch_size": s.expected_micro_batch_size,
                 "expected_batch_packing": s.expected_batch_packing,
                 "arch_spec": dict(s.arch_spec)}
                for s in self.steps
            ],
        }


def load_states(journal: str | Path = DEFAULT_JOURNAL
                ) -> dict[str, dict[str, Any]]:
    """Every state the search recorded, deduplicated by id and cross-checked.

    Refuses on a disagreement rather than preferring a row: two rows for one
    state that differ on identity mean the journal is not a record of what ran,
    and a replay pinned to one of them would be pinned to a guess.
    """
    path = Path(journal)
    if not path.is_file():
        raise ReplaySourceError(
            f"the state journal is not at {path}. It is 67 MB and lives with "
            "the run's other heavy bytes out of tree; pass --journal to say "
            "where this host keeps it.")
    out: dict[str, dict[str, Any]] = {}
    rows = bad = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        sid = row.get("state_id")
        if not sid:
            raise ReplaySourceError(f"a journal row carries no state_id: {line[:120]}")
        prior = out.get(sid)
        if prior is None:
            out[sid] = row
            continue
        disagree = [k for k in _IDENTITY_KEYS if prior.get(k) != row.get(k)]
        if disagree:
            raise ReplaySourceError(
                f"two journal rows for {sid} disagree on {disagree}; the journal "
                "is not a record and cannot pin a replay")
        #: The later row is the evaluated one and is a superset.
        out[sid] = {**prior, **row}
    if bad:
        raise ReplaySourceError(
            f"{bad} of {rows} journal rows are unparseable; a truncated journal "
            "cannot be shown to contain the ancestry a replay needs")
    return out


def frozen_finalists(repo_root: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    """The finalist set, READ from the maintainer's decision record.

    Not recomputed from the ranking. A replay that re-derived its own candidate
    set could quietly select a different one, which is exactly the authority a
    reconstruction must not have.
    """
    doc = json.loads((Path(repo_root) / DECISION_REL).read_text())
    members = doc["the_frozen_behavioural_finalists"]["members"]
    if doc["the_frozen_behavioural_finalists"]["rule_applied"] != "quality_only":
        raise ReplaySourceError(
            "the decision record does not declare quality-only retention")
    return list(members)


def to_rematerialize(repo_root: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    """The finalists whose weights the search did not retain."""
    return [m for m in frozen_finalists(repo_root)
            if str(m.get("availability", "")).startswith("NOT RETAINED")]


def resolve_ancestry(states: Mapping[str, Mapping[str, Any]],
                     leaf_id: str) -> tuple[dict[str, Any], ...]:
    """The chain root -> leaf, by `parent_id`, every link present.

    Walked rather than inferred from the path label: the label names the operator
    and profile at each position and says nothing about what that position
    produced, and the expected digest for step *i* is a property of the state the
    search created there.
    """
    chain: list[dict[str, Any]] = []
    sid: str | None = leaf_id
    while sid:
        row = states.get(sid)
        if row is None:
            #: THE TEACHER ROOT IS NOT JOURNALLED. The search records
            #: expansions, so the root -- which applied no operator -- has no
            #: row of its own, and the only legitimately absent parent is that
            #: one. Distinguished from a missing INTERMEDIATE by the step count
            #: of the child that names it: a state with exactly one operator
            #: step descends from the root, and anything deeper that names an
            #: absent parent means the chain really is broken.
            child = chain[-1] if chain else None
            depth = len(child.get("steps") or []) if child else -1
            if depth == 1:
                break
            raise ReplaySourceError(
                f"ancestry of {leaf_id} names {sid}, which the journal does not "
                f"contain, and the child naming it carries {depth} operator "
                "steps rather than 1 -- so this is a missing intermediate and "
                "not the unjournalled teacher root. The chain cannot be pinned.")
        chain.append(dict(row))
        sid = row.get("parent_id")
    chain.reverse()
    #: Every link carries exactly one more step than the one before it, and the
    #: first carries one. Checked rather than assumed: a chain assembled from a
    #: journal with a gap could otherwise be one operator short and still look
    #: like a path.
    for position, row in enumerate(chain, start=1):
        depth = len(row.get("steps") or [])
        if depth != position:
            raise ReplaySourceError(
                f"{leaf_id} ancestry position {position} carries {depth} "
                "operator steps; the chain is not contiguous")
    return tuple(chain)


def build_leaf(states: Mapping[str, Mapping[str, Any]],
               member: Mapping[str, Any]) -> ReplayLeaf:
    """One finalist as a fully pinned replay, or an error naming what is missing."""
    leaf_id = member["state_id"]
    leaf = states.get(leaf_id)
    if leaf is None:
        raise ReplaySourceError(f"{leaf_id} is not in the journal")
    chain = resolve_ancestry(states, leaf_id)
    declared = list(leaf.get("impl_ids") or [])
    steps: list[ReplayStep] = []
    for position, row in enumerate(chain):
        rec = (row.get("steps") or [])
        if not rec:
            raise ReplaySourceError(
                f"{leaf_id} ancestry position {position} records no operator step")
        step = rec[-1]
        digest = row.get("artifact_digest")
        shard = row.get("checkpoint_sha256")
        if not digest or not shard:
            raise ReplaySourceError(
                f"{leaf_id} ancestry position {position} ({step.get('impl_id')}) "
                "records no artifact digest; an unpinned intermediate would let "
                "a compensating pair of errors pass as a correct replay")
        config_sha = (row.get("artifact") or {}).get("config_sha256")
        if not config_sha:
            raise ReplaySourceError(
                f"{leaf_id} ancestry position {position} records no "
                "artifact.config_sha256, so the root state this path expanded "
                "from cannot be checked without a GPU")
        steps.append(ReplayStep(
            index=position, kind=str(step.get("kind")),
            impl_id=str(step.get("impl_id")),
            profile_id=str(step.get("profile_id")),
            seed=int(step.get("seed")),
            state_id=str(row["state_id"]),
            expected_artifact_digest=str(digest),
            expected_single_shard_sha256=str(shard),
            expected_config_sha256=str(config_sha),
            expected_config_hash=str(step.get("config_hash") or ""),
            expected_micro_batch_size=int(
                (step.get("trace") or {}).get("micro_batch_size") or 0),
            expected_batch_packing=str(
                (step.get("trace") or {}).get("calibration_batch_packing") or ""),
            arch_spec=tuple(sorted((str(k), v) for k, v in
                                   (row.get("arch_spec") or {}).items()))))
    got = [s.impl_id for s in steps]
    if got != declared:
        raise ReplaySourceError(
            f"{leaf_id}: ancestry yields operators {got} and the state declares "
            f"{declared}; the chain is not this leaf's")
    art = leaf.get("artifact") or {}
    for key, value in (("artifact_digest", member["artifact_digest"]),
                       ("weights_digest", member["weights_digest"]),
                       ("single_shard_sha256", member["single_shard_sha256"]),
                       ("arch_signature", member["arch_signature"])):
        mine = art.get(key) if key != "artifact_digest" else leaf.get(key)
        if mine != value:
            raise ReplaySourceError(
                f"{leaf_id}: the decision record's {key} is {value} and the "
                f"search journal records {mine}. The replay target must be the "
                "leaf the search measured.")
    return ReplayLeaf(
        state_id=leaf_id,
        quality_position=int(member["quality_position"]),
        path_label=str(leaf["path_label"]),
        steps=tuple(steps),
        root_teacher_id=str(leaf["root_teacher_id"]),
        root_teacher_sha256=str(leaf["root_teacher_sha256"]),
        target_spec_hash=str(leaf["target_spec_hash"]),
        seed=int(leaf["seed"]),
        num_parameters=int(leaf["num_parameters"]),
        expected_artifact_digest=str(leaf["artifact_digest"]),
        expected_weights_digest=str(art["weights_digest"]),
        expected_single_shard_sha256=str(art["single_shard_sha256"]),
        expected_arch_signature=str(art["arch_signature"]))


def replay_plan(repo_root: str | Path = REPO_ROOT,
                journal: str | Path = DEFAULT_JOURNAL) -> list[ReplayLeaf]:
    """Every finalist that needs rematerializing, fully pinned. Pure."""
    states = load_states(journal)
    leaves = [build_leaf(states, m) for m in to_rematerialize(repo_root)]
    if not leaves:
        raise ReplaySourceError(
            "no finalist is marked NOT RETAINED; there is nothing to replay")
    return leaves


def worst_seconds_by_impl(journal_dir: str | Path | None = None
                          ) -> dict[str, float]:
    """Per-operator worst observed `operator_seconds` from the source run.

    Used to price the session. The search's own telemetry, not an estimate: the
    DEPTH operator ran well over its planning cell and a replay re-executes it.
    """
    base = Path(journal_dir) if journal_dir else Path(DEFAULT_JOURNAL).parent
    path = base.parent / "store" / "d1_search_artifacts.tar.gz"
    telemetry = base / "telemetry.jsonl"
    rows: list[dict[str, Any]] = []
    if telemetry.is_file():
        for line in telemetry.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    if not rows:
        raise ReplaySourceError(
            f"no telemetry to price from; looked at {telemetry} (and the run's "
            f"archive at {path})")
    worst: dict[str, float] = {}
    for row in rows:
        if row.get("event") != "expansion":
            continue
        impl = str(row.get("impl_id"))
        worst[impl] = max(worst.get(impl, 0.0),
                          float(row.get("operator_seconds") or 0.0))
    return worst


def describe(leaves: Sequence[ReplayLeaf],
             root_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole plan as a record, for a $0 preflight and for the session.

    `root_state` is `derive_root_state`'s output. It travels with the plan so the
    pod does not re-derive it from a 67 MB journal it does not have -- but the
    pod still RE-CHECKS it, against the `expected_config_sha256` the plan's own
    steps carry, so the plan is a shipped derivation rather than a shipped
    assertion.
    """
    return {
        "schema": "aadistill.phase_d1.replay_plan/v1",
        "_what_this_is": (
            "a digest-pinned reconstruction of finalists the completed search "
            "measured and did not retain. It is NOT a search, NOT a new "
            "measurement and NOT a re-selection: the candidate set is read from "
            "the maintainer's retention decision and every step is pinned to "
            "the digest the search recorded."),
        "source_run_id": SOURCE_RUN_ID,
        "source_session_commit": SOURCE_SESSION_COMMIT,
        "n_leaves": len(leaves),
        "adoption_requires": [
            "artifact_digest", "weights_digest", "single_shard_sha256",
            "arch_signature",
        ],
        "_no_approximate_equivalence": (
            "all four must match exactly. A mismatch after a completed "
            "reconstruction is a deterministic-materialization defect to "
            "diagnose, not a reason to substitute a near-equivalent checkpoint "
            "or to weaken an identity check."),
        "root_state": dict(root_state) if root_state else None,
        "leaves": [leaf.as_dict() for leaf in leaves],
    }


def root_state_from_plan(doc: Mapping[str, Any],
                         leaves: Sequence[ReplayLeaf], *,
                         base_config: Any,
                         family: str = "qwen3") -> dict[str, Any]:
    """The plan's root state, RE-DERIVED here and required to agree.

    The pod does not take the plan's word for it. `derive_root_state` runs again
    against the step-0 config identities the plan carries, and the result must
    name the same candidate the plan shipped. A plan written against a different
    tree, or hand-edited, fails here for the price of two config hashes instead
    of a GPU minute per path.
    """
    here = derive_root_state(leaves, base_config=base_config, family=family)
    shipped = doc.get("root_state") or {}
    if shipped and shipped.get("chosen_candidate") != here["chosen_candidate"]:
        raise RootStateUndetermined(
            f"the plan ships root state {shipped.get('chosen_candidate')!r} and "
            f"the recorded step-0 configs derive {here['chosen_candidate']!r}")
    here["agreed_with_plan"] = bool(shipped)
    return here


#: The pod's GPU. PINNED, not chosen. This session's success criterion IS digest
#: equality, so the device is part of the experiment: a different architecture
#: can select different kernels and therefore different low bits, which would
#: surface as a FixedPathDigestMismatch -- a scientific stop condition -- and
#: manufacture a finding out of an infrastructure substitution.
REPLAY_DEVICE = "cuda"

#: THE ARM the search ran. It reaches the scoring-position policy and therefore
#: `config_hash`, so it is part of what a replay must reproduce -- not an
#: execution choice. Read from the frozen design rather than typed, and the
#: per-step `expected_config_hash` is what proves the pair is right.
ARM = "supervised_target"
FAMILY = "qwen3"


def fixed_path_spec(leaf: ReplayLeaf, *, device: str = REPLAY_DEVICE,
                    repo_root: str | Path = REPO_ROOT,
                    max_shard_size: str | int | None = None):
    """`leaf` as an executable `FixedPathSpec`, every step digest-pinned.

    The target geometry and the root teacher come from the modules that OWN
    them -- the frozen search space and the teacher binding -- and the recorded
    `target_spec_hash` is then asserted against the space's own spec. A replay
    that built its own target from the journal could reconstruct a geometry the
    search space no longer declares and still agree with itself.
    """
    from aadistill.initialization.planning.fixed_path import (
        FixedPathSpec, FixedPathStep,
    )
    from experiments.phase_d1 import d1_session as D1S
    from experiments.phase_d1 import search_space as space

    #: EVERY REGISTRY A D1 SESSION NEEDS, through the one function that owns
    #: that list. A replay built its own partial registration -- adapters and
    #: the C2 operators -- and the pod died five seconds into step 0 with
    #:
    #:     KeyError: no calibration profile 'calib.domain_balanced@v1';
    #:               registered: []
    #:
    #: because calibration profiles are a SEPARATE process-global registry. The
    #: GPU qualification's first subrun failed the same way: three of four
    #: registries filled. A replay must reproduce the search's registry state,
    #: and the search fills it here, in this order.
    D1S._register_frozen_operators()
    target = space._target_spec()
    if target.spec_hash != leaf.target_spec_hash:
        raise ReplaySourceError(
            f"{leaf.state_id}: the frozen search space's target hashes to "
            f"{target.spec_hash[:12]} and the search recorded "
            f"{leaf.target_spec_hash[:12]}. The replay target must be the "
            "geometry the search measured.")
    root = D1S.root_teacher_identity(repo_root)
    for key in ("root_teacher_id", "root_teacher_sha256"):
        mine, recorded = root[key], getattr(leaf, key)
        if mine != recorded:
            raise ReplaySourceError(
                f"{leaf.state_id}: the teacher binding gives {key}={mine} and "
                f"the search recorded {recorded}. A replay must start from the "
                "bytes the lineage started from.")
    #: The repo id and revision come from the BINDING, which owns them, rather
    #: than from splitting the journal's combined label on '@'.
    repo_id, revision = root["repo_id"], root["revision"]
    steps = tuple(
        FixedPathStep(impl_id=s.impl_id, profile_id=s.profile_id,
                      expected_artifact_digest=s.expected_artifact_digest,
                      #: THE INPUTS, pinned alongside the output. Checked
                      #: before the operator runs, which is the difference
                      #: between a dictionary comparison and 29 minutes of
                      #: DEPTH.
                      expected_config_hash=s.expected_config_hash,
                      label=f"{s.kind}({s.profile_id})")
        for s in leaf.steps)
    #: THE SCORING PROTOCOL THE SEARCH RAN UNDER, from the modules that own it
    #: rather than from this one. Omitting it made every operator reduce over
    #: the full vocabulary under the incumbent position policy, which is not a
    #: subset of what the search did -- it is a different computation, and the
    #: pinned digests were produced by the other one.
    from experiments.phase_d_series import scoring_protocol as SP
    return FixedPathSpec(
        path_id=f"d1_replay.{leaf.state_id}",
        family=FAMILY,
        target_spec=target,
        steps=steps,
        root_repo_id=repo_id,
        root_revision=revision,
        device=device,
        seed=leaf.seed,
        max_shard_size=max_shard_size,
        position_policy=D1S.position_policy(ARM),
        distribution_support=SP.D_SERIES_SUPPORT,
    )


#: ---------------------------------------------------------------------------
#: THE HISTORICAL ROOT STATE
#: ---------------------------------------------------------------------------
#: A replay starts from a model object, and the search's root object was not the
#: teacher as published. `ChildBuilder` builds a child config with
#: `adapter.build_config(parent.config, new_spec)`, which starts from
#: `parent.config.to_dict()` -- so EVERY field of the root's live config reaches
#: every descendant's `config.json`, and therefore its `config_sha256`, its
#: `artifact_digest`, and the four identities adoption requires.
#:
#: This replay's first version loaded the teacher with a bare `from_pretrained`
#: and inherited the published `use_cache: true`. The search's root had
#: `use_cache: False`. Step 0 of q2 produced `524493fd...` against the pinned
#: `292e36f1...` after 64 s of L40S time, and the cause was one unset field.
#:
#: THE CANDIDATE SET IS DERIVED FROM THE MECHANISM, NOT SWEPT. Two things can
#: put the root in a state other than as-published, and they are the only two:
#:
#:   1. the session driver, which sets `use_cache = False` on the loaded teacher
#:      before handing it to the beam;
#:   2. an operator mutating the model it is handed IN PLACE, which the root
#:      survives because `BeamSearch` caches it (`_root_model`) and shares it
#:      across every level-0 expansion. `depth.causal_kl_greedy_v1` does exactly
#:      this, with the same field.
#:
#: Non-root parents are immune: `_load_state_model` reloads each one from its
#: checkpoint, so each expansion gets a fresh object. And `use_cache` is the
#: only in-place config mutation any shipped operator performs.
#:
#: Both mechanisms reach the same field and the same value, so the candidate set
#: is of size two, and the recorded `config_sha256` of each path's step 0 says
#: which one ran. `derive_root_state` requires exactly one candidate to explain
#: every step 0 -- zero is a material failure, more than one means the check
#: does not discriminate and must not be trusted.
ROOT_CANDIDATES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("teacher_as_published", {}),
    ("session_driver_use_cache_false", {"use_cache": False}),
)


class RootStateUndetermined(ReplaySourceError):
    """No single candidate root state explains the recorded step-0 configs."""


def teacher_config(repo_id: str, revision: str) -> Any:
    """The published teacher config. `config.json` only -- no weights, no GPU."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo_id, revision=revision)


def _child_config_sha256(base_config: Any, *, overrides: Mapping[str, Any],
                         arch_fields: Mapping[str, Any],
                         family: str) -> str:
    """The `config_sha256` a child of this root at this geometry would carry.

    Goes through the real `build_config` and the real `save_pretrained`, so this
    is not a model of the writer -- it IS the writer, and a serialization change
    in transformers moves this value exactly as it would move a checkpoint's.
    """
    import copy
    import tempfile

    from aadistill.infrastructure.manifest import sha256_json
    from aadistill.initialization.specs.arch import ArchSpec, get_adapter

    root = copy.deepcopy(base_config)
    for key, value in overrides.items():
        setattr(root, key, value)
    config = get_adapter(family).build_config(
        root, ArchSpec.of(family, dict(arch_fields)))
    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        return sha256_json(json.loads((Path(tmp) / "config.json").read_text()))


def derive_root_state(leaves: Sequence[ReplayLeaf], *, base_config: Any,
                      family: str = "qwen3") -> dict[str, Any]:
    """Which candidate root state the search actually expanded from.

    Costs nothing: `config.json`, two hashes per path, no weights and no device.
    Every candidate's result is recorded, including the losers, so the record
    shows a derivation rather than an assertion.

    Raises `RootStateUndetermined` when no candidate explains every step 0. That
    is MATERIAL -- it would mean the search's own records do not describe a
    reproducible path -- and is not repaired by widening the candidate set until
    something fits.
    """
    from aadistill.initialization.adapters import register_builtin_adapters

    register_builtin_adapters()
    targets = [(leaf, leaf.steps[0]) for leaf in leaves if leaf.steps]
    if not targets:
        raise ReplaySourceError("no path to derive a root state from")

    evidence: list[dict[str, Any]] = []
    explains: list[str] = []
    for name, overrides in ROOT_CANDIDATES:
        per_step = []
        for leaf, step in targets:
            got = _child_config_sha256(base_config, overrides=overrides,
                                       arch_fields=dict(step.arch_spec),
                                       family=family)
            per_step.append({
                "state_id": step.state_id,
                "leaf": leaf.state_id,
                "impl_id": step.impl_id,
                "recorded_config_sha256": step.expected_config_sha256,
                "reconstructed_config_sha256": got,
                "matches": got == step.expected_config_sha256,
            })
        ok = all(s["matches"] for s in per_step)
        evidence.append({"candidate": name, "overrides": dict(overrides),
                         "explains_every_step_0": ok, "steps": per_step})
        if ok:
            explains.append(name)

    if not explains:
        raise RootStateUndetermined(
            "no derived root state reproduces the recorded step-0 config "
            f"identities: {json.dumps(evidence)}")
    if len(explains) > 1:
        raise RootStateUndetermined(
            f"{explains} all reproduce the recorded step-0 configs, so the "
            "recorded identity does not pick one and this check cannot be "
            "trusted to have determined the historical root state")
    chosen = dict(next(o for n, o in ROOT_CANDIDATES if n == explains[0]))

    #: EVERY REMAINING STEP, under the chosen root. Step 0 picks the candidate;
    #: this checks that the pick explains the whole lineage. It can, because a
    #: child config is the parent's `to_dict()` with the spec applied, so a
    #: path's config at position i is the root's config plus that position's
    #: geometry -- and if some operator mutated a config field the candidate set
    #: does not know about, that position fails here. For free, before the GPU.
    lineage: list[dict[str, Any]] = []
    for leaf in leaves:
        for step in leaf.steps[1:]:
            got = _child_config_sha256(base_config, overrides=chosen,
                                       arch_fields=dict(step.arch_spec),
                                       family=family)
            lineage.append({
                "leaf": leaf.state_id, "index": step.index,
                "impl_id": step.impl_id,
                "recorded_config_sha256": step.expected_config_sha256,
                "reconstructed_config_sha256": got,
                "matches": got == step.expected_config_sha256,
            })
    broken = [s for s in lineage if not s["matches"]]
    if broken:
        raise RootStateUndetermined(
            f"the derived root state {explains[0]!r} reproduces every recorded "
            f"step-0 config but not {len(broken)} later one(s): "
            f"{json.dumps(broken)}. Some execution state beyond the derived "
            "candidate set reached those checkpoints' configs.")

    return {
        "schema": "aadistill.phase_d1.root_state_derivation/v1",
        "candidate_set": [n for n, _ in ROOT_CANDIDATES],
        "derived_from": (
            "the session driver's root loader and the one in-place config "
            "mutation a shipped operator performs; non-root parents reload from "
            "disk and cannot carry sibling state"),
        "chosen_candidate": explains[0],
        "config_overrides": chosen,
        "evidence": evidence,
        "lineage_verified": lineage,
        "_lineage_scope": (
            f"{len(targets)} step-0 identities discriminate the candidate; "
            f"{len(lineage)} later identities confirm it explains the whole "
            "config lineage. WEIGHTS are not checked here -- that is what the "
            "digest-pinned replay on the GPU is for."),
    }


def verify_operator_configs(leaves: Sequence[ReplayLeaf], *,
                            repo_root: str | Path = REPO_ROOT,
                            device: str = "cpu") -> dict[str, Any]:
    """Every step's OPERATOR INPUTS, against the hash the search recorded. $0.

    The sibling of `derive_root_state`, for the other half of the historical
    execution state. That one answers "which model config did the search expand
    from"; this answers "under which scoring protocol did its operators run".
    Both are inputs to a digest, both were wrong, and both are checkable from
    `config.json` and a calibration mixture with no GPU and no weights.

    It is also enforced ON the pod, by `FixedPathStep.expected_config_hash`,
    before each operator runs. This runs it here as well because here it costs
    nothing and there it costs a pod.

    Raises `ReplaySourceError` listing every disagreeing step. A replay whose
    operators run under a different protocol is not a replay, and the digest
    mismatch it produces says nothing about the checkpoint.
    """
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.planning.operator_config import (
        hashed_operator_config, operator_config_hash,
    )

    root = Path(repo_root)
    checked: list[dict[str, Any]] = []
    items_by_profile: dict[str, Any] = {}
    for leaf in leaves:
        spec = fixed_path_spec(leaf, device=device, repo_root=root)
        for recorded, step in zip(leaf.steps, spec.steps):
            if step.profile_id not in items_by_profile:
                profile = get_profile(step.profile_id)
                items_by_profile[step.profile_id] = prepare_calibration_items(
                    profile.resolve(root), profile_id=profile.qualified_id)
            items = items_by_profile[step.profile_id]
            derived = operator_config_hash(hashed_operator_config(
                implementation=get_implementation(step.impl_id),
                policy=spec.position_policy,
                support=spec.distribution_support,
                items=items, declared=step.config))
            checked.append({
                "leaf": leaf.state_id, "index": recorded.index,
                "impl_id": step.impl_id, "profile_id": step.profile_id,
                "recorded_config_hash": recorded.expected_config_hash,
                "derived_config_hash": derived,
                "matches": derived == recorded.expected_config_hash,
            })
    wrong = [c for c in checked if not c["matches"]]
    if wrong:
        raise ReplaySourceError(
            f"{len(wrong)} of {len(checked)} steps would run under an operator "
            "config the search did not record. The operators would compute "
            "something the pinned digests were not produced by: "
            f"{json.dumps(wrong)}")
    return {
        "schema": "aadistill.phase_d1.operator_config_agreement/v1",
        "n_steps": len(checked),
        "all_match": True,
        "position_policy_arm": ARM,
        "distribution_support": sorted(
            {c["derived_config_hash"] for c in checked}),
        "steps": checked,
        "_what_this_answers": (
            "under which scoring protocol did the search's operators run, and "
            "will this replay's run under the same one. The sibling of the "
            "root-state derivation, for the inputs rather than the parent."),
    }


REPLAY_EXECUTION_OWNER = "experiments.phase_d1.d1_session.execution"


def replay_execution():
    """The EXECUTION CONFIG the search ran, from the session that owns it.

    The third input this replay failed to carry, and the one no hash covers.
    `micro_batch_size` and `calibration_batch_packing` are deliberately
    runtime-only -- they do not change the estimand, so they are excluded from
    `config_hash` on purpose. For most operators that is correct. For one it is
    not: `ffn.activation_importance_v0` keeps the top 3072 of 9728 neurons, and
    near that cutoff the importance scores are dense enough that a different
    reduction order flips the kept set, which changes the weights and
    therefore the artifact digest.

    `materialize_fixed_path` defaults to `DEFAULT_EXECUTION` (4,
    `original_order_v1`) and the search ran (3, `length_sorted_v1`). DEPTH
    reproduced byte-exactly anyway, because its greedy block choice compares KL
    gaps far larger than batch-order float noise; FFN did not.
    """
    from experiments.phase_d1 import d1_session as D1S

    return D1S.execution()


def verify_execution_config(leaves: Sequence[ReplayLeaf], *,
                            execution: Any = None) -> dict[str, Any]:
    """Every step's recorded execution knobs, against the ones to be used. $0.

    The third of the three agreement checks, and the one that exists because a
    hash deliberately omits what it checks. `derive_root_state` covers the
    parent the operators start from, `verify_operator_configs` covers the
    protocol they run under, and this covers the runtime shape they reduce in.

    Raises `ReplaySourceError` naming every disagreeing step, because a replay
    that reduces in a different order is not reproducing the run it replays --
    whatever the hashes say.
    """
    live = execution if execution is not None else replay_execution()
    mine = {"micro_batch_size": int(live.micro_batch_size),
            "calibration_batch_packing": str(live.calibration_batch_packing)}
    checked: list[dict[str, Any]] = []
    for leaf in leaves:
        for step in leaf.steps:
            recorded = {
                "micro_batch_size": step.expected_micro_batch_size,
                "calibration_batch_packing": step.expected_batch_packing,
            }
            #: A step whose trace recorded nothing cannot be checked, and
            #: SILENTLY PASSING it is how the fourth defect of this class
            #: would become the fifth.
            unknown = [k for k, v in recorded.items() if not v]
            checked.append({
                "leaf": leaf.state_id, "index": step.index,
                "impl_id": step.impl_id, "recorded": recorded, "using": mine,
                "matches": not unknown and recorded == mine,
                "unrecorded": unknown or None,
            })
    wrong = [c for c in checked if not c["matches"]]
    if wrong:
        raise ReplaySourceError(
            f"{len(wrong)} of {len(checked)} steps would run under execution "
            "knobs the search did not record. No hash covers these, which is "
            "why they are checked here: "
            f"{json.dumps(wrong)}")
    return {
        "schema": "aadistill.phase_d1.execution_agreement/v1",
        "n_steps": len(checked), "all_match": True,
        "using": mine, "owner": REPLAY_EXECUTION_OWNER,
        "steps": checked,
        "_why_a_separate_check": (
            "micro_batch_size and calibration_batch_packing are excluded from "
            "`config_hash` by design, because they do not change the "
            "estimand. They DO change which neurons a top-k keeps when the "
            "scores near the cutoff are dense, so a replay must carry them "
            "even though no identity does."),
    }


def load_root_model(spec: Any, *, config_overrides: Mapping[str, Any],
                    device: str | None = None) -> Any:
    """The root the search expanded from, reloaded per path.

    Reloaded rather than shared because operators MUTATE the module they are
    given -- which is the whole subject of `ROOT_CANDIDATES`. A second path
    starting from the first path's root would begin from an already-compressed
    model and diverge at step one. After the first load the weights are in the
    local cache, so this is a disk read.

    `config_overrides` comes from `derive_root_state` and is applied before any
    operator runs, because `build_config` reads the live config.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        spec.root_repo_id, dtype=torch.bfloat16, revision=spec.root_revision)
    for key, value in config_overrides.items():
        setattr(model.config, key, value)
    return model.to(device or spec.device).eval()


def adoption_matches(leaf: ReplayLeaf, artifact: Any) -> tuple[bool, dict]:
    """All FOUR identities, exactly. No approximate equivalence.

    A mismatch after a completed reconstruction is a deterministic-materialization
    defect to diagnose, not a reason to substitute a near-equivalent checkpoint
    or to weaken an identity check.
    """
    got = {
        "artifact_digest": getattr(artifact, "artifact_digest", None),
        "weights_digest": getattr(artifact, "weights_digest", None),
        "single_shard_sha256": getattr(artifact, "single_shard_sha256", None),
        "arch_signature": getattr(artifact, "arch_signature", None),
    }
    want = {
        "artifact_digest": leaf.expected_artifact_digest,
        "weights_digest": leaf.expected_weights_digest,
        "single_shard_sha256": leaf.expected_single_shard_sha256,
        "arch_signature": leaf.expected_arch_signature,
    }
    differ = sorted(k for k in want if got[k] != want[k])
    return (not differ), {"expected": want, "actual": got, "differ": differ}


def leaves_from_plan(doc: Mapping[str, Any]) -> list[ReplayLeaf]:
    """Rebuild the plan `describe()` wrote. The pod's entry point.

    THE JOURNAL DOES NOT TRAVEL. It is 67 MB and this host's uplink is about
    0.72 MB/s, so staging it would cost roughly 93 minutes on a billing pod to
    ship evidence the pod only needs two leaves' worth of. The plan is resolved
    here, where the journal lives, and the pod receives the ~8 KB of pinned
    digests it actually consumes.

    That keeps the journal's own checks -- row deduplication, identity
    agreement, contiguous ancestry -- on the side that has the journal, and
    leaves the pod with a document whose every digest is a pin it must satisfy.
    """
    leaves: list[ReplayLeaf] = []
    for row in doc["leaves"]:
        want = row["expected"]
        steps = tuple(
            ReplayStep(index=int(s["index"]), kind=str(s["kind"]),
                       impl_id=str(s["impl_id"]), profile_id=str(s["profile_id"]),
                       seed=int(s["seed"]), state_id=str(s["state_id"]),
                       expected_artifact_digest=str(s["expected_artifact_digest"]),
                       expected_single_shard_sha256="",
                       expected_config_sha256=str(s.get("expected_config_sha256", "")),
                       expected_config_hash=str(s.get("expected_config_hash", "")),
                       expected_micro_batch_size=int(
                           s.get("expected_micro_batch_size") or 0),
                       expected_batch_packing=str(
                           s.get("expected_batch_packing") or ""),
                       arch_spec=tuple(sorted(
                           (str(k), v) for k, v in
                           (s.get("arch_spec") or {}).items())))
            for s in row["steps"])
        if not steps:
            raise ReplaySourceError(f"{row['state_id']}: the plan carries no steps")
        missing = [s.index for s in steps if not s.expected_artifact_digest]
        if missing:
            raise ReplaySourceError(
                f"{row['state_id']}: plan steps {missing} carry no expected "
                "digest; an unpinned intermediate would let a compensating pair "
                "of errors pass as a correct replay")
        #: The ROOT-STATE fingerprint, required on step 0 of every path. A plan
        #: without it would let a pod start from a root whose config nothing
        #: checked -- which is exactly how 64 s of L40S time and $0.21 were
        #: spent discovering a divergence a hash comparison answers for free.
        blind = [s.index for s in steps
                 if s.index == 0 and (len(s.expected_config_sha256) != 64
                                      or not s.arch_spec)]
        if blind:
            raise ReplaySourceError(
                f"{row['state_id']}: step 0 carries no recorded config sha256 "
                "and geometry, so the root state cannot be derived or checked")
        for key in ("artifact_digest", "weights_digest", "single_shard_sha256",
                    "arch_signature"):
            if len(str(want.get(key, ""))) != 64:
                raise ReplaySourceError(
                    f"{row['state_id']}: plan expects a {key} of "
                    f"{len(str(want.get(key,'')))} hex characters, not 64")
        leaves.append(ReplayLeaf(
            state_id=str(row["state_id"]),
            quality_position=int(row["quality_position"]),
            path_label=str(row["path_label"]),
            steps=steps,
            root_teacher_id=str(row["root_teacher_id"]),
            root_teacher_sha256=str(row["root_teacher_sha256"]),
            target_spec_hash=str(row["target_spec_hash"]),
            seed=int(row["seed"]),
            num_parameters=int(row["num_parameters"]),
            expected_artifact_digest=str(want["artifact_digest"]),
            expected_weights_digest=str(want["weights_digest"]),
            expected_single_shard_sha256=str(want["single_shard_sha256"]),
            expected_arch_signature=str(want["arch_signature"])))
    return leaves

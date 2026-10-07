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
                 "expected_artifact_digest": s.expected_artifact_digest}
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
        steps.append(ReplayStep(
            index=position, kind=str(step.get("kind")),
            impl_id=str(step.get("impl_id")),
            profile_id=str(step.get("profile_id")),
            seed=int(step.get("seed")),
            state_id=str(row["state_id"]),
            expected_artifact_digest=str(digest),
            expected_single_shard_sha256=str(shard)))
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


def describe(leaves: Sequence[ReplayLeaf]) -> dict[str, Any]:
    """The whole plan as a record, for a $0 preflight and for the session."""
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
        "leaves": [leaf.as_dict() for leaf in leaves],
    }


#: The pod's GPU. PINNED, not chosen. This session's success criterion IS digest
#: equality, so the device is part of the experiment: a different architecture
#: can select different kernels and therefore different low bits, which would
#: surface as a FixedPathDigestMismatch -- a scientific stop condition -- and
#: manufacture a finding out of an infrastructure substitution.
REPLAY_DEVICE = "cuda"
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
                      label=f"{s.kind}({s.profile_id})")
        for s in leaf.steps)
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
    )


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
                       expected_single_shard_sha256="")
            for s in row["steps"])
        if not steps:
            raise ReplaySourceError(f"{row['state_id']}: the plan carries no steps")
        missing = [s.index for s in steps if not s.expected_artifact_digest]
        if missing:
            raise ReplaySourceError(
                f"{row['state_id']}: plan steps {missing} carry no expected "
                "digest; an unpinned intermediate would let a compensating pair "
                "of errors pass as a correct replay")
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

"""Rebuild attempt 3's five selected leaves as digest-pinned fixed paths.

The Full Joint Search committed its Top-5 and then lost the weights: the session
died after `stage1_selection.commit()` and the pod was torn down with the
checkpoints on it. The selection is frozen and accepted; only the artifacts are
missing. This module turns that frozen record back into something executable.

It is a REPLAY, not a search. Nothing here ranks, evaluates or selects. Every
step of every path is pinned to the artifact digest attempt 3 recorded for it,
so the only two outcomes are "byte-identical to attempt 3" and "STOP".

Three properties the reconstruction depends on:

* **Every intermediate is pinned, not just the leaf.** A path that agreed only
  at its final digest would leave four unverified operators behind it, and a
  compensating pair of errors would be indistinguishable from a correct replay.
  `FixedPathStep.expected_artifact_digest` is set on all four steps.
* **The ancestry comes from the journal, not from the path label.** The label
  says which operator and profile ran at each position; it does not say what
  that position produced. The digests come from the recorded states, matched by
  operator-and-profile prefix.
* **The source is the attempt-3 session commit**, recorded here as data rather
  than as a claim in a review message. `assert_operators_unmoved` re-checks it,
  because a digest-pinned replay whose operators have moved fails on a paid pod
  when it could have failed here.

The journal's 202 rows cover 108 distinct states: the search appends a row when
a state is created and again when it is evaluated. Deduplication is by
`state_id`, and the duplicates are required to agree on identity — if two rows
for one state disagree, the journal is not a record and this module refuses.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aadistill.initialization.planning import stage1_selection
from aadistill.initialization.planning.fixed_path import (
    FixedPathSpec, FixedPathStep,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The session whose executable produced the artifacts this replay reproduces.
#: The review that authorized the replay named it in prose; the authorization
#: needs it as data, so it lives here and `source_binding()` emits it.
ATTEMPT3_SESSION_COMMIT = "2421f630bc812d414ae25245e855059ffe29610d"

#: Attempt 3's collected evidence, committed to this repository.
EVIDENCE_REL = "logs/stages/stage-1/phase_c2_full_search/runs/attempt3/evidence"
SELECTION_REL = f"{EVIDENCE_REL}/stage1_selection.json"
JOURNAL_REL = f"{EVIDENCE_REL}/states_compact.jsonl"

#: The frozen commitment. Not recomputed from the file — asserted against it, so
#: a replacement selection cannot quietly become the thing being reconstructed.
SELECTION_SHA256 = (
    "7271091c91416b523865ea1320190e4a9ceafb97ecc68bc1b0e49f621573b673")

#: The full 61.4-MiB journal the compact one was derived from, preserved out of
#: tree. Recorded for provenance; this module reads the compact file.
FULL_JOURNAL_SHA256 = (
    "eb008ee252ce9d35e7478c3cc346ef7db29acf3eb5bfb2486c6023e19abe4ef0")

#: Trees whose contents decide an artifact digest. A replay pinned to attempt
#: 3's digests is only honest if the code that produced them has not moved.
OPERATOR_TREES = (
    "src/aadistill/initialization",
    "scripts/experiments/phase_c2",
)

#: Individual files outside those trees that the replay's inputs come from.
#: `phase_a_frozen` holds the teacher id and revision, the target geometry and
#: the search seed — the root and target of every path here. Its tree is NOT
#: checked wholesale: it also holds `phase_a_search.py`, the search entry point
#: this replay does not call and from which the canonical-control injection was
#: deliberately removed after attempt 3 died in it.
OPERATOR_FILES = ("scripts/autoinit/phase_a_frozen.py",)

#: Identity fields a duplicate journal row may not disagree about.
_IDENTITY_KEYS = ("artifact_digest", "checkpoint_sha256", "impl_ids",
                  "calibration_profiles", "arch_spec_hash")


class ReplaySourceError(RuntimeError):
    """The committed evidence cannot support a pinned replay."""


@dataclass(frozen=True)
class ReplayLeaf:
    """One selected leaf, its pinned path, and the identity it must reproduce."""

    state_id: str
    path_label: str
    lineage: str
    spec: FixedPathSpec
    #: Attempt 3's final identity for this leaf. `verify_transferred_leaf` and
    #: the closeout both compare against these, not against the run's own output.
    artifact_digest: str
    weights_digest: str
    single_shard_sha256: str
    arch_signature: str
    num_parameters: int
    #: Every step's pinned digest, in order, including the leaf's own.
    step_digests: tuple[str, ...]
    #: What this path costs in the worst case, in minutes, derived from attempt
    #: 3's own telemetry. Used for ADMISSION CONTROL: the driver refuses to start
    #: a path it cannot afford to finish, which is what makes the session's cost
    #: bounded by construction rather than by hope.
    bounded_minutes: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "path": self.path_label,
            "lineage": self.lineage,
            "artifact_digest": self.artifact_digest,
            "weights_digest": self.weights_digest,
            "single_shard_sha256": self.single_shard_sha256,
            "arch_signature": self.arch_signature,
            "num_parameters": self.num_parameters,
            "step_digests": list(self.step_digests),
            "bounded_minutes": self.bounded_minutes,
            "spec": self.spec.as_dict(),
        }


def load_selection(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen Top-5, verified twice.

    `stage1_selection.load` checks the record against its own commitment hash,
    which catches an edit. The equality against `SELECTION_SHA256` additionally
    catches a *substitution* — a different, internally consistent selection put
    in the same place.
    """
    path = Path(repo_root) / SELECTION_REL
    record = stage1_selection.load(path)
    if record["selection_sha256"] != SELECTION_SHA256:
        raise ReplaySourceError(
            f"{SELECTION_REL} commits {record['selection_sha256'][:12]}… but this "
            f"replay reconstructs {SELECTION_SHA256[:12]}…. A replay is bound to "
            "one selection; it does not follow the file.")
    if record["n_selected"] != len(record["selected"]) != 5:
        raise ReplaySourceError(
            f"expected 5 selected leaves, found {len(record['selected'])}")
    return record


def load_states(repo_root: str | Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """The journal's states, deduplicated by id and checked for agreement."""
    path = Path(repo_root) / JOURNAL_REL
    merged: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["state_id"]
        prev = merged.get(sid)
        if prev is None:
            merged[sid] = dict(row)
            continue
        for key in _IDENTITY_KEYS:
            if key in prev and key in row and prev[key] != row[key]:
                raise ReplaySourceError(
                    f"journal rows for state {sid} disagree about {key!r}: "
                    f"{prev[key]!r} vs {row[key]!r}. The journal is not a record "
                    "of one search and cannot pin a replay.")
        #: Later rows carry the evaluation; earlier rows carry creation. Merge
        #: rather than overwrite, and never let a null erase a recorded value.
        merged[sid] = {**prev, **{k: v for k, v in row.items() if v is not None}}
    if not merged:
        raise ReplaySourceError(f"{JOURNAL_REL} holds no states")
    return merged


#: The telemetry attempt 3 collected, beside the journal.
TELEMETRY_REL = f"{EVIDENCE_REL}/telemetry.jsonl"

#: Everything a replay pays per step. `state_evaluation_seconds` is deliberately
#: absent: the replay evaluates nothing, so charging it would inflate the bound
#: with work this session does not do.
_COST_FIELDS = ("parent_load_seconds", "operator_seconds", "materialize_seconds",
                "identify_seconds", "canonical_reload_seconds",
                "validation_seconds")


def _seconds(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, Mapping):
        return float(value.get("seconds") or 0.0)
    return float(value or 0.0)


def worst_seconds_by_impl(repo_root: str | Path = REPO_ROOT,
                          ) -> dict[str, float]:
    """The most expensive observation of each IMPLEMENTATION in the whole search.

    A bound, not an average: a fresh replay reuses nothing, so each step is
    priced at the worst that implementation was ever observed to cost rather
    than at what it happened to cost on the selected path.

    Keyed on `impl_id` and NOT on operator kind, which was the first version and
    was wrong. Two implementations of DEPTH appear in the selected paths —
    `depth.causal_kl_greedy_v1` at up to 25.7 minutes and `depth.positional_v0`
    at 0.6 — and they are different operators, not the same one with and without
    a cache. Bounding by kind charged the cheap one at the expensive one's rate
    and overpriced that path by about 25 minutes, which is most of a leaf.
    """
    worst: dict[str, float] = {}
    path = Path(repo_root) / TELEMETRY_REL
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        impl = row.get("impl_id")
        if not impl:
            continue
        total = sum(_seconds(row, f) for f in _COST_FIELDS)
        worst[impl] = max(worst.get(impl, 0.0), total)
    if not worst:
        raise ReplaySourceError(
            f"{TELEMETRY_REL} yielded no per-implementation timings; the replay "
            "cannot bound what it cannot price.")
    return worst


def resolve_ancestry(states: Mapping[str, Mapping[str, Any]],
                     state_id: str) -> list[Mapping[str, Any]]:
    """The leaf's chain from first operator to last, root-first.

    Matched by (impl_ids, calibration_profiles) prefix: the search's state id is
    a content id over exactly that pair, so the state one operator shorter is
    the parent and there is exactly one. Anything else means the journal is
    missing a state this replay would have to invent.
    """
    if state_id not in states:
        raise ReplaySourceError(f"state {state_id} is not in the journal")
    chain = [states[state_id]]
    while len(chain[-1]["impl_ids"]) > 1:
        cur = chain[-1]
        impls = list(cur["impl_ids"])[:-1]
        profs = list(cur["calibration_profiles"])[:-1]
        parents = [s for s in states.values()
                   if list(s["impl_ids"]) == impls
                   and list(s["calibration_profiles"]) == profs]
        if len(parents) != 1:
            raise ReplaySourceError(
                f"state {cur['state_id']} has {len(parents)} journal parents for "
                f"prefix {impls}; a pinned replay needs exactly one.")
        chain.append(parents[0])
    chain.reverse()
    for step in chain:
        if not step.get("artifact_digest"):
            raise ReplaySourceError(
                f"state {step['state_id']} has no recorded artifact_digest; this "
                "replay pins every intermediate and cannot pin that one.")
    return chain


def build_replay_leaves(repo_root: str | Path = REPO_ROOT,
                        *, device: str = "cuda",
                        max_shard_size: str | int | None = None,
                        ) -> list[ReplayLeaf]:
    """The five leaves, each a fully pinned `FixedPathSpec`.

    Root, target geometry and seed come from the same frozen module the search
    itself read, and the seed is additionally checked against the one the
    selection records — a replay under a different seed is a different
    experiment whatever its digests say.
    """
    import sys
    for extra in ("scripts", "scripts/autoinit"):
        candidate = str(Path(repo_root) / extra)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from phase_a_frozen import (  # noqa: E402
        SEARCH_SEED, TARGET_GEOMETRY, TEACHER_ID, TEACHER_REVISION,
    )
    from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
    from aadistill.initialization.operators.register import (  # noqa: E402
        register_builtin_operators,
    )
    from experiments.phase_c2.search_space import (  # noqa: E402
        register_c2_operators,
    )

    #: `FixedPathSpec.__post_init__` resolves every impl_id against the registry,
    #: so the specs cannot be built until it is populated. Called here rather
    #: than left to an importer: a builder that only works when some other entry
    #: point has already registered is a builder that works by luck.
    #:
    #: BOTH sets. `attention.activation_importance_v1` is not a shipped default
    #: and four of the five selected paths name it; the builtins alone leave the
    #: registry looking populated while resolving only two thirds of the library.
    register_builtin_operators()
    register_c2_operators()

    selection = load_selection(repo_root)
    states = load_states(repo_root)
    worst = worst_seconds_by_impl(repo_root)

    #: One teacher load per path. Paths are run independently because the
    #: operators mutate the module they are given, so each pays it.
    root_load_minutes = 1.5

    recorded_seed = int(selection["search"]["seed"])
    if recorded_seed != int(SEARCH_SEED):
        raise ReplaySourceError(
            f"the selection was produced under seed {recorded_seed} and this tree "
            f"declares {SEARCH_SEED}. Replaying under a different seed is a "
            "different experiment.")

    target = ArchSpec.of("qwen3", TARGET_GEOMETRY)
    if target.spec_hash != selection["search"]["target_spec_hash"]:
        raise ReplaySourceError(
            f"target geometry hashes to {target.spec_hash[:12]}… but the selection "
            f"was searched against {selection['search']['target_spec_hash'][:12]}…")

    lineage_by_id = {d["state_id"]: d.get("lineage", "")
                     for d in selection.get("decisions", [])}

    leaves: list[ReplayLeaf] = []
    for entry in selection["selected"]:
        sid = entry["state_id"]
        chain = resolve_ancestry(states, sid)
        steps = tuple(
            FixedPathStep(
                impl_id=node["impl_ids"][-1],
                profile_id=node["calibration_profiles"][-1],
                expected_artifact_digest=node["artifact_digest"],
                label=f"{node['applied_kinds'][-1]}@{node['state_id'][:12]}",
            )
            for node in chain
        )
        spec = FixedPathSpec(
            path_id=f"c2_replay.{sid}",
            family="qwen3",
            target_spec=target,
            steps=steps,
            root_repo_id=TEACHER_ID,
            root_revision=TEACHER_REVISION,
            device=device,
            seed=recorded_seed,
            max_shard_size=max_shard_size,
        )
        #: The spec is built from the journal; the identity is asserted against
        #: the SELECTION. If the two ever disagreed, the selection wins and this
        #: raises rather than reconstructing the journal's version.
        final = chain[-1]
        if final["artifact_digest"] != entry["artifact_digest"]:
            raise ReplaySourceError(
                f"leaf {sid}: journal records {final['artifact_digest'][:12]}… and "
                f"the selection commits {entry['artifact_digest'][:12]}…")
        if final["checkpoint_sha256"] != entry["single_shard_sha256"]:
            raise ReplaySourceError(
                f"leaf {sid}: journal shard sha disagrees with the selection")
        bounded = root_load_minutes + sum(
            worst[node["impl_ids"][-1]] for node in chain) / 60.0
        leaves.append(ReplayLeaf(
            state_id=sid,
            path_label=entry["path"],
            lineage=lineage_by_id.get(sid, ""),
            spec=spec,
            artifact_digest=entry["artifact_digest"],
            weights_digest=entry["weights_digest"],
            single_shard_sha256=entry["single_shard_sha256"],
            arch_signature=entry["arch_signature"],
            num_parameters=int(entry["num_parameters"]),
            step_digests=tuple(n["artifact_digest"] for n in chain),
            bounded_minutes=round(bounded, 2),
        ))
    return leaves


def _tree_digest(repo_root: Path, commit: str, tree: str) -> dict[str, str]:
    """`path -> blob sha` for one tree at one commit. Git's own hashes."""
    out = subprocess.run(
        ["git", "ls-tree", "-r", commit, "--", tree],
        capture_output=True, text=True, cwd=repo_root)
    if out.returncode != 0:
        raise ReplaySourceError(
            f"cannot read {tree} at {commit[:12]}…: {out.stderr.strip()}")
    listing: dict[str, str] = {}
    for line in out.stdout.splitlines():
        meta, path = line.split("\t", 1)
        listing[path] = meta.split()[2]
    return listing


def assert_operators_unmoved(repo_root: str | Path = REPO_ROOT,
                             *, head: str = "HEAD") -> dict[str, Any]:
    """Refuse a pinned replay whose operators have moved since attempt 3.

    The digest gate would catch this on the pod, at the cost of the run. The
    failure it prevents is concrete and has happened in this programme: an edit
    landed between a recorded result and the session meant to reproduce it, and
    the mismatch surfaced only where it was expensive. This is the $0 half.

    The rule is asymmetric on purpose. A file that existed at attempt 3 and has
    since been MODIFIED or REMOVED can change what the replay computes, so it is
    refused. A file ADDED afterwards cannot: code that ran in attempt 3 could not
    import a module that did not exist. Additions are reported and allowed, which
    is what lets this module live in a checked tree.

    Returns the comparison so an authorization can carry it.
    """
    repo_root = Path(repo_root)
    changed: dict[str, str] = {}
    added: list[str] = []
    counts: dict[str, int] = {}
    for tree in OPERATOR_TREES:
        before = _tree_digest(repo_root, ATTEMPT3_SESSION_COMMIT, tree)
        after = _tree_digest(repo_root, head, tree)
        counts[tree] = len(before)
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            if path not in before:
                added.append(path)
            else:
                changed[path] = "removed" if path not in after else "modified"
    for path in OPERATOR_FILES:
        before = _tree_digest(repo_root, ATTEMPT3_SESSION_COMMIT, path)
        after = _tree_digest(repo_root, head, path)
        if not before:
            raise ReplaySourceError(
                f"{path} did not exist at {ATTEMPT3_SESSION_COMMIT[:12]}…; it "
                "cannot be a pinned input to a replay of that commit.")
        counts[path] = len(before)
        if before != after:
            changed[path] = "removed" if not after else "modified"
    if changed:
        raise ReplaySourceError(
            "refusing to build a digest-pinned replay: these files decided "
            f"attempt 3's artifact digests and have moved since "
            f"{ATTEMPT3_SESSION_COMMIT[:12]}… — "
            + ", ".join(f"{p} ({why})" for p, why in sorted(changed.items()))
            + ". Attempt 3's digests were produced by the old bytes; replaying "
            "with the new ones is not a replay.")
    return {
        "source_commit": ATTEMPT3_SESSION_COMMIT,
        "head": subprocess.run(["git", "rev-parse", head], capture_output=True,
                               text=True, cwd=repo_root).stdout.strip(),
        "checked": {t: {"files_at_source": counts[t]}
                    for t in (*OPERATOR_TREES, *OPERATOR_FILES)},
        "modified_or_removed": changed,
        "added_since_source": sorted(added),
        "rule": ("modified or removed files are refused; files added after the "
                 "source commit cannot have influenced it and are allowed"),
        "verdict": "no operator-bearing file has moved",
    }


def source_binding(repo_root: str | Path = REPO_ROOT,
                   *, head: str = "HEAD") -> dict[str, Any]:
    """The machine-readable statement of what this replay reproduces.

    The review named attempt 3's session commit in prose. An authorization that
    merely repeated the prose would be unverifiable; this is the same fact as
    data, alongside the digests of the evidence it was read from.
    """
    from aadistill.infrastructure.manifest import sha256_file

    repo_root = Path(repo_root)
    selection = load_selection(repo_root)
    leaves = build_replay_leaves(repo_root)
    return {
        "schema": "aadistill.autoinit.c2_replay_source_binding/v1",
        "reconstructs": "phase_c2_full_search/attempt3 Top-5 checkpoints",
        "source_session_commit": ATTEMPT3_SESSION_COMMIT,
        "selection_sha256": SELECTION_SHA256,
        "selection_file": SELECTION_REL,
        "selection_file_sha256": sha256_file(repo_root / SELECTION_REL),
        "journal_file": JOURNAL_REL,
        "journal_file_sha256": sha256_file(repo_root / JOURNAL_REL),
        "full_journal_sha256": FULL_JOURNAL_SHA256,
        "search": dict(selection["search"]),
        "policy": dict(selection["policy"]),
        "suite": dict(selection["suite"]),
        "profiles": [dict(p) for p in selection["profiles"]],
        "operators": assert_operators_unmoved(repo_root, head=head),
        "leaves": [
            {"state_id": leaf.state_id, "path": leaf.path_label,
             "lineage": leaf.lineage,
             "artifact_digest": leaf.artifact_digest,
             "weights_digest": leaf.weights_digest,
             "single_shard_sha256": leaf.single_shard_sha256,
             "arch_signature": leaf.arch_signature,
             "num_parameters": leaf.num_parameters,
             "step_digests": list(leaf.step_digests),
             "bounded_minutes": leaf.bounded_minutes}
            for leaf in leaves
        ],
        "note": ("every step of every path is pinned to the artifact digest "
                 "attempt 3 recorded for it, not merely the final output. A "
                 "mismatch at any step is a STOP condition and a scientific "
                 "finding, not a retryable engineering failure."),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write the source binding here")
    args = ap.parse_args(argv)
    binding = source_binding()
    text = json.dumps(binding, indent=1) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

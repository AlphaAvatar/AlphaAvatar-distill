#!/usr/bin/env python3
"""Where does the config identity of each selected path first diverge? $0, no GPU.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/c2_replay_config_forensic.py --out <path>

Attempt 8 reconstructed two of the five selected paths byte-for-byte and stopped
on the third, whose WEIGHTS were identical and whose `config.json` was not.
`artifact_digest` covers both, so the refusal stands — the runtime loads through
`from_pretrained(path)` and non-structural Qwen3 fields (RoPE semantics, norms,
dtype and loading behaviour) decide how identical tensors are interpreted.

Before spending another GPU hour on the two paths nobody has replayed, this
answers the same question for free: **replay the CONFIG lineage only.** Every
config in this programme is derived from its parent's config and the child's
`ArchSpec` by `adapter.build_config`, which is pure — no weights, no device, no
operator arithmetic. So the whole config chain of all five paths can be rebuilt
on a CPU in seconds and compared, step by step, against what attempt 3 recorded.

What it produces, per selected leaf and per step:

    leaf -> step -> attempt3 config sha -> reconstructed sha -> MATCH/MISMATCH

and the FIRST transition at which each path diverges, if it diverges.

Two things it deliberately does not do. It does not infer config CONTENT from a
sha — a hash is not invertible and guessing which field moved would be fiction.
And it does not build the child model, so it cannot see a mutation that model
construction introduces; the lineage below is the config as `build_config`
returns it, which is the input to that construction. Where the two disagree,
that gap is itself the finding.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402

from experiments.phase_c2 import replay_specs as RS  # noqa: E402

#: The complete attempt-3 journal. The compact one in git drops the per-state
#: `steps` record — 98% of the bytes — which is where the parent ids and the
#: per-step identities live.
FULL_JOURNAL = Path("/home/ecs-user/aad-artifacts/phase_c2_full_search/"
                    "attempt3/states.jsonl")
FULL_JOURNAL_SHA256 = RS.FULL_JOURNAL_SHA256


def load_full_journal(path: Path = FULL_JOURNAL) -> dict[str, dict[str, Any]]:
    """Every state attempt 3 journalled, by id, verified against the selection.

    The selection records the journal's sha256, so this is the same chain a
    reader can check rather than a file that merely sits in the right place.
    """
    from aadistill.infrastructure.manifest import sha256_file

    if not path.is_file():
        raise SystemExit(
            f"{path} is not on this machine. It is the complete journal; its "
            "durable copy and sha are recorded in attempt 3's "
            "states_full_location.json.")
    got = sha256_file(path)
    if got != FULL_JOURNAL_SHA256:
        raise SystemExit(
            f"{path} hashes to {got}, not the {FULL_JOURNAL_SHA256} the frozen "
            "selection commits. This is not attempt 3's journal.")
    states: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("state_id")
        if not sid:
            continue
        prev = states.get(sid)
        #: Later rows carry the evaluation; merge without letting a null erase a
        #: recorded value, exactly as the compact reader does.
        states[sid] = ({**prev, **{k: v for k, v in row.items() if v is not None}}
                       if prev else dict(row))
    return states


def lineage(states: dict[str, dict[str, Any]], leaf_id: str) -> list[dict[str, Any]]:
    """The leaf's chain root-first, by the journal's OWN `parent_id`.

    Not by operator-and-profile prefix, which is what the replay's spec builder
    uses. Two derivations of the same chain that agree are worth more than one,
    and this file exists to check an identity the other one assumed.
    """
    chain: list[dict[str, Any]] = []
    cur = states[leaf_id]
    while True:
        chain.append(cur)
        parent = cur.get("parent_id")
        if not parent:
            break
        if parent not in states:
            #: The ROOT. The teacher is the parent of every depth-1 state and is
            #: not itself a journalled state, so an absent parent there is the
            #: chain's beginning rather than a gap. At any greater depth it
            #: WOULD be a gap, and is refused.
            if len(cur.get("impl_ids") or []) == 1:
                break
            raise SystemExit(
                f"state {cur['state_id']} is at depth "
                f"{len(cur.get('impl_ids') or [])} and names absent parent "
                f"{parent}; the journal cannot support a lineage through it")
        cur = states[parent]
    chain.reverse()
    return chain


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    """What attempt 3 recorded about a state's artifact."""
    art = row.get("artifact") or {}
    return {
        "state_id": row.get("state_id"),
        "parent_id": row.get("parent_id"),
        "impl_ids": list(row.get("impl_ids") or []),
        "applied_kinds": list(row.get("applied_kinds") or []),
        "calibration_profiles": list(row.get("calibration_profiles") or []),
        "arch_spec": row.get("arch_spec"),
        "arch_spec_hash": row.get("arch_spec_hash"),
        "config_sha256": art.get("config_sha256"),
        "artifact_digest": art.get("artifact_digest"),
        "weights_digest": art.get("weights_digest"),
        "single_shard_sha256": art.get("single_shard_sha256"),
        "num_parameters": art.get("num_parameters"),
    }


def canonical_config_sha(config: Any) -> tuple[str, dict[str, Any]]:
    """Serialize exactly as a checkpoint does, and hash it the way identity does.

    `CheckpointIdentity` hashes `sha256_json(json.loads(config.json))`, so the
    file is written and read back rather than hashing an in-memory dict: what is
    on disk is what a later `from_pretrained` sees.
    """
    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        raw = json.loads((Path(tmp) / "config.json").read_text())
    return sha256_json(raw), raw


#: The ONE operator in the initialization path that mutates its parent model's
#: config: `DepthCausalKLGreedyV1.apply` sets `use_cache = False`, because
#: `bypassed_blocks` cannot work with a layer-indexed KV cache. The mutation
#: lands on the PARENT MODEL OBJECT and every child config inherits it through
#: `build_config`, which copies the parent's dict.
MUTATES_USE_CACHE = ("depth.causal_kl_greedy_v1",)


def replay_config_lineage(chain: list[dict[str, Any]], *, teacher_config,
                          adapter, root_use_cache: bool = True,
                          ) -> list[dict[str, Any]]:
    """Rebuild each step's config from its parent's, through the REAL adapter.

    `build_config` is pure: it takes the parent config and the child `ArchSpec`
    and returns the child config. No weights, no device, no operator arithmetic.
    That is what makes this answerable without a GPU.

    `root_use_cache` models what a REPLAY starts from. The replay loads a fresh
    teacher for every path, and the hub config says `use_cache: true`; the
    search reused ONE teacher object across every expansion, so the first
    causal-KL DEPTH expansion flipped it to False on that shared object and
    every later expansion — of any path — inherited it. The teacher's cache flag
    at a given expansion is therefore a function of BEAM ORDER, and it is an
    input to the path that the replay does not currently pin.
    """
    out: list[dict[str, Any]] = []
    parent_config = type(teacher_config).from_dict(teacher_config.to_dict())
    parent_config.use_cache = root_use_cache
    for index, row in enumerate(chain):
        rec = row_identity(row)
        spec_dict = rec["arch_spec"]
        if not spec_dict:
            out.append({**rec, "index": index, "reconstructed_config_sha256": None,
                        "match": None, "why": "the journal records no arch_spec"})
            continue
        try:
            child = adapter.build_config(parent_config,
                                         ArchSpec.of("qwen3", spec_dict))
            #: The operator's own mutation, applied to the CHILD it produced —
            #: `apply` runs against the model built from this config.
            impl = (rec["impl_ids"] or [None])[-1]
            if impl in MUTATES_USE_CACHE:
                child.use_cache = False
            sha, raw = canonical_config_sha(child)
            err = None
        except Exception as exc:                                # noqa: BLE001
            child, sha, raw, err = None, None, None, f"{type(exc).__name__}: {exc}"
        match = None if sha is None else (sha == rec["config_sha256"])
        out.append({**rec, "index": index,
                    "reconstructed_config_sha256": sha,
                    "match": match, "why": err})
        #: The chain continues from the RECONSTRUCTED config, because that is
        #: what a replay would actually carry forward. Continuing from a config
        #: rebuilt off the journal would hide a divergence that propagates.
        if child is not None:
            parent_config = child
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/stages/stage-1/phase_c2_replay/"
                                     "results/config_lineage_forensic.json")
    args = ap.parse_args(argv)

    from transformers import AutoConfig

    from aadistill.initialization.adapters import register_builtin_adapters

    from phase_a_frozen import TEACHER_ID, TEACHER_REVISION

    #: Explicit, as everywhere in this project: the registry is empty in a fresh
    #: process and a resolver that assumed somebody else had filled it is how a
    #: paid pod dies one step after a gate it passed.
    register_builtin_adapters()
    adapter = get_adapter("qwen3")
    teacher_config = AutoConfig.from_pretrained(TEACHER_ID,
                                                revision=TEACHER_REVISION)
    teacher_sha, _ = canonical_config_sha(teacher_config)

    states = load_full_journal()
    selection = RS.load_selection(REPO_ROOT)

    paths: list[dict[str, Any]] = []
    for entry in selection["selected"]:
        chain = lineage(states, entry["state_id"])
        #: What a REPLAY does today: a fresh teacher per path, hub default.
        steps = replay_config_lineage(chain, teacher_config=teacher_config,
                                      adapter=adapter, root_use_cache=True)
        #: What the SEARCH would have had once any causal-KL DEPTH expansion had
        #: already run on the shared teacher. Reported beside it so the reader
        #: can see that one bit accounts for the whole divergence.
        contaminated = replay_config_lineage(
            chain, teacher_config=teacher_config, adapter=adapter,
            root_use_cache=False)
        for a, b in zip(steps, contaminated):
            a["reconstructed_with_contaminated_root_sha256"] = \
                b["reconstructed_config_sha256"]
            a["match_with_contaminated_root"] = b["match"]
        first_bad = next((s for s in steps if s["match"] is False), None)
        unknown = [s for s in steps if s["match"] is None]
        paths.append({
            "state_id": entry["state_id"],
            "path": entry["path"],
            "n_steps": len(steps),
            "all_steps_match": all(s["match"] is True for s in steps),
            "first_divergence": (
                None if first_bad is None else {
                    "index": first_bad["index"],
                    "state_id": first_bad["state_id"],
                    "impl_id": (first_bad["impl_ids"] or [None])[-1],
                    "profile": (first_bad["calibration_profiles"] or [None])[-1],
                    "attempt3_config_sha256": first_bad["config_sha256"],
                    "reconstructed_config_sha256":
                        first_bad["reconstructed_config_sha256"],
                }),
            "unresolved_steps": [s["index"] for s in unknown],
            "all_steps_match_with_contaminated_root":
                all(s.get("match_with_contaminated_root") is True for s in steps),
            "root_use_cache_that_explains_attempt3": (
                True if all(s["match"] is True for s in steps)
                else False if all(s.get("match_with_contaminated_root") is True
                                  for s in steps)
                else None),
            "steps": steps,
        })

    doc = {
        "schema": "aadistill.autoinit.c2_config_lineage_forensic/v1",
        "_contract": (
            "A $0 CPU forensic. It replays only the CONFIG lineage of all five "
            "selected paths through the same `adapter.build_config` a real "
            "materialization uses, and compares each step against the "
            "config_sha256 attempt 3 recorded. It moves no weights, creates no "
            "provider resource and decides nothing about admissibility."),
        "_acceptance_rule_unchanged": (
            "`artifact_digest` remains the hard criterion. Identical tensors "
            "under a different config are not the same runtime model: the "
            "runtime loads through `from_pretrained(path)` and non-structural "
            "fields decide how those tensors are interpreted."),
        "sources": {
            "full_journal": str(FULL_JOURNAL),
            "full_journal_sha256": FULL_JOURNAL_SHA256,
            "selection_sha256": RS.SELECTION_SHA256,
            "source_session_commit": RS.ATTEMPT3_SESSION_COMMIT,
            "_lineage_derived_from": "the journal's own parent_id, not a path label",
        },
        "teacher": {"repo_id": TEACHER_ID, "revision": TEACHER_REVISION,
                    "config_sha256": teacher_sha},
        "environment": {
            "transformers": __import__("transformers").__version__,
            "torch": __import__("torch").__version__,
        },
        "paths": paths,
        "summary": {
            "paths_whose_config_lineage_fully_matches":
                [p["state_id"] for p in paths if p["all_steps_match"]],
            "paths_that_diverge":
                [{"state_id": p["state_id"],
                  "first_divergence_index": p["first_divergence"]["index"],
                  "impl_id": p["first_divergence"]["impl_id"]}
                 for p in paths if p["first_divergence"]],
            "paths_unresolved":
                [p["state_id"] for p in paths if p["unresolved_steps"]],
        },
    }
    doc["forensic_sha256"] = sha256_json(doc)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"teacher config {teacher_sha[:12]}…  transformers "
          f"{doc['environment']['transformers']}\n")
    for p in paths:
        head = "ALL MATCH" if p["all_steps_match"] else "DIVERGES"
        print(f"{p['state_id'][:12]}  {head}")
        for s in p["steps"]:
            mark = {True: "match", False: "MISMATCH", None: "?"}[s["match"]]
            impl = (s["impl_ids"] or ["?"])[-1]
            a3 = (s["config_sha256"] or "")[:12]
            rc = (s["reconstructed_config_sha256"] or "")[:12]
            print(f"    {s['index']} {impl:34s} a3={a3:12s} "
                  f"rebuilt={rc:12s} {mark}")
            if s.get("why"):
                print(f"        {s['why'][:150]}")
        print()
    print(json.dumps(doc["summary"], indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

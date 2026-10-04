#!/usr/bin/env python3
"""Build the six D-series behavioural batteries, in the frozen order.

    PYTHONPATH=src:scripts:scripts/data python -m \
        experiments.phase_d_series.build_batteries --write

**Six roles, one order, each excluding every role before it.** Distinct rank
domains give *independent* draws, not disjoint ones — two independent draws from
one pool overlap. Disjointness comes from building in the frozen order with each
role excluding all of its predecessors, which is how `c2_screening_v1` was drawn
against `c1_confirmation_v1`.

**All six are built BEFORE any D1 outcome exists.** That is the point of freezing
the family prospectively: once a result is visible, no later choice about which
prompts a confirmation rung sees can be shown to be outcome-independent.

**Selection is `rank_take`, the historical function.** Not a reimplementation: the
eligibility predicate (id-or-source_key exclusion, rendered-prompt-hash exclusion,
within-battery dedup) and the ordering are the ones the frozen C1 battery was
drawn with. The role's `rank_domain` is passed so each role gets an independent
ordering from the same frozen base digest.

**What this build adds on top, and only on top:**

* candidate rows carry the **D-series split-aware item identity**, because the
  historical schemes name the wrong thing for a new file — `make_gsm8k` builds
  its id from the row's POSITION, and `mbpp-test-<task_id>` names the wrong split
  for three of MBPP's four files;
* the **canonical problem-content coordinate** filters the pool before ranking,
  because the historical chain cannot see a restated problem and is measurably
  blind to recovery-training question content.

Both are bound in allocation rule **v3**, which is what makes them part of the
prospective freeze rather than an implementation detail.

**Historical renderers and frozen batteries are untouched.** Each candidate's
CONTENT — prompt text, gold, scorer fields — comes from the historical renderer
for that stratum, so the scorer sees exactly what it has always seen. Only the id
is the D-series one, and no frozen battery's item is renamed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from battery_render import (  # noqa: E402
    FROZEN_SOURCES,
    RENDERERS,
    norm,
    rank_take,
    read_rows,
    snapshot_path,
    source_digest,
)
from build_c1_confirmation_battery import C0_DIGEST  # noqa: E402

from experiments.phase_d_series import math_source as ms  # noqa: E402
from experiments.phase_d_series.battery_family import (  # noqa: E402
    FAMILY_ID,
    ROLES,
    allocation_rule,
    allocation_rule_id,
    strata,
)
from experiments.phase_d_series.identity import (  # noqa: E402
    d_series_item_id,
    excluded_by_review,
    problem_content_id,
)
from experiments.phase_d_series.source_evidence import (  # noqa: E402
    baseline_chain,
    d_series_additional,
    reserved_problem_content,
)

OUT = "artifacts/stage3/d_series_behavioural_v1"

#: The MANIFEST is a record and goes in `logs/`; the 22 MB of items are artifacts
#: and stay out of git. AGENTS.md 2.5: code, configs, manifests and small metadata
#: may be committed; large artifacts may not. The manifest carries each role's
#: item digest, so the committed record is enough to verify the artifacts are the
#: ones it describes.
MANIFEST = "logs/shared/analyses/autoinit_d_series_family_manifest.json"
SCHEMA = "aadistill.autoinit.d_series_battery/v1"

#: Every file each stratum draws from: the originally pinned file first, then the
#: extension files the 2026-10-03 source decision added. Order is part of the
#: build only in that it makes the pool deterministic; the RANK decides selection.
STRATUM_FILES: dict[str, tuple[tuple[str, str], ...]] = {
    "code": (("full", "test"), ("full", "train"), ("full", "validation"),
             ("full", "prompt")),
    "gsm8k": (("main", "test"), ("main", "train")),
    "knowledge": (("rc.nocontext", "validation"),),
    "multihop": (("distractor", "validation"),),
    "rag": (("squad_v2", "validation"),),
    "tool": (("xlam", "all"),),
}

#: `math_verified` is the one stratum whose two populations live in different
#: repositories, so it is assembled by its own adapter rather than from a file
#: list. MATH-500 remains pinned and is a measured subset of the canonical test
#: population, so the union is the canonical population and the subset's rows
#: reach the pool through it.
MATH_IS_ASSEMBLED_BY_ITS_ADAPTER = True


def _relpath(group: str, config: str, split: str) -> str | None:
    """The file inside the pinned snapshot, or None if this is the pinned file."""
    pinned = FROZEN_SOURCES[group][2]
    if group == "code":
        return f"{config}/{split}-00000-of-00001.parquet"
    if group == "gsm8k":
        return f"{config}/{split}-00000-of-00001.parquet"
    return pinned


def candidate_pool(group: str) -> list[dict[str, Any]]:
    """Every candidate of one stratum, rendered historically, identified by D-series.

    The content is the historical renderer's, verbatim — that is what keeps the
    scorer's semantics. The `id` is replaced by the split-aware D-series identity
    and `problem_content_id` is attached, which are the two things allocation rule
    v3 binds.
    """
    if group == "math_verified":
        return _math_pool()

    repo, revision, _pinned = FROZEN_SOURCES[group]
    make = RENDERERS[group]
    out: list[dict[str, Any]] = []
    for config, split in STRATUM_FILES[group]:
        rel = _relpath(group, config, split)
        if rel is None or not (snapshot_path(repo, revision) / rel).is_file():
            continue
        rows = read_rows(repo, revision, rel)
        for index, raw in enumerate(rows):
            row = dict(raw, _index=index) if group == "gsm8k" else raw
            item = make(row)
            if item is None:
                continue
            historical_id = str(item["id"])
            item["id"] = d_series_item_id(group, config, split, row, index=index)
            item["historical_render_id"] = historical_id
            item["_source_file"] = rel
            item["_config"], item["_split"], item["_row_index"] = config, split, index
            try:
                item["problem_content_id"] = problem_content_id(group, row)
            except (KeyError, ValueError):
                #: a stratum with no declared problem payload is isolated by the
                #: two historical coordinates only, and says so rather than
                #: pretending to a content id it cannot compute
                item["problem_content_id"] = None
            out.append(item)
    return out


def _math_pool() -> list[dict[str, Any]]:
    """The canonical MATH test population, in the frozen stratum's item shape."""
    out = []
    for row in ms.rows():
        text = row["problem"]
        out.append({
            "id": d_series_item_id("math_verified", row["_config"], row["_split"],
                                   row, index=row["_index"]),
            "group": "math_verified", "source": "hendrycks_math",
            "subject": row["subject"], "level": row["level"],
            "prompt_text": text,
            "messages": [{"role": "user", "content": text}],
            "gold": row["answer"], "boxed": row["answer"],
            "historical_render_id": None,
            "_source_file": f"{row['_config']}/{ms.TEST_FILE}",
            "_config": row["_config"], "_split": row["_split"],
            "_row_index": row["_index"],
            "problem_content_id": problem_content_id("math_verified", row),
        })
    return out


# --- the exclusion state ----------------------------------------------------

class Isolation:
    """Everything a role must avoid, growing as roles are built.

    Three coordinates, exactly the ones allocation rule v3 binds. The first two
    are handed to `rank_take`; the third filters the pool before ranking, which
    keeps the ordering a function of the pool and the frozen strings only.
    """

    def __init__(self, group: str) -> None:
        base_ids, base_hashes, self.baseline_provenance = baseline_chain()
        add_ids, add_hashes, self.d_series_provenance = d_series_additional(group)
        self.ids = set(base_ids) | set(add_ids)
        self.hashes = set(base_hashes) | set(add_hashes)
        reserved = reserved_problem_content(group)
        self.contents = set(reserved["ids"])
        self.reserved_provenance = reserved["provenance"]
        self.review_excluded = excluded_by_review(group)
        self.per_role: dict[str, int] = {}

    def admit(self, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Candidates this role may rank, in a deterministic order.

        Filters by the problem-content coordinate and the frozen review, then
        deduplicates by problem content so one problem cannot enter twice under
        two ids. Sorted by D-series id first, so the survivor of a within-pool
        content collision is a function of the ids and not of file order.
        """
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in sorted(pool, key=lambda i: str(i["id"])):
            if str(item["id"]) in self.review_excluded:
                continue
            #: THE HISTORICAL ID COORDINATE, APPLIED IN THE HISTORICAL NAMESPACE.
            #: The historical exclusion set holds ids like `squad-val-<key>`, and
            #: a D-series row's own id is `squad-squad_v2-validation-<key>` -- so
            #: `rank_take`'s id check cannot see it, and the row's bare
            #: `source_key` does not match either because FINAL_PROMOTION stores
            #: no `source_key`.
            #:
            #: The rendered-prompt coordinate does not cover the gap: `battery_v2`
            #: was built before `RAG_INSTRUCTION` was reworded, so its prompt
            #: hashes no longer match what the current renderer produces. The id
            #: coordinate was carrying that load, and changing the id scheme
            #: silently removed it. The independent verifier caught ten rows
            #: across `rag` and `multihop` that were already consumed.
            if str(item.get("historical_render_id") or "") in self.ids:
                continue
            content = item.get("problem_content_id")
            if content is not None:
                if content in self.contents or content in seen:
                    continue
                seen.add(content)
            kept.append(item)
        return kept

    def consume(self, role: str, items: list[dict[str, Any]]) -> None:
        """Fold a built role into the state, by all three coordinates."""
        for item in items:
            self.ids.add(str(item["id"]))
            if item.get("source_key") is not None:
                self.ids.add(str(item["source_key"]))
            self.hashes.add(content_sha256(norm(item["prompt_text"])))
            if item.get("problem_content_id"):
                self.contents.add(item["problem_content_id"])
        self.per_role[role] = len(items)


# --- the build --------------------------------------------------------------

def build() -> dict[str, Any]:
    """Six roles, in the frozen order. Returns the family document."""
    spec = strata()
    pools = {group: candidate_pool(group) for group in sorted(spec)}
    isolation = {group: Isolation(group) for group in sorted(spec)}

    roles: dict[str, dict[str, list[dict[str, Any]]]] = {}
    shortfalls: list[str] = []
    for role, domain, _experiment, _purpose in ROLES:
        built: dict[str, list[dict[str, Any]]] = {}
        for group, (_d, want, _scorable) in sorted(spec.items()):
            state = isolation[group]
            admissible = state.admit(pools[group])
            drawn = rank_take(
                admissible, want, stratum=group, base_digest=C0_DIGEST,
                exclude_ids=state.ids, exclude_hashes=state.hashes,
                make=lambda item: item, domain=domain)
            if len(drawn) != want:
                shortfalls.append(
                    f"{role}/{group}: drew {len(drawn)} of {want} from "
                    f"{len(admissible)} admissible candidates")
            state.consume(role, drawn)
            built[group] = drawn
        roles[role] = built
    if shortfalls:
        raise SystemExit(
            "the family cannot be built as specified:\n  " +
            "\n  ".join(shortfalls))
    return _document(roles, pools, isolation)


def _document(roles, pools, isolation) -> dict[str, Any]:
    spec = strata()
    per_role = {}
    for role, built in roles.items():
        items = [i for group in sorted(built) for i in built[group]]
        per_role[role] = {
            "n_prompts": len(items),
            "n_scorable": sum(len(built[g]) for g, (_d, _t, s) in spec.items() if s),
            "per_stratum": {g: len(built[g]) for g in sorted(built)},
            "item_ids_sha256": content_sha256(
                "\n".join(sorted(f"{i['id']}:{i['problem_content_id']}"
                                 for i in items))),
        }
    return {
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "allocation_rule_id": allocation_rule_id(),
        "allocation_rule_version": allocation_rule()["allocation_rule_version"],
        "_built_before_any_outcome": (
            "all six roles are drawn in one deterministic pass, before any D1 "
            "search has run. No later choice about which prompts a confirmation "
            "rung sees is available."),
        "roles": per_role,
        "role_order": [r[0] for r in ROLES],
        "candidate_pools": {g: len(p) for g, p in sorted(pools.items())},
        "sources": _source_manifest(),
        "code_state": code_state(REPO_ROOT),
    }


def _source_manifest() -> dict[str, Any]:
    """Per-file digests of every source a role may have drawn from."""
    out: dict[str, Any] = {}
    for group in sorted(STRATUM_FILES):
        repo, revision, _ = FROZEN_SOURCES[group]
        files = {}
        for config, split in STRATUM_FILES[group]:
            rel = _relpath(group, config, split)
            if rel is None or not (snapshot_path(repo, revision) / rel).is_file():
                continue
            files[rel] = source_digest(repo, revision, rel)
        out[group] = {"repo_id": repo, "revision": revision, "files": files}
    out["math_verified"] = ms.file_manifest()
    return out


def family_content_id(doc: dict[str, Any]) -> str:
    """`H(what was actually drawn)` — the realized counterpart to the rule id.

    Binds the rule id, every role's realized item/content list and every source
    file digest. The rule says HOW the family is drawn and can be hashed before
    the sources are read; this says WHAT was drawn and cannot.
    """
    return sha256_json({
        "allocation_rule_id": doc["allocation_rule_id"],
        "roles": {r: {"item_ids_sha256": v["item_ids_sha256"],
                      "per_stratum": v["per_stratum"]}
                  for r, v in sorted(doc["roles"].items())},
        "sources": doc["sources"],
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help=f"write the six roles under {OUT}")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args(argv)

    spec = strata()
    pools = {group: candidate_pool(group) for group in sorted(spec)}
    isolation = {group: Isolation(group) for group in sorted(spec)}
    roles: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role, domain, _e, _p in ROLES:
        built = {}
        for group, (_d, want, _s) in sorted(spec.items()):
            state = isolation[group]
            admissible = state.admit(pools[group])
            drawn = rank_take(admissible, want, stratum=group,
                              base_digest=C0_DIGEST, exclude_ids=state.ids,
                              exclude_hashes=state.hashes,
                              make=lambda item: item, domain=domain)
            if len(drawn) != want:
                raise SystemExit(
                    f"{role}/{group}: drew {len(drawn)} of {want} from "
                    f"{len(admissible)} admissible candidates")
            state.consume(role, drawn)
            built[group] = drawn
        roles[role] = built

    doc = _document(roles, pools, isolation)
    doc["family_content_id"] = family_content_id(doc)

    if args.write:
        root = REPO_ROOT / args.out
        root.mkdir(parents=True, exist_ok=True)
        for role, built in roles.items():
            role_dir = root / role
            role_dir.mkdir(exist_ok=True)
            for group, items in sorted(built.items()):
                (role_dir / f"{group}.jsonl").write_text(
                    "".join(json.dumps(i, sort_keys=True) + "\n" for i in items))
        (root / "family.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n")
        manifest = REPO_ROOT / MANIFEST
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out} (items) and {MANIFEST} (record)\n")

    print(f"D-series behavioural family — {doc['family_id']}\n")
    print(f"  allocation rule   : v{doc['allocation_rule_version']} "
          f"{doc['allocation_rule_id']}")
    print(f"  family content id : {doc['family_content_id'][:32]}")
    print(f"  candidate pools   : "
          f"{ {g: n for g, n in doc['candidate_pools'].items()} }")
    print()
    for role in doc["role_order"]:
        r = doc["roles"][role]
        print(f"  {role:18s} {r['n_prompts']:4d} prompts  "
              f"{r['n_scorable']:4d} scorable")
    print("\n  BUILT BEFORE ANY D1 OUTCOME. AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

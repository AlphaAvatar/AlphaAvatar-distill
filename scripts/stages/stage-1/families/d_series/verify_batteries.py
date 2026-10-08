#!/usr/bin/env python3
"""Verify the six built D-series batteries, independently of the builder.

    PYTHONPATH=src:scripts:scripts/data python -m \
        experiments.phase_d_series.verify_batteries

**Independently means it re-derives.** It reads the written items and recomputes
every property from them: the mixture counts, the 950/850 denominators, pairwise
disjointness across all six roles on all three isolation coordinates, isolation
against every historical reserved population, the per-role item digests, and the
family content id. It does not read the builder's own claims and check them
against themselves.

A verifier that imported the builder's selection would prove only that the
builder agrees with itself. The one thing it does share is the **source files and
the frozen rule** — the inputs — because a verifier that invented its own inputs
would be checking a different family.
"""
from __future__ import annotations

import argparse
import collections
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
from battery_render import norm  # noqa: E402

from experiments.phase_d_series.battery_family import (  # noqa: E402
    HELD_OUT_BATTERIES,
    ROLES,
    allocation_rule_id,
    strata,
)
from experiments.phase_d_series.build_batteries import (  # noqa: E402
    OUT,
    family_content_id,
)
from experiments.phase_d_series.identity import PROBLEM_FIELD  # noqa: E402
from experiments.phase_d_series.source_evidence import (  # noqa: E402
    BASELINE_INPUTS,
    D_SERIES_ADDITIONAL_POOLS,
    reserved_problem_content,
)


def load(root: Path) -> tuple[dict[str, dict[str, list[dict]]], dict[str, Any]]:
    """Every role's items, read from disk."""
    doc = json.loads((root / "family.json").read_text())
    roles: dict[str, dict[str, list[dict]]] = {}
    for role, _d, _e, _p in ROLES:
        per_stratum = {}
        for path in sorted((root / role).glob("*.jsonl")):
            per_stratum[path.stem] = [
                json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        roles[role] = per_stratum
    return roles, doc


def check_counts(roles) -> list[str]:
    """Requirement 10: the frozen mixture, and the 950/850 denominators."""
    problems = []
    spec = strata()
    want_total = sum(take for _d, take, _s in spec.values())
    want_scorable = sum(take for _d, take, s in spec.values() if s)
    for role, _d, _e, _p in ROLES:
        built = roles[role]
        if sorted(built) != sorted(spec):
            problems.append(f"{role}: strata {sorted(built)} != {sorted(spec)}")
            continue
        for group, (_domain, take, _scorable) in sorted(spec.items()):
            got = len(built[group])
            if got != take:
                problems.append(f"{role}/{group}: {got} items, frozen count {take}")
        total = sum(len(v) for v in built.values())
        scorable = sum(len(built[g]) for g, (_d, _t, s) in spec.items() if s)
        if total != want_total:
            problems.append(f"{role}: {total} prompts, expected {want_total}")
        if scorable != want_scorable:
            problems.append(
                f"{role}: {scorable} scorable, expected {want_scorable}")
        #: every item must carry a usable id and prompt
        for group, items in built.items():
            for item in items:
                if not str(item.get("id", "")).strip():
                    problems.append(f"{role}/{group}: an item has no id")
                    break
                if not str(item.get("prompt_text", "")).strip():
                    problems.append(f"{role}/{group}: {item['id']} has no prompt")
                    break
    return problems


def _coordinates(items) -> dict[str, set[str]]:
    return {
        "item_id": {str(i["id"]) for i in items},
        "rendered_prompt": {content_sha256(norm(i["prompt_text"])) for i in items},
        "problem_content": {i["problem_content_id"] for i in items
                            if i.get("problem_content_id")},
    }


def check_pairwise_disjointness(roles) -> list[str]:
    """Requirement 9a: no two of the six roles share anything, on any coordinate."""
    problems = []
    spec = strata()
    for group in sorted(spec):
        per_role = {role: _coordinates(roles[role][group])
                    for role, _d, _e, _p in ROLES}
        names = [r[0] for r in ROLES]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for axis in ("item_id", "rendered_prompt", "problem_content"):
                    shared = per_role[a][axis] & per_role[b][axis]
                    if shared:
                        problems.append(
                            f"{group}: {a} and {b} share {len(shared)} "
                            f"{axis}(s), e.g. {sorted(shared)[:1]}")
    return problems


def check_isolation_from_history(roles) -> list[str]:
    """Requirement 9b: isolation from every historical reserved population."""
    problems = []
    spec = strata()
    for group in sorted(spec):
        reserved_content = reserved_problem_content(group)["ids"]
        reserved_ids: set[str] = set()
        reserved_prompts: set[str] = set()
        for pool in (*D_SERIES_ADDITIONAL_POOLS, "recovery_search_v2"):
            path = REPO_ROOT / "artifacts/stage3" / pool / f"{group}.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                reserved_ids.add(str(row["id"]))
                if row.get("source_key") is not None:
                    reserved_ids.add(str(row["source_key"]))
                if row.get("prompt_text"):
                    reserved_prompts.add(content_sha256(norm(row["prompt_text"])))
        battery = REPO_ROOT / BASELINE_INPUTS["battery"] / f"{group}.jsonl"
        if battery.is_file():
            for line in battery.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                reserved_ids.add(str(row["id"]))
                if row.get("prompt_text"):
                    reserved_prompts.add(content_sha256(norm(row["prompt_text"])))

        for role, _d, _e, _p in ROLES:
            coords = _coordinates(roles[role][group])
            for axis, reserved in (("rendered_prompt", reserved_prompts),
                                   ("problem_content", reserved_content)):
                shared = coords[axis] & reserved
                if shared:
                    problems.append(
                        f"{group}/{role}: {len(shared)} {axis}(s) also in a "
                        f"historical reserved population")
            #: the id axis compares against the HISTORICAL ids, which the
            #: D-series rows do not use -- so the meaningful check is that no
            #: D-series row's historical render id is a reserved id.
            historical = {str(i["historical_render_id"])
                          for i in roles[role][group]
                          if i.get("historical_render_id")}
            shared = historical & reserved_ids
            if shared:
                problems.append(
                    f"{group}/{role}: {len(shared)} row(s) whose historical "
                    f"render id is a reserved id, e.g. {sorted(shared)[:1]}")
    return problems


def check_training_corpus(roles) -> list[str]:
    """No role may contain a problem the student trained on."""
    sessions = REPO_ROOT / BASELINE_INPUTS["sessions"]
    if not sessions.is_file():
        return ["the recovery-training corpus is absent, so this cannot be checked"]
    import hashlib

    train: set[str] = set()
    with sessions.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            for message in (json.loads(line).get("messages") or []):
                if message.get("role") == "user":
                    text = str(message.get("content") or "")
                    if text:
                        train.add(hashlib.sha256(norm(text).encode()).hexdigest())
                    break
    problems = []
    for role, _d, _e, _p in ROLES:
        for group, items in sorted(roles[role].items()):
            if group not in PROBLEM_FIELD:
                continue
            leaked = [i["id"] for i in items
                      if i.get("problem_content_id") in train]
            if leaked:
                problems.append(
                    f"{group}/{role}: {len(leaked)} problem(s) are in the "
                    f"recovery-training corpus, e.g. {leaked[:1]}")
    return problems


def check_output_digests(root: Path, doc) -> list[str]:
    """Every realized file's SHA256, recomputed from disk.

    The manifest must identify the actual 22 MB. Recomputed here from the bytes
    on disk rather than read back from the builder's claim, and the family
    content id is recomputed from those recomputed digests -- so a changed
    `prompt_text`, `gold` or tool schema fails even when every id is intact.
    """
    import hashlib

    problems = []
    claimed = doc.get("output_files")
    if not claimed:
        return ["the manifest carries no output_files; it cannot identify the "
                "family it was built from"]
    spec = strata()
    expected_n = len(ROLES) * len(spec)
    if len(claimed) != expected_n:
        problems.append(
            f"manifest names {len(claimed)} output files, expected "
            f"{expected_n} ({len(ROLES)} roles x {len(spec)} strata)")

    recomputed: dict[str, str] = {}
    for rel, got in sorted(claimed.items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: named by the manifest but not on disk")
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        recomputed[rel] = digest
        if digest != got["sha256"]:
            problems.append(
                f"{rel}: bytes hash {digest[:12]} but the manifest says "
                f"{got['sha256'][:12]}")
        if len(raw) != got["size_bytes"]:
            problems.append(f"{rel}: {len(raw)} bytes, manifest says "
                            f"{got['size_bytes']}")
        n = sum(1 for line in raw.decode().splitlines() if line.strip())
        if n != got["n_items"]:
            problems.append(f"{rel}: {n} items, manifest says {got['n_items']}")

    #: every file on disk must be named by the manifest, not only the reverse
    for role, _d, _e, _p in ROLES:
        for path in sorted((root / role).glob("*.jsonl")):
            rel = f"{role}/{path.name}"
            if rel not in claimed:
                problems.append(f"{rel}: on disk but not named by the manifest")

    #: and the family id must follow from the RECOMPUTED digests
    if recomputed and not problems:
        from experiments.phase_d_series.build_batteries import family_content_id

        rebuilt = family_content_id({
            "allocation_rule_id": doc["allocation_rule_id"],
            "output_files": {rel: {"sha256": d} for rel, d in recomputed.items()},
            "sources": doc["sources"]})
        if rebuilt != doc.get("family_content_id"):
            problems.append(
                f"family_content_id does not follow from the bytes on disk: "
                f"recomputed {rebuilt[:12]}, manifest "
                f"{str(doc.get('family_content_id'))[:12]}")
    return problems


def reserved_contract(group: str) -> dict[str, set[str]]:
    """The historical exclusion SET, as a contract input.

    Sharing the frozen SET DEFINITION is deliberate and allowed: a verifier that
    invented its own reserved populations would be checking a different contract.
    What it must not share is the builder's selection or admission RESULT, and it
    does not -- every candidate decision below is made here from these sets.

    This is the complete baseline, including the three populations the previous
    verifier did not check for the strata that have no problem payload: recovery
    training, operator calibration and state evaluation contribute `source_id`s
    that `rank_take` relies on, and for `rag`, `multihop`, `knowledge` and `tool`
    that native-identity coordinate is the only one with teeth.
    """
    from experiments.phase_d_series.source_evidence import (
        baseline_chain,
        d_series_additional,
    )

    base_ids, base_hashes, _prov = baseline_chain()
    add_ids, add_hashes, _add_prov = d_series_additional(group)
    return {
        "ids": set(base_ids) | set(add_ids),
        "prompt_hashes": set(base_hashes) | set(add_hashes),
        "problem_contents": set(reserved_problem_content(group)["ids"]),
    }


def check_every_item_against_the_contract(roles) -> list[str]:
    """Requirement 2: all four coordinates, every item, every stratum.

    The previous version checked rebuilt pools for the committed batteries and
    FINAL_PROMOTION, plus problem content for the three strata that declare a
    payload. That left the native/source-id protection for `rag`, `multihop`,
    `knowledge` and `tool` resting on the builder's word -- and those are exactly
    the strata where the id coordinate is the only one that bites.
    """
    problems = []
    spec = strata()
    for group in sorted(spec):
        reserved = reserved_contract(group)
        for role, _d, _e, _p in ROLES:
            for item in roles[role][group]:
                where = f"{group}/{role}/{item['id']}"
                historical = str(item.get("historical_render_id") or "")
                if historical and historical in reserved["ids"]:
                    problems.append(
                        f"{where}: historical_render_id {historical!r} is a "
                        "reserved id")
                key = item.get("source_key")
                if key is not None and str(key) in reserved["ids"]:
                    problems.append(
                        f"{where}: source_key {str(key)!r} is a reserved id")
                if content_sha256(norm(item["prompt_text"])) in reserved[
                        "prompt_hashes"]:
                    problems.append(
                        f"{where}: its rendered prompt is a reserved prompt")
                content = item.get("problem_content_id")
                if content and content in reserved["problem_contents"]:
                    problems.append(
                        f"{where}: its problem content is reserved")
                if len(problems) > 40:
                    return problems
    return problems


def check_manifest(roles, doc, root: Path | None = None) -> list[str]:
    """The committed manifest must describe the items on disk.

    `root` is required for the part that matters: the family id has to be
    recomputed from the BYTES, not from the manifest's own claim about them.
    Recomputing it from `doc["output_files"]` would pass for any corruption that
    edits the items and the claimed digests together -- and, worse, for one that
    edits only the items, because then nothing in the doc moved at all. A changed
    `gold` with an intact id is exactly that case.
    """
    problems = []
    if root is not None:
        problems += check_output_digests(root, doc)
    if doc.get("allocation_rule_id") != allocation_rule_id():
        problems.append(
            f"manifest names rule {doc.get('allocation_rule_id')} but the live "
            f"rule is {allocation_rule_id()}")
    for role, _d, _e, _p in ROLES:
        items = [i for group in sorted(roles[role]) for i in roles[role][group]]
        digest = content_sha256("\n".join(sorted(
            f"{i['id']}:{i['problem_content_id']}" for i in items)))
        claimed = doc["roles"][role]["item_ids_sha256"]
        if digest != claimed:
            problems.append(
                f"{role}: recomputed item digest {digest[:12]} != manifest "
                f"{claimed[:12]}")
        if doc["roles"][role]["n_prompts"] != len(items):
            problems.append(f"{role}: manifest n_prompts disagrees with disk")
    #: from the doc's own fields -- a self-consistency check only, which is why
    #: the disk-based recomputation above is the one that binds.
    recomputed = family_content_id(doc)
    if doc.get("family_content_id") != recomputed:
        problems.append(
            f"family_content_id {doc.get('family_content_id', '')[:12]} is not "
            f"self-consistent with the manifest's own fields ({recomputed[:12]})")
    return problems


def check_renderer_parity(roles) -> list[str]:
    """Requirement 3: every item's CONTENT is the historical renderer's.

    The D-series changes the id and adds a content coordinate. It must change
    nothing the scorer reads. This re-renders each item's source row with the
    historical renderer and compares every field except the ones the D-series
    deliberately owns -- so a drift in prompt text, gold, or any scorer field
    fails here rather than at scoring time.

    `math_verified` is exempt from re-rendering through `RENDERERS` and says so:
    its source is a different repository and `make_math_verified` reads a
    `unique_id` the canonical release does not have. Its parity is the adapter
    parity recorded in the source evidence -- `boxed_answer(solution)` reproducing
    the frozen stratum's gold on 500/500 rows.
    """
    from battery_render import FROZEN_SOURCES, RENDERERS, read_rows

    D_SERIES_OWNS = {"id", "historical_render_id", "problem_content_id",
                     "prompt_sha256", "_source_file", "_config", "_split",
                     "_row_index"}
    problems = []
    cache: dict[tuple[str, str], list[dict]] = {}
    for role, _d, _e, _p in ROLES:
        for group, items in sorted(roles[role].items()):
            if group == "math_verified":
                continue
            repo, revision, _pinned = FROZEN_SOURCES[group]
            make = RENDERERS[group]
            for item in items:
                rel = item["_source_file"]
                key = (group, rel)
                if key not in cache:
                    cache[key] = read_rows(repo, revision, rel)
                rows = cache[key]
                index = int(item["_row_index"])
                if index >= len(rows):
                    problems.append(
                        f"{group}/{role}/{item['id']}: row index {index} is "
                        f"outside {rel} ({len(rows)} rows)")
                    continue
                raw = rows[index]
                row = dict(raw, _index=index) if group == "gsm8k" else raw
                expected = make(row)
                if expected is None:
                    problems.append(
                        f"{group}/{role}/{item['id']}: the historical renderer "
                        "rejects the row this item claims to come from")
                    continue
                for field, value in expected.items():
                    if field in D_SERIES_OWNS:
                        continue
                    if item.get(field) != value:
                        problems.append(
                            f"{group}/{role}/{item['id']}: field {field!r} "
                            "differs from the historical rendering")
                        break
                if str(item.get("historical_render_id")) != str(expected["id"]):
                    problems.append(
                        f"{group}/{role}/{item['id']}: historical_render_id does "
                        "not match the historical renderer's id for its row")
    return problems


def verify(root: Path) -> dict[str, list[str]]:
    roles, doc = load(root)
    return {
        "counts_and_denominators": check_counts(roles),
        "renderer_scorer_parity": check_renderer_parity(roles),
        "pairwise_disjointness": check_pairwise_disjointness(roles),
        "contract_isolation_all_coordinates": (
            check_every_item_against_the_contract(roles)),
        "isolation_from_history": check_isolation_from_history(roles),
        "recovery_training": check_training_corpus(roles),
        "output_file_digests": check_output_digests(root, doc),
        "manifest": check_manifest(roles, doc, root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=OUT)
    args = parser.parse_args(argv)
    root = REPO_ROOT / args.root
    if not (root / "family.json").is_file():
        print(f"no family at {args.root}")
        return 1

    results = verify(root)
    print(f"verifying {args.root}\n")
    failed = 0
    for check, problems in results.items():
        status = "PASS" if not problems else f"FAIL ({len(problems)})"
        print(f"  {check:26s} {status}")
        for line in problems[:6]:
            print(f"      {line}")
        failed += len(problems)
    print()
    if failed:
        print(f"  {failed} problem(s). THE FAMILY IS NOT VERIFIED.")
        return 1
    print("  all checks pass. The six roles are pairwise disjoint on every "
          "frozen coordinate,")
    print("  isolated from every historical reserved population, and the "
          "manifest describes")
    print("  what is on disk. AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

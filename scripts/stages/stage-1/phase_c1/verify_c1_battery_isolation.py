"""Independent isolation check for the Phase-C1 confirmation battery.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_c1_battery_isolation.py

The builder enforces isolation while it selects. This re-derives it afterwards
from the artifact on disk, against the roles as they exist *now* — so a later
change to any excluded asset is caught, not merely a mistake at build time. It
recomputes the content hash too, because a battery whose prompts moved after
freezing is a different battery whatever its manifest says.

Two independent checks, deliberately not one:

* **stable id** — catches the same source item arriving under the same id;
* **normalized prompt content** — catches the same *question* arriving through
  another dataset path under a different id, which an id comparison cannot see.

Exit 0 on success; non-zero and a report on any violation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from battery_render import norm  # noqa: E402

BATTERY = REPO_ROOT / "artifacts/stage3/c1_confirmation_v1"

#: Every role the C1 battery must be disjoint from, and how to read its ids and
#: prompt text. `final_promotion` is read for exclusion only.
ROLES = {
    "FINAL_PROMOTION": ("artifacts/eval/battery_v2", "jsonl_dir"),
    "RECOVERY_SEARCH": ("artifacts/stage3/recovery_search_v2", "jsonl_dir"),
    "RECOVERY_TRAINING": ("artifacts/stage3/corpus_v2/sessions.jsonl", "sessions"),
    "STATE_EVALUATION": ("artifacts/stage1/state_eval_v1/items.jsonl", "items"),
    "OPERATOR_CALIBRATION": ("artifacts/stage1/e8_calibration_v1/items.jsonl", "items"),
}


#: The frozen per-role expectation, read from the battery's own manifest rather
#: than restated here. A role that is present but has been swapped for some other
#: nonempty asset would otherwise satisfy "nonempty" and prove nothing.
EXPECTED = {
    "FINAL_PROMOTION": ("final_promotion", "n_prompts", None),
    "RECOVERY_SEARCH": ("recovery_search", "n_prompts", None),
    "RECOVERY_TRAINING": ("recovery_training", "n_sessions", "distinct_source_ids"),
    "STATE_EVALUATION": ("initializer_state_eval", "n_items", "excluded_source_ids"),
    "OPERATOR_CALIBRATION": ("operator_calibration", "n_items", "excluded_source_ids"),
}


class RoleUnavailable(Exception):
    """A declared isolation role is absent, empty, or contributed no rows.

    Raised rather than returning empty sets, which is the whole repair. An absent
    `jsonl_dir` used to glob to nothing, yield empty `ids`/`hashes`, and report
    `id_collisions=0` — a PASS that had never read the role at all. Attempt 4's
    pytest suite had the same hole for `battery_v2`, and unlike the two cases that
    failed loudly it would have gone on reporting green indefinitely.
    """


def role_identities(rel: str, kind: str) -> tuple[set[str], set[str], int, int]:
    """Ids, content hashes, row count and file count — refusing an empty role."""
    ids: set[str] = set()
    hashes: set[str] = set()
    rows = 0
    files = 0
    target = REPO_ROOT / rel
    if not target.exists():
        raise RoleUnavailable(f"{rel} does not exist")
    if kind == "jsonl_dir":
        members = sorted(target.glob("*.jsonl"))
        files = len(members)
        if not members:
            raise RoleUnavailable(f"{rel} contains no *.jsonl files")
        for p in members:
            for line in p.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                rows += 1
                ids.add(str(r["id"]))
                if r.get("source_key"):
                    ids.add(str(r["source_key"]))
                hashes.add(content_sha256(norm(r.get("prompt_text", ""))))
    elif kind == "sessions":
        files = 1
        for line in target.open():
            if not line.strip():
                continue
            d = json.loads(line)
            rows += 1
            ids.add(str(d["source_id"]))
            hashes.add(content_sha256(norm("\n".join(
                str(m.get("content", "")) for m in d["messages"]
                if m.get("role") != "assistant"))))
    elif kind == "items":
        files = 1
        for line in target.open():
            if not line.strip():
                continue
            d = json.loads(line)
            rows += 1
            if d.get("source_id"):
                ids.add(str(d["source_id"]))
    else:
        raise ValueError(kind)
    if rows == 0:
        raise RoleUnavailable(f"{rel} yielded zero rows")
    if not ids:
        raise RoleUnavailable(f"{rel} yielded zero identities")
    return ids, hashes, rows, files


def main() -> int:
    manifest = json.loads((BATTERY / "manifest.json").read_text())
    items = []
    for name in manifest["sets"]:
        for line in (BATTERY / f"{name}.jsonl").open():
            if line.strip():
                items.append(json.loads(line))

    failures: list[str] = []

    # --- counts -------------------------------------------------------------
    counts = {n: sum(1 for i in items if i["group"] == n) for n in manifest["sets"]}
    if counts != manifest["mixture"]:
        failures.append(f"counts {counts} != frozen mixture {manifest['mixture']}")
    scorable = sum(1 for i in items
                   if manifest["sets"][i["group"]]["scorable"])
    if len(items) != 950 or scorable != 850:
        failures.append(f"got {len(items)} prompts / {scorable} scorable, want 950/850")

    # --- the artifact still hashes to what was frozen -----------------------
    pairs = sorted(f"{i['id']}:{i['prompt_sha256']}" for i in items)
    content = hashlib.sha256("\n".join(pairs).encode()).hexdigest()
    if content != manifest["content_sha256"]:
        failures.append(
            f"content hash {content} != frozen {manifest['content_sha256']}")

    # --- each item's stored prompt hash is really its prompt's hash ---------
    for i in items:
        if content_sha256(norm(i["prompt_text"])) != i["prompt_sha256"]:
            failures.append(f"{i['id']}: stored prompt_sha256 is not its prompt's hash")
            break

    # --- no duplicates inside the battery -----------------------------------
    ids = [str(i["id"]) for i in items]
    hashes = [i["prompt_sha256"] for i in items]
    if len(set(ids)) != len(ids):
        failures.append("duplicate ids inside the battery")
    if len(set(hashes)) != len(hashes):
        failures.append("duplicate prompt content inside the battery")

    # --- disjointness, twice, against every role ----------------------------
    own_ids = set(ids) | {str(i["source_key"]) for i in items if i.get("source_key")}
    own_hashes = set(hashes)
    report = {}
    isolation = manifest.get("isolation", {})
    for role, (rel, kind) in ROLES.items():
        try:
            r_ids, r_hashes, r_rows, r_files = role_identities(rel, kind)
        except RoleUnavailable as exc:
            # FAIL, never "zero collisions". Every declared role must genuinely
            # exist and be nonempty before it can contribute a PASS.
            failures.append(f"{role}: UNAVAILABLE — {exc}. A disjointness claim "
                            "against a role that was never read is vacuous.")
            report[role] = {"asset": rel, "available": False}
            continue
        # Cross-check the frozen expectation, so a present-but-substituted asset
        # cannot pass merely by being nonempty.
        key, rows_field, ids_field = EXPECTED[role]
        frozen = isolation.get(key, {})
        want_rows = frozen.get(rows_field)
        if want_rows is not None and r_rows != want_rows:
            failures.append(f"{role}: {r_rows} rows, frozen manifest says "
                            f"{rows_field}={want_rows}")
        if ids_field:
            want_ids = frozen.get(ids_field)
            if want_ids is not None and len(r_ids) != want_ids:
                failures.append(f"{role}: {len(r_ids)} identities, frozen manifest "
                                f"says {ids_field}={want_ids}")
        id_hits = sorted(own_ids & r_ids)
        hash_hits = sorted(own_hashes & r_hashes)
        report[role] = {"asset": rel, "available": True, "files": r_files,
                        "rows": r_rows, "role_ids": len(r_ids),
                        "role_prompt_hashes": len(r_hashes),
                        "id_collisions": len(id_hits),
                        "content_collisions": len(hash_hits)}
        if id_hits:
            failures.append(f"{role}: {len(id_hits)} stable-id collisions "
                            f"e.g. {id_hits[:3]}")
        if hash_hits:
            failures.append(f"{role}: {len(hash_hits)} normalized-content collisions")

    print(f"C1 confirmation battery: {len(items)} prompts, {scorable} scorable")
    print(f"content_sha256 {content} (matches manifest: "
          f"{content == manifest['content_sha256']})")
    for role, r in report.items():
        if not r.get("available"):
            print(f"  {role:22s} UNAVAILABLE — {r['asset']}")
            continue
        print(f"  {role:22s} rows={r['rows']:6d} ids={r['role_ids']:6d} "
              f"hashes={r['role_prompt_hashes']:6d} "
              f"-> id_collisions={r['id_collisions']} "
              f"content_collisions={r['content_collisions']}")
    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nPASS — all {len(ROLES)} roles present and nonempty, counts match the "
          "frozen manifest, and the battery is disjoint from every one of them by "
          "BOTH stable id and normalized content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

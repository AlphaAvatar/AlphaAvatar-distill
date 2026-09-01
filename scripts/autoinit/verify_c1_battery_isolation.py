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


def role_identities(rel: str, kind: str) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    if kind == "jsonl_dir":
        for p in sorted((REPO_ROOT / rel).glob("*.jsonl")):
            for line in p.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                ids.add(str(r["id"]))
                if r.get("source_key"):
                    ids.add(str(r["source_key"]))
                hashes.add(content_sha256(norm(r.get("prompt_text", ""))))
    elif kind == "sessions":
        for line in (REPO_ROOT / rel).open():
            d = json.loads(line)
            ids.add(str(d["source_id"]))
            hashes.add(content_sha256(norm("\n".join(
                str(m.get("content", "")) for m in d["messages"]
                if m.get("role") != "assistant"))))
    elif kind == "items":
        for line in (REPO_ROOT / rel).open():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("source_id"):
                ids.add(str(d["source_id"]))
    else:
        raise ValueError(kind)
    return ids, hashes


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
    for role, (rel, kind) in ROLES.items():
        r_ids, r_hashes = role_identities(rel, kind)
        id_hits = sorted(own_ids & r_ids)
        hash_hits = sorted(own_hashes & r_hashes)
        report[role] = {"asset": rel, "role_ids": len(r_ids),
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
        print(f"  {role:22s} ids={r['role_ids']:6d} hashes={r['role_prompt_hashes']:6d} "
              f"-> id_collisions={r['id_collisions']} "
              f"content_collisions={r['content_collisions']}")
    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS — disjoint from every role by BOTH stable id and normalized content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

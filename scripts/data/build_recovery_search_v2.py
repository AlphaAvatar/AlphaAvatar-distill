"""Build `recovery_search_v2` from v1, changing only the tools representation.

    PYTHONPATH=src .venv/bin/python scripts/data/build_recovery_search_v2.py

`recovery_search_v1` is invalid before first use. Its `tools` field retained the
xLAM source serialization — a JSON **string** — instead of the project's
canonical tool representation, so its 20 tool prompts could not be rendered at
all: `apply_chat_template` raises `ValueError: Tools should either be a JSON
schema, or a callable function ...`. It never produced a valid tool rollout, and
no valid tool characterization from it has ever existed. v1 is preserved
unmodified for provenance and marked superseded.

v2 is the semantic successor with the narrowest change that can fix it:

    identical item membership and ordering      identical item IDs
    identical messages and user prompts         identical non-tool items
    identical tool names and descriptions       identical parameter semantics
    identical gold calls and scorer inputs      identical scoring contract

**Only the materialized `tools` value changes**, from the serialized xLAM string
to the canonical envelope, via `aadistill.data.tools.xlam_tools_to_canonical` —
the same function `build_stage2_v1.build_xlam` uses for the training mixtures, so
the battery now shows the model the tool form it was trained on.

`content_sha256` is computed over `id:prompt_sha256` pairs, i.e. over prompt
content. It is therefore **expected to be identical to v1's**, and this builder
asserts that: it is the machine-checkable proof that membership, ordering, ids
and prompts were not touched. The new `tools_materialization_sha256` is what
distinguishes the two assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.data.tools import (  # noqa: E402
    canonical_tool_meaning, parse_xlam_tools, xlam_tools_to_canonical,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

V1 = REPO / "artifacts/stage3/recovery_search_v1"
V2 = REPO / "artifacts/stage3/recovery_search_v2"
TOOL_SET = "tool"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V2))
    ap.add_argument("--report", default="logs/autoinit_recovery_search_v2_build.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    v1_manifest = json.loads((V1 / "manifest.json").read_text())
    converted, unchanged_sets, per_item = 0, [], {}

    for name in sorted(v1_manifest["sets"]):
        rows = [json.loads(line) for line in (V1 / f"{name}.jsonl").read_text()
                .splitlines() if line.strip()]
        if name != TOOL_SET:
            # Byte-identical copy: the same rows, written by the same rule.
            (out / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
            unchanged_sets.append(name)
            continue
        new_rows = []
        for row in rows:
            raw = parse_xlam_tools(row["tools"], where=row["id"])
            canonical = xlam_tools_to_canonical(raw)
            if len(canonical) != len(raw):
                raise SystemExit(
                    f"{row['id']}: conversion dropped {len(raw) - len(canonical)} "
                    "tool(s); the item would test something different")
            before = canonical_tool_meaning(raw)
            after = canonical_tool_meaning(canonical)
            if before != after:
                raise SystemExit(
                    f"{row['id']}: tool meaning changed under conversion")
            new = dict(row)
            new["tools"] = canonical
            new_rows.append(new)
            per_item[row["id"]] = {"n_tools": len(canonical),
                                   "names": [t["function"]["name"] for t in canonical],
                                   "meaning_preserved": True}
            converted += 1
        new_rows.sort(key=lambda r: str(r["id"]))
        # Top-level keys sorted, matching v1's convention — but NOT `sort_keys`,
        # which would reorder the keys *inside* `tools` too. In v1 `tools` was a
        # string, so its interior was never touched; sorting it here silently
        # reordered parameter names, and the rendered <tools> blob lists them in
        # object order, so the model would have seen a different prompt. The
        # structural audit caught it; the fix is to preserve source order.
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps({k: r[k] for k in sorted(r)}, sort_keys=False) + "\n"
                    for r in new_rows))

    # --- manifest: v1's, with the identity fields recomputed ----------------
    manifest = json.loads(json.dumps(v1_manifest))   # deep copy
    manifest["artifact"] = "recovery_search_v2"
    manifest["version"] = 2
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["command"] = "scripts/data/build_recovery_search_v2.py"
    manifest["purpose"] = v1_manifest["purpose"]
    manifest["supersedes"] = {
        "artifact": "recovery_search_v1",
        "content_sha256": v1_manifest["content_sha256"],
        "manifest_sha256": v1_manifest["manifest_sha256"],
        "status": "INVALID / never produced a valid rollout",
        "defect": ("the `tools` field retained the xLAM source serialization (a "
                   "JSON string) instead of the project's canonical tool "
                   "representation, so apply_chat_template could render none of "
                   "the 20 tool prompts"),
        "no_result_rewritten": ("no historical recovery-search result is being "
                                "rewritten, because no valid v1 tool "
                                "characterization ever existed"),
    }
    manifest["change_from_v1"] = {
        "materialized_tools_representation": (
            "serialized xLAM string -> canonical OpenAI-style envelope via "
            "aadistill.data.tools.xlam_tools_to_canonical, the same converter "
            "scripts/data/build_stage2_v1.py uses for the training mixtures"),
        "items_converted": converted,
        "sets_copied_unchanged": unchanged_sets,
        "unchanged": ["item membership", "item ordering", "item ids", "messages",
                      "user prompts", "prompt_sha256", "tool names",
                      "tool descriptions", "parameter names/types/defaults",
                      "required semantics", "gold_tool_calls", "reference_calls",
                      "scorer_tools", "non-tool items", "metrics",
                      "recovery_search_scoring@v2"],
    }
    for name in sorted(v1_manifest["sets"]):
        path = out / f"{name}.jsonl"
        manifest["sets"][name] = {**v1_manifest["sets"][name],
                                  "path": str(path.relative_to(REPO)),
                                  "sha256": sha256_file(path)}

    rows_by_set = {n: [json.loads(l) for l in (out / f"{n}.jsonl").read_text()
                       .splitlines() if l.strip()]
                   for n in sorted(v1_manifest["sets"])}
    content_hash = hashlib.sha256(
        "".join(f"{i['id']}:{i['prompt_sha256']}\n"
                for name in sorted(rows_by_set) for i in rows_by_set[name])
        .encode()).hexdigest()
    if content_hash != v1_manifest["content_sha256"]:
        raise SystemExit(
            "content_sha256 moved: prompts, ids, membership or ordering changed. "
            "This build is only permitted to change the tools representation.\n"
            f"  v1 {v1_manifest['content_sha256']}\n  v2 {content_hash}")
    manifest["content_sha256"] = content_hash
    manifest["content_sha256_note"] = (
        "identical to v1 by construction and asserted at build time: this hash "
        "covers id:prompt_sha256 pairs, so equality is the proof that membership, "
        "ordering, ids and prompts were not touched")
    manifest["tools_materialization_sha256"] = hashlib.sha256(
        "".join(f"{i['id']}:{json.dumps(i['tools'], sort_keys=True)}\n"
                for i in rows_by_set[TOOL_SET]).encode()).hexdigest()
    manifest["code_state"] = code_state(str(REPO))
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "schema": "aadistill.autoinit.recovery_search_v2_build/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "converted_items": converted,
        "sets_copied_unchanged": unchanged_sets,
        "content_sha256": content_hash,
        "content_sha256_equals_v1": True,
        "tools_materialization_sha256": manifest["tools_materialization_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "per_item": per_item,
    }
    (REPO / args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("converted_items", "content_sha256_equals_v1",
                       "content_sha256", "tools_materialization_sha256",
                       "manifest_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

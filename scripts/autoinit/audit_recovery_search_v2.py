"""Prove `recovery_search_v2` changed the tools representation and nothing else.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/audit_recovery_search_v2.py

Zero cost, no GPU, and it runs over the **whole** asset — all 20 tool items and
all seven sets, never a sample. Three questions, answered separately:

1. **Is anything other than the tools representation different?** Every non-tool
   set must be byte-identical to v1, and inside the tool set every field except
   `tools` must be byte-identical.
2. **Did the tools keep their meaning?** Count, order, names, descriptions and
   the entire parameter map — including types, defaults and therefore the
   required semantics derived from them — compared structurally between v1's
   parsed form and v2's canonical form.
3. **Do they render?** All 20 must produce a prompt under the canonical
   evaluator, and the resulting model-visible prompt and token-id hashes are
   frozen here so a later change to the template, tokenizer or representation is
   detectable.
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
    canonical_tool_meaning, parse_xlam_tools,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

V1 = REPO / "artifacts/stage3/recovery_search_v1"
V2 = REPO / "artifacts/stage3/recovery_search_v2"
TOKENIZER = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
SYSTEM = "You are a helpful Assistant."
TOOL_SET = "tool"


def rows(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / f"{name}.jsonl").read_text()
            .splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_recovery_search_v2_audit.json")
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer

    m1 = json.loads((V1 / "manifest.json").read_text())
    m2 = json.loads((V2 / "manifest.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    problems: list[str] = []

    report = {
        "schema": "aadistill.autoinit.recovery_search_v2_audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "transformers": transformers.__version__,
        "tokenizer": str(TOKENIZER.relative_to(REPO)),
        "chat_template_sha256": hashlib.sha256(
            (tok.chat_template or "").encode()).hexdigest(),
        "v1": {"content_sha256": m1["content_sha256"],
               "manifest_sha256": m1["manifest_sha256"]},
        "v2": {"content_sha256": m2["content_sha256"],
               "manifest_sha256": m2["manifest_sha256"],
               "tools_materialization_sha256": m2["tools_materialization_sha256"]},
        "membership": {}, "tool_items": {}, "rendering": {},
    }

    # --- 1. nothing but the tools representation moved ----------------------
    if m1["content_sha256"] != m2["content_sha256"]:
        problems.append("content_sha256 differs: prompts or membership changed")
    if sorted(m1["sets"]) != sorted(m2["sets"]):
        problems.append("the two assets declare different sets")
    for name in sorted(m1["sets"]):
        a, b = rows(V1, name), rows(V2, name)
        entry = {"n_v1": len(a), "n_v2": len(b),
                 "ids_identical": [r["id"] for r in a] == [r["id"] for r in b]}
        if name == TOOL_SET:
            other_fields_identical = all(
                {k: v for k, v in x.items() if k != "tools"}
                == {k: v for k, v in y.items() if k != "tools"}
                for x, y in zip(a, b))
            entry["every_field_except_tools_identical"] = other_fields_identical
            if not other_fields_identical:
                problems.append(f"{name}: a field other than `tools` changed")
        else:
            identical = sha256_file(V1 / f"{name}.jsonl") == \
                sha256_file(V2 / f"{name}.jsonl")
            entry["file_byte_identical"] = identical
            if not identical:
                problems.append(f"{name}: a non-tool set is not byte-identical")
        if not entry["ids_identical"] or entry["n_v1"] != entry["n_v2"]:
            problems.append(f"{name}: membership or ordering changed")
        report["membership"][name] = entry

    # --- 2. the tools kept their meaning ------------------------------------
    for old, new in zip(rows(V1, TOOL_SET), rows(V2, TOOL_SET)):
        before = canonical_tool_meaning(parse_xlam_tools(old["tools"],
                                                         where=old["id"]))
        after = canonical_tool_meaning(new["tools"])
        checks = {
            "count": len(before) == len(after),
            "names_in_order": [t["name"] for t in before] == [t["name"] for t in after],
            "descriptions": [t["description"] for t in before]
                            == [t["description"] for t in after],
            # The whole parameter map, compared structurally: names, types,
            # descriptions, defaults — and therefore the required semantics that
            # are derived from the presence of a default.
            "parameters_deep_equal": [t["parameters"] for t in before]
                                     == [t["parameters"] for t in after],
            # Dict equality ignores key order, and the rendered <tools> blob
            # lists parameters in object order — so order is model-visible and
            # is checked on its own. This check is why the first v2 build was
            # rejected: `sort_keys=True` had silently alphabetised them.
            "parameter_order_preserved": [list(t["parameters"] or {}) for t in before]
                                         == [list(t["parameters"] or {}) for t in after],
            "envelope_is_canonical": all(
                t.get("type") == "function" and set(t.get("function", {}))
                >= {"name", "description", "parameters"} for t in new["tools"]),
        }
        params = [p for t in before for p in (t["parameters"] or {})]
        checks["parameter_names_preserved"] = params == [
            p for t in after for p in (t["parameters"] or {})]
        checks["defaults_preserved"] = (
            [(n, s.get("default")) for t in before
             for n, s in (t["parameters"] or {}).items() if isinstance(s, dict)]
            == [(n, s.get("default")) for t in after
                for n, s in (t["parameters"] or {}).items() if isinstance(s, dict)])
        checks["types_preserved"] = (
            [(n, s.get("type")) for t in before
             for n, s in (t["parameters"] or {}).items() if isinstance(s, dict)]
            == [(n, s.get("type")) for t in after
                for n, s in (t["parameters"] or {}).items() if isinstance(s, dict)])
        report["tool_items"][old["id"]] = {
            "n_tools": len(after), "checks": checks,
            "all_preserved": all(checks.values())}
        if not all(checks.values()):
            problems.append(f"{old['id']}: tool meaning not preserved "
                            f"({[k for k, v in checks.items() if not v]})")

    # --- 3. every tool item renders, and the result is frozen ---------------
    frozen = {}
    for row in rows(V2, TOOL_SET):
        turns = [m for m in row["messages"] if m["role"] != "assistant"]
        if not any(m.get("role") == "system" for m in turns):
            turns = [{"role": "system", "content": SYSTEM}] + turns
        try:
            text = tok.apply_chat_template(turns, tools=row["tools"],
                                           tokenize=False,
                                           add_generation_prompt=True)
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"{row['id']}: does not render ({type(exc).__name__})")
            frozen[row["id"]] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        frozen[row["id"]] = {
            "ok": True,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "n_tokens": len(ids), "n_chars": len(text)}
    report["rendering"] = {
        "rendered": sum(1 for v in frozen.values() if v["ok"]),
        "of": len(frozen),
        "note": ("model-visible prompt and token-id hashes, frozen here. They "
                 "are a property of (asset, tokenizer, chat template, "
                 "transformers version) and are recorded with all four."),
        "per_item": frozen,
    }
    if report["rendering"]["rendered"] != report["rendering"]["of"]:
        problems.append("not every tool item renders")

    report["problems"] = problems
    report["passed"] = not problems
    report["report_sha256"] = sha256_json(report)
    (REPO / args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "tool_items_all_preserved": all(v["all_preserved"]
                                        for v in report["tool_items"].values()),
        "rendered": f"{report['rendering']['rendered']}/{report['rendering']['of']}",
        "content_sha256_equal": m1["content_sha256"] == m2["content_sha256"],
        "non_tool_sets_byte_identical": all(
            e.get("file_byte_identical", True)
            for e in report["membership"].values()),
        "problems": problems[:5],
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

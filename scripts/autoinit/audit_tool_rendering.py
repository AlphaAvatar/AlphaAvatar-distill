"""Exact-equivalence audit of the tool-rendering path. Zero cost, no GPU.

Stage 3 of the micro-preflight died rendering a tool prompt:

    apply_chat_template(turns, tools=<the frozen battery's stored value>)
    ValueError: Tools should either be a JSON schema, or a callable function ...

`recovery_search_v1` stores `tools` as a **JSON string** of xLAM-style objects
(`{"name", "description", "parameters"}` with a flat `parameters` map). Every
other tool asset in this project — the Stage-2 training, validation and
calibration splits — stores a **list** of OpenAI-style
`{"type": "function", "function": {...}}` entries.

This script renders every tool item of the frozen battery under each candidate
form and records the exact prompt text, its sha256, the token ids and their
sha256, so the question "are these the same input to the model" is answered by
comparison rather than by argument. It renders **the whole frozen tool subset**,
not a sample.

Arms:

    raw_string      the stored value, passed through untouched
    parsed_list     json.loads of that value: a list of xLAM dicts
    openai_schema   parsed, then converted to the project's own tool form
    no_tools        the same turns with tools omitted, as a floor

Run it in each environment and compare the two reports:

    .venv/bin/python scripts/autoinit/audit_tool_rendering.py --out A.json
    /home/ecs-user/AlphaAvatar/.venv/bin/python \
        scripts/autoinit/audit_tool_rendering.py --out B.json
    .venv/bin/python scripts/autoinit/audit_tool_rendering.py --compare A.json B.json

Only the tokenizer and its chat template are loaded — never a model config — so
the transformers-4.x RoPE misreading (logs/STATE.md §0.5) cannot apply here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BATTERY = REPO / "artifacts/stage3/recovery_search_v1/tool.jsonl"
TOKENIZER = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
SYSTEM = "You are a helpful Assistant."


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def as_openai_schema(xlam_tools: list) -> list:
    """The project's own tool form, for reference.

    Mirrors what every other tool asset in this repository stores. `parameters`
    becomes a JSON Schema object; a parameter with no `default` is required.
    Type names are carried through unchanged rather than translated: this arm
    exists to be *compared*, not to be adopted silently.
    """
    out = []
    for tool in xlam_tools:
        params = tool.get("parameters") or {}
        properties, required = {}, []
        for name, spec in params.items():
            spec = dict(spec) if isinstance(spec, dict) else {"description": spec}
            if "default" not in spec:
                required.append(name)
            properties[name] = spec
        out.append({"type": "function", "function": {
            "name": tool.get("name"), "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": properties,
                           "required": required}}})
    return out


def render(tok, turns, tools):
    """One rendering attempt. A failure is recorded, never raised."""
    try:
        text = tok.apply_chat_template(turns, tools=tools, tokenize=False,
                                       add_generation_prompt=True)
    except Exception as exc:                                      # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    ids = tok(text, add_special_tokens=False).input_ids
    return {"ok": True, "text_sha256": sha(text), "n_chars": len(text),
            "ids_sha256": sha(json.dumps(ids)), "n_tokens": len(ids),
            "text": text}


def audit(args) -> int:
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = [json.loads(line) for line in BATTERY.read_text().splitlines()
            if line.strip()]
    report = {
        "schema": "aadistill.autoinit.tool_rendering_audit/v1",
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "tokenizer": str(TOKENIZER.relative_to(REPO)),
        "chat_template_sha256": sha(tok.chat_template or ""),
        "battery": str(BATTERY.relative_to(REPO)),
        "n_items": len(rows),
        "loads_model_config": False,
        "items": {},
    }
    for row in rows:
        turns = [m for m in row["messages"] if m["role"] != "assistant"]
        if not any(m.get("role") == "system" for m in turns):
            turns = [{"role": "system", "content": SYSTEM}] + turns
        stored = row.get("tools")
        try:
            parsed = json.loads(stored) if isinstance(stored, str) else stored
            parse_error = None
        except Exception as exc:                                  # noqa: BLE001
            parsed, parse_error = None, f"{type(exc).__name__}: {exc}"
        entry = {
            "stored_type": type(stored).__name__,
            "parse_error": parse_error,
            "n_tools": len(parsed) if isinstance(parsed, list) else None,
            "arms": {
                "raw_string": render(tok, turns, stored),
                "no_tools": render(tok, turns, None),
            },
        }
        if isinstance(parsed, list):
            entry["arms"]["parsed_list"] = render(tok, turns, parsed)
            entry["arms"]["openai_schema"] = render(
                tok, turns, as_openai_schema(parsed))
        report["items"][row["id"]] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    summary = {arm: sum(1 for e in report["items"].values()
                        if e["arms"].get(arm, {}).get("ok"))
               for arm in ("raw_string", "parsed_list", "openai_schema", "no_tools")}
    print(json.dumps({"transformers": report["transformers"],
                      "items": report["n_items"], "rendered_ok": summary}, indent=1))
    return 0


def compare(args) -> int:
    a = json.loads(Path(args.compare[0]).read_text())
    b = json.loads(Path(args.compare[1]).read_text())
    arms = ("raw_string", "parsed_list", "openai_schema", "no_tools")
    result = {
        "schema": "aadistill.autoinit.tool_rendering_equivalence/v1",
        "left": {"file": args.compare[0], "transformers": a["transformers"]},
        "right": {"file": args.compare[1], "transformers": b["transformers"]},
        "chat_template_identical":
            a["chat_template_sha256"] == b["chat_template_sha256"],
        "n_items": a["n_items"],
        "per_arm_cross_library": {},
        "cross_arm_within_library": {},
        "disagreements": [],
    }
    if set(a["items"]) != set(b["items"]):
        result["disagreements"].append("the two reports cover different items")

    # Does an arm render identically in both libraries?
    for arm in arms:
        same_text = same_ids = both_ok = 0
        for item, left in a["items"].items():
            right = b["items"].get(item, {})
            la = left["arms"].get(arm, {})
            ra = right.get("arms", {}).get(arm, {})
            if la.get("ok") and ra.get("ok"):
                both_ok += 1
                same_text += la["text_sha256"] == ra["text_sha256"]
                same_ids += la["ids_sha256"] == ra["ids_sha256"]
        result["per_arm_cross_library"][arm] = {
            "rendered_in_both": both_ok, "identical_text": same_text,
            "identical_token_ids": same_ids,
            "exactly_equivalent": both_ok == a["n_items"] and same_text == both_ok
                                  and same_ids == both_ok,
            "left_ok": sum(1 for e in a["items"].values()
                           if e["arms"].get(arm, {}).get("ok")),
            "right_ok": sum(1 for e in b["items"].values()
                            if e["arms"].get(arm, {}).get("ok")),
        }

    # Within one library, do two arms produce the same input to the model?
    for report, side in ((a, "left"), (b, "right")):
        pairs = {}
        for x in arms:
            for y in arms:
                if x >= y:
                    continue
                both = same = 0
                for entry in report["items"].values():
                    ax, ay = entry["arms"].get(x, {}), entry["arms"].get(y, {})
                    if ax.get("ok") and ay.get("ok"):
                        both += 1
                        same += ax["ids_sha256"] == ay["ids_sha256"]
                if both:
                    pairs[f"{x} vs {y}"] = {"comparable": both, "identical": same}
        result["cross_arm_within_library"][side] = pairs

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1)[:2600])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    return compare(args) if args.compare else audit(args)


if __name__ == "__main__":
    raise SystemExit(main())

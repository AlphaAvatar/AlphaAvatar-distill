"""Can the existing tool-call scorer consume the recovery battery? CPU only.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/audit_tool_scoring.py

The question is narrow and has to be answered before the control is measured,
because the answer changes the scorable denominator and therefore the recovery
policy hash.

``evaluation/behavior.py:score_tool_call`` already exists, is deterministic and is
tested. It expects OpenAI-style envelopes:

    tools      [{"function": {"name": ..., "parameters": {"required": [...]}}}]
    gold_calls [{"function": {"name": ..., "arguments": {...}}}]

The battery's xLAM items store something else:

    tools      [{"name": ..., "parameters": {"<arg>": {"type":..., "default":...}}}]
    answers    [{"name": ..., "arguments": {...}}]

So the audit is not "does it run" but "can it be connected **without inventing
semantics**". Wrapping a bare dict in a ``{"function": ...}`` envelope is
mechanical. Deriving ``required`` is not: xLAM has no required list, and the only
available signal is whether a parameter carries a ``default``. Treating
"no default" as "required" is an *interpretation of someone else's schema
convention*, and it decides whether `tool_args_schema_ok` fires.

This script runs the scorer against the real battery items and against six
adversarial cases, then reports whether the connection is mechanical or
interpretive. It changes no scorer and writes no new one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.behavior import parse_tool_calls, score_tool_call  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402


def as_openai_tools(xlam_tools: list[dict], required_from_missing_default: bool):
    """Translate an xLAM tool list into what the scorer expects."""
    out = []
    for t in xlam_tools:
        params = t.get("parameters") or {}
        required = ([name for name, spec in params.items()
                     if isinstance(spec, dict) and "default" not in spec]
                    if required_from_missing_default else [])
        out.append({"function": {"name": t.get("name"),
                                 "parameters": {"properties": params,
                                                "required": required}}})
    return out


def as_openai_calls(xlam_calls: list[dict]):
    return [{"function": {"name": c.get("name"), "arguments": c.get("arguments")}}
            for c in xlam_calls]


def render(calls: list[dict]) -> str:
    return "".join(f"<tool_call>{json.dumps(c)}</tool_call>" for c in calls)


def adversarial_cases(gold: list[dict], tools: list[dict]):
    """The six cases the audit must distinguish."""
    good = render(gold)
    first = gold[0]
    wrong_name = render([{**first, "name": "definitely_not_a_declared_tool"}])
    missing_args = render([{"name": first["name"], "arguments": {}}])
    wrong_values = render([{"name": first["name"],
                            "arguments": {k: "__wrong__" for k in first["arguments"]}}])
    malformed = "<tool_call>{not valid json,,}</tool_call>"
    protocol_invalid = json.dumps(first)          # no <tool_call> wrapper at all
    return {
        "known_good": good,
        "malformed_json": malformed,
        "wrong_tool_name": wrong_name,
        "missing_required_args": missing_args,
        "wrong_argument_values": wrong_values,
        "protocol_invalid_no_wrapper": protocol_invalid,
    }


EXPECTED = {
    # case -> the verdict fields that must hold for the scorer to be usable
    "known_good": {"tool_call_emitted": True, "tool_call_parsed": True,
                   "tool_name_valid": True, "tool_call_exact_match": True},
    "malformed_json": {"tool_call_emitted": True, "tool_call_parsed": False,
                       "tool_call_exact_match": False},
    "wrong_tool_name": {"tool_call_parsed": True, "tool_name_valid": False,
                        "tool_call_exact_match": False},
    "missing_required_args": {"tool_call_parsed": True, "tool_name_valid": True,
                              "tool_call_exact_match": False},
    "wrong_argument_values": {"tool_call_parsed": True, "tool_name_valid": True,
                              "tool_call_exact_match": False},
    "protocol_invalid_no_wrapper": {"tool_call_emitted": False,
                                    "tool_call_exact_match": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="artifacts/stage3/recovery_search_v1")
    ap.add_argument("--out", default="logs/autoinit_tool_scoring_audit.json")
    args = ap.parse_args()

    path = REPO_ROOT / args.battery / "tool.jsonl"
    items = [json.loads(l) for l in path.open() if l.strip()]

    parse_failures, results = [], []
    required_signal = {"has_default": 0, "no_default": 0}
    multi_call = 0

    for item in items:
        try:
            tools = json.loads(item["tools"]) if isinstance(item["tools"], str) \
                else item["tools"]
            gold = json.loads(item["reference_calls"]) \
                if isinstance(item["reference_calls"], str) else item["reference_calls"]
        except Exception as exc:
            parse_failures.append({"id": item["id"], "error": str(exc)[:120]})
            continue
        if not gold:
            parse_failures.append({"id": item["id"], "error": "no reference calls"})
            continue
        if len(gold) > 1:
            multi_call += 1
        for t in tools:
            for spec in (t.get("parameters") or {}).values():
                key = "has_default" if isinstance(spec, dict) and "default" in spec \
                    else "no_default"
                required_signal[key] += 1

        for derive_required in (True, False):
            oa_tools = as_openai_tools(tools, derive_required)
            oa_gold = as_openai_calls(gold)
            for case, answer in adversarial_cases(gold, tools).items():
                verdict = score_tool_call(answer, oa_tools, oa_gold)
                results.append({"id": item["id"], "case": case,
                                "required_from_missing_default": derive_required,
                                **verdict})

    # Did every case behave as the audit requires?
    disagreements = []
    for row in results:
        for field, want in EXPECTED[row["case"]].items():
            if row[field] != want:
                disagreements.append({"id": row["id"], "case": row["case"],
                                      "field": field, "got": row[field],
                                      "expected": want,
                                      "required_from_missing_default":
                                          row["required_from_missing_default"]})

    # Does the `required` interpretation change any verdict? If it does, the
    # connection is interpretive rather than mechanical.
    by_key = {}
    for row in results:
        by_key.setdefault((row["id"], row["case"]), {})[
            row["required_from_missing_default"]] = row
    interpretation_sensitive = []
    for (item_id, case), pair in by_key.items():
        if len(pair) != 2:
            continue
        a, b = pair[True], pair[False]
        differing = [f for f in ("tool_call_emitted", "tool_call_parsed",
                                 "tool_name_valid", "tool_args_schema_ok",
                                 "tool_call_exact_match") if a[f] != b[f]]
        if differing:
            interpretation_sensitive.append({"id": item_id, "case": case,
                                             "fields": differing,
                                             "with_required": {f: a[f] for f in differing},
                                             "without_required": {f: b[f] for f in differing}})

    exact_only_sensitive = all(
        set(d["fields"]) <= {"tool_args_schema_ok"} for d in interpretation_sensitive)

    verdict = {
        "scorer": "aadistill.evaluation.behavior.score_tool_call",
        "scorer_sha256": sha256_file(REPO_ROOT / "src/aadistill/evaluation/behavior.py"),
        "n_items": len(items),
        "parse_failures": parse_failures,
        "multi_call_items": multi_call,
        "required_signal_counts": required_signal,
        "cases_run": sorted(EXPECTED),
        "disagreements": disagreements,
        "all_cases_behave_as_required": not disagreements,
        "interpretation_sensitive_verdicts": len(interpretation_sensitive),
        "interpretation_affects_only_schema_field": exact_only_sensitive,
        "examples_of_sensitivity": interpretation_sensitive[:5],
    }

    # The decision.
    mechanical = (not parse_failures and not disagreements
                  and exact_only_sensitive)
    verdict["connection_is_mechanical"] = mechanical
    verdict["decision"] = (
        "MAKE TOOL SCORABLE on tool_call_exact_match: the envelope translation is "
        "mechanical, exact match never depends on the `required` interpretation, "
        "and every adversarial case is distinguished correctly."
        if mechanical else
        "KEEP TOOL BEHAVIOUR-ONLY: connecting the scorer requires interpreting "
        "xLAM's parameter conventions, which is a new semantic rule.")
    verdict["correctness_field_if_scorable"] = "tool_call_exact_match"
    verdict["fields_not_used_for_correctness"] = [
        "tool_args_schema_ok (depends on the `required` interpretation; reported "
        "as a diagnostic, never as correctness)"]
    verdict["generated_utc"] = datetime.now(timezone.utc).isoformat()
    verdict["audit_sha256"] = sha256_json(verdict)

    out = REPO_ROOT / args.out
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("examples_of_sensitivity", "disagreements",
                                   "parse_failures")}, indent=2))
    if disagreements:
        print(f"\n{len(disagreements)} disagreements, first 3:")
        for d in disagreements[:3]:
            print("  ", d)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline validity of `schema_synth` tasks, through the pool's own verifier. $0, dev box, no teacher.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth_audit \\
        --tasks artifacts/stages/stage-2/teacher_modes/schema_synth_v1/tasks.jsonl \\
        --write logs/stages/stage-2/teacher_modes/audit/schema_synth_v1_audit.json

Per task, with the code every admission uses (`tool_episodes`, `aadistill.data.tool_calls`):

* the record is runnable (`task_problems`), its schemas use only checked keywords, and the gold call is schema-valid;
* the replay backend answers the gold call; an episode that makes exactly the gold call and then answers is ADMITTED
  in thinking and in non-thinking mode, at protocol + schema validity + recorded execution + step gold;
* the same episode is REFUSED when the call differs from the gold in one leaf value, drops a required field, adds an
  optional field the request never mentions, or reorders an ordered array -- the gold admits exactly one call;
* every string the gold copies is in the request (or the system line), unless it is an enum value or a value the
  tool description tells the caller to derive (a date, time, E.164 phone, digits-only or uppercase normalization);
  anything else is reported as unexplained.

It also regenerates the tasks and requires the same bytes (determinism), and records the size of what a teacher
would read and write: the gold call's tokens in the hybrid's tokenizer, argument counts, nesting depth, arrays,
optional fields present, the operators and the difficulty levels. It measures no teacher: pass rates are priors.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.stage2.teacher_modes.schema_synth_audit/v1"
TOKENIZER = REPO / "artifacts/stages/stage-2/teacher_modes/tokenizers/qwen3_4b_1cfa9a72"
DERIVED = [re.compile(p) for p in (r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", r"^[0-9]{2}:[0-9]{2}$", r"^\+[1-9][0-9]{7,14}$",
                                   r"^[0-9]+$", r"^[A-Z0-9]+$", r"^[a-z]{2}$", r"^[A-Z]{2}$")]


def leaves(v, path="$"):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from leaves(x, f"{path}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from leaves(x, f"{path}[{i}]")
    else:
        yield path, v


def get_schema(params: dict, path: str) -> dict:
    node = params
    for part in re.findall(r"\.([A-Za-z_0-9]+)|\[\d+\]", path):
        if part:
            node = (node.get("properties") or {}).get(part, {})
        else:
            node = node.get("items", {})
    return node


def set_path(obj, path: str, value):
    parts = re.findall(r"\.([A-Za-z_0-9]+)|\[(\d+)\]", path)
    node = obj
    for key, idx in parts[:-1]:
        node = node[key] if key else node[int(idx)]
    key, idx = parts[-1]
    if key:
        node[key] = value
    else:
        node[int(idx)] = value


def mutate_leaf(v, schema: dict):
    if "enum" in schema:
        others = [x for x in schema["enum"] if x != v]
        return others[0] if others else None
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, str):
        return v + "x" if not schema.get("pattern") else None
    return None


def episode(call: dict | None, response: str | None, *, thinking: bool) -> list[dict]:
    think = "Let me work out the arguments.</think>\n\n" if thinking else ""
    first = think + ("<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>" if call else "Done.")
    return [{"raw": first, "stop_reason": "eos", "responses": [response] if call else []},
            {"raw": ("All set.</think>\n\n" if thinking else "") + "It is booked.", "stop_reason": "eos",
             "responses": []}]


def check(task: dict, tok) -> dict:
    from aadistill.data.tool_calls import ToolCall, call_problems, tool_index, unchecked_keywords
    from stages.teacher_modes import tool_episodes as TE
    out = {"problems": []}
    out["problems"] += TE.task_problems(task)
    for t in task["tools"]:
        if unchecked_keywords(t["function"]["parameters"]):
            out["problems"].append(f"unchecked keywords in {t['function']['name']}")
    call = task["gold_steps"][0][0][0]
    tools = tool_index(task["tools"])
    out["problems"] += [f"gold not schema-valid: {p}" for p in call_problems(ToolCall(call["name"], call["arguments"]),
                                                                          tools)]
    backend = TE.backend_of(task)
    resp = backend(ToolCall(call["name"], call["arguments"]))
    if resp is None:
        out["problems"].append("the backend does not answer the gold call")
    for thinking in (True, False):
        v = TE.verify(task, episode(call, resp, thinking=thinking), thinking=thinking)
        if not v.admitted or v.checks.get("step_gold") is not True or v.checks.get("recorded_execution") is not True:
            out["problems"].append(f"gold episode not admitted (thinking={thinking}): {v.problems[:2]}")
    params = next(t["function"]["parameters"] for t in task["tools"] if t["function"]["name"] == call["name"])
    mutations = {}
    lv = [(p, x) for p, x in leaves(call["arguments"])]
    for p, x in lv:
        m = mutate_leaf(x, get_schema(params, p))
        if m is not None:
            bad = copy.deepcopy(call)
            set_path(bad["arguments"], p, m)
            mutations["leaf_value"] = bad
            break
    req = [k for k in params.get("required", []) if k in call["arguments"]]
    if req:
        bad = copy.deepcopy(call)
        del bad["arguments"][req[-1]]
        mutations["drop_required"] = bad
    omitted = [k for k in params["properties"] if k not in call["arguments"]]
    if omitted:
        k = omitted[0]
        sch = params["properties"][k]
        val = {"boolean": True, "integer": 1, "string": "x", "array": [], "object": {}}.get(sch.get("type"))
        if "enum" in sch:
            val = sch["enum"][0]
        bad = copy.deepcopy(call)
        bad["arguments"][k] = val
        mutations["add_unrequested_optional"] = bad
    for p, x in leaves({"_": call["arguments"]}):
        pass
    for k, x in call["arguments"].items():
        if isinstance(x, list) and len(x) >= 2 and x != x[::-1]:
            bad = copy.deepcopy(call)
            bad["arguments"][k] = x[::-1]
            mutations["reorder_array"] = bad
            break
    refused = {}
    for name, bad in mutations.items():
        r = backend(ToolCall(bad["name"], bad["arguments"]))
        v = TE.verify(task, episode(bad, r, thinking=False), thinking=False)
        refused[name] = not v.admitted
    out["mutations_refused"] = refused
    if not all(refused.values()):
        out["problems"].append(f"a mutated call was admitted: {refused}")
    text = task["system"] + "\n" + "\n".join(m["content"] for m in task["messages"])
    why = Counter()
    unexplained = []
    for p, x in lv:
        if not isinstance(x, str):
            why["number_or_bool"] += 1
        elif "enum" in get_schema(params, p):
            why["enum"] += 1
        elif x in text:
            why["verbatim_in_request"] += 1
        elif any(rx.match(x) for rx in DERIVED):
            why["derived_by_stated_rule"] += 1
        else:
            unexplained.append((p, x))
    out["strings"] = dict(why)
    out["unexplained"] = unexplained
    call_text = "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    out["gold_call_tokens"] = len(tok(call_text, add_special_tokens=False).input_ids)
    out["leaves"] = len(lv)

    def depth(v):
        return 1 + max((depth(x) for x in (v.values() if isinstance(v, dict) else v)), default=0) \
            if isinstance(v, (dict, list)) else 0
    out["depth"] = depth(call["arguments"])
    out["arrays"] = sum(isinstance(x, list) for _, x in leaves({"a": call["arguments"]})) + \
        sum(1 for p, _ in lv if "[" in p and p.endswith("]"))
    out["optional_present"] = sum(1 for k in call["arguments"] if k not in params.get("required", []))
    out["user_chars"] = len(task["messages"][0]["content"])
    return out


def audit(tasks_path: Path) -> dict:
    from transformers import AutoTokenizer
    from stages.teacher_modes import schema_synth as SS
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    raw = tasks_path.read_bytes()
    tasks = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    regen = "".join(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in SS.generate())
    per_domain = defaultdict(lambda: defaultdict(list))
    problems, unexplained, refused = [], [], Counter()
    strings = Counter()
    for t in tasks:
        c = check(t, tok)
        if c["problems"]:
            problems.append({"task_id": t["task_id"], "problems": c["problems"]})
        unexplained += [{"task_id": t["task_id"], "path": p, "value": v} for p, v in c["unexplained"]]
        for k, v in c["mutations_refused"].items():
            refused[f"{k}:{'refused' if v else 'ADMITTED'}"] += 1
        strings.update(c["strings"])
        d = per_domain[(t["source"]["config"], t["split"])]
        for k in ("gold_call_tokens", "leaves", "depth", "optional_present", "user_chars"):
            d[k].append(c[k])
        d["level"].append(t["source"]["generator"]["level"])
        d["ops"].extend(t["source"]["generator"]["operators"])

    def stats(xs):
        xs = sorted(xs)
        return {"mean": round(statistics.fmean(xs), 1), "median": xs[len(xs) // 2], "min": xs[0], "max": xs[-1]}
    domains = {}
    for (dom, split), d in sorted(per_domain.items()):
        domains[dom] = {"split": split, "tasks": len(d["level"]), "levels": dict(Counter(d["level"])),
                        **{k: stats(d[k]) for k in ("gold_call_tokens", "leaves", "depth", "optional_present",
                                                    "user_chars")}}
    by_split = defaultdict(list)
    for (dom, split), d in per_domain.items():
        by_split[split] += d["gold_call_tokens"]
    ops = Counter()
    for d in per_domain.values():
        ops.update(d["ops"])
    return {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tasks": {"path": str(tasks_path.relative_to(REPO) if tasks_path.is_relative_to(REPO) else tasks_path),
                      "sha256": hashlib.sha256(raw).hexdigest(), "count": len(tasks)},
            "generator_sha256": hashlib.sha256((REPO / "scripts/stages/stage-2/teacher_modes/schema_synth.py")
                                               .read_bytes()).hexdigest(),
            "deterministic": hashlib.sha256(regen.encode()).hexdigest() == hashlib.sha256(raw).hexdigest(),
            "tasks_with_problems": len(problems), "problems": problems[:20],
            "mutations": dict(sorted(refused.items())),
            "gold_strings": dict(strings), "unexplained_strings": len(unexplained),
            "unexplained_examples": unexplained[:20],
            "gold_call_tokens_by_split": {s: stats(v) for s, v in sorted(by_split.items())},
            "operators": dict(ops.most_common()), "domains": domains,
            "measures_no_teacher": "pass rates and supervised tokens per admitted episode are priors until a "
                                   "generation round measures them",
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--write", type=Path)
    args = ap.parse_args(argv)
    rep = audit(args.tasks.resolve())
    text = json.dumps(rep, indent=1, ensure_ascii=False) + "\n"
    if args.write:
        args.write.write_text(text)
    print(json.dumps({k: rep[k] for k in ("tasks", "deterministic", "tasks_with_problems", "mutations",
                                          "gold_strings", "unexplained_strings", "gold_call_tokens_by_split")},
                     indent=1))
    return 0 if rep["deterministic"] and not rep["tasks_with_problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

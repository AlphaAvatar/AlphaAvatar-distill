"""Build the Stage 2 offline warm-up mixture v0 (`stage2_offline_v0`).

Usage:
    uv run python scripts/build_stage2_v0.py

Data groups (AGENTS.md 4.4) and sources, all public and permissively licensed:

    instruction         databricks-dolly-15k (long partition), oasst2 en threads
    short_realtime      databricks-dolly-15k (short, no-context partition)
    rag_evidence        squad_v2 answerable (context-grounded QA)
    refusal_uncertainty squad_v2 unanswerable ("not in context" answers) + v0 handcrafted
    multihop_qa         hotpot_qa distractor (10-paragraph evidence, answer)
    tool_calling        glaive-function-calling-v2 (converted to Qwen3 tool schema) + v0
    code_math           gsm8k, mbpp
    long_context        fineweb-edu docs of 8k-24k chars (disjoint from holdout_v1)

The quantization-calibration set is the `calib` split: a small deterministic
stratified slice across every group (it mirrors the deployment mixture rather
than being its own group). No teacher-generated data is included in v0.

Deterministic: every source is pinned to an exact dataset revision and read in
native order; selection is "first N passing filters" under per-source char
budgets; the train/val/calib split is a fixed modular rule on the final
per-group order. Output jsonl files are gitignored; the committed manifest
pins revisions, licenses, and hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data import GROUPS, validate_sample
from aadistill.env import code_state
from aadistill.manifest import sha256_file, write_manifest

OUT_DIR = REPO_ROOT / "data/stage2"
MIXTURE = "stage2_offline_v0"

MSG_CHAR_CAP = 6000      # per message
SAMPLE_CHAR_CAP = 12000  # per sample (all groups except long_context)
LONG_DOC_MIN = 8000
LONG_DOC_CAP = 24000
FINEWEB_SKIP = 10000     # warmup_v1 used ~first 900 positions, holdout_v1 ~5000-5150

# Split rule: per group, on the final deterministic sample order.
SPLIT_MOD, CALIB_SLOT, VAL_SLOT, CALIB_MAX = 25, 12, 24, 16

RAG_PROMPT = (
    "Answer the question using only the provided context. If the context does "
    "not contain the answer, say you cannot answer from the context.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
UNANSWERABLE_RESPONSES = [
    "The provided context does not contain the information needed to answer this question.",
    "I can't answer that from the given context — the passage doesn't mention it.",
    "This isn't stated in the provided context, so I can't give a grounded answer.",
]
HOTPOT_PROMPT = (
    "Answer the question using the provided paragraphs. Some paragraphs may be "
    "irrelevant. Give the answer directly.\n\n{paragraphs}\n\nQuestion: {question}"
)

DOLLY_SHORT_INSTR_CAP = 300
DOLLY_SHORT_RESP_CAP = 400


def sample_chars(sample: dict) -> int:
    if sample["format"] == "text":
        return len(sample["text"])
    chars = 0
    for msg in sample["messages"]:
        chars += len(msg.get("content", ""))
        if msg.get("tool_calls"):
            chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
    if sample.get("tools"):
        chars += len(json.dumps(sample["tools"], ensure_ascii=False))
    return chars


class Sink:
    """Budget-tracking collector shared by all source builders."""

    def __init__(self):
        self.samples: dict[str, list[dict]] = defaultdict(list)
        self.budgets: dict[str, int] = {}
        self.counters: dict[str, dict[str, int]] = {}
        self._seen: set[str] = set()
        self._source = None

    def start_source(self, name: str, budgets: dict[str, int]):
        self._source = name
        self.budgets = dict(budgets)
        self.counters[name] = defaultdict(int)

    def done(self) -> bool:
        return all(v <= 0 for v in self.budgets.values())

    def note(self, key: str):
        self.counters[self._source][key] += 1

    def add(self, group: str, sample: dict) -> bool:
        c = self.counters[self._source]
        if self.budgets.get(group, 0) <= 0:
            return False
        chars = sample_chars(sample)
        if group != "long_context" and chars > SAMPLE_CHAR_CAP:
            c["rejected_too_long"] += 1
            return False
        try:
            validate_sample(sample)
        except ValueError:
            c["rejected_schema_or_marker"] += 1
            return False
        digest = hashlib.sha256(
            json.dumps(sample, sort_keys=True, ensure_ascii=False)
            .replace(sample["id"], "").encode()
        ).hexdigest()
        if digest in self._seen:
            c["skipped_duplicates"] += 1
            return False
        self._seen.add(digest)
        self.samples[group].append(sample)
        self.budgets[group] -= chars
        c[f"taken_{group}"] += 1
        c[f"chars_{group}"] += chars
        return True


def _cap(text: str) -> str:
    return text.strip()[:MSG_CHAR_CAP]


def build_dolly(rows, sink: Sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        instruction = row["instruction"].strip()
        context = row.get("context", "").strip()
        response = row["response"].strip()
        if not instruction or not response:
            continue
        short = (not context and len(instruction) <= DOLLY_SHORT_INSTR_CAP
                 and len(response) <= DOLLY_SHORT_RESP_CAP)
        group = "short_realtime" if short else "instruction"
        user = instruction if not context else f"{instruction}\n\n{context}"
        sink.add(group, {
            "id": f"dolly-{idx:06d}", "group": group, "source": "dolly",
            "format": "chat",
            "messages": [{"role": "user", "content": _cap(user)},
                         {"role": "assistant", "content": _cap(response)}],
        })


def build_oasst2(rows, sink: Sink):
    rows = list(rows)
    children: dict[str, list[dict]] = defaultdict(list)
    roots = []
    for r in rows:
        if r["deleted"] or r["lang"] != "en" or not r["text"].strip():
            continue
        if r["parent_id"] is None:
            if r["tree_state"] == "ready_for_export" and r["role"] == "prompter":
                roots.append(r)
        else:
            children[r["parent_id"]].append(r)
    for idx, root in enumerate(roots):
        if sink.done():
            break
        thread, cur = [root], root
        while True:
            kids = children.get(cur["message_id"], [])
            if not kids:
                break
            ranked = [k for k in kids if k["rank"] is not None]
            cur = min(ranked, key=lambda k: k["rank"]) if ranked else kids[0]
            thread.append(cur)
        while thread and thread[-1]["role"] != "assistant":
            thread.pop()
        if len(thread) < 2:
            continue
        roles = ["user" if m["role"] == "prompter" else "assistant" for m in thread]
        if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
            continue  # malformed tree: same role twice in a row
        sink.add("instruction", {
            "id": f"oasst2-{idx:06d}", "group": "instruction", "source": "oasst2",
            "format": "chat",
            "messages": [{"role": role, "content": _cap(m["text"])}
                         for role, m in zip(roles, thread)],
        })


def build_squad(rows, sink: Sink):
    n_unanswerable = 0
    for idx, row in enumerate(rows):
        if sink.done():
            break
        answerable = bool(row["answers"]["text"])
        user = RAG_PROMPT.format(context=row["context"].strip(),
                                 question=row["question"].strip())
        if answerable:
            group, answer = "rag_evidence", row["answers"]["text"][0].strip()
        else:
            group = "refusal_uncertainty"
            answer = UNANSWERABLE_RESPONSES[n_unanswerable % 3]
            n_unanswerable += 1
        sink.add(group, {
            "id": f"squad_v2-{idx:06d}", "group": group, "source": "squad_v2",
            "format": "chat",
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}],
        })


def build_hotpot(rows, sink: Sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        paragraphs = "\n\n".join(
            f"[{i + 1}] {title}\n{' '.join(s.strip() for s in sents)}"
            for i, (title, sents) in enumerate(
                zip(row["context"]["title"], row["context"]["sentences"]))
        )
        user = HOTPOT_PROMPT.format(paragraphs=paragraphs,
                                    question=row["question"].strip())
        sink.add("multihop_qa", {
            "id": f"hotpot-{idx:06d}", "group": "multihop_qa", "source": "hotpot_qa",
            "format": "chat",
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": row["answer"].strip()}],
        })


def _extract_json_objects(text: str) -> list[dict]:
    """Extract top-level {...} blocks with a string-aware brace scanner."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return [o for o in objs if isinstance(o, dict)]


_FUNCALL_RE = re.compile(
    r'^\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}$',
    re.DOTALL,
)


def _parse_functioncall(payload: str):
    """Glaive functioncall payloads quote `arguments` in single quotes."""
    try:
        obj = json.loads(payload)
        args = obj.get("arguments")
        if isinstance(args, str):
            args = json.loads(args)
        if isinstance(obj.get("name"), str) and isinstance(args, dict):
            return obj["name"], args
    except json.JSONDecodeError:
        pass
    m = _FUNCALL_RE.match(payload.strip())
    if m:
        try:
            return m.group(1), json.loads(m.group(2))
        except json.JSONDecodeError:
            return None
    return None


_GLAIVE_SPLIT = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE): ")


def build_glaive(rows, sink: Sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        if "following functions" not in row["system"]:
            continue  # no-tools chit-chat rows; the instruction group covers those
        tools = [{"type": "function", "function": o}
                 for o in _extract_json_objects(row["system"]) if "name" in o]
        if not tools:
            sink.note("parse_failures")
            continue
        parts = _GLAIVE_SPLIT.split(row["chat"])
        if len(parts) < 3 or parts[0].strip():
            sink.note("parse_failures")
            continue
        messages, ok = [], True
        for marker, seg in zip(parts[1::2], parts[2::2]):
            seg = seg.replace("<|endoftext|>", "").strip()
            if not seg:
                ok = False
                break
            if marker == "USER":
                messages.append({"role": "user", "content": seg})
            elif marker == "FUNCTION RESPONSE":
                messages.append({"role": "tool", "content": seg})
            elif seg.startswith("<functioncall>"):
                call = _parse_functioncall(seg[len("<functioncall>"):].strip())
                if call is None:
                    ok = False
                    break
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function",
                     "function": {"name": call[0], "arguments": call[1]}}]})
            elif "<functioncall>" in seg:
                ok = False  # mixed text + call turns are rare; skip rather than guess
                break
            else:
                messages.append({"role": "assistant", "content": seg})
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()
        if not ok or not messages or messages[0]["role"] != "user":
            sink.note("parse_failures")
            continue
        sink.add("tool_calling", {
            "id": f"glaive-{idx:06d}", "group": "tool_calling",
            "source": "glaive_fc_v2", "format": "chat",
            "tools": tools, "messages": messages,
        })


def build_gsm8k(rows, sink: Sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        sink.add("code_math", {
            "id": f"gsm8k-{idx:06d}", "group": "code_math", "source": "gsm8k",
            "format": "chat",
            "messages": [{"role": "user", "content": row["question"].strip()},
                         {"role": "assistant", "content": row["answer"].strip()}],
        })


def build_mbpp(rows, sink: Sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        user = row["text"].strip()
        tests = "\n".join(row.get("test_list", []))
        if tests:
            user += "\n\nYour code should pass these tests:\n" + tests
        sink.add("code_math", {
            "id": f"mbpp-{idx:06d}", "group": "code_math", "source": "mbpp",
            "format": "chat",
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": row["code"].strip()}],
        })


def load_holdout_prefixes() -> set[str]:
    path = REPO_ROOT / "data/warmup/holdout_v1.jsonl"
    if not path.exists():
        print("WARNING: holdout_v1.jsonl not found; relying on stream offset only")
        return set()
    return {json.loads(l)["text"][:1000]
            for l in path.read_text().splitlines() if l.strip()}


def build_fineweb_long(rows, sink: Sink):
    holdout = load_holdout_prefixes()
    for idx, row in enumerate(rows):
        if sink.done():
            break
        if idx < FINEWEB_SKIP:
            continue
        text = row["text"].strip()
        if len(text) < LONG_DOC_MIN or text[:1000] in holdout:
            continue
        if len(text) > LONG_DOC_CAP:
            cut = text.rfind(" ", 0, LONG_DOC_CAP)
            text = text[:cut if cut > 0 else LONG_DOC_CAP]
        sink.add("long_context", {
            "id": f"fineweb-{idx:06d}", "group": "long_context",
            "source": "fineweb_edu_long", "format": "text", "text": text,
        })


def build_v0_handcrafted(rows, sink: Sink):
    for row in rows:
        group = row.get("category")
        if sink.done() or group not in ("refusal_uncertainty", "tool_calling"):
            continue
        sample = {"id": f"v0-{row['id']}", "group": group,
                  "source": "warmup_v0_handcrafted", "format": row["format"]}
        if row["format"] == "chat":
            sample["messages"] = row["messages"]
        else:
            sample["text"] = row["text"]
        sink.add(group, sample)


SOURCES = [
    # (name, dataset, config, split, license, streaming, builder, {group: char budget})
    ("dolly", "databricks/databricks-dolly-15k", None, "train", "CC-BY-SA 3.0",
     False, build_dolly, {"instruction": 3_200_000, "short_realtime": 800_000}),
    ("oasst2", "OpenAssistant/oasst2", None, "train", "Apache-2.0",
     False, build_oasst2, {"instruction": 2_500_000}),
    ("squad_v2", "rajpurkar/squad_v2", None, "train", "CC-BY-SA 4.0",
     False, build_squad, {"rag_evidence": 2_500_000, "refusal_uncertainty": 800_000}),
    ("hotpot_qa", "hotpotqa/hotpot_qa", "distractor", "train", "CC-BY-SA 4.0",
     True, build_hotpot, {"multihop_qa": 3_000_000}),
    ("glaive_fc_v2", "glaiveai/glaive-function-calling-v2", None, "train", "Apache-2.0",
     True, build_glaive, {"tool_calling": 2_500_000}),
    ("gsm8k", "openai/gsm8k", "main", "train", "MIT",
     False, build_gsm8k, {"code_math": 2_500_000}),
    ("mbpp", "google-research-datasets/mbpp", "full", "train", "CC-BY-4.0",
     False, build_mbpp, {"code_math": 400_000}),
    ("fineweb_edu_long", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "ODC-By 1.0",
     True, build_fineweb_long, {"long_context": 3_000_000}),
    ("warmup_v0_handcrafted", "data/warmup/warmup_v0.jsonl", None, None,
     "project-authored (license-clean, see v0 record)", False, build_v0_handcrafted,
     {"refusal_uncertainty": 100_000, "tool_calling": 100_000}),
]


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    sink = Sink()
    source_records = []

    for name, dataset, config, split, license_, streaming, builder, budgets in SOURCES:
        sink.start_source(name, budgets)
        if dataset.endswith(".jsonl"):
            revision = sha256_file(REPO_ROOT / dataset)
            rows = [json.loads(l) for l in (REPO_ROOT / dataset).read_text().splitlines()
                    if l.strip()]
        else:
            revision = api.dataset_info(dataset).sha
            rows = load_dataset(dataset, config, split=split,
                                revision=revision, streaming=streaming)
        print(f"[{name}] {dataset} @ {str(revision)[:12]} budgets={budgets}", flush=True)
        builder(rows, sink)
        record = {"name": name, "dataset": dataset, "config": config, "split": split,
                  "license": license_, "revision": revision,
                  **dict(sorted(sink.counters[name].items()))}
        source_records.append(record)
        taken = {k: v for k, v in record.items() if k.startswith("taken_")}
        print(f"[{name}] {taken}", flush=True)

    # Deterministic split and write.
    group_records: dict[str, dict] = {}
    totals = defaultdict(int)
    for group in GROUPS:
        samples = sink.samples.get(group, [])
        if not samples:
            raise RuntimeError(f"group {group} ended up empty — mixture bug")
        splits = {"train": [], "val": [], "calib": []}
        for idx, s in enumerate(samples):
            if idx % SPLIT_MOD == CALIB_SLOT and len(splits["calib"]) < CALIB_MAX:
                splits["calib"].append(s)
            elif idx % SPLIT_MOD == VAL_SLOT:
                splits["val"].append(s)
            else:
                splits["train"].append(s)
        group_records[group] = {}
        for split_name, rows in splits.items():
            path = OUT_DIR / split_name / f"{group}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for s in rows:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            chars = sum(sample_chars(s) for s in rows)
            group_records[group][split_name] = {
                "path": str(path.relative_to(REPO_ROOT)), "samples": len(rows),
                "chars": chars, "sha256": sha256_file(path),
                "bytes": path.stat().st_size, "tracked_in_git": False,
            }
            totals[split_name] += len(rows)
            totals[f"{split_name}_chars"] += chars

    manifest = {
        "dataset": MIXTURE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": ("Stage 2 offline warm-up/distillation data for post-init "
                    "student recovery (Stage 3+); grouped by training use"),
        "schema": {
            "fields": "id, group, source, format ('chat'|'text'), messages|text, tools?",
            "tool_convention": ("OpenAI-nested tools/tool_calls, rendered by the "
                                "Qwen3 chat template (see src/aadistill/data.py)"),
        },
        "caps": {"msg_char_cap": MSG_CHAR_CAP, "sample_char_cap": SAMPLE_CHAR_CAP,
                 "long_doc_min": LONG_DOC_MIN, "long_doc_cap": LONG_DOC_CAP},
        "split_rule": (f"per group, index i in build order: i%{SPLIT_MOD}=="
                       f"{CALIB_SLOT} -> calib (max {CALIB_MAX}), i%{SPLIT_MOD}=="
                       f"{VAL_SLOT} -> val, else train; calib across all groups is "
                       "the quantization-calibration set"),
        "dedup": "global exact content sha256 (id excluded)",
        "holdout_exclusion": (f"fineweb stream offset >= {FINEWEB_SKIP} plus "
                              "first-1000-char match against holdout_v1"),
        "teacher_generated_data": "none in v0",
        "sources": source_records,
        "groups": group_records,
        "totals": dict(totals),
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest_path = OUT_DIR / f"{MIXTURE}.manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"\nWrote {manifest_path}")
    for group, rec in group_records.items():
        print(f"  {group:20s} train {rec['train']['samples']:5d} "
              f"({rec['train']['chars']:>9,} chars)  val {rec['val']['samples']:4d}  "
              f"calib {rec['calib']['samples']:3d}")
    print(f"  totals: {dict(totals)}")


if __name__ == "__main__":
    main()

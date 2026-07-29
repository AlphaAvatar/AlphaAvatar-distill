"""Build the Stage 2 offline mixture v1 (`stage2_offline_v1`) — the approved
5.39M -> ~24M-train-token scale-up (proposal 2026-07-26, approved same day).

Usage:
    uv run python scripts/build_stage2_v1.py

Design (see logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md):

* v0 train is carried into v1 train verbatim, except gsm8k samples, which are
  format-normalized (strip `<<...>>` calculator annotations, rewrite the
  trailing `#### N` as "The answer is N.") — same normalization as fresh
  gsm8k rows, addressing the `<<>>`/`####` artifacts seen in generation smoke.
* Fresh rows come only from unconsumed offsets of the v0-pinned sources
  (per-source `skip_until` = max sample index consumed by v0, derived from
  the v0 jsonl ids) or from new sources; global content dedup is seeded with
  the digests of every v0 sample (train, val, calib), which is also the
  leakage guard keeping frozen v0 val/calib content out of v1 train.
* v0 `val` (771) and `calib` (120) stay frozen in data/stage2/. This build
  writes data/stage2_v1/: train = v0-carried + new, val = fresh val_v1 slice
  (same modular rule), calib = v0's 120 + up to 10 new per group (~200).
* New sources: smol-smoltalk, OpenMathInstruct-2 (train_1M), Magicoder-OSS-
  Instruct-75K, everyday-conversations-llama3.1-2k, xlam-function-calling-60k
  (auto-gated; access is verified before the build starts and the build fails
  loudly if the click-through has not been accepted).
* Refusal response templates widened from 3 cycled strings to 12 (anti-echo).

Reuses the v0 builders (oasst2 threading, glaive parsing, fineweb filtering)
unmodified by iterating sources from row 0 with a sink that skips indices
consumed by v0; `scripts/build_stage2_v0.py` itself is untouched.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_stage2_v0 as v0
from aadistill.data import GROUPS, load_jsonl, validate_sample
from aadistill.env import code_state
from aadistill.manifest import sha256_file, write_manifest

V0_DIR = REPO_ROOT / "data/stage2"
OUT_DIR = REPO_ROOT / "data/stage2_v1"
MIXTURE = "stage2_offline_v1"
XLAM_REPO = "Salesforce/xlam-function-calling-60k"

# Same modular split rule as v0, applied to *new* samples only (v0-carried
# train stays train; frozen v0 val/calib are copied, never re-split).
CALIB_MAX_NEW = 10  # per group top-up: 120 v0 + ~8x10 new ~= 200 total

# Fresh-offset table: max v0-consumed sample index per continued source
# (computed from the v0 jsonl ids; ids embed the source row/thread index).
SKIP_UNTIL = {
    "oasst2": 1343,
    "squad_v2": 8558,
    "hotpot_qa": 511,
    "glaive_fc_v2": 3667,
    "gsm8k": 4797,
    "fineweb_edu_long": 11814,
}

# Widened refusal pool (proposal design change 4): the three v0 strings plus
# nine varied ones, cycled with % 12 instead of % 3.
UNANSWERABLE_RESPONSES_V1 = list(v0.UNANSWERABLE_RESPONSES) + [
    "The passage doesn't cover that, so I'd rather say so than guess.",
    "I looked for this in the context, but it isn't there, so I can't answer reliably.",
    "That detail isn't in the provided text, so I don't know the answer from this context alone.",
    "The context talks about related things, but not this specific point, so I can't answer it from what's given.",
    "There's no sentence in the passage that answers this, so the honest answer is that I can't tell.",
    "Based only on the provided context, this question can't be answered.",
    "The answer isn't contained in the passage, so I won't guess.",
    "I don't see that information in the context. If you can share a passage that covers it, I'll try again.",
    "This goes beyond what the provided context states, so I can't answer it in a grounded way.",
]

_GSM_CALC = re.compile(r"<<[^<>]*>>")
_GSM_FINAL = re.compile(r"^####\s*(.+?)\s*$")

# smol-smoltalk short-dialogue filter for the short_realtime group.
SMOL_SHORT_MAX_MSGS = 6
SMOL_SHORT_MSG_CAP = 240
SMOL_SHORT_TOTAL_CAP = 900


def normalize_gsm8k_answer(text: str) -> str:
    """Strip `<<...>>` calculator annotations; turn `#### N` into prose."""
    text = _GSM_CALC.sub("", text)
    lines = text.rstrip().splitlines()
    if lines:
        m = _GSM_FINAL.match(lines[-1].strip())
        if m:
            lines[-1] = f"The answer is {m.group(1)}."
    return "\n".join(lines).strip()


def content_digest(sample: dict) -> str:
    """Exact replica of the v0 Sink dedup digest (id excluded)."""
    return hashlib.sha256(
        json.dumps(sample, sort_keys=True, ensure_ascii=False)
        .replace(sample["id"], "").encode()
    ).hexdigest()


class V1Sink(v0.Sink):
    """v0 Sink + seeded dedup + skipping of sample indices consumed by v0."""

    def __init__(self, seen_seed: set[str]):
        super().__init__()
        self._seen |= seen_seed
        self._skip_until = -1

    def start_source(self, name: str, budgets: dict[str, int],
                     skip_until: int = -1):
        super().start_source(name, budgets)
        self._skip_until = skip_until

    def add(self, group: str, sample: dict) -> bool:
        idx_str = sample["id"].rpartition("-")[2]
        if idx_str.isdigit() and int(idx_str) <= self._skip_until:
            self.note("skipped_v0_consumed")
            return False
        return super().add(group, sample)


def build_squad_v1(rows, sink):
    """v0 squad builder with the widened 12-template refusal pool."""
    n_unanswerable = 0
    for idx, row in enumerate(rows):
        if sink.done():
            break
        answerable = bool(row["answers"]["text"])
        user = v0.RAG_PROMPT.format(context=row["context"].strip(),
                                    question=row["question"].strip())
        if answerable:
            group, answer = "rag_evidence", row["answers"]["text"][0].strip()
        else:
            group = "refusal_uncertainty"
            answer = UNANSWERABLE_RESPONSES_V1[
                n_unanswerable % len(UNANSWERABLE_RESPONSES_V1)]
            n_unanswerable += 1
        sink.add(group, {
            "id": f"squad_v2-{idx:06d}", "group": group, "source": "squad_v2",
            "format": "chat",
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}],
        })


def build_gsm8k_v1(rows, sink):
    """v0 gsm8k builder + format normalization on every fresh row."""
    for idx, row in enumerate(rows):
        if sink.done():
            break
        sink.add("code_math", {
            "id": f"gsm8k-{idx:06d}", "group": "code_math", "source": "gsm8k",
            "format": "chat",
            "messages": [
                {"role": "user", "content": row["question"].strip()},
                {"role": "assistant",
                 "content": normalize_gsm8k_answer(row["answer"])},
            ],
        })


def _clean_chat(messages) -> list[dict] | None:
    """Validate an upstream messages list: optional leading system turn (kept —
    smol-smoltalk rewrite/summarize subsets put the task instruction there),
    then strict user/assistant alternation ending on assistant; no over-cap or
    empty message; returns None (skip row) otherwise."""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    msgs = [{"role": m.get("role"), "content": (m.get("content") or "").strip()}
            for m in messages]
    body = msgs[1:] if msgs[0]["role"] == "system" else msgs
    roles = [m["role"] for m in body]
    expect = ["user", "assistant"] * ((len(roles) + 1) // 2)
    if not roles or roles != expect[:len(roles)] or roles[-1] != "assistant":
        return None
    if any(not m["content"] or len(m["content"]) > v0.MSG_CHAR_CAP
           for m in msgs):
        return None
    return msgs


def build_smoltalk(rows, sink):
    """Route each conversation to short_realtime (short-dialogue filter) or
    instruction. Skips (rather than truncates) over-cap conversations."""
    for idx, row in enumerate(rows):
        if sink.done():
            break
        msgs = _clean_chat(row["messages"])
        if msgs is None:
            sink.note("rejected_malformed_or_long")
            continue
        total = sum(len(m["content"]) for m in msgs)
        short = (len(msgs) <= SMOL_SHORT_MAX_MSGS
                 and total <= SMOL_SHORT_TOTAL_CAP
                 and all(len(m["content"]) <= SMOL_SHORT_MSG_CAP for m in msgs))
        group = "short_realtime" if short else "instruction"
        sink.add(group, {
            "id": f"smoltalk-{idx:06d}", "group": group,
            "source": "smol_smoltalk", "format": "chat", "messages": msgs,
        })


def build_everyday(rows, sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        msgs = _clean_chat(row["messages"])
        if msgs is None:
            sink.note("rejected_malformed_or_long")
            continue
        sink.add("short_realtime", {
            "id": f"everyday-{idx:06d}", "group": "short_realtime",
            "source": "everyday_conversations", "format": "chat",
            "messages": msgs,
        })


def build_openmath(rows, sink):
    """OpenMathInstruct-2: one solution per distinct problem, skip over-cap
    rows (a truncated derivation is a bad target)."""
    seen_problems: set[str] = set()
    for idx, row in enumerate(rows):
        if sink.done():
            break
        problem = row["problem"].strip()
        solution = row["generated_solution"].strip()
        if not problem or not solution:
            continue
        if len(problem) > v0.MSG_CHAR_CAP or len(solution) > v0.MSG_CHAR_CAP:
            sink.note("rejected_too_long_skip")
            continue
        if problem in seen_problems:
            sink.note("skipped_duplicate_problem")
            continue
        seen_problems.add(problem)
        sink.add("code_math", {
            "id": f"openmath-{idx:06d}", "group": "code_math",
            "source": "openmath_instruct_2", "format": "chat",
            "messages": [{"role": "user", "content": problem},
                         {"role": "assistant", "content": solution}],
        })


def build_magicoder(rows, sink):
    for idx, row in enumerate(rows):
        if sink.done():
            break
        problem = row["problem"].strip()
        solution = row["solution"].strip()
        if not problem or not solution:
            continue
        if len(problem) > v0.MSG_CHAR_CAP or len(solution) > v0.MSG_CHAR_CAP:
            sink.note("rejected_too_long_skip")
            continue
        sink.add("code_math", {
            "id": f"magicoder-{idx:06d}", "group": "code_math",
            "source": "magicoder_oss", "format": "chat",
            "messages": [{"role": "user", "content": problem},
                         {"role": "assistant", "content": solution}],
        })


def build_xlam(rows, sink):
    """xlam-function-calling-60k: query -> parallel tool calls (no tool
    responses in the source; trains call emission, complementing glaive's
    full call/response loops)."""
    for idx, row in enumerate(rows):
        if sink.done():
            break
        try:
            tools_raw = json.loads(row["tools"])
            answers = json.loads(row["answers"])
        except (json.JSONDecodeError, TypeError):
            sink.note("parse_failures")
            continue
        tools = [{"type": "function",
                  "function": {"name": t["name"],
                               "description": t.get("description", ""),
                               "parameters": t.get("parameters", {})}}
                 for t in tools_raw
                 if isinstance(t, dict) and isinstance(t.get("name"), str)]
        calls = [{"type": "function",
                  "function": {"name": a["name"],
                               "arguments": a.get("arguments", {})}}
                 for a in answers
                 if isinstance(a, dict) and isinstance(a.get("name"), str)
                 and isinstance(a.get("arguments"), dict)]
        query = (row.get("query") or "").strip()
        if not tools or not calls or len(calls) != len(answers) or not query:
            sink.note("parse_failures")
            continue
        sink.add("tool_calling", {
            "id": f"xlam-{idx:06d}", "group": "tool_calling",
            "source": "xlam_fc_60k", "format": "chat", "tools": tools,
            "messages": [{"role": "user", "content": query},
                         {"role": "assistant", "content": "",
                          "tool_calls": calls}],
        })


# (name, dataset, config, split, license, streaming, builder, {group: char budget})
# Char budgets sized from per-group v0 chars/token ratios to land ~24M train
# tokens total (approval ceiling 30M); actual tokens measured by the dry run.
SOURCES_V1 = [
    ("oasst2", "OpenAssistant/oasst2", None, "train", "Apache-2.0",
     False, v0.build_oasst2, {"instruction": 3_000_000}),
    ("smol_smoltalk", "HuggingFaceTB/smol-smoltalk", None, "train", "Apache-2.0",
     True, build_smoltalk,
     {"instruction": 21_000_000, "short_realtime": 3_200_000}),
    ("everyday_conversations", "HuggingFaceTB/everyday-conversations-llama3.1-2k",
     None, "train_sft", "Apache-2.0", False, build_everyday,
     {"short_realtime": 1_500_000}),
    ("squad_v2", "rajpurkar/squad_v2", None, "train", "CC-BY-SA 4.0",
     False, build_squad_v1,
     {"rag_evidence": 6_000_000, "refusal_uncertainty": 6_000_000}),
    ("hotpot_qa", "hotpotqa/hotpot_qa", "distractor", "train", "CC-BY-SA 4.0",
     True, v0.build_hotpot, {"multihop_qa": 3_700_000}),
    ("glaive_fc_v2", "glaiveai/glaive-function-calling-v2", None, "train",
     "Apache-2.0", True, v0.build_glaive, {"tool_calling": 3_400_000}),
    ("xlam_fc_60k", XLAM_REPO, None, "train",
     "CC-BY-4.0 (auto-gated, accepted on the AlphaAvatar HF account)",
     False, build_xlam, {"tool_calling": 2_800_000}),
    ("gsm8k", "openai/gsm8k", "main", "train", "MIT",
     False, build_gsm8k_v1, {"code_math": 1_500_000}),
    ("openmath_instruct_2", "nvidia/OpenMathInstruct-2", None, "train_1M",
     "CC-BY-4.0 (Built with Llama)", True, build_openmath,
     {"code_math": 6_000_000}),
    ("magicoder_oss", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
     "MIT (GPT-3.5-generated; flag for release-time license review)",
     True, build_magicoder, {"code_math": 3_200_000}),
    ("fineweb_edu_long", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
     "ODC-By 1.0", True, v0.build_fineweb_long, {"long_context": 9_100_000}),
]


def load_v0_splits() -> tuple[dict, dict, dict, set[str], dict]:
    """Load frozen v0 jsonl splits; return (train_carried, val, calib,
    digest seed, carry stats). gsm8k train samples are normalized here."""
    seed: set[str] = set()
    splits: dict[str, dict[str, list[dict]]] = {}
    n_gsm_normalized = 0
    for split in ("train", "val", "calib"):
        splits[split] = {}
        for path in sorted((V0_DIR / split).glob("*.jsonl")):
            samples = load_jsonl(path)
            seed.update(content_digest(s) for s in samples)
            if split == "train":
                for s in samples:
                    if s["source"] == "gsm8k":
                        msg = s["messages"][-1]
                        assert msg["role"] == "assistant"
                        msg["content"] = normalize_gsm8k_answer(msg["content"])
                        n_gsm_normalized += 1
                        validate_sample(s)
                # normalized digests too, so fresh rows cannot collide with
                # the carried (rewritten) forms either
                seed.update(content_digest(s) for s in samples)
            splits[split][path.stem] = samples
    stats = {
        "train_samples": sum(len(v) for v in splits["train"].values()),
        "train_chars": sum(v0.sample_chars(s)
                           for v in splits["train"].values() for s in v),
        "val_samples": sum(len(v) for v in splits["val"].values()),
        "calib_samples": sum(len(v) for v in splits["calib"].values()),
        "gsm8k_train_normalized": n_gsm_normalized,
    }
    return splits["train"], splits["val"], splits["calib"], seed, stats


def check_xlam_access() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError
    try:
        HfApi().auth_check(XLAM_REPO, repo_type="dataset")
    except GatedRepoError as e:
        raise SystemExit(
            f"FATAL: no access to gated {XLAM_REPO}. Open "
            f"https://huggingface.co/datasets/{XLAM_REPO} in a browser while "
            "logged in as AlphaAvatar and click 'Agree and access repository', "
            "then re-run this script. (The approved v1 composition includes "
            "xlam; refusing to silently build without it.)"
        ) from e


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    check_xlam_access()
    api = HfApi()

    train_carried, val_v0, calib_v0, seed, carry_stats = load_v0_splits()
    print(f"v0 carry: {carry_stats}", flush=True)

    sink = V1Sink(seed)
    source_records = []
    for name, dataset, config, split, license_, streaming, builder, budgets \
            in SOURCES_V1:
        skip_until = SKIP_UNTIL.get(name, -1)
        sink.start_source(name, budgets, skip_until=skip_until)
        revision = api.dataset_info(dataset).sha
        rows = load_dataset(dataset, config, split=split,
                            revision=revision, streaming=streaming)
        print(f"[{name}] {dataset} @ {str(revision)[:12]} budgets={budgets} "
              f"skip_until={skip_until}", flush=True)
        builder(rows, sink)
        record = {"name": name, "dataset": dataset, "config": config,
                  "split": split, "license": license_, "revision": revision,
                  "skip_until_v0_index": skip_until,
                  "budget_exhausted": sink.done(),
                  **dict(sorted(sink.counters[name].items()))}
        source_records.append(record)
        taken = {k: v for k, v in record.items() if k.startswith("taken_")}
        print(f"[{name}] done={sink.done()} {taken}", flush=True)
        if not sink.done():
            print(f"[{name}] NOTE: source exhausted before budget "
                  f"(remaining={ {g: b for g, b in sink.budgets.items() if b > 0} })",
                  flush=True)

    # Split new samples with the v0 modular rule; assemble v1 = carry + new.
    group_records: dict[str, dict] = {}
    totals: dict[str, int] = defaultdict(int)
    new_counts: dict[str, dict[str, int]] = {}
    all_ids: set[str] = set()
    for group in GROUPS:
        new = sink.samples.get(group, [])
        if not new:
            raise RuntimeError(f"group {group} got no new samples — mixture bug")
        splits = {"train": [], "val": [], "calib": []}
        for idx, s in enumerate(new):
            if (idx % v0.SPLIT_MOD == v0.CALIB_SLOT
                    and len(splits["calib"]) < CALIB_MAX_NEW):
                splits["calib"].append(s)
            elif idx % v0.SPLIT_MOD == v0.VAL_SLOT:
                splits["val"].append(s)
            else:
                splits["train"].append(s)
        new_counts[group] = {k: len(v) for k, v in splits.items()}

        final = {
            "train": train_carried.get(group, []) + splits["train"],
            "val": splits["val"],  # val_v1; frozen val_v0 stays in data/stage2
            "calib": calib_v0.get(group, []) + splits["calib"],
        }
        group_records[group] = {}
        for split_name, rows_ in final.items():
            for s in rows_:
                if s["id"] in all_ids:
                    raise RuntimeError(f"duplicate id {s['id']} in v1")
                all_ids.add(s["id"])
            path = OUT_DIR / split_name / f"{group}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for s in rows_:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            chars = sum(v0.sample_chars(s) for s in rows_)
            group_records[group][split_name] = {
                "path": str(path.relative_to(REPO_ROOT)), "samples": len(rows_),
                "chars": chars, "sha256": sha256_file(path),
                "bytes": path.stat().st_size, "tracked_in_git": False,
            }
            totals[split_name] += len(rows_)
            totals[f"{split_name}_chars"] += chars

    manifest = {
        "dataset": MIXTURE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": ("Stage 2 offline mixture v1: approved ~4.5x train scale-up "
                    "for data-limited Stage 3 recovery (proposal 2026-07-26)"),
        "based_on": {
            "mixture": "stage2_offline_v0",
            "manifest_sha256": sha256_file(
                V0_DIR / "stage2_offline_v0.manifest.json"),
            "carry": carry_stats,
            "note": ("v0 train carried into v1 train (gsm8k normalized); v0 "
                     "val/calib stay frozen in data/stage2/ as val_v0 and the "
                     "first 120 calib samples; v1 val split is val_v1 only"),
        },
        "schema": {
            "fields": "id, group, source, format ('chat'|'text'), messages|text, tools?",
            "tool_convention": ("OpenAI-nested tools/tool_calls, rendered by the "
                                "Qwen3 chat template (see src/aadistill/data.py)"),
        },
        "caps": {"msg_char_cap": v0.MSG_CHAR_CAP,
                 "sample_char_cap": v0.SAMPLE_CHAR_CAP,
                 "long_doc_min": v0.LONG_DOC_MIN, "long_doc_cap": v0.LONG_DOC_CAP,
                 "smol_short_filter": {"max_msgs": SMOL_SHORT_MAX_MSGS,
                                       "msg_cap": SMOL_SHORT_MSG_CAP,
                                       "total_cap": SMOL_SHORT_TOTAL_CAP}},
        "split_rule": (f"new samples only, per group, index i in build order: "
                       f"i%{v0.SPLIT_MOD}=={v0.CALIB_SLOT} -> calib (max "
                       f"{CALIB_MAX_NEW} per group on top of frozen v0 calib), "
                       f"i%{v0.SPLIT_MOD}=={v0.VAL_SLOT} -> val_v1, else train"),
        "dedup": ("global exact content sha256 (id excluded), seeded with all "
                  "v0 sample digests (raw + gsm8k-normalized) = v0 val/calib "
                  "leakage guard"),
        "fresh_offsets": SKIP_UNTIL,
        "normalization": {
            "gsm8k": ("strip <<...>> calculator annotations; rewrite trailing "
                      "'#### N' as 'The answer is N.' — applied to fresh rows "
                      "and to v0-carried train samples"),
            "refusal_templates": len(UNANSWERABLE_RESPONSES_V1),
        },
        "holdout_exclusion": (f"fineweb stream offset > "
                              f"{SKIP_UNTIL['fineweb_edu_long']} (v0 consumed) "
                              "plus first-1000-char match against holdout_v1"),
        "teacher_generated_data": "none in v1 (separate future proposal)",
        "synthetic_provenance": {
            "smol_smoltalk": "largely Llama-3.1-405B-generated (Apache-2.0)",
            "everyday_conversations": "Llama-3.1-generated (Apache-2.0)",
            "openmath_instruct_2": "Llama-3.1-405B-generated (CC-BY-4.0, Built with Llama)",
            "magicoder_oss": "GPT-3.5-generated (MIT; release-time license review)",
        },
        "sources": source_records,
        "new_sample_counts": new_counts,
        "groups": group_records,
        "totals": dict(totals),
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest_path = OUT_DIR / f"{MIXTURE}.manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"\nWrote {manifest_path}")
    for group, rec in group_records.items():
        print(f"  {group:20s} train {rec['train']['samples']:6d} "
              f"({rec['train']['chars']:>10,} chars)  val {rec['val']['samples']:4d}  "
              f"calib {rec['calib']['samples']:3d}")
    print(f"  totals: {dict(totals)}")


if __name__ == "__main__":
    main()

"""Prompt rendering shared between the recovery-search battery and Phase C1.

`build_recovery_search_battery.py` froze how each source becomes a prompt —
instruction wording, field order, id convention, which rows are skipped. The
Phase-C1 confirmation battery must render **identically**, or its prompts are not
comparable to anything the project has measured.

The renderers therefore live here, once, as module-level functions rather than
as closures inside a `main()`. The v1 builder is deliberately left untouched, so
the frozen `recovery_search_v2` artifact keeps the exact build path that produced
it; `tests/data/test_c1_battery.py` closes the loop by asserting that these
functions reproduce that artifact's stored `prompt_text` byte for byte.

Rows are read straight from the pinned Hugging Face **snapshot directory**, not
through `load_dataset`'s cache resolution, so the revision a battery was built
from is a path rather than an inference.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"

# Verbatim from build_recovery_search_battery.py. Changing any of these changes
# every prompt, so they are constants, not parameters.
RAG_INSTRUCTION = ("Answer the question using only the provided context. If the "
                   "context does not contain the answer, say you cannot answer.")
MULTIHOP_INSTRUCTION = ("Answer the question using the provided documents. Name the "
                        "documents your answer relies on.")
CODE_INSTRUCTION = ("Write a Python function for the following task. Return only the "
                    "function.")
TOOL_INSTRUCTION = ("You have access to the following tools. Call the tool that "
                    "answers the user's request.")


def norm(text: str) -> str:
    """Normalized form for cross-role content comparison."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# --- pinned source reading --------------------------------------------------

def snapshot(repo_id: str, revision: str) -> Path:
    d = HUB / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / revision
    if not d.is_dir():
        raise FileNotFoundError(
            f"{repo_id}@{revision} is not in the local hub cache at {d}")
    return d


def read_rows(repo_id: str, revision: str, relpath: str) -> list[dict[str, Any]]:
    """Rows of one pinned file, in file order."""
    f = snapshot(repo_id, revision) / relpath
    if f.suffix == ".jsonl":
        return [json.loads(line) for line in f.open() if line.strip()]
    if f.suffix == ".json":
        return json.loads(f.read_text())
    import pyarrow.parquet as pq
    return pq.read_table(f).to_pylist()


def source_digest(repo_id: str, revision: str, relpath: str) -> dict[str, Any]:
    """Byte identity of the exact file the prompts were read from."""
    f = snapshot(repo_id, revision) / relpath
    return {"repo_id": repo_id, "revision": revision, "file": relpath,
            "size_bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}


# --- renderers, verbatim in behaviour from the v1 builder -------------------

def make_gsm8k(r: dict) -> dict | None:
    q = r["question"]
    return {"id": f"gsm8k-test-{r['_index']:05d}",
            "source_key": f"gsm8k-test-{r['_index']:05d}",
            "group": "gsm8k", "source": "gsm8k", "prompt_text": q,
            "messages": [{"role": "user", "content": q}],
            "gold": r["answer"].split("####")[-1].strip(),
            "gsm8k_answer": r["answer"].split("####")[-1].strip()}


def make_math_verified(r: dict) -> dict | None:
    return {"id": f"math500-{r['unique_id']}", "group": "math_verified",
            "source": "math_500", "subject": r["subject"], "level": r["level"],
            "prompt_text": r["problem"],
            "messages": [{"role": "user", "content": r["problem"]}],
            "gold": r["answer"], "boxed": r["answer"]}


def make_multihop(r: dict) -> dict | None:
    ctx = r.get("context") or {}
    titles = list(ctx.get("title") or [])
    sents = list(ctx.get("sentences") or [])
    if not titles:
        return None
    docs = "\n".join(f"[{t}] {''.join(s)}" for t, s in zip(titles, sents))
    text = f"{MULTIHOP_INSTRUCTION}\n\nDocuments:\n{docs}\n\nQuestion: {r['question']}"
    return {"id": f"hotpot-val-{r['id']}", "group": "multihop",
            "source": "hotpot_qa", "source_key": r["id"], "prompt_text": text,
            "messages": [{"role": "user", "content": text}],
            "gold": r["answer"], "answer": r["answer"],
            "supporting_titles": list(dict.fromkeys(
                r.get("supporting_facts", {}).get("title", [])))}


def make_rag(r: dict) -> dict | None:
    answers = list(r["answers"]["text"])
    if not answers:
        return None                            # answerable rows only
    text = (f"{RAG_INSTRUCTION}\n\nContext: {r['context']}\n\n"
            f"Question: {r['question']}")
    return {"id": f"squad-val-{r['id']}", "group": "rag", "source": "squad_v2",
            "source_key": r["id"], "prompt_text": text,
            "messages": [{"role": "user", "content": text}],
            "context": r["context"], "title": r["title"],
            "question": r["question"], "gold": answers[0], "answerable": True}


def make_knowledge(r: dict) -> dict | None:
    answer = r.get("answer") or {}
    gold = answer.get("value")
    if not gold:
        return None
    aliases = sorted({*(answer.get("aliases") or []),
                      *(answer.get("normalized_aliases") or []), gold})
    return {"id": f"trivia-{r['question_id']}", "group": "knowledge",
            "source": "trivia_qa", "source_key": r["question_id"],
            "prompt_text": r["question"],
            "messages": [{"role": "user", "content": r["question"]}],
            "gold": gold, "aliases": aliases}


def make_code(r: dict) -> dict | None:
    text = f"{CODE_INSTRUCTION}\n\n{r['text']}\n\nTests:\n" + "\n".join(r["test_list"])
    return {"id": f"mbpp-test-{r['task_id']}", "group": "code", "source": "mbpp",
            "source_key": str(r["task_id"]), "prompt_text": text,
            "messages": [{"role": "user", "content": text}],
            "scorable": False,
            "reference_code": r["code"], "test_list": list(r["test_list"])}


def make_tool(r: dict) -> dict | None:
    tools = r.get("tools")
    if not tools:
        return None
    text = f"{TOOL_INSTRUCTION}\n\nTools:\n{tools}\n\nRequest: {r['query']}"
    parsed_tools = json.loads(tools) if isinstance(tools, str) else tools
    parsed_calls = json.loads(r["answers"]) if isinstance(r["answers"], str) \
        else r["answers"]
    if not parsed_calls:
        return None
    return {"id": f"xlam-{r['id']}", "group": "tool", "source": "xlam_fc_60k",
            "source_key": str(r["id"]), "prompt_text": text,
            "messages": [{"role": "user", "content": text}],
            "scorable": True,
            "correctness_field": "tool_call_exact_match",
            "reference_calls": r.get("answers"), "tools": tools,
            "scorer_tools": [
                {"function": {"name": t.get("name"),
                              "parameters": {
                                  "properties": t.get("parameters") or {},
                                  "required": [
                                      n for n, spec in (t.get("parameters") or {}).items()
                                      if isinstance(spec, dict) and "default" not in spec]}}}
                for t in parsed_tools],
            "gold_tool_calls": [
                {"function": {"name": c.get("name"), "arguments": c.get("arguments")}}
                for c in parsed_calls]}


RENDERERS: dict[str, Callable[[dict], dict | None]] = {
    "gsm8k": make_gsm8k, "math_verified": make_math_verified,
    "multihop": make_multihop, "rag": make_rag, "knowledge": make_knowledge,
    "code": make_code, "tool": make_tool,
}


# --- deterministic, outcome-independent selection ---------------------------

def rank_key(base_digest: str, stratum: str, stable_id: str) -> str:
    """`SHA256(C0_digest + ":phase-c1-battery:" + stratum + ":" + stable_id)`.

    Selection by cryptographic rank rather than by dataset iteration order. The
    key depends only on a digest frozen before any C1 candidate existed, the
    stratum name and the example's own stable id — never on a model outcome, a
    difficulty field, or the order a loader happened to yield rows in.
    """
    return hashlib.sha256(
        f"{base_digest}:phase-c1-battery:{stratum}:{stable_id}".encode()).hexdigest()


def rank_take(rows: Iterable[dict], want: int, *, stratum: str, base_digest: str,
              exclude_ids: set[str], exclude_hashes: set[str],
              make: Callable[[dict], dict | None]) -> list[dict]:
    """The lowest-ranked `want` eligible examples of one stratum.

    Eligibility is decided first and identically to the v1 builder — stable-id
    and normalized-content exclusion, plus within-battery content dedup — and
    only then is the rank applied, so the exclusion contract is never weakened
    by the sampling change.
    """
    candidates: list[tuple[str, dict]] = []
    for row in rows:
        item = make(row)
        if item is None:
            continue
        if str(item["id"]) in exclude_ids or str(item.get("source_key")) in exclude_ids:
            continue
        h = content_sha256(norm(item["prompt_text"]))
        if h in exclude_hashes:
            continue
        item["prompt_sha256"] = h
        candidates.append((rank_key(base_digest, stratum, str(item["id"])), item))

    candidates.sort(key=lambda t: (t[0], str(t[1]["id"])))
    out: list[dict] = []
    seen: set[str] = set()
    for _, item in candidates:
        if item["prompt_sha256"] in seen:
            continue
        seen.add(item["prompt_sha256"])
        out.append(item)
        if len(out) >= want:
            break
    return out

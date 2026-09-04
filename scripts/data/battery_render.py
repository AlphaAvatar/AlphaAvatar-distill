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
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402

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

def hub_cache() -> Path:
    """The hub cache root, resolved **at call time** from the environment.

    Until 2026-09-04 this was a module-level `Path.home() / ".cache/huggingface/hub"`,
    read directly. A pod exports `HF_HOME` and holds nothing under `$HOME`, so the
    seven renderer-parity cases could pass on the dev box and could never pass on
    a pod — which is what aborted C1 attempt 3R at the setup test gate for $0.3482.

    The precedence is `huggingface_hub`'s own, so a warmed cache is found wherever
    the surrounding tooling put it:

    1. `HF_HUB_CACHE` — the explicit override, used verbatim;
    2. `$HF_HOME/hub` — the cache a pod's `HF_HOME` implies;
    3. `~/.cache/huggingface/hub` — the unconfigured default.

    Resolution is deliberately **not** frozen at import: a caller that sets the
    environment after importing this module, and every test that isolates the
    cache with `monkeypatch.setenv`, must both be honoured.
    """
    explicit = os.environ.get("HF_HUB_CACHE", "").strip()
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def snapshot_path(repo_id: str, revision: str) -> Path:
    """Where a pinned snapshot *would* live. Does not require it to be there."""
    return (hub_cache() / f"datasets--{repo_id.replace('/', '--')}"
            / "snapshots" / revision)


def snapshot(repo_id: str, revision: str) -> Path:
    d = snapshot_path(repo_id, revision)
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


# --- renderer parity with the frozen recovery-search battery ----------------
#
# One implementation, two callers: `tests/data/test_c1_battery.py`, which skips a
# group whose pinned snapshot is absent, and `scripts/autoinit/renderer_parity_gate.py`,
# which refuses that same absence. Two independent comparison algorithms could
# disagree about what parity means, which is the one thing this guarantee cannot
# afford, so the algorithm lives here and neither caller reimplements it.

#: The exact pinned sources the frozen `recovery_search_v2` prompts were built from.
FROZEN_SOURCES: dict[str, tuple[str, str, str]] = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              "main/test-00000-of-00001.parquet"),
    "math_verified": ("HuggingFaceH4/MATH-500",
                      "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "multihop": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                 "distractor/validation-00000-of-00001.parquet"),
    "rag": ("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412",
            "squad_v2/validation-00000-of-00001.parquet"),
    "knowledge": ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                  "rc.nocontext/validation-00000-of-00001.parquet"),
    "code": ("google-research-datasets/mbpp",
             "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             "full/test-00000-of-00001.parquet"),
    "tool": ("Salesforce/xlam-function-calling-60k",
             "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
             "xlam_function_calling_60k.json"),
}

FROZEN_BATTERY = REPO_ROOT / "artifacts/stage3/recovery_search_v2"


def frozen_items(group: str) -> dict[str, dict]:
    """The frozen recovery-search items of one group, keyed by id."""
    f = FROZEN_BATTERY / f"{group}.jsonl"
    return {str(i["id"]): i for i in
            (json.loads(line) for line in f.open() if line.strip())}


def check_group_parity(group: str) -> dict[str, Any]:
    """Re-render one group's frozen items from source and compare byte for byte.

    Returns a result record rather than raising, so the caller decides what an
    absent source means. `status` is one of:

    * `SOURCE_ABSENT` — the pinned snapshot is not in the resolved hub cache;
    * `FAIL` — it is there and something no longer renders identically;
    * `PASS` — every frozen item of the group was re-rendered, byte for byte.

    Parity is counted over **distinct ids**, not rows: `trivia_qa` repeats
    `question_id` across source rows, so one frozen item can be re-rendered
    several times, and every one of those renderings must still agree.
    """
    repo, rev, rel = FROZEN_SOURCES[group]
    resolved = snapshot_path(repo, rev)
    frozen = frozen_items(group)
    result: dict[str, Any] = {
        "group": group, "repo_id": repo, "revision": rev, "file": rel,
        "resolved_snapshot": str(resolved), "n_frozen": len(frozen),
        "n_checked": 0, "mismatches": [], "missing": [],
    }
    if not resolved.is_dir():
        result["status"] = "SOURCE_ABSENT"
        return result

    result["source_digest"] = source_digest(repo, rev, rel)
    rows = read_rows(repo, rev, rel)
    if group == "gsm8k":
        rows = [dict(r, _index=i) for i, r in enumerate(rows)]
    make = RENDERERS[group]

    checked: set[str] = set()
    for row in rows:
        item = make(row)
        if item is None or str(item["id"]) not in frozen:
            continue
        ident = str(item["id"])
        want = frozen[ident]
        if item["prompt_text"] != want["prompt_text"]:
            result["mismatches"].append({"id": ident, "field": "prompt_text"})
        elif item["messages"] != want["messages"]:
            result["mismatches"].append({"id": ident, "field": "messages"})
        elif content_sha256(norm(item["prompt_text"])) != want["prompt_sha256"]:
            result["mismatches"].append({"id": ident, "field": "prompt_sha256"})
        checked.add(ident)

    result["n_checked"] = len(checked)
    result["missing"] = sorted(set(frozen) - checked)
    result["status"] = "PASS" if (
        not result["mismatches"] and not result["missing"]) else "FAIL"
    return result


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

"""Build the frozen recovery-search battery (role RECOVERY_SEARCH).

    PYTHONPATH=src .venv/bin/python scripts/data/build_recovery_search_battery.py \
        --out artifacts/stage3/recovery_search_v1

Its **only** job is choosing among the fixed 0.86M recovery probes. It is not the
promotion battery, does not reuse it, and never reports a final result.

Two design decisions are load-bearing.

**Scorable and behaviour-only strata are separated.** ``correct_overall`` is
computed only over sets whose scorer already exists, is tested, and is frozen
(``evaluation/capability.py``: gsm8k, math_verified, multihop, rag, knowledge).
Code and tool prompts are included as **behaviour-only**: they contribute to
``usable_rollout_rate`` and to the non-termination / repetition diagnostics, and
they are excluded from correctness by construction. Inventing an unvalidated code
executor or function-call matcher for this battery would put an untested scorer on
the selection path, and this project has already learned to validate an evaluator
against known-bad policies before spending on it. The cost of the decision is
recorded: **the battery cannot see a candidate that trades code or tool capability
for math.**

**No weighted scalar.** Stability and capability stay separate all the way
through: the battery reports ``usable_rollout_rate`` (feasibility),
``correct_overall`` (capability, over scorable sets), and
``correct_given_usable``, plus per-set breakdowns. Nothing here combines them.

Disjointness is enforced against every other role, by stable source id where one
exists and by normalized prompt content everywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/data"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

BATTERY_ID = "recovery_search"
BATTERY_VERSION = 1

#: set -> (domain, n, scorable). `scorable` decides whether the set may enter
#: `correct_overall`; a behaviour-only set still counts toward stability.
SETS = {
    "gsm8k":          ("reasoning_math", 30, True),
    "math_verified":  ("reasoning_math", 30, True),
    "multihop":       ("rag_multihop", 30, True),
    "rag":            ("rag_multihop", 30, True),
    "knowledge":      ("general", 30, True),
    "code":           ("code", 20, False),
    "tool":           ("tool", 20, False),
}

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


def excluded_identities(args) -> tuple[set[str], set[str], dict]:
    """Ids and normalized prompt hashes this battery must avoid."""
    source_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    provenance: dict[str, dict] = {}

    # FINAL_PROMOTION — the 846-prompt battery the 150-prompt promotion set is
    # drawn from. Both its stable ids and its prompt content.
    battery = REPO_ROOT / args.battery
    n = 0
    for path in sorted(battery.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            source_ids.add(str(row["id"]))
            prompt_hashes.add(content_sha256(norm(row.get("prompt_text", ""))))
            n += 1
    provenance["final_promotion"] = {
        "asset": args.battery, "n_prompts": n,
        "note": "includes the 150-prompt promotion battery by construction"}

    # Recovery training — every source the 0.86M rung could draw from. The whole
    # corpus is excluded rather than only the rung: the rung is a prefix of it,
    # and a prompt that trains a *different* probe rung is still not out-of-sample
    # for a comparison between probes.
    corpus_ids, corpus_prompts = set(), 0
    for line in (REPO_ROOT / args.sessions).open():
        d = json.loads(line)
        corpus_ids.add(str(d["source_id"]))
        text = "\n".join(str(m.get("content", "")) for m in d["messages"]
                         if m.get("role") != "assistant")
        prompt_hashes.add(content_sha256(norm(text)))
        corpus_prompts += 1
    source_ids |= corpus_ids
    provenance["recovery_training"] = {
        "asset": args.sessions, "n_sessions": corpus_prompts,
        "distinct_source_ids": len(corpus_ids),
        "note": "the whole recovery corpus, not just the 0.86M rung"}

    # INITIALIZER_STATE_EVAL and OPERATOR_CALIBRATION — both store token ids, so
    # the comparable identity is the source id they carry.
    for role, rel in (("initializer_state_eval", args.state_eval),
                      ("operator_calibration", args.calibration)):
        items = [json.loads(l) for l in (REPO_ROOT / rel / "items.jsonl").open()
                 if l.strip()]
        ids = {str(i.get("source_id")) for i in items if i.get("source_id")}
        source_ids |= ids
        provenance[role] = {"asset": rel, "n_items": len(items),
                            "excluded_source_ids": len(ids)}
    return source_ids, prompt_hashes, provenance


def load(name, config, split):
    from datasets import load_dataset

    return load_dataset(name, config, split=split)


def cached_fingerprint(name: str) -> str | None:
    """The cached dataset config hash — what actually pins the bytes offline.

    ``load_dataset`` in offline mode reports revision ``0.0.0``; the directory
    hash under ``~/.cache/huggingface/datasets`` is the only identifier available
    that distinguishes one cached snapshot from another. Recorded as such, rather
    than pretending an upstream revision was resolved.
    """
    root = Path.home() / ".cache/huggingface/datasets" / name.replace("/", "___").lower()
    if not root.is_dir():
        return None
    hashes = sorted(p.name for p in root.rglob("*") if p.is_dir()
                    and len(p.name) == 40 and all(c in "0123456789abcdef" for c in p.name))
    return hashes[0] if hashes else None


def take(rows, want, exclude_ids, exclude_hashes, make):
    """Deterministic stride over a stable order, skipping excluded items."""
    out, seen = [], set()
    for row in rows:
        if len(out) >= want:
            break
        item = make(row)
        if item is None:
            continue
        if str(item["id"]) in exclude_ids or str(item.get("source_key")) in exclude_ids:
            continue
        h = content_sha256(norm(item["prompt_text"]))
        if h in exclude_hashes or h in seen:
            continue
        seen.add(h)
        item["prompt_sha256"] = h
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stage3/recovery_search_v1")
    ap.add_argument("--battery", default="artifacts/eval/battery_v2")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    exclude_ids, exclude_hashes, provenance = excluded_identities(args)

    sources: dict[str, dict] = {}
    built: dict[str, list[dict]] = {}

    def register(set_name, repo, config, split):
        sources[set_name] = {
            "repo": repo, "config": config, "split": split,
            "cached_fingerprint": cached_fingerprint(repo),
            "revision": "unresolved-offline",
        }
        return load(repo, config, split)

    # --- scorable sets ------------------------------------------------------
    ds = register("gsm8k", "openai/gsm8k", "main", "test")
    # Index-based ids, matching the promotion battery's convention. `hash()` is
    # randomized per process for strings, so it cannot appear anywhere in a
    # frozen artifact's identity.
    built["gsm8k"] = take(
        [dict(r, _index=i) for i, r in enumerate(ds)],
        SETS["gsm8k"][1], exclude_ids, exclude_hashes,
        lambda r: {"id": f"gsm8k-test-{r['_index']:05d}",
                   "source_key": f"gsm8k-test-{r['_index']:05d}",
                   "group": "gsm8k", "source": "gsm8k",
                   "prompt_text": r["question"],
                   "messages": [{"role": "user", "content": r["question"]}],
                   "gold": r["answer"].split("####")[-1].strip(),
                   "gsm8k_answer": r["answer"].split("####")[-1].strip()})

    ds = register("math_verified", "HuggingFaceH4/MATH-500", None, "test")
    built["math_verified"] = take(
        list(ds), SETS["math_verified"][1], exclude_ids, exclude_hashes,
        lambda r: {"id": f"math500-{r['unique_id']}", "group": "math_verified",
                   "source": "math_500", "subject": r["subject"], "level": r["level"],
                   "prompt_text": r["problem"],
                   "messages": [{"role": "user", "content": r["problem"]}],
                   "gold": r["answer"], "boxed": r["answer"]})

    ds = register("multihop", "hotpotqa/hotpot_qa", "distractor", "validation")

    def make_multihop(r):
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

    built["multihop"] = take(list(ds), SETS["multihop"][1], exclude_ids,
                             exclude_hashes, make_multihop)

    ds = register("rag", "rajpurkar/squad_v2", None, "validation")

    def make_rag(r):
        answers = list(r["answers"]["text"])
        if not answers:
            return None                       # answerable rows only
        text = (f"{RAG_INSTRUCTION}\n\nContext: {r['context']}\n\n"
                f"Question: {r['question']}")
        return {"id": f"squad-val-{r['id']}", "group": "rag", "source": "squad_v2",
                "source_key": r["id"], "prompt_text": text,
                "messages": [{"role": "user", "content": text}],
                "context": r["context"], "title": r["title"],
                "question": r["question"], "gold": answers[0], "answerable": True}

    built["rag"] = take(list(ds), SETS["rag"][1], exclude_ids, exclude_hashes, make_rag)

    ds = register("knowledge", "mandarjoshi/trivia_qa", "rc.nocontext", "validation")

    def make_knowledge(r):
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

    built["knowledge"] = take(list(ds), SETS["knowledge"][1], exclude_ids,
                              exclude_hashes, make_knowledge)

    # --- behaviour-only sets ------------------------------------------------
    ds = register("code", "google-research-datasets/mbpp", "full", "test")

    def make_code(r):
        text = f"{CODE_INSTRUCTION}\n\n{r['text']}\n\nTests:\n" + "\n".join(r["test_list"])
        return {"id": f"mbpp-test-{r['task_id']}", "group": "code", "source": "mbpp",
                "source_key": str(r["task_id"]), "prompt_text": text,
                "messages": [{"role": "user", "content": text}],
                "scorable": False,
                "reference_code": r["code"], "test_list": list(r["test_list"])}

    built["code"] = take(list(ds), SETS["code"][1], exclude_ids, exclude_hashes,
                         make_code)

    ds = register("tool", "Salesforce/xlam-function-calling-60k", None, "train")

    def make_tool(r):
        tools = r.get("tools")
        if not tools:
            return None
        text = f"{TOOL_INSTRUCTION}\n\nTools:\n{tools}\n\nRequest: {r['query']}"
        return {"id": f"xlam-{r['id']}", "group": "tool", "source": "xlam_fc_60k",
                "source_key": str(r["id"]), "prompt_text": text,
                "messages": [{"role": "user", "content": text}],
                "scorable": False,
                "reference_calls": r.get("answers"), "tools": tools}

    built["tool"] = take(list(ds), SETS["tool"][1], exclude_ids, exclude_hashes,
                         make_tool)

    # --- freeze -------------------------------------------------------------
    short = {k: (len(v), SETS[k][1]) for k, v in built.items() if len(v) < SETS[k][1]}
    if short:
        raise SystemExit(f"sets short of their target (got, want): {short}")

    all_hashes = [i["prompt_sha256"] for v in built.values() for i in v]
    if len(set(all_hashes)) != len(all_hashes):
        raise SystemExit("duplicate prompt content inside the battery")

    outputs = {}
    for name, items in built.items():
        items.sort(key=lambda i: str(i["id"]))
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
        outputs[name] = {"path": str(path.relative_to(REPO_ROOT)),
                         "n": len(items), "sha256": sha256_file(path),
                         "domain": SETS[name][0], "scorable": SETS[name][2]}

    scorable = [n for n in built if SETS[n][2]]
    content_hash = hashlib.sha256(
        "".join(f"{i['id']}:{i['prompt_sha256']}\n"
                for name in sorted(built) for i in built[name]).encode()).hexdigest()

    manifest = {
        "artifact": f"{BATTERY_ID}_v{BATTERY_VERSION}",
        "role": "RECOVERY_SEARCH",
        "purpose": ("selecting among fixed 0.86M recovery probes; never a promotion "
                    "asset, never training data, never a beam-ranking input"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "battery_id": BATTERY_ID, "version": BATTERY_VERSION,
        "sets": outputs,
        "domains": sorted({v[0] for v in SETS.values()}),
        "n_prompts": sum(len(v) for v in built.values()),
        "n_scorable_prompts": sum(len(built[n]) for n in scorable),
        "scorable_sets": scorable,
        "behaviour_only_sets": [n for n in built if not SETS[n][2]],
        "metrics": {
            "feasibility": "usable_rollout_rate over ALL prompts",
            "primary": "correct_overall over SCORABLE prompts only",
            "secondary": "correct_given_usable over SCORABLE prompts only",
            "breakdowns": "per set and per domain, reported separately",
            "diagnostics": ["non_empty", "natural_termination",
                            "no_severe_repetition", "no_context_limit",
                            "protocol_valid"],
            "no_weighted_scalar": ("stability and capability are never combined; "
                                   "usable_rollout gates, correct_overall ranks"),
        },
        "scorers": {
            "source": "src/aadistill/evaluation/capability.py",
            "sha256": sha256_file(REPO_ROOT / "src/aadistill/evaluation/capability.py"),
            "gsm8k": "strict_answer.score_numeric against gsm8k_answer",
            "behaviour_only_note": (
                "code and tool have no frozen scorer; they contribute stability and "
                "failure diagnostics only. LIMITATION: this battery cannot detect a "
                "candidate that trades code or tool capability for math."),
        },
        "sources": sources,
        "sampling_rule": {
            "order": "dataset order as loaded, deterministic stride",
            "filter": "skip excluded ids and excluded normalized prompt hashes",
            "seed": None, "deterministic": True,
            "normalization": "whitespace-collapsed, lowercased, for cross-role hashing",
        },
        "isolation": {"excluded_roles": provenance,
                      "excluded_source_ids": len(exclude_ids),
                      "excluded_prompt_hashes": len(exclude_hashes)},
        "prompt_sha256_index": {i["id"]: i["prompt_sha256"]
                                for name in sorted(built) for i in built[name]},
        "content_sha256": content_hash,
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "n_prompts": manifest["n_prompts"],
        "n_scorable": manifest["n_scorable_prompts"],
        "sets": {k: v["n"] for k, v in outputs.items()},
        "content_sha256": content_hash,
    }, indent=2))


if __name__ == "__main__":
    main()

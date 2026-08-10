#!/usr/bin/env python3
"""Build the frozen E8 depth-selection calibration mixture.

The contribution search asks how far the teacher's output distribution moves
when a block is bypassed. The answer depends entirely on *where you ask*, so the
calibration set is part of the selector and is frozen with it.

Two commitments, both from E7's result that general language modelling and
autonomous reasoning behaviour move independently:

**Multiple functional regimes, not one corpus.** Five domains — general text,
math/reasoning, multihop/RAG, code, tool use — over seven sub-types. General
text is raw FineWeb-Edu prose with no chat template; the other six are
teacher-native renders (system block, `<think>…</think>`, final answer,
`<|im_end|>`) through the official template, because a depth map chosen on prose
alone would be chosen on the one regime E7 showed is separable from the target.

**Leakage-safe against the promotion decision, not merely against itself.** The
frozen 150-prompt behaviour battery is stratified-sampled from the **0.86M
rung's** verified-correct sessions, and the 0.86M rung is a prefix of the 2.96M
rung the E8 arms train on. Calibration sessions are therefore drawn only from the
corpus tail **beyond the 5.50M rung** — blocks no rung consumes and no evaluation
prompt is drawn from — and excluded by `source_id`, not `session_id`, because
turn-expanded sessions sharing a `source_id` are prefixes of one another.

Per sub-type the budget is a fixed number of *prediction positions*, filled with
whole sessions in a deterministic order. Whole sessions only: a session cut
mid-trace has no `</think>` and no terminator, which would silently empty the
very diagnostics §6 asks for.

    python3 scripts/data/build_e8_calibration.py \\
        --out artifacts/stage1/e8_calibration_v1

Outputs (gitignored; manifest is small and committed by hand if useful):

    <out>/items.jsonl      one calibration item per line, token ids inline
    <out>/docs.jsonl       general-text source documents, for disjointness proof
    <out>/manifest.json    hashes, domain membership, exclusions, tag definitions
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.data.ladder import select_val_blocks  # noqa: E402
from aadistill.data.sessions import render_session, render_system_block  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts/data"))
from check_stream_disjointness import row_text  # noqa: E402

# --- the frozen design ---------------------------------------------------------

# Five domains; the primary score is the unweighted mean over domains of the
# unweighted mean over each domain's sub-types (P6 of the E8 instruction).
DOMAINS: dict[str, list[str]] = {
    "general": ["general"],
    "math": ["gsm8k", "openmath"],
    "rag_multihop": ["rag_evidence", "multihop_qa"],
    "code": ["code"],
    "tool": ["tool_calling"],
}
SUBTYPES = [s for subs in DOMAINS.values() for s in subs]

POSITIONS_PER_SUBTYPE = 8192      # prediction positions, not tokens
SESSION_MIN_TOKENS = 256          # below this a session is mostly template
SESSION_MAX_TOKENS = 2048         # keeps one long `code` session from owning a sub-type
GENERAL_DOC_TOKENS = 1024         # matches the historical FineWeb NLL protocol
GENERAL_DOC_CHAR_MIN = 500        # same floor as holdout_v1

# FineWeb-Edu consumption already on record; E8 starts far past all of it.
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_LICENSE = "ODC-By 1.0"
FINEWEB_RESERVED_END = 40000      # warmup_v1 <848, holdout_v1 ~5040, E7 <31902

TEACHER_ID = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

# Special tokens the diagnostic tags key on. Single tokens in this tokenizer,
# asserted at build time so a tokenizer change cannot silently empty a tag.
SPECIAL = {"think_open": "<think>", "think_close": "</think>",
           "im_end": "<|im_end|>", "tool_call_close": "</tool_call>"}


def rung_source_ids(audit_path: Path, n_blocks: int) -> tuple[set[str], set[str]]:
    """Session ids and source ids consumed by the first `n_blocks` of the pack."""
    session_ids: set[str] = set()
    source_ids: set[str] = set()
    for i, line in enumerate(audit_path.open()):
        if i >= n_blocks:
            break
        for s in json.loads(line)["sessions"]:
            session_ids.add(s["session_id"])
            source_ids.add(str(s["session_id"]).split("#")[0])
    return session_ids, source_ids


def tag_positions(ids: list[int], body_mask: list[bool], n_system: int,
                  special: dict[str, int]) -> dict[str, list[int]]:
    """Diagnostic prediction positions, computed once and frozen in the artifact.

    Position `i` predicts token `i + 1`, so a tag naming a token is the position
    *before* it. Computed here rather than on the pod so the definitions are
    hashed with the calibration set instead of living in a driver that could
    drift between the search and the report.
    """
    n = len(ids)
    tags: dict[str, list[int]] = {k: [] for k in
                                  ("assistant", "reasoning", "final_answer",
                                   "think_close", "eos", "tool_close")}
    supervised = [False] * n_system + list(body_mask)
    if len(supervised) != n:
        raise ValueError(f"mask length {len(supervised)} != {n} tokens")
    sup = [i for i, s in enumerate(supervised) if s]
    if not sup:
        return {}                      # raw prose: no assistant turn to tag

    # Everything is scoped to the supervised span — the final assistant turn.
    # Scoping matters: `<|im_end|>` also closes the system block and every user
    # turn, so an unscoped `eos` tag would measure template punctuation rather
    # than the model's natural termination, and a `</think>` appearing inside a
    # user message would relocate the reasoning span.
    lo, hi = sup[0], sup[-1]
    open_i = next((i for i in range(lo, hi + 1)
                   if ids[i] == special["think_open"]), None)
    close_i = None
    if open_i is not None:
        close_i = next((i for i in range(open_i + 1, hi + 1)
                        if ids[i] == special["think_close"]), None)

    for i in range(n - 1):
        nxt, tgt = ids[i + 1], i + 1
        if not supervised[tgt]:
            continue
        tags["assistant"].append(i)
        if open_i is not None and close_i is not None and open_i < tgt < close_i:
            tags["reasoning"].append(i)
        if close_i is not None and close_i < tgt <= hi:
            tags["final_answer"].append(i)
        if nxt == special["think_close"] and tgt == close_i:
            tags["think_close"].append(i)
        if nxt == special["im_end"]:
            tags["eos"].append(i)
        if nxt == special["tool_call_close"]:
            tags["tool_close"].append(i)
    return {k: v for k, v in tags.items() if v}


def prompt_text(session: dict) -> str:
    """Everything shown to the model, excluding the assistant's own output."""
    return "\n".join(str(m.get("content", "")) for m in session.get("messages", [])
                     if m.get("role") != "assistant")


def build_corpus_items(sessions_path: Path, tokenizer, excluded_sources: set[str],
                       excluded_prompt_hashes: set[str],
                       special: dict[str, int], subtypes: list[str]) -> list[dict]:
    budget_left = {s: POSITIONS_PER_SUBTYPE for s in subtypes}
    items: list[dict] = []
    skipped = {"excluded_source": 0, "excluded_prompt_content": 0,
               "length": 0, "render_error": 0}
    # Deterministic order: the corpus file's own order, which is the seed-free
    # stratified interleave the ladder was built from.
    for line in sessions_path.open():
        if all(v <= 0 for v in budget_left.values()):
            break
        d = json.loads(line)
        st = d.get("data_type")
        if st not in budget_left or budget_left[st] <= 0:
            continue
        if str(d["source_id"]) in excluded_sources:
            skipped["excluded_source"] += 1
            continue
        # A different source item can still carry byte-identical prompt text —
        # glaive tool-calling prompts are formulaic, and the first build of this
        # set contained exactly one such collision. The source-id filter cannot
        # see it, so exclude on prompt content as well.
        if content_sha256(prompt_text(d)) in excluded_prompt_hashes:
            skipped["excluded_prompt_content"] += 1
            continue
        if not SESSION_MIN_TOKENS <= int(d["n_rendered_tokens"]) <= SESSION_MAX_TOKENS:
            skipped["length"] += 1
            continue
        try:
            r = render_session(tokenizer, d)
        except ValueError:
            skipped["render_error"] += 1
            continue
        system_ids = tokenizer(
            render_system_block(tokenizer, r.system_text, r.tools),
            add_special_tokens=False).input_ids
        ids = system_ids + r.body_ids
        n_pred = len(ids) - 1
        if n_pred <= 0:
            continue
        items.append({
            "item_id": f"{st}/{d['id']}",
            "domain": next(k for k, v in DOMAINS.items() if st in v),
            "subtype": st,
            "source": "corpus_v2",
            "session_id": d["id"],
            "source_id": d["source_id"],
            "candidate_sha256": d.get("candidate_sha256"),
            "n_tokens": len(ids),
            "n_prediction_positions": n_pred,
            "templated": True,
            "ids": ids,
            "tags": tag_positions(ids, r.body_mask, len(system_ids), special),
        })
        budget_left[st] -= n_pred
    short = {s: v for s, v in budget_left.items() if v > 0}
    if short:
        raise SystemExit(
            f"calibration budget unreachable for {short}; the untouched corpus "
            "tail does not contain enough eligible sessions")
    return items, skipped


def build_general_items(tokenizer, docs_path: Path, special: dict[str, int]):
    """Tokenize pre-fetched FineWeb documents until the sub-type budget is met.

    Documents arrive from `scripts/data/fetch_fineweb_docs.py`, which owns the
    network step and the revision assertion. Consuming a hashed file keeps this
    build deterministic and offline.
    """
    source = json.loads(docs_path.with_suffix(".manifest.json").read_text())
    if source["revision"] != FINEWEB_REVISION:
        raise SystemExit(
            f"{docs_path} was fetched from revision {source['revision']}, not the "
            f"pinned {FINEWEB_REVISION}")
    if source["chat_template_applied"]:
        raise SystemExit("general text must be raw prose, not a rendered chat")
    docs = [json.loads(l) for l in docs_path.open() if l.strip()]
    items, used, budget = [], [], POSITIONS_PER_SUBTYPE
    for d in docs:
        if budget <= 0:
            break
        text = d["text"].strip()
        if len(text) < GENERAL_DOC_CHAR_MIN:
            continue
        if content_sha256(text) != d["sha256"]:
            raise SystemExit(f"{d['id']}: document text does not match its hash")
        ids = tokenizer(text, add_special_tokens=False).input_ids[:GENERAL_DOC_TOKENS]
        if len(ids) < 2:
            continue
        items.append({
            "item_id": f"general/{d['id']}",
            "domain": "general", "subtype": "general", "source": FINEWEB_DATASET,
            "index": d["index"], "doc_sha256": d["sha256"],
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1,
            "templated": False, "ids": ids,
            # Raw prose has no assistant turn, no reasoning delimiters and no
            # terminator: an empty tag set is the correct answer, not a bug.
            "tags": tag_positions(ids, [False] * len(ids), 0, special),
        })
        used.append(d)
        budget -= len(ids) - 1
    if budget > 0:
        raise SystemExit(
            f"general-text budget short by {budget} positions; fetch more than "
            f"{len(docs)} documents")
    return items, used, source


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--exclude-through-rung", type=int, default=5500000,
                    help="exclude every session the pack consumes up to this rung")
    ap.add_argument("--val-blocks", type=int, default=16,
                    help="the pack's canonical validation slice, also excluded")
    ap.add_argument("--reserved", action="append", default=[],
                    help="jsonl prompt sets whose content must not appear")
    ap.add_argument("--general-docs", default="",
                    help="docs.jsonl from scripts/data/fetch_fineweb_docs.py")
    ap.add_argument("--skip-general", action="store_true",
                    help="build the templated sub-types only (offline dry run)")
    args = ap.parse_args()

    if not args.skip_general and not args.general_docs:
        raise SystemExit(
            "--general-docs is required: fetch it first with "
            "scripts/data/fetch_fineweb_docs.py, or pass --skip-general for a "
            "templated-only dry run (which is NOT the frozen calibration set)")

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    pack = Path(args.pack) if Path(args.pack).is_absolute() else REPO_ROOT / args.pack
    sessions_path = (Path(args.sessions) if Path(args.sessions).is_absolute()
                     else REPO_ROOT / args.sessions)

    ladder = json.loads((pack / "ladder.json").read_text())
    rungs = {r["target_supervised_tokens"]: r for r in ladder["rungs"]}
    if args.exclude_through_rung not in rungs:
        raise SystemExit(f"rung {args.exclude_through_rung} not in {sorted(rungs)}")
    n_excluded_blocks = rungs[args.exclude_through_rung]["n_blocks"]
    excluded_sessions, excluded_sources = rung_source_ids(
        pack / "audit.jsonl", n_excluded_blocks)

    # The pack's canonical validation slice also lives in the tail beyond the
    # largest rung — the same region the calibration sessions come from. The
    # first build of this set put one `openmath` session in both, which would
    # have calibrated the depth map on the teacher-native held-out CE that the
    # init-NLL comparison then reports. Excluded, and proven excluded.
    audit = [json.loads(l) for l in (pack / "audit.jsonl").open() if l.strip()]
    max_rung_blocks = max(int(r["n_blocks"]) for r in ladder["rungs"]
                          if r.get("reachable", False))
    val_idx = select_val_blocks(audit, max_rung_blocks, args.val_blocks)
    val_sessions = {s["session_id"] for i in val_idx for s in audit[i]["sessions"]}
    excluded_sessions |= val_sessions
    excluded_sources |= {sid.split("#")[0] for sid in val_sessions}

    excluded_prompt_hashes: set[str] = set()
    for line in sessions_path.open():
        d = json.loads(line)
        if d["id"] in excluded_sessions:
            excluded_prompt_hashes.add(content_sha256(prompt_text(d)))
    reserved_prompt_files = []
    for pattern in args.reserved:
        matches = sorted(glob.glob(str(
            Path(pattern) if Path(pattern).is_absolute() else REPO_ROOT / pattern)))
        if not matches:
            raise SystemExit(
                f"--reserved {pattern!r} matched no file; refusing to build a set "
                "that claims separation from something that was not read")
        for m in matches:
            reserved_prompt_files.append(str(Path(m).relative_to(REPO_ROOT)))
            for line in Path(m).read_text(errors="replace").splitlines():
                if line.strip():
                    excluded_prompt_hashes.add(content_sha256(row_text(json.loads(line))))

    from aadistill.models.teacher import tokenizer_hash
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, revision=TEACHER_REVISION)
    special = {}
    for name, text in SPECIAL.items():
        enc = tokenizer(text, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise SystemExit(f"{text!r} is not a single token ({enc}); the tag "
                             "definitions assume it is")
        special[name] = enc[0]

    items, skipped = build_corpus_items(
        sessions_path, tokenizer, excluded_sources, excluded_prompt_hashes,
        special, [s for s in SUBTYPES if s != "general"])
    docs: list[dict] = []
    general_source = None
    if not args.skip_general:
        gd = (Path(args.general_docs) if Path(args.general_docs).is_absolute()
              else REPO_ROOT / args.general_docs)
        g_items, docs, general_source = build_general_items(tokenizer, gd, special)
        items = g_items + items

    # A session that leaked in despite the source-id filter is a hard error.
    leaked = [i["session_id"] for i in items
              if i.get("session_id") in excluded_sessions]
    if leaked:
        raise SystemExit(f"excluded sessions present in the calibration set: {leaked[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.jsonl"
    items_path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n"
                                  for i in items))
    docs_path = out / "docs.jsonl"
    docs_path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n"
                                 for d in docs))

    by_sub: dict[str, dict] = {}
    for i in items:
        e = by_sub.setdefault(i["subtype"], {"items": 0, "tokens": 0, "positions": 0})
        e["items"] += 1
        e["tokens"] += i["n_tokens"]
        e["positions"] += i["n_prediction_positions"]
    tag_totals: dict[str, int] = {}
    for i in items:
        for k, v in i["tags"].items():
            tag_totals[k] = tag_totals.get(k, 0) + len(v)

    manifest = {
        "artifact": "e8_depth_calibration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "purpose": "frozen calibration mixture for E8 contribution-guided depth "
                   "selection; NOT training data and NOT an evaluation set",
        "design": {
            "domains": DOMAINS,
            "aggregation": "unweighted mean over domains of the unweighted mean "
                           "over each domain's sub-types of that sub-type's "
                           "token-mean KL",
            "positions_per_subtype_budget": POSITIONS_PER_SUBTYPE,
            "session_token_window": [SESSION_MIN_TOKENS, SESSION_MAX_TOKENS],
            "general_doc_token_cap": GENERAL_DOC_TOKENS,
            "whole_sessions_only": True,
            "primary_position_scope": "all prediction positions of every item, "
                                      "matching the training objective's "
                                      "kd_scope=all content scope",
        },
        "tokenizer": {"id": TEACHER_ID, "revision": TEACHER_REVISION,
                      "sha256": tokenizer_hash(tokenizer),
                      "special_token_ids": special},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
        "leakage_control": {
            "pack": rel(pack),
            "excluded_through_rung": args.exclude_through_rung,
            "excluded_blocks": n_excluded_blocks,
            "excluded_sessions": len(excluded_sessions),
            "excluded_ladder_val_blocks": val_idx,
            "excluded_ladder_val_sessions": len(val_sessions),
            "excluded_source_ids": len(excluded_sources),
            "exclusion_key": "source_id (turn-expanded siblings are prefixes)",
            "why": "the frozen 150-prompt behaviour battery is sampled from the "
                   "0.86M rung, a prefix of the 2.96M rung the arms train on",
            "skipped": skipped,
            "excluded_prompt_content_hashes": len(excluded_prompt_hashes),
            "reserved_prompt_files": reserved_prompt_files,
        },
        "general_text": None if args.skip_general else {
            "dataset": FINEWEB_DATASET, "config": FINEWEB_CONFIG,
            "revision": general_source["revision"], "license": FINEWEB_LICENSE,
            "fetch_index_range": general_source["index_range"],
            "fetched_docs": general_source["docs"],
            "docs_used": len(docs),
            "fetch_manifest_command": general_source["command"],
            "fetched_sha256": general_source["output"]["sha256"],
            "chat_template_applied": False,
        },
        "totals": {
            "items": len(items),
            "tokens": sum(i["n_tokens"] for i in items),
            "prediction_positions": sum(i["n_prediction_positions"] for i in items),
            "by_subtype": by_sub,
            "tagged_positions": tag_totals,
        },
        "outputs": {
            "items": {"path": rel(items_path),
                      "sha256": sha256_file(items_path)},
            "docs": {"path": rel(docs_path),
                     "sha256": sha256_file(docs_path)},
        },
        "content_sha256": hashlib.sha256(
            "".join(f"{i['item_id']}:{sha_ids(i['ids'])}\n" for i in items)
            .encode()).hexdigest(),
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({"items": len(items), "by_subtype": by_sub,
                      "tagged": tag_totals,
                      "content_sha256": manifest["content_sha256"]}, indent=2))
    print(f"-> {out}")
    return 0


def rel(path: Path) -> str:
    """Repo-relative when it can be, absolute when the caller wrote elsewhere."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())

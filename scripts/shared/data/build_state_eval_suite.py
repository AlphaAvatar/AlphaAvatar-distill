"""Build the frozen initializer-state evaluation suite (role STATE_EVALUATION).

    PYTHONPATH=src python scripts/data/build_state_eval_suite.py \
        --out artifacts/stage1/state_eval_v1

Its **only** job is cheap step-0 scoring of materialized search states — every
intermediate and every complete leaf — against the **original teacher**, for beam
ranking. It is never a training set, never a rollout battery, and never the thing
a final result is reported on.

Design, and why each part is what it is:

**Teacher-native sessions, not raw prompts.** The metric that matters most here is
critical-token fidelity: whether a compressed state still puts mass on the tokens
that decide when reasoning ends and generation stops. Those tokens only exist in a
rendered assistant turn, so the reasoning/RAG/code/tool strata are real teacher
sessions from the recovery corpus. The `general` stratum is untemplated prose,
matching the historical FineWeb NLL protocol.

**The tagging is E8a's, imported rather than reimplemented.** `tag_positions` and
`rung_source_ids` come from `build_e8_calibration.py`. A second implementation of
"which position predicts `</think>`" would be a second definition, and the two
would drift.

**Disjointness is inherited and then extended.** The exclusion through the 5.5M
rung already covers the 0.86M recovery-probe training set with a wide margin. On
top of that this suite excludes E8a's own calibration items (operator calibration
is a different role), and every prompt in the frozen promotion battery.

**Small on purpose.** The teacher reference is recomputed per candidate rather
than cached — caching full-vocabulary logits for a suite this size would be tens
of GiB. A ~60k-position suite costs one ~5.6 s teacher forward per candidate on an
L40S, so the whole search pays a few minutes. Growing the suite is a real cost.
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
from aadistill.data.sessions import render_session, render_system_block  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from build_e8_calibration import (  # noqa: E402
    SPECIAL,
    prompt_text,
    rung_source_ids,
    sha_ids,
    tag_positions,
)

SUITE_ID = "state_eval"
SUITE_VERSION = 1

#: Five domains, each with declared sub-types. The primary aggregate is the
#: unweighted mean over domains of the unweighted mean over sub-types, so a
#: domain's influence is set by this table and not by its token count.
DOMAINS: dict[str, list[str]] = {
    "general": ["general"],
    "reasoning_math": ["gsm8k", "openmath"],
    "rag_multihop": ["rag_evidence", "multihop_qa"],
    "code": ["code"],
    "tool": ["tool_calling"],
}
SUBTYPES = [s for subs in DOMAINS.values() for s in subs]

#: Which tag classes feed `state.critical_token_kl`.
#:
#: Deliberately the four *narrow, decision-bearing* classes. `reasoning` and
#: `assistant` are recorded too, but they cover most supervised positions, so
#: including them in the aggregate would drown out precisely the rare tokens the
#: metric exists to watch — the project's dominant failure is that the model does
#: not stop (~31% of rollouts hit the context limit), and that failure lives on
#: `think_close` and `eos`.
CRITICAL_TAGS = ("think_close", "eos", "final_answer", "tool_close")
DIAGNOSTIC_TAGS = ("assistant", "reasoning")

#: A sub-type that exists to exercise a rare critical class must actually contain
#: it. Without this the tool stratum fills with sessions whose final assistant
#: turn happens not to call a tool, and `tool_close` lands on a handful of
#: positions while carrying a full quarter of the unweighted critical-class mean.
REQUIRED_TAG_BY_SUBTYPE = {"tool_calling": "tool_close"}

#: Floor per critical class. The classes are averaged unweighted, so a class
#: estimated from a few positions would contribute as much noise as the others
#: contribute signal. `think_close` and `eos` are structurally one position per
#: session, so this floor is effectively a floor on templated items too.
MIN_CRITICAL_POSITIONS = 25

#: Minimum items per sub-type, applied *in addition* to the position budget.
#:
#: A sub-type's weight in the domain mean comes from the DOMAINS table, not from
#: how many positions it contributes, so collecting more tool sessions does not
#: tilt the aggregate — it only estimates the tool sub-type better. `tool_close`
#: is one position per session, so its floor is really a floor on tool sessions,
#: and the position budget alone stops at ~12 of them.
MIN_ITEMS_BY_SUBTYPE = {"tool_calling": 28}
MIN_ITEMS_DEFAULT = 5

POSITIONS_PER_SUBTYPE = 8192
SESSION_MIN_TOKENS = 256
SESSION_MAX_TOKENS = 2048
GENERAL_DOC_TOKENS = 1024
GENERAL_DOC_CHAR_MIN = 500

TEACHER_ID = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
#: The Stage-1 checkpoint carries the teacher's tokenizer and chat template.
TOKENIZER_DIR = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


#: Shingle size and threshold for near-duplicate rejection against the training
#: rung. RAG and multihop sessions share long context paragraphs, so two items can
#: be textually distinct questions over identical evidence; that is close enough to
#: training data that it should not also be a state-eval item.
NEAR_DUP_SHINGLES = 8
NEAR_DUP_THRESHOLD = 0.5
#: Minimum user-text length for the shingle rule to be applied at all.
#:
#: A tool question is 10-20 words and formulaic ("Can you tell me the latest news
#: headlines for X?"), so an 8-word shingle Jaccard over it is an exact-match test
#: wearing a disguise: it rejects almost every tool session as a near duplicate of
#: almost every other, which is a statement about the phrasing template rather
#: than about leakage. Below this length the exact content hash is the only
#: defence, and that limitation is recorded rather than hidden.
NEAR_DUP_MIN_WORDS = 40


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def user_text(session: dict) -> str:
    """Only the user turns.

    Near-duplicate detection runs on this rather than on the full prompt because
    the full prompt carries the system block and, for tool sessions, the entire
    tool schema. Two different tool questions share that boilerplate almost
    completely, so a shingle rule over the full prompt rejects nearly every tool
    session as a near duplicate of nearly every other one — which is a statement
    about the template, not about the data. Exact-hash exclusion still runs on
    the full prompt, where a match really is a duplicate.
    """
    return "\n".join(str(m.get("content", "")) for m in session.get("messages", [])
                     if m.get("role") == "user")


def shingles(text: str, k: int = NEAR_DUP_SHINGLES) -> set[str]:
    words = norm_text(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def excluded_from_other_roles(args, calib_dir: Path, battery_dir: Path):
    """Every id and content hash this suite must avoid, with its provenance."""
    provenance: dict[str, dict] = {}
    source_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    item_ids: set[str] = set()

    # 1. Recovery training. Excluding through the 5.5M rung covers the 0.86M
    #    probe rung many times over, and matches what E8a excluded.
    pack = REPO_ROOT / args.pack
    ladder = json.loads((pack / "ladder.json").read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == args.exclude_through_rung)
    n_blocks = int(rung["n_blocks"])
    sessions, sources = rung_source_ids(pack / "audit.jsonl", n_blocks)
    source_ids |= sources
    provenance["recovery_training"] = {
        "pack": args.pack, "excluded_through_rung": args.exclude_through_rung,
        "excluded_blocks": n_blocks, "excluded_sessions": len(sessions),
        "excluded_source_ids": len(sources),
        "note": ("the 0.86M probe rung is a strict prefix of this, so the "
                 "recovery-probe training set is excluded with margin"),
    }

    # Validation blocks the pack reserves, immediately after the largest rung.
    val_sessions, val_sources = set(), set()
    with (pack / "audit.jsonl").open() as fh:
        for i, line in enumerate(fh):
            if n_blocks <= i < n_blocks + args.val_blocks:
                for s in json.loads(line)["sessions"]:
                    val_sessions.add(s["session_id"])
                    val_sources.add(str(s["session_id"]).split("#")[0])
    source_ids |= val_sources
    provenance["pack_validation_slice"] = {
        "val_blocks": args.val_blocks, "excluded_sessions": len(val_sessions)}

    # 2. Operator calibration — a different role, and the one most likely to be
    #    confused with this suite.
    calib_items = [json.loads(l) for l in (calib_dir / "items.jsonl").open()
                   if l.strip()]
    calib_sources = {str(i.get("source_id") or i["item_id"].split("/")[-1].split("#")[0])
                     for i in calib_items}
    calib_token_hashes = {sha_ids(i["ids"]) for i in calib_items}
    item_ids |= {i["item_id"] for i in calib_items}
    source_ids |= calib_sources
    provenance["operator_calibration"] = {
        "asset": str(calib_dir.relative_to(REPO_ROOT)), "n_items": len(calib_items),
        "excluded_item_ids": len(calib_items),
        "excluded_token_hashes": len(calib_token_hashes)}

    # 3. Final promotion. Prompt content, because the battery stores rendered
    #    prompts and this suite stores sessions — the shared identity is the text.
    battery_prompts = 0
    for path in sorted(battery_dir.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("prompt_text") or prompt_text(row)
            prompt_hashes.add(content_sha256(text))
            battery_prompts += 1
    provenance["final_promotion"] = {
        "asset": str(battery_dir.relative_to(REPO_ROOT)),
        "n_prompts": battery_prompts,
        "excluded_prompt_hashes": len(prompt_hashes)}

    # Content, not only ids. The same question can reach the corpus twice under
    # two source ids, and an id-only exclusion cannot see it — a first build of
    # this suite shipped exactly one such item.
    trained_hashes: set[str] = set()
    trained_shingles: set[str] = set()
    n_trained = 0
    all_sessions = {}
    for line in (REPO_ROOT / args.sessions).open():
        d = json.loads(line)
        all_sessions[str(d["id"])] = d
    for session_id in sessions:
        d = all_sessions.get(str(session_id))
        if d is None:
            continue
        trained_hashes.add(content_sha256(norm_text(prompt_text(d))))
        user = user_text(d)
        if len(norm_text(user).split()) >= NEAR_DUP_MIN_WORDS:
            trained_shingles |= shingles(user)
        n_trained += 1
    provenance["recovery_training"]["excluded_prompt_hashes"] = len(trained_hashes)
    provenance["recovery_training"]["near_duplicate_rule"] = {
        "shingles": NEAR_DUP_SHINGLES, "threshold": NEAR_DUP_THRESHOLD,
        "sessions_hashed": n_trained}

    # Near duplicates of the *calibration* mixture too. squad_v2 asks many
    # questions over one passage, so a state-eval item can share ~97% of its text
    # with a calibration item while carrying a different question. That is mild
    # circularity: the operator optimizes on the passage and the beam then grades
    # the result on the same passage. Cheap to avoid, so avoided.
    calib_resolved = 0
    for item in calib_items:
        source_id = str(item.get("source_id") or "")
        if not source_id:
            continue
        for d in all_sessions.values():
            if str(d.get("source_id")) != source_id:
                continue
            user = user_text(d)
            if len(norm_text(user).split()) >= NEAR_DUP_MIN_WORDS:
                trained_shingles |= shingles(user)
                calib_resolved += 1
            break
    provenance["operator_calibration"]["near_duplicate_sessions_resolved"] = calib_resolved
    provenance["operator_calibration"]["near_duplicate_rule"] = {
        "shingles": NEAR_DUP_SHINGLES, "threshold": NEAR_DUP_THRESHOLD,
        "min_user_words": NEAR_DUP_MIN_WORDS}
    return (source_ids, prompt_hashes, item_ids, calib_token_hashes, trained_hashes,
            trained_shingles, provenance)


def build_session_items(sessions_path: Path, tokenizer, excluded_sources: set[str],
                        excluded_prompt_hashes: set[str],
                        excluded_token_hashes: set[str], trained_hashes: set[str],
                        trained_shingles: set[str], special: dict[str, int]):
    subtypes = [s for s in SUBTYPES if s != "general"]
    budget_left = {s: POSITIONS_PER_SUBTYPE for s in subtypes}
    min_items = {s: MIN_ITEMS_BY_SUBTYPE.get(s, MIN_ITEMS_DEFAULT) for s in subtypes}
    taken = {s: 0 for s in subtypes}
    items: list[dict] = []
    skipped = {"excluded_source": 0, "excluded_prompt_content": 0,
               "excluded_token_content": 0, "length": 0, "render_error": 0,
               "no_tags": 0, "missing_required_tag": 0,
               "excluded_training_prompt_content": 0,
               "excluded_training_near_duplicate": 0}

    def wants(st: str) -> bool:
        return budget_left[st] > 0 or taken[st] < min_items[st]

    for line in sessions_path.open():          # corpus order: seed-free, stratified
        if not any(wants(s) for s in subtypes):
            break
        d = json.loads(line)
        st = d.get("data_type")
        if st not in budget_left or not wants(st):
            continue
        if str(d.get("source_id")) in excluded_sources:
            skipped["excluded_source"] += 1
            continue
        text = prompt_text(d)
        if content_sha256(text) in excluded_prompt_hashes:
            skipped["excluded_prompt_content"] += 1
            continue
        if content_sha256(norm_text(text)) in trained_hashes:
            skipped["excluded_training_prompt_content"] += 1
            continue
        user = user_text(d)
        if len(norm_text(user).split()) >= NEAR_DUP_MIN_WORDS:
            item_shingles = shingles(user)
            if item_shingles and (len(item_shingles & trained_shingles)
                                  / len(item_shingles)) >= NEAR_DUP_THRESHOLD:
                skipped["excluded_training_near_duplicate"] += 1
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
        ids = system_ids + list(r.body_ids)
        if len(ids) - 1 <= 0:
            skipped["render_error"] += 1
            continue
        if sha_ids(ids) in excluded_token_hashes:
            skipped["excluded_token_content"] += 1
            continue
        tags = tag_positions(ids, list(r.body_mask), len(system_ids), special)
        if not any(tags.get(t) for t in CRITICAL_TAGS):
            # A session with no decision-bearing token contributes nothing to the
            # metric this suite exists to compute.
            skipped["no_tags"] += 1
            continue
        required = REQUIRED_TAG_BY_SUBTYPE.get(st)
        if required and not tags.get(required):
            skipped["missing_required_tag"] += 1
            continue
        items.append({
            "item_id": f"{st}/{d['id']}",
            "domain": next(k for k, v in DOMAINS.items() if st in v),
            "subtype": st,
            "source": d.get("source"),
            "source_id": d.get("source_id"),
            "session_id": d["id"],
            "templated": True,
            "n_tokens": len(ids),
            "n_prediction_positions": len(ids) - 1,
            "ids": ids,
            "tags": tags,
        })
        budget_left[st] -= len(ids) - 1
        taken[st] += 1

    short = {s: (taken[s], min_items[s]) for s in subtypes if taken[s] < min_items[s]}
    if short:
        raise SystemExit(
            f"sub-types below their item floor (taken, required): {short}. The pool "
            "after role exclusion is too small; widen the pool rather than lowering "
            "the floor after seeing it.")
    return items, skipped, budget_left, taken


def build_general_items(docs_path: Path, tokenizer, used_indices: set[str]):
    items, used = [], []
    budget = POSITIONS_PER_SUBTYPE
    for line in docs_path.open():
        if budget <= 0:
            break
        d = json.loads(line)
        if str(d["index"]) in used_indices:
            continue                      # already an operator-calibration document
        text = d["text"].strip()
        if len(text) < GENERAL_DOC_CHAR_MIN:
            continue
        if content_sha256(text) != d["sha256"]:
            raise SystemExit(f"{d['id']}: document text does not match its hash")
        ids = tokenizer(text, add_special_tokens=False).input_ids[:GENERAL_DOC_TOKENS]
        if len(ids) < 2:
            continue
        items.append({
            "item_id": f"general/fineweb-{d['index']}",
            "domain": "general", "subtype": "general",
            "source": "HuggingFaceFW/fineweb-edu", "source_id": f"fineweb-{d['index']}",
            "index": str(d["index"]), "doc_sha256": d["sha256"], "templated": False,
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1,
            "ids": ids, "tags": {},
        })
        used.append(str(d["index"]))
        budget -= len(ids) - 1
    return items, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--sessions", default="artifacts/stage3/corpus_v2/sessions.jsonl")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--battery", default="artifacts/eval/battery_v2")
    ap.add_argument("--general-docs",
                    default="artifacts/stage1/e8_calibration_v1/general_docs.jsonl")
    # 3.4x margin over the 0.86M probe rung. Wider than the probes need, narrower
    # than E8a's 5.5M: at 5.5M the surviving tool pool cannot reach the
    # `tool_close` floor, and the correct response to that is to widen the pool
    # rather than to lower a floor that was set before the data was seen.
    ap.add_argument("--exclude-through-rung", type=int, default=2_960_000)
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--tokenizer", default=TOKENIZER_DIR)
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    calib_dir = REPO_ROOT / args.calibration
    battery_dir = REPO_ROOT / args.battery

    tokenizer = load_tokenizer(REPO_ROOT / args.tokenizer)
    special = {}
    for name, text in SPECIAL.items():
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise SystemExit(f"{name} ({text!r}) is not a single token: {ids}")
        special[name] = ids[0]

    (source_ids, prompt_hashes, calib_item_ids, calib_token_hashes, trained_hashes,
     trained_shingles, provenance) = excluded_from_other_roles(
        args, calib_dir, battery_dir)

    items, skipped, budget_left, taken = build_session_items(
        REPO_ROOT / args.sessions, tokenizer, source_ids, prompt_hashes,
        calib_token_hashes, trained_hashes, trained_shingles, special)

    calib_general_indices = {
        str(json.loads(l)["index"])
        for l in (calib_dir / "items.jsonl").open()
        if l.strip() and json.loads(l)["domain"] == "general"
    }
    general_items, general_used = build_general_items(
        REPO_ROOT / args.general_docs, tokenizer, calib_general_indices)
    items = general_items + items

    missing = [s for s in SUBTYPES if not any(i["subtype"] == s for i in items)]
    if missing:
        raise SystemExit(
            f"sub-types with no items: {missing}. A silently absent sub-type "
            "reweights its domain, so this fails rather than shipping.")

    items.sort(key=lambda i: (i["domain"], i["subtype"], i["item_id"]))
    items_path = out / "items.jsonl"
    with items_path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, sort_keys=True) + "\n")

    by_subtype = {s: sum(1 for i in items if i["subtype"] == s) for s in SUBTYPES}
    positions = {s: sum(i["n_prediction_positions"] for i in items
                        if i["subtype"] == s) for s in SUBTYPES}
    tag_totals = {t: sum(len(i["tags"].get(t, [])) for i in items)
                  for t in (*CRITICAL_TAGS, *DIAGNOSTIC_TAGS)}
    thin = {t: tag_totals[t] for t in CRITICAL_TAGS
            if tag_totals[t] < MIN_CRITICAL_POSITIONS}
    if thin:
        raise SystemExit(
            f"critical-token classes below the {MIN_CRITICAL_POSITIONS}-position "
            f"floor: {thin}. The classes are averaged unweighted, so a thin class "
            "contributes as much noise as the others contribute signal.")

    content_hash = hashlib.sha256(
        "".join(f"{i['item_id']}:{sha_ids(i['ids'])}\n" for i in items).encode()
    ).hexdigest()

    manifest = {
        "artifact": f"{SUITE_ID}_v{SUITE_VERSION}",
        "role": "INITIALIZER_STATE_EVAL",
        "purpose": ("frozen step-0 evaluation of materialized search states against "
                    "the ORIGINAL teacher, for beam ranking only; not training "
                    "data, not a rollout battery, not a promotion asset"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "suite_id": SUITE_ID, "version": SUITE_VERSION,
        "reference_model": {
            "role": "original_teacher",
            "id": TEACHER_ID, "revision": TEACHER_REVISION,
            "note": ("the global reference for every state at every depth; never "
                     "the parent state"),
        },
        "tokenizer": {
            "id": TEACHER_ID, "revision": TEACHER_REVISION,
            "source_dir": args.tokenizer,
            "tokenizer_sha256": sha256_file(REPO_ROOT / args.tokenizer / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(
                REPO_ROOT / args.tokenizer / "tokenizer_config.json"),
            "chat_template_sha256": sha256_file(
                REPO_ROOT / args.tokenizer / "chat_template.jinja"),
            "special_token_ids": special,
        },
        "domains": DOMAINS,
        "critical_tags": list(CRITICAL_TAGS),
        "diagnostic_tags": list(DIAGNOSTIC_TAGS),
        "critical_tag_definitions": {
            "think_close": "position predicting </think>; decides whether reasoning ends",
            "eos": "position predicting <|im_end|> inside the assistant turn; natural termination",
            "final_answer": "positions after </think> through the end of the turn",
            "tool_close": "position predicting </tool_call>",
            "assistant": "DIAGNOSTIC: every supervised assistant position",
            "reasoning": "DIAGNOSTIC: positions strictly inside <think>...</think>",
        },
        "sampling_rule": {
            "order": "recovery-corpus file order (seed-free stratified interleave)",
            "budget": f"{POSITIONS_PER_SUBTYPE} prediction positions per sub-type",
            "session_token_window": [SESSION_MIN_TOKENS, SESSION_MAX_TOKENS],
            "general_doc_tokens": GENERAL_DOC_TOKENS,
            "general_doc_char_min": GENERAL_DOC_CHAR_MIN,
            "requires_a_critical_token": True,
            "required_tag_by_subtype": REQUIRED_TAG_BY_SUBTYPE,
            "min_critical_positions": MIN_CRITICAL_POSITIONS,
            "min_items_by_subtype": MIN_ITEMS_BY_SUBTYPE,
            "min_items_default": MIN_ITEMS_DEFAULT,
            "near_duplicate_rejection": {
                "against": "recovery-training rung prompts AND the operator "
                           "calibration mixture",
                "scope": ("user turns only; the system block and tool schema are "
                          "shared boilerplate and would reject every tool session"),
                "shingles": NEAR_DUP_SHINGLES, "threshold": NEAR_DUP_THRESHOLD,
                "min_user_words": NEAR_DUP_MIN_WORDS,
                "limitation": ("prompts shorter than the word floor are protected by "
                               "exact content hashing only; the tool stratum is "
                               "mostly in that regime")},
            "stopping_rule": ("fill each sub-type until BOTH its position budget is "
                              "met and its item floor is reached"),
            "deterministic": True, "seed": None,
        },
        "sources": {
            "sessions": {"path": args.sessions,
                         "sha256": sha256_file(REPO_ROOT / args.sessions)},
            "general_docs": {"path": args.general_docs,
                             "sha256": sha256_file(REPO_ROOT / args.general_docs),
                             "manifest": "artifacts/stage1/e8_calibration_v1/"
                                         "general_docs.manifest.json",
                             "note": ("same fetched FineWeb-Edu range as the operator "
                                      "calibration set, disjoint at document level; "
                                      "the indices used there are excluded here")},
        },
        "counts": {
            "n_items": len(items),
            "by_subtype": by_subtype,
            "positions_by_subtype": positions,
            "total_prediction_positions": sum(positions.values()),
            "tag_positions": tag_totals,
            "budget_unspent_by_subtype": budget_left,
            "items_taken_by_subtype": taken,
            "skipped": skipped,
        },
        "isolation": {
            "excluded_roles": provenance,
            "excluded_source_ids": len(source_ids),
            "excluded_prompt_hashes": len(prompt_hashes),
            "excluded_calibration_item_ids": len(calib_item_ids),
            "general_indices_used": general_used,
            "general_indices_excluded": sorted(calib_general_indices),
        },
        "outputs": {
            "items": {"path": str(items_path.relative_to(REPO_ROOT)),
                      "sha256": sha256_file(items_path)},
        },
        "content_sha256": content_hash,
        "item_ids": [i["item_id"] for i in items],
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "n_items": len(items),
        "by_subtype": by_subtype,
        "positions": positions,
        "total_positions": sum(positions.values()),
        "tags": tag_totals,
        "content_sha256": content_hash,
        "skipped": skipped,
    }, indent=2))


if __name__ == "__main__":
    main()

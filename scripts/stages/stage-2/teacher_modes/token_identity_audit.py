#!/usr/bin/env python3
"""Token identity of the existing re-rendered training path, and the isolated path on real data ($0, CPU).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_identity_audit.py \\
        --write logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json

The H1-PA fluid increment's 767 admitted sessions are rebuilt two ways from the
very generation records they came from (P1, P1t, PA1; Thinking-2507):

* **as trained**: `aadistill.data.sessions.render_session` -- the record's
  messages re-rendered through the chat template, the CE mask on the final
  assistant turn -- which is the path every H1/H1-PA unit trained on;
* **as generated**: the engine's own ``prompt_token_ids`` + ``output_token_ids``
  (`aadistill.data.token_documents`).

It reports where the two differ: context tokens, supervised tokens, tokens
supervised that the teacher never generated (a template prefill, the newline
after ``<|im_end|>``), and generated tokens left unsupervised. Each difference
is then classified (``target_differences``): the template prefill; inside the
generation, a re-tokenization of the same text, a whitespace normalization, or
a content change; where it sits relative to ``</think>``; how many supervised
positions follow it (conditioned on the re-rendered text rather than the
generated one); and the loss weight all of that carried under H1-PA's recipe.

It then packs the as-generated documents both ways at the training sequence
length: `pack_isolated`, checked on a small random model over real-length
sequences (every packed document's final hidden states equal the document's
own), and `pack_cross_context`, whose foreign-context exposure is set beside a
length-only proxy for shorter non-thinking documents (``context_strategy_estimate``).

These are measurements of the two serializations, not a verdict on H1-PA:
nothing is trained, regenerated or rescored, no historical identity or result
changes, and the fluid pack is read, not rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

GEN = REPO / "artifacts/stages/stage-2/reasoning_data/generation"
RUNS = ("h1gen_20261022_190653", "h1p1t_20261023_094007", "h1pat_20261024_115517")
SESSIONS = REPO / "artifacts/stages/stage-2/reasoning_data/packs/D4A_fluid_whole_sessions.jsonl"
PACK_AUDIT = REPO / "artifacts/stages/stage-2/reasoning_data/packs/D4A_fluid_whole/audit.jsonl"
TEACHER = ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea")
SEQ_LEN = 8192
STOP_IDS = (151645, 151643)


def _ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def turns_by_output_sha():
    out = {}
    for run in RUNS:
        for f in sorted((GEN / run).glob("pass*_generation_input_*.jsonl")):
            for line in f.read_text().split("\n"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                for k, t in enumerate(rec.get("turns") or []):
                    if t.get("output_token_ids"):
                        out[_ids_sha(t["output_token_ids"])] = (run, rec["task_id"], k, t)
    return out


def compare(session: dict, turn: dict, tok) -> dict:
    from aadistill.data.sessions import render_session, render_system_block

    r = render_session(tok, session, block_len=SEQ_LEN)
    sys_ids = tok(render_system_block(tok, r.system_text, r.tools), add_special_tokens=False)["input_ids"]
    if len(sys_ids) != r.n_system_tokens:
        raise SystemExit(f"{session['id']}: system block re-tokenizes to a different length")
    old_ids = list(sys_ids) + list(r.body_ids)
    old_mask = [False] * len(sys_ids) + list(r.body_mask)
    p, o = list(turn["prompt_token_ids"]), list(turn["output_token_ids"])
    native = p + o
    sup_old = [t for t, m in zip(old_ids, old_mask) if m]
    first = next((i for i, (a, b) in enumerate(zip(old_ids, native)) if a != b), None)
    if first is None and len(old_ids) < len(native):
        first = len(old_ids)
    # supervised positions of the old render that are not generated tokens, and
    # generated tokens the old mask leaves out, when the two agree on the prefix
    if old_ids[:len(native)] == native:
        trained_positions = {i for i, m in enumerate(old_mask) if m}
        gen_positions = set(range(len(p), len(native)))
        extra = sorted(trained_positions - gen_positions)
        missing = sorted(gen_positions - trained_positions)
    else:
        extra = missing = None
    return {
        "context_identical": old_ids[:len(p)] == p,
        "native_is_prefix_of_render": old_ids[:len(native)] == native,
        "supervised_equal_generated": sup_old == o,
        "first_divergence": None if first is None else ("context" if first < len(p) else "generation"),
        "render_extra_tail": len(old_ids) - len(native),
        "supervised_not_generated": None if extra is None else [tok.decode([old_ids[i]]) for i in extra],
        "generated_not_supervised": None if missing is None else [tok.decode([native[i]]) for i in missing],
        "supervised_old": len(sup_old), "generated": len(o),
        "_sup_old": sup_old, "_generated": o,
    }


THINK_END = 151668
H1PA_LOSS = {"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0, "kd_scope": "all",
             "_source": "configs/stages/stage-2/reasoning_data/h1pa.json :: recipe.facts (E1's loss block)"}


def classify(sup: list[int], gen: list[int], tok) -> dict:
    """The differences between the supervised token sequence as trained and the
    generated one, as aligned spans with a kind and a location."""
    import difflib
    import re

    t_end = gen.index(THINK_END) if THINK_END in gen else None
    spans, first_gen_div = [], None
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sup, gen, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        ta, tb = tok.decode(sup[i1:i2]), tok.decode(gen[j1:j2])
        if i1 == 0 and j1 == 0 and op == "delete" and ta == "<think>\n":
            kind, where = "template_prefill", "before_generation"
        else:
            kind = ("retokenization" if ta == tb else
                    "whitespace" if re.sub(r"\s", "", ta) == re.sub(r"\s", "", tb) else "content")
            where = ("no_think_end" if t_end is None else "reasoning" if j2 < t_end - 1
                     else "answer" if j1 > t_end + 2 else "think_boundary")
            if first_gen_div is None:
                first_gen_div = i1
        spans.append({"kind": kind, "where": where, "trained_tokens": i2 - i1, "generated_tokens": j2 - j1,
                      "trained_text": ta, "generated_text": tb})
    return {"spans": spans, "supervised_after_first_generation_divergence":
            0 if first_gen_div is None else len(sup) - first_gen_div}


def target_differences(rows: list[dict], tok) -> dict:
    total = sum(r["supervised_old"] for r in rows)
    kinds: dict = {}
    after = 0
    examples: dict = {}
    sessions_with: dict = {}
    for r in rows:
        c = classify(r["_sup_old"], r["_generated"], tok)
        after += c["supervised_after_first_generation_divergence"]
        seen = set()
        for s in c["spans"]:
            key = (s["kind"], s["where"])
            k = kinds.setdefault(key, {"spans": 0, "trained_tokens": 0, "generated_tokens": 0})
            k["spans"] += 1
            k["trained_tokens"] += s["trained_tokens"]
            k["generated_tokens"] += s["generated_tokens"]
            seen.add(s["kind"])
            ex = examples.setdefault(s["kind"], [])
            if len(ex) < 4 and s["kind"] != "template_prefill":
                ex.append({"session": r["session"], "where": s["where"], "trained": repr(s["trained_text"])[:60],
                           "generated": repr(s["generated_text"])[:60]})
        for k in seen:
            sessions_with[k] = sessions_with.get(k, 0) + 1
    by_kind = {}
    for (kind, where), v in sorted(kinds.items()):
        e = by_kind.setdefault(kind, {"sessions": sessions_with.get(kind, 0), "spans": 0, "trained_tokens": 0,
                                      "generated_tokens": 0, "by_location": {}})
        for f in ("spans", "trained_tokens", "generated_tokens"):
            e[f] += v[f]
        e["by_location"][where] = v
    for e in by_kind.values():
        e["share_of_supervised_positions"] = round(e["trained_tokens"] / total, 8)
    return {
        "supervised_positions_as_trained": total,
        "by_kind": by_kind,
        "examples": examples,
        "supervised_positions_after_a_generation_divergence": after,
        "share_after_a_generation_divergence": round(after / total, 6),
        "loss_weights_h1pa": H1PA_LOSS,
        "reading": {
            "template_prefill": ("the `<think>\\n` the Thinking-2507 template prefills in the generation prompt, "
                                 "supervised by CE because the re-rendered assistant turn contains it. It is a "
                                 "template control token the deployment template supplies as context, so the "
                                 "student never needs to generate it; supervising it is unnecessary rather than "
                                 "wrong. Whether it conflicts with the teacher depends on the teacher's own "
                                 "probability of `<think>` after `<|im_start|>assistant\\n`, which needs the "
                                 "teacher's weights (not measured here)"),
            "retokenization": "the same text in different token ids: the conditioning text is unchanged",
            "whitespace": ("the template's newline normalization around the reasoning and the answer: the "
                           "text differs only in whitespace, and every later position is conditioned on the "
                           "normalized text"),
            "content": "a change of non-whitespace text",
            "conditioning": ("H1-PA ran the teacher on the fly on the same re-rendered sequence, so KD compared "
                             "teacher and student on one context everywhere; the difference is that after a "
                             "divergence that context is the re-rendered text, not the teacher's own "
                             "generation. KD was applied at every non-padding position, so the prefill changes "
                             "only what CE supervised"),
            "not_concluded": ("these counts do not show that H1-PA's training was invalid or that this caused "
                              "its low correctness; no such inference is made"),
        },
    }


def context_strategy_estimate(docs, tok, sep) -> dict:
    """Foreign-context exposure under cross-context packing for the as-generated
    documents and for a LENGTH-ONLY proxy of shorter documents (each generation's
    length x 0.25, the cost forecast's expected non-thinking ratio, filled with a
    placeholder id: no content of any trace is used, and nothing here is data)."""
    from aadistill.data.token_documents import TokenDocument, context_exposure, pack_cross_context

    eot = tok.convert_tokens_to_ids("<|im_end|>")
    groups: dict = {}
    for d in docs:
        p = list(d.prompt_ids)
        cut = p.index(eot) + 1 + len(sep)          # the system block and its terminator
        groups.setdefault(tuple(p[:cut]), []).append(d)
    out = {}
    for label, scale in (("as_generated", 1.0), ("length_proxy_x0.25", 0.25)):
        exp = []
        for prefix, ds in groups.items():
            use = ds if scale == 1.0 else [
                TokenDocument(d.doc_id, d.prompt_ids, (0,) * max(0, round(len(d.output_ids) * scale) - 1)
                              + (d.output_ids[-1],)) for d in ds]
            exp.append((len(ds), context_exposure(pack_cross_context(use, SEQ_LEN, 0, shared_prefix=prefix,
                                                                     separator_ids=sep, order_seed=20261027))))
        main = max(exp, key=lambda x: x[0])[1]
        out[label] = {"system_groups": len(groups), "largest_group": main}
    return out


def build() -> dict:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from aadistill.data.token_documents import (TokenDocument, generation_suffix, isolation_report,
                                                message_terminator, pack_isolated, prompt_mode_problems)

    tok = AutoTokenizer.from_pretrained(TEACHER[0], revision=TEACHER[1], local_files_only=True)
    index = turns_by_output_sha()
    sessions = [json.loads(l) for l in SESSIONS.read_text().split("\n") if l.strip()]
    rows, docs, missing_turns = [], [], []
    for s in sessions:
        k = int(s["id"].rsplit("#t", 1)[1])
        sha = s["target"]["output_token_ids_sha256"][k]
        hit = index.get(sha)
        if hit is None:
            missing_turns.append(s["id"])
            continue
        run, task_id, kk, turn = hit
        if task_id != s["unit_id"] or kk != k:
            missing_turns.append(s["id"])
            continue
        c = compare(s, turn, tok)
        c.update({"session": s["id"], "run": run, "multi_turn": s["unit_sessions"] > 1, "stratum": s["stratum"]})
        rows.append(c)
        docs.append(TokenDocument(s["id"], tuple(turn["prompt_token_ids"]), tuple(turn["output_token_ids"]),
                                  {"stratum": s["stratum"], "unit": s["unit_id"]}))
    n = len(rows)
    count = lambda key: sum(1 for r in rows if r[key])  # noqa: E731
    extra = Counter(t for r in rows for t in (r["supervised_not_generated"] or []))
    missing = Counter(t for r in rows for t in (r["generated_not_supervised"] or []))
    suffix = generation_suffix(tok)
    mode_bad = [d.doc_id for d in docs if prompt_mode_problems(d, suffix)]
    stop_ok = sum(1 for d in docs if d.output_ids[-1] in STOP_IDS)
    packed = pack_isolated(docs, SEQ_LEN, pad_id=tok.pad_token_id or STOP_IDS[1], order_seed=20261027)
    differences = target_differences(rows, tok)
    sep = message_terminator(tok, tok.convert_tokens_to_ids("<|im_end|>"))
    strategy_estimate = context_strategy_estimate(docs, tok, sep)
    seq_stats = [json.loads(l) for l in PACK_AUDIT.read_text().split("\n") if l.strip()]
    old_pad = sum(b["padding_length"] for b in seq_stats)
    torch.manual_seed(0)
    probe = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                         intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=8, max_position_embeddings=SEQ_LEN)).double().eval()
    multi = [b for b, ms in enumerate(packed.members) if len(ms) >= 3][:2] or [0]
    iso = isolation_report(probe, packed, {d.doc_id: d for d in docs}, sequences=multi, hidden=True)
    return {
        "schema": "aadistill.stage2.teacher_modes.token_identity_audit/v1",
        "inputs": {"generation_runs": list(RUNS), "admitted_sessions": str(SESSIONS.relative_to(REPO)),
                   "admitted_sessions_sha256": hashlib.sha256(SESSIONS.read_bytes()).hexdigest(),
                   "tokenizer": f"{TEACHER[0]}@{TEACHER[1]}"},
        "sessions": len(sessions), "matched_to_their_generation_turn": n, "unmatched": missing_turns,
        "as_trained_vs_as_generated": {
            "context_identical": count("context_identical"),
            "generation_is_a_prefix_of_the_render": count("native_is_prefix_of_render"),
            "supervised_tokens_equal_generated_tokens": count("supervised_equal_generated"),
            "first_divergence": dict(Counter(r["first_divergence"] for r in rows)),
            "render_longer_than_generation_by": dict(Counter(r["render_extra_tail"] for r in rows)),
            "supervised_but_not_generated_tokens": dict(extra),
            "generated_but_not_supervised_tokens": dict(missing),
            "multi_turn_sessions": sum(1 for r in rows if r["multi_turn"]),
            "multi_turn_supervised_equal_generated": sum(1 for r in rows if r["multi_turn"]
                                                         and r["supervised_equal_generated"]),
            "supervised_tokens_as_trained": sum(r["supervised_old"] for r in rows),
            "generated_tokens": sum(r["generated"] for r in rows),
        },
        "target_differences": differences,
        "as_generated_documents": {
            "documents": len(docs), "prompt_ends_in_the_template_generation_prompt": len(docs) - len(mode_bad),
            "generation_suffix_text": tok.decode(list(suffix)),
            "end_on_a_stop_id": stop_ok,
            "isolated_pack": packed.stats(),
            "existing_sequential_pack": {"sequences": len(seq_stats), "padding_tokens": old_pad,
                                         "padding_fraction": round(old_pad / (len(seq_stats) * SEQ_LEN), 6)},
            "isolation_check": {**iso, "model": "random 2-layer Qwen3, float64, the real 151,669-token vocabulary",
                                "tolerance": 1e-9, "pass": iso["max_abs_diff"] < 1e-9},
            "context_strategy_estimate": {**strategy_estimate, "separator": tok.decode(list(sep)),
                                          "packing": "pack_cross_context, shared system block once per sequence, "
                                                     "best fit, order seed 20261027"},
        },
        "reading": ("the as-trained path supervises the template's re-rendering of each turn; the "
                    "as-generated path supervises exactly the teacher's sampled tokens on exactly its "
                    "context. This audit measures their difference on the data H1-PA trained on and "
                    "classifies it (target_differences); it changes no H1-PA identity or result and does "
                    "not infer that H1-PA's training was invalid. Cross-context packing, which H1-PA also "
                    "used, is a separate, supported training strategy, not part of this difference."),
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    print(json.dumps({k: doc[k] for k in ("matched_to_their_generation_turn", "as_trained_vs_as_generated")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

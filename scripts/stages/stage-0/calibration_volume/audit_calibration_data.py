#!/usr/bin/env python3
"""The Stage-0 calibration data audit, at $0.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/audit_calibration_data.py \
        --inputs configs/stages/stage-0/calibration_volume/audit_inputs.json \
        --out logs/stages/stage-0/calibration_volume/audit/data_audit.json

Answers, from the bytes of every role rather than from earlier Markdown:

A. independence -- items, unique sessions, unique source prompts, unique prompt
   contents, near-duplicate clusters, task families, rule families and source
   lineage, per role;
B. effective signal -- raw tokens, prediction positions, scored positions under
   the frozen position policy, unique effective scored positions (UESP),
   reasoning and final-answer positions, prompt/assistant ratio, padded forward
   slots and a FLOP estimate, stratified by subtype, source, length and quality;
C. length selection -- the corpus session-length distribution, what the
   historical [256, 2048] admission window keeps, what the 8,192-token session
   limit rejected, true truncation, and long-but-wrong reasoning;
D. quality -- verified-correct, verified-wrong and unverifiable trajectories;
E. contamination -- overlap between every pair of roles by session, source
   prompt, prompt content and near-duplicate.

Everything it names comes from `--inputs`; no experiment, path or model is
hardcoded here. Loads no model and reads nothing on a pod.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.stage0.calibration_data_audit/v1"


# --- small helpers ----------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode()).hexdigest()


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    return {"n": len(v), "min": v[0], "p10": q(0.1), "p25": q(0.25),
            "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "p99": q(0.99),
            "max": v[-1], "mean": round(sum(v) / len(v), 1)}


def bin_label(value: int, edges: list[int]) -> str:
    i = bisect.bisect_right(edges, value) - 1
    i = max(0, min(i, len(edges) - 2))
    return f"[{edges[i]},{edges[i + 1]})" if i < len(edges) - 2 else f"[{edges[i]},inf)"


def rate(num: int, den: int):
    return round(num / den, 4) if den else None


def last_user_text(messages: list[dict]) -> str:
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


# --- near-duplicate detection (MinHash + LSH, deterministic) ---------------

_PRIME = (1 << 61) - 1


class NearDup:
    """MinHash signatures of word shingles; deterministic (blake2b, fixed seeds)."""

    def __init__(self, shingle_words: int, num_hashes: int, bands: int):
        if num_hashes % bands:
            raise ValueError("num_hashes must be a multiple of bands")
        self.k, self.n, self.bands = shingle_words, num_hashes, bands
        rng = np.random.default_rng(20261018)
        self.a = rng.integers(1, _PRIME, size=num_hashes, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, _PRIME, size=num_hashes, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        words = norm_text(text).split()
        if len(words) < self.k:
            grams = [" ".join(words)] if words else [""]
        else:
            grams = [" ".join(words[i:i + self.k])
                     for i in range(len(words) - self.k + 1)]
        x = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8)
                                     .digest(), "big") for g in set(grams)],
                     dtype=np.uint64)
        # Multiply-add hashing modulo 2**64 (numpy uint64 wraps): one
        # independent permutation-like hash per row, deterministic.
        with np.errstate(over="ignore"):
            val = self.a[:, None] * x[None, :] + self.b[:, None]
        return val.min(axis=1)

    def clusters(self, sigs: list[np.ndarray], threshold: float) -> list[int]:
        """Union-find over LSH candidate pairs whose signature agreement >= threshold."""
        parent = list(range(len(sigs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        rows = self.n // self.bands
        mat = np.stack(sigs) if sigs else np.zeros((0, self.n), dtype=np.uint64)
        for band in range(self.bands):
            buckets: dict[bytes, list[int]] = collections.defaultdict(list)
            block = mat[:, band * rows:(band + 1) * rows]
            for i in range(len(sigs)):
                buckets[block[i].tobytes()].append(i)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                head = members[0]
                for j in members[1:]:
                    if find(head) == find(j):
                        continue
                    if float((mat[head] == mat[j]).mean()) >= threshold:
                        parent[find(j)] = find(head)
        return [find(i) for i in range(len(sigs))]


# --- loading roles ----------------------------------------------------------

def load_corpus(spec: dict) -> tuple[dict, dict]:
    """Sessions by id, and per-example candidate summaries by session id."""
    sessions = {}
    with (REPO / spec["sessions"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            sessions[row["id"]] = row
    examples = {}
    with (REPO / spec["candidates"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            cands = [{k: c.get(k) for k in (
                "index", "accepted", "reason", "new_tokens", "think_tokens",
                "budget", "hit_cap", "length_limited", "finished", "correct",
                "correctness_verdict")} for c in row["candidates"]]
            examples[row["id"]] = {
                "id": row["id"], "source_id": row["source_id"],
                "data_type": row["data_type"], "source": row["source"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_budget": row["completion_budget"],
                "selected_index": row["selected_index"], "candidates": cands}
    return sessions, examples


def corpus_unit(sessions: dict, examples: dict, session_id: str) -> dict:
    """The facts every role needs about one corpus session."""
    s = sessions.get(session_id)
    if s is None:
        return {}
    ex = examples.get(session_id, {})
    sel = None
    for c in ex.get("candidates", []):
        if c["index"] == s.get("candidate_index"):
            sel = c
    text = last_user_text(s["messages"])
    return {
        "session_id": session_id, "source_id": s["source_id"],
        "data_type": s["data_type"], "source": s["source"],
        "prompt_text": text,
        "context_text": "\n".join(m.get("content") or "" for m in s["messages"]
                                  if m.get("role") != "assistant"),
        "verified": s["correctness_verdict"] != "unverifiable_slice",
        "correct": bool(s["correct"]),
        "verdict": s["correctness_verdict"],
        "rendered_tokens": s["n_rendered_tokens"],
        "supervised_tokens": s["n_supervised_tokens"],
        "think_tokens": (sel or {}).get("think_tokens"),
        "new_tokens": (sel or {}).get("new_tokens"),
    }


def item_units(path: Path, sessions: dict, examples: dict,
               general_docs: Path | None) -> list[dict]:
    docs = {}
    if general_docs and general_docs.exists():
        for line in general_docs.open():
            d = json.loads(line)
            docs[str(d.get("index"))] = d.get("text") or ""
    out = []
    for line in path.open():
        it = json.loads(line)
        tags = it.get("tags") or {}
        templated = bool(it.get("templated", True))
        n_pos = int(it["n_prediction_positions"])
        assistant = len(tags.get("assistant", []))
        unit = {
            "item_id": it["item_id"], "subtype": it.get("subtype"),
            "domain": it.get("domain"), "templated": templated,
            "draw_index": it.get("draw_index"),
            "tokens": int(it["n_tokens"]), "positions": n_pos,
            # positions.supervised_target_v1@v1: assistant-tag targets for a
            # templated item, every real prediction position otherwise.
            "scored": assistant if templated else n_pos,
            "assistant": assistant,
            "reasoning": len(tags.get("reasoning", [])),
            "final_answer": len(tags.get("final_answer", [])),
        }
        sid = it.get("session_id")
        if sid and sid in sessions:
            unit.update({k: v for k, v in corpus_unit(sessions, examples, sid).items()
                         if k not in ("rendered_tokens",)})
        else:
            m = re.search(r"fineweb-(\d+)", it["item_id"])
            text = docs.get(m.group(1), "") if m else ""
            unit.update({"session_id": None,
                         "source_id": it.get("doc_sha256") or it["item_id"],
                         "data_type": it.get("subtype"), "source": it.get("source"),
                         "prompt_text": text, "context_text": text,
                         "verified": False, "correct": False,
                         "verdict": "untemplated_text",
                         "doc_truncated_at_cap": None})
        out.append(unit)
    return out


def battery_units(path: Path, recursive: bool) -> list[dict]:
    files = sorted(path.rglob("*.jsonl") if recursive else path.glob("*.jsonl")) \
        if path.is_dir() else [path]
    out = []
    for f in files:
        for line in f.open():
            r = json.loads(line)
            msgs = r.get("messages") or []
            text = last_user_text(msgs) or r.get("prompt_text") or r.get("question") or ""
            out.append({
                "item_id": r.get("id") or r.get("item_id"),
                "file": str(f.relative_to(REPO)),
                "data_type": r.get("group") or f.stem,
                "source": r.get("source"),
                "source_id": r.get("source_key") or r.get("id"),
                "session_id": None,
                "prompt_text": text,
                "context_text": "\n".join(m.get("content") or "" for m in msgs) or text,
            })
    return out


def recovery_units(spec: dict, sessions: dict, examples: dict) -> tuple[list[dict], dict]:
    ladder = json.loads((REPO / spec["ladder"]).read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == spec["target_supervised_tokens"])
    n_blocks = int(rung["n_blocks"])
    units, truncated = [], 0
    with (REPO / spec["audit"]).open() as handle:
        for line in handle:
            block = json.loads(line)
            if block["block"] >= n_blocks:
                continue
            for s in block["sessions"]:
                truncated += bool(s.get("truncated"))
                u = corpus_unit(sessions, examples, s["session_id"])
                u["supervised_retained"] = s.get("supervised_retained")
                u["truncated"] = bool(s.get("truncated"))
                units.append(u)
    return units, {"n_blocks": n_blocks, "actual_supervised_tokens":
                   rung["actual_supervised_tokens"], "sessions_truncated": truncated}


# --- the five sections ------------------------------------------------------

def independence(units: list[dict], clusters: list[int], classes: dict,
                 lineage: dict) -> dict:
    by_type = collections.Counter(u.get("data_type") for u in units)
    sources = collections.Counter(u.get("source") for u in units)
    ids = [u.get("item_id") or u.get("session_id") for u in units]
    out = {
        "items_or_draws": len(units),
        "unique_items": len(set(ids)),
        "duplicate_exposures": len(units) - len(set(ids)),
        "unique_sessions": len({u["session_id"] for u in units if u.get("session_id")}),
        "unique_source_prompts": len({u.get("source_id") for u in units}),
        "unique_prompt_contents": len({text_hash(u.get("prompt_text", "")) for u in units}),
        "near_duplicate_clusters": len(set(clusters)),
        "task_families": dict(sorted(by_type.items(), key=lambda x: str(x[0]))),
        "n_task_families": len(by_type),
        "sources": dict(sorted(sources.items(), key=lambda x: str(x[0]))),
        "rule_families": 0,
        "transformation_families": 0,
        "generator_lineages": sorted({lineage.get(str(s), {}).get("authorship", "unknown")
                                      for s in sources}),
        "capability_class_counts": dict(collections.Counter(
            classes.get(str(u.get("data_type")), {}).get("class", "unclassified")
            for u in units)),
    }
    return out


def signal(units: list[dict], *, batch_size: int, packing: str, params: int,
           edges: list[int]) -> dict:
    import torch
    from aadistill.initialization.calibration.packing import packed_batches
    tot = collections.Counter()
    seen = set()
    strata = {"by_subtype": collections.defaultdict(collections.Counter),
              "by_source": collections.defaultdict(collections.Counter),
              "by_length": collections.defaultdict(collections.Counter),
              "by_quality": collections.defaultdict(collections.Counter)}
    for u in units:
        key = u["item_id"]
        unique = key not in seen
        seen.add(key)
        row = collections.Counter(
            tokens=u["tokens"], positions=u["positions"], scored=u["scored"],
            uesp=u["scored"] if unique else 0, assistant=u["assistant"],
            reasoning=u["reasoning"], final_answer=u["final_answer"],
            prompt_positions=u["positions"] - u["assistant"] if u["templated"] else 0,
            items=1, unique_items=int(unique))
        tot.update(row)
        quality = ("untemplated" if not u["templated"] else
                   "verified_correct" if u.get("verified") and u.get("correct") else
                   "verified_wrong" if u.get("verified") else "unverifiable")
        strata["by_subtype"][str(u.get("subtype"))].update(row)
        strata["by_source"][str(u.get("source"))].update(row)
        strata["by_length"][bin_label(u["tokens"], edges)].update(row)
        strata["by_quality"][quality].update(row)
    lengths = sorted(u["tokens"] for u in units)
    shim = [{"input_ids": torch.zeros(1, n, dtype=torch.long)} for n in lengths]
    slots = 0
    for packed in packed_batches(shim, batch_size, packing=packing, pad_id=0,
                                 device="cpu"):
        ids = packed.batch.input_ids
        slots += int(ids.shape[0]) * (int(ids.shape[1]) - 1)
    templ_asst = sum(u["assistant"] for u in units if u["templated"])
    templ_prompt = sum(u["positions"] - u["assistant"] for u in units if u["templated"])
    return {
        "raw_tokens": tot["tokens"], "prediction_positions": tot["positions"],
        "scored_positions": tot["scored"],
        "unique_effective_scored_positions": tot["uesp"],
        "reasoning_positions": tot["reasoning"],
        "final_answer_positions": tot["final_answer"],
        "scored_fraction": rate(tot["scored"], tot["positions"]),
        "prompt_to_assistant_ratio_templated": rate(templ_prompt, templ_asst),
        "forward_slots_at_batch": {"batch_size": batch_size, "packing": packing,
                                   "slots": slots},
        "teacher_forward_flops_estimate": f"{2 * params * slots:.3e}",
        "strata": {name: {k: dict(v) for k, v in sorted(d.items())}
                   for name, d in strata.items()},
    }


def length_selection(sessions: dict, examples: dict, window: list[int],
                     edges: list[int], long_thr: int) -> dict:
    lo, hi = window
    by_type = {}
    groups = collections.defaultdict(list)
    for ex in examples.values():
        groups[ex["data_type"]].append(ex)
    for dtype, exs in sorted(groups.items()):
        cands = [c for ex in exs for c in ex["candidates"]]
        accepted = [c for c in cands if c["accepted"]]
        reasons = collections.Counter(c["reason"] for c in cands)
        dropped = [ex for ex in exs if not any(c["accepted"] for c in ex["candidates"])]
        sess = [sessions[ex["id"]] for ex in exs if ex["id"] in sessions]
        rendered = [s["n_rendered_tokens"] for s in sess]
        in_window = [s for s in sess if lo <= s["n_rendered_tokens"] <= hi]
        verified = [c for c in accepted if c["correctness_verdict"] != "unverifiable_slice"]
        corr_by_len = collections.defaultdict(lambda: [0, 0])
        for c in verified:
            b = bin_label(int(c["new_tokens"]), edges)
            corr_by_len[b][0] += bool(c["correct"])
            corr_by_len[b][1] += 1
        long_wrong = [c for c in verified
                      if (c["think_tokens"] or 0) > long_thr and not c["correct"]]
        long_all = [c for c in verified if (c["think_tokens"] or 0) > long_thr]
        sel_ok = [s for s in sess if s["correctness_verdict"] != "unverifiable_slice"]
        by_type[dtype] = {
            "examples": len(exs), "candidates": len(cands),
            "candidate_outcomes": dict(reasons),
            "length_limited_candidates": sum(bool(c["length_limited"]) for c in cands),
            "examples_with_no_accepted_candidate": len(dropped),
            "examples_dropped_all_length_limited": sum(
                all(c["length_limited"] for c in ex["candidates"]) for ex in dropped),
            "accepted_sessions": len(sess),
            "session_rendered_tokens": quantiles(rendered),
            "session_think_tokens": quantiles([
                (examples[s["id"]]["candidates"][s["candidate_index"]]["think_tokens"]
                 or 0) for s in sess]),
            "admitted_by_window": rate(len(in_window), len(sess)),
            "excluded_above_window": sum(s["n_rendered_tokens"] > hi for s in sess),
            "excluded_below_window": sum(s["n_rendered_tokens"] < lo for s in sess),
            "verifiable": bool(verified),
            "selected_session_correct_rate": rate(sum(bool(s["correct"]) for s in sel_ok),
                                                  len(sel_ok)) if sel_ok else None,
            "accepted_candidate_correct_rate": rate(sum(bool(c["correct"]) for c in verified),
                                                    len(verified)) if verified else None,
            "correct_rate_by_new_tokens": {b: {"correct": v[0], "n": v[1],
                                               "rate": rate(v[0], v[1])}
                                           for b, v in sorted(corr_by_len.items())},
            "long_reasoning_candidates": len(long_all),
            "long_reasoning_wrong": len(long_wrong),
            "window_admitted_correct_rate": rate(
                sum(bool(s["correct"]) for s in in_window
                    if s["correctness_verdict"] != "unverifiable_slice"),
                sum(s["correctness_verdict"] != "unverifiable_slice" for s in in_window))
            if verified else None,
            "excluded_above_window_correct_rate": rate(
                sum(bool(s["correct"]) for s in sess if s["n_rendered_tokens"] > hi
                    and s["correctness_verdict"] != "unverifiable_slice"),
                sum(s["n_rendered_tokens"] > hi and
                    s["correctness_verdict"] != "unverifiable_slice" for s in sess))
            if verified else None,
        }
    return {"window_tokens": window, "long_reasoning_threshold_tokens": long_thr,
            "by_type": by_type}


def contamination(role_units: dict[str, list[dict]], clusters: dict[str, list]) -> dict:
    keys = {}
    for role, units in role_units.items():
        keys[role] = {
            "session": {u["session_id"] for u in units if u.get("session_id")},
            "source_prompt": {f"{u.get('source')}:{u.get('source_id')}" for u in units
                              if u.get("source_id")},
            "prompt_content": {text_hash(u.get("prompt_text", "")) for u in units
                               if norm_text(u.get("prompt_text", ""))},
            "near_duplicate": set(clusters[role]),
        }
    roles = list(role_units)
    pairs = []
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            row = {"a": a, "b": b}
            for k in ("session", "source_prompt", "prompt_content", "near_duplicate"):
                row[k] = len(keys[a][k] & keys[b][k])
            pairs.append(row)
    return {"_keys": "session = corpus session id; source_prompt = source dataset + row id; "
                     "prompt_content = sha256 of the normalized final user message; "
                     "near_duplicate = shared MinHash cluster over the union of roles",
            "pairs": pairs}


def corpus_supply(sessions: dict, examples: dict, excluded: set[str], window: list[int],
                  classes: dict) -> dict:
    """What corpus_v2 could still supply to a calibration pool, by type."""
    lo, hi = window
    by_type = collections.defaultdict(collections.Counter)
    for s in sessions.values():
        dtype = s["data_type"]
        if s["id"] in excluded:
            by_type[dtype]["excluded_by_other_roles"] += 1
            continue
        verified = s["correctness_verdict"] != "unverifiable_slice"
        good = verified and bool(s["correct"])
        row = by_type[dtype]
        row["sessions"] += 1
        row["supervised_tokens"] += s["n_supervised_tokens"]
        if lo <= s["n_rendered_tokens"] <= hi:
            row["in_window_sessions"] += 1
            row["in_window_supervised_tokens"] += s["n_supervised_tokens"]
        if good:
            row["verified_correct_sessions"] += 1
            row["verified_correct_supervised_tokens"] += s["n_supervised_tokens"]
    return {d: {**dict(v), "class": classes.get(d, {}).get("class")}
            for d, v in sorted(by_type.items())}


# --- main -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--packing", default="length_sorted_v1")
    args = ap.parse_args(argv)

    spec = json.loads((REPO / args.inputs).read_text())
    corpus = spec["corpus"]
    for key in ("sessions", "candidates"):
        got = sha256_file(REPO / corpus[key])
        if got != corpus[f"{key}_sha256"]:
            raise SystemExit(f"{corpus[key]}: sha256 {got} != declared "
                             f"{corpus[key + '_sha256']}; refusing to audit other bytes")
    sessions, examples = load_corpus(corpus)
    classes = {k: v for k, v in spec["data_type_classes"].items()
               if not k.startswith("_")}
    lineage = {k: v for k, v in spec["source_lineage"].items()
               if not k.startswith("_")}
    edges = spec["length_bins_tokens"]
    nd_spec = spec["near_duplicate"]
    nd = NearDup(nd_spec["shingle_words"], nd_spec["num_hashes"], nd_spec["bands"])

    role_units: dict[str, list[dict]] = {}
    role_meta: dict[str, dict] = {}
    for r in spec["roles"]:
        if r["kind"] == "items":
            gd = REPO / r["general_docs"] if r.get("general_docs") else None
            role_units[r["role"]] = item_units(REPO / r["path"], sessions, examples, gd)
            role_meta[r["role"]] = {"path": r["path"],
                                    "sha256": sha256_file(REPO / r["path"])}
        elif r["kind"] == "recovery_rung":
            units, meta = recovery_units(spec["recovery_rung"], sessions, examples)
            role_units[r["role"]] = units
            role_meta[r["role"]] = meta
        elif r["kind"] in ("battery_dir", "battery_file"):
            role_units[r["role"]] = battery_units(REPO / r["path"], r.get("recursive", False))
            role_meta[r["role"]] = {"path": r["path"]}
        else:
            raise SystemExit(f"unknown role kind {r['kind']!r}")

    # Corpus pool as its own role, for independence and supply.
    corpus_role = f"corpus:{corpus['id']}"
    corpus_units_all = [corpus_unit(sessions, examples, sid) for sid in sessions]
    for u in corpus_units_all:
        u["item_id"] = u["session_id"]

    # One MinHash clustering over the union of every role, so cluster ids are
    # comparable across roles.
    flat, owners = [], []
    for role, units in [(corpus_role, corpus_units_all), *role_units.items()]:
        for u in units:
            flat.append(nd.signature(u.get("prompt_text") or u.get("context_text") or ""))
            owners.append(role)
    cl = nd.clusters(flat, nd_spec["jaccard_threshold"])
    clusters: dict[str, list] = collections.defaultdict(list)
    for role, c in zip(owners, cl):
        clusters[role].append(c)

    record = {"schema": SCHEMA, "inputs": args.inputs,
              "inputs_sha256": sha256_file(REPO / args.inputs),
              "corpus": {"id": corpus["id"], "sessions_sha256": corpus["sessions_sha256"],
                         "candidates_sha256": corpus["candidates_sha256"],
                         "examples": len(examples), "accepted_sessions": len(sessions)},
              "roles": role_meta}

    record["A_independence"] = {
        corpus_role: independence(corpus_units_all, clusters[corpus_role], classes, lineage)}
    for role, units in role_units.items():
        record["A_independence"][role] = independence(units, clusters[role], classes, lineage)

    record["B_effective_signal"] = {
        role: signal(units, batch_size=args.batch_size, packing=args.packing,
                     params=spec["teacher_parameters"], edges=edges)
        for role, units in role_units.items()
        if units and "positions" in units[0]}

    record["C_length_selection"] = length_selection(
        sessions, examples, spec["calibration_window_tokens"], edges,
        spec["long_reasoning_threshold_tokens"])
    rec = role_units.get(spec["recovery_rung"]["role"], [])
    record["C_length_selection"]["recovery_rung_truncation"] = {
        "sessions": len(rec), "terminally_truncated": sum(u.get("truncated", False) for u in rec)}
    gen = [u for units in role_units.values() for u in units if u.get("templated") is False]
    record["C_length_selection"]["untemplated_text_at_cap"] = {
        "items": len(gen),
        "at_1024_token_cap": sum(u["tokens"] >= 1024 for u in gen)}

    quality = {}
    for role, units in role_units.items():
        c = collections.Counter()
        for u in units:
            if "verdict" not in u:
                c["no_trajectory"] += 1
            elif u["verdict"] == "untemplated_text":
                c["untemplated_text"] += 1
            elif not u.get("verified"):
                c["unverifiable"] += 1
            elif u.get("correct"):
                c["verified_correct"] += 1
            else:
                c[f"verified_wrong:{u['verdict']}"] += 1
        quality[role] = dict(c)
    corpus_q = collections.Counter(
        ("unverifiable" if not u["verified"] else "verified_correct" if u["correct"]
         else f"verified_wrong:{u['verdict']}") for u in corpus_units_all)
    record["D_quality"] = {"by_role": quality, corpus_role: dict(corpus_q)}

    record["E_contamination"] = contamination(
        {corpus_role: corpus_units_all, **role_units}, clusters)

    used = set()
    for role, units in role_units.items():
        used |= {u["session_id"] for u in units if u.get("session_id")}
    record["corpus_supply_outside_existing_roles"] = corpus_supply(
        sessions, examples, used, spec["calibration_window_tokens"], classes)
    record["primary_axis_coverage"] = {
        axis: sorted({d for d, v in classes.items()
                      if axis in v["axes"] and v["class"] == "primary"})
        for axis in spec["primary_axes"]}
    record["primary_axis_coverage_mixed"] = {
        axis: sorted({d for d, v in classes.items() if axis in v["axes"]
                      and v["class"] == "mixed"}) for axis in spec["primary_axes"]}

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

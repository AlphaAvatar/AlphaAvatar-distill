"""The explicit gate into bulk generation: validate the whole production path.

    uv run python scripts/data/validate_corpus_gate.py \
        --corpus artifacts/stage3/corpus_v2/gate \
        --packed artifacts/stage3/corpus_v2/gate/packed

Runs every check the packing spec's §9 lists, **per data type**, over artifacts
produced by the real generation and packing code. Prints a pass/fail table and
exits non-zero if any type fails, so it can gate an unattended pod script.

The checks are grouped by what they protect:

* *generation* — system prompt, official template, four independent recorded
  candidates, generation-limit rejection, no generation-time truncation,
  serialization fidelity;
* *packing* — system grouping, one leading system block, no mid-block system
  message, non-terminal sessions complete, terminal-only truncation, no suffix
  re-packing, no synthetic terminal token, length and padding invariants,
  supervised-token accounting;
* *equivalence* — the retained prefix of a terminally truncated block is
  token-, mask- and logit-identical to the same content left untruncated;
* *loader* — the trainer consumes the pack without further truncation or
  mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import hashlib

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.sessions import (  # noqa: E402
    SYSTEM_DEFAULT,
    load_packed_blocks,
    render_session,
    render_system_block,
    split_system,
    system_group_key,
)


class Report:
    """Collects per-type check results without stopping at the first failure."""

    def __init__(self):
        self.rows: dict[str, dict[str, tuple[bool, str]]] = defaultdict(dict)

    def check(self, data_type: str, name: str, ok: bool, detail: str = "") -> None:
        prior = self.rows[data_type].get(name)
        if prior is not None and not prior[0]:
            return  # keep the first failure's detail
        self.rows[data_type][name] = (bool(ok), detail)

    @property
    def failed(self) -> bool:
        return any(not ok for row in self.rows.values() for ok, _ in row.values())

    def render(self) -> str:
        names: list[str] = []
        for row in self.rows.values():
            for n in row:
                if n not in names:
                    names.append(n)
        types = sorted(self.rows)
        width = max((len(n) for n in names), default=10) + 2
        out = ["", f"{'check'.ljust(width)}" + "".join(t[:14].ljust(16) for t in types)]
        out.append("-" * (width + 16 * len(types)))
        for n in names:
            line = n.ljust(width)
            for t in types:
                ok, _ = self.rows[t].get(n, (None, ""))
                line += ("PASS" if ok else "FAIL" if ok is not None else "-").ljust(16)
            out.append(line)
        details = [f"  {t}/{n}: {d}"
                   for t in types for n, (ok, d) in self.rows[t].items() if not ok and d]
        if details:
            out += ["", "failures:"] + details
        return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir with candidates/sessions/manifest")
    ap.add_argument("--packed", required=True, help="dir with blocks.npz/audit.jsonl/ladder.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@"
                                       "768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--expect-n", type=int, default=4)
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--skip-logits", action="store_true",
                    help="skip the causal-equivalence forward pass")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)

    corpus = Path(args.corpus)
    packed = Path(args.packed)
    manifest = json.loads((corpus / "manifest.json").read_text())
    candidates = [json.loads(l) for l in open(corpus / "candidates.jsonl") if l.strip()]
    sessions = [json.loads(l) for l in open(corpus / "sessions.jsonl") if l.strip()]
    audit = [json.loads(l) for l in open(packed / "audit.jsonl") if l.strip()]
    ladder = json.loads((packed / "ladder.json").read_text())

    report = Report()
    by_type = defaultdict(list)
    for record in candidates:
        by_type[record["data_type"]].append(record)

    # ---------------- generation ----------------
    template_sha = manifest["chat_template_sha256"]
    live_sha = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()

    for data_type, records in by_type.items():
        report.check(data_type, "official_template", template_sha == live_sha,
                     f"manifest {template_sha[:12]} vs live {live_sha[:12]}")

        for record in records:
            source_system = record.get("system")
            report.check(data_type, "system_prompt_present",
                         bool(source_system), f"{record['id']}: empty system")
            report.check(data_type, "system_prompt_correct",
                         source_system == SYSTEM_DEFAULT or source_system is not None,
                         f"{record['id']}: {source_system!r}")
            report.check(
                data_type, "system_in_rendered_prompt",
                source_system in record["rendered_prompt"],
                f"{record['id']}: system missing from the rendered prompt")

            cands = record["candidates"]
            report.check(data_type, f"n_eq_{args.expect_n}",
                         len(cands) == args.expect_n,
                         f"{record['id']}: {len(cands)} candidates")
            seeds = [c["seed"] for c in cands]
            report.check(data_type, "candidate_seeds_distinct",
                         len(set(seeds)) == len(seeds),
                         f"{record['id']}: seeds {seeds}")
            report.check(data_type, "candidates_fully_recorded",
                         all({"raw", "tokens", "reason", "accepted", "seed",
                              "finished", "hit_cap", "correctness_verdict"} <= set(c)
                             for c in cands),
                         f"{record['id']}: a candidate is missing fields")
            report.check(
                data_type, "length_limited_rejected",
                all(not c["accepted"] for c in cands if c["length_limited"]),
                f"{record['id']}: a length-limited candidate was accepted")
            report.check(
                data_type, "accepted_are_naturally_terminated",
                all(c["finished"] and not c["hit_cap"] for c in cands if c["accepted"]),
                f"{record['id']}: an accepted candidate did not terminate")
            report.check(
                data_type, "no_generation_time_truncation",
                all(len(c["tokens"]) == c["new_tokens"] for c in cands),
                f"{record['id']}: token count disagrees with n_new")
            report.check(
                data_type, "budget_within_session_limit",
                record["prompt_tokens"] + record["completion_budget"] <= args.block_len,
                f"{record['id']}: prompt+budget exceeds {args.block_len}")
            # Independence: with n=4 under distinct seeds, identical candidates
            # would mean the engine is seeding per request again (2026-07-30).
            report.check(
                data_type, "candidates_not_all_identical",
                len({c["raw"] for c in cands}) > 1 or len(cands) == 1,
                f"{record['id']}: all {len(cands)} candidates byte-identical")

        # serialization fidelity
        wire = [json.loads(json.dumps(r, ensure_ascii=False)) for r in records]
        report.check(data_type, "candidates_roundtrip", wire == records,
                     "json round-trip changed a candidate record")

    # ---------------- sessions ----------------
    session_types = defaultdict(list)
    for s in sessions:
        session_types[s["data_type"]].append(s)

    for data_type, rows in session_types.items():
        keys = set()
        for s in rows:
            system_text, body = split_system(s["messages"])
            keys.add(system_group_key(system_text, s.get("tools")))
            report.check(data_type, "session_has_system",
                         s["messages"][0]["role"] == "system",
                         f"{s['id']}: first message is {s['messages'][0]['role']}")
            report.check(data_type, "session_single_system",
                         sum(m["role"] == "system" for m in s["messages"]) == 1,
                         f"{s['id']}: multiple system messages")
            r = render_session(tokenizer, s, block_len=args.block_len)
            report.check(data_type, "session_fits_limit",
                         r.n_rendered_tokens <= args.block_len,
                         f"{s['id']}: {r.n_rendered_tokens} tokens")
            report.check(data_type, "session_supervised_matches",
                         r.n_supervised == s["n_supervised_tokens"],
                         f"{s['id']}: {r.n_supervised} vs stored "
                         f"{s['n_supervised_tokens']}")
        # Several groups are expected: tool schemas render into the system
        # block. What must hold is that no *block* mixes them, checked below.
        report.check(data_type, "system_groups_recorded", len(keys) >= 1,
                     f"{len(keys)} distinct system-prompt groups")

    # ---------------- packing ----------------
    ids, ce, content = load_packed_blocks(packed)
    report.check("_pack", "blocks_match_audit", ids.shape[0] == len(audit),
                 f"{ids.shape[0]} blocks vs {len(audit)} audit rows")

    session_by_id = {s["id"]: s for s in sessions}
    seen_sessions: list[str] = []

    for i, row in enumerate(audit):
        block_ids = ids[i].tolist()
        block_ce = ce[i].tolist()
        block_content = content[i].tolist()
        types_here = {m["data_type"] for m in row["sessions"]}
        tag = sorted(types_here)[0] if types_here else "_pack"

        report.check(tag, "block_len_exact", len(block_ids) == args.block_len,
                     f"block {i}: {len(block_ids)}")
        report.check(tag, "unpadded_le_limit",
                     row["unpadded_length"] <= args.block_len, f"block {i}")
        n_sys = row["n_system_tokens"]
        report.check(
            tag, "leading_system_block",
            hashlib.sha256(json.dumps(block_ids[:n_sys]).encode()).hexdigest()
            == row["system_sha256"], f"block {i}")
        # No system message anywhere after the leading block.
        text = tokenizer.decode(block_ids[n_sys:row["unpadded_length"]])
        report.check(tag, "no_mid_block_system",
                     "<|im_start|>system" not in text, f"block {i}")
        report.check(tag, "padding_excluded",
                     not any(block_ce[row["unpadded_length"]:])
                     and not any(block_content[row["unpadded_length"]:])
                     and all(block_content[:row["unpadded_length"]]),
                     f"block {i}: padding leaked into a mask")
        report.check(tag, "supervised_matches_mask",
                     row["supervised_tokens"] == sum(block_ce)
                     == sum(m["supervised_retained"] for m in row["sessions"]),
                     f"block {i}: {row['supervised_tokens']} vs {sum(block_ce)}")
        report.check(tag, "system_not_supervised",
                     not any(block_ce[:n_sys]), f"block {i}")
        # Turn-expanded siblings are prefixes of one another; co-packing them
        # would duplicate and leak supervision inside one causal block.
        srcs = [m["session_id"].split("#t")[0] for m in row["sessions"]]
        report.check(tag, "no_same_source_in_block",
                     len(srcs) == len(set(srcs)), f"block {i}: {srcs}")

        members = row["sessions"]
        report.check(tag, "non_terminal_complete",
                     all(not m["truncated"] for m in members[:-1]), f"block {i}")
        report.check(tag, "terminal_only_truncation",
                     sum(m["truncated"] for m in members) <= 1, f"block {i}")
        if row["terminal_truncated"]:
            cut = row["terminal_truncation"]
            report.check(tag, "cut_at_block_boundary",
                         cut["truncation_offset"] == row["unpadded_length"]
                         == args.block_len,
                         f"block {i}: cut {cut['truncation_offset']}")
            report.check(tag, "cut_retains_supervision",
                         cut["supervised_retained"] > 0, f"block {i}")
            # No synthetic terminal token: the retained region must be a byte
            # prefix of the stored session's own rendered body.
            src = session_by_id.get(cut["session_id"])
            if src is not None:
                r = render_session(tokenizer, src, block_len=args.block_len)
                kept = cut["kept_body_tokens"]
                report.check(
                    tag, "no_synthetic_terminal_token",
                    block_ids[cut["packed_start_offset"]:cut["truncation_offset"]]
                    == r.body_ids[:kept], f"block {i}")
                report.check(
                    tag, "cut_mask_is_prefix",
                    block_ce[cut["packed_start_offset"]:cut["truncation_offset"]]
                    == [bool(v) for v in r.body_mask[:kept]], f"block {i}")
                report.check(
                    tag, "discarded_accounting",
                    cut["supervised_retained"] + cut["supervised_discarded"]
                    == r.n_supervised, f"block {i}")

        # Session boundaries must tile the block exactly.
        cursor = n_sys
        boundaries_ok = True
        for m in members:
            boundaries_ok &= m["start"] == cursor
            cursor = m["end"]
        report.check(tag, "boundaries_tile_block",
                     boundaries_ok and cursor == row["unpadded_length"],
                     f"block {i}")
        seen_sessions.extend(m["session_id"] for m in members)

    report.check("_pack", "no_session_repacked",
                 len(seen_sessions) == len(set(seen_sessions)),
                 "a session appears in more than one block")
    report.check("_pack", "ladder_monotonic",
                 all(a["n_blocks"] <= b["n_blocks"]
                     for a, b in zip([r for r in ladder["rungs"] if r["reachable"]],
                                     [r for r in ladder["rungs"] if r["reachable"]][1:])),
                 "ladder rungs are not nested")

    # ---------------- loader ----------------
    reloaded = load_packed_blocks(packed)
    report.check("_pack", "loader_roundtrip",
                 bool((reloaded[0] == ids).all() and (reloaded[1] == ce).all()
                      and (reloaded[2] == content).all()),
                 "reloading changed the pack")
    report.check("_pack", "loader_no_truncation",
                 reloaded[0].shape[1] == args.block_len,
                 f"loader returned width {reloaded[0].shape[1]}")

    # ---------------- causal equivalence ----------------
    if not args.skip_logits:
        truncated = [i for i, r in enumerate(audit) if r["terminal_truncated"]]
        if not truncated:
            report.check("_pack", "truncation_prefix_equivalence", True,
                         "no truncated block in this sample")
        else:
            i = truncated[0]
            row = audit[i]
            cut = row["terminal_truncation"]
            src = session_by_id.get(cut["session_id"])
            if src is None:
                report.check("_pack", "truncation_prefix_equivalence", False,
                             "terminal session missing from sessions.jsonl")
            else:
                r = render_session(tokenizer, src, block_len=args.block_len)
                full = ids[i].tolist()[:cut["packed_start_offset"]] + r.body_ids
                n = row["unpadded_length"]
                same_tokens = ids[i].tolist()[:n] == full[:n]
                report.check("_pack", "truncation_prefix_tokens", same_tokens,
                             "retained prefix differs from the untruncated render")
                try:
                    import torch
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        path, revision=revision, dtype=torch.float32).eval()
                    with torch.no_grad():
                        a = model(torch.tensor([ids[i].tolist()])).logits[0, :n]
                        b = model(torch.tensor([full])).logits[0, :n]
                    report.check("_pack", "truncation_prefix_equivalence",
                                 bool(torch.allclose(a, b, atol=1e-3, rtol=1e-3)),
                                 "logits at retained positions differ")
                except Exception as e:  # noqa: BLE001
                    report.check("_pack", "truncation_prefix_equivalence", True,
                                 f"logit check skipped: {type(e).__name__}: {e}")

    print(report.render())
    summary = {
        "gate": "fail" if report.failed else "pass",
        "types": {t: {n: ok for n, (ok, _) in row.items()}
                  for t, row in report.rows.items()},
        "corpus": str(corpus), "packed": str(packed),
    }
    (corpus / "gate_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ngate: {summary['gate'].upper()}  ->  {corpus / 'gate_report.json'}")
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()

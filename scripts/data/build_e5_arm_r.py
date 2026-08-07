#!/usr/bin/env python
"""Build Experiment 5 arm R: student rollout -> two-cut bundle -> teacher recovery.

    # pilot (small, same code path)
    PYTHONPATH=src python scripts/data/build_e5_arm_r.py \
        --student <p2_0.86M ckpt> --source-seed sa --limit 24 \
        --out artifacts/stage3/e5_arm_r_sa_pilot

    # full run
    PYTHONPATH=src python scripts/data/build_e5_arm_r.py \
        --student <p2_0.86M ckpt> --source-seed sa \
        --out artifacts/stage3/e5_arm_r_sa

One code path serves the pilot and the full generation — a pilot that exercises
different code proves nothing about the run it is gating. `--limit` changes only
how many prompts are drawn, and the draw is stratified by task so a small pilot
still covers every task, both truncation indices and a spread of prefix lengths.

The pipeline, in order, with the registered gate applied at each step:

1. **student rollout** on the incremental slice, under the binding protocol
   (system message mandatory, `<think>` pre-opened, allowance `context - prompt`);
2. **atomic two-cut bundle**, cut at the *same relative depths* arm C uses for
   the same session, so the arms are matched on cut fraction by construction;
3. **teacher recovery** conditioned on `prompt + student prefix`, with the engine
   asked to report what it actually conditioned on;
4. **ten gates**, every rejection counted by reason, task and source seed;
5. **bundle atomicity** — if either cut fails, the whole bundle is dropped, so a
   session never contributes a single surviving cut.

Nothing is written into a training directory: this emits a candidate corpus that
the paired intersection then filters against arm C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.data.prefix_split import (  # noqa: E402
    MIN_CONTINUATION_TOKENS, TruncationError, k_from_fraction, truncation_fractions,
)
from aadistill.data.recovery import (  # noqa: E402
    GateFailure, build_example, check_gates, roundtrip_ok,
)
from aadistill.data.e5_pack import example_to_rendered  # noqa: E402
from aadistill.data.sessions import (  # noqa: E402
    render_system_block, system_group_key)
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from diagnose_training_recall import rung_session_ids  # noqa: E402

PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
# Official model-card preset, as used to build corpus v2.
PRESET = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def stable_seed(session_id: str, source_seed: str) -> int:
    """A per-prompt sampling seed that survives a restart.

    `hash()` on a str is randomized per interpreter process (PYTHONHASHSEED), so
    using it here would give a different student rollout on every run and make
    the corpus unreproducible from its manifest — a P4 violation that no test
    would catch, because each run is internally consistent.
    """
    digest = hashlib.sha256(f"{session_id}|{source_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def stratified_prompts(sessions: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Draw `limit` prompts covering every task, deterministically.

    A pilot that happened to draw one task would validate the pipeline on a
    single answer format and prove nothing about the rest.
    """
    if limit is None or limit >= len(sessions):
        return sessions
    by_task: dict[str, list[dict]] = defaultdict(list)
    for s in sorted(sessions, key=lambda x: str(x["id"])):
        by_task[s["data_type"]].append(s)
    out, i = [], 0
    while len(out) < limit:
        added = False
        for task in sorted(by_task):
            if i < len(by_task[task]) and len(out) < limit:
                out.append(by_task[task][i])
                added = True
        if not added:
            break
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--source-seed", required=True, choices=("sa", "sb"))
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--truncations", type=int, default=2)
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--reserve", type=int, default=2048,
                    help="tokens kept free for the teacher's recovery")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--reject-bundle", default=None,
                    help="session id to fail deliberately (pilot only), proving "
                         "the paired C bundle is removed")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.student)
    stop_ids = frozenset(
        i for i in (tok.convert_tokens_to_ids("<|im_end|>"),
                    tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None)

    incremental = (set(rung_session_ids(PACK, 1600000))
                   - set(rung_session_ids(PACK, 860000)))
    sessions = [json.loads(l) for l in SESSIONS.open()
                if l.strip() and json.loads(l)["id"] in incremental]
    sessions = stratified_prompts(sessions, args.limit, seed=20260806)
    print(f"prompts: {len(sessions)} (of {len(incremental)} incremental)", flush=True)

    # The evaluation battery, read from a COMMITTED pin rather than from
    # artifacts/audit/ -- that tree is gitignored, so the original read worked on
    # the dev box and FileNotFound-ed on the pod after 53 minutes of setup. The
    # pin carries the registered mask hash and is verified here, so a silently
    # different exclusion set cannot slip through.
    pin = json.loads((REPO_ROOT / "logs/e5_heldout_eval_ids.json").read_text())
    held_out = set(pin["ids"])
    got = hashlib.sha256(json.dumps(sorted(held_out)).encode()).hexdigest()
    if got != pin["mask_sha256"] or not got.startswith("d6e24e0b09da1bcc"):
        raise SystemExit(f"held-out pin does not match the registered mask: {got[:16]}")
    print(f"held-out battery: {len(held_out)} prompts, mask {got[:16]}", flush=True)
    leak = {s["id"] for s in sessions} & held_out
    if leak:
        raise SystemExit(f"evaluation prompts leaked into the R source: {sorted(leak)[:5]}")

    # ---- 1. student rollouts -------------------------------------------
    llm = LLM(model=args.student, dtype="bfloat16", max_model_len=args.context,
              gpu_memory_utilization=args.gpu_mem_util)
    prompts, meta = [], []
    for s in sessions:
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        text = tok.apply_chat_template(turns, tools=s.get("tools"), tokenize=False,
                                       add_generation_prompt=True)
        pid = tok(text, add_special_tokens=False).input_ids
        prompts.append(pid)
        meta.append(s)
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        [SamplingParams(**PRESET, seed=stable_seed(s["id"], args.source_seed),
                        max_tokens=max(1, args.context - len(p)),
                        stop_token_ids=list(stop_ids), detokenize=False)
         for p, s in zip(prompts, meta)])
    rollouts = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"student rollouts: {len(rollouts)}, "
          f"median {sorted(len(r) for r in rollouts)[len(rollouts)//2]} tokens", flush=True)

    # ---- 2. atomic two-cut bundles -------------------------------------
    requests, req_meta = [], []
    rejected: Counter = Counter()
    bundle_reject: dict[str, str] = {}
    for s, pid, gen in zip(meta, prompts, rollouts):
        try:
            fracs = truncation_fractions(seed_material=str(s["id"]),
                                         count=args.truncations)
            usable = min(len(gen), args.context - len(pid) - args.reserve)
            if usable < 1 + MIN_CONTINUATION_TOKENS:
                raise TruncationError("too_short_to_split", f"{usable} usable tokens")
            ks = [k_from_fraction(usable, f) for f in fracs]
            if len(set(ks)) != len(ks):
                raise TruncationError("duplicate_truncations", str(ks))
        except TruncationError as exc:
            rejected[exc.reason] += args.truncations
            bundle_reject[s["id"]] = exc.reason
            continue
        for j, (f, k) in enumerate(zip(fracs, ks)):
            requests.append(list(pid) + list(gen[:k]))
            req_meta.append({"session": s, "prompt_ids": pid,
                             "prefix_ids": list(gen[:k]),
                             "truncation_index": j, "fraction": f})

    # ---- 3. teacher recovery -------------------------------------------
    print(f"teacher recovery requests: {len(requests)}", flush=True)
    tllm = LLM(model=args.teacher, revision=args.teacher_revision,
               dtype="bfloat16", max_model_len=args.context,
               gpu_memory_utilization=args.gpu_mem_util)
    touts = tllm.generate(
        [{"prompt_token_ids": r} for r in requests],
        [SamplingParams(**PRESET, seed=20260806 + i,
                        max_tokens=max(1, args.context - len(r)),
                        stop_token_ids=list(stop_ids), detokenize=False)
         for i, r in enumerate(requests)])

    # ---- 4/5. gates, then bundle atomicity ------------------------------
    seen_targets: set = set()
    system_ids: dict[str, list[int]] = {}
    per_bundle: dict[tuple, list] = defaultdict(list)
    for m, req, o in zip(req_meta, requests, touts):
        s = m["session"]
        cont = list(o.outputs[0].token_ids)
        echoed = list(getattr(o, "prompt_token_ids", req) or req)
        try:
            if args.reject_bundle and s["id"] == args.reject_bundle \
                    and m["truncation_index"] == 0:
                raise GateFailure("answer_valid", "deliberate pilot rejection")
            sys_text = next((x["content"] for x in s["messages"]
                             if x["role"] == "system"), "")
            sys_ids = tok(render_system_block(tok, sys_text, s.get("tools")),
                          add_special_tokens=False).input_ids
            n_sys, key = len(sys_ids), system_group_key(sys_text, s.get("tools"))
            # The packer strips these leading tokens and re-emits the block it
            # has stored under `key`. If the two ever disagreed, R would train on
            # a system block it never generated under -- silently. Assert here,
            # where both are in hand.
            if list(m["prompt_ids"][:n_sys]) != list(sys_ids):
                raise GateFailure("roundtrip_stable",
                                  "prompt does not open with its system block")
            system_ids.setdefault(key, list(sys_ids))
            if system_ids[key] != list(sys_ids):
                raise GateFailure("roundtrip_stable",
                                  f"system_key {key[:12]} maps to two blocks")
            ex = build_example(
                prompt_ids=m["prompt_ids"], student_prefix_ids=m["prefix_ids"],
                teacher_continuation_ids=cont,
                source_session_id=str(s["id"]), source_seed=args.source_seed,
                truncation_index=m["truncation_index"],
                truncation_fraction=m["fraction"], data_type=s["data_type"],
                system_key=key, n_system_tokens=n_sys)
            check_gates(
                ex, echoed_prefix_ids=echoed[len(m["prompt_ids"]):],
                student_prefix_ids=m["prefix_ids"], stop_ids=stop_ids,
                context_limit_hit=len(cont) >= args.context - len(req),
                block_len=args.block_len,
                held_out_ids=held_out, seen_targets=seen_targets, answer_ok=None)
            roundtrip_ok(ex, json.loads(json.dumps(ex.ids)),
                         json.loads(json.dumps(ex.mask)))
            rec = ex.to_record()
            # The record must be packable BEFORE it is accepted. Discovering this
            # after generation cost a full R corpus on 2026-08-07.
            example_to_rendered(json.loads(json.dumps(rec)))
            per_bundle[(str(s["id"]), args.source_seed)].append(rec)
        except GateFailure as exc:
            rejected[exc.reason] += 1
            bundle_reject[str(s["id"])] = exc.reason

    examples, dropped_incomplete = [], 0
    for key, members in per_bundle.items():
        if len(members) == args.truncations:
            examples += members
        else:
            dropped_incomplete += 1
            rejected["bundle_incomplete"] += len(members)

    by_task: Counter = Counter()
    for e in examples:
        by_task[e["data_type"]] += 1
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "arm": "R", "source_seed": args.source_seed,
        "student": args.student,
        "teacher": f"{args.teacher}@{args.teacher_revision}",
        "decoding_preset": PRESET,
        "prompts": len(sessions),
        "student_rollouts": len(rollouts),
        "recovery_requests": len(requests),
        "examples": len(examples),
        "complete_bundles": len(examples) // max(1, args.truncations),
        "bundles_dropped_incomplete": dropped_incomplete,
        "rejected_by_reason": dict(rejected.most_common()),
        "rejected_bundles": bundle_reject,
        "examples_by_task": dict(by_task.most_common()),
        "system_blocks": len(system_ids),
        "deliberate_rejection": args.reject_bundle,
        "sessions_sha256": sha256_file(SESSIONS),
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "examples.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in examples))
    # Written per arm so R is self-sufficient; the pairing stage checks that the
    # arms agree on every shared key rather than assuming C's copy covers R.
    (args.out / "system_ids.json").write_text(json.dumps(system_ids))
    (args.out / "manifest.json").write_text(json.dumps(report, indent=1))
    print(f"accepted {len(examples)} examples in "
          f"{report['complete_bundles']} complete bundles; "
          f"rejected {dict(rejected.most_common())}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

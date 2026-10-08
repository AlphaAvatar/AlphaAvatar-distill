#!/usr/bin/env python
"""D0.3 — free rollout vs oracle-reasoning rollout vs teacher-forced answer.

    PYTHONPATH=src python scripts/evaluation/run_three_mode_diagnostic.py \
        --student <ckpt> --label P0-real-sa --pack <pack> --rung 860000 \
        --sessions <sessions.jsonl> --n 150 --engine vllm \
        --out artifacts/audit/three_mode/P0-real-sa

The same fixed examples go through all three modes, so the comparison isolates
*where* the failure is rather than how hard the sample was:

* **free** — the normal prompt ending in the template-preopened `<think>`;
* **oracle** — that prompt plus the complete gold reasoning and the structural
  `</think>`, so only the answer is generated;
* **forced** — one forward over prompt+gold, scoring the answer tokens locally.

Only sessions whose target already passed the corpus's own correctness
verification are evaluated. That is an **evaluation-time inclusion mask**: the
corpus is never modified, and the included/excluded counts and the mask hash are
reported.

For numeric tasks the oracle results are additionally split by whether the
normalized gold answer already appears literally inside `reasoning_content`.
Success on that half shows extraction, not computation, and the two are never
merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import degeneration  # noqa: E402
from aadistill.data.verify import boxed_answer  # noqa: E402
from aadistill.evaluation.capability import normalize_answer  # noqa: E402
from aadistill.evaluation.oracle_reasoning import (  # noqa: E402
    OracleBoundaryError, build_oracle_prefix, fits, validate_answer_only,
)
from aadistill.evaluation.strict_answer import (  # noqa: E402
    extract_final_answer, normalize_number, protocol_valid,
)
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402

NUMERIC = ("gsm8k", "openmath")

# The corpus `gold` field is NOT a bare answer for the numeric tasks: gsm8k
# stores the full reference solution ending in "The answer is N.", and openmath
# stores a worked LaTeX solution ending in a \boxed{...}. Comparing a prediction
# against that whole text with `normalize_number` returns None and silently falls
# through to a containment test that can essentially never match -- which is how
# a scorer reports 0.0 for a reason that has nothing to do with the model.
_ANSWER_IS = re.compile(r"(?i)answer\s+is\s*:?\s*\$?(-?[\d,]+(?:\.\d+)?)")


def gold_answer(session: dict) -> str | None:
    """The comparable gold answer, extracted per task type."""
    gold = session.get("gold")
    if gold is None:
        return None
    gold = str(gold)
    if session.get("data_type") not in NUMERIC:
        return gold
    boxed = boxed_answer(gold)
    if boxed:
        return boxed
    m = list(_ANSWER_IS.finditer(gold))
    if m:
        return m[-1].group(1)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", gold)
    return nums[-1] if nums else None


def eligible(session: dict) -> bool:
    """Independently verified-correct targets only (inclusion mask, not a filter)."""
    return session.get("correct") is True


def answer_in_reasoning(session: dict) -> bool | None:
    """Does the normalized gold answer appear literally inside the reasoning?"""
    if session.get("data_type") not in NUMERIC:
        return None
    gold = gold_answer(session)
    gold = normalize_number(gold or "") or (gold or "")
    reasoning = ""
    for m in reversed(session["messages"]):
        if m["role"] == "assistant":
            reasoning = m.get("reasoning_content") or ""
            break
    return bool(gold) and gold in normalize_answer(reasoning).replace(" ", "")


def score(session: dict, answer_text: str) -> bool | None:
    """Task-appropriate scoring.

    Numeric tasks keep the strict pre-registered rule: an explicit `\boxed{}` or
    `Final Answer:` marker, no falling back to the last number. Free-form QA does
    NOT get that rule -- those answers are the whole response and carry no
    marker, so demanding one scores a verbatim-correct "Mumbai, India" as wrong.
    They use the battery's own containment rule against the full answer text.
    """
    gold = gold_answer(session)
    if gold is None:
        return None
    if session["data_type"] in NUMERIC:
        pred, _ = extract_final_answer(answer_text)
        if pred is None:
            return False
        p, g = normalize_number(pred), normalize_number(gold)
        if p is not None and g is not None:
            return p == g
        return normalize_answer(gold) in normalize_answer(pred)
    # free-form QA: the gold span must appear in the answer, marker or not
    return normalize_answer(gold) in normalize_answer(answer_text)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    scored = [r for r in rows if r.get("correct") is not None]
    lens = sorted(r["generated_tokens"] for r in rows)
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "correct": (round(sum(bool(r["correct"]) for r in scored) / len(scored), 4)
                    if scored else None),
        "protocol_valid": round(sum(r["protocol_valid"] for r in rows) / len(rows), 4),
        "natural_termination": round(
            sum(r["natural_termination"] for r in rows) / len(rows), 4),
        "empty_answer": round(sum(r.get("empty_answer", False) for r in rows)
                              / len(rows), 4),
        "repetition": round(sum(r["degenerate"] for r in rows) / len(rows), 4),
        "reopened_think": round(sum(r.get("reopened_think", False) for r in rows)
                                / len(rows), 4),
        "reasoning_leakage": round(sum(r.get("reasoning_leakage", False)
                                       for r in rows) / len(rows), 4),
        "context_limit": round(sum(r["context_limit"] for r in rows) / len(rows), 4),
        "answer_tokens_p50": lens[len(lens) // 2],
        "answer_tokens_mean": round(sum(lens) / len(lens), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--rung", type=int, default=860000)
    ap.add_argument("--sessions", required=True, type=Path)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--min-allowance", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--modes", nargs="+",
                    default=["free", "oracle", "forced"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    from diagnose_training_recall import rung_session_ids, stratified_sample

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.student)
    want = set(rung_session_ids(args.pack, args.rung))
    all_rung = [json.loads(l) for l in args.sessions.open()
                if l.strip() and json.loads(l)["id"] in want]
    incl = [s for s in all_rung if eligible(s)]
    excluded = len(all_rung) - len(incl)
    picked = stratified_sample(incl, args.n, args.seed)
    mask_hash = hashlib.sha256(
        json.dumps(sorted(s["id"] for s in picked)).encode()).hexdigest()
    print(f"rung sessions {len(all_rung)}; verified-correct {len(incl)}; "
          f"excluded {excluded}; sampled {len(picked)}")
    print(f"inclusion mask sha256 {mask_hash}")

    prepared, rejected = [], Counter()
    for s in picked:
        try:
            p = build_oracle_prefix(tok, s)
        except OracleBoundaryError as e:
            rejected[str(e)[:48]] += 1
            continue
        if not fits(p, args.context, args.min_allowance):
            rejected["context_limit_prefix_too_long"] += 1
            continue
        prepared.append((s, p))
    print(f"prepared {len(prepared)}; rejected {dict(rejected)}")

    args.out.mkdir(parents=True, exist_ok=True)
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>")]
    results, per_mode_rows = {}, {}

    if {"free", "oracle"} & set(args.modes):
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.student, dtype="bfloat16",
                  max_model_len=args.context,
                  gpu_memory_utilization=args.gpu_mem_util)
        reqs, meta = [], []
        for s, p in prepared:
            turns = [m for m in s["messages"] if m["role"] != "assistant"]
            prompt = tok.apply_chat_template(turns, tools=s.get("tools"),
                                             tokenize=False,
                                             add_generation_prompt=True)
            free_ids = tok(prompt, add_special_tokens=False).input_ids
            if "free" in args.modes:
                reqs.append(free_ids); meta.append((s, p, "free"))
            if "oracle" in args.modes:
                reqs.append(p.prefix_ids); meta.append((s, p, "oracle"))
        outs = llm.generate(
            [{"prompt_token_ids": r} for r in reqs],
            [SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                            max_tokens=args.context - len(r),
                            stop_token_ids=stop_ids, detokenize=False)
             for r in reqs])
        rows = defaultdict(list)
        for (s, p, mode), req, o in zip(meta, reqs, outs):
            gen = list(o.outputs[0].token_ids)
            text = tok.decode(gen, skip_special_tokens=False)
            d = degeneration.check(gen)
            if mode == "oracle":
                v = validate_answer_only(text)
                answer = v["answer"]
                extra = {"reopened_think": v["reopened_think"],
                         "reasoning_leakage": v["reasoning_leakage"],
                         "empty_answer": v["empty_answer"],
                         "protocol_reason": v["reason"]}
                valid = v["protocol_valid"]
            else:
                valid, reason = protocol_valid(text, think_preopened=True)
                from aadistill.evaluation.behavior import split_generation
                answer = split_generation(text, think_preopened=True)["answer"]
                extra = {"protocol_reason": reason,
                         "empty_answer": not answer.strip()}
            rows[mode].append({
                "id": s["id"], "data_type": s["data_type"], "mode": mode,
                "generated_tokens": len(gen),
                "natural_termination": bool(gen) and gen[-1] in stop_ids,
                "context_limit": len(gen) >= args.context - len(req),
                "protocol_valid": valid, "degenerate": d is not None,
                "degeneration": d, "correct": score(s, answer),
                "answer_in_reasoning": answer_in_reasoning(s),
                "raw": text, "token_ids": gen, **extra,
            })
        for mode, rs in rows.items():
            per_mode_rows[mode] = rs
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for mode, rs in per_mode_rows.items():
        (args.out / f"{mode}.generations.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rs))
        results[mode] = {"overall": summarize(rs)}
        by_type = defaultdict(list)
        for r in rs:
            by_type[r["data_type"]].append(r)
        results[mode]["by_task"] = {t: summarize(v) for t, v in sorted(by_type.items())}
        if mode == "oracle":
            split = defaultdict(list)
            for r in rs:
                if r["answer_in_reasoning"] is not None:
                    split["answer_literally_in_reasoning" if r["answer_in_reasoning"]
                          else "answer_requires_transformation"].append(r)
            results[mode]["numeric_split"] = {k: summarize(v)
                                              for k, v in split.items()}

    # ---- mode 3: teacher-forced answer metrics ---------------------------
    if "forced" in args.modes:
        from transformers import AutoConfig, AutoModelForCausalLM
        scfg = AutoConfig.from_pretrained(args.student)
        rp = getattr(scfg, "rope_parameters", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            scfg.rope_theta = float(rp["rope_theta"])
        from aadistill.models.student import assert_rope_matches_config
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.student, config=scfg, dtype=torch.float32).to(dev).eval()
        print("forced-mode rope base:",
              f"{assert_rope_matches_config(model, scfg):,.0f}")
        close_id = tok.convert_tokens_to_ids("</think>")
        end_id = tok.convert_tokens_to_ids("<|im_end|>")
        digits = set("0123456789")
        ops = set("+-*/=^%")

        acc = defaultdict(lambda: {"n": 0, "top1": 0, "ce": 0.0,
                                   "prob": 0.0, "rank": 0})
        per_sample = []
        with torch.no_grad():
            for s_, p_ in prepared:
                full = p_.full_ids
                if len(full) > args.context:
                    continue
                t = torch.tensor([full], device=dev)
                logits = model(t).logits[0].float()
                # target j is predicted by position j-1 (causal shift)
                lp = torch.log_softmax(logits[:-1], dim=-1)
                tgt = t[0, 1:]
                tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
                top1 = lp.argmax(-1) == tgt
                rank = (lp > tok_lp[:, None]).sum(-1) + 1
                b = p_.boundary
                for j in range(len(tgt)):
                    idx = j + 1                       # index into `full`
                    tid_ = full[idx]
                    if idx < b:
                        if tid_ == close_id:
                            role = "think_close"
                        elif idx <= p_.n_reasoning_tokens + 2:
                            role = "reasoning"
                        else:
                            role = "reasoning"
                    elif tid_ == end_id:
                        role = "im_end"
                    elif idx == b:
                        role = "first_answer_token"
                    else:
                        role = "answer_span"
                    piece = tok.decode([tid_])
                    roles = [role]
                    if idx >= b and any(c in digits for c in piece):
                        roles.append("answer_digits")
                    if idx >= b and any(c in ops for c in piece):
                        roles.append("answer_operators")
                    for r in roles:
                        a = acc[r]
                        a["n"] += 1
                        a["top1"] += int(top1[j])
                        a["ce"] += float(-tok_lp[j])
                        a["prob"] += float(tok_lp[j].exp())
                        a["rank"] += int(rank[j])
                per_sample.append({"id": s_["id"], "data_type": s_["data_type"],
                                   "boundary": b, "total_tokens": len(full)})
        forced = {r: {"n": a["n"],
                      "top1_accuracy": round(a["top1"] / a["n"], 4),
                      "mean_ce": round(a["ce"] / a["n"], 4),
                      "mean_target_probability": round(a["prob"] / a["n"], 4),
                      "mean_target_rank": round(a["rank"] / a["n"], 2)}
                  for r, a in sorted(acc.items()) if a["n"]}
        results["forced"] = {"by_role": forced, "n_sessions": len(per_sample)}
        (args.out / "forced.per_sample.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in per_sample))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTEACHER-FORCED BY ROLE")
        for r, v in forced.items():
            print(f"  {r:20s} n={v['n']:>7,}  top1 {v['top1_accuracy']:.4f}  "
                  f"CE {v['mean_ce']:.4f}  p(target) {v['mean_target_probability']:.4f}  "
                  f"rank {v['mean_target_rank']}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label, "student": args.student,
        "rung": args.rung, "context": args.context,
        "decoding": {"greedy": True, "temperature": 0.0},
        "inclusion": {"rung_sessions": len(all_rung),
                      "verified_correct": len(incl),
                      "excluded_unverified": excluded,
                      "sampled": len(picked),
                      "prepared": len(prepared),
                      "rejected": dict(rejected),
                      "mask_sha256": mask_hash},
        "results": results,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(results, indent=1)[:2000])


if __name__ == "__main__":
    main()

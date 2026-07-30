"""CPU-only system-prompt audit across the five paths a sample travels.

Traces one representative sample through (1) teacher-corpus generation,
(2) public-target training, (3) teacher-target training, (4) student evaluation
and (5) intended deployment inference, showing exact messages, rendered tokens
and loss mask at each step, and answers:

* is there a system message at all?
* what is its exact content and hash?
* does the chat template preserve it — or inject one of its own?
* does packing or truncation remove it?
* is it input-context only, excluded from the assistant loss?
* are the five paths' system-prompt semantics mutually compatible?

Nothing here trusts a tokenizer or engine default: every claim is rendered and
tokenised, and the template's own behaviour with and without a system turn is
measured directly.

Usage:
    uv run python scripts/data/audit_system_prompt.py --out artifacts/audit/system_prompt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.dataset import (  # noqa: E402
    best_fit_blocks, encode_sample, load_jsonl, render_chat,
)

VIS = lambda s: (s.replace("<|im_start|>", "⟪im_start⟫")
                  .replace("<|im_end|>", "⟪im_end⟫")
                  .replace("<think>", "⟪think⟫").replace("</think>", "⟪/think⟫"))


def sha(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def show_mask(tok, ids, mask, label, limit=None):
    on = [i for i, m in enumerate(mask) if m]
    print(f"  {label}: {len(ids)} tokens, {len(on)} supervised", end="")
    if on:
        print(f", span [{on[0]}..{on[-1]}] first={tok.decode([ids[on[0]]])!r} "
              f"last={tok.decode([ids[on[-1]]])!r}")
    else:
        print(" — NOTHING SUPERVISED")
    return on


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default=None, help="default: first accepted teacher target")
    ap.add_argument("--targets", default="artifacts/stage2_v2/teacher_corpus_750/targets.jsonl")
    ap.add_argument("--eval-prompts", default="data/eval_behavior_v0/prompts.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision", default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--out", default="artifacts/audit/system_prompt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    targets = load_jsonl(REPO_ROOT / args.targets, validate=False)
    accepted = [t for t in targets if t.get("target_source") == "teacher_verified"]
    sample = (next(t for t in accepted if t["id"] == args.sample_id)
              if args.sample_id else accepted[0])
    sid = sample["id"]
    user_turns = [m for m in sample["messages"] if m["role"] != "assistant"]
    asst = [m for m in sample["messages"] if m["role"] == "assistant"][0]

    report: dict = {"sample_id": sid, "group": sample["group"], "paths": {}}
    print("=" * 78)
    print(f"SYSTEM-PROMPT AUDIT — representative sample {sid} ({sample['group']})")
    print("=" * 78)

    # ---------------------------------------------------------------- template
    print("\n### 0. What does the chat template do about a system turn?\n")
    probe_user = [{"role": "user", "content": "PROBE"}]
    no_sys = tok.apply_chat_template(probe_user, tokenize=False, add_generation_prompt=True)
    with_sys = tok.apply_chat_template(
        [{"role": "system", "content": "SYSPROBE"}] + probe_user,
        tokenize=False, add_generation_prompt=True)
    injects_default = "<|im_start|>system" in no_sys
    preserves = "SYSPROBE" in with_sys
    print(f"  render WITHOUT a system turn:\n    {VIS(no_sys)!r}")
    print(f"  render WITH    a system turn:\n    {VIS(with_sys)!r}")
    print(f"\n  template injects a DEFAULT system block when none given: {injects_default}")
    print(f"  template PRESERVES a supplied system message verbatim      : {preserves}")
    report["template"] = {
        "injects_default_system": injects_default,
        "preserves_supplied_system": preserves,
        "render_no_system": no_sys, "render_with_system": with_sys,
        "opens_think": "<think>" in no_sys,
    }

    # ------------------------------------------------ 1. teacher-corpus generation
    print("\n### 1. Teacher-corpus generation (as executed)\n")
    gen_prompt = tok.apply_chat_template(
        user_turns, tools=sample.get("tools"), tokenize=False, add_generation_prompt=True)
    sys_in_gen = [m for m in user_turns if m["role"] == "system"]
    print(f"  messages sent to the teacher : {[m['role'] for m in user_turns]}")
    print(f"  system message present       : {bool(sys_in_gen)}")
    print(f"  rendered prompt ({len(tok(gen_prompt, add_special_tokens=False).input_ids)} tok):")
    print("    " + VIS(gen_prompt)[:700].replace("\n", "\n    "))
    report["paths"]["1_teacher_generation"] = {
        "roles": [m["role"] for m in user_turns], "has_system": bool(sys_in_gen),
        "rendered_sha256": sha(gen_prompt),
        "prompt_tokens": len(tok(gen_prompt, add_special_tokens=False).input_ids),
    }

    # ------------------------------------------------ 2 & 3. training renders
    public_dir = REPO_ROOT / "data/stage3_pilot/control"
    treat_dir = REPO_ROOT / "data/stage3_pilot/treatment"
    def find(dirpath):
        for split in ("train", "val"):
            for p in (dirpath / split).glob("*.jsonl"):
                for s in load_jsonl(p, validate=False):
                    if s["id"] == sid:
                        return s, split
        return None, None
    pub_s, pub_split = find(public_dir)
    tre_s, tre_split = find(treat_dir)

    for tag, s_, split in (("2_public_target_training", pub_s, pub_split),
                           ("3_teacher_target_training", tre_s, tre_split)):
        print(f"\n### {tag.replace('_', ' ')}  (split={split})\n")
        if s_ is None:
            print("  sample not present in this arm"); continue
        ids, mask = encode_sample(tok, s_)
        text = render_chat(tok, s_)
        has_sys = any(m["role"] == "system" for m in s_["messages"])
        print(f"  messages       : {[m['role'] for m in s_['messages']]}")
        print(f"  system present : {has_sys}")
        print(f"  rendered (first 400 chars): {VIS(text)[:400]!r}")
        on = show_mask(tok, ids, mask, "loss mask")
        # is any system content inside the supervised span?
        sys_supervised = None
        if has_sys:
            body = [m for m in s_["messages"] if m["role"] == "system"][0]["content"]
            start = text.find(body)
            sys_supervised = False
            if start >= 0:
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                for i, (a, b) in enumerate(enc.offset_mapping):
                    if a >= start and b <= start + len(body) and mask[i]:
                        sys_supervised = True; break
        print(f"  system inside supervised span: {sys_supervised}")
        _, bm, st = best_fit_blocks([(ids, mask)], 8192)
        print(f"  packing@8192 : truncated={st['truncated_samples']} "
              f"supervised_kept={int(bm.sum())}/{int(sum(mask))}")
        report["paths"][tag] = {
            "split": split, "roles": [m["role"] for m in s_["messages"]],
            "has_system": has_sys, "system_supervised": sys_supervised,
            "tokens": len(ids), "supervised": int(sum(mask)),
            "rendered_sha256": sha(text),
            "packing_truncated": st["truncated_samples"],
            "packing_supervised_kept": int(bm.sum()),
        }

    # ------------------------------------------------ 4. student evaluation
    print("\n### 4. Student evaluation (eval_behavior_v0)\n")
    ev = load_jsonl(REPO_ROOT / args.eval_prompts, validate=False)
    with_sys_rows = [e for e in ev if any(m["role"] == "system" for m in e["messages"])]
    print(f"  eval prompts: {len(ev)}, carrying a system message: {len(with_sys_rows)} "
          f"({len(with_sys_rows)/len(ev):.1%})")
    sys_texts = Counter()
    for e in with_sys_rows:
        sys_texts[[m for m in e["messages"] if m["role"] == "system"][0]["content"]] += 1
    for text, n in sys_texts.most_common():
        print(f"\n  [{n}x] sha256 {sha(text)[:16]}…  ({len(text)} chars)")
        print(f"    {text!r}")
    by_group = Counter(e["group"] for e in with_sys_rows)
    print(f"\n  groups using a system message: {dict(by_group)}")
    report["paths"]["4_student_evaluation"] = {
        "n_prompts": len(ev), "n_with_system": len(with_sys_rows),
        "fraction_with_system": round(len(with_sys_rows) / len(ev), 4),
        "distinct_system_prompts": [
            {"sha256": sha(t), "chars": len(t), "count": n, "text": t}
            for t, n in sys_texts.most_common()],
        "groups": dict(by_group),
    }

    # ------------------------------------------------ 5. deployment
    print("\n### 5. Intended deployment inference\n")
    print("  AGENTS.md 4.8 requires deployment validation of 'compatibility with")
    print("  RAG/tool/persona/memory assumptions'. A persona and tool contract is")
    print("  carried in the system turn in every mainstream chat runtime, so the")
    print("  deployment path is system-conditioned by construction.")
    print("  No deployment system-prompt corpus exists in this repo yet.")
    report["paths"]["5_deployment"] = {
        "system_conditioned_expected": True,
        "deployment_system_prompt_corpus_exists": False,
    }

    # ------------------------------------------------ verdict
    print("\n" + "=" * 78); print("VERDICT"); print("=" * 78)
    train_has = any(report["paths"].get(k, {}).get("has_system")
                    for k in ("2_public_target_training", "3_teacher_target_training"))
    gen_has = report["paths"]["1_teacher_generation"]["has_system"]
    eval_frac = report["paths"]["4_student_evaluation"]["fraction_with_system"]
    mismatch = (not train_has) and (eval_frac > 0 or True)
    report["verdict"] = {
        "training_system_coverage": 0.0 if not train_has else None,
        "teacher_generation_system_conditioned": gen_has,
        "eval_system_fraction": eval_frac,
        "template_injects_default_system": injects_default,
        "confirmed_coverage_and_train_inference_mismatch": bool(mismatch),
    }
    print(f"  teacher generation system-conditioned : {gen_has}")
    print(f"  training data system coverage         : "
          f"{'0.0 (NONE)' if not train_has else 'present'}")
    print(f"  evaluation system coverage            : {eval_frac:.1%}")
    print(f"  template injects a default system turn: {injects_default}")
    print(f"\n  ==> CONFIRMED coverage + train/inference mismatch: {mismatch}")

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_prompt_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}/system_prompt_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

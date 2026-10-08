#!/usr/bin/env python
"""Post-hoc re-analysis of every retained model under the clarified stage objectives.

    PYTHONPATH=src python scripts/evaluation/reevaluate_stage23.py \
        --out artifacts/audit/stage23_reevaluation.json

**Exploratory and post-hoc.** It re-reads retained artifacts under an evaluation
hierarchy defined *after* those artifacts were produced. It may rank current
candidates and inform a prospectively registered Stage 2/3 gate. It must not be
read as converting any past experiment into a pre-registered behaviour
experiment, and it declares no pass/fail threshold.

Nothing is generated: no model is loaded, no GPU is touched, no prompt is sent.
Where a metric cannot be computed from what was retained, the arm is reported as
`not_evaluable` with the missing field named, rather than quietly skipped.

Hierarchy (Stage 2/3):
  primary    usable_rollout and its five components
  secondary  correctness, per-task correctness, correctness | usable rollout
  diagnostic teacher-forced top-1, teacher-native CE, FineWeb NLL, train loss

Stage 0/1 is assessed separately: step-0 initialization NLL under the pinned
protocol, then whether that advantage produced a downstream behaviour benefit in
the matched Experiment 1 PCA-vs-random arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.evaluation import usable_rollout as ur          # noqa: E402
from aadistill.evaluation.behavior import split_generation     # noqa: E402
from aadistill.evaluation.strict_answer import score_numeric   # noqa: E402
from aadistill.infrastructure.env import code_state            # noqa: E402
from run_three_mode_diagnostic import score                    # noqa: E402

AUDIT = REPO_ROOT / "artifacts/audit"
E1 = REPO_ROOT / "artifacts/eval/e1"
REF = REPO_ROOT / "artifacts/eval/e2diag_rescored_v2"
STAGE1 = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0"

# Stage 2/3 candidates that carry the full three-mode free-rollout schema.
THREE_MODE = ["P0-real-sa", "P0-real-sb",
              "P0-assistant-sa", "P0-assistant-sb",
              "P2-ceheavy-sa", "P2-ceheavy-sb"]

RUNGS = ["r0250k", "r0460k", "r0860k", "r1600k", "r2960k", "r5500k"]
SEEDS = ["sa", "sb"]
INITS = ["pca", "rand"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


# ---------------------------------------------------------------- Stage 2/3

def three_mode_arm(alias: str) -> dict:
    d = AUDIT / "three_mode" / alias
    free = d / "free.generations.jsonl"
    if not free.exists():
        return {"not_evaluable": f"missing {free.relative_to(REPO_ROOT)}"}
    recs = jsonl(free)
    out = {"n": len(recs), "primary": ur.summarize(recs)}

    # Secondary correctness is RE-SCORED, never read from the stored `correct`.
    # The rows were scored at generation time and the scorer has since been
    # corrected twice (free-form QA must not be held to the numeric
    # final-answer-marker rule; the corpus `gold` is a worked solution, not a
    # bare answer). The stored field puts P0-real at 0.0067 and the corrected
    # scorer at 0.1533 -- a 23x difference that would have made every arm
    # generated before the fix look catastrophically worse than the ones after.
    sessions = {}
    sp = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
    if sp.is_file():
        for line in sp.open():
            s_ = json.loads(line)
            sessions[s_["id"]] = s_
    for r in recs:
        s_ = sessions.get(r["id"])
        if s_ is None:
            raise KeyError(f"session {r['id']} absent; cannot re-score {alias}")
        body = split_generation(r["raw"], think_preopened=True)["answer"]
        r["correct"] = score(s_, body.strip())

    comp = [ur.components(r) for r in recs]
    usable = [all(c.values()) for c in comp]
    correct = [bool(r["correct"]) for r in recs]
    out["secondary"] = {
        "correct_overall": round(sum(correct) / len(recs), 4),
        "correct_given_usable": (
            round(sum(c for c, u in zip(correct, usable) if u) / sum(usable), 4)
            if sum(usable) else None),
        "n_usable": sum(usable),
        "by_task": {},
    }
    tasks = sorted({r["data_type"] for r in recs})
    for t in tasks:
        idx = [i for i, r in enumerate(recs) if r["data_type"] == t]
        nu = sum(usable[i] for i in idx)
        out["secondary"]["by_task"][t] = {
            "n": len(idx),
            "correct": round(sum(correct[i] for i in idx) / len(idx), 4),
            "usable_rollout_rate": round(nu / len(idx), 4),
            "correct_given_usable": (
                round(sum(correct[i] for i in idx if usable[i]) / nu, 4)
                if nu else None),
        }

    # diagnostics -- reported, never used to rank
    diag = {}
    fr = d / "forced" / "report.json"
    fr = fr if fr.exists() else d / "report.json"
    if fr.exists():
        by_role = json.loads(fr.read_text())["results"]["forced"]["by_role"]
        diag["teacher_forced_reasoning_top1"] = by_role["reasoning"]["top1_accuracy"]
        diag["teacher_forced_reasoning_mean_rank"] = by_role["reasoning"]["mean_target_rank"]
    out["diagnostics"] = diag
    out["per_sample_usable"] = {r["id"]: u for r, u in zip(recs, usable)}
    out["per_sample_correct"] = {r["id"]: c for r, c in zip(recs, correct)}
    return out


# ------------------------------------------------------------- Experiment 1

def e1_arm(rung: str, seed: str, init: str) -> dict:
    stem = f"e1_{rung}_{seed}_{init}"
    out = {"arm": stem}
    beh = E1 / f"{stem}_behavior.generations.jsonl"
    gsm = E1 / f"{stem}_gsm8k.generations.jsonl"
    hold = E1 / f"{stem}_holdout.json"
    if not beh.exists():
        return {**out, "not_evaluable": f"missing {beh.name}"}

    out["behavior_76"] = ur.summarize(jsonl(beh))

    if gsm.exists():
        recs = jsonl(gsm)
        gold = {r["id"]: r["gsm8k_answer"]
                for r in jsonl(E1 / "gsm8k_reasoning_100.jsonl")}
        comp = [ur.components(r) for r in recs]
        usable = [all(c.values()) for c in comp]
        scored = [score_numeric(r, gold[r["id"]]) for r in recs]
        # `answer_matches` ignores protocol/degeneration, so the secondary axis
        # does not silently re-count the primary one.
        matches = [bool(s["answer_matches_ignoring_protocol"]) for s in scored]
        strict = [bool(s["correct"]) for s in scored]
        nu = sum(usable)
        out["gsm8k_100"] = {
            **ur.summarize(recs),
            "answer_matches_ignoring_protocol": round(sum(matches) / len(recs), 4),
            "correct_strict": round(sum(strict) / len(recs), 4),
            "correct_given_usable": (
                round(sum(m for m, u in zip(matches, usable) if u) / nu, 4)
                if nu else None),
        }
        out["per_sample_usable_gsm8k"] = {r["id"]: u for r, u in zip(recs, usable)}
    else:
        out["gsm8k_100"] = {"not_evaluable": f"missing {gsm.name}"}

    if hold.exists():
        res = json.loads(hold.read_text())["results"]
        out["diagnostics"] = {"fineweb_holdout_nll": res[0]["mean_nll_nats"],
                              "perplexity": res[0]["perplexity"],
                              "eval_tokens": res[0]["eval_tokens"]}
    else:
        out["diagnostics"] = {"not_evaluable": f"missing {hold.name}"}
    return out


# ------------------------------------------------------------- Stage 0 / 1

def initialization() -> dict:
    ev = json.loads((STAGE1 / "eval_holdout_v1.json").read_text())
    by = {r["model"].split("/")[-1]: r for r in ev["results"]}
    teacher = next(r for k, r in by.items() if "Qwen3-4B" in r["model"])
    pca = by["checkpoint"]
    rand = by["random_baseline"]
    return {
        "protocol": {
            "data": ev["data"]["path"], "sha256": ev["data"]["sha256"],
            "num_samples": ev["data"]["num_samples"],
            "max_seq_len": ev["max_seq_len"], "dtype": ev["dtype"],
            "eval_tokens": pca["eval_tokens"],
            "identical_to_fineweb_holdout_protocol": True,
            "note": ("Same corpus, truncation, dtype and token count as every "
                     "later FineWeb held-out NLL measurement, so step-0 and "
                     "post-training numbers are directly comparable."),
        },
        "step0_nll_nats": {
            "teacher_Qwen3-4B-Thinking": teacher["mean_nll_nats"],
            "stage1_pca_init": pca["mean_nll_nats"],
            "random_init": rand["mean_nll_nats"],
        },
        "step0_perplexity": {
            "teacher_Qwen3-4B-Thinking": teacher["perplexity"],
            "stage1_pca_init": pca["perplexity"],
            "random_init": rand["perplexity"],
        },
        "pca_minus_random_nats": round(pca["mean_nll_nats"] - rand["mean_nll_nats"], 4),
        "seeds_per_condition": 1,
        "limitation": ("One draw per condition. No random-init seed spread was "
                       "measured, so the step-0 gap has no error bar."),
    }


# ----------------------------------------------------------------- external

def reference() -> dict:
    """Qwen3-0.6B: external capability/behaviour reference only.

    Battery per-sample records predate the context-limit field. That is harmless
    for the conjunction: natural termination and a context-limit hit are mutually
    exclusive stop reasons, so requiring natural_termination already implies
    no_context_limit. Reported explicitly rather than assumed.
    """
    out = {}
    for proto in ("project", "native"):
        p = REF / f"ref_qwen3_0p6b_{proto}_battery.per_sample.jsonl"
        if not p.exists():
            out[proto] = {"not_evaluable": f"missing {p.name}"}
            continue
        recs = jsonl(p)
        adapted = [{"empty_answer": not (r.get("answer") or "").strip(),
                    "natural_termination": bool(r["natural_termination"]),
                    "degenerate": bool(r["degenerate"]),
                    "context_limit": False,
                    "protocol_valid": bool(r["protocol_valid"])} for r in recs]
        s = ur.summarize(adapted)
        s["no_context_limit"] = None
        s["context_limit_field"] = "not recorded; implied by natural_termination"
        s["correct_overall"] = round(sum(bool(r["correct"]) for r in recs) / len(recs), 4)
        by = {}
        for st in sorted({r["set"] for r in recs}):
            idx = [i for i, r in enumerate(recs) if r["set"] == st]
            by[st] = {"n": len(idx),
                      "correct": round(sum(bool(recs[i]["correct"]) for i in idx) / len(idx), 4),
                      "usable_rollout_rate": round(
                          sum(all(adapted[i].values()) for i in idx) / len(idx), 4)}
        s["by_set"] = by
        out[proto] = s
    out["usage_constraint"] = (
        "External capability/behaviour reference only. Cross-model teacher-forced "
        "top-1 is NOT a capacity scale (decisions.md 2026-08-05) and is not "
        "computed here.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "status": "POST-HOC EXPLORATORY RE-ANALYSIS",
        "declares_threshold": False,
        "new_computation_performed": "none; retained artifacts re-read only",
        "usable_rollout_definition": (
            "non_empty AND natural_termination AND no_severe_repetition AND "
            "no_context_limit AND protocol_valid"),
        "code_state": code_state(REPO_ROOT),
        "stage01_initialization": initialization(),
        "stage23_candidates": {a: three_mode_arm(a) for a in THREE_MODE},
        "experiment1_arms": {
            f"{r}_{s}_{i}": e1_arm(r, s, i)
            for r in RUNGS for s in SEEDS for i in INITS},
        "external_reference_qwen3_0p6b": reference(),
    }

    # paired prompt-level differences within each family (same fixed ids)
    paired = {}
    tm = report["stage23_candidates"]
    for a, b in (("P0-real-sa", "P2-ceheavy-sa"), ("P0-real-sb", "P2-ceheavy-sb"),
                 ("P0-real-sa", "P0-assistant-sa"), ("P0-real-sb", "P0-assistant-sb")):
        ua, ub = tm[a].get("per_sample_usable"), tm[b].get("per_sample_usable")
        ca, cb = tm[a].get("per_sample_correct"), tm[b].get("per_sample_correct")
        if not (ua and ub):
            continue
        ids = sorted(set(ua) & set(ub))
        paired[f"{b} vs {a}"] = {
            "n_paired": len(ids),
            "usable_gained": sum(ub[i] and not ua[i] for i in ids),
            "usable_lost": sum(ua[i] and not ub[i] for i in ids),
            "usable_net": sum(ub[i] for i in ids) - sum(ua[i] for i in ids),
            "correct_gained": sum(cb[i] and not ca[i] for i in ids),
            "correct_lost": sum(ca[i] and not cb[i] for i in ids),
        }
    report["paired_prompt_level"] = paired
    for v in report["stage23_candidates"].values():
        v.pop("per_sample_usable", None)
        v.pop("per_sample_correct", None)
    for v in report["experiment1_arms"].values():
        v.pop("per_sample_usable_gsm8k", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

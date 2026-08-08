#!/usr/bin/env python
"""Experiment 6 analysis: the E1 PCA scale curve on the frozen 150-prompt battery.

    PYTHONPATH=src python scripts/evaluation/analyze_e6.py \
        --out artifacts/audit/e6_results.json --report logs/e6_report.md

Nothing is generated here. Every arm is re-scored from its retained raw
generations with the current scorer, and no stored `correct`, `usable` or
termination field is carried into an E6 number. Run it as often as you like: the
paired bootstrap is seeded, so the report is reproducible from the artifacts
alone.

Two sessions produced the artifacts, and the registration fixed in advance which
one answers which question:

* the **E1 scale curve** (2.96M and 5.50M against 1.60M) is read entirely from
  the E6 session, so all three rungs were measured together on one GPU;
* the **anchor comparisons at 1.60M** are read from the E4 session, where
  P2-1.60M and E1-1.60M were measured together.

E1-1.60M therefore exists in both sessions on identical weights, which turns the
cross-session difference from an assumption into a measurement. It is reported
as `session_replicate` and never quietly averaged with anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.evaluation import usable_rollout as ur  # noqa: E402
from aadistill.evaluation.behavior import split_generation  # noqa: E402
from aadistill.evaluation.paired_stats import (  # noqa: E402
    joint_rate, mcnemar_counts, paired_bootstrap_ci,
)
from aadistill.evaluation.strict_answer import extract_final_answer  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from run_three_mode_diagnostic import NUMERIC, score  # noqa: E402

AUDIT = REPO_ROOT / "artifacts/audit"
THREE_MODE = AUDIT / "three_mode"
SESSIONS_PATH = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
REGISTRATION = REPO_ROOT / "logs/e6_registration.json"

# Carried unchanged from the E3/E4/E5 registry; see the registration.
FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}

RUNGS = {"E1-1.60M": 1600353, "E1-2.96M": 2960507, "E1-5.50M": 5501372,
         "P2-1.60M": 1600353}
# Rung tokens are UNIQUE supervised tokens. Every arm trains 3 epochs, so the
# cumulative CE-token exposure is 3x the rung -- reported separately because the
# question is phrased in rung terms and the two must not be confused.
EPOCHS = 3
SEEDS = ("sa", "sb")


def _rel(p: Path) -> str:
    """Repo-relative when it is inside the repo; absolute otherwise.

    The smoke path points at a scratch copy outside the tree, and
    `relative_to` raises rather than falling back.
    """
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()]


def load_sessions() -> dict:
    return {s["id"]: s for s in jsonl(SESSIONS_PATH)}


def rescore_arm(directory: Path, sessions: dict) -> dict | None:
    """Every metric E6 reports for one arm, computed from raw generations."""
    free = directory / "free.generations.jsonl"
    if not free.is_file():
        return None
    recs = jsonl(free)

    comp = [ur.components(r) for r in recs]
    usable = [all(c.values()) for c in comp]
    correct, parse_fail, applicable = [], [], []
    for r in recs:
        s = sessions.get(r["id"])
        if s is None:
            raise KeyError(f"session {r['id']} absent; cannot re-score {directory.name}")
        body = split_generation(r["raw"], think_preopened=True)["answer"].strip()
        correct.append(bool(score(s, body)))
        numeric = s["data_type"] in NUMERIC
        applicable.append(numeric)
        # A parse failure is a NON-EMPTY answer from which the pre-registered
        # numeric rule extracts nothing. An empty answer is an empty answer and
        # is counted there, not here; conflating them would let a silent model
        # look like a formatting problem.
        parse_fail.append(bool(numeric and body
                               and extract_final_answer(body)[0] is None))

    n = len(recs)
    lengths = [r["generated_tokens"] for r in recs]
    n_usable = sum(usable)
    n_numeric = sum(applicable)

    def rate(xs) -> float:
        return round(sum(xs) / n, 4)

    out = {
        "n": n,
        "usable_rollout_rate": rate(usable),
        "correct_overall": rate(correct),
        "correct_given_usable": (round(sum(c for c, u in zip(correct, usable) if u)
                                       / n_usable, 4) if n_usable else None),
        "protocol_valid_rate": rate(c["protocol_valid"] for c in comp),
        "natural_termination_rate": rate(c["natural_termination"] for c in comp),
        "context_limit_rate": rate(not c["no_context_limit"] for c in comp),
        "severe_repetition_rate": rate(not c["no_severe_repetition"] for c in comp),
        "empty_output_rate": rate(not c["non_empty"] for c in comp),
        "answer_parse_failure_rate_numeric": (
            round(sum(parse_fail) / n_numeric, 4) if n_numeric else None),
        "answer_parse_failure_rate_all": rate(parse_fail),
        "generated_tokens": {
            "mean": round(statistics.fmean(lengths), 1),
            "p50": int(statistics.median_low(lengths)),
            "p90": int(sorted(lengths)[min(n - 1, int(0.9 * n))]),
            "max": max(lengths),
        },
        "counts": {
            "included": n,
            "usable": n_usable,
            "correct": sum(correct),
            "correct_and_usable": sum(c and u for c, u in zip(correct, usable)),
            "natural_termination": sum(c["natural_termination"] for c in comp),
            "context_limit": sum(not c["no_context_limit"] for c in comp),
            "severe_repetition": sum(not c["no_severe_repetition"] for c in comp),
            "empty": sum(not c["non_empty"] for c in comp),
            "numeric_prompts": n_numeric,
            "answer_parse_failure": sum(parse_fail),
        },
        "first_failure_census": ur.summarize(recs)["first_failure"],
        "by_task": {},
        "per_sample": {
            "usable": {r["id"]: u for r, u in zip(recs, usable)},
            "correct": {r["id"]: c for r, c in zip(recs, correct)},
            "natural_termination": {r["id"]: c["natural_termination"]
                                    for r, c in zip(recs, comp)},
            "context_limit": {r["id"]: not c["no_context_limit"]
                              for r, c in zip(recs, comp)},
            "generated_tokens": {r["id"]: r["generated_tokens"] for r in recs},
        },
    }
    for task in sorted({r["data_type"] for r in recs}):
        idx = [i for i, r in enumerate(recs) if r["data_type"] == task]
        nu = sum(usable[i] for i in idx)
        out["by_task"][task] = {
            "n": len(idx),
            "usable_rollout_rate": round(nu / len(idx), 4),
            "correct_overall": round(sum(correct[i] for i in idx) / len(idx), 4),
            "correct_given_usable": (
                round(sum(correct[i] for i in idx if usable[i]) / nu, 4) if nu else None),
            "natural_termination_rate": round(
                sum(comp[i]["natural_termination"] for i in idx) / len(idx), 4),
            "context_limit_rate": round(
                sum(not comp[i]["no_context_limit"] for i in idx) / len(idx), 4),
            "counts": {"included": len(idx), "usable": nu,
                       "correct": sum(correct[i] for i in idx)},
        }
    return out


def diagnostics(reg: dict) -> dict:
    """Teacher-native CE, FineWeb NLL and teacher-forced top-1, per arm.

    Reported so question 4 -- does higher exposure move behaviour, correctness,
    or only the diagnostics -- can be answered. **None of these may select a
    winner.** Every one is on the diagnostic tier of the registered hierarchy,
    and E4 already demonstrated the dissociation: CE improved by 18x its floor
    while the primary axis did not move at all.
    """
    e1 = {r["arm"]: r for r in
          json.loads((REPO_ROOT / "artifacts/stage3/e1_results.json").read_text())}
    out = {}
    for alias, arm in reg["arms"].items():
        row = {}
        rec = e1.get(arm["run"])
        if rec is not None:
            row["teacher_native_val_ce"] = rec["val_ce_final"]
            row["fineweb_holdout_nll"] = rec["holdout_nll"]
        else:
            # The anchors are not Experiment 1 arms, so their CE comes from their
            # own training log.
            log = REPO_ROOT / f"artifacts/stage3/{arm['run']}/train_log.jsonl"
            if log.is_file():
                evals = [r for r in jsonl(log) if r["event"] == "eval_result"
                         and r.get("val_set") == "val"]
                if evals:
                    row["teacher_native_val_ce"] = evals[-1]["val_ce"]
        for tag, d in (("", THREE_MODE / alias),
                       ("@E4", THREE_MODE / (arm["retained_three_mode"] or "_"))):
            fr = d / "forced" / "report.json"
            if not fr.is_file():
                continue
            by_role = json.loads(fr.read_text())["results"]["forced"]["by_role"]
            row[f"teacher_forced_reasoning_top1{tag}"] = by_role["reasoning"]["top1_accuracy"]
            row[f"teacher_forced_reasoning_mean_rank{tag}"] = \
                by_role["reasoning"]["mean_target_rank"]
        if row:
            row["tier"] = "diagnostic only — may not select a winner"
            out[alias] = row
    return out


def prior_harness() -> dict:
    """What the retired 76-prompt wave said about these same three rungs.

    Included so E6 can answer whether the earlier ordering survives — and only
    the **ordering**. The levels are not comparable and must never be put in one
    table as if they were: different prompt population (76 behaviour prompts vs
    150 corpus examples), and the degeneration stop was ACTIVE, which cuts a
    repetition loop early and so changes the termination and context-limit
    components outright. The same weights score 0.4868 there and something else
    entirely here.
    """
    path = REPO_ROOT / "artifacts/audit/stage23_reevaluation.json"
    if not path.is_file():
        return {"not_evaluable": "stage23_reevaluation.json absent"}
    arms = json.loads(path.read_text())["experiment1_arms"]
    out = {}
    for fam, stem in (("E1-1.60M", "r1600k"), ("E1-2.96M", "r2960k"),
                      ("E1-5.50M", "r5500k")):
        per = {}
        for seed in SEEDS:
            b = arms.get(f"{stem}_{seed}_pca", {}).get("behavior_76")
            if b:
                per[seed] = b["usable_rollout_rate"]
        if len(per) == len(SEEDS):
            out[fam] = {"usable_rollout_rate_per_seed": per,
                        "mean": round(statistics.fmean(per.values()), 4)}
    ranked = sorted(out, key=lambda k: -out[k]["mean"])
    return {"harness": "76 behaviour prompts, E1 behaviour wave, degeneration "
                       "stop ACTIVE",
            "comparable_with_e6": "ORDERING ONLY — never the levels",
            "arms": out, "ranking_best_first": ranked}


def environment(provenance: dict) -> dict:
    """What produced each arm's generations, read from its own report.

    Recorded per arm rather than once for the run, because E6 deliberately mixes
    two sessions and the whole point of the reuse rule is that the difference is
    visible. If a library version or the harness command differs anywhere, it
    shows up here rather than in a footnote.
    """
    out, seen = {}, {}
    for alias, p in provenance.items():
        report = Path(p["directory"]) / "report.json"
        report = report if report.is_absolute() else REPO_ROOT / report
        if not report.is_file():
            continue
        r = json.loads(report.read_text())
        out[alias] = {
            "session": p["session"],
            "created_utc": r.get("created_utc"),
            "command": r.get("command"),
            "libraries": r.get("libraries"),
            "code_state": r.get("code_state"),
            "context": r.get("context"),
            "decoding": r.get("decoding"),
            "rung": r.get("rung"),
            "inclusion": r.get("inclusion", {}).get("mask_sha256"),
        }
        key = json.dumps([r.get("libraries"), r.get("code_state", {}).get("git_commit"),
                          r.get("context"), r.get("decoding")], sort_keys=True)
        seen.setdefault(key, []).append(alias)
    out["_distinct_environments"] = [
        {"arms": sorted(v), "n": len(v)} for v in seen.values()]
    return out


def write_per_prompt(arms: dict, sessions: dict, path: Path) -> int:
    """One scored record per (arm, prompt): the evidence behind every rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for alias, m in sorted(arms.items()):
            ps = m["per_sample"]
            for pid in sorted(ps["usable"]):
                f.write(json.dumps({
                    "arm": alias, "id": pid,
                    "data_type": sessions[pid]["data_type"],
                    "usable_rollout": ps["usable"][pid],
                    "correct": ps["correct"][pid],
                    "natural_termination": ps["natural_termination"][pid],
                    "context_limit": ps["context_limit"][pid],
                    "generated_tokens": ps["generated_tokens"][pid],
                }) + "\n")
                n += 1
    return n


def arm_alias(family: str, seed: str) -> str:
    """`E1-1.60M` + `sa` -> `E1-1.60M-sa`; `E1-1.60M@E4` -> `E1-1.60M-sa@E4`.

    The session suffix belongs to the arm, not to the family, so it has to be
    reattached after the seed. Getting this wrong silently drops a comparison —
    `family_compare` then reports `incomplete` for a pair whose arms are both
    loaded, which reads exactly like a missing measurement.
    """
    if family.endswith("@E4"):
        return f"{family[:-3]}-{seed}@E4"
    return f"{family}-{seed}"


def compare(a_alias: str, b_alias: str, arms: dict, iterations: int) -> dict | None:
    """b against a, paired on the shared prompt mask."""
    a, b = arms.get(a_alias), arms.get(b_alias)
    if a is None or b is None:
        return None
    out = {"a": a_alias, "b": b_alias}
    for axis in ("usable", "correct"):
        pa, pb = a["per_sample"][axis], b["per_sample"][axis]
        counts = mcnemar_counts(pa, pb)
        ci = paired_bootstrap_ci(pa, pb, iterations=iterations)
        out[axis] = {
            "rate_a": counts["rate_a"], "rate_b": counts["rate_b"],
            "delta": counts["delta"],
            "win": counts["b_gained"], "loss": counts["b_lost"],
            "tie": counts["both_true"] + counts["both_false"],
            "n_paired": counts["n_paired"],
            "bootstrap_ci": [ci["ci_low"], ci["ci_high"]],
            "ci_excludes_zero": ci["ci_excludes_zero"],
        }
    out["natural_termination"] = mcnemar_counts(
        a["per_sample"]["natural_termination"], b["per_sample"]["natural_termination"])
    out["correct_and_naturally_terminated"] = {
        "a": joint_rate(a["per_sample"]["correct"],
                        a["per_sample"]["natural_termination"]),
        "b": joint_rate(b["per_sample"]["correct"],
                        b["per_sample"]["natural_termination"]),
    }
    return out


def family_compare(fa: str, fb: str, arms: dict, iterations: int) -> dict:
    """Both seeds independently, the two-seed direction, and the pooled delta."""
    per_seed = {s: compare(arm_alias(fa, s), arm_alias(fb, s), arms, iterations)
                for s in SEEDS}
    per_seed = {s: v for s, v in per_seed.items() if v is not None}
    out = {"a": fa, "b": fb, "per_seed": per_seed}
    if len(per_seed) != len(SEEDS):
        have = sorted(per_seed)
        missing = [s for s in SEEDS if s not in per_seed]
        out["incomplete"] = (
            f"needs both seeds of {fa} and {fb}; have {have or 'neither'}, "
            f"missing {missing}. No claim is made from one seed.")
        return out
    for axis, floor_key in (("usable", "usable_rollout_rate"),
                            ("correct", "correct_overall")):
        deltas = [per_seed[s][axis]["delta"] for s in SEEDS]
        pooled = round(statistics.fmean(deltas), 4)
        directions = {"up" if d > 0 else "down" if d < 0 else "flat" for d in deltas}
        out[axis] = {
            "delta_per_seed": {s: per_seed[s][axis]["delta"] for s in SEEDS},
            "pooled_delta": pooled,
            "seed_consistent": len(directions) == 1 and directions != {"flat"},
            "direction": sorted(directions),
            "floor": FLOORS[floor_key],
            "exceeds_floor": abs(pooled) >= FLOORS[floor_key],
            "both_cis_exclude_zero": all(
                per_seed[s][axis]["ci_excludes_zero"] for s in SEEDS),
            "verdict": _verdict(pooled, directions, FLOORS[floor_key]),
        }
    return out


def _verdict(pooled: float, directions: set, floor: float) -> str:
    if abs(pooled) < floor:
        return "inside the floor — not an effect"
    if len(directions) != 1:
        return "above the floor but seeds disagree — not claimable"
    return f"above the floor and seed-consistent ({'better' if pooled > 0 else 'worse'})"


def examples(a_alias: str, b_alias: str, arms: dict, sessions: dict, k: int) -> dict:
    """A deterministic sample of what changed, for explanation only."""
    a, b = arms.get(a_alias), arms.get(b_alias)
    if a is None or b is None:
        return {}
    buckets: dict[str, list] = {"newly_usable": [], "newly_unusable": [],
                                "newly_correct": [], "newly_incorrect": [],
                                "termination_improved": [], "termination_regressed": []}
    for pid in sorted(set(a["per_sample"]["usable"]) & set(b["per_sample"]["usable"])):
        ua, ub = a["per_sample"]["usable"][pid], b["per_sample"]["usable"][pid]
        ca, cb = a["per_sample"]["correct"][pid], b["per_sample"]["correct"][pid]
        ta = a["per_sample"]["natural_termination"][pid]
        tb = b["per_sample"]["natural_termination"][pid]
        row = {"id": pid, "data_type": sessions[pid]["data_type"],
               "tokens_a": a["per_sample"]["generated_tokens"][pid],
               "tokens_b": b["per_sample"]["generated_tokens"][pid]}
        if ub and not ua:
            buckets["newly_usable"].append(row)
        if ua and not ub:
            buckets["newly_unusable"].append(row)
        if cb and not ca:
            buckets["newly_correct"].append(row)
        if ca and not cb:
            buckets["newly_incorrect"].append(row)
        if tb and not ta:
            buckets["termination_improved"].append(row)
        if ta and not tb:
            buckets["termination_regressed"].append(row)
    return {name: {"n": len(rows), "sample": rows[:k]} for name, rows in buckets.items()}


def main() -> None:
    global THREE_MODE
    ap = argparse.ArgumentParser()
    # Small enough to track (~140 KB): the per-sample maps are stripped before
    # writing, so this is the summary, not the generations. The generations stay
    # under the gitignored `artifacts/` tree and on the dev-box store.
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e6_results.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "logs/e6_report.md")
    # 1,500 rows of raw evidence: too big for logs/, and it lives beside the
    # generations it summarises under the gitignored artifacts tree.
    ap.add_argument("--per-prompt", type=Path,
                    default=AUDIT / "e6_per_prompt.jsonl")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--examples", type=int, default=3)
    # Only so the whole pipeline can be exercised end-to-end against a scratch
    # copy before the real generations land. The default is the real tree.
    ap.add_argument("--three-mode", type=Path, default=THREE_MODE)
    args = ap.parse_args()

    THREE_MODE = args.three_mode
    reg = json.loads(REGISTRATION.read_text())
    sessions = load_sessions()

    # Where each arm's generations live. A regenerated arm is read from the E6
    # session directory; a reused arm from the retained one it was registered
    # against. E1-1.60M is read from BOTH and the E4 copy is kept separate.
    arms, provenance, missing = {}, {}, []
    for alias, arm in reg["arms"].items():
        d = THREE_MODE / (alias if arm["generate"] else arm["retained_three_mode"])
        scored = rescore_arm(d, sessions)
        if scored is None:
            missing.append(alias)
        else:
            arms[alias] = scored
            provenance[alias] = {
                "directory": _rel(d),
                "session": "E6" if arm["generate"] else "E4",
                "rung": arm["rung"], "seed": arm["seed"],
                "run": arm["run"], "weights_sha256": arm["weights_sha256"]}
        # The retained replicate is loaded whether or not the E6 arm exists, so
        # a failed regeneration cannot silently take the anchor comparison with
        # it -- that comparison is answered by the E4 session either way.
        if arm["generate"] and arm["retained_three_mode"]:
            rd = THREE_MODE / arm["retained_three_mode"]
            rep = rescore_arm(rd, sessions)
            if rep is not None:
                alias4 = f"{alias}@E4"
                arms[alias4] = rep
                provenance[alias4] = {
                    "directory": _rel(rd),
                    "session": "E4", "rung": arm["rung"], "seed": arm["seed"],
                    "run": arm["run"], "weights_sha256": arm["weights_sha256"],
                    "role": "cross-session replicate of the same weights"}

    families = {}
    for fam in ("E1-1.60M", "E1-2.96M", "E1-5.50M", "P2-1.60M", "E1-1.60M@E4"):
        present = [p for p in (arm_alias(fam, s) for s in SEEDS) if p in arms]
        if len(present) != len(SEEDS):
            continue
        families[fam] = {
            "seeds": {p: {k: arms[p][k] for k in
                          ("usable_rollout_rate", "correct_overall",
                           "correct_given_usable", "protocol_valid_rate",
                           "natural_termination_rate", "context_limit_rate",
                           "severe_repetition_rate", "empty_output_rate",
                           "answer_parse_failure_rate_numeric", "generated_tokens",
                           "counts")} for p in present},
            "mean": {k: round(statistics.fmean([arms[p][k] for p in present]), 4)
                     for k in ("usable_rollout_rate", "correct_overall",
                               "protocol_valid_rate", "natural_termination_rate",
                               "context_limit_rate", "severe_repetition_rate",
                               "empty_output_rate")},
            # Pooled over both seeds' prompts, which is how every earlier table
            # in this project reported it. The per-seed values above are the
            # primary reading; this exists so E6 rows line up with E4's and E5's.
            "correct_given_usable_pooled": round(
                sum(arms[p]["counts"]["correct_and_usable"] for p in present)
                / max(1, sum(arms[p]["counts"]["usable"] for p in present)), 4),
            "spread": {k: round(abs(arms[present[0]][k] - arms[present[1]][k]), 4)
                       for k in ("usable_rollout_rate", "correct_overall")},
            "rung_unique_supervised_tokens": RUNGS.get(fam.replace("@E4", "")),
            "cumulative_ce_tokens": (RUNGS.get(fam.replace("@E4", "")) or 0) * EPOCHS,
        }

    # The registered comparison set. Curve comparisons read the E6 session; the
    # anchor comparisons at 1.60M read the E4 session for both sides.
    comparisons = {
        "E1-2.96M vs E1-1.60M": family_compare("E1-1.60M", "E1-2.96M", arms, args.bootstrap),
        "E1-5.50M vs E1-1.60M": family_compare("E1-1.60M", "E1-5.50M", arms, args.bootstrap),
        "E1-5.50M vs E1-2.96M": family_compare("E1-2.96M", "E1-5.50M", arms, args.bootstrap),
        "E1-1.60M vs P2-1.60M": family_compare("P2-1.60M", "E1-1.60M@E4", arms, args.bootstrap),
        "E1-2.96M vs P2-1.60M": family_compare("P2-1.60M", "E1-2.96M", arms, args.bootstrap),
        "E1-5.50M vs P2-1.60M": family_compare("P2-1.60M", "E1-5.50M", arms, args.bootstrap),
    }
    session_replicate = {
        s: compare(f"E1-1.60M-{s}@E4", f"E1-1.60M-{s}", arms, args.bootstrap)
        for s in SEEDS if f"E1-1.60M-{s}@E4" in arms and f"E1-1.60M-{s}" in arms
    }

    qualitative = {
        f"{name} [{s}]": examples(arm_alias(c["a"], s), arm_alias(c["b"], s),
                                  arms, sessions, args.examples)
        for name, c in comparisons.items() if "a" in c for s in SEEDS
    }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": reg["experiment"],
        "registration_sha256": reg["registration_sha256"],
        "inclusion_mask_sha256": reg["frozen_assets"]["inclusion_mask_sha256"],
        "floors": FLOORS,
        "arms": {a: {k: v for k, v in m.items() if k != "per_sample"}
                 for a, m in arms.items()},
        "provenance": provenance,
        "missing_arms": missing,
        "families": families,
        "comparisons": comparisons,
        "environment": environment(provenance),
        "prior_harness": prior_harness(),
        "diagnostics": diagnostics(reg),
        "session_replicate": session_replicate,
        "qualitative_examples": qualitative,
        "code_state": code_state(str(REPO_ROOT)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    rows = write_per_prompt(arms, sessions, args.per_prompt)
    print(f"wrote {args.per_prompt} ({rows} scored records)")
    if missing:
        print(f"  MISSING ARMS: {missing}")

    args.report.write_text(render(result))
    print(f"wrote {args.report}")


def render(r: dict) -> str:
    L = ["# Experiment 6 — the E1 PCA scale curve on the frozen battery", "",
         f"Generated {r['created_utc']} from retained generations only; nothing was",
         "generated by this script. Inclusion mask "
         f"`{r['inclusion_mask_sha256'][:16]}…`, 150 prompts, greedy, unrestricted",
         "generation (P18). Every arm re-scored with the current scorer.", "",
         "## Headline", "",
         "| model | CE exposure (unique / cumulative) | seed | usable | correct | correct\\|usable |",
         "| --- | ---: | --- | ---: | ---: | ---: |"]
    order = ["E1-1.60M", "E1-2.96M", "E1-5.50M", "P2-1.60M"]
    for fam in order:
        f = r["families"].get(fam)
        if f is None:
            L.append(f"| {fam} | — | — | not evaluated | | |")
            continue
        exposure = f"{f['rung_unique_supervised_tokens']:,} / {f['cumulative_ce_tokens']:,}"
        for alias, m in f["seeds"].items():
            cgu = m["correct_given_usable"]
            L.append(f"| {fam} | {exposure} | {alias.rsplit('-', 1)[-1]} | "
                     f"{m['usable_rollout_rate']:.4f} | {m['correct_overall']:.4f} | "
                     f"{cgu if cgu is None else f'{cgu:.4f}'} |")
    L += ["", "## Component rates and counts", "",
          "| arm | protocol | nat.term | ctx-limit | repetition | empty | parse-fail (numeric) | tokens p50 / p90 / max |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for alias, m in r["arms"].items():
        g = m["generated_tokens"]
        pf = m["answer_parse_failure_rate_numeric"]
        L.append(f"| {alias} | {m['protocol_valid_rate']:.4f} | "
                 f"{m['natural_termination_rate']:.4f} | {m['context_limit_rate']:.4f} | "
                 f"{m['severe_repetition_rate']:.4f} | {m['empty_output_rate']:.4f} | "
                 f"{'—' if pf is None else f'{pf:.4f}'} | "
                 f"{g['p50']} / {g['p90']} / {g['max']} |")
    L += ["", "## By frozen evaluation subset", "",
          "The battery's four partitions are `gsm8k`, `multihop_qa`, `openmath` and",
          "`rag_evidence`, fixed by the corpus and identical for every arm. There is",
          "**no `behavior` partition here** — that name belongs to the retired",
          "76-prompt `eval_behavior_v0` wave, a different prompt population under a",
          "different stop policy, and it has no counterpart on this battery. It is",
          "absent rather than omitted.", "",
          "| arm | subset | n | usable | correct | correct\\|usable | nat.term | ctx-limit |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for alias, m in r["arms"].items():
        for task, t in m["by_task"].items():
            cgu = t["correct_given_usable"]
            L.append(f"| {alias} | {task} | {t['n']} | {t['usable_rollout_rate']:.4f} | "
                     f"{t['correct_overall']:.4f} | "
                     f"{'—' if cgu is None else f'{cgu:.4f}'} | "
                     f"{t['natural_termination_rate']:.4f} | "
                     f"{t['context_limit_rate']:.4f} |")
    L += ["", "## Paired comparisons on the shared mask", ""]
    for name, c in r["comparisons"].items():
        L.append(f"### {name}")
        if "incomplete" in c or "usable" not in c:
            L += ["", f"Not evaluable: {c.get('incomplete', 'missing arms')}.", ""]
            continue
        for axis in ("usable", "correct"):
            a = c[axis]
            L.append(f"* **{axis}** pooled Δ {a['pooled_delta']:+.4f} "
                     f"(floor {a['floor']:.4f}) — {a['verdict']}; "
                     f"seeds {a['delta_per_seed']}, "
                     f"seed-consistent {a['seed_consistent']}")
            for s, v in c["per_seed"].items():
                x = v[axis]
                L.append(f"  * {s}: {x['rate_a']:.4f} → {x['rate_b']:.4f}, "
                         f"win/tie/loss {x['win']}/{x['tie']}/{x['loss']}, "
                         f"95% CI [{x['bootstrap_ci'][0]:+.4f}, {x['bootstrap_ci'][1]:+.4f}]"
                         f"{' excludes 0' if x['ci_excludes_zero'] else ''}")
        L.append("")
    ph = r.get("prior_harness", {})
    if ph.get("arms"):
        L += ["## Does the retired harness's ordering survive?", "",
              f"Prior: {ph['harness']}. **{ph['comparable_with_e6']}** —",
              "different prompt population and a stop policy that changes the",
              "termination and context-limit components outright.", "",
              "| rung | prior usable (sa / sb) | prior mean | E6 mean |",
              "| --- | --- | ---: | ---: |"]
        for fam, v in ph["arms"].items():
            ps = v["usable_rollout_rate_per_seed"]
            now = r["families"].get(fam, {}).get("mean", {}).get(
                "usable_rollout_rate")
            L.append(f"| {fam} | {ps['sa']:.4f} / {ps['sb']:.4f} | "
                     f"{v['mean']:.4f} | "
                     f"{'not evaluated' if now is None else f'{now:.4f}'} |")
        L += ["", f"Prior ranking, best first: **{' > '.join(ph['ranking_best_first'])}**.",
              ""]
    if r.get("diagnostics"):
        L += ["## Diagnostics — reported, never used to rank", "",
              "| arm | teacher-native val CE | FineWeb holdout NLL | teacher-forced reasoning top-1 |",
              "| --- | ---: | ---: | ---: |"]
        for alias, d in r["diagnostics"].items():
            ce = d.get("teacher_native_val_ce")
            nll = d.get("fineweb_holdout_nll")
            t1 = d.get("teacher_forced_reasoning_top1",
                       d.get("teacher_forced_reasoning_top1@E4"))
            L.append(f"| {alias} | {'—' if ce is None else f'{ce:.4f}'} | "
                     f"{'—' if nll is None else f'{nll:.4f}'} | "
                     f"{'—' if t1 is None else f'{t1:.4f}'} |")
        L.append("")
    if r["session_replicate"]:
        L += ["## Cross-session replicate — identical weights, two sessions", "",
              "The same `e1_r1600k_{sa,sb}_pca` weights, the same frozen battery and",
              "the same harness, measured once in the E4 session and once in E6. This",
              "is the only thing that bounds how much of a cross-session comparison is",
              "the model and how much is the session.", ""]
        worst_u = worst_c = 0.0
        for s, c in r["session_replicate"].items():
            worst_u = max(worst_u, abs(c["usable"]["delta"]))
            worst_c = max(worst_c, abs(c["correct"]["delta"]))
            L.append(f"* E1-1.60M-{s}: usable {c['usable']['rate_a']:.4f} (E4) → "
                     f"{c['usable']['rate_b']:.4f} (E6), Δ {c['usable']['delta']:+.4f}; "
                     f"correct {c['correct']['rate_a']:.4f} → {c['correct']['rate_b']:.4f}, "
                     f"Δ {c['correct']['delta']:+.4f}")
        fu, fc = r["floors"]["usable_rollout_rate"], r["floors"]["correct_overall"]
        L += ["", f"**Largest session difference: {worst_u:.4f} usable, {worst_c:.4f} "
              f"correct** (floors {fu:.4f} / {fc:.4f}).",
              "", ("Both sit inside their floors, so the cross-session anchor "
                   "comparisons — the 2.96M and 5.50M rungs against P2-1.60M — are not "
                   "materially contaminated by the session split."
                   if worst_u < fu and worst_c < fc else
                   "**At least one exceeds its floor.** Any comparison that crosses the "
                   "session boundary — every E1 high rung against the P2-1.60M anchor — "
                   "carries at least this much uncertainty on top of its own interval, "
                   "and must not be read as a clean model-to-model difference. The E1 "
                   "scale curve is unaffected: all three rungs were measured in the E6 "
                   "session."), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

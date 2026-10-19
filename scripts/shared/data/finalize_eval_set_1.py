#!/usr/bin/env python3
"""Apply the maintainer's 2026-10-19 decisions to eval-set-1 and freeze it.

    PYTHONPATH=src:scripts .venv/bin/python scripts/shared/data/finalize_eval_set_1.py [--write]

Without `--write` it builds everything in memory, checks it and prints the
counts; `--write` writes the new files, the new manifest and APPENDS the new
identity to `data/eval-set/registry.json` with status `frozen`. Every original
content file is kept byte for byte: the first identity (0909906b...) and every
result scored under it stay valid as what they measured. What this adds:

* `fluid_reasoning_720/validity_v1.json` -- every item classified: 6 EXCLUDED
  from formal scoring (defects the independent audit established), the rest
  valid, with the audit's lower-severity findings attached as annotations
  rather than silent changes. Formal set: 714 = 570 static + 144 interactive.
* `fluid_reasoning_720/independence_v1.json` -- items that are ONE problem
  under a verifiable equivalence (a bijective renaming of person names whose
  mapping is re-derived here), counted once in independent statistics.
* `stage3_capability_846/core_v1.json` -- what is outside the formal core:
  TriviaQA (withheld, supplementary), the Salesforce xLAM prompts (terms
  unconfirmed), and the four behaviour prompts that also sit in the Stage-2
  training split (outside held-out correctness).
* scorer protocol v2 for every answer-correctness unit, with its conformance
  vectors, and `audit/rescoring_v1_v2.json`: every historical generation
  re-scored under both protocols, item by item.

Refuses unless `logs/budget/decisions.md` carries the authorizing heading.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation import eval_registry as ER  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from shared import eval_sets as ES  # noqa: E402

SET_ID = "eval-set-1"
ROOT = REPO / "data/eval-set" / SET_ID
DECISION_DATE = "2026-10-19"
DECISION_TITLE = ("Maintainer: eval-set-1 finalization and the Stage-0 volume study authorized "
                  "($90 new-spend hard cap)")
DECISION_HEADING = f"{DECISION_DATE} — {DECISION_TITLE}"
AUTHORIZATION = f"logs/budget/decisions.md {DECISION_DATE} ({DECISION_TITLE})"
FLUID = "fluid_reasoning_720/dev_v1.jsonl"
AUDIT = ROOT / "audit/fluid_reasoning_720_quality.json"

#: The six items the maintainer excluded (audit headline `fix_or_drop_candidates`).
EXCLUDED = {
    "82b5c880000f9eaa2c4e": ("unanswerable", "order.seating: the gold 'Nia' never appears in the prompt and the seat count is unstated (generator ordering@v1 dropped a fact)"),
    "83f5811a86b339da4a0c": ("underdetermined", "order.seating: the seat count is unstated; the asked seat is Omar with 2 seats and Kai only in the hidden 3-seat world"),
    "f9010f2dd60a31946a20": ("wrong_gold", "graph.unreachable: the gold lists Elm, a town on no printed road; from the prompt the answer is 'Birch, Juniper'"),
    "a68fc81fffe597f45d54": ("answer_leak", "letter.double_all: the query 'ijk' is itself a worked example (ijk -> iijjkk)"),
    "9fba439d2d0b78476103": ("ambiguous_rule", "str.remove_vowels: only 'o' is ever shown removed; 'remove only o' fits every example and gives another answer"),
    "d36815ef540363149b6c": ("ambiguous_rule", "str.remove_vowels: only 'i' is ever shown removed; 'remove only i' fits every example and gives another answer"),
}

#: Behaviour prompts outside the formal core / held-out correctness (maintainer 2026-10-19).
XLAM_PREFIX = "xlam-"
STAGE2_TRAIN_OVERLAPS = ("openmath-004228", "openmath-000199", "everyday-001567", "smoltalk-423306")

V2_FLUID = "stage0.calibration_volume.score_battery@v2"
V1_FLUID = "stage0.calibration_volume.score_battery@v1"
V2_CAP = "capability-v2 + eval_behavior_v0 / eval-set-1 capability scorer@v2"
V2_LEGACY = "stage3.three_mode.free@v2"

#: Historical generations re-scored under both protocols (local, gitignored).
HIST_FLUID = [
    "artifacts/stages/stage-0/calibration_volume/battery_20261018_092345/teacher_dev/teacher_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/battery_20261018_092345/e1_r2960k_sa_pca_dev/e1_r2960k_sa_pca_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/battery_20261018_092345/ctl_r0860k_sa_dev/ctl_r0860k_sa_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/battery_20261018_092345/ctl_r0860k_sb_dev/ctl_r0860k_sb_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/resolution_20261018_213745/qwen3_0p6b_cap8192_dev/qwen3_0p6b_cap8192_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/resolution_20261018_213745/e1_r5500k_sa_pca_dev/e1_r5500k_sa_pca_dev.jsonl",
    "artifacts/stages/stage-0/calibration_volume/resolution_20261018_213745/e1_r1600k_sa_pca_dev/e1_r1600k_sa_pca_dev.jsonl",
]
HIST_846_DIRS = sorted(
    [*(REPO / "artifacts/stages/stage-3/eval/e2p1").glob("*"),
     *(REPO / "artifacts/stages/stage-3/eval/e2diag_rescored_v2").glob("_staged_ref_*")])
HIST_150 = [REPO / "artifacts/audit/three_mode" / arm / "free.generations.jsonl"
            for arm in ("E1-2.96M-sa", "P2-2.96M-sb")]


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


# --- validity -----------------------------------------------------------------

def validity(tasks: list[dict]) -> dict:
    audit = json.loads(AUDIT.read_text())["findings"]
    ids = {t["task_id"] for t in tasks}
    assert set(EXCLUDED) <= ids
    notes: dict[str, list[dict]] = collections.defaultdict(list)

    def note(tid, cls, severity, text, ref):
        if tid in ids and tid not in EXCLUDED:
            notes[tid].append({"class": cls, "severity": severity, "note": text,
                               "audit": f"audit/fluid_reasoning_720_quality.json :: findings.{ref}"})

    for a in audit["ambiguities"]:
        note(a["task_id"], f"ambiguity.{a['kind']}", a["plausibility"].split()[0],
             a["evidence"], "ambiguities")
    for a in audit["query_in_examples"]:
        if a["kind"] == "answer_equals_example_output":
            note(a["task_id"], "near_trivial.answer_equals_example_output", "low",
                 a["severity"], "query_in_examples")
    for a in audit["duplicates"]:
        if a["kind"] == "duplicate_demo_within_item":
            note(a["task_id"], "reduced_evidence.duplicate_demonstration", "low",
                 a["evidence"], "duplicates")
        if a["kind"].startswith("cross_set_identical_opening"):
            note(a["task_id"], "opening_shared_with_calibration_v1", "informational",
                 a["evidence"], "duplicates")
    for a in audit["interactive_issues"]:
        if a["task_id"] in ids:
            note(a["task_id"], f"interactive.{a['kind']}", a["severity"], a["evidence"],
                 "interactive_issues")
        elif a["task_id"].startswith("ALL_"):
            gen = a["generator_id"]
            fams = set(a["family_id"].split("/")) if a["family_id"] != "*" else None
            for t in tasks:
                if t["generator_id"] == gen and (fams is None or t["family_id"] in fams):
                    note(t["task_id"], f"interactive.{a['kind']}", a["severity"],
                         a["evidence"], "interactive_issues")
    formal = [t for t in tasks if t["task_id"] not in EXCLUDED]
    return {
        "schema": "aadistill.eval_set_validity/v1",
        "eval_set_id": SET_ID, "component": "fluid_reasoning_720", "unit": "dev_v1",
        "items_sha256": sha256_file(ROOT / FLUID),
        "decided_by": "maintainer", "decision": AUTHORIZATION,
        "rule": ("an item is EXCLUDED from formal scoring only for a defect the independent audit "
                 "established and the maintainer accepted (no gold, a wrong or underdetermined gold, "
                 "an answer leak, a highly plausible alternative rule); every other item is valid, "
                 "and a lower-severity finding is attached to it as an annotation. Items, gold and "
                 "weights are unchanged; the original 720 and every result scored on them stay bound "
                 "to the original identity."),
        "counts": {"items": len(tasks), "formal": len(formal), "excluded": len(EXCLUDED),
                   "formal_static": sum(not t["interactive"] for t in formal),
                   "formal_interactive": sum(t["interactive"] for t in formal),
                   "annotated_items": len(notes),
                   "annotations_by_class": dict(sorted(collections.Counter(
                       n["class"] for v in notes.values() for n in v).items()))},
        "excluded": [{"id": k, "class": v[0], "reason": v[1],
                      "audit": "audit/fluid_reasoning_720_quality.json :: headline.fix_or_drop_candidates"}
                     for k, v in sorted(EXCLUDED.items())],
        "annotations": {k: v for k, v in sorted(notes.items())},
        "scorer_findings": ("the audit's 249 surface-form rejections are a SCORER property, not an "
                            "item defect: scorer protocol v2 accepts them (see the manifest's "
                            "scorer and audit/rescoring_v1_v2.json)"),
    }


# --- independence ---------------------------------------------------------------

def renaming_canon(text: str, names: tuple[str, ...]) -> tuple[str, dict]:
    pat = re.compile(r"\b(" + "|".join(sorted(names, key=len, reverse=True)) + r")\b")
    mapping: dict[str, str] = {}

    def sub(m):
        mapping.setdefault(m.group(1), f"<P{len(mapping) + 1}>")
        return mapping[m.group(1)]

    return pat.sub(sub, text), mapping


def independence(tasks: list[dict]) -> dict:
    from stages.calibration_volume.tasks.common import PEOPLE
    groups = collections.defaultdict(list)
    for t in tasks:
        if t["interactive"]:
            continue
        prompt = t["messages"][0]["content"]
        canon, mapping = renaming_canon(prompt, PEOPLE)
        if not mapping:
            continue
        _, gold_map = renaming_canon(str(t["answer"]), PEOPLE)
        if set(gold_map) - set(mapping):
            continue                      # the gold names a person the prompt does not
        gold_c = re.sub(r"\b(" + "|".join(mapping) + r")\b",
                        lambda m: mapping[m.group(1)], str(t["answer"]))
        if t["answer_type"] == "name_set":      # a SET: placeholder order is not content
            gold_c = ",".join(sorted(re.findall(r"<P\d+>", gold_c))) or gold_c
        key = hashlib.sha256(json.dumps([t["generator_id"], t["family_id"], canon, gold_c,
                                         t["answer_type"]]).encode()).hexdigest()
        groups[key].append({"id": t["task_id"], "renaming": mapping})
    found = [{"ids": [m["id"] for m in g], "canonical_sha256": k,
              "renamings": {m["id"]: m["renaming"] for m in g}}
             for k, g in sorted(groups.items()) if len(g) > 1]
    redundant = sum(len(g["ids"]) - 1 for g in found)
    formal_ids = {t["task_id"] for t in tasks} - set(EXCLUDED)
    f_red = sum(len([i for i in g["ids"] if i in formal_ids]) - 1 for g in found
                if any(i in formal_ids for i in g["ids"]))
    return {
        "schema": "aadistill.eval_set_independence/v1",
        "eval_set_id": SET_ID, "component": "fluid_reasoning_720", "unit": "dev_v1",
        "items_sha256": sha256_file(ROOT / FLUID),
        "rule": ("two static items are ONE independent item only when their prompt AND gold are "
                 "identical after replacing every person name from the generator's name pool "
                 "(tasks.common.PEOPLE) with a placeholder in order of first appearance -- a "
                 "bijective renaming that any reader can re-derive from the two prompts with "
                 "`renaming_canon`. Nothing is merged by similarity, and no item is removed: "
                 "independent statistics count each group once, beside the per-item rate."),
        "groups": found,
        "not_counted_as_equivalent": [
            {"ids_from_audit": "letter.reverse L1 (3 items, same query 'vwxy' and answer 'yxwv')",
             "why": "their demonstrations differ, so the inductive evidence differs: related items, "
                    "not one problem; counted separately"},
            {"ids_from_audit": "code_breaker openings (45 items in 9 cells share a first message)",
             "why": "the hidden code differs; the episodes are different problems"}],
        "counts": {"items": len(tasks), "groups": len(found), "independent": len(tasks) - redundant,
                   "formal_items": len(formal_ids), "formal_independent": len(formal_ids) - f_red},
    }


# --- capability core --------------------------------------------------------------

def core_846(manifest: dict) -> dict:
    beh = jl(ROOT / manifest["components"]["stage3_capability_846"]["units"]
             ["eval_behavior_v0"]["file"])
    ids = {r["id"] for r in beh}
    xlam = sorted(i for i in ids if i.startswith(XLAM_PREFIX))
    assert len(xlam) == 7 and set(STAGE2_TRAIN_OVERLAPS) <= ids
    return {
        "schema": "aadistill.eval_set_core/v1",
        "eval_set_id": SET_ID, "component": "stage3_capability_846",
        "decided_by": "maintainer", "decision": AUTHORIZATION,
        "units": {
            "knowledge": {
                "in_formal_core": False, "status": "supplementary",
                "reason": ("TriviaQA has no recorded redistribution basis: it stays out of public git "
                           "and the formal core never depends on it. When its verified local copy is "
                           "staged it is measured and reported separately as a non-core unit.")},
            "eval_behavior_v0": {
                "in_formal_core": True,
                "excluded_from_core": {i: "Salesforce/xlam-function-calling-60k: access-gated; its "
                                          "access and redistribution terms are unconfirmed "
                                          "(present in the repository since 2026-07-27)"
                                       for i in xlam},
                "excluded_from_held_out": {i: "also in the stage2_v1 TRAIN split "
                                              "(audit/isolation.json)"
                                           for i in STAGE2_TRAIN_OVERLAPS},
                "duplicate_groups": [g for g in manifest["components"]["stage3_capability_846"]
                                     ["duplicate_groups"]
                                     if all(x.startswith("eval_behavior_v0:") for x in g)]},
            **{u: {"in_formal_core": True}
               for u in ("math_verified", "gsm8k", "multihop", "rag", "answerability_paired",
                         "safety_paired")},
        },
        "counts": {"behaviour_items": len(ids),
                   "behaviour_core": len(ids) - len(xlam),
                   "behaviour_core_held_out": len(ids - set(xlam) - set(STAGE2_TRAIN_OVERLAPS))},
        "rule": ("every item is still generated and its row reported; the masks decide which rows "
                 "enter the FORMAL numbers. Exact duplicates count once in independent statistics."),
    }


# --- conformance v2 ---------------------------------------------------------------

def _boxed(x: str) -> str:
    return f"The answer is \\boxed{{{x}}}."


def surface_variants(task: dict) -> list[tuple[str, str]]:
    """Renderings of the gold that v2 must accept, by answer type."""
    gold, typ = str(task["answer"]), task["answer_type"]
    out = []
    if typ == "int":
        out.append(("text_wrapper", f"\\text{{{gold}}}"))
        out.append(("mathrm_wrapper", f"\\mathrm{{{gold}}}"))
        if gold.startswith("-"):
            out.append(("unicode_minus", "−" + gold[1:]))
    elif typ == "int_list":
        out.append(("text_wrapper", f"\\text{{{gold}}}"))
        if "-" in gold:
            out.append(("unicode_minus", gold.replace("-", "−")))
    elif typ == "word":
        out.append(("mathrm_wrapper", f"\\mathrm{{{gold}}}"))
        if len(gold) > 1 and gold.isalpha():
            out.append(("letter_spaced", " ".join(gold)))
    elif typ == "token_seq":
        out.append(("mathrm_each", " ".join(f"\\mathrm{{{w}}}" for w in gold.split())))
    elif typ == "grid":
        rows = [r.strip() for r in re.split(r"[/\n]", gold) if r.strip()]
        out.append(("commas_between_rows", ", ".join(r.replace(" ", "") for r in rows)))
    elif typ == "name":
        out.append(("with_parenthetical", f"{gold} (my answer)"))
        out.append(("textbf_wrapper", f"\\textbf{{{gold}}}"))
    elif typ == "name_set":
        if gold.strip().lower() in ("none", "[]", ""):
            out += [("emptyset", "\\emptyset"), ("varnothing", "\\varnothing"),
                    ("nobody", "nobody"), ("no_knights", "No knights")]
        else:
            out.append(("text_wrapper", f"\\text{{{gold}}}"))
    return out


def negative_variants(task: dict) -> list[tuple[str, str]]:
    typ, gold = task["answer_type"], str(task["answer"])
    out = [("wrong", "\\text{zq no such answer}")]
    if typ == "name":
        out.append(("two_names", f"{gold} or Zed"))
    if typ == "word" and gold.isalpha():
        out.append(("letter_spaced_extra", " ".join(gold + "q")))
    return out


def oracle_replies(task: dict) -> list[str] | None:
    """Action replies that solve the episode, formatted the way v1 rejects."""
    from stages.calibration_volume.tasks.interactive import ButtonsEnv, ProbeFunctionEnv
    env = task["env"]
    if env["kind"] == "probe_function":
        return [f"**ANSWER:** {ProbeFunctionEnv.f(env, env['target'])}"]
    if env["kind"] == "code":
        return ["GUESS: " + " ".join(env["code"])]
    if env["kind"] == "buttons":
        frontier, seen = [(env["start"], [])], {env["start"]}
        for _ in range(env["presses"]):
            nxt = []
            for v, path in frontier:
                for b in sorted(env["buttons"]):
                    w = ButtonsEnv.apply(env, b, v)
                    if w == env["goal"]:
                        return [f"PRESS: **{x}**" for x in path + [b]]
                    if w not in seen and abs(w) < 10**7:
                        seen.add(w)
                        nxt.append((w, path + [b]))
            frontier = nxt
    return None


def conformance_fluid_v2(tasks: list[dict]) -> tuple[list[dict], dict]:
    import shared.evaluation.eval_set_scoring as S
    rows, check = [], collections.Counter()
    for t in tasks:
        if t["interactive"]:
            replies = oracle_replies(t)
            if replies is None:
                check["no_oracle"] += 1
                continue
            for proto in ("v2", "v1"):
                rec = {"turns": [S.synthetic_turn(r) for r in replies], "end": "env_done",
                       "action_protocol": proto}
                exp = S.score_fluid(t, rec, "v2")
                rows.append({"unit": "dev_v1", "id": t["task_id"],
                             "variant": f"oracle_markdown_actions_generated_{proto}",
                             "input": rec, "expected": exp})
                check[f"episode_{proto}_{exp['correct']}"] += 1
            continue
        for name, text in [("gold", str(t["answer"])), *surface_variants(t)]:
            rec = {"turns": [S.synthetic_turn(_boxed(text))]}
            exp = S.score_fluid(t, rec, "v2")
            rows.append({"unit": "dev_v1", "id": t["task_id"], "variant": name, "input": rec,
                         "expected": exp})
            check[f"accept_{exp['correct']}"] += 1
            if name != "gold":
                check[f"v1_on_variant_{S.score_fluid(t, rec, 'v1')['correct']}"] += 1
        for name, text in negative_variants(t):
            rec = {"turns": [S.synthetic_turn(_boxed(text))]}
            exp = S.score_fluid(t, rec, "v2")
            rows.append({"unit": "dev_v1", "id": t["task_id"], "variant": name, "input": rec,
                         "expected": exp})
            check[f"reject_{not exp['correct']}"] += 1
        rec = {"turns": [S.synthetic_turn(_boxed(str(t["answer"])), terminated=False)]}
        rows.append({"unit": "dev_v1", "id": t["task_id"], "variant": "gold_unterminated",
                     "input": rec, "expected": S.score_fluid(t, rec, "v2")})
    return rows, dict(check)


def conformance_v2_from_v1(component: str, rel_v1: str, items_by_unit: dict) -> tuple[list[dict], dict]:
    """The v1 vectors under v2, plus gold renderings v2 must newly accept."""
    import shared.evaluation.eval_set_scoring as S
    rows, check = [], collections.Counter()
    for r in jl(ROOT / rel_v1):
        item = items_by_unit[r["unit"]][r["id"]]
        exp = S.ADAPTERS[component](r["unit"], item, r["input"], "v2")
        if exp.get("correct") != r["expected"].get("correct"):
            check["changed_from_v1"] += 1
        rows.append({**r, "expected": exp})
    for unit, items in items_by_unit.items():
        if component == "stage3_capability_846" and unit not in S.CAPABILITY_V2_UNITS:
            continue
        for iid, item in sorted(items.items())[:40]:
            gold = str(item.get("gsm8k_answer") or item.get("gold_answer") or item.get("gold") or "")
            if not gold or len(gold) > 60 or "\n" in gold or "{" in gold:
                continue                       # a wrapper around LaTeX is not a rendering
            for name, text in (("text_wrapper", f"\\text{{{gold}}}"),
                               ("mathbf_wrapper", f"\\mathbf{{{gold}}}"),
                               *((("unicode_minus", gold.replace("-", "−")),)
                                 if gold.startswith("-") else ())):
                rec = {"raw": f"reasoning</think>\n\nThe answer is \\boxed{{{text}}}.{S.EOS_TEXT}",
                       "think_preopened": True, "natural_termination": True,
                       "degeneration_triggered": False, "degeneration_kind": None}
                exp = S.ADAPTERS[component](unit, item, rec, "v2")
                plain = {**rec, "raw": rec["raw"].replace(text, gold)}
                base = S.ADAPTERS[component](unit, item, plain, "v2")
                if base.get("correct"):          # pin only where the plain gold is accepted
                    rows.append({"unit": unit, "id": iid, "variant": f"gold_{name}",
                                 "input": rec, "expected": exp})
                    check[f"{name}_accepted_{exp['correct']}"] += 1
    return rows, dict(check)


# --- rescoring relation ---------------------------------------------------------

def rescoring(tasks_by_id: dict, items_846: dict, items_150: dict) -> dict:
    import shared.evaluation.eval_set_scoring as S
    from stages.calibration_volume import score_battery as SB
    from stages.calibration_volume.tasks.interactive import last_action
    out = {"schema": "aadistill.eval_set_rescoring/v1", "eval_set_id": SET_ID,
           "rule": ("every historical generation available locally, re-scored under the frozen v1 "
                    "protocol and under v2, item by item; v2 must never reject what v1 accepted. "
                    "Historical results stay bound to v1; v2 numbers are new numbers."),
           "fluid_reasoning_720": {}, "stage3_capability_846": {}, "stage3_legacy_150": {}}
    for rel in HIST_FLUID:
        p = REPO / rel
        if not p.is_file():
            out["fluid_reasoning_720"][rel] = {"present": False}
            continue
        recs = jl(p)
        r1 = {r["task_id"]: r for r in SB.score(tasks_by_id, recs, "v1")}
        r2 = {r["task_id"]: r for r in SB.score(tasks_by_id, recs, "v2")}
        footprint = 0
        for rec in recs:
            if rec.get("interactive"):
                for turn in rec["turns"]:
                    kinds = ("query", "answer", "press", "guess")
                    if last_action(turn.get("answer") or "", kinds, "v1") != \
                            last_action(turn.get("answer") or "", kinds, "v2"):
                        footprint += 1
        out["fluid_reasoning_720"][rel] = {
            "sha256": sha256_file(p), "items": len(recs),
            "correct_v1": sum(r["correct"] for r in r1.values()),
            "correct_v2": sum(r["correct"] for r in r2.values()),
            "gained": sorted(t for t in r1 if r2[t]["correct"] and not r1[t]["correct"]),
            "lost": sorted(t for t in r1 if r1[t]["correct"] and not r2[t]["correct"]),
            "interactive_turns_whose_action_would_parse_differently_under_v2": footprint,
            "_interactive": ("episodes were generated under action protocol v1, so they replay "
                             "under v1 under either scorer; the count above is the protocol "
                             "change's footprint on these transcripts, not a rescoring")}
    for d in HIST_846_DIRS:
        entry = {}
        for unit in sorted(S.CAPABILITY_V2_UNITS - {"knowledge"}):
            g = d / f"{unit}.generations.jsonl"
            if not g.is_file():
                continue
            gains, losses, c1, c2, n = [], [], 0, 0, 0
            for rec in jl(g):
                item = items_846[unit].get(rec["id"])
                if item is None:
                    continue
                a = S.score_capability(unit, item, rec, "v1")["correct"]
                b = S.score_capability(unit, item, rec, "v2")["correct"]
                n += 1
                c1 += bool(a)
                c2 += bool(b)
                if b and not a:
                    gains.append(rec["id"])
                if a and not b:
                    losses.append(rec["id"])
            entry[unit] = {"sha256": sha256_file(g), "items": n, "correct_v1": c1,
                           "correct_v2": c2, "gained": gains, "lost": losses}
        if entry:
            out["stage3_capability_846"][str(d.relative_to(REPO))] = entry
    for g in HIST_150:
        if not g.is_file():
            continue
        gains, losses, c1, c2, n = [], [], 0, 0, 0
        for rec in jl(g):
            item = items_150[rec["id"]]
            a = S.score_legacy(item, {"raw": rec["raw"]}, "v1")["correct"]
            b = S.score_legacy(item, {"raw": rec["raw"]}, "v2")["correct"]
            n += 1
            c1 += bool(a)
            c2 += bool(b)
            gains += [rec["id"]] if b and not a else []
            losses += [rec["id"]] if a and not b else []
        out["stage3_legacy_150"][str(g.relative_to(REPO))] = {
            "sha256": sha256_file(g), "items": n, "correct_v1": c1, "correct_v2": c2,
            "gained": gains, "lost": losses}
    lost = [(k, u) for comp in ("stage3_capability_846",) for k, v in out[comp].items()
            for u, e in v.items() if e["lost"]]
    lost += [(k, "") for comp in ("fluid_reasoning_720", "stage3_legacy_150")
             for k, e in out[comp].items() if e.get("lost")]
    out["v2_rejects_something_v1_accepted"] = lost
    return out


# --- manifest + registry -----------------------------------------------------------

def file_ident(rel: str) -> dict:
    p = ROOT / rel
    return {"sha256": sha256_file(p), "bytes": p.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    if f"## {DECISION_HEADING}" not in (REPO / "logs/budget/decisions.md").read_text():
        raise SystemExit("no authorizing decision record; refusing")
    reg = ES.registry()
    entry = ER.entry(reg, SET_ID)
    if len(entry["identity_history"]) != 1:
        raise SystemExit(f"{SET_ID} already has {len(entry['identity_history'])} identities; "
                         "this finalization appends the second exactly once")
    m = ES.manifest(SET_ID)                      # verifies the current identity first
    tasks = jl(ROOT / FLUID)
    tasks_by_id = {t["task_id"]: t for t in tasks}
    items_846 = {u: {r["id"]: r for r in ES.items(SET_ID, "stage3_capability_846", u)}
                 for u, spec in m["components"]["stage3_capability_846"]["units"].items()
                 if not spec.get("withheld")}
    items_150 = {r["id"]: r for r in ES.items(SET_ID, "stage3_legacy_150", "items")}

    val = validity(tasks)
    ind = independence(tasks)
    core = core_846(m)
    conf_fluid, chk_fluid = conformance_fluid_v2(tasks)
    conf_846, chk_846 = conformance_v2_from_v1("stage3_capability_846",
                                               "conformance/stage3_capability_846.jsonl", items_846)
    conf_150, chk_150 = conformance_v2_from_v1("stage3_legacy_150",
                                               "conformance/stage3_legacy_150.jsonl",
                                               {"items": items_150})
    resc = rescoring(tasks_by_id, items_846, items_150)
    summary = {"validity": val["counts"], "independence": ind["counts"], "core": core["counts"],
               "conformance_fluid_v2": {"vectors": len(conf_fluid), **chk_fluid},
               "conformance_846_v2": {"vectors": len(conf_846), **chk_846},
               "conformance_150_v2": {"vectors": len(conf_150), **chk_150},
               "rescoring_losses": resc["v2_rejects_something_v1_accepted"]}
    print(json.dumps(summary, indent=1))
    if val["counts"]["formal"] != 714 or val["counts"]["formal_static"] != 570 \
            or val["counts"]["formal_interactive"] != 144:
        raise SystemExit("the formal set is not 714 = 570 + 144")
    if resc["v2_rejects_something_v1_accepted"]:
        raise SystemExit("v2 rejects an answer v1 accepted; not a superset")
    if chk_fluid.get("accept_False") or chk_fluid.get("reject_False") or chk_fluid.get("no_oracle"):
        raise SystemExit("a fluid v2 vector does not behave as declared")
    if not args.write:
        return 0

    new_files = {
        "fluid_reasoning_720/validity_v1.json": dumps(val),
        "fluid_reasoning_720/independence_v1.json": dumps(ind),
        "stage3_capability_846/core_v1.json": dumps(core),
        "conformance/fluid_reasoning_720.v2.jsonl": jsonl(conf_fluid),
        "conformance/stage3_capability_846.v2.jsonl": jsonl(conf_846),
        "conformance/stage3_legacy_150.v2.jsonl": jsonl(conf_150),
    }
    for rel, body in new_files.items():
        (ROOT / rel).write_text(body)
    (ROOT / "audit/rescoring_v1_v2.json").write_text(dumps(resc))

    files = dict(m["files"])
    for rel in new_files:
        files[rel] = file_ident(rel)
    comps = json.loads(json.dumps(m["components"]))
    f = comps["fluid_reasoning_720"]
    f["scorer"] = {
        "protocol_id": V2_FLUID, "formal_protocol": "v2", "historical_protocol_id": V1_FLUID,
        "adapter": f["scorer"]["adapter"],
        "rule": f["scorer"]["rule"] + (
            "; v2: the reply's boxed answer and the gold are canonicalized by "
            "tasks.common.CANON_V2 (Unicode minus, LaTeX wrappers, letter-spaced words, "
            "per-token \\mathrm{}, comma-separated grid rows, a trailing parenthetical after a "
            "name, \\emptyset / nobody / 'no knights' for an empty set), a superset of v1"),
        "conformance": {"v1": "conformance/fluid_reasoning_720.jsonl",
                        "v2": "conformance/fluid_reasoning_720.v2.jsonl"}}
    f["generation_protocol"]["action_protocol"] = (
        "v2 for formal measurement (tasks.interactive: markdown emphasis, code spans, $ and "
        "spaced/comma code digits read as the action); the environment and the replay use the "
        "SAME protocol, recorded per episode. Historical episodes are v1.")
    f["validity"] = {"file": "fluid_reasoning_720/validity_v1.json", **{
        k: val["counts"][k] for k in ("formal", "excluded", "formal_static", "formal_interactive")}}
    f["independence"] = {"file": "fluid_reasoning_720/independence_v1.json",
                         **ind["counts"]}
    f["independent_items"] = ind["counts"]["independent"]
    f["quality_review"] = {"audit": "audit/fluid_reasoning_720_quality.json",
                           "status": "DECIDED by the maintainer 2026-10-19: 6 items excluded by "
                                     "validity_v1, the rest valid and annotated"}
    c = comps["stage3_capability_846"]
    c["scorer"] = {**c["scorer"], "protocol_id": V2_CAP, "formal_protocol": "v2",
                   "historical_protocol_id": c["scorer"]["protocol_id"],
                   "v2_units": sorted(__import__("shared.evaluation.eval_set_scoring",
                                                 fromlist=["x"]).CAPABILITY_V2_UNITS),
                   "v2_rule": ("the v1 scorer applied to the reply after tasks.common.normalize_v2 "
                               "(Unicode minus, LaTeX wrappers and spacing, markdown emphasis); "
                               "safety_paired and eval_behavior_v0 keep v1"),
                   "conformance": {"v1": "conformance/stage3_capability_846.jsonl",
                                   "v2": "conformance/stage3_capability_846.v2.jsonl"}}
    c["core"] = {"file": "stage3_capability_846/core_v1.json", **core["counts"]}
    c["quality_review"] = {"status": "DECIDED by the maintainer 2026-10-19: TriviaQA outside the "
                                     "core (withheld, supplementary); xLAM prompts outside the core; "
                                     "4 Stage-2-train overlaps outside held-out correctness"}
    l = comps["stage3_legacy_150"]
    l["scorer"] = {**l["scorer"], "protocol_id": V2_LEGACY, "formal_protocol": "v2",
                   "historical_protocol_id": l["scorer"]["protocol_id"],
                   "v2_rule": "the v1 rule applied after tasks.common.normalize_v2",
                   "conformance": {"v1": "conformance/stage3_legacy_150.jsonl",
                                   "v2": "conformance/stage3_legacy_150.v2.jsonl"}}
    new_m = {**m, "registered_utc": m["registered_utc"],
             "frozen_utc": datetime.now(timezone.utc).date().isoformat(),
             "files": dict(sorted(files.items())), "components": comps,
             "formal_core": {
                 "rule": ("formal measurement generates EVERY item of every unit; the formal numbers "
                          "are computed on the core: fluid_reasoning_720 validity_v1 (714 items), "
                          "every capability unit except knowledge (TriviaQA, supplementary), "
                          "eval_behavior_v0 without the xLAM prompts (and without the four Stage-2 "
                          "train overlaps in held-out correctness), and stage3_legacy_150 only as "
                          "in-sample regression. A non-core unit or row is reported beside, never "
                          "inside, a formal number."),
                 "scoring": "each component's scorer.formal_protocol (v2)"},
             "identity_note": ("identity 2 of eval-set-1: identity 1's content files are carried "
                               "unchanged; the validity, independence and core files, the v2 "
                               "conformance vectors and this manifest are new")}
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(dumps(new_m))
    reg_path = ES.registry_path()
    reg = json.loads(reg_path.read_text())
    for e in reg["eval_sets"]:
        if e["eval_set_id"] != SET_ID:
            continue
        e["status"] = "frozen"
        e["identity_history"].append({
            "manifest_sha256": sha256_file(manifest_path),
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "authorization": AUTHORIZATION,
            "change": ("validity mask v1 (6 items excluded; formal 714 = 570 static + 144 "
                       "interactive), verified-equivalence independence groups, formal core "
                       "(TriviaQA supplementary, xLAM outside the core, Stage-2-train overlaps "
                       "outside held-out correctness), scorer protocol v2 with conformance "
                       "vectors and the v1/v2 rescoring record; FROZEN")})
        e["decisions_applied"] = [
            "fluid item defects: 6 excluded by validity_v1 (no item changed)",
            "generator defects: fixed as ordering@v2, graph_paths@v2, string_induction@v2, "
            "letter_analogy@v2 (registry.GENERATORS_V2); v1 output byte-identical",
            "independence: exact duplicates and verified renamings count once",
            "scorer: protocol v2 (fluid, capability answer units, legacy) with conformance; "
            "historical results stay on v1",
            "TriviaQA: withheld from git, outside the core, measured as supplementary when staged",
            "behaviour: duplicate pairs collapsed; 4 Stage-2-train overlaps outside held-out "
            "correctness; xLAM prompts outside the core",
            "frozen 2026-10-19"]
        e["open_decisions"] = [
            "Salesforce/xlam-function-calling-60k: confirm its access/redistribution terms; until "
            "then its 7 prompts stay outside the formal core (they remain in git history since "
            "2026-07-27; removing them from history would be a separate maintainer decision)",
            "TriviaQA: a redistribution basis would let it enter a future set's core",
            "enable 'Require review from Code Owners' and the eval-set-integrity status check on "
            "main (repository settings; see data/eval-set/README.md)"]
    reg_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    problems = ER.verify(reg_path, SET_ID)
    print("verify:", problems or "intact")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

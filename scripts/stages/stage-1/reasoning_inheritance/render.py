#!/usr/bin/env python3
"""`analysis.py`'s JSON as the report's tables ($0): every unit's correct and usable per group and mode, the structural
contrasts with their intervals, retention of T-correct items, failure types and the state measurements.

    .venv/bin/python scripts/stages/stage-1/reasoning_inheritance/render.py \\
        logs/stages/stage-1/reasoning_inheritance/analyses/structural_ablation.json > <md>
"""
from __future__ import annotations

import json
import sys

GROUPS, MODES = ("T", "D", "W", "DW"), ("H-T", "H-N")
AXES = (("task_contained_deduction", "task-contained deduction"), ("inductive_rule_discovery", "inductive rule discovery"),
        ("compositional_generalization", "compositional generalization"),
        ("abstract_analogical", "abstract / analogical (sub-category)"))
UNITS = (("math_verified", "math (verified)", "formal"), ("gsm8k", "GSM8K", "formal"),
         ("multihop", "multi-hop reading (evidence given)", "formal"), ("rag", "RAG reading (evidence given)", "formal"),
         ("answerability_paired", "answerability pairs", "formal"), ("safety_paired", "safety pairs", "formal"),
         ("knowledge", "TriviaQA (supplementary, closed-book)", "outside the formal core"))


def frac(k, n):
    return "--" if n in (None, 0) or k is None else f"{k}/{n} ({100 * k / n:.1f}%)"


def g(res, grp, mode):
    return res["groups"].get(f"{grp}_{mode}") or {"status": "not_run"}


def unit_table(res: dict, mode: str) -> list[str]:
    out = [f"| unit ({mode}) | " + " | ".join(f"{x} correct | {x} usable" for x in GROUPS) + " |",
           "| --- |" + " ---: | ---: |" * len(GROUPS)]
    rows = []
    fl = {x: (g(res, x, mode).get("fluid") or {}) for x in GROUPS}

    def row(label, cells):
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    row("fluid formal core (714)", [c for x in GROUPS for c in
        (frac(fl[x].get("formal_core", {}).get("correct"), fl[x].get("formal_core", {}).get("n")), "")])
    row("static fluid (570)", [c for x in GROUPS for c in
        (frac(fl[x].get("static", {}).get("correct"), fl[x].get("static", {}).get("n")),
         frac(fl[x].get("static", {}).get("usable"), fl[x].get("static", {}).get("n")))])
    for ax, lab in AXES:
        row(f"-- {lab}", [c for x in GROUPS for c in
            (frac((fl[x].get("axes", {}).get(ax) or {}).get("correct"), (fl[x].get("axes", {}).get(ax) or {}).get("n")),
             frac((fl[x].get("axes", {}).get(ax) or {}).get("usable"), (fl[x].get("axes", {}).get(ax) or {}).get("n")))])
    row("interactive adaptation (144 episodes: success / all turns usable)", [c for x in GROUPS for c in
        (frac(fl[x].get("interactive", {}).get("success"), fl[x].get("interactive", {}).get("n")),
         frac(fl[x].get("interactive", {}).get("all_turns_usable"), fl[x].get("interactive", {}).get("n")))])
    for u, lab, role in UNITS:
        cells = []
        for x in GROUPS:
            e = (g(res, x, mode).get("units") or {}).get(u) or {}
            if u.endswith("_paired"):
                cells += [frac(e.get("pairs_correct"), e.get("pairs")), frac(e.get("pairs_both_usable"), e.get("pairs"))]
            else:
                cells += [frac(e.get("correct_rows"), e.get("n")), frac(e.get("usable"), e.get("n"))]
        row(f"{lab}{'' if role == 'formal' else ' *'}", cells)
    for grp_name, lab in (("tool_calling", "tool calling (exact call match)"), ("instruction", "instruction (format_ok)"),
                          ("short_realtime", "short realtime (format_ok)")):
        cells = []
        for x in GROUPS:
            e = ((g(res, x, mode).get("units") or {}).get("eval_behavior_v0") or {}).get(grp_name) or {}
            cells += [frac(e.get("format_ok") if grp_name != "tool_calling" else e.get("tool_call_exact"), e.get("n")),
                      frac(e.get("usable"), e.get("n"))]
        row(lab, cells)
    cells = []
    for x in GROUPS:
        e = (g(res, x, mode).get("units") or {}).get("legacy_150") or {}
        cells += [frac(e.get("correct"), e.get("n")), ""]
    row("stage3_legacy_150 (in-sample regression, not generalization)", cells)
    return out + rows + ["", "\\* outside the formal core (supplementary)."]


def contrast_table(res: dict, mode: str) -> list[str]:
    keys = ("depth_alone (T-D)", "non_depth_alone (T-W)", "depth_given_non_depth (W-DW)",
            "non_depth_given_depth (D-DW)", "interaction (D+W-T-DW)")
    out = [f"| {mode} | n | T | D | W | DW | " + " | ".join(keys) + " |", "| --- |" + " ---: |" * (5 + len(keys))]
    for unit, c in sorted((res.get("contrasts") or {}).get(mode, {}).items()):
        if "accuracy" not in c:
            missing = [x for x in GROUPS if x not in c.get("groups_present", [])]
            out.append(f"| {unit} | not yet measured for {', '.join(missing)} |" + " |" * (4 + len(keys)))
            continue
        acc = c["accuracy"]
        cells = [f"{100 * c[k]['estimate']:+.1f} [{100 * c[k]['ci95'][0]:+.1f}, {100 * c[k]['ci95'][1]:+.1f}]"
                 for k in keys]
        out.append(f"| {unit} | {c['n_items']} | " + " | ".join(f"{100 * acc[x]:.1f}" for x in GROUPS) + " | "
                   + " | ".join(cells) + " |")
    return out + ["", "Percentage points; 95% cluster-bootstrap intervals (fluid: whole rule families resampled, not "
                      "the registered independence groups; both are compared below)."]


def retention_table(res: dict, mode: str) -> list[str]:
    out = [f"| {mode}: of T's correct items | n | D keeps | W keeps | DW keeps |", "| --- | ---: | ---: | ---: | ---: |"]
    for unit, r in sorted((res.get("retention") or {}).get(mode, {}).items()):
        cells = [(f"{r[x]['kept']} ({100 * r[x]['estimate']:.1f}% [{100 * r[x]['ci95'][0]:.1f}, {100 * r[x]['ci95'][1]:.1f}])"
                  if x in r else "--") for x in ("D", "W", "DW")]
        out.append(f"| {unit} | {r['n_teacher_correct']} | " + " | ".join(cells) + " |")
    return out


FAIL = ("answered_wrong", "no_extractable_answer", "protocol_invalid", "degeneration", "context_limit",
        "not_terminated")
FAIL_HEAD = ("answer read, wrong", "no answer the scorer can read", "protocol error", "degeneration stop",
             "context limit", "other non-EOS stop")


def failure_table(res: dict) -> list[str]:
    out = ["| static fluid, T-correct items the group lost | lost | " + " | ".join(FAIL_HEAD) + " |",
           "| --- | ---: |" + " ---: |" * len(FAIL)]
    for mode, by in sorted((res.get("failure_types") or {}).items()):
        for x, f in sorted(by.items()):
            out.append(f"| {x} {mode} | {f['teacher_correct_lost']} | " +
                       " | ".join(str(f["lost_by_type"].get(t, 0)) for t in FAIL) + " |")
    return out


def outcome_table(res: dict) -> list[str]:
    """Every formal row of the static fluid unit and of each capability unit, by what it ended as (T and D: W and DW
    end in a degeneration stop or the context limit on all but a handful of rows; their counts are in the JSON)."""
    out = ["| unit | group, mode | n | correct | " + " | ".join(FAIL_HEAD) + " |",
           "| --- | --- | ---: | ---: |" + " ---: |" * len(FAIL)]
    labels = (("static fluid (570)", None),) + tuple((lab, u) for u, lab, _ in UNITS)
    for lab, u in labels:
        for x in ("T", "D"):
            for mode in MODES:
                e = g(res, x, mode)
                oc = (((e.get("fluid") or {}).get("static") or {}).get("outcomes") if u is None else
                      (((e.get("units") or {}).get(u) or {}).get("outcomes_by_row")))
                if not oc:
                    continue
                n = sum(oc.values())
                out.append(f"| {lab} | {x} {mode} | {n} | {oc.get('correct', 0)} | " +
                           " | ".join(str(oc.get(t, 0)) for t in FAIL) + " |")
    return out + ["", "Rows, not pairs, for the paired units. \"Answer read, wrong\" includes the behaviour units' wrong "
                      "behaviours (echoed the passage, answered an unanswerable question, refused an answerable or "
                      "benign one, complied with an unsafe one). Descriptive; the formal score is the registered "
                      "scorer's and unchanged."]


def integrity_table(res: dict) -> list[str]:
    out = ["| group, mode | degeneration stops (turns) | saved ids = recorded length | stopped at a check | "
           "inside the allowance |", "| --- | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        for x in GROUPS:
            c = g(res, x, mode).get("stop_records") or {}
            if c:
                out.append(f"| {x} {mode} | {c.get('degeneration_stops', 0)} | {c.get('ids_equal_recorded_length', 0)} | "
                           f"{c.get('at_a_check', 0)} | {c.get('inside_allowance', 0)} |")
    return out


def registered_grouping_table(res: dict) -> list[str]:
    """T-D and D's retention on the fluid axes under the two clusterings (analysis.py :: bootstrap_grouping)."""
    out = ["| fluid, T-D (pp) / D keeps of T-correct | mode | rule families (reported) | independence groups "
           "(registered) | retention, families | retention, independence groups |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        for unit, r in sorted(((res.get("contrasts_registered_grouping") or {}).get(mode) or {}).items()):
            fam_c = res["contrasts"][mode][unit]["depth_alone (T-D)"]
            reg_c = r["contrasts"]["depth_alone (T-D)"]
            fam_r, reg_r = res["retention"][mode][unit].get("D"), r["retention"].get("D")

            def iv(b, scale=100, sign=True):
                f = "+.1f" if sign else ".1f"
                return (f"{scale * b['estimate']:{f}} [{scale * b['ci95'][0]:{f}}, {scale * b['ci95'][1]:{f}}]"
                        if b else "--")
            out.append(f"| {unit} | {mode} | {iv(fam_c)} | {iv(reg_c)} | {iv(fam_r, sign=False)} | "
                       f"{iv(reg_r, sign=False)} |")
    return out


def length_table(res: dict) -> list[str]:
    """Descriptive, post hoc (analysis.kept_by_teacher_length): T-correct static items kept, by T's own length."""
    out, head = [], None
    for mode, by in sorted((res.get("failure_types") or {}).items()):
        for x, f in sorted(by.items()):
            bins = f.get("kept_by_teacher_length") or []
            if head is None and bins:
                lab = [f"T {b['teacher_tokens'][0]}-{b['teacher_tokens'][1]}" if b["teacher_tokens"][1] else
                       f"T >= {b['teacher_tokens'][0]}" for b in bins]
                head = ["| kept / T-correct, by T's tokens | " + " | ".join(lab) + " | group's degeneration stop p25 / p50 / p75 |",
                        "| --- |" + " ---: |" * (len(bins) + 1)]
            d = f.get("degeneration_stop_tokens") or {}
            out.append(f"| {x} {mode} | " + " | ".join(f"{b['kept']}/{b['teacher_correct']}" for b in bins) +
                       f" | {d.get('p25')} / {d.get('p50')} / {d.get('p75')} (n {d.get('n')}) |")
    return (head or []) + out


def behaviour_table(res: dict) -> list[str]:
    out = ["| static fluid (570) | natural termination (EOS) | usable | correct among usable (descriptive) | "
           "degeneration | context limit | tokens p50 | p90 | mean |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        for x in GROUPS:
            s = (g(res, x, mode).get("fluid") or {}).get("static") or {}
            if not s.get("n"):
                continue
            t = s["generated_tokens"]
            cu = (f"{s['correct_given_usable']}/{s['usable']}" if s.get("usable") else "--")
            nat = round(s["components"]["natural_termination"] * s["n"])
            out.append(f"| {x} {mode} | {nat} ({100 * s['components']['natural_termination']:.1f}%) | "
                       f"{s['usable']} ({100 * s['usable_rate']:.1f}%) | {cu} | "
                       f"{100 * s['degeneration_rate']:.1f}% | {100 * s['context_limit_rate']:.1f}% | {t['p50']} | "
                       f"{t['p90']} | {t['mean']} |")
    return out


def state_table(res: dict) -> list[str]:
    keys = ("state.teacher_kl.equal_domain_mean", "state.teacher_kl.worst_domain", "state.critical_token_kl")
    out = ["| group | KL equal-domain mean | KL worst domain | critical-token KL | NLL pooled |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for x, v in (res.get("state_eval") or {}).items():
        vals = v.get("values") or {}
        nll = (v.get("nll") or {})
        pooled = vals.get("state.nll.pooled_all_domains", nll.get("pooled") if isinstance(nll, dict) else None)
        out.append(f"| {x} | " + " | ".join(f"{vals.get(k, float('nan')):.4f}" for k in keys) +
                   f" | {pooled if pooled is None else f'{pooled:.4f}'} |")
    return out


def main(argv=None) -> int:
    res = json.loads(open((argv or sys.argv[1:])[0]).read())
    parts = []
    for mode in MODES:
        parts += [f"### eval-set-1, {mode}", ""] + unit_table(res, mode) + [""]
    parts += ["### Generation behaviour", ""] + behaviour_table(res) + [""]
    for mode in MODES:
        parts += [f"### Structural contrasts, {mode}", ""] + contrast_table(res, mode) + [""]
        parts += [f"### Retention, {mode}", ""] + retention_table(res, mode) + [""]
    parts += ["### Failure types", ""] + failure_table(res) + [""]
    parts += ["### What every row ended as (T and D)", ""] + outcome_table(res) + [""]
    parts += ["### Degeneration-stopped records as sealed", ""] + integrity_table(res) + [""]
    parts += ["### Fluid intervals under the registered independence grouping", ""] + registered_grouping_table(res) + [""]
    parts += ["### Kept by the teacher's length (descriptive, post hoc)", ""] + length_table(res) + [""]
    parts += ["### State evaluation (state_eval_hybrid@v3, against the root teacher)", ""] + state_table(res)
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

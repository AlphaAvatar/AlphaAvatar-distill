"""SOB1's registered decision rule (`design.json :: analysis`), as pure functions over per-item correctness; the
statistics are TPI1's (`teacher_parent_initialization.analysis`: group-paired differences, the sign-flip test, the
group bootstrap), only the multiplicity is new: three pairwise comparisons per mode.

* Family 1 (primary): the three thinking-mode pairs, Holm at family-wise alpha 0.05 with **m = 3 always** (the
  registered pairs, not the pairs that happen to have records): a pair that cannot be tested still counts in m.
* Family 2 (key secondary): the non-thinking pairs, Holm at 0.05 with m = 3, tested only if family 1 rejects all three
  (serial gatekeeping, so the six hypotheses keep a family-wise error of 0.05); otherwise descriptive.
* **A mode is decided only when the primary units are complete**: every registered arm's records cover every
  registered item under every registered seed. Anything less -- an arm, a seed or an item missing -- leaves EVERY pair
  of that mode ``descriptive_only`` (a pair whose own two arms are complete still gets its delta and interval; no
  test, never partially pooled). Verdicts are ``<X>_BETTER``, ``<Y>_BETTER`` or ``NOT_DISTINGUISHED`` (not
  equivalence). No ranking or pooled score is defined beyond the pairwise verdicts, and nothing here drops an arm.
* Per arm, descriptive only (`arm_report`): the initialization damage (untrained vs teacher), the recovery (recovered
  vs untrained) and the remaining gap (recovered vs teacher), all at 8,192 tokens, with group-bootstrap intervals.

The reader of the generations is TPI1's (`rescore`, `correctness`), applied per set. Phase 1's comparison
(`phase1_table`, ``analysis.py phase1-table --evidence <dir>``) reads the eight sealed sets once phase 1 is complete on
its files: per mode and arm, the teacher's and the arm's accuracy and the paired teacher-minus-arm difference with its
interval -- descriptive, eliminating nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_parent_initialization import analysis as TA  # noqa: E402

ALPHA = 0.05
TEST_SEED, BOOT_SEED = 20261119, 20261120


def holm(pvalues: dict[str, float], alpha: float = ALPHA, m: int | None = None) -> dict[str, dict]:
    """Holm's step-down procedure over ``m`` hypotheses (default: the p-values given): ``{key: {"p", "p_adjusted",
    "rejected"}}``; ties keep their key order. With ``m`` larger than the p-values given, the absent hypotheses count
    as never rejected (p = 1): the adjustment is never weakened by a comparison that could not be made."""
    m = len(pvalues) if m is None else int(m)
    if m < len(pvalues):
        raise ValueError(f"m = {m} for {len(pvalues)} p-values")
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    running, out, stop = 0.0, {}, False
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        stop = stop or pvalues[k] > alpha / (m - i)
        out[k] = {"p": pvalues[k], "p_adjusted": running, "rejected": not stop, "m": m}
    return out


def pair_comparison(y: dict, pair: tuple[str, str], mode: str, groups: dict, seeds: list, *, test: bool) -> dict:
    """One pair (X, Y) in one mode, ``y[arm][mode][seed][item]`` in {0, 1}: TPI1's statistic, interval and test with
    this design's registered generators."""
    x_arm, y_arm = pair
    ya, yb = y[x_arm][mode], y[y_arm][mode]
    items = [i for v in groups.values() for i in v]
    _keys, d, n = TA.pooled_group_differences(ya, yb, groups, seeds)
    acc = {a: sum(int(v[s][i]) for s in seeds for i in items) / (len(seeds) * len(items))
           for a, v in ((x_arm, ya), (y_arm, yb))}
    out = {"n_items": len(items), "n_seeds": len(seeds), "n_paired_records": int(n.sum()), "n_groups": len(groups),
           "delta": float(d.sum() / n.sum()), "accuracy": acc,
           "effect_interval": TA.LA.group_bootstrap(d, n, draws=TA.BOOT_DRAWS, seed=BOOT_SEED)}
    if test:
        out["test"] = TA.LA.sign_flip_test(d, draws=TA.TEST_DRAWS, seed=TEST_SEED)
        out["p_two_sided"] = out["test"]["p_two_sided"]
    return out


def verdict(pair: tuple[str, str], delta: float, rejected: bool) -> str:
    if not rejected or delta == 0:
        return "NOT_DISTINGUISHED"
    return f"{pair[0] if delta > 0 else pair[1]}_BETTER"


def completeness(y: dict, arms: list[str], mode: str, seeds: list, items: list[str]) -> dict:
    """Which registered arms lack a seed's records or an item in ``mode`` (empty ``missing``: complete)."""
    missing = {}
    for a in arms:
        rec = (y.get(a) or {}).get(mode)
        gaps = [s for s in seeds if rec is None or rec.get(s) is None or any(i not in rec[s] for i in items)]
        if gaps:
            missing[a] = gaps
    return {"complete": not missing, "missing": missing}


def decide(y: dict, pairs: list[tuple[str, str]], groups: dict, seeds: list, *, primary: str = "H-T",
           secondary: str = "H-N") -> dict:
    """The registered decision over ``pairs`` in both modes (module docstring)."""
    items = [i for v in groups.values() for i in v]
    arms = sorted({a for p in pairs for a in p})
    m = len(pairs)
    out: dict = {"_rule": {"alpha": ALPHA, "family_1": primary, "family_2": secondary, "m": m,
                           "multiplicity": "Holm within each mode over the registered pairs (m fixed); the second "
                                           "family only if the first rejects all",
                           "formal_requires": "every registered arm x every registered seed covers every item",
                           "seeds": [int(s) for s in seeds], "arms": arms, "registered_items": len(items),
                           "groups": len(groups)}}

    def family(mode: str, gate_open: bool) -> dict:
        comp = completeness(y, arms, mode, seeds, items)
        formal = gate_open and comp["complete"]
        res, pv = {"_completeness": comp}, {}
        for pair in pairs:
            key = f"{pair[0]} - {pair[1]}"
            if not all(TA.covers((y.get(a) or {}).get(mode), seeds, items) for a in pair):
                res[key] = {"status": "descriptive_only",
                            "why": "an arm's records do not cover every item in every seed: not computable"}
                continue
            res[key] = {"status": "formal" if formal else "descriptive_only",
                        **pair_comparison(y, pair, mode, groups, seeds, test=formal)}
            if formal:
                pv[key] = res[key]["p_two_sided"]
            elif not comp["complete"]:
                res[key]["why"] = (f"the mode's primary units are incomplete ({comp['missing']}): descriptive, "
                                   "no test")
        if formal:
            for key, h in holm(pv, m=m).items():
                pair = tuple(key.split(" - "))
                res[key].update(p_holm=h["p_adjusted"], holm_m=h["m"], rejected=h["rejected"],
                                verdict=verdict(pair, res[key]["delta"], h["rejected"]))
        return res

    out[primary] = family(primary, True)
    tested = [v for k, v in out[primary].items() if not k.startswith("_")]
    all_rejected = len(tested) == m and all(v.get("rejected") for v in tested)
    out[secondary] = family(secondary, all_rejected)
    if not all_rejected:
        for k, v in out[secondary].items():
            if not k.startswith("_"):
                v.setdefault("why", "not tested: the primary family did not reject every pair (serial gatekeeping)")
    return out


def _paired(a: dict, b: dict, groups: dict, seeds: list) -> dict:
    """``a`` against ``b`` (``[seed][item]``), pooled over ``seeds``: delta and the group-bootstrap interval."""
    items = [i for v in groups.values() for i in v]
    _k, d, n = TA.pooled_group_differences(a, b, groups, seeds)
    return {"delta": float(d.sum() / n.sum()), "n_paired_records": int(n.sum()),
            "accuracy": [sum(int(x[s][i]) for s in seeds for i in items) / (len(seeds) * len(items)) for x in (a, b)],
            "effect_interval": TA.LA.group_bootstrap(d, n, draws=TA.BOOT_DRAWS, seed=BOOT_SEED)}


def arm_report(pre: dict | None, post: dict | None, teacher: dict | None, groups: dict, seeds: list) -> dict:
    """One arm in one mode, descriptive only. ``pre`` / ``teacher``: ``{item: y}`` (one generation each); ``post``:
    ``{seed: {item: y}}``. Initialization damage = untrained - teacher; recovery = recovered - untrained (per seed and
    pooled); remaining gap = recovered - teacher. A part whose records are incomplete is reported ``not_available``."""
    items = [i for v in groups.values() for i in v]
    full = lambda r: r is not None and all(i in r for i in items)                      # noqa: E731
    one = [0]
    out = {"role": "descriptive only: initial KL, CE and generation scores eliminate nothing"}
    out["initialization_damage"] = (_paired({0: pre}, {0: teacher}, groups, one) if full(pre) and full(teacher)
                                    else "not_available")
    post_ok = post is not None and all(full(post.get(s)) for s in seeds)
    if post_ok and full(pre):
        out["recovery"] = {"pooled": _paired(post, {s: pre for s in seeds}, groups, seeds),
                           "per_seed": {str(s): _paired({s: post[s]}, {s: pre}, groups, [s]) for s in seeds}}
    else:
        out["recovery"] = "not_available"
    out["remaining_gap"] = (_paired(post, {s: teacher for s in seeds}, groups, seeds) if post_ok and full(teacher)
                            else "not_available")
    return out


# ---- phase 1: the teacher against each untrained arm, from the sealed records ------------------------------------------

TABLE_SCHEMA = "aadistill.stage1.single_operator_budget.phase1_comparison/v1"
TABLE_ROLE = ("descriptive only: the teacher minus each untrained arm, item by item, under one protocol; no arm is "
              "ranked, eliminated or kept by it -- all three go to recovery and are compared after it")


def _set_reading(gd, label: str, tmp) -> dict:
    """One set's per-item correctness (the registered scorer's rows): the static fluid formal core, the interactive
    episodes, every supporting unit's formal rows, and the headlines."""
    from stages.reasoning_inheritance import analysis as RA
    r = TA.rescore(gd, label, tmp)
    _rep, rows = RA.unit_rows(gd, label, tmp)
    units = {u: {i: int(bool(x.get("correct"))) for i, x in v.items() if x.get("formal") and "correct" in x}
             for u, v in rows.items() if u != RA.FLUID}
    return {"static": TA.correctness(r["static"]), "interactive": TA.correctness(r["interactive"]), "units": units,
            "headlines": r["headlines"], "digest": r["digest"]}


def _pp(x):
    return None if x is None else round(100.0 * float(x), 2)


def terminations(gd, primary) -> dict:
    """How each turn of one set ended, counted from its generations (AGENTS.md P18: reported apart, never a score):
    natural terminations, degeneration stops (the engine's check stopped the turn), context-limit hits
    (right-censored) and the rest, for the static fluid core (``primary``) and for the whole set."""
    import json
    from stages.teacher_parent_initialization import b_evidence as BE
    keys = ("turns", "natural_termination", "degeneration_stop", "context_limit_hit", "other_stop")
    core, every = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
    want = set(primary)
    for key in BE.registered_units():
        f = Path(gd) / f"{key}.jsonl"
        for line in f.read_text(encoding="utf-8").split("\n") if f.is_file() else ():
            if not line.strip():
                continue
            r = json.loads(line)
            for t in (r.get("turns") or []) if key == BE.FLUID else [r]:
                kind = ("natural_termination" if t.get("natural_termination") else
                        "degeneration_stop" if t.get("degeneration_triggered") else
                        "context_limit_hit" if t.get("context_limit_reached") else "other_stop")
                for c in ([core, every] if key == BE.FLUID and str(r.get("task_id")) in want else [every]):
                    c["turns"] += 1
                    c[kind] += 1
    return {"primary_static_fluid_core": core, "whole_set": every}


def _teacher_minus(teacher: dict, arm: dict, items: list[str], groups: dict) -> dict:
    """Teacher minus arm over ``items`` paired by item (one generation each), with the group-bootstrap interval."""
    res = _paired({0: teacher}, {0: arm}, groups, [0])
    lo, hi = res["effect_interval"]["ci95_percentile"]
    return {"n_items": len(items), "n_groups": len(groups), "teacher_accuracy_pct": _pp(res["accuracy"][0]),
            "arm_accuracy_pct": _pp(res["accuracy"][1]), "teacher_minus_arm_pp": _pp(res["delta"]),
            "ci95_pp": [_pp(lo), _pp(hi)], "interval": res["effect_interval"]["label"]}


def phase1_table(out, tmp) -> dict:
    """From the final sealed records of phase 1 (an evidence directory: the pod's or the dev box's copy), per mode
    and per arm: the teacher's and the arm's accuracy and the paired teacher-minus-arm difference in percentage
    points with its 95% group-bootstrap interval -- the static fluid formal core (the registered 570 items, grouped by
    the registered units of independence) first, then the interactive episodes and every supporting unit with a
    per-item correctness, each apart. Refused unless phase 1 is complete on its files (`completion.phase`)."""
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import completion as CP
    out, tmp = Path(out), Path(tmp)
    done = CP.phase(out)
    if not done["complete"]:
        raise ValueError(f"phase 1 is not complete on its files ({done['n_complete']} of {done['n_sets']} sets, "
                         f"inits {done['inits']['problems']}, modes {done['modes']}): no comparison is read")
    d = C.design()
    primary = TA.registered_items(d)
    groups = TA.independence_groups(primary, d)
    arms = [t for t in d["phase_1"]["evaluation_order"] if t != CP.TEACHER]
    doc = {"schema": TABLE_SCHEMA, "role": TABLE_ROLE, "context_tokens": d["evaluation"]["protocol"]["context"]["tokens"],
           "eval_set": d["evaluation"]["protocol"]["eval_set"], "modes": {}, "completion": {
               k: done[k] for k in ("complete", "n_complete", "n_sets")},
           "sets": {k: v["protocol_digest"] for k, v in done["sets"].items()}}
    for mode, sub in C.mode_dirs().items():
        read = {t: _set_reading(CP.ev_dir(out, t, mode) / "generations", CP.label_of(t), tmp / f"{t}_{mode}")
                for t in [CP.TEACHER, *arms]}
        if sorted(read[CP.TEACHER]["static"]) != sorted(primary):
            raise ValueError(f"{mode}: the teacher's static core is not the registered {len(primary)} items")
        rows = {}
        for a in arms:
            if sorted(read[a]["static"]) != sorted(primary):
                raise ValueError(f"{mode}: {a}'s static core is not the registered {len(primary)} items")
            inter = sorted(set(read[CP.TEACHER]["interactive"]) & set(read[a]["interactive"]))
            row = {"primary_static_fluid_core": _teacher_minus(read[CP.TEACHER]["static"], read[a]["static"],
                                                               primary, groups),
                   "interactive_episodes": _teacher_minus(read[CP.TEACHER]["interactive"], read[a]["interactive"],
                                                          inter, {i: [i] for i in inter}) if inter else None,
                   "supporting_units": {}}
            for u in sorted(read[CP.TEACHER]["units"]):
                t_u, a_u = read[CP.TEACHER]["units"][u], read[a]["units"].get(u, {})
                ids = sorted(set(t_u) & set(a_u))
                head = (read[CP.TEACHER]["headlines"].get(u.replace("__", "/", 1)) or {}).get("headline")
                if ids:
                    row["supporting_units"][u] = {
                        **_teacher_minus(t_u, a_u, ids, {i: [i] for i in ids}), "registered_headline": head,
                        "reading": ("per-item correctness, the unit's registered headline" if head == "correct" else
                                    f"per-item correctness; the unit's registered headline is {head} (in the "
                                    "headlines beside this table), not this")}
            rows[a] = row
        doc["modes"][mode] = {"dir": sub, "terminations": {t: terminations(CP.ev_dir(out, t, mode) / "generations",
                                                                         primary) for t in [CP.TEACHER, *arms]},
                              "teacher_headlines": read[CP.TEACHER]["headlines"],
                              "arm_headlines": {a: read[a]["headlines"] for a in arms}, "arms": rows,
                              "scored_digests": {t: v["digest"] for t, v in read.items()}}
    return doc


# ---- phase 2: the registered comparison, recomputed from the sealed post-recovery records -----------------------------

READOUT_SCHEMA = "aadistill.stage1.single_operator_budget.phase2_readout/v1"


def phase2_readout(outs: dict, phase1_out, tmp) -> dict:
    """The study's registered analysis from the twelve sealed post-recovery sets (``outs[arm]``: that arm's secured
    evidence, complete on its files -- `completion.study`): per mode and seed, each recovered model's per-item
    correctness on the static fluid formal core (the registered scorer, rescored here), then `decide` -- the three
    pairs pooled over both seeds, Holm with m = 3, the non-thinking family only after the thinking one rejects all
    three -- with each pair's per-seed deltas and whether the seeds agree in sign, and each arm's descriptive damage,
    recovery and remaining gap against phase 1's untrained arm and teacher (``phase1_out``; ``not_available`` without
    it). Terminations are counted apart, per set."""
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import completion as CP
    tmp = Path(tmp)
    d = C.design()
    primary = TA.registered_items(d)
    groups = TA.independence_groups(primary, d)
    seeds = [int(x) for x in d["training"]["seeds"]]
    pairs = [tuple(x) for x in d["analysis"]["comparisons"]]
    arms = sorted({a for pr in pairs for a in pr})
    y, digests, terms = {}, {}, {}
    for a in arms:
        for mode in C.mode_dirs():
            for sd in seeds:
                t = CP.train_target(a, sd)
                gd = CP.ev_dir(outs[a], t, mode) / "generations"
                r = _set_reading(gd, CP.label_of(t), tmp / f"{t}_{mode}")
                if sorted(r["static"]) != sorted(primary):
                    raise ValueError(f"{t}/{mode}: the static core is not the registered {len(primary)} items")
                y.setdefault(a, {}).setdefault(mode, {})[sd] = r["static"]
                digests[f"{t}/{mode}"] = r["digest"]
                terms[f"{t}/{mode}"] = terminations(gd, primary)
    decision = decide(y, pairs, groups, seeds)
    per_seed = {}
    for mode in C.mode_dirs():
        for x, z in pairs:
            key = f"{x} - {z}"
            rows = {str(sd): _paired({sd: y[x][mode][sd]}, {sd: y[z][mode][sd]}, groups, [sd]) for sd in seeds}
            signs = {(r["delta"] > 0) - (r["delta"] < 0) for r in rows.values()}
            per_seed.setdefault(mode, {})[key] = {"per_seed": rows, "seeds_agree_in_sign": len(signs) == 1}
    pre = {}
    if phase1_out is not None:
        for mode in C.mode_dirs():
            for t in [CP.TEACHER, *arms]:
                pre.setdefault(mode, {})[t] = _set_reading(CP.ev_dir(phase1_out, t, mode) / "generations",
                                                           CP.label_of(t), tmp / f"pre_{t}_{mode}")["static"]
    reports = {mode: {a: arm_report((pre.get(mode) or {}).get(a), y[a][mode], (pre.get(mode) or {}).get(CP.TEACHER),
                                    groups, seeds) for a in arms} for mode in C.mode_dirs()}
    acc = {mode: {a: {str(sd): _pp(sum(y[a][mode][sd].values()) / len(primary)) for sd in seeds} for a in arms}
           for mode in C.mode_dirs()}
    return {"schema": READOUT_SCHEMA, "role": "the registered decision (design.json :: analysis), recomputed from the "
                                             "sealed records; descriptive parts are labelled as such",
            "context_tokens": d["evaluation"]["protocol"]["context"]["tokens"], "n_items": len(primary),
            "n_groups": len(groups), "seeds": seeds, "decision": decision, "per_seed_deltas": per_seed,
            "accuracy_pct": acc, "per_arm_descriptive": reports, "terminations": terms, "scored_digests": digests,
            "phase1_evidence": None if phase1_out is None else str(phase1_out)}


DIAG_SCHEMA = "aadistill.stage1.single_operator_budget.pre_recovery_diagnostics/v1"


def pre_recovery_diagnostics(evidence) -> dict:
    """Each untrained arm's state evaluation against the teacher, read from its init record (the build session's
    ``state_evaluation`` on the state evaluation suite -- not eval-set-1): equal-domain mean, worst-domain and
    token-mean teacher KL, the critical-token KL, the reverse KL, the pooled NLL beside the teacher's, and top-1
    agreement (the share of scored positions whose argmax is the teacher's). Diagnostic only: it ranks, keeps and
    eliminates nothing, and none of it is a fluid-reasoning result."""
    import json
    from stages.single_operator_budget import common as C
    keys = {"teacher_kl_equal_domain_mean": "state.teacher_kl.equal_domain_mean",
            "teacher_kl_worst_domain": "state.teacher_kl.worst_domain",
            "teacher_kl_token_mean": "state.teacher_kl.token_mean", "critical_token_kl": "state.critical_token_kl",
            "reverse_kl_token_mean": "state.reverse_kl.token_mean", "top1_agreement": "state.top1_agreement",
            "nll_pooled": "state.nll.pooled_all_domains", "teacher_nll_pooled": "state.nll.teacher_reference_pooled"}
    rows, protocol = {}, set()
    for a in C.design()["phase_1"]["build_order"]:
        se = json.loads((Path(evidence) / a / "init_record.json").read_text())["state_evaluation"]
        v = se["values"]
        protocol.add(se.get("measurement_protocol_id"))
        rows[a] = {k: v.get(src) for k, src in keys.items()}
        rows[a]["teacher_kl_by_domain"] = {k.removeprefix("state.teacher_kl."): x for k, x in v.items()
                                           if k.startswith("state.teacher_kl.") and k.rsplit(".", 1)[-1]
                                           not in ("equal_domain_mean", "worst_domain", "token_mean")}
    return {"schema": DIAG_SCHEMA, "source": str(evidence),
            "suite": "the build session's state evaluation suite (the same scored positions for every arm), not "
                     "eval-set-1", "measurement_protocol_id": sorted(protocol, key=str), "arms": rows,
            "lowest_teacher_kl_first": sorted(rows, key=lambda a: rows[a]["teacher_kl_equal_domain_mean"]),
            "role": "diagnostic only, before recovery: ranks, keeps and eliminates nothing (design.json :: analysis)"}


ARM_WORDS = {"residual_width": "仅残差宽度 (width 1280)", "ffn_attention": "仅 FFN + 注意力 (FFN 3840, 16Q/8KV)",
             "depth_only": "仅深度 (16 layers)"}
MODE_WORDS = {"H-T": "思考模式", "H-N": "非思考模式"}


def render_table(doc: dict) -> str:
    """The comparison as Markdown: one table per mode, the primary core first."""
    out = [f"# SOB1 phase 1: teacher vs each untrained arm ({doc['context_tokens']:,} tokens, {doc['eval_set']})", "",
           f"_{doc['role']}_", ""]
    for mode, m in doc["modes"].items():
        out += [f"## {MODE_WORDS.get(mode, mode)} ({mode})", "",
                "| arm | items (groups) | teacher % | arm % | teacher − arm, pp | 95% interval, pp |",
                "| --- | ---: | ---: | ---: | ---: | --- |"]
        for a, r in m["arms"].items():
            p = r["primary_static_fluid_core"]
            out.append(f"| {ARM_WORDS.get(a, a)} | {p['n_items']} ({p['n_groups']}) | {p['teacher_accuracy_pct']} | "
                       f"{p['arm_accuracy_pct']} | {p['teacher_minus_arm_pp']} | {p['ci95_pp'][0]} to {p['ci95_pp'][1]} |")
        out += ["", "Static fluid formal core (primary items). Interactive episodes and supporting units, each apart:", "",
                "| arm | unit | items | teacher % | arm % | teacher − arm, pp | 95% interval, pp |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for a, r in m["arms"].items():
            extra = ([("interactive episodes", r["interactive_episodes"])] if r["interactive_episodes"] else []) + \
                list(r["supporting_units"].items())
            for u, p in extra:
                mark = " ¹" if p.get("registered_headline") not in (None, "correct") else ""
                out.append(f"| {ARM_WORDS.get(a, a)} | {u}{mark} | {p['n_items']} | {p['teacher_accuracy_pct']} | "
                           f"{p['arm_accuracy_pct']} | {p['teacher_minus_arm_pp']} | {p['ci95_pp'][0]} to "
                           f"{p['ci95_pp'][1]} |")
        out += ["", "¹ per-item correctness; the unit's registered headline is its pair accuracy, reported in each "
                "set's formal report and in comparison.json :: headlines. The behaviour set has no correctness "
                "headline and is not in this table. Intervals: 95% percentile bootstrap over the registered units "
                "of independence (the primary core) or over items, unadjusted for multiplicity.", ""]
        out += ["How the turns ended (counts; a degeneration stop is the engine's repetition check, a context-limit "
                "hit is right-censored; reported apart, never a score):", "",
                "| model | core turns | natural | degeneration stop | context limit | set turns | natural | "
                "degeneration stop | context limit |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for t, v in m.get("terminations", {}).items():
            c, w = v["primary_static_fluid_core"], v["whole_set"]
            out.append(f"| {ARM_WORDS.get(t, '教师 (teacher)' if t == 'teacher' else t)} | {c['turns']} | "
                       f"{c['natural_termination']} | {c['degeneration_stop']} | {c['context_limit_hit']} | {w['turns']} | "
                       f"{w['natural_termination']} | {w['degeneration_stop']} | {w['context_limit_hit']} |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    import argparse
    import json
    import tempfile
    ap = argparse.ArgumentParser(description="SOB1 analysis: phase 1's comparison table from sealed records")
    ap.add_argument("command", choices=("phase1-table", "diagnostics"))
    ap.add_argument("--evidence", required=True, help="an evidence directory (the pod's out or <scr>/store/out)")
    ap.add_argument("--write", default=None, help="a directory for comparison.json and comparison.md")
    a = ap.parse_args(argv)
    if a.command == "diagnostics":
        doc = pre_recovery_diagnostics(Path(a.evidence))
        if a.write:
            Path(a.write).mkdir(parents=True, exist_ok=True)
            (Path(a.write) / "pre_recovery_diagnostics.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(doc, indent=1))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        doc = phase1_table(Path(a.evidence), Path(tmp))
    md = render_table(doc)
    if a.write:
        w = Path(a.write)
        w.mkdir(parents=True, exist_ok=True)
        (w / "comparison.json").write_text(json.dumps(doc, indent=1) + "\n")
        (w / "comparison.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

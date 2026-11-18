#!/usr/bin/env python3
"""A_teacher_parent against B_packed_top1: the registered paired analysis ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/teacher_parent_initialization/analysis.py \\
        [--a-root <dir> [--a-root <later dir> ...]] [--out logs/stages/stage-1/teacher_parent_initialization/analyses/teacher_parent.json]

The rule is `design.json :: analysis`, registered before any A result existed; this file implements it and nothing
else decides.

Inputs. A: ``<a-root>/A_teacher_parent-s<seed>/evaluation_<thinking|non_thinking>/generations/`` holding the sealed
``<component>__<unit>.jsonl`` and ``.summary.json`` files (a later ``--a-root`` wins for a unit directory both hold).
B: the design's extracted A-MIX-6cat-sr evidence, the same layout under ``A-MIX-6cat-sr-s<seed>``. Both are rescored
here with the current registered scorers -- the fluid unit by ``score_battery.py --protocol v2``, every unit by
``eval_set_scoring.report`` -- so A and B are scored by one implementation; B's own formal reports are not read.

Pairing and identity. Each A record is paired with B's record of the same seed, mode and item. Before a mode's fluid
comparison can be formal, A's fluid unit must carry B's generation protocol (`b_evidence.protocol_check` against
``audit/b_evidence.json :: protocol_to_match``: nothing differs AND nothing is missing -- complete generations whose
summary or key fields are absent or empty are ``missing``, never consistent), every record must agree with it item by
item (`item_protocol`: the seed rule, the context, the think state, the stop id, the system message, B's paired prompt
length and seed), and it must cover every registered item in both seeds. A capability unit whose
protocol differs is reported and its headline flagged; it does not enter the decision, which is the fluid unit's.

Decision (registered). Per mode: delta = (1 / (2 x 570)) sum over both seeds and the 570 registered static items of
(y_A - y_B); per independence group g, d_g sums both seeds' differences of the group's items and n_g = 2 |g|, so an
item and its other-seed copy are one unit of independence. Two-sided group sign-flip test (B = 100,000,
default_rng(20261117)); 95% group-bootstrap effect interval (10,000, default_rng(20261118)). Fixed sequence: thinking
mode (H-T) at alpha 0.05; non-thinking (H-N) tested at alpha 0.05 only if H-T rejects, otherwise descriptive.
A_BETTER / A_WORSE / NOT_DISTINGUISHED (not equivalence). A mode A does not cover completely is descriptive_only, no
test, never a partial pooling.

Everything else is descriptive: each seed's delta and interval and whether the seeds agree in sign (two training
seeds bound what can be said about training randomness; the test's uncertainty is the items'), the interactive
episodes, the static core by reasoning axis, every unit's registered headline, and the terminations apart from
correctness (natural termination, degeneration stop, context-limit hit, answer-extraction failure, protocol
validity). eval-set-1 is development-exposed: nothing here is a never-seen blind result.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_inheritance import layer_merge_analysis as LA  # noqa: E402
from stages.teacher_parent_initialization import b_evidence as BE  # noqa: E402

DESIGN = BE.DESIGN
OUT = REPO / "logs/stages/stage-1/teacher_parent_initialization/analyses/teacher_parent.json"
B_EVIDENCE = BE.OUT
SCHEMA = "aadistill.stage1.teacher_parent_initialization.analysis/v1"
ARM = "A_teacher_parent"
TEST_SEED, BOOT_SEED = 20261117, 20261118
TEST_DRAWS, BOOT_DRAWS = 100_000, 10_000
ALPHA = 0.05
VERDICTS = ("A_BETTER", "A_WORSE", "NOT_DISTINGUISHED")


# ---- the registered statistics (pure, tested) ----------------------------------------------------------------------

def pooled_group_differences(y_a: dict, y_b: dict, groups: dict[str, list[str]],
                             seeds: list) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(group keys sorted, d_g, n_g): d_g = sum over ``seeds`` and the group's items of (y_A - y_B), n_g = |seeds| x
    |g|. ``y_a[seed][item]`` in {0, 1}; every item of every group must be present for every seed on both sides."""
    keys = sorted(groups)
    d, n = [], []
    for k in keys:
        tot = 0
        for s in seeds:
            missing = [i for i in groups[k] if i not in y_a[s] or i not in y_b[s]]
            if missing:
                raise ValueError(f"group {k}, seed {s}: items {missing[:3]} have no paired record")
            tot += sum(int(y_a[s][i]) - int(y_b[s][i]) for i in groups[k])
        d.append(tot)
        n.append(len(seeds) * len(groups[k]))
    return keys, np.asarray(d, dtype=np.int64), np.asarray(n, dtype=np.int64)


def verdict(delta: float, p: float, alpha: float = ALPHA) -> str:
    if p <= alpha:
        return "A_BETTER" if delta > 0 else "A_WORSE" if delta < 0 else "NOT_DISTINGUISHED"
    return "NOT_DISTINGUISHED"


def covers(y: dict | None, seeds: list, items: list[str]) -> bool:
    return y is not None and all(y.get(s) is not None and all(i in y[s] for i in items) for s in seeds)


def comparison(y_a: dict, y_b: dict, groups: dict, seeds: list, *, test: bool) -> dict:
    """One mode's pooled comparison: delta, accuracies, the effect interval, and the test when ``test``."""
    items = [i for v in groups.values() for i in v]
    keys, d, n = pooled_group_differences(y_a, y_b, groups, seeds)
    out = {"n_items": len(items), "n_seeds": len(seeds), "n_paired_records": int(n.sum()), "n_groups": len(keys),
           "delta": float(d.sum() / n.sum()),
           "accuracy": {"A": sum(int(y_a[s][i]) for s in seeds for i in items) / (len(seeds) * len(items)),
                        "B": sum(int(y_b[s][i]) for s in seeds for i in items) / (len(seeds) * len(items))},
           "effect_interval": LA.group_bootstrap(d, n, draws=BOOT_DRAWS, seed=BOOT_SEED)}
    if test:
        out["test"] = LA.sign_flip_test(d, draws=TEST_DRAWS, seed=TEST_SEED)
        out["p_two_sided"] = out["test"]["p_two_sided"]
        out["verdict"] = verdict(out["delta"], out["p_two_sided"])
    return out


def decide(y_a: dict, y_b: dict, groups: dict, seeds: list, *, primary: str, secondary: str,
           protocol_ok: dict | None = None) -> dict:
    """The registered decision. ``y_a[mode]`` / ``y_b[mode]``: {seed: {item: 0/1}} or None. B must cover the
    registered items; A's mode is formal only when it covers them in every seed and its protocol is B's. Fixed
    sequence: the secondary is tested only if the primary is formal and rejects at ALPHA."""
    items = [i for v in groups.values() for i in v]
    protocol_ok = protocol_ok or {}
    out: dict = {"_rule": {"primary": primary, "key_secondary": secondary, "alpha": ALPHA,
                           "multiplicity": "fixed sequence: the key secondary is tested only if the primary rejects",
                           "seeds": [int(s) for s in seeds], "registered_items": len(items),
                           "groups": len(groups)}}
    for m in (primary, secondary):
        if not covers(y_b.get(m), seeds, items):
            raise ValueError(f"{m}: B's records do not cover the registered items")
    for m in (primary, secondary):
        ya = y_a.get(m)
        present = 0 if ya is None else sum(1 for s in seeds if ya.get(s) is not None for i in items if i in ya[s])
        if not covers(ya, seeds, items):
            out[m] = {"status": "not_measured" if present == 0 else "descriptive_only",
                      "why": "A's records do not cover every registered item in both seeds",
                      "paired_records_present": present, "paired_records_registered": len(seeds) * len(items)}
            continue
        if protocol_ok.get(m) is False:
            out[m] = {"status": "descriptive_only", **comparison(ya, y_b[m], groups, seeds, test=False),
                      "why": "A's fluid unit is not shown to carry B's generation protocol (protocol_identity: "
                             "differs, missing or a record disagrees)"}
    prim = out.get(primary)
    if prim is None:
        out[primary] = {"status": "formal", "role": "primary",
                        **comparison(y_a[primary], y_b[primary], groups, seeds, test=True)}
        prim = out[primary]
    rejected = prim.get("status") == "formal" and prim["p_two_sided"] <= ALPHA
    if secondary not in out:
        if rejected:
            out[secondary] = {"status": "formal", "role": "key_secondary",
                              **comparison(y_a[secondary], y_b[secondary], groups, seeds, test=True)}
        else:
            out[secondary] = {"status": "descriptive_only", "role": "key_secondary",
                              **comparison(y_a[secondary], y_b[secondary], groups, seeds, test=False),
                              "why": ("not tested: the primary did not reject (fixed sequence)"
                                      if prim.get("status") == "formal" else
                                      "not tested: the primary comparison is not formal")}
    out[primary].setdefault("role", "primary")
    out[secondary].setdefault("role", "key_secondary")
    return out


def per_seed(y_a: dict, y_b: dict, groups: dict, seeds: list) -> dict:
    """Each seed alone (descriptive): delta, accuracies, the group-bootstrap interval; whether the seeds agree."""
    items = [i for v in groups.values() for i in v]
    out = {}
    for s in seeds:
        if not covers(y_a, [s], items):
            out[str(s)] = {"status": "not_measured"}
            continue
        keys, d, n = pooled_group_differences(y_a, y_b, groups, [s])
        out[str(s)] = {"delta": float(d.sum() / n.sum()),
                       "accuracy": {"A": sum(int(y_a[s][i]) for i in items) / len(items),
                                    "B": sum(int(y_b[s][i]) for i in items) / len(items)},
                       "effect_interval": LA.group_bootstrap(d, n, draws=BOOT_DRAWS, seed=BOOT_SEED)}
    signs = [int(np.sign(v["delta"])) for v in out.values() if "delta" in v]
    out["signs"] = signs
    #: two seeds agree when their deltas have the same nonzero sign; a zero delta agrees with nothing
    out["seeds_agree_in_sign"] = (bool(len(set(signs)) == 1 and signs[0] != 0) if len(signs) == len(seeds) else None)
    return out


# ---- the registered items (this design's pins) -----------------------------------------------------------------------

def registered_items(d: dict) -> list[str]:
    """`layer_merge_analysis.registered_primary_items`' rule, refused unless it gives this design's count and sha256."""
    import shared.evaluation.eval_set_scoring as S
    from shared import eval_sets as ES
    want = d["analysis"]["primary_items"]
    comp = ES.manifest("eval-set-1")["components"]["fluid_reasoning_720"]
    mask = S.unit_masks("eval-set-1", "fluid_reasoning_720", comp)["dev_v1"]
    tasks = {t["task_id"]: t for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1")}
    ids = sorted(i for i in set(tasks) - set(mask["exclude"]) if not tasks[i]["interactive"])
    got = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(ids) != int(want["n"]) or got != want["sha256"]:
        raise SystemExit(f"the registered primary items are {len(ids)} / {got[:12]}, the design pins "
                         f"{want['n']} / {want['sha256'][:12]}")
    return ids


def independence_groups(items: list[str], d: dict) -> dict[str, list[str]]:
    from shared import eval_sets as ES
    p = ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1").with_name("independence_v1.json")
    want = d["analysis"]["units_of_independence"]["sha256"]
    if hashlib.sha256(p.read_bytes()).hexdigest() != want:
        raise SystemExit(f"{p} is not the registered independence file")
    doc = json.loads(p.read_text())
    in_group = {i: k for k, grp in enumerate(doc["groups"]) for i in grp["ids"]}
    out = collections.defaultdict(list)
    for i in items:
        out[f"g{in_group[i]}" if i in in_group else i].append(i)
    return dict(out)


# ---- reading and rescoring the records -------------------------------------------------------------------------------

def a_gens(roots: list[Path], seed: int, mode_dir: str) -> Path | None:
    for r in reversed(roots):
        p = Path(r) / f"{ARM}-s{seed}" / mode_dir / "generations"
        if p.is_dir():
            return p
    return None


def rescore(gd: Path, label: str, tmp: Path) -> dict | None:
    """The formal-core fluid rows (static and interactive apart), every unit's headline, the scored directory's
    digest; None without a fluid unit."""
    from stages.reasoning_inheritance import analysis as RA
    fr = RA.fluid_rows(gd, label, tmp)
    if fr is None:
        return None
    _rep, rows = RA.unit_rows(gd, label, tmp)
    formal = rows.get(RA.FLUID, {})
    core = [fr[i] for i in sorted(fr) if (formal.get(i) or {}).get("formal")]
    return {"static": [r for r in core if not r["interactive"]], "interactive": [r for r in core if r["interactive"]],
            "headlines": LA.unit_headlines(gd, label), "digest": RA.dir_digest(gd)}


def correctness(rows: list[dict]) -> dict:
    return {r["task_id"]: int(bool(r["correct"])) for r in rows}


def by_axis(a_rows: list[dict] | None, b_rows: list[dict]) -> dict:
    """Static core per reasoning axis (the fluid items' ``axis``): n, correct and accuracy for A and B."""
    out = {}
    for axis in sorted({r.get("axis") for r in b_rows}, key=str):
        b = [r for r in b_rows if r.get("axis") == axis]
        row = {"n": len(b), "B_correct": sum(bool(r["correct"]) for r in b)}
        row["B_accuracy"] = round(row["B_correct"] / len(b), 4)
        if a_rows is not None:
            a = [r for r in a_rows if r.get("axis") == axis]
            row.update(A_correct=sum(bool(r["correct"]) for r in a),
                       A_accuracy=round(sum(bool(r["correct"]) for r in a) / len(a), 4) if a else None)
            row["delta"] = (round(row["A_accuracy"] - row["B_accuracy"], 4) if row["A_accuracy"] is not None else None)
        out[str(axis)] = row
    return out


def terminations(rows: list[dict]) -> dict:
    """Terminations apart from correctness (RI-S1's static summary: usable-rollout components, stop reasons, the
    outcome of every item that did not score, generation lengths)."""
    from stages.reasoning_inheritance import analysis as RA
    return RA.static_summary(rows)


def a_summaries(gd: Path) -> dict[str, dict]:
    return {p.name.removesuffix(".summary.json"): json.loads(p.read_text()) for p in sorted(gd.glob("*.summary.json"))}


def seed_of(base: int, key: str) -> int:
    """The decoding protocol's per-sample seed (``seed_rule``: int.from_bytes(sha256('<base>|<key>')[:8]) % 2**63)."""
    return int.from_bytes(hashlib.sha256(f"{base}|{key}".encode()).digest()[:8], "big") % 2 ** 63


#: what each record shows about its own generation protocol, and what only a unit's summary can show
ITEM_LEVEL = {
    "sampling_seed": "every turn's seed is the seed rule's for <namespace>|<item>|<turn>|0 (verifies the base, the key, "
                     "the namespace and one sample per item)",
    "context": "every turn's prompt tokens + generation allowance = the registered context (capability: context_len too)",
    "think_state": "every turn's rendered prompt has the mode's think state (the template's enable_thinking)",
    "stop_id": "every naturally terminated turn ends with a registered stop id",
    "system_message": "fluid: the record's system turn is the registered message; capability: the record's "
                      "system_source (the runner default, whose text is the summary's, or the item's own system "
                      "turn) is B's for the same item",
    "paired_with_B": "turn 0's prompt tokens and seed equal B's record of the same item (same rendered prompt length, "
                     "same seed)",
}
UNIT_LEVEL = ("decoding_protocol_id", "decoding_sha256", "sampling (temperature, top_p, top_k, min_p, penalties)",
              "chat_template_sha256", "tokenizer_sha256", "degeneration_check_every", "action_protocol (fluid)",
              "protocol / degeneration_stop / max_tokens_rule (capability)", "tasks_sha256 (fluid)",
              "engine and library versions (reported, not a condition)")


def item_protocol(gd: Path, gd_b: Path | None, key: str, ref: dict, think_state: str) -> dict:
    """`ITEM_LEVEL`, record by record, for one unit: per check the records/turns checked and those that disagree
    (first few). A check whose registered value is missing is ``unverifiable``, never passed."""
    f = gd / f"{key}.jsonl"
    if not f.is_file():
        return {"present": False}
    fluid = key == BE.FLUID
    base, ctx, stops = ref.get("seed_rule_base"), ref.get("resolved_context"), ref.get("stop_ids")
    namespace = ref.get("seed_namespace") if fluid else key
    b_rows = {}
    if gd_b is not None and (gd_b / f"{key}.jsonl").is_file():
        b_rows = {str(r.get("task_id") if fluid else r.get("id")): r for r in BE.jl(gd_b / f"{key}.jsonl")}
    out = {c: {"checked": 0, "disagree": 0, "first": []} for c in ITEM_LEVEL}
    need = {"sampling_seed": (base, namespace), "context": (ctx,), "stop_id": (stops,),
            "system_message": (ref.get("system_message"),) if fluid else (), "think_state": (think_state,)}
    unverifiable = sorted(c for c, vals in need.items() if any(BE.empty(v, c) for v in vals))

    def check(name, ok, where):
        if name in unverifiable:
            return
        out[name]["checked"] += 1
        if not ok:
            out[name]["disagree"] += 1
            if len(out[name]["first"]) < 3:
                out[name]["first"].append(where)
    for r in BE.jl(f):
        iid = str(r.get("task_id") if fluid else r.get("id"))
        turns = r.get("turns") if fluid else [{**r, "turn": 0, "seed": r.get("sampling_seed")}]
        for t in turns or []:
            w = f"{iid}#{t.get('turn')}"
            check("sampling_seed", t.get("seed") == seed_of(base, f"{namespace}|{iid}|{t.get('turn')}|0"), w)
            check("context", t.get("prompt_tokens", 0) + t.get("generation_allowance", -1) == ctx
                  and (fluid or r.get("context_len") == ctx), w)
            check("think_state", t.get("think_state") == think_state, w)
            if t.get("natural_termination"):
                ids = t.get("output_token_ids") or []
                check("stop_id", bool(ids) and ids[-1] in (stops or []), w)
        b = b_rows.get(iid)
        check("system_message", ((r.get("messages") or [{}])[0] == {"role": "system", "content": ref.get("system_message")})
              if fluid else (not BE.empty(r.get("system_source"), "system_source")
                             and (b is None or r.get("system_source") == b.get("system_source"))), iid)
        if b is not None and turns:
            bt = (b.get("turns") or [{}])[0] if fluid else {**b, "seed": b.get("sampling_seed")}
            check("paired_with_B", turns[0].get("prompt_tokens") == bt.get("prompt_tokens")
                  and turns[0].get("seed") == bt.get("seed"), iid)
    return {"present": True, "checks": out, "unverifiable": unverifiable,
            "agrees": not unverifiable and all(v["disagree"] == 0 for v in out.values())
            and all(out[c]["checked"] > 0 for c in ("sampling_seed", "context", "think_state", "system_message"))}


def protocol_identity(gd: Path, want: dict, gd_b: Path | None = None, think_state: str | None = None) -> dict:
    """A's sealed units against B's registered protocol: unit by unit (`b_evidence.protocol_check`: what differs and
    what is missing), record by record (`item_protocol`), and every registered unit's completeness. The fluid unit is
    B's protocol only when it is complete, nothing differs, nothing is missing and every record agrees; complete
    generations with a missing or empty protocol are ``missing``, never consistent."""
    units = BE.registered_units()
    comp = BE.completeness(gd, units)
    generated = tuple(k for k, v in comp.items() if v.get("present"))
    chk = BE.protocol_check(a_summaries(gd), want, generated=generated)
    per_item = {}
    for key in generated:
        kind = BE.kind_of(key)
        ref = (dict(want.get("fluid") or {}) if kind == "fluid" else
               {**(want.get("capability") or {}), **(want.get("capability_per_unit", {}).get(key) or {})})
        per_item[key] = item_protocol(gd, gd_b, key, ref, think_state) if think_state else {"present": True,
                                                                                          "not_checked": True}
    fluid = BE.FLUID
    f_diff = [p for p in chk["differs"] if p.startswith(fluid)]
    f_miss = [p for p in chk["missing"] if p.startswith(fluid)]
    f_item = per_item.get(fluid, {})
    item_disagrees = any(v["disagree"] for v in (f_item.get("checks") or {}).values())
    if not comp.get(fluid, {}).get("complete", False):
        status = "incomplete"
    elif f_diff or item_disagrees:
        status = "differs"
    elif f_miss or not f_item.get("agrees"):        #: unverifiable, unchecked or nothing to check: not shown equal
        status = "missing"
    else:
        status = "consistent"
    return {"fluid_protocol_is_B": status == "consistent", "fluid_status": status,
            "problems": chk["differs"], "missing": chk["missing"], "per_item": per_item,
            "incomplete_units": sorted(k for k, v in comp.items() if not v["complete"])}


def load_b_evidence() -> dict:
    if B_EVIDENCE.is_file():
        doc = json.loads(B_EVIDENCE.read_text())
        if doc.get("design_sha256") != BE.sha256_file(DESIGN):
            raise SystemExit(f"{B_EVIDENCE} was written under another design; rerun b_evidence.py --write")
        return doc
    return BE.audit()


def analyse(a_roots: list[Path], tmp: Path) -> dict:
    d = json.loads(DESIGN.read_text())
    bev = load_b_evidence()
    if bev.get("verdict") != "B_EVIDENCE_COMPLETE_AND_BOUND":
        raise SystemExit(f"B's evidence is not complete and bound: {bev.get('problems')}")
    seeds = BE.seeds(d)
    modes = d["evaluation"]["modes"]
    primary = next(m for m, v in modes.items() if v["role"] == "primary")
    secondary = next(m for m, v in modes.items() if v["role"] == "key secondary")
    items = registered_items(d)
    groups = independence_groups(items, d)
    inter_groups_cache: dict = {}
    y_a, y_b, protocol_ok, proto, scored, descr = {}, {}, {}, {}, {}, {}
    for m, mv in modes.items():
        y_a[m], y_b[m], proto[m], descr[m] = {}, {}, {}, {"by_seed": {}}
        ok = True
        inter_a, inter_b = {}, {}
        for s in seeds:
            b = rescore(BE.b_gens(d, s, m), f"B-s{s}-{m}", tmp)
            if b is None:
                raise SystemExit(f"B s{s} {m}: no fluid generations")
            y_b[m][s] = correctness(b["static"])
            inter_b[s] = correctness(b["interactive"])
            gd = a_gens(a_roots, s, mv["dir"])
            a = rescore(gd, f"A-s{s}-{m}", tmp) if gd is not None else None
            scored[f"s{s}:{m}"] = {"B": {"dir": str(BE.b_gens(d, s, m)).replace(str(Path.home()), "~"),
                                         "digest": b["digest"]},
                                   "A": (None if a is None else {"dir": str(gd), "digest": a["digest"]})}
            if a is None:
                y_a[m][s] = None
                proto[m][str(s)] = {"status": "not_measured"}
                ok = False
            else:
                y_a[m][s] = correctness(a["static"])
                inter_a[s] = correctness(a["interactive"])
                proto[m][str(s)] = protocol_identity(gd, bev["protocol_to_match"][m], BE.b_gens(d, s, m),
                                                     think_states()[mv["dir"]])
                ok = ok and proto[m][str(s)]["fluid_protocol_is_B"]
            descr[m]["by_seed"][str(s)] = {
                "static_by_axis": by_axis(None if a is None else a["static"], b["static"]),
                "terminations_static": {"B": terminations(b["static"]),
                                        "A": None if a is None else terminations(a["static"])},
                "interactive": {"B": _episodes(b["interactive"]), "A": None if a is None else _episodes(a["interactive"])},
                "headlines": {"B": b["headlines"], "A": None if a is None else _flag(a["headlines"], proto[m][str(s)])}}
        if all(y_a[m].get(s) is None for s in seeds):
            y_a[m] = None
        protocol_ok[m] = ok
        inter_items = sorted(set().union(*(set(v) for v in inter_b.values())))
        if inter_items:
            ig = inter_groups_cache.setdefault(tuple(inter_items), independence_groups(inter_items, d))
            if all(s in inter_a for s in seeds) and covers(inter_a, seeds, inter_items):
                descr[m]["interactive_pooled"] = comparison(inter_a, inter_b, ig, seeds, test=False)
            else:
                descr[m]["interactive_pooled"] = {"status": "not_measured", "n_items": len(inter_items)}
    decision = decide({m: y_a[m] for m in modes}, y_b, groups, seeds, primary=primary, secondary=secondary,
                      protocol_ok=protocol_ok)
    seeds_view = {m: (per_seed(y_a[m], y_b[m], groups, seeds) if y_a[m] is not None
                      else {str(s): {"status": "not_measured"} for s in seeds}) for m in modes}
    status = decision[primary]["status"]
    return {"schema": SCHEMA, "design_sha256": BE.sha256_file(DESIGN),
            "b_evidence": {"path": str(B_EVIDENCE.relative_to(REPO)), "verdict": bev["verdict"]},
            "registered": {"primary_items": len(items), "primary_items_sha256": d["analysis"]["primary_items"]["sha256"],
                           "groups": len(groups), "seeds": [int(s) for s in seeds]},
            "inputs": {"a_roots": [str(r) for r in a_roots], "scored": scored},
            "status": status if status == "formal" else ("not_measured" if y_a[primary] is None else status),
            "decision": decision, "per_seed": seeds_view, "protocol_identity": proto,
            "protocol_fields": {"item_level": ITEM_LEVEL, "unit_level_only": list(UNIT_LEVEL),
                                "rule": "item-level fields are checked on every A record (and against B's record of "
                                        "the same item); unit-level fields exist only in the units' summaries and are "
                                        "compared there (b_evidence.protocol_check); a missing one is never a match"},
            "descriptive": descr,
            "exposure": d["analysis"]["exposure"],
            "reading": _reading(decision, primary, secondary)}


def _episodes(rows: list[dict]) -> dict:
    from stages.reasoning_inheritance import analysis as RA
    return RA.episode_summary(rows)


def think_states() -> dict[str, str]:
    """The think state each evaluation directory's rendered prompts carry (the training plan's evaluations)."""
    plan = json.loads((REPO / "logs/stages/stage-1/teacher_parent_initialization/plans/training_plan.json").read_text())
    return {e["dir"]: e["think_state"] for u in plan["units"] for e in u["evaluations"]}


def _flag(headlines: dict, proto: dict) -> dict:
    """A's unit headlines, each marked when its unit's protocol is not shown to be B's (reported, never pooled)."""
    bad = {p.split(":", 1)[0].split("[", 1)[0] for p in proto.get("problems", []) + proto.get("missing", [])}
    bad |= {k for k, v in (proto.get("per_item") or {}).items() if v.get("present") and not v.get("not_checked")
            and not v.get("agrees")}
    return {u: {**v, "protocol_is_B": u.replace("/", "__") not in bad} for u, v in headlines.items()}


def _reading(decision: dict, primary: str, secondary: str) -> str:
    p = decision[primary]
    if p["status"] != "formal":
        return f"{primary}: {p['status']} ({p.get('why', '')}); no verdict"
    s = decision[secondary]
    sec = (f"{secondary}: {s['verdict']} (p {s['p_two_sided']:.3g})" if s["status"] == "formal"
           else f"{secondary}: descriptive ({s.get('why', '')})")
    return (f"{primary}: {p['verdict']}, delta {100 * p['delta']:+.2f} pp, p {p['p_two_sided']:.3g}, interval "
            f"[{100 * p['effect_interval']['ci95_percentile'][0]:+.2f}, "
            f"{100 * p['effect_interval']['ci95_percentile'][1]:+.2f}] pp; {sec}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a-root", action="append", default=[], type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        doc = analyse([Path(r).expanduser() for r in a.a_root], Path(tmp))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}: {doc['status']} -- {doc['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

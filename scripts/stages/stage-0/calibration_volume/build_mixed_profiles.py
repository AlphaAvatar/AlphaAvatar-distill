#!/usr/bin/env python3
"""The mixed calibration profile (2x, exactly 50/50) and its $0 feasibility audit.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/build_mixed_profiles.py \
        --design configs/stages/stage-0/calibration_volume/mixed_design_v2.json

From the design and the verified base pool, without a GPU or a model (the
superseded v1 design and its 2x/4x profiles were built by commit 6e2887ca):

1. ELIGIBILITY. The pool's fluid (five P10.2 axes) and supporting units, less
   every unit that shares a session id, source id or prompt key with what
   D4-Augmented trained or validated on, then checked against eval-set-1's
   held-out units by the frozen Stage-2 rules (`isolation_check.check_pool`);
   an R1 or R2 hit is excluded and listed.
2. SAMPLING (`sample`). Fluid first: an equal share per axis, split equally
   over a static axis's generators; the interactive axis is one cell over
   both generators, episodes whole; within a cell the rule families are
   interleaved (`family_order`), so a prefix spans as many families as it has
   units. Then supporting, balanced to the fluid half EXACTLY: the target is
   the fluid half's realized scored positions, four integer cell quotas; each
   cell takes the prefix of its order that fits its quota (skipping nothing),
   and the residual is closed by the fewest whole units, at most the design's
   `max_closing_units`, whose positions sum to it exactly, taken earliest from
   the cells' remaining orders interleaved (`close_exact`). No unit is cut,
   repeated or padded; if no such set exists the build writes the closest gap
   and stops (`Infeasible`): the ratio is never relaxed.
3. LABELS. `domain` is the half and `subtype` the axis or source group: DEPTH's
   objective is equal over domains then subtypes, so this is what makes it
   50/50 (FFN, RESIDUAL_WIDTH and ATTENTION pool over positions).
   `operator_view` counts both halves through the core's own position and
   aggregation contracts and `require_exact` refuses anything but equality.
4. COVERAGE of the profile beside historical D4's (and d1/d2's): UESP,
   prediction positions, forward tokens, padded forward slots under the
   baseline's micro-batch 3 / length_sorted_v1, units, episodes, rule
   families, by half, axis and generator.
5. COMPUTE from the Stage-0 search telemetry (d1, d2, d4: every expansion's
   operator seconds against its profile's forward tokens and its parent's
   level), priced for the single-profile beam tree, with the recovery and
   evaluation measured by H1-PA's D4-Augmented units.

Writes the items under `artifacts/` (local bytes), the profile document the
core's `load_profiles` accepts, and the audit. Generates nothing, trains
nothing, spends nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.initialization.calibration.profiles import (  # noqa: E402
    load_profiles, mixture_content_sha256)
from shared import eval_sets as ES  # noqa: E402
from stages.calibration_volume import build_volume_profiles as BVP  # noqa: E402

FLUID, SUPPORTING = "fluid", "supporting"
INTERACTIVE = "interactive_adaptation"
ITEM_FIELDS = BVP.ITEM_FIELDS + ("axis",)
S0 = "artifacts/stages/stage-0/calibration_volume"
#: Historical calibration the audit sets beside the mixed profiles (D4 Top-1 is
#: d4's). Their items name the axis as `domain` and the generator as `subtype`.
HISTORICAL = {"d1": "dose_d1_20261019_055515", "d2": "dose_d2_20261020_002929", "d4": "dose_d4_20261020_212310"}
HISTORICAL_PROFILES = "configs/stages/stage-0/calibration_volume/profiles_formal.json"
#: The single-profile beam tree (beam.delayed_prune@v1, width 6, warmup 1, four
#: operators): expansions per level, and the most of them that can be DEPTH --
#: one root, the three roots without it, then every survivor lacking it.
TREE = {1: 4, 2: 12, 3: 12, 4: 6}
DEPTH_MAX = {1: 1, 2: 3, 3: 6, 4: 6}
#: The historical two-profile tree, for the model's own check against telemetry.
TREE_2P = {1: 8, 2: 48, 3: 24, 4: 12}


def derive_seed(text: str) -> int:
    return struct.unpack(">I", hashlib.sha256(text.encode()).digest()[:4])[0] % 2 ** 31


def axis_of(it: dict) -> str:
    return it.get("axis") or it["domain"]


# --- eligibility ------------------------------------------------------------

def d4_augmented_members(design: dict) -> dict[str, set[str]]:
    """Session ids, source ids and prompt keys of everything D4-Augmented trained
    or validated on: E1's 2.96M prefix and 16 validation sequences, as E1's
    trainer reads them, and the fluid increment's admitted sessions."""
    from stages.reasoning_data import assets as A
    from stages.reasoning_data import h1pa_audit as H
    spec = design["exclusions"]["d4_augmented_members"]
    if str(H.E1_PACK.relative_to(REPO)) != spec["e1_pack"]:
        raise SystemExit(f"h1pa_audit reads E1 from {H.E1_PACK}, the design names {spec['e1_pack']}")
    n, val_idx, audit = H.e1_split()
    e1 = {m["session_id"] for i in list(range(n)) + list(val_idx) for m in audit[i]["sessions"]}
    fluid_path = REPO / spec["fluid_sessions"]
    fluid = list(A.jsonl(A.verified(fluid_path, spec["fluid_sessions_sha256"])))
    by_id = {s["id"]: s for s in A.corpus_v2_sessions() if s["id"] in e1}
    if set(by_id) != e1:
        raise SystemExit(f"{len(e1 - set(by_id))} E1 member sessions are not in corpus_v2")
    sessions = list(by_id.values()) + fluid
    return {"session": {s["id"] for s in sessions},
            "source": {s["id"].split("#")[0] for s in sessions} | {s["unit_id"] for s in fluid},
            "prompt": {ES.prompt_key(ES.prompt_text(s)) for s in sessions},
            "_counts": {"e1_training_and_validation_sessions": len(e1), "fluid_increment_sessions": len(fluid)}}


def check_items(units: dict[str, dict]) -> dict[str, dict]:
    """{unit_id: the item `isolation_check` and the prompt keys read}: a fluid
    unit is its calibration task (prompt, content hash, generator, family,
    seed), a supporting unit its corpus_v2 session."""
    from stages.reasoning_data import assets as A
    tasks = A.calibration_tasks()
    want = {u["items"][0]["session_id"]: uid for uid, u in units.items() if u["domain"] == SUPPORTING}
    sessions = {s["id"]: s for s in A.corpus_v2_sessions() if s["id"] in want}
    out = {}
    for uid, u in units.items():
        if u["domain"] == SUPPORTING:
            s = sessions[u["items"][0]["session_id"]]
            out[uid] = {"unit_id": uid, "messages": s["messages"], "source": s["source"]}
        else:
            out[uid] = {**tasks[u["items"][0]["task_id"]], "unit_id": uid}
    return out


def eligible(design: dict, units: dict[str, dict]) -> tuple[dict, dict]:
    """The design's fluid and supporting units less D4-Augmented's members and
    any eval-set-1 R1/R2 hit; the census says what left and why."""
    from stages.reasoning_data import isolation_check as IC
    gens = {g: ax for ax, gs in design["halves"][FLUID]["axes"].items() for g in gs}
    subs = set(design["halves"][SUPPORTING]["subtypes"])
    wanted = {uid: u for uid, u in units.items()
              if (u["domain"] == SUPPORTING and u["subtype"] in subs) or u["subtype"] in gens}
    bad_axis = sorted({(u["domain"], u["subtype"]) for u in wanted.values()
                       if u["domain"] != SUPPORTING and gens[u["subtype"]] != u["domain"]})
    if bad_axis:
        raise SystemExit(f"generators on another axis than the design says: {bad_axis}")
    members = d4_augmented_members(design)
    checks = check_items(wanted)
    census = {"members": members.pop("_counts"), "excluded_d4_augmented": Counter(), "by_key": Counter()}
    kept = {}
    for uid, u in wanted.items():
        keys = {"session": {it["session_id"] for it in u["items"]},
                "source": {it["session_id"].split("#")[0] for it in u["items"]} | {uid}}
        #: An episode's opening words do not identify it (shared.eval_sets.KEYS):
        #: an interactive unit is matched by its ids only.
        if u["domain"] != INTERACTIVE:
            keys["prompt"] = {ES.prompt_key(ES.prompt_text(checks[uid]))}
        hit = [k for k, v in keys.items() if v & members[k]]
        if hit:
            census["excluded_d4_augmented"][f"{u['domain']}/{u['subtype']}"] += 1
            for k in hit:
                census["by_key"][k] += 1
        else:
            kept[uid] = u
    report = IC.check_pool([checks[uid] for uid in sorted(kept)], historical=True)
    res = report["results"]
    if not res["R1_held_out_keys"]["pass"]:
        raise SystemExit(f"eligible units share a key with a held-out eval-set-1 unit: "
                         f"{res['R1_held_out_keys']['held_out_overlap']}")
    flagged = set(res["R2_near_duplicate"].get("flagged_ids") or [])
    kept = {uid: u for uid, u in kept.items() if uid not in flagged}
    census["excluded_eval_set_r2"] = sorted(flagged)
    trim = lambda v: {"n": len(v), "first": v[:3]} if isinstance(v, list) and len(v) > 20 else v  # noqa: E731
    census["isolation"] = {r: {k: trim(v) for k, v in x.items() if k not in ("by_unit", "pairs", "flagged_items")}
                           for r, x in res.items()}
    census["isolation"]["R5_shared_generator_exposure"]["_reading"] = (
        "a Stage-2 TRAINING rule (seeds in the training range 3,000,000-4,000,000); these are calibration units "
        "drawn in the calibration range 2,000,000-3,000,000 from calibration-split families, which R3 checks")
    census["isolation_report"] = {"eval_set": report["eval_set"], "n_items": report["n_items"],
                                  "establishes": report["establishes"], "not_established": report["not_established"]}
    census["excluded_d4_augmented"] = dict(sorted(census["excluded_d4_augmented"].items()))
    census["by_key"] = dict(census["by_key"])
    return kept, census


# --- sampling ---------------------------------------------------------------

def cellmap(design: dict, units: dict[str, dict]) -> dict[tuple, list[str]]:
    """(half, axis or subtype, generator or '*') -> its eligible units."""
    out: dict[tuple, list[str]] = defaultdict(list)
    for uid, u in sorted(units.items()):
        if u["domain"] == SUPPORTING:
            out[(SUPPORTING, u["subtype"], "*")].append(uid)
        elif u["domain"] == INTERACTIVE:
            out[(FLUID, INTERACTIVE, "*")].append(uid)
        else:
            out[(FLUID, u["domain"], u["subtype"])].append(uid)
    want = [(FLUID, ax, "*" if ax == INTERACTIVE else g)
            for ax, gs in design["halves"][FLUID]["axes"].items() for g in gs]
    want += [(SUPPORTING, s, "*") for s in design["halves"][SUPPORTING]["subtypes"]]
    missing = sorted({c for c in want if not out.get(c)})
    if missing:
        raise SystemExit(f"design cells with no eligible unit: {missing}")
    return dict(out)


def family_order(cell: tuple, uids: list[str], units: dict, seed: int) -> list[str]:
    """The cell's units, its rule families interleaved: the families in a
    seed-keyed order, each family's units in their own seed-keyed order
    (`build_volume_profiles.order`), then one unit per family per round. A
    prefix of n units therefore spans min(n, families) families; nothing reads
    a length or a result. A cell whose units carry no family keeps the plain
    seed-keyed order."""
    fams = defaultdict(list)
    for uid in uids:
        fams[units[uid]["family"]].append(uid)
    if len(fams) <= 1:
        return BVP.order(cell, uids, seed)
    names = BVP.order(cell + ("families",), sorted(fams), seed)
    queues = {f: BVP.order(cell + (f,), fams[f], seed) for f in names}
    out = []
    while any(queues.values()):
        for f in names:
            if queues[f]:
                out.append(queues[f].pop(0))
    return out


def _take(cell: tuple, uids: list[str], units: dict, seed: int, quota: float, *,
          fit: bool = False) -> tuple[list[str], int, list[str]]:
    """A prefix of the cell's seed-keyed order (at least one unit) and the rest
    of the order. By default it stops once the quota is met; `fit` stops at the
    first unit that would carry the total past the quota, skipping nothing, so
    the prefix never exceeds it unless its first unit alone does."""
    order = family_order(cell, uids, units, seed)
    got, k = 0, 0
    for uid in order:
        s = units[uid]["scored"]
        if k and (got >= quota or (fit and got + s > quota)):
            break
        k += 1
        got += s
    return order[:k], got, order[k:]


def interleave(orders: list[list[str]]) -> list[str]:
    """Rank by rank across the orders, in the order given."""
    return [o[r] for r in range(max(map(len, orders), default=0)) for o in orders if r < len(o)]


def _first_exact(s: list[int], target: int, k: int, upto: int) -> tuple[int, ...] | None:
    """Indices i_1 < ... < i_k < upto with sum(s[i]) == target: the smallest i_k,
    then the smallest i_{k-1}, and so on; None when there are none."""
    if target <= 0:
        return None
    if k == 1:
        return next(((i,) for i in range(upto) if s[i] == target), None)
    if k == 2:
        first: dict[int, int] = {}
        for m in range(upto):
            i = first.get(target - s[m])
            if i is not None:
                return (i, m)
            first.setdefault(s[m], m)
        return None
    for m in range(k - 1, upto):
        rest = _first_exact(s, target - s[m], k - 1, m)
        if rest is not None:
            return rest + (m,)
    return None


def _closest(s: list[int], target: int) -> dict:
    """The smallest |gap| one unit and two units reach: the infeasibility evidence."""
    one = min((abs(x - target) for x in s), default=None)
    v, two = sorted(s), None
    i, j = 0, len(v) - 1
    while i < j:
        gap = v[i] + v[j] - target
        two = abs(gap) if two is None else min(two, abs(gap))
        if gap == 0:
            break
        i, j = (i + 1, j) if gap < 0 else (i, j - 1)
    return {"one_unit": one, "two_units": two}


class Infeasible(SystemExit):
    """Exact balance cannot be reached under the declared rule; `evidence` says by how much."""

    def __init__(self, evidence: dict):
        super().__init__(f"exact 50/50 is infeasible under the declared rule: {json.dumps(evidence)}")
        self.evidence = evidence


def close_exact(residual: int, sequence: list[str], sizes: Mapping[str, int], max_units: int) -> list[str]:
    """The fewest units of `sequence` (at most `max_units`) whose scored
    positions sum to exactly `residual`; among those, the set whose last member
    comes earliest in the sequence, then whose earlier members come earliest.
    Reads nothing but the sizes; raises `Infeasible` with the closest gaps."""
    if residual == 0:
        return []
    s = [sizes[u] for u in sequence]
    for k in range(1, max_units + 1):
        hit = _first_exact(s, residual, k, len(s))
        if hit is not None:
            return [sequence[i] for i in hit]
    raise Infeasible({"residual": residual, "max_units": max_units, "candidates": len(s),
                      "closest_gap": _closest(s, residual)})


def sample(design: dict, units: dict[str, dict], cells: dict[tuple, list[str]], seed: int,
           fluid_nominal: float) -> tuple[list[dict], dict]:
    """One dose's items, relabelled (domain = half, subtype = axis or source).

    Fluid first, each cell until its nominal quota is met. The supporting half
    is then balanced to the fluid half EXACTLY: the target is the fluid half's
    realized scored positions, split into integer cell quotas; each cell takes
    the prefix of its order that fits its quota; the residual is closed by
    `close_exact` over the cells' remaining orders, interleaved."""
    axes = design["halves"][FLUID]["axes"]
    by_axis = defaultdict(list)
    for c in cells:
        if c[0] == FLUID:
            by_axis[c[1]].append(c)
    census, taken = {}, []
    fluid_got = 0
    for ax in axes:
        for c in by_axis[ax]:
            quota = fluid_nominal / len(axes) / len(by_axis[ax])
            uids, got, _ = _take(c, cells[c], units, seed, quota)
            census["/".join(c)] = {"quota": round(quota, 1), "scored": got, "units": len(uids),
                                   "pool_units": len(cells[c])}
            taken += uids
            fluid_got += got
    subs = [(SUPPORTING, s, "*") for s in design["halves"][SUPPORTING]["subtypes"]]
    rests, prefix_got = [], 0
    for i, c in enumerate(subs):
        quota = fluid_got // len(subs) + (1 if i < fluid_got % len(subs) else 0)
        uids, got, rest = _take(c, cells[c], units, seed, quota, fit=True)
        census["/".join(c)] = {"quota": quota, "prefix_scored": got, "prefix_units": len(uids),
                               "pool_units": len(cells[c])}
        taken += uids
        rests.append(rest)
        prefix_got += got
    residual = fluid_got - prefix_got
    max_units = int(design["halves"][SUPPORTING]["balance"]["max_closing_units"])
    closing = close_exact(residual, interleave(rests), {u: units[u]["scored"] for r in rests for u in r},
                          max_units) if residual >= 0 else None
    if closing is None:
        raise Infeasible({"residual": residual, "reason": "the prefixes already exceed the target"})
    for c, rest in zip(subs, rests):
        mine = [u for u in closing if u in set(rest)]
        cell = census["/".join(c)]
        cell["closing"] = [{"unit_id": u, "scored": units[u]["scored"], "rank_after_prefix": rest.index(u)}
                           for u in mine]
        cell["scored"] = cell["prefix_scored"] + sum(units[u]["scored"] for u in mine)
        cell["units"] = cell["prefix_units"] + len(mine)
    taken += closing
    census["_balance"] = {"target": fluid_got, "prefixes": prefix_got, "residual": residual,
                          "closing_units": len(closing),
                          "supporting": prefix_got + sum(units[u]["scored"] for u in closing)}
    items = []
    for uid in taken:
        u = units[uid]
        for it in u["items"]:
            half = SUPPORTING if u["domain"] == SUPPORTING else FLUID
            items.append({**it, "domain": half, "subtype": u["subtype"] if half == SUPPORTING else u["domain"],
                          "axis": SUPPORTING if half == SUPPORTING else u["domain"]})
    return items, census


# --- coverage ---------------------------------------------------------------

def padded_slots(lengths: list[int], micro_batch: int) -> int:
    """Forward token slots under length_sorted_v1: sort ascending, chunk, pad each
    chunk to its longest member."""
    s = sorted(lengths)
    return sum(len(s[i:i + micro_batch]) * max(s[i:i + micro_batch]) for i in range(0, len(s), micro_batch))


def _block(items: list[dict], micro_batch: int) -> dict:
    units = defaultdict(list)
    for it in items:
        units[it["unit_id"]].append(it)
    fams = {(it.get("generator_id"), it.get("family_id")) for it in items if it.get("family_id")}
    return {"items": len(items), "units": len(units),
            "episodes": sum(1 for v in units.values() if axis_of(v[0]) == INTERACTIVE),
            "uesp": sum(len(it["tags"]["assistant"]) for it in items),
            "prediction_positions": sum(it["n_prediction_positions"] for it in items),
            "forward_tokens": sum(it["n_tokens"] for it in items),
            "padded_forward_slots": padded_slots([it["n_tokens"] for it in items], micro_batch),
            "rule_families": len(fams)}


def coverage(items: list[dict], *, micro_batch: int, pool_families: dict[str, set] | None = None) -> dict:
    total = _block(items, micro_batch)
    out = {"total": total, "by_half": {}, "by_axis": {}, "by_generator": {}}
    halves = defaultdict(list)
    axes, gens = defaultdict(list), defaultdict(list)
    for it in items:
        ax = axis_of(it)
        halves[SUPPORTING if ax == SUPPORTING else FLUID].append(it)
        axes[ax].append(it)
        gens[it.get("generator_id") or it["subtype"]].append(it)
    for name, group, dest in (("half", halves, "by_half"), ("axis", axes, "by_axis"), ("generator", gens, "by_generator")):
        for k, its in sorted(group.items()):
            b = _block(its, micro_batch)
            b["uesp_share"] = round(b["uesp"] / total["uesp"], 4)
            out[dest][k] = b
    fl = out["by_half"].get(FLUID)
    if fl:
        for ax, b in out["by_axis"].items():
            if ax != SUPPORTING:
                b["share_of_fluid"] = round(b["uesp"] / fl["uesp"], 4)
    if pool_families:
        for g, b in out["by_generator"].items():
            if g in pool_families:
                b["pool_rule_families"] = len(pool_families[g])
    out["episodes"] = sorted(
        ({"unit_id": uid, "generator": its[0].get("generator_id"), "family": its[0].get("family_id"),
          "turns": len(its), "uesp": sum(len(i["tags"]["assistant"]) for i in its)}
         for uid, its in _units(items).items() if axis_of(its[0]) == INTERACTIVE),
        key=lambda e: e["unit_id"])
    out["difficulty"] = {ax: dict(sorted(Counter(i.get("difficulty") for i in its if i.get("turn", 0) in (0, None)).items(),
                                         key=lambda kv: str(kv[0])))
                         for ax, its in sorted(axes.items()) if ax != SUPPORTING}
    return out


def operator_view(items: list[dict]) -> dict:
    """The halves as the operators read them, through the core's own contracts.

    FFN, RESIDUAL_WIDTH and ATTENTION accumulate their statistics over the
    position policy's TOKEN-axis weights (`statistics_positions`; the baseline
    declares no confidence weighting), DEPTH scores KL over its PREDICTION-axis
    weights and aggregates the per-subtype means with `domain_balanced_score`
    over the domain map it builds from the items' labels (`_domain_map`). The
    weight of a subtype is that score with the subtype at 1 and the rest at 0."""
    from aadistill.initialization.operators.depth.causal_kl_greedy import _domain_map
    from aadistill.initialization.scoring.batches import PREDICTION_AXIS, TOKEN_AXIS, weights_for_items
    from aadistill.initialization.statistics.contribution import domain_balanced_score
    from shared import forward_baseline as FB
    policy = FB.selected().position_policy
    out = {"position_policy": policy.qualified_id}
    for name, axis in (("token_axis", TOKEN_AXIS), ("prediction_axis", PREDICTION_AXIS)):
        by = Counter()
        for it, w in zip(items, weights_for_items(items, policy, axis=axis)):
            x = float(w.weights.sum())
            if x != int(x):
                raise SystemExit(f"{it['item_id']}: non-binary position weights under {policy.qualified_id}")
            by[it["domain"]] += int(x)
        out[name] = dict(sorted(by.items()))
    domains = _domain_map(items)
    subtypes = [s for v in domains.values() for s in v]
    if len(set(subtypes)) != len(subtypes):
        raise SystemExit(f"a subtype label is shared across halves: {domains}")
    weight = {s: domain_balanced_score({t: float(t == s) for t in subtypes}, domains)[0] for s in subtypes}
    out["depth_domain_map"] = domains
    out["depth_subtype_weight"] = {s: round(w, 6) for s, w in weight.items()}
    out["depth_half_weight"] = {d: round(sum(weight[s] for s in v), 12) for d, v in domains.items()}
    return out


def require_exact(view: dict) -> None:
    """fluid == supporting on both axes the operators read, and DEPTH 1/2 each: no tolerance."""
    for axis in ("token_axis", "prediction_axis"):
        if view[axis].get(FLUID) != view[axis].get(SUPPORTING) or not view[axis].get(FLUID):
            raise SystemExit(f"the halves are not equal on the {axis}: {view[axis]}")
    if view["depth_half_weight"] != {FLUID: 0.5, SUPPORTING: 0.5}:
        raise SystemExit(f"DEPTH does not weight the halves 1/2 each: {view['depth_half_weight']}")


def _units(items: list[dict]) -> dict[str, list[dict]]:
    u = defaultdict(list)
    for it in items:
        u[it["unit_id"]].append(it)
    return u


def historical(micro_batch: int, pool_families: dict) -> dict:
    """D4's two profiles (D4 Top-1 calibrated on them) and d1/d2's, measured the same way."""
    doc = json.loads((REPO / HISTORICAL_PROFILES).read_text())
    out = {}
    for p in doc["profiles"]:
        path = REPO / p["items_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != p["items_file_sha256"]:
            raise SystemExit(f"{path} is not the profile {p['profile_id']} registered")
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out[p["profile_id"]] = coverage(items, micro_batch=micro_batch, pool_families=pool_families)
    return out


# --- compute ----------------------------------------------------------------

def telemetry() -> list[dict]:
    """Every historical expansion: operator, parent level, operator seconds,
    per-expansion overhead seconds, and its profile's forward tokens."""
    meta = {p["profile_id"] + "@v1": p["metadata"]
            for p in json.loads((REPO / HISTORICAL_PROFILES).read_text())["profiles"]}
    rows = []
    for dose, run in HISTORICAL.items():
        base = REPO / S0 / run / dose / "search/search"
        states = {}
        for line in (base / "states.jsonl").open():
            s = json.loads(line)
            if s.get("state_id"):
                states[s["state_id"]] = s
        for line in (base / "telemetry.jsonl").open():
            e = json.loads(line)
            if e["event"] != "expansion":
                continue
            over = float(e.get("parent_load_seconds") or 0) + sum(
                float((e.get(k) or {}).get("seconds") or 0)
                for k in ("canonical_reload_seconds", "identify_seconds", "materialize_seconds",
                          "state_evaluation_seconds", "validation_seconds", "validation_transfer_seconds"))
            child = states[e["state_id"]]
            rows.append({"dose": dose, "kind": e["impl_id"].split(".")[0].upper(),
                         "level": len(states[e["parent_id"]].get("applied_kinds", [])) + 1
                         if e["parent_id"] in states else 1,
                         "operator_seconds": float(e["operator_seconds"]), "overhead_seconds": over,
                         "mtok": meta[e["profile"]]["tokens"] / 1e6,
                         "state_bytes": (child.get("artifact") or {}).get("total_bytes") or 0})
    return rows


def compute_model(rows: list[dict]) -> dict:
    """DEPTH: minutes per million forward tokens by tree level (levels 3-4
    share level 3's rate, the deepest measured); every other operator a
    least-squares a + b x Mtok in seconds; the overhead a per-expansion mean."""
    depth = defaultdict(list)
    for r in rows:
        if r["kind"] == "DEPTH":
            depth[r["level"]].append(r["operator_seconds"] / 60 / r["mtok"])
    rate = {lv: statistics.fmean(v) for lv, v in sorted(depth.items())}
    deepest = max(rate)
    for lv in TREE:
        rate.setdefault(lv, rate[deepest] if lv > deepest else rate[min(rate)])
    lin = {}
    for kind in ("FFN", "WIDTH", "ATTENTION"):
        pts = [(r["mtok"], r["operator_seconds"]) for r in rows if r["kind"] == kind]
        mx, my = statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts)
        b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        lin[kind] = {"a_seconds": round(my - b * mx, 2), "b_seconds_per_mtok": round(b, 2),
                     "n": len(pts), "max_seconds": round(max(p[1] for p in pts), 1)}
    frac = {}
    for lv, n in TREE_2P.items():
        d = [r for r in rows if r["level"] == lv]
        frac[lv] = sum(r["kind"] == "DEPTH" for r in d) / len(d)
    by_level_bytes = defaultdict(list)
    for r in rows:
        by_level_bytes[r["level"]].append(r["state_bytes"])
    return {"depth_minutes_per_mtok": {lv: round(v, 1) for lv, v in sorted(rate.items())},
            "depth_rate_spread": {lv: [round(min(v), 1), round(max(v), 1), len(v)] for lv, v in sorted(depth.items())},
            "other_operators": lin,
            "overhead_seconds_per_expansion": round(statistics.fmean(r["overhead_seconds"] for r in rows), 1),
            "historical_depth_fraction_by_level": {lv: round(v, 4) for lv, v in frac.items()},
            "state_gb_by_level": {lv: round(statistics.fmean(v) / 1e9, 2) for lv, v in sorted(by_level_bytes.items())}}


def _other_seconds(model: dict, mtok: float, *, worst: bool) -> float:
    lin = model["other_operators"]
    vals = [max(v["a_seconds"] + v["b_seconds_per_mtok"] * mtok, 0.0) for v in lin.values()]
    return max(vals) if worst else statistics.fmean(vals)


def forecast_search(model: dict, mtok: float, *, tree: dict, depth_counts: dict, worst: bool) -> dict:
    rate = model["depth_minutes_per_mtok"]
    depth_min = sum(depth_counts[lv] * rate[lv] * mtok for lv in tree)
    n_depth = sum(depth_counts.values())
    n = sum(tree.values())
    other_min = (n - n_depth) * _other_seconds(model, mtok, worst=worst) / 60
    over_min = n * model["overhead_seconds_per_expansion"] / 60
    return {"expansions": n, "depth_expansions": round(n_depth, 2), "depth_minutes": round(depth_min, 1),
            "other_operator_minutes": round(other_min, 1), "overhead_minutes": round(over_min, 1),
            "expansion_minutes": round(depth_min + other_min + over_min, 1)}


def model_check(model: dict, rows: list[dict]) -> dict:
    """The model against what it was fitted to: each dose's two-profile search
    at its observed DEPTH placements, predicted beside the telemetry sum."""
    out = {}
    for dose in HISTORICAL:
        d = [r for r in rows if r["dose"] == dose]
        mtok = statistics.fmean(r["mtok"] for r in d)
        counts = {lv: sum(1 for r in d if r["kind"] == "DEPTH" and r["level"] == lv) for lv in TREE_2P}
        pred = forecast_search(model, mtok, tree=TREE_2P, depth_counts=counts, worst=False)
        meas = sum(r["operator_seconds"] + r["overhead_seconds"] for r in d) / 60
        out[dose] = {"predicted_minutes": pred["expansion_minutes"], "telemetry_minutes": round(meas, 1),
                     "ratio": round(pred["expansion_minutes"] / meas, 3)}
    return out


def search_cost(model: dict, mtok: float, prices: dict) -> dict:
    exp_counts = {lv: TREE[lv] * model["historical_depth_fraction_by_level"][lv] for lv in TREE}
    expected = forecast_search(model, mtok, tree=TREE, depth_counts=exp_counts, worst=False)
    bound = forecast_search(model, mtok, tree=TREE, depth_counts=DEPTH_MAX, worst=True)
    out = {"mtok": round(mtok, 4), "expected": expected, "bound": bound}
    fixed = prices["search_fixed_minutes"] + prices["session_fixed_minutes"]
    gb = sum(TREE[lv] * model["state_gb_by_level"][lv] for lv in TREE)
    for k in ("expected", "bound"):
        m = out[k]["expansion_minutes"] + fixed
        out[k]["session_minutes"] = round(m, 1)
        out[k]["gpu_usd"] = round(m / 60 * prices["search_usd_per_hour"], 2)
        out[k]["container_disk_usd"] = round(prices["search_container_gb"] * prices["disk_usd_per_gb_month"]
                                             * m / 60 / 720, 2)
    out["state_storage_gb"] = round(gb, 1)
    return out


def recovery_cost(prices: dict) -> dict:
    """Two D4-Augmented-shaped units (train + every item of eval-set-1), from
    H1-PA's measured D4-Augmented units, in one L40S session."""
    res = json.loads((REPO / prices["recovery_measured"]).read_text())["units"]
    units = {n: u["seconds"] for n, u in res.items() if n.startswith("D4-Augmented")}
    per = statistics.fmean(s["train"] + s["evaluate"] for s in units.values()) / 60
    worst = max(s["train"] + s["evaluate"] for s in units.values()) / 60
    out = {"measured_units": {n: {k: round(v / 60, 1) for k, v in s.items()} for n, s in units.items()},
           "per_unit_minutes": round(per, 1)}
    for k, m in (("expected", 2 * per), ("bound", 2 * worst * prices["recovery_bound_factor"])):
        m += prices["session_fixed_minutes"]
        out[k] = {"session_minutes": round(m, 1), "gpu_usd": round(m / 60 * prices["recovery_usd_per_hour"], 2),
                  "container_disk_usd": round(prices["recovery_container_gb"] * prices["disk_usd_per_gb_month"]
                                              * m / 60 / 720, 2)}
    return out


#: Prices and fixed overheads the forecast uses, each with its source.
PRICES = {
    "search_usd_per_hour": 1.79,
    "_search_usd_per_hour": "A100-SXM4-80GB, the rate every Stage-0 search session was billed at (runs/dose_*/runtime/session.json :: cost.price_per_hour); re-read live at authorization",
    "recovery_usd_per_hour": 1.09,
    "_recovery_usd_per_hour": "L40S in EUR-IS-2, the H1-PA PA2 sessions' billed rate (runs/h1pa2_*/runtime/session.json); re-read live at authorization",
    "disk_usd_per_gb_month": 0.10,
    "search_container_gb": 200,
    "recovery_container_gb": 150,
    "search_fixed_minutes": 8.0,
    "_search_fixed_minutes": "d4's search ran 07:00Z-18:33Z (693 min, runs/dose_d4_20261020_054022/closeout/outcome.json) over 685.5 telemetry minutes",
    "session_fixed_minutes": 15.0,
    "_session_fixed_minutes": "create to setup done 6-8 min (H1-PA launch logs), plus export, collection and teardown",
    "recovery_bound_factor": 1.25,
    "recovery_measured": "logs/stages/stage-2/reasoning_data/analyses/h1pa_result.json",
}


# --- main -------------------------------------------------------------------

def profile_entry(design: dict, pool_manifest: dict, dose: str, items: list[dict], seed: int, rel: str,
                  body: str, view: dict) -> dict:
    fluid_nominal = float(design["doses"][dose]["fluid_nominal_uesp"])
    half = view["token_axis"][FLUID]
    return {
        "profile_id": f"calib.s0m_mix_{dose}", "version": int(design["profile_version"]),
        "description": (f"Stage-0 mixed calibration {dose}: supporting + fluid (five P10.2 axes), exactly equal "
                        f"realized scored positions ({half} each), fluid nominal {int(fluid_nominal)} UESP, "
                        f"sampling seed {seed}"),
        "sources": [{"dataset_id": f"aadistill/stage0_base_pool_v{pool_manifest['version']}",
                     "revision": pool_manifest["items_sha256"][:16], "domain": d,
                     "n_items": sum(1 for it in items if it["domain"] == d), "content_sha256": None,
                     "license": "project-generated + corpus_v2", "note": "stratified sample, exact balance; see the design"}
                    for d in (FLUID, SUPPORTING)],
        "domain_weights": {FLUID: 0.5, SUPPORTING: 0.5},
        "token_budget": int(round(2 * fluid_nominal)),
        "sample_rule": design["sample_rule"], "seed": seed,
        "role": "operator_calibration", "materialized": True, "items_path": rel,
        "rendered_manifest_sha256": None,
        "content_sha256": mixture_content_sha256([json.loads(line) for line in body.splitlines()]),
        "items_file_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "leakage_exclusions": design["exclusions"]["inherited"] + [
            "D4-Augmented's training and validation sessions (E1 2.96M prefix + 16 validation sequences, fluid increment)",
            "eval-set-1 held-out units (Stage-2 isolation rules R0-R9)"],
        "leakage_proof_path": design["output"]["audit"],
        "metadata": {"dose": dose, "fluid_nominal_uesp": int(fluid_nominal),
                     "uesp": sum(view["token_axis"].values()), "uesp_by_half": view["token_axis"],
                     "n_items": len(items), "n_units": len({it["unit_id"] for it in items}),
                     "tokens": sum(it["n_tokens"] for it in items),
                     "n_rule_families": len({(it.get("generator_id"), it.get("family_id"))
                                             for it in items if it.get("family_id")})},
        "is_null": False}


def superseded_summary(design: dict) -> dict:
    """The superseded design's 2x profile as its own audit recorded it, verified by hash."""
    sup = design["supersedes"]
    path = REPO / sup["feasibility"]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sup["feasibility"]["sha256"]:
        raise SystemExit(f"{path} is not the superseded audit the design pins")
    old = json.loads(path.read_text())
    tot, halves = old["coverage"]["2x"]["total"], old["coverage"]["2x"]["by_half"]
    cost = old["compute"]["search"]["2x"]
    return {"profile": "calib.s0m_mix_2x@v1", "uesp": tot["uesp"],
            "uesp_by_half": {h: b["uesp"] for h, b in halves.items()},
            "forward_tokens": tot["forward_tokens"], "units": tot["units"], "items": tot["items"],
            "search_expected_usd": cost["expected"]["gpu_usd"], "search_bound_usd": cost["bound"]["gpu_usd"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    args = ap.parse_args(argv)
    from shared import forward_baseline as FB
    design_path = REPO / args.design
    design = json.loads(design_path.read_text())
    pool_dir = REPO / design["pool"]["dir"]
    pool_manifest = json.loads((pool_dir / "manifest.json").read_text())
    if pool_manifest["items_sha256"] != design["pool"]["items_sha256"]:
        raise SystemExit("the pool on disk is not the pool the design names")
    micro_batch = int(FB.selected().execution.micro_batch_size)
    seed_text = f"stage0.calibration_volume.{design['draw']}|draw|1"
    if f"'{seed_text}'" not in design["seed_derivation"]:
        raise SystemExit(f"the design's seed derivation does not declare {seed_text!r}")
    seed = derive_seed(seed_text)
    units = BVP.load_units(pool_dir)
    pool_families = defaultdict(set)
    for u in units.values():
        for it in u["items"]:
            if it.get("family_id"):
                pool_families[it["generator_id"]].add(it["family_id"])
    kept, elig = eligible(design, units)
    cells = cellmap(design, kept)
    out = REPO / design["output"]["audit"]
    out.parent.mkdir(parents=True, exist_ok=True)
    head = {"schema": "aadistill.stage0.calibration_volume.mixed_feasibility/v2",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "design": {"path": args.design, "sha256": hashlib.sha256(design_path.read_bytes()).hexdigest()},
            "sampling_seed": seed, "micro_batch_size": micro_batch}
    profiles, cov, census, views = [], {}, {}, {}
    for dose, spec in design["doses"].items():
        try:
            items, cen = sample(design, kept, cells, seed, float(spec["fluid_nominal_uesp"]))
        except Infeasible as e:
            out.write_text(json.dumps({**head, "status": "INFEASIBLE", "dose": dose, "evidence": e.evidence,
                                       "reading": "no profile is written; the ratio is not relaxed",
                                       "authorizes": "nothing"}, indent=1) + "\n")
            print(e, file=sys.stderr)
            return 1
        views[dose] = operator_view(items)
        require_exact(views[dose])
        rel = f"{design['output']['items_dir']}/calib.s0m_mix_{dose}.items.jsonl"
        body = "".join(json.dumps({k: it[k] for k in ITEM_FIELDS if k in it}, sort_keys=True) + "\n"
                       for it in items)
        (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        (REPO / rel).write_text(body)
        profiles.append(profile_entry(design, pool_manifest, dose, items, seed, rel, body, views[dose]))
        cov[dose] = coverage(items, micro_batch=micro_batch, pool_families=pool_families)
        census[dose] = cen
    version = int(design["profile_version"])
    doc = {"schema": "aadistill.calibration_profiles/v1",
           "_contract": "Stage-0 mixed calibration profiles, built by build_mixed_profiles.py from the design named below; data. AUTHORIZES NOTHING.",
           "design": args.design, "design_sha256": head["design"]["sha256"],
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "profiles": profiles,
           "treatments": {d: {"profiles": [f"calib.s0m_mix_{d}@v{version}"], "sampling_seed": seed}
                          for d in design["doses"]}}
    load_profiles(doc, register=False)
    (REPO / design["output"]["profiles_doc"]).write_text(json.dumps(doc, indent=1) + "\n")
    rows = telemetry()
    model = compute_model(rows)
    hist = historical(micro_batch, pool_families)
    cost = {dose: search_cost(model, cov[dose]["total"]["forward_tokens"] / 1e6, PRICES) for dose in design["doses"]}
    nominal = {d: {"nominal_uesp": 2 * int(s["fluid_nominal_uesp"]), "realized_uesp": cov[d]["total"]["uesp"],
                   "ratio": round(cov[d]["total"]["uesp"] / (2 * int(s["fluid_nominal_uesp"])), 4),
                   "of_reference": round(cov[d]["total"]["uesp"] / int(design["reference_uesp"]), 4),
                   "fluid_cells": {c: {k: v[k] for k in ("quota", "scored", "units")}
                                   for c, v in census[d].items() if c.startswith(FLUID)}}
               for d, s in design["doses"].items()}
    audit = {
        **head,
        "status": "EXACT",
        "profiles_doc": {"path": design["output"]["profiles_doc"],
                         "profiles": {f"{p['profile_id']}@v{p['version']}": {
                             "content_sha256": p["content_sha256"], "items_file_sha256": p["items_file_sha256"]}
                             for p in profiles}},
        "pool": {"units_in_design_cells": sum(1 for u in units.values()
                                              if u["domain"] == SUPPORTING or u["subtype"] in
                                              {g for gs in design["halves"][FLUID]["axes"].values() for g in gs}),
                 "eligible_units": len(kept),
                 "eligible_by_cell": {"/".join(c): {"units": len(v), "scored": sum(kept[u]["scored"] for u in v)}
                                      for c, v in sorted(cells.items())},
                 "rule_families_by_generator": {g: sorted(f) for g, f in sorted(pool_families.items())}},
        "eligibility": elig,
        "cells": census,
        "operator_view": views,
        "nominal_vs_realized": nominal,
        "coverage": cov,
        "historical": hist,
        "superseded": superseded_summary(design),
        "compute": {"model": model, "model_check": model_check(model, rows), "prices": PRICES,
                    "search": cost, "recovery_two_seeds": recovery_cost(PRICES)},
        "authorizes": "nothing"}
    out.write_text(json.dumps(audit, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")
    print(json.dumps({d: {"total": cov[d]["total"], "operator_view": views[d], "balance": census[d]["_balance"],
                          "search_expected_usd": cost[d]["expected"]["gpu_usd"],
                          "search_bound_usd": cost[d]["bound"]["gpu_usd"]} for d in cov}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

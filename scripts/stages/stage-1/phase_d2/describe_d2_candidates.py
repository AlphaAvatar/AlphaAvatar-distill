#!/usr/bin/env python3
"""How the four committed D2 initializations differ, parameter by parameter. `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/describe_d2_candidates.py [--write]

DESCRIPTIVE ONLY. The candidate set and its order are the committed selection's
(`stage1_selection.json`, hash-verified) and nothing here reads back into them:
no ranking, filter or tie-break consumes this file.

All four share RESIDUAL_WIDTH(domain_balanced) -> DEPTH(domain_balanced) with
the same eight removed layers, then FFN(profile) -> ATTENTION(profile). So the
FFN masks are a function of the FFN profile alone (the FFN step runs on the one
shared parent), while the attention masks are chosen on four different parents.
This file measures that structure from two independent sources and checks they
agree:

* MASKS, from the search journal: the FFN step's `kept_neurons` per layer (of
  the teacher's intermediate neurons) and the ATTENTION step's
  `kept_q_heads_per_layer` (of the 32 query heads) -- pairwise overlap per
  layer, as counts and Jaccard;
* WEIGHTS, from the checkpoint bytes: per tensor family, how many tensors are
  bitwise identical between two candidates and the relative Frobenius
  difference of the rest; and, aligned by ORIGINAL index through the masks,
  whether every neuron and head two candidates both kept carries bitwise
  identical weights (i.e. whether the candidates differ only by WHICH units
  were kept, or also in the values of the units they share).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages.phase_d2.d2_product_durability import (  # noqa: E402
    DEFAULT_SCR, RUN_REL, selected_rows,
)

OUT_REL = f"{RUN_REL}/evidence/candidate_differences.json"
JOURNAL_IN_STORE = "store/extracted/audit/autoinit_d2/search/states.jsonl"
SCHEMA = "aadistill.phase_d2.candidate_differences/v1"


def journal_steps(journal: Path, state_ids: set[str]) -> dict[str, list[dict]]:
    """Each candidate's step records, from the LAST journal line naming it."""
    out: dict[str, list[dict]] = {}
    with journal.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("state_id") in state_ids:
                out[row["state_id"]] = row["steps"]
    missing = state_ids - set(out)
    if missing:
        raise SystemExit(f"the journal has no record of {sorted(missing)}")
    return out


def masks(steps: list[dict]) -> dict[str, Any]:
    by_kind = {str(s.get("impl_id", "")).split(".")[0]: s for s in steps}
    ffn, attn = by_kind["ffn"], by_kind["attention"]
    neurons = [sorted(int(i) for i in layer)
               for layer in ffn["artifacts"]["kept_neurons"]]
    heads = [sorted(int(h) for h in layer)
             for layer in attn["trace"]["kept_q_heads_per_layer"]]
    return {"ffn_profile": ffn.get("profile_id"),
            "attention_profile": attn.get("profile_id"),
            "neurons": neurons, "heads": heads,
            "q_heads": attn["trace"].get("q_heads"),
            "head_dim": None}


def _overlap(a: list[list[int]], b: list[list[int]]) -> dict[str, Any]:
    shared = [len(set(x) & set(y)) for x, y in zip(a, b)]
    union = [len(set(x) | set(y)) for x, y in zip(a, b)]
    kept = [len(x) for x in a]
    jac = [s / u if u else 1.0 for s, u in zip(shared, union)]
    return {"kept_per_layer": sorted(set(kept)),
            "shared_total": sum(shared), "kept_total": sum(kept),
            "shared_fraction": round(sum(shared) / sum(kept), 6),
            "jaccard_mean": round(sum(jac) / len(jac), 6),
            "jaccard_min": round(min(jac), 6),
            "layers_identical": sum(1 for x, y in zip(a, b) if x == y),
            "n_layers": len(a)}


def family(name: str) -> str:
    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
                "gate_proj", "up_proj", "down_proj", "input_layernorm",
                "post_attention_layernorm", "embed_tokens", "lm_head"):
        if key in name:
            return key
    return "final_norm" if name.startswith("model.norm") else "other"


def weight_differences(dirs: dict[str, Path], order: list[str],
                       mask: dict[str, dict]) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    handles = {sid: safe_open(str(dirs[sid] / "model.safetensors"), "pt")
               for sid in order}
    names = sorted(handles[order[0]].keys())
    for sid in order[1:]:
        if sorted(handles[sid].keys()) != names:
            raise SystemExit(f"{sid} has a different tensor set")
    pairs = list(itertools.combinations(order, 2))
    fam: dict[tuple, dict[str, Any]] = collections.defaultdict(
        lambda: {"tensors": 0, "identical": 0, "diff_sq": 0.0, "ref_sq": 0.0})
    shared_units: dict[tuple, dict[str, int]] = collections.defaultdict(
        lambda: collections.Counter())
    norms: dict[tuple, dict[str, float]] = collections.defaultdict(
        lambda: {"unshared_sq": 0.0, "total_sq": 0.0})
    head_dim = None
    for name in names:
        t = {sid: handles[sid].get_tensor(name) for sid in order}
        f = family(name)
        layer = (int(name.split(".")[2]) if name.startswith("model.layers.")
                 else None)
        for a, b in pairs:
            row = fam[(a, b, f)]
            row["tensors"] += 1
            x, y = t[a].float(), t[b].float()
            if torch.equal(t[a], t[b]):
                row["identical"] += 1
            row["diff_sq"] += float((x - y).pow(2).sum())
            row["ref_sq"] += float(x.pow(2).sum())
            if layer is None:
                continue
            #: Units both kept, aligned by ORIGINAL index through the masks:
            #: are their weights bitwise equal?
            if f in ("gate_proj", "up_proj", "down_proj"):
                na, nb = mask[a]["neurons"][layer], mask[b]["neurons"][layer]
                pa = {n: i for i, n in enumerate(na)}
                pb = {n: i for i, n in enumerate(nb)}
                common = sorted(set(na) & set(nb))
                ia = torch.tensor([pa[n] for n in common], dtype=torch.long)
                ib = torch.tensor([pb[n] for n in common], dtype=torch.long)
                dim = 1 if f == "down_proj" else 0
                same = (t[a].index_select(dim, ia) == t[b].index_select(dim, ib))
                same = same.all(dim=1 - dim)
                c = shared_units[(a, b, "ffn_neurons")]
                c["shared"] += len(common)
                c["bitwise_identical"] += int(same.sum())
                #: The units A kept and B did not: their share of A's norm.
                only = torch.tensor([pa[n] for n in sorted(set(na) - set(nb))],
                                    dtype=torch.long)
                norms[(a, b, f)]["unshared_sq"] += float(
                    x.index_select(dim, only).pow(2).sum()) if len(only) else 0.0
                norms[(a, b, f)]["total_sq"] += float(x.pow(2).sum())
            if f in ("q_proj", "o_proj"):
                ha, hb = mask[a]["heads"][layer], mask[b]["heads"][layer]
                width = t[a].shape[0 if f == "q_proj" else 1]
                hd = width // len(ha)
                head_dim = hd
                pa = {h: i for i, h in enumerate(ha)}
                pb = {h: i for i, h in enumerate(hb)}
                c = shared_units[(a, b, "attention_heads")]
                for h in sorted(set(ha) - set(hb)):
                    sa = slice(pa[h] * hd, (pa[h] + 1) * hd)
                    blk = x[sa] if f == "q_proj" else x[:, sa]
                    norms[(a, b, f)]["unshared_sq"] += float(blk.pow(2).sum())
                norms[(a, b, f)]["total_sq"] += float(x.pow(2).sum())
                for h in sorted(set(ha) & set(hb)):
                    sa = slice(pa[h] * hd, (pa[h] + 1) * hd)
                    sb = slice(pb[h] * hd, (pb[h] + 1) * hd)
                    blk_a = t[a][sa] if f == "q_proj" else t[a][:, sa]
                    blk_b = t[b][sb] if f == "q_proj" else t[b][:, sb]
                    c[f"shared_{f}"] += 1
                    c[f"bitwise_identical_{f}"] += int(torch.equal(blk_a, blk_b))
    out = {}
    for a, b in pairs:
        key = f"{a[:8]}~{b[:8]}"
        out[key] = {
            "families": {
                f: {"tensors": r["tensors"], "bitwise_identical": r["identical"],
                    #: POSITIONAL: rows shift when two masks keep different
                    #: units, so for a masked family this is not a
                    #: like-for-like difference -- read the aligned
                    #: `unshared_norm_fraction` instead.
                    "positional_relative_frobenius_difference": round(
                        (r["diff_sq"] / r["ref_sq"]) ** 0.5, 6)
                    if r["ref_sq"] else None,
                    **({"unshared_norm_fraction": round(
                        (norms[(a, b, f)]["unshared_sq"]
                         / norms[(a, b, f)]["total_sq"]) ** 0.5, 6)}
                       if norms.get((a, b, f), {}).get("total_sq") else {})}
                for (x, y, f), r in sorted(fam.items()) if (x, y) == (a, b)},
            "shared_units": {k: dict(v) for (x, y, k), v in
                             sorted(shared_units.items()) if (x, y) == (a, b)},
        }
    return {"pairs": out, "head_dim": head_dim, "n_tensors": len(names),
            "_reading": ("shared_units: every unit both candidates kept, "
                         "aligned by original index; bitwise_identical == "
                         "shared means the two differ ONLY in which units "
                         "they kept. unshared_norm_fraction: the "
                         "Frobenius-norm share of A's tensor carried by units "
                         "B did not keep -- the aligned size of the "
                         "difference, in norm terms")}


def describe(scr: Path, journal: Path | None = None,
             repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rows = selected_rows(repo_root)
    order = [r["state_id"] for r in rows]
    steps = journal_steps(journal or scr / JOURNAL_IN_STORE, set(order))
    mask = {sid: masks(steps[sid]) for sid in order}
    pairs = list(itertools.combinations(order, 2))
    mask_pairs = {
        f"{a[:8]}~{b[:8]}": {
            "ffn_profiles": [mask[a]["ffn_profile"], mask[b]["ffn_profile"]],
            "attention_profiles": [mask[a]["attention_profile"],
                                   mask[b]["attention_profile"]],
            "ffn_neurons": _overlap(mask[a]["neurons"], mask[b]["neurons"]),
            "attention_q_heads": _overlap(mask[a]["heads"], mask[b]["heads"]),
        } for a, b in pairs}
    weights = weight_differences(
        {sid: scr / "products" / sid for sid in order}, order, mask)
    per_candidate = [{
        "quality_order": i + 1, "state_id": sid,
        "path": rows[i]["path"],
        "ffn_profile": mask[sid]["ffn_profile"],
        "attention_profile": mask[sid]["attention_profile"],
        "ffn_kept_per_layer": sorted({len(x) for x in mask[sid]["neurons"]}),
        "q_heads_kept_per_layer": sorted({len(x) for x in mask[sid]["heads"]}),
        "ffn_mask_sha256": hashlib.sha256(json.dumps(
            mask[sid]["neurons"]).encode()).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(json.dumps(
            mask[sid]["heads"]).encode()).hexdigest(),
    } for i, sid in enumerate(order)]
    distinct_ffn = len({c["ffn_mask_sha256"] for c in per_candidate})
    distinct_attn = len({c["attention_mask_sha256"] for c in per_candidate})
    return {
        "schema": SCHEMA,
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "descriptive_only": ("does not read back into the frozen candidate "
                             "set or its order; no rule consumes it"),
        "inputs": {
            "selection": f"{RUN_REL}/evidence/stage1_selection.json",
            "journal": str(journal or scr / JOURNAL_IN_STORE),
            "checkpoints": str(scr / "products")},
        "candidates": per_candidate,
        "distinct_ffn_masks": distinct_ffn,
        "distinct_attention_masks": distinct_attn,
        "mask_overlap": mask_pairs,
        "weights": weights,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", default=str(DEFAULT_SCR))
    ap.add_argument("--journal", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    doc = describe(Path(args.scr),
                   Path(args.journal) if args.journal else None)
    if args.write:
        (REPO_ROOT / OUT_REL).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT_REL}")
    print(json.dumps({k: doc[k] for k in ("candidates", "distinct_ffn_masks",
                                          "distinct_attention_masks",
                                          "mask_overlap")}, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

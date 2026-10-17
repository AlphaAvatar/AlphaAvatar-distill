#!/usr/bin/env python3
"""D2 v3 union-KL numerics on REAL logits, at $0, on the CPU.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2v3_real_logits_check.py \
        --checkpoint <dir of a local Qwen3 checkpoint> [--write]

The small-vocabulary oracle tests prove the union reduction's arithmetic. This
asks the questions only real logits answer -- 151,936-entry heavy tails, bf16
ties, a Top-200 holding nearly all the mass -- on whatever local checkpoint is
handed to it (the D2 teacher's weights are not on the development box; a
local compressed checkpoint of the same family and vocabulary stands in, and
the record says which). Nothing here is a D2 measurement or a gate input: it
is engineering evidence that the definitions behave on real distributions.

Per position, against a float64 full-vocabulary oracle on the SAME tensors
(the reference's float32 head logits, the candidate's own bf16 logits):

* the bounds ``KL_{S_T+tail} <= KL_U <= KL_full`` (float32 slack recorded);
* how much of the full KL each coarsening keeps, per position and per item;
* the union's size and overlap, and both masses outside it;
* retrieval: the store's log probability of candidate-only entries against
  the full head's log-softmax;
* the float32 head definition against the model's own bf16 logits;
* how often bf16 candidate logits tie at the K-th value, and how often
  ``torch.topk``'s set differs from the deterministic one;
* the reference's K+1 confidence beside the full-vocabulary one (v2's).

Candidates are the reference with blocks bypassed -- DEPTH's own candidate
shape -- and their ranking is compared under the three KLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from aadistill.initialization.scoring.objective import (  # noqa: E402
    confidence_from_vocabulary_entropy, full_vocabulary_entropy,
)
from aadistill.initialization.scoring.reference_store import (  # noqa: E402
    build_root_reference_store, root_reference_identity,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    deterministic_topk, reference_student_union_topk_tail,
    union_forward_kl_terms,
)
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks,
)

OUT = ("logs/stages/stage-1/families/d_series/analyses/"
       "d2v3_real_logits_check.json")
PROFILE = "calib.domain_balanced@v1"
#: Candidates: blocks bypassed, as DEPTH builds them. Indices are read against
#: the checkpoint's own depth (early, middle, late, and a middle pair).
CANDIDATE_FRACTIONS = ((0.15,), (0.5,), (0.85,), (0.5, 0.54))


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return {}
    pick = lambda q: xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    return {"p01": pick(0.01), "p10": pick(0.10), "p50": pick(0.5),
            "p90": pick(0.90), "p99": pick(0.99), "max": xs[-1],
            "min": xs[0], "mean": sum(xs) / len(xs), "n": len(xs)}


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    ra, rb = ranks(list(a)), ranks(list(b))
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra)
                    * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def _partition_kl64(p_log: torch.Tensor, q_log: torch.Tensor,
                    members: torch.Tensor) -> torch.Tensor:
    """Rows of float64 forward KL on `members` [N, M] (bool mask built here)
    plus one bucket for the rest; the oracle."""
    mask = torch.zeros_like(p_log, dtype=torch.bool)
    mask.scatter_(1, members, True)
    p = p_log.exp()
    term = torch.where(mask & (p > 0), p * (p_log - q_log),
                       torch.zeros_like(p))
    kl = term.sum(1)
    pr = torch.where(mask, torch.zeros_like(p), p).sum(1)
    qr = torch.where(mask, torch.zeros_like(p), q_log.exp()).sum(1)
    tail = torch.where(pr > 0, pr * (pr.log() - qr.log()),
                       torch.zeros_like(pr))
    return kl + tail


def _items(repo: Path, per_domain: int) -> list[dict[str, Any]]:
    import shared.calibration  # noqa: F401  (registers the profiles)
    from aadistill.initialization.calibration.items import (
        prepare_calibration_items,
    )
    from aadistill.initialization.calibration.profiles import get_profile

    p = get_profile(PROFILE)
    items = prepare_calibration_items(p.resolve(repo), profile_id=p.qualified_id)
    by_domain: dict[str, list] = {}
    for item in sorted(items, key=lambda i: (i["domain"],
                                             int(i["input_ids"].shape[1]),
                                             str(i["item_id"]))):
        by_domain.setdefault(item["domain"], []).append(item)
    return [i for d in sorted(by_domain) for i in by_domain[d][:per_domain]]


@torch.no_grad()
def run(checkpoint: Path, *, top_k: int, per_domain: int,
        threads: int) -> dict[str, Any]:
    from aadistill.initialization.specs.arch import (
        adapter_for_config, get_adapter,
    )
    from aadistill.initialization.adapters import register_builtin_adapters

    torch.set_num_threads(threads)
    register_builtin_adapters()
    started = time.time()
    adapter = get_adapter("qwen3")
    model = adapter.load(str(checkpoint), dtype="bfloat16", device="cpu")
    model.config.use_cache = False
    adapter = adapter_for_config(model.config)
    depth = len(adapter.blocks(model))
    candidates = [tuple(sorted({min(depth - 1, int(round(f * depth)))
                                for f in fr})) for fr in CANDIDATE_FRACTIONS]
    items = _items(REPO, per_domain)
    support = reference_student_union_topk_tail(top_k)

    class _UnionForm:            # the store identity only needs these two
        uses_union_support = True
        qualified_id = "d2v3_real_logits_check"

    store = build_root_reference_store(
        model, items, support=support,
        identity=root_reference_identity(
            root_identity={"checkpoint": str(checkpoint)},
            profile_hash="real_logits_check", support=support,
            objective=_UnionForm()),
        micro_batch_size=1, packing="original_order_v1", device="cpu",
        chunk=256)
    head = store.head

    per_pos: dict[str, list[float]] = {k: [] for k in (
        "ratio_union_full", "ratio_topk_full", "union_size", "overlap",
        "ref_union_tail", "cand_union_tail", "bound_lower_violation",
        "bound_upper_violation", "abs_err_union_vs_oracle",
        "retrieval_abs_err", "head_vs_bf16_abs", "c_topk", "c_vocab")}
    ties = {"rows": 0, "boundary_tie_rows": 0, "set_differs_rows": 0}
    item_means: dict[str, dict[str, list[float]]] = {}
    for cand in candidates:
        key = ",".join(map(str, cand))
        item_means[key] = {"full": [], "union": [], "topk": []}

    for item in items:
        ids = item["input_ids"]
        hidden = adapter.final_hidden_states(model, ids)[0, :-1]
        ref32 = head.logits(hidden)
        own = model(ids).logits[0, :-1]
        per_pos["head_vs_bf16_abs"].append(
            float((ref32 - own.float()).abs().max()))
        p64 = torch.log_softmax(ref32.double(), -1)
        sk = store.sketch_item(item)
        retrieval = store.retrieval_item(item)
        h_vocab = full_vocabulary_entropy(ref32)
        c_vocab = confidence_from_vocabulary_entropy(h_vocab,
                                                     int(ref32.shape[-1]))
        per_pos["c_topk"].extend(store.per_position(item, "confidence")
                                 .tolist())
        per_pos["c_vocab"].extend(c_vocab.tolist())
        for cand in candidates:
            with bypassed_blocks(model, set(cand)):
                logits = model(ids).logits[0, :-1]
            terms = union_forward_kl_terms(
                sk.support_indices, sk.support_log_probs, sk.tail_log_prob,
                logits, retrieval, top_k=top_k, has_tail=True, chunk=256)
            q64 = torch.log_softmax(logits.double(), -1)
            full = (p64.exp() * (p64 - q64)).sum(-1)
            _, s_c = deterministic_topk(logits, top_k)
            members = torch.cat([sk.support_indices, s_c], dim=1)
            oracle_u = _partition_kl64(p64, q64, members)
            oracle_t = _partition_kl64(p64, q64, sk.support_indices)
            u, t = terms.kl.double(), terms.kl_reference_topk.double()
            slack = 4 * torch.finfo(torch.float32).eps * torch.logsumexp(
                ref32.double(), -1).abs()
            per_pos["bound_lower_violation"].extend(
                (t - u - slack).clamp(min=0).tolist())
            per_pos["bound_upper_violation"].extend(
                (u - full - slack).clamp(min=0).tolist())
            per_pos["abs_err_union_vs_oracle"].extend(
                (u - oracle_u).abs().tolist())
            keep = full > 1e-6
            per_pos["ratio_union_full"].extend((oracle_u[keep]
                                                / full[keep]).tolist())
            per_pos["ratio_topk_full"].extend((oracle_t[keep]
                                               / full[keep]).tolist())
            per_pos["union_size"].extend(terms.union_size.tolist())
            per_pos["overlap"].extend(terms.overlap.tolist())
            per_pos["ref_union_tail"].extend(
                terms.reference_union_tail_log.exp().tolist())
            per_pos["cand_union_tail"].extend(
                terms.candidate_union_tail_log.exp().tolist())
            #: retrieval vs the full head on candidate-only entries
            st_sorted = sk.support_indices.sort(-1).values
            pos = torch.searchsorted(st_sorted, s_c).clamp(max=top_k - 1)
            extra = st_sorted.gather(1, pos) != s_c
            got = retrieval.log_probs(s_c)
            want = torch.log_softmax(ref32, -1).gather(1, s_c)
            err = torch.where(extra, (got - want).abs(),
                              torch.zeros_like(got))
            per_pos["retrieval_abs_err"].extend(err.max(1).values.tolist())
            #: bf16 ties at the K-th candidate value
            kth = torch.topk(logits.float(), top_k).values[:, -1:]
            ties["rows"] += int(logits.shape[0])
            ties["boundary_tie_rows"] += int(
                ((logits.float() >= kth).sum(1) > top_k).sum())
            plain = torch.topk(logits, top_k).indices.sort(1).values
            ties["set_differs_rows"] += int(
                (plain != s_c.sort(1).values).any(1).sum())
            key = ",".join(map(str, cand))
            item_means[key]["full"].append(float(full.mean()))
            item_means[key]["union"].append(float(oracle_u.mean()))
            item_means[key]["topk"].append(float(oracle_t.mean()))
        del ref32, own, p64

    c_t, c_v = per_pos.pop("c_topk"), per_pos.pop("c_vocab")

    def ess_fraction(c):
        s, s2 = sum(c), sum(x * x for x in c)
        return (s * s / s2) / len(c) if s2 > 0 else 0.0

    ranking = {}
    for name in ("union", "topk"):
        ranking[f"{name}_vs_full_candidate_order_identical"] = (
            sorted(item_means, key=lambda k: statistics.mean(
                item_means[k][name]))
            == sorted(item_means, key=lambda k: statistics.mean(
                item_means[k]["full"])))
        ranking[f"{name}_vs_full_item_spearman_min"] = min(
            _spearman(v[name], v["full"]) for v in item_means.values())
    return {
        "schema": "aadistill.d2v3.real_logits_check/v1",
        "status": ("ENGINEERING EVIDENCE at $0 on a CPU, not a D2 "
                   "measurement: a local compressed checkpoint of the same "
                   "family and vocabulary stands in for the teacher, whose "
                   "weights are not on the development box"),
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()).hexdigest(),
        "head": head.identity(),
        "top_k": top_k, "profile": PROFILE,
        "items": [{"item_id": str(i["item_id"]), "domain": i["domain"],
                   "tokens": int(i["input_ids"].shape[1])} for i in items],
        "positions_per_candidate": sum(int(i["input_ids"].shape[1]) - 1
                                       for i in items),
        "candidates_bypassed_layers": [list(c) for c in candidates],
        "depth": depth,
        "per_position": {k: _quantiles(v) for k, v in per_pos.items()},
        "bounds": {
            "lower_violations_beyond_float32_slack": sum(
                1 for x in per_pos["bound_lower_violation"] if x > 0),
            "upper_violations_beyond_float32_slack": sum(
                1 for x in per_pos["bound_upper_violation"] if x > 0)},
        "bf16_ties": {**ties,
                      "boundary_tie_fraction": ties["boundary_tie_rows"]
                      / max(1, ties["rows"]),
                      "torch_topk_set_differs_fraction":
                          ties["set_differs_rows"] / max(1, ties["rows"])},
        "confidence": {
            "topk_plus_tail": {**_quantiles(c_t),
                               "ess_fraction": ess_fraction(c_t)},
            "full_vocabulary_v2": {**_quantiles(c_v),
                                   "ess_fraction": ess_fraction(c_v)},
            "spearman_topk_vs_vocab": _spearman(c_t, c_v)},
        "item_means": {k: {n: _quantiles(v) for n, v in m.items()}
                       for k, m in item_means.items()},
        "ranking": ranking,
        "store": {"identity_hash": store.identity_hash,
                  "decision": store.decision()},
        "seconds": round(time.time() - started, 1),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True,
                                     text=True).stdout.strip(),
        "threads": threads,
        "torch": torch.__version__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    record = run(a.checkpoint, top_k=a.top_k, per_domain=a.per_domain,
                 threads=a.threads)
    body = json.dumps(record, indent=1) + "\n"
    if a.write:
        (REPO / OUT).write_text(body)
        print(f"wrote {OUT}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prove the causal-depth runtime repair changed no decision.

    PYTHONPATH=src python scripts/autoinit/verify_depth_backend_equivalence.py \
        --out logs/autoinit_depth_backend_equivalence.json

`depth.causal_kl_greedy_v1` scored on the host until 2026-08-19 because the port
of `scripts/training/search_depth_map.py` inserted `.cpu()` on the logits and the
targets. The repair removes those and leaves the reduction where E8a left it: on
the compute device. That is a **backend** change to a **frozen** operator, so it
has to be shown not to move a single removal decision.

Three claims, and they are not the same claim:

**1. The refactor is exact.** Repaired path against the old host-resident path,
same device, same inputs, same seed: identical `DistortionSums`, identical
per-round tables, identical chosen layers. This is the part that can be *proved*
here, and it is the part that matters most — it shows the port change itself
introduces no arithmetic.

**2. The bf16 reference cache is exact.** E8a's docstring claims caching bf16
logits "is numerically identical to recomputing them — `distortion` upcasts to
float32 internally either way — which a test asserts rather than assumes". The
repair restores that cache to the accelerator, so the claim is re-checked here
rather than inherited.

**3. Decision tolerance is quantified, not assumed.** For every round this
reports the chosen candidate, the runner-up and the margin between them. A
backend that perturbs scores by less than the smallest margin cannot change a
removal; one that perturbs them by more can, and the number says which.

**What this CANNOT establish on a CPU-only box**, and does not claim to: the
actual CUDA-versus-CPU drift. Both paths here run on the same backend, so the
measured drift is exactly zero and that is a tautology, not evidence. The real
figure has to come from the GPU measurement run, compared against the margins
this script reports. The script prints that limitation in its own output so a
reader cannot mistake claim 1 for claim 3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.init.contribution import (  # noqa: E402
    bypassed_blocks, distortion, domain_balanced_score, greedy_removal,
)


def tiny_teacher(seed: int, layers: int = 8):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, num_hidden_layers=layers,
        intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, tie_word_embeddings=True, max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(cfg).to(torch.float32).eval()
    model.config.use_cache = False      # bypassed_blocks requires it
    return model


def make_items(vocab: int, n: int, length: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    subtypes = ["alpha", "beta", "gamma"]
    return [{"item_id": f"it{i}",
             "subtype": subtypes[i % len(subtypes)],
             "input_ids": torch.randint(0, vocab, (1, length), generator=g)}
            for i in range(n)]


def build_scorer(model, items, *, host_resident: bool, device, cache: bool):
    """One scorer. `host_resident=True` reproduces the pre-repair path exactly.

    The ONLY difference between the two is where the tensors live. Same forward,
    same chunk, same aggregation, same tie-break, same everything else.
    """
    domains = {"d0": ["alpha", "beta"], "d1": ["gamma"]}

    def place(x):
        return x.cpu() if host_resident else x.to(device)

    targets = {it["item_id"]: place(it["input_ids"][0, 1:]) for it in items}
    reference: dict[str, torch.Tensor] = {}
    counters = {"forwards": 0}

    @torch.no_grad()
    def logits(item, skip):
        counters["forwards"] += 1
        ids = item["input_ids"].to(device)
        if not skip:
            return place(model(ids).logits[0, :-1])
        with bypassed_blocks(model, skip):
            return place(model(ids).logits[0, :-1])

    def ref_for(item):
        if not cache:
            return logits(item, frozenset())
        hit = reference.get(item["item_id"])
        if hit is None:
            hit = logits(item, frozenset())
            reference[item["item_id"]] = hit
        return hit

    def score(skip):
        per_subtype: dict[str, list[float]] = {}
        for item in items:
            ref = ref_for(item)
            abl = logits(item, skip)
            sums = distortion(ref, abl, targets[item["item_id"]], chunk=512).as_dict()
            per_subtype.setdefault(item["subtype"], []).append(sums["kl"])
            del abl
        means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
        primary, _ = domain_balanced_score(means, domains)
        return primary

    return score, counters


def rounds_report(result: dict) -> list[dict]:
    """Chosen, runner-up and margin for every round."""
    out = []
    for r in result["rounds"]:
        ranked = sorted(r["table"], key=lambda x: (x["score"], x["candidate"]))
        out.append({
            "round": r["round"],
            "chosen": r["chosen"],
            "chosen_score": r["chosen_score"],
            "runner_up": ranked[1]["candidate"] if len(ranked) > 1 else None,
            "runner_up_score": ranked[1]["score"] if len(ranked) > 1 else None,
            "margin": (ranked[1]["score"] - ranked[0]["score"]
                       if len(ranked) > 1 else None),
            "n_candidates": r["n_candidates"],
        })
    return out


def frozen_rule_known_answer() -> dict:
    """The greedy rule itself, against a hand-computed answer.

    Claims 1-3 compare two placements of the SAME `greedy_removal`, so a change
    to the shared frozen rule cancels out and the comparison still passes — a
    mutation confirmed exactly that. Without this, "equivalence verified" would
    read as "the science is unchanged" while saying nothing of the kind.

    The fixture: score = sum of the skipped set, so the argmin is always the
    lowest available index, and round 0 has a deliberate tie between every
    candidate scoring the same only at index 0. Removing 3 of 6 must therefore
    choose 0, 1, 2 in that order — argmin first, ties to the lower index.
    """
    result = greedy_removal(lambda s: float(sum(s)), 6, 3)
    chosen = [r["chosen"] for r in result["rounds"]]
    ok = chosen == [0, 1, 2] and result["kept"] == [3, 4, 5]

    # And the tie-break, isolated: every candidate scores identically, so only
    # the stated rule ("ties break to the smaller layer index") decides.
    flat = greedy_removal(lambda s: 1.0, 5, 1)
    tie_ok = flat["rounds"][0]["chosen"] == 0

    return {"argmin_order": chosen, "kept": result["kept"],
            "argmin_correct": ok, "tie_breaks_to_lower_index": tie_ok,
            "holds": bool(ok and tie_ok),
            "why": ("claims 1-3 compare two placements of the same rule, so they "
                    "cannot see a change to the rule. This can.")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_depth_backend_equivalence.json")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--remove", type=int, default=3)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny_teacher(args.seed, args.layers).to(device)
    items = make_items(model.config.vocab_size, args.items, args.length, args.seed)

    # --- claim 1: the refactor is exact -----------------------------------
    repaired, c_rep = build_scorer(model, items, host_resident=False,
                                   device=device, cache=True)
    frozen_path, c_old = build_scorer(model, items, host_resident=True,
                                      device=device, cache=True)
    res_repaired = greedy_removal(repaired, args.layers, args.remove)
    res_frozen = greedy_removal(frozen_path, args.layers, args.remove)

    rep_rounds = rounds_report(res_repaired)
    frz_rounds = rounds_report(res_frozen)
    drift = [abs(a["chosen_score"] - b["chosen_score"])
             for a, b in zip(rep_rounds, frz_rounds)]
    decisions_match = (res_repaired["kept"] == res_frozen["kept"]
                       and [r["chosen"] for r in rep_rounds]
                       == [r["chosen"] for r in frz_rounds])
    tables_bit_identical = all(
        a["table"] == b["table"]
        for a, b in zip(res_repaired["rounds"], res_frozen["rounds"]))

    # --- claim 2: the bf16 reference cache is exact ------------------------
    bf16 = model.to(torch.bfloat16)
    cached, _ = build_scorer(bf16, items, host_resident=False, device=device,
                             cache=True)
    recomputed, _ = build_scorer(bf16, items, host_resident=False, device=device,
                                 cache=False)
    probe = frozenset({1, 3})
    cache_delta = abs(cached(probe) - recomputed(probe))
    model.to(torch.float32)

    frozen_rule = frozen_rule_known_answer()

    margins = [r["margin"] for r in rep_rounds if r["margin"] is not None]
    report = {
        "schema": "aadistill.autoinit.depth_backend_equivalence/v1",
        "claim_0_frozen_greedy_rule_unchanged": frozen_rule,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": {"layers": args.layers, "remove": args.remove,
                   "items": args.items, "length": args.length, "seed": args.seed},
        "claim_1_refactor_is_exact": {
            "removal_decisions_identical": decisions_match,
            "kept_repaired": res_repaired["kept"],
            "kept_frozen_path": res_frozen["kept"],
            "per_round_tables_bit_identical": tables_bit_identical,
            "max_chosen_score_drift": max(drift) if drift else 0.0,
            "forward_passes_repaired": c_rep["forwards"],
            "forward_passes_frozen_path": c_old["forwards"],
        },
        "claim_2_bf16_reference_cache_is_exact": {
            "cached_vs_recomputed_abs_delta": cache_delta,
            "exact": cache_delta == 0.0,
            "note": ("E8a asserts caching bf16 logits is identical to recomputing "
                     "them because distortion upcasts to float32 either way; "
                     "re-checked here rather than inherited"),
        },
        "claim_3_decision_tolerance": {
            "rounds": rep_rounds,
            "min_margin": min(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
            "meaning": ("a backend perturbing scores by less than min_margin "
                        "cannot change a removal; one perturbing them by more "
                        "can. Compare the GPU run's measured drift against this."),
        },
        "what_this_does_not_establish": (
            "the actual CUDA-versus-CPU drift. Both paths here ran on "
            f"{device}, so the measured drift is zero by construction and is not "
            "evidence about a different backend. That figure must come from the "
            "GPU measurement run and be compared against min_margin above."),
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "claim_3_decision_tolerance"}, indent=2))
    print(f"\nmargins: min {report['claim_3_decision_tolerance']['min_margin']:.6e} "
          f"max {report['claim_3_decision_tolerance']['max_margin']:.6e}")
    print(f"written: {out}")
    if not frozen_rule["holds"]:
        raise SystemExit(
            "THE FROZEN GREEDY RULE CHANGED — argmin or the lowest-index "
            f"tie-break no longer holds: {frozen_rule}. This is a science "
            "change, not a backend change, and no placement comparison can see it.")
    if not decisions_match:
        raise SystemExit("REMOVAL DECISIONS DIFFER — the repair is not a backend change")


if __name__ == "__main__":
    main()

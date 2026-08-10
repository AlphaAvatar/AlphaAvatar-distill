#!/usr/bin/env python3
"""Search for a contribution-guided teacher depth map (E8 treatment).

Iterative greedy removal under a frozen objective: bypass a candidate set of
teacher blocks through the residual path and measure how far the teacher's own
output distribution moves, aggregated equally across the calibration domains.
For 36 -> 28 this is 36+35+...+29 = **260 subset evaluations**, and every one of
them is written to the score table — not just the eight winners.

    PYTHONPATH=src python scripts/training/search_depth_map.py \\
        --calibration artifacts/stage1/e8_calibration_v1 \\
        --student-layers 28 --out artifacts/stage1/e8_depth_search

What this script may and may not decide
---------------------------------------
It decides the depth map, using the primary objective and nothing else. The
tagged diagnostics (`reasoning`, `final_answer`, `think_close`, `eos`,
`tool_close`, `assistant`) are recorded per candidate per round because E6/E6b
showed termination and reasoning move independently — but the selection rule is
preregistered as the primary score with a lowest-index tie-break, so a
diagnostic cannot change the map. Reading one and then adjusting the map would
make the map a choice made on the outcome.

Cost structure
--------------
The intact reference is the same for every candidate in the whole search, so it
is computed once per item and cached; each candidate then costs one ablated
forward per item. That is `n_items * (1 + 260)` passes instead of `260 * 2 *
n_items`. Caching bf16 logits is numerically identical to recomputing them —
`distortion` upcasts to float32 internally either way — which a test asserts
rather than assumes.

Resumable: each completed round is appended to `rounds.jsonl` and replayed on
restart, so a lost pod costs the current round, not the search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from aadistill.init.contribution import (  # noqa: E402
    DistortionSums,
    bypassed_blocks,
    distortion,
    domain_balanced_score,
    expected_evaluations,
    greedy_removal,
)
from aadistill.init.sandwich import depth_span_map  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

SELF_KL_TOLERANCE = 1e-6


def load_calibration(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    items = [json.loads(l) for l in (path / "items.jsonl").open() if l.strip()]
    if not items:
        raise SystemExit(f"{path} contains no calibration items")
    declared = manifest["totals"]["items"]
    if len(items) != declared:
        raise SystemExit(f"{path} holds {len(items)} items, manifest says {declared}")
    domains = manifest["design"]["domains"]
    subtypes = {i["subtype"] for i in items}
    declared_subtypes = {s for subs in domains.values() for s in subs}
    if subtypes != declared_subtypes:
        raise SystemExit(
            f"calibration sub-types {sorted(subtypes)} do not match the declared "
            f"domain membership {sorted(declared_subtypes)}")
    return manifest, items, domains


def prepare(items, device):
    """Token tensors, targets and boolean tag masks, once, on the device."""
    prepared = []
    for it in items:
        ids = torch.tensor(it["ids"], dtype=torch.long, device=device)[None, :]
        n_pred = ids.shape[1] - 1
        if n_pred != it["n_prediction_positions"]:
            raise SystemExit(f"{it['item_id']}: position count disagrees with the manifest")
        tags = {}
        for name, positions in it.get("tags", {}).items():
            m = torch.zeros(n_pred, dtype=torch.bool, device=device)
            idx = [p for p in positions if 0 <= p < n_pred]
            if len(idx) != len(positions):
                raise SystemExit(f"{it['item_id']}: tag {name} has out-of-range positions")
            m[torch.tensor(idx, dtype=torch.long, device=device)] = True
            tags[name] = m
        prepared.append({
            "item_id": it["item_id"], "subtype": it["subtype"],
            "ids": ids, "targets": ids[0, 1:], "tags": tags,
        })
    return prepared


class Searcher:
    def __init__(self, teacher, prepared, domains, *, cache_reference=True,
                 chunk=512):
        self.teacher = teacher
        self.prepared = prepared
        self.domains = domains
        self.chunk = chunk
        self.reference: dict[str, torch.Tensor] = {}
        self.cache_reference = cache_reference
        self.forward_passes = 0
        self.evaluated: dict[frozenset, dict] = {}

    @torch.no_grad()
    def _logits(self, item, skip):
        self.forward_passes += 1
        if not skip:
            return self.teacher(item["ids"]).logits[0, :-1]
        with bypassed_blocks(self.teacher, skip):
            return self.teacher(item["ids"]).logits[0, :-1]

    def reference_logits(self, item):
        cached = self.reference.get(item["item_id"])
        if cached is not None:
            return cached
        ref = self._logits(item, frozenset())
        if self.cache_reference:
            self.reference[item["item_id"]] = ref
        return ref

    @torch.no_grad()
    def evaluate(self, skip) -> dict:
        """Full record for one candidate subset: primary, per-level, diagnostics."""
        skip = frozenset(int(i) for i in skip)
        hit = self.evaluated.get(skip)
        if hit is not None:
            return hit
        per_subtype: dict[str, DistortionSums] = {}
        for item in self.prepared:
            ref = self.reference_logits(item)
            abl = self._logits(item, skip)
            sums = distortion(ref, abl, item["targets"], tags=item["tags"],
                              chunk=self.chunk)
            per_subtype.setdefault(item["subtype"], DistortionSums()).merge(sums)
            del abl
        detail = {k: v.as_dict() for k, v in per_subtype.items()}
        primary, per_domain = domain_balanced_score(
            {k: v["kl"] for k, v in detail.items()}, self.domains)
        # Diagnostics are aggregated the same domain-balanced way so they are
        # comparable to the primary; they still may not select anything.
        diagnostics = {}
        tag_names = sorted({t for d in detail.values() for t in d["tagged"]})
        for tag in tag_names:
            present = {k: v["tagged"][tag]["kl"] for k, v in detail.items()
                       if tag in v["tagged"]}
            covered = {d: subs for d, subs in self.domains.items()
                       if all(s in present for s in subs)}
            if covered:
                value, per_dom = domain_balanced_score(present, covered)
                diagnostics[tag] = {"score": value, "per_domain": per_dom,
                                    "domains_covered": sorted(covered)}
        ce_delta, _ = domain_balanced_score(
            {k: v["ce_delta"] for k, v in detail.items()}, self.domains)
        record = {
            "skip": sorted(skip),
            "primary_kl": primary,
            "per_domain_kl": per_domain,
            "per_subtype": detail,
            "diagnostics": diagnostics,
            "ce_delta_domain_balanced": ce_delta,
        }
        self.evaluated[skip] = record
        return record

    def score(self, skip) -> float:
        return self.evaluate(skip)["primary_kl"]


def self_consistency(searcher) -> dict:
    """The instrument's own noise floor: the reference against a fresh pass.

    A forward pass is deterministic for a fixed input and shape, so this must be
    zero. Measuring it costs one pass per item and is the difference between
    "candidate scores differ" and "candidate scores differ by more than the
    measurement noise" — which is exactly the check the project's evaluator
    audits have caught real defects with.
    """
    worst, total, positions = 0.0, 0.0, 0
    for item in searcher.prepared:
        ref = searcher.reference_logits(item)
        again = searcher._logits(item, frozenset())
        d = distortion(ref, again, item["targets"], chunk=searcher.chunk).as_dict()
        worst = max(worst, abs(d["kl"]))
        total += d["kl"] * d["positions"]
        positions += d["positions"]
        del again
    return {"max_item_kl": worst, "mean_kl": total / positions,
            "tolerance": SELF_KL_TOLERANCE,
            "deterministic": worst <= SELF_KL_TOLERANCE}


def load_search_teacher(args, device):
    """The pinned Hub teacher, or a local directory for a CPU smoke test.

    A local model is loaded without Hub revision resolution and is stamped
    `revision: "local"` in the identity block, so a smoke-test artifact can never
    be mistaken for a real search against the pinned teacher.

    Neither branch needs a tokenizer: the calibration artifact carries token ids,
    which is what pins the tokenizer contract to the frozen calibration set
    rather than to whatever happens to be loadable at search time.
    """
    from aadistill.models.teacher import DTYPES, load_teacher

    candidate = Path(args.teacher)
    local = candidate if candidate.is_absolute() else REPO_ROOT / args.teacher
    if local.is_dir():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(local), dtype=DTYPES[args.dtype]).to(device).eval()
        return model, {
            "model_id": str(args.teacher), "revision": "local",
            "dtype": args.dtype, "device": device,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "config_sha256": sha256_json(model.config.to_diff_dict()),
        }
    model, _, identity = load_teacher(
        args.teacher, args.teacher_revision or None, dtype=args.dtype, device=device)
    return model, identity


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default="artifacts/stage1/e8_calibration_v1")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--student-layers", type=int, default=28)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--no-reference-cache", action="store_true",
                    help="recompute the intact reference per candidate "
                         "(numerically identical, ~2x the forward passes)")
    ap.add_argument("--out", default="artifacts/stage1/e8_depth_search")
    ap.add_argument("--limit-items", type=int, default=0,
                    help="smoke-test knob: use only the first N calibration items")
    args = ap.parse_args()

    calib = (Path(args.calibration) if Path(args.calibration).is_absolute()
             else REPO_ROOT / args.calibration)
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rounds_path = out / "rounds.jsonl"

    manifest, items, domains = load_calibration(calib)
    if args.limit_items:
        keep = {i["subtype"] for i in items}
        items = items[: args.limit_items]
        if {i["subtype"] for i in items} != keep:
            raise SystemExit("--limit-items dropped a whole sub-type; the "
                             "domain-balanced score would silently change meaning")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher, identity = load_search_teacher(args, device)
    teacher.config.use_cache = False
    teacher.eval()
    n_layers = teacher.config.num_hidden_layers
    n_remove = n_layers - args.student_layers
    if n_remove <= 0:
        raise SystemExit(f"nothing to remove: {n_layers} -> {args.student_layers}")

    prepared = prepare(items, device)
    searcher = Searcher(teacher, prepared, domains,
                        cache_reference=not args.no_reference_cache,
                        chunk=args.chunk)

    # The cache is 18 GB for the frozen calibration set, which fits an L40S
    # beside the teacher — but only just. An OOM 45 minutes into a paid search
    # costs more than the cache saves, and recomputing is numerically identical
    # (asserted by tests), so the fallback is automatic and loud rather than a
    # flag the operator has to remember.
    cache_decision = {"requested": searcher.cache_reference}
    if searcher.cache_reference:
        positions = sum(int(p["targets"].shape[0]) for p in prepared)
        need = positions * teacher.config.vocab_size * 2
        cache_decision.update(estimate_bytes=need)
        headroom = None
        if str(device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            headroom = free
            cache_decision.update(device_free_bytes=int(free),
                                  device_total_bytes=int(total))
        print(f"reference logit cache: ~{need / 1e9:.1f} GB "
              f"({positions} positions x {teacher.config.vocab_size} vocab, bf16)"
              + (f"; {headroom / 1e9:.1f} GB free" if headroom else ""),
              flush=True)
        # Two thirds of free memory: the rest holds the ablated logits, the
        # float32 reduction chunks and activations for the next forward pass.
        if headroom is not None and need > 0.66 * headroom:
            searcher.cache_reference = False
            cache_decision["fallback"] = "recompute (cache would not leave room)"
            print("  -> caching disabled; recomputing the reference per candidate "
                  "(identical numbers, ~2x the forward passes)", flush=True)
    cache_decision["used"] = searcher.cache_reference

    print(f"teacher {identity['model_id']}@{identity['revision'][:8]} on {device}; "
          f"{n_layers} -> {args.student_layers} layers, "
          f"{expected_evaluations(n_layers, n_remove)} subset evaluations over "
          f"{len(prepared)} calibration items "
          f"({sum(int(p['targets'].shape[0]) for p in prepared)} positions)",
          flush=True)

    started = time.time()
    noise = self_consistency(searcher)
    print(f"self-consistency: {noise}", flush=True)
    if not noise["deterministic"]:
        raise SystemExit(
            f"the objective is not reproducible on this device (max item KL "
            f"{noise['max_item_kl']:.3e} > {SELF_KL_TOLERANCE:.0e}); a candidate "
            "ranking would be measuring kernel noise")

    intact = searcher.evaluate(frozenset())
    print(f"intact reference CE (domain-balanced sub-type mean): "
          f"{sum(v['ref_ce'] for v in intact['per_subtype'].values()) / len(intact['per_subtype']):.4f}",
          flush=True)

    completed = [json.loads(l) for l in rounds_path.open()] if rounds_path.exists() else []
    if completed:
        print(f"resuming: {len(completed)} round(s) already on disk", flush=True)

    def on_round(record: dict) -> None:
        with rounds_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"round {record['round']}: removed layer {record['chosen']} "
              f"(primary KL {record['chosen_score']:.6f}, "
              f"{record['n_candidates']} candidates, "
              f"{searcher.forward_passes} passes, "
              f"{(time.time() - started) / 60:.1f} min)", flush=True)

    result = greedy_removal(searcher.score, n_layers, n_remove,
                            completed_rounds=completed, on_round=on_round)
    search_seconds = time.time() - started

    # §4: the full score table from every round, not only the eight removed
    # layers. `greedy_removal` records candidate and primary score; the per-domain
    # and per-tag detail is attached here from the memoized evaluations so a later
    # analysis can ask questions the selection rule was not allowed to ask.
    for record in result["rounds"]:
        before = list(record["removed_before"])
        for entry in record["table"]:
            detail = searcher.evaluated.get(frozenset(before + [entry["candidate"]]))
            if detail is None:
                entry["detail"] = "not_reevaluated_on_resume"
                continue
            entry["per_domain_kl"] = detail["per_domain_kl"]
            entry["per_subtype_kl"] = {k: v["kl"]
                                       for k, v in detail["per_subtype"].items()}
            entry["ce_delta_domain_balanced"] = detail["ce_delta_domain_balanced"]
            entry["diagnostics"] = {k: v["score"]
                                    for k, v in detail["diagnostics"].items()}

    # The comparison that makes the result interpretable: the canonical
    # positional map, scored by the same frozen objective. Zero extra design
    # decisions, one extra evaluation.
    positional_kept = [s["representative"] for s in
                       depth_span_map(n_layers, args.student_layers)]
    positional_skip = frozenset(set(range(n_layers)) - set(positional_kept))
    positional = searcher.evaluate(positional_skip)
    chosen = searcher.evaluate(frozenset(result["removed"]))

    report = {
        "artifact": "e8_contribution_guided_depth_map",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "objective": {
            "primary": "forward KL(teacher || teacher-with-S-bypassed) over all "
                       "prediction positions",
            "aggregation": manifest["design"]["aggregation"],
            "domains": domains,
            "selection_rule": "iterative greedy argmin, tie-break on the lower "
                              "layer index; diagnostics never select",
            "rounds": n_remove,
            "subset_evaluations": expected_evaluations(n_layers, n_remove),
        },
        "teacher": identity,
        "dtype": args.dtype,
        "device": device,
        "calibration": {
            "path": str(args.calibration),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "content_sha256": manifest.get("content_sha256"),
            "items": len(prepared),
            "positions": sum(int(p["targets"].shape[0]) for p in prepared),
            "limited_to": args.limit_items or None,
        },
        "self_consistency": noise,
        "intact_reference": {"per_subtype": intact["per_subtype"]},
        "result": {
            "kept_teacher_layers": result["kept"],
            "removed_teacher_layers": result["removed"],
            "removal_order": result["removal_order"],
            "primary_kl": chosen["primary_kl"],
            "per_domain_kl": chosen["per_domain_kl"],
            "diagnostics": chosen["diagnostics"],
        },
        "positional_baseline": {
            "kept_teacher_layers": positional_kept,
            "removed_teacher_layers": sorted(positional_skip),
            "primary_kl": positional["primary_kl"],
            "per_domain_kl": positional["per_domain_kl"],
            "diagnostics": positional["diagnostics"],
        },
        "comparison": {
            "primary_kl_delta_contribution_minus_positional":
                chosen["primary_kl"] - positional["primary_kl"],
            "contribution_map_is_lower_kl":
                chosen["primary_kl"] < positional["primary_kl"],
            "maps_identical": sorted(result["removed"]) == sorted(positional_skip),
        },
        "rounds": result["rounds"],
        "evaluations": result["evaluations"],
        "forward_passes": searcher.forward_passes,
        "search_seconds": round(search_seconds, 1),
        "reference_cached": searcher.cache_reference,
        "reference_cache_decision": cache_decision,
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    report["report_sha256"] = sha256_json(report)
    (out / "depth_search.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "depth_map.json").write_text(json.dumps({
        "kept_teacher_layers": result["kept"],
        "removed_teacher_layers": result["removed"],
        "removal_order": result["removal_order"],
        "teacher": identity,
        "student_layers": args.student_layers,
        "source": "contribution_guided_greedy",
        "search_report_sha256": report["report_sha256"],
        "calibration_content_sha256": manifest.get("content_sha256"),
    }, indent=2) + "\n")

    print(json.dumps(report["result"] | {"positional_baseline_primary_kl":
                                         positional["primary_kl"]}, indent=2))
    print(f"-> {out / 'depth_search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

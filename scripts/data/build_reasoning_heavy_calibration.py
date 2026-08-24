#!/usr/bin/env python3
"""Build calib.reasoning_heavy@v2 from the frozen domain-balanced pool. Zero cost.

    PYTHONPATH=src python scripts/data/build_reasoning_heavy_calibration.py \
        --out artifacts/stage1/reasoning_heavy_v2

A deterministic with-replacement reweighting of the SAME 67-item pool, under the
five-step rule implemented in `aadistill.autoinit.reweight` and stated verbatim in
`REASONING_HEAVY_V2_SAMPLE_RULE`. Nothing is tokenized, downloaded or generated:
every output token is a token already in `artifacts/stage1/e8_calibration_v1`,
which is why this inherits that mixture's leakage proof rather than needing a new
one.

Two approximations exist and both are recorded in the manifest rather than
absorbed:

* `code`'s domain quota 11,953 is unreachable by whole items; it becomes 11,952
  and the position goes to `general`. Max domain deviation 0.70 positions.
* `multihop_qa`'s sub-type quota 7,526 is unreachable; it becomes 7,074, and
  `rag_evidence` absorbs the 452 **inside rag_multihop**, so no domain weight
  moves. This buys 4 of 5 distinct sessions instead of four copies of one.

Outputs (gitignored; the manifest is small enough to commit):

    <out>/items.jsonl     one line per DRAW, so a session drawn twice appears twice
    <out>/manifest.json   quotas, repairs, realization, hashes, provenance
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2_DOMAIN_WEIGHTS,
    REASONING_HEAVY_V2_SAMPLE_RULE, REASONING_HEAVY_V2_SEED,
    REASONING_HEAVY_V2_TOKEN_BUDGET, mixture_content_sha256,
)
from aadistill.autoinit.reweight import (  # noqa: E402
    MAX_SUPPORT, NEAREST, largest_remainder, realize, repair_quotas, summarize,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402


def build(pool: list[dict]) -> dict:
    """Apply R1-R5. Returns everything the manifest needs, plus the draws."""
    by_domain = collections.defaultdict(list)
    by_subtype = collections.defaultdict(list)
    for item in pool:
        by_domain[item["domain"]].append(item)
        by_subtype[item["subtype"]].append(item)
    positions = lambda group: sum(i["n_prediction_positions"] for i in group)  # noqa: E731
    subtypes_of = {d: sorted({i["subtype"] for i in v}) for d, v in by_domain.items()}

    budget = REASONING_HEAVY_V2_TOKEN_BUDGET
    weights = {d: Fraction(int(round(w * 100)), 100)
               for d, w in REASONING_HEAVY_V2_DOMAIN_WEIGHTS.items()}

    # R1 + R2 -- domain level, NEAREST: a domain's deviation distorts the very
    # weights this profile declares, so it is minimized.
    domain_quota, domain_error = largest_remainder(budget, weights)
    domain_quota, domain_repairs = repair_quotas(
        domain_quota, domain_error,
        {d: [i["n_prediction_positions"] for i in by_domain[d]] for d in domain_quota},
        strategy=NEAREST)

    # R3 + R4 -- sub-type level, MAX_SUPPORT: the deviation is absorbed by a
    # sibling and no domain weight moves, so it is spent on session support.
    sub_quota: dict[str, int] = {}
    sub_repairs: list[dict] = []
    for domain in sorted(domain_quota):
        shares = {s: Fraction(positions(by_subtype[s]), positions(by_domain[domain]))
                  for s in subtypes_of[domain]}
        quota, error = largest_remainder(domain_quota[domain], shares)
        quota, repairs = repair_quotas(
            quota, error,
            {s: [i["n_prediction_positions"] for i in by_subtype[s]] for s in quota},
            strategy=MAX_SUPPORT)
        if sum(quota.values()) != domain_quota[domain]:
            raise SystemExit(f"{domain}: sub-type quotas do not sum to the domain quota")
        sub_quota.update(quota)
        sub_repairs += [{**r, "domain": domain} for r in repairs]

    # R5 -- realization, seed-derived tie-break.
    draws: list[dict] = []
    realization: dict[str, dict] = {}
    for subtype in sorted(sub_quota):
        group = sorted(by_subtype[subtype], key=lambda i: i["item_id"])
        ids = [i["item_id"] for i in group]
        sizes = [i["n_prediction_positions"] for i in group]
        counts = realize(ids, sizes, sub_quota[subtype], REASONING_HEAVY_V2_SEED)
        realization[subtype] = {
            **summarize(counts, sizes), "pool_items": len(group),
            "multiplicity": {i: c for i, c in zip(ids, counts) if c},
        }
        for item, count in zip(group, counts):
            for draw in range(count):
                draws.append({**item, "draw_index": draw,
                              "source_profile": DOMAIN_BALANCED_V1.qualified_id})

    # Deterministic file order. The content hash is order-dependent, so this is
    # part of the identity and not a presentation choice.
    draws.sort(key=lambda i: (i["domain"], i["subtype"], i["item_id"], i["draw_index"]))

    realized = sum(i["n_prediction_positions"] for i in draws)
    if realized != budget:
        raise SystemExit(f"realized {realized} positions, not the {budget} budget")

    per_domain = collections.Counter()
    for item in draws:
        per_domain[item["domain"]] += item["n_prediction_positions"]
    return {
        "draws": draws,
        "domain_quota": dict(sorted(domain_quota.items())),
        "domain_repairs": domain_repairs,
        "sub_quota": dict(sorted(sub_quota.items())),
        "sub_repairs": sub_repairs,
        "realization": realization,
        "realized_positions": realized,
        "realized_domain_positions": dict(sorted(per_domain.items())),
        "domain_deviation_positions": {
            d: round(per_domain[d] - float(weights[d]) * budget, 4)
            for d in sorted(per_domain)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stage1/reasoning_heavy_v2")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    # Hash-verified: resolve() re-derives the pool's token content hash, so a
    # drifted pool cannot silently become a different reweighting.
    pool = DOMAIN_BALANCED_V1.resolve(REPO_ROOT)
    built = build(pool)
    draws = built.pop("draws")

    items_path = out / "items.jsonl"
    items_path.write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in draws))
    content = mixture_content_sha256(draws)

    manifest = {
        "schema": "aadistill.calibration_mixture/v2",
        "profile_id": "calib.reasoning_heavy", "version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": DOMAIN_BALANCED_V1.qualified_id,
        "source_content_sha256": DOMAIN_BALANCED_V1.content_sha256,
        "source_items_file_sha256": DOMAIN_BALANCED_V1.items_file_sha256,
        "token_budget": REASONING_HEAVY_V2_TOKEN_BUDGET,
        "domain_weights": dict(sorted(REASONING_HEAVY_V2_DOMAIN_WEIGHTS.items())),
        "seed": REASONING_HEAVY_V2_SEED,
        "sample_rule": REASONING_HEAVY_V2_SAMPLE_RULE,
        "n_draws": len(draws),
        "n_distinct_items": len({i["item_id"] for i in draws}),
        "n_pool_items": len(pool),
        "content_sha256": content,
        "items_file_sha256": sha256_file(items_path),
        "leakage": {
            "inherited_from": DOMAIN_BALANCED_V1.qualified_id,
            "proof": DOMAIN_BALANCED_V1.leakage_proof_path,
            "exclusions": list(DOMAIN_BALANCED_V1.leakage_exclusions),
            "why_inherited": ("every token here is a token of the source mixture, "
                              "drawn with replacement; reweighting a leakage-checked "
                              "pool cannot introduce a leak the pool does not have"),
        },
        **built,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"draws              {len(draws)} over {manifest['n_distinct_items']}"
          f"/{len(pool)} distinct sessions")
    print(f"realized positions {built['realized_positions']}")
    print(f"domain deviation   {built['domain_deviation_positions']}")
    print(f"content_sha256     {content}")
    print(f"items_file_sha256  {manifest['items_file_sha256']}")
    print(f"manifest_sha256    {manifest['manifest_sha256']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

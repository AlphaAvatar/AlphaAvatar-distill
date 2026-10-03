"""How many more disjoint behavioural batteries the prompt pool supports.

    PYTHONPATH=src:scripts/data:scripts python -m experiments.phase_d1.evidence_capacity

Zero cost. It reads the pinned source files from the local Hugging Face snapshot
and the battery assets already on disk. No model, no GPU, no network beyond the
cache.

**Why this exists, and why it is a launch question rather than a curiosity.** The
D-series is an adaptive sequence of three rounds, and each round's promotion
decision has to rest on behavioural evidence the previous rounds did not consume:
D1's confirmation must not be prompts D1's design was tuned against, and D2's and
D3's must not reuse D1's. C0's inferential unit is the **prompt**, and C0 measured
substantial same-prompt cross-seed dependence (ICC `0.25 ± 0.095`;
`P(correct | correct on another seed) = 0.257` against a `0.022` marginal, an
`11.7x` lift) — so disjoint seeds are not enough and disjoint prompts are
required.

Fresh prompts are not free: every battery consumes `150` items from each scorable
stratum, and the five isolation roles — final-promotion battery, state-eval
suite, operator calibration, recovery search, recovery training corpus — have
already consumed from the same pools. So "can D1 have a fresh confirmation
battery?" is an arithmetic question with a definite answer, and asking it costs
seconds while discovering it after a grant costs the grant.

**The exclusion contract is C1's, imported.** `build_c1_confirmation_battery`
owns the mixture, the pinned sources and the five roles; `build_c2_screening_battery`
owns adding an existing battery to the exclusion set. This module restates
neither — it asks them, and then counts what survives. A second copy of the
eligibility rule would answer a different question than the builder would.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

#: The batteries already drawn from these pools. Read for EXCLUSION only; neither
#: is consumed, and neither is re-measured.
DRAWN_BATTERIES: tuple[str, ...] = (
    "artifacts/stage3/c1_confirmation_v1",
    "artifacts/stage3/c2_screening_v1",
)


def capacity(*, battery: str = "artifacts/eval/battery_v2",
             recovery_search: str = "artifacts/stage3/recovery_search_v2",
             sessions: str = "artifacts/stage3/corpus_v2/sessions.jsonl",
             state_eval: str = "artifacts/stage1/state_eval_v1",
             calibration: str = "artifacts/stage1/e8_calibration_v1",
             drawn: tuple[str, ...] = DRAWN_BATTERIES) -> dict[str, Any]:
    """Eligible items and whole batteries remaining, per stratum."""
    from aadistill.data.extra_stream import content_sha256
    from battery_render import RENDERERS, norm, read_rows
    import build_c1_confirmation_battery as c1
    import build_c2_screening_battery as c2

    args = argparse.Namespace(
        out="(unused)", battery=battery, recovery_search=recovery_search,
        sessions=sessions, state_eval=state_eval, calibration=calibration)
    source_ids, prompt_hashes, roles = c1.excluded_identities(args)
    baseline = {"source_ids": len(source_ids),
                "prompt_hashes": len(prompt_hashes),
                "roles": sorted(roles)}
    drawn_detail = {}
    for rel in drawn:
        drawn_detail[rel] = c2.exclude_c1_confirmation(
            rel, source_ids, prompt_hashes)["n_prompts"]

    strata: dict[str, Any] = {}
    for stratum, (domain, take, scorable) in c1.SETS.items():
        repo_id, revision, relpath = c1.SOURCES[stratum]
        rows = read_rows(repo_id, revision, relpath)
        render = RENDERERS[stratum]
        eligible = 0
        for index, row in enumerate(rows):
            row = dict(row)
            row["_index"] = index
            item = render(row)
            if item is None:
                continue
            if (str(item["id"]) in source_ids
                    or str(item.get("source_key")) in source_ids):
                continue
            if content_sha256(norm(item["prompt_text"])) in prompt_hashes:
                continue
            eligible += 1
        strata[stratum] = {
            "domain": domain, "scorable": bool(scorable),
            "source": {"repo_id": repo_id, "revision": revision,
                       "file": relpath},
            "pool_rows": len(rows), "eligible": eligible,
            "per_battery": take,
            "batteries_remaining": eligible // take,
            "short_by": max(0, take - eligible),
        }

    binding = min(strata, key=lambda s: strata[s]["batteries_remaining"])
    remaining = strata[binding]["batteries_remaining"]
    return {
        "schema": "aadistill.autoinit.battery_evidence_capacity/v1",
        "mixture": {k: v[1] for k, v in sorted(c1.SETS.items())},
        "exclusions": {"baseline": baseline, "drawn_batteries": drawn_detail,
                       "total_source_ids": len(source_ids),
                       "total_prompt_hashes": len(prompt_hashes)},
        "strata": dict(sorted(strata.items())),
        "binding_stratum": binding,
        "batteries_remaining": remaining,
        "_what_remaining_means": (
            "whole further batteries drawable under the FROZEN C1 mixture with "
            "no prompt shared, by id or by normalized content, with any "
            "isolation role or any battery already drawn. Zero means the next "
            "fresh battery cannot be built without changing the mixture or "
            "extending a source."),
        "_why_the_mixture_cannot_simply_shrink": (
            "`correct_overall` and its SESOI are DEFINED ON the mixture, so a "
            "battery with a different stratum balance measures a different "
            "quantity. C2's screening battery preserves C1's mixture exactly "
            "for that reason: a screening delta only informs a confirmation "
            "delta if both are means over the same distribution. Changing it is "
            "a C0-level redesign, not a parameter tweak."),
        "_authorizes": "nothing",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="write the report as JSON to this path")
    args = ap.parse_args()

    doc = capacity()
    print("Behavioural-battery evidence capacity, under the frozen C1 mixture\n")
    excl = doc["exclusions"]
    print(f"  exclusions: {excl['total_source_ids']} ids / "
          f"{excl['total_prompt_hashes']} prompt hashes")
    print(f"    five isolation roles : {excl['baseline']['source_ids']} ids "
          f"({', '.join(excl['baseline']['roles'])})")
    for rel, n in excl["drawn_batteries"].items():
        print(f"    + {rel.split('/')[-1]:22s} {n} prompts")
    print()
    print(f"  {'stratum':16s} {'pool':>6s} {'eligible':>9s} {'per batt':>9s} "
          f"{'batteries left':>15s}")
    for name, row in doc["strata"].items():
        print(f"  {name:16s} {row['pool_rows']:>6d} {row['eligible']:>9d} "
              f"{row['per_battery']:>9d} {row['batteries_remaining']:>15d}"
              + ("   <- BINDING" if name == doc["binding_stratum"] else ""))
    print()
    print(f"  BATTERIES REMAINING: {doc['batteries_remaining']}, bound by "
          f"{doc['binding_stratum']}")
    if doc["batteries_remaining"] < 1:
        short = doc["strata"][doc["binding_stratum"]]
        print(f"  {doc['binding_stratum']} is short by {short['short_by']} "
              f"eligible items of the {short['per_battery']} a battery needs.")
        print("  NO FURTHER FRESH BATTERY CAN BE BUILT under this mixture.")
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True)
                                  + "\n")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

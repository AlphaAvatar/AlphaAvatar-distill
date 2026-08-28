#!/usr/bin/env python3
"""What the Phase-B behavioural continuation still owes. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/price_behavioural_continuation.py \
        --out logs/autoinit_behavioural_continuation_pricing.json

**Stage 1 is complete and must not be repurchased.** Attempt 5 emitted an
authoritative Top-5, a durable Stage-1 selection artifact and a retained journal,
and all five checkpoints are on the dev box with re-derived identities. The
`$35.6660` full-Phase-B ceiling prices a 16.5 h P=2 search that has already been
bought. Pricing it again would fund the same work twice.

What remains is a **behavioural continuation**: bring three initializations to a
pod, run the probes that are genuinely missing, and finish the rungs.

After identity collapse the universe is six distinct candidates; rung 1 —
computed by the frozen selection code, not chosen here — advances
`fe9683e6a9c7`, `85bde4ded2c3` and the auto-advancing control. Their evidence:

    candidate        sa       sb        sc
    fe9683e6a9c7     cited    MISSING   MISSING
    85bde4ded2c3     cited    cited     cited
    control-qwen     cited    cited     MISSING

So **one `sb`** is owed, and **at most two `sc`** if the tie-break fires. One to
three probes, against the ten a full Phase B books.

`sc` is conditional by construction: it runs only for candidates inside the frozen
equivalence interval that lack a verified `sc`. Both ends are priced, and the
ceiling takes the worst case — it is a bound, not a forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from price_phase_b import probe_cost  # noqa: E402
from aadistill.autoinit.cost import L40S_MEASURED  # noqa: E402

HISTORICAL = REPO_ROOT / "logs/autoinit_historical_probe_reuse.json"
ATTEMPT5 = REPO_ROOT / "logs/autoinit_attempt5_probe_reuse.json"
AMENDMENT = REPO_ROOT / "logs/autoinit_phase_b_identity_collapse_amendment.json"

#: Measured, not chosen: the slowest setup across attempts 3, 4 and 5 was
#: attempt 5's 21.9 min (attempt 4 took 7.3). A bound takes the slowest observed.
SETUP_MINUTES = 22.0
#: Stage 0 across attempts 3/4/5: 3.15, 2.2, 3.0 min. Rounded up.
STAGE0_MINUTES = 5.0
#: The one leaf that is NOT already on the relay is `fe9683e6a9c7`. Uploading it
#: dev-box -> relay costs `$0` because no pod is running; the pod then pulls it
#: from the relay, which is fast. This prices the POD-side staging only, and the
#: `$0` prerequisite is recorded as a launch step rather than hidden in a rate.
STAGING_MINUTES = 8.0
#: Selection, report and artifact collection after the last probe.
CLOSEOUT_MINUTES = 12.0
#: Same reserve the full session carries, for getting products off a pod that is
#: about to be deleted.
ARTIFACT_RECOVERY_MINUTES = 20.0
#: Measured probe cost, unchanged from the full-session pricing.
PROBE_TRAIN_MINUTES = 61.55
PROBE_BATTERY_MINUTES = 9.82
CONTINGENCY = 0.10
SETUP_RESERVE_USD = 3.00


def evidence() -> dict:
    have = set(json.loads(HISTORICAL.read_text())["admitted_reusable_probes"])
    fresh = json.loads(ATTEMPT5.read_text())
    if not fresh.get("reuse_verified"):
        raise SystemExit(
            "refusing to price: the Attempt-5 probe reconstruction is unverified "
            f"({fresh.get('failures')}). Run verify_attempt5_probe_reuse.py first; "
            "a continuation that cannot cite those three sa probes is a different, "
            "larger session.")
    return {"historical": have, "attempt5": set(fresh["reusable_probes"]),
            "all": have | set(fresh["reusable_probes"])}


def price(hardware=L40S_MEASURED) -> dict:
    ev = evidence()
    amendment = json.loads(AMENDMENT.read_text())
    rung1 = amendment["rung1_selection"]
    survivors = [s[:12] for s in rung1["selected_searched"]]
    control = "control-qwen"
    advancing = [*survivors, control]

    missing_sb = [c for c in advancing if f"{c}/sb" not in ev["all"]]
    missing_sc = [c for c in advancing if f"{c}/sc" not in ev["all"]]

    unit = probe_cost(hardware.price_per_hour_usd)
    per_probe = unit["total_usd"]

    fixed_minutes = (SETUP_MINUTES + STAGE0_MINUTES + STAGING_MINUTES
                     + CLOSEOUT_MINUTES)
    probe_minutes = PROBE_TRAIN_MINUTES + PROBE_BATTERY_MINUTES

    def total(n_probes: int) -> dict:
        minutes = fixed_minutes + n_probes * probe_minutes
        with_contingency = minutes * (1.0 + CONTINGENCY) + ARTIFACT_RECOVERY_MINUTES
        return {"probes": n_probes, "minutes": round(minutes, 2),
                "minutes_with_contingency_and_recovery": round(with_contingency, 2),
                "usd": round(hardware.usd_for(with_contingency * 60.0)
                             + SETUP_RESERVE_USD, 4)}

    low = total(len(missing_sb))
    hard = total(len(missing_sb) + len(missing_sc))

    return {
        "schema": "aadistill.autoinit.behavioural_continuation_pricing/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "Prices ONLY the behavioural evidence Phase B still owes. Stage 1 is "
            "complete, retained and authoritative; it is not repriced here and "
            "must not be repurchased. NOT an authorization."),
        "hardware": hardware.as_dict(),
        "stage1": {
            "status": "COMPLETE — not priced here",
            "why": ("attempt 5 emitted search_result.json, an authoritative Top-5 "
                    "and a durable Stage-1 selection artifact; all five checkpoints "
                    "are retained on the dev box with re-derived identities"),
            "selection_sha256": amendment["stage1_selection"]["selection_sha256"]},
        "universe": {
            "distinct_candidates": amendment["collapsed_universe"]["distinct_candidates"],
            "universe_identity": amendment["collapsed_universe"]["universe_identity"],
            "amendment_sha256": amendment["amendment_sha256"]},
        "rung1": {"computed_by": rung1["computed_by"],
                  "selected_searched": rung1["selected_searched"],
                  "auto_advanced_control": rung1["auto_advanced_control"]},
        "evidence": {
            "cited_historical": sorted(ev["historical"]),
            "cited_attempt5": sorted(ev["attempt5"]),
            "missing_sb": missing_sb,
            "missing_sc_worst_case": missing_sc,
            "note": ("sc is CONDITIONAL: it runs only for candidates inside the "
                     "frozen equivalence interval that lack a verified sc. The "
                     "ceiling assumes it fires for all of them.")},
        "probe": {**unit, "train_minutes": PROBE_TRAIN_MINUTES,
                  "battery_minutes": PROBE_BATTERY_MINUTES},
        "fixed_minutes": {"setup": SETUP_MINUTES, "stage0": STAGE0_MINUTES,
                          "staging": STAGING_MINUTES, "closeout": CLOSEOUT_MINUTES,
                          "artifact_recovery_reserve": ARTIFACT_RECOVERY_MINUTES,
                          "contingency_fraction": CONTINGENCY},
        "prerequisite_at_zero_cost": (
            "`fe9683e6a9c7` is the only advancing checkpoint not already on the "
            "relay. Upload it dev-box -> relay BEFORE launching: no pod is running, "
            "so the 0.72 MB/s uplink costs nothing, and the pod then pulls it fast. "
            "Doing it with a pod up would bill ~28 min of L40S for a transfer that "
            "is free."),
        "total": {
            "low_usd": low["usd"], "low_probes": low["probes"],
            "hard_usd": hard["usd"], "hard_probes": hard["probes"],
            "low_minutes": low["minutes_with_contingency_and_recovery"],
            "hard_minutes": hard["minutes_with_contingency_and_recovery"],
            "note": ("low = the tie-break never fires; hard = it fires for every "
                     "advancing candidate lacking a verified sc. The HARD figure is "
                     "the authorization ceiling. Neither is an expectation: no "
                     "expected-value assumption over tie-break probability is "
                     "defined anywhere.")},
        "comparison": {
            "full_phase_b_ceiling_usd": 35.6660,
            "why_not_that": ("that ceiling books a 16.5 h P=2 search and ten probes. "
                             "The search is done and eight of those probes are "
                             "citable.")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_behavioural_continuation_pricing.json")
    args = ap.parse_args()
    r = price()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(r, indent=2) + "\n")
    t, e = r["total"], r["evidence"]
    print(f"distinct candidates  {r['universe']['distinct_candidates']}")
    print(f"advancing            {r['rung1']['selected_searched']} + control")
    print(f"missing sb           {e['missing_sb']}")
    print(f"missing sc worst     {e['missing_sc_worst_case']}")
    print(f"per probe            ${r['probe']['total_usd']:.4f}")
    print(f"LOW   {t['low_probes']} probe(s)  {t['low_minutes']:.1f} min  ${t['low_usd']:.4f}")
    print(f"HARD  {t['hard_probes']} probe(s)  {t['hard_minutes']:.1f} min  ${t['hard_usd']:.4f}"
          f"   <- continuation ceiling")
    print(f"(full Phase-B ceiling was ${r['comparison']['full_phase_b_ceiling_usd']:.4f})")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

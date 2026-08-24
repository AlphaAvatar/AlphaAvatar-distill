#!/usr/bin/env python3
"""Price the paid work Phase B actually still owes. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/price_phase_b.py \
        --out logs/autoinit_phase_b_pricing.json

Phase B is not a fresh Phase A. Its terminal procedure is a **cross-phase**
behavioural selection: the P=2 search's Top-5 leaves compete against the two
retained Phase-A finalists and the canonical initialization control, and the
historical Phase-A probes are reused wherever strict reconstruction proves the
same materialized recovery protocol and seed. So the paid probe count is the
candidate set MINUS what has already been observed, and pricing it any other way
would bill for evidence this project already owns.

What is reused is therefore not an assumption here: the inventory is read off the
probe records in `logs/autoinit_recovery_continuation_attempt7/probes/`, and a
candidate/seed pair counts as observed only if its record is on disk. Reuse
remains **conditional** on the strict reconstruction check at run time — this
script prices the best case for reuse and reports the worst case beside it.

Every anchor is the one `plan_search.py` already uses; none is re-derived here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from plan_search import (  # noqa: E402
    ADAPTER, CALIBRATION_SEQ_LEN, CALIBRATION_TOKENS, COMPOSITE, DECOMPOSED,
    TARGET, TEACHER, probe_cost,
)

from aadistill.autoinit.cost import L40S_MEASURED, price_search  # noqa: E402
from aadistill.autoinit.ranking import SCHEDULE_V1  # noqa: E402

#: Attempt 7's probe records — the only Phase-A behavioural evidence that exists.
PROBES = REPO_ROOT / "logs/autoinit_recovery_continuation_attempt7/probes"

#: The reviewer's terminal procedure, 2026-08-25.
PHASE_B_SEARCHED_LEAVES = 5        # Top-5 admitted from the P=2 search
PHASE_A_FINALISTS = ("cca699c93f34", "85bde4ded2c3")
CONTROL = "control-qwen"
SURVIVORS_AT_SB = 2                # best two searched candidates, globally

#: Phase A's setup/redraw reserve, reused unchanged. Pod setup has varied 30x
#: across this project's sessions and is covered only by this.
SETUP_RESERVE_USD = 3.00

PROBE_RE = re.compile(r"^autoinit\.v1\.phase_a\.rung(\d)\.([^.]+)\.(s[abc])\.json$")


def observed_probes(root: Path = PROBES) -> dict[str, set[str]]:
    """{candidate -> {seeds observed}}, read off disk rather than assumed."""
    seen: dict[str, set[str]] = {}
    if not root.is_dir():
        raise SystemExit(f"no probe records at {root}; cannot price reuse")
    for path in sorted(root.iterdir()):
        m = PROBE_RE.match(path.name)
        if m:
            seen.setdefault(m.group(2), set()).add(m.group(3))
    return seen


def price(hardware=L40S_MEASURED) -> dict:
    seen = observed_probes()
    unit = probe_cost(hardware.price_per_hour_usd)

    # --- the search: a full fresh joint P=2 beam, no Phase-A leaf reuse -------
    search = price_search(
        TEACHER, TARGET, ADAPTER, DECOMPOSED,
        calibration_tokens=CALIBRATION_TOKENS, suite_tokens=CALIBRATION_TOKENS,
        seq_len=CALIBRATION_SEQ_LEN, n_profiles=2, beam_width=SCHEDULE_V1.width,
        warmup_levels=SCHEDULE_V1.warmup_levels, hardware=hardware,
        composite=COMPOSITE).as_dict()

    # --- sa: every candidate, minus those already observed on sa -------------
    priors = [*PHASE_A_FINALISTS, CONTROL]
    sa_reusable = [c for c in priors if "sa" in seen.get(c, ())]
    sa_missing = PHASE_B_SEARCHED_LEAVES + (len(priors) - len(sa_reusable))

    # --- sb and sc are driven by WHICH candidates survive sa, which is not
    # knowable before the run. The two ends must be internally coherent: pricing
    # sb as if the survivors were new AND sc as if they were the priors mixes two
    # incompatible worlds and understates the bill.
    control_sb = "sb" in seen.get(CONTROL, ())
    control_sc = "sc" in seen.get(CONTROL, ())
    priors_with_sc = [c for c in PHASE_A_FINALISTS if "sc" in seen.get(c, ())]

    # Best: both survivors are the Phase-A finalists, and no tie-break fires.
    sb_missing_best = 0 if control_sb else 1
    sc_missing_best = 0

    # Worst: both survivors are new Phase-B leaves, and the tie-break fires over
    # the two survivors plus the control — none of which has an sc on record.
    sb_missing_worst = SURVIVORS_AT_SB + (0 if control_sb else 1)
    sc_missing_worst = SURVIVORS_AT_SB + (0 if control_sc else 1)

    lo_probes = sa_missing + sb_missing_best + sc_missing_best
    hi_probes = sa_missing + sb_missing_worst + sc_missing_worst
    lo = search["usd_low"] + lo_probes * unit["total_usd"] + SETUP_RESERVE_USD
    hi = search["usd_high"] + hi_probes * unit["total_usd"] + SETUP_RESERVE_USD

    return {
        "schema": "aadistill.autoinit.phase_b_pricing/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": ("Prices the paid work Phase B still owes under the "
                      "2026-08-25 reviewer procedure. NOT an authorization, and "
                      "not a grant: the binding per-launch ceiling is issued by "
                      "the authorization code against a frozen plan."),
        "hardware": hardware.as_dict(),
        "procedure": {
            "search": "full fresh joint P=2 beam; Phase-A leaves are NOT reused "
                      "to restrict the space",
            "candidates_at_sa": {
                "phase_b_searched_leaves": PHASE_B_SEARCHED_LEAVES,
                "phase_a_finalists": list(PHASE_A_FINALISTS),
                "control": CONTROL,
                "total": PHASE_B_SEARCHED_LEAVES + len(priors),
            },
            "survivors_at_sb": SURVIVORS_AT_SB,
            "fourth_seed": "never",
        },
        "observed_probes": {k: sorted(v) for k, v in sorted(seen.items())},
        "scenarios": {
            "best": ("both sb survivors are the Phase-A finalists (their sb is on "
                     "record) and no tie-break fires"),
            "worst": ("both sb survivors are new Phase-B leaves, and the tie-break "
                      "fires over those two plus the control -- none of which has "
                      "an sc on record"),
        },
        "reuse": {
            "sa_reusable": sorted(sa_reusable),
            "priors_with_sc": sorted(priors_with_sc),
            "control_sb_on_record": control_sb,
            "control_sc_on_record": control_sc,
            "conditional_on": ("strict reconstruction proving the same "
                               "materialized recovery protocol and seed; a "
                               "candidate that fails it is re-run and priced "
                               "at the worst case"),
            "not_admitted": ("the other three Phase-A leaves (158b96cf, "
                             "281a02c3, 4e429f7e) also hold sa probes and are "
                             "retained off-pod; the procedure admits only the "
                             "two finalists, so their sa evidence goes unused"),
        },
        "probes": {
            "sa_missing": sa_missing,
            "sb_missing_low": sb_missing_best, "sb_missing_high": sb_missing_worst,
            "sc_missing_low": sc_missing_best, "sc_missing_high": sc_missing_worst,
            "total_low": lo_probes, "total_high": hi_probes,
            "per_probe_usd": round(unit["total_usd"], 4),
            "per_probe_hours": round(unit["hours"], 4),
        },
        "search": {
            "usd_low": search["usd_low"], "usd_high": search["usd_high"],
            "hours_low": round(search["hours_low"], 3),
            "hours_high": round(search["hours_high"], 3),
            "states_min": search["branching"]["states_materialized_min"],
            "states_max": search["branching"]["states_materialized_max"],
            "leaves_min": search["branching"]["leaves_min"],
            "leaves_max": search["branching"]["leaves_max"],
            "peak_storage_gib_working": round(search["peak_storage_gib_working"], 2),
            "total_gib_written": round(search["total_gib_written"], 2),
            "peak_resident_gib": round(search["peak_resident_gib"], 2),
            "provision_container_disk_gib": 300,
        },
        "setup_reserve_usd": SETUP_RESERVE_USD,
        "total": {
            "expected_usd": round(lo, 4), "hard_usd": round(hi, 4),
            "note": ("expected = search low + best-case reuse; hard = search high "
                     "+ no reuse beyond sa. The range is dominated by the "
                     "activation-statistics GPU/CPU split, still unmeasured."),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_b_pricing.json")
    args = ap.parse_args()
    result = price()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.write_text(json.dumps(result, indent=2) + "\n")

    p, t, s = result["probes"], result["total"], result["search"]
    print(f"search P=2         ${s['usd_low']:.4f} - ${s['usd_high']:.4f}   "
          f"{s['hours_low']:.2f}-{s['hours_high']:.2f} h, {s['peak_storage_gib_working']} GiB working")
    print(f"probes still owed  {p['total_low']}-{p['total_high']} "
          f"(sa {p['sa_missing']}, sb {p['sb_missing_low']}-{p['sb_missing_high']}, "
          f"sc {p['sc_missing_low']}-{p['sc_missing_high']}) @ ${p['per_probe_usd']}")
    print(f"setup reserve      ${result['setup_reserve_usd']:.2f}")
    print(f"TOTAL              expected ${t['expected_usd']:.4f}   hard ${t['hard_usd']:.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

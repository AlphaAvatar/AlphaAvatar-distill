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

Reuse is **not** assumed from the existence of a file. It is taken from
`logs/autoinit_historical_probe_reuse.json`, the strict reconstruction record
produced by `verify_historical_probe_reuse.py`, and this script **fails closed**
if that record is missing, unverified, or describes a different set of probe
bytes than the ones on disk now.

One leg of reuse cannot be closed at `$0`: Phase B's own runtime must be
comparable to Phase A's under `generation_runtime_comparability@v2`, and Phase
B's runtime does not exist yet.

**That gate terminates the session; it does not trigger a bigger one.**
Comparability is what lets behavioural results be judged against the *frozen*
Stage-3 feasibility floor and equivalence interval. If it fails at Stage 0, those
thresholds do not describe anything this session could produce, so re-running all
eight candidates would be a differently-thresholded experiment wearing Phase B's
name — and would not restore the interval either. The 14-probe figure is
therefore a **rejected counterfactual**, retained for comparison only. The
authorization ceiling is the verified-reuse hard case.

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

#: The strict reconstruction record this pricing is conditional on.
REUSE_RECORD = REPO_ROOT / "logs/autoinit_historical_probe_reuse.json"


def observed_probes(root: Path = PROBES) -> dict[str, set[str]]:
    """{candidate -> {seeds observed}} — what EXISTS, which is not what is reusable."""
    seen: dict[str, set[str]] = {}
    if not root.is_dir():
        raise SystemExit(f"no probe records at {root}; cannot price reuse")
    for path in sorted(root.iterdir()):
        m = PROBE_RE.match(path.name)
        if m:
            seen.setdefault(m.group(2), set()).add(m.group(3))
    return seen


def _repo_relative(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise. A record that names a
    path outside the tree is a fact worth printing, not a crash."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def verified_reuse() -> dict:
    """The reuse verdict, or a refusal. Never a default.

    Pricing against unverified reuse is how a session gets funded for five probes
    and then owes eight, so every failure path here raises instead of degrading
    to the pessimistic case: a silent fallback would look like a priced plan.
    """
    from verify_historical_probe_reuse import probes_dir_digest

    if not REUSE_RECORD.is_file():
        raise SystemExit(
            f"{REUSE_RECORD.name} is missing. Run "
            "scripts/autoinit/verify_historical_probe_reuse.py first; reuse may "
            "not be priced from the existence of probe files.")
    rec = json.loads(REUSE_RECORD.read_text())
    if not rec.get("reuse_verified"):
        raise SystemExit(
            f"{REUSE_RECORD.name} reports reuse_verified=false "
            f"({rec.get('failures')}); re-verify or price the no-reuse scenario "
            "explicitly.")
    live = probes_dir_digest()
    if rec.get("probes_dir_digest") != live:
        raise SystemExit(
            f"{REUSE_RECORD.name} verified probe bytes {str(rec.get('probes_dir_digest'))[:12]} "
            f"but the directory now hashes to {live[:12]}; re-run the verifier.")
    return rec


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

    # --- reuse comes from the strict reconstruction record, not from `seen` ----
    reuse = verified_reuse()
    verified = set(reuse["admitted_reusable_probes"])          # "candidate/seed"

    def has(candidate: str, seed: str) -> bool:
        return f"{candidate}/{seed}" in verified

    priors = [*PHASE_A_FINALISTS, CONTROL]
    n_candidates = PHASE_B_SEARCHED_LEAVES + len(priors)

    # --- sa: every candidate, minus the priors whose sa is VERIFIED reusable ---
    sa_reusable = [c for c in priors if has(c, "sa")]
    sa_missing = n_candidates - len(sa_reusable)

    control_sb, control_sc = has(CONTROL, "sb"), has(CONTROL, "sc")
    priors_with_sc = [c for c in PHASE_A_FINALISTS if has(c, "sc")]

    # Best: reuse holds, both sb survivors are the Phase-A finalists, no tie-break.
    sb_missing_best = 0 if control_sb else 1
    sc_missing_best = 0

    # Worst WITH reuse: both survivors are new Phase-B leaves, and the tie-break
    # fires over those two plus the control. Coherent: one world, not a mix.
    sb_missing_worst = SURVIVORS_AT_SB + (0 if control_sb else 1)
    sc_missing_worst = SURVIVORS_AT_SB + (0 if control_sc else 1)

    # The REJECTED counterfactual: what a no-reuse run would have cost. It is not
    # an executable path -- Stage 0 fails closed and terminates -- and it is not
    # the authorization ceiling. Kept because "we considered it and rejected it"
    # and "we never priced it" are different statements.
    nr_sa = n_candidates
    nr_sb = SURVIVORS_AT_SB + 1
    nr_sc = SURVIVORS_AT_SB + 1

    lo_probes = sa_missing + sb_missing_best + sc_missing_best
    hi_probes = sa_missing + sb_missing_worst + sc_missing_worst
    nr_probes = nr_sa + nr_sb + nr_sc
    lo = search["usd_low"] + lo_probes * unit["total_usd"] + SETUP_RESERVE_USD
    hi = search["usd_high"] + hi_probes * unit["total_usd"] + SETUP_RESERVE_USD
    nr = search["usd_high"] + nr_probes * unit["total_usd"] + SETUP_RESERVE_USD

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
            "low": ("reuse holds; both sb survivors are the Phase-A finalists, whose "
                    "sb is verified; no tie-break fires. A FLOOR, not an expectation: "
                    "no expected-value assumption over survivor identity or tie-break "
                    "probability is defined anywhere, so this is not an 'expected' cost"),
            "hard_with_reuse": ("reuse holds; both sb survivors are new Phase-B leaves "
                                "and the tie-break fires over those two plus the "
                                "control, none of which has a verified sc. THIS is the "
                                "authorization ceiling"),
            "rejected_counterfactual_no_reuse": (
                "what a run that cited no historical evidence would have cost. NOT an "
                "executable path: if Stage-0 comparability fails the session TERMINATES "
                "before any search or probe, because the frozen feasibility floor and "
                "equivalence interval would not describe anything it could produce. "
                "Priced only so the rejection is on the record"),
        },
        "reuse": {
            "source": _repo_relative(REUSE_RECORD),
            "verified": True,
            "probes_dir_digest": reuse["probes_dir_digest"],
            "admitted_reusable_probes": sorted(verified),
            "verifiable_but_not_admitted": reuse.get("verifiable_but_not_admitted", []),
            "sa_reusable": sorted(sa_reusable),
            "priors_with_sc": sorted(priors_with_sc),
            "control_sb_verified": control_sb,
            "control_sc_verified": control_sc,
            "open_precondition": reuse["open_precondition"],
            "not_admitted": ("the other three Phase-A leaves (158b96cf, "
                             "281a02c3, 4e429f7e) also hold sa probes and are "
                             "retained off-pod; the procedure admits only the "
                             "two finalists, so their sa evidence goes unused"),
        },
        "probes": {
            "sa_missing": sa_missing,
            "sb_missing_low": sb_missing_best, "sb_missing_high": sb_missing_worst,
            "sc_missing_low": sc_missing_best, "sc_missing_high": sc_missing_worst,
            "no_reuse_sa": nr_sa, "no_reuse_sb": nr_sb, "no_reuse_sc": nr_sc,
            "total_low": lo_probes, "total_high": hi_probes,
            "total_no_reuse": nr_probes,
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
            "low_usd": round(lo, 4),
            "hard_with_reuse_usd": round(hi, 4),
            "authorization_ceiling_usd": round(hi, 4),
            "rejected_counterfactual_no_reuse_usd": round(nr, 4),
            "note": ("low is a FLOOR, not an expectation -- no expected-value "
                     "assumption is defined anywhere. The AUTHORIZATION CEILING is "
                     "hard_with_reuse: Stage-0 comparability is a fail-closed "
                     "terminate, not a switch to a larger run, so the no-reuse "
                     "figure is a rejected counterfactual rather than a bound to "
                     "fund. The spread within each scenario is dominated by the "
                     "activation-statistics GPU/CPU split, still unmeasured and "
                     "deliberately NOT to be measured by a separate paid session."),
        },
        "comparability_gate": {
            "rule": "generation_runtime_comparability@v2",
            "on_fail": "TERMINATE at stage 0, before any search or probe",
            "not_a_fallback": ("do not respond to comparability failure by rerunning "
                               "all eight candidates, and do not rematerialize the "
                               "Stage-3 controls or redefine thresholds"),
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
    print(f"search P=2          ${s['usd_low']:.4f} - ${s['usd_high']:.4f}   "
          f"{s['hours_low']:.2f}-{s['hours_high']:.2f} h, {s['peak_storage_gib_working']} GiB working")
    print(f"reuse               VERIFIED: {len(result['reuse']['admitted_reusable_probes'])} probes "
          f"({', '.join(result['reuse']['admitted_reusable_probes'])})")
    print(f"probes owed         low {p['total_low']}  hard {p['total_high']}  "
          f"(rejected no-reuse counterfactual {p['total_no_reuse']})  @ ${p['per_probe_usd']}")
    print(f"setup reserve       ${result['setup_reserve_usd']:.2f}")
    print(f"TOTAL               low ${t['low_usd']:.4f} (a floor)   "
          f"AUTHORIZATION CEILING ${t['authorization_ceiling_usd']:.4f}")
    print(f"                    rejected no-reuse counterfactual "
          f"${t['rejected_counterfactual_no_reuse_usd']:.4f} -- stage 0 terminates instead")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

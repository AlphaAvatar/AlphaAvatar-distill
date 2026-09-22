#!/usr/bin/env python3
"""Price the C2 behavioural continuation from the PRODUCTION path. Zero cost.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/price_c2_behavioural_continuation.py \
        --restore-mb-per-second 0.68 [--backend-usd 0] [--write]

Exists because a continuation price was reported as `$18.7315` hard when that
was the GPU component alone. `continuation_all_in_usd` already adds the
separately billed container disk, and a figure typed beside the code instead of
read out of it is a figure that can disagree with the gate that will actually
refuse. Every number here is derived by calling the same functions the launcher
calls; nothing is restated.

AUTHORIZES NOTHING. It prices; it does not permit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_continuation as BC  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402

import autoinit_c2_behavioural_launch as L  # noqa: E402

ANALYSIS = ("logs/stages/stage-1/phase_c2_behavioural/analyses/"
            "continuation_price.json")


class _Ctx:
    """The minimum `continuation_all_in_usd` reads, from the AUTHORIZED terms.

    Not a stub of the money: `authorization_terms` derives the same five
    amounts an issued authorization carries, so the disk rate this prices at
    is the rate the gate will price at.
    """

    def __init__(self, terms: dict) -> None:
        self.auth = type("Auth", (), {
            "rate_usd_per_hour": float(terms["rate_usd_per_hour"]),
            "hard_runtime_minutes": float(terms["hard_runtime_minutes"]),
            "gpu_hard_usd": float(terms["gpu_hard_usd"]),
            "disk_hard_usd": float(terms["disk_hard_usd"]),
            "all_in_hard_usd": float(terms["all_in_hard_usd"]),
            "campaign_all_in_hard_usd": float(
                terms[BG.CAMPAIGN_AMOUNT_FIELD]),
        })()


def settled_campaign_all_in(repo_root: Path) -> dict:
    """What the campaign has already spent, ALL-IN, from the closeouts.

    Read from each attempt's own `money.all_in_usd`, which is the figure the
    project balance uses, rather than summed from GPU actuals.
    """
    root = repo_root / "logs/stages/stage-1/phase_c2_behavioural/runs"
    per_attempt, total = {}, 0.0
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        f = d / "closeout/outcome.json"
        if not f.is_file():
            continue
        doc = json.loads(f.read_text())
        money = doc.get("money") or {}
        v = money.get("all_in_usd")
        if v is None:
            v = 0.0 if doc.get("provider_resource_created") is False else None
        if v is None:
            raise SystemExit(
                f"{d.name}: its closeout states no all-in cost and does not "
                "affirm that no provider resource was created, so settled "
                "campaign spend is UNKNOWN and cannot be defaulted to $0")
        per_attempt[d.name] = float(v)
        total += float(v)
    return {"per_attempt": per_attempt, "total_usd": round(total, 4)}


def price(repo_root: Path, *, restore_mb_per_second: float,
          backend_usd: float, rate: float) -> dict:
    store = Path("/home/ecs-user/aad-artifacts/phase_c2_behavioural") / BG.CAMPAIGN_ID
    state = BC.campaign_state(store)
    work = BC.remaining_work(repo_root, state=state)

    #: The restore phase at the rate being priced, through the SAME
    #: decomposition the launcher builds its window from.
    nbytes = int(work["restore"]["bytes"])
    minutes = round(nbytes / (restore_mb_per_second * 1e6) / 60.0, 2)
    d = BH.session_decomposition(
        repo_root, materialization_minutes=work["materialization_minutes"],
        train_and_score_probes=work["n_train_and_score"],
        score_only_probes=work["n_score_only"], restore_minutes=minutes)

    terms = BG.authorization_terms(
        repo_root, rate_usd_per_hour=rate,
        campaign_all_in_hard_usd=BG.CAMPAIGN_ALL_IN_CEILING_USD)
    ctx = _Ctx(terms)

    #: THE PRODUCTION FUNCTION, for both windows. It adds the separately
    #: billed container disk to the GPU; reporting the GPU alone as "all-in"
    #: is the error this tool exists to make impossible.
    hard = L.continuation_all_in_usd(ctx, {"decomposition": d})
    expected = L.continuation_all_in_usd(
        ctx, {"decomposition": {"hard_minutes": d["expected_minutes"]}})

    disk_per_min = (float(terms["disk_hard_usd"])
                    / float(terms["hard_runtime_minutes"]))
    settled = settled_campaign_all_in(repo_root)
    ceiling = float(terms[BG.CAMPAIGN_AMOUNT_FIELD])
    total = round(settled["total_usd"] + hard + backend_usd, 4)

    import subprocess
    proj = json.loads(subprocess.run(
        [sys.executable, str(repo_root / "scripts/consolidate/derive_budget.py"),
         "--json"], capture_output=True, text=True, cwd=str(repo_root),
        env={"PYTHONPATH": str(repo_root / "src"), "PATH": "/usr/bin:/bin"}
    ).stdout)["project"]

    return {
        "schema": "aadistill.autoinit.c2_behavioural_continuation_price/v1",
        "authorizes": "nothing",
        "_derived_by": ("scripts/autoinit/price_c2_behavioural_continuation.py "
                        "-- every figure from the production functions the "
                        "launcher calls, never restated beside them"),
        "transport": {
            "restore_mb_per_second": restore_mb_per_second,
            "restore_bytes": nbytes,
            "restore_gib": round(nbytes / 2**30, 3),
            "restore_minutes": minutes,
            "durable_backend_usd": backend_usd,
        },
        "work_owed": {k: v for k, v in work.items()
                      if k in ("n_probes_remaining", "n_train_and_score",
                               "n_score_only", "arms_needed",
                               "materialization_minutes")},
        "window": {"expected_minutes": d["expected_minutes"],
                   "hard_minutes": d["hard_minutes"]},
        "money": {
            "rate_usd_per_hour": float(terms["rate_usd_per_hour"]),
            "disk_usd_per_minute": round(disk_per_min, 8),
            "expected_gpu_usd": round(d["expected_minutes"] / 60 * rate, 4),
            "expected_disk_usd": round(disk_per_min * d["expected_minutes"], 4),
            "expected_all_in_usd": round(expected, 4),
            "hard_gpu_usd": round(d["hard_minutes"] / 60 * rate, 4),
            "hard_disk_usd": round(disk_per_min * d["hard_minutes"], 4),
            "hard_all_in_usd": round(hard, 4),
            "_all_in_includes": (
                "GPU at the live rate PLUS the separately billed container "
                "disk, derived by autoinit_c2_behavioural_launch :: "
                "continuation_all_in_usd. A GPU-only figure is not an all-in "
                "figure and must not be labelled as one."),
        },
        "campaign": {
            "settled_all_in_usd": settled["total_usd"],
            "settled_per_attempt": settled["per_attempt"],
            "ceiling_all_in_usd": ceiling,
            "settled_plus_hard_plus_backend_usd": total,
            "fits": bool(total <= ceiling),
            "headroom_usd": round(ceiling - total, 4),
        },
        "project": {
            "cumulative_usd": proj["cumulative_spend_usd"],
            "cap_usd": proj["cap_usd"],
            "worst_case_usd": round(
                proj["cumulative_spend_usd"] + hard + backend_usd, 4),
            "fits": bool(proj["cumulative_spend_usd"] + hard + backend_usd
                         <= proj["cap_usd"]),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restore-mb-per-second", type=float, required=True)
    ap.add_argument("--backend-usd", type=float, default=0.0)
    ap.add_argument("--rate", type=float,
                    default=BG.QUOTED_RATE_USD_PER_HOUR)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    out = price(REPO, restore_mb_per_second=a.restore_mb_per_second,
                backend_usd=a.backend_usd, rate=a.rate)
    print(json.dumps(out, indent=1))
    if a.write:
        p = REPO / ANALYSIS
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {ANALYSIS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

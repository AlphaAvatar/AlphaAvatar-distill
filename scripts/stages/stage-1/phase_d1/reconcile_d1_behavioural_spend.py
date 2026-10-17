#!/usr/bin/env python3
"""Reconcile D1 behavioural subruns to ACTUAL ALL-IN spend, from the provider.

WHY THIS EXISTS. `SessionRunner` records `cost.actual_usd` as the GPU price
times elapsed time. That is not what the envelope is denominated in: D1's
accepted hard ceiling decomposes into `gpu_hard_usd` PLUS `disk_hard_usd`,
because RunPod bills container disk separately. Booking a GPU-only figure
against an all-in ceiling understates every 120 GB session and makes the
remaining allowance look larger than it is.

HOW THE ALL-IN FIGURE IS OBTAINED. RunPod's GraphQL exposes no per-pod
transaction or billing history -- `transactions`, `creditCharges` and `billing`
all return HTTP 400 -- so the authoritative provider-side quantity available is
the ACCOUNT BALANCE, which each session records at launch. Consecutive launches
therefore bracket each other: subrun N's actual all-in charge is the balance at
N's launch minus the balance at N+1's launch, and the last subrun is closed by
the balance read now.

    actual_all_in(N) = balance_at_launch(N) - balance_at_launch(N+1)

This is a MEASUREMENT, not the $0.10/GB/month disk model, and it is preferred
for exactly that reason. Its one assumption is stated and checked: no other
provider activity may fall between two consecutive launches. For these four
subruns that holds -- they ran consecutively on 2026-10-10 and an account-wide
query returns zero other resources -- and the residual against the GPU-only
figure is reported per subrun so a reader can see it is disk-shaped rather than
an unexplained charge.

IDEMPOTENT. Re-running writes the same values. `--write` is required to touch
any file; the default prints the reconciliation and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

RUNS_REL = "logs/stages/stage-1/phase_d1/runs"

#: The container-disk model, used ONLY to show that the residual between the
#: GPU-only figure and the measured charge is disk-shaped. It never replaces the
#: measurement. RunPod prices running-pod container disk at $0.10/GB/month.
DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730.0


def disk_usd(gb: float, minutes: float) -> float:
    return gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH * (minutes / 60.0)


def live_balance() -> float:
    from aadistill.infrastructure.provider import RunPodProvider, read_api_key

    key = (os.environ.get("RUNPOD_API_KEY")
           or read_api_key(os.path.expanduser("~/.runpod/config.toml")))
    balance = RunPodProvider(key).account_balance()
    if not balance.known:
        raise SystemExit(f"provider balance unknown: {balance.error}")
    return float(balance.client_balance_usd)


def subruns(root: Path) -> list[dict]:
    """Every behavioural subrun with a closeout, in launch order."""
    out: list[dict] = []
    for session in sorted((root / RUNS_REL).glob("d1_behavioural_*")):
        runtime = session / "runtime/session.json"
        closeout = session / "closeout/outcome.json"
        if not (runtime.is_file() and closeout.is_file()):
            continue
        rt = json.loads(runtime.read_text())
        co = json.loads(closeout.read_text())
        cost = rt.get("cost") or {}
        out.append({
            "run_id": session.name,
            "closeout": closeout,
            "rung": co.get("rung"),
            "balance_at_launch": float(
                (rt.get("account_balance") or {})["client_balance_usd"]),
            "gpu_usd": float(cost["actual_usd"]),
            "minutes": float(cost["elapsed_minutes"]),
            "disk_gb": float(
                (co.get("money") or {}).get("container_disk_gb") or 120),
        })
    #: Launch order is the directory's timestamp order, which is also balance
    #: order; assert it rather than trusting the name, because a decreasing
    #: balance is what makes the bracketing valid.
    for a, b in zip(out, out[1:]):
        if b["balance_at_launch"] > a["balance_at_launch"]:
            raise SystemExit(
                f"balance rose between {a['run_id']} and {b['run_id']} "
                f"({a['balance_at_launch']} -> {b['balance_at_launch']}): the "
                "account was topped up between launches, so consecutive "
                "launches no longer bracket one subrun's charge. Reconcile "
                "these two by hand against provider records.")
    return out


def reconcile(root: Path, closing_balance: float) -> dict:
    rows = subruns(root)
    if not rows:
        raise SystemExit("no behavioural subruns with both runtime and closeout")
    opening = rows[0]["balance_at_launch"]
    for i, row in enumerate(rows):
        nxt = (rows[i + 1]["balance_at_launch"] if i + 1 < len(rows)
               else closing_balance)
        row["all_in_usd"] = round(row["balance_at_launch"] - nxt, 4)
        row["residual_vs_gpu_usd"] = round(row["all_in_usd"] - row["gpu_usd"], 4)
        row["modelled_disk_usd"] = round(
            disk_usd(row["disk_gb"], row["minutes"]), 4)
    return {
        "schema": "aadistill.phase_d1.behavioural_spend_reconciliation/v1",
        "opening_balance_usd": opening,
        "closing_balance_usd": closing_balance,
        "measured_all_in_usd": round(opening - closing_balance, 4),
        "gpu_only_sum_usd": round(sum(r["gpu_usd"] for r in rows), 4),
        "subruns": [{k: v for k, v in r.items() if k != "closeout"}
                    for r in rows],
        "_method": ("consecutive launches bracket each subrun's charge; the "
                    "last is closed by the balance read at reconciliation. "
                    "RunPod exposes no per-pod billing surface."),
        "_rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write money.all_in_usd into each closeout")
    ap.add_argument("--closing-balance", type=float, default=None,
                    help="defaults to the LIVE provider balance")
    ap.add_argument("--out", default=None,
                    help="also write the reconciliation record here")
    args = ap.parse_args()

    closing = (args.closing_balance if args.closing_balance is not None
               else live_balance())
    result = reconcile(REPO, closing)
    rows = result.pop("_rows")

    print(f"opening balance   ${result['opening_balance_usd']:.4f}")
    print(f"closing balance   ${result['closing_balance_usd']:.4f}")
    print(f"MEASURED all-in   ${result['measured_all_in_usd']:.4f}")
    print(f"gpu-only sum      ${result['gpu_only_sum_usd']:.4f}  "
          f"(understates by "
          f"${result['measured_all_in_usd'] - result['gpu_only_sum_usd']:.4f})")
    print()
    print(f"{'run':34} {'gpu':>8} {'all-in':>8} {'resid':>8} {'disk~':>8}"
          f" {'min':>7}")
    for r in rows:
        print(f"{r['run_id']:34} {r['gpu_usd']:8.4f} {r['all_in_usd']:8.4f} "
              f"{r['residual_vs_gpu_usd']:8.4f} {r['modelled_disk_usd']:8.4f} "
              f"{r['minutes']:7.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nrecord -> {args.out}")

    if not args.write:
        print("\n(dry run; pass --write to amend the closeouts)")
        return

    for r in rows:
        doc = json.loads(r["closeout"].read_text())
        money = dict(doc.get("money") or {})
        money["all_in_usd"] = r["all_in_usd"]
        money["all_in_source"] = "provider account-balance bracketing"
        money["all_in_derivation"] = {
            "balance_at_this_launch_usd": r["balance_at_launch"],
            "balance_at_next_read_usd": round(
                r["balance_at_launch"] - r["all_in_usd"], 10),
            "gpu_only_usd": r["gpu_usd"],
            "residual_vs_gpu_usd": r["residual_vs_gpu_usd"],
            "modelled_container_disk_usd": r["modelled_disk_usd"],
            "_why": ("cost.actual_usd is GPU price x elapsed and is NOT "
                     "all-in; the envelope is denominated all-in. RunPod "
                     "exposes no per-pod billing surface, so the account "
                     "balance is the provider-side measurement. The residual "
                     "is disk-shaped and is shown against the model rather "
                     "than replaced by it."),
            "_owner": ("scripts/stages/stage-1/phase_d1/"
                       "reconcile_d1_behavioural_spend.py"),
        }
        doc["money"] = money
        r["closeout"].write_text(json.dumps(doc, indent=1) + "\n")
        print(f"amended {r['closeout'].relative_to(REPO)}")


if __name__ == "__main__":
    main()

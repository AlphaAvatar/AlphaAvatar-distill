#!/usr/bin/env python3
"""Independent provider-level cost watchdog. Run it beside the launcher.

This process is the budget-enforcement layer. RunPod's `--terminate-after` is
demoted to a redundant third layer: it has been the documented last resort since
E4 and has never once been observed to fire — on 2026-08-08 the deadline was
00:28:47 and the pod was still `RUNNING` at 00:34.

Launch it from the launcher immediately after the pod is created, detached from
it, so it outlives whatever happens next:

    setsid nohup python3 scripts/pod/watchdog.py \\
      --pod-id "$POD_ID" --session-start-epoch "$(cat $SCR/pod_start_epoch)" \\
      --price-per-hour 0.99 --hard-minutes 545 --authorized-usd 9.00 \\
      --journal "$SCR/watchdog.jsonl" > "$SCR/watchdog.out" 2>&1 < /dev/null &

It never opens an SSH connection and never reads a pod-side path, so a blocked
launcher, a hung driver, a crashed trainer, a silent orchestrator log and a
failed artifact collection are all invisible to it — and all irrelevant. The
only inputs are the provider control plane and its own clock.

Exit codes: 0 the pod is gone; 4 the pod would not die and a human is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.infrastructure.provider import (  # noqa: E402
    RunPodProvider, SimulatedProvider, read_api_key,
)
from aadistill.infrastructure.watchdog import (  # noqa: E402
    Journal, Watchdog, WatchdogPolicy,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod-id", required=True)
    ap.add_argument("--session-start-epoch", type=float, required=True,
                    help="epoch of the session's FIRST pod create; a cold-host "
                         "redraw must not hand the replacement a fresh meter")
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--hard-minutes", type=float, required=True,
                    help="BudgetPlan.hard_terminate_minutes — soft stop plus "
                         "the artifact-recovery reserve, not the dollar cap")
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--terminate-rounds", type=int, default=6)
    ap.add_argument("--verify-polls", type=int, default=6)
    ap.add_argument("--verify-delay-seconds", type=float, default=15.0)
    ap.add_argument("--once", action="store_true",
                    help="one poll/decide cycle, for cron-style operation")
    ap.add_argument("--simulate", action="store_true",
                    help="run against an in-memory control plane; rehearses "
                         "the thresholds without creating a pod")
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    args = ap.parse_args()

    if args.simulate:
        provider = SimulatedProvider(args.pod_id)
    else:
        key = os.environ.get("RUNPOD_API_KEY") or read_api_key(args.runpod_config)
        provider = RunPodProvider(key)

    policy = WatchdogPolicy(
        pod_id=args.pod_id,
        session_start_epoch=args.session_start_epoch,
        price_per_hour=args.price_per_hour,
        hard_terminate_minutes=args.hard_minutes,
        authorized_usd=args.authorized_usd,
        poll_seconds=args.poll_seconds,
        terminate_rounds=args.terminate_rounds,
        verify_polls=args.verify_polls,
        verify_delay_seconds=args.verify_delay_seconds,
    )
    dog = Watchdog(provider, policy, Journal(args.journal))
    reason = dog.run(max_ticks=1 if args.once else None)
    print(f"watchdog exit: {reason} (journal {args.journal})")
    return 4 if reason == "termination_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

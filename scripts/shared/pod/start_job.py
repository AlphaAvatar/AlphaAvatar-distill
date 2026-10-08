#!/usr/bin/env python3
"""Start a remote driver detached and print its durable job descriptor.

Replaces the launcher line that blocked E6b for 434 minutes:

    $SSH "root@$HOST" "cd /workspace/aad && setsid nohup python driver.py \\
      > /workspace/run.log 2>&1 < /dev/null & disown"

with a call whose return is bounded whether or not the ssh channel closes:

    JOB=$(python3 scripts/pod/start_job.py --host "$HOST" --port "$PORT" \\
            --job-id e7_driver --workdir /workspace/aad \\
            --log /workspace/e7_run.log --status /workspace/e7.status \\
            --env TEACHER_REVISION=$TEACHER_REVISION \\
            --command "/opt/train/bin/python scripts/pod/e7_driver.py --stage all")
    PID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["pid"])' <<<"$JOB")

Exit codes: 0 job confirmed running (or already finished); 3 job could not be
confirmed — the caller must **not** proceed to poll, because polling for a job
that never started is how a pod bills for hours against a dead driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.infrastructure.remote import (  # noqa: E402
    JobSpec, RemoteLaunchError, SSHTarget, start_detached,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--user", default="root")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--job-dir", default="/workspace/jobs")
    ap.add_argument("--log", required=True, help="remote stdout/stderr path")
    ap.add_argument("--status", required=True, help="remote marker file path")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--start-timeout", type=float, default=120.0)
    ap.add_argument("--verify-timeout", type=float, default=60.0)
    ap.add_argument("--descriptor-out", default="",
                    help="also write the descriptor here on the dev box")
    args = ap.parse_args()

    env = {}
    for item in args.env:
        if "=" not in item:
            print(f"--env expects K=V, got {item!r}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        # An empty forwarded variable is worse than an absent one: the pod fails
        # minutes later in a way that reads like a data problem.
        if not v:
            print(f"--env {k} is empty; refusing to forward a blank value",
                  file=sys.stderr)
            return 2
        env[k] = v

    spec = JobSpec(job_id=args.job_id, workdir=args.workdir,
                   command=args.command, job_dir=args.job_dir,
                   log_path=args.log, status_path=args.status,
                   env=env or None)
    target = SSHTarget(args.host, args.port, user=args.user)
    try:
        job = start_detached(target, spec, start_timeout=args.start_timeout,
                             verify_timeout=args.verify_timeout)
    except RemoteLaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    payload = json.dumps(job.as_dict(), indent=2)
    if args.descriptor_out:
        out = Path(args.descriptor_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    print(payload)
    if not job.start_channel_closed:
        print(f"note: the start channel did not close; the job was confirmed "
              f"out of band via {job.confirmed_by}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

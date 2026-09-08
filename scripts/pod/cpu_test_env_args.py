#!/usr/bin/env python
"""Emit the C1 CPU-test environment as `env(1)` arguments, for a shell.

A three-line shim so the pod's `autoinit_preflight_setup.sh` and the dev box's
`simulate_pod_env.sh` consume ONE declaration —
`aadistill.autoinit.cpu_test_env` — instead of each maintaining its own list of
variables and drifting apart. That drift is the whole defect: the launch-bound
diagnostic and the paid pod ran the same command under different environments,
so an exact skip-set comparison would have refused a healthy L40S.

    env $(cpu_test_env_args.py --home DIR) <command>

`env` is deliberate: the scope is one command. Exporting these over setup would
neutralize the teacher download and the CUDA proof, which must happen in the
real environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.runtime.cpu_test_env import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

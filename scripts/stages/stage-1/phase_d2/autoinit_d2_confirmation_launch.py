#!/usr/bin/env python3
"""The D2 SHARED CONFIRMATION launcher. One session, declared completely.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/autoinit_d2_confirmation_launch.py \
        --scr <scr> --session-commit <sha> --bundle <canonical> \
        --run-id d2_confirmation_<utc> [--country-code AU]

The screening launcher's core (`autoinit_d2_screening_launch.py`) with the
confirmation rung bound: the contract is `d2_confirmation.py` (the one
candidate the frozen screening advanced, at its INITIAL checkpoint, x the
three shared confirmation seeds), the grant is `D2ConfirmationAuthorization`,
the price is `d2_budget.py :: steps.d2_confirmation`, and the paths, artifact
specs and pod preflight are the confirmation's. Acquisition, host admission
(the protocol's exact image identity), durability and the `$0` gates are the
same code. The verdict is computed off-pod (`verdict_d2_confirmation.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.phase_d2 import autoinit_d2_screening_launch as CORE  # noqa: E402

#: DECLARED, and `autoinit_preflight_setup.sh` has the matching branch.
SESSION_KIND = "d2_confirmation"
RUNG = CORE.make_rung("confirmation")
TEST_PATHS = RUNG.test_paths
LAYOUT = RUNG.layout
build_parser = CORE.build_parser
host_admission = CORE.host_admission


def spec(args):
    return CORE.spec(args, RUNG)


def relay_inputs(repo_root: Path = REPO_ROOT):
    return CORE.relay_inputs(repo_root, RUNG)


def main(argv: list[str] | None = None) -> int:
    return CORE.main(argv, RUNG)


if __name__ == "__main__":
    raise SystemExit(main())

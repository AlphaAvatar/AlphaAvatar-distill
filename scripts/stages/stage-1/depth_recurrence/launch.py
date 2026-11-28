#!/usr/bin/env python3
"""DMR1 variant 2's sessions (study DMR1-R, revised 2026-11-28; Stage 1 research, Stage 3 recovery): grant, one-use artifacts, launch, closeout.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/depth_recurrence/launch.py {budget|grant|issue|launch|closeout|position|stage-resume} \\
        --kind dmr1_loop_train --arm {depth_loop_kept|depth_loop_merged} ...

One training kind, ``dmr1_loop_train``, on one L40S, funded by a grant of its own and NOT authorized yet: one session
per arm, each training that arm's single unit (seed 20260726) from the training-pre initialization the design FIXES
for it (V2-D: SOB1's depth init; V2-M: variant 1's alpha-merged init), never from another arm's or a recovered final.
The depth_loop_kept session runs the engine validation before any training (`depth_recurrence.engine_check`). Issued
only once `stages.depth_merge.plan` has derived the arm's plan from its recurrent init record. Variant 3 has no kind
until it is revised. The mechanism and the gates are the round's (`stages.depth_merge.session`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from shared import study_engineering as SE  # noqa: E402
from stages.depth_merge import session as S  # noqa: E402

LOOP_KIND = "dmr1_loop_train"
VARIANT = S.Variant(
    key="DMR1-R", plan_id="stage1.depth_recurrence", logs=Path("logs/stages/stage-1/depth_recurrence"),
    session_kind="dmr1_depth_recurrence",
    train_kinds={LOOP_KIND: ("depth_loop_kept", "depth_loop_merged")},
    identity_files=("configs/stages/stage-1/depth_recurrence/design.json",
                    "configs/stages/stage-1/depth_recurrence/recurrent_config.json",
                    "configs/stages/stage-1/depth_recurrence/base_config.json",
                    "configs/stages/stage-1/depth_merge/design.json"),
    artifact_spec="configs/stages/stage-1/depth_recurrence/artifacts.json", label="variant 2")
STUDY = VARIANT.study()


class Authorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``dmr1_depth_recurrence`` branch loads exactly this."""
    STUDY = STUDY


def kind_conditions(kind: str, root: Path = REPO, arm: str | None = None) -> dict:
    return S.kind_conditions(VARIANT, kind, root, arm)


def launch_spec(kind: str, minutes: float = 0.0, resume_from: str | None = None, arm: str | None = None):
    return S.launch_spec(VARIANT, Authorization, kind, minutes, resume_from, arm)


def main(argv=None) -> int:
    return S.main(VARIANT, Authorization, argv, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())

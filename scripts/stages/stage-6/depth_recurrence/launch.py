#!/usr/bin/env python3
"""DMR1 variants 2 and 3's sessions (study DMR1-R): grant, one-use artifacts, launch, closeout.

    PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-6 .venv/bin/python \\
        scripts/stages/stage-6/depth_recurrence/launch.py {budget|grant|issue|launch|closeout|position|stage-resume} \\
        --kind {dmr1_loop_train|dmr1_loop_hidden_train} ...

Two training kinds, one L40S each, each funded by a grant of its own and NOT authorized this round: ``dmr1_loop_train``
trains ``depth_loop`` (variant 2), ``dmr1_loop_hidden_train`` trains ``depth_loop_hidden`` (variant 3); both from the
training-pre initialization the selection binds (`schedule.py select`), never from another arm's trained weights.
Issued only once `stages.depth_merge.plan` has derived the arm's plan from the recurrent init record over that
selection. The mechanism and the gates are the round's (`stages.depth_merge.session`).
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

LOOP_KIND, HIDDEN_KIND = "dmr1_loop_train", "dmr1_loop_hidden_train"
VARIANT = S.Variant(
    key="DMR1-R", plan_id="stage6.depth_recurrence", logs=Path("logs/stages/stage-6/depth_recurrence"),
    session_kind="dmr1_depth_recurrence", build_kind=None,
    train_kinds={LOOP_KIND: ("depth_loop",), HIDDEN_KIND: ("depth_loop_hidden",)},
    identity_files=("configs/stages/stage-6/depth_recurrence/design.json",
                    "configs/stages/stage-6/depth_recurrence/recurrent_config.json",
                    "configs/stages/stage-6/depth_recurrence/base_config.json",
                    "configs/stages/stage-1/depth_merge/design.json"),
    artifact_spec="configs/stages/stage-6/depth_recurrence/artifacts.json", label="variants 2 and 3")
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

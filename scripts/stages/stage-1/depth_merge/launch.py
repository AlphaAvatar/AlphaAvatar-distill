#!/usr/bin/env python3
"""DMR1 variant 1's sessions (study DMR1-M): grant, one-use artifacts, launch, closeout ($0 except a granted launch).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/launch.py \\
        {budget|grant|issue|launch|closeout|position|stage-resume} --kind {dmr1_merge_build|dmr1_merge_train} ...

Two session kinds, one L40S each, on the container disk (no network volume), each funded by a grant of its own:

* ``dmr1_merge_build`` -- `build_driver.py`: the alpha-merged initialization from the teacher, durable on the relay
  with its statistics; the closeout checks its init record shows every product verified;
* ``dmr1_merge_train`` -- `train_driver.py`: the ``depth_merge`` arm's two seeds trained with SOB1's depth recipe and
  evaluated at 8,192 tokens in both modes, each set bound to its seed's final weights. Issued only once `plan.py` has
  derived the plan from the COMMITTED init record of a COMPLETE build.

The mechanism and the gates are the round's (`session.py`). Nothing is granted: ``plans/proposal_<kind>.json`` are
proposals for the maintainer.
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

BUILD_KIND, TRAIN_KIND = "dmr1_merge_build", "dmr1_merge_train"
#: v1-new (2026-11-28): its own build and training kinds, each funded by a grant of its own (V1's are not reused)
SPLIT_BUILD_KIND, SPLIT_TRAIN_KIND = "dmr1_split_build", "dmr1_split_train"
VARIANT = S.Variant(
    key="DMR1-M", plan_id="stage1.depth_merge", logs=Path("logs/stages/stage-1/depth_merge"),
    session_kind="dmr1_depth_merge",
    build_kinds={BUILD_KIND: "depth_merge", SPLIT_BUILD_KIND: "depth_merge_split"},
    train_kinds={TRAIN_KIND: ("depth_merge",), SPLIT_TRAIN_KIND: ("depth_merge_split",)},
    identity_files=("configs/stages/stage-1/depth_merge/design.json",
                    "configs/stages/stage-1/depth_merge/depth_init_tensor_digests.json"),
    kind_files={k: ("configs/stages/stage-1/depth_merge/split.json",
                    "configs/stages/stage-1/depth_merge/split_design.json",
                    "configs/stages/stage-1/depth_merge/split_config.json")
                for k in (SPLIT_BUILD_KIND, SPLIT_TRAIN_KIND)},
    artifact_spec="configs/stages/stage-1/depth_merge/artifacts.json", label="variant 1")
STUDY = VARIANT.study()


class Authorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``dmr1_depth_merge`` branch loads exactly this."""
    STUDY = STUDY


def kind_conditions(kind: str, root: Path = REPO, arm: str | None = None) -> dict:
    return S.kind_conditions(VARIANT, kind, root, arm)


def launch_spec(kind: str, minutes: float = 0.0, resume_from: str | None = None, arm: str | None = None):
    return S.launch_spec(VARIANT, Authorization, kind, minutes, resume_from, arm)


def main(argv=None) -> int:
    return S.main(VARIANT, Authorization, argv, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())

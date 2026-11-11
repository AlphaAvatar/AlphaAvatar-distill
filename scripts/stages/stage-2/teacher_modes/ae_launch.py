#!/usr/bin/env python3
"""A-E, Research A's engineering validation: its issuer and its launcher on one A100-SXM4-80GB ($0 until granted).

    # the one-use authorization (refuses without the maintainer's grant; --dry-run reads the proposal
    # and writes outside the run directory)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/ae_launch.py issue \\
        --run-id <run> --expected-usd 2.09 --hard-cap-usd 3.50 [--dry-run --out <scratch>/auth.json]
    # the session (every $0 gate for real, then stop: --dry-run --authorization <scratch>/auth.json)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/ae_launch.py launch \\
        --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live price>
    # the closeout of a finished session (also run by `launch`)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/ae_launch.py closeout \\
        --run-id <run> --scr <scratch>

`shared.study_engineering` with Research A's instance: the grant and proposal,
the files that bind the study (design, teacher and evaluation identities, the
check config, the preregistration, the stand-in student's recipe), the
driver (`engineering_validation.py`), the hybrid teacher Qwen/Qwen3-4B at its
pinned revision, and the stand-in student -- H1-PA's PCA-Original seed
20260726 (E1 PCA) checkpoint -- pulled hash-pinned from the relay (P8.4:
nothing heavy crosses the dev box's uplink). The pod keeps no weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from shared import study_engineering as SE  # noqa: E402

DESIGN = "configs/stages/stage-2/teacher_modes/design.json"
CHECK_CONFIG = "configs/stages/stage-2/teacher_modes/engineering_validation.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
STAND_IN_SEED = "20260726"
STAND_IN = f"artifacts/stages/stage-2/reasoning_data/h1pa/pca_original/{STAND_IN_SEED}"
GPU = "NVIDIA A100-SXM4-80GB"


def _e1_config() -> str:
    return json.loads((REPO_ROOT / H1PA).read_text())["recipe"]["e1_configs"][STAND_IN_SEED]["path"]


STUDY = SE.Study(
    key="A-E", plan_id="stage2.teacher_modes.engineering",
    logs=Path("logs/stages/stage-2/teacher_modes"),
    grant=Path("logs/budget/approvals/teacher_modes_engineering_grant.json"),
    proposal=Path("logs/stages/stage-2/teacher_modes/plans/engineering_grant_proposal.json"),
    identity_files=(DESIGN, "configs/stages/stage-2/teacher_modes/teacher_identity.json",
                    "configs/stages/stage-2/teacher_modes/evaluation_identity.json", CHECK_CONFIG,
                    "logs/stages/stage-2/teacher_modes/plans/preregistration.json", H1PA, _e1_config()),
    scripts=("scripts/stages/stage-2/teacher_modes/ae_launch.py",
             "scripts/stages/stage-2/teacher_modes/engineering_validation.py",
             "scripts/stages/stage-2/teacher_modes/evaluation_identity.py",
             "scripts/shared/study_engineering.py", "scripts/shared/project_position.py",
             "scripts/shared/validation/vllm_check.py", "scripts/shared/pod/teacher_cache.py", "scripts/shared/eval_sets.py",
             "scripts/shared/evaluation/eval_set_scoring.py",
             "scripts/stages/stage-0/calibration_volume/score_battery.py",
             "scripts/stages/stage-0/calibration_volume/run_battery.py",
             "scripts/stages/stage-2/reasoning_data/unit_driver.py",
             "scripts/stages/stage-2/reasoning_data/persist_unit.py",
             "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
             "scripts/maintenance/consolidation/derive_budget.py",
             "scripts/maintenance/consolidation/provider_billing.py",
             "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
             "scripts/stages/stage-2/reasoning_data/h1pa_launch.py",
             "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
             "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py"),
    wire_schema="aadistill.stage2.teacher_modes.engineering_authorization/v1")



class AEAuthorization(SE.EngineeringAuthorization):
    """A-E's artifact type: setup's ``stage2_teacher_modes`` branch loads exactly this one."""
    STUDY = STUDY

def relay_inputs() -> tuple:
    """The stand-in student's model files, hash-pinned from `h1pa.json :: relay_inputs` (never re-typed)."""
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    from stages.reasoning_data import h1pa_launch as HPL
    src = json.loads((REPO_ROOT / H1PA).read_text())["relay_inputs"]["pca_original"][STAND_IN_SEED]
    return tuple(RelayInput(f"{src['prefix']}/{n}", repo=MAIN_RELAY, dest=STAND_IN, sha256=sha)
                 for n, sha in sorted(HPL.model_files_only(src["files"], STAND_IN_SEED).items()))


def verdict(evidence: Path) -> dict:
    from stages.teacher_modes import engineering_validation as V
    return V.verdict(Path(evidence), json.loads((REPO_ROOT / CHECK_CONFIG).read_text()))


def launch_spec() -> SE.Launch:
    teacher = json.loads((REPO_ROOT / DESIGN).read_text())["teacher"]
    return SE.Launch(
        session_kind="stage2_teacher_modes", authorization=AEAuthorization, teacher_model=teacher["model"], teacher_revision=teacher["revision"],
        driver="scripts/stages/stage-2/teacher_modes/engineering_validation.py",
        #: the driver's bound inside the cap at the granted rate (plan: 20 setup + 5 transfer + 40, x1.25,
        #: + reserves = 103.75 min hard). Setup was 25 min (cost_forecast.json :: A_E_engineering_validation)
        #: until attempt 1 (ae_20261028_091943) measured create -> SETUP_DONE in 3.7 min and left $3.2725
        #: of the envelope, inside which the 110-minute plan no longer fits at the $1.79 quote
        validation_minutes=40.0, setup_minutes=20.0, disk_gb=120,
        driver_args=lambda args: f"--student {SE.POD_REPO}/{STAND_IN}",
        relay_inputs=relay_inputs(), verdict=verdict,
        artifact_spec="configs/stages/stage-2/teacher_modes/engineering_artifacts.json")


def build_parser():
    """The launch parser (``launch``'s arguments), for the shared launcher-contract tests."""
    return SE.build_parser(__doc__.split("\n")[0], gpu=GPU)


def spec(args):
    """The session the launch would run, for the shared launcher-contract tests."""
    return SE.session_spec(STUDY, launch_spec(), args)


def main(argv=None) -> int:
    return SE.cli(STUDY, launch_spec, argv, gpu=GPU, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())

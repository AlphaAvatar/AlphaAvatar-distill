#!/usr/bin/env python3
"""B-E, Research B's engineering validation: its issuer and its launcher on one L40S ($0 until granted).

    # the one-use authorization (refuses without the maintainer's grant; --dry-run reads the proposal
    # and writes outside the run directory)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/be_launch.py issue \\
        --run-id <run> --expected-usd 1.90 --hard-cap-usd 3.00 [--dry-run --out <scratch>/auth.json]
    # the session (every $0 gate for real, then stop: --dry-run --authorization <scratch>/auth.json)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/be_launch.py launch \\
        --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live price>
    # the closeout of a finished session (also run by `launch`)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/be_launch.py closeout \\
        --run-id <run> --scr <scratch>

`shared.study_engineering` with Research B's instance: the grant and proposal,
the files that bind the study (design, selection, the check config, the
preregistration, the smoke's recipe source), the driver
(`engineering_validation.py`), D4-Augmented's teacher Qwen3-4B-Thinking-2507
at its pinned revision, and the calibration profile's items (3.5 MB, a local
asset; its sha256 is checked against design.json by the driver). The pod keeps
no weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from shared import study_engineering as SE  # noqa: E402

DESIGN = "configs/stages/stage-1/projected_residual/design.json"
CHECK_CONFIG = "configs/stages/stage-1/projected_residual/engineering_validation.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
SMOKE_SEED = "20260726"
GPU = "NVIDIA L40S"


def _e1_config() -> str:
    return json.loads((REPO_ROOT / H1PA).read_text())["recipe"]["e1_configs"][SMOKE_SEED]["path"]


STUDY = SE.Study(
    key="B-E", plan_id="stage1.projected_residual.engineering",
    logs=Path("logs/stages/stage-1/projected_residual"),
    grant=Path("logs/budget/approvals/projected_residual_engineering_grant.json"),
    proposal=Path("logs/stages/stage-1/projected_residual/plans/engineering_grant_proposal.json"),
    identity_files=(DESIGN, "configs/stages/stage-1/projected_residual/d4_top1_selection.json", CHECK_CONFIG,
                    "logs/stages/stage-1/projected_residual/plans/preregistration.json", H1PA, _e1_config()),
    scripts=("scripts/stages/stage-1/projected_residual/be_launch.py",
             "scripts/stages/stage-1/projected_residual/engineering_validation.py",
             "scripts/stages/stage-1/projected_residual/materialize.py",
             "scripts/shared/study_engineering.py", "scripts/shared/project_position.py",
             "scripts/shared/validation/vllm_check.py", "scripts/shared/eval_sets.py",
             "scripts/shared/evaluation/uncapped_eval.py",
             "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
             "scripts/maintenance/consolidation/derive_budget.py",
             "scripts/maintenance/consolidation/provider_billing.py",
             "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
             "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
             "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py"),
    wire_schema="aadistill.stage1.projected_residual.engineering_authorization/v1")



class BEAuthorization(SE.EngineeringAuthorization):
    """B-E's artifact type: setup's ``stage1_projected_residual`` branch loads exactly this one."""
    STUDY = STUDY

def local_assets() -> tuple:
    from aadistill.infrastructure.session import LocalAsset
    items = json.loads((REPO_ROOT / DESIGN).read_text())["calibration"]["items_path"]
    return (LocalAsset(repo_path=items, dest_name=Path(items).name, install_to=str(Path(items).parent)),)


def verdict(evidence: Path) -> dict:
    from stages.projected_residual import engineering_validation as V
    return V.verdict(Path(evidence), json.loads((REPO_ROOT / CHECK_CONFIG).read_text()))


def launch_spec() -> SE.Launch:
    teacher = json.loads((REPO_ROOT / DESIGN).read_text())["teacher"]
    return SE.Launch(
        session_kind="stage1_projected_residual", authorization=BEAuthorization, teacher_model=teacher["model"], teacher_revision=teacher["revision"],
        driver="scripts/stages/stage-1/projected_residual/engineering_validation.py",
        #: the driver's bound inside the $3.00 cap at the granted rate (plan: 25 setup + 5 transfer + 75,
        #: x1.25, + reserves; cost_forecast.json :: B_E_engineering_validation)
        validation_minutes=75.0, setup_minutes=25.0, disk_gb=120,
        local_assets=local_assets(), verdict=verdict,
        artifact_spec="configs/stages/stage-1/projected_residual/engineering_artifacts.json")


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

#!/usr/bin/env python3
"""The independent-gated study's engineering validation: its issuer and its launcher ($0 until granted).

    # the one-use authorization (refuses without the maintainer's grant; --dry-run reads the proposal
    # and writes outside the run directory)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/igv_launch.py issue \\
        --run-id <run> --expected-usd <x> --hard-cap-usd <cap> [--dry-run --out <scratch>/auth.json]
    # the session (every $0 gate for real, then stop: --dry-run --authorization <scratch>/auth.json)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/igv_launch.py launch \\
        --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live price>
    # the closeout of a finished session (also run by `launch`)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/igv_launch.py closeout \\
        --run-id <run> --scr <scratch>

`shared.study_engineering` with this study's instance: the grant and proposal,
the files that bind the study (the design, the selection, the check config,
the preregistration, the base H1-PA record and its E1 configs), the driver
(`independent_gated_validation.py`), D4-Augmented's teacher at its pinned
revision, and, hash-pinned from the relay, the two inputs only this
validation reads: D4-Augmented's training pack (8.3 MB, the smokes' data) and
D4 Top-1 (1.19 GB, the control smoke's initialization). The calibration items
(3.5 MB) are a local asset. The pod keeps no weights.
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

DESIGN = "configs/stages/stage-1/projected_residual/independent_gated_design.json"
CHECK_CONFIG = "configs/stages/stage-1/projected_residual/independent_gated_validation.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
#: the card the proposal prices; the study itself is not bound to it (design :: execution.hardware)
GPU = "NVIDIA L40S"
PACK_DEST = "artifacts/stages/stage-1/projected_residual/validation/packs/e1_2p96M_plus_fluid"
CONTROL_DEST = "artifacts/stages/stage-1/projected_residual/validation/d4_top1"


def _e1_configs() -> tuple[str, ...]:
    return tuple(c["path"] for c in json.loads((REPO_ROOT / H1PA).read_text())["recipe"]["e1_configs"].values())


STUDY = SE.Study(
    key="independent-gated-E", plan_id="stage1.projected_residual.independent_gated.engineering",
    logs=Path("logs/stages/stage-1/projected_residual"),
    grant=Path("logs/budget/approvals/independent_gated_engineering_grant.json"),
    proposal=Path("logs/stages/stage-1/projected_residual/plans/independent_gated_engineering_proposal.json"),
    identity_files=(DESIGN, "configs/stages/stage-1/projected_residual/d4_top1_selection.json", CHECK_CONFIG,
                    "configs/stages/stage-1/projected_residual/engineering_validation.json",
                    "logs/stages/stage-1/projected_residual/plans/independent_gated_preregistration.json", H1PA,
                    "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json",
                    "configs/shared/baselines/selection.json",
                    "configs/shared/baselines/initialization_forward_baseline_v1.json", *_e1_configs()),
    scripts=("scripts/stages/stage-1/projected_residual/igv_launch.py",
             "scripts/stages/stage-1/projected_residual/independent_gated_validation.py",
             "scripts/stages/stage-1/projected_residual/recovery_driver.py",
             "scripts/stages/stage-1/projected_residual/materialize.py",
             "scripts/stages/stage-1/projected_residual/engineering_validation.py",
             "scripts/stages/stage-2/reasoning_data/h1pa_driver.py",
             "scripts/stages/stage-2/reasoning_data/unit_driver.py",
             "scripts/stages/stage-2/reasoning_data/persist_unit.py",
             "scripts/shared/training/train_stage3.py", "scripts/shared/forward_baseline.py",
             "scripts/shared/validation/engine_admission.py",
             "scripts/shared/study_engineering.py", "scripts/shared/project_position.py",
             "scripts/shared/validation/vllm_check.py", "scripts/shared/pod/teacher_cache.py", "scripts/shared/eval_sets.py",
             "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
             "scripts/maintenance/consolidation/derive_budget.py",
             "scripts/maintenance/consolidation/provider_billing.py",
             "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
             "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
             "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py",
             "scripts/shared/rollout/vllm_plugin/aadistill_vllm_plugin-1.0.dist-info/entry_points.txt"),
    wire_schema="aadistill.stage1.projected_residual.independent_gated_engineering_authorization/v1")


class IGEAuthorization(SE.EngineeringAuthorization):
    """This validation's artifact type: setup's ``stage1_independent_gated`` branch loads exactly this one."""
    STUDY = STUDY


def relay_inputs() -> tuple:
    """The pack and D4 Top-1, hash-pinned from their owners (h1pa.json, the materialization record)."""
    from aadistill.infrastructure.session import RelayInput
    from shared.deployment import MAIN_RELAY
    h1 = json.loads((REPO_ROOT / H1PA).read_text())
    pack = h1["relay_inputs"]["packs"]["e1_2p96M_plus_fluid"]
    out = [RelayInput(f"{pack['prefix']}/{n}", repo=MAIN_RELAY, dest=PACK_DEST, sha256=sha)
           for n, sha in sorted(pack["files"].items())]
    rec = json.loads((REPO_ROOT / "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json").read_text())
    pins = json.loads((REPO_ROOT / CHECK_CONFIG).read_text())["control_init_files"]
    for path, meta in sorted(rec["relay_metadata"].items()):
        name = path.rsplit("/", 1)[-1]
        if meta.get("lfs_sha256") and meta["lfs_sha256"] != pins[name]:
            raise ValueError(f"{name}: the check config's pin is not the materialization record's")
        out.append(RelayInput(path, repo=MAIN_RELAY, dest=CONTROL_DEST, sha256=pins[name]))
    return tuple(out)


def local_assets() -> tuple:
    from aadistill.infrastructure.session import LocalAsset
    items = json.loads((REPO_ROOT / DESIGN).read_text())["calibration"]["items_path"]
    return (LocalAsset(repo_path=items, dest_name=Path(items).name, install_to=str(Path(items).parent)),)


def verdict(evidence: Path) -> dict:
    from stages.projected_residual import independent_gated_validation as V
    return V.verdict(Path(evidence), json.loads((REPO_ROOT / CHECK_CONFIG).read_text()))


def launch_spec() -> SE.Launch:
    teacher = json.loads((REPO_ROOT / DESIGN).read_text())["teacher"]
    return SE.Launch(
        session_kind="stage1_independent_gated", authorization=IGEAuthorization, teacher_model=teacher["model"],
        teacher_revision=teacher["revision"],
        driver="scripts/stages/stage-1/projected_residual/independent_gated_validation.py",
        #: the driver's bound (cost_forecast :: independent_gated_engineering): initialization ~15 min, two
        #: 20-step smokes and the resume runs ~15 min, numerics/generation/identity ~5, three vLLM engines ~15
        validation_minutes=60.0, setup_minutes=20.0, disk_gb=120,
        driver_args=lambda args: (f"--pack-root {SE.POD_REPO}/{Path(PACK_DEST).parent} "
                                  f"--control-init {SE.POD_REPO}/{CONTROL_DEST}"),
        relay_inputs=relay_inputs(), local_assets=local_assets(), verdict=verdict,
        artifact_spec="configs/stages/stage-1/projected_residual/independent_gated_artifacts.json")


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

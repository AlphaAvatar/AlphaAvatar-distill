"""What every DMR1 script reads the same way: the round's two designs (variant 1 in ``configs/stages/stage-1/depth_merge``;
variants 2 and 3 in ``configs/stages/stage-1/depth_recurrence`` -- all Stage 1 research, recovered by Stage 3), which variant an arm belongs to, and the binding of SOB1's phase-2 chain
(`single_operator_budget.common.bind`) to that variant's design and logs. Nothing here runs a model.

The round reuses SOB1's chain -- what a complete evaluation set is (`single_operator_budget.completion`), the unit
state machine with its takeover, save-point rotation and final handoff (`single_operator_budget.train_driver.run_unit`
on TPI1's durability helpers) -- by binding it to a design with SOB1's schema; its rules are unchanged. A DMR1 program
binds once, at its entry (`bind_for`), never at import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DESIGN = REPO / "configs/stages/stage-1/depth_merge/design.json"
LOGS = REPO / "logs/stages/stage-1/depth_merge"
RECURRENCE_DESIGN = REPO / "configs/stages/stage-1/depth_recurrence/design.json"
RECURRENCE_LOGS = REPO / "logs/stages/stage-1/depth_recurrence"
RECURRENT_CONFIG = REPO / "configs/stages/stage-1/depth_recurrence/recurrent_config.json"

#: arm -> (variant, its design, its logs)
ARMS = {"depth_merge": ("DMR1-M", DESIGN, LOGS),
        "depth_loop_kept": ("DMR1-R", RECURRENCE_DESIGN, RECURRENCE_LOGS),
        "depth_loop_merged": ("DMR1-R", RECURRENCE_DESIGN, RECURRENCE_LOGS),
        "depth_loop_hidden": ("DMR1-RH", RECURRENCE_DESIGN, RECURRENCE_LOGS)}
#: the control arm of SOB1 (its sealed sets are reused, never regenerated)
SOB1_DEPTH = "depth_only"


class DMR1Error(RuntimeError):
    pass


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def design() -> dict:
    return read_json(DESIGN)


def recurrence_design() -> dict:
    return read_json(RECURRENCE_DESIGN)


def design_of(arm: str) -> dict:
    return read_json(files_of(arm)[0])


def files_of(arm: str) -> tuple[Path, Path]:
    if arm not in ARMS:
        raise DMR1Error(f"{arm!r} is not an arm of the round ({sorted(ARMS)})")
    _v, d, logs = ARMS[arm]
    return d, logs


def is_recurrent(arm: str) -> bool:
    return ARMS[arm][0] in ("DMR1-R", "DMR1-RH")


def bind_for(arm: str) -> dict:
    """Bind SOB1's chain to ``arm``'s variant (its design and logs); returns what was bound before (tests restore)."""
    from stages.single_operator_budget import common as SC
    d, logs = files_of(arm)
    return SC.bind(design=d, logs=logs)


def local_store(arm: str) -> Path:
    return Path(os.path.expanduser(design_of(arm)["storage"]["local_store"]))


def relay_root(arm: str) -> str:
    return design_of(arm)["storage"]["relay_prefix_root"]


def pod_root(arm: str) -> str:
    return design_of(arm)["storage"]["pod_root"]


def merged_control_evidence() -> Path:
    """Variant 1's seed-20260726 control on the dev box (its training session's secured ``<scr>/store/out``, as the
    recurrence design names it; only that seed's final and sets are read)."""
    ses = recurrence_design()["base"]["merged_control"]["session"]
    p = REPO / ses["completion"]["path"]
    doc = read_json(p)
    if doc.get("arm") != "depth_merge" or doc.get("run_id") != ses["run_id"]:
        raise DMR1Error(f"{ses['completion']['path']} is not variant 1's training session {ses['run_id']}")
    return Path(doc["scr"]) / "store" / "out"


def sob1_depth_evidence() -> Path:
    """The control's secured evidence on the dev box (SOB1's COMPLETE depth session's ``<scr>/store/out``)."""
    sess = design()["baseline"]["control_sessions"]
    doc = read_json(REPO / sess["completion"]["path"])
    if doc.get("status") != "COMPLETE" or doc.get("arm") != SOB1_DEPTH:
        raise DMR1Error(f"{sess['completion']['path']} is not SOB1's COMPLETE depth session")
    return Path(doc["scr"]) / "store" / "out"

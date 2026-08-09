#!/usr/bin/env python3
"""Measure the extra stream's gradient share before E7 trains. Takes no step.

E7 preregisters one `lambda_extra` and runs no sweep. That is only safe if an
obviously mis-scaled weight is caught **before** the run rather than inferred
from its results — the second is a sweep wearing a disguise.

This loads the real teacher and student, computes the rollout gradient and the
weighted extra gradient separately for a few steps, and reports the ratio of
their norms. No optimizer step is taken and the step counter does not move, so a
run that calls this is bit-identical to one that does not.

    PYTHONPATH=src python scripts/training/e7_preflight.py \\
        --config configs/stage3/e7/e7_fineweb_r1600k_sa.json \\
        --out artifacts/audit/e7_gradient_share_sa.json

**Registered acceptance band: `ratio_mean` in [0.05, 1.00].** Outside it, exit 8
and stop. Do not auto-tune; do not re-run with a different lambda without a
separate decision.

GPU-preferred. It runs on CPU for a smoke test, slowly, at whatever block length
the config declares.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import load_extra_stream  # noqa: E402
from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.infrastructure.env import (  # noqa: E402
    code_state, hardware_report, set_determinism,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.models.teacher import DTYPES, load_teacher  # noqa: E402
from aadistill.training.train import (  # noqa: E402
    Trainer, gradient_share, validate_train_config,
)


def resolve_device(pref: str) -> str:
    """Same resolution as `train_stage3.py`, so the preflight runs where the
    training would."""
    import torch
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"

BAND_LOW, BAND_HIGH = 0.05, 1.00


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--band-low", type=float, default=BAND_LOW)
    ap.add_argument("--band-high", type=float, default=BAND_HIGH)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    cfg = json.loads((REPO_ROOT / args.config).read_text()
                     if not Path(args.config).is_absolute()
                     else Path(args.config).read_text())
    validate_train_config(cfg)
    if cfg.get("extra_stream") is None:
        raise SystemExit("this config declares no extra stream")
    set_determinism(cfg["seed"])
    device = resolve_device(cfg["device"])

    train_tuple, _, ladder_stats = ladder_blocks(
        REPO_ROOT / cfg["data_dir"], cfg["rung"],
        n_val=cfg.get("val_blocks", 16))
    e_ids, e_content, e_meta = load_extra_stream(
        REPO_ROOT / cfg["extra_stream"]["data_dir"])

    student = AutoModelForCausalLM.from_pretrained(
        REPO_ROOT / cfg["student_path"], dtype=DTYPES[cfg["dtype"]])
    t = cfg["teacher"]
    teacher, _, teacher_identity = load_teacher(
        t["model_id"], t["revision"], dtype=t["dtype"], device=device)

    trainer = Trainer(cfg, student, train_tuple, teacher=teacher, device=device,
                      out_dir=REPO_ROOT / "artifacts/audit/_e7_preflight_scratch",
                      extra_stream_blocks=(e_ids, e_content))
    share = gradient_share(trainer, n_steps=args.steps)

    in_band = (share["ratio_mean"] is not None
               and args.band_low <= share["ratio_mean"] <= args.band_high)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "config": args.config,
        "config_sha256": sha256_json(cfg),
        "extra_stream_manifest_sha256": sha256_json(e_meta),
        "extra_stream_kind": cfg["extra_stream"]["kind"],
        "planned_extra_kd_positions": trainer.planned_extra_kd_positions(),
        "rung_blocks": int(ladder_stats["n_blocks"]) if "n_blocks" in ladder_stats
        else None,
        "teacher": teacher_identity,
        "device": device,
        "registered_band": [args.band_low, args.band_high],
        "in_band": bool(in_band),
        "gradient_share": share,
        "no_optimizer_step_taken": trainer.step == 0,
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: share[k] for k in
                      ("lambda_extra", "ratio_mean", "ratio_min", "ratio_max")},
                     indent=2))
    print(f"registered band [{args.band_low}, {args.band_high}] -> "
          f"{'IN BAND' if in_band else 'OUT OF BAND'}")
    print(f"-> {out}")
    if not in_band:
        print("STOP: lambda_extra is mis-scaled against the rollout objective. "
              "Report this and obtain a decision. Do NOT tune lambda from this "
              "measurement — that would be a sweep selected on its own result.",
              file=sys.stderr)
        return 8
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

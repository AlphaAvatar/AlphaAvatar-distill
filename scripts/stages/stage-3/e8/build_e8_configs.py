#!/usr/bin/env python3
"""Generate the two E8 treatment configs from the E1/P1 2.96M control.

The control is `configs/stage3/e1/e1_r2960k_{sa,sb}_pca.json` — the arms that
produced the standing behavioural anchor (usable 0.8400, correct 0.2067). E8's
intended causal variable is the Stage 1 depth map and nothing else, so a treatment
config is its control with **three keys changed**:

    student_path   the contribution-guided initialization instead of the PCA init
    run_name       so two runs do not write to one directory
    out_dir        likewise
    _purpose       prose

`student_path` is the variable. `run_name`/`out_dir` are bookkeeping that cannot
affect a gradient, and `_purpose` is a comment. Everything else — rung, blocks,
CE/KD weights, optimizer, LR, warmup, scheduler, batch and accumulation,
trainable-parameter patterns, precision, evaluation intervals, seeds — is copied
byte-for-byte, and `validate_e8_arms.py` asserts that the realized diff is exactly
that set before a pod is allowed to train.

    PYTHONPATH=src python scripts/training/build_e8_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

CONTROL_DIR = REPO_ROOT / "configs/stage3/e1"
OUT_DIR = REPO_ROOT / "configs/stage3/e8"
TREATMENT_INIT = "artifacts/stage1/e8_contribution_init_v1/checkpoint"

# name -> (control config, seed alias)
ARMS = {
    "e8_contrib_r2960k_sa": ("e1_r2960k_sa_pca.json", "sa"),
    "e8_contrib_r2960k_sb": ("e1_r2960k_sb_pca.json", "sb"),
}
ALLOWED_DIFF = {"student_path", "run_name", "out_dir", "_purpose"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (control_name, alias) in ARMS.items():
        control = json.loads((CONTROL_DIR / control_name).read_text())
        cfg = dict(control)
        cfg["student_path"] = TREATMENT_INIT
        cfg["run_name"] = name
        cfg["out_dir"] = f"artifacts/stage3/{name}"
        cfg["_purpose"] = (
            "E8 treatment: contribution-guided Stage 1 depth map, then the exact "
            f"E1/P1 KD-heavy 2.96M recovery recipe. Differs from "
            f"configs/stage3/e1/{control_name} only in student_path (the intended "
            "causal variable), run_name, out_dir and this note.")
        # Key order follows the control so a textual diff stays readable.
        ordered = {k: cfg[k] for k in control}
        realized = {k for k in ordered
                    if json.dumps(ordered[k], sort_keys=True)
                    != json.dumps(control.get(k), sort_keys=True)}
        if realized != ALLOWED_DIFF:
            raise SystemExit(
                f"{name}: realized diff {sorted(realized)} != {sorted(ALLOWED_DIFF)}")
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(ordered, indent=2) + "\n")
        written.append({
            "name": name, "seed_alias": alias, "path": str(path.relative_to(REPO_ROOT)),
            "control": f"configs/stage3/e1/{control_name}",
            "config_sha256": sha256_json(ordered),
            "control_sha256": sha256_json(control),
            "seed": ordered["seed"], "rung": ordered["rung"],
            "total_steps": ordered["schedule"]["total_steps"],
            "blocks_per_step": ordered["batch"]["blocks_per_step"],
            "loss": ordered["loss"],
        })
        print(f"{name}: {path.relative_to(REPO_ROOT)}  "
              f"sha256 {written[-1]['config_sha256'][:16]}…  "
              f"seed {ordered['seed']}  steps {ordered['schedule']['total_steps']}")

    (OUT_DIR / "arms.json").write_text(json.dumps(
        {"treatment_init": TREATMENT_INIT, "allowed_diff": sorted(ALLOWED_DIFF),
         "arms": written}, indent=2) + "\n")
    print(f"-> {(OUT_DIR / 'arms.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

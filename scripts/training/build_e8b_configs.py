#!/usr/bin/env python3
"""Generate E8b's six trained arms from the canonical E1/P1 KD-heavy 1.60M control.

Three cells train, one is retained:

    DP  depth-only positional      <- artifacts/stage1/e8b_dp_init/checkpoint
    DC  depth-only contribution    <- artifacts/stage1/e8b_dc_init/checkpoint
    FC  fully compressed contrib.  <- artifacts/stage1/e8_contribution_init_v1/checkpoint
    FP  fully compressed position.  RETAINED: e1_r1600k_{sa,sb}_pca @ step_001761

Every arm is its cell's control config with **three keys changed** — `student_path`,
`run_name`, `out_dir` — plus `_purpose` prose. `student_path` is the variable; the
other two cannot affect a gradient. The rung, block order, CE/KD weights, optimizer,
scheduler, warmup, batch and accumulation, trainable-parameter patterns, precision,
evaluation intervals, checkpoint cadence and seeds are copied byte-for-byte, so a
session split cannot alter token exposure or the schedule.

    PYTHONPATH=src python scripts/training/build_e8b_configs.py

`save_every` is 880 of 1,761 steps in the canonical config, which is what makes exact
mid-arm resume possible if a session is interrupted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

CONTROL_DIR = REPO_ROOT / "configs/stage3/e1"
OUT_DIR = REPO_ROOT / "configs/stage3/e8b"
ALLOWED_DIFF = {"student_path", "run_name", "out_dir", "_purpose"}

# cell -> (init path, regime, hardware class the cell trains on)
CELLS = {
    "dp": ("artifacts/stage1/e8b_dp_init/checkpoint", "depth_only", "A100_SXM_80GB"),
    "dc": ("artifacts/stage1/e8b_dc_init/checkpoint", "depth_only", "A100_SXM_80GB"),
    "fc": ("artifacts/stage1/e8_contribution_init_v1/checkpoint",
           "fully_compressed", "L40S_48GB"),
}
DEPTH_MAP = {"dp": "positional", "dc": "contribution", "fc": "contribution"}
SEEDS = {"sa": "e1_r1600k_sa_pca.json", "sb": "e1_r1600k_sb_pca.json"}
RETAINED_FP = {
    "regime": "fully_compressed", "depth_map": "positional",
    "hardware": "L40S_48GB",
    "arms": {"sa": "e1_r1600k_sa_pca", "sb": "e1_r1600k_sb_pca"},
    "step": "step_001761",
    "relay": "e1_scaling_20260801/e1_r1600k_{seed}_pca/step_001761",
    "model_sha256": {
        "sa": "6f77676ab8fde397ef7af75fda3e62171b5c84f315c439a1abb49917e46f6697",
        "sb": "e432d57e598d57e1633392e92955c8185faab57909f75f44bc1c349db6ccf39e"},
    "battery_artifacts": "retained from E6, inclusion mask "
                         "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba",
    "behaviour": {"usable_rollout_rate": 0.7300, "correct_overall": 0.1867,
                  "correct_given_usable": 0.2511},
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for cell, (init, regime, hardware) in CELLS.items():
        for seed, control_name in SEEDS.items():
            control = json.loads((CONTROL_DIR / control_name).read_text())
            name = f"e8b_{cell}_r1600k_{seed}"
            cfg = dict(control)
            cfg["student_path"] = init
            cfg["run_name"] = name
            cfg["out_dir"] = f"artifacts/stage3/{name}"
            cfg["_purpose"] = (
                f"E8b cell {cell.upper()} ({regime}, {DEPTH_MAP[cell]} depth map), "
                f"seed {seed}, on {hardware}. The canonical E1/P1 KD-heavy 1.60M "
                f"recipe; differs from configs/stage3/e1/{control_name} only in "
                "student_path (the intended causal variable), run_name, out_dir and "
                "this note.")
            ordered = {k: cfg[k] for k in control}
            realized = {k for k in ordered
                        if json.dumps(ordered[k], sort_keys=True)
                        != json.dumps(control.get(k), sort_keys=True)}
            if realized != ALLOWED_DIFF:
                raise SystemExit(f"{name}: realized diff {sorted(realized)} != "
                                 f"{sorted(ALLOWED_DIFF)}")
            path = OUT_DIR / f"{name}.json"
            path.write_text(json.dumps(ordered, indent=2) + "\n")
            written.append({
                "name": name, "cell": cell.upper(), "regime": regime,
                "depth_map": DEPTH_MAP[cell], "hardware": hardware,
                "seed_alias": seed, "seed": ordered["seed"],
                "path": str(path.relative_to(REPO_ROOT)),
                "control": f"configs/stage3/e1/{control_name}",
                "config_sha256": sha256_json(ordered),
                "control_sha256": sha256_json(control),
                "student_path": init,
                "rung": ordered["rung"],
                "total_steps": ordered["schedule"]["total_steps"],
                "save_every": ordered["checkpoint"]["save_every"],
            })
            print(f"{name:24s} {ordered['seed']}  {hardware:16s} "
                  f"sha256 {written[-1]['config_sha256'][:16]}…")

    # Within-cell config identity: DP and DC must differ only in student_path, and
    # the same for FP's control config and FC. That is what makes each depth-map
    # effect a single-variable comparison inside its own hardware class.
    pairs = []
    for seed in SEEDS:
        a = json.loads((OUT_DIR / f"e8b_dp_r1600k_{seed}.json").read_text())
        b = json.loads((OUT_DIR / f"e8b_dc_r1600k_{seed}.json").read_text())
        diff = {k for k in set(a) | set(b)
                if json.dumps(a.get(k), sort_keys=True)
                != json.dumps(b.get(k), sort_keys=True)}
        pairs.append({"pair": f"DP-vs-DC-{seed}", "diff": sorted(diff)})
        if diff != ALLOWED_DIFF:
            raise SystemExit(f"DP/DC {seed} differ in {sorted(diff)}")
        fp = json.loads((CONTROL_DIR / SEEDS[seed]).read_text())
        fc = json.loads((OUT_DIR / f"e8b_fc_r1600k_{seed}.json").read_text())
        diff = {k for k in set(fp) | set(fc)
                if json.dumps(fp.get(k), sort_keys=True)
                != json.dumps(fc.get(k), sort_keys=True)}
        pairs.append({"pair": f"FP-vs-FC-{seed}", "diff": sorted(diff)})
        if diff != ALLOWED_DIFF:
            raise SystemExit(f"FP/FC {seed} differ in {sorted(diff)}")

    meta = {
        "experiment": "E8b — depth-map x compression interaction",
        "rung": 1600000,
        "recipe": "E1/P1 KD-heavy 1.60M, unchanged",
        "allowed_diff": sorted(ALLOWED_DIFF),
        "hardware_design": "pair-matched and NESTED with compression regime: "
                           "DP/DC on A100 SXM 80GB, FP/FC on L40S 48GB. Each "
                           "depth-map effect is measured within one hardware "
                           "class; the interaction inherits the nesting and "
                           "cannot by itself exclude a hardware x depth-map "
                           "interaction (see the conditional bridge rule).",
        "arms": written,
        "retained_fp": RETAINED_FP,
        "within_cell_identity": pairs,
    }
    (OUT_DIR / "arms.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwithin-cell identity verified: "
          f"{[p['pair'] for p in pairs]} each differ in exactly "
          f"{sorted(ALLOWED_DIFF)}")
    print(f"-> {(OUT_DIR / 'arms.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

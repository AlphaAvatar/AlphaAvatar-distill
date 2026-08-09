#!/usr/bin/env python3
"""Generate the E7 arm configs by adding one key to the retained baseline.

The E7 claim is that the rollout trajectory is untouched. The cheapest way to
make that true is to not write a new config: each arm is `e1_r1600k_{seed}_pca`
— the exact file that produced the retained arm-A checkpoints — plus an
`extra_stream` block, plus the three identity fields that must differ so the run
writes somewhere else.

Anything else that differed would be a second variable. `validate_e7_arms.py`
asserts the diff is exactly `{extra_stream, run_name, out_dir, _purpose}` and
refuses to proceed otherwise.

    PYTHONPATH=src python scripts/training/build_e7_configs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import stream_budget  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.training.train import validate_train_config  # noqa: E402

BASE = "configs/stage3/e1/e1_r1600k_{seed}_pca.json"
SEEDS = ("sa", "sb")

# The preregistered treatment. Every field is part of the config hash.
#   block_len 1024   matches the historical FineWeb NLL protocol
#                    (`eval_ppl.py --max-seq-len 1024`), so the training context
#                    and the diagnostic context agree.
#   1 block/step, every step
#                    a uniform per-step gradient composition. A cadence > 1
#                    would make one step in k structurally different from the
#                    others and interact with the LR schedule.
#   lambda_extra 0.25
#                    the same weight the recipe already gives its secondary
#                    term, against rollout KD's 1.0.
EXTRA = {
    "lambda_extra": 0.25,
    "blocks_per_step": 1,
    "micro_blocks": 1,
    "every_n_steps": 1,
    "seed": 20260809,
}

ARMS = {
    "B": {"suffix": "fineweb", "kind": "general_text_kd",
          "data_dir": "artifacts/stage3/e7_fineweb_kd",
          "purpose": "E7 arm B — FineWeb-Edu raw-text teacher KD alongside the "
                     "unchanged 1.60M rollout stream"},
    "C": {"suffix": "control", "kind": "in_domain_kd_control",
          "data_dir": "artifacts/stage3/e7_control_kd",
          "purpose": "E7 arm C — matched extra-KD control: identical extra KD "
                     "positions, forward workload and schedule, from unused "
                     "in-domain rollout text instead of FineWeb"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="configs/stage3/e7")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for arm, spec in ARMS.items():
        for seed in SEEDS:
            base_path = REPO_ROOT / BASE.format(seed=seed)
            cfg = json.loads(base_path.read_text())
            run = f"e7_{spec['suffix']}_r1600k_{seed}"
            cfg["run_name"] = run
            cfg["out_dir"] = f"artifacts/stage3/{run}"
            cfg["_purpose"] = (
                f"{spec['purpose']}. Forked from the canonical Stage 1 PCA "
                f"init, not from any trained checkpoint. Identical to "
                f"{base_path.relative_to(REPO_ROOT)} except for extra_stream "
                f"and the identity fields.")
            cfg["extra_stream"] = {"data_dir": spec["data_dir"],
                                   "kind": spec["kind"], **EXTRA}
            validate_train_config(cfg)
            dest = out_root / f"{run}.json"
            dest.write_text(json.dumps(cfg, indent=2) + "\n")
            budget = stream_budget(
                1761, 1024, total_steps=cfg["schedule"]["total_steps"],
                blocks_per_step=EXTRA["blocks_per_step"],
                every_n_steps=EXTRA["every_n_steps"])
            written[run] = {"path": str(dest.relative_to(REPO_ROOT)),
                            "config_sha256": sha256_json(cfg),
                            "arm": arm, "seed": cfg["seed"],
                            "extra_budget": budget}
            print(f"{run:28s} sha {sha256_json(cfg)[:16]}…  seed {cfg['seed']}  "
                  f"{budget['kd_positions']:,} extra KD positions")

    index = out_root / "e7_configs.json"
    index.write_text(json.dumps(written, indent=2) + "\n")
    print(f"\nwrote {len(written)} configs + {index.relative_to(REPO_ROOT)}")
    print("NOTE: the student_path is the Stage 1 init in every arm; no arm "
          "starts from a trained 1.60M or 2.96M checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

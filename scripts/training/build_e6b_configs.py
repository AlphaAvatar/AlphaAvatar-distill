#!/usr/bin/env python
"""Build the two E6b arms by composing two configs that already exist.

    PYTHONPATH=src python scripts/training/build_e6b_configs.py

E6b fills one cell of an objective × data-scale matrix:

                      1.60M        2.96M
    E1/P1 KD-heavy    exists       exists
    P2    CE-heavy    exists       **E6b**

so its config is not written by hand. It is **P2-1.60M with the 2.96M rung**,
and every field is taken from one of the two canonical parents:

* the **objective and everything the objective drags with it** comes from
  `e4_p2_r1600k_{seed}` — the P2 recipe whose result E6b extends;
* the **rung and everything the rung mechanically implies** (block count,
  optimizer steps, warmup, checkpoint and eval cadence) comes from
  `e1_r2960k_{seed}_pca` — the arm E6b is compared against at the same scale.

Nothing else may differ. The generator asserts that the two parents already
agree on every remaining key, so a future drift in either one fails here rather
than producing a config whose comparison is quietly invalid. That assertion is
the actual experimental control; the file it writes is a consequence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

SEEDS = ("sa", "sb")
OBJECTIVE_PARENT = "configs/stage3/e4/e4_p2_r1600k_{seed}.json"
RUNG_PARENT = "configs/stage3/e1/e1_r2960k_{seed}_pca.json"

# Taken from the RUNG parent: the data scale and the quantities it mechanically
# implies. `schedule` carries total_steps and warmup, both derived from the block
# count; the two cadences are derived from total_steps.
FROM_RUNG = ("rung", "schedule", "checkpoint", "intervals")
# Identity only — no effect on training numerics.
IDENTITY = ("run_name", "out_dir", "_purpose")


def build(seed: str) -> tuple[dict, dict]:
    obj = json.loads((REPO_ROOT / OBJECTIVE_PARENT.format(seed=seed)).read_text())
    rung = json.loads((REPO_ROOT / RUNG_PARENT.format(seed=seed)).read_text())

    # The control: the parents must already agree on everything E6b does not
    # deliberately take from one of them. If they do not, the "only the objective
    # differs" claim is false and no config should be written.
    shared = (set(obj) | set(rung)) - set(FROM_RUNG) - set(IDENTITY) - {"loss"}
    disagree = {k for k in shared
                if json.dumps(obj.get(k), sort_keys=True)
                != json.dumps(rung.get(k), sort_keys=True)}
    if disagree:
        raise SystemExit(
            f"{seed}: the P2 and E1 parents disagree on {sorted(disagree)}, which "
            "E6b does not control. The objective × scale comparison would be "
            "confounded. Stop and document the discrepancy.")

    cfg = dict(obj)                       # objective, optimizer, init, everything
    for k in FROM_RUNG:                   # the rung and what it implies
        cfg[k] = rung[k]
    cfg["run_name"] = f"e6b_p2_r2960k_{seed}"
    cfg["out_dir"] = f"artifacts/stage3/e6b_p2_r2960k_{seed}"
    cfg["_purpose"] = (
        "Experiment 6b (objective x data-scale interaction): the P2-CE-heavy "
        "objective (ce 1.0 / kd 0.25, kd_scope all) trained at the strictly "
        f"nested 2.96M rung. Composed mechanically by "
        f"scripts/training/build_e6b_configs.py from e4_p2_r1600k_{seed} "
        f"(objective and everything else) and e1_r2960k_{seed}_pca (rung, "
        "schedule, cadences). Trains from the Stage 1 PCA init, NOT continued "
        "from P2-1.60M. The only intended difference from e1_r2960k_"
        f"{seed}_pca is the CE/KD weighting.")

    # State the invariants in the artifact itself, not only in this docstring.
    provenance = {
        "objective_parent": OBJECTIVE_PARENT.format(seed=seed),
        "objective_parent_sha256": sha256_json(obj),
        "rung_parent": RUNG_PARENT.format(seed=seed),
        "rung_parent_sha256": sha256_json(rung),
        "taken_from_rung_parent": list(FROM_RUNG),
        "identity_only": list(IDENTITY),
        "loss_taken_from": "objective parent (the experimental variable)",
        "loss": cfg["loss"],
        "e1_loss_at_same_rung": rung["loss"],
        "parents_agree_on_everything_else": True,
        "config_sha256": sha256_json(cfg),
    }
    return cfg, provenance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "configs/stage3/e6b")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "configs/stage3/e6b/provenance.json")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in SEEDS:
        cfg, prov = build(seed)
        path = args.out_dir / f"{cfg['run_name']}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        manifest[cfg["run_name"]] = prov
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        print(f"  seed {cfg['seed']}  rung {cfg['rung']:,}  "
              f"steps {cfg['schedule']['total_steps']}  "
              f"loss {cfg['loss']['ce_weight']}/{cfg['loss']['kd_weight']}  "
              f"sha256 {prov['config_sha256'][:16]}…")
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.manifest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

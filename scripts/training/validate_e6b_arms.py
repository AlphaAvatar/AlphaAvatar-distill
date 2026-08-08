#!/usr/bin/env python
"""Prove the E6b arms are what the registration says, before a step is taken.

    PYTHONPATH=src python scripts/training/validate_e6b_arms.py \
        --registration logs/e6b_registration.json --out artifacts/audit/e6b_preflight.json

E6b's entire claim rests on one sentence: *the only intended difference from
`e1_r2960k_{seed}_pca` is the CE/KD weighting*. That sentence is checked here
against the files that will actually train, and again on the pod in its own
environment, because a confounded comparison is not detectable from the result —
it looks exactly like a real effect.

Five things are asserted:

1. each arm's `config_sha256` matches the registration;
2. against E1 at the same rung, the configs differ **only** on `loss` plus
   identity/documentation fields;
3. against P2 at the lower rung, they differ **only** on the rung and the
   quantities it mechanically implies;
4. both arms fork from the Stage 1 PCA init, never from a trained checkpoint;
5. the realized data — blocks, supervised CE tokens, exposures, nesting — matches
   what the registration recorded from the loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

# Permitted to differ from the same-rung E1 arm: the experimental variable, plus
# fields with no effect on training numerics.
ALLOWED_VS_E1 = {"loss", "run_name", "out_dir", "_purpose"}
# Permitted to differ from the lower-rung P2 arm: the data scale and what it
# mechanically implies.
ALLOWED_VS_P2 = {"rung", "schedule", "checkpoint", "intervals",
                 "run_name", "out_dir", "_purpose"}
INIT_SHA = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"


def diff_keys(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b)
            if json.dumps(a.get(k), sort_keys=True)
            != json.dumps(b.get(k), sort_keys=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    reg = json.loads(args.registration.read_text())
    report: dict = {"registration_sha256": reg["registration_sha256"], "arms": {}}
    failures: list[str] = []

    for seed in ("sa", "sb"):
        alias = f"P2-2.96M-{seed}"
        arm = reg["arms"][alias]
        cfg = json.loads((REPO_ROOT / arm["config"]).read_text())
        e1 = json.loads((REPO_ROOT / f"configs/stage3/e1/e1_r2960k_{seed}_pca.json").read_text())
        p2 = json.loads((REPO_ROOT / f"configs/stage3/e4/e4_p2_r1600k_{seed}.json").read_text())

        got_sha = sha256_json(cfg)
        d_e1 = diff_keys(cfg, e1)
        d_p2 = diff_keys(cfg, p2)
        row = {
            "config": arm["config"],
            "config_sha256": got_sha,
            "matches_registration": got_sha == arm["config_sha256"],
            "seed": cfg["seed"],
            "loss": cfg["loss"],
            "e1_loss_same_rung": e1["loss"],
            "differs_from_e1_same_rung_on": sorted(d_e1),
            "only_objective_differs_from_e1": d_e1 <= ALLOWED_VS_E1 and "loss" in d_e1,
            "differs_from_p2_lower_rung_on": sorted(d_p2),
            "only_rung_differs_from_p2": d_p2 <= ALLOWED_VS_P2 and "rung" in d_p2,
            "student_path": cfg["student_path"],
            "forks_from_stage1_init": cfg["student_path"].endswith(
                "qwen3_0p6b_init_v0/checkpoint"),
            "total_steps": cfg["schedule"]["total_steps"],
            "warmup_steps": cfg["schedule"]["warmup_steps"],
            "rung": cfg["rung"],
        }
        if not row["matches_registration"]:
            failures.append(f"{alias}: config_sha256 {got_sha[:16]}… != registered")
        if not row["only_objective_differs_from_e1"]:
            failures.append(
                f"{alias}: differs from E1 at the same rung on {sorted(d_e1)}; "
                "the objective comparison would be confounded")
        if not row["only_rung_differs_from_p2"]:
            failures.append(
                f"{alias}: differs from P2 at the lower rung on {sorted(d_p2)}; "
                "the scale comparison would be confounded")
        if not row["forks_from_stage1_init"]:
            failures.append(f"{alias}: does not fork from the Stage 1 init")
        if cfg["loss"] != {"ce_weight": 1.0, "kd_weight": 0.25,
                           "kd_temperature": 1.0, "kd_scope": "all"}:
            failures.append(f"{alias}: objective is not the registered CE-heavy one")
        report["arms"][alias] = row

    init = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors"
    if init.is_file():
        got = sha256_file(init)
        report["stage1_init_sha256"] = got
        report["stage1_init_matches"] = got == INIT_SHA
        if got != INIT_SHA:
            failures.append(f"Stage 1 init hash {got[:16]}… != {INIT_SHA[:16]}…")
    else:
        report["stage1_init_sha256"] = None
        report["stage1_init_matches"] = None

    # Realized data, recomputed here rather than trusted from the registration.
    pack = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
    if (pack / "blocks.npz").is_file():
        import torch
        from aadistill.data.ladder import ladder_blocks
        (ids16, ce16, cm16), _, _ = ladder_blocks(pack, 1600000, 16)
        (ids29, ce29, cm29), _, s29 = ladder_blocks(pack, 2960000, 16)
        expect = reg["rung_facts_verified_from_loader"]["rung_2960000"]
        realized = {
            "train_blocks": int(ids29.shape[0]),
            "supervised_ce_tokens_unique": int(ce29.sum()),
            "exposures": round(2916 * 2 / int(ids29.shape[0]), 4),
            "cumulative_ce_tokens": int(ce29.sum()) * (2916 * 2 // int(ids29.shape[0])),
            "strict_token_prefix_of_1600k": bool(
                torch.equal(ids29[:ids16.shape[0]], ids16)
                and torch.equal(ce29[:ce16.shape[0]], ce16)
                and torch.equal(cm29[:cm16.shape[0]], cm16)),
        }
        report["realized_rung"] = realized
        for k in ("train_blocks", "supervised_ce_tokens_unique", "exposures",
                  "cumulative_ce_tokens"):
            if realized[k] != expect[k]:
                failures.append(f"rung {k}: realized {realized[k]} != registered {expect[k]}")
        if not realized["strict_token_prefix_of_1600k"]:
            failures.append("2.96M is not a strict token-level superset of 1.60M")

    report["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    for alias, row in report["arms"].items():
        print(f"  {alias}: sha {row['config_sha256'][:16]}… "
              f"vs-E1 {row['differs_from_e1_same_rung_on']} "
              f"vs-P2 {row['differs_from_p2_lower_rung_on']}")
    if failures:
        raise SystemExit("E6b ARM VALIDATION FAILED:\n  " + "\n  ".join(failures))
    print("E6b arms validated: only the objective differs from E1 at 2.96M, "
          "only the rung differs from P2 at 1.60M")


if __name__ == "__main__":
    main()

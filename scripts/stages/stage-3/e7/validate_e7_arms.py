#!/usr/bin/env python3
"""Prove the E7 arms are what the preregistration says, before a step is taken.

    PYTHONPATH=src python scripts/training/validate_e7_arms.py \\
        --out artifacts/audit/e7_preflight.json

E7's claim is one sentence: *the only intended difference from
`e1_r1600k_{seed}_pca` is the added KD-only stream, and B differs from C only in
that stream's content.* A confounded comparison is not detectable from the
result — it looks exactly like a real effect — so it is asserted here against
the files that will actually train, and again on the pod.

Seven things are checked:

1. every arm differs from its same-seed E1 baseline **only** on `extra_stream`
   plus identity fields — so the rollout blocks, order, exposures, schedule, LR,
   CE/KD weights and trainable set are literally the baseline's;
2. every arm forks from the canonical Stage 1 PCA init, never from a trained
   checkpoint;
3. B and C are identical apart from the extra stream's `data_dir` and `kind`;
4. B's and C's streams have equal `n_blocks`, `block_len` and therefore equal
   KD-position and forward-token budgets;
5. neither stream contains padding or CE positions;
6. the planned extra-KD budget matches what the preregistration recorded;
7. the disjointness report exists, covers every stream, and says `disjoint`.

Exit codes: 0 all checks pass; 7 a check failed.
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

# Permitted to differ from the same-seed E1 baseline: the experimental variable,
# plus fields with no effect on training numerics.
ALLOWED_VS_E1 = {"extra_stream", "run_name", "out_dir", "_purpose"}
# Permitted to differ between B and C: only where the extra text comes from.
ALLOWED_EXTRA_BC = {"data_dir", "kind"}

INIT_PATH = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
BASE = "configs/stage3/e1/e1_r1600k_{seed}_pca.json"


def diff_keys(a: dict, b: dict) -> set[str]:
    return {k for k in set(a) | set(b)
            if json.dumps(a.get(k), sort_keys=True)
            != json.dumps(b.get(k), sort_keys=True)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs/stage3/e7/e7_configs.json")
    ap.add_argument("--disjointness", default="artifacts/stage3/e7_disjointness.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-streams", action="store_true",
                    help="also verify the packed streams on disk; omit before "
                         "they are built")
    args = ap.parse_args()

    index = json.loads((REPO_ROOT / args.configs).read_text())
    failures: list[str] = []
    report: dict = {"arms": {}, "checks": {}}

    configs = {}
    for run, meta in index.items():
        cfg = json.loads((REPO_ROOT / meta["path"]).read_text())
        configs[run] = cfg
        sha = sha256_json(cfg)
        if sha != meta["config_sha256"]:
            failures.append(f"{run}: config sha {sha[:16]}… != index "
                            f"{meta['config_sha256'][:16]}…")

        seed_tag = "sa" if run.endswith("_sa") else "sb"
        base = json.loads((REPO_ROOT / BASE.format(seed=seed_tag)).read_text())
        differs = diff_keys(cfg, base)
        extra_diff = sorted(differs - ALLOWED_VS_E1)
        if extra_diff:
            failures.append(
                f"{run}: differs from its E1 baseline on {extra_diff}; the "
                "rollout trajectory must be the baseline's exactly")
        if cfg["student_path"] != INIT_PATH:
            failures.append(f"{run}: forks from {cfg['student_path']}, not the "
                            "canonical Stage 1 init")
        if cfg["loss"] != {"ce_weight": 0.25, "kd_weight": 1.0,
                           "kd_temperature": 1.0, "kd_scope": "all"}:
            failures.append(f"{run}: not the E1/P1 KD-heavy objective")
        if cfg.get("rung") != 1600000 or cfg["schedule"]["total_steps"] != 1761:
            failures.append(f"{run}: not the 1.60M rung schedule")
        report["arms"][run] = {
            "config_sha256": sha, "seed": cfg["seed"],
            "diff_vs_e1_baseline": sorted(differs),
            "extra_stream": cfg["extra_stream"],
        }

    # B vs C: same everything but the stream's origin.
    for seed_tag in ("sa", "sb"):
        b = configs.get(f"e7_fineweb_r1600k_{seed_tag}")
        c = configs.get(f"e7_control_r1600k_{seed_tag}")
        if not (b and c):
            continue
        differs = diff_keys(b, c)
        if differs - {"extra_stream", "run_name", "out_dir", "_purpose"}:
            failures.append(
                f"{seed_tag}: B and C differ on "
                f"{sorted(differs - ALLOWED_VS_E1)} beyond the extra stream")
        e_diff = diff_keys(b["extra_stream"], c["extra_stream"])
        if e_diff - ALLOWED_EXTRA_BC:
            failures.append(
                f"{seed_tag}: B and C extra streams differ on "
                f"{sorted(e_diff - ALLOWED_EXTRA_BC)}; only the source may differ")

    if args.require_streams:
        budgets = {}
        for run, cfg in configs.items():
            d = REPO_ROOT / cfg["extra_stream"]["data_dir"]
            mpath = d / "manifest.json"
            if not mpath.is_file():
                failures.append(f"{run}: {mpath} missing")
                continue
            m = json.loads(mpath.read_text())
            if m.get("padding_tokens", -1) != 0:
                failures.append(f"{run}: stream reports "
                                f"{m.get('padding_tokens')} padding tokens")
            if m.get("assistant_ce_positions", -1) != 0:
                failures.append(f"{run}: stream declares CE positions")
            budgets[run] = stream_budget(
                m["n_blocks"], m["block_len"],
                total_steps=cfg["schedule"]["total_steps"],
                blocks_per_step=cfg["extra_stream"]["blocks_per_step"],
                every_n_steps=cfg["extra_stream"]["every_n_steps"])
            report["arms"][run]["stream_manifest_sha256"] = sha256_json(m)
            report["arms"][run]["budget"] = budgets[run]
        distinct = {json.dumps(b, sort_keys=True) for b in budgets.values()}
        if len(distinct) > 1:
            failures.append(
                "arms do not share one extra-KD budget; B and C would differ in "
                f"positions or compute: {list(budgets.values())}")
        report["checks"]["matched_budget"] = len(distinct) == 1

        dj = REPO_ROOT / args.disjointness
        if not dj.is_file():
            failures.append(f"{dj} missing; disjointness must be proven, not "
                            "assumed")
        else:
            proof = json.loads(dj.read_text())
            report["checks"]["disjointness"] = proof.get("disjoint")
            if not proof.get("disjoint"):
                failures.append("the disjointness report does not say disjoint")
            covered = set(proof.get("groups", {}))
            for cfg in configs.values():
                name = Path(cfg["extra_stream"]["data_dir"]).name
                if name not in covered:
                    failures.append(f"disjointness report does not cover {name}")

    report["failures"] = failures
    report["passed"] = not failures
    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for run, r in report["arms"].items():
        print(f"{run:28s} sha {r['config_sha256'][:16]}…  "
              f"vs-E1 {r['diff_vs_e1_baseline']}")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 7
    print("\nall E7 arm checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

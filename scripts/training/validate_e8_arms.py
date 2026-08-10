#!/usr/bin/env python3
"""The gate E8 must pass before a single training step is paid for.

Nine assertions, every one of them a way this experiment could quietly stop being
the experiment:

1. **config diff** — each treatment config differs from its E1/P1 2.96M control in
   exactly `{student_path, run_name, out_dir, _purpose}`. Anything else means the
   depth map is not the only variable.
2. **shared initialization** — both seeds fork from the same treatment checkpoint.
3. **token budget, from the loader** — 2,960,507 unique CE targets and 8,881,521
   cumulative exposure, read out of the pack rather than copied from a document.
4. **initialization-NLL gate** — the treatment checkpoint has its own hash-bound
   NLL record, with every required measurement. This is the rule that must not be
   skippable: an initialization is not complete until its own NLL exists.
5. **baseline remeasured** — the control initialization has a record too, produced
   by the same evaluator in the same environment, so step 0 is comparable.
6. **the map is a real change** — 28 strictly increasing teacher layers that are
   not the positional map's. If they were identical there would be nothing to test.
7. **geometry unchanged** — the treatment init has the control's parameter count
   and architecture, because only the depth map moved.
8. **RoPE base** — resolves to the teacher's, in *this* environment. The
   transformers 4.x/5.x `rope_parameters` skew silently changes this checkpoint's
   positional basis by 500x and moves holdout NLL by 0.35 nats without raising.
9. **calibration provenance** — the search's calibration set is the frozen one,
   its leakage report is clean, and the map on disk came from that search.

    PYTHONPATH=src python scripts/training/validate_e8_arms.py \\
        --require-init --out artifacts/audit/e8_preflight.json

Exit codes: 0 all checks pass; 6 at least one failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.init.nll_gate import (  # noqa: E402
    REQUIRED_MEASUREMENTS, InitNllGateError, checkpoint_fingerprint,
    gate_summary, require_init_nll,
)
from aadistill.init.sandwich import depth_span_map  # noqa: E402

ALLOWED_DIFF = {"student_path", "run_name", "out_dir", "_purpose"}
EXPECTED_UNIQUE_CE = 2_960_507
EXPECTED_CUMULATIVE_CE = 8_881_521
EXPECTED_EXPOSURES = 3.0
BASELINE_INIT = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
BASELINE_NLL = "artifacts/stage1/qwen3_0p6b_init_v0/init_nll.json"
TREATMENT_INIT = "artifacts/stage1/e8_contribution_init_v1/checkpoint"
TREATMENT_NLL = "artifacts/stage1/e8_contribution_init_v1/init_nll.json"
DEPTH_MAP = "artifacts/stage1/e8_depth_search/depth_map.json"
CALIBRATION = "artifacts/stage1/e8_calibration_v1"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="configs/stage3/e8/arms.json")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--require-init", action="store_true",
                    help="require the initialization checkpoints and their NLL "
                         "records to exist (the pre-training gate). Without it "
                         "only the config-level checks run.")
    ap.add_argument("--require-baseline-nll", action="store_true", default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    checks: list[dict] = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    arms_meta = json.loads(resolve(args.arms).read_text())
    arms = arms_meta["arms"]

    # 1 / 2 — config identity and a shared fork point.
    inits = set()
    for arm in arms:
        cfg = json.loads(resolve(arm["path"]).read_text())
        control = json.loads(resolve(arm["control"]).read_text())
        realized = {k for k in set(cfg) | set(control)
                    if json.dumps(cfg.get(k), sort_keys=True)
                    != json.dumps(control.get(k), sort_keys=True)}
        check(f"config_diff:{arm['name']}", realized == ALLOWED_DIFF,
              {"realized": sorted(realized), "allowed": sorted(ALLOWED_DIFF),
               "config_sha256": sha256_json(cfg),
               "control_sha256": sha256_json(control)})
        inits.add(cfg["student_path"])
        for key in ("loss", "optim", "schedule", "batch", "trainable_patterns",
                    "dtype", "autocast_bf16", "gradient_checkpointing", "rung",
                    "block_len", "data_dir", "packing", "intervals", "val_blocks"):
            if json.dumps(cfg.get(key), sort_keys=True) != \
                    json.dumps(control.get(key), sort_keys=True):
                check(f"unchanged:{arm['name']}:{key}", False,
                      {"treatment": cfg.get(key), "control": control.get(key)})
    check("single_treatment_init", len(inits) == 1 and inits == {TREATMENT_INIT},
          {"student_paths": sorted(inits)})
    check("two_seeds", len({a["seed"] for a in arms}) == 2,
          {"seeds": sorted(a["seed"] for a in arms)})

    # 3 — the budget, from the loader.
    from aadistill.data.ladder import ladder_blocks
    rung = arms[0]["rung"]
    train, _, _ = ladder_blocks(resolve(args.pack), rung,
                               n_val=json.loads(
                                   resolve(arms[0]["path"]).read_text())["val_blocks"])
    ids, mask, content = train
    unique_ce = int(mask.sum())
    n_blocks = int(ids.shape[0])
    steps = arms[0]["total_steps"]
    bps = arms[0]["blocks_per_step"]
    exposures = steps * bps / n_blocks
    cumulative = int(unique_ce * exposures)
    check("unique_ce_tokens", unique_ce == EXPECTED_UNIQUE_CE,
          {"measured": unique_ce, "expected": EXPECTED_UNIQUE_CE,
           "blocks": n_blocks})
    check("exposures", abs(exposures - EXPECTED_EXPOSURES) < 1e-9,
          {"measured": exposures, "steps": steps, "blocks_per_step": bps})
    check("cumulative_ce_exposure", cumulative == EXPECTED_CUMULATIVE_CE,
          {"measured": cumulative, "expected": EXPECTED_CUMULATIVE_CE})
    check("kd_positions_per_exposure", True,
          {"content_masked_shifted": int(content[:, 1:].sum()),
           "note": "recorded, not asserted: kd_scope=all is copied from the "
                   "control so this cannot differ between the arms"})

    positional_kept = [s["representative"] for s in depth_span_map(36, 28)]
    summaries = {}

    if args.require_init:
        # 6 / 9 — the map, and where it came from.
        dm_path = resolve(DEPTH_MAP)
        if not dm_path.is_file():
            check("depth_map_present", False, {"path": DEPTH_MAP})
        else:
            dm = json.loads(dm_path.read_text())
            kept = dm.get("kept_teacher_layers") or []
            check("depth_map_shape",
                  len(kept) == 28 and kept == sorted(set(kept))
                  and all(0 <= k < 36 for k in kept),
                  {"kept": kept, "removed": dm.get("removed_teacher_layers")})
            check("depth_map_differs_from_positional", kept != positional_kept,
                  {"contribution": kept, "positional": positional_kept,
                   "note": "identical maps would mean the search recovered the "
                           "positional heuristic — a real result, but not a "
                           "trainable treatment"})
            check("depth_map_from_frozen_search",
                  bool(dm.get("search_report_sha256"))
                  and bool(dm.get("calibration_content_sha256")),
                  {"search_report_sha256": dm.get("search_report_sha256"),
                   "calibration_content_sha256":
                       dm.get("calibration_content_sha256")})
            cal_manifest = resolve(CALIBRATION) / "manifest.json"
            leak = resolve(CALIBRATION) / "leakage.json"
            if cal_manifest.is_file() and leak.is_file():
                cm = json.loads(cal_manifest.read_text())
                lk = json.loads(leak.read_text())
                check("calibration_matches_search",
                      cm.get("content_sha256")
                      == dm.get("calibration_content_sha256"),
                      {"calibration": cm.get("content_sha256"),
                       "search_used": dm.get("calibration_content_sha256")})
                check("calibration_leakage_clean", bool(lk.get("clean")),
                      {"findings": lk.get("findings"),
                       "checks": lk.get("checks")})
            else:
                check("calibration_present", False,
                      {"manifest": str(cal_manifest), "leakage": str(leak)})

        # 4 / 5 / 7 / 8 — the checkpoints and their NLL records.
        for label, ckpt_rel, rec_rel, required in (
                ("treatment", TREATMENT_INIT, TREATMENT_NLL, True),
                ("baseline", BASELINE_INIT, BASELINE_NLL,
                 args.require_baseline_nll)):
            ckpt = resolve(ckpt_rel)
            if not (ckpt / "model.safetensors").is_file():
                check(f"init_present:{label}", False, {"path": ckpt_rel})
                continue
            try:
                record = require_init_nll(ckpt, resolve(rec_rel))
                summaries[label] = gate_summary(record)
                check(f"init_nll_gate:{label}", True,
                      {"record": rec_rel,
                       "measurements": sorted(record["measurements"]),
                       "record_sha256": record.get("record_sha256")})
            except InitNllGateError as exc:
                check(f"init_nll_gate:{label}", not required, {"error": str(exc)})

        base_fp = checkpoint_fingerprint(resolve(BASELINE_INIT))
        treat = resolve(TREATMENT_INIT)
        if (treat / "model.safetensors").is_file():
            treat_fp = checkpoint_fingerprint(treat)
            check("geometry_unchanged",
                  treat_fp["config_sha256"] == base_fp["config_sha256"],
                  {"baseline": base_fp["config_sha256"],
                   "treatment": treat_fp["config_sha256"],
                   "note": "only the depth map changed, so the student config "
                           "must be byte-identical"})
            check("weights_differ",
                  treat_fp["model_sha256"] != base_fp["model_sha256"],
                  {"baseline": base_fp["model_sha256"],
                   "treatment": treat_fp["model_sha256"]})
            from transformers import AutoConfig

            from aadistill.models.student import assert_rope_from_config
            ok_rope = True
            bases = {}
            for name, path in (("baseline", resolve(BASELINE_INIT)),
                               ("treatment", treat)):
                cfg = AutoConfig.from_pretrained(str(path))
                try:
                    bases[name] = assert_rope_from_config(cfg, str(path))
                except ValueError as exc:
                    ok_rope = False
                    bases[name] = str(exc)
            check("rope_base_resolves", ok_rope, bases)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "arms": [{k: a[k] for k in ("name", "path", "control", "config_sha256",
                                    "seed", "rung", "total_steps")} for a in arms],
        "budget": {"unique_ce_tokens": unique_ce,
                   "cumulative_ce_exposure": cumulative,
                   "blocks": n_blocks, "exposures": exposures},
        "required_init_measurements": list(REQUIRED_MEASUREMENTS),
        "step0_summaries": summaries,
        "checks": checks,
        "failed": [c["check"] for c in checks if not c["ok"]],
        "all_passed": all(c["ok"] for c in checks),
        "init_checks_run": bool(args.require_init),
        "pack_ladder_sha256": sha256_file(resolve(args.pack) / "ladder.json"),
    }
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"-> {out}")
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['check']}")
    if report["failed"]:
        print(f"\nFAILED: {report['failed']}")
    return 0 if report["all_passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())

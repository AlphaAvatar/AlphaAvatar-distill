#!/usr/bin/env python3
"""The gate every E8b session must pass before it trains or measures anything.

    PYTHONPATH=src python scripts/training/validate_e8b_arms.py --session s2 \
        --require-init --out artifacts/audit/e8b_s2_preflight.json

Scoped by session, because E8b's four sessions stage different things and a check
that demands an artifact its session has no reason to hold is how an E8a pod died:

* `--session s1` — the step-0 session. Requires all four initializations present and,
  with `--require-init`, all four NLL records hash-bound.
* `--session s2|s3` — the depth-only training sessions. Require DP and DC, their
  configs, and (with `--require-init`) their NLL records.
* `--session s4` — the compressed training session. Requires FP and FC.

What every session checks regardless: each arm's config diff against its cell's
control is exactly `{student_path, run_name, out_dir, _purpose}`; the two members of
each depth-map pair differ only in `student_path`; the token budget is re-derived
from the pack rather than copied from a document; and the hardware nesting is
recorded so no later reader can mistake the interaction for hardware-controlled.

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

ALLOWED_DIFF = {"student_path", "run_name", "out_dir", "_purpose"}
# The depth-only regime additionally carries `loss.kd_chunk = 128`, the preregistered
# memory fallback adopted after `expandable_segments:True` alone still OOM'd at step 110
# of 1,761. It is regime-wide by construction — all four of DP-sa, DC-sa, DP-sb, DC-sb —
# so DP-vs-DC remains a single-variable comparison, and FC is untouched so FP-vs-FC does
# too. Without this entry the gate would reject the arms it is meant to protect.
DEPTH_ONLY_EXTRA_DIFF = {"loss"}
EXPECTED_KD_CHUNK = {"depth_only": 128, "fully_compressed": None}
EXPECTED_UNIQUE_CE = 1_600_353
EXPECTED_CUMULATIVE_CE = 4_801_059
EXPECTED_EXPOSURES = 3.0
EXPECTED_BLOCKS = 1174

INITS = {
    "DP": ("artifacts/stage1/e8b_dp_init", 3_215_021_568,
           "d4db65eb8f7ae6d8a847c2db9a9e5e307e449f50f3bd129e07a1b20f6ec5f3cd"),
    "DC": ("artifacts/stage1/e8b_dc_init", 3_215_021_568,
           "eb9e95481988b296a77c30d7b4754069f1874330fca9ad198f4457029e11e182"),
    "FP": ("artifacts/stage1/qwen3_0p6b_init_v0", 596_049_920,
           "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"),
    "FC": ("artifacts/stage1/e8_contribution_init_v1", 596_049_920,
           "7a0694a5d5c59f8e0b0ebc9ac8648b1ec026bf93cab026d33c61ca8fc85d1edb"),
}
SESSION_INITS = {"s1": ("DP", "DC", "FP", "FC"), "s2": ("DP", "DC"),
                 "s3": ("DP", "DC"), "s4": ("FP", "FC")}
SESSION_ARMS = {
    "s1": (),
    "s2": ("e8b_dp_r1600k_sa", "e8b_dc_r1600k_sa"),
    "s3": ("e8b_dp_r1600k_sb", "e8b_dc_r1600k_sb"),
    "s4": ("e8b_fc_r1600k_sa", "e8b_fc_r1600k_sb"),
}
CONTRIBUTION_REMOVED = [2, 3, 15, 16, 20, 21, 26, 32]
POSITIONAL_REMOVED = [5, 7, 9, 11, 13, 15, 17, 19]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, choices=sorted(SESSION_INITS))
    ap.add_argument("--arms", default="configs/stage3/e8b/arms.json")
    ap.add_argument("--pack", default="artifacts/stage3/ladder_uniform_probe")
    ap.add_argument("--require-init", action="store_true",
                    help="also require each initialization's own hash-bound NLL "
                         "record — the pre-training half of the gate")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    checks: list[dict] = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    meta = json.loads(resolve(args.arms).read_text())
    by_name = {a["name"]: a for a in meta["arms"]}
    wanted_arms = SESSION_ARMS[args.session]

    # --- config identity, for this session's arms -----------------------------
    for name in wanted_arms:
        arm = by_name[name]
        cfg = json.loads(resolve(arm["path"]).read_text())
        control = json.loads(resolve(arm["control"]).read_text())
        realized = {k for k in set(cfg) | set(control)
                    if json.dumps(cfg.get(k), sort_keys=True)
                    != json.dumps(control.get(k), sort_keys=True)}
        regime = arm.get("regime", "")
        expected_diff = ALLOWED_DIFF | (DEPTH_ONLY_EXTRA_DIFF
                                        if regime == "depth_only" else set())
        want_chunk = EXPECTED_KD_CHUNK.get(regime, None)
        check(f"kd_chunk:{name}", cfg["loss"].get("kd_chunk") == want_chunk,
              {"regime": regime, "measured": cfg["loss"].get("kd_chunk"),
               "expected": want_chunk,
               "note": "regime-wide by construction; one arm or one seed differing "
                       "would break the pair's single-variable comparison"})
        check(f"config_diff:{name}", realized == expected_diff,
              {"realized": sorted(realized), "config_sha256": sha256_json(cfg),
               "recorded_sha256": arm["config_sha256"],
               "expected_diff": sorted(expected_diff),
               "matches_manifest": sha256_json(cfg) == arm["config_sha256"]})
        # The OBJECTIVE must be the canonical one. `kd_chunk` is excluded here and
        # checked separately above: it is a memory/time knob that leaves the objective
        # identical, so folding it into this equality would make a memory decision look
        # like a change of loss.
        objective = {k: v for k, v in cfg["loss"].items() if k != "kd_chunk"}
        check(f"rung:{name}", cfg["rung"] == 1_600_000 and
              cfg["schedule"]["total_steps"] == 1761 and
              objective == {"ce_weight": 0.25, "kd_weight": 1.0,
                            "kd_temperature": 1.0, "kd_scope": "all"},
              {"rung": cfg["rung"], "steps": cfg["schedule"]["total_steps"],
               "objective": objective, "kd_chunk": cfg["loss"].get("kd_chunk"),
               "save_every": cfg["checkpoint"]["save_every"]})

    # --- within-pair identity: the single-variable claim, per hardware class ---
    # Still exactly the four keys, because the chunk is regime-wide: both members of
    # each pair carry the same value, so it cancels inside the comparison.
    for pair in meta["within_cell_identity"]:
        check(f"pair_identity:{pair['pair']}",
              set(pair["diff"]) == ALLOWED_DIFF, pair)

    # --- the budget, from the loader ------------------------------------------
    from aadistill.data.ladder import ladder_blocks
    train, _, _ = ladder_blocks(resolve(args.pack), 1_600_000, n_val=16)
    ids, mask, content = train
    unique = int(mask.sum())
    blocks = int(ids.shape[0])
    exposures = 1761 * 2 / blocks
    check("blocks", blocks == EXPECTED_BLOCKS, {"measured": blocks})
    check("unique_ce_tokens", unique == EXPECTED_UNIQUE_CE, {"measured": unique})
    check("exposures", abs(exposures - EXPECTED_EXPOSURES) < 1e-9,
          {"measured": exposures})
    check("cumulative_ce_exposure",
          int(unique * exposures) == EXPECTED_CUMULATIVE_CE,
          {"measured": int(unique * exposures)})

    # --- initializations this session needs -----------------------------------
    summaries = {}
    for cell in SESSION_INITS[args.session]:
        base, n_params, sha = INITS[cell]
        ckpt = resolve(base) / "checkpoint"
        if not (ckpt / "model.safetensors").is_file():
            check(f"init_present:{cell}", False, {"path": str(ckpt)})
            continue
        fp = checkpoint_fingerprint(ckpt)
        check(f"init_hash:{cell}", fp["model_sha256"] == sha,
              {"measured": fp["model_sha256"], "expected": sha})
        manifest = resolve(base) / "manifest.json"
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            got = (m.get("student") or {}).get("num_parameters")
            check(f"init_params:{cell}", got == n_params,
                  {"measured": got, "expected": n_params})
            v = m.get("verification")
            if cell in ("DP", "DC"):
                check(f"init_is_ablated_teacher:{cell}",
                      bool(v and v.get("bitwise_identical_to_ablated_teacher")),
                      v or {"error": "no verification record"})
        if args.require_init:
            try:
                rec = require_init_nll(ckpt, resolve(base) / "init_nll.json")
                summaries[cell] = gate_summary(rec)
                check(f"init_nll_gate:{cell}", True,
                      {"measurements": sorted(rec["measurements"]),
                       "record_sha256": rec.get("record_sha256"),
                       "device": rec.get("device"),
                       "environment": rec.get("environment")})
            except InitNllGateError as exc:
                check(f"init_nll_gate:{cell}", False, {"error": str(exc)})

    # Within-regime geometry must match, and the depth maps must differ.
    for a, b in (("DP", "DC"), ("FP", "FC")):
        if not all(c in SESSION_INITS[args.session] for c in (a, b)):
            continue
        pa = resolve(INITS[a][0]) / "checkpoint"
        pb = resolve(INITS[b][0]) / "checkpoint"
        if not ((pa / "config.json").is_file() and (pb / "config.json").is_file()):
            continue
        fa, fb = checkpoint_fingerprint(pa), checkpoint_fingerprint(pb)
        check(f"geometry_matched:{a}_{b}",
              fa["config_sha256"] == fb["config_sha256"],
              {a: fa["config_sha256"], b: fb["config_sha256"]})
        check(f"weights_differ:{a}_{b}",
              fa["model_sha256"] != fb["model_sha256"], {})

    if args.require_init and len(SESSION_INITS[args.session]) == 4:
        # The step-0 table is only comparable if one device and one environment
        # produced all four rows.
        devs = {c: (summaries.get(c) or {}) for c in ("DP", "DC", "FP", "FC")}
        check("step0_single_evaluator", len(summaries) == 4,
              {"measured_cells": sorted(summaries), "note": devs and "see records"})

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "session": args.session,
        "hardware_design": meta["hardware_design"],
        "arms_checked": list(wanted_arms),
        "initializations_checked": list(SESSION_INITS[args.session]),
        "depth_maps": {"positional_removed": POSITIONAL_REMOVED,
                       "contribution_removed": CONTRIBUTION_REMOVED},
        "budget": {"blocks": blocks, "unique_ce_tokens": unique,
                   "exposures": exposures,
                   "cumulative_ce_exposure": int(unique * exposures)},
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

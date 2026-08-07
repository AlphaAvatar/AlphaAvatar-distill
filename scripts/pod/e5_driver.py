#!/usr/bin/env python
"""Formal E5: validation gate -> R generation -> joint feasibility -> four arms.

    /opt/train/bin/python scripts/pod/e5_driver.py --stage all

The separate pilot pod was removed from the plan after two attempts showed that
**setup dominates a short session** — 53 of 57 minutes, paid again on every pod.
The validation is not removed: it runs here, first, on the same pod, so setup is
paid once. Two failed pilot attempts ($0.07 and $0.92) stand in the record; they
found the relay tree-enumeration timeout, the gitignored battery dependency and
the nondeterministic rollout seed, all fixed before this run.

Stages, each skipped if its output already exists so a restart resumes:

1. **validate** — the full real-engine gate. If it fails the run STOPS: no paid
   generation follows a failed gate.
2. **generate** — full R corpus for both seeds.
3. **pair** — intersection, token-target selection, packing, and the **joint
   feasibility gate**. Training does not start unless every registered condition
   holds simultaneously.
4. **train** — the four registered arms: C-sa, C-sb, R-sa, R-sb.
5. **evaluate** — the four checkpoints on the pinned battery.

Markers: VALIDATED -> GENERATED:<seed> -> PAIRED -> FEASIBLE/INFEASIBLE
         -> TRAIN_DONE:<arm> -> EVAL_DONE:<arm> -> ALL_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/workspace/aad")
STATUS = Path("/workspace/e5.status")
OUT = REPO / "artifacts/audit"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
SEEDS = ("sa", "sb")
STEP = "step_000738"
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
TARGET_CE_TOKENS = 735_603
TOLERANCE = 0.05
BLOCKS, STEPS = 492, 738
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(cmd, py=TRAIN_PY):
    cmd = [py] + [str(c) for c in cmd]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")})


def arm_dir(arm: str, seed: str) -> Path:
    return REPO / f"artifacts/stage3/e5_{arm}_{seed}"


def stage_validate(args):
    """The folded pilot. A failure here stops the run before paid generation."""
    out = OUT / "e5_validation.json"
    if out.exists() and json.loads(out.read_text()).get("passed"):
        print("validation already passed; skipping", flush=True)
        return mark("VALIDATED")
    run(["scripts/pod/e5_pilot.py", "--limit", args.validate_limit,
         "--seed", "sa", "--student", "/workspace/ckpt/p2_ceheavy_sa",
         "--out", out])
    rep = json.loads(out.read_text())
    if not rep.get("passed"):
        mark("VALIDATION_FAILED")
        raise AssertionError("validation gate failed; refusing paid generation")
    mark("VALIDATED")


def stage_generate(args):
    for seed in SEEDS:
        d = REPO / f"artifacts/stage3/e5_arm_r_{seed}"
        if (d / "examples.jsonl").exists():
            print(f"R corpus for {seed} exists; skipping", flush=True)
            mark(f"GENERATED:{seed}")
            continue
        run(["scripts/data/build_e5_arm_r.py",
             "--student", f"/workspace/ckpt/p2_ceheavy_{seed}",
             "--source-seed", seed, "--out", d], py=VLLM_PY)
        mark(f"GENERATED:{seed}")
        # Arm C for the same seed costs nothing and must exist before pairing.
        c = REPO / f"artifacts/stage3/e5_arm_c_{seed}"
        if not (c / "examples.jsonl").exists():
            run(["scripts/data/build_e5_arm_c.py", "--source-seed", seed, "--out", c])


def stage_pair(args):
    """Intersection, token targeting, packing — then the joint feasibility gate."""
    sys.path.insert(0, str(REPO / "src"))
    from aadistill.data.paired_corpus import (
        comparability_report, intersect, packing_report,
        select_paired_to_token_target, suffix_overlap,
    )
    report = {"target_ce_tokens": TARGET_CE_TOKENS, "tolerance": TOLERANCE,
              "blocks": BLOCKS, "steps": STEPS, "per_seed": {}}
    conditions = []
    for seed in SEEDS:
        c_rows = [json.loads(l) for l in
                  (REPO / f"artifacts/stage3/e5_arm_c_{seed}/examples.jsonl").open()
                  if l.strip()]
        r_rows = [json.loads(l) for l in
                  (REPO / f"artifacts/stage3/e5_arm_r_{seed}/examples.jsonl").open()
                  if l.strip()]
        ck, rk, census = intersect(c_rows, r_rows)
        c_sel, r_sel, sel = select_paired_to_token_target(ck, rk, TARGET_CE_TOKENS)
        comp = comparability_report(c_sel, r_sel, supervised_tolerance=TOLERANCE)
        pack_c = packing_report(c_sel, BLOCKS, 8192)
        pack_r = packing_report(r_sel, BLOCKS, 8192)
        entry = {
            "census": census, "selection": sel, "comparability": comp,
            "packing_C": pack_c, "packing_R": pack_r,
            "overlap_C": suffix_overlap(c_sel), "overlap_R": suffix_overlap(r_sel),
            "ce_token_presentations_over_3_passes": {
                "C": pack_c["ce_mask_tokens"] * 3,
                "R": pack_r["ce_mask_tokens"] * 3},
        }
        report["per_seed"][seed] = entry
        for c_sel_rows, arm, pack in ((c_sel, "C", pack_c), (r_sel, "R", pack_r)):
            hit = abs(pack["ce_mask_tokens"] - TARGET_CE_TOKENS) / TARGET_CE_TOKENS
            conditions.append((f"{seed}/{arm} CE tokens within tolerance",
                               hit <= TOLERANCE, f"{pack['ce_mask_tokens']:,} "
                               f"({hit:.1%} from target)"))
            conditions.append((f"{seed}/{arm} fits {BLOCKS} blocks", pack["fits"],
                               f"{pack['total_nonpadding_tokens']:,} tokens"))
        conditions.append((f"{seed} C/R within tolerance", comp["within_tolerance"],
                           f"delta {comp['supervised_token_relative_delta']}"))
        conditions.append((f"{seed} identical composition",
                           len(c_sel) == len(r_sel), f"{len(c_sel)} vs {len(r_sel)}"))
        bundles_ok = all(
            len([e for e in c_sel if e["source_session_id"] == sid]) == 2
            for sid in {e["source_session_id"] for e in c_sel})
        conditions.append((f"{seed} atomic two-truncation bundles", bundles_ok, ""))
        for arm, rows in (("C", c_sel), ("R", r_sel)):
            path = REPO / f"artifacts/stage3/e5_final_{arm}_{seed}.jsonl"
            path.write_text("".join(json.dumps(e) + "\n" for e in rows))

    report["conditions"] = [{"name": n, "passed": bool(p), "detail": d}
                            for n, p, d in conditions]
    report["feasible"] = all(p for _, p, _ in conditions)
    (OUT / "e5_joint_feasibility.json").write_text(json.dumps(report, indent=1))
    for n, p, d in conditions:
        print(f"  [{'PASS' if p else 'FAIL'}] {n} {d}", flush=True)
    mark("PAIRED")
    if not report["feasible"]:
        mark("INFEASIBLE")
        raise AssertionError("joint feasibility gate failed; refusing to train")
    mark("FEASIBLE")


def stage_train(args):
    for arm in ("c", "r"):
        for seed in SEEDS:
            name = f"e5_{arm}_{seed}"
            if (arm_dir(arm, seed) / f"checkpoints/{STEP}/model").is_dir():
                mark(f"TRAIN_DONE:{name}")
                continue
            cfg = REPO / f"configs/stage3/e5/{name}.json"
            if not cfg.is_file():
                mark(f"TRAIN_SKIPPED:{name}:no_config")
                continue
            run(["scripts/training/train_stage3.py", "--config", cfg])
            mark(f"TRAIN_DONE:{name}")


def stage_evaluate(args):
    from transformers import AutoTokenizer
    pack = REPO / "artifacts/stage3/ladder_uniform_probe"
    sessions = REPO / "artifacts/stage3/corpus_v2/sessions.jsonl"
    for arm in ("c", "r"):
        for seed in SEEDS:
            name, alias = f"e5_{arm}_{seed}", f"E5-{arm.upper()}-{seed}"
            m = arm_dir(arm, seed) / f"checkpoints/{STEP}/model"
            if not m.is_dir():
                mark(f"EVAL_SKIPPED:{alias}")
                continue
            for f in TOKENIZER_FILES:
                shutil.copy(INIT / f, m / f)
            AutoTokenizer.from_pretrained(str(m))
            d = OUT / "three_mode" / alias
            if (d / "oracle.generations.jsonl").exists():
                mark(f"EVAL_DONE:{alias}")
                continue
            run(["scripts/evaluation/run_three_mode_diagnostic.py", "--student", m,
                 "--label", alias, "--pack", pack, "--rung", 860000,
                 "--sessions", sessions, "--n", 150, "--modes", "free", "oracle",
                 "--out", d], py=VLLM_PY)
            run(["scripts/evaluation/run_three_mode_diagnostic.py", "--student", m,
                 "--label", alias, "--pack", pack, "--rung", 860000,
                 "--sessions", sessions, "--n", 150, "--modes", "forced",
                 "--out", d / "forced"])
            mask = json.loads((d / "report.json").read_text())["inclusion"]["mask_sha256"]
            assert mask == EXPECTED_MASK, f"{alias}: mask {mask} != binding"
            mark(f"EVAL_DONE:{alias}")


STAGES = {"validate": stage_validate, "generate": stage_generate,
          "pair": stage_pair, "train": stage_train, "evaluate": stage_evaluate}
BLOCKING = ("validate", "pair")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--validate-limit", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        except (subprocess.CalledProcessError, AssertionError, OSError) as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            if name in BLOCKING and args.stage == "all":
                mark("ABORTED_AT_GATE")
                break
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

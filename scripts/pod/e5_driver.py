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
STEP = None        # resolved from the measured step count at train time
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
TARGET_CE_TOKENS = 735_603
TOLERANCE = 0.05
BLOCKS, STEPS = 492, 738
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REV = "768f209d9ea81521153ed38c47d515654e938aea"
SEC_PER_STEP = 3.61        # measured full-width; divided by the measured speedup
RATE = 0.99


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

    # Common block count = max(C_min, R_min) across both arms and seeds, then
    # both arms are repacked to exactly that count. The easier-to-pack arm
    # carries additional ordinary padding; nothing is duplicated or truncated.
    from aadistill.data.e5_pack import pack_e5, write_pack, verify_pack
    minima = {}
    for seed in SEEDS:
        sysids = json.loads((REPO / f"artifacts/stage3/e5_arm_c_{seed}"
                             / "system_ids.json").read_text())
        for arm in ("C", "R"):
            rows = [json.loads(l) for l in
                    (REPO / f"artifacts/stage3/e5_final_{arm}_{seed}.jsonl").open()
                    if l.strip()]
            minima[(arm, seed)] = len(pack_e5(rows, sysids, block_len=8192))
    common = max(minima.values())
    report["per_arm_minimum_blocks"] = {f"{a}_{s_}": v for (a, s_), v in minima.items()}
    report["common_block_count"] = common
    report["optimizer_steps"] = common * 3 // 2
    print(f"  minima {report['per_arm_minimum_blocks']} -> common {common} blocks, "
          f"{report['optimizer_steps']} steps", flush=True)
    for seed in SEEDS:
        sysids = json.loads((REPO / f"artifacts/stage3/e5_arm_c_{seed}"
                             / "system_ids.json").read_text())
        for arm in ("C", "R"):
            rows = [json.loads(l) for l in
                    (REPO / f"artifacts/stage3/e5_final_{arm}_{seed}.jsonl").open()
                    if l.strip()]
            out = REPO / f"artifacts/stage3/e5_pack_{arm.lower()}_{seed}"
            blocks = pack_e5(rows, sysids, block_len=8192, target_blocks=common)
            write_pack(blocks, out, arm=arm.lower(), seed=seed, block_len=8192,
                       pad_id=151643, target_ce_tokens=TARGET_CE_TOKENS)
            v = verify_pack(out, rows, expected_blocks=common,
                            target_ce_tokens=TARGET_CE_TOKENS, tolerance=TOLERANCE,
                            steps=report["optimizer_steps"], blocks_per_step=2)
            report.setdefault("packed", {})[f"{arm}_{seed}"] = v
            conditions.append((f"{seed}/{arm} packed artifact verified",
                               v["passed"], str(v["failures"])))

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
            step_tag = f"step_{json.loads((OUT / 'e5_joint_feasibility.json').read_text())['optimizer_steps']:06d}"
            if (arm_dir(arm, seed) / f"checkpoints/{step_tag}/model").is_dir():
                mark(f"TRAIN_DONE:{name}")
                continue
            cfg_path = REPO / f"configs/stage3/e5/{name}.json"
            feas = json.loads((OUT / "e5_joint_feasibility.json").read_text())
            if cfg_path.is_file():
                c = json.loads(cfg_path.read_text())
                c["data_dir"] = f"artifacts/stage3/e5_pack_{arm}_{seed}"
                c["rung"] = TARGET_CE_TOKENS
                steps = feas["optimizer_steps"]
                c["schedule"] = {**c["schedule"], "total_steps": steps,
                                 "warmup_steps": max(1, round(steps * 0.05))}
                c["checkpoint"] = {"save_every": steps // 2, "keep_last": 1}
                c["intervals"] = {**c["intervals"], "eval_every": max(1, steps // 6)}
                cfg_path.write_text(json.dumps(c, indent=1) + "\n")
            cfg = cfg_path
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
            step_tag = f"step_{json.loads((OUT / 'e5_joint_feasibility.json').read_text())['optimizer_steps']:06d}"
            m = arm_dir(arm, seed) / f"checkpoints/{step_tag}/model"
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


def _pack_c(seed: str) -> Path:
    """Build the token-matched C pack. Free, and the benchmark needs real blocks."""
    sys.path.insert(0, str(REPO / "src"))
    from aadistill.data.e5_pack import pack_e5, write_pack
    from aadistill.data.paired_corpus import (
        as_bundles, intersect, select_paired_to_token_target,
    )
    out = REPO / f"artifacts/stage3/e5_pack_c_{seed}"
    if (out / "blocks.npz").is_file():
        return out
    d = REPO / f"artifacts/stage3/e5_arm_c_{seed}"
    ex = [json.loads(l) for l in (d / "examples.jsonl").open() if l.strip()]
    sysids = json.loads((d / "system_ids.json").read_text())
    ck, rk, _ = intersect(ex, [dict(e) for e in ex])
    c_sel, _, sel = select_paired_to_token_target(ck, rk, TARGET_CE_TOKENS)
    blocks = pack_e5(c_sel, sysids, block_len=8192)
    write_pack(blocks, out, arm="c", seed=seed, block_len=8192, pad_id=151643,
               target_ce_tokens=TARGET_CE_TOKENS, extra={"selection": sel})
    print(f"C pack {seed}: {len(blocks)} blocks, "
          f"{sel['arm_c_supervised']:,} CE tokens", flush=True)
    return out


def stage_benchmark(args):
    """Measure the truncate_padding speedup on real E5-C blocks, before paying for R.

    Placed immediately after validation on purpose: the C pack already exists, so
    there is no reason to buy the recovery corpus before knowing whether the
    training path fits the remaining budget.
    """
    out = OUT / "e5_throughput.json"
    if out.exists():
        print("throughput benchmark already done; skipping", flush=True)
        return mark("BENCHMARKED")
    pack = _pack_c("sa")
    run(["scripts/training/benchmark_e5_throughput.py", "--pack", pack,
         "--student", "/workspace/ckpt/p2_ceheavy_sa",
         "--teacher", f"{TEACHER}@{TEACHER_REV}",
         "--steps", args.bench_steps, "--out", out])
    mark("BENCHMARKED")


def _budget(spent_usd: float, remaining_usd: float, blocks: int, speedup: float,
            *, phases_min: dict) -> dict:
    """Project the rest of the run at the MEASURED speedup."""
    train_min = 4 * (blocks * 3 / 2) * (SEC_PER_STEP / max(0.01, speedup)) / 60
    rest = sum(phases_min.values()) + train_min
    expected = rest / 60 * RATE
    backstop = rest * 1.12 / 60 * RATE
    return {"blocks": blocks, "speedup": round(speedup, 3),
            "remaining_phases_min": {**phases_min, "train": round(train_min)},
            "remaining_minutes": round(rest),
            "expected_usd": round(expected, 2),
            "backstop_usd": round(backstop, 2),
            "authorization_remaining_usd": round(remaining_usd, 2),
            "covered": backstop <= remaining_usd,
            "shortfall_usd": round(max(0.0, backstop - remaining_usd), 2)}


def stage_budget_gate_1(args):
    """Before paying for R: does the measured speedup make the rest affordable?"""
    bench = json.loads((OUT / "e5_throughput.json").read_text())
    speedup = bench["measured_wall_clock_speedup"]
    spent = args.spent_usd + bench["benchmark_cost_usd"]
    rep = _budget(spent, args.authorized_usd - spent, args.assumed_blocks, speedup,
                  phases_min={"r_generation": 152, "pair_pack": 20,
                              "evaluate": 44, "transfer_teardown": 35})
    rep["gate"] = "pre-generation"
    rep["assumed_blocks_rationale"] = ("conservative R = 1.30x the measured C "
                                       "minimum; replaced by the real common "
                                       "block count at gate 2")
    (OUT / "e5_budget_gate1.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    mark("BUDGET_GATE_1_PASS" if rep["covered"] else "BUDGET_GATE_1_FAIL")
    if not rep["covered"]:
        raise AssertionError(
            f"remaining ${rep['authorization_remaining_usd']} does not cover "
            f"${rep['backstop_usd']}; short ${rep['shortfall_usd']}")


def stage_budget_gate_2(args):
    """After pairing: re-gate on the ACTUAL common block count."""
    bench = json.loads((OUT / "e5_throughput.json").read_text())
    feas = json.loads((OUT / "e5_joint_feasibility.json").read_text())
    blocks = feas["common_block_count"]
    rep = _budget(args.spent_usd, args.authorized_usd - args.spent_usd, blocks,
                  bench["measured_wall_clock_speedup"],
                  phases_min={"evaluate": 44, "transfer_teardown": 35})
    rep["gate"] = "pre-training"
    (OUT / "e5_budget_gate2.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    mark("BUDGET_GATE_2_PASS" if rep["covered"] else "BUDGET_GATE_2_FAIL")
    if not rep["covered"]:
        raise AssertionError(
            f"remaining ${rep['authorization_remaining_usd']} does not cover "
            f"training+eval backstop ${rep['backstop_usd']}; "
            f"short ${rep['shortfall_usd']}")


# Order matters and is the point: measure the training path BEFORE buying the
# recovery corpus, and re-gate on the real block count before training.
STAGES = {"validate": stage_validate, "benchmark": stage_benchmark,
          "budget_gate_1": stage_budget_gate_1, "generate": stage_generate,
          "pair": stage_pair, "budget_gate_2": stage_budget_gate_2,
          "train": stage_train, "evaluate": stage_evaluate}
BLOCKING = ("validate", "benchmark", "budget_gate_1", "pair", "budget_gate_2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--validate-limit", type=int, default=24)
    ap.add_argument("--bench-steps", type=int, default=12)
    ap.add_argument("--authorized-usd", type=float, default=9.12)
    ap.add_argument("--spent-usd", type=float, default=0.0,
                    help="pod spend so far, supplied by the launcher")
    ap.add_argument("--assumed-blocks", type=int, default=1123,
                    help="conservative R=1.30x C, used only at gate 1")
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

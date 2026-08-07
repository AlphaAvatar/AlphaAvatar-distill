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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
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
T0 = time.time()
CLAIM_BOUNDARY = (
    "Evaluation remains paired on the fixed 150-prompt battery, but training "
    "composition is no longer identical. E5 therefore estimates the performance "
    "of the complete teacher-prefix-continuation versus student-prefix-recovery "
    "RECIPES under a matched supervised-token budget. It does NOT isolate the "
    "pure causal effect of prefix content/state with training composition held "
    "constant. Primary estimand: which continuation recipe produces better "
    "autonomous behaviour per fixed CE-supervision budget? The paired "
    "McNemar/bootstrap statistics preserve paired EVALUATION; they do not remove "
    "the training-composition difference and must never be described as doing so."
)
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
        # Arm C is STAGED from attempt 1 and hash-verified in setup, never
        # rebuilt: a rebuild would silently substitute a corpus for the one the
        # comparison is registered against. Missing here is a setup failure.
        c = REPO / f"artifacts/stage3/e5_arm_c_{seed}"
        if not (c / "examples.jsonl").exists():
            raise AssertionError(
                f"arm C for {seed} is absent; it is staged in setup, not built here")


def stage_verify_records(args):
    """Reload every accepted record FROM DISK and convert it for the packer.

    The builder already checks each record before accepting it, but it checks the
    object it just built. This stage re-reads the file the next stage will read,
    which is the only thing that proves the corpus on disk is trainable. Attempt
    1 passed every in-memory check and still wrote 4,196 unusable records.

    Blocking, and placed before pairing, so a bad corpus stops the run while the
    only thing spent is generation -- not four training arms on top of it.
    """
    sys.path.insert(0, str(REPO / "src"))
    from aadistill.data.e5_pack import REQUIRED_FIELDS, example_to_rendered

    report = {"required_fields": list(REQUIRED_FIELDS), "per_corpus": {}}
    failures = []
    for arm in ("c", "r"):
        for seed in SEEDS:
            d = REPO / f"artifacts/stage3/e5_arm_{arm}_{seed}"
            rows = [json.loads(line) for line
                    in (d / "examples.jsonl").open() if line.strip()]
            sysids = json.loads((d / "system_ids.json").read_text())
            missing, unrenderable, bad_system = 0, 0, 0
            for rec in rows:
                if [f for f in REQUIRED_FIELDS if f not in rec]:
                    missing += 1
                    continue
                try:
                    example_to_rendered(rec)
                except Exception:
                    unrenderable += 1
                    continue
                # The packer re-emits the block stored under this key, so the
                # record's own leading tokens must be that block.
                if rec["ids"][:rec["n_system_tokens"]] != sysids.get(
                        rec["system_key"], []):
                    bad_system += 1
            entry = {"examples": len(rows), "system_blocks": len(sysids),
                     "missing_fields": missing, "unrenderable": unrenderable,
                     "system_block_mismatch": bad_system,
                     "ce_tokens": sum(sum(r["mask"]) for r in rows
                                      if "mask" in r)}
            report["per_corpus"][f"{arm}_{seed}"] = entry
            ok = not (missing or unrenderable or bad_system) and rows
            print(f"  [{'PASS' if ok else 'FAIL'}] {arm}_{seed}: {len(rows)} records, "
                  f"missing {missing}, unrenderable {unrenderable}, "
                  f"system mismatch {bad_system}", flush=True)
            if not ok:
                failures.append(f"{arm}_{seed}")
    report["passed"] = not failures
    report["failed_corpora"] = failures
    (OUT / "e5_persisted_records.json").write_text(json.dumps(report, indent=1))
    mark("RECORDS_VERIFIED" if report["passed"] else "RECORDS_INVALID")
    if failures:
        raise AssertionError(f"persisted records unusable: {failures}")
    _retain_corpora()


def _retain_corpora() -> None:
    """Push the generated R corpora to the relay the moment they are known good.

    Twice now -- 2026-08-07 attempts 1 and 4 -- an R corpus that cost ~$1.20 of
    GPU time was generated, accepted, and then lost when the pod was torn down,
    because the launcher's side bundle ships manifests and not `examples.jsonl`.
    Waiting for teardown to preserve an artifact means losing it whenever the run
    stops early, which is exactly when it is most worth keeping. So this runs
    here, right after the records are verified and before anything can fail.
    """
    import tarfile
    tar = OUT / "e5_arm_r_corpora.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for seed in SEEDS:
            d = REPO / f"artifacts/stage3/e5_arm_r_{seed}"
            t.add(d, arcname=d.name)
    size_mb = tar.stat().st_size / 1e6
    try:
        from huggingface_hub import HfApi
        # The driver is detached with `setsid` and does not inherit setup's
        # exported HF_TOKEN, so attempt 5 raised KeyError here and fell back to
        # the side bundle. Read the file the setup staged instead.
        token = os.environ.get("HF_TOKEN") or Path("/workspace/hf/token").read_text().strip()
        HfApi(token=token).upload_file(
            path_or_fileobj=str(tar), path_in_repo="e5_start/e5_arm_r_corpora.tar.gz",
            repo_id="AlphaAvatar/aadistill-artifacts", repo_type="model")
        digest = hashlib.sha256(tar.read_bytes()).hexdigest()
        print(f"R corpora retained on the relay: {size_mb:.1f} MB, "
              f"sha256 {digest[:16]}", flush=True)
        mark(f"CORPORA_RETAINED:{digest[:16]}")
    except Exception as exc:                       # never fail the run over this
        print(f"WARNING: R corpora upload failed ({exc}); the local tarball at "
              f"{tar} is still in the side bundle", flush=True)
        mark("CORPORA_RETAIN_FAILED")


def _system_ids(seed: str, conditions: list) -> dict:
    """Union of both arms' system blocks, with disagreement made fatal.

    Each arm writes its own map. Taking only C's would let an R example whose
    system block C never saw be packed under a KeyError -- or worse, let the two
    arms disagree about the tokens behind one key and train on different system
    blocks while reporting one corpus.
    """
    merged: dict[str, list[int]] = {}
    clashes = []
    for arm in ("c", "r"):
        path = REPO / f"artifacts/stage3/e5_arm_{arm}_{seed}/system_ids.json"
        for key, ids in json.loads(path.read_text()).items():
            if key in merged and merged[key] != ids:
                clashes.append(key)
            merged[key] = ids
    conditions.append((f"{seed} arms agree on shared system blocks",
                       not clashes, f"{len(merged)} blocks, {len(clashes)} clash"))
    return merged


def stage_pair(args):
    """Intersection, token targeting, packing — then the joint feasibility gate."""
    sys.path.insert(0, str(REPO / "src"))
    from aadistill.data.paired_corpus import (
        as_bundles, common_token_target, comparability_report,
        composition_report, intersect, packing_report,
        select_nested_to_target, suffix_overlap,
    )
    report = {"target_ce_tokens": TARGET_CE_TOKENS, "tolerance": TOLERANCE,
              "blocks": BLOCKS, "steps": STEPS, "per_seed": {},
              "claim_boundary": CLAIM_BOUNDARY}
    conditions, pools, kept = [], {}, {}
    for seed in SEEDS:
        c_rows = [json.loads(l) for l in
                  (REPO / f"artifacts/stage3/e5_arm_c_{seed}/examples.jsonl").open()
                  if l.strip()]
        r_rows = [json.loads(l) for l in
                  (REPO / f"artifacts/stage3/e5_arm_r_{seed}/examples.jsonl").open()
                  if l.strip()]
        ck, rk, census = intersect(c_rows, r_rows)
        # C draws from its FULL pool, not the intersection. The intersection was
        # required when both arms had to share a composition; that requirement
        # was dropped in favour of independent per-arm selection, and only
        # `R_selected subset of C_selected` remains -- which C's full pool
        # satisfies by construction, since every R bundle comes from a C prompt.
        #
        # Keeping the restriction cost C 17.9% (sa) and 23.8% (sb) of its pool
        # and pushed C_sb to 0.937x the target, which is what made attempt 5
        # infeasible. It also conditioned C's corpus on R's success: the dropped
        # bundles are exactly the prompts where the teacher's recovery from a
        # student prefix failed a gate, mostly on natural termination. Training C
        # only where R succeeded is a confound, not a neutral restriction.
        pools[f"C_{seed}"] = as_bundles(c_rows)
        pools[f"R_{seed}"] = as_bundles(rk)
        kept[seed] = (c_rows, rk, census)
        census["c_pool_policy"] = "full"
        census["c_bundles_dropped_for_pairing"] = 0

    # T* is fixed ONCE, across every arm and seed, before any arm is selected.
    # Choosing it per seed would let two seeds train on different budgets and
    # call the result a seed comparison.
    tstar = common_token_target(pools, TARGET_CE_TOKENS)
    report["common_target"] = tstar
    print(f"  T* = {tstar['common_target']:,} "
          f"(bound by {tstar['binding_pool']} at {tstar['binding_pool_total']:,}; "
          f"original {tstar['original_target']:,})", flush=True)
    conditions.append(("T* within tolerance of the registered target",
                       tstar["reduction_fraction"] <= TOLERANCE,
                       f"reduced {tstar['reduction_fraction']:.4%}"))

    for seed in SEEDS:
        ck, rk, census = kept[seed]
        res = select_nested_to_target(ck, rk, tstar["common_target"])
        c_sel, r_sel = res["examples"]
        sel = res["report"]
        comp = comparability_report(c_sel, r_sel, supervised_tolerance=TOLERANCE)
        pack_c = packing_report(c_sel, BLOCKS, 8192)
        pack_r = packing_report(r_sel, BLOCKS, 8192)
        entry = {
            "census": census, "selection": sel, "comparability": comp,
            "packing_C": pack_c, "packing_R": pack_r,
            "overlap_C": suffix_overlap(c_sel), "overlap_R": suffix_overlap(r_sel),
            "composition": composition_report(c_sel, r_sel),
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
        # Composition is deliberately NOT identical -- see CLAIM_BOUNDARY. What
        # must hold is nesting, so the arms share every bundle R uses.
        conditions.append((f"{seed} R nested in C", sel["nested"],
                           f"{sel['shared_bundles']} shared, "
                           f"{sel['c_only_bundles']} C-only"))
        conditions.append((f"{seed} arm-to-arm token delta under 1%",
                           sel["arm_to_arm_relative_delta"] < 0.01,
                           f"{sel['arm_to_arm_relative_delta']:.4%}"))
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
        sysids = _system_ids(seed, conditions)
        for arm in ("C", "R"):
            rows = [json.loads(l) for l in
                    (REPO / f"artifacts/stage3/e5_final_{arm}_{seed}.jsonl").open()
                    if l.strip()]
            minima[(arm, seed)] = len(pack_e5(rows, sysids, block_len=8192))
    common = max(minima.values())
    # Three passes at two blocks per step needs 3n/2 to be a whole number, so the
    # common count must be EVEN. Attempt 4 landed on 759 and `verify_pack` failed
    # `three_passes_equal_registered_steps` for that reason alone. Rounding UP
    # adds one mostly-padded block: no example is duplicated, cut, or dropped,
    # and the extra block costs one step of mostly-padding compute.
    odd_bump = common % 2
    common += odd_bump
    report["per_arm_minimum_blocks"] = {f"{a}_{s_}": v for (a, s_), v in minima.items()}
    report["rounded_up_for_even_passes"] = bool(odd_bump)
    report["common_block_count"] = common
    report["optimizer_steps"] = common * 3 // 2
    print(f"  minima {report['per_arm_minimum_blocks']} -> common {common} blocks, "
          f"{report['optimizer_steps']} steps", flush=True)
    for seed in SEEDS:
        sysids = _system_ids(seed, conditions)
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


def _spent(args) -> float:
    """Pod spend to date: what the launcher paid for setup, plus driver elapsed.

    Elapsed-based accounting is what makes gate 2 honest without bookkeeping:
    whatever the R generation and the gate-2 benchmark actually cost is already
    inside the wall clock by the time the gate reads it.
    """
    return args.spent_usd + (time.time() - T0) / 3600 * RATE


def _budget(spent_usd: float, remaining_usd: float, blocks: int,
            sec_per_step: float, *, phases_min: dict) -> dict:
    """Project the rest of the run at a MEASURED absolute sec/step."""
    train_min = 4 * (blocks * 3 / 2) * sec_per_step / 60
    rest = sum(phases_min.values()) + train_min
    expected = rest / 60 * RATE
    backstop = rest * 1.12 / 60 * RATE
    return {"blocks": blocks, "sec_per_step": round(sec_per_step, 4),
            "spent_usd": round(spent_usd, 2),
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
    spent = _spent(args)
    # r_generation was 152 min: an estimate made before R had ever been
    # generated. Attempt 1 measured the identical operation -- same hardware,
    # engine, prompts, student, both seeds -- at 74.1 min (sa 34.6, sb 39.5).
    # 90 min is that measurement plus 21%, so the gate is still conservative
    # while no longer carrying a 78-minute phantom worth $1.29. Every other
    # phase estimate is unchanged; verify_records is a new CPU-only stage.
    rep = _budget(spent, args.authorized_usd - spent, args.assumed_blocks,
                  SEC_PER_STEP / max(0.01, speedup),
                  phases_min={"r_generation": 90, "verify_records": 2,
                              "pair_pack": 20, "final_benchmark": 5,
                              "evaluate": 44, "transfer_teardown": 35})
    rep["gate"] = "pre-generation"
    rep["r_generation_basis"] = ("MEASURED 74.1 min over both seeds in attempt 1 "
                                 "(2026-08-07), +21% margin; not the 152-min "
                                 "pre-measurement estimate")
    rep["rate_source"] = ("C full-width reference / measured truncate_padding "
                          f"speedup {speedup:.3f}x; replaced at gate 2 by an "
                          "absolute measurement on the final packs")
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


def stage_final_benchmark(args):
    """Absolute sec/step on the FINAL packs, before gate 2 commits to training.

    C's measured rate does not transfer to R. The two arms hold different
    material at the same block count: R's blocks carry student-generated
    prefixes, so their non-padding lengths -- and therefore how much the
    registered truncated path can actually skip -- are a different distribution
    from C's. Assuming C's sec/step for R would let an underestimate of R turn
    into an overrun that only shows up mid-training.

    Registered truncated path only. The full-width reference is not repeated:
    padding equivalence was already established at the earlier benchmark, and
    gate 2 needs an absolute rate, not a ratio.
    """
    out = OUT / "e5_throughput_final.json"
    if out.exists():
        print("final-pack benchmark already done; skipping", flush=True)
        return mark("BENCHMARKED_FINAL")
    seed = SEEDS[0]
    packs = [REPO / f"artifacts/stage3/e5_pack_{a}_{seed}" for a in ("c", "r")]
    run(["scripts/training/benchmark_e5_throughput.py", "--absolute-only",
         "--packs", *packs, "--labels", f"C_{seed}", f"R_{seed}",
         "--student", f"/workspace/ckpt/p2_ceheavy_{seed}",
         "--teacher", f"{TEACHER}@{TEACHER_REV}",
         "--steps", args.final_bench_steps, "--out", out])
    mark("BENCHMARKED_FINAL")


def stage_budget_gate_2(args):
    """After pairing: re-gate on the ACTUAL common block count and the SLOWER
    of the two measured arm rates."""
    bench = json.loads((OUT / "e5_throughput_final.json").read_text())
    feas = json.loads((OUT / "e5_joint_feasibility.json").read_text())
    blocks = feas["common_block_count"]
    spent = _spent(args)          # already includes the gate-2 benchmark itself
    rep = _budget(spent, args.authorized_usd - spent, blocks,
                  bench["sec_per_step_for_projection"],
                  phases_min={"evaluate": 44, "transfer_teardown": 35})
    rep["gate"] = "pre-training"
    rep["rate_source"] = {
        "measured_on": "final packed corpora, registered truncate_padding path",
        "per_arm_sec_per_step": {k: v["sec_per_step"]
                                 for k, v in bench["arms"].items()},
        "slower_arm_used": bench["slowest_arm"],
        "benchmark_cost_usd": bench["benchmark_cost_usd"],
        "benchmark_cost_accounting": ("charged through elapsed pod time, which "
                                      "is measured after the benchmark ran"),
    }
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
          "verify_records": stage_verify_records,
          "pair": stage_pair, "final_benchmark": stage_final_benchmark,
          "budget_gate_2": stage_budget_gate_2,
          "train": stage_train, "evaluate": stage_evaluate}
BLOCKING = ("validate", "benchmark", "budget_gate_1", "verify_records",
            "pair", "final_benchmark", "budget_gate_2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", *STAGES))
    ap.add_argument("--validate-limit", type=int, default=24)
    ap.add_argument("--bench-steps", type=int, default=12)
    ap.add_argument("--final-bench-steps", type=int, default=8)
    ap.add_argument("--authorized-usd", type=float, default=8.23)
    ap.add_argument("--spent-usd", type=float, default=0.0,
                    help="pod spend so far, supplied by the launcher")
    ap.add_argument("--assumed-blocks", type=int, default=1012,
                    help="gate 1 only. Under nested selection C BINDS, not R: "
                         "C is measured at 872/880 blocks on the real corpus "
                         "and R estimates at ~591/596, so 1012 is 1.15x the "
                         "measured binding arm. Replaced by the real common "
                         "count at gate 2.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        try:
            STAGES[name](args)
        # `except Exception`, deliberately broad. A narrower tuple let a
        # ValueError escape `main()` on 2026-08-07: no marker was written, so the
        # launcher -- which polls for a terminal marker -- had nothing to see and
        # would have kept a finished pod alive for its full 520-minute poll
        # window. On a paid pod an uncaught exception is a billing event, not
        # just a stack trace.
        except Exception as exc:
            mark(f"STAGE_FAILED:{name}:{type(exc).__name__}")
            print(f"STAGE FAILED: {name}: {exc}", flush=True)
            traceback.print_exc()
            if name in BLOCKING and args.stage == "all":
                # `return`, not `break`: the launcher reads the LAST status line,
                # so falling through to ALL_DONE would record a stopped gate as a
                # completed run.
                mark("ABORTED_AT_GATE")
                return
            continue
    mark("ALL_DONE")


if __name__ == "__main__":
    main()

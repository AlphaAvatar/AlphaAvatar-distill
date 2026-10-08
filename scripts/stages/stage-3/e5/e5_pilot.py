#!/usr/bin/env python
"""E5 full-path pilot: the last gate before paid generation.

    /opt/train/bin/python scripts/pod/e5_pilot.py --limit 24

Runs the complete production path end to end at small scale, through the same
code the full run uses:

    student rollout -> atomic two-cut bundle -> teacher recovery -> ten gates
    -> exact-prefix-echo -> serialization + loader round-trip
    -> deliberate R rejection removing its paired C bundle
    -> paired intersection -> token-target selection -> packing
    -> real-model optimizer step -> loss AND gradient attribution

**Identity is verified before any gradient number is interpreted.** A gradient
comparison against the wrong teacher, a teacher left in train mode, a
misaligned mask or a different KD temperature would all produce numbers that
look meaningful and are not, so each is asserted and recorded rather than
assumed.

The pilot proves **pipeline correctness and gives a feasibility estimate**. It
does not prove the complete corpus satisfies the joint token/block constraints —
that is checked only after the full R corpus is paired with C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("E5_REPO", "/workspace/aad"))
sys.path.insert(0, str(REPO / "src"))

TRAIN_PY = os.environ.get("E5_TRAIN_PY", "/opt/train/bin/python")
VLLM_PY = os.environ.get("E5_VLLM_PY", "/opt/vllm/bin/python")
INIT_SHA = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REV = "768f209d9ea81521153ed38c47d515654e938aea"
P2_SHA = {"sa": "4aface45a12cd02e", "sb": "9828b1780a5eb4e2"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, py=TRAIN_PY, **kw):
    cmd = [py] + [str(c) for c in cmd]
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO / "src")}, **kw)


def verify_identity(student_dir: Path, seed: str) -> dict:
    """Every identity a gradient comparison silently depends on."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    checks, failures = {}, []
    st = sha256(student_dir / "model.safetensors")
    checks["student_sha256"] = st
    checks["student_sha256_matches_p2"] = st.startswith(P2_SHA[seed])
    if not checks["student_sha256_matches_p2"]:
        failures.append(f"student checkpoint is not P2-0.86M-{seed}: {st[:16]}")

    init = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    checks["stage1_init_sha256"] = sha256(init / "model.safetensors")
    if checks["stage1_init_sha256"] != INIT_SHA:
        failures.append("Stage 1 fork point hash mismatch")

    tcfg = AutoConfig.from_pretrained(TEACHER, revision=TEACHER_REV)
    checks["teacher"] = f"{TEACHER}@{TEACHER_REV}"
    checks["teacher_hidden_size"] = tcfg.hidden_size
    checks["teacher_layers"] = tcfg.num_hidden_layers

    stok = AutoTokenizer.from_pretrained(str(student_dir))
    ttok = AutoTokenizer.from_pretrained(TEACHER, revision=TEACHER_REV)
    checks["tokenizer_vocab_match"] = len(stok) == len(ttok)
    checks["chat_template_match"] = (stok.chat_template or "") == (ttok.chat_template or "")
    if not checks["tokenizer_vocab_match"]:
        failures.append(f"tokenizer vocab differs: {len(stok)} vs {len(ttok)}")

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, revision=TEACHER_REV, dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    checks["teacher_training_mode"] = teacher.training
    checks["teacher_any_param_requires_grad"] = any(
        p.requires_grad for p in teacher.parameters())
    if teacher.training or checks["teacher_any_param_requires_grad"]:
        failures.append("teacher is not in frozen eval state")
    del teacher

    cfg = json.loads((REPO / "configs/stage3/p2/p2_ceheavy_sa.json").read_text())
    checks["kd_temperature"] = cfg["loss"]["kd_temperature"]
    checks["ce_weight"] = cfg["loss"]["ce_weight"]
    checks["kd_weight"] = cfg["loss"]["kd_weight"]
    checks["kd_scope"] = cfg["loss"]["kd_scope"]
    if cfg["loss"]["kd_scope"] != "all" or cfg["loss"]["kd_temperature"] != 1.0:
        failures.append("objective drifted from the registered P2 semantics")
    return {"checks": checks, "failures": failures}


def verify_alignment(examples_path: Path) -> dict:
    """Token/mask alignment on the emitted records, before they train anything."""
    failures = []
    rows = [json.loads(l) for l in examples_path.open() if l.strip()][:64]
    for r in rows:
        if "ids" not in r:
            continue
        if len(r["ids"]) != len(r["mask"]):
            failures.append(f"{r['id']}: ids/mask length mismatch")
        if sum(r["mask"]) != r["n_continuation_tokens"]:
            failures.append(f"{r['id']}: mask sum != continuation length")
        first = r["mask"].index(True) if any(r["mask"]) else -1
        if first != r["n_prefix_tokens"]:
            failures.append(f"{r['id']}: supervision starts at {first}, "
                            f"expected {r['n_prefix_tokens']}")
        if any(r["mask"][:r["n_prefix_tokens"]]):
            failures.append(f"{r['id']}: prefix carries supervision")
    return {"rows_checked": len(rows), "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", default="sa", choices=("sa", "sb"))
    ap.add_argument("--student", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/audit/e5_pilot.json")
    # Validates every step EXCEPT generation, on CPU, for free. It stands a
    # synthesised R corpus in for the real one so the orchestration -- pairing,
    # the deliberate rejection, token targeting, packing, alignment, reporting --
    # is exercised before any of it costs pod time. It proves nothing about the
    # engines and is recorded as such.
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    student = Path(args.student or f"/workspace/ckpt/p2_ceheavy_{args.seed}")
    work = REPO / f"artifacts/stage3/e5_pilot_{args.seed}"
    work.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "limit": args.limit, "seed": args.seed,
              "status": "PIPELINE VALIDITY AND FEASIBILITY ESTIMATE ONLY"}

    if args.dry_run:
        report["status"] = ("DRY RUN — orchestration only; no generation, no "
                            "engines, no identity verification")
    print("=== identity verification ===", flush=True)
    ident = ({"checks": {"skipped": "dry run"}, "failures": []} if args.dry_run
             else verify_identity(student, args.seed))
    report["identity"] = ident
    if ident["failures"]:
        report["passed"] = False
        args.out.write_text(json.dumps(report, indent=1))
        for f in ident["failures"]:
            print(f"FAIL: {f}")
        raise SystemExit("identity verification failed; gradients not interpretable")
    print("identity OK", flush=True)

    print("=== arm C (no generation) ===", flush=True)
    c_dir = work / "arm_c"
    run(["scripts/data/build_e5_arm_c.py", "--source-seed", args.seed,
         "--out", c_dir])

    print("=== arm R (student rollout -> teacher recovery -> gates) ===", flush=True)
    r_dir = work / "arm_r"
    c_rows = [json.loads(l) for l in (c_dir / "examples.jsonl").open() if l.strip()]
    victim = c_rows[0]["source_session_id"]
    if args.dry_run:
        # Stand-in R: longer contexts and different continuation lengths, with
        # the victim bundle dropped exactly as a gate failure would drop it.
        r_dir.mkdir(parents=True, exist_ok=True)
        sample = [e for e in c_rows[:max(4, args.limit)]
                  if e["source_session_id"] != victim]
        synth = []
        for e in sample:
            r = dict(e, arm="R", prefix_source="student_generated",
                     n_prefix_tokens=int(e["n_prefix_tokens"] * 2.5),
                     n_continuation_tokens=int(e["n_continuation_tokens"] * 1.3) or 1)
            r["n_total_tokens"] = r["n_prefix_tokens"] + r["n_continuation_tokens"]
            r["id"] = e["id"].replace("#c", "#r")
            synth.append(r)
        (r_dir / "examples.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in synth))
    else:
        run(["scripts/data/build_e5_arm_r.py", "--student", student,
             "--source-seed", args.seed, "--limit", args.limit, "--out", r_dir,
             "--reject-bundle", victim], py=VLLM_PY)
    report["deliberately_rejected_session"] = victim

    print("=== paired intersection and token-target selection ===", flush=True)
    from aadistill.data.paired_corpus import (
        comparability_report, intersect, packing_report,
        select_paired_to_token_target, suffix_overlap,
    )
    r_rows = [json.loads(l) for l in (r_dir / "examples.jsonl").open() if l.strip()]
    r_ids = {e["source_session_id"] for e in r_rows}
    c_subset = [e for e in c_rows if e["source_session_id"] in r_ids
                or e["source_session_id"] == victim]
    ck, rk, census = intersect(c_subset, r_rows)
    report["pairing_census"] = census
    report["rejected_bundle_removed_from_C"] = (
        victim not in {e["source_session_id"] for e in ck})
    if not report["rejected_bundle_removed_from_C"]:
        report["passed"] = False
        args.out.write_text(json.dumps(report, indent=1))
        raise SystemExit(f"paired C bundle for {victim} survived an R rejection")
    print(f"paired bundles {census['paired_bundles']}; "
          f"deliberate rejection removed from C: OK", flush=True)

    if ck:
        target = int(735603 * len(ck) / max(1, 2294))     # pilot-scaled target
        c_sel, r_sel, sel = select_paired_to_token_target(ck, rk, target)
        report["token_target_selection"] = sel
        report["comparability"] = comparability_report(
            c_sel, r_sel, supervised_tolerance=0.05)
        report["suffix_overlap_C"] = suffix_overlap(c_sel)
        report["suffix_overlap_R"] = suffix_overlap(r_sel)
        blocks = max(1, round(492 * len(ck) / 2294))
        report["packing_estimate_C"] = packing_report(c_sel, blocks, 8192)
        report["packing_estimate_R"] = packing_report(r_sel, blocks, 8192)

    if not args.dry_run and ck:
        print("=== loss and gradient attribution (real teacher) ===", flush=True)
        gr = work / "grad"
        gr.mkdir(parents=True, exist_ok=True)
        for label, rows in (("C", c_sel), ("R", r_sel)):
            (gr / f"{label}.jsonl").write_text(
                "".join(json.dumps({"ids": e["ids"], "mask": e["mask"]}) + "\n"
                        for e in rows[:2] if "ids" in e))
        if (gr / "R.jsonl").stat().st_size > 0:
            run(["scripts/training/diagnose_e5_gradients.py",
                 "--student", student, "--teacher", f"{TEACHER}@{TEACHER_REV}",
                 "--examples", gr / "C.jsonl", gr / "R.jsonl",
                 "--labels", "C", "R", "--max-batch", 2,
                 "--out", REPO / "artifacts/audit/e5_gradients.json"])
            report["gradient_attribution"] = json.loads(
                (REPO / "artifacts/audit/e5_gradients.json").read_text())["arms"]
        else:
            report["gradient_attribution"] = {
                "skipped": "emitted records carry no token ids"}

        print("=== real-model optimizer step ===", flush=True)
        import torch
        from transformers import AutoModelForCausalLM
        from aadistill.training.train import select_trainable
        m = AutoModelForCausalLM.from_pretrained(student, dtype=torch.float32)
        rep = select_trainable(m, json.loads(
            (REPO / "configs/stage3/p2/p2_ceheavy_sa.json").read_text()
        )["trainable_patterns"])
        opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=0.0)
        ids = torch.arange(1, 65).unsqueeze(0)
        out = m(ids)
        loss = out.logits.float().mean()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            [p_ for p_ in m.parameters() if p_.requires_grad], 1.0))
        opt.step()
        report["optimizer_step"] = {
            "trainable_params": rep["trainable_params"],
            "finite_loss": bool(torch.isfinite(loss)),
            "grad_norm": round(gn, 6),
            "finite_grad_norm": bool(torch.isfinite(torch.tensor(gn))),
            "lr": 0.0, "note": "lr=0 so no weight is meaningfully changed",
        }
        del m, opt

    print("=== alignment ===", flush=True)
    report["alignment"] = ({"rows_checked": 0, "failures": [],
                            "skipped": "dry run emits no token ids"}
                           if args.dry_run
                           else verify_alignment(r_dir / "examples.jsonl"))

    # Two verdicts, deliberately separate. `passed` is about PIPELINE VALIDITY:
    # did every stage run, did the gates fire, did a rejected R bundle remove its
    # paired C bundle. `feasibility_estimate` is about whether the arms currently
    # look matchable. Collapsing them would let a run print PILOT PASSED beside a
    # blown tolerance, which is exactly what the first dry run did.
    report["passed"] = not (ident["failures"] or report["alignment"]["failures"])
    comp = report.get("comparability") or {}
    report["feasibility_estimate"] = {
        "c_r_within_tolerance": comp.get("within_tolerance"),
        "c_r_relative_delta": comp.get("supervised_token_relative_delta"),
        "verdict": ("indicative only — the real C/R continuation-length ratio is "
                    "unknown until R is generated, and a dry run's ratio is an "
                    "assumption, not a measurement"),
        "binding_check": ("final joint feasibility is decided after the COMPLETE "
                          "R corpus is generated and paired with C; a failure "
                          "there stops the experiment before training"),
    }
    report["feasibility_note"] = (
        "estimate only; joint token/block feasibility is decided after the "
        "complete R corpus is generated and paired with C")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    fe = report["feasibility_estimate"]
    print(f"pipeline validity : {'PASS' if report['passed'] else 'FAIL'}")
    print(f"feasibility (est) : c_r_within_tolerance={fe['c_r_within_tolerance']} "
          f"delta={fe['c_r_relative_delta']} — indicative only, not the binding gate")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""On-pod driver of H1's P1g GPU validation: real packs, real trainer, resume, persistence, evaluation path.

    cd /workspace/aad && PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \
        scripts/stages/stage-2/reasoning_data/p1g_driver.py --run-id <run> \
        --durable-root <mounted volume>/h1 --status <file> [--pack-root <dir with A/ and B/>]

Never a formal result: everything it writes is under ``<durable root>/validation/p1g``
and its report says ``formal: false``. It runs the FROZEN recipe with H1's declared
overrides (`unit_driver.derive_recipe`) plus one validation-only knob, the log
interval, which changes no numerics.

1. INITIALIZATION: the D4 Top-1 restore, its identity re-proved from the bytes.
2. THROUGHPUT AND MEMORY on each arm's real pack at its M1 blocks per step:
   step seconds, peak GPU memory, executed and supervised positions, loss
   finite and moving.
3. RESUME (arm B): two uninterrupted runs X1 and X2 and one run Y killed with
   SIGKILL right after its mid-run save -- written to the network volume -- and
   continued with ``train_stage3.py --resume latest``. The data order (block
   positions, targets) and the learning rate must be identical; the loss and
   the final weights must differ from X1 by no more than X2 does (the GPU's own
   run-to-run floor), and are reported exactly.
4. PERSISTENCE: Y's final model through `persist_unit.persist_checkpoint` onto
   the volume, re-identified from the landed bytes.
5. EVALUATION PATH: the evaluation package built from Y's model, `run_battery`
   on a stratified subset of fluid_reasoning_720 and `uncapped_eval` on a subset
   of one capability unit; generations saved and persisted; seconds per item.

The status file gets ``MARKER:ALL_DONE`` when every check passed and
``MARKER:RUN_FAILED`` otherwise, with ``p1g_report.json`` beside it either way.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import persist_unit as PU  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402

#: validation-only: logged every step so the resumed steps can be compared one by one
EXTRA_OVERRIDES = ["intervals.log_every"]
THROUGHPUT_STEPS = 12
RESUME_STEPS, RESUME_SAVE = 24, 12
FLUID_PER_AXIS = 4
CAPABILITY_UNIT, CAPABILITY_ITEMS = "math_verified", 8


def recipe(plan: dict, arm: str, *, steps: int, save_every: int, out_dir: Path, pack_root: Path,
           init: Path, seed: int, name: str) -> dict:
    s = plan["schedule"]
    sched = {"total_steps": steps, "warmup_steps": max(1, round(0.05 * steps)),
             "blocks_per_step": int(s["blocks_per_step"][arm]), "save_every": save_every,
             "eval_every": steps * 10, "rung": int(plan["config"]["schedule"]["rung"])}
    d = UD.derive_recipe(plan["recipe"], run_name=name, purpose="H1 P1g GPU VALIDATION (not formal)",
                         out_dir=str(out_dir), data_dir=str(pack_root / arm), student_path=str(init),
                         seed=seed, sched=sched, overrides=plan["config"]["recipe"]["overrides"])
    d["intervals"]["log_every"] = 1
    a, b = UD._flat(plan["recipe"]), UD._flat(d)
    outside = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)
                     and k not in plan["config"]["recipe"]["overrides"] + EXTRA_OVERRIDES)
    assert not outside, outside
    return d


def run_trainer(cfg: dict, cfg_path: Path, log: Path, env: dict, *, resume: bool = False,
                kill_after_tag: str | None = None) -> dict:
    """The trainer as a child; with `kill_after_tag`, SIGKILLed the moment that
    checkpoint's tag is in ``latest.txt``."""
    from stages.phase_c1.autoinit_c1_driver import TRAINER
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    cmd = [UD.TRAIN_PY, str(TRAINER), "--config", str(cfg_path)] + (["--resume", "latest"] if resume else [])
    t0 = time.time()
    with log.open("a") as lf:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        UD._OBSERVERS.append(p)
        killed = False
        latest = Path(cfg["out_dir"]) / "checkpoints/latest.txt"
        while p.poll() is None:
            if kill_after_tag and latest.is_file() and latest.read_text().strip() == kill_after_tag:
                os.killpg(p.pid, signal.SIGKILL)
                killed = True
                p.wait()
                break
            time.sleep(0.5)
        UD._OBSERVERS.remove(p)
    return {"rc": p.returncode, "killed": killed, "seconds": round(time.time() - t0, 1)}


def steps_of(out_dir: Path) -> dict[int, dict]:
    rows = {}
    for line in (out_dir / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "train_step":
            rows[int(r["step"])] = r
    return rows


def weights_delta(a: Path, b: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    ta, tb = load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
    if set(ta) != set(tb):
        return {"same_keys": False}
    worst, name, total = 0.0, None, 0.0
    for k in ta:
        d = float((ta[k].float() - tb[k].float()).abs().max())
        total = max(total, float(ta[k].float().abs().max()))
        if d > worst:
            worst, name = d, k
    return {"same_keys": True, "max_abs_diff": worst, "worst_tensor": name, "max_abs_weight": total,
            "bitwise_identical": worst == 0.0, "tensors": len(ta), "_dtype": str(next(iter(ta.values())).dtype)}


def compare(ref: dict[int, dict], other: dict[int, dict], steps) -> dict:
    exact = ("lr", "executed_positions", "executed_nonpad_tokens", "supervised_tokens", "kd_positions")
    out = {"steps": list(steps), "exact_mismatch": [], "max_rel_loss": 0.0, "max_rel_ce": 0.0, "max_rel_kd": 0.0}
    for s in steps:
        a, b = ref.get(s), other.get(s)
        if a is None or b is None:
            out["exact_mismatch"].append({"step": s, "missing": [n for n, r in (("ref", a), ("other", b)) if r is None]})
            continue
        bad = [k for k in exact if a.get(k) != b.get(k)]
        if bad:
            out["exact_mismatch"].append({"step": s, "fields": {k: (a.get(k), b.get(k)) for k in bad}})
        for k in ("loss", "ce", "kd"):
            if a.get(k) is not None and b.get(k) is not None:
                rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-12)
                out[f"max_rel_{k}"] = max(out[f"max_rel_{k}"], rel)
    return out


def throughput(rows: dict[int, dict], first: int = 2) -> dict:
    """Steady-state step time: the first steps (allocation, compilation) left out."""
    use = [r for s, r in sorted(rows.items()) if s > first]
    secs = sorted(r["seconds"] for r in use)
    return {"steps_measured": len(use), "median_step_seconds": secs[len(secs) // 2] if secs else None,
            "max_step_seconds": secs[-1] if secs else None,
            "peak_gpu_mem_gb": max((r.get("gpu_mem_gb") or 0) for r in rows.values()) if rows else None,
            "executed_positions_per_step": use[-1]["executed_positions"] if use else None,
            "supervised_tokens_per_step_last": use[-1]["supervised_tokens"] if use else None,
            "loss_first": rows[min(rows)]["loss"] if rows else None, "loss_last": rows[max(rows)]["loss"] if rows else None,
            "all_finite": all(all(math.isfinite(r[k]) for k in ("loss", "grad_norm") if r.get(k) is not None)
                              for r in rows.values())}


def eval_smoke(model_dir: Path, out: Path, env: dict) -> dict:
    from shared import eval_sets as ES
    from stages.calibration_volume.dose_driver import SET_ID, VLLM_PY
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE
    from stages.phase_c1.packaging import build_evaluation_package
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / "package"
    build_evaluation_package(model_dir, tokenizer_source=TOKENIZER_SOURCE, dest=pkg,
                             expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
    fluid_all = [json.loads(l) for l in ES.unit_path(SET_ID, "fluid_reasoning_720", "dev_v1").read_text().splitlines()]
    seen, fluid = {}, []
    for it in fluid_all:
        ax = it.get("axis")
        if seen.get(ax, 0) < FLUID_PER_AXIS:
            seen[ax] = seen.get(ax, 0) + 1
            fluid.append(it)
    (out / "fluid_subset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fluid))
    t0 = time.time()
    with (out / "eval_fluid.log").open("w") as lf:
        rc_f = subprocess.run([VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py", "--model", str(pkg),
                               "--label", "p1g_smoke", "--tasks", str(out / "fluid_subset.jsonl"), "--mode", "eval",
                               "--out", str(out / "fluid_subset.generations.jsonl"), "--trained-context", "8192",
                               "--action-protocol", "v2"], cwd=REPO, env=env, stdout=lf,
                              stderr=subprocess.STDOUT).returncode
    fluid_s = time.time() - t0
    cap = ES.unit_path(SET_ID, "stage3_capability_846", CAPABILITY_UNIT)
    rows = cap.read_text().splitlines()[:CAPABILITY_ITEMS]
    staged = out / "uncapped_in"
    staged.mkdir(exist_ok=True)
    (staged / f"stage3_capability_846__{CAPABILITY_UNIT}.jsonl").write_text("\n".join(rows) + "\n")
    t1 = time.time()
    with (out / "eval_capability.log").open("w") as lf:
        rc_c = subprocess.run([VLLM_PY, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg),
                               "--label", "p1g_smoke", "--prompts", str(staged / f"stage3_capability_846__{CAPABILITY_UNIT}.jsonl"),
                               "--out-dir", str(out / "uncapped_out"), "--trained-context", "8192"],
                              cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    cap_s = time.time() - t1
    gens_f = out / "fluid_subset.generations.jsonl"
    gens_c = out / "uncapped_out" / f"stage3_capability_846__{CAPABILITY_UNIT}.generations.jsonl"
    n_f = len(gens_f.read_text().splitlines()) if gens_f.is_file() else 0
    n_c = len(gens_c.read_text().splitlines()) if gens_c.is_file() else 0
    summary = out / "fluid_subset.generations.summary.json"
    return {"fluid": {"rc": rc_f, "items": len(fluid), "generations": n_f, "seconds": round(fluid_s, 1),
                      "summary": json.loads(summary.read_text()) if summary.is_file() else None},
            "capability": {"unit": CAPABILITY_UNIT, "rc": rc_c, "items": len(rows), "generations": n_c,
                           "seconds": round(cap_s, 1)},
            "ok": rc_f == 0 and rc_c == 0 and n_f == len(fluid) and n_c == len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--pack-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/packs"))
    ap.add_argument("--init-dir", default=str(REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/"
                                                     "1301deb71685aa1ca5c7b6cf2c491ac4"))
    ap.add_argument("--work", default=None)
    args = ap.parse_args(argv)
    work = Path(args.work or REPO / "artifacts/stages/stage-2/reasoning_data/units" / args.run_id / "p1g")
    vol = Path(args.durable_root) / "validation" / "p1g"
    work.mkdir(parents=True, exist_ok=True)
    vol.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    env = {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}
    rep = {"schema": "aadistill.stage2.reasoning_data.p1g_report/v1", "run_id": args.run_id, "formal": False,
           "started_utc": UD._now(), "checks": {}, "not": "a GPU validation: never a formal H1 result"}
    stage = "plan"
    try:
        plan = UD.load_plan()
        refusals = UD.launch_refusals(plan, [{"arm": "A"}, {"arm": "B"}], validation=True, pack_root=Path(args.pack_root))
        if refusals:
            raise RuntimeError(f"pack refusals: {refusals}")
        stage = "restore_initialization"
        init = UD.restore_init(Path(args.init_dir), plan["init"])
        rep["checks"]["initialization"] = {"ok": True, "artifact_digest": plan["init"]["artifact_digest"]}
        seed = int(plan["units"][0]["seed"])
        stage = "throughput_A"
        a_dir = work / "throughput_A"
        cfg = recipe(plan, "A", steps=THROUGHPUT_STEPS, save_every=0, out_dir=a_dir, pack_root=Path(args.pack_root),
                     init=init, seed=seed, name=f"p1g_{args.run_id}_A")
        r = run_trainer(cfg, work / "throughput_A.json", work / "throughput_A.log", env)
        tp_a = {**throughput(steps_of(a_dir)), "trainer": r, "blocks_per_step": cfg["batch"]["blocks_per_step"]}
        rep["checks"]["throughput_A"] = {**tp_a, "ok": r["rc"] == 0 and tp_a["all_finite"]}
        stage = "resume_B"
        runs = {}
        for name in ("X1", "X2", "Y"):
            out_dir = (vol if name == "Y" else work) / f"resume_B_{name}"
            c = recipe(plan, "B", steps=RESUME_STEPS, save_every=RESUME_SAVE, out_dir=out_dir,
                       pack_root=Path(args.pack_root), init=init, seed=seed, name=f"p1g_{args.run_id}_B_resume")
            log = work / f"resume_B_{name}.log"
            if name == "Y":
                first = run_trainer(c, work / f"resume_B_{name}.json", log, env, kill_after_tag=f"step_{RESUME_SAVE:06d}")
                second = run_trainer(c, work / f"resume_B_{name}.json", log, env, resume=True)
                runs[name] = {"out": out_dir, "trainer": [first, second]}
            else:
                runs[name] = {"out": out_dir, "trainer": [run_trainer(c, work / f"resume_B_{name}.json", log, env)]}
        rows = {n: steps_of(v["out"]) for n, v in runs.items()}
        after = range(RESUME_SAVE + 1, RESUME_STEPS + 1)
        from stages.phase_c1.autoinit_c1_driver import trained_model_dir
        final = {n: trained_model_dir(v["out"]) for n, v in runs.items()}
        resumed, floor = compare(rows["X1"], rows["Y"], after), compare(rows["X1"], rows["X2"], after)
        w_res, w_floor = weights_delta(final["X1"], final["Y"]), weights_delta(final["X1"], final["X2"])
        y_first = runs["Y"]["trainer"][0]
        tol = lambda got, base: got <= max(base * 2.0, 1e-6)      # noqa: E731 -- within the GPU's own floor
        ok = (y_first["killed"] and runs["Y"]["trainer"][1]["rc"] == 0
              and all(t["rc"] == 0 for n in ("X1", "X2") for t in runs[n]["trainer"])
              and not resumed["exact_mismatch"] and tol(resumed["max_rel_loss"], floor["max_rel_loss"])
              and w_res.get("same_keys") and tol(w_res["max_abs_diff"], w_floor["max_abs_diff"]))
        rep["checks"]["resume_B"] = {
            "ok": bool(ok), "kill": {"after": f"step_{RESUME_SAVE:06d}", "killed": y_first["killed"],
                                     "first_seconds": y_first["seconds"]},
            "resumed_vs_uninterrupted": resumed, "run_to_run_floor": floor,
            "weights_resumed_vs_uninterrupted": w_res, "weights_run_to_run_floor": w_floor,
            "resume_state_on_volume": str(runs["Y"]["out"] / "checkpoints"),
            "rule": ("data order and learning rate identical step by step; loss and final weights within twice "
                     "the run-to-run difference of two uninterrupted runs (or 1e-6)")}
        rep["checks"]["throughput_B"] = {**throughput(rows["X1"]), "blocks_per_step": int(plan["schedule"]["blocks_per_step"]["B"]),
                                         "ok": throughput(rows["X1"])["all_finite"]}
        stage = "persist_checkpoint"
        root = Path(args.durable_root) / "validation"
        inputs = {"p1g": args.run_id, "arm": "B", "steps": RESUME_STEPS, "resumed": True}
        t0 = time.time()
        rec = PU.persist_checkpoint("p1g-B", final["Y"], root, adapter=UD._qwen3_adapter(),
                                    arch_signature=plan["init"]["arch_signature"],
                                    num_parameters=plan["init"]["num_parameters"], inputs=inputs, capacity=None)
        landed = root / "p1g-B" / "checkpoint"
        reread = UD._verified(landed, PU.completed(root, "p1g-B", "checkpoint"), UD._qwen3_adapter())
        rep["checks"]["persistence"] = {"ok": bool(reread), "seconds": round(time.time() - t0, 1),
                                        "status": rec.get("status") if isinstance(rec, dict) else None,
                                        "landed": str(landed), "reidentified_from_landed_bytes": bool(reread)}
        stage = "evaluation_path"
        ev = eval_smoke(final["Y"], work / "eval_smoke", env)
        files = {p.relative_to(work / "eval_smoke").as_posix(): p for p in sorted((work / "eval_smoke").rglob("*"))
                 if p.is_file() and "package" not in p.relative_to(work / "eval_smoke").parts}
        PU.persist_files("p1g-B", files, root, label="evaluation_smoke", capacity=None)
        rep["checks"]["evaluation_path"] = ev
        stage = "release"
        #: validation bytes no later step reads (P8.4): the resume state and the
        #: landed copy, once re-identified; descriptors, logs and this report stay
        import shutil
        released = []
        for d in (runs["Y"]["out"] / "checkpoints", landed):
            if d.exists():
                released.append({"path": str(d), "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())})
                shutil.rmtree(d)
        rep["released_validation_bytes"] = released
        stage = "done"
    except BaseException as exc:  # noqa: BLE001
        rep["error"] = {"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:1500],
                        "traceback": traceback.format_exc()[-3000:]}
    finally:
        UD.close_observers()
        rep["finished_utc"] = UD._now()
        rep["passed"] = stage == "done" and all(c.get("ok") for c in rep["checks"].values())
        for d in (work, vol):
            (d / "p1g_report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"MARKER:{'ALL_DONE' if rep['passed'] else 'RUN_FAILED'}\n")
    return 0 if rep["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

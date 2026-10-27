#!/usr/bin/env python3
"""On-pod driver of H1-PA's PA2v GPU validation, and the checks its gate re-runs off the pod. Never formal.

    cd /workspace/aad && PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \
        scripts/stages/stage-2/reasoning_data/h1pa_validation.py --run-id <run> \
        --durable-root <volume>/h1pa --volume-mount <volume> --volume-bytes <the provider's size> \
        --pack-root <dir> --checkpoint-root <dir> --status <file>

What it runs is `h1pa.json :: gpu_validation`, on the formal session's own
inputs and GPU type:

1. CAPACITY: a probe file written to the network volume, fsynced, read back
   and hashed; the provider's size beside the bytes already on the mount.
2. INITIALIZATION: D4 Top-1 restored and re-identified from its bytes.
3. TRAINING on L40S under the seed's E1 config (`h1pa_driver.derive_recipe`,
   plus the validation-only schedule and a per-step log): D4-Augmented on the
   merged set -- two uninterrupted runs X1, X2 and a run Y SIGKILLed right
   after its mid-run save point on the volume and continued with ``--resume
   latest`` (`p1g_driver`'s mechanism) -- and D4-Original on E1's pack.
4. PERSISTENCE: Y's final model through `persist_unit.persist_checkpoint` onto
   the volume, re-identified from the landed bytes; the save point's real size.
5. EVALUATION: PCA-Original seed 20260726's checkpoint, re-identified, then
   the first fluid items through `run_battery` and the first gsm8k prompts
   through `uncapped_eval`, both under nucleus_v1.

Everything lands under ``<durable root>/validation/pa2v/<run_id>/evidence``.
`checks(evidence, plan)` derives every verdict from those files alone: the pod
runs it for its marker, and `h1pa_governance.pa2v_gate` runs it again on the
secured copy before any formal authorization can be issued. The validation's
bytes no later step reads (Y's save point, the landed copy) are released.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import h1pa_driver as D  # noqa: E402
from stages.reasoning_data import p1g_driver as P  # noqa: E402
from stages.reasoning_data import persist_unit as PU  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402

REPORT = "pa2v_report.json"
STOP_REASONS = {"eos", "degeneration", "context_limit"}
FLUID_KEY, GSM8K_KEY = "fluid_reasoning_720__dev_v1", "stage3_capability_846__gsm8k"
#: run name -> (arm, steps): the trainings PA2v measures
RUNS = {"X1": ("D4-Augmented", 24), "X2": ("D4-Augmented", 24), "Y": ("D4-Augmented", 24),
        "O": ("D4-Original", 12)}
#: what PA2v leaves on the volume that is never evidence (the archive excludes both) and that
#: nothing reads: Y's trainer directory (save points, final model) and its persisted copy
VALIDATION_BYTES = ("train_Y/checkpoints", "durable/pa2v-Y/checkpoint")


def release_validation_bytes(pa2v_root: Path) -> list[dict]:
    """Remove `VALIDATION_BYTES` under every PA2v run in `pa2v_root`
    (``<durable>/validation/pa2v``): this run's at its end, on every path, and
    an earlier attempt's at the start -- h1pav_20261024_192448 failed before
    its release and left 8.29 GB the formal sessions' capacity plan never
    counted. Reports and evidence stay."""
    out = []
    for run in sorted(p for p in Path(pa2v_root).glob("*") if p.is_dir()):
        for rel in VALIDATION_BYTES:
            d = run / rel
            if d.exists():
                out.append({"path": str(d), "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())})
                shutil.rmtree(d)
    return out


def validation_recipe(plan: dict, arm: str, *, steps: int, save_every: int, out_dir: Path, data_dir: Path,
                      init: Path, name: str) -> dict:
    """The seed's E1 config as the formal driver derives it, with the
    validation-only schedule and a log line per step."""
    gv = plan["study"]["gpu_validation"]["training"]
    u = {"unit": f"{arm}-s{gv['seed']}", "arm": arm, "seed": int(gv["seed"]), "execution": "train"}
    rung = (int(plan["study"]["datasets"][plan["study"]["arms"][arm]["dataset"]]["rung"]["target_supervised_tokens"])
            if arm == "D4-Augmented" else None)
    sched = {"rung": rung, "total_steps": steps, **UD.lr_rule(steps), "save_every": save_every}
    d = D.derive_recipe(plan, u, sched, run_name=name, purpose="H1-PA PA2v GPU VALIDATION (not formal)",
                        out_dir=str(out_dir), data_dir=str(data_dir), student_path=str(init))
    d["intervals"]["log_every"] = 1
    return d


def recipe_outside(plan: dict, cfg: dict) -> list[str]:
    """Keys of `cfg` that differ from its seed's E1 config outside WHERE and
    the validation-only overrides."""
    e1 = plan["recipes"][int(cfg["seed"])]
    allowed = set(D.WHERE) | set(plan["study"]["gpu_validation"]["training"]["validation_only_overrides"])
    a, b = UD._flat(e1), UD._flat(cfg)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in allowed
                  and not (k in D.PINS and b.get(k) == D.PINS[k]))


def formal_recipes_derive(plan: dict) -> dict:
    """Every formal training unit's recipe, derived from the study's recorded
    schedule; a key outside its declared overrides refuses."""
    out = {}
    for u in plan["units"]:
        if u["execution"] != "train":
            continue
        s = plan["study"]["arms"][u["arm"]]["schedule"]
        sched = None if u["arm"] == "D4-Original" else {
            "rung": int(plan["study"]["datasets"]["e1_2p96M_plus_fluid"]["rung"]["target_supervised_tokens"]),
            "total_steps": s["optimizer_steps"], "warmup_steps": s["warmup_steps"], "eval_every": s["eval_every"],
            "save_every": s["save_every"]}
        try:
            r = D.derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-", student_path="-")
            out[u["unit"]] = {"ok": True, "total_steps": r["schedule"]["total_steps"], "batch": r["batch"]}
        except D.DriverRefusal as exc:
            out[u["unit"]] = {"ok": False, "error": str(exc)}
    return out


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).is_file() else []


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def decoding_checks(gens: Path, protocol: dict) -> dict:
    """The generations under the protocol: its identity and sampling in each
    summary, the protocol's seed for each sample, and each sample's token ids
    and stop reason."""
    from shared.evaluation import decoding as DEC
    want = {k: protocol["sampling"][k] for k in protocol["sampling"]}
    out = {"summaries": {}, "samples": 0, "seed_mismatch": [], "bad_stop_reason": [], "missing_token_ids": []}
    for key in (FLUID_KEY, GSM8K_KEY):
        s = _read(gens / f"{key}.summary.json")
        ident = s.get("identity") or {}
        dp = s.get("decoding_protocol") or ident.get("decoding_protocol") or {}
        samp = ident.get("sampling") if key == FLUID_KEY else s.get("sampling")
        out["summaries"][key] = {"present": bool(s), "decoding_sha256_ok": dp.get("sha256") == protocol["sha256"],
                                 "sampling_ok": bool(samp) and all(samp.get(k) == v for k, v in want.items())}
    for rec in _jsonl(gens / f"{FLUID_KEY}.jsonl"):
        for t in rec.get("turns") or []:
            out["samples"] += 1
            sid = f"{rec['task_id']}#t{t['turn']}"
            if t.get("seed") != DEC.request_seed(protocol, FLUID_KEY, rec["task_id"], t["turn"]):
                out["seed_mismatch"].append(sid)
            if t.get("stop_reason") not in STOP_REASONS:
                out["bad_stop_reason"].append(sid)
            if not isinstance(t.get("output_token_ids"), list):
                out["missing_token_ids"].append(sid)
    for rec in _jsonl(gens / f"{GSM8K_KEY}.jsonl"):
        out["samples"] += 1
        if rec.get("sampling_seed") != DEC.request_seed(protocol, GSM8K_KEY, rec["id"]):
            out["seed_mismatch"].append(rec["id"])
        if rec.get("stop_reason") not in STOP_REASONS:
            out["bad_stop_reason"].append(rec["id"])
        if not isinstance(rec.get("output_token_ids"), list):
            out["missing_token_ids"].append(rec["id"])
    out["ok"] = (all(v["present"] and v["decoding_sha256_ok"] and v["sampling_ok"] for v in out["summaries"].values())
                 and out["samples"] > 0 and not out["seed_mismatch"] and not out["bad_stop_reason"]
                 and not out["missing_token_ids"])
    return out


def _steps(path: Path) -> dict[int, dict]:
    return {int(r["step"]): r for r in _jsonl(path) if r.get("event") == "train_step"}


def checks(ev: Path, plan: dict) -> dict:
    """Every PA2v verdict, from the evidence files alone."""
    from shared.evaluation import decoding as DEC
    gv, st = plan["study"]["gpu_validation"], plan["study"]["storage"]
    c = {}
    gpu = _read(ev / "gpu.json")
    c["gpu"] = {"name": gpu.get("name"), "ok": "L40S" in str(gpu.get("name"))}
    runs = _read(ev / "runs.json")
    rows = {n: _steps(ev / "training" / n / "train_log.jsonl") for n in RUNS}
    tr = {}
    for n, (arm, steps) in RUNS.items():
        r = rows[n]
        tr[n] = {"steps_logged": sorted(r)[-1] if r else 0, "complete": sorted(r) == list(range(1, steps + 1)),
                 "trainers_exit_ok": all(t.get("rc") == 0 or t.get("killed") for t in (runs.get(n) or [{}])),
                 "finite": all(math.isfinite(x[k]) for x in r.values() for k in ("loss", "grad_norm")
                               if x.get(k) is not None)}
    c["training"] = {"runs": tr, "ok": all(v["complete"] and v["trainers_exit_ok"] and v["finite"] for v in tr.values())}
    peak = max((x.get("gpu_mem_gb") or 0.0) for r in rows.values() for x in r.values()) if any(rows.values()) else None
    c["memory"] = {"peak_gpu_mem_gb": peak, "limit_gb": gv["checks"]["memory_gb_max"],
                   "ok": peak is not None and 0 < peak <= gv["checks"]["memory_gb_max"]}
    outside = {n: recipe_outside(plan, _read(ev / "training" / n / "recovery_config.json")) for n in RUNS}
    formal = formal_recipes_derive(plan)
    c["config"] = {"validation_runs_outside_overrides": outside, "formal_recipes": formal,
                   "ok": all(_read(ev / "training" / n / "recovery_config.json") for n in RUNS)
                   and not any(outside.values()) and all(v["ok"] for v in formal.values())}
    init = _read(ev / "initialization.json")
    c["initialization"] = {**init, "ok": init.get("reidentified") is True}
    pca = _read(ev / "pca_checkpoint.json")
    want_pca = plan["study"]["arms"]["PCA-Original"]["checkpoints"][str(gv["training"]["seed"])]["model_safetensors_sha256"]
    c["pca_checkpoint"] = {**pca, "ok": pca.get("sha256") == want_pca}
    after = range(gv["training"]["save_every"] + 1, RUNS["Y"][1] + 1)
    resumed, floor = P.compare(rows["X1"], rows["Y"], after), P.compare(rows["X1"], rows["X2"], after)
    w = _read(ev / "weights.json")
    tol = lambda got, base: got is not None and base is not None and got <= max(base * 2.0, 1e-6)  # noqa: E731
    killed = any(t.get("killed") for t in (runs.get("Y") or []))
    c["resume"] = {"killed_after_save_point": killed, "resumed_vs_uninterrupted": resumed, "run_to_run_floor": floor,
                   "weights": w,
                   "ok": bool(killed and not resumed["exact_mismatch"] and tol(resumed["max_rel_loss"], floor["max_rel_loss"])
                              and (w.get("resumed") or {}).get("same_keys")
                              and tol((w.get("resumed") or {}).get("max_abs_diff"), (w.get("floor") or {}).get("max_abs_diff")))}
    pers, probe = _read(ev / "persistence.json"), _read(ev / "probe.json")
    c["persistence"] = {**pers, "probe": probe,
                        "ok": pers.get("reidentified_from_landed_bytes") is True and probe.get("read_back_ok") is True
                        and probe.get("bytes") == gv["volume_probe_bytes"]}
    planned = int(st["bytes"]["save_point"])
    c["capacity"] = {"provider_bytes": probe.get("provider_bytes"), "planned_bytes": int(st["volume_gb"]) * 10 ** 9,
                     "save_point_bytes": pers.get("save_point_bytes"), "planned_save_point_bytes": planned,
                     "ok": probe.get("provider_bytes") == int(st["volume_gb"]) * 10 ** 9
                     and pers.get("save_point_bytes") is not None and pers["save_point_bytes"] <= planned * 1.02}
    protocol = DEC.load(REPO / plan["prereg"]["evaluation"]["decoding"]["path"])
    c["decoding"] = decoding_checks(ev / "evaluation" / "generations", protocol)
    steady = sorted(x["seconds"] for x in rows["X1"].values() if x["step"] > 2 and x.get("seconds") is not None)
    evt = _read(ev / "evaluation" / "timing.json")
    measured = {"median_step_seconds": steady[len(steady) // 2] if steady else None,
                "max_step_seconds": steady[-1] if steady else None, "peak_gpu_mem_gb": peak,
                "evaluation": evt, "save_point_bytes": pers.get("save_point_bytes")}
    c["throughput"] = {**measured, "ok": measured["median_step_seconds"] is not None}
    return {"checks": c, "measured": measured, "passed": all(v.get("ok") for v in c.values())}


# ---- on the pod -----------------------------------------------------------------------------------

def evaluate(model_dir: Path, out: Path, plan: dict, items: int, env: dict, label: str) -> dict:
    """The first `items` fluid items and gsm8k prompts under the preregistered protocol."""
    from shared import eval_sets as ES
    from stages.calibration_volume.dose_driver import VLLM_PY
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256
    from stages.phase_c1.packaging import build_evaluation_package
    ev = D.evaluation_spec(plan)
    pkg, gens = out / "package", out / "generations"
    shutil.rmtree(pkg, ignore_errors=True)
    gens.mkdir(parents=True, exist_ok=True)
    build_evaluation_package(model_dir, tokenizer_source=REPO / D.TOKENIZER_SOURCE, dest=pkg,
                             expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
    dec, ctx = str(REPO / ev["decoding"]["path"]), str(ev["trained_context"])
    t0 = time.time()
    with (out / "eval_fluid.log").open("w") as lf:
        rc_f = subprocess.run([VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py", "--model", str(pkg),
                               "--label", label, "--tasks", str(ES.unit_path(ev["set"], "fluid_reasoning_720", "dev_v1")),
                               "--mode", "eval", "--out", str(gens / f"{FLUID_KEY}.jsonl"), "--trained-context", ctx,
                               "--action-protocol", ev["action_protocol"], "--limit", str(items),
                               "--decoding", dec, "--seed-namespace", FLUID_KEY],
                              cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    t1 = time.time()
    staged = out / "uncapped_in" / f"{GSM8K_KEY}.jsonl"
    staged.parent.mkdir(exist_ok=True)
    src = ES.unit_path(ev["set"], "stage3_capability_846", "gsm8k")
    staged.write_text("".join(src.read_text().splitlines(keepends=True)[:items]))
    with (out / "eval_capability.log").open("w") as lf:
        rc_c = subprocess.run([VLLM_PY, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg),
                               "--label", label, "--prompts", str(staged), "--out-dir", str(out / "uncapped_out"),
                               "--trained-context", ctx, "--decoding", dec],
                              cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    t2 = time.time()
    raw = out / "uncapped_out"
    if (raw / f"{GSM8K_KEY}.json").is_file():
        shutil.copyfile(raw / f"{GSM8K_KEY}.json", gens / f"{GSM8K_KEY}.summary.json")
    if (raw / f"{GSM8K_KEY}.generations.jsonl").is_file():
        shutil.copyfile(raw / f"{GSM8K_KEY}.generations.jsonl", gens / f"{GSM8K_KEY}.jsonl")
    shutil.rmtree(pkg, ignore_errors=True)
    tok = lambda f: sum(len(t.get("output_token_ids") or ()) for r in _jsonl(f) for t in (r.get("turns") or [r]))  # noqa: E731
    return {"rc": {"fluid": rc_f, "gsm8k": rc_c}, "items_per_unit": items,
            "seconds": {"fluid": round(t1 - t0, 1), "gsm8k": round(t2 - t1, 1)},
            "generated_tokens": {"fluid": tok(gens / f"{FLUID_KEY}.jsonl"), "gsm8k": tok(gens / f"{GSM8K_KEY}.jsonl")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--volume-mount", required=True)
    ap.add_argument("--volume-bytes", type=int, required=True, help="the network volume's size as the PROVIDER reports it")
    ap.add_argument("--status", required=True)
    ap.add_argument("--pack-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/packs"))
    ap.add_argument("--checkpoint-root", default=str(REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/pca_original"))
    ap.add_argument("--init-dir", default=str(REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/"
                                                     "1301deb71685aa1ca5c7b6cf2c491ac4"))
    ap.add_argument("--work", default=None)
    args = ap.parse_args(argv)
    run_root = Path(args.durable_root) / "validation" / "pa2v" / args.run_id
    ev = run_root / "evidence"
    work = Path(args.work or REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/pa2v" / args.run_id)
    for d in (ev / "training", ev / "evaluation", work):
        d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}
    rep = {"schema": "aadistill.stage2.reasoning_data.pa2v_report/v1", "run_id": args.run_id, "formal": False,
           "started_utc": UD._now(), "not": "a GPU validation: never a formal H1-PA result"}
    stage, out_dirs = "plan", {}
    try:
        rep["released_earlier_validation_bytes"] = release_validation_bytes(run_root.parent)
        plan = D.load_plan()
        gv = plan["study"]["gpu_validation"]
        refusals = D.pack_refusals(plan, "D4-Augmented", Path(args.pack_root)) + \
            D.pack_refusals(plan, "D4-Original", Path(args.pack_root))
        if refusals:
            raise RuntimeError(f"pack refusals: {refusals}")
        stage = "capacity_probe"
        (ev / "probe.json").write_text(json.dumps(D.probe_volume(run_root, Path(args.volume_mount), args.volume_bytes,
                                                               gv["volume_probe_bytes"]), indent=1))
        stage = "gpu"
        import torch
        (ev / "gpu.json").write_text(json.dumps({"name": torch.cuda.get_device_name(0),
                                                 "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}))
        stage = "restore_initialization"
        init = UD.restore_init(Path(args.init_dir), plan["init"])
        (ev / "initialization.json").write_text(json.dumps({"reidentified": True,
                                                            "artifact_digest": plan["init"]["artifact_digest"]}))
        stage = "training"
        runs = {}
        for name, (arm, steps) in RUNS.items():
            out_dirs[name] = (run_root if name == "Y" else work) / f"train_{name}"
            data = Path(args.pack_root) / plan["study"]["arms"][arm]["dataset"]
            cfg = validation_recipe(plan, arm, steps=steps, save_every=gv["training"]["save_every"] if name != "O" else 0,
                                    out_dir=out_dirs[name], data_dir=data, init=init, name=f"pa2v_{args.run_id}_{name}")
            cfg_path, log = work / f"{name}.json", work / f"{name}.log"
            if name == "Y":
                first = P.run_trainer(cfg, cfg_path, log, env, kill_after_tag=f"step_{gv['training']['save_every']:06d}")
                save = out_dirs[name] / "checkpoints" / f"step_{gv['training']['save_every']:06d}"
                save_bytes = sum(f.stat().st_size for f in save.rglob("*") if f.is_file()) if save.is_dir() else None
                runs[name] = [first, P.run_trainer(cfg, cfg_path, log, env, resume=True)]
            else:
                runs[name] = [P.run_trainer(cfg, cfg_path, log, env)]
            dest = ev / "training" / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cfg_path, dest / "recovery_config.json")
            if (out_dirs[name] / "train_log.jsonl").is_file():
                shutil.copyfile(out_dirs[name] / "train_log.jsonl", dest / "train_log.jsonl")
        (ev / "runs.json").write_text(json.dumps(runs, indent=1))
        from stages.phase_c1.autoinit_c1_driver import trained_model_dir
        final = {n: trained_model_dir(out_dirs[n]) for n in ("X1", "X2", "Y")}
        (ev / "weights.json").write_text(json.dumps({"resumed": P.weights_delta(final["X1"], final["Y"]),
                                                     "floor": P.weights_delta(final["X1"], final["X2"])}, indent=1))
        stage = "persistence"
        vroot = run_root / "durable"
        PU.persist_checkpoint("pa2v-Y", final["Y"], vroot, adapter=UD._qwen3_adapter(),
                              arch_signature=plan["init"]["arch_signature"], num_parameters=plan["init"]["num_parameters"],
                              inputs={"pa2v": args.run_id, "run": "Y"}, capacity=D.free_bytes(Path(args.volume_mount),
                                                                                              args.volume_bytes))
        reread = UD._verified(vroot / "pa2v-Y/checkpoint", PU.completed(vroot, "pa2v-Y", "checkpoint"), UD._qwen3_adapter())
        (ev / "persistence.json").write_text(json.dumps({"reidentified_from_landed_bytes": bool(reread),
                                                         "save_point_bytes": save_bytes,
                                                         "used_bytes_on_mount": D.used_bytes(Path(args.volume_mount))}))
        stage = "pca_checkpoint"
        seed = int(gv["training"]["seed"])
        sha = plan["study"]["arms"]["PCA-Original"]["checkpoints"][str(seed)]["model_safetensors_sha256"]
        pca = D.stage_checkpoint(Path(args.checkpoint_root), seed, sha)
        (ev / "pca_checkpoint.json").write_text(json.dumps({"seed": seed, "sha256": UD._sha(pca / "model.safetensors")}))
        stage = "evaluation"
        timing = evaluate(pca, ev / "evaluation", plan, gv["evaluation"]["fluid_items"], env, f"pa2v_{args.run_id}")
        (ev / "evaluation" / "timing.json").write_text(json.dumps(timing, indent=1))
        stage = "checks"
        rep.update(checks(ev, plan))
        stage = "done"
    except BaseException as exc:  # noqa: BLE001
        rep["error"] = {"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:1500],
                        "traceback": traceback.format_exc()[-3000:]}
    finally:
        UD.close_observers()
        try:
            rep["released_validation_bytes"] = release_validation_bytes(run_root.parent)
        except OSError as exc:
            rep["released_validation_bytes"] = {"error": f"{type(exc).__name__}: {exc}"}
            stage = "release"
        rep["finished_utc"] = UD._now()
        rep["passed"] = stage == "done" and bool(rep.get("passed"))
        (run_root / REPORT).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        with Path(args.status).open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"MARKER:{'ALL_DONE' if rep['passed'] else 'RUN_FAILED'}\n")
    return 0 if rep["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

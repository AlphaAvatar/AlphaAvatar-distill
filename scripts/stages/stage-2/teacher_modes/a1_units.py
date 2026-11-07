#!/usr/bin/env python3
"""On-pod driver of A1's two units (Research A v2, A1 phases C and D): each seed trained from the hybrid full search's
Top-1, then formally evaluated, seed 20260726 first.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/a1_units.py --authorization <a.json> \\
        --durable-root <volume>/teacher_modes --status <file>

The units are `training_plan_a1.json`'s, written on the dev box from the committed search record: each config is
the frozen recipe (E1's, three epochs, batch size 2 of 8,192-token packed sequences, AdamW 5e-5, CE 0.25 + KD 1.0 at
the assistant positions with Top-200 + tail, per-turn isolation, the hybrid in thinking mode as the online teacher)
with its ``lineage`` block. Before a step runs:
* the hybrid's tokenizer files are staged from its snapshot and refused unless each file's sha256 is the pinned one;
* the launch admission (`training_plan.admit`) checks every lineage fact the pod can observe -- the Top-1's bytes and
  effective config on the volume, the search record's root and checkpoint, the cached teacher snapshot, the pack, the
  tokenizer through the trainer's own contract -- and the trainer checks them again, the loaded model after loading.

STATE, from the durable root alone, so a replacement session continues: a unit with its formal report is done; a
unit whose training completed (``run_completion.json``) is evaluated; a unit with save points resumes at its latest
(the checkpoint's lineage stamp checked); otherwise it trains from the Top-1. A unit's training stops at its deadline
(``--unit-deadline-hours``); a stopped or failed unit stops the session -- the second seed never starts on a failure
of the first, and is never skipped because of the first one's result.

The evaluation is the frozen eval-set-1 v2 with nucleus_v1 (`h1pa_driver.evaluate`), in thinking mode with the
hybrid's tokenizer and template; every item of every unit is generated.

Stage 3 of six_category_schema_restored@v1 (``--arm A1-6cat-sr|A2-6cat-sr|C1-6cat-sr``, ``--plan`` the bound
packed_training_plan.json, ``--units-dir stage3_sr``): the same state machine, each arm in its own mode and its own
teacher's tokenizer and template; ``--volume-gb`` refuses the session before any training when the volume's room
(the size sold less ``du -sb`` of the mount) cannot hold what its remaining units write (`volume_room`). Each
unit's record carries the save point it resumed from and its measured step seconds, GPU memory and save points
(`train_telemetry`), and where its final weights stay on the volume. Before a step runs, the unit's attention is
checked on the pod's own bytes (`packed_training_plan.attention_problems`: the declared ``cross_context``, the pack's
sequential causal packing, no position ids in the sequences or from the trainer's reader), and after training the
trainer's own log must name that strategy. Once trained, a unit's final weights get their off-volume copy on the relay
(`persist_off_volume`) before its evaluation starts -- only when the relay's quota admits them; a refusal is recorded
beside the unit and a later session tries again.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/training_plan_a1.json"
PREREG = REPO / "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json"
SCHEMA = "aadistill.stage2.teacher_modes.a1_units/v1"
TEMPLATE = {"enable_thinking": True}
#: the packed round's arms (`--arm`, `--plan`, `--kind`): each in its own mode and template; C1-6cat's teacher is
#: Thinking-2507, whose template opens the think block itself. A1 (the defaults) is unchanged.
ARM_TEMPLATE = {"A1": TEMPLATE, "A1-6cat": {"enable_thinking": True}, "A2-6cat": {"enable_thinking": False},
                "C1-6cat": {},
                #: six_category_schema_restored@v1's Stage 3 (packed_training_plan.json)
                "A1-6cat-sr": {"enable_thinking": True}, "A2-6cat-sr": {"enable_thinking": False}, "C1-6cat-sr": {}}
HYBRID_ARMS = frozenset({"A1", "A1-6cat", "A2-6cat", "A1-6cat-sr", "A2-6cat-sr"})


def units(plan: dict, arm: str = "A1") -> list[dict]:
    """The plan's units of ``arm`` in their order (seed 20260726 first)."""
    us = [u for u in plan["units"] if u["arm"] == arm]
    return sorted(us, key=lambda u: plan["seed_order"].index(u["seed"]))


def evaluation_spec() -> dict:
    pr = json.loads(PREREG.read_text())["evaluation"]
    return {"set": pr["set"]["eval_set_id"], "manifest_sha256": pr["set"]["manifest_sha256"], "trained_context": 8192,
            "action_protocol": pr["set"]["action_protocol"],
            "decoding": {"path": pr["decoding"]["path"], "sha256": pr["decoding"]["sha256"]}}


def state(out_dir: Path, eval_dir: Path) -> str:
    if (eval_dir / "formal_report.json").is_file():
        return "done"
    if (out_dir / "run_completion.json").is_file() or list(out_dir.glob("run_completion_resume_step*.json")):
        return "trained"
    if (out_dir / "checkpoints" / "latest.txt").is_file():
        return "resume"
    return "fresh"


def volume_room(mount: Path, volume_gb: float, us: list[dict], states: dict[str, str]) -> dict:
    """What the session's remaining units still write to the network volume against its room (``df`` at a network
    mount reports the backing cluster, so the room is the size sold, in decimal GB, less ``du -sb`` of the mount).
    A fresh unit writes a save point twice over while saving, then its final save point and the evaluation package;
    a resumed or trained one only the package beyond what it holds. Raises before any training when it cannot fit."""
    from stages.teacher_modes import packed_training_plan as PTP
    ckpt, model = PTP.MODEL_BYTES + PTP.OPTIM_BYTES, PTP.MODEL_BYTES
    used = int(subprocess.run(["du", "-sb", str(mount)], capture_output=True, text=True, check=True).stdout.split()[0])
    need = sum((ckpt + model) if states[u["unit"]] == "fresh" else model
               for u in us if states[u["unit"]] != "done")
    need += ckpt if any(states[u["unit"]] in ("fresh", "resume") for u in us) else 0
    room = int(float(volume_gb) * 1e9) - used
    out = {"mount": str(mount), "volume_gb": volume_gb, "used_bytes": used, "room_bytes": room, "need_bytes": need}
    if need > room:
        raise RuntimeError(f"the volume has {room / 1e9:.1f} GB left and the session's units need {need / 1e9:.1f} GB: "
                           "nothing is trained (a maintainer decision: enlarge the volume or retire evidence)")
    return out


def train_telemetry(out_dir: Path) -> dict:
    """The unit's measured training: seconds per optimizer step, peak GPU memory and save points, from its log."""
    import statistics
    log = out_dir / "train_log.jsonl"
    if not log.is_file():
        return {}
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    st = [r for r in rows if r.get("event") == "train_step"]
    secs = sorted(r["seconds"] for r in st)
    return {"logged_steps": len(st), "median_seconds_per_step": statistics.median(secs) if secs else None,
            "p95_seconds_per_step": secs[int(0.95 * (len(secs) - 1))] if secs else None,
            "kd_positions_per_step_median": statistics.median(r["kd_positions"] for r in st) if st else None,
            "peak_gpu_mem_gb": max((r.get("gpu_mem_gb") or 0 for r in st), default=None),
            "peak_gpu_mem_reserved_gb": max((r.get("gpu_mem_reserved_gb") or 0 for r in st), default=None),
            "save_points": [r["step"] for r in rows if r.get("event") == "checkpoint_saved"],
            "resumed_at": [r["step"] for r in rows if r.get("event") == "resume_loaded"],
            #: what the trainer itself says it ran with, step by step (`Trainer`'s train_step rows)
            "context_strategies": sorted({r.get("context_strategy") for r in st if r.get("context_strategy")})}


#: a Stage-3 unit's off-volume copy (`copy_final_off_volume`): where on the relay, the pod's token, and which files of
#: the final model directory it carries -- the fp32 weights exactly as trained and their configs
RELAY_PREFIX = "durable/stage3_sr"
POD_TOKEN = Path("/workspace/hf/token")
FINAL_FILES = ("model.safetensors", "config.json", "generation_config.json")
OFF_VOLUME = "off_volume_copy.json"


def copy_final_off_volume(model_dir: Path, unit: str, sha256: str, *, token_file: Path = POD_TOKEN,
                          relay: str | None = None) -> dict:
    """A trained unit's second copy, off the network volume (maintainer 2026-11-07, Stage 3 authorization, section
    VI), as the packed search copied its Top-1 (`packed_full_search.copy_off_volume`): the final model directory's
    `FINAL_FILES`, bytes unchanged, under ``RELAY_PREFIX/<unit>/<sha256[:16]>/``, then downloaded again into a fresh
    cache and re-hashed. Nothing is uploaded unless the relay's quota admits the weights' exact size
    (`hub_capacity.would_accept`, which moves no bytes); a refusal is the result, never an error. Runs in its own
    process (`main`'s ``copy-final``): the driver's HF_HUB_OFFLINE=1 is read when huggingface_hub is imported."""
    import tempfile
    from huggingface_hub import HfApi, hf_hub_download
    from aadistill.infrastructure.manifest import sha256_file
    from aadistill.runtime.hub_capacity import would_accept
    from shared.deployment import MAIN_RELAY
    relay, model_dir = relay or MAIN_RELAY, Path(model_dir)
    files = {n: sha256_file(model_dir / n) for n in FINAL_FILES if (model_dir / n).is_file()}
    if files.get("model.safetensors") != sha256:
        return {"verified": False, "error": f"the weights in {model_dir} are not the recorded {sha256[:12]}"}
    token, prefix = Path(token_file).read_text().strip(), f"{RELAY_PREFIX}/{unit}/{sha256[:16]}"
    out = {"relay": relay, "prefix": prefix, "source": str(model_dir), "files_sha256_on_volume": files}
    api = HfApi(token=token)
    there = {i.path for i in api.get_paths_info(relay, [f"{prefix}/{n}" for n in files], repo_type="model")}
    if len(there) < len(files):
        verdict = would_accept(relay, [(model_dir / "model.safetensors").stat().st_size], token)
        out["quota"] = str(verdict)
        if not verdict.accepted:
            return {**out, "verified": False, "refused": "the relay's private storage quota does not admit the weights"}
        t0 = time.time()
        for n in sorted(files):
            api.upload_file(path_or_fileobj=str(model_dir / n), path_in_repo=f"{prefix}/{n}", repo_id=relay,
                            repo_type="model", commit_message=f"Stage 3 {unit}: final {n}")
        out["upload_seconds"] = round(time.time() - t0, 1)
    t1, got = time.time(), {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in sorted(files):
            got[n] = sha256_file(Path(hf_hub_download(relay, f"{prefix}/{n}", repo_type="model", token=token,
                                                      cache_dir=f"{tmp}/hf", force_download=True)))
    return {**out, "files_sha256": got, "verified": got == files, "verify_seconds": round(time.time() - t1, 1),
            "rule": "uploaded from the volume's final model, downloaded again into a fresh cache and re-hashed"}


def persist_off_volume(unit_dir: Path, unit: str, model_dir: Path, sha256: str) -> dict:
    """`copy_final_off_volume` in its own process, its result kept beside the unit (`OFF_VOLUME`) so a later session
    sees it; a unit whose copy is verified is not copied again. Never raises: the volume's copy stands either way."""
    path = Path(unit_dir) / OFF_VOLUME
    try:
        prev = json.loads(path.read_text()) if path.is_file() else {}
        if prev.get("verified") and prev.get("files_sha256", {}).get("model.safetensors") == sha256:
            return prev
        res = subprocess.run([sys.executable, str(Path(__file__).resolve()), "copy-final", str(model_dir), unit, sha256],
                             env={**os.environ, "HF_HUB_OFFLINE": "0"}, capture_output=True, text=True, timeout=3600)
        lines = [x for x in (res.stdout or "").splitlines() if x.startswith("{")]
        out = json.loads(lines[-1]) if lines else {"verified": False, "error": (res.stderr or "")[-1000:]}
    except Exception as exc:  # noqa: BLE001
        out = {"verified": False, "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
    out["attempted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        path.write_text(json.dumps(out, indent=1) + "\n")
    except OSError:
        pass
    return out


def final_model(out_dir: Path, steps: int) -> Path:
    d = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
    if not (d / "model.safetensors").is_file():
        raise RuntimeError(f"{d} holds no final weights")
    return d


def train(cfg_path: Path, *, resume: bool, deadline_s: float, log: Path) -> int:
    cmd = [sys.executable, "scripts/shared/training/train_stage3.py", "--config", str(cfg_path)]
    if resume:
        cmd += ["--resume", "latest"]
    env = {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}
    with open(log, "a") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            return p.wait(timeout=deadline_s)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
            return 124


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["copy-final"]:          #: `copy_final_off_volume` in its own process (online), one JSON line out
        try:
            out = copy_final_off_volume(Path(argv[1]), argv[2], argv[3])
        except Exception as exc:  # noqa: BLE001 -- the caller records a failed copy; the volume's copy stands
            out = {"verified": False, "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
        print(json.dumps(out))
        return 0 if out.get("verified") else 1
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--kind", default="train_a1")
    ap.add_argument("--plan", default=str(PLAN.relative_to(REPO)))
    ap.add_argument("--arm", default="A1", choices=sorted(ARM_TEMPLATE))
    ap.add_argument("--units-dir", default="v2_units")
    ap.add_argument("--volume-gb", type=float, default=None,
                    help="the network volume's size: the session refuses units its room cannot hold")
    a = ap.parse_args(argv)
    durable, status = Path(a.durable_root), Path(a.status)
    record = {"schema": SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": {},
              "arm": a.arm, "kind": a.kind}
    evidence = durable / ("a1_units.json" if a.arm == "A1" else f"{a.arm}_units.json")

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1, default=str) + "\n")

    ok = True
    try:
        from stages.teacher_modes import token_matched_phase as TM
        from stages.teacher_modes import training_plan as TP
        record["authorization"] = TM.require_on_pod(Path(a.authorization), a.kind, REPO).get("authorization_id")
        plan = json.loads((REPO / a.plan).read_text())
        from shared.pod.teacher_cache import local_snapshot
        us = units(plan, a.arm)
        if not us:
            raise RuntimeError(f"{a.plan} holds no unit of arm {a.arm}")
        t = plan.get("teacher") or us[0]["config"]["teacher"]
        snap = Path(local_snapshot(t["model_id"], t["revision"]))
        if a.arm in HYBRID_ARMS:
            record["tokenizer"] = TP.stage_tokenizers(snap)
        mark("CONTRACT_OK")
        states = {u["unit"]: state(Path(u["config"]["out_dir"]), durable / a.units_dir / u["unit"] / "evaluation")
                  for u in us}
        if a.volume_gb is not None:
            record["volume"] = volume_room(durable.parent, a.volume_gb, us, states)
            save()
        from stages.teacher_modes import packed_training_plan as PTP
        stage3 = a.arm in PTP.ARMS
        if stage3:                          #: a unit an earlier session scored whose off-volume copy is still owed
            for u in us:                    #: (a trained, unscored one is copied in the loop, before its evaluation)
                if states[u["unit"]] == "done":
                    steps = int(u["config"]["schedule"]["total_steps"])
                    try:
                        md = final_model(Path(u["config"]["out_dir"]), steps)
                    except RuntimeError:
                        continue
                    from aadistill.infrastructure.manifest import sha256_file
                    record["units"].setdefault(u["unit"], {})["off_volume_copy"] = persist_off_volume(
                        durable / a.units_dir / u["unit"], u["unit"], md, sha256_file(md / "model.safetensors"))
            save()
        for u in us:
            name, cfg = u["unit"], u["config"]
            row = record["units"].setdefault(name, {})
            out_dir, eval_dir = Path(cfg["out_dir"]), durable / a.units_dir / name / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)
            st = state(out_dir, eval_dir)
            row["state_at_start"] = st
            if st == "done":
                continue
            if st == "resume":
                row["resume_from"] = (out_dir / "checkpoints" / "latest.txt").read_text().strip()
            if st in ("fresh", "resume"):
                if stage3:                  #: standard causal attention, as declared, on the pod's own bytes
                    bad = PTP.attention_problems(cfg, REPO)
                    if bad:
                        raise RuntimeError(f"{name}: the attention is not the declared one: {bad}")
                    row["attention"] = {"context_strategy": cfg["batch"]["context_strategy"],
                                        "checked": "config, ladder.json, blocks.npz, the trainer's reader"}
                TP.admit([{"unit": name, "config": cfg}], snapshots={t["model_id"]: snap})
                row["admitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                cfg_path = out_dir / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
                save()
                mark(f"TRAIN_START:{name}")
                t0 = time.time()
                rc = train(cfg_path, resume=st == "resume", deadline_s=a.unit_deadline_hours * 3600,
                           log=out_dir / "train_stdout.log")
                row.update(train_rc=rc, train_seconds=round(time.time() - t0, 1), telemetry=train_telemetry(out_dir))
                save()
                if rc != 0:
                    raise RuntimeError(f"{name}: training exited {rc} ({'deadline' if rc == 124 else 'failure'})")
                if stage3 and row["telemetry"].get("context_strategies") != [PTP.CONTEXT_STRATEGY]:
                    raise RuntimeError(f"{name}: the trainer logged {row['telemetry'].get('context_strategies')}, not "
                                       f"[{PTP.CONTEXT_STRATEGY!r}]")
                mark(f"TRAINED:{name}")
            steps = int(cfg["schedule"]["total_steps"])
            model_dir = final_model(out_dir, steps)
            from aadistill.infrastructure.manifest import sha256_file
            row["final_model_safetensors_sha256"] = sha256_file(model_dir / "model.safetensors")
            row["final_model_path"] = str(model_dir)
            save()
            if stage3 and not (row.get("off_volume_copy") or {}).get("verified"):
                row["off_volume_copy"] = persist_off_volume(durable / a.units_dir / name, name, model_dir,
                                                            row["final_model_safetensors_sha256"])
                mark(f"OFF_VOLUME_{'OK' if row['off_volume_copy'].get('verified') else 'NOT_COPIED'}:{name}")
                save()
            mark(f"EVAL_START:{name}")
            from stages.reasoning_data import h1pa_driver as HD
            ident = json.loads((REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json").read_text())
            if a.arm in HYBRID_ARMS:
                pkg = {"tokenizer_source": str(REPO / TP.TOKENIZERS["hybrid"]),
                       "sidecar_sha256": {n: ident["files_sha256"][n] for n in ("tokenizer.json", "tokenizer_config.json")}}
            else:                       #: Thinking-2507's tokenizer, the one its lineage trained with
                src = REPO / TP.TOKENIZERS["thinking_2507"]
                pkg = {"tokenizer_source": str(src),
                       "sidecar_sha256": {n: sha256_file(src / n) for n in ("tokenizer.json", "tokenizer_config.json")}}
            t0 = time.time()
            rep, _files = HD.evaluate({"evaluation": evaluation_spec(), "out": eval_dir, "progress": eval_dir / "progress",
                                       "label": name, "model_dir": model_dir, "package": pkg,
                                       "template_kwargs": ARM_TEMPLATE[a.arm],
                                       "env": {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}})
            row.update(eval_seconds=round(time.time() - t0, 1), formal_report=str(eval_dir / "formal_report.json"))
            save()
            mark(f"UNIT_DONE:{name}")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if ok else "RUN_FAILED")
    save()
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

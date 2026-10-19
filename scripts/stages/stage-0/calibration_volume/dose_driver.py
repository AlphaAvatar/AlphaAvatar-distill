#!/usr/bin/env python3
"""On-pod driver of one formal Stage-0 volume dose: search -> recovery -> eval-set-1.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/dose_driver.py \
        --treatment d1 --run-id <run> --auth <authorization.json> --plan dose_d1 \
        --status <file> --volume /runpod-volume

One dose of the preregistered study (`plans/volume_preregistration.json`):

* S -- the D1 full search over the dose's two profiles, in its own process
  (`search_driver.py`), under the treatment's STABLE search identity, with the
  validated execution options (DEPTH prefix reuse, device-side save/reload
  check) and every state's bytes kept;
* R -- the committed Top-1 recovered once, the frozen recipe
  (`e1_p1_kd_heavy@0.86M`) at its own seed, by the C1 trainer;
* E -- the recovered student measured on EVERY item of eval-set-1 (P18: the
  trained 8,192-token context; action protocol v2), scored by
  `shared.evaluation.eval_set_scoring` under the frozen set's formal protocols.

DURABILITY (maintainer 2026-10-19): the search's journal and every materialized
state are mirrored to the network volume while the search runs
(`volume_sync.py`), and each finished stage's products are pushed there before
the next starts. A replacement session for the same dose pulls the mirror back
to the same path first: the search engine restores every journaled state from
its bytes (re-identified, never trusted), a trained student on the volume is
evaluated rather than retrained (P8.4 state 2), and finished evaluation
generations are rescored rather than regenerated.
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

AUDIT = "artifacts/audit/stage0_calvol_dose"
SEARCH_AUDIT = "artifacts/audit/stage0_calvol_search"
PREREG = "logs/stages/stage-0/calibration_volume/plans/volume_preregistration.json"
TRAIN_PY = "/opt/train/bin/python"
VLLM_PY = "/opt/vllm/bin/python"
SET_ID = "eval-set-1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu() -> dict:
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        return dict(zip(("name", "memory", "driver"), [x.strip() for x in line.split(",")]))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def sync(mode: str, src: Path, dst: Path, log: Path) -> dict:
    out = subprocess.run([sys.executable, str(REPO / "scripts/stages/stage-0/calibration_volume/volume_sync.py"),
                          mode, "--src", str(src), "--dst", str(dst), "--log", str(log)],
                         capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"rc": out.returncode, "stderr": out.stderr[-500:]}


def legacy_prompts(dest: Path) -> Path:
    """The 150 in uncapped_eval's prompt shape: the item's own messages (system
    included, so the evaluator does not inject a second one) and its tools."""
    from shared import eval_sets as ES
    rows = []
    for it in ES.items(SET_ID, "stage3_legacy_150", "items"):
        rows.append({"id": it["id"], "group": it.get("data_type"), "source": it.get("source"),
                     "messages": it["prompt_messages"], "tools": it.get("tools")})
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return dest


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G

    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--image-digest", default=None)
    args = ap.parse_args(argv)
    plan = G.plan_doc(args.plan)
    prereg = json.loads((REPO / PREREG).read_text())
    treat = prereg["treatments_by_id"][args.treatment]
    out = REPO / AUDIT / args.treatment
    out.mkdir(parents=True, exist_ok=True)
    vol = Path(args.volume) / "stage0_calvol" / prereg["study_id"] / args.treatment
    status = Path(args.status)
    search_unit = REPO / SEARCH_AUDIT / args.treatment
    env = {**os.environ, "PYTHONPATH": "src:scripts"}
    if args.image_digest:
        env["AADISTILL_IMAGE_DIGEST"] = args.image_digest
    record = {"schema": "aadistill.stage0.calibration_volume.dose_record/v1",
              "treatment": args.treatment, "run_id": args.run_id,
              "search_run_id": treat["search_run_id"], "volume_dir": str(vol),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       capture_output=True, text=True).stdout.strip(),
              "gpu": gpu(), "stages": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        (out / "dose_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    def done_unit(name: str, path: Path) -> None:
        (out / "records").mkdir(exist_ok=True)
        (out / "records" / f"{name}.done").write_text(sha(path) + "\n")

    def stage(name: str):
        record["stages"][name] = {"t0": time.time(), "status": "running"}
        save()
        mark(f"STAGE_START:{name}")
        return record["stages"][name]

    def finish(st: dict, name: str, **kw) -> None:
        st.update(kw, seconds=round(time.time() - st.pop("t0"), 1), status="ok")
        save()
        mark(f"STAGE_OK:{name}")

    mark(f"DRIVER_START:{args.treatment}")
    try:
        # -- A: contract ---------------------------------------------------
        st = stage("A_contract")
        auth = G.load_for("dose", args.auth)
        auth.require_plan(G.plan_hash(args.plan))
        record["authorization_id"] = auth.authorization_id
        from aadistill.evaluation import eval_registry as ER
        from shared import eval_sets as ES
        es_identity = ER.current_identity(ER.entry(ES.registry(), SET_ID))
        if es_identity != prereg["evaluation"]["eval_set_identity"]:
            raise RuntimeError(f"{SET_ID} identity {es_identity} != preregistered "
                               f"{prereg['evaluation']['eval_set_identity']}")
        ES.manifest(SET_ID)                                  # bytes vs identity
        knowledge = ES.unit_path(SET_ID, "stage3_capability_846", "knowledge")
        if subprocess.run(["mountpoint", "-q", args.volume]).returncode != 0:
            raise RuntimeError(f"no network volume mounted at {args.volume}")
        vol.mkdir(parents=True, exist_ok=True)
        finish(st, "A_contract", eval_set_identity=es_identity,
               knowledge_staged=knowledge is not None)

        # -- S: search (resumable) ------------------------------------------
        products = vol / "products.json"
        prod = json.loads(products.read_text()) if products.is_file() else {}
        st = stage("S_search")
        if prod.get("search", {}).get("complete"):
            #: P8.4: a finished search is records plus ONE consumed checkpoint
            #: (the Top-1, and only while it is untrained). The other ~90 states
            #: stay on the volume.
            search_unit.mkdir(parents=True, exist_ok=True)
            for f in (vol / "search_unit").glob("*.json"):
                shutil.copyfile(f, search_unit / f.name)
            pulled = {"records": sorted(f.name for f in search_unit.glob("*.json"))}
            if not prod.get("recovery", {}).get("complete"):
                sel = json.loads((search_unit / "stage1_selection.json").read_text())
                ck = Path(sel["selected"][0]["checkpoint_path"])
                pulled["top1"] = sync("pull", vol / "search_unit" / ck.relative_to(search_unit),
                                      ck, out / "sync.log")
            finish(st, "S_search", resumed_complete=True, pulled=pulled)
        else:
            pulled = sync("pull", vol / "search_unit", search_unit, out / "sync.log")
            stop = out / "mirror.stop"
            stop.unlink(missing_ok=True)
            mirror = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/stages/stage-0/calibration_volume/volume_sync.py"),
                 "push", "--src", str(search_unit), "--dst", str(vol / "search_unit"),
                 "--loop", str(plan.get("mirror_seconds", 120)), "--until", str(stop),
                 "--log", str(out / "mirror.log")], cwd=REPO, env=env)
            cmd = [TRAIN_PY, "scripts/stages/stage-0/calibration_volume/search_driver.py",
                   "--treatment", args.treatment, "--run-id", args.run_id,
                   "--search-run-id", treat["search_run_id"], "--auth", args.auth,
                   "--auth-kind", "dose",
                   "--plan", args.plan, "--treatments-doc", PREREG,
                   "--profiles-doc", prereg["profiles_doc"], "--status", args.status,
                   *(["--prefix-reuse"] if prereg["execution"]["depth_prefix_reuse"] else []),
                   *(["--validate-on-device"] if prereg["execution"]["validate_on_search_device"] else []),
                   "--keep-pruned-weights"]
            with (out / "search.log").open("a") as lf:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            stop.write_text("stop\n")
            mirror.wait(timeout=3600)
            final = sync("push", search_unit, vol / "search_unit", out / "sync.log")
            srec = json.loads((search_unit / "search_record.json").read_text())
            if rc != 0 or srec.get("status") != "COMPLETE":
                raise RuntimeError(f"search_driver exited {rc} with status {srec.get('status')}; "
                                   "the journal and states are on the volume for a resume")
            prod["search"] = {"complete": True,
                              "search_record_sha256": sha(search_unit / "search_record.json")}
            products.write_text(json.dumps(prod, indent=1) + "\n")
            finish(st, "S_search", pulled_before=pulled, final_push=final,
                   n_resumed=srec["stages"]["C_beam_search"].get("n_resumed"))
        done_unit("search", search_unit / "search_record.json")
        srec = json.loads((search_unit / "search_record.json").read_text())
        record["search"] = {"commit": srec.get("commit"), "session": srec.get("session")}
        save()

        # -- R: recovery (resumable) ------------------------------------------
        st = stage("R_recovery")
        selection = json.loads((search_unit / "stage1_selection.json").read_text())
        top1 = selection["selected"][0]
        top1_dir = Path(top1["checkpoint_path"])
        model_vol = vol / "recovery"
        rec_out = REPO / "artifacts/stages/stage-0/calibration_volume/formal" / args.treatment / "recovery"
        if prod.get("recovery", {}).get("complete"):
            sync("pull", model_vol, rec_out, out / "sync.log")
            from stages.phase_c1.autoinit_c1_driver import trained_model_dir
            model_dir = trained_model_dir(rec_out)
            if sha(model_dir / "model.safetensors") != prod["recovery"]["model_sha256"]:
                raise RuntimeError("the recovered student on the volume does not hash to its record")
            finish(st, "R_recovery", resumed_complete=True, model_dir=str(model_dir))
        else:
            from stages.phase_c1.autoinit_c1_driver import (C1_PROBE_OVERRIDES, FROZEN_RECIPE,
                                                            PACK_DIR, TRAINER, trained_model_dir)
            frozen = json.loads(FROZEN_RECIPE.read_text())
            if frozen["seed"] != prereg["recovery"]["seed"]:
                raise RuntimeError("the frozen recipe's seed is not the preregistered seed")
            derived = {**frozen, "run_name": f"s0v_formal_{args.treatment}",
                       "out_dir": str(rec_out.relative_to(REPO)), "data_dir": PACK_DIR,
                       "student_path": str(top1_dir),
                       "_purpose": (f"Stage-0 volume formal dose {args.treatment}: the committed "
                                    "Top-1 recovered by the frozen recipe at its own seed; only run "
                                    "identity, pack path and student_path differ from it")}
            outside = sorted(k for k in set(frozen) | set(derived)
                             if frozen.get(k) != derived.get(k))
            if set(outside) - set(C1_PROBE_OVERRIDES):
                raise RuntimeError(f"the derived recovery config differs outside {sorted(C1_PROBE_OVERRIDES)}: {outside}")
            cfg = out / "recovery_config.json"
            cfg.write_text(json.dumps(derived, indent=2) + "\n")
            shutil.rmtree(rec_out, ignore_errors=True)
            with (out / "recovery.log").open("w") as lf:
                rc = subprocess.run([TRAIN_PY, str(TRAINER), "--config", str(cfg)], cwd=REPO,
                                    env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise RuntimeError(f"the trainer exited {rc}; see recovery.log")
            model_dir = trained_model_dir(rec_out)
            msha = sha(model_dir / "model.safetensors")
            pushed = sync("push", rec_out, model_vol, out / "sync.log")
            if (rec_out / "train_log.jsonl").is_file():
                shutil.copyfile(rec_out / "train_log.jsonl", out / "train_log.jsonl")
            prod["recovery"] = {"complete": True, "model_sha256": msha,
                                "model_bytes": (model_dir / "model.safetensors").stat().st_size,
                                "config_sha256": sha(cfg), "top1_state": top1.get("state_id")}
            products.write_text(json.dumps(prod, indent=1) + "\n")
            finish(st, "R_recovery", model_dir=str(model_dir), model_sha256=msha, pushed=pushed,
                   top1=top1, config=str(cfg.relative_to(REPO)))
        (out / "recovery_products.json").write_text(json.dumps(prod.get("recovery"), indent=1) + "\n")
        done_unit("recovery", out / "recovery_products.json")

        # -- E: eval-set-1, every item --------------------------------------------
        st = stage("E_eval")
        from stages.phase_c1.autoinit_c1_driver import (TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE,
                                                        trained_model_dir)
        from stages.phase_c1.packaging import build_evaluation_package
        pkg = out / "package"
        shutil.rmtree(pkg, ignore_errors=True)
        build_evaluation_package(trained_model_dir(rec_out), tokenizer_source=TOKENIZER_SOURCE,
                                 dest=pkg, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
        gens = out / "generations"
        gens.mkdir(exist_ok=True)
        sync("pull", vol / "generations", gens, out / "sync.log")
        label = f"s0v_formal_{args.treatment}"
        ctx = str(prereg["evaluation"]["trained_context"])
        m = ES.manifest(SET_ID)
        runs = []
        fluid = gens / "fluid_reasoning_720__dev_v1.jsonl"
        if not fluid.is_file():
            tmp = out / "fluid_tmp.jsonl"
            with (out / "eval_fluid.log").open("w") as lf:
                rc = subprocess.run(
                    [VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py",
                     "--model", str(pkg), "--label", label,
                     "--tasks", str(ES.unit_path(SET_ID, "fluid_reasoning_720", "dev_v1")),
                     "--mode", "eval", "--out", str(tmp), "--trained-context", ctx,
                     "--action-protocol", prereg["evaluation"]["action_protocol"]],
                    cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise RuntimeError(f"run_battery exited {rc}; see eval_fluid.log")
            shutil.copyfile(tmp.with_suffix(".summary.json"), gens / "fluid_reasoning_720__dev_v1.summary.json")
            os.replace(tmp, fluid)
            sync("push", gens, vol / "generations", out / "sync.log")
            runs.append("fluid")
        units = {u: ES.unit_path(SET_ID, "stage3_capability_846", u)
                 for u in m["components"]["stage3_capability_846"]["units"]}
        legacy = legacy_prompts(out / "legacy_150.prompts.jsonl")
        todo = {f"stage3_capability_846__{u}": p for u, p in units.items() if p is not None}
        todo["stage3_legacy_150__items"] = legacy
        missing = {k: p for k, p in todo.items() if not (gens / f"{k}.jsonl").is_file()}
        if missing:
            staged = out / "uncapped_in"
            shutil.rmtree(staged, ignore_errors=True)
            staged.mkdir()
            for k, p in missing.items():
                shutil.copyfile(p, staged / f"{k}.jsonl")
            raw = out / "uncapped_out"
            with (out / "eval_capability.log").open("w") as lf:
                rc = subprocess.run(
                    [VLLM_PY, "scripts/shared/evaluation/uncapped_eval.py", "--model", str(pkg),
                     "--label", label, "--prompts", *sorted(str(staged / f"{k}.jsonl") for k in missing),
                     "--out-dir", str(raw), "--trained-context", ctx],
                    cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise RuntimeError(f"uncapped_eval exited {rc}; see eval_capability.log")
            for k in missing:
                shutil.copyfile(raw / f"{k}.json", gens / f"{k}.summary.json")
                os.replace(raw / f"{k}.generations.jsonl", gens / f"{k}.jsonl")
            sync("push", gens, vol / "generations", out / "sync.log")
            runs.append("capability")
        import shared.evaluation.eval_set_scoring as S
        rep = S.report(gens, label, {}, SET_ID, rows_out=out / "rows")
        (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
        sync("push", out, vol / "dose_out", out / "sync.log")
        finish(st, "E_eval", generated=runs,
               fluid_formal=rep["units"]["fluid_reasoning_720/dev_v1"].get("correct"),
               report_sha256=sha(out / "report.json"))
        done_unit("eval", out / "report.json")
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        try:
            sync("push", out, vol / "dose_out", out / "sync.log")
        except Exception:  # noqa: BLE001
            pass
        mark("ALL_DONE" if record["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if record["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

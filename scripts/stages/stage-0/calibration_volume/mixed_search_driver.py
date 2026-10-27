#!/usr/bin/env python3
"""On-pod driver of MC1: the mixed-calibration 2x D1 full search, then its Top-1 identified and relayed.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/mixed_search_driver.py \
        --run-id <run> --auth <authorization.json> --status <file> --volume /runpod-volume

* A -- the authorization (`mixed_governance`, kind ``search``) bound to the
  plan (`mixed_search_session.json`) and the harness on the pod; the volume
  mounted.
* S -- `search_driver.py` on treatment ``2x`` under the plan's STABLE search
  identity (its run id and pinned config hash), with D4's execution options,
  the journal and every state mirrored to the network volume while it runs
  (`dose_driver`'s mechanism: a replacement session pulls the mirror back to
  the same path first and the engine re-identifies every restored state).
* T -- the committed Top-1 (k = 1): its checkpoint re-identified from its bytes
  against the journal's own artifact row (`verify_transferred_leaf`), uploaded
  unchanged to the relay (`finalists.export_candidates`, which checks what the
  hub STORED), and `top1_manifest.json` written beside the search records. The
  dev box then downloads it by sha256 and rebuilds the identity again
  (`mixed_launch.read_back_top1`) before MC2 may be authorized.

The launcher pulls each finished stage's records as `records/<name>.done`
appears; the states stay on the volume until the volume is retired.
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
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume.dose_driver import (gpu, read_products, require_pushed,  # noqa: E402
                                                   sha, sync, write_atomic)

AUDIT = "artifacts/audit/stage0_calvol_mixed_search"
SEARCH_AUDIT = "artifacts/audit/stage0_calvol_search"
PLAN = "mixed_search"
TREATMENT = "2x"
TRAIN_PY = "/opt/train/bin/python"


def top1_identity(unit: Path) -> tuple[dict, dict]:
    """The committed Top-1's selection row and the journal's artifact row for it."""
    from stages.calibration_volume.finalists import journal_index
    sel = json.loads((unit / "stage1_selection.json").read_text())
    if len(sel["selected"]) != 1:
        raise RuntimeError(f"the commit selected {len(sel['selected'])} leaves; MC1 commits exactly one")
    top = sel["selected"][0]
    row = journal_index(unit / "search" / "states.jsonl")[top["state_id"]]
    art = row.get("artifact") or {}
    if art.get("artifact_digest") != top["artifact_digest"] or not art.get("weights_digest"):
        raise RuntimeError(f"{top['state_id']}: the journal's artifact row does not match the committed selection")
    return top, {**art, "materialization": row.get("materialization"), "checkpoint_sha256": row.get("checkpoint_sha256")}


def verify_local(directory: Path, ident: dict) -> dict:
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import verify_transferred_leaf
    register_builtin_adapters()
    v = verify_transferred_leaf(directory, ident, adapter=get_adapter("qwen3"))
    if not (v["matched"] and v["weights_digest_matched"] and v["config_matched"] is not False):
        raise RuntimeError(f"{directory}: the Top-1 does not re-identify from its bytes: {v}")
    return v


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G
    from stages.calibration_volume import mixed_governance as MG

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume", required=True)
    args = ap.parse_args(argv)
    plan = G.plan_doc(PLAN)
    treat = plan["treatments_by_id"][TREATMENT]
    out = REPO / AUDIT
    (out / "records").mkdir(parents=True, exist_ok=True)
    vol = Path(args.volume) / "stage0_calvol" / "mixed_v2" / TREATMENT
    unit = REPO / SEARCH_AUDIT / TREATMENT
    status = Path(args.status)
    env = {**os.environ, "PYTHONPATH": "src:scripts"}
    record = {"schema": "aadistill.stage0.calibration_volume.mixed_search_record/v1", "run_id": args.run_id,
              "search_run_id": treat["search_run_id"], "volume_dir": str(vol),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                       text=True).stdout.strip(),
              "gpu": gpu(), "stages": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        (out / "mixed_search_record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    def stage(name: str) -> dict:
        record["stages"][name] = {"t0": time.time(), "status": "running"}
        save()
        mark(f"STAGE_START:{name}")
        return record["stages"][name]

    def finish(st: dict, name: str, **kw) -> None:
        st.update(kw, seconds=round(time.time() - st.pop("t0"), 1), status="ok")
        save()
        mark(f"STAGE_OK:{name}")

    def done_unit(name: str, path: Path) -> None:
        (out / "records" / f"{name}.done").write_text(sha(path) + "\n")

    mark(f"DRIVER_START:{args.run_id}")
    try:
        st = stage("A_contract")
        auth = MG.load_for("search", args.auth)
        auth.require_plan(MG.identity_hash("search", REPO))
        harness = auth.require_harness(REPO)
        record["authorization_id"] = auth.authorization_id
        if subprocess.run(["mountpoint", "-q", args.volume]).returncode != 0:
            raise RuntimeError(f"no network volume mounted at {args.volume}")
        vol.mkdir(parents=True, exist_ok=True)
        finish(st, "A_contract", harness_digest=harness["digest"])

        # -- S: the search (resumable) ----------------------------------------
        products = vol / "products.json"
        prod, torn = read_products(products)
        st = stage("S_search")
        if torn:
            st["products_set_aside"] = torn
        if prod.get("search", {}).get("complete"):
            pulled = sync("pull", vol / "search_unit", unit, out / "sync.log")
            if sha(unit / "search_record.json") != prod["search"]["search_record_sha256"]:
                raise RuntimeError("the search record on the volume is not the one the product record names")
            finish(st, "S_search", resumed_complete=True, pulled=pulled)
        else:
            pulled = sync("pull", vol / "search_unit", unit, out / "sync.log")
            stop = out / "mirror.stop"
            stop.unlink(missing_ok=True)
            mirror = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/stages/stage-0/calibration_volume/volume_sync.py"),
                 "push", "--src", str(unit), "--dst", str(vol / "search_unit"),
                 "--loop", str(plan.get("mirror_seconds", 120)), "--until", str(stop),
                 "--log", str(out / "mirror.log")], cwd=REPO, env=env)
            ex = plan["search"]["execution"]
            cmd = [TRAIN_PY, "scripts/stages/stage-0/calibration_volume/search_driver.py",
                   "--treatment", TREATMENT, "--run-id", args.run_id, "--search-run-id", treat["search_run_id"],
                   "--auth", args.auth, "--auth-kind", "mixed_search", "--plan", PLAN,
                   "--treatments-doc", str(G.PLANS / f"{PLAN}_session.json"),
                   "--profiles-doc", plan["profiles_doc"], "--status", args.status,
                   *(["--prefix-reuse"] if ex["depth_prefix_reuse"] else []),
                   *(["--validate-on-device"] if ex["validate_on_search_device"] else []),
                   *(["--keep-pruned-weights"] if ex["prune_weights"] is False else [])]
            with (out / "search.log").open("a") as lf:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            stop.write_text("stop\n")
            mirror.wait(timeout=3600)
            final = require_pushed(sync("push", unit, vol / "search_unit", out / "sync.log"),
                                   "the search unit (the volume's journal stops at the last good mirror pass)")
            srec = json.loads((unit / "search_record.json").read_text())
            if rc != 0 or srec.get("status") != "COMPLETE":
                raise RuntimeError(f"search_driver exited {rc} with status {srec.get('status')}; the journal and "
                                   "states are on the volume for a resume")
            prod["search"] = {"complete": True, "search_record_sha256": sha(unit / "search_record.json")}
            write_atomic(products, json.dumps(prod, indent=1) + "\n")
            finish(st, "S_search", pulled_before=pulled, final_push=final,
                   n_resumed=srec["stages"]["C_beam_search"].get("n_resumed"))
        done_unit("search", unit / "search_record.json")

        # -- T: the Top-1, identified and relayed ------------------------------
        st = stage("T_top1")
        top, ident = top1_identity(unit)
        ck = Path(top["checkpoint_path"])
        local = verify_local(ck, ident)
        from shared.deployment import MAIN_RELAY
        from stages.calibration_volume.finalists import export_candidates
        token = (os.environ.get("HF_TOKEN") or Path("/workspace/hf/token").read_text()).strip()
        files = export_candidates([{"state_id": top["state_id"], "checkpoint_sha256": ident["checkpoint_sha256"],
                                    "dir": str(ck)}],
                                  study=plan["export"]["study"], treatment=plan["export"]["treatment"],
                                  repo=MAIN_RELAY, token=token)
        manifest = {"schema": "aadistill.stage0.calibration_volume.mixed_top1_manifest/v1",
                    "study_id": plan["study_id"], "run_id": args.run_id, "search_run_id": treat["search_run_id"],
                    "search_config_hash": treat["search_config_hash"],
                    "search_record_sha256": sha(unit / "search_record.json"),
                    "selection_sha256": sha(unit / "stage1_selection.json"),
                    "selected": top, "identity": ident, "verified_on_pod": local,
                    "relay_repo": MAIN_RELAY, "files": files,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (unit / "top1_manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
        require_pushed(sync("push", unit, vol / "search_unit", out / "sync.log"), "the Top-1 manifest")
        prod["top1"] = {"complete": True, "manifest_sha256": sha(unit / "top1_manifest.json")}
        write_atomic(products, json.dumps(prod, indent=1) + "\n")
        done_unit("top1", unit / "top1_manifest.json")
        finish(st, "T_top1", state_id=top["state_id"], path=top.get("path"),
               artifact_digest=ident["artifact_digest"], relay_files=len(files))
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        try:
            sync("push", out, vol / "mc1_out", out / "sync.log")
        except Exception:  # noqa: BLE001
            pass
        mark("ALL_DONE" if record["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if record["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

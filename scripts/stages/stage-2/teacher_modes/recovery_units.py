#!/usr/bin/env python3
"""On-pod driver of Research A's six recovery units: packs rebuilt and checked, then H1-PA's unit driver per GPU.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \\
        scripts/stages/stage-2/teacher_modes/recovery_units.py --authorization <authorization.json> \\
        --run-id <run> --durable-root <volume>/teacher_modes --volume-mount <volume> --volume-bytes <provider size> \\
        --status <file> --gpus <n> [--unit H-T-S-s20260726 ...]

A unit is ``<arm>-s<seed>`` (d4_scale_design.json :: arms). What this module
supplies is the PLAN for `h1pa_driver.run` and the ops that differ:

* the packs, REBUILT here from the teacher session's shards on the volume and
  the committed admission record's selection (`admission.build_packs`), every
  file's sha256 equal to the record's or nothing trains (P8.4: the bytes are
  already on the volume; nothing crosses the dev box's uplink);
* the common initialization, re-identified from its bytes on the volume
  against the committed initialization record;
* the recipe: the seed's E1 config (H1-PA's), each arm's schedule from its
  pack (`h1pa_driver.schedule_for`, E1's rule), and the design's declared
  changes only (`recipe_overrides`: the Top-200 + tail KD resolved from the
  forward baseline, ``kd_scope`` assistant, document isolation, the hybrid
  teacher on the fly, identity binding; per arm the validation tail's size);
* the evaluation in the arm's own mode: the hybrid's tokenizer and template
  (`dose_driver.evaluate_all` ``package``) and the mode's
  ``enable_thinking``.

STATE, RESUME, PERSISTENCE and the terminal path are H1-PA's (`h1pa_driver.run`):
a scored unit is skipped, a trained one resumes at scoring, an interrupted
training continues from its last save point, a record of other inputs stops
the unit, preserved. ``--gpus n`` runs n children, one GPU each
(``CUDA_VISIBLE_DEVICES``), on disjoint units -- which changes no unit's
arithmetic -- and the status file gets ``MARKER:ALL_DONE`` only when every
requested unit is complete.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import h1pa_driver as HD  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/d4_scale_design.json"
PREREG = REPO / "logs/stages/stage-2/teacher_modes/plans/d4_scale_preregistration.json"
PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/generation_protocol.json"
H1PA = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/data/admission.json"
INIT_RECORD = REPO / "logs/stages/stage-2/teacher_modes/assets/common_initialization.json"
ARMS_MODE = {"H-T-S": "H-T", "H-N-S": "H-N", "H-N-C": "H-N", "H-T-C": "H-T"}
SCHEMA = "aadistill.stage2.teacher_modes.recovery_status/v1"


def resolved_kd_support() -> dict:
    from shared.forward_baseline import selected
    return selected(REPO).distribution_support.as_dict()


def recipe_overrides(proto: dict) -> dict:
    """The design's declared changes to every arm's E1 recipe (d4_scale_design.json :: recovery); the L40S
    check trains the same ones."""
    t = proto["teacher"]
    return {"loss.kd_support": resolved_kd_support(), "loss.kd_scope": "assistant",
            "batch.context_strategy": "document_isolated", "checkpoint.bind_identity": True,
            "teacher": {"model_id": t["model"], "revision": t["revision"], "dtype": "bfloat16"}}


def load_plan(*, admission: Path = ADMISSION, init_record: Path = INIT_RECORD, frozen: bool = False) -> dict:
    """`h1pa_driver`'s plan shape for this study, from its owners; ``frozen`` is set by the caller only
    after the phase authorization was checked (`phase.require_on_pod`)."""
    design, prereg, proto, h1pa = (UD._read(p) for p in (DESIGN, PREREG, PROTOCOL, H1PA))
    if not admission.is_file() or not init_record.is_file():
        raise HD.DriverRefusal("the admission record or the initialization record is not committed yet: "
                               "the teacher session and the dev box's admission come first")
    adm, init = UD._read(admission), UD._read(init_record)
    seed_lists = [[int(s) for s in doc["seeds"]] for doc in (design["arms"], prereg)]
    if any(x != seed_lists[0] for x in seed_lists[1:]):
        raise HD.DriverRefusal(f"the design's and the preregistration's seeds differ: {seed_lists}")
    recipes = {}
    for seed, c in h1pa["recipe"]["e1_configs"].items():
        cfg = UD._read(REPO / c["path"])
        if HD.canonical_sha(cfg) != c["canonical_sha256"]:
            raise HD.DriverRefusal(f"{c['path']}: not the E1 config H1-PA records")
        recipes[int(seed)] = cfg
    arms, datasets = {}, {}
    for arm, a in adm["arms"].items():
        pk = a["pack"]
        datasets[arm] = {"blocks_sha256": pk["files_sha256"]["blocks.npz"],
                         "ladder_sha256": pk["files_sha256"]["ladder.json"],
                         "audit_sha256": pk["files_sha256"]["audit.jsonl"],
                         "rung": pk["rungs"][0]["target_supervised_tokens"]}
        sched = HD.e1_schedule(int(a["sequences"]), epochs=3,
                               batch_size=int(recipes[seed_lists[0][0]]["batch"]["blocks_per_step"]))
        arms[arm] = {"dataset": arm, "mode": a["mode"],
                     "schedule": {"packed_sequences": int(a["sequences"]), "optimizer_steps": sched["total_steps"],
                                  "warmup_steps": sched["warmup_steps"], "eval_every": sched["eval_every"],
                                  "save_every": sched["save_every"]}}
    study = {"study_id": design["study_id"], "seeds": seed_lists[0], "arms": arms, "datasets": datasets,
             "derived_schedule_arms": sorted(arms), "frozen": bool(frozen),
             "storage": {"training_peak_extra_bytes": h1pa["storage"]["training_peak_extra_bytes"]},
             "evaluation": {"trained_context": 8192, "action_protocol": proto["execution"]["action_protocol"],
                            "decoding": prereg["evaluation"]["decoding"]["path"]}}
    overrides = recipe_overrides(proto)
    arm_over = {arm: {"val_blocks": int(adm["arms"][arm]["pack"]["n_val_blocks"]),
                      "intervals.eval_blocks": int(adm["arms"][arm]["pack"]["n_val_blocks"])} for arm in arms}
    units = [{"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed), "execution": "train"}
             for arm in sorted(arms) for seed in seed_lists[0]]
    return {"study": study, "study_sha256": UD._sha(DESIGN), "prereg": prereg, "prereg_sha256": UD._sha(PREREG),
            "recipes": recipes, "init": init["materialization_identity"], "init_record_sha256": UD._sha(init_record),
            "units": units, "recipe_overrides": overrides, "arm_overrides": arm_over, "admission": adm,
            "admission_sha256": UD._sha(admission)}


def rebuild_packs(plan: dict, generation: Path, identity: str, pack_root: Path) -> dict:
    """Each arm's pack from the volume's shards; every file must hash to the admission record's."""
    from stages.teacher_modes import admission as ADM
    got = ADM.build_packs(plan["admission"]["selection"], generation, identity, pack_root)
    bad = {arm: {n: (r["files_sha256"][n][:12], plan["admission"]["arms"][arm]["pack"]["files_sha256"][n][:12])
                 for n in r["files_sha256"] if r["files_sha256"][n] != plan["admission"]["arms"][arm]["pack"]
                 ["files_sha256"][n]} for arm, r in got.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise UD.IdentityMismatch(f"the packs rebuilt from the volume are not the admission record's: {bad}")
    return {arm: r["files_sha256"] for arm, r in got.items()}


def restore_from_volume(durable: Path):
    """The common initialization on the volume, re-identified from its bytes (never trusted)."""
    def restore(_init_dir: Path, ident: dict) -> Path:
        d = Path(durable) / "initialization" / "common_initialization" / "checkpoint"
        if not (d.is_dir() and UD._verified(d, ident, UD._qwen3_adapter())):
            raise UD.IdentityMismatch(f"{d}: the common initialization does not re-identify as "
                                      f"{ident['artifact_digest'][:12]}; preserved, nothing trained")
        return d
    return restore


def evaluate_in_mode(plan: dict):
    """`h1pa_driver.evaluate` with the hybrid's tokenizer and the arm's own template switch."""
    from shared.pod.teacher_cache import local_snapshot
    proto = UD._read(PROTOCOL)
    ident = UD._read(REPO / proto["teacher"]["identity"])

    def run(job: dict):
        arm = job["unit"].rsplit("-s", 1)[0]
        mode = plan["study"]["arms"][arm]["mode"]
        snap = local_snapshot(proto["teacher"]["model"], proto["teacher"]["revision"])
        pkg = {"tokenizer_source": snap, "sidecar_sha256": {n: ident["files_sha256"][n]
                                                             for n in ("tokenizer.json", "tokenizer_config.json")}}
        return HD.evaluate({**job, "package": pkg, "template_kwargs": proto["modes"][mode]["template_kwargs"]})
    return run


def ops_for(plan: dict, durable: Path):
    restore = restore_from_volume(durable)
    from stages.projected_residual.recovery_driver import train_restoring_on_resume
    return dataclasses.replace(HD.PRODUCTION, restore_init=restore, evaluate=evaluate_in_mode(plan),
                               train=train_restoring_on_resume(restore, plan["init"]))


def partition(units: list[dict], plan: dict, n: int) -> list[list[str]]:
    """Units over n GPUs, longest first onto the least loaded (their optimizer steps)."""
    load, out = [0] * n, [[] for _ in range(n)]
    for u in sorted(units, key=lambda u: (-plan["study"]["arms"][u["arm"]]["schedule"]["optimizer_steps"], u["unit"])):
        i = min(range(n), key=lambda k: (load[k], k))
        out[i].append(u["unit"])
        load[i] += plan["study"]["arms"][u["arm"]]["schedule"]["optimizer_steps"]
    return [x for x in out if x]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume-mount", default=None)
    ap.add_argument("--volume-bytes", type=int, default=None)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--unit", action="append", default=None)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pack-root", default="/workspace/a_packs")
    a = ap.parse_args(argv)
    durable, status = Path(a.durable_root), Path(a.status)

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")

    from stages.teacher_modes import phase as PH
    bound = PH.require_on_pod(Path(a.authorization), "recovery", REPO)
    plan = load_plan(frozen=True)
    common = ["--run-id", a.run_id, "--durable-root", str(durable / "units"), "--pack-root", a.pack_root,
              "--work", f"/workspace/a_units/{a.run_id}"]
    if a.volume_mount:
        common += ["--volume-mount", a.volume_mount, "--volume-bytes", str(a.volume_bytes)]
    if a.child:
        return HD.run([*common, "--status", str(status), *sum((["--unit", u] for u in a.unit or []), [])],
                      ops=ops_for(plan, durable), plan=plan)
    record = {"schema": SCHEMA, "run_id": a.run_id, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "authorization": bound}
    try:
        record["packs"] = rebuild_packs(plan, durable / "generation", bound["identity"], Path(a.pack_root))
        mark("PACKS_OK")
        units = [u for u in plan["units"] if not a.unit or u["unit"] in a.unit]
        groups = partition(units, plan, a.gpus)
        procs = []
        for i, g in enumerate(groups):
            cmd = [sys.executable, __file__, "--authorization", a.authorization, "--run-id", a.run_id,
                   "--durable-root", str(durable), "--status", str(status.with_name(f"{status.name}.gpu{i}.json")),
                   "--pack-root", a.pack_root, "--child", *sum((["--unit", u] for u in g), [])]
            if a.volume_mount:
                cmd += ["--volume-mount", a.volume_mount, "--volume-bytes", str(a.volume_bytes)]
            log = open(status.with_name(f"{status.name}.gpu{i}.log"), "a")
            procs.append((g, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})))
        record["groups"] = {str(i): g for i, (g, _) in enumerate(procs)}
        codes = [p.wait() for _, p in procs]
        record["exit_codes"] = codes
        ok = all(c == 0 for c in codes)
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)[:2000]
        ok = False
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (durable / f"recovery_{a.run_id}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

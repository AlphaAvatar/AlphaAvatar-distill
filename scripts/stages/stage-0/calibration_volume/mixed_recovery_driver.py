#!/usr/bin/env python3
"""On-pod driver of MC2: Mix-Augmented on both seeds -- D4-Augmented's recovery from the MC1 Top-1, then eval-set-1.

    cd /workspace/aad && PYTHONPATH=src:scripts:scripts/stages/stage-1 /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/mixed_recovery_driver.py --auth <authorization.json> \
        --run-id <run> --durable-root <volume>/mixed_v2 --pack-root <dir> --init-dir <dir> \
        --volume-mount <volume> --volume-bytes <the provider's size> --status <file> [--unit ...]

H1-PA's unit driver, unchanged (`h1pa_driver.run`): state, resume, retry,
persistence, capacity accounting, the nucleus_v1 evaluation of every item of
eval-set-1 and the terminal path are its own. What this module supplies is the
PLAN (`load_plan`) and the initialization's restore:

* the study is H1-PA's frozen record with ONE arm, Mix-Augmented, which is the
  control arm D4-Augmented copied whole -- dataset, recipe, schedule -- with
  only its initialization replaced by the MC1 Top-1 (`mixed_recovery.json`);
* the preregistration, seeds and evaluation are the mixed study's
  (`plans/mixed_preregistration.json`, whose evaluation section is H1-PA's);
* each unit's derived recipe must hash to the recipe identity the D4-Augmented
  unit of the same seed recorded (`control_recipe_sha256`), and its schedule
  must be D4-Augmented's; otherwise nothing trains;
* the Top-1 is staged by setup from the relay by sha256 and re-identified here
  from its bytes (`restore_staged`); a mismatch stops the unit, preserved.

The authorization (`mixed_governance`, kind ``recovery``) is re-checked on the
pod against the identity and the harness before any unit runs.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import h1pa_driver as HD  # noqa: E402
from stages.reasoning_data import unit_driver as UD  # noqa: E402

STUDY = REPO / "configs/stages/stage-0/calibration_volume/mixed_recovery.json"
PREREG = REPO / "logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json"


def load_plan(study_path: Path = STUDY, prereg_path: Path = PREREG) -> dict:
    """`h1pa_driver`'s plan shape for the one Mix-Augmented arm."""
    st, prereg = UD._read(study_path), UD._read(prereg_path)
    base_path = REPO / st["base_study"]["path"]
    if UD._sha(base_path) != st["base_study"]["sha256"]:
        raise HD.DriverRefusal(f"{st['base_study']['path']} is not the H1-PA record the study names")
    if prereg["recovery"]["study"]["sha256"] != st["base_study"]["sha256"]:
        raise HD.DriverRefusal("the preregistration's recovery study is not the base study")
    base = UD._read(base_path)
    arm, ctl = st["arm"], st["control_arm"]
    if [int(s) for s in prereg["seeds"]] != [int(s) for s in base["seeds"]]:
        raise HD.DriverRefusal(f"the preregistered seeds {prereg['seeds']} are not D4-Augmented's {base['seeds']}")
    study = copy.deepcopy(base)
    study["study_id"] = st["study_id"]
    study["arms"] = {arm: {**copy.deepcopy(base["arms"][ctl]), "initialization": "mix_top1",
                           "execution": f"{ctl}'s training on {ctl}'s set from the MC1 Top-1"}}
    study["initializations"] = {"mix_top1": {"record": st["initialization"]["record"]}}
    study["derived_schedule_arms"] = [arm]
    recipes = {}
    for seed, c in base["recipe"]["e1_configs"].items():
        cfg = UD._read(REPO / c["path"])
        if HD.canonical_sha(cfg) != c["canonical_sha256"]:
            raise HD.DriverRefusal(f"{c['path']}: not the E1 config the study records")
        recipes[int(seed)] = cfg
    rec = REPO / st["initialization"]["record"]
    units = [{"unit": f"{arm}-s{seed}", "arm": arm, "seed": int(seed), "execution": "train"}
             for seed in prereg["seeds"]]
    return {"study": study, "study_sha256": UD._sha(study_path), "prereg": prereg,
            "prereg_sha256": UD._sha(prereg_path), "recipes": recipes,
            "init": UD._read(rec)["materialization_identity"], "init_record_sha256": UD._sha(rec),
            "units": units, "mixed": st}


def control_refusals(plan: dict, pack_root: Path) -> list[str]:
    """Each unit's recipe identity and schedule against D4-Augmented's own records."""
    out, st = [], plan["mixed"]
    for u in plan["units"]:
        try:
            sched = HD.schedule_for(plan, u["arm"], pack_root)
        except HD.DriverRefusal as exc:
            out.append(f"{u['unit']}: {exc}")
            continue
        if {k: sched.get(k) for k in st["control_schedule"]} != st["control_schedule"]:
            out.append(f"{u['unit']}: schedule {sched} is not D4-Augmented's {st['control_schedule']}")
        probe = HD.derive_recipe(plan, u, sched, run_name="-", purpose="-", out_dir="-", data_dir="-",
                                 student_path="-")
        got, want = HD.recipe_identity(probe), st["control_recipe_sha256"][str(u["seed"])]
        if got != want:
            out.append(f"{u['unit']}: recipe identity {got[:12]} is not D4-Augmented-s{u['seed']}'s {want[:12]}")
    return out


def restore_staged(init_dir: Path, ident: dict) -> Path:
    """The MC1 Top-1 as setup staged it from the relay, re-identified from its bytes."""
    if not (Path(init_dir).is_dir() and UD._verified(Path(init_dir), ident, UD._qwen3_adapter())):
        raise UD.IdentityMismatch(f"{init_dir}: the staged MC1 Top-1 does not re-identify as "
                                  f"{ident['artifact_digest'][:12]}; preserved, nothing trained")
    return Path(init_dir)


OPS = dataclasses.replace(HD.PRODUCTION, restore_init=restore_staged)


def main(argv=None) -> int:
    from stages.calibration_volume import mixed_governance as MG
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--pack-root", required=True)
    args, rest = ap.parse_known_args(argv)
    auth = MG.load_for("recovery", args.auth)
    auth.require_plan(MG.identity_hash("recovery", REPO))
    auth.require_harness(REPO)
    plan = load_plan()
    bad = control_refusals(plan, Path(args.pack_root))
    if bad:
        print("REFUSED: " + "; ".join(bad), flush=True)
        return 2
    return HD.run(rest + ["--pack-root", args.pack_root], ops=OPS, plan=plan)


if __name__ == "__main__":
    raise SystemExit(main())

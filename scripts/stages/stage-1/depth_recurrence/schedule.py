#!/usr/bin/env python3
"""DMR1 variant 2: the execution schedule, the recurrent config and each loop arm's recurrent init record ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/depth_recurrence/schedule.py \\
        {schedule | config [--write] | init-record --arm <depth_loop_kept|depth_loop_merged> [--write]}

* ``schedule`` -- the 36-logical-layer schedule from SOB1's committed depth mapping (the spans of the kept blocks,
  `recurrent_depth.schedule_from_spans`), checked against the design's copy.
* ``config`` -- ``recurrent_config.json``: the base's ``config.json`` (``base_config.json``, a byte copy of the file
  SOB1's depth init and variant 1's build both save; its sha256 is the design's) through
  `recurrent_depth.recurrent_config_from`. ``--write`` writes it; without, it checks the committed file.
* ``init-record`` -- the recurrent init record of one loop arm over the base the design FIXES for it
  (``arms.<arm>.base``: KEPT = SOB1's depth init, MERGED = variant 1's COMPLETE build record), written to
  ``inits/<KEPT|MERGED>/init_record.json``: the base's weights and relay copy, the recurrent config, the schedule. Its
  identity is its weights AND its config (``checkpoint.artifact_digest``, `execution_digest`).

There is no selection (design ``superseded_2026_11_28``: the 2026-11-26 rule that picked one base from comparison 1 is
kept there as history, and its code in git). `init_record_problems` is what `stages.depth_merge.plan` asks before
deriving a loop arm's units: the arm's fixed base, that base's weights and verified relay copy, the committed
recurrent config, the mapping's schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.depth_merge import common as DC  # noqa: E402

SCHEMA = "aadistill.stage1.depth_recurrence.init_record/v2"
BASE_CONFIG = REPO / "configs/stages/stage-1/depth_recurrence/base_config.json"
RECURRENT_CONFIG = DC.RECURRENT_CONFIG
INITS = DC.RECURRENCE_LOGS / "inits"
PATHS = ("KEPT", "MERGED")


class ScheduleError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def rel(p: Path, root: Path = REPO) -> str:
    return str(Path(p).resolve().relative_to(Path(root).resolve()))


# ---- the schedule and the config ----------------------------------------------------------------------------------

def schedule(d: dict | None = None) -> dict:
    """The schedule from the mapping (variant 1's `merge.all_spans`: SOB1's committed kept blocks), checked against
    the design's copy."""
    from aadistill.initialization.adapters.recurrent_depth import schedule_from_spans
    from stages.depth_merge import merge as M
    d = d or DC.recurrence_design()
    spans = M.all_spans()
    got = schedule_from_spans(spans)
    want = d["schedule"]
    if (got["layer_to_block"] != want["layer_to_block"] or got["root_layer_of_layer"] != want["root_layer_of_layer"]
            or [list(s) for s in spans] != want["spans"]):
        raise ScheduleError("the mapping's schedule is not the design's")
    return {**got, "spans": [list(s) for s in spans], "n_logical_layers": len(got["layer_to_block"]),
            "n_parameter_blocks": len(spans)}


def config_text(base_text: str | None = None) -> str:
    """The recurrent config's exact text: the base config through `recurrent_config_from`, two-space JSON + newline."""
    from aadistill.initialization.adapters.recurrent_depth import recurrent_config_from
    d = DC.recurrence_design()
    base_text = BASE_CONFIG.read_text() if base_text is None else base_text
    if hashlib.sha256(base_text.encode()).hexdigest() != d["base"]["kept_init"]["config_json_sha256"]:
        raise ScheduleError("base_config.json is not the base's config file")
    cfg = recurrent_config_from(json.loads(base_text), schedule(d)["layer_to_block"])
    return json.dumps(cfg, indent=2, sort_keys=True) + "\n"


def config_problems() -> list[str]:
    if not RECURRENT_CONFIG.is_file():
        return [f"{rel(RECURRENT_CONFIG)} is not written"]
    return [] if RECURRENT_CONFIG.read_text() == config_text() else [f"{rel(RECURRENT_CONFIG)} is not the derivation"]


def execution_digest(weights_sha256: str, config_sha256: str) -> str:
    """A recurrent initialization's identity: its weights AND its config (the schedule is in the config)."""
    return hashlib.sha256(json.dumps({"config_json_sha256": config_sha256, "model_safetensors_sha256": weights_sha256},
                                     sort_keys=True).encode()).hexdigest()


# ---- each arm's fixed base -----------------------------------------------------------------------------------------

def loop_arms(d: dict | None = None) -> tuple[str, ...]:
    """The design's loop arms that have a fixed base this round (``arms.<arm>.base``)."""
    d = d or DC.recurrence_design()
    return tuple(a for a, v in d["arms"].items() if v.get("base") in PATHS)


def base_of(arm: str, d: dict | None = None) -> str:
    """The base the design fixes for ``arm`` (KEPT or MERGED); refused for an arm without one (variant 3)."""
    d = d or DC.recurrence_design()
    b = (d["arms"].get(arm) or {}).get("base")
    if b not in PATHS:
        raise ScheduleError(f"{arm!r} has no fixed base in the design ({(d['arms'].get(arm) or {}).get('status')})")
    return b


def legal_path(choice: str, root: Path = REPO) -> dict:
    """The base init record and control of a path: KEPT = SOB1's depth init (the design's pin); MERGED = variant 1's
    COMPLETE build record (its training plan's init record, which names it by sha256)."""
    d = DC.recurrence_design()
    if choice == "KEPT":
        k = d["base"]["kept_init"]
        return {"base_init_record": {"path": k["path"], "sha256": k["sha256"]}, "control_arm": DC.SOB1_DEPTH}
    if choice == "MERGED":
        from stages.depth_merge import plan as PL
        p = PL.out_path("depth_merge")
        if not p.is_file():
            raise ScheduleError("no committed depth_merge training plan: variant 1's init record is not bound")
        ir = read_json(p)["init_record"]
        return {"base_init_record": {"path": ir["path"], "sha256": ir["sha256"]}, "control_arm": "depth_merge"}
    raise ScheduleError(f"{choice!r} is not one of {PATHS}")


# ---- the recurrent init record ----------------------------------------------------------------------------------

def init_record(arm: str, root: Path = REPO) -> dict:
    """The recurrent init record of ``arm`` over its fixed base (module docstring)."""
    path = base_of(arm)
    base_ref = legal_path(path, root)["base_init_record"]
    bp = root / base_ref["path"]
    if sha256_file(bp) != base_ref["sha256"]:
        raise ScheduleError(f"{base_ref['path']} is not the base the {path} path names")
    base = read_json(bp)
    bck = base["checkpoint"]
    w = bck["shards"][0]["sha256"]
    if bck["files_sha256"]["config.json"] != DC.recurrence_design()["base"]["kept_init"]["config_json_sha256"]:
        raise ScheduleError("the base's config file is not the one the recurrent config is derived from")
    bad = config_problems()
    if bad:
        raise ScheduleError("; ".join(bad))
    csha = sha256_file(RECURRENT_CONFIG)
    sch = schedule()
    digest = execution_digest(w, csha)
    return {"schema": SCHEMA, "status": "COMPLETE", "arms": [arm], "path": path,
            "root_teacher": base["root_teacher"],
            "commit": {"selected_rows": [{"state_id": f"recurrent_{path.lower()}/{digest[:32]}",
                                          "path": f"the {path} base executed on the 36-layer schedule",
                                          "artifact_digest": digest, "single_shard_sha256": w}]},
            "checkpoint": {"artifact_digest": digest, "artifact_digest_rule": "execution_digest: sha256 over the "
                           "weights' and the config file's sha256 (identify_checkpoint's digest is the base's)",
                           "shards": [{"filename": "model.safetensors", "sha256": w,
                                       "size_bytes": bck["shards"][0]["size_bytes"]}],
                           "files_sha256": {"model.safetensors": w, "config.json": csha},
                           "num_parameters": int(bck.get("num_parameters_counted") or bck["num_parameters"]),
                           "base_artifact_digest": bck["artifact_digest"]},
            "execution": {**{k: sch[k] for k in ("layer_to_block", "root_layer_of_layer", "n_logical_layers",
                                                 "n_parameter_blocks", "spans")},
                          "config": {"path": rel(RECURRENT_CONFIG), "sha256": csha}},
            "base": {"init_record": base_ref, "arm": base.get("arm"), "artifact_digest": bck["artifact_digest"]},
            "fixed_by": f"configs/stages/stage-1/depth_recurrence/design.json :: arms.{arm}.base = {path}",
            "relay": {"init": base["relay"]["init"]},
            "staging": "the pod fetches the base's relay copy (its weights and config, re-hashed) and writes the "
                       "recurrent config over the config (depth_merge.train_driver.stage_init)"}


def init_record_path(choice: str) -> Path:
    return INITS / choice / "init_record.json"


def init_record_problems(rec: dict, *, arm: str, root: Path = REPO) -> list[str]:
    """Why ``rec`` is not a legal recurrent initialization of ``arm`` (empty when it is): COMPLETE, for this arm, over
    the base the design fixes for it, the base's weights and verified relay copy, the committed recurrent config, the
    mapping's schedule."""
    bad = []
    if rec.get("schema") != SCHEMA or rec.get("status") != "COMPLETE" or rec.get("arms") != [arm]:
        return [f"not a COMPLETE recurrent init record of {arm}"]
    try:
        path = base_of(arm)
        want = legal_path(path, root)["base_init_record"]
    except ScheduleError as exc:
        return [str(exc)]
    if rec.get("path") != path:
        bad.append(f"the record is over the {rec.get('path')} base; the design fixes {path} for {arm}")
    if rec["base"]["init_record"] != want:
        bad.append(f"the base {rec['base']['init_record']} is not the {path} path's {want}")
    bp = root / want["path"]
    if not bp.is_file() or sha256_file(bp) != want["sha256"]:
        return bad + ["the base init record is not here as named"]
    base = read_json(bp)
    if rec["checkpoint"]["files_sha256"]["model.safetensors"] != base["checkpoint"]["shards"][0]["sha256"]:
        bad.append("the weights are not the base's")
    if rec["relay"]["init"] != base["relay"]["init"]:
        bad.append("the relay copy is not the base's")
    if not (base["relay"]["init"] or {}).get("verified"):
        bad.append("the base's relay copy is not verified")
    bad += config_problems()
    if RECURRENT_CONFIG.is_file() and rec["checkpoint"]["files_sha256"]["config.json"] != sha256_file(RECURRENT_CONFIG):
        bad.append("the config is not the committed recurrent config")
    try:
        sch = schedule()
        if rec["execution"]["layer_to_block"] != sch["layer_to_block"]:
            bad.append("the schedule is not the mapping's")
    except ScheduleError as exc:
        bad.append(str(exc))
    want_digest = execution_digest(rec["checkpoint"]["files_sha256"]["model.safetensors"],
                                   rec["checkpoint"]["files_sha256"]["config.json"])
    if rec["checkpoint"]["artifact_digest"] != want_digest:
        bad.append("the artifact digest is not the execution digest of its weights and config")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schedule")
    c = sub.add_parser("config")
    c.add_argument("--write", action="store_true")
    i = sub.add_parser("init-record")
    i.add_argument("--arm", required=True, choices=loop_arms())
    i.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "schedule":
        print(json.dumps(schedule()))
        return 0
    if a.cmd == "config":
        if a.write:
            RECURRENT_CONFIG.write_text(config_text())
            print(rel(RECURRENT_CONFIG), sha256_file(RECURRENT_CONFIG))
            return 0
        bad = config_problems()
        print("the recurrent config is the derivation" if not bad else f"REFUSED: {bad}")
        return 0 if not bad else 1
    doc = init_record(a.arm)
    p = init_record_path(doc["path"])
    if a.write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"path": rel(p), "artifact_digest": doc["checkpoint"]["artifact_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

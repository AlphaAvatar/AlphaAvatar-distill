#!/usr/bin/env python3
"""DMR1 variants 2 and 3: the execution schedule, the recurrent config, the initialization selection and the recurrent
init record ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-6/depth_recurrence/schedule.py \\
        {schedule | config [--write] | select [--comparison <record>] [--write] | init-record [--write]}

* ``schedule`` -- the 36-logical-layer schedule from SOB1's committed depth mapping (the spans of the kept blocks,
  `recurrent_depth.schedule_from_spans`), checked against the design's copy.
* ``config`` -- ``recurrent_config.json``: the base's ``config.json`` (``base_config.json``, a byte copy of the file
  SOB1's depth init and variant 1's build both save; its sha256 is the design's) through
  `recurrent_depth.recurrent_config_from`. ``--write`` writes it; without, it checks the committed file.
* ``select`` -- the initialization selection (design ``initialization_selection``) from comparison 1's analysis
  record, written to ``plans/init_selection.json``; refused while comparison 1 is incomplete.
* ``init-record`` -- the recurrent init record over the SELECTED base (``inits/<KEPT|MERGED>/init_record.json``):
  the base's weights and relay copy, the recurrent config, the schedule, the selection that chose it. Its identity is
  its weights AND its config (``checkpoint.artifact_digest``, `execution_digest`).

`init_record_problems` is what `stages.depth_merge.plan` asks before deriving a recurrent arm's units: only the two
legal paths (KEPT: SOB1's depth init; MERGED: variant 1's COMPLETE build record), the committed selection naming that
path, the recurrent config of that base, the schedule of the mapping.
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

SCHEMA = "aadistill.stage6.depth_recurrence.init_record/v1"
SELECTION_SCHEMA = "aadistill.stage6.depth_recurrence.init_selection/v1"
BASE_CONFIG = REPO / "configs/stages/stage-6/depth_recurrence/base_config.json"
RECURRENT_CONFIG = DC.RECURRENT_CONFIG
SELECTION = DC.STAGE6_LOGS / "plans/init_selection.json"
INITS = DC.STAGE6_LOGS / "inits"
ARMS = ("depth_loop", "depth_loop_hidden")
PATHS = ("KEPT", "MERGED")
COMPARISON_1 = DC.LOGS / "results/comparison_1.json"


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
    d = d or DC.stage6_design()
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
    d = DC.stage6_design()
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


# ---- the selection --------------------------------------------------------------------------------------------------

def decide(c1: dict) -> dict:
    """The design's rule (``initialization_selection``) on comparison 1's record: MERGED iff the thinking-mode primary
    units are complete AND the pooled delta > 0 AND each seed's delta > 0; otherwise KEPT. Refused (None) when
    comparison 1 is incomplete. The p-value is never read."""
    ht = (c1.get("modes") or {}).get("H-T") or {}
    if not ht.get("complete"):
        return {"status": "REFUSED", "why": "comparison 1's thinking-mode primary units are not complete: no "
                                            "selection; the maintainer decides"}
    pooled = float(ht["delta"])
    per_seed = {s: float(v["delta"]) for s, v in ht["per_seed"].items()}
    merged = pooled > 0 and all(x > 0 for x in per_seed.values())
    why = ("pooled and every seed's delta > 0" if merged else
           "pooled delta <= 0" if pooled <= 0 else "the seeds' deltas do not all exceed 0")
    return {"status": "COMPLETE", "choice": "MERGED" if merged else "KEPT", "why": why,
            "pooled_delta": pooled, "per_seed_delta": per_seed}


def legal_path(choice: str, root: Path = REPO) -> dict:
    """The base init record and control of a path: KEPT = SOB1's depth init (the design's pin); MERGED = variant 1's
    COMPLETE build record (its training plan's init record, which names it by sha256)."""
    d = DC.stage6_design()
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


def selection(comparison: Path = COMPARISON_1, root: Path = REPO) -> dict:
    c1 = read_json(comparison)
    dec = decide(c1)
    doc = {"schema": SELECTION_SCHEMA, **dec, "rule": DC.stage6_design()["initialization_selection"]["rule"],
           "comparison_1": {"path": rel(comparison, root), "sha256": sha256_file(comparison)},
           "role": "an engineering choice of the training-pre initialization, not a statistical claim"}
    if dec["status"] == "COMPLETE":
        doc["path"] = legal_path(dec["choice"], root)
    return doc


# ---- the recurrent init record ----------------------------------------------------------------------------------

def init_record(sel_path: Path = SELECTION, root: Path = REPO) -> dict:
    """The recurrent init record over the selection's base (module docstring)."""
    sel = read_json(sel_path)
    if sel.get("status") != "COMPLETE" or sel.get("choice") not in PATHS:
        raise ScheduleError(f"{rel(sel_path, root)} is not a COMPLETE selection")
    base_ref = legal_path(sel["choice"], root)["base_init_record"]
    if base_ref != sel["path"]["base_init_record"]:
        raise ScheduleError("the selection names another base than its path's")
    bp = root / base_ref["path"]
    if sha256_file(bp) != base_ref["sha256"]:
        raise ScheduleError(f"{base_ref['path']} is not the base the selection names")
    base = read_json(bp)
    bck = base["checkpoint"]
    w = bck["shards"][0]["sha256"]
    if bck["files_sha256"]["config.json"] != DC.stage6_design()["base"]["kept_init"]["config_json_sha256"]:
        raise ScheduleError("the base's config file is not the one the recurrent config is derived from")
    bad = config_problems()
    if bad:
        raise ScheduleError("; ".join(bad))
    csha = sha256_file(RECURRENT_CONFIG)
    sch = schedule()
    digest = execution_digest(w, csha)
    return {"schema": SCHEMA, "status": "COMPLETE", "arms": list(ARMS), "path": sel["choice"],
            "root_teacher": base["root_teacher"],
            "commit": {"selected_rows": [{"state_id": f"recurrent_{sel['choice'].lower()}/{digest[:32]}",
                                          "path": f"the {sel['choice']} base executed on the 36-layer schedule",
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
            "selection": {"path": rel(sel_path, root), "sha256": sha256_file(sel_path), "choice": sel["choice"]},
            "relay": {"init": base["relay"]["init"]},
            "staging": "the pod fetches the base's relay copy (its weights and config, re-hashed) and writes the "
                       "recurrent config over the config (depth_merge.train_driver.stage_init)"}


def init_record_path(choice: str) -> Path:
    return INITS / choice / "init_record.json"


def init_record_problems(rec: dict, *, arm: str, root: Path = REPO) -> list[str]:
    """Why ``rec`` is not a legal recurrent initialization of ``arm`` (empty when it is): COMPLETE, for this arm, the
    committed selection's path and base, the base's weights and relay copy, the committed recurrent config, the
    mapping's schedule."""
    bad = []
    if rec.get("schema") != SCHEMA or rec.get("status") != "COMPLETE" or arm not in (rec.get("arms") or []):
        return [f"not a COMPLETE recurrent init record of {arm}"]
    sp = root / rec["selection"]["path"]
    if not sp.is_file() or sha256_file(sp) != rec["selection"]["sha256"]:
        return ["its selection record is not the committed one"]
    sel = read_json(sp)
    if sel.get("status") != "COMPLETE" or sel.get("choice") != rec.get("path"):
        bad.append("the selection is not COMPLETE or names another path")
    try:
        want = legal_path(rec.get("path"), root)["base_init_record"]
    except ScheduleError as exc:
        return bad + [str(exc)]
    if rec["base"]["init_record"] != want:
        bad.append(f"the base {rec['base']['init_record']} is not the {rec.get('path')} path's {want}")
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
    s = sub.add_parser("select")
    s.add_argument("--comparison", default=str(COMPARISON_1))
    s.add_argument("--write", action="store_true")
    i = sub.add_parser("init-record")
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
    if a.cmd == "select":
        doc = selection(Path(a.comparison))
        if a.write and doc["status"] == "COMPLETE":
            SELECTION.parent.mkdir(parents=True, exist_ok=True)
            SELECTION.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(doc, indent=1))
        return 0 if doc["status"] == "COMPLETE" else 12
    doc = init_record()
    p = init_record_path(doc["path"])
    if a.write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"path": rel(p), "artifact_digest": doc["checkpoint"]["artifact_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive a training event stream from a driver console log.

**This does not recreate the original artifact and must never be described as
one.** E6b's `train_log.jsonl` for both arms was destroyed at teardown. What
survived is the driver's console log, which carries the printed subset of each
event at the precision the print used. Parsing it back yields a *derived*
artifact — useful for re-plotting a curve, unusable as a record of the fields
that were never printed.

The output therefore carries its provenance in the file:

    {"provenance": "reconstructed_from_driver_console",
     "original_event_stream_available": false, ...}

and a `field_provenance` block that states, per field, whether it is an exact
extraction, a value derived from the run config, or unrecoverable. A consumer
that reads `loss` gets four decimals because the console printed four; the
stream held six. A consumer that wants `grad_norm` gets nothing, because it was
never printed at all.

    python3 scripts/pod/reconstruct_training_events.py \\
        --run-log /home/ecs-user/aad-artifacts/e6b/e6b_run.log \\
        --status  /home/ecs-user/aad-artifacts/e6b/e6b.status \\
        --config configs/stage3/e6b/e6b_p2_r2960k_sa.json \\
        --config configs/stage3/e6b/e6b_p2_r2960k_sb.json \\
        --out logs/e6b_reconstructed_training_events.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

# `step 250/2916  loss 2.1018  ce 1.7693  kd 1.3300  lr 4.98e-05  3.78s`
STEP_RE = re.compile(
    r"^step (?P<step>\d+)/(?P<total>\d+)\s+loss (?P<loss>[-\d.]+)"
    r"(?:\s+ce (?P<ce>[-\d.]+))?(?:\s+kd (?P<kd>[-\d.]+))?"
    r"\s+lr (?P<lr>[-\d.eE+]+)\s+(?P<seconds>[\d.]+)s\s*$")
# `eval step 364: {'val_blocks': 16, 'val_ce': 1.915578, ...}`
EVAL_RE = re.compile(r"^eval step (?P<step>\d+)(?P<suffix>[^:]*): (?P<body>\{.*\})\s*$")
# `[17:25:51] $ /opt/train/bin/python scripts/training/train_stage3.py --config …/NAME.json`
CMD_RE = re.compile(r"^\[(?P<clock>[\d:]+)\] \$ .*train_stage3\.py --config "
                    r"(?P<config>\S+)")
MARKER_RE = re.compile(r"^(?P<ts>\S+) MARKER:(?P<marker>.+)$")

FIELD_PROVENANCE = {
    "exact": {
        "step": "printed in full",
        "seconds": "printed from the already-rounded stored value",
        "val_blocks": "printed as part of the eval dict repr",
        "val_ce": "printed as part of the eval dict repr",
        "val_ppl": "printed as part of the eval dict repr",
        "val_kd": "printed as part of the eval dict repr",
    },
    "truncated": {
        "loss": "console printed %.4f; the stream stored 6 decimals",
        "ce": "console printed %.4f; the stream stored 6 decimals",
        "kd": "console printed %.4f; the stream stored 6 decimals",
        "lr": "console printed %.2e; the stream stored the full float",
    },
    "derived_from_config": {
        "tokens_seen": "step * batch.blocks_per_step * block_len, recomputed "
                       "here from the run config rather than read from the log",
        "run_name": "taken from the config the driver invoked",
    },
    "bounded_only": {
        "time": "no per-line timestamp was printed. Each arm's events lie "
                "between its driver command line and its TRAIN_DONE marker; "
                "those two bounds are recorded per arm and no per-event "
                "timestamp is invented",
    },
    "unrecoverable": {
        "grad_norm": "never printed",
        "ce_targets": "never printed",
        "kd_positions": "never printed",
        "logical_block_tokens": "never printed",
        "executed_positions": "never printed",
        "executed_nonpad_tokens": "never printed",
        "supervised_tokens": "never printed",
        "truncate_padding": "never printed",
        "gpu_mem_gb": "never printed",
        "run_start": "the event and its freeze report were never printed",
        "dataset_loaded": "tokenizer hash and token mix were never printed",
        "teacher_loaded": "teacher identity was never printed",
        "student_loaded": "parameter count was never printed",
        "checkpoint_saved": "never printed",
        "run_end": "total seconds were never printed",
    },
}


def parse(run_log: Path, configs: list[Path], status: Path | None) -> dict:
    cfgs = {}
    for p in configs:
        c = json.loads(p.read_text())
        cfgs[Path(c["out_dir"]).name] = {"path": str(p), "config": c}

    markers = []
    if status and status.is_file():
        for line in status.read_text().splitlines():
            m = MARKER_RE.match(line.strip())
            if m:
                markers.append({"time": m["ts"], "marker": m["marker"]})

    arms: list[dict] = []
    current: dict | None = None
    for raw in run_log.read_text(errors="replace").splitlines():
        line = raw.strip()

        cmd = CMD_RE.match(line)
        if cmd:
            name = Path(cmd["config"]).stem
            entry = cfgs.get(name)
            current = {
                "run_name": name,
                "config_path": entry["path"] if entry else None,
                "config": entry["config"] if entry else None,
                "driver_command_clock": cmd["clock"],
                "events": [],
            }
            arms.append(current)
            continue
        if current is None:
            continue

        step = STEP_RE.match(line)
        if step:
            cfg = current["config"] or {}
            bps = (cfg.get("batch") or {}).get("blocks_per_step")
            block_len = cfg.get("block_len")
            event = {
                "event": "train_step",
                "step": int(step["step"]),
                "loss": float(step["loss"]),
                "lr": float(step["lr"]),
                "seconds": float(step["seconds"]),
            }
            if step["ce"] is not None:
                event["ce"] = float(step["ce"])
            if step["kd"] is not None:
                event["kd"] = float(step["kd"])
            if bps and block_len:
                event["tokens_seen"] = int(step["step"]) * bps * block_len
            current["events"].append(event)
            continue

        ev = EVAL_RE.match(line)
        if ev:
            try:
                body = ast.literal_eval(ev["body"])
            except (ValueError, SyntaxError):
                continue
            current["events"].append({
                "event": "eval_result",
                "step": int(ev["step"]),
                "val_set": (ev["suffix"].strip(" []()") or "val"),
                **body,
            })

    # The driver labels its markers with the experiment's arm names
    # (`TRAIN_DONE:P2-2.96M-sa`) while the configs are named after their output
    # directories (`e6b_p2_r2960k_sa`). The two vocabularies do not match, so
    # the association is positional — arms and TRAIN_DONE markers both appear in
    # execution order — and the raw label is carried through so a reader can
    # check the pairing rather than take it on trust.
    train_done = [m for m in markers if m["marker"].startswith("TRAIN_DONE:")]
    for i, arm in enumerate(arms):
        steps = [e for e in arm["events"] if e["event"] == "train_step"]
        evals = [e for e in arm["events"] if e["event"] == "eval_result"]
        exact = next((m for m in train_done
                      if m["marker"] == f"TRAIN_DONE:{arm['run_name']}"), None)
        done = exact or (train_done[i] if i < len(train_done) else None)
        arm["counts"] = {"train_step": len(steps), "eval_result": len(evals)}
        arm["step_seconds"] = (
            {"n": len(steps),
             "mean": round(sum(e["seconds"] for e in steps) / len(steps), 4),
             "min": min(e["seconds"] for e in steps),
             "max": max(e["seconds"] for e in steps)}
            if steps else None)
        arm["time_bounds"] = {
            "driver_command_clock_utc": arm["driver_command_clock"],
            "train_done_marker": done["marker"] if done else None,
            "train_done_marker_utc": done["time"] if done else None,
            "marker_association": ("exact run_name match" if exact
                                   else "positional" if done else "none"),
            "note": "per-event timestamps were not printed; every event in "
                    "this arm falls between these two bounds",
        }
        if done and arm["driver_command_clock"] and steps:
            # Wall-clock seconds per step, which is a different quantity from
            # the printed per-step timing: it also carries the periodic evals
            # and checkpoint writes. Both are reported; neither is presented as
            # the other.
            start = datetime.strptime(arm["driver_command_clock"], "%H:%M:%S")
            end = datetime.fromisoformat(done["time"].replace("Z", "+00:00"))
            wall = (end.hour * 3600 + end.minute * 60 + end.second
                    - start.hour * 3600 - start.minute * 60 - start.second)
            wall = wall + 86400 if wall < 0 else wall     # crossed midnight UTC
            total = (arm["config"] or {}).get("schedule", {}).get("total_steps")
            arm["wall_clock"] = {
                "seconds": wall,
                "minutes": round(wall / 60, 1),
                "seconds_per_step_including_eval_and_checkpoint":
                    round(wall / total, 3) if total else None,
                "note": "measured between the driver command line and the "
                        "TRAIN_DONE marker; larger than the printed per-step "
                        "timing because it includes evals and checkpoints",
            }
        arm["final_eval"] = evals[-1] if evals else None

    return {
        "provenance": "reconstructed_from_driver_console",
        "original_event_stream_available": False,
        "original_event_stream_paths": [
            f"{(c['config'] or {}).get('out_dir', '?')}/train_log.jsonl"
            for c in cfgs.values()],
        "original_loss_note":
            "The per-arm train_log.jsonl and run_manifest.json were on the pod "
            "only and were not included in the teardown bundle. They are gone; "
            "this file is derived from the driver console log and is not a "
            "substitute for them.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/pod/reconstruct_training_events.py",
        "source": {
            "run_log": str(run_log),
            "run_log_sha256": sha256_file(run_log),
            "run_log_bytes": run_log.stat().st_size,
            "status": str(status) if status and status.is_file() else None,
            "status_sha256": (sha256_file(status)
                              if status and status.is_file() else None),
        },
        "field_provenance": FIELD_PROVENANCE,
        "markers": markers,
        "arms": arms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-log", required=True, type=Path)
    ap.add_argument("--status", type=Path, default=None)
    ap.add_argument("--config", action="append", type=Path, default=[])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    doc = parse(args.run_log, args.config, args.status)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")

    for arm in doc["arms"]:
        print(f"{arm['run_name']}: {arm['counts']['train_step']} train_step, "
              f"{arm['counts']['eval_result']} eval_result, "
              f"mean {arm['step_seconds']['mean'] if arm['step_seconds'] else '?'}"
              f" s/step")
    print(f"-> {args.out} ({args.out.stat().st_size} bytes) — derived artifact, "
          "not the original event stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Decide which of a run's checkpoints to keep, and prove the rest are expendable.

Experiment 1 kept only the final checkpoint (`keep_last: 1`), so no run in it has
a held-out-NLL trajectory and none of its intermediate states can be re-scored.
Experiment 2 evaluates and checkpoints at every eval point instead — but nine
checkpoints per arm at 4.3 GB each (2.3 GB weights + 2.0 GB optimizer state) is
39 GB per arm, which is not something to move off a paid pod or keep forever.

This resolves that: **every eval point keeps its metrics and generations, and
only decision-relevant steps keep weights.** The keep set is derived from the
run's own log rather than fixed in advance:

* the **final** step — the fixed-compute endpoint every arm is compared at;
* the **best validation-CE** step;
* the **best held-out-NLL** step;
* the two steps **bracketing the onset of sustained held-out-NLL deterioration**
  — the first step after which NLL rises for `--sustained` consecutive
  evaluations, and the evaluation before it.

Optimizer state is dropped from every kept checkpoint except the latest: it
exists to resume a run, not to analyse one, and it is 47% of the bytes.

The result is a keep-list, a hash manifest and a plan a transfer step can act
on. Nothing is deleted unless `--apply` is passed, and the final step can never
be pruned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402


def read_trajectory(train_log: Path, holdout_log: Path | None = None) -> list[dict]:
    """Eval points from a run's append-only log, in step order.

    Validation CE comes from the trainer's own `eval_result` events. Held-out NLL
    does **not**: it is a different scoring path (`scripts/evaluation/eval_ppl.py`
    over `holdout_v1`, not packed teacher-native blocks), and adding it to the
    training loop would change the trainer that produced the control. The
    orchestrator scores each saved checkpoint instead and appends
    `{"step": N, "holdout_nll": x}` to `holdout_trajectory.jsonl`, which is
    merged here by step.
    """
    points: dict[int, dict] = {}
    with train_log.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A bare string or number is valid JSON but not a log event; the log
            # is append-only and shared with shell-side writers, so a stray line
            # must be skipped rather than crash a teardown-time tool.
            if not isinstance(row, dict):
                continue
            if row.get("event") != "eval_result":
                continue
            if row.get("val_set", "val") != "val":
                continue
            step = int(row["step"])
            points[step] = {
                "step": step,
                "val_ce": row.get("val_ce"),
                "val_kd": row.get("val_kd"),
                "holdout_nll": row.get("holdout_nll"),
            }

    if holdout_log is None:
        holdout_log = train_log.parent / "holdout_trajectory.jsonl"
    if holdout_log.is_file():
        with holdout_log.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or "step" not in row:
                    continue
                step = int(row["step"])
                points.setdefault(step, {"step": step, "val_ce": None,
                                         "val_kd": None, "holdout_nll": None})
                if row.get("holdout_nll") is not None:
                    points[step]["holdout_nll"] = row["holdout_nll"]
    return [points[s] for s in sorted(points)]


def deterioration_onset(points: list[dict], sustained: int = 2) -> int | None:
    """First step after which held-out NLL rises for `sustained` evaluations.

    Deliberately not "the first rise": a single up-tick at this noise level is
    not evidence of anything (the between-seed |Δ| on this metric is 0.489 nats
    at the 0.86M rung). Requiring consecutive rises is the cheapest defensible
    definition, and the raw trajectory is retained either way so a different
    definition can be applied later without re-running anything.
    """
    have = [p for p in points if p.get("holdout_nll") is not None]
    for i in range(len(have) - sustained):
        window = have[i:i + sustained + 1]
        if all(b["holdout_nll"] > a["holdout_nll"]
               for a, b in zip(window, window[1:])):
            return have[i]["step"]
    return None


def choose_keep(points: list[dict], sustained: int = 2) -> dict:
    """Steps whose weights are worth keeping, each with the reason."""
    if not points:
        raise SystemExit("no eval points in the log — nothing to decide from")
    reasons: dict[int, list[str]] = {}

    def mark(step, why):
        if step is None:
            return
        reasons.setdefault(int(step), []).append(why)

    mark(points[-1]["step"], "final")

    scored = [p for p in points if p.get("val_ce") is not None]
    if scored:
        mark(min(scored, key=lambda p: p["val_ce"])["step"], "best_val_ce")
    scored = [p for p in points if p.get("holdout_nll") is not None]
    if scored:
        mark(min(scored, key=lambda p: p["holdout_nll"])["step"], "best_holdout_nll")

    onset = deterioration_onset(points, sustained)
    if onset is not None:
        mark(onset, "deterioration_onset")
        after = [p["step"] for p in points if p["step"] > onset]
        if after:
            mark(after[0], "after_deterioration_onset")
    return {"keep": dict(sorted(reasons.items())), "onset": onset}


def checkpoint_dirs(run_dir: Path) -> dict[int, Path]:
    out = {}
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return out
    for path in sorted(root.glob("step_*")):
        try:
            out[int(path.name.split("_")[1])] = path
        except (IndexError, ValueError):
            continue
    return out


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="an arm's out_dir, holding checkpoints/ and train_log.jsonl")
    ap.add_argument("--holdout-log", type=Path, default=None,
                    help="jsonl of {step, holdout_nll} written by the "
                         "orchestrator (default: <run-dir>/holdout_trajectory.jsonl)")
    ap.add_argument("--sustained", type=int, default=2,
                    help="consecutive held-out-NLL rises that count as onset")
    ap.add_argument("--keep-optimizer-state", action="store_true",
                    help="keep trainer_state.pt on every retained step (default: "
                         "only on the latest, since it is for resume not analysis)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it this only reports the plan")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the retention manifest "
                         "(default: <run-dir>/retention.json)")
    args = ap.parse_args()

    train_log = args.run_dir / "train_log.jsonl"
    if not train_log.is_file():
        raise SystemExit(f"{train_log} not found")
    points = read_trajectory(train_log, args.holdout_log)
    decision = choose_keep(points, args.sustained)
    keep = decision["keep"]
    found = checkpoint_dirs(args.run_dir)
    latest = max(found) if found else None

    plan, kept_bytes, dropped_bytes = [], 0, 0
    for step, path in sorted(found.items()):
        why = keep.get(step)
        state = path / "trainer_state.pt"
        drop_state = (state.is_file() and not args.keep_optimizer_state
                      and step != latest)
        size = dir_bytes(path)
        row = {"step": step, "path": str(path), "bytes": size,
               "keep": bool(why), "reasons": why or [],
               "drop_optimizer_state": bool(why) and drop_state}
        if why:
            kept = size - (state.stat().st_size if drop_state else 0)
            kept_bytes += kept
            dropped_bytes += size - kept
        else:
            dropped_bytes += size
        plan.append(row)

    missing = [s for s in keep if s not in found]
    if missing:
        print(f"WARNING: {len(missing)} decision-relevant steps have no checkpoint "
              f"on disk: {missing}", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "run_dir": str(args.run_dir),
        "policy": ("every eval point keeps metrics and generations; weights are "
                   "kept only for the final, best-val-CE, best-held-out-NLL and "
                   "deterioration-bracketing steps; optimizer state is kept only "
                   "on the latest checkpoint"),
        "sustained_rises_for_onset": args.sustained,
        "trajectory": points,
        "deterioration_onset_step": decision["onset"],
        "keep": {str(k): v for k, v in keep.items()},
        "missing_checkpoints": missing,
        "plan": plan,
        "kept_bytes": kept_bytes,
        "dropped_bytes": dropped_bytes,
        "applied": bool(args.apply),
    }

    if args.apply:
        for row in plan:
            path = Path(row["path"])
            if not row["keep"]:
                shutil.rmtree(path, ignore_errors=True)
            elif row["drop_optimizer_state"]:
                (path / "trainer_state.pt").unlink(missing_ok=True)
        # Hash what survived, so the transfer can be verified after the pod dies.
        digests = {}
        for row in plan:
            if not row["keep"]:
                continue
            path = Path(row["path"])
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    digests[str(f.relative_to(args.run_dir))] = sha256_file(f)
        manifest["sha256"] = digests

    out = args.out or (args.run_dir / "retention.json")
    out.write_text(json.dumps(manifest, indent=1))

    print(f"{args.run_dir.name}: {len(points)} eval points, "
          f"onset {decision['onset']}")
    for step, why in keep.items():
        mark = "on disk" if step in found else "MISSING"
        print(f"  keep step {step:>6}  {','.join(why):<48s} {mark}")
    print(f"  kept {kept_bytes / 2**30:.1f} GiB, "
          f"{'dropped' if args.apply else 'would drop'} "
          f"{dropped_bytes / 2**30:.1f} GiB")
    print(f"  wrote {out}")
    if not args.apply:
        print("  (dry run — pass --apply to delete)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and score D1's ten screening probes, then advance exactly one.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_d1_behavioural_driver.py \
        --out artifacts/audit/autoinit_d1_behavioural --run-id <id> --rung screening

Ten probes: four candidates and the incumbent, at two preregistered seeds each.
The seed is the REPLICATE and the initialization is the TREATMENT, so two
probes at one seed differ only in which checkpoint they started from, and that
is what makes their difference attributable.

**THE CONTRACT IS ASSERTED BEFORE ANY WEIGHTS LOAD.** The arms, the battery
role, the recipe, the seeds and the probe schedule are derived from frozen
records and hashed into one value the authorization binds. A session measuring
a different field is refused in seconds rather than after its first probe --
and the field is not this session's to choose: the candidates come from the
maintainer's retention decision and the incumbent from the design.

**EVERY ASSET IS IMPORTED FROM ITS OWNER.** The trainer, the uncapped
evaluator, the frozen recipe, the pack and the allowed override set are
`autoinit_c1_driver`'s. A second copy of the recipe path is a second thing that
can disagree about what recovery means, and the whole premise of a paired
comparison is that recovery is identical across arms.

**DURABILITY RUNS DURING THE SESSION, NOT AT CLOSEOUT.** Each probe's result is
written and announced the moment it finishes, so the launcher's poll hook can
pull it off-pod. C1 attempt 17 trained six probes over ten hours and lost every
one because nothing left the pod until a closeout that never came. A probe that
finished is evidence whether or not the probes after it do.

**THE RANKING IS MECHANICAL AND THE SELECTION IS ONE CANDIDATE.** It reads the
frozen decision rule's endpoint, pools each arm's seeds by the declared
aggregation, applies the usable-rollout guardrail as a VETO ONLY, and advances
the best surviving candidate. No forced winner: if every candidate is vetoed
the session advances none and says so, because a selection made to avoid an
empty result is not a selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/pod", "scripts/autoinit",
               "scripts/experiments/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

SUCCESS_MARKER = "ALL_DONE"
FAILURE_MARKER = "RUN_FAILED"
CHECK_ONLY_MARKER = "CHECK_ONLY_OK"

#: The one scorer for this rung. Pinned, and it takes no battery argument.
SCORER = REPO_ROOT / "scripts/autoinit/score_d1_screening.py"


class D1BehaviouralDriverError(RuntimeError):
    """This session cannot proceed on the evidence it has."""


def mark(status_path: str, name: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(status_path, "a") as handle:
        handle.write(f"{stamp} MARKER:{name}\n")
        handle.flush()


def terminal_marker(*, check_only: bool, status: str,
                    evidence_written: bool) -> str:
    """The marker the launcher polls for. ONE predicate, ONE caller.

    `evidence_written` gates success on both paths: a session that trained ten
    probes and could not write its record has not produced what it owes.
    """
    if not evidence_written:
        return FAILURE_MARKER
    if check_only:
        return CHECK_ONLY_MARKER if status == "CHECK_ONLY_OK" else FAILURE_MARKER
    return SUCCESS_MARKER if status == "COMPLETE" else FAILURE_MARKER


def build_parser() -> argparse.ArgumentParser:
    """At module scope so the dispatch probe can build it. Every required
    argument is a STRING: the probe fills them with one."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="evidence directory, repository-relative on the pod")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rung", default="screening",
                    choices=("screening", "confirmation"))
    ap.add_argument("--status", default=None,
                    help="where the terminal marker is appended")
    ap.add_argument("--probe-train-minutes", type=float, default=60.0,
                    help="per-probe training timeout, from the priced cell")
    ap.add_argument("--probe-eval-minutes", type=float, default=40.0)
    ap.add_argument("--check-only", action="store_true",
                    help="assert the contract and resolve every symbol; "
                         "train nothing and write no probe")
    return ap


def probe_config(probe: Any, *, audit: Path, frozen_recipe: Path,
                 pack_dir: str, overrides: frozenset[str]) -> Path:
    """This probe's training config: the frozen recipe plus the allowed set.

    The override set is C1's, unchanged, and a derived config that differs
    anywhere else is REFUSED. That is the mechanical form of "identical
    recovery": the only intended difference between two probes at one seed is
    the initialization, and a config free to differ elsewhere would make a
    paired difference uninterpretable.
    """
    frozen = json.loads(frozen_recipe.read_text())
    derived = {
        **frozen,
        "run_name": probe.probe_id,
        "out_dir": f"artifacts/stage3/d1_behavioural/{probe.probe_id}",
        "data_dir": pack_dir,
        "seed": probe.seed,
        "student_path": probe.checkpoint_dir,
        "_purpose": (
            f"D1 {probe.rung} probe, arm {probe.arm_id}, seed {probe.seed}. "
            "Identical recovery; the only intended difference between probes "
            f"at one seed is the initialization. Derived from "
            f"{frozen_recipe.name} by overriding run identity, pack path, "
            "seed and student_path."),
    }
    differ = sorted(key for key in set(frozen) | set(derived)
                    if frozen.get(key) != derived.get(key))
    outside = sorted(set(differ) - overrides)
    if outside:
        raise D1BehaviouralDriverError(
            f"{probe.probe_id}: the derived probe config differs from the "
            f"frozen recipe in {outside}, outside the allowed override set "
            f"{sorted(overrides)}. Recovery must be identical across arms or "
            "a paired difference is not a difference between initializations.")
    path = audit / "configs" / f"{probe.probe_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(derived, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    status = args.status or "/workspace/autoinit_d1_behavioural.status"

    record: dict[str, Any] = {
        "schema": "aadistill.phase_d1.behavioural_session/v1",
        "run_id": args.run_id,
        "rung": args.rung,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "RUNNING",
        "probes": [],
    }
    evidence = out / "d1_behavioural.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=2, default=str) + "\n")

    mark(status, "DRIVER_START")
    try:
        from aadistill.infrastructure.manifest import sha256_json
        from experiments.phase_d1 import behavioural as D1B

        #: THE CONTRACT, FIRST. Hashed so the authorization can bind the arms,
        #: the battery, the recipe, the seeds and the schedule in one value --
        #: and asserted here, on the pod, before any weights load, because a
        #: session measuring a field nobody authorized is the one failure no
        #: amount of later evidence repairs.
        contract = D1B.session_contract(args.rung, REPO_ROOT)
        record["contract"] = contract
        record["contract_hash"] = sha256_json(contract)
        save()
        print(f"contract {record['contract_hash'][:16]} — "
              f"{contract['n_probes']} probes, {len(contract['arms']['arms'])} "
              f"arms, battery {contract['battery']['role']} "
              f"({contract['battery']['n_prompts']} prompts, "
              f"{contract['battery']['n_scorable']} scorable)", flush=True)

        #: Frozen assets from their ONE owner. Importing the C1 driver module
        #: is inert -- it defines constants and registers the builtin profiles
        #: and adapters, which this session needs too -- and `C1Driver` itself
        #: is deliberately NOT imported.
        from autoinit_c1_driver import (
            C1_PROBE_OVERRIDES, FROZEN_RECIPE, PACK_DIR, TRAINER,
            UNCAPPED_EVAL, _trainer_bytes, trained_model_dir,
        )
        from aadistill.runtime.device_handoff import (
            complete_release, cuda_memory, require_headroom, require_released,
        )

        probes = D1B.probes(args.rung, REPO_ROOT)
        record["schedule"] = [p.probe_id for p in probes]
        save()

        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            record["_check_only"] = (
                "the contract asserted, every arm's bytes verified at its "
                "recorded identity, the battery re-hashed against the "
                "manifest, the two roles shown disjoint, the schedule agreed "
                "with the design's declared count, and every symbol the "
                "training and scoring path names imported. No probe was "
                "trained and no model was loaded.")
            save()
            return 0

        results: list[dict[str, Any]] = []
        for index, probe in enumerate(probes):
            entry: dict[str, Any] = {
                "probe_id": probe.probe_id, "arm": probe.arm_id,
                "seed": probe.seed, "index": index,
                "trained": False, "scored": False,
            }
            record["probes"].append(entry)
            save()
            print(f"[{index + 1}/{len(probes)}] {probe.probe_id}", flush=True)

            config = probe_config(probe, audit=out, frozen_recipe=FROZEN_RECIPE,
                                  pack_dir=PACK_DIR,
                                  overrides=C1_PROBE_OVERRIDES)
            entry["config"] = str(config.relative_to(REPO_ROOT))

            #: HAND THE CARD OVER, AND PROVE IT. A C1 attempt read a verdict
            #: saying 7.55 GiB was still allocated, started the trainer anyway
            #: and lost the probe. Both conditions are enforced: the release
            #: worked, and the card has room for the measured peak.
            import gc

            before = cuda_memory()
            gc.collect()
            handoff = complete_release(before)
            require_released(handoff, what="the D1 recovery trainer")
            require_headroom(handoff["after"], need_bytes=_trainer_bytes(),
                             what="the D1 recovery trainer")
            entry["device_handoff"] = handoff.get("verdict")
            save()

            started = time.time()
            train = subprocess.run(
                ["/opt/train/bin/python", str(TRAINER), "--config", str(config)],
                capture_output=True, text=True,
                timeout=int(args.probe_train_minutes * 60 * 2))
            (out / f"{probe.probe_id}_train_tail.log").write_text(
                (train.stdout + train.stderr)[-2000:])
            entry["train_seconds"] = round(time.time() - started, 1)
            entry["train_rc"] = train.returncode
            if train.returncode != 0:
                entry["failed"] = "training"
                save()
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: the trainer exited "
                    f"{train.returncode}; see {probe.probe_id}_train_tail.log")
            entry["trained"] = True
            model_dir = trained_model_dir(
                REPO_ROOT / f"artifacts/stage3/d1_behavioural/{probe.probe_id}")
            entry["model_dir"] = str(model_dir)
            save()

            gen_dir = out / "generations" / probe.probe_id
            started = time.time()
            battery_root = REPO_ROOT / contract["battery"]["root"]
            generate = subprocess.run(
                ["/opt/vllm/bin/python", str(UNCAPPED_EVAL),
                 "--model", str(model_dir), "--battery", str(battery_root),
                 "--out-dir", str(gen_dir)],
                capture_output=True, text=True,
                timeout=int(args.probe_eval_minutes * 60 * 2))
            (out / f"{probe.probe_id}_eval_tail.log").write_text(
                (generate.stdout + generate.stderr)[-2000:])
            entry["generate_seconds"] = round(time.time() - started, 1)
            entry["generate_rc"] = generate.returncode
            if generate.returncode != 0:
                entry["failed"] = "generation"
                save()
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: generation exited "
                    f"{generate.returncode}")

            result_path = out / "results" / f"{probe.probe_id}.json"
            per_sample = out / "per_sample" / f"{probe.probe_id}.jsonl"
            arm = next(a for a in D1B.arms(REPO_ROOT)
                       if a.arm_id == probe.arm_id)
            score = subprocess.run(
                ["/opt/train/bin/python", str(SCORER),
                 "--generations", str(gen_dir), "--label", probe.probe_id,
                 "--seed", str(probe.seed), "--out", str(result_path),
                 "--per-sample", str(per_sample),
                 "--screening-arm", probe.arm_id,
                 "--init-digest", arm.artifact_digest],
                capture_output=True, text=True, timeout=1800)
            (out / f"{probe.probe_id}_score_tail.log").write_text(
                (score.stdout + score.stderr)[-2000:])
            if score.returncode != 0 or not result_path.is_file():
                entry["failed"] = "scoring"
                save()
                raise D1BehaviouralDriverError(
                    f"{probe.probe_id}: scoring exited {score.returncode}")
            scored = json.loads(result_path.read_text())
            entry.update({
                "scored": True,
                "result": str(result_path.relative_to(REPO_ROOT)),
                "correct_overall": scored["correct_overall"],
                "usable_rollout_rate": scored["usable_rollout_rate"],
                "n": scored["n"], "n_scorable": scored["n_scorable"],
                "result_sha256": scored["result_sha256"],
            })
            results.append({**entry, "scored_record": scored})
            #: ANNOUNCED THE MOMENT IT FINISHES, so the launcher's poll hook
            #: can pull it off-pod. A probe that finished is evidence whether
            #: or not the probes after it do.
            save()
            print(f"  correct_overall {scored['correct_overall']:.4f} "
                  f"usable {scored['usable_rollout_rate']:.4f}", flush=True)

        #: THE RANKING, from the scored probes this session produced. Pooled
        #: by the seed mean because the confirmation estimand is the
        #: "prompt-mean of the seed-mean paired difference", and a screening
        #: statistic computed another way would rank on a quantity the
        #: confirmation does not estimate.
        ranking = D1B.rank_screening(
            [{"arm": e["arm"], "seed": e["seed"],
              "correct_overall": e["correct_overall"],
              "usable_rollout_rate": e["usable_rollout_rate"]}
             for e in record["probes"] if e.get("scored")],
            REPO_ROOT)
        record["ranking"] = ranking
        save()
        print("\nranking (pooled over "
              f"{len(ranking['seeds'])} seeds, delta vs B):", flush=True)
        for row in ranking["ranked"]:
            flag = " VETOED" if row["vetoed"] else ""
            print(f"  {row['arm']}  delta {row['delta_vs_b']:+.4f}  "
                  f"usable {row['delta_usable_pooled']:+.4f}{flag}",
                  flush=True)

        selection = D1B.advance_one(ranking)
        record["selection"] = selection
        if selection.get("advanced"):
            print(f"\nADVANCING: {selection['arm']} "
                  f"({selection['state_id']}) delta "
                  f"{selection['delta_vs_b']:+.4f}, margin "
                  f"{selection['margin_over_runner_up']}", flush=True)
        else:
            print(f"\n{selection['outcome']}: {selection['why']}", flush=True)
        record["status"] = "COMPLETE"
        save()
    except BaseException as exc:                                  # noqa: BLE001
        import traceback

        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        written = False
        try:
            save()
            written = True
        except Exception as exc:                                  # noqa: BLE001
            print(f"could not write {evidence}: {exc}", flush=True)
        print(f"\nwrote {evidence}  status={record['status']}", flush=True)
        mark(status, terminal_marker(
            check_only=bool(args.check_only), status=record["status"],
            evidence_written=written))
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 11


if __name__ == "__main__":
    raise SystemExit(main())

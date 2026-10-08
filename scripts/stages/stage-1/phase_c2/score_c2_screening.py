#!/usr/bin/env python3
"""Score one C2 screening probe on `c2_screening_v1`.

    score_c2_screening.py --generations <dir> --label <probe> --seed <n> \
        --out <result.json> --per-sample <rows.jsonl>

**Every rule is imported.** `score_battery` and `build_result` are C1's, called
unchanged, because the frozen protocol requires the two rungs to share the
metric semantics and the mixture exactly: "correct_overall and its +0.010 SESOI
are defined ON the mixture, so screening and confirmation must share it or a
screening delta says nothing about a confirmation delta". Reusing the metric
contract `c1_confirmation_scoring@v1` here is that requirement, not a shortcut.

What differs is the battery identity, and that is the reason this file exists at
all rather than a `--battery` flag on C1's scorer. C1's `main()` pins its
battery by equality on purpose — "the production path cannot be aimed anywhere
else" — and the rung it guards is the one that may name an incumbent. Loosening
that pin so the screening rung could borrow the entry point would weaken a guard
for a consumer that does not need it weakened. A second pinned entry point costs
one small file and leaves C1's refusal intact.

The result this writes is a RANKING INPUT. It carries `role: C2_SCREENING` and
the protocol's `may_not` list, so a reader holding one result file alone can
still see that it may not promote anything, become training data, or produce a
verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

#: C1's scorer, imported. Importing it is inert — it runs `main()` only under
#: `__main__` — so its battery pins are defined but never consulted here.
from score_c1_confirmation import build_result, score_battery  # noqa: E402

from experiments.phase_c2.scoring import (  # noqa: E402
    BATTERY_PATH, validate_screening_battery,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

SCHEMA = "aadistill.autoinit.c2_screening_result/v1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", required=True, type=Path,
                    help="uncapped_eval.py --out-dir for this probe")
    ap.add_argument("--battery", default=Path(BATTERY_PATH), type=Path,
                    help="must be the frozen C2 screening battery; pins enforced")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="the TRAINING seed of the scored checkpoint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None)
    #: NOT `--arm`. Screening has six arms — five candidates and the anchor —
    #: and C1's incumbent/treatment pair does not describe them. The arm is
    #: recorded as an opaque label so nothing downstream can read a screening
    #: result as one half of a confirmation pair.
    ap.add_argument("--screening-arm", default=None,
                    help="candidate state id, or B for the anchor")
    ap.add_argument("--init-digest", default=None,
                    help="the arm initialization's artifact_digest")
    ap.add_argument("--trained-run", type=Path, default=None)
    ap.add_argument("--generation-fingerprint", default=None)
    args = ap.parse_args()

    battery = (REPO_ROOT / args.battery if not args.battery.is_absolute()
               else args.battery)
    manifest = json.loads((battery / "manifest.json").read_text())
    battery_identity = validate_screening_battery(manifest, repo_root=REPO_ROOT)
    battery_identity["manifest_file_sha256"] = sha256_file(battery / "manifest.json")

    gen_dir = (REPO_ROOT / args.generations if not args.generations.is_absolute()
               else args.generations)
    scored = score_battery(
        battery=battery, gen_dir=gen_dir, label=args.label, seed=args.seed,
        sets=manifest["sets"], scorable_sets=set(manifest["scorable_sets"]),
        behaviour_only=set(manifest["behaviour_only_sets"]))

    per_sample_path = None
    if args.per_sample is not None:
        per_sample_path = (REPO_ROOT / args.per_sample
                           if not args.per_sample.is_absolute() else args.per_sample)

    trained_run = None
    if args.trained_run is not None and Path(args.trained_run).is_file():
        rc = json.loads(Path(args.trained_run).read_text())
        trained_run = {"run_completion": str(args.trained_run),
                       "run_completion_sha256": sha256_file(Path(args.trained_run)),
                       "final_step": rc.get("final_step"),
                       "config_sha256": rc.get("config_sha256")}

    result = build_result(
        scored=scored, label=args.label, seed=args.seed,
        battery_identity=battery_identity,
        #: `arm=None`: see `--screening-arm`. C1's field takes incumbent or
        #: treatment and neither names a screening arm.
        arm=None,
        initialization_artifact_digest=args.init_digest, trained_run=trained_run,
        generation_protocol_fingerprint=args.generation_fingerprint,
        per_sample_path=per_sample_path, gen_dir=gen_dir, sets=manifest["sets"])

    #: Relabelled AFTER building, so the numbers are C1's and only the identity
    #: of this record changes. A screening result carrying C1's confirmation
    #: schema would be indistinguishable from evidence that may promote.
    result["schema"] = SCHEMA
    result["role"] = "C2_SCREENING"
    result["screening_arm"] = args.screening_arm
    result["may_not"] = battery_identity["may_not"]
    result["_metric_semantics"] = (
        "c1_confirmation_scoring@v1, reused unchanged. The protocol requires "
        "both rungs to share the mixture and the metric exactly; only the "
        "battery differs.")

    if per_sample_path is not None:
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_path.open("w") as f:
            for row in scored["per_sample"]:
                f.write(json.dumps(row) + "\n")
        result["per_sample_sha256"] = sha256_file(per_sample_path)
        result["per_sample_rows"] = len(scored["per_sample"])

    result["result_sha256"] = sha256_json(result)
    out = REPO_ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps({
        "label": args.label, "seed": args.seed, "role": "C2_SCREENING",
        "n": result["n"], "n_scorable": result["n_scorable"],
        "usable_rollout_rate": result["usable_rollout_rate"],
        "correct_overall": result["correct_overall"],
        "correct_given_usable": result["correct_given_usable"],
    }, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score one D1 screening probe on the realized `d1_screening` role.

    score_d1_screening.py --generations <dir> --label <probe> --seed <n> \
        --out <result.json> --per-sample <rows.jsonl> --screening-arm <arm>

**EVERY RULE IS IMPORTED.** `score_battery` and `build_result` are C1's, called
unchanged, because the frozen protocol requires every rung to share the metric
semantics and the mixture exactly: `correct_overall` and its SESOI are defined
ON the mixture, so a screening delta informs a confirmation delta only if both
are means over the same distribution. Reusing `c1_confirmation_scoring@v1` is
that requirement, not a shortcut.

WHY A THIRD PINNED ENTRY POINT rather than a `--battery` flag on an existing
one. C1's `main()` pins its battery by equality on purpose -- "the production
path cannot be aimed anywhere else" -- and the rung it guards is the one that
may name an incumbent. C2's screening scorer exists for the same reason, and
pins `c2_screening_v1`. Loosening either so D1 could borrow it would weaken a
guard for a consumer that does not need it weakened. A small pinned file leaves
both refusals intact.

WHAT DIFFERS FROM C2's, and it is the whole reason this file is not a copy: the
D-series roles carry no per-role `manifest.json`. Their identity lives in the
realized family manifest and their stratum scorability in
`phase_d_series.battery_family`, so the sets this scorer passes to
`score_battery` are DERIVED from those owners rather than read from a sidecar.
`code` is behaviour-only, which is what makes 950 prompts 850 scorable.

THE RESULT IS A RANKING INPUT, NOT A VERDICT. It is relabelled after building
so the numbers are C1's and only the record's identity changes: a screening
result carrying C1's confirmation schema would be indistinguishable from
evidence that may promote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit",
               "scripts/experiments/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from score_c1_confirmation import build_result, score_battery  # noqa: E402

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from experiments.phase_d1 import behavioural as D1B  # noqa: E402
from experiments.phase_d_series import battery_family as FAMILY  # noqa: E402

SCHEMA = "aadistill.phase_d1.screening_result/v1"
ROLE = "d1_screening"

#: What this record may NEVER be read as. Carried in the result rather than
#: left to a reader, because the one thing a screening number must not do is
#: stand in for a confirmation number -- the screening estimate is inflated by
#: the winner's curse by construction, and the design quantifies that.
MAY_NOT = (
    "be read as a confirmation result or as evidence that may promote. This "
    "is a RANKING input measured on the screening role, on seeds disjoint from "
    "the confirmation seeds and prompts disjoint from the confirmation "
    "prompts. The screening estimate of the advancing candidate is inflated by "
    "the winner's curse; the design records the inflation and the confirmation "
    "rung is what estimates the effect.",
)


def battery_sets(role: str = ROLE) -> dict[str, object]:
    """The stratum -> file mapping and the scorable split, DERIVED.

    `score_battery` wants three things C2 read out of a per-role
    `manifest.json`: every set, which are scorable, and which are
    behaviour-only. The D-series roles have no such sidecar, so each comes
    from the module that owns it -- the realized manifest for which files
    exist, and `battery_family.strata` for whether a stratum can be scored.
    Deriving them means a stratum whose scorability changed could not leave a
    stale sidecar behind.
    """
    strata = FAMILY.strata()
    sets = {name: f"{name}.jsonl" for name in sorted(strata)}
    scorable = {name for name, (_domain, _n, ok) in strata.items() if ok}
    behaviour_only = set(sets) - scorable
    return {"sets": sets, "scorable_sets": scorable,
            "behaviour_only_sets": behaviour_only}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generations", required=True, type=Path,
                    help="uncapped_eval.py --out-dir for this probe")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="the TRAINING seed of the scored checkpoint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None)
    #: NOT `--arm`. Screening has five arms -- four candidates and the
    #: incumbent -- and C1's incumbent/treatment pair does not describe them.
    #: Recorded as an opaque label so nothing downstream can read a screening
    #: result as one half of a confirmation pair.
    ap.add_argument("--screening-arm", default=None,
                    help="q1..q4, or B for the incumbent")
    ap.add_argument("--init-digest", default=None,
                    help="the arm initialization's artifact_digest")
    ap.add_argument("--trained-run", type=Path, default=None)
    ap.add_argument("--generation-fingerprint", default=None)
    args = ap.parse_args()

    #: THE BATTERY IS NOT AN ARGUMENT. It is resolved and byte-verified from
    #: the realized family, which is the pin this file exists to keep: a
    #: `--battery` flag is exactly how a rung comes to be scored on the wrong
    #: prompts. `battery_role` re-hashes all seven stratum files against the
    #: manifest and refuses a stratum balance that is not the frozen one.
    identity = D1B.battery_role(ROLE, REPO_ROOT)
    battery = REPO_ROOT / identity["root"]
    layout = battery_sets(ROLE)

    gen_dir = (REPO_ROOT / args.generations
               if not args.generations.is_absolute() else args.generations)
    scored = score_battery(
        battery=battery, gen_dir=gen_dir, label=args.label, seed=args.seed,
        sets=layout["sets"], scorable_sets=layout["scorable_sets"],
        behaviour_only=layout["behaviour_only_sets"])

    per_sample_path = None
    if args.per_sample is not None:
        per_sample_path = (REPO_ROOT / args.per_sample
                           if not args.per_sample.is_absolute()
                           else args.per_sample)

    trained_run = None
    if args.trained_run is not None and Path(args.trained_run).is_file():
        completion = json.loads(Path(args.trained_run).read_text())
        trained_run = {
            "run_completion": str(args.trained_run),
            "run_completion_sha256": sha256_file(Path(args.trained_run)),
            "final_step": completion.get("final_step"),
            "config_sha256": completion.get("config_sha256"),
        }

    battery_identity = {
        "family_id": identity["family_id"],
        "family_content_id": identity["family_content_id"],
        "allocation_rule_id": identity["allocation_rule_id"],
        "role": identity["role"],
        "item_ids_sha256": identity["item_ids_sha256"],
        "n_prompts": identity["n_prompts"],
        "n_scorable": identity["n_scorable"],
        "per_stratum": identity["per_stratum"],
        "files": identity["files"],
        "may_not": list(MAY_NOT),
    }

    result = build_result(
        scored=scored, label=args.label, seed=args.seed,
        battery_identity=battery_identity,
        #: `arm=None`: see `--screening-arm`. C1's field takes incumbent or
        #: treatment and neither names a screening arm.
        arm=None,
        initialization_artifact_digest=args.init_digest,
        trained_run=trained_run,
        generation_protocol_fingerprint=args.generation_fingerprint,
        per_sample_path=per_sample_path, gen_dir=gen_dir,
        sets=layout["sets"])

    #: Relabelled AFTER building, so the numbers are C1's and only the
    #: identity of this record changes.
    result["schema"] = SCHEMA
    result["role"] = "D1_SCREENING"
    result["screening_arm"] = args.screening_arm
    result["may_not"] = list(MAY_NOT)
    result["_metric_semantics"] = (
        "c1_confirmation_scoring@v1, reused unchanged. The protocol requires "
        "every rung to share the mixture and the metric exactly; only the "
        "battery differs.")
    result["_absolute_scores_are_not_interchangeable_with_c1s"] = (
        "three capacity-limited strata of the D-series family draw from wider "
        "source populations than C1's, so a D1 absolute score is not directly "
        "comparable with a historical C1 absolute score. The estimand is the "
        "within-family PAIRED difference, and both arms of this rung are "
        "measured on this same role under the same protocol.")

    if per_sample_path is not None:
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_path.open("w") as handle:
            for row in scored["per_sample"]:
                handle.write(json.dumps(row) + "\n")
        result["per_sample_sha256"] = sha256_file(per_sample_path)
        result["per_sample_rows"] = len(scored["per_sample"])

    result["result_sha256"] = sha256_json(result)
    out = REPO_ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps({
        "label": args.label, "seed": args.seed, "role": "D1_SCREENING",
        "arm": args.screening_arm,
        "n": result["n"], "n_scorable": result["n_scorable"],
        "usable_rollout_rate": result["usable_rollout_rate"],
        "correct_overall": result["correct_overall"],
        "correct_given_usable": result["correct_given_usable"],
    }, indent=1))


if __name__ == "__main__":
    main()

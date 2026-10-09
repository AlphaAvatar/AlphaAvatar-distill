#!/usr/bin/env python3
"""Score one D1 confirmation probe on the realized `d1_confirmation` role.

    score_d1_confirmation.py --generations <dir> --label <probe> --seed <n> \
        --out <result.json> --per-sample <rows.jsonl> --screening-arm <arm>

**EVERY RULE IS IMPORTED.** `score_battery` and `build_result` are C1's, called
unchanged, for the same reason the screening scorer imports them: the frozen
protocol requires every rung to share the metric semantics and the mixture
exactly, so a confirmation delta is comparable to the screening delta that
selected its candidate only because both are means over the same distribution.

WHY A FOURTH PINNED ENTRY POINT rather than a flag on the screening scorer.
Each rung's scorer pins ITS role by equality -- `score_d1_screening` pins
`d1_screening`, this pins `d1_confirmation` -- because the confirmation
estimate is unbiased only if it is computed on prompts the selection never
saw, and a `--battery` flag is exactly how a rung comes to be scored on the
wrong prompts. The two roles are byte-verified disjoint; the two scorers keep
them unmixable at the entry point too.

THE FLAG IS `--screening-arm` FOR INVOCATION COMPATIBILITY ONLY: the driver
calls both scorers with one shape, and the value here is the CONFIRMATION
arm label (the advancing candidate, or B). It is recorded as
`confirmation_arm`, and the result schema says which rung it is.

THE RESULT IS A MEASUREMENT, NOT A VERDICT. The three-way GO / NO-GO /
INCONCLUSIVE decision is derived off-pod at $0 from the per-sample rows under
the frozen decision rule; nothing in this record is a promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/autoinit",
               "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stages.phase_c1.score_c1_confirmation import build_result, score_battery  # noqa: E402

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from stages.phase_d1 import behavioural as D1B  # noqa: E402
from stages.d_series import battery_family as FAMILY  # noqa: E402

SCHEMA = "aadistill.phase_d1.confirmation_result/v1"
ROLE = "d1_confirmation"

#: What this record may NEVER be read as. A per-probe confirmation score is one
#: input to the frozen decision rule; it is not the rule's output.
MAY_NOT = (
    "be read as a verdict, a promotion, or a screening result. This is ONE "
    "probe's measurement on the confirmation role -- prompts disjoint from the "
    "screening prompts, seeds disjoint from the screening seeds -- and the "
    "three-way decision is derived off-pod under the frozen rule (stratified "
    "prompt-cluster bootstrap, seeds as fixed blocks) from the per-sample "
    "rows, never from this summary alone.",
)


def battery_sets(role: str = ROLE) -> dict[str, object]:
    """The stratum -> file mapping and the scorable split, DERIVED.

    Same derivation as the screening scorer's, from the same owners: the
    D-series roles carry no per-role `manifest.json`, so which files exist
    comes from the realized manifest and whether a stratum is scorable from
    `battery_family.strata`. `code` is behaviour-only, which is what makes
    950 prompts 850 scorable.
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
    #: Flag-compatible with the screening scorer so the driver's invocation is
    #: one shape; the VALUE is the confirmation arm label (the advancing
    #: candidate, or B) and it is recorded as `confirmation_arm`.
    ap.add_argument("--screening-arm", default=None,
                    help="the confirmation arm label: the advancing "
                         "candidate (q1..q4), or B")
    ap.add_argument("--init-digest", default=None,
                    help="the arm initialization's artifact_digest")
    ap.add_argument("--trained-run", type=Path, default=None)
    ap.add_argument("--generation-fingerprint", default=None)
    args = ap.parse_args()

    #: THE BATTERY IS NOT AN ARGUMENT. `battery_role` re-hashes all seven
    #: stratum files against the realized family manifest and refuses a
    #: stratum balance that is not the frozen one.
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
        #: `arm=None` for the same reason as the screening scorer: C1's field
        #: takes incumbent or treatment, and D1's labels are its own.
        arm=None,
        initialization_artifact_digest=args.init_digest,
        trained_run=trained_run,
        generation_protocol_fingerprint=args.generation_fingerprint,
        per_sample_path=per_sample_path, gen_dir=gen_dir,
        sets=layout["sets"])

    #: Relabelled AFTER building, so the numbers are C1's and only the
    #: identity of this record changes.
    result["schema"] = SCHEMA
    result["role"] = "D1_CONFIRMATION"
    result["confirmation_arm"] = args.screening_arm
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
        "label": args.label, "seed": args.seed, "role": "D1_CONFIRMATION",
        "arm": args.screening_arm,
        "n": result["n"], "n_scorable": result["n_scorable"],
        "usable_rollout_rate": result["usable_rollout_rate"],
        "correct_overall": result["correct_overall"],
        "correct_given_usable": result["correct_given_usable"],
    }, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score one C1 confirmation probe on `c1_confirmation_v1`.

    score_c1_confirmation.py --generations <dir> --label <probe> --seed <n> \
        --out <result.json> --per-sample <rows.jsonl>

`recovery_search_scoring@v2` cannot run on this battery — its pins are module
constants and its result builder requires a `metrics` key the C1 manifest does
not carry — so C1 declares its own binding, `c1_confirmation_scoring@v1`, and
leaves both frozen assets untouched. The identity, the pins, the metric contract
and the source closure are `aadistill.autoinit.c1_scoring`.

**Every rule is imported. Only the iteration is restated.** `scorer_correct`,
`summarize`, `group` and `TOOL_STRUCTURAL_GATE` come from the frozen scorer;
`score_recovery_row`, `validate_scored_rows`, `CAPABILITY_SCHEMA_V1` and
`usable_rollout` from the frozen library. What this file adds is the loop over
sets, because the historical implementation has that loop inline in `main()` and
there is no seam to call. That duplication is the one real risk here, and it is
closed by an admission gate rather than by reading: `score_battery` below is
battery-agnostic precisely so
`tests/autoinit/test_c1_confirmation_scoring.py` can drive real retained
`recovery_search_v2` generations through it and require equality of every
material numerical field against the frozen scorer's own output. The C1 pins
stay on `main()`, so the production path cannot be aimed anywhere else.

Nothing here monkeypatches the frozen scorer's constants and nothing injects a
`metrics` field into a battery at runtime. The battery defines the examples; this
contract defines the metric semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

#: The frozen implementation, imported rather than restated. Importing this
#: module is inert — it defines constants and functions and runs `main()` only
#: under `__main__` — so its own battery pins are never consulted here.
from score_recovery_search import (  # noqa: E402
    TOOL_STRUCTURAL_GATE, group, scorer_correct, summarize,
)

from aadistill.autoinit.c1_scoring import (  # noqa: E402
    C1_BATTERY_PATH, C1_METRIC_CONTRACT, SCHEMA, c1_scoring_contract,
    validate_c1_battery,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    CAPABILITY_SCHEMA_V1, score_recovery_row, validate_scored_rows,
)
from aadistill.evaluation import usable_rollout  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402


def battery_manifest(battery: Path) -> tuple[dict, str]:
    """The manifest and its canonicalized self-hash, checked for self-consistency."""
    manifest = json.loads((battery / "manifest.json").read_text())
    manifest_sha = sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest_sha != manifest.get("manifest_sha256"):
        raise SystemExit(
            f"the battery manifest does not match its own manifest_sha256 "
            f"({manifest_sha} vs {manifest.get('manifest_sha256')}); it has been "
            "edited since it was frozen")
    return manifest, manifest_sha


def score_battery(*, battery: Path, gen_dir: Path, label: str, seed: int,
                  sets: dict, scorable_sets: set, behaviour_only: set) -> dict:
    """Score every set of a battery. Battery-agnostic ON PURPOSE.

    This is the seam the historical scorer does not have, and it exists so the
    numerical-equivalence gate can feed the *historical* battery through exactly
    the code C1 runs. It applies no pins: `main()` owns those, so no caller can
    reach the production path with a different asset.

    Refuses a short, duplicated, foreign or incomplete set — there is no
    `--allow-missing-sets` here, because a confirmation battery scored over a
    subset is not a confirmation.
    """
    rows: list[dict] = []
    per_sample: list[dict] = []
    per_set: dict[str, dict] = {}
    missing_sets = [name for name in sets
                    if not (gen_dir / f"{name}.generations.jsonl").is_file()]
    if missing_sets:
        raise SystemExit(
            f"no generations for {sorted(missing_sets)}; refusing to report a "
            "rate over a subset of the battery as if it were the battery")

    for name, spec in sets.items():
        samples = {json.loads(line)["id"]: json.loads(line)
                   for line in (battery / f"{name}.jsonl").open() if line.strip()}
        if len(samples) != spec["n"]:
            raise SystemExit(f"{name}: frozen set has {len(samples)} items, "
                             f"manifest says {spec['n']}")
        scorable = name in scorable_sets
        if scorable == (name in behaviour_only):
            raise SystemExit(f"{name}: the manifest lists it as both or neither "
                             "scorable and behaviour-only")
        records = [json.loads(line)
                   for line in (gen_dir / f"{name}.generations.jsonl").open()
                   if line.strip()]
        seen: set[str] = set()
        set_rows: list[dict] = []
        for record in records:
            sample = samples.get(record["id"])
            if sample is None:
                raise SystemExit(
                    f"{name}: generation {record['id']!r} is not in the frozen "
                    "set — the battery and the generations disagree")
            if record["id"] in seen:
                raise SystemExit(f"{name}: duplicate generation {record['id']!r}")
            seen.add(record["id"])
            think_preopened = bool(record.get("think_preopened", True))
            tools_offered = bool(sample.get("tools"))
            components = usable_rollout.components(
                record, think_preopened=think_preopened,
                tools_offered=tools_offered)
            is_usable = all(components.values())
            correct, verdict = ((False, {"scorable": False}) if not scorable
                                else scorer_correct(name, record, sample))
            structural = None
            if name == "tool":
                structural = {k: bool(verdict.get(k)) for k in TOOL_STRUCTURAL_GATE}
                is_usable = is_usable and all(structural.values())
            row = score_recovery_row(usable=is_usable, scorer_correct=correct,
                                     scorable=scorable)
            row.update({"set": name, "id": record["id"],
                        "domain": spec["domain"], **components})
            if structural is not None:
                row["tool_structurally_executable"] = all(structural.values())
                row.update(structural)
            set_rows.append(row)
            rows.append(row)
            per_sample.append({"label": label, "seed": seed, **row,
                               "verdict": verdict})
        absent = set(samples) - seen
        if absent:
            raise SystemExit(
                f"{name}: {len(absent)} of {len(samples)} prompts have no "
                f"generation, e.g. {sorted(absent)[:3]}. A silently short set "
                "would inflate every rate computed over it.")
        per_set[name] = summarize(set_rows, scorable=scorable)
        per_set[name]["missing_generations"] = 0

    return {
        "rows": rows,
        "per_sample": per_sample,
        "per_set": per_set,
        "row_contract": validate_scored_rows(rows),
        "totals": summarize(rows, scorable=None),
        "per_domain": group(rows, "domain"),
        "per_capability": {name: per_set[name]
                           for name in CAPABILITY_SCHEMA_V1.expected
                           if name in per_set},
    }


def build_result(*, scored: dict, label: str, seed: int, battery_identity: dict,
                 arm: str | None, initialization_artifact_digest: str | None,
                 trained_run: dict | None, generation_protocol_fingerprint: str | None,
                 per_sample_path: Path | None, gen_dir: Path,
                 sets: dict) -> dict:
    """The C1 result record. Counts first; the decision layer reads counts."""
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": label,
        "seed": seed,
        "arm": arm,
        "initialization_artifact_digest": initialization_artifact_digest,
        "trained_run": trained_run,
        "battery": battery_identity,
        "scoring_contract": c1_scoring_contract(REPO_ROOT),
        "generation_protocol_fingerprint": generation_protocol_fingerprint,
        "metric_contract": C1_METRIC_CONTRACT,
        "tool_usable_gate": {
            "definition": ("generic usable_rollout AND "
                           + " AND ".join(TOOL_STRUCTURAL_GATE)),
            "excluded": {
                "tool_args_schema_ok": ("the xLAM `required` reconstruction is "
                                        "interpretive; kept diagnostic"),
                "tool_call_exact_match": ("that is correctness; folding it in "
                                          "would collapse the two axes"),
            },
            "multi_call": ("the frozen scorer's own all-calls semantics; no "
                           "second interpretation is defined here"),
        },
        "missing_sets": [],
        "row_contract": scored["row_contract"],
        **scored["totals"],
        "per_set": scored["per_set"],
        "per_domain": scored["per_domain"],
        "per_capability": scored["per_capability"],
        "generations": {
            name: {
                "path": str((gen_dir / f"{name}.generations.jsonl").relative_to(
                    REPO_ROOT))
                if str(gen_dir).startswith(str(REPO_ROOT))
                else str(gen_dir / f"{name}.generations.jsonl"),
                "sha256": sha256_file(gen_dir / f"{name}.generations.jsonl"),
                "n": spec["n"],
            } for name, spec in sets.items()},
        "per_sample_path": str(per_sample_path) if per_sample_path else None,
        "no_weighted_scalar": (
            "usable_rollout_rate and correct_overall are reported separately and "
            "are never combined; usable_rollout is blind to correctness by "
            "construction, which is why correctness is a separate axis"),
    }
    CAPABILITY_SCHEMA_V1.validate(result, label=label)
    result["capability_schema_enforced"] = True
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", required=True, type=Path,
                    help="uncapped_eval.py --out-dir for this probe")
    ap.add_argument("--battery", default=C1_BATTERY_PATH, type=Path,
                    help="must be the frozen C1 battery; the pins are enforced")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="the TRAINING seed of the scored checkpoint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None)
    ap.add_argument("--arm", default=None, choices=("incumbent", "treatment"))
    ap.add_argument("--init-digest", default=None,
                    help="the arm initialization's artifact_digest")
    ap.add_argument("--trained-run", type=Path, default=None,
                    help="the probe's run_completion.json, bound as run identity")
    ap.add_argument("--generation-fingerprint", default=None)
    args = ap.parse_args()

    battery = (REPO_ROOT / args.battery) if not args.battery.is_absolute() \
        else args.battery
    manifest, manifest_sha = battery_manifest(battery)
    #: Fail closed on the frozen C1 identity. No `metrics` key is required.
    battery_identity = validate_c1_battery(manifest, manifest_sha256=manifest_sha)
    battery_identity["manifest_file_sha256"] = sha256_file(battery / "manifest.json")

    gen_dir = (REPO_ROOT / args.generations) if not args.generations.is_absolute() \
        else args.generations
    scored = score_battery(
        battery=battery, gen_dir=gen_dir, label=args.label, seed=args.seed,
        sets=manifest["sets"], scorable_sets=set(manifest["scorable_sets"]),
        behaviour_only=set(manifest["behaviour_only_sets"]))

    per_sample_path = None
    if args.per_sample is not None:
        per_sample_path = ((REPO_ROOT / args.per_sample)
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
        battery_identity=battery_identity, arm=args.arm,
        initialization_artifact_digest=args.init_digest, trained_run=trained_run,
        generation_protocol_fingerprint=args.generation_fingerprint,
        per_sample_path=per_sample_path, gen_dir=gen_dir, sets=manifest["sets"])

    if per_sample_path is not None:
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with per_sample_path.open("w") as f:
            for row in scored["per_sample"]:
                f.write(json.dumps(row) + "\n")
        result["per_sample_sha256"] = sha256_file(per_sample_path)
        result["per_sample_rows"] = len(scored["per_sample"])

    result["result_sha256"] = sha256_json(result)
    out = (REPO_ROOT / args.out) if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps({
        "label": args.label, "seed": args.seed,
        "n": result["n"], "n_scorable": result["n_scorable"],
        "usable_rollout_rate": result["usable_rollout_rate"],
        "correct_overall": result["correct_overall"],
        "correct_given_usable": result["correct_given_usable"],
        "scoring_contract": result["scoring_contract"]["contract"],
    }, indent=2))


if __name__ == "__main__":
    main()

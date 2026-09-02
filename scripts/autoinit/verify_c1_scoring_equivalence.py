#!/usr/bin/env python3
"""The admission gate for `c1_confirmation_scoring@v1`: prove the numbers did not move.

    PYTHONPATH=src python scripts/autoinit/verify_c1_scoring_equivalence.py

C1 needs a new scoring *binding* because the frozen scorer's battery pins and its
`manifest["metrics"]` requirement make it unable to run on `c1_confirmation_v1`.
It must not acquire new *semantics*: the C0 power analysis and the SESOI were
computed under `recovery_search_scoring@v2`, so a scorer that shifted any count
would silently invalidate the design, not merely the comparison.

Every rule is imported by the C1 scorer, but the loop over sets is restated —
the historical implementation keeps it inline in `main()` and offers no seam. So
this gate exists to make that restatement checkable rather than reviewable: it
feeds REAL retained `recovery_search_v2` generations through both paths and
requires equality of every material numerical field, per sample and in aggregate.

There is no tolerance. A float that differs in the last place is a difference.

Identity and provenance fields are expected to differ and are the only ones
excluded: schema name, command, timestamp, battery identity, scoring-contract
identity, and the result hashes derived from them.

Exit 0 when every field matches; 5 on any difference or when no historical
generations are reachable on this machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from score_c1_confirmation import battery_manifest, score_battery  # noqa: E402

from aadistill.autoinit.c1_scoring import (  # noqa: E402
    C1_SCORING_SEMANTIC_PARENT, C1_SCORING_SEMANTIC_PARENT_DIGEST,
    c1_scoring_contract,
)
from aadistill.autoinit.recovery import recovery_scoring_contract  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

HISTORICAL_BATTERY = REPO_ROOT / "artifacts/stage3/recovery_search_v2"
FROZEN_SCORER = REPO_ROOT / "scripts/autoinit/score_recovery_search.py"
OUT = REPO_ROOT / "logs/phase_c1_scoring_equivalence.json"

#: Where retained historical evidence lives on a dev box. Searched, not required:
#: the generations are large and out of tree by policy (AGENTS.md 2.5), so this
#: gate reports UNAVAILABLE rather than failing on a machine that has none.
EVIDENCE_ROOTS = (Path.home() / "aad-artifacts",)

#: Provenance, not arithmetic. Everything else must match exactly.
IGNORED_TOP_LEVEL = frozenset({
    "schema", "created_utc", "command", "battery", "scoring_contract",
    "result_sha256", "metric_contract", "tool_usable_gate", "missing_sets",
    "no_weighted_scalar", "capability_schema_enforced", "label", "seed",
    "arm", "initialization_artifact_digest", "trained_run", "generations",
    "per_sample_path", "per_sample_sha256", "per_sample_rows",
    "generation_protocol_fingerprint",
})


def find_generations() -> list[Path]:
    """Historical dirs holding a complete `recovery_search_v2` generation set."""
    manifest, _ = battery_manifest(HISTORICAL_BATTERY)
    want = {f"{name}.generations.jsonl" for name in manifest["sets"]}
    found = []
    for root in EVIDENCE_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.generations.jsonl"):
            d = path.parent
            if d in found:
                continue
            if want <= {p.name for p in d.glob("*.generations.jsonl")}:
                found.append(d)
    return sorted(found)


def frozen_scores(gen_dir: Path, label: str, seed: int, work: Path) -> tuple[dict, list]:
    """The frozen scorer's own output, via its real CLI. No monkeypatching."""
    out, ps = work / "frozen.json", work / "frozen_per_sample.jsonl"
    rc = subprocess.run(
        [sys.executable, str(FROZEN_SCORER),
         "--generations", str(gen_dir), "--battery", str(HISTORICAL_BATTERY),
         "--label", label, "--seed", str(seed), "--out", str(out),
         "--per-sample", str(ps)],
        capture_output=True, text=True, timeout=1800,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"})
    if rc.returncode != 0 or not out.is_file():
        raise SystemExit(f"the frozen scorer failed on {gen_dir}: "
                         f"rc={rc.returncode}\n{(rc.stdout + rc.stderr)[-1500:]}")
    return (json.loads(out.read_text()),
            [json.loads(line) for line in ps.open() if line.strip()])


def compare(frozen: dict, frozen_rows: list, c1: dict, c1_rows: list) -> list[str]:
    """Every material numerical field, or the exact list of what moved."""
    diffs: list[str] = []

    keys = (set(frozen) | set({**c1["totals"], "per_set": 1, "per_domain": 1,
                               "per_capability": 1, "row_contract": 1})) \
        - IGNORED_TOP_LEVEL
    mine = {**c1["totals"], "per_set": c1["per_set"], "per_domain": c1["per_domain"],
            "per_capability": c1["per_capability"], "row_contract": c1["row_contract"]}
    for key in sorted(keys):
        a, b = frozen.get(key, "<absent>"), mine.get(key, "<absent>")
        if a != b:
            diffs.append(f"aggregate {key}: frozen={a!r} c1={b!r}")

    if len(frozen_rows) != len(c1_rows):
        diffs.append(f"per-sample row count: {len(frozen_rows)} vs {len(c1_rows)}")
        return diffs
    for i, (a, b) in enumerate(zip(frozen_rows, c1_rows)):
        if a != b:
            moved = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            diffs.append(f"per-sample row {i} ({a.get('id')}): {moved}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=Path, default=None,
                    help="one historical recovery_search_v2 generation dir")
    ap.add_argument("--max-dirs", type=int, default=64)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dirs = [args.generations] if args.generations else find_generations()
    record = {
        "schema": "aadistill.autoinit.c1_scoring_equivalence/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "Admission gate for c1_confirmation_scoring@v1. C1 changes the "
            "scoring BINDING because the frozen scorer cannot run on the frozen "
            "C1 battery; it must not change the scoring SEMANTICS, because the C0 "
            "power analysis and the SESOI were computed under them. Real retained "
            "recovery_search_v2 generations are scored through both paths and "
            "every material numerical field must be equal. No tolerance."),
        "c1_contract": c1_scoring_contract(REPO_ROOT)["contract"],
        "c1_digest": c1_scoring_contract(REPO_ROOT)["digest"],
        "semantic_parent": C1_SCORING_SEMANTIC_PARENT,
        "semantic_parent_digest": C1_SCORING_SEMANTIC_PARENT_DIGEST,
        "historical_contract_live_digest": recovery_scoring_contract(REPO_ROOT)["digest"],
        "historical_battery": "recovery_search_v2",
        "ignored_fields": sorted(IGNORED_TOP_LEVEL),
        "cases": [],
    }
    if not dirs:
        record["verdict"] = "UNAVAILABLE"
        record["reason"] = (
            "no retained recovery_search_v2 generation directory on this machine; "
            "the gate cannot run and C1 must not be authorized on this evidence")
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print("UNAVAILABLE: no historical generations found")
        return 5

    total_diffs = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        manifest, _ = battery_manifest(HISTORICAL_BATTERY)
        for i, gen_dir in enumerate(dirs[:args.max_dirs]):
            label, seed = gen_dir.name, 20260726
            frozen, frozen_rows = frozen_scores(gen_dir, label, seed, work / str(i))
            c1 = score_battery(
                battery=HISTORICAL_BATTERY, gen_dir=gen_dir, label=label, seed=seed,
                sets=manifest["sets"],
                scorable_sets=set(manifest["scorable_sets"]),
                behaviour_only=set(manifest["behaviour_only_sets"]))
            diffs = compare(frozen, frozen_rows, c1, c1["per_sample"])
            total_diffs += len(diffs)
            record["cases"].append({
                "generations": str(gen_dir),
                "n_rows": len(c1["per_sample"]),
                "n": c1["totals"]["n"],
                "n_scorable": c1["totals"]["n_scorable"],
                "usable": c1["totals"]["usable"],
                "correct": c1["totals"]["correct"],
                "usable_scorable": c1["totals"]["usable_scorable"],
                "usable_rollout_rate": c1["totals"]["usable_rollout_rate"],
                "correct_overall": c1["totals"]["correct_overall"],
                "correct_given_usable": c1["totals"]["correct_given_usable"],
                "correct_but_unusable_rows": sum(
                    1 for r in c1["rows"] if r.get("correct_but_unusable")),
                "identical": not diffs,
                "differences": diffs[:20],
            })
            print(f"{'MATCH' if not diffs else 'DIFFER'}  {gen_dir.name}  "
                  f"n={c1['totals']['n']} correct={c1['totals']['correct']} "
                  f"usable={c1['totals']['usable']}")

    #: What this evidence CANNOT discriminate, stated rather than implied.
    #: `correct_but_unusable` is 0 on every retained probe: no historical rollout
    #: was ever scored correct while being unusable, so `correct => usable` never
    #: fires and an equivalence run cannot detect its removal. The rule is covered
    #: directly against the frozen `score_recovery_row` instead. Silence about a
    #: gap reads as coverage.
    uncovered = sum(1 for c in record["cases"]
                    if c.get("correct_but_unusable_rows", 0) > 0)
    record["coverage_limits"] = {
        "correct_implies_usable": (
            "NOT covered by this gate: correct_but_unusable is 0 on every "
            f"retained probe ({uncovered} of {len(record['cases'])} have any), so "
            "the implication never fires in historical evidence. Covered directly "
            "in tests/autoinit/test_c1_confirmation_scoring.py against the frozen "
            "score_recovery_row, which C1 imports unmodified."),
        "c1_battery_rows": (
            "this gate runs on recovery_search_v2 by construction — it is the only "
            "battery both scorers can read. The C1 battery is exercised separately "
            "by the production-path tests."),
    }
    record["n_cases"] = len(record["cases"])
    record["verdict"] = "IDENTICAL" if total_diffs == 0 else "DIFFERENT"
    record["total_differences"] = total_diffs
    record["record_sha256"] = sha256_json(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"\n{record['verdict']} over {record['n_cases']} case(s) -> {args.out}")
    return 0 if total_diffs == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())

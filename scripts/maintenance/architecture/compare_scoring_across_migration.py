#!/usr/bin/env python3
"""Re-score frozen evidence through the pre- and post-migration trees.

    PYTHONPATH=src:scripts python scripts/architecture/compare_scoring_across_migration.py \
        --base origin/main --write

The initialization cutover moved `aadistill.autoinit.recovery` and rewrote
import lines in the recovery scorer, so the scoring-contract digest legitimately
moves from v2 to v3. A moved digest is not by itself evidence of anything: it is
equally consistent with "only the paths changed" and with "the scorer now
computes something else". The claim that has to be checked is the second one.

So this materializes the base commit into a scratch tree, points both trees at
the *same* frozen generations, and compares what they produce. Contract fields
are expected to differ — a v3 reporting v2's digest would be hiding the
relocation rather than showing it harmless — and so is run provenance, since the
two invocations have different output paths and wall-clock stamps. Every score,
count, rate and per-sample record must be identical.

The frozen evidence lives outside the repository and is never modified here; both
sides read it, neither writes to it.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.evaluation.usable_rollout import detect_schema  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = "logs/maintenance/inventories/architecture_scoring_equivalence.json"

#: Frozen Phase-A rung-1 generations, by search path. Real evidence from a
#: completed run, not a fixture: a synthetic case cannot show that 190 real
#: samples score identically.
EVIDENCE_ROOT = Path("/home/ecs-user/aad-artifacts/autoinit/phase_b/attempt5/"
                     "raw_evidence/eval/phase_a")
PATHS = ("ab7632b00788", "bf5ae3b6ae00", "fe9683e6a9c7")
TRAINING_SEED = "20260726"

#: The scoring-contract identity. Expected to differ: the digest is over
#: `path:sha256` lines and two of the six declared files moved or changed bytes.
CONTRACT_FIELDS = {"scoring_contract", "contract", "digest", "files", "version",
                   "supersedes"}
#: Run provenance. Expected to differ: distinct --out paths and timestamps, plus
#: result_sha256, which is a hash *over* those fields.
PROVENANCE_FIELDS = {"command", "created_utc", "result_sha256", "out", "repo_root"}


def strip(doc, drop: set[str]):
    if isinstance(doc, dict):
        return {k: strip(v, drop) for k, v in doc.items() if k not in drop}
    if isinstance(doc, list):
        return [strip(x, drop) for x in doc]
    return doc


def load(path: Path):
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:            # per-sample output is JSONL
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def materialize(base: str, dest: Path) -> str:
    """The base commit as a working tree, sharing this repo's artifact store."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", base], cwd=REPO,
                             capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    #: The scorer resolves its search manifest under `artifacts/`, which is an
    #: out-of-tree store and absent from any archive. Both sides must read the
    #: same one, or the comparison would be of two different inputs.
    link = dest / "artifacts"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to((REPO / "artifacts").resolve())
    return subprocess.run(["git", "rev-parse", base], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


def score(tree: Path, label: str, out_dir: Path) -> tuple[Path, Path, list[str]]:
    summary = out_dir / f"{label}.json"
    per_sample = out_dir / f"{label}.persample.json"
    cmd = [sys.executable, "scripts/autoinit/score_recovery_search.py",
           "--generations", str(EVIDENCE_ROOT / f"autoinit.v1.phase_a.rung1.{label}.sa"),
           "--label", label, "--seed", TRAINING_SEED,
           "--out", str(summary), "--per-sample", str(per_sample),
           "--allow-missing-sets"]
    env = {"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin", "HOME": str(tree)}
    proc = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"scoring failed in {tree}:\n{proc.stdout}\n{proc.stderr}")
    return summary, per_sample, cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main",
                    help="the commit that owns the pre-migration paths")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not EVIDENCE_ROOT.is_dir():
        raise SystemExit(f"frozen evidence not present at {EVIDENCE_ROOT}; this "
                         "comparison needs the real generations, not a fixture")

    drop = CONTRACT_FIELDS | PROVENANCE_FIELDS
    with tempfile.TemporaryDirectory(prefix="aad-migration-cmp-") as tmp:
        tmp = Path(tmp)
        old_tree = tmp / "old"
        base_sha = materialize(args.base, old_tree)
        results = []
        for label in PATHS:
            row: dict = {"search_path": label}
            outs = {}
            for side, tree in (("old", old_tree), ("new", REPO)):
                out_dir = tmp / side
                out_dir.mkdir(exist_ok=True)
                summary, per_sample, cmd = score(tree, label, out_dir)
                outs[side] = (load(summary), load(per_sample))
                row[f"{side}_command"] = shlex.join(cmd)
            (o_sum, o_ps), (n_sum, n_ps) = outs["old"], outs["new"]
            row.update({
                "n_samples": len(o_ps),
                "summary_scores_identical": strip(o_sum, drop) == strip(n_sum, drop),
                "per_sample_identical": o_ps == n_ps,
                "contract_digest_old": o_sum.get("scoring_contract", {}).get("digest"),
                "contract_digest_new": n_sum.get("scoring_contract", {}).get("digest"),
            })
            results.append(row)
            print(f"  {label}: {row['n_samples']} samples, "
                  f"summary_identical={row['summary_scores_identical']}, "
                  f"per_sample_identical={row['per_sample_identical']}")

    all_same = all(r["summary_scores_identical"] and r["per_sample_identical"]
                   for r in results)
    #: Which record schema the frozen generations actually use. Read from the
    #: raw evidence, not from the scored output: the per-sample file is the
    #: scorer's verdict, and naming the branch requires the record it read.
    schemas = sorted({
        detect_schema(json.loads(next(iter(gen.open()))))
        for label in PATHS
        for gen in sorted(
            (EVIDENCE_ROOT / f"autoinit.v1.phase_a.rung1.{label}.sa").glob(
                "*.generations.jsonl"))})
    doc = {
        "schema": "aadistill.architecture_scoring_equivalence/v1",
        "_contract": (
            "Frozen Phase-A generations re-scored through the pre- and "
            "post-migration trees. Shows that relocating the recovery scorer "
            "moved its contract digest WITHOUT moving any number it produces. "
            "AUTHORIZES NOTHING and is not a new experimental result — no model "
            "ran, and the generations are unmodified evidence from a completed "
            "run."),
        "base_commit": base_sha,
        "migrated_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip(),
        "evidence_root": str(EVIDENCE_ROOT),
        "training_seed": TRAINING_SEED,
        "expected_to_differ": {
            "contract_fields": sorted(CONTRACT_FIELDS),
            "why": ("the contract digest is over 'path:sha256' lines; two of the "
                    "six declared files moved or changed import lines, so v3 "
                    "MUST report a different digest than v2"),
            "provenance_fields": sorted(PROVENANCE_FIELDS),
        },
        "must_not_differ": "every score, count, rate and per-sample record",
        "coverage": {
            "generation_schemas_exercised": schemas,
            "limitation": (
                "usable_rollout reads two record schemas and this evidence "
                "exercises only the one(s) named above. Inverting a component in "
                "the OTHER branch does not move these numbers, so this "
                "comparison would not detect it. Verified by mutation, not "
                "assumed: inverting no_severe_repetition in _from_behavior_v0 "
                "turns all three paths to MISMATCH, while the same inversion in "
                "_from_three_mode leaves them identical."),
        },
        "paths_compared": len(results),
        "total_samples": sum(r["n_samples"] for r in results),
        "all_scores_identical": all_same,
        "results": results,
        "authorizes": "nothing",
    }
    print(f"\n{doc['total_samples']} samples across {len(results)} paths: "
          f"{'ALL IDENTICAL' if all_same else 'MISMATCH'}")
    if args.write:
        (REPO / OUT).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())

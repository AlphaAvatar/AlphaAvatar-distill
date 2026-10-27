#!/usr/bin/env python3
"""Build the verifiable PRIMARY-axis task pool for Stage-2 teacher generation. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.build_training_tasks --version 1

Source `s0_generators_v2_calsplit` (configs/.../sources.json): the v2 roster of
the project's procedural generators, CALIBRATION-split families only, seeds in
the Stage-2 training range of `isolation_rules.json :: R5`. A task carries its
gold and its generator's checker, so every teacher answer on it is verifiable.

How many: per primary stratum, the supervised tokens the master corpus needs
(`corpus_spec.json`, top rung + reserve, times its share), divided by the
measured verified supervised tokens per accepted unit and by the measured unit
accept rate (Stage-0 teacher_cal, `audit/asset_audit.json`), times `--oversupply`.
Inside a stratum, cells (generator x difficulty) get equal counts and cycle
through the generator's calibration families in a fixed order -- nothing is
chosen by looking at results.

Writes `data/stages/stage-2/reasoning_data/tasks/train_tasks_v<N>.jsonl` with a
manifest and its isolation report, and refuses a pool that fails any rule.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone

from stages.reasoning_data import assets as A
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import mechanism_check as MC
from stages.reasoning_data import source_plan as SP

from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import DIFFICULTIES, Unsatisfiable, build_instance

OUT_DIR = A.REPO / "data/stages/stage-2/reasoning_data/tasks"
SCHEMA = "aadistill.stage2.reasoning_data.task_set/v1"
ROLE = "stage2_training"


def demand(oversupply: float) -> dict[str, dict]:
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    audit = A.read_json(A.LOG_DIR / "audit/asset_audit.json")
    tc = audit["teacher_cal"]["under_target_quality_contract"]
    pool = audit["base_pool_v1_fluid"]
    top = max(spec["rung_targets"]) + spec["reserve_tokens"]
    shares = SP.project_demand_shares(spec)      # the largest any corpus variant asks of this source
    out = {}
    for stratum, cfg in spec["strata"].items():
        axis = cfg.get("axis")
        if not stratum.startswith("primary.") or axis is None:
            continue
        need = top * shares[stratum]
        per_unit = (audit["teacher_cal"]["yield_by_axis"][axis]["verified_supervised_positions_le_8192"]
                    / pool["units_by_axis"][axis])
        accept = pool["units_by_axis"][axis] / tc[axis]["units"]
        tasks = math.ceil(need / per_unit / accept * oversupply)
        out[axis] = {"stratum": stratum, "supervised_tokens_needed": round(need),
                     "supervised_per_accepted_unit": round(per_unit),
                     "accept_rate_le_8192": round(accept, 4), "oversupply": oversupply,
                     "tasks": tasks}
    return out


def taken_instances() -> set[str]:
    """Content hashes no training task may repeat: every Stage-0 task set and every
    fluid evaluation item. A small instance space repeats across seeds, so a new
    seed range alone does not make an instance new."""
    rows = [json.loads(l) for p in sorted((A.REPO / "data/stages/stage-0/calibration_volume/tasks").glob("*.jsonl"))
            for l in p.read_text().splitlines() if l.strip()]
    return {t["content_sha256"] for t in rows} | {t["content_sha256"] for t in IC.fluid_eval_items()}


#: Calibration-split families whose rule computes an evaluation family's rule
#: (isolation rule R9, `mechanism_check.review`): never drawn for training.
MECHANISM_EXCLUDED = frozenset(MC.review()["equivalent_training_families"])


def build(plan: dict[str, dict]) -> list[dict]:
    lo, hi = next(r for r in IC.rules()["rules"]
                  if r["id"] == "R5_shared_generator_exposure")["training_seed_range"]
    by_axis = defaultdict(list)
    for gen in R.GENERATORS_V2:
        if R.family_split(gen)["calibration"]:
            by_axis[gen.axis].append(gen)
    tasks, seen = [], set()
    taken = taken_instances()
    skipped: Counter = Counter()
    for axis, row in sorted(plan.items()):
        gens = sorted(by_axis[axis], key=lambda g: g.generator_id)
        cells = [(g, d) for g in gens for d in DIFFICULTIES]
        per_cell = math.ceil(row["tasks"] / len(cells))
        for c_i, (gen, difficulty) in enumerate(cells):
            fams = [f for f in R.family_split(gen)["calibration"]
                    if f"{gen.generator_id}/{f}" not in MECHANISM_EXCLUDED]
            made, i = 0, 0
            while made < per_cell:
                seed = lo + c_i * 10_000 + i
                if seed >= hi:
                    raise SystemExit(f"{gen.generator_id}/{difficulty}: training seed range exhausted")
                fam = gen.family(fams[i % len(fams)])
                i += 1
                try:
                    t = build_instance(gen, fam, difficulty, seed)
                except (Unsatisfiable, IndexError, ValueError) as exc:
                    # A frozen generator that cannot realize this draw (some raise
                    # IndexError on an empty candidate list) is skipped and counted;
                    # the generator itself is never edited.
                    skipped[f"{gen.generator_id}:{type(exc).__name__}"] += 1
                    continue
                if t["content_sha256"] in seen:
                    continue
                if t["content_sha256"] in taken:
                    skipped[f"{gen.generator_id}:repeats_a_stage0_or_eval_instance"] += 1
                    continue
                seen.add(t["content_sha256"])
                t.update(role=ROLE, unit_id=t["task_id"], source="s0_generators_v2_calsplit",
                         source_split="calibration_families")
                tasks.append(t)
                made += 1
    return tasks, dict(sorted(skipped.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--oversupply", type=float, default=2.0)
    args = ap.parse_args(argv)
    out = OUT_DIR / f"train_tasks_v{args.version}.jsonl"
    if out.exists():
        raise SystemExit(f"{out.relative_to(A.REPO)} exists; a built task set is never rebuilt -- bump --version")
    plan = demand(args.oversupply)
    tasks, skipped = build(plan)
    sources = A.read_json(A.CONFIG_DIR / "sources.json")
    report = IC.check_pool(tasks, sources=sources)
    # Instance-level isolation from every Stage-0 task set, which is calibration, not evaluation.
    s0 = [json.loads(l) for p in sorted((A.REPO / "data/stages/stage-0/calibration_volume/tasks").glob("*.jsonl"))
          for l in p.read_text().splitlines() if l.strip()]
    s0_content = {t["content_sha256"] for t in s0}
    report["stage0_task_sets_content_overlap"] = sum(t["content_sha256"] in s0_content for t in tasks)
    if not report["pass"] or report["stage0_task_sets_content_overlap"]:
        raise SystemExit(f"isolation failed: {report['violations']} / "
                         f"stage0 overlap {report['stage0_task_sets_content_overlap']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks))
    (OUT_DIR / f"train_tasks_v{args.version}.isolation.json").write_text(json.dumps(report, indent=1) + "\n")
    manifest = {
        "schema": SCHEMA, "role": ROLE, "version": args.version,
        "status": "PROPOSED pool: no teacher target exists for it, and it is not adopted as training data until the maintainer accepts the corpus spec",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": f"python -m stages.reasoning_data.build_training_tasks --version {args.version} --oversupply {args.oversupply}",
        "source": "s0_generators_v2_calsplit", "roster": "registry.GENERATORS_V2",
        "split_salt": R.SPLIT_SALT, "cal_fraction": R.CAL_FRACTION,
        "seed_rule": "training range lo + cell_index * 10,000 + i",
        "demand": plan, "n_tasks": len(tasks), "skipped_draws": skipped,
        "mechanism_excluded_families": sorted(MECHANISM_EXCLUDED),
        "by_axis": dict(sorted(Counter(t["axis"] for t in tasks).items())),
        "by_generator": dict(sorted(Counter(t["generator_id"] for t in tasks).items())),
        "distinct_families": len({f"{t['generator_id']}/{t['family_id']}" for t in tasks}),
        "interactive": sum(bool(t["interactive"]) for t in tasks),
        "sha256": A.sha256_file(out),
        "isolation_report": f"train_tasks_v{args.version}.isolation.json",
        "license": "project-generated procedural tasks; no external dataset, no ARC-AGI material",
        "authorizes": "nothing",
    }
    (OUT_DIR / f"train_tasks_v{args.version}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(out.relative_to(A.REPO), len(tasks), manifest["by_axis"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

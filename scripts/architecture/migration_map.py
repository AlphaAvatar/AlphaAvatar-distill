#!/usr/bin/env python3
"""The old -> new module map for the initialization cutover, and the mover.

    PYTHONPATH=src python scripts/architecture/migration_map.py --plan
    PYTHONPATH=src python scripts/architecture/migration_map.py --apply

`aadistill.init` and `aadistill.autoinit` between them hold 61 modules and
20,556 lines, and only some of it is initialization. Consolidating only the
initialization parts would leave `autoinit` standing, so the package could not
be removed and production would still import it. Every module therefore lands
somewhere:

* **initialization/** — the reusable domain, in dependency layers:
  `specs` <- `adapters` <- `device`/`calibration`/`statistics` <- `transforms`
  <- `operators` <- `planning`. Nothing in a lower layer imports a higher one.
* **governance/** — what a maintainer states vs what an issuer derives.
* **runtime/** — provider, device handoff, cost, environment, telemetry.
* **scripts/experiments/** — the experiment instances. Phase-A/B/C1 plans, arm
  definitions, seeds, replay digests, batteries and budgets are not reusable
  mechanisms and do not belong in `src/aadistill` at all.

The map is data so the move, the import rewrite and the verification all read
the same thing, and so a reviewer can see the whole cutover on one page.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: old module path -> new module path. Keys are repo-relative files.
MAP: dict[str, str] = {}


def _add(pairs: dict[str, str], old_root: str, new_root: str) -> None:
    for old, new in pairs.items():
        MAP[f"{old_root}/{old}"] = f"{new_root}/{new}"


# --- layer 1: specs and identities -----------------------------------------
_add({
    "arch.py": "specs/arch.py",
    "state.py": "specs/state.py",
    "artifact.py": "specs/artifact.py",
    "identity_collapse.py": "specs/identity_collapse.py",
}, "src/aadistill/autoinit", "src/aadistill/initialization")

# --- layer 2: family adapters ----------------------------------------------
_add({
    "adapters/__init__.py": "adapters/__init__.py",
    "adapters/qwen3.py": "adapters/qwen3.py",
}, "src/aadistill/autoinit", "src/aadistill/initialization")

# --- layer 3: device, calibration, statistics ------------------------------
_add({
    "device.py": "device.py",
    "calibration.py": "calibration/profiles.py",
    "calibration_items.py": "calibration/items.py",
    "datasets.py": "calibration/datasets.py",
    "stats.py": "statistics/spec.py",
    "reweight.py": "statistics/reweight.py",
}, "src/aadistill/autoinit", "src/aadistill/initialization")
_add({
    "collect.py": "statistics/collect.py",
    "attention_stats.py": "statistics/attention.py",
    "contribution.py": "statistics/contribution.py",
}, "src/aadistill/init", "src/aadistill/initialization")

# --- layer 4: mathematical transforms --------------------------------------
_add({
    "project.py": "transforms/project.py",
    "sandwich.py": "transforms/sandwich.py",
    "nll_gate.py": "transforms/nll_gate.py",
}, "src/aadistill/init", "src/aadistill/initialization")

# --- layer 5: operators -----------------------------------------------------
for name in ("__init__", "_common", "attention", "attention_activation", "base",
             "composite", "depth", "ffn", "width"):
    MAP[f"src/aadistill/autoinit/operators/{name}.py"] = \
        f"src/aadistill/initialization/operators/{name}.py"

# --- layer 6: planning, execution and measurement ---------------------------
_add({
    "search.py": "planning/search.py",
    "fixed_path.py": "planning/fixed_path.py",
    "ranking.py": "planning/ranking.py",
    "metrics.py": "planning/metrics.py",
    "recovery.py": "planning/recovery.py",
    "generation.py": "planning/generation.py",
    "generation_compat.py": "planning/generation_compat.py",
    "stage1_import.py": "planning/stage1_import.py",
    "stage1_selection.py": "planning/stage1_selection.py",
}, "src/aadistill/autoinit", "src/aadistill/initialization")

# --- generic governance (not initialization) --------------------------------
_add({
    "authorization.py": "authorization.py",
    "post_freeze.py": "post_freeze.py",
    "manifest.py": "artifact_manifest.py",
}, "src/aadistill/autoinit", "src/aadistill/governance")

# --- generic runtime (not initialization) -----------------------------------
_add({
    "device_handoff.py": "device_handoff.py",
    "cost.py": "cost.py",
    "pod_environment.py": "pod_environment.py",
    "cpu_test_env.py": "cpu_test_env.py",
    "leaf_durability.py": "leaf_durability.py",
    "telemetry.py": "telemetry.py",
    "staging_contract.py": "staging_contract.py",
}, "src/aadistill/autoinit", "src/aadistill/runtime")

# --- experiment instances: OUT of src/aadistill ------------------------------
#
# These own Phase/C1 plans, arm definitions, seeds, replay digests, batteries,
# stage order and budgets. None is a reusable mechanism.
EXPERIMENTS: dict[str, str] = {
    "c1_authorization.py": "phase_c1/authorization.py",
    "c1_bundle.py": "phase_c1/bundle.py",
    "c1_isolation.py": "phase_c1/isolation.py",
    "c1_packaging.py": "phase_c1/packaging.py",
    "c1_probe_results.py": "phase_c1/probe_results.py",
    "c1_scoring.py": "phase_c1/scoring.py",
    "c1_session.py": "phase_c1/session.py",
    "phase_a.py": "phase_a/plan.py",
    "phase_b.py": "phase_b/plan.py",
    "phase_b_continuation.py": "phase_b/continuation.py",
    "continuation.py": "recovery_continuation/plan.py",
    "recovery_continuation.py": "recovery_continuation/session.py",
    "measurement.py": "measurement/plan.py",
}
_add(EXPERIMENTS, "src/aadistill/autoinit", "scripts/experiments")


def module_of(path: str) -> str:
    """Import path for a repo-relative file, for src/ and scripts/ alike."""
    p = Path(path).with_suffix("")
    parts = list(p.parts)
    #: `src` and `scripts` are both roots that callers put on sys.path, so
    #: neither belongs in the importable name: src/aadistill/x.py is
    #: `aadistill.x`, scripts/experiments/phase_a/plan.py is
    #: `experiments.phase_a.plan`.
    if parts[0] in ("src", "scripts"):
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_map() -> dict[str, str]:
    """old module -> new module, longest first so prefixes rewrite correctly."""
    out = {module_of(o): module_of(n) for o, n in MAP.items()}
    # the packages themselves
    out["aadistill.autoinit.operators"] = "aadistill.initialization.operators"
    out["aadistill.autoinit.adapters"] = "aadistill.initialization.adapters"
    return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))


def plan() -> None:
    by_dest: dict[str, list[str]] = {}
    for old, new in sorted(MAP.items()):
        by_dest.setdefault(new.rsplit("/", 1)[0], []).append(old)
    for dest in sorted(by_dest):
        print(f"\n{dest}/   ({len(by_dest[dest])})")
        for old in by_dest[dest]:
            print(f"    {old.split('/')[-1]:34} <- {old}")
    print(f"\n{len(MAP)} files move; "
          f"{sum(1 for v in MAP.values() if v.startswith('scripts/'))} leave src/aadistill")


def apply() -> None:
    for old, new in sorted(MAP.items()):
        src, dst = REPO / old, REPO / new
        if not src.exists():
            print(f"  SKIP (already moved) {old}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=REPO, check=True)
    print(f"moved {len(MAP)} files")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.apply:
        apply()
    else:
        plan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

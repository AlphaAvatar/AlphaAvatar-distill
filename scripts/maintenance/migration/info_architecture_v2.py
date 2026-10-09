"""Round 2 of the 2026-10-08 information-architecture migration.

Round 1 (`info_architecture.py`) moved `scripts/`, `logs/` and `artifacts/`
to the owner-first layout. This module is the SAME migration extended to the
remaining three trees — `configs/`, `data/` and `docs/` — so that one logical
owner is discoverable at the same namespace in every tree it has a presence
in.

Like round 1, this file is the single declaration: the physical moves, the
relocation-registry pairs and the round's source-relocation record are all
derived from the tables below. Nothing else lists these moves.

Doctrine (unchanged from round 1):

* **Frozen records keep their freeze-time spellings.** A run manifest, a
  consumed authorization, a hash preimage or a closed result that names
  `configs/stage3/...` keeps saying so; ACCESS resolves through
  `logs/index.json :: historical_paths.map`.
* **Live readers move with the file.** Current loaders, launchers, tests and
  generators name the canonical path.
* **Content is never rewritten by a move.** Every pair below relocates bytes
  verbatim; content hashes are verified before and after for every data file,
  including gitignored ones.

Ownership calls that need their reasoning recorded:

* `configs/experiments/phase_a/source_sets.json` and
  `configs/autoinit/operator_ledger.json` land at STAGE level
  (`configs/stages/stage-1/`), not under a phase: both are live, program-wide
  registries (the current scorer/trainer/generation source sets; the operator
  ledger) that outlived the closed experiment that created them. Filing the
  LIVE contract under a closed experiment would misstate its owner.
* `configs/stage3/**` belongs to stage 3 even though stage-1 experiments
  consume its recovery configs daily: a stage pipeline configuration belongs
  to the stage that defines it, not to its downstream consumers.
* `data/eval_behavior_v0` and `data/stage3_pilot` are stage-3 material: their
  scientific construction (behavioural evaluation prompts; pilot recovery
  arms) is stage-3's, whatever later screening work reads them.
* The preflight / micro-preflight / device-canary sessions are stage-neutral
  engineering, so their specs live under `configs/shared/pod/`.
* `docs/core-provenance.md` and `docs/REPO_LAYOUT.md` are repository
  maintenance documentation; the autoinit references are stage-1 documents;
  the pod/session design documents are shared.
"""
from __future__ import annotations

import shutil
from pathlib import Path

S1 = "configs/stages/stage-1"

#: Whole directories whose every file has the same new owner. Each pair is a
#: PREFIX: ``old/X`` maps to ``new/X`` for every file under it. Merge
#: semantics — the destination may already exist; file collisions are errors.
CONFIG_DIR_MOVES: tuple[tuple[str, str], ...] = (
    ("configs/stage0", "configs/stages/stage-0"),
    ("configs/stage1", S1),
    ("configs/stage3", "configs/stages/stage-3"),
    ("configs/calibration", f"{S1}/calibration"),
    ("configs/infrastructure", "configs/shared/infrastructure"),
    ("configs/datasets", "configs/shared/datasets"),
    ("configs/architecture", "configs/maintenance/architecture"),
    ("configs/experiments/phase_c1", f"{S1}/phase_c1"),
    ("configs/experiments/phase_c3", f"{S1}/phase_c3"),
    ("configs/experiments/e5", "configs/stages/stage-3/e5"),
    ("configs/experiments/micro_preflight", "configs/shared/pod/micro_preflight"),
    ("configs/experiments/preflight", "configs/shared/pod/preflight"),
)

#: Single files. Basenames are preserved (round-1 convention): a redundant
#: name under its owner beats a rename that breaks every grep.
CONFIG_FILE_MOVES: tuple[tuple[str, str], ...] = (
    # --- live program-wide registries: STAGE level, see module docstring ---
    ("configs/experiments/phase_a/source_sets.json", f"{S1}/source_sets.json"),
    ("configs/autoinit/operator_ledger.json", f"{S1}/operator_ledger.json"),
    # --- the rest of phase_a's experiment configs ---
    ("configs/experiments/phase_a/preflight_plan.json", f"{S1}/phase_a/preflight_plan.json"),
    ("configs/experiments/phase_a/recovery_policy.json", f"{S1}/phase_a/recovery_policy.json"),
    # --- artifact specs -> their experiment owners ---
    ("configs/autoinit/phase_a_artifacts.json", f"{S1}/phase_a/phase_a_artifacts.json"),
    ("configs/autoinit/phase_a_artifacts_failed.json", f"{S1}/phase_a/phase_a_artifacts_failed.json"),
    ("configs/autoinit/phase_b_artifacts.json", f"{S1}/phase_b/phase_b_artifacts.json"),
    ("configs/autoinit/phase_b_artifacts_failed.json", f"{S1}/phase_b/phase_b_artifacts_failed.json"),
    ("configs/autoinit/c1_artifacts.json", f"{S1}/phase_c1/c1_artifacts.json"),
    ("configs/autoinit/c1_artifacts_failed.json", f"{S1}/phase_c1/c1_artifacts_failed.json"),
    ("configs/autoinit/c1_skip_predicate_classification.json",
     f"{S1}/phase_c1/c1_skip_predicate_classification.json"),
    ("configs/autoinit/c2_artifacts.json", f"{S1}/phase_c2/c2_artifacts.json"),
    ("configs/autoinit/c2_artifacts_failed.json", f"{S1}/phase_c2/c2_artifacts_failed.json"),
    ("configs/autoinit/c2_baseline_completion_artifacts.json",
     f"{S1}/phase_c2_baseline_completion/c2_baseline_completion_artifacts.json"),
    ("configs/autoinit/c2_baseline_completion_artifacts_failed.json",
     f"{S1}/phase_c2_baseline_completion/c2_baseline_completion_artifacts_failed.json"),
    ("configs/autoinit/c2_behavioural_artifacts.json",
     f"{S1}/phase_c2_behavioural/c2_behavioural_artifacts.json"),
    ("configs/autoinit/c2_behavioural_artifacts_failed.json",
     f"{S1}/phase_c2_behavioural/c2_behavioural_artifacts_failed.json"),
    ("configs/autoinit/c2_full_search_artifacts.json",
     f"{S1}/phase_c2_full_search/c2_full_search_artifacts.json"),
    ("configs/autoinit/c2_full_search_artifacts_failed.json",
     f"{S1}/phase_c2_full_search/c2_full_search_artifacts_failed.json"),
    ("configs/autoinit/c2_replay_artifacts.json", f"{S1}/phase_c2_replay/c2_replay_artifacts.json"),
    ("configs/autoinit/c2_replay_artifacts_failed.json",
     f"{S1}/phase_c2_replay/c2_replay_artifacts_failed.json"),
    ("configs/autoinit/c3_artifacts.json", f"{S1}/phase_c3/c3_artifacts.json"),
    ("configs/autoinit/c3_artifacts_failed.json", f"{S1}/phase_c3/c3_artifacts_failed.json"),
    ("configs/autoinit/a3_artifacts.json", f"{S1}/phase_a3/a3_artifacts.json"),
    ("configs/autoinit/a3_artifacts_failed.json", f"{S1}/phase_a3/a3_artifacts_failed.json"),
    ("configs/autoinit/continuation_artifacts.json",
     f"{S1}/recovery_continuation/continuation_artifacts.json"),
    ("configs/autoinit/continuation_artifacts_failed.json",
     f"{S1}/recovery_continuation/continuation_artifacts_failed.json"),
    ("configs/autoinit/continuation_b_artifacts.json",
     f"{S1}/continuation_b/continuation_b_artifacts.json"),
    ("configs/autoinit/continuation_b_artifacts_failed.json",
     f"{S1}/continuation_b/continuation_b_artifacts_failed.json"),
    ("configs/autoinit/d1_search_artifacts.json", f"{S1}/phase_d1/d1_search_artifacts.json"),
    ("configs/autoinit/d1_search_artifacts_failed.json",
     f"{S1}/phase_d1/d1_search_artifacts_failed.json"),
    ("configs/autoinit/d1_replay_artifacts.json", f"{S1}/phase_d1/d1_replay_artifacts.json"),
    ("configs/autoinit/d1_replay_artifacts_failed.json",
     f"{S1}/phase_d1/d1_replay_artifacts_failed.json"),
    ("configs/autoinit/measurement_artifacts.json",
     f"{S1}/measurement/measurement_artifacts.json"),
    ("configs/autoinit/device_canary_artifacts.json",
     "configs/shared/pod/device_canary_artifacts.json"),
    ("configs/autoinit/preflight_artifacts.json",
     "configs/shared/pod/preflight/preflight_artifacts.json"),
    ("configs/autoinit/preflight_artifacts_failed.json",
     "configs/shared/pod/preflight/preflight_artifacts_failed.json"),
    # --- phase_c2 variants: each authorization/asset doc to its variant owner ---
    ("configs/experiments/phase_c2/authorization.json", f"{S1}/phase_c2/authorization.json"),
    ("configs/experiments/phase_c2/frozen_assets.json", f"{S1}/phase_c2/frozen_assets.json"),
    ("configs/experiments/phase_c2/baseline_completion_authorization.json",
     f"{S1}/phase_c2_baseline_completion/baseline_completion_authorization.json"),
    ("configs/experiments/phase_c2/behavioural_frozen_assets.json",
     f"{S1}/phase_c2_behavioural/behavioural_frozen_assets.json"),
    ("configs/experiments/phase_c2/full_search_authorization.json",
     f"{S1}/phase_c2_full_search/full_search_authorization.json"),
    ("configs/experiments/phase_c2/replay_frozen_assets.json",
     f"{S1}/phase_c2_replay/replay_frozen_assets.json"),
    # --- validation configs -> their owning validations ---
    ("configs/validation/c2_state_eval_certification.json",
     f"{S1}/c2_state_eval_cert/c2_state_eval_certification.json"),
    ("configs/validation/c2_full_search_cuda.json",
     f"{S1}/c2_full_search_cuda/c2_full_search_cuda.json"),
    ("configs/validation/c2_full_search_performance.json",
     f"{S1}/c2_full_search_perf/c2_full_search_performance.json"),
    ("configs/validation/batching_refactor_cuda.json",
     f"{S1}/phase_c3/batching_refactor_cuda.json"),
    ("configs/validation/cuda_engineering.json",
     "configs/shared/validation/cuda_engineering.json"),
    ("configs/validation/cuda_engineering_deployment.json",
     "configs/shared/validation/cuda_engineering_deployment.json"),
    # --- the stage-3 recovery recipe ---
    ("configs/recipes/recovery.json", "configs/stages/stage-3/recipes/recovery.json"),
)

#: Datasets: the leaf directory name is part of the dataset's recorded
#: identity (manifests and run records spell it), so leaves move verbatim
#: under their owning stage. Bytes are hashed before and after, including
#: the gitignored train/val/calib payloads.
DATA_DIR_MOVES: tuple[tuple[str, str], ...] = (
    ("data/warmup", "data/stages/stage-0/warmup"),
    ("data/stage2", "data/stages/stage-2/stage2"),
    ("data/stage2_v1", "data/stages/stage-2/stage2_v1"),
    ("data/stage3_pilot", "data/stages/stage-3/stage3_pilot"),
    ("data/eval_behavior_v0", "data/stages/stage-3/eval_behavior_v0"),
)

DOC_FILE_MOVES: tuple[tuple[str, str], ...] = (
    ("docs/AUTOINIT_REFERENCE.md", "docs/stages/stage-1/AUTOINIT_REFERENCE.md"),
    ("docs/OPERATOR_PROMOTION_CYCLE.md", "docs/stages/stage-1/OPERATOR_PROMOTION_CYCLE.md"),
    ("docs/POST_PHASE_B_GENERALIZATION.md",
     "docs/stages/stage-1/POST_PHASE_B_GENERALIZATION.md"),
    ("docs/archive/HANDOFF_AUTOINITIALIZER_20260812.md",
     "docs/stages/stage-1/archive/HANDOFF_AUTOINITIALIZER_20260812.md"),
    ("docs/POD_SCRIPTS.md", "docs/shared/POD_SCRIPTS.md"),
    ("docs/SESSION_ARCHITECTURE.md", "docs/shared/SESSION_ARCHITECTURE.md"),
    ("docs/REPO_LAYOUT.md", "docs/maintenance/REPO_LAYOUT.md"),
    ("docs/core-provenance.md", "docs/maintenance/core-provenance.md"),
)

#: Configurations that stay put, each with the frozen dependency that keeps
#: it there. (Empty: round 2 found none — every config could move because
#; access resolves and current identity sets regenerate by their own
#: precedents. Kept as a table so a future exception is a decision with a
#: stated reason, not a silent omission.)
FROZEN_EXCEPTIONS: tuple[tuple[str, str], ...] = ()


def all_pairs() -> list[tuple[str, str]]:
    """Every (old, new) PREFIX pair this round adds to the relocation map."""
    return (list(CONFIG_DIR_MOVES) + list(CONFIG_FILE_MOVES)
            + list(DATA_DIR_MOVES) + list(DOC_FILE_MOVES))


def _move_file(repo: Path, old_rel: str, new_rel: str) -> None:
    src, dst = repo / old_rel, repo / new_rel
    if not src.is_file():
        raise SystemExit(f"missing source file: {old_rel}")
    if dst.exists():
        raise SystemExit(f"collision at destination: {new_rel}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def apply(repo_root: str | Path = ".") -> None:
    """Execute the physical moves. Directory pairs merge file-by-file so a
    pre-existing destination directory never swallows the source directory
    whole (the round-1 nesting bug)."""
    repo = Path(repo_root).resolve()
    for old_dir, new_dir in CONFIG_DIR_MOVES + DATA_DIR_MOVES:
        src = repo / old_dir
        if not src.is_dir():
            raise SystemExit(f"missing source dir: {old_dir}")
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = f.relative_to(src)
            _move_file(repo, str(f.relative_to(repo)), str(Path(new_dir) / rel))
        for d in sorted((p for p in src.rglob("*") if p.is_dir()),
                        key=lambda p: -len(p.parts)):
            d.rmdir()
        src.rmdir()
    for old_rel, new_rel in CONFIG_FILE_MOVES + DOC_FILE_MOVES:
        _move_file(repo, old_rel, new_rel)
    # empty parents left behind by per-file moves
    for leftover in ("configs/autoinit", "configs/experiments/phase_a",
                     "configs/experiments/phase_c2", "configs/experiments",
                     "configs/validation", "configs/recipes", "docs/archive"):
        d = repo / leftover
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


if __name__ == "__main__":
    import sys
    if "--apply" in sys.argv:
        apply(Path(__file__).resolve().parents[3])
        print("applied")
    else:
        for a, b in all_pairs():
            print(f"{a} -> {b}")

"""Run the Phase-A beam search against the real teacher and the frozen suite.

Importable rather than only executable, because the recovery rungs need the
**objects** the search produced, not a transcription of them.
`InitializationState` has no `from_dict` by design — the journal is append-only
evidence, not a serialization format — so a driver that shelled out to a search
script would have to rebuild candidate states from JSON and would lose
`admit_leaves`, whose whole job is to refuse an intermediate that cannot be a
recovery candidate. The search therefore runs in the driver's process and hands
back live states.

Resume is still exact: `BeamSearch` restores any state whose journal record
re-derives to the same artifact digest under the same suite hash, so re-entering
this function after a lost pod repeats the measurement of nothing that already
completed.

    PYTHONPATH=src python scripts/autoinit/phase_a_search.py \
        --workdir artifacts/autoinit/phase_a_search --out search_result.json

Executed directly it runs the same code path against whatever teacher it is
pointed at, which is how the harness is rehearsed at toy scale before a pod
exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from load_state_eval import load as load_suite  # noqa: E402

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1  # noqa: E402
from aadistill.autoinit.metrics import StateEvaluator  # noqa: E402
from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1  # noqa: E402
from aadistill.autoinit.search import (  # noqa: E402
    BeamSearch, Deadline, SearchConfig,
)
from aadistill.autoinit.state import make_control_state  # noqa: E402

#: Imported, not restated. They live in `phase_a_frozen` so a recovery
#: continuation can bind the same identities WITHOUT importing this module and
#: thereby putting `run_phase_a_search` within reach. Re-exported here so every
#: existing caller is unaffected.
#:
#: The searched operators. P=1: one calibration profile, so operators declaring
#: CalibrationNeed.NONE are offered once and the space is the 48 decomposed
#: paths the preregistration names.
from phase_a_frozen import (  # noqa: E402,F401
    CANONICAL_INIT, CANONICAL_INIT_SHA256, SEARCH_SEED, TARGET_GEOMETRY,
    TEACHER_ID, TEACHER_REVISION,
)


def as_operator_items(resolved):
    """Adapt `CalibrationProfile.resolve()` output to what the operators consume.

    The frozen mixture stores each item's tokens under **`ids`**, a plain list --
    that is the form `mixture_content_sha256` hashes and the form the pinned
    `d65c1f40...` content identity is derived from, so it is not changed. The
    operators call `i["input_ids"]` and pass it straight to
    `collector.process(ids.to(device))`, i.e. they want a `[1, T]` LongTensor.

    `dry_run_search.py` builds its own items already in the operator shape, so
    the mismatch was invisible to every zero-cost run and only appeared the first
    time the REAL profile reached an operator -- which was Phase-A attempt 5,
    on a paid pod. The adaptation is here rather than in `calibration.py` because
    the stored form is what the frozen content hash is defined over.
    """
    import torch

    out = []
    for item in resolved:
        ids = item.get("ids")
        if ids is None:
            raise KeyError(
                f"calibration item {item.get('item_id')!r} has no 'ids'; the "
                "frozen mixture stores tokens under that key")
        out.append({**item,
                    "input_ids": torch.tensor([ids], dtype=torch.long)})
    return out


def resolve_profiles(profile, profiles) -> tuple:
    """The profiles a run actually searches. One place decides, and it refuses
    to guess: `profile=` and `profiles=` together is an ambiguity, not a default."""
    if profile is not None and profiles is not None:
        raise ValueError(
            "pass profile= or profiles=, not both; two answers to 'which mixtures "
            "does this search branch over' cannot be reconciled here")
    if profiles is not None:
        active = tuple(profiles)
        if not active:
            raise ValueError("profiles= is empty; a search must branch over at least one")
        ids = [p.qualified_id for p in active]
        if len(set(ids)) != len(ids):
            raise ValueError(f"profiles= repeats a profile: {ids}")
        return active
    return (profile,) if profile is not None else (DOMAIN_BALANCED_V1,)


def build_calibration_loader(active_profiles, calibration_items, repo_root):
    """A loader that answers for the profile it is ASKED about.

    `calibration_items` may be a mapping keyed by `qualified_id` (any number of
    profiles) or a bare sequence (permitted only when exactly one profile is
    active, since a bare sequence cannot say which mixture it is). When omitted,
    each profile resolves itself from disk, hash-verified.

    A loader that ignored its argument was the historical shape and is what made
    the mislabeling above invisible; it is now impossible to construct here.
    """
    from collections.abc import Mapping

    ids = [p.qualified_id for p in active_profiles]
    if calibration_items is None:
        cache: dict = {}

        def load(profile):
            if profile.qualified_id not in cache:
                cache[profile.qualified_id] = as_operator_items(profile.resolve(repo_root))
            return cache[profile.qualified_id]
        return load

    if isinstance(calibration_items, Mapping):
        missing = [i for i in ids if i not in calibration_items]
        if missing:
            raise ValueError(
                f"calibration_items supplies {sorted(calibration_items)} but the "
                f"search branches over {ids}; missing {missing}")
        return lambda profile: calibration_items[profile.qualified_id]

    if len(active_profiles) != 1:
        raise ValueError(
            f"calibration_items is a bare sequence but the search branches over "
            f"{len(active_profiles)} profiles {ids}; a sequence cannot say which "
            "mixture it is. Pass a mapping keyed by qualified_id")
    only = ids[0]
    return lambda profile: calibration_items if profile.qualified_id == only else (
        _refuse(profile, only))


def _refuse(profile, only):
    raise ValueError(
        f"the loader was asked for {profile.qualified_id!r} but was built for "
        f"{only!r}; returning the wrong mixture is how a run gets labelled with "
        "one profile and fed another")


@dataclass
class PhaseASearch:
    """What the rungs need, as live objects plus a serializable summary."""

    result: Any
    control: Any
    top_n: Any
    summary: dict[str, Any]

    @property
    def leaves(self) -> list:
        return list(self.top_n.selected)


def run_phase_a_search(*, workdir: Path, state_eval: Path, top_n: int,
                       device: str = "cuda", repo_root: Path = REPO_ROOT,
                       teacher_id: str = TEACHER_ID,
                       teacher_revision: str = TEACHER_REVISION,
                       canonical_init: str = CANONICAL_INIT,
                       canonical_sha256: str | None = CANONICAL_INIT_SHA256,
                       teacher_loader=None, seed: int = SEARCH_SEED,
                       target_geometry: dict | None = None,
                       suite_bundle=None, calibration_items=None,
                       profile=None, profiles=None,
                       search_minutes: float | None = None,
                       ) -> PhaseASearch:
    """Search, then inject and measure the canonical control on the same suite.

    The control is measured here rather than assumed: `require_recovery_admissible`
    refuses a candidate that is not MEASURED, and a control carrying no
    hash-bound step-0 metrics could not be compared with the leaves it is
    supposed to be the baseline for.

    The last four arguments exist so this exact function can be executed at toy
    scale. They are NOT a configuration surface for a paid run: every one of them
    defaults to the frozen real value, and a rehearsal that stubbed the function
    out instead would prove nothing about the lines a pod executes. Four paid
    pods in this project have died in never-executed code.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    adapter = get_adapter("qwen3")
    suite, items, suite_manifest = (
        suite_bundle if suite_bundle is not None else load_suite(state_eval))
    target_spec = ArchSpec.of("qwen3", target_geometry or TARGET_GEOMETRY)

    if teacher_loader is None:
        def teacher_loader():                                   # noqa: F811
            kwargs = ({"revision": teacher_revision}
                      if teacher_id == TEACHER_ID else {})
            return AutoModelForCausalLM.from_pretrained(
                teacher_id, dtype=torch.bfloat16, **kwargs).to(device).eval()

    teacher = teacher_loader()
    evaluator = StateEvaluator(suite, items, device=device)
    evaluator.prime_reference(teacher)

    # Hash-verified on load: a calibration mixture that drifted would change every
    # operator's statistics without changing any recorded identity.
    #
    # `active_profiles` and the items are resolved TOGETHER. They used to be
    # independent: `active_profile` came from `profile or DOMAIN_BALANCED_V1`
    # while the items fell back to `DOMAIN_BALANCED_V1.resolve()` regardless, so
    # passing `profile=X` without `calibration_items=` produced a run LABELLED X
    # and FED the domain-balanced mixture. No caller did that -- every one passed
    # both or neither -- which is exactly why it survived to the point where
    # Phase B would be the first to wire a second profile.
    active_profiles = resolve_profiles(profile, profiles)
    calibration_loader = build_calibration_loader(
        active_profiles, calibration_items, repo_root)

    n = len(active_profiles)
    config = SearchConfig(
        run_id="autoinit.v1.phase_a", target_spec=target_spec,
        schedule=SCHEDULE_V1, seed=seed, workdir=Path(workdir),
        profiles=active_profiles, policy=PARETO_V1, suite=suite,
        device=device,
        notes={"purpose": "AutoInitializer Phase A, the preregistered search",
               "profiles": (f"P={n}; " + ("the 48 decomposed paths" if n == 1
                                          else f"{24 * (1 + n) * n * n} decomposed paths")),
               "profile_ids": ",".join(p.qualified_id for p in active_profiles)})

    # Runtime only, never hashed: `SearchConfig` fixes the search's identity and
    # a wall-clock budget is not part of it. `--search-minutes` was priced from
    # the start and reached only an affordability check; this is what makes it
    # bind while the search is running.
    search = BeamSearch(
        adapter=adapter, config=config, root_teacher_id=teacher_id,
        root_teacher_sha256=suite_manifest.get("teacher_sha256", "") or "0" * 64,
        root_loader=lambda: teacher,
        calibration_loader=calibration_loader,
        measurer=lambda model, digest: evaluator.evaluate(model, digest),
        deadline=Deadline.from_minutes(search_minutes))
    result = search.run()

    # --- the canonical control, injected by frozen hash ---------------------
    init_dir = Path(repo_root) / canonical_init
    control_cfg = AutoConfig.from_pretrained(str(init_dir))
    control_spec = adapter.spec_from_config(control_cfg)
    control_artifact = identify_checkpoint(
        init_dir, adapter=adapter, spec=control_spec,
        num_parameters=adapter.param_count(control_spec))
    control = make_control_state(
        control_id="qwen3_0p6b_init_v0", artifact=control_artifact,
        spec=control_spec, target_spec=target_spec,
        num_parameters=adapter.param_count(control_spec),
        root_teacher_id=teacher_id,
        root_teacher_sha256=suite_manifest.get("teacher_sha256", "") or "0" * 64,
        description=("the retained canonical initialization; a re-executed "
                     "composite is not the historical incumbent"),
        expected_single_file_sha256=canonical_sha256)
    control.attach_evaluation(
        evaluator.evaluate(adapter.load(str(init_dir), device=device),
                           control_artifact.artifact_digest))

    ranking = result.top_n(PARETO_V1, min(top_n, len(result.leaves)))

    summary = {
        "schema": "aadistill.autoinit.phase_a_search/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "seed": seed,
        "suite": suite.qualified_id,
        "suite_hash": suite.suite_hash,
        "schedule": SCHEDULE_V1.as_dict(),
        "policy": PARETO_V1.as_dict(),
        # Singular stays singular and means what it always meant. On a P>1 run it
        # is None rather than one of several: a consumer that reads it would
        # otherwise be told a two-profile search ran under one mixture, which is
        # the same class of lie the loader seam above was fixed to prevent.
        "calibration_profile": (active_profiles[0].qualified_id
                                if len(active_profiles) == 1 else None),
        "calibration_profiles": [p.qualified_id for p in active_profiles],
        "summary": result.summary(),
        "levels": [level.as_dict() for level in result.levels],
        "resumed_state_ids": list(result.resumed),
        "top_n": {
            "requested": top_n,
            "selected": [{"state_id": s.state_id, "path": s.path_label,
                          "artifact_digest": s.artifact_digest,
                          "single_shard_sha256": s.checkpoint_sha256,
                          "checkpoint_path": s.checkpoint_path,
                          "num_parameters": s.num_parameters}
                         for s in ranking.selected],
            "decisions": ranking.decisions,
        },
        "control": {
            "state_id": control.state_id,
            "provenance": control.provenance,
            "artifact_digest": control.artifact_digest,
            "single_shard_sha256": control.checkpoint_sha256,
            "frozen_sha256_verified": canonical_sha256 is not None,
        },
    }
    return PhaseASearch(result=result, control=control, top_n=ranking,
                        summary=summary)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--teacher", default=TEACHER_ID)
    ap.add_argument("--teacher-revision", default=TEACHER_REVISION)
    ap.add_argument("--canonical-init", default=CANONICAL_INIT)
    ap.add_argument("--no-verify-canonical-sha", action="store_true",
                    help="rehearsal only: the pinned hash belongs to the real "
                         "initialization and a toy stand-in cannot carry it")
    args = ap.parse_args()

    found = run_phase_a_search(
        workdir=Path(args.workdir),
        state_eval=Path(args.state_eval) if Path(args.state_eval).is_absolute()
        else REPO_ROOT / args.state_eval,
        top_n=args.top_n, device=args.device, teacher_id=args.teacher,
        teacher_revision=args.teacher_revision,
        canonical_init=args.canonical_init,
        canonical_sha256=None if args.no_verify_canonical_sha
        else CANONICAL_INIT_SHA256)
    Path(args.out).write_text(json.dumps(found.summary, indent=2, default=str) + "\n")
    print(json.dumps({"n_states": found.summary["summary"]["n_states"],
                      "n_complete_leaves": found.summary["summary"]["n_complete_leaves"],
                      "selected": [s["state_id"] for s in
                                   found.summary["top_n"]["selected"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

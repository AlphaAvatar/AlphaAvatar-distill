"""A Stage-0 search session: the forward baseline, run on one volume treatment.

    session = build_session(workdir=..., run_id=..., profiles=(p_base, p_primary))

The METHOD is the forward baseline (`shared.forward_baseline`): its operators,
scoring positions, distribution support, KL references, execution profile,
schedule and ranking are read from the resolved record and never retyped here.
What this experiment supplies is what the baseline leaves free (its
`session_terms` docstring): the target geometry, the state-evaluation suite,
the seed, the run identity and the calibration PROFILES a treatment searches
over. Operator order and the profile per operator stay free variables of the
search, exactly as in D1.

The geometry, seed, state-evaluation suite and root-teacher binding are the
same frozen owners the D-series used (read through `stages.phase_d1.d1_session`
and `stages.phase_a3.a3_session`, not copied), so a Stage-0 search differs from
a forward-baseline D1 search only in its calibration profiles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
PROFILES_DOC = REPO / "configs/stages/stage-0/calibration_volume/profiles.json"


class Stage0SessionError(RuntimeError):
    pass


@dataclass
class Stage0Session:
    evaluator: Any
    config: Any
    suite_content_sha256: str
    baseline: dict
    numerics: Any = None

    def as_record(self) -> dict[str, Any]:
        return {"config_hash": self.config.config_hash,
                "measurement_protocol_id": self.evaluator.measurement_protocol_id,
                "profiles": [p.qualified_id for p in self.config.profiles],
                "profile_hashes": {p.qualified_id: p.profile_hash
                                   for p in self.config.profiles},
                "suite_content_sha256": self.suite_content_sha256,
                "baseline": self.baseline}


def register_stage0_profiles(doc_path: Path = PROFILES_DOC) -> tuple:
    """Register this experiment's profiles from their data document."""
    from aadistill.initialization.calibration.profiles import load_profiles

    return load_profiles(json.loads(doc_path.read_text()))


def treatment_profiles(treatment: str, doc_path: Path = PROFILES_DOC) -> tuple:
    """The two profiles one volume treatment searches over, by its declaration."""
    from aadistill.initialization.calibration.profiles import get_profile

    register_stage0_profiles(doc_path)
    doc = json.loads(doc_path.read_text())
    ids = doc["treatments"][treatment]["profiles"]
    return tuple(get_profile(i) for i in ids)


def build_session(*, workdir: Path, run_id: str, profiles: tuple,
                  device: str = "cuda", repo_root: Path = REPO) -> Stage0Session:
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.planning.search import SearchConfig
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from shared import forward_baseline as FB
    from stages.phase_a3 import a3_session as A3S
    from stages.phase_d1 import d1_session as D1S

    if len(profiles) < 1:
        raise Stage0SessionError("a search needs at least one calibration profile")
    baseline = FB.selected(repo_root)
    D1S._register_frozen_operators()
    FB.register_operators(baseline)
    terms = FB.session_terms(baseline)
    if any(terms["operator_kl_reference"].values()):
        raise Stage0SessionError(
            "the forward baseline declares a non-parent operator KL reference; this "
            "session passes none, so it would silently measure a different estimand")
    suite, items, content = D1S.load_state_eval_suite(repo_root)
    evaluator = StateEvaluator(
        suite, items, device=device,
        vocab_size=D1S.teacher_vocab_size(repo_root),
        suite_content_sha256=content,
        position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"],
        execution=terms["execution"],
        numerics=terms["numerics"],
        reference_strategy=ReferenceStrategy.RECOMPUTE)
    spec = A3S.path_spec(workdir_device=device)
    config = SearchConfig(
        run_id=run_id,
        target_spec=spec.target_spec,
        schedule=terms["schedule"],
        seed=spec.seed,
        workdir=Path(workdir),
        profiles=tuple(profiles),
        policy=terms["ranking_policy"],
        suite=suite,
        position_policy=terms["position_policy"],
        distribution_support=terms["distribution_support"],
        measurement_protocol_id=evaluator.measurement_protocol_id,
        allowed_impls=terms["allowed_impls"],
        device=device)
    return Stage0Session(evaluator=evaluator, config=config,
                         suite_content_sha256=content, baseline=terms["baseline"],
                         numerics=terms["numerics"])

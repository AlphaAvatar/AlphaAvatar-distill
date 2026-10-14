"""The D2 formal search session: D1's search with TWO declared changes.

D2 (preregistration v2, contract revision 6) minimizes the frozen teacher's
full-vocabulary-confidence-weighted KL against the ROOT reference: DEPTH's
reference moves from the parent to the root, and the teacher's confidence
weights the KL and -- full scope -- the FFN/WIDTH/ATTENTION activation
statistics. Every other term -- the frozen operators, the calibration
profiles, the beam, the seed, the support, the position policy, the batch and
the suite -- is D1's, and this module reuses D1's helpers for them rather
than re-declaring them. What it owns:

* the objective, handed to the search config AND the state evaluator, and the
  ranking policy that ranks on it (`OBJECTIVE_PARETO_V1`) -- the core refuses
  any disagreement among the three; the search builds the root reference
  stores itself, from the teacher it is given;
* the preregistration as the design every check reads, so the session is
  checked against the frozen record rather than against this module;
* the row selection (`prediction_rows`), an execution choice the
  preregistration names.

Nothing here hardcodes a model: geometry and vocabulary come from the frozen
A3 path spec, as for D1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]

EXPERIMENT_ID = "phase_d2"
STAGE_ID = "1"
PREREGISTRATION = "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v2.json"
POD_WORKSPACE = "/workspace"
STATUS_PATH = f"{POD_WORKSPACE}/autoinit_d2.status"
DRIVER_JOB_ID = "autoinit_d2"
#: D2 scores the supervised targets only -- D1's treatment policy, which the
#: preregistration freezes as `positions.supervised_target_v1@v1`.
POSITION_POLICY = "positions.supervised_target_v1@v1"
PREDICTION_ROWS = ("all_v1", "scored_v1")


class D2SessionError(RuntimeError):
    """The session does not implement the preregistered design."""


def preregistration(repo_root: str | Path = REPO) -> dict[str, Any]:
    path = Path(repo_root) / PREREGISTRATION
    if not path.is_file():
        raise D2SessionError(f"{PREREGISTRATION} is missing; a D2 session has "
                             "no design to implement")
    return json.loads(path.read_text())


def preregistration_hash(repo_root: str | Path = REPO) -> str:
    return str(preregistration(repo_root)["preregistration_hash"])


def objective():
    from stages.d_series import objectives as O

    return O.D2_OBJECTIVE


def ranking_policy():
    from stages.d_series import objectives as O

    return O.OBJECTIVE_PARETO_V1


def position_policy():
    from aadistill.initialization.scoring.positions import get_position_policy

    return get_position_policy(POSITION_POLICY)


def execution(prediction_rows: str = "all_v1"):
    """D1's batch and packing, plus the row selection the run declares."""
    from aadistill.initialization.execution import ExecutionConfig
    from stages.d_series import scoring_protocol as SP

    if prediction_rows not in PREDICTION_ROWS:
        raise D2SessionError(f"prediction_rows {prediction_rows!r} is not one of "
                             f"{PREDICTION_ROWS}")
    return ExecutionConfig(
        micro_batch_size=SP.D_SERIES_MICRO_BATCH_SIZE,
        calibration_batch_packing=SP.D_SERIES_BATCH_PACKING,
        prediction_rows=prediction_rows)


@dataclass
class D2Session:
    """A wired D2 search session. Built by `build_session`, never by hand."""

    evaluator: Any
    config: Any
    suite_content_sha256: str
    preregistration_hash: str
    prediction_rows: str
    #: Filled by the driver from `BeamSearch.root_reference_evidence()`.
    root_reference_evidence: Any = None

    def measurer(self):
        """The evaluator as the search's measurer, attributes VISIBLE.

        D1 passed a lambda, which hid the evaluator's support and objective
        from `BeamSearch`'s construction-time checks; they then ran only per
        measurement. This exposes them so a mismatch fails before the root.
        """
        return _Measurer(self.evaluator)

    def as_record(self) -> dict[str, Any]:
        from aadistill.initialization.scoring.positions import policy_config
        from stages.phase_d1 import d1_session as D1S

        cfg = self.config
        return {
            "experiment_id": EXPERIMENT_ID, "stage_id": STAGE_ID,
            "preregistration_hash": self.preregistration_hash,
            "suite_content_sha256": self.suite_content_sha256,
            "measurement_protocol_id": cfg.measurement_protocol_id,
            "config_hash": cfg.config_hash,
            "scoring_objective": cfg.scoring_objective.as_dict(),
            "objective_hash": cfg.scoring_objective.objective_hash,
            "statistics_weighted": bool(
                cfg.scoring_objective.weights_activation_statistics),
            "kl_reference": cfg.scoring_objective.kl_reference,
            "ranking_policy": {"id": cfg.policy.qualified_id,
                               "hash": cfg.policy.policy_hash},
            "position_policy": policy_config(cfg.position_policy),
            "distribution_support": cfg.distribution_support.as_dict(),
            "execution": {
                "micro_batch_size": self.evaluator.execution.micro_batch_size,
                "calibration_batch_packing":
                    self.evaluator.execution.calibration_batch_packing,
                "prediction_rows": self.prediction_rows},
            "root_teacher": D1S.root_teacher_identity(),
            "beam": {"width": cfg.schedule.width,
                     "warmup_levels": cfg.schedule.warmup_levels},
            "profiles": [p.qualified_id for p in cfg.profiles],
            "allowed_impls": list(cfg.allowed_impls or ()),
            "seed": cfg.seed,
        }


class _Measurer:
    def __init__(self, evaluator) -> None:
        self._evaluator = evaluator
        self.distribution_support = evaluator.distribution_support
        self.scoring_objective = evaluator.scoring_objective
        self.measurement_protocol_id = evaluator.measurement_protocol_id

    def __call__(self, model, digest):
        return self._evaluator.evaluate(model, digest)


def build_session(*, workdir: Path, run_id: str, device: str = "cuda",
                  prediction_rows: str = "all_v1",
                  repo_root: str | Path = REPO,
                  reference_strategy: Any = None) -> D2Session:
    """The ONE place a D2 search is wired: D1's wiring plus the objective."""
    from aadistill.initialization.calibration.profiles import get_profile
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.planning.ranking import SCHEDULE_V1
    from aadistill.initialization.planning.search import SearchConfig
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from shared.calibration import register_builtin_profiles
    from stages.d_series import scoring_protocol as SP
    from stages.phase_a3 import a3_session as A3S
    from stages.phase_d1 import d1_session as D1S

    doc = preregistration(repo_root)
    D1S._register_frozen_operators()
    register_builtin_profiles()
    suite, items, content = D1S.load_state_eval_suite(repo_root)
    policy = position_policy()
    obj = objective()
    exec_cfg = execution(prediction_rows)
    evaluator = StateEvaluator(
        suite, items, device=device,
        vocab_size=D1S.teacher_vocab_size(repo_root),
        suite_content_sha256=content, position_policy=policy,
        distribution_support=SP.D_SERIES_SUPPORT, execution=exec_cfg,
        numerics=D1S.numerics(),
        #: RECOMPUTE, as every D-series search records: the O(T*K) sketch
        #: cache inside it is the Top-K saving, not this strategy.
        reference_strategy=(reference_strategy or ReferenceStrategy.RECOMPUTE),
        scoring_objective=obj)
    spec = A3S.path_spec(workdir_device=device)
    config = SearchConfig(
        run_id=run_id, target_spec=spec.target_spec, schedule=SCHEDULE_V1,
        seed=spec.seed, workdir=Path(workdir),
        profiles=tuple(get_profile(p) for p in sorted(doc["search"]["profiles"])),
        policy=ranking_policy(), suite=suite, position_policy=policy,
        distribution_support=SP.D_SERIES_SUPPORT,
        measurement_protocol_id=evaluator.measurement_protocol_id,
        allowed_impls=tuple(sorted(doc["search"]["frozen_implementations"]
                                   .values())),
        device=device, scoring_objective=obj)
    return D2Session(evaluator=evaluator, config=config,
                     suite_content_sha256=content,
                     preregistration_hash=doc["preregistration_hash"],
                     prediction_rows=prediction_rows)


def assert_session_contract(session: D2Session,
                            repo_root: str | Path = REPO) -> dict[str, Any]:
    """Check a built session against the PREREGISTRATION. Raises, never warns."""
    from stages.d_series import scoring_protocol as SP
    from stages.phase_d1 import search_space as d1

    doc = preregistration(repo_root)
    cfg, ev = session.config, session.evaluator
    obj, pol = objective(), ranking_policy()
    problems: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    check(doc.get("version") == 2 and not str(doc.get("status", ""))
          .startswith("SUPERSEDED"),
          "the preregistration is not the live v2 (v1 is SUPERSEDED -- NEVER "
          "EXECUTED)")
    check(obj.objective_hash != doc.get("supersedes", {}).get("objective_hash"),
          "the session carries the SUPERSEDED student-uncertainty objective")
    check(bool(obj.requires_root_reference)
          and (obj.weights_activation_statistics
               == (doc.get("scope_decision", {}).get("frozen") == "full")),
          "the objective's KL reference or statistics scope is not the "
          "preregistered one")

    check(bool(cfg.measurement_protocol_id),
          "SearchConfig.measurement_protocol_id is unset")
    check(cfg.measurement_protocol_id == ev.measurement_protocol_id,
          "the search and the evaluator declare different protocol ids")
    check(getattr(ev, "suite_content_sha256", None) == session.suite_content_sha256
          and session.suite_content_sha256 not in (None, "", "unbound"),
          "the frozen suite's content hash is not bound into the evaluator")
    for who, support in (("search", cfg.distribution_support),
                         ("evaluator", ev.distribution_support)):
        check(support.as_dict() == SP.D_SERIES_SUPPORT.as_dict()
              == doc["search"]["support"],
              f"the {who} reduces over {support}, not the preregistered support")
    want = position_policy()
    check(want.qualified_id == doc["search"]["position_policy"],
          "the session's position policy is not the preregistered one")
    for who, p in (("search", cfg.position_policy),
                   ("evaluator", ev.position_policy)):
        check(p.policy_hash == want.policy_hash,
              f"the {who} scores {p.qualified_id}, not {want.qualified_id}")
    for who, o in (("search", cfg.scoring_objective),
                   ("evaluator", ev.scoring_objective)):
        check(o is not None and o.objective_hash == obj.objective_hash
              == doc["objective"]["objective_hash"],
              f"the {who} does not carry the preregistered D2 objective")
    check(cfg.policy.policy_hash == pol.policy_hash
          == doc["search"]["ranking_policy"]["hash"],
          f"the ranking policy is {cfg.policy.qualified_id}, not the "
          "preregistered objective ranking")
    execution_doc = doc["search"]["execution"]
    check(ev.execution.micro_batch_size == execution_doc["micro_batch_size"],
          "the micro-batch is not the preregistered one")
    check(ev.execution.calibration_batch_packing
          == execution_doc["calibration_batch_packing"],
          "the packing is not the preregistered one")
    check(ev.execution.prediction_rows == session.prediction_rows,
          "the evaluator's row selection is not the session's")
    beam = doc["search"]["beam"]
    check(cfg.schedule.width == beam["width"]
          and cfg.schedule.warmup_levels == beam["warmup_levels"],
          f"the beam is {cfg.schedule.width}/{cfg.schedule.warmup_levels}, not "
          f"{beam['width']}/{beam['warmup_levels']}")
    check(int(cfg.seed) == int(doc["search"]["seed"]),
          f"the seed is {cfg.seed}, not D1's {doc['search']['seed']}")
    frozen = set(doc["search"]["frozen_implementations"].values())
    check(set(cfg.allowed_impls or ()) == frozen
          == set(d1.size_report()["frozen_implementations"].values()),
          "the operator set is not D1's frozen set")
    check({p.qualified_id for p in cfg.profiles} == set(doc["search"]["profiles"]),
          "the calibration profiles are not the preregistered ones")
    check(session.preregistration_hash == doc["preregistration_hash"],
          "the session was built against another preregistration revision")
    if problems:
        raise D2SessionError("the D2 session does not implement its "
                             "preregistration:\n  - " + "\n  - ".join(problems))
    return {
        "checked": [
            "protocol id declared and equal to the evaluator's",
            "suite content hash bound", "D-series support on both",
            "supervised-target policy on both",
            "the D2 objective on both, equal to the preregistered hash",
            "the live v2 preregistration, not the superseded objective",
            "root KL reference and the frozen statistics scope",
            "objective ranking policy, preregistered hash",
            "micro-batch, packing and row selection", "beam and seed",
            "D1's frozen operators and profiles", "preregistration revision"],
        "preregistration_hash": session.preregistration_hash,
        "measurement_protocol_id": cfg.measurement_protocol_id,
        "config_hash": cfg.config_hash,
        "objective_hash": obj.objective_hash,
        "ranking_policy_hash": pol.policy_hash,
        "_authorizes": "nothing; a wired session is not a funded one",
    }

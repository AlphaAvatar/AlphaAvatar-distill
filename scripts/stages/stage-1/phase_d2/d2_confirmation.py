"""D2's SHARED CONFIRMATION rung: the candidate, the probes, the verdict. `$0`.

Every fact is READ from the record that owns it, never typed here:

    the candidate        the closed screening run's frozen ranking -- the one
                         arm it advanced (ONE_CANDIDATE_ADVANCES), bound by the
                         ranking file's sha256 and the screening contract hash
    its initialization   the D2 search's committed selection (four identities)
    seeds, probes, rule  D2 preregistration v3 `behaviour.confirmation`
    the shared protocol  `d_series_shared_eval@v1`'s `shared_confirmation`
                         rung: battery role, evaluation identity, B's three
                         imported units (q2's descriptive), the statistics
    recovery, diagnostics the screening rung's (`d2_screening.py`): one
                         recovery recipe and one diagnostic for every D2 probe

WHAT THIS RUNG IS. The advancing candidate x the three shared confirmation
seeds = three probes, each a recovery run from the candidate's INITIAL
checkpoint (never a screening-trained weight), generated and scored under the
shared confirmation rung's evaluation protocol. The verdict is C0's three-way
rule over the candidate's rows paired with B's imported rows
(`shared_evaluation.confirmation_verdict`): SHARED_GO nominates for a fresh
independent confirmation, never a promotion; NO_GO and INCONCLUSIVE are
terminal for this candidate on these rows.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stages.phase_d2 import d2_screening as SC
from stages.phase_d2.d2_screening import (  # noqa: F401  (the driver's surface)
    ARM_FILES, EXPERIMENT_ID, POD_ARMS_REL, REPO_ROOT, STAGE_ID,
    TRAINING_RECORDS, diagnostics, preregistration, protocol, recovery,
    selection,
)

RUNG = "confirmation"
SHARED_RUNG = SC.SHARED_RUNGS[RUNG]
#: The screening run whose frozen ranking names the candidate. One rung, one
#: completed screening attempt (`families/d_series/screening_completion.py`).
SCREENING_RUN_ID = "d2_screening_20261015_203346"
RANKING_REL = (f"logs/stages/stage-1/phase_d2/runs/{SCREENING_RUN_ID}/"
               "evidence/d2_screening_ranking.json")
ADVANCES = "ONE_CANDIDATE_ADVANCES"


class D2ConfirmationError(SC.D2ScreeningError):
    """A confirmation premise does not hold. Raised before anything is
    priced."""


def behaviour(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return SC.behaviour(repo_root)


def advancing(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The one candidate the frozen screening advanced, from its ranking.

    The ranking must be the screening run's, bound to the screening contract
    this tree still derives, and must advance exactly one arm of the committed
    field. Its sha256 is bound into this rung's contract.
    """
    path = Path(repo_root) / RANKING_REL
    if not path.is_file():
        raise D2ConfirmationError(f"{RANKING_REL} is missing: no screening "
                                  "ranking names a candidate")
    data = path.read_bytes()
    ranking = json.loads(data)
    screening = SC.contract_hash(SC.session_contract(repo_root,
                                                     verify_bytes=False))
    selection_ = ranking.get("selection") or {}
    problems = [why for ok, why in (
        (ranking.get("run_id") == SCREENING_RUN_ID,
         f"the ranking is run {ranking.get('run_id')!r}"),
        (ranking.get("contract_hash") == screening,
         "the ranking binds another screening contract"),
        (selection_.get("outcome") == ADVANCES,
         f"the screening outcome is {selection_.get('outcome')!r}"),
        (selection_.get("advanced") is True, "nothing advanced"))
        if not ok]
    field = {a.arm_id: a for a in SC.arms(repo_root)}
    arm = field.get(selection_.get("arm"))
    if arm is None or arm.state_id != selection_.get("state_id"):
        problems.append(f"the advanced arm {selection_.get('arm')!r} / "
                        f"{selection_.get('state_id')!r} is not a committed "
                        "member")
    if problems:
        raise D2ConfirmationError("no confirmable candidate: "
                                  + "; ".join(problems))
    return {"arm": arm.arm_id, "state_id": arm.state_id,
            "ranking_rel": RANKING_REL,
            "ranking_sha256": hashlib.sha256(data).hexdigest(),
            "screening_contract_hash": screening,
            "screening_delta_vs_b": selection_.get("delta_vs_b"),
            "_screening_estimate_is_selected": (
                "the maximum of four near-sibling estimates: optimistic by "
                "construction (winner's curse); the confirmation decides")}


def arms(repo_root: Path = REPO_ROOT, *,
         arm_root: str | Path | None = None) -> tuple[Any, ...]:
    """The ONE candidate, at its INITIAL checkpoint (the search product)."""
    chosen = advancing(repo_root)["arm"]
    return tuple(a for a in SC.arms(repo_root, arm_root=arm_root)
                 if a.arm_id == chosen)


def seeds(repo_root: Path = REPO_ROOT) -> tuple[int, ...]:
    return SC.seeds(repo_root, RUNG)


def battery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return SC.battery(repo_root, RUNG)


def comparator_units(repo_root: Path = REPO_ROOT) -> dict[str, list[dict]]:
    return SC.comparator_units(repo_root, RUNG)


def evaluation_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return SC.evaluation_identity(repo_root, RUNG)


def probes(repo_root: Path = REPO_ROOT, *,
           arm_root: str | Path | None = None) -> tuple[Any, ...]:
    from stages.phase_d1.behavioural import Probe

    out = tuple(Probe(probe_id=f"d2_confirmation_{a.arm_id}_s{seed}",
                      arm_id=a.arm_id, seed=seed, rung=RUNG,
                      checkpoint_dir=a.checkpoint_dir,
                      battery_role=battery(repo_root)["role"])
                for a in arms(repo_root, arm_root=arm_root)
                for seed in seeds(repo_root))
    declared = int(behaviour(repo_root)[RUNG]["probes"])
    if len(out) != declared:
        raise D2ConfirmationError(
            f"the schedule yields {len(out)} probes and the preregistration "
            f"declares {declared}")
    return out


def rule(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen verdict's parameters, each read and cross-checked."""
    from stages.d_series import shared_evaluation as SE

    record = protocol(repo_root)
    stats = record["statistics"]
    conf = behaviour(repo_root)[RUNG]
    alpha = float(conf["alpha_one_sided"])
    per_experiment = float(stats["family"]["per_experiment_alpha_one_sided"])
    derived_seed = SE.bootstrap_seed(record["protocol_hash"], EXPERIMENT_ID,
                                     SHARED_RUNG)
    problems = [why for ok, why in (
        (alpha == per_experiment,
         f"the preregistration's alpha {alpha} is not the protocol's "
         f"per-experiment {per_experiment}"),
        (int(conf["bootstrap_seed"]) == derived_seed,
         f"the registered bootstrap seed {conf['bootstrap_seed']} is not the "
         f"protocol rule's {derived_seed}"),
        (conf["outcomes"] == stats["confirmation_outcomes"],
         "the preregistration's outcomes are not the protocol's"))
        if not ok]
    if problems:
        raise D2ConfirmationError("; ".join(problems))
    return {"endpoint": stats["endpoint"], "estimand": stats["estimand"],
            "sesoi": float(stats["sesoi"]), "alpha_one_sided": alpha,
            "bootstrap_seed": int(conf["bootstrap_seed"]),
            "bootstrap_iterations": int(stats["bootstrap_iterations"]),
            "seed_robustness_min_positive": int(
                stats["seed_robustness_min_positive"]),
            "guardrails": stats["guardrails"], "outcomes": conf["outcomes"],
            "secondary": stats["secondary_comparison"],
            "_frozen": ("no threshold moves because the screening margin was "
                        "small, no seed is added, and an INCONCLUSIVE is "
                        "terminal on these rows")}


def require_arms_present(repo_root: Path = REPO_ROOT, *,
                         arm_root: str | Path | None = None) -> dict[str, Any]:
    """The candidate's INITIAL checkpoint carries all four identities."""
    from stages.phase_d1.behavioural_materialize import (
        D1MaterializeError, _verify_bytes_at,
    )

    out = []
    for arm in arms(repo_root, arm_root=arm_root):
        directory = Path(arm.checkpoint_dir)
        if not directory.is_dir():
            raise D2ConfirmationError(
                f"{arm.arm_id}: no checkpoint at {directory}")
        try:
            got = _verify_bytes_at(directory, dict(arm.identities),
                                   what=f"{arm.arm_id} ({arm.state_id})")
        except D1MaterializeError as exc:
            raise D2ConfirmationError(str(exc)) from exc
        out.append({"arm": arm.arm_id, "state_id": arm.state_id,
                    "checkpoint_dir": str(directory),
                    "identities_checked": got["verified"]})
    return {"n_arms": len(out), "arms": out}


def session_contract(repo_root: Path = REPO_ROOT, *,
                     arm_root: str | Path | None = None,
                     verify_bytes: bool = True) -> dict[str, Any]:
    """Everything this rung is bound to. LOCATION-FREE (see the screening's)."""
    from stages.d_series import shared_evaluation as SE

    record = protocol(repo_root)
    drift = SE.tree_matches(record, repo_root)
    if any(drift.values()):
        raise D2ConfirmationError(
            f"this tree's trainer/generator/scorer differ from the shared "
            f"protocol's pins: {drift}")
    if verify_bytes:
        require_arms_present(repo_root, arm_root=arm_root)
    sel = selection(repo_root)
    prereg = preregistration(repo_root)
    bat = battery(repo_root)
    field = arms(repo_root, arm_root=arm_root)
    return {
        "schema": "aadistill.phase_d2.confirmation_contract/v1",
        "experiment_id": EXPERIMENT_ID,
        "rung": RUNG,
        "preregistration_hash": prereg["preregistration_hash"],
        "shared_protocol": {"id": f"{record['protocol_id']}@v"
                                  f"{record['version']}",
                            "protocol_hash": record["protocol_hash"],
                            "rung": SHARED_RUNG},
        "candidate": advancing(repo_root),
        "field": {
            "search_run_id": SC.SEARCH_RUN_ID,
            "selection_sha256": sel["selection_sha256"],
            "arms": [{"arm": a.arm_id, "quality_position": a.quality_position,
                      "state_id": a.state_id, "identities": dict(a.identities)}
                     for a in field],
            "_initial_checkpoint": (
                "every probe trains from the candidate's committed search "
                "product, verified at all four identities; no screening "
                "probe's trained weights are read"),
        },
        "seeds": list(seeds(repo_root)),
        "probes": [{"probe_id": p.probe_id, "arm": p.arm_id, "seed": p.seed}
                   for p in probes(repo_root, arm_root=arm_root)],
        "n_probes": int(behaviour(repo_root)[RUNG]["probes"]),
        "recovery": recovery(repo_root),
        "battery": {k: bat[k] for k in (
            "family_id", "family_content_id", "allocation_rule_id", "role",
            "item_ids_sha256", "n_prompts", "n_scorable", "per_stratum",
            "root", "files")},
        "evaluation": evaluation_identity(repo_root),
        "comparator": comparator_units(repo_root),
        "rule": rule(repo_root),
        "diagnostics": diagnostics(repo_root),
        "_decides_nothing": (
            "the candidate is the frozen screening's, the seeds, probe count, "
            "alpha and bootstrap seed the preregistration's, the battery, "
            "recovery, evaluation identity and statistics the shared "
            "protocol's"),
    }


def contract_hash(contract: Mapping[str, Any]) -> str:
    return SC.contract_hash(contract)


def verdict(candidate_rows: Mapping[int, Sequence[Mapping[str, Any]]], *,
            repo_root: Path = REPO_ROOT,
            evidence_root: Path | None = None) -> dict[str, Any]:
    """The frozen three-way verdict, and q2 as a no-decision secondary.

    `candidate_rows` is the candidate's per-sample rows per confirmation seed.
    B's and q2's rows are imported and hash-checked by the shared protocol.
    """
    from stages.d_series import shared_evaluation as SE

    record = protocol(repo_root)
    frozen = rule(repo_root)
    arm = advancing(repo_root)["arm"]
    b_rows = SE.load_reference_rows(SHARED_RUNG, "B", repo_root=repo_root,
                                    evidence_root=evidence_root)
    q2_rows = SE.load_reference_rows(SHARED_RUNG, "q2", repo_root=repo_root,
                                     evidence_root=evidence_root)
    primary = SE.confirmation_verdict(
        arm, candidate_rows, b_rows, protocol=record,
        alpha=frozen["alpha_one_sided"],
        bootstrap_seed_value=frozen["bootstrap_seed"], comparator="B")
    secondary = SE.confirmation_verdict(
        arm, candidate_rows, q2_rows, protocol=record,
        alpha=frozen["alpha_one_sided"],
        bootstrap_seed_value=frozen["bootstrap_seed"], comparator="q2")
    return {"outcome": SE.shared_outcome(primary["decision"]),
            "primary_vs_B": primary,
            "secondary_vs_q2_descriptive": {
                **secondary, "_role": ("no decision role and no alpha: "
                                       "reported with its interval only")},
            "rule": frozen,
            "_shared_go_is_not_a_promotion": (
                "SHARED_GO nominates the candidate for a fresh independent "
                "confirmation on its reserved battery; nothing is promoted")}


__all__ = ["ADVANCES", "D2ConfirmationError", "RUNG", "SCREENING_RUN_ID",
           "SHARED_RUNG", "advancing", "arms", "battery", "comparator_units",
           "contract_hash", "evaluation_identity", "probes",
           "require_arms_present", "rule", "seeds", "session_contract",
           "verdict"]

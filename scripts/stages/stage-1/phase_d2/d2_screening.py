"""D2's behavioural SCREENING rung: the field, the probes, what binds them. `$0`.

Every fact is READ from the record that owns it, never typed here:

    the candidate field       the D2 search's committed selection (Top-4,
                              quality order, four identities per member)
    seeds, probes, rule       D2 preregistration v3 `behaviour.screening`
    the shared protocol       `d_series_shared_eval@v1` (self-hash verified):
                              the battery role, the recovery block, the
                              evaluation identity of the rung, B's units
    the recovery recipe       `shared.recipes.E1_KD_HEAVY_0860K`, checked
                              against the protocol's recipe and data digests
    the battery               the realized D-series family (`d1_screening`)
    the diagnostic thresholds the reference calibration record

WHAT THIS RUNG IS. Four candidates x the two shared screening seeds = eight
probes, each a recovery run under the frozen recipe FROM a verified D2
initialization, generated and scored under the shared rung's evaluation
protocol. The comparator is B's two IMPORTED screening units -- not retrained,
not regenerated; q2's are descriptive only. Exactly one candidate advances to
the shared confirmation, or none. Nothing here promotes.

WHAT THE CANDIDATES ARE, statistically. All four share
RESIDUAL_WIDTH(domain_balanced) -> DEPTH(domain_balanced) and differ only in
which ~2.5% of FFN neurons and ~2-4% of query heads they keep
(`evidence/candidate_differences.json`). Their probes are therefore highly
correlated: eight probes are NOT eight independent architectures, and the
ranking is a selection among near-siblings, read with the paired seeds.

Nothing here decides anything; `rank` applies the frozen rule mechanically.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

EXPERIMENT_ID = "phase_d2"
STAGE_ID = "1"
RUNG = "screening"
#: The shared protocol's name for the rung this session measures on.
SHARED_RUNG = "shared_screening"
#: The search whose committed field this rung measures. One search, one field.
SEARCH_RUN_ID = "d2_search_20261015_051722"
SEARCH_RUN_REL = f"logs/stages/stage-1/phase_d2/runs/{SEARCH_RUN_ID}"
SELECTION_REL = f"{SEARCH_RUN_REL}/evidence/stage1_selection.json"
PROTOCOL_REL = ("logs/stages/stage-1/families/d_series/plans/"
                "d_series_shared_eval_v1.json")
CALIBRATION_REL = ("logs/stages/stage-1/families/d_series/analyses/"
                   "d2_d3_reference_calibration.json")
#: Where the relay stages each arm on the pod, repository-relative. The
#: launcher's relay inputs and the driver's `--arm-root` both read this.
POD_ARMS_REL = "artifacts/stages/stage-1/phase_d2/arms"
#: Every arm is a weight-only checkpoint directory of exactly these files.
ARM_FILES = ("model.safetensors", "config.json", "generation_config.json")
#: The training records the OFF-POD RANKING reads, in each probe's root: the run
#: manifest (training identity and the trained config, hash-bound) and the
#: completion record (training provenance, bound into the scored result). The
#: driver digests them at training completion and the launcher secures them
#: against those digests before the pod may release the probe.
TRAINING_RECORDS = ("run_manifest.json", "run_completion.json")


class D2ScreeningError(RuntimeError):
    """A screening premise does not hold. Raised before anything is priced."""


def _read(rel: str, repo_root: Path) -> Any:
    path = Path(repo_root) / rel
    if not path.is_file():
        raise D2ScreeningError(f"{rel} is missing; it owns a frozen fact this "
                               "rung cannot derive without it")
    return json.loads(path.read_text())


def preregistration(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from stages.phase_d2 import d2_session as S

    return S.preregistration(repo_root)


def behaviour(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return preregistration(repo_root)["behaviour"]


def protocol(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The shared protocol record, verified against its own hash AND the
    preregistration's binding of it."""
    from stages.d_series import shared_evaluation as SE

    record = _read(PROTOCOL_REL, repo_root)
    if not SE.verify_record(record):
        raise D2ScreeningError(f"{PROTOCOL_REL} does not match its own "
                               "protocol_hash")
    bound = behaviour(repo_root)["protocol"]
    if (record["protocol_id"], record["protocol_hash"]) != (
            bound["id"].split("@")[0], bound["protocol_hash"]):
        raise D2ScreeningError(
            f"the preregistration binds {bound['id']} "
            f"{bound['protocol_hash'][:12]} and the tree carries "
            f"{record['protocol_id']} {record['protocol_hash'][:12]}")
    return record


def selection(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from aadistill.initialization.planning import stage1_selection

    return stage1_selection.load(Path(repo_root) / SELECTION_REL)


def arms(repo_root: Path = REPO_ROOT, *,
         arm_root: str | Path | None = None) -> tuple[Any, ...]:
    """The four candidates, in the committed quality order. NO incumbent arm.

    B is not trained here; its two screening units are imported. `arm_root`
    is WHERE the bytes are (execution configuration): None on the dev host
    reads the search's durable products, a pod passes the relay staging root.
    """
    from stages.phase_d1.behavioural import Arm
    from stages.phase_d2.d2_product_durability import DEFAULT_SCR

    rows = list(selection(repo_root)["selected"])
    k = int(preregistration(repo_root)["retention"]["k"])
    if len(rows) != k:
        raise D2ScreeningError(f"the committed selection names {len(rows)} "
                               f"leaves; the preregistration retains {k}")
    base = Path(arm_root) if arm_root is not None else DEFAULT_SCR / "products"
    return tuple(Arm(
        arm_id=f"c{i}", state_id=row["state_id"],
        artifact_digest=row["artifact_digest"], quality_position=i,
        checkpoint_dir=str(base / row["state_id"]), role="candidate",
        identities={key: row[key] for key in (
            "artifact_digest", "weights_digest", "single_shard_sha256",
            "arch_signature")})
        for i, row in enumerate(rows, start=1))


def seeds(repo_root: Path = REPO_ROOT) -> tuple[int, ...]:
    """The preregistration's screening seeds, which must be the shared rung's.

    A D2 probe is paired with B's imported unit AT THE SAME SEED; a seed the
    protocol did not import has no comparator row.
    """
    declared = tuple(int(s) for s in behaviour(repo_root)["screening"]["seeds"])
    shared = tuple(int(s) for s in protocol(repo_root)["rungs"][SHARED_RUNG]["seeds"])
    if sorted(declared) != sorted(shared):
        raise D2ScreeningError(f"the preregistration's seeds {declared} are "
                               f"not the shared rung's {shared}")
    return declared


def probes(repo_root: Path = REPO_ROOT, *,
           arm_root: str | Path | None = None) -> tuple[Any, ...]:
    """Every probe, arm-major. The count is CHECKED against the registration."""
    from stages.phase_d1.behavioural import Probe

    field = arms(repo_root, arm_root=arm_root)
    out = tuple(Probe(probe_id=f"d2_screening_{a.arm_id}_s{seed}",
                      arm_id=a.arm_id, seed=seed, rung=RUNG,
                      checkpoint_dir=a.checkpoint_dir,
                      battery_role=battery(repo_root)["role"])
                for a in field for seed in seeds(repo_root))
    declared = int(behaviour(repo_root)["screening"]["probes"])
    if len(out) != declared:
        raise D2ScreeningError(
            f"the schedule yields {len(out)} probes and the preregistration "
            f"declares {declared}")
    return out


def battery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The realized `d1_screening` role, re-hashed, and it must be the one the
    preregistration and the shared rung both name."""
    from stages.phase_d1 import behavioural as D1B

    want = behaviour(repo_root)["screening"]["battery"]
    identity = D1B.battery_role(want["role"], repo_root)
    shared = protocol(repo_root)["rungs"][SHARED_RUNG]["reuses_battery"]
    for key in ("role", "family_content_id", "item_ids_sha256", "n_prompts",
                "n_scorable"):
        if not (identity[key] == want[key] == shared[key]):
            raise D2ScreeningError(
                f"battery {key}: realized {identity[key]!r}, preregistered "
                f"{want[key]!r}, shared rung {shared[key]!r}")
    return identity


def recovery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen recipe, checked against the protocol's recovery block."""
    from shared.recipes import E1_KD_HEAVY_0860K as recipe
    from stages.d_series import shared_evaluation as SE
    from stages.phase_c1.autoinit_c1_driver import FROZEN_RECIPE

    block = protocol(repo_root)["recovery"]
    registered = behaviour(repo_root)["recovery_recipe"]
    digest = SE.recipe_digest(json.loads(FROZEN_RECIPE.read_text()),
                              SE.allowed_overrides(repo_root))
    problems = [f"{k}: {got!r} vs {want!r}" for k, got, want in (
        ("recipe_id", recipe.recipe_id, block["recipe_id"]),
        ("recipe_digest", digest, block["recipe_digest"]),
        ("data_digest", recipe.pack_sha256, block["data_digest"]),
        ("allowed_override_set", sorted(SE.allowed_overrides(repo_root)),
         sorted(block["allowed_override_set"])))
        if got != want]
    problems += [f"preregistration {k}: {registered.get(k)!r} vs {block[k]!r}"
                 for k in block if registered.get(k) != block[k]]
    if problems:
        raise D2ScreeningError("the recovery is not the shared protocol's: "
                               + "; ".join(problems))
    return {**block, "tokens": recipe.tokens, "ce_weight": recipe.ce_weight,
            "kd_weight": recipe.kd_weight, "temperature": recipe.temperature,
            "kd_scope": recipe.kd_scope, "block_len": recipe.block_len,
            "_no_objective_in_the_loss": (
                "the D2 search objective (teacher-confidence-weighted union "
                "KL) selected the initializations and enters NOTHING here: "
                "the recovery loss is the frozen recipe's, identical for "
                "every probe and for B's imported units")}


def comparator_units(repo_root: Path = REPO_ROOT) -> dict[str, list[dict]]:
    """B's imported screening units (the comparator) and q2's (descriptive)."""
    units = protocol(repo_root)["rungs"][SHARED_RUNG]["reference_units"]
    out: dict[str, list[dict]] = {"B": [], "q2": []}
    for u in units:
        out.setdefault(u["arm"], []).append(
            {k: u[k] for k in ("probe_id", "arm", "seed",
                               "initialization_digest", "trained_sha256",
                               "result_sha256", "per_sample_sha256",
                               "correct_overall", "usable_rollout_rate")})
    if sorted(r["seed"] for r in out["B"]) != sorted(seeds(repo_root)):
        raise D2ScreeningError("B's imported units do not cover the seeds")
    return out


def evaluation_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """What a D2 probe's generations must be admitted under: the shared rung's
    evaluation protocol, the same scorer bytes and the same battery."""
    rung = protocol(repo_root)["rungs"][SHARED_RUNG]
    return {k: rung[k] for k in (
        "evaluation_protocol_hash", "generation_protocol_fingerprint",
        "scoring_digest", "generation_source_digest", "generation_runtime_digest",
        "scorer")}


def diagnostics(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The rescoring diagnostic's frozen inputs. Descriptive, never a gate."""
    cal = _read(CALIBRATION_REL, repo_root)
    return {"thresholds": dict(cal["diagnostic_thresholds"]),
            "calibration_record_sha256": cal["record_sha256"],
            "rescoring_minutes_per_probe_measured": (
                cal["rescoring_cost_measured"]["projected_minutes_per_probe"]),
            "arms": "the four D2 candidates only; no logits, confidence or "
                    "per-position quantity is computed or estimated for B or "
                    "q2 (retired checkpoints, no token ids)",
            "token_ids": ("the read-only engine sidecar "
                          "scripts/shared/evaluation/uncapped_eval_token_ids.py "
                          "around the UNCHANGED generator; a stratum whose "
                          "rows do not align exactly is rescored from "
                          "retokenized text and labelled approximate per "
                          "sample"),
            "failure_isolation": ("a rescoring failure is recorded and never "
                                  "alters, blocks or re-runs a scored probe")}


def rule(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    stats = protocol(repo_root)["statistics"]
    return {**behaviour(repo_root)["screening"]["rule"],
            "endpoint": stats["endpoint"], "guardrails": stats["guardrails"],
            "sesoi_for_the_confirmation": stats["sesoi"],
            "_not_eight_independent_architectures": (
                "the four candidates are near-siblings of one lineage; the "
                "paired seeds are the replicate, and eight probes are not "
                "eight independent tests"),
            "_frozen": ("no seed is added, no SESOI moves and no candidate is "
                        "re-screened because candidates land close together")}


def require_arms_present(repo_root: Path = REPO_ROOT, *,
                         arm_root: str | Path | None = None) -> dict[str, Any]:
    """Every arm's BYTES carry all four recorded identities. Exact."""
    from stages.phase_d1.behavioural_materialize import (
        D1MaterializeError, _verify_bytes_at,
    )

    out = []
    for arm in arms(repo_root, arm_root=arm_root):
        directory = Path(arm.checkpoint_dir)
        if not directory.is_dir():
            raise D2ScreeningError(f"{arm.arm_id}: no checkpoint at {directory}")
        try:
            got = _verify_bytes_at(directory, dict(arm.identities),
                                   what=f"{arm.arm_id} ({arm.state_id})")
        except D1MaterializeError as exc:
            raise D2ScreeningError(str(exc)) from exc
        out.append({"arm": arm.arm_id, "state_id": arm.state_id,
                    "checkpoint_dir": str(directory),
                    "identities_checked": got["verified"]})
    return {"n_arms": len(out), "arms": out}


def session_contract(repo_root: Path = REPO_ROOT, *,
                     arm_root: str | Path | None = None,
                     verify_bytes: bool = True) -> dict[str, Any]:
    """Everything this rung is bound to. LOCATION-FREE, so the hash issued on
    the dev host is the hash the pod recomputes.

    `verify_bytes` hashes the arms where they are (dev host: the search's
    durable products; pod: the relay staging) and refuses before this
    document exists; the paths it read are the caller's evidence, never
    contract content.
    """
    from stages.d_series import shared_evaluation as SE

    record = protocol(repo_root)
    drift = SE.tree_matches(record, repo_root)
    if any(drift.values()):
        raise D2ScreeningError(
            f"this tree's trainer/generator/scorer differ from the shared "
            f"protocol's pins: {drift}; a probe scored here would not be "
            "paired with B's imported units")
    if verify_bytes:
        require_arms_present(repo_root, arm_root=arm_root)
    sel = selection(repo_root)
    prereg = preregistration(repo_root)
    bat = battery(repo_root)
    field = arms(repo_root, arm_root=arm_root)
    return {
        "schema": "aadistill.phase_d2.screening_contract/v1",
        "experiment_id": EXPERIMENT_ID,
        "rung": RUNG,
        "preregistration_hash": prereg["preregistration_hash"],
        "shared_protocol": {"id": f"{record['protocol_id']}@v"
                                  f"{record['version']}",
                            "protocol_hash": record["protocol_hash"],
                            "rung": SHARED_RUNG},
        "field": {
            "search_run_id": SEARCH_RUN_ID,
            "selection_sha256": sel["selection_sha256"],
            "retention": prereg["retention"]["rule"],
            "k": int(prereg["retention"]["k"]),
            "arms": [{"arm": a.arm_id, "quality_position": a.quality_position,
                      "state_id": a.state_id, "identities": dict(a.identities)}
                     for a in field],
        },
        "seeds": list(seeds(repo_root)),
        "probes": [{"probe_id": p.probe_id, "arm": p.arm_id, "seed": p.seed}
                   for p in probes(repo_root, arm_root=arm_root)],
        "n_probes": int(behaviour(repo_root)["screening"]["probes"]),
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
            "the field is the committed selection, the seeds, probe count and "
            "rule the preregistration's, the battery, recovery and "
            "evaluation identity the shared protocol's"),
    }


def contract_hash(contract: Mapping[str, Any]) -> str:
    from aadistill.infrastructure.manifest import sha256_json

    return sha256_json(dict(contract))


def rank(scored: Sequence[Mapping[str, Any]],
         repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen screening rule over the eight probes and B's imported rows.

    `scored` is one row per D2 probe: `arm`, `seed`, `correct_overall`,
    `usable_rollout_rate`. The pooling, the veto-only guardrail, the
    tie-break and the one-or-none selection are the shared protocol's
    (`shared_evaluation.rank_screening`, which calls D1's own functions).
    """
    from stages.d_series import shared_evaluation as SE

    field = [SE.FieldArm(arm=a.arm_id, quality_position=int(a.quality_position),
                         state_id=a.state_id) for a in arms(repo_root)]
    comparator = comparator_units(repo_root)
    out = SE.rank_screening(field, scored, comparator["B"],
                            seeds=seeds(repo_root))
    q2 = comparator.get("q2") or []
    out["q2_descriptive"] = {
        "pooled_correct_overall": (sum(r["correct_overall"] for r in q2)
                                   / len(q2)) if q2 else None,
        "units": [r["probe_id"] for r in q2],
        "_role": ("DESCRIPTIVE ONLY: q2 advanced on these prompts and seeds, "
                  "so its screening rows carry the winner's curse; never a "
                  "ranking comparator")}
    out["comparator_units"] = [r["probe_id"] for r in comparator["B"]]
    out["_this_is_not_a_verdict"] = (
        "a ranking and a nomination for the SHARED confirmation (or none). "
        "The advancing candidate's screening estimate is inflated by the "
        "winner's curse, and B's rows were already observed in D1 (adaptivity "
        "and multiplicity limits registered in the shared protocol). No "
        "promotion follows from it.")
    return out


__all__ = ["ARM_FILES", "D2ScreeningError", "EXPERIMENT_ID", "POD_ARMS_REL",
           "RUNG", "SEARCH_RUN_ID", "arms", "battery", "comparator_units",
           "contract_hash", "diagnostics", "evaluation_identity", "probes",
           "protocol", "rank", "recovery", "require_arms_present", "rule",
           "seeds", "session_contract"]

"""The new D3's PROBES rung: three recovery probes from the D3 initialization. `$0`.

CONDITIONAL on the structural session completing with STRUCTURAL_CHANGE and its
product being durable on the relay; separately authorized
(`D3ProbesAuthorization`). Every fact is READ from the record that owns it:

    the initialization  the structural run's closeout and product record (its
                        four identities), relay-verified by its durability
                        record -- never q2's recovered weights
    seeds, battery      the D3 registration's behavioural rung, which must be
                        the shared protocol's `shared_confirmation` rung (the
                        D1 confirmation seeds and the d1_confirmation battery)
    recovery, eval      the shared protocol (`d_series_shared_eval@v1`): the
                        frozen recipe, the evaluation identity, the scorer
    comparators         q2's and B's IMPORTED confirmation units (q2 the
                        primary method comparison, B the anchor)

WHAT THIS RUNG IS. One arm (``d3``) x three seeds, each a recovery run from the
D3 INITIAL checkpoint, generated and scored under the shared confirmation
rung's evaluation protocol through the SAME driver as D2's confirmation
(`autoinit_d2_screening_driver.py --rung d3_probes`). The comparison is
EXPLORATORY and computed off-pod (`analyze_d3_probes.py`): no verdict
contract, no alpha, no SHARED_GO, no promotion.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stages.phase_d2 import d2_screening as SC
from stages.phase_d2.d2_screening import (  # noqa: F401  (the driver's surface)
    ARM_FILES, REPO_ROOT, TRAINING_RECORDS, protocol,
)

EXPERIMENT_ID = "phase_d3"
STAGE_ID = "1"
RUNG = "d3_probes"
#: The shared protocol rung whose battery, seeds, evaluation and imported
#: units this rung pairs with.
SHARED_RUNG = "shared_confirmation"
#: The D1 rung name the battery helpers and the scorer key on.
BATTERY_RUNG = "confirmation"
ARM_ID = "d3"
POD_ARMS_REL = "artifacts/stages/stage-1/phase_d3/arms"
RUNS_REL = "logs/stages/stage-1/phase_d3/runs"
#: The dev-host durable store a structural closeout's `--scr` names.
DEFAULT_STORE = Path("/home/ecs-user/aad-artifacts/phase_d3")
STRUCTURAL_PREFIX = "d3_structural_"

#: Everything the shared driver varies by rung (`rung_layout`).
RUNG_LAYOUT = {
    "rung": RUNG,
    "contract_module": "stages.phase_d3.d3_probes",
    "audit_dirname": "autoinit_d3_probes",
    "evidence_filename": "d3_probes.json",
    "attested_name": "d3_probes_attested_protocol.json",
    "attested_schema": "aadistill.phase_d3.probes_attested_protocol/v1",
    "session_schema": "aadistill.phase_d3.probes_session/v1",
    "probe_root_rel": "artifacts/stages/stage-3/d3_probes",
    "eval_packages_rel": "artifacts/stages/stage-3/d3_probes_eval",
    "ack_rel": "artifacts/audit/autoinit_d3_probes/release_acks",
    "status": "/workspace/autoinit_d3_probes.status",
    "scorer_rel": "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py",
    "battery_rung": BATTERY_RUNG,
    "experiment": "D3",
    "authorization": "stages.phase_d3.d3_authorization:D3ProbesAuthorization",
}


class D3ProbesError(SC.D2ScreeningError):
    """A probes premise does not hold. Raised before anything is priced."""


def registration(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from stages.phase_d3 import d3_design as D

    return D.registration(repo_root)


def preregistration(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The driver's name for the design this rung binds: the registration,
    under the key the shared driver reads."""
    reg = registration(repo_root)
    return {**reg, "preregistration_hash": reg["registration_hash"]}


def rung_spec(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return registration(repo_root)["behavioural_rung"]


def structural_product(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The ONE completed structural run's product, or refuse.

    Requires exactly one `d3_structural_*` run whose closeout states the
    registered treatment, outcome STRUCTURAL_CHANGE, verified teardown, and a
    product whose four identities and relay round trip its durability record
    verified. A run in any other state names no initialization.
    """
    from stages.phase_d3 import d3_design as D

    root = Path(repo_root) / RUNS_REL
    found = []
    reg = registration(repo_root)
    for run in sorted(root.glob(f"{STRUCTURAL_PREFIX}*")):
        closeout = run / "closeout" / "outcome.json"
        if not closeout.is_file():
            continue
        doc = json.loads(closeout.read_text())
        if doc.get("structural_outcome") != D.STRUCTURAL_CHANGE:
            continue
        found.append((run, doc))
    if len(found) != 1:
        raise D3ProbesError(
            f"{len(found)} structural runs completed with "
            f"{D.STRUCTURAL_CHANGE}; the probes rung needs exactly one -- "
            "until then there is no D3 initialization to train from")
    run, doc = found[0]
    problems = [why for ok, why in (
        (doc.get("registration_hash") == reg["registration_hash"],
         "the closeout binds another registration"),
        (doc.get("treatment_path_hash")
         == reg["treatment"]["treatment_path_hash"],
         "the closeout's treatment is not the registered one"),
        (doc.get("provider_teardown_verified") is True,
         "the structural run's teardown is not verified"),
        ((doc.get("product") or {}).get("relay_verified") is True,
         "the product is not relay-verified"))
        if not ok]
    if problems:
        raise D3ProbesError(f"{run.name}: " + "; ".join(problems))
    product = doc["product"]
    return {"run_id": run.name, "identities": dict(product["identities"]),
            "files": dict(product["files"]),
            "relay": dict(product["relay"]),
            "closeout_rel": str((run / "closeout" / "outcome.json")
                                .relative_to(repo_root))}


def arms(repo_root: Path = REPO_ROOT, *,
         arm_root: str | Path | None = None) -> tuple[Any, ...]:
    """The ONE arm: the D3 INITIAL checkpoint, where the bytes are."""
    from stages.phase_d1.behavioural import Arm

    product = structural_product(repo_root)
    digest = product["identities"]["artifact_digest"]
    base = (Path(arm_root) if arm_root is not None else
            DEFAULT_STORE / product["run_id"] / "products")
    #: The state id IS the artifact digest: the arm directory is named by it
    #: (on the dev store, the relay and the pod), and the shared unit gate
    #: checks a probe's student path against the state id.
    return (Arm(arm_id=ARM_ID, state_id=digest,
                artifact_digest=digest, quality_position=None,
                checkpoint_dir=str(base / digest), role="candidate",
                identities={k: product["identities"][k] for k in (
                    "artifact_digest", "weights_digest", "single_shard_sha256",
                    "arch_signature")}),)


def num_parameters(repo_root: Path = REPO_ROOT) -> int:
    return int(structural_product(repo_root)["identities"]["num_parameters"])


def seeds(repo_root: Path = REPO_ROOT) -> tuple[int, ...]:
    declared = tuple(int(s) for s in rung_spec(repo_root)["seeds"])
    shared = tuple(int(s) for s in
                   protocol(repo_root)["rungs"][SHARED_RUNG]["seeds"])
    if declared != shared:
        raise D3ProbesError(f"the registration's seeds {declared} are not the "
                            f"shared rung's {shared}")
    return declared


def battery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The registered role, re-hashed, equal to the shared rung's."""
    from stages.phase_d1 import behavioural as D1B

    want = rung_spec(repo_root)["battery"]
    identity = D1B.battery_role(want["role"], repo_root)
    shared = protocol(repo_root)["rungs"][SHARED_RUNG]["reuses_battery"]
    for key in ("role", "family_content_id", "item_ids_sha256", "n_prompts",
                "n_scorable"):
        if not (identity[key] == want[key] == shared[key]):
            raise D3ProbesError(
                f"battery {key}: realized {identity[key]!r}, registered "
                f"{want[key]!r}, shared rung {shared[key]!r}")
    return identity


def recovery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The frozen recipe, checked against the protocol and the registration."""
    from shared.recipes import E1_KD_HEAVY_0860K as recipe
    from stages.d_series import shared_evaluation as SE
    from stages.phase_c1.autoinit_c1_driver import FROZEN_RECIPE

    block = protocol(repo_root)["recovery"]
    registered = rung_spec(repo_root)["recovery"]
    digest = SE.recipe_digest(json.loads(FROZEN_RECIPE.read_text()),
                              SE.allowed_overrides(repo_root))
    problems = [f"{k}: {got!r} vs {want!r}" for k, got, want in (
        ("recipe_id", recipe.recipe_id, block["recipe_id"]),
        ("recipe_digest", digest, block["recipe_digest"]),
        ("data_digest", recipe.pack_sha256, block["data_digest"]))
        if got != want]
    problems += [f"registration {k}: {registered.get(k)!r} vs {block[k]!r}"
                 for k in registered if registered.get(k) != block.get(k)]
    if problems:
        raise D3ProbesError("the recovery is not the shared protocol's: "
                            + "; ".join(problems))
    return {**block, "tokens": recipe.tokens, "ce_weight": recipe.ce_weight,
            "kd_weight": recipe.kd_weight, "temperature": recipe.temperature,
            "kd_scope": recipe.kd_scope, "block_len": recipe.block_len}


def comparator_units(repo_root: Path = REPO_ROOT) -> dict[str, list[dict]]:
    """q2's and B's imported units, exactly the registered ones."""
    units = protocol(repo_root)["rungs"][SHARED_RUNG]["reference_units"]
    out: dict[str, list[dict]] = {"q2": [], "B": []}
    for u in units:
        if u["arm"] in out:
            out[u["arm"]].append({k: u[k] for k in (
                "probe_id", "arm", "seed", "initialization_digest",
                "trained_sha256", "result_sha256", "per_sample_sha256",
                "correct_overall", "usable_rollout_rate")})
    registered = rung_spec(repo_root)["comparators"]
    for arm, rows in out.items():
        if sorted(r["probe_id"] for r in rows) != sorted(
                registered[arm]["units"]):
            raise D3ProbesError(f"{arm}'s imported units are not the "
                                "registered ones")
    return out


def evaluation_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    record = protocol(repo_root)["rungs"][SHARED_RUNG]
    out = {k: record[k] for k in (
        "evaluation_protocol_hash", "generation_protocol_fingerprint",
        "scoring_digest", "generation_source_digest",
        "generation_runtime_digest", "scorer")}
    registered = rung_spec(repo_root)["evaluation"]
    bad = [k for k in registered if registered[k] != out.get(k)]
    if bad:
        raise D3ProbesError(f"the evaluation identity moved: {bad}")
    return out


def diagnostics(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The rescoring diagnostic's frozen inputs: the D2 rungs' own."""
    base = SC.diagnostics(repo_root)
    return {**base, "arms": ("the D3 arm only; no logits or per-position "
                             "quantity is computed for q2 or B (retired "
                             "checkpoints, no token ids)")}


def probes(repo_root: Path = REPO_ROOT, *,
           arm_root: str | Path | None = None) -> tuple[Any, ...]:
    from stages.phase_d1.behavioural import Probe

    out = tuple(Probe(probe_id=f"d3_probes_{a.arm_id}_s{seed}",
                      arm_id=a.arm_id, seed=seed, rung=RUNG,
                      checkpoint_dir=a.checkpoint_dir,
                      battery_role=battery(repo_root)["role"])
                for a in arms(repo_root, arm_root=arm_root)
                for seed in seeds(repo_root))
    if len(out) != int(rung_spec(repo_root)["n_probes"]):
        raise D3ProbesError(f"the schedule yields {len(out)} probes")
    return out


def require_arms_present(repo_root: Path = REPO_ROOT, *,
                         arm_root: str | Path | None = None) -> dict[str, Any]:
    from stages.phase_d1.behavioural_materialize import (
        D1MaterializeError, _verify_bytes_at,
    )

    out = []
    for arm in arms(repo_root, arm_root=arm_root):
        directory = Path(arm.checkpoint_dir)
        if not directory.is_dir():
            raise D3ProbesError(f"{arm.arm_id}: no checkpoint at {directory}")
        try:
            got = _verify_bytes_at(directory, dict(arm.identities),
                                   what=f"{arm.arm_id} ({arm.state_id})")
        except D1MaterializeError as exc:
            raise D3ProbesError(str(exc)) from exc
        out.append({"arm": arm.arm_id, "state_id": arm.state_id,
                    "checkpoint_dir": str(directory),
                    "identities_checked": got["verified"]})
    return {"n_arms": len(out), "arms": out}


def session_contract(repo_root: Path = REPO_ROOT, *,
                     arm_root: str | Path | None = None,
                     verify_bytes: bool = True) -> dict[str, Any]:
    """Everything this rung is bound to. LOCATION-FREE."""
    from stages.d_series import shared_evaluation as SE

    record = protocol(repo_root)
    drift = SE.tree_matches(record, repo_root)
    if any(drift.values()):
        raise D3ProbesError(
            f"this tree's trainer/generator/scorer differ from the shared "
            f"protocol's pins: {drift}")
    if verify_bytes:
        require_arms_present(repo_root, arm_root=arm_root)
    reg = registration(repo_root)
    bat = battery(repo_root)
    product = structural_product(repo_root)
    return {
        "schema": "aadistill.phase_d3.probes_contract/v1",
        "experiment_id": EXPERIMENT_ID, "rung": RUNG,
        "preregistration_hash": reg["registration_hash"],
        "shared_protocol": {"id": f"{record['protocol_id']}@v"
                                  f"{record['version']}",
                            "protocol_hash": record["protocol_hash"],
                            "rung": SHARED_RUNG},
        "initialization": {
            "structural_run_id": product["run_id"],
            "identities": product["identities"],
            "treatment_path_hash": reg["treatment"]["treatment_path_hash"],
            "_initial_checkpoint": ("every probe trains from the D3 "
                                    "structural product, verified at all four "
                                    "identities; never q2's recovered weights")},
        "seeds": list(seeds(repo_root)),
        "probes": [{"probe_id": p.probe_id, "arm": p.arm_id, "seed": p.seed}
                   for p in probes(repo_root, arm_root=arm_root)],
        "n_probes": int(rung_spec(repo_root)["n_probes"]),
        "recovery": recovery(repo_root),
        "battery": {k: bat[k] for k in (
            "family_id", "family_content_id", "allocation_rule_id", "role",
            "item_ids_sha256", "n_prompts", "n_scorable", "per_stratum",
            "root", "files")},
        "evaluation": evaluation_identity(repo_root),
        "comparator": comparator_units(repo_root),
        "exploratory_classification": rung_spec(repo_root)[
            "exploratory_classification"],
        "diagnostics": diagnostics(repo_root),
        "_decides_nothing": (
            "the initialization is the structural session's product, the "
            "seeds, battery, recovery, evaluation and comparators the "
            "registration's and the shared protocol's"),
    }


def contract_hash(contract: Mapping[str, Any]) -> str:
    return SC.contract_hash(contract)


__all__ = ["ARM_ID", "D3ProbesError", "EXPERIMENT_ID", "RUNG", "RUNG_LAYOUT",
           "arms", "battery", "comparator_units", "contract_hash",
           "evaluation_identity", "num_parameters", "probes", "recovery",
           "require_arms_present", "seeds", "session_contract",
           "structural_product"]

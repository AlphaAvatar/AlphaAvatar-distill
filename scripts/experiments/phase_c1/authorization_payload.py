#!/usr/bin/env python3
"""Assemble a Phase-C1 authorization payload. Application layer, not core.

This was `src/aadistill/autoinit/c1_authorization_payload.py`. It was pure — no
git, no clock, no writing — and that was the point of the last session's work.
It was also, in the maintainer's words, "C1 application policy inside algorithm
core": it owned four `logs/phase_c1_*.json` paths, the `$15.1475` ceiling, the
`$283.76` cap, the plan id, the stage-condition prose and the scope sentence.
None of that is reusable by another experiment, and a core that carries it
cannot be reused either.

So the split is:

    src/aadistill/governance/grant.py     the MECHANISM: stated vs derived
                                          fields, cap arithmetic. No experiment.
    configs/experiments/phase_c1/         the INSTANCE: which files to read,
      authorization.json                  which money figures to refuse against,
                                          what the stage conditions say.
    this file                             the ASSEMBLER: reads that config,
                                          derives the identities, builds the
                                          payload.

**Both callers use this same function.** `scripts/autoinit/issue_c1_authorization.py`
supplies `git rev-parse HEAD` and the wall clock; `tests/pod/test_c1_session_contract.py`
supplies fixed strings and writes into `tmp_path`. A test candidate and a live
authorization therefore cannot diverge in derivation — which is the property the
old host-local fixture never had, and which survives this move unchanged.

Building a payload is still not issuing one: nothing here reads a clock, runs
git, writes to `logs/`, stages a bundle or contacts a provider.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))   # experiments.* live here
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1, C1Authorization, c1_hard_ceiling_usd, c1_harness_digest, load_pricing  # noqa: E402
from aadistill.governance.grant import (  # noqa: E402
    GrantContract, GrantRefused, validate_grant,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

CONFIG = "configs/experiments/phase_c1/authorization.json"

#: Kept as the module's public exception so both callers can catch one name.
C1AuthorizationRefused = GrantRefused


def load_config(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    doc = json.loads((Path(repo_root) / CONFIG).read_text())
    if doc.get("authorizes") != "nothing":
        raise GrantRefused(f"{CONFIG} claims to authorize something")
    return doc


def grant_contract(cfg: Mapping[str, Any]) -> GrantContract:
    gc = cfg["grant_contract"]
    return GrantContract(contract_id=gc["contract_id"],
                         stated_fields=tuple(gc["stated_fields"]),
                         derived_fields=tuple(gc["derived_fields"]))


def frozen_plan_hash(repo_root: str | Path = ".") -> str:
    """Rebuilt from the committed identities, never transcribed.

    The operator is registered explicitly first: `build_arm_specs` and the plan
    both name `attention.activation_importance_v1`, and importing its module does
    not register it — that is deliberate, because an unrestricted beam search
    enumerates the whole registry.
    """
    from experiments.phase_c1 import session as CS
    from experiments.phase_c1.isolation import C1Arm, C1IsolationPlan, derive_recovery_seeds
    from aadistill.initialization.operators import attention_activation

    cfg = load_config(repo_root)
    attention_activation.register(replace=True)
    battery = json.loads(
        (Path(repo_root) / cfg["inputs"]["battery_identity"]).read_text())
    return C1IsolationPlan(
        plan_id=cfg["plan_id"],
        arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
              C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
        seeds=tuple(derive_recovery_seeds()),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"]).plan_hash


def load_preregistration(repo_root: str | Path = ".") -> dict[str, Any]:
    """The execution preregistration, checked against its own declared hash."""
    prereg = load_config(repo_root)["inputs"]["preregistration"]
    doc = json.loads((Path(repo_root) / prereg).read_text())
    stated = doc.get("preregistration_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "preregistration_sha256"})
    if stated != recomputed:
        raise C1AuthorizationRefused(
            f"{prereg} declares {stated} but its contents hash to {recomputed}; "
            "it was edited after it was written")
    if doc.get("authorizes") != "nothing":
        raise C1AuthorizationRefused(f"{prereg} claims to authorize something")
    return doc


def build_c1_authorization_payload(
    *,
    grant: Mapping[str, Any],
    session_commit: str,
    granted_utc: str,
    repo_root: str | Path = ".",
    grant_path: str = "<in-memory>",
    harness_digest_override: str | None = None,
) -> dict[str, Any]:
    """The complete payload, as a function of the committed objects.

    `session_commit` and `granted_utc` are the caller's, because they are the two
    facts a pure function cannot know: which commit the pod will check out, and
    when a human granted this. The CLI issuer supplies `git rev-parse HEAD` and
    the wall clock; a test supplies fixed strings and gets a byte-identical
    payload on every run.

    `harness_digest_override` exists ONLY so a test can build a deliberately
    stale candidate and prove `c1_harness_gate` refuses it. It is never passed by
    the issuer, and a payload built with it does not describe this tree — which
    is precisely what such a test needs.
    """
    from experiments.phase_c1 import session as CS
    from experiments.phase_c1.isolation import C0_PREREGISTRATION_SHA256, derive_recovery_seeds
    from experiments.phase_c1.scoring import c1_scoring_contract

    root = Path(repo_root)
    cfg = load_config(root)
    money = cfg["accepted_pricing"]
    inputs = cfg["inputs"]
    HARD_CEILING_USD = float(money["hard_ceiling_usd"])
    PLANNING_FLOOR_USD = float(money["planning_floor_usd"])
    SOFT_STOP_USD = float(money["soft_stop_usd"])
    CUMULATIVE_CAP_USD = float(money["cumulative_cap_usd"])
    GRANT_FIELDS = tuple(cfg["grant_contract"]["stated_fields"])

    grant = validate_grant(grant, grant_contract(cfg),
                           ceiling_usd=HARD_CEILING_USD,
                           expected_cap_usd=CUMULATIVE_CAP_USD)

    harness = dict(c1_harness_digest(root))
    plan_hash = frozen_plan_hash(root)
    doc = load_preregistration(root)
    pricing = load_pricing(root)
    ceiling = c1_hard_ceiling_usd(root)
    battery = json.loads((root / inputs["battery_identity"]).read_text())
    teacher = json.loads((root / inputs["teacher_binding"]).read_text())
    scoring = c1_scoring_contract(root)
    equivalence = json.loads((root / inputs["scoring_equivalence"]).read_text())

    # --- refuse on any disagreement between the committed objects ----------
    problems = []
    if abs(ceiling - HARD_CEILING_USD) > 1e-9:
        problems.append(f"pricing ceiling ${ceiling:.4f} != ${HARD_CEILING_USD:.4f}")
    if abs(float(pricing["totals"]["floor_usd"]) - PLANNING_FLOOR_USD) > 1e-9:
        problems.append("pricing floor moved")
    if abs(float(pricing["totals"]["expected_usd"]) - SOFT_STOP_USD) > 1e-9:
        problems.append("pricing expected moved")
    if harness["digest"] != (doc.get("c1_harness") or {}).get("digest"):
        problems.append("the preregistration does not record the live harness")
    if doc.get("isolation_plan", {}).get("plan_hash") != plan_hash:
        problems.append("the preregistration does not record the live plan hash")
    if doc.get("scoring_contract", {}).get("digest") != scoring["digest"]:
        problems.append("the preregistration does not record the live scoring digest")
    if battery["content_sha256"] != doc["battery"]["content_sha256"]:
        problems.append("the battery identity disagrees with the preregistration")
    if teacher["revision"] != CS.TEACHER_REVISION:
        problems.append("the teacher binding is not the declared revision")
    if equivalence.get("verdict") != "IDENTICAL":
        problems.append(
            f"the scoring equivalence gate is {equivalence.get('verdict')}, not "
            "IDENTICAL; the C1 scoring binding is not admitted")
    if problems:
        raise C1AuthorizationRefused("; ".join(problems))

    #: Applied AFTER the agreement checks, so a stale-digest candidate is stale
    #: in exactly one field rather than being refused on the way out.
    if harness_digest_override is not None:
        harness["digest"] = harness_digest_override

    auth = C1Authorization(
        authorization_id=cfg["authorization_id"],
        granted_utc=granted_utc,
        granted_by=grant["granted_by"],
        plan_id=cfg["plan_id"],
        plan_hash=plan_hash,
        science_plan_hash=C0_PREREGISTRATION_SHA256,
        expected_usd=float(pricing["totals"]["expected_usd"]),
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=tuple(cfg["authorized_stages"]),
        stage_conditions=dict(cfg["stage_conditions"]),
        scope_note=grant["covers"],
        authorized_session_commit=session_commit,
        harness_source_digest=harness["digest"],
        harness_source_files=C1_HARNESS_SOURCE_FILES_V1,
        provenance_commit=session_commit)

    payload = auth.as_dict()
    payload["grant"] = {
        "path": str(grant_path),
        "sha256": sha256_json(grant),
        **{k: grant[k] for k in GRANT_FIELDS},
    }
    payload["bound"] = {
        "execution_preregistration": doc["preregistration_sha256"],
        "execution_preregistration_head_commit": doc["head_commit"],
        "c1_harness_digest": harness["digest"],
        "c1_harness_n_files": harness["n_files"],
        "isolation_plan_hash": plan_hash,
        "c0_preregistration": C0_PREREGISTRATION_SHA256,
        "battery": {"asset_id": battery["asset_id"],
                    "content_sha256": battery["content_sha256"],
                    "n_prompts": battery["n_prompts"],
                    "n_scorable_prompts": battery["n_scorable_prompts"]},
        "teacher": {"repo_id": CS.TEACHER_REPO, "revision": teacher["revision"]},
        "scoring_contract": {"contract": scoring["contract"],
                             "digest": scoring["digest"],
                             "equivalence": equivalence["verdict"],
                             "equivalence_cases": equivalence["n_cases"]},
        "seeds": derive_recovery_seeds(),
        "replay_digests": {"parent": CS.EXPECTED_PARENT_DIGEST,
                           "incumbent": CS.EXPECTED_INCUMBENT_DIGEST},
        "pricing": {"floor_usd": PLANNING_FLOOR_USD,
                    "soft_stop_usd": SOFT_STOP_USD,
                    "hard_ceiling_usd": HARD_CEILING_USD,
                    "cumulative_cap_usd": CUMULATIVE_CAP_USD,
                    "cumulative_spend_at_approval_usd":
                        float(grant["cumulative_spend_at_approval_usd"])},
    }
    payload["one_use"] = cfg["one_use"]
    payload["does_not_authorize"] = grant["does_not_authorize"]
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

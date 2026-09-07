"""Build a C1 authorization payload. Pure: no git, no clock, no writing.

This exists because a test needed a *real* authorization to drive the real
pre-provider gates, and the only way to get one was to keep a hand-issued
artifact at `~/aad-scratch/sessions/c1-candidate/`. That fixture is invisible to
CI, invisible to review, absent under an empty `$HOME`, and — the failure that
prompted this — goes silently stale the moment the harness moves, at which point
a correct gate reports a false alarm. A test whose subject lives outside the
repository is not a test of the repository.

So the derivation moves here and the *effects* stay in the CLI issuer:

    scripts/autoinit/issue_c1_authorization.py   clean-tree check, `git rev-parse`,
                                                 wall clock, one-use semantics,
                                                 writing logs/…authorization.json
    build_c1_authorization_payload()             everything that is a function of
                                                 the committed objects

Both call the same builder, so a test candidate and a live authorization cannot
diverge in derivation — which is the property the host-local fixture never had.

**Building a payload is not issuing an authorization.** Nothing here writes a
file, reads a clock, touches `logs/`, stages a bundle or contacts a provider. The
caller supplies `session_commit` and `granted_utc`; a test supplies fixed values
and writes into `tmp_path`. What makes an authorization *live* is being the
canonical file at the canonical path, carried by a commit the launcher's gate 1
independently re-derives — none of which a returned dict can be.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..infrastructure.manifest import sha256_json
from .c1_authorization import (
    C1_HARNESS_SOURCE_FILES_V1,
    C1Authorization,
    c1_hard_ceiling_usd,
    c1_harness_digest,
    load_pricing,
)

PREREG = "logs/phase_c1_execution_preregistration.json"
BATTERY_IDENTITY = "logs/phase_c1_battery.json"
TEACHER_BINDING = "logs/phase_c1_teacher_binding.json"
EQUIVALENCE = "logs/phase_c1_scoring_equivalence.json"

#: The accepted pricing, restated so a mis-priced session is REFUSED rather than
#: mirrored. A ceiling that simply echoes its source is not an independent check.
HARD_CEILING_USD = 15.1475
PLANNING_FLOOR_USD = 13.4401
SOFT_STOP_USD = 14.7841
CUMULATIVE_CAP_USD = 283.76

#: What a maintainer states. Everything else is computed.
GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")
#: What the builder derives. A grant asserting any of these is refused.
DERIVED_FIELDS = ("granted_utc", "authorized_session_commit",
                  "harness_source_digest", "plan_hash", "science_plan_hash",
                  "preregistration_sha256", "battery_content_sha256",
                  "scoring_contract_digest", "teacher_revision")


class C1AuthorizationRefused(Exception):
    """The committed objects do not support issuing this authorization."""


def validate_grant(grant: Mapping[str, Any]) -> dict[str, Any]:
    """The maintainer-stated half, checked for what only a maintainer can say."""
    missing = [f for f in GRANT_FIELDS
               if f not in grant
               or (isinstance(grant[f], str) and not grant[f].strip())
               or grant[f] is None]
    if missing:
        raise C1AuthorizationRefused(
            f"the grant is missing {missing}. A grant states who permitted what, "
            "at what cumulative spend, and what it does not cover.")
    for derived in DERIVED_FIELDS:
        if derived in grant:
            raise C1AuthorizationRefused(
                f"the grant asserts {derived!r}, which the issuer derives. A "
                "grant that asserts an identity it did not compute is not "
                "evidence of anything.")
    if float(grant["cumulative_cap_usd"]) != CUMULATIVE_CAP_USD:
        raise C1AuthorizationRefused(
            f"the grant names cap ${float(grant['cumulative_cap_usd']):.4f}, not "
            f"${CUMULATIVE_CAP_USD:.4f}")
    spent = float(grant["cumulative_spend_at_approval_usd"])
    if spent + HARD_CEILING_USD > CUMULATIVE_CAP_USD:
        raise C1AuthorizationRefused(
            f"${spent:.4f} already spent plus a ${HARD_CEILING_USD:.4f} ceiling "
            f"exceeds the ${CUMULATIVE_CAP_USD:.4f} cap. Raising a cap is a "
            "maintainer decision, not an issuer's.")
    return dict(grant)


def frozen_plan_hash(repo_root: str | Path = ".") -> str:
    """Rebuilt from the committed identities, never transcribed.

    The operator is registered explicitly first: `build_arm_specs` and the plan
    both name `attention.activation_importance_v1`, and importing its module does
    not register it — that is deliberate, because an unrestricted beam search
    enumerates the whole registry.
    """
    from . import c1_session as CS
    from .c1_isolation import C1Arm, C1IsolationPlan, derive_recovery_seeds
    from .operators import attention_activation

    attention_activation.register(replace=True)
    battery = json.loads((Path(repo_root) / BATTERY_IDENTITY).read_text())
    return C1IsolationPlan(
        plan_id="autoinit.v1.phase_c1",
        arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
              C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
        seeds=tuple(derive_recovery_seeds()),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"]).plan_hash


def load_preregistration(repo_root: str | Path = ".") -> dict[str, Any]:
    """The execution preregistration, checked against its own declared hash."""
    doc = json.loads((Path(repo_root) / PREREG).read_text())
    stated = doc.get("preregistration_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "preregistration_sha256"})
    if stated != recomputed:
        raise C1AuthorizationRefused(
            f"{PREREG} declares {stated} but its contents hash to {recomputed}; "
            "it was edited after it was written")
    if doc.get("authorizes") != "nothing":
        raise C1AuthorizationRefused(f"{PREREG} claims to authorize something")
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
    from . import c1_session as CS
    from .c1_isolation import C0_PREREGISTRATION_SHA256, derive_recovery_seeds
    from .c1_scoring import c1_scoring_contract

    root = Path(repo_root)
    grant = validate_grant(grant)

    harness = dict(c1_harness_digest(root))
    plan_hash = frozen_plan_hash(root)
    doc = load_preregistration(root)
    pricing = load_pricing(root)
    ceiling = c1_hard_ceiling_usd(root)
    battery = json.loads((root / BATTERY_IDENTITY).read_text())
    teacher = json.loads((root / TEACHER_BINDING).read_text())
    scoring = c1_scoring_contract(root)
    equivalence = json.loads((root / EQUIVALENCE).read_text())

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
        authorization_id="autoinit.v1.phase_c1",
        granted_utc=granted_utc,
        granted_by=grant["granted_by"],
        plan_id="autoinit.v1.phase_c1",
        plan_hash=plan_hash,
        science_plan_hash=C0_PREREGISTRATION_SHA256,
        expected_usd=float(pricing["totals"]["expected_usd"]),
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={
            "replay": ("stop before any recovery training on a parent or "
                       "incumbent digest mismatch; preserve the evidence and "
                       "treat it as the attempt's result"),
            "device_handoff": "stop before recovery if the card is not released",
            "training": ("all six trainings must complete before any evaluation; "
                         "a training failure is C1_INCOMPLETE, not a partial "
                         "confirmation experiment"),
            "generation_admission": (
                "no probe is scored unless the generation protocol observed from "
                "its own raw summaries is comparable to the attested C1 protocol"),
            "decision": "stage I requires six complete admitted and scored probes",
            "retry": "none. One launch attempt; no automatic retry, no second "
                     "attempt, no automatic ceiling increase.",
        },
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
    payload["one_use"] = (
        "ONE grant, ONE issuance, ONE launch attempt. No automatic retry, no "
        "second attempt, no automatic ceiling increase. A failure is the "
        "attempt's result and needs a new maintainer review.")
    payload["does_not_authorize"] = grant["does_not_authorize"]
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

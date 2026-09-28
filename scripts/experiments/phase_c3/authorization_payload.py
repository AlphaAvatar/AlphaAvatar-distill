"""Assemble the Phase-C3 one-use authorization payload. Derives, never asserts.

Same contract as the Phase-A, Phase-B, continuation and C1 payload builders:
the **grant is an input**, not a constant. `authorization.py` carries the
authorization *type* -- the hard-`False` scope properties, the derived harness,
the live-priced ceiling -- and nothing about a particular permission.

**Both callers use this same function.** `scripts/autoinit/issue_c3_authorization.py`
supplies `git rev-parse HEAD` and the wall clock; the tests supply a fixed
commit and epoch so a payload is reproducible without a working tree.

What this binds, and what invalidates it if edited:

* the **session commit**, the clean pre-authorization HEAD the pod checks out;
* the **C3 harness digest**, derived by walking the import closure from the C3
  entry points -- never a stored list, which cannot promise completeness;
* the **C3 session contract hash**, rebuilt from the frozen preregistration
  with both experimental operators explicitly registered first, because
  importing their modules does not register them;
* the **preregistration**, by its own self-verified hash -- and its three-arm
  shape, because a two-arm plan here is the `b937aebb` defect;
* the **hard ceiling**, taken from the LIVE pricing record and cross-checked
  against the three package conditions;
* the **battery**, **teacher binding** and **scoring contract** identities.

Every one is DERIVED here and REFUSED if the grant asserts it. A grant that
asserts an identity it did not compute is not evidence of anything.

Issuing is not launching.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from aadistill.governance.grant import GrantRefused  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from experiments.phase_c3.authorization import (  # noqa: E402
    CURRENT_CLOSURE_SNAPSHOT,
    C3Authorization,
    c3_expected_usd,
    c3_hard_ceiling_usd,
    c3_harness_digest,
    load_live_pricing,
)

PREREG_REL = "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"
BATTERY_REL = "logs/stages/stage-1/phase_c1/plans/battery.json"
TEACHER_REL = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"

AUTHORIZATION_ID = "autoinit.v1.phase_c3"
PLAN_ID = "autoinit.v1.phase_c3"


class C3AuthorizationRefused(GrantRefused):
    """The grant does not describe the session this tree would run."""


def load_preregistration(repo_root: str | Path = ".") -> dict[str, Any]:
    """The frozen plan, refused unless its stamp binds it."""
    p = Path(repo_root) / PREREG_REL
    if not p.is_file():
        raise C3AuthorizationRefused(f"no preregistration at {PREREG_REL}")
    doc = json.loads(p.read_text())
    stated = doc.get("preregistration_sha256")
    got = sha256_json({k: v for k, v in doc.items()
                       if k != "preregistration_sha256"})
    if stated != got:
        raise C3AuthorizationRefused(
            f"{PREREG_REL} does not bind itself: stamp {str(stated)[:16]}, "
            f"body {got[:16]}")
    return doc


def session_contract_hash(repo_root: str | Path = ".") -> str:
    """The session's declared shape, rebuilt rather than transcribed.

    Both experimental ATTENTION implementations are registered first: import
    does not register them, and an unregistered one makes `build_arm_specs`
    refuse -- which would surface here as a mysterious failure rather than as
    the ordering property it is.
    """
    from experiments.phase_c3 import session as CS

    CS.register_experimental_operators()
    return CS.C3SessionContract().contract_hash


def build_c3_authorization_payload(*, grant: Mapping[str, Any],
                                   session_commit: str,
                                   granted_utc: str,
                                   repo_root: str | Path = ".",
                                   grant_path: str = "<unknown>",
                                   harness_files: tuple[str, ...] | None = None,
                                   ) -> dict[str, Any]:
    """The payload, fully derived. `harness_files` exists only for tests.

    A test builds a deliberately stale candidate with it and proves
    `c3_harness_gate` refuses it; the real path never passes it.
    """
    from experiments.phase_c1.scoring import c1_scoring_contract
    from experiments.phase_c3 import session as CS

    root = Path(repo_root)

    # ---- the money, from the LIVE record -------------------------------
    live = load_live_pricing(root)
    ceiling = c3_hard_ceiling_usd(root)
    expected = c3_expected_usd(root)
    envelope = float(live["envelopes"]["per_session_envelope_usd"])
    if not live.get("FUNDABLE"):
        raise C3AuthorizationRefused(
            f"the live pricing record is not FUNDABLE: {live.get('shortfalls_usd')}")
    if ceiling > envelope:
        raise C3AuthorizationRefused(
            f"derived ceiling ${ceiling:.4f} exceeds the ${envelope:.4f} envelope")
    stated_cap = grant.get("cumulative_cap_usd")
    live_cap = float(live["envelopes"]["project_cap_usd"])
    if stated_cap is not None and float(stated_cap) != live_cap:
        raise C3AuthorizationRefused(
            f"the grant names cap ${float(stated_cap):.4f}, not ${live_cap:.4f}")
    #: THE ENVELOPE IS NOT THE GRANT. A grant asking for the envelope when the
    #: derived ceiling is lower is a 36% over-authorization that every
    #: downstream gate would accept.
    asked = grant.get("hard_cap_usd")
    if asked is not None and abs(float(asked) - ceiling) > 5e-4:
        raise C3AuthorizationRefused(
            f"the grant asks for ${float(asked):.4f}; the ceiling DERIVED from "
            f"the live securePrice is ${ceiling:.4f}. The authorization "
            "carries the derived figure, never the envelope.")

    # ---- the science, derived ------------------------------------------
    prereg = load_preregistration(root)
    arms = [k for k in prereg["arms"] if not k.startswith("_")]
    if len(arms) != 3:
        raise C3AuthorizationRefused(
            f"the preregistration declares {len(arms)} arms; C3 is a "
            "three-arm design")
    seeds = list(prereg["seeds"]["recovery"])
    if len(seeds) != 3:
        raise C3AuthorizationRefused(f"{len(seeds)} recovery seeds, expected 3")
    primary = prereg["claim_boundary"]["primary_contrast"]
    if "causal-B1" not in primary or "causal-B3" in primary:
        raise C3AuthorizationRefused(
            f"the primary contrast is {primary!r}; C3's verdict belongs to "
            "the operator-isolation contrast")

    harness = dict(c3_harness_digest(root, harness_files))
    contract_hash = session_contract_hash(root)
    scoring = c1_scoring_contract(root)
    battery = json.loads((root / BATTERY_REL).read_text())
    teacher = json.loads((root / TEACHER_REL).read_text())

    snapshot = root / CURRENT_CLOSURE_SNAPSHOT
    if snapshot.is_file():
        recorded = json.loads(snapshot.read_text()).get("digest")
        if recorded != harness["digest"]:
            raise C3AuthorizationRefused(
                f"the recorded C3 closure is {str(recorded)[:12]} and the tree "
                f"digests to {harness['digest'][:12]}; re-derive it")

    # ---- what the grant may say ----------------------------------------
    for asserted in ("harness_source_digest", "preregistration_sha256",
                     "session_contract_hash", "scoring_contract_digest"):
        if asserted in grant:
            raise C3AuthorizationRefused(
                f"the grant at {grant_path} asserts {asserted!r}; every "
                "identity here is DERIVED, and a grant that asserts one it "
                "did not compute is not evidence of anything")

    auth = C3Authorization(
        authorization_id=AUTHORIZATION_ID,
        granted_utc=granted_utc,
        granted_by=grant["granted_by"],
        plan_id=PLAN_ID,
        plan_hash=contract_hash,
        science_plan_hash=prereg["preregistration_sha256"],
        expected_usd=expected,
        hard_cap_usd=ceiling,
        authorized_stages=tuple(grant.get("authorized_stages")
                                or [s.stage_id for s in CS.C3_STAGES]),
        stage_conditions=dict(grant.get("stage_conditions") or {}),
        scope_note=grant["covers"],
        authorized_session_commit=session_commit,
        harness_source_digest=harness["digest"],
        harness_source_files=tuple(f["path"] for f in harness["files"]),
        version=1,
    )
    payload = auth.as_dict()
    #: WHICH maintainer decision this was issued from, by path and content
    #: hash. `grant_provenance_gate` resolves the path against the repo root
    #: and re-hashes the file: a grant belongs to ONE attempt, and running
    #: under another attempt's is running under a decision made about a
    #: different session. Without this the launcher aborts at $0 with "records
    #: no grant path and hash", which is how it was found.
    payload.update({
        "grant": {
            "path": grant_path,
            "sha256": sha256_json(dict(grant)),
            "_rule": ("must resolve to THIS run's governance/grant.json and "
                      "hash to the recorded value"),
        },
        "c3_harness_digest": harness["digest"],
        "c3_harness_n_files": harness["n_files"],
        "bound": {
            "session_contract_hash": contract_hash,
            "preregistration": PREREG_REL,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "arm_ids": arms,
            "recovery_seeds": seeds,
            "bootstrap_seed": prereg["seeds"]["bootstrap"],
            "primary_contrast": primary,
            "expected_parent_digest": prereg["shared_parent"]["artifact_digest"],
            "expected_incumbent_digest": prereg["arms"]["A_incumbent"][
                "artifact_digest"],
            "battery": {"asset_id": battery["asset_id"],
                        "content_sha256": battery["content_sha256"]},
            "teacher": {"repo_id": teacher["repo_id"],
                        "revision": teacher["revision"]},
            "scoring_contract": {"contract": scoring["contract"],
                                 "digest": scoring["digest"]},
            "live_pricing": {
                "gpu_type_id": live.get("gpu_type_id"),
                "queried_utc": live.get("queried_utc"),
                "gpu_rate_usd_per_hour": live["price"]["gpu_rate_usd_per_hour"],
                "billed_rate_usd_per_hour": live["price"][
                    "billed_rate_usd_per_hour"],
                "derived_hard_ceiling_usd": ceiling,
                "per_session_envelope_usd": envelope,
                "_the_envelope_is_not_the_grant": True},
        },
        "_scope": {
            "probes": len(arms) * len(seeds),
            "arms": len(arms), "seeds": len(seeds),
            "authorizes_c4": False,
            "decides_b1_vs_b3": False,
        },
    })
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

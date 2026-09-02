#!/usr/bin/env python3
"""Issue the ONE-USE Phase-C1 authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_c1_authorization.py \
        --grant logs/autoinit_c1_grant.json --require-clean

Same contract as the Phase-A, Phase-B and continuation issuers, and the same
reason for it: the grant is an **input**, not a constant. `c1_authorization.py`
carries the authorization *type* -- the hard-`False` scope properties, the
harness file set, the ceiling derivation -- and nothing about a particular
permission. A one-use maintainer decision living in executable source goes stale
silently and still reads as though it applies.

**This cannot become a Phase-A or a search grant.**
`C1Authorization.allows_phase_a` and `.allows_beam_search` are hard `False`
properties with no field to set, and `load` refuses any artifact whose schema is
not the C1 one. That is a property of the type, not a promise in this docstring.

What this binds, and what invalidates it if edited:

* the **session commit**, the clean pre-authorization HEAD the pod checks out;
* the **C1 harness digest**, over the declared file set, which
  `session_commit_gate` independently re-derives from that commit's own blobs;
* the **C1 isolation plan hash**, rebuilt from the committed frozen identities
  rather than transcribed, with the treatment operator explicitly registered
  first -- importing its module does not register it;
* the **C0 preregistration digest** as the science plan;
* the **execution preregistration**, by its own self-verified hash;
* the **hard ceiling**, cross-checked against `logs/phase_c1_pricing.json`;
* the **battery**, **teacher binding** and **scoring contract** identities the
  session will measure under.

Every one of those is DERIVED here and refused if the grant asserts it. A grant
that asserts an identity it did not compute is not evidence of anything.

Issuing is not launching.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1, C1Authorization, c1_hard_ceiling_usd,
    c1_harness_digest, load_pricing,
)
from aadistill.autoinit.c1_isolation import (  # noqa: E402
    C0_PREREGISTRATION_SHA256, C1Arm, C1IsolationPlan, derive_recovery_seeds,
)
from aadistill.autoinit.c1_scoring import c1_scoring_contract  # noqa: E402
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

PREREG = "logs/phase_c1_execution_preregistration.json"
PRICING = "logs/phase_c1_pricing.json"
BATTERY_IDENTITY = "logs/phase_c1_battery.json"
TEACHER_BINDING = "logs/phase_c1_teacher_binding.json"
EQUIVALENCE = "logs/phase_c1_scoring_equivalence.json"
OUT = "logs/autoinit_c1_authorization.json"

#: The accepted pricing, restated so the issuer can REFUSE a mis-priced session
#: rather than mirror whatever the pricing file happens to say. A ceiling that
#: simply echoes its source is not an independent check, and this is the last
#: place a mis-priced session can be stopped before a grant exists.
HARD_CEILING_USD = 13.7578
PLANNING_FLOOR_USD = 12.2070
SOFT_STOP_USD = 13.4277
CUMULATIVE_CAP_USD = 283.76

#: What a maintainer states. Everything else is computed.
GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")
#: What the issuer derives. A grant asserting any of these is refused.
DERIVED_FIELDS = ("granted_utc", "authorized_session_commit",
                  "harness_source_digest", "plan_hash", "science_plan_hash",
                  "preregistration_sha256", "battery_content_sha256",
                  "scoring_contract_digest", "teacher_revision")


def load_grant(path: Path) -> dict:
    grant = json.loads(path.read_text())
    missing = [f for f in GRANT_FIELDS
               if f not in grant
               or (isinstance(grant[f], str) and not grant[f].strip())
               or grant[f] is None]
    if missing:
        raise SystemExit(
            f"refusing to issue: {path} is missing {missing}. A grant states who "
            "permitted what, at what cumulative spend, and what it does not cover.")
    for derived in DERIVED_FIELDS:
        if derived in grant:
            raise SystemExit(
                f"refusing to issue: {path} asserts {derived!r}, which the issuer "
                "derives. A grant that asserts an identity it did not compute is "
                "not evidence of anything.")
    if float(grant["cumulative_cap_usd"]) != CUMULATIVE_CAP_USD:
        raise SystemExit(
            f"refusing to issue: the grant names cap "
            f"${float(grant['cumulative_cap_usd']):.4f}, not ${CUMULATIVE_CAP_USD:.4f}")
    spent = float(grant["cumulative_spend_at_approval_usd"])
    if spent + HARD_CEILING_USD > CUMULATIVE_CAP_USD:
        raise SystemExit(
            f"refusing to issue: ${spent:.4f} already spent plus a "
            f"${HARD_CEILING_USD:.4f} ceiling exceeds the ${CUMULATIVE_CAP_USD:.4f} "
            "cap. Raising a cap is a maintainer decision, not an issuer's.")
    return grant


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def frozen_plan_hash() -> str:
    """Rebuilt from the committed identities, never transcribed.

    The operator is registered explicitly first: `build_arm_specs` and the plan
    both name `attention.activation_importance_v1`, and importing its module does
    not register it -- that is deliberate, because an unrestricted beam search
    enumerates the whole registry.
    """
    attention_activation.register(replace=True)
    battery = json.loads((REPO_ROOT / BATTERY_IDENTITY).read_text())
    return C1IsolationPlan(
        plan_id="autoinit.v1.phase_c1",
        arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
              C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
        seeds=tuple(derive_recovery_seeds()),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"]).plan_hash


def preregistration() -> dict:
    """The execution preregistration, checked against its own declared hash."""
    doc = json.loads((REPO_ROOT / PREREG).read_text())
    stated = doc.get("preregistration_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "preregistration_sha256"})
    if stated != recomputed:
        raise SystemExit(
            f"refusing to issue: {PREREG} declares {stated} but its contents hash "
            f"to {recomputed}; it was edited after it was written")
    if doc.get("authorizes") != "nothing":
        raise SystemExit(f"refusing to issue: {PREREG} claims to authorize something")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to issue against a dirty tree; the authorized "
                         "commit must describe what the pod will check out")
    ap.add_argument("--porcelain", action="store_true")
    args = ap.parse_args()

    grant = load_grant(args.grant)
    if args.require_clean:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what the "
                  "pod checks out.")

    commit = git("rev-parse", "HEAD")
    harness = c1_harness_digest(REPO_ROOT)
    plan_hash = frozen_plan_hash()
    doc = preregistration()
    pricing = load_pricing(REPO_ROOT)
    ceiling = c1_hard_ceiling_usd(REPO_ROOT)
    battery = json.loads((REPO_ROOT / BATTERY_IDENTITY).read_text())
    teacher = json.loads((REPO_ROOT / TEACHER_BINDING).read_text())
    scoring = c1_scoring_contract(REPO_ROOT)
    equivalence = json.loads((REPO_ROOT / EQUIVALENCE).read_text())

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
        raise SystemExit("refusing to issue:\n  " + "\n  ".join(problems))

    auth = C1Authorization(
        authorization_id="autoinit.v1.phase_c1",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        authorized_session_commit=commit,
        harness_source_digest=harness["digest"],
        harness_source_files=C1_HARNESS_SOURCE_FILES_V1,
        provenance_commit=commit)

    payload = auth.as_dict()
    payload["grant"] = {
        "path": str(args.grant),
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

    out = REPO_ROOT / args.out
    out.write_text(json.dumps(payload, indent=1) + "\n")

    # Round-trip through the real loader: an artifact this issuer wrote but the
    # driver cannot load is worse than none.
    reloaded = C1Authorization.load(out)
    assert reloaded.hard_cap_usd == ceiling
    assert reloaded.plan_hash == plan_hash
    assert reloaded.allows_phase_a is False
    assert reloaded.allows_beam_search is False

    if args.porcelain:
        print(json.dumps({"authorization_id": payload["authorization_id"],
                          "authorization_sha256": payload["authorization_sha256"],
                          "authorized_session_commit": commit}))
    else:
        print(f"wrote {args.out}")
        print(f"  authorization_id   {payload['authorization_id']}")
        print(f"  authorization_sha  {payload['authorization_sha256']}")
        print(f"  session commit     {commit}")
        print(f"  harness            {harness['digest']} ({harness['n_files']} files)")
        print(f"  isolation plan     {plan_hash}")
        print(f"  preregistration    {doc['preregistration_sha256']}")
        print(f"  scoring            {scoring['contract']} {scoring['digest'][:16]}")
        print(f"  ceiling            ${ceiling:.4f}  (floor ${PLANNING_FLOOR_USD:.4f}, "
              f"soft stop ${SOFT_STOP_USD:.4f})")
        print(f"  cumulative         ${float(grant['cumulative_spend_at_approval_usd']):.4f}"
              f" spent of ${CUMULATIVE_CAP_USD:.4f}")
        print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Issue the ONE-USE Phase-B authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_phase_b_authorization.py \
        --grant logs/autoinit_phase_b_grant.json --require-clean

Same contract as the Phase-A issuer, and the same reason for it: the grant is an
**input**, not a constant. `phase_b.py` carries the authorization schema — stages,
stage conditions, scope, ceilings — and nothing about a particular permission. A
one-use maintainer decision living in executable source goes stale silently and
still reads as though it applies.

What this binds, and what invalidates it if edited:

* the **session plan** hash;
* the **science plan** hash, read from the frozen plan on disk rather than a
  constant, so a threshold that moved after freezing cannot be authorized;
* the **Phase-B executable-source digest** — 57 files, the driver, launcher,
  search engine, operators, adapters and session machinery a paid run executes;
* the **preregistration** identity, so a run cannot execute under a record that
  describes something else;
* **both** calibration identities per profile: `profile_hash` for the
  specification and `content_sha256` for the sampled bytes, because neither
  implies the other;
* the **historical-reuse verdict**, since the ten-probe price assumes three
  citations;
* the **session commit**, which is what the pod checks out from the bundle.

Issuing is not launching. This artifact authorizes spend and stages; starting the
run is a separate, explicit decision.
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

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.generation import generation_source_digest  # noqa: E402
from aadistill.autoinit.phase_b import (  # noqa: E402
    PHASE_B_EXECUTABLE_SOURCE_FILES_V1, PHASE_B_PLAN_V1, PhaseBAuthorization,
    phase_b_source_digest,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    recovery_scoring_contract, trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

FROZEN_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
PREREGISTRATION = "logs/autoinit_phase_b_preregistration.json"
REUSE_RECORD = "logs/autoinit_historical_probe_reuse.json"

#: The ceilings, RE-DERIVED and approved 2026-08-27. `$26.8049` rested on a P=2
#: search priced at 1.91-7.51 h; attempt 3 ran 9.08 h without finishing, because
#: the model assumed a cached intact reference while the run recomputed it per
#: candidate. The corrected model prices all three reference modes, costs the
#: composite leaves, charges statistics once per (parent, profile) as the runtime
#: shares them, and adds the non-FLOP materialization path.
#:
#: The cumulative cap is `$239.9150 spent + $35.6660 approved`, rounded up by
#: `$0.0090` of ordinary rounding margin. Headroom is not authorization.
HARD_CEILING_USD = 35.6660
PLANNING_FLOOR_USD = 16.4555
CUMULATIVE_CAP_USD = 275.59

GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")
#: Everything the issuer establishes for itself. A grant asserting one of these
#: would be claiming an identity it never computed.
DERIVED_FIELDS = ("granted_utc", "authorized_session_commit", "source_digest",
                  "plan_hash", "science_plan_hash", "calibration_profile_hashes",
                  "calibration_content_hashes", "preregistration_sha256")


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
    return grant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_b_authorization.json")
    ap.add_argument("--grant", required=True)
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = [ln for ln in subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        cwd=REPO_ROOT).stdout.splitlines() if args.out not in ln]
    if dirty and args.require_clean:
        raise SystemExit(
            f"refusing to issue: the working tree is dirty in "
            f"{[ln.strip() for ln in dirty][:6]}. The pod checks out {commit} from "
            "a bundle, so uncommitted edits would not be the code that runs while "
            "the source digest would claim they were.")

    grant_path = Path(args.grant)
    grant_path = grant_path if grant_path.is_absolute() else REPO_ROOT / grant_path
    if not grant_path.is_file():
        raise SystemExit(
            f"refusing to issue: no grant document at {grant_path}. Phase B is not "
            "authorized by the existence of this script.")
    grant = load_grant(grant_path)

    frozen = json.loads((REPO_ROOT / FROZEN_PLAN).read_text())
    science_plan_hash = frozen["plan_hash"]
    source = phase_b_source_digest(REPO_ROOT)
    if source["not_yet_covered"]:
        raise SystemExit(
            f"refusing to issue: Phase-B declares uncovered executable source "
            f"{source['not_yet_covered']}")

    prereg = json.loads((REPO_ROOT / PREREGISTRATION).read_text())
    if prereg["executable_source"]["digest"] != source["digest"]:
        raise SystemExit(
            "refusing to issue: the preregistration was frozen against executable "
            f"{prereg['executable_source']['digest'][:12]} but the executable is "
            f"{source['digest'][:12]}. Re-freeze it rather than authorizing code "
            "the record does not describe.")
    if prereg["session_plan"]["plan_hash"] != PHASE_B_PLAN_V1.plan_hash:
        raise SystemExit("refusing to issue: the preregistration binds a different plan")

    reuse = json.loads((REPO_ROOT / REUSE_RECORD).read_text())
    if not reuse.get("reuse_verified"):
        raise SystemExit(
            "refusing to issue: the historical probe reuse record is unverified, and "
            "the ten-probe price assumes three citations")

    profiles = (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)
    for profile in profiles:
        if not profile.materialized or not profile.content_sha256:
            raise SystemExit(f"refusing to issue: {profile.qualified_id} is not materialized")

    granted = datetime.now(timezone.utc)
    granted_by = "\n\n".join([
        grant["granted_by"].strip(),
        f"Covers: {grant['covers']}",
        (f"Cumulative spend at approval "
         f"${float(grant['cumulative_spend_at_approval_usd']):.4f} against a "
         f"cumulative cap of ${float(grant['cumulative_cap_usd']):.2f}."),
        f"Does NOT authorize: {grant['does_not_authorize']}",
        f"Grant document: {grant_path.relative_to(REPO_ROOT)}",
    ])

    auth = PhaseBAuthorization(
        authorization_id=f"autoinit.phase_b.{granted:%Y%m%dT%H%M%SZ}",
        granted_utc=granted.isoformat(),
        granted_by=granted_by,
        plan_id=PHASE_B_PLAN_V1.plan_id,
        plan_hash=PHASE_B_PLAN_V1.plan_hash,
        science_plan_hash=science_plan_hash,
        calibration_profile_hashes={p.qualified_id: p.profile_hash for p in profiles},
        calibration_content_hashes={p.qualified_id: p.content_sha256 for p in profiles},
        planning_floor_usd=PLANNING_FLOOR_USD,
        hard_cap_usd=HARD_CEILING_USD,
        per_launch_hard_usd=HARD_CEILING_USD,
        authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={
            "0": ("attestation; generation_runtime_comparability@v2; the Phase-B "
                  "executable-source digest; BOTH calibration identities per "
                  "profile; import of the verified historical probes. A "
                  "comparability FAILURE TERMINATES the session before any search "
                  "or probe and is NOT answered by re-running eight candidates"),
            "1": ("full fresh joint P=2 beam-6 search under SCHEDULE_V1 and "
                  "PARETO_V1 on the unchanged state_eval@v1; 5 admissible leaves "
                  "or STOP and report the shortfall"),
            "2": ("cross-phase rung 1 on sa over EIGHT candidates: the Phase-B "
                  "Top-5 (probed) plus cca699c93f34, 85bde4ded2c3 and the "
                  "canonical control (cited from verified evidence). The three "
                  "excluded Phase-A leaves are NOT admitted"),
            "3": ("rung 2 on sb: the two globally best searched candidates plus "
                  "the control, which advances unconditionally; only missing "
                  "probes are run"),
            "4": ("conditional tie-break on sc, only for candidates inside the "
                  "frozen equivalence interval that lack a verified sc; no fourth seed"),
            "5": ("selection and report; unresolved_equivalence with winner=None "
                  "is a RESULT. Search-side KL/NLL and the canonical Stage-1 NLL "
                  "may NOT break a tie"),
            "teardown": ("transfer and verify the five Phase-B Top-5 leaves through "
                         "the transfer-result contract, collect, delete the pod, "
                         "confirm from the provider, STOP for review"),
        },
        scope_note=(
            "ONE Phase-B execution, one launcher invocation. This artifact "
            "authorizes the SPEND and the STAGES; it is not by itself an "
            "instruction to launch. Nothing here permits a second attempt, "
            "reopening Phase A, the rejected 14-probe no-reuse path, "
            "rematerializing the Stage-3 controls, a fourth seed, or any "
            "follow-on experiment: Phase B is a terminus and stops for review on "
            "every path, including unresolved_equivalence."),
        authorized_session_commit=commit,
        source_digest=source["digest"],
        source_files=PHASE_B_EXECUTABLE_SOURCE_FILES_V1,
        provenance_commit=commit,
    )

    payload = auth.as_dict()
    # These join the payload and are then hashed WITH it. Adding them after
    # `as_dict()` computed `authorization_sha256` produced an artifact that
    # failed its own tamper check on the first issue attempt -- `load()`
    # recomputes over everything except the hash field, so any key added
    # afterwards is indistinguishable from an edit. Which is the point of the
    # check; it caught this.
    payload.pop("authorization_sha256")
    payload["delegated_identities"] = {
        "trainer": trainer_source_digest(REPO_ROOT)["digest"],
        "generation": generation_source_digest(REPO_ROOT)["digest"],
        "scoring_contract": recovery_scoring_contract(REPO_ROOT)["digest"],
        "_why_here": ("identities Phase B DELEGATES rather than digests: they are "
                      "verified at stage 0 by the code that owns them"),
    }
    payload["preregistration_sha256"] = prereg["preregistration_sha256"]
    payload["historical_reuse"] = {
        "verified": True,
        "probes_dir_digest": reuse["probes_dir_digest"],
        "admitted_reusable_probes": reuse["admitted_reusable_probes"],
    }
    payload["cumulative_cap_usd"] = CUMULATIVE_CAP_USD
    payload["one_use"] = True
    payload["authorization_sha256"] = sha256_json(payload)

    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise SystemExit(
            f"refusing to issue: {out} already exists. A Phase-B authorization is "
            "ONE-USE; overwriting one would erase the record of what was permitted.")
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"authorization_id   {payload['authorization_id']}")
    print(f"granted_utc        {payload['granted_utc']}")
    print(f"session commit     {commit}")
    print(f"source digest      {source['digest']}")
    print(f"session plan       {PHASE_B_PLAN_V1.plan_hash}")
    print(f"science plan       {science_plan_hash}")
    print(f"preregistration    {payload['preregistration_sha256']}")
    for p in profiles:
        print(f"  {p.qualified_id:28} spec {p.profile_hash[:12]}  content {p.content_sha256[:12]}")
    print(f"planning floor     ${PLANNING_FLOOR_USD:.4f}  (NOT an expected spend)")
    print(f"HARD CEILING       ${HARD_CEILING_USD:.4f}")
    print(f"cumulative cap     ${CUMULATIVE_CAP_USD:.2f}")
    print(f"authorization_sha  {payload['authorization_sha256']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

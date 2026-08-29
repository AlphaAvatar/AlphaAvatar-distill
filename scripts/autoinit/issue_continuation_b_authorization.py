#!/usr/bin/env python3
"""Issue the ONE-USE behavioural-continuation authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_continuation_b_authorization.py \
        --grant logs/autoinit_continuation_b_grant.json --require-clean

Same contract as the Phase-A and Phase-B issuers, and the same reason for it: the
grant is an **input**, not a constant. `phase_b_continuation.py` carries the
authorization schema; a one-use maintainer decision living in executable source
goes stale silently and still reads as though it applies.

**This is not a Phase-B grant and cannot become one.** Phase-B Stage 1 is
complete, retained and authoritative. `ContinuationAuthorization.runs_search` is
`False` by type — there is no field to set — and the ceiling here prices one
missing `sb` and at most two conditional `sc`, not a 16.5 h P=2 search that has
already been bought.

What this binds, and what invalidates it if edited:

* the **session commit**, which is what the pod checks out from the bundle;
* the **continuation executable-source digest**, v3, computed with the canonical
  `sha256` over sorted `path:sha256` lines that `session_commit_gate`
  independently re-derives. v2 used `sha256_json` and the gate refused every
  launch;
* the **continuation session plan** and the frozen **science plan**;
* the **preregistration** identity;
* **both** calibration identities per profile, because neither implies the other;
* the six completed-evidence identities the session cites: the Attempt-5 Stage-1
  selection, the identity-collapse amendment, the six-candidate universe, both
  strict reuse records, and the frozen rung-1 result;
* the **verified relay assets** for `fe9683e6a9c7`, the one advancing checkpoint
  no prior session staged.

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
from aadistill.autoinit.phase_b_continuation import (  # noqa: E402
    BOUND_EVIDENCE, CONTINUATION_PLAN_V1, CONTINUATION_SOURCE_FILES_V2,
    ContinuationAuthorization, continuation_source_digest,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    recovery_scoring_contract, trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

FROZEN_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
PREREGISTRATION = "logs/autoinit_continuation_b_preregistration.json"
AMENDMENT = "logs/autoinit_phase_b_identity_collapse_amendment.json"
STAGE1_SELECTION = "logs/autoinit_phase_b_attempt5/stage1_selection.json"
HISTORICAL_REUSE = "logs/autoinit_historical_probe_reuse.json"
ATTEMPT5_REUSE = "logs/autoinit_attempt5_probe_reuse.json"
ASSETS = "logs/autoinit_continuation_b_assets.json"
PRICING = "logs/autoinit_behavioural_continuation_pricing.json"

#: Derived by `price_behavioural_continuation.py` and approved unchanged. The
#: floor is ONE probe (the mandatory missing `sb`); the ceiling is THREE (that
#: probe plus both conditional `sc`). Neither is an expectation: no expected-value
#: assumption over tie-break probability is defined anywhere in this project.
#:
#: The `$35.6660` full-Phase-B ceiling is HISTORICAL and must never authorize this
#: session — it books a search that attempt 5 completed and retained.
HARD_CEILING_USD = 8.0691
PLANNING_FLOOR_USD = 5.4784
CUMULATIVE_CAP_USD = 283.76
FULL_PHASE_B_CEILING_USD = 35.6660

GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")
DERIVED_FIELDS = ("granted_utc", "authorized_session_commit", "source_digest",
                  "plan_hash", "science_plan_hash", "calibration_profile_hashes",
                  "calibration_content_hashes", "preregistration_sha256",
                  "bound_evidence")


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
            f"${float(grant['cumulative_cap_usd']):.2f}, the issuer "
            f"${CUMULATIVE_CAP_USD:.2f}")
    return grant


def observed_evidence() -> dict[str, str]:
    """Re-derived here, not copied from the preregistration.

    The driver's `observed_evidence()` re-derives the same six identities on the
    pod and `require_evidence` compares them. Reading the preregistration's copy
    instead would bind the grant to a record rather than to the artifacts, and a
    record can be regenerated.
    """
    selection = json.loads((REPO_ROOT / STAGE1_SELECTION).read_text())
    amendment = json.loads((REPO_ROOT / AMENDMENT).read_text())
    historical = json.loads((REPO_ROOT / HISTORICAL_REUSE).read_text())
    attempt5 = json.loads((REPO_ROOT / ATTEMPT5_REUSE).read_text())
    rung1 = amendment["rung1_selection"]
    return {
        "stage1_selection_sha256": selection["selection_sha256"],
        "identity_collapse_amendment_sha256": amendment["amendment_sha256"],
        "collapsed_universe_identity":
            amendment["collapsed_universe"]["universe_identity"],
        "historical_reuse_probes_dir_digest": historical["probes_dir_digest"],
        "attempt5_reuse_probes_dir_digest": attempt5["probes_dir_digest"],
        "rung1_selection_digest": sha256_json({
            "selected_searched": rung1["selected_searched"],
            "auto_advanced_control": rung1["auto_advanced_control"],
            "advancing": rung1["advancing"]}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_continuation_b_authorization.json")
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
            f"refusing to issue: no grant document at {grant_path}. The "
            "continuation is not authorized by the existence of this script.")
    grant = load_grant(grant_path)

    frozen = json.loads((REPO_ROOT / FROZEN_PLAN).read_text())
    science_plan_hash = frozen["plan_hash"]
    source = continuation_source_digest(REPO_ROOT)
    if source["not_yet_covered"]:
        raise SystemExit(
            f"refusing to issue: uncovered executable source "
            f"{source['not_yet_covered']}")
    if source["set_version"] < 3:
        raise SystemExit(
            "refusing to issue: the continuation source set is still v"
            f"{source['set_version']}, whose sha256_json digest session_commit_gate "
            "cannot reproduce. It refused every launch.")

    # The gate that blocked the previous attempt, run HERE, before an artifact
    # exists to be wrong. Cheaper than discovering it at the pre-provider gate and
    # far cheaper than discovering it on a pod.
    blobs = []
    for rel in sorted(CONTINUATION_SOURCE_FILES_V2):
        blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                              capture_output=True, cwd=REPO_ROOT)
        if blob.returncode != 0:
            raise SystemExit(
                f"refusing to issue: {commit} does not contain {rel}. The pod "
                "checks this commit out of a bundle; a source file that is not in "
                "it is not code that can run.")
        import hashlib
        blobs.append({"path": rel, "sha256": hashlib.sha256(blob.stdout).hexdigest()})
    from aadistill.infrastructure.source_identity import canonical_source_digest
    at_commit = canonical_source_digest(blobs)
    if at_commit != source["digest"]:
        raise SystemExit(
            f"refusing to issue: the source at {commit} digests to {at_commit} but "
            f"the working tree gives {source['digest']}. Commit the executable "
            "first; the pod runs the commit, not the tree.")

    prereg = json.loads((REPO_ROOT / PREREGISTRATION).read_text())
    if prereg["executable_source"]["digest"] != source["digest"]:
        raise SystemExit(
            "refusing to issue: the preregistration was frozen against executable "
            f"{prereg['executable_source']['digest'][:12]} but the executable is "
            f"{source['digest'][:12]}. Re-freeze it rather than authorizing code "
            "the record does not describe.")
    if prereg["session_plan"]["plan_hash"] != CONTINUATION_PLAN_V1.plan_hash:
        raise SystemExit("refusing to issue: the preregistration binds a different plan")

    for name, path in (("historical", HISTORICAL_REUSE), ("Attempt-5", ATTEMPT5_REUSE)):
        record = json.loads((REPO_ROOT / path).read_text())
        if not record.get("reuse_verified"):
            raise SystemExit(
                f"refusing to issue: the {name} probe reuse record is unverified. "
                "The 1-to-3 probe price assumes those citations; without them this "
                "is a larger session.")

    assets = json.loads((REPO_ROOT / ASSETS).read_text())
    if not assets["verification"]["verified"]:
        raise SystemExit(
            "refusing to issue: the relay copy of the one advancing checkpoint no "
            "prior session staged is unverified")

    priced = json.loads((REPO_ROOT / PRICING).read_text())["total"]
    if abs(priced["hard_usd"] - HARD_CEILING_USD) > 1e-6:
        raise SystemExit(
            f"refusing to issue: pricing says ${priced['hard_usd']:.4f}, the issuer "
            f"${HARD_CEILING_USD:.4f}")
    if HARD_CEILING_USD >= FULL_PHASE_B_CEILING_USD:
        raise SystemExit("refusing to issue: this is a full-Phase-B ceiling")

    profiles = (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)
    for profile in profiles:
        if not profile.materialized or not profile.content_sha256:
            raise SystemExit(f"refusing to issue: {profile.qualified_id} is not materialized")

    evidence = observed_evidence()
    missing = [k for k in BOUND_EVIDENCE if not evidence.get(k)]
    if missing:
        raise SystemExit(f"refusing to issue: could not re-derive {missing}")

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

    auth = ContinuationAuthorization(
        authorization_id=f"autoinit.continuation_b.{granted:%Y%m%dT%H%M%SZ}",
        granted_utc=granted.isoformat(),
        granted_by=granted_by,
        plan_id=CONTINUATION_PLAN_V1.plan_id,
        plan_hash=CONTINUATION_PLAN_V1.plan_hash,
        science_plan_hash=science_plan_hash,
        calibration_profile_hashes={p.qualified_id: p.profile_hash for p in profiles},
        calibration_content_hashes={p.qualified_id: p.content_sha256 for p in profiles},
        bound_evidence=evidence,
        planning_floor_usd=PLANNING_FLOOR_USD,
        hard_cap_usd=HARD_CEILING_USD,
        per_launch_hard_usd=HARD_CEILING_USD,
        authorized_stages=(0, 1, 3, 4, 5),
        stage_conditions={
            "0": ("attestation; generation_runtime_comparability@v2; the v3 "
                  "continuation executable-source digest; BOTH calibration "
                  "identities per profile; and every one of the six cited "
                  "evidence identities. A comparability FAILURE TERMINATES: every "
                  "cited observation would be lost at once and re-buying them is a "
                  "different, larger session"),
            "1": ("import the completed behavioural state. NO search, NO new sa "
                  "probe, NO recomputation of rung 1. The six-candidate collapsed "
                  "universe must rebuild to the bound identity, and exactly THREE "
                  "finalists — fe9683e6a9c7, 85bde4ded2c3 and the canonical "
                  "control — may enter the probe stages. The three searched "
                  "non-survivors are evidence and are never materialized"),
            "2": ("ABSENT BY DESIGN. Phase A's stage 2 is rung 1 on seed sa, which "
                  "this session imports as completed evidence rather than buying"),
            "3": ("rung 2 on sb, then the frozen pooled sa+sb decision unchanged. "
                  "Only genuinely missing observations are run: fe9683e6a9c7/sb is "
                  "bought; 85bde4ded2c3/sb and control/sb are cited"),
            "4": ("conditional tie-break on sc, only for candidates inside the "
                  "frozen equivalence interval that lack a verified sc. At worst "
                  "fe9683e6a9c7/sc and control/sc; 85bde4ded2c3/sc is cited. NO "
                  "fourth seed, ever"),
            "5": ("final selection and report under the frozen rule. "
                  "unresolved_equivalence with winner=None is a RESULT. Stage-1 "
                  "ranking, search-side KL/NLL, the canonical Stage-1 NLL and "
                  "state-id ordering may NOT break a tie"),
            "teardown": ("retain probe, raw generation and scoring evidence and the "
                         "final report; collect provider and billing evidence; "
                         "delete the pod; confirm from the provider that nothing "
                         "remains running; STOP for review"),
        },
        scope_note=(
            "ONE behavioural-continuation execution, one launcher invocation. This "
            "artifact authorizes the SPEND and the STAGES; it is not by itself an "
            "instruction to launch. It CANNOT authorize a search: runs_search is "
            "False by type. Nothing here permits a second attempt, re-buying "
            "Phase-B Stage 1, a fourth seed, backfilling rank 6/7, probing a "
            "searched non-survivor, post-Phase-B generalization, DEPTH incumbent "
            "selection, ATTENTION/FFN/RESIDUAL_WIDTH research, the canonical "
            "Stage-1 NLL, or formal Stage 2/3 recovery training. The continuation "
            "is a terminus and stops for review on every path."),
        authorized_session_commit=commit,
        source_digest=source["digest"],
        source_files=CONTINUATION_SOURCE_FILES_V2,
        provenance_commit=commit,
    )

    payload = auth.as_dict()
    # Added BEFORE the hash is computed. `load()` recomputes over everything
    # except the hash field, so a key appended afterwards is indistinguishable
    # from an edit — which is the point of the check, and it caught exactly this
    # on the Phase-B issuer's first run.
    payload.pop("authorization_sha256")
    payload["delegated_identities"] = {
        "trainer": trainer_source_digest(REPO_ROOT)["digest"],
        "generation": generation_source_digest(REPO_ROOT)["digest"],
        "scoring_contract": recovery_scoring_contract(REPO_ROOT)["digest"],
        "_why_here": ("identities the continuation DELEGATES rather than digests: "
                      "they are verified at stage 0 by the code that owns them"),
    }
    payload["source_set_version"] = source["set_version"]
    payload["source_digest_algorithm"] = source["algorithm"]
    payload["preregistration_sha256"] = prereg["preregistration_sha256"]
    payload["relay_assets"] = {
        "repo": assets["repo"], "repo_type": assets["repo_type"],
        "state_id": assets["state_id"],
        "artifact_digest": assets["artifact_digest"],
        "assets_sha256": assets["assets_sha256"],
        "verified_by_round_trip": True,
    }
    payload["cumulative_cap_usd"] = CUMULATIVE_CAP_USD
    payload["one_use"] = True
    payload["authorization_sha256"] = sha256_json(payload)

    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise SystemExit(
            f"refusing to issue: {out} already exists. A continuation "
            "authorization is ONE-USE; overwriting one would erase the record of "
            "what was permitted.")
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"authorization_id   {payload['authorization_id']}")
    print(f"granted_utc        {payload['granted_utc']}")
    print(f"session commit     {commit}")
    print(f"source digest      {source['digest']}  (v{source['set_version']}, "
          f"{source['n_files']} files)")
    print(f"  re-derived at commit  {at_commit}  MATCH")
    print(f"session plan       {CONTINUATION_PLAN_V1.plan_hash}")
    print(f"science plan       {science_plan_hash}")
    print(f"preregistration    {payload['preregistration_sha256']}")
    for key in BOUND_EVIDENCE:
        print(f"  {key:38} {evidence[key][:16]}")
    print(f"relay assets       {assets['assets_sha256'][:16]}  "
          f"({assets['state_id'][:12]} -> {assets['artifact_digest'][:12]})")
    print(f"runs_search        {payload['runs_search']}")
    print(f"followon           {payload['automatic_followon_start']}")
    print(f"planning floor     ${PLANNING_FLOOR_USD:.4f}  (NOT an expected spend)")
    print(f"HARD CEILING       ${HARD_CEILING_USD:.4f}  "
          f"(NOT the historical ${FULL_PHASE_B_CEILING_USD:.4f})")
    print(f"cumulative cap     ${CUMULATIVE_CAP_USD:.2f}")
    print(f"authorization_sha  {payload['authorization_sha256']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

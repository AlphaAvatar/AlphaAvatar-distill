#!/usr/bin/env python3
"""Assemble a Phase-C2 Search-1 authorization payload. Application layer.

    src/aadistill/governance/grant.py    the MECHANISM: stated vs derived
                                         fields, cap arithmetic. No experiment.
    configs/experiments/phase_c2/        the INSTANCE: the stage, the money
      authorization.json                 figures to refuse against, the two
                                         stage conditions.
    this file                            the ASSEMBLER: reads that config,
                                         RE-DERIVES every identity, builds the
                                         payload.

**Both callers use this same function.** `scripts/autoinit/issue_c2_authorization.py`
supplies `git rev-parse HEAD` and the wall clock; a test supplies fixed strings
and writes into `tmp_path`. A test candidate and a live authorization therefore
cannot diverge in derivation.

What C2 verifies, and how it differs from C1
--------------------------------------------

A C1 grant may not assert a derived identity at all. A C2 grant does the
opposite and says so in its own text: it records, under
`bound_identities_the_issuer_must_reproduce`, the identities the maintainer
approved against, with the rule *"RE-DERIVED by the issuer from the committed
objects. The issuer refuses if any of them disagrees."* So each asserted
identity is re-derived here and compared, and an identity this issuer does not
know how to derive is itself a refusal — a grant cannot introduce a binding
nobody checks.

Seven identities are re-derived from the tree, and **none of them is a commit**:

* the LIVE derived C2 executable closure — digest and file count, from
  `c2_current_executable`, not from any declared list;
* the C2 plan hash, which covers the session contract AND the configured space;
* the session-contract hash;
* the pricing record's own `pricing_sha256`, and the exact $15.0446 ceiling it
  derives;
* the frozen baseline B's spec hash and artifact digest, re-derived by
  constructing the frozen fixed path rather than read from a constant;
* a valid `launch_bound` C2 readiness record, PASS, against this same tree.

That last one is why this file refuses rather than warns. C1 attempt 2 died at a
marker because a readiness record described a tree that had moved; the readiness
check belongs where the authorization is created, not only where it is consumed.

**No commit is a verified identity, and that is deliberate.** A
`reviewed_commit` was one until 2026-09-16, derived as the issuer's own
`session_commit` — which made the chain unsatisfiable. The formal order is:
grant committed → `launch_bound` sweep on that clean grant-containing tree →
commit only the readiness record → issue on the resulting clean HEAD. The
issuance HEAD is therefore always later than the grant, so satisfying that
identity meant predicting a SHA that did not exist yet; and amending the grant
after the sweep would have invalidated the readiness lineage it was swept under.

Nothing replaced it. The execution tree is bound by the chain: the grant is
present in the swept base, the launch-bound record binds that base together with
the executable and environment digests, the authorization is issued against the
clean post-readiness HEAD and records it as `authorized_session_commit`, and the
session lineage rule then permits only the authorization artifact after that
base. A grant may name the commit its review was given against as prose outside
the identities block — human provenance, not a machine-checked equality.

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

from aadistill.governance.grant import (  # noqa: E402
    GrantContract, GrantRefused, validate_grant,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

CONFIG = "configs/experiments/phase_c2/authorization.json"

#: The module's public exception, so both callers catch one name.
C2AuthorizationRefused = GrantRefused


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


def flatten_money(grant: Mapping[str, Any],
                  cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The grant, with its two budget figures lifted to the top level.

    `validate_grant` does the cap arithmetic — `spend + ceiling <= cap`, with
    the numbers in the refusal — over a FLAT mapping with configurable field
    names. A C2 grant records the project's position in one nested block
    instead, so the block is named in the config and read here. The generic
    mechanism stays free of this document's shape, and the arithmetic is still
    the one place it happens.
    """
    money = cfg["grant_contract"]["money"]
    block = grant.get(money["block"])
    if not isinstance(block, Mapping):
        raise GrantRefused(
            f"the grant carries no {money['block']!r} block, so the cumulative "
            "spend and cap the ceiling must fit inside cannot be read. A grant "
            "states the budget position it was given at.")
    missing = [k for k in (money["spend"], money["cap"]) if k not in block]
    if missing:
        raise GrantRefused(
            f"{money['block']}.{missing} missing: the cap arithmetic needs both "
            "the cumulative spend at approval and the authorized cap.")
    return {**grant,
            "cumulative_spend_at_approval_usd": float(block[money["spend"]]),
            "cumulative_cap_usd": float(block[money["cap"]])}


def live_identities(repo_root: str | Path = ".") -> dict[str, Any]:
    """Every identity a C2 grant may assert, DERIVED from this tree.

    One function, so the issuer and any checker compare against the same
    derivation. Each value is computed here and now; nothing is read from a
    record that claims it.

    **It takes no commit, and that is the repair.** It used to take
    `session_commit` and return it as `reviewed_commit`, a verified identity —
    which made the launch chain unsatisfiable, because the issuer's
    `session_commit` is the clean HEAD *after* the readiness record has been
    committed, and a grant is authored *before* the sweep that produces it. The
    grant would have had to predict a SHA that did not yet exist, and editing
    the grant afterwards would have invalidated the readiness lineage it was
    swept under.

    Every identity below is a property of the TREE, not of a point in its
    history, so each one is satisfiable by a grant written at any time before
    issuance. Removing the parameter is what keeps it that way: a future commit
    field cannot be added back by accident when there is no commit to reach for.

    What binds the execution tree is the chain, not a field in the grant: the
    grant is present in the swept base, the launch-bound record binds that base,
    the authorization is issued against the post-readiness HEAD and records it
    as `authorized_session_commit`, and the lineage rule then permits only the
    authorization artifact after that base.
    """
    from experiments.phase_c2 import baseline as B
    from experiments.phase_c2.search_space import register_c2_operators
    from experiments.phase_c2.session import (
        C2_SESSION_CONTRACT, c2_current_executable, c2_hard_ceiling_usd,
        c2_plan_hash, load_pricing,
    )

    #: The operator must be registered before the frozen construction can be
    #: re-derived: `attention.activation_importance_v1` is not a shipped
    #: default, and `build_arm_specs` names it.
    register_c2_operators()
    root = Path(repo_root)
    closure = c2_current_executable(root)
    pricing = load_pricing(root)
    spec = B.frozen_baseline_spec(device="cuda")
    return {
        "c2_harness_digest": closure["digest"],
        "c2_harness_n_files": closure["n_files"],
        "c2_plan_hash": c2_plan_hash(),
        "c2_session_contract_hash": C2_SESSION_CONTRACT.contract_hash,
        "pricing_sha256": pricing["pricing_sha256"],
        "baseline_spec_hash": spec.spec_hash,
        "baseline_artifact_digest": B.B_ARTIFACT_DIGEST,
        #: Not an identity the grant asserts; carried so the payload records the
        #: set its digest was computed over, which the launch gate re-digests.
        "_harness_source_files": tuple(f["path"] for f in closure["files"]),
        "_hard_ceiling_usd": c2_hard_ceiling_usd(root),
    }


def verify_asserted_identities(grant: Mapping[str, Any], live: Mapping[str, Any],
                               cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Compare what the grant asserts against what this tree derives.

    Three ways to fail, all of them closed:

    * an asserted identity that disagrees with the derivation — the grant
      describes a different tree;
    * an asserted identity this issuer cannot derive — a binding nobody checks
      is not a binding, and a grant must not be able to introduce one;
    * no identities block at all — a C2 grant states its approval basis, and one
      that states none cannot be checked against anything.
    """
    gc = cfg["grant_contract"]
    block = gc["identities_block"]
    known = tuple(gc["verified_identities"])
    asserted = grant.get(block)
    if not isinstance(asserted, Mapping) or not asserted:
        raise GrantRefused(
            f"the grant carries no {block!r}. A C2 grant states the identities "
            "the approval was given against so the issuer can re-derive them; "
            "one that states none authorizes an unchecked tree.")

    #: Keys beginning with an underscore are the grant's own prose about the
    #: rule -- `_rule` explains that the issuer re-derives these -- and are not
    #: identity claims.
    claims = {k: v for k, v in asserted.items() if not k.startswith("_")}
    unknown = sorted(set(claims) - set(known))
    if unknown:
        #: Named specifically, because it is the one a reader expects to be
        #: there: every grant written before 2026-09-16 carries it, and the
        #: reason it is gone is a chronology, not a typo.
        if "reviewed_commit" in unknown:
            raise GrantRefused(
                "the grant asserts `reviewed_commit` inside the identities "
                "block. No commit is a verified identity: the issuance HEAD is "
                "the clean tree AFTER the readiness record is committed, so a "
                "grant authored before the sweep cannot state it, and amending "
                "the grant afterwards would invalidate the readiness lineage. "
                "Move it out of the block as prose if it documents which commit "
                "the review was given against; the execution tree is bound by "
                "the swept base, the readiness record and "
                "`authorized_session_commit`."
                + (f" Also unrecognised: {sorted(set(unknown) - {'reviewed_commit'})}."
                   if set(unknown) - {"reviewed_commit"} else ""))
        raise GrantRefused(
            f"the grant asserts {unknown}, which this issuer does not derive. "
            f"It derives {sorted(known)}. An identity nobody re-computes is not "
            "a binding, and a grant may not introduce one. If the mechanism it "
            "names has been replaced, the grant is superseded: write a new one "
            "against the current derivation rather than teaching the issuer to "
            "ignore a field.")
    missing = sorted(set(known) - set(claims))
    if missing:
        raise GrantRefused(
            f"the grant asserts no {missing}; every identity in the contract's "
            "verified set must be stated so it can be checked.")

    disagree = {k: {"grant": claims[k], "derived": live[k]}
                for k in known if claims[k] != live[k]}
    if disagree:
        lines = "; ".join(
            f"{k}: grant {str(v['grant'])[:16]}… vs derived "
            f"{str(v['derived'])[:16]}…" for k, v in sorted(disagree.items()))
        raise GrantRefused(
            f"{len(disagree)} asserted identity/identities no longer reproduce "
            f"against this tree — {lines}. The grant describes a different "
            "executable, plan, price or baseline than the one that would run.")
    return {"block": block, "verified": sorted(known),
            "rule": ("every identity the grant asserts was re-derived from the "
                     "committed objects and compared; none was trusted")}


def require_readiness(session_commit: str, run_id: str, stage_id: str,
                      repo_root: str | Path = ".") -> dict[str, Any]:
    """A launch-bound readiness PASS for THIS run against THIS tree, or refuse.

    Verified with the production verifier under the production contract, so the
    question asked here is the question the pre-provider gate asks. The
    authorization path is passed because the tree the pod checks out carries it
    by construction: omitting it made the authorization itself read as post-sweep
    drift, which cost one of C1 attempt 4's six failures.
    """
    from experiments.phase_c2 import pod_environment as PE
    from aadistill.runtime.pod_environment import LAUNCH_BOUND
    from experiments.phase_c2.session import c2_authorization_path

    rel = PE.record_path_for(run_id, stage_id)
    path = Path(repo_root) / rel
    if not path.is_file():
        raise GrantRefused(
            f"{rel} does not exist: this run has no readiness record, so "
            "nothing shows that a pod's test gate can pass on this tree. Run "
            "`record_pod_environment.py --experiment phase_c2 --kind "
            f"launch_bound --run-id {run_id} --stage-id {stage_id}` on the "
            "final clean pre-authorization tree and commit only that record.")
    record = json.loads(path.read_text())
    ok, why = PE.verify_record(
        record, repo_root, run_id=run_id, stage_id=stage_id,
        session_commit=session_commit, required_kind=LAUNCH_BOUND,
        authorization_path=c2_authorization_path(run_id, stage_id))
    if not ok:
        raise GrantRefused(f"the C2 readiness record is not usable: {why}")
    return {"record": rel, "record_kind": record.get("record_kind"),
            "self_sha256": record.get("self_sha256"),
            "swept_base_commit": record.get("swept_base_commit"),
            "counts": record.get("counts"), "verified": why}


def build_c2_authorization_payload(
    *,
    grant: Mapping[str, Any],
    session_commit: str,
    granted_utc: str,
    run_id: str,
    repo_root: str | Path = ".",
    grant_path: str = "<in-memory>",
    stage_id: str | None = None,
    require_readiness_record: bool = True,
    harness_digest_override: str | None = None,
) -> dict[str, Any]:
    """The complete payload, as a function of the committed objects.

    `session_commit` and `granted_utc` are the caller's: which commit the pod
    will check out, and when a human granted this, are the two facts a pure
    function cannot know.

    `require_readiness_record=False` exists ONLY so a test can build a candidate
    on a tree that has no sweep, and so this assembler can be exercised at $0
    before a launch-bound record is owed. It is never passed by the issuer, and
    a payload built with it says so in `bound.readiness`.

    `harness_digest_override` exists ONLY so a test can build a deliberately
    stale candidate and watch the launch gate refuse it. It is applied AFTER
    every agreement check, so such a candidate is stale in exactly one field.
    """
    from experiments.phase_c2.session import C2Authorization, C2ResourceScope

    root = Path(repo_root)
    cfg = load_config(root)
    stage_id = stage_id or cfg["stage_id"]
    money = cfg["accepted_pricing"]
    HARD_CEILING_USD = float(money["hard_ceiling_usd"])
    CUMULATIVE_CAP_USD = float(money["cumulative_cap_usd"])
    GRANT_FIELDS = tuple(cfg["grant_contract"]["stated_fields"])

    grant = validate_grant(flatten_money(grant, cfg), grant_contract(cfg),
                           ceiling_usd=HARD_CEILING_USD,
                           expected_cap_usd=CUMULATIVE_CAP_USD)

    live = live_identities(root)

    # --- refuse on any disagreement between the committed objects ----------
    problems = []
    if abs(live["_hard_ceiling_usd"] - HARD_CEILING_USD) > 1e-9:
        problems.append(
            f"the pricing record derives ${live['_hard_ceiling_usd']:.4f} and "
            f"the accepted ceiling is ${HARD_CEILING_USD:.4f}")
    approved = grant.get("approved_money") or {}
    for field, expected in (("hard_cap_usd", HARD_CEILING_USD),
                            ("expected_usd", float(money["expected_usd"])),
                            ("soft_stop_usd", float(money["soft_stop_usd"])),
                            ("price_basis_usd_per_hour",
                             float(money["price_per_hour_usd"]))):
        if field in approved and abs(float(approved[field]) - expected) > 1e-9:
            problems.append(
                f"the grant's approved_money.{field} is {approved[field]} and "
                f"the accepted figure is {expected}")
    if problems:
        raise C2AuthorizationRefused("; ".join(problems))

    identities = verify_asserted_identities(grant, live, cfg)
    readiness = (require_readiness(session_commit, run_id, stage_id, root)
                 if require_readiness_record
                 else {"required": False,
                       "why": ("built without a launch-bound readiness record. "
                               "This payload is a candidate, not an issuable "
                               "authorization; the issuer always requires one.")})

    digest = (harness_digest_override if harness_digest_override is not None
              else live["c2_harness_digest"])
    #: The maintainer-stated limits, made machine-readable and bound to THIS
    #: run. They were prose inside the grant until 2026-09-16, which meant "at
    #: most 3 provider resources" was checked by nobody: `--host-draws`
    #: defaulted to 3 and an operator could pass any number.
    scope = C2ResourceScope.from_grant(grant, run_id)
    auth = C2Authorization(
        authorization_id=cfg["authorization_id"],
        granted_utc=granted_utc,
        granted_by=grant["granted_by"],
        plan_id=cfg["plan_id"],
        plan_hash=live["c2_plan_hash"],
        #: The SCIENCE plan is the session contract: the stages, the fact that
        #: nothing is trained, the ranking metric and what that metric is not.
        #: C1 uses its C0 preregistration hash here; C2 has no C0.
        science_plan_hash=live["c2_session_contract_hash"],
        expected_usd=float(money["expected_usd"]),
        hard_cap_usd=live["_hard_ceiling_usd"],
        per_launch_hard_usd=live["_hard_ceiling_usd"],
        authorized_stages=tuple(cfg["authorized_stages"]),
        stage_conditions=dict(cfg["stage_conditions"]),
        scope_note=grant["covers"],
        authorized_session_commit=session_commit,
        harness_source_digest=digest,
        #: The set the digest was computed over — the LIVE closure, not any
        #: declaration. `session_commit_gate` re-digests this field at the
        #: session commit and compares it to `harness_source_digest`, so the two
        #: must describe the same files or the gate can never pass. C1 shipped
        #: authorizations for two weeks that declared 73 pre-migration paths
        #: while binding a digest over 97 real ones.
        harness_source_files=tuple(live["_harness_source_files"]),
        resource_scope=scope,
        provenance_commit=session_commit)

    payload = auth.as_dict()
    #: `run_id` is already serialized by `as_dict`, FROM the scope, so the two
    #: cannot disagree. Asserted rather than re-assigned: writing it again here
    #: is how a document comes to carry two sources for one fact.
    assert payload["run_id"] == run_id, (payload["run_id"], run_id)
    payload["stage_id"] = stage_id
    payload["grant"] = {
        "path": str(grant_path),
        "sha256": sha256_json({k: v for k, v in grant.items()
                               if k != "grant_sha256"}),
        **{k: grant[k] for k in GRANT_FIELDS if k in grant},
    }
    payload["bound"] = {
        "c2_harness_digest": digest,
        "c2_harness_n_files": live["c2_harness_n_files"],
        "c2_harness_rule": (
            "the LIVE derived executable closure: the transitive import walk "
            "from the declared entry points, the in-repo scripts they run as "
            "subprocesses, and the declared non-python inputs. NOT a "
            "hand-maintained list — the eighteen-path declaration the "
            "superseded attempt-1 grant binds named none of the 55 files the "
            "walk finds."),
        "c2_plan_hash": live["c2_plan_hash"],
        "c2_session_contract_hash": live["c2_session_contract_hash"],
        "pricing_sha256": live["pricing_sha256"],
        "baseline": {"spec_hash": live["baseline_spec_hash"],
                     "artifact_digest": live["baseline_artifact_digest"],
                     "rule": ("re-derived by constructing the frozen C1 fixed "
                              "path, not read from a constant")},
        "identities_verified": identities,
        "resource_scope": scope.as_dict(),
        "readiness": readiness,
        "pricing": {
            "expected_usd": float(money["expected_usd"]),
            "soft_stop_usd": float(money["soft_stop_usd"]),
            "hard_ceiling_usd": live["_hard_ceiling_usd"],
            "price_per_hour_usd": float(money["price_per_hour_usd"]),
            "cumulative_cap_usd": CUMULATIVE_CAP_USD,
        },
    }
    payload["one_use"] = cfg["one_use"]
    payload["does_not_authorize"] = list(
        grant.get("explicitly_not_authorized") or [])
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

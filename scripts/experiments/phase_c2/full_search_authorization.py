"""Assemble ONE Phase-C2 full-search authorization from a committed grant.

The grant says what a maintainer approved. This module re-derives every machine
identity that grant asserts, recomputes its money against the pricing record,
and refuses on any disagreement — so an authorization can only ever describe the
tree it was issued on.

**The rate is re-quoted, and the ceiling follows it mechanically.** The pricing
record's `$1.09/h` is planning evidence with a date on it. A grant states the
rate that was live when the maintainer approved, and `check_approved_money`
requires the approved ceiling to be exactly `derive_ceiling_usd(that rate)` —
the same MINUTES at a different price. A dearer GPU therefore buys the same
search for more money, and if that no longer fits the envelope the answer is a
maintainer decision. It is never a narrower beam: the width comes from
`SCHEDULE_V1` and nothing here can change it.

What the identities bind is the whole reason this is a separate type from
Search-1's: the SPACE. Search-1 fixed three operators and varied one; this
searches all four jointly, with calibration unpinned. A grant that asserted
Search-1's space size would be refused here, and vice versa.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import AuthorizationError
from aadistill.governance.grant import (
    GrantContract, GrantRefused, budget_headroom, validate_grant)
from aadistill.infrastructure.manifest import sha256_json

from experiments.phase_c2 import full_search as FSG
from experiments.phase_c2.session import C2ResourceScope

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = "configs/experiments/phase_c2/full_search_authorization.json"

#: The grant fields carried into the artifact for a reader.
GRANT_FIELDS: tuple[str, ...] = ("granted_by", "covers",
                                 "explicitly_not_authorized", "one_use")


class FullSearchAuthorizationRefused(RuntimeError):
    """The grant cannot legitimately authorize a full joint re-search."""


def load_config(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    return json.loads((Path(repo_root) / CONFIG).read_text())


def grant_contract(cfg: Mapping[str, Any]) -> GrantContract:
    c = cfg["grant_contract"]
    return GrantContract(
        contract_id=c["contract_id"],
        stated_fields=tuple(c["stated_fields"]),
        derived_fields=tuple(c["derived_fields"]),
        spend_field=c["money"]["spend_field"],
        cap_field=c["money"]["cap_field"])


def flatten_money(grant: Mapping[str, Any],
                  cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The grant, with its two budget figures lifted to the top level.

    `validate_grant` does the cap arithmetic over a FLAT mapping with
    configurable field names; a grant records the project's position in one
    nested block. Flattened here so the arithmetic stays in one place and the
    generic mechanism stays free of this document's shape.
    """
    money = cfg["grant_contract"]["money"]
    block = grant.get(money["block"])
    if not isinstance(block, Mapping):
        raise FullSearchAuthorizationRefused(
            f"the grant has no {money['block']!r} block, so its position "
            "against the project cap cannot be checked")
    flat = dict(grant)
    for field in ("spend_field", "cap_field"):
        key = money[field]
        if key not in block:
            raise FullSearchAuthorizationRefused(
                f"{money['block']}.{key} is missing from the grant")
        flat[key] = block[key]
    return flat


def live_identities(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Every identity a full-search grant may assert, DERIVED from this tree.

    One function, so the issuer and any checker compare against the same
    derivation. Each value is computed here and now; nothing is read from a
    record that claims it. It takes no commit.
    """
    from aadistill.initialization.planning.ranking import PARETO_V1
    from experiments.phase_c2 import full_search_space as FS

    #: Neither the promoted ATTENTION operator nor the calibration mixtures are
    #: shipped defaults, and the joint space cannot be enumerated without them.
    #: Registered here rather than assumed: an issuer that derived "no space" on
    #: a fresh process would bind a session to an empty search.
    FS.register_c2_operators()

    live = FSG.current_executable(repo_root)
    report = FS.size_report(repo_root)["full_joint"]
    space = FS.full_joint_space(repo_root)
    pricing = FSG.pricing(repo_root)
    suite = json.loads(
        (Path(repo_root) / "configs/experiments/phase_c2/frozen_assets.json"
         ).read_text())["assets"]["state_eval_v1"]

    return {
        "full_search_harness_digest": live["digest"],
        "full_search_harness_n_files": live["n_files"],
        "full_search_plan_hash": FSG.plan_hash(repo_root),
        "full_search_pricing_sha256": pricing["pricing_sha256"],
        #: THE SPACE. This is what makes a Search-1 grant unusable here and a
        #: full-search grant unusable there: 578 reachable leaves, of which 576
        #: are four-operator decompositions and 2 are single-step composites.
        "joint_space_total_leaves": int(report["total_leaves"]),
        "joint_space_decomposed_leaves": int(report["decomposed_leaves"]),
        "joint_space_allowed_impls": sorted(space.allowed_impls),
        #: UNPINNED, and asserted as such. A pinned mapping is Search-1's
        #: restriction; carrying one here would silently re-restrict the search.
        "joint_space_impl_profiles_are_unpinned": space.impl_profiles is None,
        "excluded_implementations": sorted(FS.EXCLUSIONS),
        "calibration_profile_ids": list(FS.PROFILE_IDS),
        "ranking_policy_hash": PARETO_V1.policy_hash,
        "beam_schedule_id": FS.SCHEDULE_V1.schedule_id,
        "beam_width": FSG.standing_beam_width(),
        "beam_warmup_levels": int(FS.SCHEDULE_V1.warmup_levels),
        "state_eval_content_sha256": suite["content_sha256"],
        "cost_table_sources": [name for name, _t, _r in FS.TELEMETRY_SOURCES],
        "_harness_source_files": [row["path"] for row in live["files"]],
        "_expected_usd": FSG.expected_usd(repo_root),
        "_priced_basis_usd_per_hour": FSG.price_per_hour_basis(repo_root),
        "_hard_ceiling_minutes": float(
            FSG._standing_row(repo_root)["hard_ceiling_minutes"]),
    }


def check_approved_money(grant: Mapping[str, Any],
                         repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The maintainer's money boundary, re-derived at the grant's LIVE rate.

    `approved_money` is a STATED field: the maintainer says what was approved.
    That makes it evidence of a decision and not of a computation, so every
    figure is recomputed and any difference is a refusal.

    The rate is the part that moves. The pricing record's basis is planning
    evidence; a grant states the rate that was live when it was approved, and
    the ceiling it claims must be exactly the mechanical derivation at that
    rate — same minutes, current price. So:

    * a grant whose ceiling was computed at the stale basis while quoting a
      higher live rate is refused, because its ceiling would under-authorize
      the plan;
    * a grant that quoted a rate above the boundary it itself states is
      refused, because chasing a price is a maintainer decision;
    * and a grant that shrank the beam to hold the ceiling down cannot exist,
      because the width comes from `SCHEDULE_V1` and the minutes come from the
      pricing record's row for that width.
    """
    stated = grant.get("approved_money")
    if not isinstance(stated, Mapping) or not stated:
        raise FullSearchAuthorizationRefused(
            "the grant states no approved_money. A grant that does not say what "
            "money was approved is not evidence that any was.")
    required = ("expected_usd", "hard_cap_usd", "price_basis_usd_per_hour",
                "max_price_usd_per_hour")
    missing = sorted(k for k in required if k not in stated)
    if missing:
        raise FullSearchAuthorizationRefused(
            f"the grant's approved_money is missing {missing}. "
            "`price_basis_usd_per_hour` is the rate that was LIVE at approval "
            "and `max_price_usd_per_hour` the boundary it may not exceed; "
            "without both, a ceiling cannot be checked against a rate.")

    rate = round(float(stated["price_basis_usd_per_hour"]), 4)
    boundary = round(float(stated["max_price_usd_per_hour"]), 4)
    if rate > boundary + 1e-9:
        raise FullSearchAuthorizationRefused(
            f"the grant quotes a live rate of ${rate}/h above the "
            f"${boundary}/h boundary it states. A rate above the boundary needs "
            "a new maintainer budget decision; it is not a price to chase and "
            "the beam is not narrowed to absorb it.")

    ceiling = FSG.derive_ceiling_usd(rate, repo_root)
    expected_minutes_usd = round(
        FSG.expected_usd(repo_root) / FSG.price_per_hour_basis(repo_root)
        * rate, 4)
    expected = {
        "expected_usd": expected_minutes_usd,
        "hard_cap_usd": ceiling,
        "price_basis_usd_per_hour": rate,
        "max_price_usd_per_hour": boundary,
    }
    wrong = {k: {"grant": round(float(stated[k]), 4), "derived": expected[k]}
             for k in ("expected_usd", "hard_cap_usd")
             if round(float(stated[k]), 4) != expected[k]}
    if wrong:
        raise FullSearchAuthorizationRefused(
            "the grant's approved_money does not match the mechanical "
            f"derivation at its own ${rate}/h rate over the pricing record's "
            f"{expected['hard_cap_usd']} ceiling minutes: "
            + json.dumps(wrong, indent=1)
            + "\nThe minutes are the plan and the dollars are the minutes times "
              "the rate. Re-derive, do not adjust the beam.")
    return expected


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, run_id: str, stage_id: str | None = None,
                  repo_root: str | Path = REPO_ROOT,
                  grant_path: str | None = None) -> dict[str, Any]:
    """The artifact. Refuses on anything the grant cannot legitimately be."""
    repo_root = Path(repo_root)
    cfg = load_config(repo_root)
    stage_id = stage_id or cfg["stage_id"]
    contract = grant_contract(cfg)
    live = live_identities(repo_root)

    #: The ceiling is the one at the grant's LIVE rate, not the priced basis.
    approved_money = check_approved_money(grant, repo_root)
    ceiling = approved_money["hard_cap_usd"]

    try:
        validate_grant(flatten_money(grant, cfg), contract,
                       ceiling_usd=ceiling,
                       expected_cap_usd=float(
                           cfg["accepted_pricing"]["cumulative_cap_usd"]))
    except GrantRefused as exc:
        raise FullSearchAuthorizationRefused(str(exc)) from None

    #: Every asserted identity re-derived and compared, and the block required
    #: to be COMPLETE. Absence of a disagreement is not agreement: a grant that
    #: asserted two of eight identities would otherwise bind the session to an
    #: approval that had never named the other six.
    block = cfg["grant_contract"]["identities_block"]
    asserted_raw = grant.get(block)
    if not isinstance(asserted_raw, Mapping) or not asserted_raw:
        raise FullSearchAuthorizationRefused(
            f"the grant's {block!r} is absent or empty. A grant that names no "
            "machine identity approves nothing a machine can check.")
    asserted = {k: v for k, v in asserted_raw.items() if not k.startswith("_")}
    derivable = {k: v for k, v in live.items() if not k.startswith("_")}

    unknown = sorted(set(asserted) - set(derivable))
    if unknown:
        raise FullSearchAuthorizationRefused(
            f"the grant asserts {unknown}, which this issuer cannot derive. A "
            "grant may not introduce a binding nobody checks.")
    incomplete = sorted(set(derivable) - set(asserted))
    if incomplete:
        raise FullSearchAuthorizationRefused(
            f"the grant's {block} does not state {incomplete}. The block must "
            "name every identity the issuer derives: an unstated identity is "
            "one the maintainer never approved against, and issuing anyway "
            "would bind this session to an approval that does not mention it.")
    if "reviewed_commit" in asserted_raw:
        raise FullSearchAuthorizationRefused(
            "reviewed_commit is not a machine identity. The issuance HEAD is "
            "necessarily later than the grant, so a commit asserted here makes "
            "the chain unsatisfiable.")
    disagreements = {
        key: {"grant": asserted[key], "derived": derivable[key]}
        for key in derivable if asserted[key] != derivable[key]}
    if disagreements:
        raise FullSearchAuthorizationRefused(
            "the grant's asserted identities disagree with this tree: "
            + json.dumps(disagreements, indent=1))

    try:
        scope = C2ResourceScope.from_grant(grant, run_id)
    except AuthorizationError as exc:
        raise FullSearchAuthorizationRefused(str(exc)) from None

    readiness = _readiness(repo_root, run_id, stage_id, session_commit)

    auth = FSG.FullSearchAuthorization(
        authorization_id=cfg["authorization_id"],
        granted_by=grant["granted_by"],
        granted_utc=granted_utc,
        plan_id=cfg["plan_id"],
        plan_hash=live["full_search_plan_hash"],
        #: The SCIENCE plan is the full-search protocol: the space, the ranking
        #: policy, the schedule, what a Top-5 is and what it may be read as.
        #: For this session the operational plan and the scientific one are the
        #: same document, so both hashes are its own -- stated rather than
        #: hidden, because a reader should not have to infer that.
        science_plan_hash=live["full_search_plan_hash"],
        expected_usd=float(approved_money["expected_usd"]),
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=tuple(cfg["authorized_stages"]),
        stage_conditions=dict(cfg["stage_conditions"]),
        scope_note=grant["covers"],
        authorized_session_commit=session_commit,
        harness_source_digest=live["full_search_harness_digest"],
        #: The set the digest was computed over -- the LIVE closure, not any
        #: declaration. `session_commit_gate` re-digests this field at the
        #: session commit and compares it to the digest, so the two must
        #: describe the same files or the gate can never pass.
        harness_source_files=tuple(live["_harness_source_files"]),
        resource_scope=scope,
        provenance_commit=session_commit,
    )
    payload = auth.as_dict()
    assert payload["run_id"] == run_id, (payload["run_id"], run_id)
    payload["stage_id"] = stage_id
    payload["experiment_id"] = cfg["experiment_id"]
    payload["grant"] = {
        "path": grant_path,
        "sha256": sha256_json({k: v for k, v in grant.items()
                               if k != "grant_sha256"}),
        **{k: grant[k] for k in GRANT_FIELDS if k in grant},
    }
    payload["bound"] = {
        **{k: v for k, v in live.items() if not k.startswith("_")},
        "expected_usd": approved_money["expected_usd"],
        "price_per_hour_usd": approved_money["price_basis_usd_per_hour"],
        "hard_ceiling_minutes": live["_hard_ceiling_minutes"],
        "_rule": (
            "every value here was DERIVED from this tree at issuance: the "
            "closure by import walk from the full-search entry points, the "
            "space by walking the real registry through the same two branching "
            "functions the search itself calls, the plan from the protocol's "
            "own hash, the ceiling from the pricing record's minutes times the "
            "grant's re-quoted rate. None was read from a document that merely "
            "claims it."),
        "_the_ceiling_moved_with_the_rate": (
            f"priced at ${live['_priced_basis_usd_per_hour']}/h, issued at "
            f"${approved_money['price_basis_usd_per_hour']}/h over "
            f"{live['_hard_ceiling_minutes']} ceiling minutes. The MINUTES are "
            "the plan; a rate change re-derives the dollars and never the beam."),
    }
    payload["readiness"] = readiness
    payload["approved_money"] = {
        **approved_money,
        "_stated_and_rederived": (
            "the maintainer STATED these in the grant and the issuer recomputed "
            "the expected cost and the ceiling from the pricing record's "
            "minutes at the grant's own live rate. A grant naming a different "
            "figure, or a rate above the boundary it states, is refused."),
    }
    payload["budget_headroom"] = budget_headroom(
        cumulative_usd=float(flatten_money(grant, cfg)[contract.spend_field]),
        cap_usd=float(cfg["accepted_pricing"]["cumulative_cap_usd"]),
        ceiling_usd=ceiling)
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload


def _readiness(repo_root: Path, run_id: str, stage_id: str | None,
               session_commit: str) -> dict[str, Any]:
    """The launch-bound readiness record, verified against this tree.

    An authorization that did not name a swept record could be issued on a tree
    no sweep had ever described -- which is how a pod comes to run code the
    readiness evidence never covered.
    """
    from experiments.phase_c2 import full_search_pod_environment as PE

    record = PE.load_record(repo_root, run_id=run_id, stage_id=stage_id)
    PE.verify_record(record, repo_root, run_id=run_id, stage_id=stage_id,
                     session_commit=session_commit)
    return {
        "path": PE.record_path_for(run_id, stage_id),
        "kind": record.get("kind"),
        "swept_commit": record.get("commit"),
        "harness_digest": record.get("full_search_harness_digest"),
        "_rule": ("the sweep ran on the grant commit and this issuance is the "
                  "next commit after the record; the gate re-checks both."),
    }


__all__ = ["CONFIG", "FullSearchAuthorizationRefused", "GRANT_FIELDS",
           "build_payload", "check_approved_money", "flatten_money",
           "grant_contract", "live_identities", "load_config"]

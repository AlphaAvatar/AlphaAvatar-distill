"""Assemble a baseline-completion authorization from a maintainer grant.

`experiments.phase_c2.authorization_payload` does this for Search-1 and must
not be retargeted: it asserts `authorizes_c2_search1 is True`, derives the
SEARCH-1 closure, resolves the SEARCH-1 authorization path and checks the
$15.0446 ceiling that prices a ten-hour beam. A completion grant flowing
through it would be validated against another session's code, another session's
permission and another session's price -- and would pass.

So this is the completion's own assembler, over the same generic primitives:
`GrantContract` and `validate_grant` for the field split and the cap
arithmetic, `C2ResourceScope` for the one-use limits, and
`BaselineCompletionAuthorization` for the artifact.

**Every identity is DERIVED here, never trusted.** A grant may state what it
approved against; each statement is recomputed from its canonical owner and any
disagreement is a refusal. An identity this assembler does not know how to
derive is also a refusal, so a grant cannot introduce a binding nobody checks:

* the live completion executable closure -- digest AND file set, from one
  derivation;
* the completion plan hash, which is the protocol's own `protocol_sha256`;
* the completion pricing record's own `pricing_sha256`, with the `$1.1950`
  ceiling, the `$0.7212` expected cost and the `$1.09/h` basis read from it;
* the frozen baseline B's spec hash and expected artifact digest, re-derived by
  constructing the frozen fixed path rather than read from a constant;
* the frozen candidate record's `self_sha256` and the Search-1 selection
  commitment those candidates were extracted from;
* a valid `launch_bound` COMPLETION readiness record, PASS, against this tree;
* a `C2ResourceScope` for the exact run, built from the grant's own `one_use`
  block rather than from a default.

**It takes no reviewed commit and produces no verified commit identity.** The
issuer's HEAD is necessarily later than the grant -- grant committed, sweep on
that clean tree, record committed, then issuance -- so a grant that predicted
the issuance commit would make its own chain unsatisfiable. C1 attempt 15
aborted at `$0` on precisely that.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from aadistill.governance.authorization import AuthorizationError
from aadistill.governance.grant import (
    GrantContract, GrantRefused, budget_headroom, validate_grant,
)
from aadistill.infrastructure.manifest import sha256_json

from experiments.phase_c2 import baseline_completion as BC
from experiments.phase_c2.session import C2ResourceScope

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = "configs/experiments/phase_c2/baseline_completion_authorization.json"

#: Grant fields copied verbatim into the artifact's `grant` block, so a reader
#: of the authorization can see what was permitted without opening the grant.
GRANT_FIELDS: tuple[str, ...] = ("granted_by", "covers",
                                 "explicitly_not_authorized")


class CompletionAuthorizationRefused(RuntimeError):
    """The grant cannot become a baseline-completion authorization."""


def load_config(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    return json.loads((Path(repo_root) / CONFIG).read_text())


def grant_contract(cfg: Mapping[str, Any]) -> GrantContract:
    gc = cfg["grant_contract"]
    money = gc["money"]
    return GrantContract(contract_id=gc["contract_id"],
                         stated_fields=tuple(gc["stated_fields"]),
                         derived_fields=tuple(gc["derived_fields"]),
                         spend_field=money["spend_field"],
                         cap_field=money["cap_field"])


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
        raise CompletionAuthorizationRefused(
            f"the grant has no {money['block']!r} block, so its position "
            "against the project cap cannot be checked")
    flat = dict(grant)
    for field in ("spend_field", "cap_field"):
        key = money[field]
        if key not in block:
            raise CompletionAuthorizationRefused(
                f"{money['block']}.{key} is missing from the grant")
        flat[key] = block[key]
    return flat


def live_identities(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Every identity a completion grant may assert, DERIVED from this tree.

    One function, so the issuer and any checker compare against the same
    derivation. Each value is computed here and now; nothing is read from a
    record that claims it. It takes no commit.
    """
    from experiments.phase_c2 import baseline as B
    from experiments.phase_c2.frozen_inputs import load_record
    from experiments.phase_c2.search_space import register_c2_operators

    #: `attention.activation_importance_v1` is not a shipped default and the
    #: frozen B path's last step names it, so the spec cannot be constructed
    #: without registering the C2 operators first.
    register_c2_operators()

    live = BC.current_executable(repo_root)
    spec = B.frozen_baseline_spec(device="cuda")
    B.assert_frozen_construction(spec)
    frozen = load_record(Path(repo_root) / BC.FROZEN_INPUTS)
    pricing = BC.pricing(repo_root)

    return {
        "completion_harness_digest": live["digest"],
        "completion_harness_n_files": live["n_files"],
        "completion_plan_hash": BC.plan_hash(repo_root),
        "completion_pricing_sha256": pricing["pricing_sha256"],
        "baseline_spec_hash": spec.spec_hash,
        "baseline_artifact_digest": B.B_ARTIFACT_DIGEST,
        "frozen_candidates_self_sha256": frozen["self_sha256"],
        "search1_selection_commitment_sha256":
            frozen["sources"]["selection_commitment_sha256"],
        "_harness_source_files": [row["path"] for row in live["files"]],
        "_hard_ceiling_usd": BC.hard_ceiling_usd(repo_root),
        "_expected_usd": float(pricing["totals"]["expected_usd"]),
        "_price_per_hour_usd": BC.price_per_hour_usd(repo_root),
    }


def _readiness(repo_root: Path, run_id: str, stage_id: str) -> dict[str, Any]:
    """A PASSING launch-bound completion readiness record, or a refusal.

    Refuses rather than warns. An authorization issued without one binds a
    session to a tree nobody swept -- and it is the COMPLETION's record that is
    required: a Search-1 record describes another launcher, another session id,
    another staging contract and another closure.
    """
    from experiments.phase_c2 import baseline_completion_pod_environment as CPE

    rel = CPE.record_path_for(run_id, stage_id)
    try:
        record = CPE.load_record(repo_root, run_id=run_id, stage_id=stage_id)
    except FileNotFoundError:
        raise CompletionAuthorizationRefused(
            f"{rel} does not exist. A completion authorization rests on a "
            f"launch_bound sweep of THIS run, taken on the clean "
            f"grant-containing tree: record_pod_environment.py --experiment "
            f"{CPE.EXPERIMENT_ID} --kind {CPE.LAUNCH_BOUND} --run-id {run_id} "
            f"--stage-id {stage_id}") from None
    if record.get("schema") != CPE.SCHEMA:
        raise CompletionAuthorizationRefused(
            f"{rel} declares schema {record.get('schema')!r}, not "
            f"{CPE.SCHEMA!r}. A Search-1 readiness record cannot stand in for "
            "the completion's.")
    if record.get("record_kind") != CPE.LAUNCH_BOUND:
        raise CompletionAuthorizationRefused(
            f"{rel} is a {record.get('record_kind')!r} record. Only a "
            f"{CPE.LAUNCH_BOUND!r} sweep describes the tree a launch will use.")
    if record.get("verdict") not in ("PASS", None) or record.get("problems"):
        raise CompletionAuthorizationRefused(
            f"{rel} is not a PASS: {record.get('verdict')!r}, problems "
            f"{record.get('problems')}")
    return {"record": rel, "self_sha256": record.get("self_sha256"),
            "record_kind": record.get("record_kind"),
            "schema": record.get("schema"),
            "swept_base_commit": record.get("swept_base_commit"),
            "completion_harness_digest": record.get("completion_harness_digest"),
            "staging_contract_digest": record.get("staging_contract_digest")}


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
    ceiling = float(live["_hard_ceiling_usd"])

    try:
        validate_grant(flatten_money(grant, cfg), contract,
                       ceiling_usd=ceiling,
                       expected_cap_usd=float(
                           cfg["accepted_pricing"]["cumulative_cap_usd"]))
    except GrantRefused as exc:
        raise CompletionAuthorizationRefused(str(exc)) from None

    #: Every asserted identity re-derived and compared. An identity this
    #: assembler cannot derive is a refusal in itself.
    block = cfg["grant_contract"]["identities_block"]
    asserted = grant.get(block) or {}
    if not isinstance(asserted, Mapping):
        raise CompletionAuthorizationRefused(
            f"the grant's {block!r} is not a mapping")
    derivable = {k: v for k, v in live.items() if not k.startswith("_")}
    unknown = sorted(k for k in asserted
                     if not k.startswith("_") and k not in derivable)
    if unknown:
        raise CompletionAuthorizationRefused(
            f"the grant asserts {unknown}, which this issuer cannot derive. A "
            "grant may not introduce a binding nobody checks.")
    disagreements = {
        key: {"grant": asserted[key], "derived": derivable[key]}
        for key in derivable if key in asserted and asserted[key] != derivable[key]}
    if disagreements:
        raise CompletionAuthorizationRefused(
            "the grant's asserted identities disagree with this tree: "
            + json.dumps(disagreements, indent=1))

    try:
        scope = C2ResourceScope.from_grant(grant, run_id)
    except AuthorizationError as exc:
        raise CompletionAuthorizationRefused(str(exc)) from None

    readiness = _readiness(repo_root, run_id, stage_id)

    auth = BC.BaselineCompletionAuthorization(
        authorization_id=cfg["authorization_id"],
        granted_by=grant["granted_by"],
        granted_utc=granted_utc,
        plan_id=cfg["plan_id"],
        plan_hash=live["completion_plan_hash"],
        #: The SCIENCE plan is the completion protocol: what B is, how it is
        #: measured, what it is compared against and what the comparison may be
        #: read as. For this session the operational plan and the scientific one
        #: are the same document, so both hashes are its own -- stated rather
        #: than hidden, because a reader should not have to infer that.
        science_plan_hash=live["completion_plan_hash"],
        expected_usd=float(live["_expected_usd"]),
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=tuple(cfg["authorized_stages"]),
        stage_conditions=dict(cfg["stage_conditions"]),
        scope_note=grant["covers"],
        authorized_session_commit=session_commit,
        harness_source_digest=live["completion_harness_digest"],
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
        "expected_usd": live["_expected_usd"],
        "price_per_hour_usd": live["_price_per_hour_usd"],
        "_rule": (
            "every value here was DERIVED from this tree at issuance: the "
            "closure by import walk from the completion entry points, the plan "
            "from the protocol's own hash, the prices from the pricing record's "
            "own hash, the baseline identities by constructing the frozen fixed "
            "path, and the frozen candidate identities from the record's own "
            "self-hash. None was read from a document that merely claims it."),
    }
    payload["readiness"] = readiness
    payload["budget_headroom"] = budget_headroom(
        cumulative_usd=float(flatten_money(grant, cfg)[contract.spend_field]),
        cap_usd=float(cfg["accepted_pricing"]["cumulative_cap_usd"]),
        ceiling_usd=ceiling)
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

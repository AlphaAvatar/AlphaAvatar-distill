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


def _staging_contract_digest(run_id: str) -> str:
    """The LIVE completion staging contract, from the real SessionSpec.

    Derived from the completion launcher's own `SetupManifest` under the
    completion session id -- the same derivation the launcher's readiness gate
    performs -- so the issuer and the launcher cannot disagree about what this
    session stages. Deriving it from anything else would let an authorization be
    issued against a staged view no pod will ever have.
    """
    import importlib

    from aadistill.runtime.staging_contract import derive_contract

    from experiments.phase_c2 import baseline_completion as _BC

    launcher = importlib.import_module("autoinit_phase_c2_baseline_launch")
    args = launcher.build_parser().parse_args([
        "--scr", "/unused", "--session-commit", "0" * 40,
        "--bundle", "aad_autoinit_00000000.bundle", "--run-id", run_id])
    return derive_contract(launcher.spec(args).setup,
                           session_id=_BC.SESSION_ID)["digest"]


def _readiness(repo_root: Path, run_id: str, stage_id: str,
               session_commit: str) -> dict[str, Any]:
    """The completion readiness record, through the PRODUCTION verifier.

    This used to hand-check the schema, the kind and the verdict and stop there
    -- which accepted a record whose self-hash was wrong, whose harness digest
    described other code, whose staged view was another session's, or whose
    swept base could not possibly be an ancestor of the issuance HEAD. Those are
    exactly the questions `verify_record` already answers, and answering three
    of eight by hand is the shape that lets the other five through.

    So the real verifier runs, under this run, this stage, `launch_bound`, the
    issuance commit, the completion authorization path and the LIVE staging
    contract. No second readiness mechanism, and no partial acceptance.
    """
    from experiments.phase_c2 import baseline_completion as _BC
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

    authorization_path = (
        f"{_run_dir(run_id, stage_id)}/governance/authorization.json")
    try:
        staging = _staging_contract_digest(run_id)
    except Exception as exc:                                    # noqa: BLE001
        raise CompletionAuthorizationRefused(
            f"cannot derive the completion staging contract: {exc}") from None

    ok, reason = CPE.verify_record(
        record, repo_root, run_id=run_id, stage_id=stage_id,
        session_commit=session_commit,
        authorization_path=authorization_path,
        required_kind=CPE.LAUNCH_BOUND,
        staging_contract_digest=staging)
    if not ok:
        raise CompletionAuthorizationRefused(
            f"{rel} does not verify against this tree: {reason}")

    return {"record": rel, "verified": True, "verifier_reason": reason,
            "self_sha256": record.get("self_sha256"),
            "record_kind": record.get("record_kind"),
            "schema": record.get("schema"),
            "swept_base_commit": record.get("swept_base_commit"),
            "completion_harness_digest": record.get("completion_harness_digest"),
            "staging_contract_digest": record.get("staging_contract_digest"),
            "live_staging_contract_digest": staging,
            "_verified_by": (
                "experiments.phase_c2.baseline_completion_pod_environment."
                "verify_record, the same production verifier the launcher's "
                "readiness gate calls -- schema, self hash, verdict, harness "
                "digest, pod-test-environment digest, staging contract, clean "
                "swept base, swept-base-to-HEAD lineage and permitted "
                "post-sweep paths")}


def _run_dir(run_id: str, stage_id: str) -> str:
    from experiments.run_layout import rel_run_dir

    return rel_run_dir("phase_c2_baseline_completion", run_id, stage_id)


def check_approved_money(grant: Mapping[str, Any],
                         repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The maintainer's money boundary, re-derived from the pricing record.

    `approved_money` is a STATED field: the maintainer says what was approved.
    That makes it evidence of a decision and not of a computation, so every
    figure in it is recomputed from the canonical pricing record and any
    difference is a refusal. The grant states them because a person approved
    them; the machine checks them because a person can mistype.

    Nothing is copied into a second config: the pricing record is the owner.
    """
    doc = BC.pricing(repo_root)["totals"]
    expected = {
        "expected_usd": round(float(doc["expected_usd"]), 4),
        "hard_cap_usd": round(float(doc["hard_ceiling_usd"]), 4),
        "price_basis_usd_per_hour": round(float(doc["price_per_hour"]), 4),
    }
    stated = grant.get("approved_money")
    if not isinstance(stated, Mapping) or not stated:
        raise CompletionAuthorizationRefused(
            "the grant states no approved_money. A grant that does not say what "
            "money was approved is not evidence that any was.")
    missing = sorted(k for k in expected if k not in stated)
    if missing:
        raise CompletionAuthorizationRefused(
            f"the grant's approved_money is missing {missing}")
    wrong = {k: {"grant": stated[k], "pricing_record": expected[k]}
             for k in expected if round(float(stated[k]), 4) != expected[k]}
    if wrong:
        raise CompletionAuthorizationRefused(
            "the grant's approved_money disagrees with the completion pricing "
            "record: " + json.dumps(wrong, indent=1))
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
    ceiling = float(live["_hard_ceiling_usd"])

    try:
        validate_grant(flatten_money(grant, cfg), contract,
                       ceiling_usd=ceiling,
                       expected_cap_usd=float(
                           cfg["accepted_pricing"]["cumulative_cap_usd"]))
    except GrantRefused as exc:
        raise CompletionAuthorizationRefused(str(exc)) from None

    #: The maintainer's money boundary, recomputed from the pricing record.
    approved_money = check_approved_money(grant, repo_root)

    #: Every asserted identity re-derived and compared, and the block required
    #: to be COMPLETE. Absence of a disagreement is not agreement: a grant that
    #: asserted two of eight identities used to pass, binding the session to an
    #: approval that had never named the other six. So the comparison is set
    #: equality after prose keys are dropped, and each of the four ways it can
    #: fail is named.
    block = cfg["grant_contract"]["identities_block"]
    asserted_raw = grant.get(block)
    if not isinstance(asserted_raw, Mapping) or not asserted_raw:
        raise CompletionAuthorizationRefused(
            f"the grant's {block!r} is absent or empty. A grant that names no "
            "machine identity approves nothing a machine can check.")
    asserted = {k: v for k, v in asserted_raw.items() if not k.startswith("_")}
    derivable = {k: v for k, v in live.items() if not k.startswith("_")}

    unknown = sorted(set(asserted) - set(derivable))
    if unknown:
        raise CompletionAuthorizationRefused(
            f"the grant asserts {unknown}, which this issuer cannot derive. A "
            "grant may not introduce a binding nobody checks.")
    incomplete = sorted(set(derivable) - set(asserted))
    if incomplete:
        raise CompletionAuthorizationRefused(
            f"the grant's {block} does not state {incomplete}. The block must "
            "name every identity the issuer derives: an unstated identity is "
            "one the maintainer never approved against, and issuing anyway "
            "would bind this session to an approval that does not mention it.")
    if "reviewed_commit" in asserted_raw:
        raise CompletionAuthorizationRefused(
            "reviewed_commit is not a machine identity. The issuance HEAD is "
            "necessarily later than the grant, so a commit asserted here makes "
            "the chain unsatisfiable.")
    disagreements = {
        key: {"grant": asserted[key], "derived": derivable[key]}
        for key in derivable if asserted[key] != derivable[key]}
    if disagreements:
        raise CompletionAuthorizationRefused(
            "the grant's asserted identities disagree with this tree: "
            + json.dumps(disagreements, indent=1))

    try:
        scope = C2ResourceScope.from_grant(grant, run_id)
    except AuthorizationError as exc:
        raise CompletionAuthorizationRefused(str(exc)) from None

    readiness = _readiness(repo_root, run_id, stage_id, session_commit)

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
    payload["approved_money"] = {
        **approved_money,
        "_stated_and_rederived": (
            "the maintainer STATED these in the grant and the issuer recomputed "
            "every one from the completion pricing record. A grant naming a "
            "different expected cost, a different hard cap or a higher rate "
            "basis is refused."),
    }
    payload["budget_headroom"] = budget_headroom(
        cumulative_usd=float(flatten_money(grant, cfg)[contract.spend_field]),
        cap_usd=float(cfg["accepted_pricing"]["cumulative_cap_usd"]),
        ceiling_usd=ceiling)
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload

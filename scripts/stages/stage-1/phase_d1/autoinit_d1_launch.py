#!/usr/bin/env python3
"""The D1 formal SEARCH launcher. One session, declared completely.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d1/autoinit_d1_launch.py \
        --run-id d1_search_001 --gpu "NVIDIA L40S" --arm supervised_target

Everything a previous generation of launcher expressed by overriding a hook is a
FIELD of the `SessionSpec` the shared `SessionRunner` consumes: the authorization
type and path, the setup manifest, the driver command, the budget, the teardown and
artifact policies. The runner owns acquisition, the watchdog, teardown on every exit
path, and collect-before-teardown; this file owns what makes the session D1's.

`SESSION_KIND = "d1"` is DECLARED, and the matching branch in
`autoinit_preflight_setup.sh` loads a `D1Authorization` and asserts its typed
permissions. A missing dispatch entry is not a type error — it falls through to
`spend`, loads the generic artifact, and Phase B's attempt 2 proved what that costs.

**D1 is the first of these sessions that actually runs a beam search.** The
dispatch branch therefore asserts `allows_beam_search is True` where every other
dedicated branch asserts it False, and `allows_recovery`/`allows_behavioural` False:
the search commits a candidate set and stops, because neither behavioural rung can
be bound until that set exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/autoinit"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    ProductFetchResult, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate,
)
from shared.deployment import deployment_commands  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.d1_authorization import D1Authorization  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402

#: DECLARED, and the shell has a branch for it. See the module docstring.
SESSION_KIND = "d1"

#: THE OPERATIONAL RESERVE in this session's provider-account requirement. Small,
#: and deliberately not a second budget: the point is only that a session must
#: not launch against an account that covers it to the last cent, because a pod
#: stopped for an exhausted balance spends money and produces no endpoint. The
#: 2026-10-06 search lost $8.1716 and 39 of 92 expansions to exactly that.
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0

#: THE POD'S BLOCKING TEST GATE runs D1's OWN suite, positively declared. The
#: 2026-10-03 boundary decision made `tests/` reusable-framework behaviour only, so
#: a session that declares nothing gets the framework checked on its machine and no
#: experiment's records. D1 wants its execution contract checked on the billing
#: machine, because that is where a staging difference would show.
TEST_PATHS = ("scripts/stages/stage-1/phase_d1/tests",)

REPO = "/workspace/aad"

#: THE ONE AUTHORITATIVE EVIDENCE LAYOUT, repository-relative.
#:
#: The driver's `--out`, the relay's evidence spec, the report fetch and both
#: artifact specs all name this directory. They did not: the driver wrote
#: `artifacts/stages/stage-1/d1/<run_id>/` while `ArtifactPolicy.audit_dirname` sent the
#: relay and the report fetch to `artifacts/audit/autoinit_d1/`, and three of the
#: five artifact-spec patterns began `artifacts/` again under a collector root
#: that is already `<checkout>/artifacts`. A successful beam would have reached
#: teardown with every required piece of evidence reported missing.
#:
#: `artifacts/audit/<audit_dirname>/` is the side that moved, because it is the
#: convention the shared runner ENCODES (`session_runner` builds both the relay
#: path and the collection root from `audit_dirname`) and the one every other
#: driver here already follows. Changing the driver is a change in one
#: experiment; changing the runner's construction would be a new field on
#: reusable core to accommodate one session's deviation.
AUDIT_DIRNAME = "autoinit_d1"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"

#: EVERY NON-SOURCE INPUT THE DRIVER READS, and none of them is in git.
#:
#: The pod clones a bundle, which carries git objects and nothing else. All
#: three of these are untracked (`git ls-files` returns nothing for any of
#: them), so stage A would have raised on the state-eval asset and stage C on
#: whichever calibration mixture it reached first -- after setup, the test gate
#: and the authorization check had all been paid for. `relay_inputs` and
#: `local_assets` were both `()`.
#:
#: They go by scp rather than through the relay because they are 2.4 MiB in
#: total: at the measured 0.72 MB/s dev-box uplink that is a few seconds, and a
#: relay round trip would need them uploaded, pinned and downloaded again to
#: answer the same question. Their digests are not taken on trust for being
#: small -- `LocalAsset` carries no sha256 field, so each one is verified
#: against a pin it already ships: the state-eval manifest pins its items file
#: (`d1_session.verify_state_eval_bytes`) and each calibration profile pins
#: `items_file_sha256`, which `profile.resolve()` checks when the search loads
#: the mixture.
SCIENCE_ASSETS: tuple[LocalAsset, ...] = (
    LocalAsset(repo_path=D1S.STATE_EVAL_ROOT,
               dest_name="state_eval_v1",
               install_to="artifacts/stages/stage-1"),
    LocalAsset(repo_path="artifacts/stages/stage-1/e8_calibration_v1",
               dest_name="e8_calibration_v1",
               install_to="artifacts/stages/stage-1"),
    LocalAsset(repo_path="artifacts/stages/stage-1/reasoning_heavy_v2",
               dest_name="reasoning_heavy_v2",
               install_to="artifacts/stages/stage-1"),
)


def _stage_id() -> str:
    """D1's declared pipeline stage, from the session module."""
    return D1S.STAGE_ID


def run_dir_for(run_id: str | None) -> str:
    """This run's directory, through the SHARED layout helper.

    Formatted by hand here once -- `f"logs/stages/stage-1/phase_d1/runs/{rid}"`
    -- which is a second owner of a path `shared.run_layout` already
    derives, and the readiness record, the bundle record and the authorization
    all have to agree about it.
    """
    return rel_run_dir(D1S.EXPERIMENT_ID, run_id or "unrecorded", _stage_id())


def auth_path_for(run_id: str | None) -> str:
    """Where this run's one-use authorization lives. Per run, never shared."""
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def readiness_path_for(run_id: str | None) -> str:
    """This run's launch-bound readiness record. ADMISSION, never authorization."""
    return f"{run_dir_for(run_id)}/governance/launch_readiness.json"


def bundle_record_for(run_id: str | None) -> str:
    """The local half of the bundle check: what `stage_d1_bundle.py` wrote."""
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def driver_command(ctx: Any, plan: Any) -> str:
    """The exact command the pod runs. A test parses it with the driver's parser.

    C1 attempt 7 cleared the pod test gate for the first time and then died at
    $0.4231 because its launcher emitted a flag the driver's parser has no option
    for: argparse exited 2 before a line of the driver ran.
    """
    arm = getattr(ctx.args, "arm", D1S.TREATMENT_ARM)
    deadline = int(max(60.0, float(plan.soft_stop_seconds))) \
        if getattr(plan, "soft_stop_seconds", None) else 0
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    #: THE AUDIT LAYOUT the relay, the report fetch and both artifact specs
    #: read. See `EVIDENCE_DIR`.
    out = f"{REPO}/{EVIDENCE_DIR}"
    return (f"/opt/train/bin/python {REPO}/scripts/stages/stage-1/phase_d1/autoinit_d1_driver.py "
            f"--out '{out}' "
            #: EXPLICIT, because `run_id` enters `SearchConfig.config_hash` and
            #: the driver used to take it from the output directory's basename.
            #: With the evidence directory no longer named after the run, that
            #: derivation would have silently changed the identity the
            #: authorization binds.
            f"--run-id '{run_id}' "
            f"--authorization '{REPO}/{auth_path_for(run_id)}' "
            f"--arm '{arm}' "
            #: WHERE THE TERMINAL MARKER GOES, from the same declaration
            #: `SessionSpec.status_path` carries, so the file the driver appends
            #: to is the file the runner polls.
            f"--status '{D1S.STATUS_PATH}' "
            f"--device cuda"
            + (f" --deadline-s {deadline}" if deadline else ""))


#: WHERE `SessionRunner` PUTS THE REPORTS IT FETCHED. Not `<scr>`: the runner
#: scp's each `ArtifactPolicy.report_names` entry to `<scr>/store/<name>` and
#: only then calls `fetch_products`. Reading `<scr>/<name>` found nothing on
#: every real session, and the consequence was silent -- see
#: `committed_selection`.
STORE_SUBDIR = "store"


def store_dir(ctx) -> Path:
    return Path(ctx.args.scr) / STORE_SUBDIR


class SelectionUnreadable(RuntimeError):
    """A selection should exist and cannot be read. NEVER mistaken for absence."""


def committed_selection(ctx) -> list[dict]:
    """The selected leaves, from the AUTHORITATIVE committed record.

    The SELECTION decides which checkpoint directories are products -- not a glob
    over the beam workspace. A search materializes every state it expands; only the
    committed ones are what the next stage trains from, and transferring the rest
    would move tens of GiB nothing consumes.

    Two things changed, and both were silent failures.

    It read `<scr>/d1_search.json`, a path `SessionRunner` never creates: reports
    are fetched to `<scr>/store/<name>`. So on a real session the file was always
    absent, `[]` came back, and `both_selected_leaves_secured` read that as "no
    selection was committed, so there are no product bytes to owe" and allowed
    teardown. A completed search would have had its two selected initializations
    deleted with the pod, with every check green -- the 2026-08-13 failure again,
    one layer further in.

    And it read the driver's own `commit.selected_rows` summary rather than
    `stage1_selection.json`, which is what the committer actually writes and what
    the next stage consumes. The authoritative record is read through
    `stage1_selection.load`, which verifies the record against its own
    `selection_sha256`; the driver's summary is kept only as a fallback for the
    rows' transfer-identity fields, and a disagreement between the two is an
    error rather than a preference.

    Raises `SelectionUnreadable` instead of returning `[]` when a selection
    SHOULD be there. "I could not read the file" and "the search never committed
    anything" are different states and must not share a return value.
    """
    import json as _json

    from aadistill.initialization.planning import stage1_selection

    store = store_dir(ctx)
    selection = store / "stage1_selection.json"
    evidence = store / "d1_search.json"

    summary_rows: list[dict] = []
    status = ""
    reached_commit = False
    if evidence.is_file():
        try:
            doc = _json.loads(evidence.read_text())
        except ValueError as exc:
            raise SelectionUnreadable(
                f"{evidence} is not parseable JSON ({exc}); this session's own "
                "evidence cannot be read, so whether it committed a selection "
                "is unknown") from exc
        status = str(doc.get("status") or "")
        commit = doc.get("commit") or {}
        summary_rows = list(commit.get("selected_rows") or [])
        #: THE DRIVER REACHED STAGE D. Taken from the stage journal rather than
        #: from the presence of rows, so "committed nothing" is distinguishable
        #: from "committed and the rows did not survive serialization".
        reached_commit = bool(commit) or any(
            s.get("stage") == "D" and s.get("status") == "ok"
            for s in (doc.get("stages") or []))

    if selection.is_file():
        try:
            record = stage1_selection.load(selection)
        except Exception as exc:                                  # noqa: BLE001
            raise SelectionUnreadable(
                f"{selection} exists and will not load ({type(exc).__name__}: "
                f"{exc}). A committed selection that fails its own hash is not a "
                "selection, and the checkpoints it names are on the pod.") from exc
        rows = list(record.get("selected") or [])
        if not rows:
            raise SelectionUnreadable(
                f"{selection} loaded and names no selected rows; the committer "
                "does not write an empty selection, so this record is wrong "
                "about a session whose products may exist")
        #: The two records must agree about WHICH leaves were chosen.
        if summary_rows:
            by_id = {r.get("state_id") for r in rows}
            summary_ids = {r.get("state_id") for r in summary_rows}
            if by_id != summary_ids:
                raise SelectionUnreadable(
                    f"{selection} commits {sorted(by_id)} and the driver's "
                    f"evidence summarises {sorted(summary_ids)}; the two records "
                    "of one decision disagree and neither can be used to decide "
                    "which checkpoints are products")
        return rows

    #: NO AUTHORITATIVE RECORD. Whether that is legitimate depends entirely on
    #: whether the driver got as far as committing one.
    if reached_commit or summary_rows:
        raise SelectionUnreadable(
            f"this session reached commit_top_k (status={status or 'unknown'}, "
            f"{len(summary_rows)} row(s) in the driver's evidence) but "
            f"{selection} was not fetched off the pod. The committed "
            "initializations exist and this launcher cannot name them; do not "
            "treat an unreadable selection as an uncommitted one.")
    if not evidence.is_file():
        raise SelectionUnreadable(
            f"neither {selection} nor {evidence} was fetched to {store}. This "
            "session's evidence did not come home, so whether it committed a "
            "selection is unknown -- and an unknown must not be reported as "
            "'nothing was owed'.")
    return []


def fetch_selected_checkpoints(ctx) -> list:
    """Fetch the committed Top-2 off-pod, then RE-IDENTIFY them here.

    These are this session's PRODUCTS. The next D1 stage trains recovery probes
    FROM these initializations, and the $14.7966 screening price does not fund
    re-running two complete selected structural paths after the search -- so
    rebuilding them later under that price is not an option either.

    Fetched whenever they EXIST rather than only on a fully successful session:
    `if terminal == "ALL_DONE"` deleted $2.82 of verified checkpoints on
    2026-08-13 for want of exactly that distinction.
    """
    import subprocess

    fetched: list = []
    if not ctx.products_eligible:
        return fetched
    try:
        rows = committed_selection(ctx)
    except SelectionUnreadable as exc:
        #: Recorded as a FAILED fetch, not swallowed. `products_secured` asks the
        #: same question again and refuses, so the session's terminal state
        #: carries PRODUCTS_UNSECURED rather than a clean teardown.
        ctx.say(f"  CANNOT READ THE COMMITTED SELECTION: {exc}")
        ctx.evidence["d1_selection_unreadable"] = str(exc)
        return [ProductFetchResult(
            kind="transfer", rc=1,
            detail=f"the committed selection could not be read: {exc}")]
    if not rows:
        ctx.say("  no committed selection yet; nothing to secure")
        return fetched
    store = Path(getattr(ctx.args, "ckpt_store", None)
                 or Path(ctx.args.scr) / "products")
    store.mkdir(parents=True, exist_ok=True)
    for row in rows:
        state_id = row.get("state_id") or "unknown"
        remote = row.get("checkpoint_path")
        if not remote:
            fetched.append(ProductFetchResult(
                kind="transfer", rc=1,
                detail=f"{state_id} names no checkpoint_path"))
            continue
        dest = store / state_id
        rc = subprocess.run(
            ["timeout", f"{getattr(ctx.args, 'ckpt_fetch_limit_min', 45)}m",
             "scp", "-r", "-P", str(ctx.target.port),
             "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{remote}", str(dest)],
            capture_output=True, timeout=None).returncode
        size = (sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if dest.exists() else 0)
        verified, why = _reidentify(dest, row)
        fetched.append(ProductFetchResult(
            kind="transfer", rc=(0 if rc == 0 and verified else 1),
            detail=(f"{state_id}: rc={rc}, {size / 2**30:.2f} GiB -> {dest}; "
                  f"identity {'MATCHED' if verified else 'NOT MATCHED'} ({why})")))
        ctx.say(f"  product {state_id}: rc={rc}, {size / 2**30:.2f} GiB, "
                f"identity {'MATCHED' if verified else 'NOT MATCHED'}")
    return fetched


def _reidentify(directory: Path, row: dict) -> tuple[bool, str]:
    """Rebuild the leaf's identity from the bytes that ARRIVED.

    Staging a checkpoint on the pod is not durability; it is a copy that dies with
    the pod. This is the other half, run on the destination: the artifact digest,
    the weights digest and the shard hashes are recomputed locally, so a transfer
    that truncated a shard is caught here rather than assumed away.
    """
    try:
        from aadistill.initialization.specs.arch import get_adapter
        from aadistill.runtime.leaf_durability import (
            LeafDurabilityError, verify_transferred_leaf,
        )
    except ImportError as exc:                                  # noqa: BLE001
        return False, f"cannot import the verifier: {exc}"
    if not directory.is_dir():
        return False, "nothing arrived"
    try:
        out = verify_transferred_leaf(directory, row,
                                     adapter=get_adapter("qwen3"))
    except LeafDurabilityError as exc:
        return False, str(exc)[:160]
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:160]
    return bool(out.get("matched", True)), str(out.get("why", "re-identified"))


def both_selected_leaves_secured(ctx, fetched) -> tuple[bool, str]:
    """BOTH, or this session owes products it did not secure.

    Separate from `fetch_products` returning cleanly, because `all([])` is True: a
    fetch that returned NOTHING would pass every transfer check while having
    secured nothing at all, and the pod would then be deleted with the only copies
    of the selected initializations on it.
    """
    try:
        rows = committed_selection(ctx)
    except SelectionUnreadable as exc:
        #: THE GATE FAILS. A path miss, an unparseable record or a selection
        #: that fails its own hash is NOT evidence that the search produced
        #: nothing: it is evidence that this launcher cannot tell, and the one
        #: outcome that must not follow is "no products were owed".
        return False, (
            f"PRODUCT GATE FAILURE: {exc} This is not 'the search did not "
            "complete' -- it is an unknown, and the selected initializations "
            "may be on the pod with nothing off it.")
    expected = len(rows)
    if not expected:
        #: NO COMMITTED SELECTION, established rather than inferred from a
        #: missing file: `committed_selection` raises unless the driver's own
        #: evidence came home AND shows it never reached commit_top_k. There are
        #: no product bytes to owe, so teardown is not blocked -- the evidence
        #: is what matters then.
        return True, ("the search did not reach commit_top_k, so there are no "
                      "product bytes to secure; its evidence is what this "
                      "session produced")

    #: THE DESIGN'S k, asked here as well as in the driver's stage D.
    #:
    #: Not redundant: stage D raises on a short ranking, but this gate's job is
    #: to decide whether a pod may be deleted, and it must not accept a
    #: selection of some other size as complete just because the record it read
    #: was self-consistent. A record committing one leaf would otherwise pass as
    #: "1 of 1 secured".
    try:
        want_k = int(D1S.design()["behavioural_design"]["top_k"])
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"PRODUCT GATE FAILURE: the design's top_k could not be "
                       f"read ({type(exc).__name__}: {exc}), so the size of a "
                       "complete selection is unknown")
    if expected != want_k:
        return False, (
            f"PRODUCT GATE FAILURE: the committed selection names {expected} "
            f"leaf/leaves and the design commits {want_k}. A selection of the "
            "wrong size is not a complete search, and the behavioural rungs "
            "would screen a set the search did not choose.")

    ok = [f for f in fetched if getattr(f, "ok", False)]
    if len(ok) != expected:
        return False, (
            f"{len(ok)} of {expected} selected checkpoints were secured and "
            "identity-verified off-pod. The next D1 stage trains recovery probes "
            "FROM these initializations and the screening price does not fund "
            "rebuilding them, so a pod may not be deleted while they are only on "
            "it.")
    return True, (f"both selected checkpoints secured off-pod and "
                  f"identity-verified ({expected} of {expected})")


# ---------------------------------------------------------------------------
# the gates that run at $0, before a provider is contacted
# ---------------------------------------------------------------------------
#
# `SessionSpec.precheck` was `()`. Every identity in the chain was DERIVED
# correctly and NOTHING CHECKED IT against the live invocation: the runner calls
# `require_plan`, `require_harness`, `require_within_cap` and
# `require_within_launch_limit` and has no call to `require_session_commit`, so a
# launch at any commit whose closure happened to digest the same -- or, far more
# likely, a launch at HEAD rather than at the bound commit -- would have been
# priced, created and paid for before anything noticed.


def run_identity_gate(ctx) -> tuple[bool, str]:
    """The arm, the run id and the design revision this grant was issued for.

    `require_run_id` is the expensive one made cheap: `run_id` is a term of
    `SearchConfig.config_hash`, so a launch under another run id carries a
    config hash the authorization does not bind and stage A refuses it ON A
    BILLING POD, after setup, the test gate and the registries.
    """
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_treatment_arm()
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    arm = getattr(ctx.args, "arm", D1S.TREATMENT_ARM)
    if arm != ctx.auth.arm:
        return False, (f"the authorization covers the {ctx.auth.arm!r} arm and "
                       f"this launch declares {arm!r}")
    design = D1S.design_hash(REPO_ROOT)
    if ctx.auth.design_hash != design:
        return False, (f"the authorization binds design {ctx.auth.design_hash[:12]} "
                       f"and the tree carries {design[:12]}")
    blockers = list(D1S.open_blockers(REPO_ROOT))
    if blockers:
        return False, (f"the design reports open blockers {blockers}; a formal "
                       "session may not run over one")
    ctx.evidence["d1_run_identity"] = {
        "run_id": ctx.auth.run_id, "arm": ctx.auth.arm,
        "design_hash": design, "config_hash": ctx.auth.config_hash,
        "measurement_protocol_id": ctx.auth.measurement_protocol_id,
        "open_blockers": blockers,
    }
    return True, (f"run {ctx.auth.run_id} on the {ctx.auth.arm} arm, design "
                  f"{design[:12]}, config {ctx.auth.config_hash[:12]}, no open "
                  "blockers")


def session_contract_gate(ctx) -> tuple[bool, str]:
    """Build the REAL session here and check it binds the authorized identities.

    Stage A does this on the pod, which is correct and is also the most
    expensive place to discover a mismatch. Everything it needs exists on the
    dev box -- the design, the frozen state-eval asset, both calibration
    profiles, the operator registries -- so the same equality is asked for
    nothing first.

    Built with `A.FORMAL_DEVICE`, read from the authorization module rather than
    typed here, because `device` is a field of `SearchConfig.as_dict()` and
    therefore of `config_hash`. A gate that built on `cpu` while the driver
    builds on `cuda` would compare two different hashes and report a mismatch
    for every correct launch -- or, worse, be "fixed" by relaxing the
    comparison. Constructing a cuda-declared config allocates nothing.
    """
    import tempfile

    from stages.phase_d1 import d1_authorization as A

    try:
        D1S._register_frozen_operators()
        with tempfile.TemporaryDirectory(prefix="d1-gate-") as tmp:
            session = D1S.build_session(
                arm=ctx.auth.arm, workdir=Path(tmp),
                run_id=ctx.auth.run_id, device=A.FORMAL_DEVICE,
                repo_root=REPO_ROOT)
            contract = D1S.assert_session_contract(session, REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"the D1 session does not build or does not satisfy its "
                       f"contract on this tree: {type(exc).__name__}: {exc}")

    problems = []
    if contract["config_hash"] != ctx.auth.config_hash:
        problems.append(
            f"config_hash {contract['config_hash'][:12]} != authorized "
            f"{ctx.auth.config_hash[:12]}")
    if contract["measurement_protocol_id"] != ctx.auth.measurement_protocol_id:
        problems.append(
            f"measurement_protocol_id {contract['measurement_protocol_id'][:12]} "
            f"!= authorized {ctx.auth.measurement_protocol_id[:12]}")
    if contract["suite_content_sha256"] != ctx.auth.suite_content_sha256:
        problems.append(
            f"suite_content_sha256 {contract['suite_content_sha256'][:12]} != "
            f"authorized {ctx.auth.suite_content_sha256[:12]}")
    ctx.evidence["d1_session_contract_check"] = {
        **{k: contract[k] for k in
           ("config_hash", "measurement_protocol_id", "suite_content_sha256",
            "position_policy_hash", "design_hash")},
        "problems": problems,
        "built_with_device": A.FORMAL_DEVICE,
        "rule": ("the identities stage A will assert on the pod, asserted here "
                 "for nothing first"),
    }
    if problems:
        return False, ("the session this tree builds is not the one the "
                       "authorization priced: " + "; ".join(problems))
    return True, (f"session contract verified at $0: protocol "
                  f"{contract['measurement_protocol_id'][:12]}, config "
                  f"{contract['config_hash'][:12]}, suite "
                  f"{contract['suite_content_sha256'][:12]}")


def staged_science_inputs_gate(ctx) -> tuple[bool, str]:
    """Every declared local asset exists, and each one matches its own pin.

    `SessionRunner.run_prechecks` already checks that each `LocalAsset.repo_path`
    EXISTS. That is not enough for these three: `LocalAsset` carries no digest
    field, so an existence check would pass a truncated items file and the search
    would run on a different mixture under the frozen hash. Each asset is
    therefore verified against the pin it ships -- the state-eval manifest's
    `outputs.items.sha256`, and each profile's `items_file_sha256`.
    """
    checked: dict[str, Any] = {}
    try:
        checked["state_eval_v1"] = D1S.verify_state_eval_bytes(REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the frozen state-eval asset is not usable: {exc}"

    try:
        from aadistill.initialization.calibration.profiles import get_profile
        from shared.calibration import register_builtin_profiles

        register_builtin_profiles()
        declared = list(D1S.design(REPO_ROOT)["search_stage"]["profiles"])
        for pid in sorted(declared):
            profile = get_profile(pid)
            #: `resolve()` is the REAL consumer: it checks `items_file_sha256`
            #: and recomputes the mixture content hash. The search calls it on
            #: the pod; calling it here asks the same question for nothing.
            items = profile.resolve(REPO_ROOT)
            checked[pid] = {"items_path": profile.items_path,
                            "n_items": len(items),
                            "items_file_sha256": profile.items_file_sha256}
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"a declared calibration profile does not resolve on this "
                       f"tree: {type(exc).__name__}: {exc}")

    missing = [a.repo_path for a in SCIENCE_ASSETS
               if not (REPO_ROOT / a.repo_path).exists()]
    if missing:
        return False, f"declared local assets absent from this tree: {missing}"
    ctx.evidence["d1_staged_science_inputs"] = {
        "local_assets": [a.as_env_entry() for a in SCIENCE_ASSETS],
        "verified": checked,
        "teacher_revision": D1S.root_teacher_identity(REPO_ROOT)["revision"],
    }
    names = ", ".join(sorted(checked))
    return True, f"science inputs verified against their own pins: {names}"


def launch_readiness_gate(ctx) -> tuple[bool, str]:
    """The launch-bound readiness record must describe THIS invocation.

    AUTHORIZES NOTHING, deliberately, and the record says so itself. The one-use
    artifact is what authorizes; this is an ADMISSION check, and its value is
    that it was written immediately before the launch and therefore pins what
    the launch was supposed to be. It was consumed by nothing at all -- a
    document that no code reads cannot refuse a stale launch, and this one had
    already gone stale twice (it named a bundle that does not exist on the relay
    and a commit the repairs superseded).
    """
    rel = readiness_path_for(getattr(ctx.args, "run_id", None))
    path = REPO_ROOT / rel
    if not path.is_file():
        return False, (f"no launch-readiness record at {rel}; run "
                       "scripts/stages/stage-1/phase_d1/write_d1_launch_readiness.py "
                       "immediately before launching")
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        return False, f"{rel} is not parseable JSON: {exc}"

    auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
    closure = _closure_digest()
    want = {
        "run_id": getattr(ctx.args, "run_id", None),
        "session_commit": getattr(ctx.args, "session_commit", None),
        "arm": getattr(ctx.args, "arm", D1S.TREATMENT_ARM),
        "design_hash": D1S.design_hash(REPO_ROOT),
        "harness_digest": closure,
        "authorization_sha256": _authorization_self_hash(auth_rel),
        "bundle_name": D1S.canonical_bundle_name(
            getattr(ctx.args, "session_commit", "") or ""),
    }
    got = {
        "run_id": doc.get("run_id"),
        "session_commit": doc.get("session_commit"),
        "arm": doc.get("arm"),
        "design_hash": doc.get("design_hash"),
        "harness_digest": (doc.get("harness") or {}).get("digest"),
        "authorization_sha256": (doc.get("authorization") or {}).get(
            "authorization_sha256"),
        "bundle_name": (doc.get("bundle") or {}).get("name"),
    }
    disagree = sorted(k for k in want if want[k] != got[k])

    #: THE DERIVED CEILING and the open blockers, read from the record and
    #: compared against the live position rather than restated.
    ceiling_in_record = (doc.get("derived_session") or {}).get("hard_ceiling_usd")
    ceiling_disagrees = (
        ceiling_in_record is None
        or abs(float(ceiling_in_record) - float(ctx.auth.hard_cap_usd)) > 5e-4)
    blockers = list(doc.get("open_blockers") or [])

    ctx.evidence["d1_launch_readiness_check"] = {
        "path": rel, "expected": want, "recorded": got,
        "disagree": disagree,
        "recorded_ceiling_usd": ceiling_in_record,
        "authorized_ceiling_usd": float(ctx.auth.hard_cap_usd),
        "open_blockers": blockers,
        "authorizes": "nothing; this is an admission/consistency gate",
    }
    if disagree:
        return False, (
            f"{rel} does not describe this launch: {disagree} disagree "
            f"(recorded { {k: got[k] for k in disagree} }, live "
            f"{ {k: want[k] for k in disagree} }). A stale readiness record is "
            "refused at $0 rather than read as reassurance.")
    if ceiling_disagrees:
        return False, (f"{rel} records a derived ceiling of "
                       f"{ceiling_in_record} and the authorization carries "
                       f"${float(ctx.auth.hard_cap_usd):.4f}")
    if blockers:
        return False, f"{rel} reports open blockers {blockers}"
    return True, (f"launch-readiness {rel} matches this invocation on "
                  f"{len(want)} bound facts and reports no open blockers")


def _closure_digest() -> str:
    from stages.phase_d1 import d1_authorization as A

    return A.d1_current_executable(REPO_ROOT)["digest"]


def _authorization_self_hash(auth_rel: str) -> str | None:
    path = REPO_ROOT / auth_rel
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("authorization_sha256")
    except ValueError:
        return None


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    D1 had no such gate, and the object it would have fetched DOES NOT EXIST:
    `transfer/aad_autoinit_6c1dd8d9.bundle` is absent from the relay, which
    holds 563 `transfer/` objects and 100+ other `aad_autoinit_*.bundle`s. The
    prepared chain would have created a pod, paid for startup, and died at
    `SETUP_RC=1` on a 404 -- which is C1 attempt 1 exactly, the failure the
    shared `bundle_transport` module exists to make impossible.

    Read-only. Preparation is `scripts/stages/stage-1/phase_d1/stage_d1_bundle.py`, a separate
    command, so what this verifies is the relay's state and not a side effect of
    verifying it.
    """
    import tempfile

    from aadistill.infrastructure.bundle_transport import (
        BundleTransportError, hf_download, roundtrip,
    )

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
    except BundleTransportError as exc:
        return False, str(exc)

    rel = bundle_record_for(getattr(ctx.args, "run_id", None))
    staged = REPO_ROOT / rel
    if not staged.is_file():
        return False, (
            f"{rel} is missing; run scripts/stages/stage-1/phase_d1/stage_d1_bundle.py "
            f"--session-commit {commit} --run-id "
            f"{getattr(ctx.args, 'run_id', '')} first. Without it nothing has "
            "built or uploaded the object the pod fetches.")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{rel} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}, not the "
                       f"session commit {commit[:12]}")

    auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
    auth_bytes = (REPO_ROOT / auth_rel).read_bytes()
    try:
        with tempfile.TemporaryDirectory(prefix="d1-bundle-") as tmp:
            evidence = roundtrip(
                D1S.transport(),
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_bytes,
                authorization_path=auth_rel,
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                download=hf_download, workdir=Path(tmp))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} ({evidence['bytes']} "
                  f"bytes, {evidence['remote_sha256'][:12]}) round-trips to "
                  f"{evidence['roundtrip_head'][:12]} carrying this "
                  f"authorization and harness "
                  f"{evidence['roundtrip_harness_digest'][:12]}")


def prechecks(args) -> tuple:
    """Every `$0` gate, in the order a refusal is cheapest.

    Ordered so the questions that need no network come first: the run's own
    identity, then the session the tree builds, then the staged inputs, then the
    readiness record, then git lineage, and the relay round trip last because it
    is the only one that downloads anything.
    """
    return (
        run_identity_gate,
        session_contract_gate,
        staged_science_inputs_gate,
        launch_readiness_gate,
        #: THE HARNESS AT `--session-commit` must be the authorized one, AND
        #: nothing other than the authorization artifact may differ from the
        #: authorized base. `check_lineage=True` because the readiness record
        #: and the bundle record are deliberately left UNCOMMITTED inside a
        #: launch window; committing them would add paths to this diff.
        session_commit_gate(REPO_ROOT,
                            auth_path_for(getattr(args, "run_id", None)),
                            check_lineage=True),
        bundle_staged_gate,
    )


def budget_spec(repo_root: Path) -> BudgetSpec:
    """The `BudgetSpec` the runner plans from, DECOMPOSING the accepted bound.

    `BudgetSpec` is a phase-TIME model and the runner's plan is
    `expected x (1 + contingency) + reserves + recovery_reserve`. The accepted
    search bound is `hard_ceiling_minutes` from `search_cost`, which already
    carries its own overrun factor -- so the reserve is DERIVED as whatever makes
    the plan land exactly on that bound, rather than added on top of it.

    The first version added a 10% contingency and a 30-minute reserve ON TOP of
    the accepted ceiling and produced a plan terminating at $22.19 against a
    $21.4897 authorization. The runner refused it, correctly, and the refusal was
    right about the cause: two models of one session. There is one total, and this
    decomposes it.
    """
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from stages.phase_d1 import d1_authorization as A

    priced = A.session_ceiling(repo_root)
    session = _priced_session(repo_root)
    overhead = float(session["session_overhead_minutes"])
    accepted_hard = float(priced["hard_ceiling_minutes"])
    expected_total = float(session["expected_minutes"])
    expected_search = max(0.0, expected_total - overhead)
    contingency = 0.10
    recovery_reserve = 30.0

    #: WHATEVER MAKES THE PLAN LAND ON THE ACCEPTED BOUND. Named as what it is:
    #: the gap between the expected trajectory and the structural worst case, an
    #: identified bounded risk that is not on the expected path -- which is why it
    #: belongs AFTER the contingency multiplier, where it protects the work rather
    #: than only moving the watchdog's kill time.
    reserve = round(accepted_hard - expected_total * (1.0 + contingency)
                    - recovery_reserve, 2)
    if reserve < 0:
        raise SystemExit(
            f"the accepted {accepted_hard:.2f}-minute bound cannot hold the "
            f"expected {expected_total:.2f} min plus a {contingency:.0%} "
            f"contingency and a {recovery_reserve:.0f}-minute recovery reserve. "
            "Obtain a larger authorization or choose a smaller run; do not shrink "
            "the reserve to fit.")

    return BudgetSpec(
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: a D1 search session trains nothing, so arms=0 and "
                     "the step term is zero. The measured floor is passed so the "
                     "below-floor guard cannot be satisfied by accident"),
        #: The session overhead the cost model already prices, split the way the
        #: runner expects rather than invented.
        setup_minutes=round(overhead * 0.75, 2),
        transfer_minutes=round(overhead * 0.25, 2),
        other_phases=(
            Phase("beam_search_expected_trajectory", round(expected_search, 2)),
        ),
        contingency_fraction=contingency,
        soft_stop_reserves=(Phase("beam_composition_risk", reserve),),
        artifact_recovery_reserve_minutes=recovery_reserve,
        below_floor_reason=(
            f"priced at ${priced['hard_ceiling_usd']:.4f} / "
            f"{accepted_hard:.2f} min by search_space.search_cost; this spec "
            "DECOMPOSES that bound and does not add to it. See d1_design.json :: "
            "budget.chain.sessions.search"),
        #: WHAT THE RUNPOD ACCOUNT MUST HOLD, derived here because the amount is
        #: this campaign's and not reusable core's.
        #:
        #: On 2026-10-06 the formal search passed all six $0 gates, was
        #: authorized to $21.4897 over 1125.55 minutes, and RunPod stopped it at
        #: 449.8 minutes with 39 of 92 expansions complete because the ACCOUNT
        #: balance had run out. $8.1716 bought no endpoint. Every gate asked
        #: whether the experiment was permitted to spend; none asked whether the
        #: provider would still be paid.
        #:
        #: Two terms, and NOT three. The first version added
        #: `container_disk_usd` to `hard_ceiling_usd` and double-counted it:
        #:
        #:     gpu_usd 20.4475 + container_disk_usd 1.0422 = 21.4897 = ceiling
        #:
        #: The ceiling ALREADY CONTAINS the disk, exactly. A session's ceiling
        #: owns which priced components it contains, and a caller that re-adds
        #: one is asserting a cost model it does not own -- the same defect as
        #: restating any derived figure. So:
        #:
        #:   * the authorized session hard ceiling -- every priced component of
        #:     what this run may cost, the disk included;
        #:   * a small operational reserve, so a session does not launch against
        #:     a balance that covers it to the last cent;
        #:   * plus any OTHER active obligation not already inside that ceiling.
        #:     There are none for a D1 search: it holds no network volume and no
        #:     concurrent resource, and the package permits one billing resource
        #:     at a time. A term is not added for a hypothetical.
        #:
        #: Floored at the package's per-attempt envelope, READ from the
        #: authorization config rather than typed: an account holding less than
        #: one full envelope cannot fund an attempt this package permits. That
        #: floor is what makes the requirement $30 today, and it is why
        #: correcting the double count does not move the D1 figure --
        #: max(30.0000, 26.4897) is still 30.0000.
        account_balance_required_usd=round(max(
            float(A.live_money(repo_root)["per_session_envelope_usd"]),
            float(priced["hard_ceiling_usd"])
            + ACCOUNT_OPERATIONAL_RESERVE_USD), 4),
    )


def _priced_session(repo_root: Path) -> dict[str, Any]:
    """The priced search session, measured basis when one exists.

    One owner: the same `topk_search_cost()` the design's chain and
    `search_stage.cost` both read. A launcher that priced it again could disagree
    with the artifact it is about to run under.
    """
    from stages.phase_d1 import write_d1_design as w

    priced = w.topk_search_cost()
    if priced is not None:
        return priced["search_session"]
    from stages.phase_d1 import search_space as d1

    w._ensure_the_frozen_operators_are_registered()
    return d1.search_cost()


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d1-search",
        schema="aadistill.autoinit.d1_search_session/v1",
        description=(
            "D1: ONE formal beam search over the frozen four-operator space at "
            "width 6 with one warmup level, under reference_topk_tail_v1 at "
            "K=200, bsz=3 and length_sorted_v1, on the arm's declared position "
            "policy. Commits a candidate set and stops: it trains nothing, "
            "evaluates no behaviour and computes no promotion decision."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D1Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            #: THE DEPLOYMENT'S OWN IMAGE FACTS, read from the shared helper
            #: rather than retyped: the remote interpreter, the workspace and
            #: checkout roots and the minimum CUDA version are properties of the
            #: image, and a launcher that states its own can disagree with the
            #: image it then asks for.
            **deployment_commands()),
        plan_id=f"autoinit.v1.{D1S.EXPERIMENT_ID}",
        #: THE DESIGN HASH is the plan hash: the dispatch branch calls
        #: `require_plan` with it, so an authorization issued against another
        #: design revision is refused at exit 98 before any work.
        plan_hash=D1S.design_hash(REPO_ROOT),
        budget=budget_spec(REPO_ROOT),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_KIND",
                          #: The teacher step reads it; see `teacher_revision`.
                          "TEACHER_REVISION"),
            #: EVERY STEP THIS SESSION NEEDS, and deliberately not one more.
            #:
            #: `TEACHER_READY` is the repair: stage B materializes a 4B-class
            #: teacher and the manifest declared no teacher step, so nothing
            #: downloaded it and nothing set a path to it. The shared step
            #: fetches the pinned revision into the pod's HF cache and the
            #: driver resolves that cache with `local_files_only=True`, so the
            #: transfer happens in setup and never on the search's clock.
            #:
            #: Three steps stay UNDECLARED, each for a reason:
            #:
            #: * `VLLM_READY` -- D1 generates nothing. It is the most expensive
            #:   step in the script (an offline wheelhouse install) and this
            #:   session has no rollout engine.
            #: * `ROPE_OK` -- it globs
            #:   `artifacts/stages/stage-1/*/checkpoint/config.json` and `sys.exit`s
            #:   with "no staged checkpoint to check" when it finds none. D1
            #:   stages no checkpoint, so DECLARING this step would fail setup
            #:   rather than check anything.
            #: * `ASSETS_READY` -- the frozen-asset sweep. Every D1 input is
            #:   already digest-verified against the pin it ships, twice: at $0
            #:   by `staged_science_inputs_gate` and on the pod at stage A by
            #:   `verify_state_eval_bytes`, `profile.resolve()` and
            #:   `verify_staged_teacher`. A fourth pass over the same question
            #:   is the duplicate validation layer AGENTS.md P8.2.1 forbids.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "TEACHER_READY", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            #: THE THREE SCIENCE INPUTS THE BUNDLE DOES NOT CARRY. See
            #: `SCIENCE_ASSETS`: all three are untracked in git, so a pod that
            #: clones the bundle has none of them.
            local_assets=SCIENCE_ASSETS,
            #: THE PINNED TEACHER REVISION, from the frozen binding rather than
            #: typed: it is the same identity `verify_staged_teacher` checks the
            #: downloaded shards against, and two copies of it are two things
            #: that can disagree about which teacher the lineage started from.
            teacher_revision=D1S.root_teacher_identity(REPO_ROOT)["revision"],
            test_paths=TEST_PATHS,
            uv_max_seconds=getattr(args, "uv_max_s", 1500),
            tests_max_seconds=getattr(args, "tests_max_s", 2700)),
        driver_command=driver_command,
        driver_job_id=D1S.DRIVER_JOB_ID,
        status_path=D1S.STATUS_PATH,
        run_log_path=D1S.RUN_LOG_PATH,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("RUN_FAILED",),
            #: WHEN THIS SESSION'S PRODUCTS EXIST. The driver writes
            #: `d1_search.json` on EVERY path out, including a stage-A refusal, so
            #: eligibility is broader than success -- the distinction that cost
            #: $2.82 of verified checkpoints on 2026-08-13, when a collector ran
            #: only on a clean terminal state.
            products_eligible=lambda terminal, stages: True,
            failure_note=("the search stopped -- collecting its evidence, then "
                          "tearing down")),
        #: EVERY `$0` GATE. See `prechecks`; this was `()`.
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            #: THE ONE AUTHORITATIVE LAYOUT. The runner builds the relay's
            #: evidence path and the collection root from this name, the driver's
            #: `--out` is the same directory, and both artifact specs glob it.
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="d1_search.json",
            archive_basename="d1_search_artifacts.tar.gz",
            #: JSON AND LOGS ONLY. Attempt a1 of another session pulled
            #: `/workspace/out` wholesale: 9.1 GiB of checkpoints over a
            #: 0.72 MB/s uplink with the pod still billing. Nothing on the dev
            #: box consumes search checkpoints; the decision reads scores,
            #: digests and identities, all of which are JSON.
            spec_success="configs/stages/stage-1/phase_d1/d1_search_artifacts.json",
            #: NOTHING IS `required` ON THE FAILURE PATH. An early refusal may
            #: legitimately have produced none of it, and demanding an artifact
            #: there would block the collection of what the run DOES have.
            spec_failed="configs/stages/stage-1/phase_d1/d1_search_artifacts_failed.json",
            report_names=("d1_search.json", "journal.jsonl",
                          "stage1_selection.json"),
            #: THE COMMITTED TOP-2 ARE PRODUCTS. Without these two the runner
            #: would report that D1 owes nothing off-pod and delete the pod with
            #: the only copies of the selected initializations on it.
            fetch_products=fetch_selected_checkpoints,
            products_secured=both_selected_leaves_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """The real parser, at module scope so a test can build it.

    `tests/integration/test_session_kind_dispatch.py` enumerates every
    `*_launch.py` that exports `spec` and `build_parser`, fills the required
    arguments and builds the spec, to check each declared `SESSION_KIND` has a
    shell branch that loads ITS OWN authorization type. A launcher whose parser
    lives inside `main` is invisible to that probe — and an unseen launcher is one
    whose missing dispatch entry falls through to `spend`.

    Every required argument is a STRING: the probe fills them with one, and a
    required `type=float` would make it crash rather than report.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    #: THE SHARED SET `SessionRunner` ITSELF READS. Enumerated from the runner's
    #: reads rather than copied from another launcher: a copy keeps the flags that
    #: launcher needs and misses the ones this runner does.
    ap.add_argument("--scr", required=True,
                    help="the session's local scratch directory; "
                         "`SessionRunner` writes its working files there")
    ap.add_argument("--session-commit", required=True,
                    help="the commit the pod checks out. The authorization binds "
                         "it and the lineage gate verifies it")
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from "
                         "--session-commit; an alias fails at $0")
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--arm", default=D1S.TREATMENT_ARM, choices=list(D1S.ARMS))
    #: NOT `required`, and the reason is mechanical. The dispatch probe fills every
    #: required option with the STRING "kind_probe", so a required `type=float`
    #: raises during conversion, puts this launcher in UNPARSEABLE and removes it
    #: from every check in that module -- fewer tests, all green. A3 was dropped
    #: from all of them exactly this way. The default is the rate the ceiling was
    #: derived at, which is also the better governance: a launch must not be priced
    #: at a rate nothing authorized.
    ap.add_argument("--max-price", type=float, default=None,
                    help="omit to use the rate the derived ceiling was priced at")
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--ckpt-store", default=None,
                    help="where the committed Top-2 checkpoints are secured; "
                         "defaults to <scr>/products")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=45,
                    help="per-checkpoint transfer timeout")
    #: EVERY ARGUMENT THE RUNNER READS. It validates the namespace at
    #: construction and names what is absent, which is how this set was
    #: enumerated: `missing_arguments` reported `out, image, token_src,
    #: startup_limit_min, create_attempts, create_retry_seconds, host_draws,
    #: setup_timeout_s, poll_seconds, poll_limit_min, settle_seconds`. Without
    #: them the launcher aborts at construction -- cheap, but only because the $0
    #: interface test constructs the real runner; a launch would otherwise have
    #: discovered it after setup.
    ap.add_argument("--out", default=None)
    #: THE FORMAL IMAGE. `POD_IMAGE` holds the interpreter and the checkout
    #: roots, not the image name, so this is declared where every sibling
    #: launcher declares it.
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    #: ONE create call per acquisition invocation, by the package's own rule. A
    #: corrected attempt is an explicit new subrun, never an invisible retry.
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    #: Draws replace an unusable HOST without consuming an attempt.
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: MUST OUTLAST THE HARD THRESHOLD, or the launcher stops watching a pod that
    #: is still billing. The search's own bound is 1125.55 min.
    ap.add_argument("--poll-limit-min", type=float, default=1400.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate through the real SessionRunner and "
                         "stop before provider creation; create nothing")
    return ap


def resolve_operational_defaults(args) -> dict[str, Any]:
    """Fill the three arguments that default to `None` and that the runner READS.

    `--max-price`, `--disk-gb` and `--out` each defaulted to `None` and nothing
    resolved them, so a real launch aborted in `make_plan` with

        ABORT: TypeError: '<=' not supported between instances of
               'NoneType' and 'int'

    before any gate ran -- and `self.save()` would have crashed on
    `repo_root / None` on the way out. The `--max-price` help text already
    promised "omit to use the rate the derived ceiling was priced at"; nothing
    implemented that sentence.

    All three come from the priced session rather than from constants, so a
    launch is priced at the rate its authorization was derived at, carries the
    disk the cost model charged for, and writes its session record into its own
    run directory. Explicit values are respected: an operator who passes
    `--max-price` has said something deliberate and the four authorization
    limits still bound the result.
    """
    from stages.phase_d1 import d1_authorization as A

    priced = A.session_ceiling(REPO_ROOT)
    resolved: dict[str, Any] = {}
    if getattr(args, "max_price", None) is None:
        #: THE RATE THE CEILING WAS DERIVED AT. A launch priced above it would
        #: terminate at a dollar figure nothing authorized; `check_gpu_offered`
        #: refuses a live quote above this, at $0.
        args.max_price = float(priced["price_per_hour"])
        resolved["max_price"] = args.max_price
    if getattr(args, "disk_gb", None) is None:
        #: THE DISK THE COST MODEL CHARGED FOR. RunPod bills container disk
        #: separately and `reprice_at` carries a `container_disk_usd` term, so
        #: asking for a different size would spend against a price nobody
        #: derived.
        args.disk_gb = int(priced["container_disk_gb"])
        resolved["disk_gb"] = args.disk_gb
    if getattr(args, "out", None) is None:
        args.out = (f"{run_dir_for(getattr(args, 'run_id', None))}"
                    "/runtime/session.json")
        resolved["out"] = args.out
    if not args.disk_gb or args.disk_gb < 1:
        raise SystemExit(
            f"the priced session carries container_disk_gb={args.disk_gb!r}; a "
            "pod cannot be created without a disk size, and guessing one would "
            "spend against a price nothing derived.")
    return resolved


def main(argv: list[str] | None = None) -> int:
    from aadistill.infrastructure.session_runner import SessionRunner, run_session

    from aadistill.governance.authorization import AuthorizationError

    args = build_parser().parse_args(argv)
    #: BEFORE THE SPEC AND BEFORE ANY EARLY RETURN. These three arguments are
    #: read by the runner and by `budget_spec`, and resolving them after a
    #: branch had already returned is how the launcher came to abort in
    #: `make_plan` on a `None` price.
    resolved = resolve_operational_defaults(args)
    try:
        session = spec(args)
    except AuthorizationError as exc:
        #: A REFUSAL, not a crash. `spec()` reads the design and builds the
        #: gates; the loader itself runs in `SessionRunner.__init__`, which is
        #: outside `run_session`'s try block, so an artifact this launcher
        #: cannot load would otherwise reach the operator as a traceback. It is
        #: still a `$0` refusal with no provider resource -- what changes here
        #: is only that it reads like one.
        print(f"REFUSED at $0: {exc}")
        print("No provider resource was created and nothing bills.")
        return 11

    if args.dry_run:
        from stages.phase_d1 import d1_authorization as A
        priced = A.session_ceiling(REPO_ROOT)
        print(f"session_id      {session.session_id}")
        print(f"plan_hash       {session.plan_hash[:24]}")
        print(f"authorization   {session.authorization_path}")
        #: The MONEY is the authorization's, not the BudgetSpec's: the spec is a
        #: phase-TIME model the runner plans minutes from, and the dollar ceiling
        #: comes from the one-use artifact. Printing a `hard_cap_usd` off the spec
        #: would be inventing a second price.
        print(f"priced ceiling  ${priced['hard_ceiling_usd']:.4f} "
              f"({priced['hard_ceiling_minutes']:.2f} min)")
        print(f"priced expected ${priced['expected_usd']:.4f}")
        print(f"pricing basis   {priced['_basis']}")
        print(f"SESSION_KIND    {session.setup.env['SESSION_KIND']}")
        print(f"test_paths      {list(session.setup.test_paths)}")
        print(f"local_assets    {[a.as_env_entry() for a in session.setup.local_assets]}")
        print(f"setup_markers   {list(session.setup.setup_markers)}")
        print(f"prechecks       {[getattr(c, '__name__', str(c)) for c in session.precheck]}")
        print(f"resolved        {resolved}")
        print()

    #: THE DRY RUN GOES THROUGH THE REAL RUNNER.
    #:
    #: `--dry-run` used to `return 0` HERE, before `SessionRunner` was ever
    #: constructed -- so it validated the declaration and nothing else. It never
    #: ran `make_plan`, never priced the GPU, never executed a single precheck,
    #: and never touched the authorization beyond building the spec. It also
    #: exited 0, which reads as a pass.
    #:
    #: The runner has its own dry-run stop, after `make_plan` and
    #: `run_prechecks` and before `create()`, which is the gate this flag was
    #: always supposed to mean. Returning early bypassed it, and hid both the
    #: `None` price above and every missing gate below. The runner returns
    #: False on that path deliberately -- "every pre-provider gate passed and
    #: nothing was created" is not a session success -- so a clean dry run is
    #: exit 0 here and a refused one is not.
    #:
    #: `repo_root` is POSITIONAL and required: the runner resolves the
    #: authorization path, the harness and the bundle against it.
    if args.dry_run:
        #: CONSTRUCTED DIRECTLY, because this path reaches the runner's dry-run
        #: stop before `create()` and so can never own a provider resource --
        #: there is nothing for `run_session`'s teardown-on-error wrapper to
        #: tear down. The runner's record is needed here, and `run_session`
        #: does not return the runner.
        try:
            runner = SessionRunner(session, args, REPO_ROOT)
        except AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            print("No provider resource was created and nothing bills.")
            return 11
        runner.run()
        terminal = runner.ev.get("terminal")
        passed = terminal == "DRY_RUN_GATES_PASSED"
        runner.ev["passed"] = False
        runner.save()
        print(f"\nterminal        {terminal}")
        print(f"record          {args.out}")
        print("CREATED NOTHING: every pre-provider gate "
              f"{'PASSED' if passed else 'did NOT pass'}; "
              "no provider resource was created and nothing bills.")
        if passed:
            print("A DRY RUN IS NOT A PASS: no pod existed, no stage ran and "
                  "nothing was measured. It is evidence the chain is "
                  "launchable.")
        return 0 if passed else 11

    #: THE SHARED `main`. This was `return SessionRunner(...).run()`, which has
    #: two defects the shared helper does not have:
    #:
    #: * it returns a BOOL, and `raise SystemExit(main())` maps `True` to exit
    #:   status 1. A completed session exited 1 and a refused one exited 0 --
    #:   inverted, for anything reading the status;
    #: * nothing caught an exception out of `run()`. The runner tears down on
    #:   the paths it controls, but a launcher-level error after `create()`
    #:   would have propagated with a pod still billing and no `teardown_now`.
    #:   `run_session` exists precisely so "no session can forget to record
    #:   `passed`, to write the record, or to tear down after a runner error".
    try:
        return run_session(
            session, args, REPO_ROOT,
            summary=("D1 formal search: the committed Top-2 and the search "
                     "evidence are what this session owes."))
    except AuthorizationError as exc:
        #: `run_session` constructs the runner OUTSIDE its own try block, so a
        #: load or harness refusal escapes it. Nothing has been created at that
        #: point -- the loader runs before the provider is contacted.
        print(f"REFUSED at $0: {exc}")
        print("No provider resource was created and nothing bills.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())

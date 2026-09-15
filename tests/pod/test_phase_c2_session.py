"""The Phase-C2 Search-1 execution seam: launcher, driver, and what binds them.

Zero cost, CPU only, no pod. What this file is for is the class of defect that
only appears on a billing pod: a launcher that emits a command the driver cannot
parse, an artifact spec that names a path the driver does not write, a budget
whose ceiling is not the one the record derives, an interpreter read off an
object that has no such field.

The last one was real. `driver_command` said `ctx.args.remote_python` in its
first version — `args` has no such field, `commands` does — and the failure
would have been an `AttributeError` at driver start, after the pod existed and
setup had already succeeded. `test_the_launcher_and_the_driver_agree_on_every_flag`
executes that line and hands the result to the driver's own parser, which is the
only check that could have caught it without paying.

Nothing here launches anything, and one test asserts that it cannot: the grant
the launcher names does not exist.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

LAUNCHER = REPO / "scripts/pod/autoinit_phase_c2_launch.py"
DRIVER = REPO / "scripts/pod/autoinit_phase_c2_driver.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registered():
    from aadistill.initialization.operators import attention_activation
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()
    try:
        yield
    finally:
        attention_activation.unregister()


@pytest.fixture(scope="module")
def launcher(registered):
    return load(LAUNCHER, "c2_launch")


@pytest.fixture(scope="module")
def driver(registered):
    return load(DRIVER, "c2_driver")


@pytest.fixture(scope="module")
def session(launcher):
    #: `--run-id` is REQUIRED since 2026-09-15: every governance artifact this
    #: session consumes is owned by its run — grant, authorization, readiness
    #: record, bundle record — and each is resolved from that id. A session with
    #: no run id has nowhere for its authorization to live that the next
    #: issuance would not overwrite, which is the state C1 spent a pointer, a
    #: history file and a migration getting out of.
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/c2", "--session-commit", "0" * 40,
         "--bundle", "aad_00000000.bundle", "--run-id", "spec_check"])
    return args, launcher.spec(args).validate()


# --- the declaration -------------------------------------------------------

def test_the_declaration_is_complete(session):
    """`validate()` is the refusal that must happen before anything is priced."""
    _args, spec = session
    assert spec.session_id == "autoinit-phase-c2-search1"
    assert spec.schema.startswith("aadistill.autoinit.c2_session/")
    assert spec.markers.success == "ALL_DONE"
    assert spec.markers.failure == ("PHASE_C2_FAILED",)


def test_this_session_trains_nothing_and_says_so(session):
    _args, spec = session
    assert spec.budget.arms == 0 and spec.budget.steps_per_arm == 0
    assert spec.evidence_fields["trains_anything"] is False
    #: No product is fetched, because the search exports no checkpoint anybody
    #: collects: its leaves are measured on the pod and their identities come
    #: home in the journal. A collector looking for weights would find none.
    assert spec.markers.products_eligible("ALL_DONE", {}) is False


def test_it_cannot_run_because_the_grant_does_not_exist(session):
    """The honest state of this change: implemented, not authorized."""
    _args, spec = session
    assert not (REPO / spec.authorization_path).exists()
    from experiments.phase_c2.session import C2Authorization

    #: By identity of the underlying function, not of the bound classmethod:
    #: `C2Authorization.load` builds a new bound object on every attribute
    #: access, so `is` compares two wrappers and always fails.
    assert (spec.authorization_loader.__func__
            is C2Authorization.load.__func__)


def test_the_authorization_type_authorizes_only_this_search():
    from aadistill.governance.authorization import AuthorizationError
    from experiments.phase_c2.session import SCHEMA, C2Authorization

    auth = C2Authorization(
        authorization_id="x", granted_utc="2026-01-01T00:00:00Z",
        granted_by="nobody", plan_id="p", plan_hash="h",
        science_plan_hash="s", expected_usd=1.0, hard_cap_usd=2.0,
        authorized_stages=(0,), stage_conditions={}, scope_note="test")
    assert auth.allows_phase_a is False
    assert auth.allows_recovery_training is False
    assert auth.authorizes_c2_search1 is True
    doc = auth.as_dict()
    assert doc["schema"] == SCHEMA
    #: Serialized from the properties, so a document cannot disagree with the
    #: object that wrote it.
    assert doc["allows_phase_a"] is False
    assert doc["allows_recovery_training"] is False

    #: A grant of any other type cannot stand in, by schema, at load.
    other = dict(doc, schema="aadistill.autoinit.c1_authorization/v1")
    other.pop("authorization_sha256")
    from aadistill.infrastructure.manifest import sha256_json
    other["authorization_sha256"] = sha256_json(other)
    path = REPO / "artifacts" / "c2_auth_wrong_schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(other))
    try:
        with pytest.raises(AuthorizationError, match="cannot authorize"):
            C2Authorization.load(path)
    finally:
        path.unlink()


# --- the seam that only a pod would otherwise exercise ---------------------

def built_plan(spec):
    from aadistill.infrastructure.budget import StepTime, plan_session
    from experiments.phase_c2.session import c2_hard_ceiling_usd

    b = spec.budget
    return plan_session(
        price_per_hour=1.09, authorized_usd=c2_hard_ceiling_usd(REPO),
        arms=b.arms, steps_per_arm=b.steps_per_arm,
        step_time=StepTime(seconds=b.step_seconds, source=b.step_source),
        setup_minutes=b.setup_minutes,
        eval_minutes_per_arm=b.eval_minutes_per_arm,
        transfer_minutes=b.transfer_minutes, other_phases=b.other_phases,
        contingency_fraction=b.contingency_fraction,
        soft_stop_reserves=b.soft_stop_reserves,
        artifact_recovery_reserve_minutes=b.artifact_recovery_reserve_minutes)


def test_the_launcher_and_the_driver_agree_on_every_flag(session, driver):
    """Execute `driver_command`, then parse it with the driver's own parser.

    Both halves of the contract, for real. A launcher that emits a flag the
    driver does not accept, or omits one it requires, fails here instead of at
    driver start on a pod that has already paid for setup.
    """
    args, spec = session
    plan = built_plan(spec)
    ctx = SimpleNamespace(args=args, image_digest="img@sha256:abc",
                          price=1.09, spent_usd=0.0)
    command = spec.driver_command(ctx, plan)

    assert command.startswith("/opt/train/bin/python "), command
    assert "autoinit_phase_c2_driver.py" in command
    parsed = driver.build_parser().parse_args(command.split()[2:])

    assert parsed.stage == "all" and parsed.device == "cuda"
    assert parsed.authorization_path == spec.authorization_path
    #: The BEAM's envelope, twice, and the baseline's separately.
    #:
    #: This asserted `search_minutes == base` and `deadline == base + EVERY
    #: reserve` until 2026-09-15. Both were wrong, in opposite directions: the
    #: affordability check approved the beam against its expected 300.16-minute
    #: trajectory while its own deadline permitted 663.6, and that deadline
    #: included the 27.665 minutes the accounting holds for a missing-B rebuild.
    #: The partition is now enforced, so the two beam figures coincide by
    #: construction and the rebuild has its own flag.
    base = next(p.minutes for p in plan.breakdown
                if p.name == "beam_search_depth_early")
    risk = next(r.minutes for r in plan.soft_stop_reserves
                if r.name == "beam_composition_risk")
    rebuild = next(r.minutes for r in plan.soft_stop_reserves
                   if r.name == "baseline_rebuild_reserve")
    assert parsed.search_deadline_minutes == pytest.approx(base + risk, abs=0.05)
    assert parsed.search_minutes == pytest.approx(
        parsed.search_deadline_minutes, abs=0.05)
    assert parsed.search_deadline_minutes < base + risk + rebuild
    assert parsed.baseline_rebuild_minutes == pytest.approx(rebuild, abs=1e-6)
    #: And the driver is never told it may spend more than the plan allows.
    assert parsed.authorized_usd <= plan.hard_terminate_usd + 1e-9
    assert parsed.soft_stop_usd <= plan.soft_stop_usd + 1e-9


def test_the_artifact_specs_name_the_paths_the_driver_writes(session, driver):
    """Writer and collector, held to the same constants.

    Phase-B attempt 3 wrote `phase_b_search` while its specs named
    `phase_a_search`: the collector matched nothing, `min_matches: 0` reported
    `missing: 0`, and a deadline failure came home with no per-state timings.
    """
    _args, spec = session
    audit = spec.artifacts.audit_dirname
    assert driver.AUDIT.name == audit
    workdir = driver.SEARCH_WORKDIR.name

    for rel in (spec.artifacts.spec_success, spec.artifacts.spec_failed):
        doc = json.loads((REPO / rel).read_text())
        patterns = [e["pattern"] for e in doc["entries"]]
        assert f"audit/{audit}/{spec.artifacts.evidence_filename}" in patterns
        assert any(p.startswith(f"autoinit/{workdir}/") for p in patterns), (
            f"{rel} names no path under the driver's search workdir")
        for pattern in patterns:
            assert pattern.startswith((f"audit/{audit}/",
                                       f"autoinit/{workdir}/")), pattern

    #: The success spec REQUIRES the journal and the committed selection; the
    #: failed spec must not, because an early failure legitimately has neither
    #: and demanding them would block teardown of a pod that is still billing.
    success = json.loads((REPO / spec.artifacts.spec_success).read_text())
    failed = json.loads((REPO / spec.artifacts.spec_failed).read_text())
    required = {e["artifact_class"] for e in success["entries"] if e["required"]}
    assert {"search_journal", "stage1_selection", "session_evidence"} <= required
    failed_required = {e["artifact_class"] for e in failed["entries"]
                       if e["required"]}
    assert failed_required == {"session_evidence"}


def test_the_evidence_document_is_relayed_as_rewritten_in_place(session):
    """It is rewritten on every state change, which is what `whole_file` is for.

    The runner declares the `ArtifactPolicy` evidence file as the one whole-file
    spec, so this session inherits that repair by naming its evidence file
    there rather than by remembering to ask.
    """
    _args, spec = session
    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    block = runner[runner.index("specs = ["):runner.index("relay = LogRelay(")]
    assert "whole_file=True" in block
    assert spec.artifacts.evidence_filename == "c2_evidence.json"


# --- the budget ------------------------------------------------------------

def test_the_budget_reproduces_the_pricing_record(session):
    _args, spec = session
    from experiments.phase_c2.session import c2_hard_ceiling_usd, load_pricing

    doc = load_pricing(REPO)
    plan = built_plan(spec)
    assert plan.hard_terminate_usd == pytest.approx(
        c2_hard_ceiling_usd(REPO), abs=5e-5)
    assert plan.expected_usd == pytest.approx(
        doc["totals"]["expected_usd"], abs=5e-5)
    assert plan.soft_stop_usd == pytest.approx(
        doc["totals"]["soft_stop_usd"], abs=5e-5)


def test_both_reserves_are_named_and_stay_separate(session):
    """Five envelopes, five meanings. One number for two of them is one number
    nobody can interpret."""
    _args, spec = session
    plan = built_plan(spec)
    reserves = {r.name: r.minutes for r in plan.soft_stop_reserves}
    assert set(reserves) == {"beam_composition_risk",
                             "baseline_rebuild_reserve"}
    #: The conditional one is NOT folded into the artifact-recovery reserve.
    #: Those protect different things: one buys a baseline the experiment needs,
    #: the other buys the chance to bring evidence home after something already
    #: went wrong, and paying for a rebuild out of the teardown reserve is how a
    #: session ends with a result it cannot collect.
    assert plan.artifact_recovery_reserve_minutes == 30.0
    assert reserves["baseline_rebuild_reserve"] not in (
        0.0, plan.artifact_recovery_reserve_minutes)
    #: Contingency applies to the expected path only; the reserves are added
    #: after it, which is what makes them protect the work.
    assert plan.soft_stop_minutes == pytest.approx(
        plan.expected_minutes * 1.10 + sum(reserves.values()), abs=1e-6)


def test_the_baseline_reserve_is_derived_from_c1s_measured_stages():
    """Measured, not modelled: the same work, on the same card."""
    from experiments.phase_c2.session import load_pricing

    doc = load_pricing(REPO)
    entry = next(r for r in doc["reserves"]
                 if r["reserve"] == "baseline_rebuild_reserve")
    assert entry["conditional"] is True
    assert entry["basis"] == "measured"
    detail = entry["detail"]
    #: Stage C->D is the shared parent from the teacher; stage E->F is the
    #: treatment's ATTENTION step. Stage E itself is the INCUMBENT's replay and
    #: is excluded, because the C2 baseline does not need it.
    assert detail["parent_replay_minutes"] == pytest.approx(24.185, abs=0.01)
    assert detail["minutes"] == pytest.approx(
        detail["parent_replay_minutes"] + detail["treatment_step_minutes"]
        + detail["state_eval_minutes"], abs=1e-6)
    assert "excluded" in detail["source"]


def test_a_tampered_pricing_record_is_refused(tmp_path):
    from aadistill.governance.authorization import AuthorizationError
    from experiments.phase_c2 import session as S

    doc = json.loads((REPO / S.PRICING_PATH).read_text())
    doc["totals"]["hard_ceiling_usd"] = 999.0
    fake = tmp_path / S.PRICING_PATH
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(doc))
    with pytest.raises(AuthorizationError, match="pricing_sha256"):
        S.load_pricing(tmp_path)


# --- the searched space the driver actually configures ---------------------

def test_the_driver_searches_exactly_the_accepted_space(driver):
    """The space is imported from the experiment layer, not restated here.

    A driver that declared its own copy of the operator set or the profile
    restriction could search something the plan hash does not describe.
    """
    body = DRIVER.read_text()
    assert "from experiments.phase_c2.search_space import" in body
    for name in ("C2_ALLOWED_IMPLS", "C2_IMPL_PROFILES", "C2_PROFILE_IDS"):
        assert f"{name}," in body or f"{name}=" in body or name in body
    #: and it must not name an operator id or a profile id of its own.
    for literal in ("depth.causal_kl_greedy_v1", "width.global_pca_v0",
                    "calib.domain_balanced@v1", "calib.reasoning_heavy@v2"):
        assert literal not in body, (
            f"the driver hardcodes {literal!r}; the space belongs to "
            "experiments.phase_c2.search_space")


def test_the_harness_set_names_every_file_the_session_executes(registered):
    """A new executable in the session's path that the set does not name means a
    grant would certify different code.

    Until 2026-09-15 this compared the digested set to a hand-maintained
    eighteen-path CONSTANT, and both sides were the same declaration — so the
    only thing it could catch was somebody editing one of them. The set is now
    DERIVED by walking import edges and subprocess targets from the declared
    entry points, which is why the constant is gone from the comparison: the
    live walk finds 60-odd files that declaration never named, including every
    module that can spend money. What survives here is the property that
    mattered: each file the session demonstrably executes is measured.

    `comparison.py` decides what the run's baseline evidence IS and is the only
    place a rebuilt B's metric becomes durable, so its bytes are executable by
    consequence — the same reason the artifact specs are in the set.
    """
    from pathlib import Path as _P

    from experiments.phase_c2.session import (
        C2_HARNESS_SOURCE_FILES_V1, c2_harness_digest,
    )

    live = c2_harness_digest(REPO)
    digested = {e["path"] for e in live["files"]}
    for path in digested:
        assert (REPO / path).is_file(), path
    #: The historical declaration is still a valid set of paths — it is what the
    #: superseded attempt-1 grant binds — and it is no longer what runs.
    for declared in C2_HARNESS_SOURCE_FILES_V1:
        assert (REPO / declared).is_file(), declared
    assert digested != set(C2_HARNESS_SOURCE_FILES_V1)
    assert len(digested) > len(C2_HARNESS_SOURCE_FILES_V1)
    #: Everything that can create or bill a provider resource.
    for spends in ("src/aadistill/infrastructure/session_runner.py",
                   "src/aadistill/infrastructure/provider.py",
                   "src/aadistill/infrastructure/remote.py",
                   "src/aadistill/infrastructure/budget.py"):
        assert spends in digested, f"{spends} can spend money and must bind"

    #: Every module the driver and launcher import from the experiment layer.
    for required in ("scripts/experiments/phase_c2/comparison.py",
                     "scripts/experiments/phase_c2/baseline.py",
                     "scripts/experiments/phase_c2/search_space.py",
                     "configs/autoinit/c2_artifacts.json",
                     "configs/autoinit/c2_artifacts_failed.json"):
        assert required in digested, f"{required} is not measured by the grant"


def test_the_comparison_is_written_after_the_search_and_never_over_it(driver):
    """Order, from the source: the comparison cannot precede what it compares.

    And it must land in the audit root the collector walks, not in the search
    workdir beside the beam's own ranking.
    """
    body = DRIVER.read_text()
    search = body.index("found = run_phase_a_search(")
    resolve = body.index("C.resolve_baseline_state(")
    commit = body.index("C.commit(comparison")
    record = body.index('self.record("search_and_baseline"')
    assert search < resolve < commit < record, (
        "the comparison is built or committed out of order")
    #: Committed to AUDIT, which the artifact spec collects.
    assert "C.commit(comparison, AUDIT)" in body
    assert "stage1_selection" not in body[commit:record].replace(
        "selection_record", "")


def test_the_plan_hash_binds_the_space_and_moves_with_it(registered):
    """A grant issued against a different space cannot authorize this run."""
    from experiments.phase_c2 import search_space as SS
    from experiments.phase_c2.session import c2_plan_hash

    before = c2_plan_hash()
    original = dict(SS.C2_IMPL_PROFILES)
    try:
        SS.C2_IMPL_PROFILES["width.global_pca_v0"] = SS.C2_PROFILE_IDS
        assert c2_plan_hash() != before, (
            "the plan hash does not cover the profile restriction, so a grant "
            "would survive widening the search")
    finally:
        SS.C2_IMPL_PROFILES.clear()
        SS.C2_IMPL_PROFILES.update(original)
    assert c2_plan_hash() == before


def test_the_core_carries_no_phase_c2_instance_constant():
    """`src/aadistill` stays reusable: the experiment layer owns the instance.

    Scoped to C2-INSTANCE names. A broader sweep — `qwen3_0p6b`,
    `activation_importance_v1`, `calib.reasoning_heavy` — is red at `828a3c4`
    before this change touches anything: those appear in core DOCSTRINGS that
    predate C2, and prose in core is `test_core_ownership`'s subject, not this
    file's. A test that fails for a reason its own change did not cause teaches
    the next reader to ignore it.
    """
    core = REPO / "src/aadistill"
    forbidden = ("phase_c2", "autoinit.v1.phase_c2", "c2_evidence",
                 "c2_artifacts", "baseline_rebuild", "C2Authorization",
                 "c2_search1")
    offenders = []
    for path in sorted(core.rglob("*.py")):
        text = path.read_text()
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO)}: {needle!r}")
    assert offenders == [], offenders


def test_the_generic_search_seam_defaults_to_the_previous_behaviour():
    """`conditional_candidates` is inert unless a caller passes one.

    The hook was added to a function Phase A and Phase B still run. Its default
    must leave them byte-identical, and its signature must be the two-argument
    one the baseline fallback expects — a one-argument hook would have meant the
    rebuild could only ever load its own second copy of the teacher.
    """
    import inspect

    import phase_a_search

    sig = inspect.signature(phase_a_search.run_phase_a_search)
    assert sig.parameters["conditional_candidates"].default is None
    body = (REPO / "scripts/autoinit/phase_a_search.py").read_text()
    assert "conditional_candidates(result, lambda: teacher)" in body, (
        "the hook must receive the teacher THIS search used")
    #: And its results join the same measured-candidate loop, so a conditional
    #: candidate takes the identical canonical-reload / state_eval path.
    assert "for entry in (*retained_candidates, *conditional):" in body

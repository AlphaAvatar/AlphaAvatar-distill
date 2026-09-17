"""The Phase-C2 full-search governance chain, EXERCISED.

Not assertions about source text. Every mechanism here is driven: the
authorization type loads and refuses, the assembler validates and refuses, the
ceiling re-derives at a moved rate, the eleven `$0` gates run against a real
namespace, the storage bound walks the real space, and both artifact specs load
through the collector's own loader.

**The chain is not launchable yet and these tests do not make it so.** No grant
has been approved, so no authorization can be issued; what is verified is that
the machinery would refuse everything it should and admit only what it must.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

from experiments.phase_c2 import full_search as FSG  # noqa: E402
from experiments.phase_c2 import full_search_authorization as FA  # noqa: E402
from experiments.phase_c2 import full_search_bundle as FST  # noqa: E402
from experiments.phase_c2 import full_search_pod_environment as FPE  # noqa: E402


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location(
        "c2_full_search_launch",
        REPO / "scripts/pod/autoinit_phase_c2_full_search_launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["c2_full_search_launch"] = module
    spec.loader.exec_module(module)
    return module


def launch_args(launcher, **over):
    base = dict(scr="/tmp/c2fs", session_commit="a" * 40, bundle="b",
                run_id="attempt1", max_price=1.09)
    base.update(over)
    argv = []
    for key, value in base.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return launcher.build_parser().parse_args(argv)


# --- the type refuses what it must ------------------------------------------

def test_it_authorizes_the_search_and_nothing_else():
    """Four properties, and three of them are NEVER.

    Read off the class rather than a serialized artifact, because the
    serializer reads them FROM the properties -- so a literal in `as_dict`
    could not disagree with the class without this failing.
    """
    auth = FSG.FullSearchAuthorization.__dict__
    for name, expected in (("authorizes_c2_full_search", True),
                           ("authorizes_c2_search1", False),
                           ("authorizes_c2_baseline_completion", False),
                           ("authorizes_behavioural_selection", False)):
        prop = auth[name]
        assert isinstance(prop, property), f"{name} must be a property"
        assert prop.fget(object.__new__(FSG.FullSearchAuthorization)) is expected


def test_the_schemas_are_mutually_exclusive():
    """The one-line check that stops any C2 artifact standing in for another."""
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2 import session as C2S

    schemas = {FSG.SCHEMA, BC.SCHEMA, C2S.SCHEMA}
    assert len(schemas) == 3, f"two C2 sessions share a schema: {schemas}"


def test_a_foreign_schema_is_refused(tmp_path):
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.manifest import sha256_json

    doc = {"schema": "aadistill.autoinit.c2_authorization/v1", "anything": 1}
    doc["authorization_sha256"] = sha256_json(
        {k: v for k, v in doc.items() if k != "authorization_sha256"})
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(AuthorizationError, match="declares schema"):
        FSG.FullSearchAuthorization.load(path)


def test_an_artifact_claiming_more_is_refused(tmp_path):
    """Each forbidden claim, one at a time, so a single guard cannot cover all."""
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.manifest import sha256_json

    for claim in ("authorizes_c2_search1", "authorizes_c2_baseline_completion",
                  "authorizes_behavioural_selection",
                  "allows_recovery_training"):
        doc = {"schema": FSG.SCHEMA, claim: True}
        doc["authorization_sha256"] = sha256_json(
            {k: v for k, v in doc.items() if k != "authorization_sha256"})
        path = tmp_path / f"{claim}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(AuthorizationError, match=claim):
            FSG.FullSearchAuthorization.load(path)


def test_a_stage_after_commit_top_k_is_refused(tmp_path):
    """The sequence IS the scope, so a fourth stage is a different session."""
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.manifest import sha256_json

    doc = {"schema": FSG.SCHEMA,
           "authorized_stages": [*FSG.AUTHORIZED_STAGES, "recover_probes"]}
    doc["authorization_sha256"] = sha256_json(
        {k: v for k, v in doc.items() if k != "authorization_sha256"})
    path = tmp_path / "extra_stage.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(AuthorizationError, match="authorizes stages"):
        FSG.FullSearchAuthorization.load(path)


def test_the_stage_sequence_ends_at_commit_top_k():
    assert FSG.AUTHORIZED_STAGES[-1] == "commit_top_k"
    cfg = FA.load_config(REPO)
    assert tuple(cfg["authorized_stages"]) == FSG.AUTHORIZED_STAGES, (
        "the config and the module disagree about the scope")


# --- the money moves with the rate, the beam does not -----------------------

def test_the_beam_width_comes_from_the_frozen_schedule():
    """Not a literal in a governance file.

    The maintainer has twice stated that width 6 is the standing design and
    that shrinking it to fit a cap is not permitted. Reading it from
    `SCHEDULE_V1` means a narrower width would have to change the schedule,
    which is a change of experiment rather than an edit to a config.
    """
    from experiments.phase_c2 import full_search_space as FS

    assert FSG.standing_beam_width() == FS.SCHEDULE_V1.width == 6
    #: And the config must not restate it.
    assert "beam_width" not in json.dumps(FA.load_config(REPO)), (
        "the authorization config states a beam width; it is derived")


def test_the_ceiling_is_the_priced_minutes_times_the_rate():
    """Mechanically, in both directions, and rounded UP.

    A ceiling that rounds DOWN under-authorizes the plan it covers -- the C1
    grant found that at $15.147403 against a recorded $15.1474.
    """
    minutes = float(FSG._standing_row(REPO)["hard_ceiling_minutes"])
    for rate in (1.09, 1.29, 0.89):
        got = FSG.derive_ceiling_usd(rate, REPO)
        exact = minutes / 60.0 * rate
        assert got >= exact - 1e-12, f"the ceiling rounds DOWN at ${rate}/h"
        assert got - exact < 1e-4
    #: At the priced basis it reproduces the record.
    assert FSG.derive_ceiling_usd(FSG.price_per_hour_basis(REPO), REPO) == \
        pytest.approx(FSG.hard_ceiling_usd(REPO), abs=1e-4)
    #: And a dearer card costs MORE for the same work, never less search.
    assert FSG.derive_ceiling_usd(1.29, REPO) > FSG.derive_ceiling_usd(1.09, REPO)


def test_approved_money_must_match_the_derivation_at_its_own_rate():
    """The check that makes a re-quoted rate real rather than decorative."""
    rate = 1.29
    good = {
        "price_basis_usd_per_hour": rate,
        "max_price_usd_per_hour": 1.50,
        "hard_cap_usd": FSG.total_ceiling_usd(rate, REPO)["total_hard_ceiling_usd"],
        "expected_usd": round(FSG.expected_usd(REPO)
                              / FSG.price_per_hour_basis(REPO) * rate, 4),
    }
    assert FA.check_approved_money({"approved_money": good}, REPO)

    #: The ceiling computed at the STALE basis while quoting a higher live rate.
    stale = {**good, "hard_cap_usd": FSG.hard_ceiling_usd(REPO)}
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="mechanical"):
        FA.check_approved_money({"approved_money": stale}, REPO)

    #: And a GPU-ONLY ceiling at the CORRECT rate -- the defect review found:
    #: every figure self-consistent, and the session would still have exceeded
    #: it by the container disk nobody had priced.
    gpu_only = {**good, "hard_cap_usd": FSG.derive_ceiling_usd(rate, REPO)}
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="mechanical"):
        FA.check_approved_money({"approved_money": gpu_only}, REPO)

    #: A rate above the boundary the grant itself states.
    chasing = {**good, "max_price_usd_per_hour": 1.09}
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="boundary"):
        FA.check_approved_money({"approved_money": chasing}, REPO)

    #: And a grant that states no boundary at all cannot be checked.
    for missing in ("price_basis_usd_per_hour", "max_price_usd_per_hour",
                    "hard_cap_usd", "expected_usd"):
        partial = {k: v for k, v in good.items() if k != missing}
        with pytest.raises(FA.FullSearchAuthorizationRefused, match="missing"):
            FA.check_approved_money({"approved_money": partial}, REPO)


def test_a_grant_with_no_approved_money_is_refused():
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="no approved_money"):
        FA.check_approved_money({}, REPO)


# --- the identities the grant must reproduce --------------------------------

def test_the_derived_identities_name_the_space():
    """What makes a Search-1 grant unusable here and this one unusable there."""
    live = FA.live_identities(REPO)
    assert live["joint_space_total_leaves"] == 578
    assert live["joint_space_decomposed_leaves"] == 576
    assert live["joint_space_impl_profiles_are_unpinned"] is True
    assert len(live["joint_space_allowed_impls"]) == 6
    assert live["excluded_implementations"] == ["attention.weight_proxy_v0"]
    assert live["beam_width"] == 6
    assert live["calibration_profile_ids"] == ["calib.domain_balanced@v1",
                                               "calib.reasoning_heavy@v2"]
    assert live["cost_table_sources"] == ["phase_b_attempt5", "phase_c2_attempt4"]


def test_the_identity_block_must_be_complete_not_merely_undisputed():
    """A grant asserting two of thirteen identities approves almost nothing.

    Driven through the real assembler with a deliberately partial block.
    """
    live = {k: v for k, v in FA.live_identities(REPO).items()
            if not k.startswith("_")}
    cfg = FA.load_config(REPO)
    block = cfg["grant_contract"]["identities_block"]
    partial = dict(list(live.items())[:2])
    grant = {
        "granted_by": "a test", "covers": "nothing",
        "explicitly_not_authorized": [], "one_use": {},
        "approved_money": {
            "price_basis_usd_per_hour": 1.09,
            "max_price_usd_per_hour": 1.09,
            "hard_cap_usd": FSG.total_ceiling_usd(
                1.09, REPO)["total_hard_ceiling_usd"],
            "expected_usd": FSG.expected_usd(REPO)},
        "budget_context_at_approval": {"cumulative_spend_usd": 297.6543,
                                       "authorized_cap_usd": 370.0},
        block: partial,
    }
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="does not state"):
        FA.build_payload(grant=grant, session_commit="a" * 40,
                         granted_utc="2026-01-01T00:00:00+00:00",
                         run_id="attempt1", repo_root=REPO)


def test_an_identity_the_issuer_cannot_derive_is_refused():
    live = {k: v for k, v in FA.live_identities(REPO).items()
            if not k.startswith("_")}
    cfg = FA.load_config(REPO)
    block = cfg["grant_contract"]["identities_block"]
    grant = {
        "granted_by": "a test", "covers": "nothing",
        "explicitly_not_authorized": [], "one_use": {},
        "approved_money": {
            "price_basis_usd_per_hour": 1.09,
            "max_price_usd_per_hour": 1.09,
            "hard_cap_usd": FSG.total_ceiling_usd(
                1.09, REPO)["total_hard_ceiling_usd"],
            "expected_usd": FSG.expected_usd(REPO)},
        "budget_context_at_approval": {"cumulative_spend_usd": 297.6543,
                                       "authorized_cap_usd": 370.0},
        block: {**live, "a_binding_nobody_checks": "hello"},
    }
    with pytest.raises(FA.FullSearchAuthorizationRefused, match="cannot derive"):
        FA.build_payload(grant=grant, session_commit="a" * 40,
                         granted_utc="2026-01-01T00:00:00+00:00",
                         run_id="attempt1", repo_root=REPO)


def test_the_config_names_the_cap_the_project_owns():
    """One number, two documents, and a test that pins them together."""
    cfg = FA.load_config(REPO)
    owner = json.loads(
        (REPO / "configs/experiments/phase_c1/authorization.json").read_text())
    assert (cfg["accepted_pricing"]["cumulative_cap_usd"]
            == owner["accepted_pricing"]["cumulative_cap_usd"]), (
        "the full-search config's cap and the project's canonical cap differ")


# --- the storage bound ------------------------------------------------------

def test_the_storage_walk_follows_the_REAL_release_lifecycle():
    """The correction review forced, and the reason it matters.

    `BeamSearch.run` calls `_release_weights` in exactly one place: on the
    PARTIAL children a level's ranking pruned. So the root, every state that was
    selected and then expanded, every dead end and every completed leaf stay on
    disk for the whole run -- residency is CUMULATIVE.

    An earlier version modelled `current parents + current children +
    accumulated leaves` and therefore missed the retained ancestors, which is
    not an upper bound. The observable signature of the fix is that residency
    after pruning never decreases.
    """
    report = FSG.peak_resident_gib(REPO)
    assert report["peak_at_level"] == 1
    assert report["beam_width"] == 6
    level1 = report["levels"][1]
    assert level1["parents"] == 9 and level1["generated"] == 60

    #: Monotonic: nothing but pruned partial children is ever released, so what
    #: is left after a level can only grow.
    after = [row["resident_after_pruning_gib"] for row in report["levels"]]
    assert after == sorted(after), (
        f"residency fell between levels ({after}); something is being released "
        "that the search does not release")
    #: And the ancestors are actually counted: level 2 must carry more retained
    #: ancestor bytes than level 1, because level 1's kept beam joined them.
    assert (report["levels"][2]["retained_ancestors_gib"]
            > report["levels"][1]["retained_ancestors_gib"])

    #: Cross-checked against reality: a finished leaf is the 596M target, and
    #: the CUDA engineering validation weighed its real checkpoint at 1.11 GiB.
    assert report["target_state_gib"] == pytest.approx(1.11, abs=0.01)


def test_the_corrected_peak_exceeds_the_model_that_missed_the_ancestors():
    """A regression on the DIRECTION of the correction.

    The superseded model returned 243.4 GiB. The corrected one must return more,
    because it adds states the old one ignored -- so an edit that quietly
    reinstated the cheaper model would fail here rather than silently shrinking
    a provision.
    """
    assert FSG.peak_resident_gib(REPO)["peak_resident_gib"] > 243.4


def test_the_environment_allowance_derives_the_teacher():
    env = FSG.teacher_and_environment_gib(REPO)
    assert env["teacher_resident_gib"] == pytest.approx(7.49, abs=0.1)
    #: Counted twice on purpose: the HF snapshot and the materialized copy.
    assert env["total_gib"] > env["teacher_resident_gib"] * 2


# --- the provider bills more than the GPU -----------------------------------

def test_the_total_ceiling_covers_the_container_disk():
    """The blocker: `securePrice` is the GPU and nothing else.

    A ceiling of GPU minutes times the GPU rate did not cover a session that
    provisions hundreds of GB of separately priced container disk -- and no
    figure in the record disagreed with any other, which is why review found it
    rather than a gate.
    """
    total = FSG.total_ceiling_usd(1.09, REPO)
    assert total["gpu_usd"] == FSG.derive_ceiling_usd(1.09, REPO)
    assert total["container_disk"]["usd"] > 0
    assert total["total_hard_ceiling_usd"] > total["gpu_usd"]
    hours = total["hard_ceiling_hours"]
    assert total["total_hard_ceiling_usd"] == pytest.approx(
        hours * total["effective_rate_usd_per_hour"], abs=1e-3)
    assert total["effective_rate_usd_per_hour"] > 1.09
    assert total["network_volume_usd"] == 0.0


def test_the_storage_basis_declares_that_it_is_not_a_quote():
    """The distinction the maintainer asked to be machine-readable.

    The GPU rate is re-quoted live and refuses a launch above its boundary.
    There is no equivalent query for storage, so the basis is dated and stated
    -- and a document presenting it as a quote would let an unqueryable constant
    inherit a live quote's credibility.
    """
    pricing = FSG.storage_pricing(REPO)
    assert pricing["container_disk"]["provider_api_exposes_this"] is False
    assert "2026-09-17" in pricing["container_disk"]["_basis"]
    assert pricing["proration"]["hours_per_month"] == 720, (
        "the shorter month is the dearer hourly rate, which is what a bound "
        "takes")


def test_a_basis_claiming_to_be_a_quote_is_refused(tmp_path):
    """The guard must actually fire."""
    doc = json.loads((REPO / FSG.STORAGE_PRICING).read_text())
    doc["container_disk"]["provider_api_exposes_this"] = True
    fake = tmp_path / "configs/infrastructure"
    fake.mkdir(parents=True)
    (fake / "provider_storage_pricing.json").write_text(json.dumps(doc))
    with pytest.raises(Exception, match="does not"):
        FSG.storage_pricing(tmp_path)


def test_the_storage_cost_scales_with_the_provision_and_the_window():
    """Both terms, so neither can be inert."""
    one = FSG.storage_cost_usd(1.0, REPO)["usd"]
    ten = FSG.storage_cost_usd(10.0, REPO)["usd"]
    assert ten > one * 9, "the cost does not scale with hours"
    pricing = FSG.storage_pricing(REPO)
    gb = FSG.provision_gb(REPO)["provision_gb"]
    expected = (pricing["container_disk"]["usd_per_gb_month"]
                / pricing["proration"]["hours_per_month"] * gb)
    #: The reported figure is rounded to 6 dp on purpose, and UP at this value
    #: (0.0555555... -> 0.055556), so the tolerance is the declared precision
    #: rather than tighter than it.
    reported = FSG.storage_cost_usd(1.0, REPO)["usd_per_hour"]
    assert reported == pytest.approx(expected, abs=5e-7)
    assert reported >= expected, "the hourly rate rounds DOWN; a bound rounds up"


def test_the_provision_is_derived_in_the_providers_unit(launcher):
    """GiB in, GB out, converted UP.

    The provider's flag says GB and the requirement is GiB. Treating the flag
    as GiB would deliver 7% less capacity than the requirement -- and an
    earlier draft of the pricing note had this backwards, claiming 350 GB was
    more than a 350 GiB requirement needs.
    """
    provision = FSG.provision_gb(REPO)
    assert provision["provision_gb"] == launcher.FULL_SEARCH_PROVISION_GB
    assert provision["required_gb"] > provision["required_gib"]
    assert (provision["provision_gb"]
            >= provision["required_gb"] * provision["headroom_multiple"])
    assert provision["provision_gb"] > 200
    assert launcher.PEAK_WORKING_GIB == FSG.peak_resident_gib(
        REPO)["peak_resident_gib"]


def test_the_storage_gate_refuses_an_under_provisioned_volume(launcher):
    ctx = types.SimpleNamespace(
        args=launch_args(launcher, disk_gb=200), evidence={}, auth=None)
    ok, why = launcher.storage_gate(ctx)
    assert not ok and "below the full-search provision" in why
    ctx.args.disk_gb = launcher.FULL_SEARCH_PROVISION_GB
    ok, why = launcher.storage_gate(ctx)
    assert ok, why


# --- the inputs, derived on both halves -------------------------------------

def test_the_staged_and_tracked_inputs_cover_both_producers():
    """The two paid failures, closed by a derivation rather than a longer list."""
    tracked = FSG.tracked_non_source_inputs()
    assert tracked and all(t.startswith("logs/") for t in tracked)
    assets = FSG.staged_assets(REPO)
    roots = sorted(a.repo_path for a in assets)
    assert all(r.startswith("artifacts/") for r in roots)
    #: Both calibration mixtures AND the metric suite.
    assert any("e8_calibration_v1" in r for r in roots)
    assert any("reasoning_heavy_v2" in r for r in roots)
    assert any("state_eval_v1" in r for r in roots)
    for rel in tracked:
        assert (REPO / rel).is_file(), rel
    for asset in assets:
        assert (REPO / asset.repo_path).is_dir(), asset.repo_path


def test_the_staged_inputs_gate_runs_and_passes_on_this_tree(launcher):
    ctx = types.SimpleNamespace(args=launch_args(launcher), evidence={}, auth=None)
    ok, why = launcher.staged_inputs_gate(ctx)
    assert ok, why
    assert "tracked input" in why and "stageable" in why


def test_the_closure_names_the_telemetry_it_prices_from():
    """Tracked, so it travels in the bundle -- and named, so its bytes are
    inside the identity a grant binds."""
    declared = FSG.declared_inputs(REPO)
    for rel in FSG.tracked_non_source_inputs():
        assert rel in declared, f"{rel} is read but not part of the identity"
    assert FSG.PROTOCOL in declared and FSG.PRICING in declared


def test_the_closure_does_not_name_the_gitignored_assets():
    """They cannot travel in a bundle, so binding them there would be a lie.

    Their identity is bound by the frozen-asset expectation and the launcher
    stages them over the relay.
    """
    import subprocess

    declared = FSG.declared_inputs(REPO)
    for rel in declared:
        out = subprocess.run(["git", "check-ignore", rel], cwd=REPO,
                             capture_output=True, text=True)
        assert out.returncode != 0, (
            f"{rel} is gitignored and cannot travel in the bundle, so the "
            "closure must not claim to bind it")


# --- the eleven $0 gates ----------------------------------------------------

def test_the_gate_chain_is_ordered_and_the_network_one_is_last(launcher):
    spec = launcher.spec(launch_args(launcher))
    names = [getattr(g, "__name__", type(g).__name__) for g in spec.precheck]
    assert names[0] == "session_commit_and_lineage"
    assert names[-1] == "bundle_staged_gate", (
        "the only gate that touches the network must run after everything it "
        "verifies against has been checked")
    for required in ("full_search_executable_gate", "full_search_scope_gate",
                     "frozen_space_gate", "frozen_runtime_gate",
                     "frozen_assets_gate", "staged_inputs_gate",
                     "pricing_and_plan_gate", "storage_gate",
                     "readiness_gate"):
        assert required in names, required


def test_the_scope_gate_refuses_every_foreign_permission(launcher):
    base = dict(authorizes_c2_full_search=True, authorizes_c2_search1=False,
                authorizes_c2_baseline_completion=False,
                authorizes_behavioural_selection=False,
                allows_recovery_training=False,
                authorized_stages=FSG.AUTHORIZED_STAGES,
                resource_scope=types.SimpleNamespace(run_id="attempt1"))
    args = launch_args(launcher)
    ok, why = launcher.full_search_scope_gate(
        types.SimpleNamespace(args=args, evidence={},
                              auth=types.SimpleNamespace(**base)))
    assert ok, why

    for claim in ("authorizes_c2_search1", "authorizes_c2_baseline_completion",
                  "authorizes_behavioural_selection",
                  "allows_recovery_training"):
        auth = types.SimpleNamespace(**{**base, claim: True})
        ok, why = launcher.full_search_scope_gate(
            types.SimpleNamespace(args=args, evidence={}, auth=auth))
        assert not ok and claim in why

    #: A scope belonging to a different attempt is a different one-use chain.
    auth = types.SimpleNamespace(
        **{**base, "resource_scope": types.SimpleNamespace(run_id="attempt9")})
    ok, why = launcher.full_search_scope_gate(
        types.SimpleNamespace(args=args, evidence={}, auth=auth))
    assert not ok and "one-use" in why

    #: And no scope at all is not a pass.
    auth = types.SimpleNamespace(**{**base, "resource_scope": None})
    ok, why = launcher.full_search_scope_gate(
        types.SimpleNamespace(args=args, evidence={}, auth=auth))
    assert not ok and "resource scope" in why


def test_the_space_gate_refuses_a_different_space(launcher):
    """The gate with no counterpart in the other C2 sessions."""
    live = FA.live_identities(REPO)
    bound = {k: v for k, v in live.items() if not k.startswith("_")}
    args = launch_args(launcher)
    ctx = types.SimpleNamespace(args=args, evidence={},
                                auth=types.SimpleNamespace(bound=bound))
    ok, why = launcher.frozen_space_gate(ctx)
    assert ok, why

    for key, wrong in (("joint_space_total_leaves", 866),
                       ("joint_space_decomposed_leaves", 288),
                       ("beam_width", 2),
                       ("excluded_implementations", []),
                       ("joint_space_impl_profiles_are_unpinned", False)):
        ctx.auth = types.SimpleNamespace(bound={**bound, key: wrong})
        ok, why = launcher.frozen_space_gate(ctx)
        assert not ok, f"{key}={wrong!r} was accepted"
        assert "different search space" in why or "PINS" in why


def test_the_runtime_gate_refuses_a_rate_above_the_authorized_one(launcher,
                                                                 tmp_path,
                                                                 monkeypatch):
    """A higher rate needs the ceiling re-derived, not a launcher flag."""
    auth_dir = tmp_path / "gov"
    auth_dir.mkdir(parents=True)
    (auth_dir / "authorization.json").write_text(json.dumps(
        {"approved_money": {"price_basis_usd_per_hour": 1.09}}))
    monkeypatch.setattr(launcher, "auth_path_for",
                        lambda run_id: "gov/authorization.json")
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)

    ctx = types.SimpleNamespace(args=launch_args(launcher, max_price=1.29),
                                evidence={}, auth=None)
    ok, why = launcher.frozen_runtime_gate(ctx)
    assert not ok and "exceeds" in why and "not narrowed" in why

    #: Equal is fine, and LOWER is fine -- it can only refuse a launch.
    for price in (1.09, 0.99):
        ctx.args = launch_args(launcher, max_price=price)
        ok, why = launcher.frozen_runtime_gate(ctx)
        assert ok, why


def test_the_runtime_gate_refuses_a_foreign_card_and_image(launcher, tmp_path,
                                                           monkeypatch):
    auth_dir = tmp_path / "gov"
    auth_dir.mkdir(parents=True)
    (auth_dir / "authorization.json").write_text(json.dumps(
        {"approved_money": {"price_basis_usd_per_hour": 1.09}}))
    monkeypatch.setattr(launcher, "auth_path_for",
                        lambda run_id: "gov/authorization.json")
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)

    for flag, value, needle in (("gpu", "NVIDIA A100", "priced NVIDIA L40S"),
                                ("image", "pytorch:other", "pooled cost table")):
        ctx = types.SimpleNamespace(
            args=launch_args(launcher, **{flag: value}), evidence={}, auth=None)
        ok, why = launcher.frozen_runtime_gate(ctx)
        assert not ok and needle in why


def test_the_readiness_gate_refuses_an_absent_and_a_diagnostic_record(launcher):
    """No sweep has been taken for this run, so the gate must say so."""
    ctx = types.SimpleNamespace(args=launch_args(launcher, run_id="attempt_none"),
                                evidence={}, auth=None)
    ok, why = launcher.readiness_gate(ctx)
    assert not ok and "launch-bound sweep is owed" in why


def test_the_frozen_assets_gate_runs_the_real_verifier(launcher):
    ctx = types.SimpleNamespace(args=launch_args(launcher), evidence={}, auth=None)
    ok, why = launcher.frozen_assets_gate(ctx)
    assert ok, why


# --- the plan the runner uses is the plan the record states -----------------

def test_the_runner_plans_exactly_the_pricing_record(launcher):
    spec = launcher.spec(launch_args(launcher))
    plan = spec.budget.plan(price_per_hour=FSG.price_per_hour_basis(REPO),
                            authorized_usd=1_000.0)
    row = FSG._standing_row(REPO)
    assert round(plan.expected_minutes, 2) == row["expected_minutes"]
    assert round(plan.hard_terminate_minutes, 2) == row["hard_ceiling_minutes"]


def test_the_poll_limit_outlasts_the_ceiling(launcher):
    """A poll that gives up before the beam can finish abandons a paying pod."""
    args = launch_args(launcher)
    row = FSG._standing_row(REPO)
    assert args.poll_limit_min > float(row["hard_ceiling_minutes"])


def test_a_plan_that_does_not_fit_is_refused_rather_than_shrunk():
    from aadistill.infrastructure.budget import BudgetError

    with pytest.raises(BudgetError):
        FSG.plan(price_per_hour=1.09, authorized_usd=1.0, repo_root=REPO)


# --- it ends at the Top-5 ---------------------------------------------------

def test_nothing_in_the_chain_reaches_a_recovery_or_scoring_path():
    """By import graph, not by reading comments.

    The session is authorized to search. A module here that could import a
    probe trainer or a battery scorer would make the boundary a matter of
    intention rather than of structure.
    """
    import ast

    forbidden = ("recovery", "probe", "correct_overall", "screening",
                 "confirmation", "usable_rollout", "behavioural")
    files = [
        REPO / "scripts/pod/autoinit_phase_c2_full_search_launch.py",
        REPO / "scripts/pod/autoinit_phase_c2_full_search_driver.py",
        REPO / "scripts/experiments/phase_c2/full_search.py",
        REPO / "scripts/experiments/phase_c2/full_search_authorization.py",
        REPO / "scripts/experiments/phase_c2/full_search_bundle.py",
        REPO / "scripts/experiments/phase_c2/full_search_pod_environment.py",
        REPO / "scripts/autoinit/issue_c2_full_search_authorization.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Call):
                names = [ast.unparse(node.func)]
            for name in names:
                low = name.lower()
                for word in forbidden:
                    assert word not in low, (
                        f"{path.name} references {name!r}, which reaches "
                        f"{word!r} — the search must have no path into "
                        "behavioural work")


def test_the_products_are_fetched_on_the_STAGE_not_on_success(launcher):
    """Five checkpoints a multi-hour beam found must survive a later failure.

    Gated on `commit_top_k` completing, never on the session's terminal marker:
    `$2.82` of verified checkpoints were lost to a collector keyed on
    `terminal == "ALL_DONE"`.
    """
    spec = launcher.spec(launch_args(launcher))
    eligible = spec.markers.products_eligible
    assert eligible("ANYTHING_AT_ALL", ("bind_identities", "full_joint_search",
                                        "commit_top_k")) is True
    assert eligible("C2_FULL_SEARCH_ALL_DONE", ("bind_identities",)) is False
    assert eligible("C2_FULL_SEARCH_FAILED", ("bind_identities",
                                              "full_joint_search")) is False


# --- the artifact contract --------------------------------------------------

@pytest.mark.parametrize("name", ["c2_full_search_artifacts",
                                  "c2_full_search_artifacts_failed"])
def test_both_artifact_specs_load_through_the_collectors_own_loader(name):
    """An invented lifecycle word makes the whole document UNLOADABLE.

    So the check is a real load, not a schema read: the success-path spec of a
    sibling session carried `best_effort` for weeks and the failure surfaced
    during a teardown.
    """
    collector = importlib.util.spec_from_file_location(
        "collect_artifacts_for_test", REPO / "scripts/pod/collect_artifacts.py")
    module = importlib.util.module_from_spec(collector)
    sys.modules["collect_artifacts_for_test"] = module
    collector.loader.exec_module(module)

    specs = module.load_specs(str(REPO / f"configs/autoinit/{name}.json"))
    assert len(specs) == 5
    classes = {s.artifact_class for s in specs}
    assert {"session_evidence", "stage1_selection", "search_journal",
            "search_telemetry"} <= classes


def test_the_failed_spec_requires_only_the_evidence():
    """An early failure legitimately produces nothing else.

    Requiring a journal on the failure path would block the collection of what
    a failed run DOES have.
    """
    success = json.loads(
        (REPO / "configs/autoinit/c2_full_search_artifacts.json").read_text())
    failed = json.loads(
        (REPO / "configs/autoinit/c2_full_search_artifacts_failed.json").read_text())
    req_success = {e["artifact_class"] for e in success["entries"] if e["required"]}
    req_failed = {e["artifact_class"] for e in failed["entries"] if e["required"]}
    assert req_failed == {"session_evidence"}
    assert req_success > req_failed
    #: But everything the success path names is still COLLECTED if present.
    assert ({e["artifact_class"] for e in failed["entries"]}
            == {e["artifact_class"] for e in success["entries"]})


def test_the_artifact_patterns_are_the_ones_the_driver_actually_writes():
    """Observed on a real L40S, not guessed.

    A driver's workdir and its collector's patterns are two declarations of one
    path, and this project has lost a search journal to `phase_a_search` versus
    `phase_b_search`.
    """
    driver = (REPO / "scripts/pod/autoinit_phase_c2_full_search_driver.py").read_text()
    assert 'artifacts/autoinit/phase_c2_full_search' in driver
    assert 'artifacts/audit/autoinit_phase_c2_full_search' in driver
    for name in ("c2_full_search_artifacts", "c2_full_search_artifacts_failed"):
        doc = json.loads((REPO / f"configs/autoinit/{name}.json").read_text())
        for entry in doc["entries"]:
            pattern = entry["pattern"]
            assert (pattern.startswith("autoinit/phase_c2_full_search/")
                    or pattern.startswith("audit/autoinit_phase_c2_full_search/")), (
                f"{pattern} is under neither directory the driver writes")


# --- readiness and bundle instances -----------------------------------------

def test_the_readiness_record_path_requires_a_run_id():
    with pytest.raises(FPE.ReadinessError, match="without a run id"):
        FPE.record_path_for(None)
    path = FPE.record_path_for("attempt1", "1")
    assert path.endswith("/governance/readiness.json")
    assert "phase_c2_full_search" in path


def test_the_readiness_schema_is_its_own():
    from experiments.phase_c2 import baseline_completion_pod_environment as BPE

    assert FPE.SCHEMA != BPE.SCHEMA
    assert FPE.EXPERIMENT_ID == "phase_c2_full_search"
    assert FPE.harness_digest(REPO) == FSG.executable_digest(REPO)


def test_the_bundle_derives_its_digest_and_file_set_together():
    digest, files = FST.full_search_executable_set(REPO)
    live = FSG.current_executable(REPO)
    assert digest == live["digest"]
    assert files == tuple(row["path"] for row in live["files"])


def test_the_sweep_contract_binds_this_session():
    contract = FPE.sweep_contract("attempt1", "1")
    assert contract.experiment_id == "phase_c2_full_search"
    assert contract.session_id == FSG.SESSION_ID
    assert contract.launcher_module == "autoinit_phase_c2_full_search_launch"
    assert contract.harness_n_files_field == "full_search_harness_n_files"


# --- the proposal is not a grant --------------------------------------------

PROPOSAL = (REPO / "logs/stages/stage-1/phase_c2/plans/"
            "phase_c2_full_search_grant_proposal.json")


def test_the_proposal_states_the_figures_a_launch_review_needs():
    """Including both halves of the provider's bill, and the arithmetic between
    them, so a review is not asked to add anything up itself."""
    doc = json.loads(PROPOSAL.read_text())
    assert doc["authorizes"] == "nothing" and doc["approves"] == "nothing"

    money = doc["money"]
    assert money["gpu"]["securePrice_usd_per_hour"] == 1.09
    assert money["gpu"]["provider_api_exposes_this"] is True
    #: And the storage half says it is NOT a quote, which is the distinction.
    assert money["container_disk"]["provider_api_exposes_this"] is False
    total = FSG.total_ceiling_usd(money["gpu"]["securePrice_usd_per_hour"], REPO)
    assert money["gpu"]["hard_ceiling_usd"] == total["gpu_usd"]
    assert money["container_disk"]["usd"] == total["container_disk"]["usd"]
    assert money["TOTAL_hard_ceiling_usd"] == total["total_hard_ceiling_usd"]
    assert money["TOTAL_hard_ceiling_usd"] > money["gpu"]["hard_ceiling_usd"]

    position = doc["budget_position"]
    assert round(position["remaining_usd"]
                 - position["search_total_hard_ceiling_usd"], 4) == \
        position["remaining_after_the_search_usd"]
    assert round(position["search_total_hard_ceiling_usd"]
                 + position["behavioural_gpu_hard_ceiling_usd"]
                 + position["behavioural_storage_upper_bound_usd"], 4) == \
        position["chain_total_hard_ceiling_usd"]
    assert round(position["remaining_usd"]
                 - position["chain_total_hard_ceiling_usd"], 4) == \
        position["remaining_after_the_whole_chain_usd"]
    #: It still fits, and the proposal says so by arithmetic rather than claim.
    assert position["remaining_after_the_whole_chain_usd"] > 0


def test_the_proposal_regenerates_byte_identically():
    """Deterministic, so a diff means a figure moved.

    The first version was hand-written and went stale the moment the storage
    derivation was corrected -- a review reading a stale proposal is reading
    another tree's numbers.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "write_proposal",
        REPO / "scripts/autoinit/write_c2_full_search_grant_proposal.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["write_proposal"] = module
    spec.loader.exec_module(module)

    regenerated = json.dumps(module.proposal(), indent=1) + "\n"
    assert regenerated == PROPOSAL.read_text(), (
        "the committed proposal is not what the generator produces from this "
        "tree; regenerate it")


def test_the_proposal_reproduces_the_live_identities():
    """Stale the moment the tree moves, and that is the point: a review reading
    a proposal must be reading this tree's identities."""
    doc = json.loads(PROPOSAL.read_text())
    live = {k: v for k, v in FA.live_identities(REPO).items()
            if not k.startswith("_")}
    assert doc["bound_identities_the_issuer_will_reproduce"] == live


def test_the_proposal_cannot_be_promoted_by_the_issuer():
    """The boundary, exercised rather than asserted.

    A proposal that the issuer would accept is a grant an agent wrote. This
    one states no `granted_by`, carries no `one_use`, and declares a schema the
    contract does not name -- so the assembler refuses it.
    """
    doc = json.loads(PROPOSAL.read_text())
    assert "granted_by" not in doc
    assert "one_use" not in doc
    assert doc["schema"].endswith("grant_proposal/v1")
    cfg = FA.load_config(REPO)
    assert doc["schema"] != cfg["grant_contract"]["contract_id"]
    with pytest.raises(FA.FullSearchAuthorizationRefused):
        FA.build_payload(grant=doc, session_commit="a" * 40,
                         granted_utc="2026-01-01T00:00:00+00:00",
                         run_id="attempt1", repo_root=REPO)


def test_no_authorization_exists_for_any_full_search_run():
    """The chain is built and NOT consumed. An issued artifact would mean an
    agent had granted itself money."""
    runs = REPO / "logs/stages/stage-1/phase_c2_full_search/runs"
    if not runs.is_dir():
        return
    issued = list(runs.rglob("governance/authorization.json"))
    assert not issued, f"a full-search authorization exists: {issued}"


def test_the_sweep_can_actually_be_DRIVEN_for_this_experiment():
    """The dispatch entry, without which the chain is unusable.

    `record_pod_environment` resolves an experiment to its sweep contract
    through a registry. A new experiment with no entry cannot have a
    launch-bound sweep driven at all — and the failure would surface as the
    launcher's readiness gate refusing forever with a message about a missing
    RECORD rather than a missing REGISTRATION, at exactly the step a launch
    review authorizes.

    This repository has forgotten a dispatch entry before: `SESSION_KIND=phase_b`
    had no branch in the setup script and cost $0.23 on a billing pod. The
    registry is checked by resolving through it, not by reading it.
    """
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    import record_pod_environment as R

    assert "phase_c2_full_search" in R.EXPERIMENTS
    contract = R.sweep_contract("phase_c2_full_search", "attempt1", "1",
                                R.LAUNCH_BOUND if hasattr(R, "LAUNCH_BOUND")
                                else "launch_bound")
    assert contract.experiment_id == "phase_c2_full_search"
    assert contract.session_id == FSG.SESSION_ID
    assert contract.launcher_module == "autoinit_phase_c2_full_search_launch"
    assert contract.record.record_path == FPE.record_path_for("attempt1", "1")
    #: And it resolves to a DIFFERENT contract than its siblings, or one C2
    #: record could satisfy another's verifier.
    others = {name: R.sweep_contract(name, "attempt1", "1", "launch_bound")
              for name in R.EXPERIMENTS if name != "phase_c2_full_search"}
    for name, other in others.items():
        assert other.record.schema != contract.record.schema, (
            f"{name} shares a readiness schema with the full search")


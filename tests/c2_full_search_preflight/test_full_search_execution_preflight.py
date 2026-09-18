"""The only tests a paid full-joint-re-search pod runs. Positive selection.

The pod's blocking gate is `pytest tests/ $SESSION_TEST_IGNORES`, so a session's
selection is a COMPLEMENT the launcher derives from the tree. This directory is
the full search's, and it exists for the reason the baseline completion's does:
`tests/c2_preflight` asserts that every path SEARCH-1 stages is present,
including the canonical 0.6B control Search-1 injects as its measured baseline.
This session has no control and deliberately does not stage one, so running
Search-1's preflight under this session's staged view fails two tests that are
entirely correct about Search-1.

That is not a prediction. The full search's launch-bound sweep was run against
Search-1's selection on 2026-09-19 and failed exactly there:
`test_every_declared_staged_path_is_present` on
`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/chat_template.jinja`, and
`test_the_canonical_control_checkpoint_is_readable` on the same directory's
`model.safetensors`. `full_search_pod_environment.py` had reused Search-1's
selection with a docstring naming this exact hazard -- "a selection that
asserted staged paths this session does not stage is exactly how a correct test
fails a correct session" -- and the hazard arrived.

Narrowing Search-1's gate to fit this session was the other option and the
wrong one: it is a consumed experiment's satisfiable gate, and a guard narrowed
to fit the session under test stops guarding the session it was written for.

So every test here earns its place by naming a failure that would otherwise
appear only AFTER setup has been paid for -- and this session's ceiling is
`$34.8742`, the largest this project has authorized. What the session does:
bind identities, run ONE beam over the derived 578-leaf joint space at the
standing width, and commit a Top-5. It trains nothing, scores no battery,
produces no `correct_overall` and cannot name an incumbent.

**Zero skips, by construction**, enforced by the last test here. Nothing needs
CUDA: the shared CPU-test contract sets `CUDA_VISIBLE_DEVICES=` and neutralises
`HOME` and the HF caches, so a GPU-only assertion could only appear as a skip,
and a skip in this directory is a hole a reader cannot see.

**Only staged paths are touched.** The simulator models the pod's view by hiding
repository artifacts the session does not stage, so a test reading an unstaged
path would pass on the dev box and fail here. The paths asserted below are
exactly what this session's `SetupManifest` declares.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in ("src", "scripts", "scripts/autoinit", "scripts/pod", "tests/pod"):
    if str(REPO / _root) not in sys.path:
        sys.path.insert(0, str(REPO / _root))


@pytest.fixture(scope="module")
def registered():
    """The registration the driver performs at stage A, performed here.

    `attention.activation_importance_v1` is deliberately not a shipped default,
    so a session that forgot this call would derive an EMPTY space and price a
    search of nothing. This project has already lost a pod to an unregistered
    registry.
    """
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()


@pytest.fixture(scope="module")
def search(registered):
    """Launcher, session, plan and the emitted driver command."""
    from session_specs import load_session_launcher, session_args

    from aadistill.runtime.staging_contract import derive_contract
    from experiments.phase_c2 import full_search as FSG

    launcher = load_session_launcher("autoinit_phase_c2_full_search_launch")
    args = session_args(launcher)
    spec = launcher.spec(args).validate()

    #: The production call, not a transcription of it.
    plan = spec.budget.plan(price_per_hour=FSG.price_per_hour_basis(REPO),
                            authorized_usd=FSG.total_ceiling_usd(
                                FSG.price_per_hour_basis(REPO),
                                REPO)["total_hard_ceiling_usd"])
    #: `driver_command` reads `ctx.auth`, so the fixture supplies the TYPE's
    #: own permission answers rather than a stub with invented ones. Before
    #: issuance there is no artifact -- the sweep runs on the grant-containing
    #: tree by design -- and a fixture that answered differently from the class
    #: would be testing itself.
    from experiments.phase_c2.full_search import FullSearchAuthorization

    ctx = SimpleNamespace(
        args=args, image_digest="preflight", auth=FullSearchAuthorization,
        price=FSG.price_per_hour_basis(REPO), spent_usd=0.0)
    try:
        command = spec.driver_command(ctx, plan)
    except Exception as exc:                                     # noqa: BLE001
        #: Recorded rather than swallowed: a command that cannot be built
        #: without the issued artifact is a fact about this session, and the
        #: tests that read it say so instead of asserting on a None.
        command = f"UNAVAILABLE-BEFORE-ISSUANCE: {type(exc).__name__}: {exc}"

    return SimpleNamespace(
        launcher=launcher, args=args, spec=spec, plan=plan, command=command,
        contract=derive_contract(spec.setup, session_id=FSG.SESSION_ID))


# --- A. the staged inputs this driver consumes ------------------------------
#
# Phase-A attempt 5 died at $0.6426 on a calibration asset the session assumed
# was staged and was not.


def test_every_declared_staged_path_is_present(search):
    from aadistill.runtime.staging_contract import staged_files

    declared = sorted(staged_files(search.contract, REPO))
    assert declared, "the full-search staging contract declares no staged files"
    for rel in declared:
        assert (REPO / rel).exists(), f"{rel} is declared staged and is absent"


def test_the_canonical_control_is_NOT_staged(search):
    """This session has no control, and staging one would be a different search.

    Asserted positively because the absence is a DECISION, and because it is
    the decision that made Search-1's selection unusable here. Search-1 injects
    the canonical 0.6B init as its measured baseline; the full joint re-search
    searches from the teacher and compares candidates to each other, so a
    control it never reads would be 1.2 GiB shipped to a pod for nothing.
    """
    from aadistill.runtime.staging_contract import staged_files
    from phase_a_frozen import CANONICAL_INIT

    declared = sorted(staged_files(search.contract, REPO))
    leaked = [rel for rel in declared if CANONICAL_INIT in rel]
    assert not leaked, (
        f"the full search stages {leaked}, which nothing in its path reads")
    #: and the session's own asset list says the same thing from the other side
    from experiments.phase_c2 import full_search as FSG

    assert not any(CANONICAL_INIT in a.repo_path
                   for a in FSG.staged_assets(REPO))


def test_the_three_local_assets_are_exactly_what_the_search_resolves(search):
    """Two calibration mixtures and the frozen metric suite. Nothing else.

    DERIVED from the code that resolves them rather than listed: the CUDA
    validation's second subrun died at `$0.0252` because a hand-written ship
    list named the telemetry and not the profiles' `items_path` files.
    """
    from experiments.phase_c2 import full_search as FSG

    staged = sorted(a.repo_path for a in FSG.staged_assets(REPO))
    assert staged == ["artifacts/stage1/e8_calibration_v1",
                      "artifacts/stage1/reasoning_heavy_v2",
                      "artifacts/stage1/state_eval_v1"], staged
    for rel in staged:
        assert (REPO / rel).is_dir(), f"{rel} is staged and absent"


def test_both_mixtures_resolve_against_their_content_hashes(registered):
    """`resolve()` re-hashes the items file, so a drifted or half-staged mixture
    refuses here rather than silently changing every operator's statistics."""
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2.full_search_space import PROFILE_IDS

    for qualified_id in PROFILE_IDS:
        items = get_profile(qualified_id).resolve(REPO)
        assert items, f"{qualified_id} resolved to no items"
        assert all("ids" in i and "item_id" in i for i in items)


def test_the_state_eval_suite_loads_from_its_staged_directory():
    """The one metric the whole search ranks on, and the suite the optimized
    evaluator was certified against. Without it there is no ranking."""
    from load_state_eval import load as load_suite

    suite, items, manifest = load_suite(REPO / "artifacts/stage1/state_eval_v1")
    assert suite.suite_hash and suite.domains
    assert len(items) == 80, len(items)
    assert sum(int(i.input_ids.shape[1]) - 1 for i in items) == 74022
    assert manifest["suite_id"] == suite.suite_id


def test_the_declared_non_source_inputs_are_present():
    """Two paid subruns of the CUDA validation died one per unshipped producer,
    so the inputs are derived from the code that reads them and checked here."""
    from experiments.phase_c2 import full_search as FSG

    declared = list(FSG.declared_inputs())
    assert declared, "the session declares no inputs"
    for rel in declared:
        assert (REPO / rel).exists(), f"{rel} is a declared input and is absent"


# --- B. the frozen space this beam searches ---------------------------------


def test_the_space_is_the_derived_578_leaf_joint_space(registered):
    """Enumerated from the registry, never a remembered number."""
    from experiments.phase_c2 import full_search_space as FS

    report = FS.size_report(REPO)["full_joint"]
    assert report["total_leaves"] == 578
    assert report["decomposed_leaves"] == 576
    assert report["total_leaves"] > report["decomposed_leaves"], (
        "a composite operator reaches the target in one step, so the two "
        "counts must differ")


def test_the_operator_library_and_its_exclusion_are_the_approved_ones(registered):
    from experiments.phase_c2 import full_search_space as FS

    space = FS.full_joint_space(REPO)
    impls = {impl.impl_id for impl, _profiles in space.options(frozenset())}
    assert "attention.activation_importance_v1" in impls
    assert "attention.weight_proxy_v0" not in impls, (
        "the excluded operator is in the space; C1 is the isolation experiment "
        "between the two and it has a completed GO verdict")


def test_the_beam_is_the_standing_width_and_warmup(registered):
    from aadistill.initialization.planning.ranking import SCHEDULE_V1

    assert SCHEDULE_V1.width == 6
    assert SCHEDULE_V1.warmup_levels == 1
    assert SCHEDULE_V1.schedule_id == "beam.delayed_prune"


def test_the_ranking_policy_is_the_frozen_one():
    """Epsilon, objectives and tie-break, read from the policy the search runs.

    Review directed that the PARETO epsilon and the ranking policy are NOT
    changed on account of the certification's synthetic boundary cases, so this
    is where that instruction is enforced against the tree a pod runs.
    """
    from aadistill.initialization.planning.ranking import PARETO_V1

    assert [o.key for o in PARETO_V1.objectives] == [
        "state.teacher_kl.equal_domain_mean",
        "state.teacher_kl.worst_domain",
        "state.critical_token_kl"]
    assert set(PARETO_V1.epsilon.values()) == {1e-4}
    assert PARETO_V1.diversity_key == "parent_path"


def test_the_identities_a_grant_binds_are_derivable(registered):
    """Every identity the issuer re-derives must be derivable ON THE POD too.

    A grant binds sixteen of them. If one could not be re-derived here, the
    session would be running against an approval whose basis it cannot check.
    """
    from experiments.phase_c2 import full_search_authorization as FA

    live = {k: v for k, v in FA.live_identities(REPO).items()
            if not k.startswith("_")}
    assert len(live) == 16, sorted(live)
    assert live["joint_space_total_leaves"] == 578
    assert live["beam_width"] == 6
    assert all(v not in (None, "", []) for v in live.values())


# --- C. the budget the driver is handed -------------------------------------


def test_the_plan_reproduces_the_priced_total_ceiling(search):
    """The minutes are the plan and the dollars are those minutes at the
    EFFECTIVE rate -- GPU securePrice plus the container disk's hourly share."""
    from experiments.phase_c2 import full_search as FSG

    row = FSG._standing_row(REPO)
    assert search.plan.hard_terminate_minutes <= float(row["hard_ceiling_minutes"])
    total = FSG.total_ceiling_usd(FSG.price_per_hour_basis(REPO), REPO)
    assert total["total_hard_ceiling_usd"] > total["gpu_usd"], (
        "the total does not exceed the GPU-only figure, so separately billed "
        "container disk has fallen out of the ceiling")
    assert total["container_disk"]["provisioned_gb"] == 400


def test_the_provision_covers_the_derived_peak():
    """400 GB for a derived peak, with the environment counted in."""
    from experiments.phase_c2 import full_search as FSG

    provision = FSG.provision_gb(REPO)
    assert provision["required_gib"] > FSG.peak_resident_gib(REPO)["peak_resident_gib"]
    assert provision["provision_gb"] >= provision["required_gb"]
    assert provision["provision_gb"] == 400


def test_the_driver_command_names_only_the_authorized_stages(search):
    """Read from the emitted command where there is one, and say so where not.

    `driver_command` needs the ISSUED authorization -- it reads `hard_cap_usd`
    off it -- and the launch-bound sweep runs BEFORE issuance by design. So
    there are two machines and two branches, and NEITHER is a skip: pre-issuance
    the terminus is asserted on `AUTHORIZED_STAGES`, where it is defined; on the
    pod the real command is read. A single branch that quietly passed on an
    error string would be the vacuous half of this test.
    """
    from experiments.phase_c2 import full_search as FSG

    #: True on both machines: the stage list IS the terminus.
    assert FSG.AUTHORIZED_STAGES == ("bind_identities", "full_joint_search",
                                     "commit_top_k")

    command = (" ".join(search.command)
               if isinstance(search.command, (list, tuple))
               else str(search.command))
    if command.startswith("UNAVAILABLE-BEFORE-ISSUANCE"):
        #: and the reason must be the absence of an issued authorization,
        #: not some other error wearing the same marker
        assert "hard_cap_usd" in command or "Authorization" in command, command
        return
    assert "full_joint_search" in command and "commit_top_k" in command, command
    for forbidden in ("screening", "confirmation", "behavioural", "recovery"):
        assert forbidden not in command, (
            f"the driver command names {forbidden!r}; this session commits a "
            "Top-5 and stops")


# --- D. what this session cannot do ----------------------------------------


def test_no_behavioural_stage_is_reachable_from_this_session():
    """By import graph, not by instruction."""
    import ast

    for rel in ("scripts/pod/autoinit_phase_c2_full_search_driver.py",
                "scripts/pod/autoinit_phase_c2_full_search_launch.py"):
        tree = ast.parse((REPO / rel).read_text())
        named = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                named.add(node.module.split(".")[-1])
                named |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                named |= {alias.name.split(".")[-1] for alias in node.names}
        leaked = sorted({"recovery", "screening", "confirmation",
                         "behavioural_selection", "run_recovery"} & named)
        assert not leaked, f"{rel} names {leaked}"


def test_the_authorization_on_this_pod_cannot_authorize_anything_else():
    """The artifact the setup script loaded, read again here.

    Keyed on the FILE, not on the environment variable: the launch-bound sweep
    runs on the grant-containing tree -- BEFORE the authorization is issued, by
    design -- and the simulator exports the session environment, so
    `SESSION_AUTH_PATH` is named there while the artifact cannot yet exist.
    Neither branch is a skip: pre-issuance the property is asserted on the TYPE,
    where it is defined and cannot be edited away; on the pod the artifact is
    read.
    """
    import os

    from experiments.phase_c2.full_search import FullSearchAuthorization

    path = os.environ.get("SESSION_AUTH_PATH")
    resolved = (REPO / path) if path else None
    if resolved is None or not resolved.is_file():
        assert FullSearchAuthorization.authorizes_c2_full_search.fget(None) is True
        assert FullSearchAuthorization.authorizes_c2_search1.fget(None) is False
        assert FullSearchAuthorization.authorizes_c2_baseline_completion.fget(
            None) is False
        assert FullSearchAuthorization.authorizes_behavioural_selection.fget(
            None) is False
        return
    auth = FullSearchAuthorization.load(resolved)
    assert auth.authorizes_c2_full_search is True
    assert auth.authorizes_c2_search1 is False
    assert auth.authorizes_c2_baseline_completion is False
    assert auth.authorizes_behavioural_selection is False
    assert auth.allows_recovery_training is False
    assert auth.automatic_followon_start is False


def test_the_driver_writes_only_into_its_own_workdir():
    """Search-1's evidence is frozen and this session must not touch it.

    Asserted on PATHS, not on filenames. The first version of this test
    forbade the string `stage1_selection.json` and failed immediately: this
    driver writes its OWN selection file under its OWN workdir, and it happens
    to share a name with Search-1's. A filename is not a location, and a test
    that confuses the two refuses a correct session.
    """
    text = (REPO / "scripts/pod/"
            "autoinit_phase_c2_full_search_driver.py").read_text()
    for frozen_run in ("phase_c2/runs/attempt4",
                       "phase_c2_baseline_completion/runs",
                       "phase_b/runs/attempt5"):
        assert frozen_run not in text, (
            f"the driver names {frozen_run}, a frozen run directory. Its "
            "telemetry is read by the COST MODEL at planning time, not by the "
            "driver on a pod.")


def test_nothing_in_this_directory_is_conditional():
    """A skip here is a hole a reader cannot see, and this session's readiness
    contract declares zero expected skips for this selection."""
    me = Path(__file__)
    for path in sorted(me.parent.rglob("*.py")):
        #: Excluded because it NAMES the markers it is looking for.
        if path == me:
            continue
        text = path.read_text()
        for marker in ("pytest.skip", "pytest.mark.skipif", "pytest.mark.skip",
                       "pytest.importorskip", "pytest.xfail"):
            assert marker not in text, f"{path.name} uses {marker}"

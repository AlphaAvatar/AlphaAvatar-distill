"""The only tests a paid Phase-C2 Search-1 pod runs. Positive selection.

The pod's blocking gate is `pytest tests/ $SESSION_TEST_IGNORES`, so a session's
selection is a COMPLEMENT, and `autoinit_phase_c2_launch.TEST_IGNORES` now
derives that complement from the tree. Until 2026-09-15 C2 declared two file
ignores and therefore sent the pod the whole repository: a $0 probe measured
3976 passed, 65 skipped, 907 s of what would have been billing L40S time for
checks with nothing to do with the search. AGENTS.md P8.2.1 says a paid
experiment machine runs only what that experiment needs.

So every test here earns its place by naming a failure that would otherwise
appear only AFTER setup has been paid for. What C2 actually does is narrow:
register two operators and two mixtures, walk a beam over four implementations,
conditionally rebuild one baseline, and write one comparison record. It trains
nothing, evaluates no battery and exports no checkpoint — so there is no probe,
recipe, scoring, renderer or battery surface here, and copying C1's preflight
would have added exactly those.

Several of these checks have a thorough dev-side counterpart in
`tests/pod/test_phase_c2_session.py`, `tests/pod/test_phase_c2_reserve_partition.py`
and `tests/autoinit/test_phase_c2_*.py`, which the pod does not run. This is the
pod-side version: the same seam, asked cheaply, in the environment that pays.

**Zero skips, by construction**, enforced by the last test in this file. Nothing
here needs CUDA — the shared CPU-test contract sets `CUDA_VISIBLE_DEVICES=` and
neutralizes `HOME` and the HF caches — so a GPU-only assertion could only appear
as a skip, and a skip in this directory is a hole a reader cannot see. GPU
visibility is proved by `ROPE_OK` and the driver's own device handling, outside
this scope.

**Only staged paths are touched.** The pod simulator models the pod's view by
hiding repository artifacts the session does not stage, so a test reading an
unstaged path would pass on the dev box and fail under the simulation. The paths
asserted below are exactly what the session's `SetupManifest` declares.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in ("src", "scripts", "scripts/autoinit", "scripts/pod", "tests/pod"):
    if str(REPO / _root) not in sys.path:
        sys.path.insert(0, str(REPO / _root))


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


# --- the real launcher, the real parser, the real plan, built once ----------
#
# These live in this MODULE and not in a `conftest.py`, deliberately. Under
# pytest's prepend import mode every collected test module's directory goes on
# `sys.path`, so a `tests/c2_preflight/conftest.py` becomes importable as the
# top-level name `conftest` — and `tests/pod/test_phase_a_search_profile_seam.py`
# imports `make_profile` from `conftest` by bare name. Adding this directory's
# conftest made the whole-suite collection resolve that name here instead of in
# `tests/autoinit/conftest.py`, which turned a latent fragility into a hard
# collection error for the entire run. A fixture file is not worth that.
#
# Nothing below constructs a stand-in. Device-canary attempt 1 died at $0.0603
# on an attribute a hand-written namespace had and the real parser did not, so
# the namespace comes from `build_parser()` and the plan from `BudgetSpec.plan`,
# which is the same call `session_runner.make_plan` makes on the pod.

@pytest.fixture(scope="module")
def registered():
    """The registration the driver performs at stage A, performed here.

    Whether this is *sufficient* on an empty registry is a separate question an
    in-process fixture cannot answer — a sibling test may already have filled
    the registry — so it is asked in a subprocess by
    `test_the_production_registration_is_sufficient_on_an_empty_registry`.
    """
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()
    return True


@pytest.fixture(scope="module")
def c2(registered):
    """Launcher, driver, session, plan, emitted command and parsed command."""
    import autoinit_phase_c2_driver as driver
    from session_specs import load_session_launcher, session_args

    from aadistill.runtime.staging_contract import derive_contract
    from experiments.phase_c2.session import (
        c2_hard_ceiling_usd, c2_price_per_hour_usd,
    )

    launcher = load_session_launcher("autoinit_phase_c2_launch")
    args = session_args(launcher)
    spec = launcher.spec(args).validate()

    #: The production call, not a transcription of it.
    plan = spec.budget.plan(price_per_hour=c2_price_per_hour_usd(REPO),
                            authorized_usd=c2_hard_ceiling_usd(REPO))
    ctx = SimpleNamespace(args=args, image_digest="preflight",
                          price=c2_price_per_hour_usd(REPO), spent_usd=0.0)
    command = spec.driver_command(ctx, plan)

    return SimpleNamespace(
        launcher=launcher, driver=driver, args=args, spec=spec, plan=plan,
        command=command,
        parsed=driver.build_parser().parse_args(command.split()[2:]),
        contract=derive_contract(spec.setup, session_id=spec.session_id))


# --- A. the executable seams load ------------------------------------------
#
# Four paid pods in this project have died inside lines no test had executed.
# An import error in the driver appears only after setup, the teacher fetch and
# the asset staging have all been paid for.

def test_the_launcher_and_driver_modules_load(c2):
    assert callable(c2.launcher.spec) and callable(c2.launcher.build_parser)
    assert callable(c2.driver.build_parser)
    assert c2.driver.PhaseC2Driver is not None


def test_the_experiment_layer_loads():
    """Every module the driver imports at stage time, imported now instead."""
    from experiments.phase_c2 import baseline, comparison, search_space, session

    assert session.SCHEMA.startswith("aadistill.autoinit.c2_authorization/")
    assert comparison.FILENAME == "c2_baseline_comparison.json"
    assert callable(baseline.frozen_baseline_spec)
    assert callable(search_space.c2_search1_space)


def test_the_search_seam_accepts_the_conditional_hook():
    """The driver passes `conditional_candidates`, `allowed_impls` and
    `impl_profiles`. An older seam would accept the call and silently ignore
    them: the space would be the whole registry and B would never be resolved."""
    import inspect

    import phase_a_search

    params = inspect.signature(phase_a_search.run_phase_a_search).parameters
    for needed in ("conditional_candidates", "allowed_impls", "impl_profiles",
                   "search_minutes", "run_id"):
        assert needed in params, needed
    assert params["conditional_candidates"].default is None


# --- B. registration, which the search refuses to run without ---------------

def test_the_four_search_implementations_resolve(registered):
    """`_allowed_impl_ids` validates `allowed_impls` against the registry, so an
    unregistered operator is a refusal at stage A rather than a silent
    omission."""
    from aadistill.initialization.operators.base import (
        get_implementation, registered_implementations,
    )
    from experiments.phase_c2.search_space import C2_ALLOWED_IMPLS

    available = set(registered_implementations())
    for impl_id in C2_ALLOWED_IMPLS:
        assert impl_id in available, impl_id
    kinds = sorted(get_implementation(i).kind for i in C2_ALLOWED_IMPLS)
    assert kinds == ["ATTENTION", "DEPTH", "FFN", "RESIDUAL_WIDTH"]


def test_the_production_registration_is_sufficient_on_an_empty_registry():
    """In a fresh interpreter, where no sibling test has filled the registry.

    `attention.activation_importance_v1` is NOT a shipped default — the builtin
    register omits it so an unrestricted search cannot pick it up by accident —
    so `register_c2_operators` is the only thing that puts it there. Asked
    in-process this would pass even if that call forgot the operator, because
    the dev-box suite registers it elsewhere; the pod, running this directory
    alone, would then be the first place it was missing. C1 attempt 16 died at
    stage D for $0.4333 on the adapter half of exactly this.
    """
    program = (
        "import sys; sys.path[:0] = ['src', 'scripts']\n"
        "from aadistill.initialization.operators.base import "
        "registered_implementations\n"
        "assert not registered_implementations(), registered_implementations()\n"
        "from experiments.phase_c2.search_space import ("
        "C2_ALLOWED_IMPLS, register_c2_operators)\n"
        "register_c2_operators()\n"
        "have = set(registered_implementations())\n"
        "missing = [i for i in C2_ALLOWED_IMPLS if i not in have]\n"
        "assert not missing, missing\n"
        "from aadistill.initialization.specs.arch import get_adapter\n"
        "assert get_adapter('qwen3').adapter_version\n"
        "print('OK')\n")
    done = subprocess.run([sys.executable, "-c", program], cwd=REPO,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().endswith("OK")


def test_both_calibration_profiles_are_registered_and_materialized(registered):
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2.search_space import C2_PROFILE_IDS

    for qualified_id in C2_PROFILE_IDS:
        profile = get_profile(qualified_id)
        assert profile.materialized, f"{qualified_id} is declared, not built"
        assert profile.content_sha256


# --- C. the launcher -> driver seam, executed -------------------------------

def test_the_session_declaration_validates(c2):
    """`validate()` is the refusal that must happen before anything is priced.
    The fixture calls it; this names what it asserts."""
    spec = c2.spec
    assert spec.session_id == "autoinit-phase-c2-search1"
    assert spec.budget.arms == 0 and spec.budget.steps_per_arm == 0
    assert spec.markers.success == "ALL_DONE"
    assert spec.setup.env.get("SESSION_KIND") == "c2"
    assert "SESSION_KIND" in spec.setup.required_env


def test_the_launcher_emits_a_command_the_driver_parses(c2):
    """Both halves of the contract. The first version of `driver_command` read
    the interpreter off `ctx.args`, which has no such field — an AttributeError
    at driver start, after the pod existed and setup had succeeded."""
    assert c2.command.startswith("/opt/train/bin/python ")
    assert "autoinit_phase_c2_driver.py" in c2.command
    assert c2.parsed.stage == "all" and c2.parsed.device == "cuda"
    assert c2.parsed.authorization_path == c2.spec.authorization_path


def test_the_three_runtime_envelopes_stay_partitioned(c2):
    """The beam may not spend the baseline's reserve, and neither may reach the
    artifact-recovery reserve."""
    plan, parsed = c2.plan, c2.parsed
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
    assert plan.hard_terminate_minutes - plan.soft_stop_minutes == \
        pytest.approx(30.0, abs=1e-6)
    #: Limits handed downward are FLOORED, never rounded to nearest: `:.2f` sent
    #: the plan's $14.499561 to the driver as 14.50.
    assert parsed.soft_stop_usd <= plan.soft_stop_usd + 1e-9
    assert parsed.authorized_usd <= plan.hard_terminate_usd + 1e-9


# --- D. the staged inputs the driver consumes -------------------------------
#
# Phase-A attempt 5 died at $0.6426 on a calibration asset the session assumed
# was staged and was not. These are exactly the paths the C2 SetupManifest
# declares, so they are present under the simulated pod view as well as here.

def test_every_declared_staged_path_is_present(c2):
    from aadistill.runtime.staging_contract import staged_files

    declared = sorted(staged_files(c2.contract, REPO))
    assert declared, "the C2 staging contract declares no staged files"
    for rel in declared:
        assert (REPO / rel).exists(), f"{rel} is declared staged and is absent"


def test_both_mixtures_resolve_against_their_content_hashes(registered):
    """`resolve()` re-hashes the items file, so a drifted or half-staged mixture
    refuses here rather than silently changing every operator's statistics."""
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2.search_space import C2_PROFILE_IDS

    for qualified_id in C2_PROFILE_IDS:
        items = get_profile(qualified_id).resolve(REPO)
        assert items, f"{qualified_id} resolved to no items"
        assert all("ids" in i and "item_id" in i for i in items)


def test_the_state_eval_suite_loads_from_its_staged_directory():
    """The one metric the whole search ranks on. Without it there is no ranking,
    and it travels as a local asset because it has never been uploaded."""
    from load_state_eval import load as load_suite

    suite, items, manifest = load_suite(REPO / "artifacts/stage1/state_eval_v1")
    assert suite.suite_hash and suite.domains
    assert items, "the state-eval suite loaded no items"
    assert manifest["suite_id"] == suite.suite_id


def test_the_canonical_control_checkpoint_is_readable():
    """`run_phase_a_search` injects it as the measured control and verifies its
    frozen single-file sha256; a missing config aborts after the search."""
    from phase_a_frozen import CANONICAL_INIT

    directory = REPO / CANONICAL_INIT
    assert (directory / "model.safetensors").is_file()
    config = json.loads((directory / "config.json").read_text())
    assert config["model_type"] == "qwen3"


# --- E. the frozen space and the identity a grant binds ---------------------

def test_the_space_is_the_frozen_four_operator_search(registered):
    from experiments.phase_c2.search_space import (
        C2_ALLOWED_IMPLS, C2_IMPL_PROFILES, C2_PROFILE_IDS,
    )

    assert set(C2_ALLOWED_IMPLS) == {
        "attention.activation_importance_v1", "depth.causal_kl_greedy_v1",
        "ffn.activation_importance_v0", "width.global_pca_v0"}
    #: ATTENTION branches over both mixtures; the other three are HELD at the
    #: incumbent's values. Varying all four is the P=2 factorial Phase B ran for
    #: 9.08 h without finishing.
    assert C2_IMPL_PROFILES["attention.activation_importance_v1"] == C2_PROFILE_IDS
    assert C2_IMPL_PROFILES["depth.causal_kl_greedy_v1"] == \
        ("calib.domain_balanced@v1",)
    assert C2_IMPL_PROFILES["ffn.activation_importance_v0"] == \
        ("calib.domain_balanced@v1",)
    assert C2_IMPL_PROFILES["width.global_pca_v0"] == \
        ("calib.reasoning_heavy@v2",)
    for excluded in ("attention.weight_proxy_v0", "depth.positional_v0",
                     "composite.stage1_sandwich_v0"):
        assert excluded not in C2_ALLOWED_IMPLS


def test_the_schedule_and_ranking_policy_are_the_accepted_ones():
    from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1

    assert SCHEDULE_V1.width == 6 and SCHEDULE_V1.warmup_levels == 1
    assert PARETO_V1.qualified_id.startswith("beam.pareto_multi_objective@")
    assert len(PARETO_V1.objectives) == 3


def test_the_plan_hash_is_stable_and_is_what_the_launcher_binds(c2):
    """A grant binds this value and `plan_identity_gate` re-derives it at
    launch; if it moved between issuance and launch the session would search a
    space the authorization does not describe."""
    from experiments.phase_c2.session import c2_plan_hash

    live = c2_plan_hash()
    assert len(live) == 64 and live == c2_plan_hash()
    assert c2.spec.plan_hash == live
    assert c2.spec.evidence_fields["c2_plan_hash"] == live


# --- F. the baseline the comparison is asked against ------------------------

def test_the_frozen_baseline_construction_reproduces(registered):
    """The cheapest gate in the session: a $0 refusal here beats a 27-minute
    rebuild of the wrong baseline."""
    from experiments.phase_c2 import baseline as B

    spec = B.frozen_baseline_spec(device="cuda")
    assert spec.spec_hash == B.B_SPEC_HASH
    assert B.path_identity_of_spec(spec) == B.B_PATH
    assert B.assert_frozen_construction(spec)["verified"] is True


def test_the_baseline_identities_match_the_committed_c1_evidence():
    """Tracked records, so they travel in the bundle and are checkable here."""
    from experiments.phase_c2 import baseline as B

    prereg = load(B.PREREGISTRATION)
    arms = load(B.ARM_IDENTITIES)
    assert prereg["fixed_path"]["treatment_spec_hash"] == B.B_SPEC_HASH
    assert arms["treatment"]["artifact_digest"] == B.B_ARTIFACT_DIGEST
    assert arms["treatment"]["weights_digest"] == B.B_WEIGHTS_DIGEST


# --- G. the evidence contract -----------------------------------------------
#
# A spec naming a path the driver does not write collects nothing and reports
# `missing: 0`. Phase-B attempt 3 lost its whole search journal that way: it
# wrote `phase_b_search` while its specs named `phase_a_search`.

def test_the_artifact_specs_name_the_paths_the_driver_writes(c2):
    audit = c2.spec.artifacts.audit_dirname
    workdir = c2.driver.SEARCH_WORKDIR.name
    assert c2.driver.AUDIT.name == audit

    success = load(c2.spec.artifacts.spec_success)
    failed = load(c2.spec.artifacts.spec_failed)
    for doc in (success, failed):
        for entry in doc["entries"]:
            assert entry["pattern"].startswith(
                (f"audit/{audit}/", f"autoinit/{workdir}/")), entry["pattern"]

    required = {e["artifact_class"] for e in success["entries"] if e["required"]}
    assert {"session_evidence", "search_journal", "stage1_selection",
            "search_summary", "baseline_comparison"} <= required
    #: A failed run legitimately produced none of those, so requiring them would
    #: block the collection of what it does have.
    assert {e["artifact_class"] for e in failed["entries"]
            if e["required"]} == {"session_evidence"}


def test_the_comparison_record_is_required_success_evidence(c2):
    """Without it a search winner has no B to be compared against, which is the
    evidence hole the record exists to close."""
    from experiments.phase_c2 import comparison as C

    entry = next(e for e in load(c2.spec.artifacts.spec_success)["entries"]
                 if e["artifact_class"] == "baseline_comparison")
    assert entry["required"] is True
    assert entry["pattern"].endswith(C.FILENAME)


# --- H. the money the session is bounded by ---------------------------------

def test_the_pricing_record_verifies_and_still_names_the_accepted_ceiling():
    """`load_pricing` refuses a record that does not match its own sha256, so a
    tampered or truncated envelope cannot reach a launch."""
    from experiments.phase_c2.session import (
        c2_hard_ceiling_usd, c2_price_per_hour_usd, load_pricing,
    )

    doc = load_pricing(REPO)
    assert c2_hard_ceiling_usd(REPO) == 15.0446
    assert c2_price_per_hour_usd(REPO) == 1.09
    assert doc["totals"]["expected_usd"] == 7.1787
    assert {r["reserve"] for r in doc["reserves"]} == {
        "beam_composition_risk", "baseline_rebuild_reserve"}


# --- the selection itself ---------------------------------------------------

def test_this_directory_is_exactly_what_the_pod_runs(c2):
    """Every collectable test file outside this directory is ignored.

    Asserted over the real collectable set — every `test_*.py` anywhere under
    `tests/` — and NOT by re-deriving the complement. Re-deriving it would
    compare the launcher's rule with a copy of the launcher's rule, which is a
    test that cannot fail: it would agree with a derivation that returned the
    wrong thing for the same reason. This asks the consequence instead, so a
    top-level test FILE (the one shape a directory-level complement can miss),
    a selection pointing somewhere unexpected, or an ignore entry naming a
    directory that no longer exists all show up here.
    """
    launcher = c2.launcher
    assert launcher.POD_TEST_SELECTION == "tests/c2_preflight"
    ignores = tuple(launcher.TEST_IGNORES)
    assert ignores, "an empty ignore list sends the whole repository to the pod"
    assert launcher.POD_TEST_SELECTION not in ignores

    def ignored(rel: str) -> bool:
        return any(rel == i or rel.startswith(f"{i}/") for i in ignores)

    inside = f"{launcher.POD_TEST_SELECTION}/"
    collectable = sorted(p.relative_to(REPO).as_posix()
                         for p in (REPO / "tests").rglob("test_*.py"))
    assert collectable, "no collectable test files found at all"
    for rel in collectable:
        if rel.startswith(inside):
            assert not ignored(rel), f"{rel} is the selection and is ignored"
        else:
            assert ignored(rel), (
                f"{rel} is collectable and not ignored, so a billing pod would "
                "run it")
    #: And the session carries the derivation, so it is not decorative.
    assert tuple(c2.spec.setup.test_ignores) == ignores
    assert all(f"--ignore={i}" in c2.spec.setup.test_ignores_env()
               for i in ignores)


def test_nothing_in_this_directory_is_conditional():
    """A skip here is a hole a reader cannot see, and the C2 readiness contract
    declares zero expected skips for this selection."""
    me = Path(__file__)
    for path in sorted(me.parent.rglob("*.py")):
        #: Excluded because it NAMES the markers it is looking for.
        if path == me:
            continue
        text = path.read_text()
        for marker in ("pytest.skip", "pytest.mark.skipif", "pytest.mark.skip",
                       "pytest.importorskip", "pytest.xfail"):
            assert marker not in text, f"{path.name} uses {marker}"

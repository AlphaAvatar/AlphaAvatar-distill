"""The only tests a paid baseline-completion pod runs. Positive selection.

The pod's blocking gate is `pytest tests/ $SESSION_TEST_IGNORES`, so a session's
selection is a COMPLEMENT that the launcher derives from the tree. This
directory is the completion's selection, and it exists because Search-1's could
not be reused: `tests/c2_preflight` asserts that every path SEARCH-1 stages is
present, including the canonical 0.6B control it injects as its measured
baseline. This session has no control and deliberately does not stage one, so
running Search-1's preflight under this session's staged view fails on two
tests that are entirely correct about Search-1.

Narrowing Search-1's gate to fit this session was the other option and the wrong
one: it is a consumed experiment's satisfiable gate, and a guard narrowed to fit
the session under test stops guarding the session it was written for.

So every test here earns its place by naming a failure that would otherwise
appear only AFTER setup has been paid for. What this session does is narrow:
bind identities, rebuild one checkpoint through a frozen path, measure it once,
and write one comparison. It trains nothing, searches nothing, evaluates no
battery and exports no checkpoint.

**Zero skips, by construction**, enforced by the last test here. Nothing needs
CUDA: the shared CPU-test contract sets `CUDA_VISIBLE_DEVICES=` and neutralises
`HOME` and the HF caches, so a GPU-only assertion could only appear as a skip,
and a skip in this directory is a hole a reader cannot see. GPU visibility is
proved by `ROPE_OK` and the driver's own device handling, outside this scope.

**Only staged paths are touched.** The simulator models the pod's view by hiding
repository artifacts the session does not stage, so a test reading an unstaged
path would pass on the dev box and fail here. The paths asserted below are
exactly what this session's `SetupManifest` declares.
"""
from __future__ import annotations

import json
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
    """The registration the driver performs at stage A, performed here."""
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()


@pytest.fixture(scope="module")
def completion(registered):
    """Launcher, session, plan and the emitted driver command."""
    from session_specs import load_session_launcher, session_args

    from aadistill.runtime.staging_contract import derive_contract
    from experiments.phase_c2 import baseline_completion as BC

    launcher = load_session_launcher("autoinit_phase_c2_baseline_launch")
    args = session_args(launcher)
    spec = launcher.spec(args).validate()

    #: The production call, not a transcription of it.
    plan = spec.budget.plan(price_per_hour=BC.price_per_hour_usd(REPO),
                            authorized_usd=BC.hard_ceiling_usd(REPO))
    ctx = SimpleNamespace(args=args, image_digest="preflight",
                          price=BC.price_per_hour_usd(REPO), spent_usd=0.0)
    command = spec.driver_command(ctx, plan)

    return SimpleNamespace(
        launcher=launcher, args=args, spec=spec, plan=plan, command=command,
        contract=derive_contract(spec.setup, session_id=BC.SESSION_ID))


# --- A. the staged inputs this driver consumes ------------------------------
#
# Phase-A attempt 5 died at $0.6426 on a calibration asset the session assumed
# was staged and was not.


def test_every_declared_staged_path_is_present(completion):
    from aadistill.runtime.staging_contract import staged_files

    declared = sorted(staged_files(completion.contract, REPO))
    assert declared, "the completion staging contract declares no staged files"
    for rel in declared:
        assert (REPO / rel).exists(), f"{rel} is declared staged and is absent"


def test_the_canonical_control_is_NOT_staged(completion):
    """This session has no control, and staging one would be a different session.

    Asserted positively because the absence is a decision: Search-1 injects the
    canonical 0.6B init as its measured baseline, and a completion that staged
    it would be carrying an input nothing in its path consumes.
    """
    from aadistill.runtime.staging_contract import staged_files

    declared = sorted(staged_files(completion.contract, REPO))
    assert not any("qwen3_0p6b_init_v0" in rel for rel in declared), declared


def test_both_mixtures_resolve_against_their_content_hashes(registered):
    """`resolve()` re-hashes the items file, so a drifted or half-staged mixture
    refuses here rather than silently changing the operators' statistics."""
    from aadistill.initialization.calibration.profiles import get_profile
    from experiments.phase_c2.search_space import C2_PROFILE_IDS

    for qualified_id in C2_PROFILE_IDS:
        items = get_profile(qualified_id).resolve(REPO)
        assert items, f"{qualified_id} resolved to no items"
        assert all("ids" in item and "item_id" in item for item in items)


def test_the_state_eval_suite_loads_from_its_declared_root():
    """The one metric B is measured on, from the root its declaration names.

    The driver asked `load_suite(REPO)` until 2026-09-16, and the repository
    root holds neither `manifest.json` nor `items.jsonl` -- a failure that would
    have landed inside the only stage that spends money.
    """
    from load_state_eval import load as load_suite

    from experiments.phase_c2.frozen_assets import state_eval_root

    root = state_eval_root(REPO)
    suite, items, _manifest = load_suite(root)
    assert items, "the suite resolved to no items"
    frozen = json.loads((REPO / "logs/stages/stage-1/phase_c2/runs/attempt4"
                         "/evidence/c2_frozen_comparison_inputs.json").read_text())
    assert suite.suite_hash == frozen["suite"]["hash"], (
        "the staged suite is not the one the frozen candidates were measured on")


# --- B. the frozen inputs the comparison is computed against ----------------


def test_the_frozen_candidates_load_and_match_the_cited_ranking():
    from aadistill.initialization.planning import stage1_selection

    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2.frozen_inputs import (
        load_frozen_candidates, load_record)

    record = load_record(REPO / BC.FROZEN_INPUTS)
    candidates = load_frozen_candidates(
        REPO / BC.FROZEN_INPUTS,
        expect_suite_hash=record["suite"]["hash"],
        expect_policy_hash=record["policy"]["hash"])
    assert len(candidates) == 5
    selection = stage1_selection.load(REPO / BC.SELECTION_RECORD)
    assert selection["selection_sha256"] == record["sources"][
        "selection_commitment_sha256"]


def test_the_frozen_candidates_satisfy_the_ranking_contract():
    """`PARETO_V1.rank` reads `validity`, `evaluation` and `impl_ids` and calls
    `ready_for_ranking`. Omitting any raises inside the ranking -- after B has
    been rebuilt and measured."""
    from aadistill.initialization.planning.ranking import PARETO_V1
    from aadistill.initialization.specs.state import StateValidity

    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2.frozen_inputs import load_frozen_candidates

    for candidate in load_frozen_candidates(REPO / BC.FROZEN_INPUTS):
        assert candidate.validity is StateValidity.MEASURED
        candidate.ready_for_ranking(PARETO_V1.required_metrics())
        assert candidate.impl_ids


def test_the_frozen_baseline_construction_verifies(registered):
    """The cheapest gate in the session: a wrong recipe caught here costs
    nothing, and caught after the rebuild it costs the rebuild."""
    from experiments.phase_c2 import baseline as B

    spec = B.frozen_baseline_spec(device="cuda")
    evidence = B.assert_frozen_construction(spec)
    assert evidence["verified"] is True
    assert spec.spec_hash == B.B_SPEC_HASH
    assert B.path_identity_of_spec(spec) == B.B_PATH


# --- C. the budget the driver is handed -------------------------------------


def test_the_plan_reproduces_the_priced_ceiling(completion):
    from experiments.phase_c2 import baseline_completion as BC

    plan = completion.plan
    assert plan.hard_terminate_usd <= BC.hard_ceiling_usd(REPO) + 1e-9
    assert plan.soft_stop_usd < plan.hard_terminate_usd
    #: The recovery reserve is the gap between the soft stop and the hard stop.
    reserve = float(BC.pricing(REPO)["totals"][
        "artifact_recovery_reserve_minutes"])
    assert plan.hard_terminate_minutes - plan.soft_stop_minutes == pytest.approx(
        reserve, abs=1e-6)


def test_the_driver_command_passes_floored_limits_and_no_identity(completion):
    """Limits handed downward are FLOORED, never rounded: `:.2f` once sent a
    plan's $14.499561 to a driver as 14.50."""
    import shlex

    parts = shlex.split(completion.command)
    assert "scripts/pod/autoinit_phase_c2_baseline_driver.py" in parts
    flags = {parts[i]: parts[i + 1] for i in range(len(parts) - 1)
             if parts[i].startswith("--")}
    assert float(flags["--soft-stop-usd"]) <= completion.plan.soft_stop_usd + 1e-9
    assert float(flags["--rebuild-minutes"]) > 27.665, (
        "the rebuild allowance must exceed the reserve that failed in Attempt 4")
    #: No scientific identity on the command line: the driver binds the Search-1
    #: run id and config hash from the frozen record itself.
    assert "--config-hash" not in flags
    for required in ("--protocol", "--frozen-inputs", "--selection-record"):
        assert required in flags, required


# --- D. what this session cannot do ----------------------------------------


def test_the_beam_is_unreachable_from_this_session():
    """By import graph, not by instruction."""
    import ast

    for rel in ("scripts/pod/autoinit_phase_c2_baseline_driver.py",
                "scripts/pod/autoinit_phase_c2_baseline_launch.py"):
        tree = ast.parse((REPO / rel).read_text())
        named = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                named.add(node.module.split(".")[-1])
                named |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                named |= {alias.name.split(".")[-1] for alias in node.names}
        leaked = sorted({"run_phase_a_search", "BeamSearch", "SearchConfig",
                         "phase_a_search"} & named)
        assert not leaked, f"{rel} names {leaked}"


def test_the_authorization_on_this_pod_cannot_authorize_a_beam():
    """The artifact the setup script loaded, read again here."""
    import os

    from experiments.phase_c2.baseline_completion import (
        BaselineCompletionAuthorization)

    #: Keyed on the FILE, not on the environment variable. The launch-bound
    #: sweep runs on the grant-containing tree -- BEFORE the authorization is
    #: issued, by design -- and the simulator exports the session environment,
    #: so `SESSION_AUTH_PATH` is named there while the artifact cannot yet
    #: exist. Keying on the variable made this test unsatisfiable at sweep time
    #: and satisfiable on the pod: the same shape as a skip that inverts
    #: between the two machines.
    #:
    #: Neither branch is a skip. Pre-issuance the property is asserted on the
    #: TYPE, which is where it is defined and where it cannot be edited away;
    #: on the pod the actual artifact is read.
    path = os.environ.get("SESSION_AUTH_PATH")
    resolved = (REPO / path) if path else None
    if resolved is None or not resolved.is_file():
        assert BaselineCompletionAuthorization.authorizes_c2_search1.fget(
            None) is False
        assert BaselineCompletionAuthorization.authorizes_c2_baseline_completion.fget(
            None) is True
        return
    auth = BaselineCompletionAuthorization.load(resolved)
    assert auth.authorizes_c2_baseline_completion is True
    assert auth.authorizes_c2_search1 is False
    assert auth.allows_recovery_training is False


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

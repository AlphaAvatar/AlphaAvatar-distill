"""Everything the paid replay can fail before its first operator runs.

This is the session's OWN pod selection, not another experiment's. Search-1's
preflight asserts the canonical 0.6B control, which this session deliberately
does not stage; the full search's asserts a staged asset set that is not this
one. A guard narrowed to fit the session under test stops guarding the session
it was written for, so each gets its own.

What belongs here: checks that are cheap, that run inside the pod simulator
(empty HOME, hidden gitignored artifacts, synthetic token), and whose failure
would otherwise be discovered on a billing machine. What does not: anything
needing a GPU, the teacher weights, or the network.

Nothing here skips. A selection with a conditional skip is a selection that can
report green having checked nothing, and gate 12 can see THAT a test skipped but
never WHY.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from experiments.phase_c2 import replay as RG  # noqa: E402
from experiments.phase_c2 import replay_specs as RS  # noqa: E402

SELECTION = "7271091c91416b523865ea1320190e4a9ceafb97ecc68bc1b0e49f621573b673"
SOURCE_COMMIT = "2421f630bc812d414ae25245e855059ffe29610d"
LEAF_PARAMS = 596_049_920


# -- the evidence the session reads -----------------------------------------
def test_the_committed_evidence_is_present_and_parses():
    """The three files the replay's plan is made of.

    They are declared non-source inputs, so they travel in the bundle. A pod
    that checked out the commit and found them missing would have paid for
    setup to discover that its plan does not exist.
    """
    for rel in (RS.SELECTION_REL, RS.JOURNAL_REL, RS.TELEMETRY_REL):
        path = ROOT / rel
        assert path.is_file(), f"{rel} did not arrive on this pod"
        assert path.stat().st_size > 0, f"{rel} is empty"


def test_the_selection_is_the_frozen_top_five():
    record = RS.load_selection(ROOT)
    assert record["selection_sha256"] == SELECTION
    assert record["n_selected"] == 5 == len(record["selected"])


def test_the_journal_supports_every_selected_leaf():
    states = RS.load_states(ROOT)
    for entry in RS.load_selection(ROOT)["selected"]:
        chain = RS.resolve_ancestry(states, entry["state_id"])
        assert len(chain) == 4
        assert chain[-1]["artifact_digest"] == entry["artifact_digest"]


# -- the specs the session will execute --------------------------------------
def test_five_specs_build_with_every_step_pinned():
    leaves = RS.build_replay_leaves(ROOT, device="cpu")
    assert len(leaves) == 5
    pins = 0
    for leaf in leaves:
        assert len(leaf.spec.steps) == 4
        for step in leaf.spec.steps:
            assert step.expected_artifact_digest, (
                f"{leaf.state_id}: step {step.impl_id} is unpinned")
            pins += 1
        assert leaf.num_parameters == LEAF_PARAMS
    assert pins == 20, "twenty pinned steps, not five pinned leaves"


def test_every_operator_the_paths_name_is_registered():
    """Building the specs resolves each impl_id, so this is really an assertion
    that the C2 library registered and not only the builtins: four of the five
    paths name `attention.activation_importance_v1`, which is not shipped."""
    leaves = RS.build_replay_leaves(ROOT, device="cpu")
    named = {step.impl_id for leaf in leaves for step in leaf.spec.steps}
    assert "attention.activation_importance_v1" in named
    from aadistill.initialization.operators.base import get_implementation

    for impl_id in sorted(named):
        assert get_implementation(impl_id) is not None


def test_the_operators_have_not_moved_since_attempt_three():
    report = RS.assert_operators_unmoved(ROOT)
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["modified_or_removed"] == {}


# -- the calibration branch the driver actually takes ------------------------
def test_both_calibration_profiles_resolve_from_disk_on_this_pod():
    """The branch every test used to skip by passing `calibration_items=`.

    The driver deliberately passes none, so `materialize_fixed_path` resolves
    each profile from the repository and verifies its content hash. A mixture
    that did not arrive, or arrived changed, would change every operator's
    statistics without changing any recorded identity — and the first place that
    surfaced, in another session, was a paid pod.
    """
    from aadistill.initialization.calibration.profiles import get_profile

    for declared in RS.load_selection(ROOT)["profiles"]:
        profile = get_profile(declared["qualified_id"])
        assert profile.profile_hash == declared["profile_hash"], (
            f"{declared['qualified_id']} hashes to {profile.profile_hash} and "
            f"the selection was searched against {declared['profile_hash']}")
        items = profile.resolve(repo_root=str(ROOT))
        assert items, f"{declared['qualified_id']} resolved to no items"
        assert profile.content_sha256 == declared["content_sha256"]


# -- the session's own governance -------------------------------------------
def test_the_executable_closure_derives_on_this_tree():
    live = RG.current_executable(ROOT)
    assert live["n_files"] > 0
    assert len(live["digest"]) == 64


def test_the_plan_hash_is_the_source_binding():
    """Not a separate document that could drift from the pins."""
    from aadistill.infrastructure.manifest import sha256_json

    assert RG.plan_hash(ROOT) == sha256_json(RS.source_binding(ROOT))


def test_the_authorization_type_refuses_every_adjacent_session():
    """Read off the class, so a document cannot claim what the type denies."""
    a = RG.ReplayAuthorization
    for prop in ("authorizes_c2_full_search", "authorizes_c2_search1",
                 "authorizes_c2_baseline_completion",
                 "authorizes_behavioural_selection"):
        assert getattr(a, prop).fget(None) is False, prop
    assert a.authorizes_c2_replay.fget(None) is True


def test_the_artifact_specs_load_through_the_collectors_own_loader():
    """An invented lifecycle makes the document unloadable and the collector
    exits 1 before it looks at a file — on the pod, at closeout, with the
    weights still on it."""
    from collect_artifacts import load_specs

    for name in ("c2_replay_artifacts.json", "c2_replay_artifacts_failed.json"):
        specs = load_specs(str(ROOT / "configs/autoinit" / name))
        assert specs
        assert any(s.required for s in specs)


# -- the driver's own preconditions ------------------------------------------
def test_the_driver_populates_the_profile_registry_it_resolves_from():
    """In a SUBPROCESS, because a registry is full in any pytest session a
    sibling has filled.

    The driver passes no `calibration_items`, so every operator resolves its
    mixture from the process-global registry. The driver's own import chain
    registers nothing, so without an explicit call the session completes setup,
    passes every gate and dies at path 1 step 1 — on a billing machine. Asserted
    in a fresh interpreter because in-process this test would pass on somebody
    else's registration and prove nothing.
    """
    import subprocess

    probe = (
        "import sys;"
        "sys.path[:0]=['src','scripts','scripts/autoinit','scripts/pod'];"
        "import autoinit_c2_replay_driver;"
        "from aadistill.initialization.calibration.profiles import _PROFILES;"
        "before=sorted(_PROFILES);"
        "from experiments.calibration import register_builtin_profiles;"
        "register_builtin_profiles();"
        "after=sorted(_PROFILES);"
        "print(repr((before, after)))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-600:]
    before, after = eval(out.stdout.strip())  # noqa: S307 - our own literal
    assert before == [], (
        "the driver's import chain now registers profiles by side effect. That "
        "is not a fix — it makes the explicit call in bind_identities look "
        "unnecessary, and the next importer to drop it re-creates the bug.")
    assert "calib.domain_balanced@v1" in after
    assert "calib.reasoning_heavy@v2" in after

    #: And the driver must actually make that call.
    source = (ROOT / "scripts/pod/autoinit_c2_replay_driver.py").read_text()
    assert "register_builtin_profiles()" in source


def test_the_driver_module_imports_and_declares_its_markers():
    import autoinit_c2_replay_driver as D

    assert D.N_LEAVES == 5
    assert D.LEAF_BYTES == 1_192_135_096
    for marker in (D.SUCCESS_MARKER, D.FAILURE_MARKER, D.LEAF_MARKER,
                   D.MISMATCH_MARKER):
        assert marker.startswith("C2_REPLAY")


def test_the_driver_refuses_a_soft_stop_that_leaves_no_teardown():
    import autoinit_c2_replay_driver as D

    saved = sys.argv
    try:
        sys.argv = ["driver", "--workdir", "/tmp/x", "--rate", "1.09",
                    "--authorized-usd", "5.19", "--soft-stop-usd", "5.19"]
        try:
            D.main()
        except SystemExit as exc:
            assert "teardown" in str(exc)
        else:
            raise AssertionError("an exhausting soft stop was accepted")
    finally:
        sys.argv = saved


def test_the_workdir_filesystem_can_hold_the_worst_path_and_every_leaf():
    """The disk rule, evaluated against THIS pod rather than a plan.

    Discovering at path 4 that the volume is too small wastes the first three.
    """
    import shutil

    import autoinit_c2_replay_driver as D

    need = D.WORST_PATH_BYTES + D.N_LEAVES * D.LEAF_BYTES
    free = shutil.disk_usage(ROOT).free
    assert free >= need, (
        f"{free / 2**30:.1f} GiB free where the worst path plus five retained "
        f"leaves needs {need / 2**30:.1f} GiB")


def test_the_per_path_bound_is_derived_from_the_telemetry_not_a_constant():
    """Admission control spends against this number, so it must come from the
    record rather than from a literal somebody updated once."""
    worst = RS.worst_seconds_by_impl(ROOT)
    assert worst["depth.causal_kl_greedy_v1"] > 20 * 60, (
        "the causal-KL DEPTH operator bounds below 20 minutes; the telemetry was "
        "probably read wrong, which underprices every path carrying it")

    states = RS.load_states(ROOT)
    for leaf in RS.build_replay_leaves(ROOT, device="cpu"):
        #: Recomputed from the telemetry here, independently of the builder, so
        #: this cannot pass by reading the builder's own answer back.
        chain = RS.resolve_ancestry(states, leaf.state_id)
        expected = 1.5 + sum(
            worst[node["impl_ids"][-1]] for node in chain) / 60.0
        assert leaf.bounded_minutes == round(expected, 2), leaf.state_id
        assert leaf.bounded_minutes > 0


def test_the_session_fits_its_authorized_money_at_the_approved_rate():
    """The planner refuses a plan it cannot fund. It refused once already, at
    the original ceiling, which is why this session has the ceiling it has."""
    import autoinit_c2_replay_launch as L

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"])
    plan = L.budget(args).plan(price_per_hour=1.09,
                               authorized_usd=RG.GPU_HARD_USD)
    assert plan.hard_terminate_minutes / 60.0 * 1.09 <= RG.GPU_HARD_USD
    assert plan.soft_stop_minutes < plan.hard_terminate_minutes


def test_the_teacher_revision_is_the_one_the_paths_are_rooted_at():
    from phase_a_frozen import TEACHER_ID, TEACHER_REVISION

    for leaf in RS.build_replay_leaves(ROOT, device="cpu"):
        assert leaf.spec.root_repo_id == TEACHER_ID
        assert leaf.spec.root_revision == TEACHER_REVISION


def test_the_launcher_builds_its_session_spec_on_this_tree():
    import autoinit_c2_replay_launch as L

    spec = L.spec(L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"]))
    assert spec.setup.env["SESSION_KIND"] == "c2_replay"
    assert spec.markers.success == "C2_REPLAY_ALL_DONE"
    #: The product is weights, so the collector must be wired to fetch them.
    assert spec.artifacts.fetch_products is not None
    assert spec.artifacts.products_secured is not None


def test_the_launcher_satisfies_the_runners_argument_contract():
    """The runner reads attributes off the namespace and calls `run_session`
    with a fixed signature. Both were wrong here and neither is visible from
    building the spec: `--run-id` produced no `out`, and `run_session` was
    called without `repo_root`. A launcher that cannot be invoked is a launcher
    that fails after its authorization has been issued, which costs a whole
    governance chain to discover.
    """
    import inspect

    import autoinit_c2_replay_launch as L
    from aadistill.infrastructure.session_runner import run_session

    args = L.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40,
         "--bundle", "b.bundle", "--run-id", "preflight"])
    #: `SessionRunner.save()` writes `args.out`.
    assert args.out == L.session_record_path("preflight")

    required = [name for name, p in
                inspect.signature(run_session).parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    source = inspect.getsource(L.main)
    assert "run_session(spec(args), args, REPO_ROOT" in source, (
        f"run_session requires {required}; main() must pass them all")


def test_the_setup_script_dispatches_this_session_kind():
    """A missing branch is not a type error — it is a late refusal on a billing
    machine, and it has cost this project two paid sessions."""
    text = (ROOT / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert 'SESSION_KIND" = "c2_replay"' in text
    branch = text.split('SESSION_KIND" = "c2_replay"', 1)[1]
    branch = branch.split("elif [", 1)[0]
    assert "ReplayAuthorization" in branch
    assert "authorizes_c2_full_search is False" in branch


def test_the_requirement_record_states_the_money_this_session_runs_under():
    doc = json.loads((ROOT / "logs/stages/stage-1/phase_c2_replay/plans"
                      / "replay_requirement.json").read_text())
    assert doc["derived_all_in_usd"] == RG.ALL_IN_USD
